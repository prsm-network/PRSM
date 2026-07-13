"""Sprint 540 — Pattern A daemon-mediated bridge (deposit half).

Replaces the polygon_mumbai-era FTNSBridge scaffold (sprint 539
investigation) with a daemon-side flow: linked-address registry
in LocalLedger + InboundMonitor credit hook.

These pins cover:
  1. LocalLedger.link_eth_address — basic + idempotent re-link
  2. LocalLedger.wallet_for_eth_address — reverse lookup
  3. LocalLedger.eth_address_for_wallet — forward lookup
  4. UNIQUE constraint on eth_address (one wallet per addr)
  5. BRIDGE_DEPOSIT / BRIDGE_WITHDRAW transaction types exist
  6. InboundMonitor._credit_deposit — credits linked wallet on
     detected inbound; no-op for unlinked address; dedup on
     repeat tx_hash
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.local_ledger import LocalLedger, TransactionType


@pytest.fixture
def tmp_db_path():
    with tempfile.NamedTemporaryFile(
        suffix=".db", delete=False,
    ) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
async def ledger(tmp_db_path):
    led = LocalLedger(db_path=tmp_db_path)
    await led.initialize()
    yield led
    if led._db is not None:
        await led._db.close()


# ── TransactionType ──────────────────────────────────────


def test_bridge_transaction_types_exist():
    assert TransactionType.BRIDGE_DEPOSIT.value == "bridge_deposit"
    assert TransactionType.BRIDGE_WITHDRAW.value == "bridge_withdraw"


# ── link_eth_address ─────────────────────────────────────


@pytest.mark.asyncio
async def test_link_eth_address_creates_wallet_if_missing(ledger):
    """Linking auto-creates the wallet row (matches credit/debit
    auto-create semantics)."""
    await ledger.link_eth_address("alice", "0x" + "a" * 40)
    assert await ledger.wallet_exists("alice")


@pytest.mark.asyncio
async def test_link_eth_address_round_trip(ledger):
    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    await ledger.link_eth_address("alice", addr)
    # Forward lookup (returns lowercased)
    assert await ledger.eth_address_for_wallet("alice") == addr.lower()
    # Reverse lookup
    assert await ledger.wallet_for_eth_address(addr) == "alice"
    # Reverse lookup is case-insensitive on input
    assert await ledger.wallet_for_eth_address(
        addr.upper(),
    ) == "alice"


@pytest.mark.asyncio
async def test_link_eth_address_rejects_bad_format(ledger):
    with pytest.raises(ValueError):
        await ledger.link_eth_address("alice", "not-an-address")
    with pytest.raises(ValueError):
        await ledger.link_eth_address("alice", "")
    with pytest.raises(ValueError):
        await ledger.link_eth_address("", "0x" + "a" * 40)


@pytest.mark.asyncio
async def test_link_eth_address_refuses_cross_wallet_relink(ledger):
    """sp1441 (deposit-audit hardening) — re-linking an address ALREADY bound to a
    DIFFERENT wallet is REFUSED (deposit-theft guard), not silently moved. The credit
    path resolves the wallet at scan time, so a silent move let whoever links a known
    deposit address LAST steal its next inbound transfer. alice keeps her address."""
    addr = "0x" + "a" * 40
    await ledger.link_eth_address("alice", addr)
    with pytest.raises(ValueError):
        await ledger.link_eth_address("bob", addr)
    # alice still owns it; bob got nothing.
    assert await ledger.wallet_for_eth_address(addr) == "alice"
    assert await ledger.eth_address_for_wallet("alice") == addr
    assert await ledger.eth_address_for_wallet("bob") is None


@pytest.mark.asyncio
async def test_link_eth_address_same_wallet_relink_is_idempotent(ledger):
    """sp1441 — re-linking the SAME address to the SAME wallet is a harmless no-op."""
    addr = "0x" + "a" * 40
    await ledger.link_eth_address("alice", addr)
    await ledger.link_eth_address("alice", addr)  # must not raise
    assert await ledger.wallet_for_eth_address(addr) == "alice"


@pytest.mark.asyncio
async def test_link_eth_address_wallet_can_rebind_to_a_fresh_address(ledger):
    """sp1441 — a wallet rebinding ITS OWN link to a NEW (unclaimed) address is unaffected
    by the guard (only ANOTHER wallet's claimed address is protected)."""
    a1, a2 = "0x" + "a" * 40, "0x" + "b" * 40
    await ledger.link_eth_address("alice", a1)
    await ledger.link_eth_address("alice", a2)   # rebind to a fresh address — allowed
    assert await ledger.eth_address_for_wallet("alice") == a2
    assert await ledger.wallet_for_eth_address(a1) is None


@pytest.mark.asyncio
async def test_unknown_eth_address_returns_none(ledger):
    assert await ledger.wallet_for_eth_address(
        "0x" + "f" * 40,
    ) is None


# ── InboundMonitor._credit_deposit ────────────────────────


@pytest.mark.asyncio
async def test_credit_deposit_linked_address_credits_wallet(
    tmp_db_path,
):
    """Detected inbound from linked address → BRIDGE_DEPOSIT
    credit on linked wallet."""
    from prsm.economy.ftns_onchain import InboundMonitor

    led = LocalLedger(db_path=tmp_db_path)
    await led.initialize()
    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    await led.link_eth_address("alice", addr)
    bal_before = await led.get_balance("alice")
    assert bal_before == 0.0

    mock_ledger = MagicMock()
    mock_ledger.w3 = None  # not used by _credit_deposit
    mon = InboundMonitor(
        mock_ledger, interval_seconds=60, local_ledger=led,
    )
    transfer = {
        "tx_hash": "0x" + "ab" * 32,
        "from_address": addr,
        "amount_ftns": 1.5,
        "block_number": 46175008,
    }
    await mon._credit_deposit(transfer)

    assert await led.get_balance("alice") == 1.5
    # Dedup: re-running same transfer should NOT double-credit
    await mon._credit_deposit(transfer)
    assert await led.get_balance("alice") == 1.5


@pytest.mark.asyncio
async def test_credit_deposit_unlinked_address_noop(tmp_db_path):
    """Inbound from an UNLINKED address must NOT credit anywhere.
    Logs a warning + skips."""
    from prsm.economy.ftns_onchain import InboundMonitor

    led = LocalLedger(db_path=tmp_db_path)
    await led.initialize()
    # Don't link the sender
    mock_ledger = MagicMock()
    mock_ledger.w3 = None
    mon = InboundMonitor(
        mock_ledger, interval_seconds=60, local_ledger=led,
    )
    transfer = {
        "tx_hash": "0x" + "cd" * 32,
        "from_address": "0x" + "f" * 40,
        "amount_ftns": 5.0,
        "block_number": 46175008,
    }
    await mon._credit_deposit(transfer)
    # No wallet was credited
    assert await led.wallet_for_eth_address(
        "0x" + "f" * 40,
    ) is None


@pytest.mark.asyncio
async def test_credit_deposit_no_local_ledger_noop():
    """If InboundMonitor was constructed without local_ledger,
    _credit_deposit is a silent no-op (backwards compat)."""
    from prsm.economy.ftns_onchain import InboundMonitor

    mock_ledger = MagicMock()
    mock_ledger.w3 = None
    mon = InboundMonitor(mock_ledger, interval_seconds=60)
    # No local_ledger passed; should not raise
    await mon._credit_deposit({
        "tx_hash": "0x" + "ef" * 32,
        "from_address": "0x" + "a" * 40,
        "amount_ftns": 1.0,
        "block_number": 100,
    })


@pytest.mark.asyncio
async def test_credit_deposit_zero_amount_skips(tmp_db_path):
    """Edge: amount=0 transfers (shouldn't happen on-chain but
    defensive) must not credit anything."""
    from prsm.economy.ftns_onchain import InboundMonitor

    led = LocalLedger(db_path=tmp_db_path)
    await led.initialize()
    addr = "0x" + "a" * 40
    await led.link_eth_address("alice", addr)
    mon = InboundMonitor(
        MagicMock(), interval_seconds=60, local_ledger=led,
    )
    await mon._credit_deposit({
        "tx_hash": "0x" + "ff" * 32,
        "from_address": addr,
        "amount_ftns": 0,
        "block_number": 100,
    })
    assert await led.get_balance("alice") == 0.0
