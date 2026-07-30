"""Sprint 1492 — a dropped on-chain payment leg must not be invisible.

Established by an adversarial multi-agent trace (26 agents) of the paid-dispatch
money path, which corrected TWO prior recorded conclusions of mine:

  * BLOCKER-0 "the provider is never paid" is FALSE for the live default rail —
    compute_requester transfers AND gossips, so the payee's own node credits it.
  * My CORRECTION was also false: `broadcast_tx` does NOT credit a remote payee.
    It routes to BatchSettlementManager.enqueue -> _resolve_address, which accepts
    only 0x addresses (>=40 chars) or this node's own id. A peer node_id is 32 hex
    chars with no 0x, so it resolves to None, enqueue returns False at DEBUG, and
    node.py discards that return entirely.

So an escrow release to a REMOTE payee logs a cheerful "Escrow released: … ->
<provider>" while the on-chain leg silently never ran. That invisibility is why
the class survived three prior audits — this makes it observable.

The distinction matters: skipping an INTERNAL wallet (escrow-<uuid>, system) is
correct and routine, so it must stay quiet or the warning becomes noise.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.economy.batch_settlement import (
    BatchSettlementManager,
    _looks_like_node_id,
)

PEER_NODE_ID = "d437aa67d99cff4a6a17179f5c731b77"        # 32 hex, real shape
ADDR = "0x" + "a1" * 20


# ── the shape test ──────────────────────────────────────────────────

def test_recognises_a_real_peer_node_id():
    assert _looks_like_node_id(PEER_NODE_ID)


def test_does_not_mistake_internal_wallets_for_node_ids():
    """These are skipped BY DESIGN and must stay at DEBUG."""
    for internal in ("escrow-6f1b2c3d-4e5f-6789-abcd-ef0123456789",
                     "system", "treasury", ""):
        assert not _looks_like_node_id(internal), internal


def test_does_not_mistake_an_eth_address_for_a_node_id():
    assert not _looks_like_node_id(ADDR)
    assert not _looks_like_node_id("0x" + "a" * 30)


def test_rejects_wrong_length_and_non_hex():
    assert not _looks_like_node_id("abc123")                  # too short
    assert not _looks_like_node_id("z" * 32)                  # not hex
    assert not _looks_like_node_id("d437aa67d99cff4a6a17179f5c731b7")   # 31


# ── the log-level split ─────────────────────────────────────────────

def _mgr(node_id="mynode", connected=ADDR):
    m = object.__new__(BatchSettlementManager)
    m._node_id = node_id
    m._connected_address = connected
    m._settled_ids = set()
    m._pending = {}
    led = MagicMock()
    led._is_initialized = True
    m._ftns_ledger = led
    m._lock = asyncio.Lock()
    m._queue = []
    m._persist = MagicMock()
    m._mark_settled = MagicMock()
    from prsm.economy.batch_settlement import SettlementMode
    m.mode = SettlementMode.MANUAL
    m.flush_threshold = 1e9
    m.max_queue_size = 10**6
    return m


def _tx(to_wallet, amount=5.0, tx_id="tx-1"):
    t = MagicMock()
    t.tx_id, t.to_wallet, t.from_wallet, t.amount = tx_id, to_wallet, "me", amount
    return t


@pytest.mark.asyncio
async def test_a_dropped_PEER_payment_warns_loudly(caplog):
    """★ THE fix. Was DEBUG and discarded — an operator had no way to learn the
    on-chain half of a real payment never ran."""
    with caplog.at_level(logging.WARNING):
        ok = await _mgr().enqueue(_tx(PEER_NODE_ID, amount=12.5))
    assert ok is False
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "NO on-chain leg" in msg
    assert PEER_NODE_ID[:20] in msg
    assert "only on" in msg and "ledger" in msg      # names the consequence
    assert "12.5" in msg                             # and the amount at risk


@pytest.mark.asyncio
async def test_an_internal_wallet_skip_stays_QUIET(caplog):
    """★ The other half — escrow wallets have no on-chain identity by design.
    Warning on those would bury the real signal in noise."""
    with caplog.at_level(logging.WARNING):
        ok = await _mgr().enqueue(_tx("escrow-6f1b2c3d-4e5f-6789-abcd-ef0123"))
    assert ok is False
    assert not [r for r in caplog.records if "NO on-chain leg" in r.getMessage()]


@pytest.mark.asyncio
async def test_a_payable_address_is_not_warned_about(caplog):
    """A resolvable payee must produce no warning at all."""
    with caplog.at_level(logging.WARNING):
        await _mgr().enqueue(_tx(ADDR))
    assert not [r for r in caplog.records if "NO on-chain leg" in r.getMessage()]


@pytest.mark.asyncio
async def test_the_nodes_OWN_id_resolves_and_is_not_warned_about(caplog):
    """_resolve_address maps this node's own id to its connected address, so it
    is payable and must not trip the warning."""
    m = _mgr(node_id=PEER_NODE_ID)
    with caplog.at_level(logging.WARNING):
        await m.enqueue(_tx(PEER_NODE_ID))
    assert not [r for r in caplog.records if "NO on-chain leg" in r.getMessage()]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
