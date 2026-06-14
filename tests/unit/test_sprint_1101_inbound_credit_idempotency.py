"""Sprint 1101 — fold the Domain-05 review F1 (MEDIUM, WIRED): make the inbound
bridge-deposit credit IDEMPOTENT at the durable ledger.

The Domain-05 review found that `InboundMonitor._credit_deposit` credits the off-chain
ledger (irreversible) BEFORE persisting the dedup mark, and `LocalLedger.credit` mints a
fresh-UUID tx with no natural key. So a crash in that window + the sprint-543 restart
catch-up scan re-presents the same on-chain Transfer → it is credited a SECOND time, and
that off-chain balance is withdrawable as real on-chain FTNS (drains the operator pool).

The fix is defense BELOW the checkpoint store: `LocalLedger.credit(idempotency_key=...)`
derives a DETERMINISTIC tx_id, so a replayed credit collides on the PRIMARY KEY and is a
no-op (returns the existing tx). The monitor passes
``bridge-deposit:<recipient>:<tx_hash>`` so even if the mark is lost the ledger itself
de-duplicates. Exactly-once instead of at-least-once.
"""
from __future__ import annotations

import pytest

from prsm.node.local_ledger import LocalLedger, TransactionType


async def _fresh_ledger() -> LocalLedger:
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    await ledger.create_wallet("alice", "Alice")
    return ledger


# ── Ledger-level idempotency (the durable backstop) ─────────────────────────────

@pytest.mark.asyncio
async def test_credit_with_idempotency_key_is_no_op_on_replay():
    ledger = await _fresh_ledger()
    key = "bridge-deposit:0xrecipient:0xtxhash"
    t1 = await ledger.credit(
        wallet_id="alice", amount=1.0,
        tx_type=TransactionType.BRIDGE_DEPOSIT, idempotency_key=key)
    # replay the SAME deposit (crash-window / catch-up re-scan)
    t2 = await ledger.credit(
        wallet_id="alice", amount=1.0,
        tx_type=TransactionType.BRIDGE_DEPOSIT, idempotency_key=key)
    assert t1.tx_id == t2.tx_id              # same deterministic tx, not a new one
    assert await ledger.get_balance("alice") == 1.0  # credited ONCE, not twice


@pytest.mark.asyncio
async def test_credit_without_idempotency_key_unchanged():
    """Default behavior must be untouched: two un-keyed credits = two txs."""
    ledger = await _fresh_ledger()
    await ledger.credit(wallet_id="alice", amount=1.0, tx_type=TransactionType.BRIDGE_DEPOSIT)
    await ledger.credit(wallet_id="alice", amount=1.0, tx_type=TransactionType.BRIDGE_DEPOSIT)
    assert await ledger.get_balance("alice") == 2.0


@pytest.mark.asyncio
async def test_distinct_deposits_each_credit():
    ledger = await _fresh_ledger()
    await ledger.credit(wallet_id="alice", amount=1.0,
                        tx_type=TransactionType.BRIDGE_DEPOSIT, idempotency_key="dep:a")
    await ledger.credit(wallet_id="alice", amount=2.0,
                        tx_type=TransactionType.BRIDGE_DEPOSIT, idempotency_key="dep:b")
    assert await ledger.get_balance("alice") == 3.0


# ── End-to-end: the monitor's crash-window double-credit is now impossible ───────

class _StubLedger:
    """Minimal `self.ledger` for InboundMonitor — only `_connected_address`
    is read by `_credit_deposit`."""
    def __init__(self, recipient):
        self.node_id = "stub"
        self._connected_address = recipient


@pytest.mark.asyncio
async def test_monitor_credit_deposit_idempotent_even_with_no_checkpoint_store():
    """The F1 scenario reduced to its core: with NO checkpoint store (so the ONLY
    backstop is the ledger) and a cleared in-memory dedup set (a restart), crediting
    the SAME transfer twice must credit exactly once."""
    from prsm.economy.ftns_onchain import InboundMonitor

    sender = "0xCba5b409a504480d4C969a47FC74cd8c109F8B15"
    recipient = "0x4acdE458766C704B2511583572303e77109cFFE8"
    tx_hash = "0x0a0d63e6f63515a3bf94af4b1e506be58c4ced357e6ca453466534984a57bc84"

    ledger = await _fresh_ledger()
    await ledger.link_eth_address("alice", sender)

    monitor = InboundMonitor(
        ledger=_StubLedger(recipient),
        local_ledger=ledger,
        checkpoint_store=None,  # no restart-dedup → ledger is the only line of defense
    )
    transfer = {"from_address": sender, "amount_ftns": 1.0, "tx_hash": tx_hash}

    await monitor._credit_deposit(transfer)
    assert await ledger.get_balance("alice") == 1.0

    # simulate restart: in-memory dedup set is empty again, same tx re-presented
    monitor._credited_tx_hashes.clear()
    await monitor._credit_deposit(transfer)

    assert await ledger.get_balance("alice") == 1.0, "crash-window double-credit!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
