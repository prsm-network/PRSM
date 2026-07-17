"""Sprint 1473 — bridge-deposit dedup must distinguish distinct Transfer logs within one tx.

Bridge-deposit audit (wf_1160e260) MEDIUM: all three dedup layers (in-memory _credited_tx_hashes,
the checkpoint_store credited_deposits table, and the ledger idempotency_key) keyed on
(recipient, tx_hash) with NO log_index. A single tx that emits MULTIPLE Transfers to the recipient
(a multisend/disperse/router, or two distinct linked senders in one batched tx) yields N transfers
sharing tx_hash — the first credits, transfers 2..N short-circuit at the dedup guard → the 2nd+
deposits are silently DROPPED (depositor fund-loss; the operator holds the full on-chain sum).

Fix: fold log_index into the dedup key (a composite `tx_hash:log_index` string — no schema change),
so 'same tx re-presented' still dedups but 'distinct log within the same tx' credits.
"""
from __future__ import annotations

import pytest

from prsm.economy.ftns_onchain import InboundMonitor
from prsm.node.local_ledger import LocalLedger

pytestmark = pytest.mark.asyncio


class _StubLedger:
    def __init__(self, recipient):
        self.node_id = "stub"
        self._connected_address = recipient


async def _monitor_and_ledger():
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    monitor = InboundMonitor(
        ledger=_StubLedger("0x" + "cc" * 20), local_ledger=ledger, checkpoint_store=None)
    return monitor, ledger


async def test_two_transfers_same_tx_distinct_log_index_both_credit():
    monitor, ledger = await _monitor_and_ledger()
    sender_a, sender_b = "0x" + "a1" * 20, "0x" + "b2" * 20
    await ledger.link_eth_address("walletA", sender_a)
    await ledger.link_eth_address("walletB", sender_b)
    tx = "0x" + "de" * 32

    # ONE tx emits two Transfers to the recipient (a batched/multisend deposit) — distinct log_index.
    await monitor._credit_deposit(
        {"from_address": sender_a, "amount_ftns": 10.0, "tx_hash": tx, "log_index": 0})
    await monitor._credit_deposit(
        {"from_address": sender_b, "amount_ftns": 20.0, "tx_hash": tx, "log_index": 1})

    assert await ledger.get_balance("walletA") == 10.0     # first Transfer
    assert await ledger.get_balance("walletB") == 20.0     # ★ 2nd Transfer NOT dropped


async def test_same_tx_same_log_index_still_dedups():
    # A re-presented (tx_hash, log_index) — the restart catch-up case — must remain exactly-once.
    monitor, ledger = await _monitor_and_ledger()
    sender = "0x" + "a1" * 20
    await ledger.link_eth_address("walletA", sender)
    tx = "0x" + "de" * 32
    transfer = {"from_address": sender, "amount_ftns": 5.0, "tx_hash": tx, "log_index": 0}

    await monitor._credit_deposit(transfer)
    monitor._credited_tx_hashes.clear()                    # simulate restart (in-memory dedup lost)
    await monitor._credit_deposit(transfer)                # same (tx, log_index) → no-op

    assert await ledger.get_balance("walletA") == 5.0      # credited exactly once


async def test_legacy_transfer_without_log_index_still_credits_once():
    # Back-compat: a transfer dict with no log_index (default 0) still credits + dedups.
    monitor, ledger = await _monitor_and_ledger()
    sender = "0x" + "a1" * 20
    await ledger.link_eth_address("walletA", sender)
    tx = "0x" + "de" * 32
    transfer = {"from_address": sender, "amount_ftns": 3.0, "tx_hash": tx}   # no log_index key

    await monitor._credit_deposit(transfer)
    await monitor._credit_deposit(transfer)
    assert await ledger.get_balance("walletA") == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
