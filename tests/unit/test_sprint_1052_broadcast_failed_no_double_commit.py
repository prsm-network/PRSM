"""Sprint 1052 — BroadcastFailed-that-landed double-commit guard (live-rail money safety).

Closes deferred review #4. On the LIVE mainnet rail we hit RPC-response-loss-class
issues (sp1047/1048): commitBatch's send_raw_transaction can THROW
(BroadcastFailedError, no tx_hash) even when the tx actually landed. The prior
behavior dropped the commit-intent and left the receipts in the accumulator for a
blind retry → re-committing the same receipts as a DISTINCT on-chain batchId (the
registry has no content dedup) → DOUBLE-SETTLE of real escrow.

Fix (prefer-no-double-spend, mirrors the OnChainPending quarantine philosophy): on
BroadcastFailed, KEEP the durable commit-intent (root-based recovery marker) + POP
the receipts from the accumulator so they're NEVER blindly re-committed.
recover_committing_intents then scans by root and ADOPTS the batch if the broadcast
actually landed (no double), else the intent stays surfaced (funds_in_flight) for
operator attention. OnChainReverted (mined+atomically-reverted = nothing committed)
still clears the intent + retries — that one is genuinely safe.
"""
from __future__ import annotations

import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import (
    AccumulatorConfig, BatchedReceipt, ReceiptAccumulator)
from prsm.settlement.client import BatchSettlementClient
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError, OnChainRevertedError)

ONE_FTNS = 10**18
PROVIDER = "0x" + "b" * 40
REQUESTER = "0x" + "a" * 40


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _batched(i=0):
    r = ShardExecutionReceipt(
        job_id=f"job-{i}", shard_index=i, provider_id="p",
        provider_pubkey_b64=b64encode(b"pk").decode(),
        output_hash=hashlib.sha256(f"o{i}".encode()).hexdigest(),
        executed_at_unix=1700000000 + i, signature=b64encode(b"s").decode())
    return BatchedReceipt(receipt=r, requester_address=REQUESTER,
                          provider_address=PROVIDER, value_ftns=ONE_FTNS,
                          local_escrow_id=f"esc-{i}")


def _acc():
    return ReceiptAccumulator(AccumulatorConfig(count_threshold=1))


def _contract():
    m = AsyncMock()
    m.address = PROVIDER
    m.is_finalizable.return_value = False
    return m


def test_broadcast_failed_keeps_intent_and_quarantines_receipts():
    """BroadcastFailed → intent RETAINED (recovery marker) + accumulator POPPED
    (no blind re-commit). The committed list is empty."""
    c = BatchSettlementClient(_acc(), _contract(), PROVIDER)
    c._contract.commit_batch.side_effect = BroadcastFailedError("broadcast failed")
    _run(c.accumulate(_batched(0)))
    out = _run(c.commit_ready_batches())
    assert out == []
    assert len(c.committing_intents()) == 1          # intent retained for recovery
    assert c._accumulator.total_receipt_count() == 0  # receipts quarantined (not re-committable)


def test_broadcast_failed_then_re_poll_does_not_double_commit():
    """After a BroadcastFailed, a subsequent commit_ready_batches must NOT re-commit
    (the receipts were quarantined) — so commit_batch is called exactly once."""
    c = BatchSettlementClient(_acc(), _contract(), PROVIDER)
    c._contract.commit_batch.side_effect = BroadcastFailedError("boom")
    _run(c.accumulate(_batched(0)))
    _run(c.commit_ready_batches())
    _run(c.commit_ready_batches())   # second poll
    assert c._contract.commit_batch.await_count == 1   # NOT re-committed


def test_recover_adopts_broadcast_failed_that_landed():
    """recover_committing_intents scans the retained intent by root; if the broadcast
    actually landed, ADOPT it (into _tracked) — single settlement, no double."""
    landed = hashlib.sha256(b"landed").digest()
    contract = _contract()
    contract.commit_batch.side_effect = BroadcastFailedError("boom")
    contract.find_committed_batch_by_root = AsyncMock(return_value=(landed, 1700009999))
    c = BatchSettlementClient(_acc(), contract, PROVIDER)
    _run(c.accumulate(_batched(0)))
    _run(c.commit_ready_batches())                 # → intent retained, receipts quarantined
    assert c.get_tracked(landed) is None
    adopted = _run(c.recover_committing_intents())
    assert adopted == 1
    assert c.get_tracked(landed) is not None        # adopted, not re-committed
    assert c.committing_intents() == []


def test_recover_adopt_pops_accumulator():
    """Defensive: if receipts are still in the accumulator when recover adopts an
    intent, recover pops them (so a later commit can't re-commit the adopted batch)."""
    from prsm.settlement.client import CommitIntent
    from prsm.settlement.accumulator import TriggerReason
    landed = hashlib.sha256(b"x").digest()
    contract = _contract()
    contract.find_committed_batch_by_root = AsyncMock(return_value=(landed, 1))
    c = BatchSettlementClient(_acc(), contract, PROVIDER)
    _run(c.accumulate(_batched(0)))                  # receipts in accumulator
    ready = c._accumulator.ready_batches()[0]
    leaf_hashes, root = c._build_leaves_root(ready)
    c._committing[c._intent_key(ready.key, root)] = c._build_intent(ready, leaf_hashes, root)
    assert c._accumulator.total_receipt_count() == 1
    _run(c.recover_committing_intents())
    assert c.get_tracked(landed) is not None
    assert c._accumulator.total_receipt_count() == 0  # popped on adopt


def test_reverted_clears_intent_and_retries():
    """OnChainReverted (mined + atomically reverted, nothing committed) → clear the
    intent + keep the receipts for a safe retry (distinct from BroadcastFailed)."""
    c = BatchSettlementClient(_acc(), _contract(), PROVIDER)
    c._contract.commit_batch.side_effect = OnChainRevertedError("reverted")
    _run(c.accumulate(_batched(0)))
    _run(c.commit_ready_batches())
    assert c.committing_intents() == []               # intent cleared
    assert c._accumulator.total_receipt_count() == 1   # receipts retained for retry


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
