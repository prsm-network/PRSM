"""Sprint 1022 — settlement commit must never double-settle on a pending tx.

(Tier-1 audit follow-up; the gating should-fix from the sp1021 Web3SettlementContractClient
adversarial review.)

A commitBatch that is broadcast but whose receipt times out raises OnChainPendingError
— the tx MAY still mine. The on-chain batchId is derived from a monotonic per-provider
counter + block.number with NO content/merkle-root dedup (verified against
BatchSettlementRegistry.sol), so blindly retrying the same receipts mints a SECOND
distinct batch that settles FTNS again → double payment. commit_ready_batches
previously caught OnChainPendingError under its generic safe-to-retry handler.

Fix: distinguish a broadcast-but-unconfirmed failure (an exception carrying .tx_hash —
the documented OnChainPendingError contract) from safe-to-retry errors
(BroadcastFailedError / OnChainRevertedError, which carry no tx_hash). On the former,
QUARANTINE the batch (remove it from the accumulator so it can't be re-committed) and
record it for reconcile_pending_commits(), which adopts the batch into the tracked set
if the original tx actually landed.
"""
from __future__ import annotations

import asyncio
import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import (
    AccumulatorConfig,
    BatchedReceipt,
    ReceiptAccumulator,
)
from prsm.settlement.client import BatchSettlementClient
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
)

ONE_FTNS = 10 ** 18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40


def _make_batched(i: int = 0) -> BatchedReceipt:
    receipt = ShardExecutionReceipt(
        job_id=f"job-{i}", shard_index=i, provider_id="pid",
        provider_pubkey_b64=b64encode(b"pk").decode(),
        output_hash=hashlib.sha256(f"o{i}".encode()).hexdigest(),
        executed_at_unix=1_700_000_000 + i, signature=b64encode(b"sig").decode(),
    )
    return BatchedReceipt(
        receipt=receipt, requester_address=REQUESTER_A,
        provider_address=PROVIDER_ADDR, value_ftns=ONE_FTNS, local_escrow_id=f"e{i}",
    )


def _run(coro):
    return asyncio.run(coro)


def _client(commit_side_effect, *, get_committed_for_tx=None):
    # count_threshold=1 → a single receipt becomes a ready batch immediately.
    cfg = AccumulatorConfig(count_threshold=1, time_threshold_seconds=3600, value_threshold_ftns=10 ** 30)
    acc = ReceiptAccumulator(cfg)
    contract = AsyncMock()
    contract.commit_batch.side_effect = commit_side_effect
    if get_committed_for_tx is not None:
        contract.get_committed_batch_for_tx.side_effect = get_committed_for_tx
    client = BatchSettlementClient(acc, contract, PROVIDER_ADDR)
    return acc, contract, client


def test_pending_commit_quarantined_not_retried():
    async def commit_pending(**kw):
        raise OnChainPendingError("receipt unknown", tx_hash="0xPENDINGTX")

    acc, contract, client = _client(commit_pending)
    _run(client.accumulate(_make_batched(0)))

    out = _run(client.commit_ready_batches())
    assert out == []  # nothing successfully committed
    # The batch was POPPED from the accumulator (NOT retained) → it can never be
    # re-committed, which is what prevents the double settlement.
    assert acc.total_receipt_count() == 0
    pend = client.pending_commits()
    assert len(pend) == 1
    assert pend[0].tx_hash == "0xPENDINGTX"

    # A second poll must NOT attempt commit_batch again (no double-commit).
    contract.commit_batch.reset_mock()
    contract.commit_batch.side_effect = commit_pending
    _run(client.commit_ready_batches())
    contract.commit_batch.assert_not_called()


def test_safe_to_retry_errors_still_retained_and_retried():
    calls = {"n": 0}

    async def commit_broadcast_fail(**kw):
        calls["n"] += 1
        raise BroadcastFailedError("network down")  # no tx_hash → safe to retry

    acc, contract, client = _client(commit_broadcast_fail)
    _run(client.accumulate(_make_batched(0)))

    _run(client.commit_ready_batches())
    assert acc.total_receipt_count() == 1  # retained (chain saw nothing)
    assert client.pending_commits() == []  # not quarantined

    _run(client.commit_ready_batches())  # retries
    assert calls["n"] == 2


def test_reconcile_adopts_landed_pending_batch():
    landed_id = b"\x0a" * 32

    async def commit_pending(**kw):
        raise OnChainPendingError("unknown", tx_hash="0xLANDED")

    async def get_for_tx(tx_hash):
        return (landed_id, 1_700_001_234) if tx_hash == "0xLANDED" else None

    acc, contract, client = _client(commit_pending, get_committed_for_tx=get_for_tx)
    _run(client.accumulate(_make_batched(0)))
    _run(client.commit_ready_batches())
    assert len(client.pending_commits()) == 1

    n = _run(client.reconcile_pending_commits())
    assert n == 1
    assert client.pending_commits() == []
    adopted = client.get_tracked(landed_id)
    assert adopted is not None
    assert adopted.batch_id == landed_id
    assert adopted.commit_timestamp == 1_700_001_234
    assert adopted.tx_hash == "0xLANDED"


def test_reconcile_leaves_unlanded_pending_quarantined():
    async def commit_pending(**kw):
        raise OnChainPendingError("unknown", tx_hash="0xMAYBE")

    async def get_for_tx(tx_hash):
        return None  # tx not (yet) landed — pending or dropped: stay conservative

    acc, contract, client = _client(commit_pending, get_committed_for_tx=get_for_tx)
    _run(client.accumulate(_make_batched(0)))
    _run(client.commit_ready_batches())

    n = _run(client.reconcile_pending_commits())
    assert n == 0
    assert len(client.pending_commits()) == 1  # still quarantined, never auto-re-committed
    assert client.tracked_batches() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
