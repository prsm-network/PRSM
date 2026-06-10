"""Sprint 1053 — settlement client concurrency lock (closes deferred review sp1039 #8).

On the live rail, accumulate() runs from the /compute/inference settle path while the
background poll loop runs commit/recover/finalize on the SAME client. They mutate
shared state (_accumulator / _tracked / _committing / _pending_commits) across await
points, so a concurrent accumulate during a commit's network await could interleave
and drop the freshly-added receipt (the committed PendingBatch is popped, taking the
concurrently-added receipt with it). A coarse asyncio.Lock serializes the
state-mutating methods. accumulate is fail-open + runs AFTER the off-chain release,
so blocking it for a poll cycle is acceptable.
"""
from __future__ import annotations

import asyncio
import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import (
    AccumulatorConfig, BatchedReceipt, ReceiptAccumulator)
from prsm.settlement.client import BatchSettlementClient

ONE_FTNS = 10**18
PROVIDER = "0x" + "b" * 40
REQUESTER = "0x" + "a" * 40


def _batched(i=0):
    r = ShardExecutionReceipt(
        job_id=f"job-{i}", shard_index=i, provider_id="p",
        provider_pubkey_b64=b64encode(b"pk").decode(),
        output_hash=hashlib.sha256(f"o{i}".encode()).hexdigest(),
        executed_at_unix=1700000000 + i, signature=b64encode(b"s").decode())
    return BatchedReceipt(receipt=r, requester_address=REQUESTER,
                          provider_address=PROVIDER, value_ftns=ONE_FTNS,
                          local_escrow_id=f"esc-{i}")


def test_concurrent_accumulate_during_commit_not_dropped():
    """A receipt accumulated WHILE a commit is mid-flight (holding the lock) must
    not be lost: with the lock, the accumulate runs only AFTER the commit pops its
    batch, so the new receipt lands in a fresh PendingBatch."""
    async def scenario():
        started = asyncio.Event()
        proceed = asyncio.Event()

        contract = AsyncMock()
        contract.address = PROVIDER
        contract.is_finalizable.return_value = False

        async def slow_commit(**kw):
            started.set()
            await proceed.wait()                 # hold the commit open mid-await
            return (hashlib.sha256(b"A").digest(), 1700000000)
        contract.commit_batch.side_effect = slow_commit

        c = BatchSettlementClient(
            ReceiptAccumulator(AccumulatorConfig(count_threshold=1)), contract, PROVIDER)
        await c.accumulate(_batched(0))           # batch A ready (key = requester,provider)

        commit_task = asyncio.create_task(c.commit_ready_batches())
        await started.wait()                      # commit in-flight, holding the lock
        acc_task = asyncio.create_task(c.accumulate(_batched(1)))  # same key, concurrent
        await asyncio.sleep(0.01)                 # let acc_task try to run (it must block)
        # with the lock, acc_task is still waiting; A's batch not yet contaminated
        proceed.set()                             # finish the commit
        await commit_task
        await acc_task

        # A committed + popped; B accumulated AFTER → survives in a fresh batch.
        assert c.status()["tracked_batches"] == 1
        assert c._accumulator.total_receipt_count() == 1   # B not dropped
        return True

    assert asyncio.run(scenario()) is True


def test_lock_does_not_break_sequential_flow():
    """The lock must not deadlock the normal sequential accumulate→commit path."""
    async def scenario():
        contract = AsyncMock()
        contract.address = PROVIDER
        contract.is_finalizable.return_value = False
        contract.commit_batch.side_effect = lambda **kw: (
            hashlib.sha256(b"x").digest(), 1)
        c = BatchSettlementClient(
            ReceiptAccumulator(AccumulatorConfig(count_threshold=1)), contract, PROVIDER)
        await c.accumulate(_batched(0))
        out = await c.commit_ready_batches()
        await c.reconcile_finalized()
        return len(out)
    assert asyncio.run(scenario()) == 1


def test_client_has_async_lock():
    contract = AsyncMock()
    contract.address = PROVIDER
    c = BatchSettlementClient(ReceiptAccumulator(), contract, PROVIDER)
    assert isinstance(c._lock, asyncio.Lock)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
