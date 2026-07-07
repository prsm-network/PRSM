"""Sprint 1407 — force-flush commit for a ceremony/canary: commit a single small batch now.

The accumulator only marks a batch READY at count(1000)/time(1h)/value(100 FTNS). A 1-job, 1-FTNS
canary hits none, so commit_ready_batches returned []. `force` evaluates readiness as-of a far-future
time so the TIME trigger fires for every pending batch — the controlled one-shot the ceremony uses.
"""
from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import AccumulatorConfig, BatchedReceipt, ReceiptAccumulator


def _receipt():
    return ShardExecutionReceipt(
        job_id="j1", shard_index=0, provider_id="p" * 32, provider_pubkey_b64="PUB",
        output_hash="abc", executed_at_unix=1, signature="SIG")


def test_far_future_time_forces_readiness():
    acc = ReceiptAccumulator(AccumulatorConfig())   # defaults 1000 / 3600s / 100 FTNS
    acc.add(BatchedReceipt(
        receipt=_receipt(), requester_address="0xR", provider_address="0xP",
        value_ftns=10 ** 18, local_escrow_id="job-j1"), at_unix=1000)
    # a small batch just after add is NOT ready under any threshold:
    assert acc.ready_batches(at_unix=1005) == []
    # evaluated far in the future → the TIME trigger fires → ready (this is what force=True does):
    ready = acc.ready_batches(at_unix=2 ** 62)
    assert len(ready) == 1
    assert ready[0].trigger.value == "time"
    assert ready[0].batch.total_value_ftns == 10 ** 18


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
