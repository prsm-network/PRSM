"""Sprint 1051 — settlement-rail observability (closes deferred review items
sp1039 #9 settlement-off-reason + sp1040 #13 stale-intent surface).

The on-chain settlement rail is now live on mainnet (a committed batch sits PENDING
for the 3-day window), so an operator needs read-only visibility into its lifecycle
state: is it enabled + write-capable, how many batches are tracked/finalized, how
many commits are quarantined (broadcast-but-unconfirmed) or orphaned (commit-intent
WAL entries awaiting chain-scan recovery), and whether durable state is on. A
non-zero pending/committing count is the operator's signal that funds may be
mid-flight and need attention.
"""
from __future__ import annotations

import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock, MagicMock

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import (
    AccumulatorConfig, BatchedReceipt, ReceiptAccumulator)
from prsm.settlement.client import BatchSettlementClient

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


def _contract(with_address="0x" + "b" * 40):
    m = AsyncMock()
    n = {"i": 0}

    async def commit(**kw):
        n["i"] += 1
        return (hashlib.sha256(f"b{n['i']}".encode()).digest(), 1700000000 + n["i"])
    m.commit_batch.side_effect = commit
    m.is_finalizable.return_value = False
    # .address is a plain attribute (write-capable iff a key is configured)
    m.address = with_address
    return m


def test_status_reports_lifecycle_counts():
    client = BatchSettlementClient(
        ReceiptAccumulator(AccumulatorConfig(count_threshold=1)), _contract(), PROVIDER)
    _run(client.accumulate(_batched(0)))
    _run(client.commit_ready_batches())   # → 1 tracked
    st = client.status()
    assert st["provider_address"] == PROVIDER
    assert st["tracked_batches"] == 1
    assert st["pending_commits"] == 0
    assert st["committing_intents"] == 0
    assert st["finalized_locally"] == 0
    assert st["write_capable"] is True


def test_status_write_capable_false_without_key():
    client = BatchSettlementClient(
        ReceiptAccumulator(), _contract(with_address=None), PROVIDER)
    assert client.status()["write_capable"] is False


def test_status_durable_state_flag(tmp_path):
    from prsm.settlement.state_store import SettlementStateStore
    store = SettlementStateStore(tmp_path / "s.json")
    c1 = BatchSettlementClient(ReceiptAccumulator(), _contract(), PROVIDER)
    c2 = BatchSettlementClient(ReceiptAccumulator(), _contract(), PROVIDER,
                               state_store=store)
    assert c1.status()["durable_state"] is False
    assert c2.status()["durable_state"] is True


def test_status_surfaces_pending_and_intents():
    """Quarantined (pending) + orphaned (committing-intent) commits are surfaced —
    the operator's stranded-funds signal."""
    from prsm.settlement.client import PendingCommit, CommitIntent
    from prsm.settlement.accumulator import TriggerReason
    client = BatchSettlementClient(ReceiptAccumulator(), _contract(), PROVIDER)
    client._pending_commits["0xtx"] = PendingCommit(
        accumulator_key=(REQUESTER, PROVIDER, b"\x00" * 32, 0), tx_hash="0xtx",
        merkle_root=b"\x01" * 32, leaf_hashes=(b"\x02" * 32,), receipt_count=1,
        total_value_ftns=ONE_FTNS, provider_address=PROVIDER,
        requester_address=REQUESTER, trigger_reason=TriggerReason.COUNT)
    client._committing["root|x"] = CommitIntent(
        accumulator_key=(REQUESTER, PROVIDER, b"\x00" * 32, 0),
        merkle_root=b"\x03" * 32, leaf_hashes=(b"\x04" * 32,), receipt_count=1,
        total_value_ftns=ONE_FTNS, provider_address=PROVIDER,
        requester_address=REQUESTER, trigger_reason=TriggerReason.COUNT)
    st = client.status()
    assert st["pending_commits"] == 1
    assert st["committing_intents"] == 1
    assert st["funds_in_flight"] is True   # pending or intents non-zero


def test_get_settlement_status_disabled_when_off():
    from prsm.settlement.client_wiring import get_settlement_status
    st = get_settlement_status(None)
    assert st["enabled"] is False
    assert "reason" in st


def test_get_settlement_status_enabled_wraps_client():
    from prsm.settlement.client_wiring import get_settlement_status
    client = BatchSettlementClient(ReceiptAccumulator(), _contract(), PROVIDER)
    st = get_settlement_status(client)
    assert st["enabled"] is True
    assert st["tracked_batches"] == 0
    assert st["write_capable"] is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
