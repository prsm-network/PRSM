"""Sprint 1476 — BatchSettlementManager durable persistence (audit wf_530e5cd6 #5).

The batch-settlement audit's deferred MEDIUM: the queue, in-flight tracker, and
dedup set were in-memory ONLY, so a process restart (i.e. every deploy) silently
dropped owed on-chain payouts, voiding the sp914/sp917/sp1475 re-queue safety net.

sp1476 persists all three durably (atomic JSON) and restores them on construction:
  - a queued-but-unbroadcast payout survives restart → retried, not lost;
  - a broadcast-but-unconfirmed (in-flight) payout survives → the reconciler
    resolves it (confirmed→drop, reverted/dead→re-queue), never a silent drop;
  - the dedup set survives → an already-settled tx_id is NOT re-enqueued →
    re-broadcast → double-pay;
  - the flush path persists the queue-CLEAR BEFORE broadcasting, so a crash
    mid-flush cannot re-broadcast a transfer that already went out (favours a
    recoverable loss over an irreversible double-pay).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prsm.economy.batch_settlement import (
    BatchSettlementManager,
    PendingTransfer,
    SettlementMode,
)

pytestmark = pytest.mark.asyncio

_FROM = "0x" + "a" * 40
_TO = "0x" + "b" * 40


class _FakeLedger:
    """transfer() returns the configured outcome kind."""
    def __init__(self, kind="pending"):
        self._is_initialized = True
        self.kind = kind
        self._connected_address = _FROM
        self.calls = 0

    async def transfer(self, *, job_id, to_address, amount_ftns):
        self.calls += 1
        if self.kind == "none":
            return None
        return SimpleNamespace(status=self.kind, tx_hash="0x" + "f" * 64,
                               block_number=1, nonce=5)


def _mgr(persist_dir, kind="pending"):
    return BatchSettlementManager(
        ftns_ledger=_FakeLedger(kind), node_id="n1", connected_address=_FROM,
        mode=SettlementMode.MANUAL, persist_dir=str(persist_dir),
    )


def _txn(tx_id="t1", amount=5.0, to=_TO):
    return SimpleNamespace(tx_id=tx_id, from_wallet=_FROM, to_wallet=to,
                           amount=amount, description="d")


# ───────────────────── restore queued payout across restart ─────────────────

async def test_queued_payout_survives_restart(tmp_path):
    mgr1 = _mgr(tmp_path)
    assert await mgr1.enqueue(_txn(tx_id="t1", amount=7.0)) is True
    assert len(mgr1._queue) == 1
    # A fresh manager (simulating a process restart / redeploy) restores it.
    mgr2 = _mgr(tmp_path)
    assert len(mgr2._queue) == 1
    assert mgr2._queue[0].to_wallet == _TO
    assert mgr2._queue[0].amount == 7.0


async def test_in_memory_when_no_persist_dir(tmp_path):
    # persist_dir "" / ":memory:" → no file, no cross-instance restore.
    mgr1 = _mgr(":memory:")
    await mgr1.enqueue(_txn())
    assert mgr1._persist_path is None
    mgr2 = _mgr(":memory:")
    assert mgr2._queue == []


# ───────────────────── dedup survives restart (no double-pay) ───────────────

async def test_settled_id_survives_restart_prevents_reenqueue(tmp_path):
    """★ An already-queued/settled tx_id must remain deduped across a restart,
    else a re-enqueue of the same logical payment re-broadcasts → double-pay."""
    mgr1 = _mgr(tmp_path)
    await mgr1.enqueue(_txn(tx_id="pay-1", amount=3.0))
    mgr1._queue.clear()          # simulate it having been settled + cleared
    mgr1._persist()
    mgr2 = _mgr(tmp_path)
    assert "pay-1" in mgr2._settled_ids
    # Re-presenting the SAME tx_id after restart is deduped (not re-queued).
    assert await mgr2.enqueue(_txn(tx_id="pay-1", amount=3.0)) is False
    assert mgr2._queue == []


# ───────────────────── flush persists CLEAR before broadcast ────────────────

async def test_flush_pending_persists_in_flight_not_queue(tmp_path):
    """★ After a flush that broadcasts a pending tx, a restart restores the tx in
    the IN-FLIGHT tracker (for reconciliation) but NOT the queue — so it is never
    re-broadcast (double-pay), yet never silently dropped."""
    mgr1 = _mgr(tmp_path, kind="pending")
    mgr1._queue.append(PendingTransfer(tx_id="t1", from_wallet=_FROM,
                                       to_wallet=_TO, amount=5.0, job_id="j1"))
    await mgr1.flush()
    assert mgr1._queue == [] and len(mgr1._in_flight) == 1
    mgr2 = _mgr(tmp_path)
    assert mgr2._queue == [], "a broadcast payout must NOT be re-queued (double-pay)"
    assert len(mgr2._in_flight) == 1
    assert mgr2._in_flight[0]["tx_hash"] == "0x" + "f" * 64
    assert mgr2._in_flight[0]["nonce"] == 5


async def test_flush_never_broadcast_requeue_survives_restart(tmp_path):
    """A never-broadcast (transfer None) payout is re-queued AND persisted → a
    restart still owes + retries it."""
    mgr1 = _mgr(tmp_path, kind="none")
    mgr1._queue.append(PendingTransfer(tx_id="t1", from_wallet=_FROM,
                                       to_wallet=_TO, amount=5.0, job_id="j1"))
    await mgr1.flush()
    assert len(mgr1._queue) == 1          # re-queued
    mgr2 = _mgr(tmp_path)
    assert len(mgr2._queue) == 1          # survived restart
    assert mgr2._queue[0].amount == 5.0


# ───────────────────── bounded dedup (no unbounded growth) ──────────────────

async def test_settled_ids_bounded_fifo_eviction(tmp_path):
    mgr = _mgr(tmp_path)
    mgr._max_settled_ids = 3
    for i in range(5):
        mgr._mark_settled(f"tx{i}")
    assert len(mgr._settled_ids) == 3
    assert "tx0" not in mgr._settled_ids and "tx1" not in mgr._settled_ids
    assert "tx4" in mgr._settled_ids     # newest kept


async def test_requeue_persisted_incrementally_within_flush(tmp_path):
    """★ persistence-review fix — a never-broadcast re-queue is persisted
    IMMEDIATELY (durable self._queue + persist), not accumulated in a loop-local
    list flushed only after the whole broadcast loop. So a crash mid-flush cannot
    silently lose the owed payout. We assert net#1's re-queue is already durable
    at the moment net#2 is being broadcast (i.e. before flush() returns)."""
    addr_a, addr_b = "0x" + "a1" * 20, "0x" + "b2" * 20
    seen = {}

    class _MidLoopLedger:
        def __init__(self):
            self._is_initialized = True
            self._connected_address = _FROM
            self.n = 0

        async def transfer(self, *, job_id, to_address, amount_ftns):
            self.n += 1
            if self.n == 2:
                # Simulate a restart mid-loop: a fresh manager from the same dir
                # must ALREADY see net#1's re-queued owed payout.
                probe = _mgr(tmp_path, kind="none")
                seen["queue_len_midloop"] = len(probe._queue)
            return None   # both never-broadcast → both re-queued

    mgr = BatchSettlementManager(
        ftns_ledger=_MidLoopLedger(), node_id="n1", connected_address=_FROM,
        mode=SettlementMode.MANUAL, persist_dir=str(tmp_path))
    mgr._queue.append(PendingTransfer(tx_id="t1", from_wallet=_FROM,
                                      to_wallet=addr_a, amount=3.0, job_id="j1"))
    mgr._queue.append(PendingTransfer(tx_id="t2", from_wallet=_FROM,
                                      to_wallet=addr_b, amount=2.0, job_id="j2"))
    await mgr.flush()
    assert seen["queue_len_midloop"] >= 1, \
        "net#1's re-queue must be durable BEFORE the flush loop finishes"
    # After the full flush both are durably re-queued.
    final = _mgr(tmp_path)
    assert len(final._queue) == 2


async def test_corrupt_persist_file_starts_empty(tmp_path):
    (tmp_path / "batch_settlement.json").write_text("{ not json")
    mgr = _mgr(tmp_path)   # must not raise
    assert mgr._queue == [] and mgr._in_flight == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
