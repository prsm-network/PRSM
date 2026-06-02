"""Sprint 930 — batch-settlement concurrency hardening (money-rail review).

Two confirmed findings in BatchSettlementManager:

1. DOUBLE-REQUEUE / DOUBLE-PAY (critical). flush() calls reconcile_in_flight()
   at its start (line 239), OUTSIDE self._lock, and a threshold-triggered
   flush (asyncio.create_task in enqueue) can run concurrently with the
   periodic flush. Two concurrent reconciles both iterate the SAME _in_flight
   list, both poll the same reverted tx's receipt (the poll awaits, so both are
   in-flight at once), and both re-queue the owed payout — so the next flush
   broadcasts it TWICE → double-pay. (Note: a concurrent _track_in_flight APPEND
   during a single reconcile is NOT lost — Python's index-based list iteration
   visits it — so the real bug is concurrent reconciles, not the append.)
   Fix: a dedicated reconcile lock serializes reconcile_in_flight against itself;
   the second reconcile then snapshots an _in_flight with the entry already
   resolved, so it cannot re-queue it again.

2. JOB-ID COLLISION (audit corruption). Settlement job_ids were
   f"batch-{int(time.time())}" — whole-second precision — so two net transfers
   in the same flush second collided, and the on-chain audit table (job_id
   PRIMARY KEY, INSERT OR REPLACE) silently overwrote the first row, losing its
   tx_hash/amount and corrupting reconciliation. Fix: per-transfer uuid job_ids.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from prsm.economy.batch_settlement import (
    BatchSettlementManager,
    PendingTransfer,
    SettlementMode,
)

_FROM = "0x" + "a" * 40
_TO = "0x" + "b" * 40
_TO2 = "0x" + "c" * 40


class _GatedEth:
    """Receipt source whose poll optionally blocks on a gate, so two concurrent
    reconciles can be held mid-poll to expose the double-requeue race."""

    def __init__(self, receipts, gate=None):
        self._receipts = receipts
        self._gate = gate

    def get_transaction_receipt(self, h):
        if self._gate is not None:
            self._gate.wait(timeout=5)
        r = self._receipts.get(h)
        if isinstance(r, Exception):
            raise r
        return r


def _mgr(receipts=None, *, transfer=None, gate=None):
    led = MagicMock()
    led.w3 = SimpleNamespace(eth=_GatedEth(receipts or {}, gate))
    if transfer is not None:
        led.transfer = transfer
    return BatchSettlementManager(
        ftns_ledger=led, node_id="n1", connected_address=_FROM,
        mode=SettlementMode.MANUAL,
    )


@pytest.mark.asyncio
async def test_concurrent_reconciles_do_not_double_requeue():
    # A reverted in-flight tx, reconciled by TWO concurrent flushes. It must be
    # re-queued exactly ONCE — re-queuing it twice double-pays the payee.
    gate = threading.Event()
    mgr = _mgr({"0xrev": {"status": 0}}, gate=gate)
    mgr._track_in_flight(_FROM, _TO, 5.0, "0xrev")

    t1 = asyncio.create_task(mgr.reconcile_in_flight())
    t2 = asyncio.create_task(mgr.reconcile_in_flight())
    await asyncio.sleep(0.05)   # let both reach the (gated) receipt poll
    gate.set()
    await asyncio.gather(t1, t2)

    requeued = [p for p in mgr._queue if p.to_wallet == _TO and p.amount == 5.0]
    assert len(requeued) == 1, f"reverted tx re-queued {len(requeued)}x (double-pay)"
    assert mgr._in_flight == []


@pytest.mark.asyncio
async def test_concurrent_track_during_reconcile_is_preserved():
    # Defense-in-depth: an entry tracked by a concurrent flush mid-reconcile
    # must survive the reconcile's in-flight rebuild.
    gate = threading.Event()
    mgr = _mgr({"0xA": None}, gate=gate)   # A: no receipt yet → still pending
    mgr._track_in_flight(_FROM, _TO, 5.0, "0xA")

    task = asyncio.create_task(mgr.reconcile_in_flight())
    await asyncio.sleep(0.05)              # reconcile now blocked polling A
    mgr._track_in_flight(_FROM, _TO2, 9.0, "0xB")   # appended mid-reconcile
    gate.set()
    await task

    hashes = {e["tx_hash"] for e in mgr._in_flight}
    assert "0xB" in hashes, "concurrently-tracked entry was dropped"
    assert "0xA" in hashes


@pytest.mark.asyncio
async def test_batch_transfers_get_distinct_job_ids():
    # Two net transfers in one flush (same wall-clock second) must get DISTINCT
    # job_ids, or the on-chain audit row (job_id PK, INSERT OR REPLACE) silently
    # overwrites the first.
    seen = []

    async def fake_transfer(*, job_id, to_address, amount_ftns):
        seen.append(job_id)
        return SimpleNamespace(status="confirmed", tx_hash="0x" + "d" * 64, block_number=1)

    mgr = _mgr(transfer=fake_transfer)
    mgr._queue.append(PendingTransfer(
        tx_id="t1", from_wallet=_FROM, to_wallet=_TO, amount=5.0, job_id="j1"))
    mgr._queue.append(PendingTransfer(
        tx_id="t2", from_wallet=_FROM, to_wallet=_TO2, amount=7.0, job_id="j2"))

    await mgr.flush()

    assert len(seen) == 2
    assert len(set(seen)) == 2, f"settlement job_ids collided: {seen}"


@pytest.mark.asyncio
async def test_single_reverted_reconcile_still_requeues_once():
    # Sanity: the reconcile lock must not break the normal single-reconcile path.
    mgr = _mgr({"0xrev": {"status": 0}})
    mgr._track_in_flight(_FROM, _TO, 4.0, "0xrev")
    out = await mgr.reconcile_in_flight()
    assert out["reverted"] == 1
    assert sum(1 for p in mgr._queue if p.to_wallet == _TO) == 1
    assert mgr._in_flight == []
