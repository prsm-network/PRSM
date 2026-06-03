"""Sprint 955 — bounded node shutdown drain.

PRSMNode.stop() awaits its daemon tasks to wind down. The task-await loop used
`asyncio.wait_for(task, timeout=5.0)`, which on timeout CANCELS the task and then
AWAITS the cancellation to complete — so a task stuck in an uncancellable await
(e.g. a blocking executor RPC, or a libp2p subprocess teardown) makes wait_for
itself hang, and node shutdown never returns. (Surfaced while probing the live-P2P
e2e suite: node.stop() parked awaiting PendingWithdrawReconciler.run_forever.)

Same class as sp953's transport fix, one layer up. The robust primitive is
`_drain_task_bounded`: wait up to the timeout via `asyncio.wait` (which returns at
the deadline regardless), and on timeout request-cancel but ABANDON the task
(never await the cancellation), guaranteeing a bounded return.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from prsm.node.node import _drain_task_bounded


@pytest.mark.asyncio
async def test_completed_task_returns_true_fast():
    async def quick():
        return 42
    task = asyncio.ensure_future(quick())
    t0 = time.monotonic()
    ok = await _drain_task_bounded(task, timeout=2.0, name="quick")
    assert ok is True
    assert time.monotonic() - t0 < 1.0  # returned ~immediately, not at timeout


@pytest.mark.asyncio
async def test_hanging_task_is_bounded_and_abandoned():
    """A task that will not finish (and resists cancellation) must NOT hang the
    drain — it returns False within ~timeout."""
    started = asyncio.Event()

    async def stubborn():
        # Hang on a never-resolving Future (NOT asyncio.sleep — the suite mocks
        # sleep to be instant; a bare Future await is unaffected, so this models
        # a genuinely-stuck task). Swallow the first cancellation and await
        # another never-Future, modelling a task that resists cancellation
        # (e.g. stuck in an uncancellable executor call).
        loop = asyncio.get_event_loop()
        started.set()
        try:
            await loop.create_future()
        except asyncio.CancelledError:
            await loop.create_future()

    task = asyncio.ensure_future(stubborn())
    await started.wait()
    t0 = time.monotonic()
    ok = await _drain_task_bounded(task, timeout=0.3, name="stubborn")
    elapsed = time.monotonic() - t0
    assert ok is False
    assert elapsed < 2.0, f"drain hung {elapsed:.1f}s — not bounded"
    # Clean up the abandoned task so the test loop can close.
    task.cancel()
    try:
        await asyncio.wait({task}, timeout=0.1)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_none_task_is_noop():
    assert await _drain_task_bounded(None, timeout=1.0, name="none") is True


@pytest.mark.asyncio
async def test_task_raising_is_swallowed_returns_true():
    """A task that finishes by raising (non-cancel) is treated as drained — its
    exception is swallowed (shutdown is best-effort), and the drain returns True."""
    async def boom():
        raise RuntimeError("subsystem stop failed")
    task = asyncio.ensure_future(boom())
    ok = await _drain_task_bounded(task, timeout=2.0, name="boom")
    assert ok is True
