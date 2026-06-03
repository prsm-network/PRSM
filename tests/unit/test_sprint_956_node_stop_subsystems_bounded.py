"""Sprint 956 — comprehensive bounded node shutdown.

sp955 bounded node.stop()'s daemon-task DRAIN loop. sp956 extends the bound to
the SUBSYSTEM stops (transport, ledger, bittorrent, compute providers, etc.) —
the awaits most likely to hang on an external/uncancellable dependency (a libp2p
subprocess teardown, a stuck SQLite close, a libtorrent shutdown, a chain RPC).
Each is now wrapped in `_await_bounded(coro, timeout, name)`, a thin coroutine
form of `_drain_task_bounded`: run the stop coroutine as a task, drain it up to
the timeout, and ABANDON it (cancel-request only) if it doesn't finish — so no
single subsystem can hang the whole shutdown.

Normal shutdown is unchanged (a subsystem that stops promptly is reaped
immediately). Only a genuinely-stuck subsystem is abandoned (the process is
exiting; an orphaned task is harmless — and an abandoned `ledger.close()` is safe
because SQLite's WAL recovers on the next open).
"""
from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from prsm.node.node import PRSMNode, _await_bounded


@pytest.mark.asyncio
async def test_await_bounded_fast_coro_returns_true():
    async def quick():
        return "ok"
    t0 = time.monotonic()
    ok = await _await_bounded(quick(), timeout=2.0, name="quick")
    assert ok is True
    assert time.monotonic() - t0 < 1.0


@pytest.mark.asyncio
async def test_await_bounded_hanging_coro_is_bounded():
    """A stop coroutine that never finishes (and resists cancellation) must NOT
    hang the drain — returns False within ~timeout."""
    loop = asyncio.get_event_loop()
    started = asyncio.Event()

    async def stuck():
        started.set()
        try:
            await loop.create_future()  # never resolves (not asyncio.sleep — suite mocks it)
        except asyncio.CancelledError:
            await loop.create_future()  # resist cancellation

    coro = stuck()
    t0 = time.monotonic()
    ok = await _await_bounded(coro, timeout=0.3, name="stuck")
    elapsed = time.monotonic() - t0
    assert ok is False
    assert elapsed < 2.0, f"drain hung {elapsed:.1f}s — not bounded"


@pytest.mark.asyncio
async def test_await_bounded_raising_coro_swallowed_true():
    async def boom():
        raise RuntimeError("subsystem stop failed")
    ok = await _await_bounded(boom(), timeout=2.0, name="boom")
    assert ok is True


def test_node_stop_wraps_key_subsystem_stops():
    """Structural pin: the highest hang-risk subsystem stops (transport, ledger,
    bittorrent client) must be bounded via _await_bounded in Node.stop, not bare
    awaits that could hang shutdown indefinitely."""
    src = inspect.getsource(PRSMNode.stop)
    assert "_await_bounded(self.transport.stop()" in src, (
        "transport.stop() must be bounded (libp2p subprocess teardown can hang)"
    )
    assert "_await_bounded(self.ledger.close()" in src, (
        "ledger.close() must be bounded (a stuck SQLite close can hang shutdown)"
    )
    assert "_await_bounded(self.bt_client.shutdown()" in src, (
        "bt_client.shutdown() must be bounded (libtorrent teardown can hang)"
    )
    # And the bare unbounded forms must be gone for those.
    assert "await self.transport.stop()" not in src
    assert "await self.ledger.close()" not in src
