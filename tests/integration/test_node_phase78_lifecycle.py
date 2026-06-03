"""End-to-end integration test for the Phase 7/8 + watcher
lifecycle wiring in `Node.start` and `Node.stop`.

Where the unit tests at tests/unit/test_node_phase78_wiring.py +
test_node_event_watcher_wiring.py exercise the BUILDER helpers in
isolation, this suite verifies the LIFECYCLE WIRING:

  - All 5 daemon attrs (2 schedulers + 3 watchers) are referenced
    in BOTH Node.start (for asyncio.create_task launch) AND
    Node.stop (for graceful shutdown). Catches miswiring (e.g.,
    forgetting to add a watcher to the stop block).
  - The runtime stop pattern (signal stop → await with 5s timeout
    → suppress CancelledError) actually completes cleanly within
    a reasonable bound when applied to a fleet of stub daemons.

A full Node.initialize is intentionally NOT exercised here —
that's a heavy bootstrap path with ledger / transport / BitTorrent
/ etc. dependencies that aren't relevant to the daemon-lifecycle
question. The unit-test suites cover the builders' construction
contracts; this suite covers the lifecycle-wiring contract that
sits one level up.
"""
from __future__ import annotations

import asyncio
import inspect
import re
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.node import PRSMNode as Node, _drain_task_bounded


# ──────────────────────────────────────────────────────────────────────
# Test 1: static wiring integrity
# ──────────────────────────────────────────────────────────────────────


# Each daemon attr must appear in BOTH Node.start (launch path) AND
# Node.stop (shutdown path). Catches miswiring at code-review time
# rather than waiting for a real incident.
DAEMON_ATTRS = [
    "_heartbeat_scheduler",
    "_compensation_scheduler",
    "_key_distribution_watcher",
    "_storage_slashing_watcher",
    "_compensation_distributor_watcher",
]


class TestStaticWiringIntegrity:
    """Static introspection: each daemon attr referenced in both
    start + stop. Code-level invariant; runs fast."""

    def test_each_daemon_attr_in_start(self):
        start_src = inspect.getsource(Node.start)
        missing = [a for a in DAEMON_ATTRS if a not in start_src]
        assert not missing, (
            f"Daemon attrs missing from Node.start: {missing}. "
            f"Each must be referenced for asyncio.create_task launch."
        )

    def test_each_daemon_attr_in_stop(self):
        stop_src = inspect.getsource(Node.stop)
        missing = [a for a in DAEMON_ATTRS if a not in stop_src]
        assert not missing, (
            f"Daemon attrs missing from Node.stop: {missing}. "
            f"Each must be referenced for graceful shutdown."
        )

    def test_each_daemon_task_attr_in_stop(self):
        # Companion task slots must also be teardown-referenced.
        stop_src = inspect.getsource(Node.stop)
        for attr in DAEMON_ATTRS:
            task_attr = f"{attr}_task"
            assert task_attr in stop_src, (
                f"Task slot {task_attr} missing from Node.stop. "
                f"Each daemon's asyncio task must be awaited "
                f"with timeout in the shutdown loop."
            )

    def test_stop_drains_daemon_tasks_with_bounded_abandon(self):
        # Sprint 955 — Node.stop must BOUND its daemon-task drains so a
        # misbehaving/stuck daemon can't hang shutdown. It uses
        # _drain_task_bounded (a 5.0s bound), which on timeout cancel-requests
        # and ABANDONS the task — strictly stronger than the prior
        # asyncio.wait_for(task, 5.0), which cancelled THEN awaited the
        # cancellation and so could itself hang on a task that resists cancel
        # (an uncancellable executor RPC / subprocess teardown).
        stop_src = inspect.getsource(Node.stop)
        assert "_drain_task_bounded" in stop_src, (
            "Node.stop daemon-task drain must use _drain_task_bounded to "
            "bound the wait; without it a stuck daemon hangs shutdown."
        )
        assert "5.0" in stop_src, (
            "the daemon-task drain bound (5.0s) must be present in Node.stop."
        )
        # The primitive must ABANDON (not await) a non-finishing task — that is
        # the load-bearing difference from wait_for.
        drain_src = inspect.getsource(_drain_task_bounded)
        assert "asyncio.wait(" in drain_src, (
            "_drain_task_bounded must use asyncio.wait (returns at the deadline "
            "regardless), NOT wait_for (which awaits the cancellation)."
        )
        assert "task.cancel()" in drain_src, (
            "_drain_task_bounded must cancel-request the stuck task on timeout."
        )


# ──────────────────────────────────────────────────────────────────────
# Test 2: runtime wiring contract — stub fleet completes within bound
# ──────────────────────────────────────────────────────────────────────


class _StubDaemon:
    """Minimal stub of HeartbeatScheduler / *Watcher etc. — has a
    run_forever coroutine that awaits a stop signal, and a stop()
    method that sets the signal."""

    def __init__(self, name: str):
        self.name = name
        self._stop_event = asyncio.Event()
        self.tick_count = 0
        self.stopped = False

    async def run_forever(self):
        while not self._stop_event.is_set():
            self.tick_count += 1
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=0.05,
                )
            except asyncio.TimeoutError:
                continue
        self.stopped = True

    async def stop(self):
        self._stop_event.set()


class TestRuntimeStopContract:
    """Apply the same stop-pattern Node.stop uses to a fleet of stub
    daemons. Verifies the 5-task graceful-stop completes within the
    timeout bound and leaves the fleet in stopped state."""

    @pytest.mark.asyncio
    async def test_5_daemon_fleet_stops_cleanly_within_10s(self):
        # Build a fleet matching Node's daemon set.
        fleet = {name: _StubDaemon(name) for name in DAEMON_ATTRS}
        tasks = {
            name: asyncio.create_task(d.run_forever())
            for name, d in fleet.items()
        }

        # Let them tick a few times so they're actually running.
        await asyncio.sleep(0.1)
        for name, d in fleet.items():
            assert d.tick_count >= 1, (
                f"daemon {name} never ticked — sanity-check failure"
            )

        # Now apply the SAME stop pattern Node.stop uses (mirrors
        # prsm/node/node.py:2138+ as of 2026-05-08).
        for d in fleet.values():
            await d.stop()
        for task in tasks.values():
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5.0)

        # Verify all daemons cleanly stopped.
        for name, d in fleet.items():
            assert d.stopped, f"daemon {name} did not reach stopped state"

    @pytest.mark.asyncio
    async def test_fleet_with_one_misbehaving_daemon_still_completes(self):
        """If one daemon ignores stop(), the per-task timeout=5.0 bounds
        the wait. Verifies the timeout protects against a misbehaving
        daemon hanging the entire shutdown."""
        good = _StubDaemon("good")
        bad = _StubDaemon("bad")

        # Patch bad.stop to NOT actually set the stop event — simulates
        # a buggy daemon.
        bad.stop = AsyncMock()

        good_task = asyncio.create_task(good.run_forever())
        bad_task = asyncio.create_task(bad.run_forever())
        await asyncio.sleep(0.05)

        # Apply the bounded-wait pattern.
        await good.stop()
        await bad.stop()

        loop_start = asyncio.get_running_loop().time()
        for task in (good_task, bad_task):
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=0.5)
        elapsed = asyncio.get_running_loop().time() - loop_start

        # Total wait bounded by 2 × timeout (loop iterates over all
        # tasks). Real Node uses 5.0s; we test with 0.5s for speed
        # but verify the bound holds.
        assert elapsed < 2.0, (
            f"Fleet shutdown took {elapsed:.2f}s — the bounded-wait "
            f"pattern is broken (one bad daemon hung the rest)."
        )
        assert good.stopped
        # Cleanup the still-running bad task.
        bad_task.cancel()
        with suppress(asyncio.CancelledError):
            await bad_task


# ──────────────────────────────────────────────────────────────────────
# Test 3: initialize-time builder integration
# ──────────────────────────────────────────────────────────────────────


class TestInitializeBuildsAllDaemonSlots:
    """Verifies that the initialize block constructing the 5 daemon
    slots exists in source. This is a complement to the unit tests
    that exercise each builder in isolation — here we confirm
    Node.initialize actually invokes all 5 builders."""

    def test_initialize_calls_all_5_builders(self):
        init_src = inspect.getsource(Node.initialize)
        builders = [
            "_build_compensation_distributor_client_or_none",
            "_build_storage_slashing_client_or_none",
            "_build_compensation_scheduler_or_none",
            "_build_heartbeat_scheduler_or_none",
            "_build_key_distribution_client_or_none",
            "_build_key_distribution_watcher_or_none",
            "_build_storage_slashing_watcher_or_none",
            "_build_compensation_distributor_watcher_or_none",
        ]
        missing = [b for b in builders if b not in init_src]
        assert not missing, (
            f"Builder helpers missing from Node.initialize: {missing}. "
            f"Without these calls, the env-var-driven daemon "
            f"construction never happens."
        )

    def test_initialize_assigns_5_task_slots_to_none(self):
        """Each daemon's task slot must be pre-allocated to None at
        initialize-time so Node.stop can safely getattr without a
        defensive hasattr check."""
        init_src = inspect.getsource(Node.initialize)
        for attr in DAEMON_ATTRS:
            task_attr = f"self.{attr}_task = None"
            assert task_attr in init_src, (
                f"Task slot {task_attr} not pre-allocated at "
                f"initialize time. Without this, Node.stop's "
                f"getattr(...) call could raise AttributeError "
                f"when the daemon was never constructed."
            )
