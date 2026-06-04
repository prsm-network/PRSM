"""HeartbeatScheduler — async daemon that calls
StorageSlashingClient.record_heartbeat() on cadence.

Closes the deferred-follow-on item from EXPLOIT_RESPONSE_PLAYBOOK_
ANNEX_2026_05.md §6.2: providers running v1.7.0 had to invoke
record_heartbeat externally (cron, manual, custom service) until a
daemon shipped. Without the daemon, providers become vulnerable to
permissionless slash_for_missing_heartbeat() once their grace window
elapses.

Tests use a stub StorageSlashingClient to verify the daemon's
loop shape, error swallowing, and stop semantics without requiring
a live chain.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List
from unittest.mock import MagicMock

import pytest

from prsm.economy.web3.heartbeat_scheduler import HeartbeatScheduler
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)


# ──────────────────────────────────────────────────────────────────────
# Stub StorageSlashingClient
# ──────────────────────────────────────────────────────────────────────


class _FakeSlashingClient:
    """Records calls + lets tests inject per-call outcomes.

    `outcomes` is a list of values: each is either
      - a (tx_hash_hex, TransferStatus) tuple — happy path
      - an Exception class to instantiate and raise
      - an Exception instance to raise directly
    Consumed FIFO; once exhausted, subsequent calls return a default
    happy result.
    """

    def __init__(self, outcomes=None):
        self._outcomes = list(outcomes or [])
        self.calls: List[None] = []
        self.address = "0x" + "11" * 20

    def record_heartbeat(self):
        self.calls.append(None)
        if not self._outcomes:
            return ("0x" + "ab" * 32, TransferStatus.CONFIRMED)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, type) and issubclass(outcome, Exception):
            raise outcome("injected")
        return outcome


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_requires_client(self):
        with pytest.raises(TypeError):
            HeartbeatScheduler()  # type: ignore[call-arg]

    def test_default_interval_is_positive(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.interval_seconds > 0

    def test_custom_interval(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client, interval_seconds=10.0)
        assert scheduler.interval_seconds == 10.0

    def test_zero_interval_rejected(self):
        client = _FakeSlashingClient()
        with pytest.raises(ValueError, match="interval"):
            HeartbeatScheduler(client=client, interval_seconds=0)

    def test_negative_interval_rejected(self):
        client = _FakeSlashingClient()
        with pytest.raises(ValueError, match="interval"):
            HeartbeatScheduler(client=client, interval_seconds=-5)


# ──────────────────────────────────────────────────────────────────────
# Single tick
# ──────────────────────────────────────────────────────────────────────


class TestSingleTick:
    @pytest.mark.asyncio
    async def test_happy_path_calls_record_heartbeat(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_happy_path_increments_success_counter(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.success_count == 0
        await scheduler.tick()
        assert scheduler.success_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_failure_swallowed(self):
        client = _FakeSlashingClient(outcomes=[BroadcastFailedError])
        scheduler = HeartbeatScheduler(client=client)
        # MUST NOT raise — daemon stays alive.
        await scheduler.tick()
        assert scheduler.success_count == 0
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_pending_error_swallowed_but_logged_loud(self, caplog):
        # OnChainPendingError is the concerning case — receipt unknown.
        # The daemon should NOT crash but SHOULD log at WARNING+.
        client = _FakeSlashingClient(
            outcomes=[OnChainPendingError("pending", tx_hash="0xdead")],
        )
        scheduler = HeartbeatScheduler(client=client)
        with caplog.at_level(logging.WARNING):
            await scheduler.tick()
        assert any(
            "pending" in r.message.lower() or "unknown" in r.message.lower()
            for r in caplog.records
        )
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_reverted_error_swallowed(self):
        client = _FakeSlashingClient(outcomes=[OnChainRevertedError])
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_unexpected_exception_swallowed(self):
        # If client raises an unexpected type, daemon stays alive.
        client = _FakeSlashingClient(outcomes=[RuntimeError("weird")])
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_callback_fires_on_success(self):
        client = _FakeSlashingClient()
        events = []

        async def cb(tx_hash):
            events.append(tx_hash)

        scheduler = HeartbeatScheduler(client=client, on_success=cb)
        await scheduler.tick()
        assert len(events) == 1
        assert events[0].startswith("0x")

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash_daemon(self):
        client = _FakeSlashingClient()

        def cb(tx_hash):
            raise RuntimeError("callback exploded")

        scheduler = HeartbeatScheduler(client=client, on_success=cb)
        # Must not propagate — the bug is in the operator's callback,
        # not the daemon, and the daemon must keep running.
        await scheduler.tick()
        assert scheduler.success_count == 1


# ──────────────────────────────────────────────────────────────────────
# Sprint 399 — last_tick_at heartbeat tracking
# ──────────────────────────────────────────────────────────────────────


class TestLastTickAtTracking:
    """The daemon's task can be running (asyncio task not
    done) yet not making forward progress because the
    underlying chain RPC is failing every tick. Sprint
    392's hard-won observability lesson on the bootstrap-
    server side: surface heartbeat AGE, not just
    'is-running'. Mirrors that on the operator-node side
    starting with HeartbeatScheduler — the canonical
    silent-economic-failure case (no chain heartbeat =
    no compensation epoch credit)."""

    def test_last_tick_at_initially_none(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.last_tick_at is None

    @pytest.mark.asyncio
    async def test_successful_tick_bumps_last_tick_at(self):
        from datetime import datetime, timezone
        before = datetime.now(timezone.utc)
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        after = datetime.now(timezone.utc)
        assert scheduler.last_tick_at is not None
        assert before <= scheduler.last_tick_at <= after

    @pytest.mark.asyncio
    async def test_broadcast_failure_does_not_bump_last_tick_at(self):
        """If the chain RPC fails, the daemon hasn't actually
        recorded a heartbeat on-chain. last_tick_at should
        NOT advance — otherwise operators looking at the
        timestamp think things are fine while their node
        misses compensation epochs."""
        client = _FakeSlashingClient(outcomes=[BroadcastFailedError])
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert scheduler.last_tick_at is None

    @pytest.mark.asyncio
    async def test_pending_error_does_not_bump_last_tick_at(self):
        """Same logic as broadcast failure — receipt unknown
        = forward progress unknown = don't claim success."""
        client = _FakeSlashingClient(
            outcomes=[OnChainPendingError("pending", tx_hash="0xdead")],
        )
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert scheduler.last_tick_at is None

    @pytest.mark.asyncio
    async def test_reverted_error_does_not_bump_last_tick_at(self):
        client = _FakeSlashingClient(outcomes=[OnChainRevertedError])
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert scheduler.last_tick_at is None

    @pytest.mark.asyncio
    async def test_unexpected_exception_does_not_bump_last_tick_at(self):
        client = _FakeSlashingClient(outcomes=[RuntimeError("weird")])
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert scheduler.last_tick_at is None

    @pytest.mark.asyncio
    async def test_multiple_successes_advance_last_tick_at(self):
        """Each successful tick should advance the timestamp,
        not stay frozen at the first success."""
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        first = scheduler.last_tick_at
        # Force a small gap so timestamps are distinguishable
        import asyncio as _asyncio
        await _asyncio.sleep(0.001)
        await scheduler.tick()
        second = scheduler.last_tick_at
        assert second > first

    def test_last_tick_age_seconds_none_when_never_ticked(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.last_tick_age_seconds is None

    @pytest.mark.asyncio
    async def test_last_tick_age_seconds_after_tick(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        age = scheduler.last_tick_age_seconds
        assert age is not None
        # Just ticked — age should be small
        assert 0 <= age < 1.0


# ──────────────────────────────────────────────────────────────────────
# Run loop
# ──────────────────────────────────────────────────────────────────────


class TestRunForever:
    @pytest.mark.asyncio
    async def test_run_forever_exits_when_stop_called(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client, interval_seconds=0.05)

        task = asyncio.create_task(scheduler.run_forever())
        # Let it tick at least once.
        await asyncio.sleep(0.15)
        await scheduler.stop()
        # run_forever should now return promptly.
        await asyncio.wait_for(task, timeout=1.0)

        # At least one heartbeat occurred.
        assert len(client.calls) >= 1

    @pytest.mark.asyncio
    async def test_multiple_direct_ticks_increment_counters(self):
        # Avoids loop-timing flakiness under pytest-asyncio fixture
        # overhead. Direct `tick()` calls verify the cadence-relevant
        # property: successive ticks accumulate counter state.
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client, interval_seconds=60.0)

        for _ in range(5):
            await scheduler.tick()

        assert scheduler.success_count == 5
        assert scheduler.failure_count == 0
        assert len(client.calls) == 5

    @pytest.mark.asyncio
    async def test_multiple_direct_ticks_keep_running_through_failures(self):
        # Mix successes + failures via outcome injection. The
        # cadence-irrelevant property under test: scheduler counters
        # correctly classify each call regardless of order, and no
        # exception escapes tick() to crash a hypothetical loop.
        client = _FakeSlashingClient(outcomes=[
            BroadcastFailedError,
            ("0x" + "01" * 32, TransferStatus.CONFIRMED),
            OnChainRevertedError,
            ("0x" + "02" * 32, TransferStatus.CONFIRMED),
            OnChainPendingError("pending", tx_hash="0xfe"),
            ("0x" + "03" * 32, TransferStatus.CONFIRMED),
        ])
        scheduler = HeartbeatScheduler(client=client, interval_seconds=60.0)

        for _ in range(6):
            await scheduler.tick()

        # 3 successes, 3 failures (broadcast + revert + pending).
        assert scheduler.success_count == 3
        assert scheduler.failure_count == 3


# ──────────────────────────────────────────────────────────────────────
# sp999 — grace re-tune (false-slash defense)
# ──────────────────────────────────────────────────────────────────────


class _GraceClient(_FakeSlashingClient):
    """Slashing client with a mutable on-chain heartbeat grace, for the re-tune
    tests. `grace_raises=True` simulates a transient RPC read failure."""

    def __init__(self, grace_seconds, outcomes=None):
        super().__init__(outcomes)
        self._grace = grace_seconds
        self.grace_raises = False

    def heartbeat_grace_seconds(self):
        if self.grace_raises:
            raise RuntimeError("rpc down")
        return self._grace


class TestGraceRetune:
    """sp999 — a HEALTHY daemon must shrink its auto-tuned heartbeat interval when
    governance reduces the on-chain grace (incident-response setHeartbeatGrace),
    or it can be falsely slashed for a 'missed' heartbeat despite heartbeating
    exactly as configured (its frozen cadence falls outside the tightened window)."""

    def test_auto_tuned_interval_shrinks_when_grace_drops(self):
        client = _GraceClient(grace_seconds=86400)   # 24h grace
        s = HeartbeatScheduler(client=client)        # auto-tune → 86400/4 = 6h
        assert s.interval_seconds == 21600.0
        client._grace = 3600                          # governance cuts grace to 1h
        s._maybe_retune()
        # 3600/4 = 900s (15 min) — back inside the tightened (2h) slash window.
        assert s.interval_seconds == 900.0

    def test_retune_never_grows_interval(self):
        client = _GraceClient(grace_seconds=3600)    # 1h → 900s
        s = HeartbeatScheduler(client=client)
        assert s.interval_seconds == 900.0
        client._grace = 86400                         # grace GREW
        s._maybe_retune()
        assert s.interval_seconds == 900.0            # unchanged — never grow

    def test_retune_noop_on_transient_read_error(self):
        client = _GraceClient(grace_seconds=86400)
        s = HeartbeatScheduler(client=client)         # 6h
        client.grace_raises = True                     # RPC blip during re-tune
        s._maybe_retune()
        # A transient read MUST NOT spuriously shrink (else one blip pins the
        # daemon at a tight cadence forever — only a genuine grace cut shrinks).
        assert s.interval_seconds == 21600.0

    def test_operator_pinned_interval_warns_but_not_shrunk(self, caplog):
        import logging as _logging
        client = _GraceClient(grace_seconds=86400)
        # Operator-pinned interval (not auto-tuned) — their choice is preserved.
        s = HeartbeatScheduler(client=client, interval_seconds=21600.0)
        client._grace = 3600                            # grace cut below the cadence
        with caplog.at_level(_logging.CRITICAL):
            s._maybe_retune()
        assert s.interval_seconds == 21600.0          # preserved
        assert any(
            "EXCEEDS the live slash window" in r.message
            for r in caplog.records
        )
