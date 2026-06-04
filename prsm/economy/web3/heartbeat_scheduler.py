"""Heartbeat scheduler — async daemon for StorageSlashingClient.

Closes the deferred-follow-on item from
``docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md`` §6.2:
storage providers running v1.7.0 had to invoke ``record_heartbeat``
externally (cron, manual, custom service) until a daemon shipped.
Without periodic heartbeats they become vulnerable to permissionless
``slash_for_missing_heartbeat()`` once their grace window elapses.

Mirrors ``prsm/emission/watcher.py`` (EmissionWatcher) — async
asyncio.Event-driven loop, exception-swallowing, optional callback,
graceful stop.

Activation pattern:

    from prsm.economy.web3.storage_slashing import StorageSlashingClient
    from prsm.economy.web3.heartbeat_scheduler import HeartbeatScheduler

    client = StorageSlashingClient(rpc_url=..., contract_address=...,
                                   private_key=...)
    scheduler = HeartbeatScheduler(client=client,
                                   interval_seconds=900)  # 15 min
    asyncio.create_task(scheduler.run_forever())
    # ... later ...
    await scheduler.stop()

Cadence guidance: choose ``interval_seconds`` to be substantially
shorter than ``client.heartbeat_grace_seconds()`` so a single missed
tick (RPC outage, container restart) does not push the provider
past the grace window. Default ``interval_seconds`` is 900 (15 min) —
appropriate for the contract's MIN_HEARTBEAT_GRACE = 1 hour. Operators
running with a longer grace can lengthen the interval; the daemon
does not auto-tune.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Union

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
)


logger = logging.getLogger(__name__)


SuccessCallback = Callable[[str], Union[None, Awaitable[None]]]
"""Called as ``callback(tx_hash_hex)`` after each successful heartbeat."""

CriticalCallback = Callable[[int], None]
"""sp939 — called as ``callback(consecutive_failures)`` when heartbeats have
failed CRITICAL_CONSECUTIVE_FAILURES times in a row (slashing imminent — the
operator should intervene before the on-chain grace window elapses)."""


class HeartbeatScheduler:
    """Periodically calls ``client.record_heartbeat()``.

    Construction:
        client: StorageSlashingClient (must have private_key set).
        interval_seconds: poll cadence. Default 900s = 15 min.
        on_success: optional callback fired with tx_hash on success.

    The scheduler swallows all exceptions from the client — a failure
    on one tick does not crash the loop. Failure modes:

      - BroadcastFailedError: log + counter; next tick retries.
      - OnChainPendingError: log at WARNING (receipt unknown); next
        tick retries (heartbeat is idempotent — another call resets
        the timestamp).
      - OnChainRevertedError: log + counter; next tick retries
        (recordHeartbeat has no real revert path on-chain, so this
        would indicate a deeper problem, but the daemon stays alive).
      - Unexpected exceptions: log + counter; next tick retries.

    success_count + failure_count are exposed for operator telemetry.
    """

    # Default fallback when auto-tune is unavailable. Matches the
    # contract's MIN_HEARTBEAT_GRACE = 1 hour: 3600 / AUTO_TUNE_DIVISOR
    # = 900s.
    DEFAULT_INTERVAL_SECONDS = 900.0

    # Auto-tune ratio: heartbeats per grace window. 4 = "miss up to
    # 3 ticks before grace expires" — defense against transient RPC
    # failures, container bounces, etc.
    AUTO_TUNE_DIVISOR = 4

    # Auto-tune floor: defends against runaway tight cadence if the
    # operator misconfigured grace_seconds to a very small value.
    # 60s is short enough to be useful + long enough to avoid
    # hammering the chain.
    AUTO_TUNE_MIN_INTERVAL_SECONDS = 60.0

    # sp939 — consecutive failures before escalating to CRITICAL + on_critical.
    # With the grace/4 auto-tune (3-tick buffer), 3 consecutive misses means
    # ~3/4 of the grace window has elapsed with no on-chain progress — roughly
    # one interval before the provider becomes slashable, i.e. the last moment
    # an operator can still intervene.
    CRITICAL_CONSECUTIVE_FAILURES = 3

    # sp999 — the loop wakes at LEAST this often to re-read the on-chain heartbeat
    # grace + (auto-tune mode) shrink the interval, INDEPENDENT of the heartbeat
    # cadence. The contract's MIN_HEARTBEAT_GRACE is 1 hour, so a tightened slash
    # window is >= grace*slashGraceMultiplier (default 2) = 2h; re-checking every
    # 15 min detects a governance grace reduction + re-tunes well inside it, even
    # when the heartbeat interval itself is hours (a long-running daemon that
    # auto-tuned to grace/4 = 6h off a 24h grace was otherwise stranded at a
    # cadence WIDER than the live window after an incident-response grace cut).
    GRACE_RECHECK_CEILING_SECONDS = 900.0

    def __init__(
        self,
        client,
        *,
        interval_seconds: Optional[float] = None,
        on_success: Optional[SuccessCallback] = None,
        on_critical: Optional[CriticalCallback] = None,
    ) -> None:
        # interval_seconds resolution:
        #   None (default) → auto-tune from client.heartbeat_grace_seconds()
        #     with floor at AUTO_TUNE_MIN_INTERVAL_SECONDS
        #   numeric        → operator-supplied; use directly (must be > 0)
        #
        # Auto-tune fallback to DEFAULT_INTERVAL_SECONDS (900s) when:
        #   - client lacks heartbeat_grace_seconds() method
        #   - method raises (e.g., RPC down at construction)
        #   - method returns non-positive (operator misconfig)
        # sp999 — remember whether the interval was auto-tuned (None) vs
        # operator-pinned. Only the auto-tuned path self-corrects (shrinks) when
        # governance reduces the on-chain grace; a pinned interval is the
        # operator's choice, so we only warn (loudly) if it falls behind the window.
        self._auto_tuned = interval_seconds is None
        if interval_seconds is None:
            interval_seconds = self._auto_tune_from_client(client)
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be > 0, got {interval_seconds}"
            )
        self._client = client
        self._interval = float(interval_seconds)
        self._on_success = on_success
        self._on_critical = on_critical
        self._stop_event = asyncio.Event()
        self.success_count = 0
        self.failure_count = 0
        # sp939 — consecutive failures since the last success. Reset on success;
        # crossing CRITICAL_CONSECUTIVE_FAILURES escalates (slashing imminent).
        self.consecutive_failures = 0
        # Sprint 399 — track last successful tick timestamp.
        # /health/detailed and Prometheus surface this so
        # operators can distinguish "task running but no
        # forward progress" (chain RPC failing every tick)
        # from "task running healthily." Mirrors sprint 392's
        # observability discipline from the bootstrap-server
        # side onto operator-node daemons. None until first
        # successful tick. Bumped ONLY on success — RPC
        # failure paths leave it stale, which is the signal.
        self.last_tick_at: Optional[datetime] = None

    @classmethod
    def _auto_tune_from_client(cls, client) -> float:
        """Read client.heartbeat_grace_seconds() and compute the
        proportional interval. Falls back to DEFAULT_INTERVAL_SECONDS
        on any error path."""
        if not hasattr(client, "heartbeat_grace_seconds"):
            logger.warning(
                "HeartbeatScheduler: client lacks heartbeat_grace_seconds() "
                "method; auto-tune unavailable, using "
                "fallback interval=%ss",
                cls.DEFAULT_INTERVAL_SECONDS,
            )
            return cls.DEFAULT_INTERVAL_SECONDS
        try:
            grace = client.heartbeat_grace_seconds()
        except Exception as exc:
            logger.warning(
                "HeartbeatScheduler: heartbeat_grace_seconds() raised "
                "%s: %s; auto-tune fallback to interval=%ss",
                type(exc).__name__, exc, cls.DEFAULT_INTERVAL_SECONDS,
            )
            return cls.DEFAULT_INTERVAL_SECONDS
        if not isinstance(grace, (int, float)) or grace <= 0:
            logger.warning(
                "HeartbeatScheduler: heartbeat_grace_seconds() returned "
                "non-positive value %r; treating as misconfig, "
                "auto-tune fallback to interval=%ss",
                grace, cls.DEFAULT_INTERVAL_SECONDS,
            )
            return cls.DEFAULT_INTERVAL_SECONDS
        tuned = max(
            grace / cls.AUTO_TUNE_DIVISOR,
            cls.AUTO_TUNE_MIN_INTERVAL_SECONDS,
        )
        logger.info(
            "HeartbeatScheduler: auto-tuned interval=%ss from grace=%ss "
            "(ratio 1/%d, floor %ss)",
            tuned, grace, cls.AUTO_TUNE_DIVISOR,
            cls.AUTO_TUNE_MIN_INTERVAL_SECONDS,
        )
        return tuned

    @property
    def interval_seconds(self) -> float:
        return self._interval

    @property
    def last_tick_age_seconds(self) -> Optional[float]:
        """Seconds since last successful tick, or None if no
        tick has ever succeeded. Surfaced via /health/detailed
        and Prometheus subsystem labeled gauge."""
        if self.last_tick_at is None:
            return None
        return (
            datetime.now(timezone.utc) - self.last_tick_at
        ).total_seconds()

    def _maybe_retune(self) -> None:
        """sp999 — re-read the on-chain heartbeat grace and SHRINK the auto-tuned
        interval if governance reduced it (incident-response setHeartbeatGrace).

        A long-running daemon froze its interval at construction. If grace is later
        cut (e.g. 24h → 1h), a HEALTHY provider heartbeating on the old cadence can
        fall OUTSIDE the new slash window and be permissionlessly slashed despite
        being honest. Re-reading here keeps the auto-tuned cadence inside the live
        window. Reads grace DIRECTLY (not via the error-masking auto-tune helper)
        so a transient RPC blip is a NO-OP — it must never spuriously shrink the
        interval, only a genuine grace reduction does. Only ever shrinks (never
        grows past the operator/auto value)."""
        client = self._client
        if not hasattr(client, "heartbeat_grace_seconds"):
            return
        try:
            grace = client.heartbeat_grace_seconds()
        except Exception:  # noqa: BLE001 — transient read: keep current interval
            return
        if not isinstance(grace, (int, float)) or grace <= 0:
            return
        retuned = max(
            grace / self.AUTO_TUNE_DIVISOR, self.AUTO_TUNE_MIN_INTERVAL_SECONDS,
        )
        if retuned >= self._interval:
            return  # grace unchanged or larger — never grow
        if self._auto_tuned:
            logger.warning(
                "HeartbeatScheduler: on-chain heartbeat grace dropped — shrinking "
                "auto-tuned interval %.0fs → %.0fs to stay inside the live slash "
                "window (a healthy provider must keep heartbeating within it).",
                self._interval, retuned,
            )
            self._interval = float(retuned)
        else:
            logger.critical(
                "HeartbeatScheduler: on-chain heartbeat grace dropped — the "
                "operator-pinned interval %.0fs now likely EXCEEDS the live slash "
                "window (auto-tune would use %.0fs). A HEALTHY provider may be "
                "slashed for a missed heartbeat; lower the configured interval or "
                "switch to auto-tune (unset PRSM_HEARTBEAT_INTERVAL_SECONDS).",
                self._interval, retuned,
            )

    async def run_forever(self) -> None:
        """Run the heartbeat loop until ``stop()`` is called.

        The heartbeat fires every ``self._interval``, but the loop WAKES at least
        every ``GRACE_RECHECK_CEILING_SECONDS`` to re-read the on-chain grace and
        (auto-tune mode) shrink the interval — so a governance grace reduction is
        detected + reacted to inside the live slash window even when the heartbeat
        cadence is hours (sp999)."""
        self._stop_event.clear()
        last_tick_mono: Optional[float] = None
        while not self._stop_event.is_set():
            self._maybe_retune()
            now = time.monotonic()
            if last_tick_mono is None or (now - last_tick_mono) >= self._interval:
                await self.tick()
                last_tick_mono = time.monotonic()
            # Sleep until the next heartbeat is due, capped so we re-check grace
            # frequently regardless of the (possibly hours-long) heartbeat cadence.
            due_in = self._interval - (time.monotonic() - last_tick_mono)
            nap = min(due_in, self.GRACE_RECHECK_CEILING_SECONDS)
            if nap <= 0:
                nap = self.GRACE_RECHECK_CEILING_SECONDS
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=nap,
                )
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        """Signal the loop to exit at the next iteration boundary."""
        self._stop_event.set()

    def _note_failure(self) -> None:
        """sp939 — record a failed tick and escalate when slashing is imminent.

        Increments the cumulative + consecutive counters; once the consecutive
        streak reaches CRITICAL_CONSECUTIVE_FAILURES, logs CRITICAL and invokes
        the optional on_critical alert hook so the operator can intervene before
        the on-chain grace window elapses and the stake becomes slashable."""
        self.failure_count += 1
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.CRITICAL_CONSECUTIVE_FAILURES:
            logger.critical(
                "HeartbeatScheduler: %d consecutive heartbeat failures — the "
                "provider stake is at risk of being slashed for a missed "
                "heartbeat once the on-chain grace window elapses. Operator "
                "intervention needed (check RPC / wallet gas / connectivity).",
                self.consecutive_failures,
            )
            if self._on_critical is not None:
                try:
                    self._on_critical(self.consecutive_failures)
                except Exception:  # pragma: no cover — alert hook must not crash the loop
                    logger.exception(
                        "HeartbeatScheduler: on_critical callback raised"
                    )

    async def tick(self) -> None:
        """Run one heartbeat attempt. Always returns; never raises.

        Public for unit testing; production code should call
        ``run_forever()`` instead.
        """
        try:
            tx_hash, _status = self._client.record_heartbeat()
        except OnChainPendingError as exc:
            # Concerning but not fatal — receipt unknown means the tx
            # may or may not have landed. Next tick will resubmit.
            self._note_failure()
            logger.warning(
                "heartbeat tx pending (receipt unknown): %s; will retry "
                "next tick (heartbeat is idempotent)",
                exc,
            )
            return
        except BroadcastFailedError as exc:
            self._note_failure()
            logger.info(
                "heartbeat broadcast failed: %s; will retry next tick", exc,
            )
            return
        except OnChainRevertedError as exc:
            self._note_failure()
            logger.warning(
                "heartbeat reverted unexpectedly (recordHeartbeat has no "
                "real revert path): %s",
                exc,
            )
            return
        except Exception:  # pragma: no cover — defensive
            self._note_failure()
            logger.exception("heartbeat tick failed with unexpected exception")
            return

        self.success_count += 1
        self.consecutive_failures = 0  # sp939 — forward progress; clear the streak
        # Sprint 399 — only bump on actual on-chain success.
        # All failure paths above return early without
        # updating this timestamp, which is what lets ops
        # alerts catch "loop running but no forward progress."
        self.last_tick_at = datetime.now(timezone.utc)
        logger.info("heartbeat ok: %s", tx_hash)

        if self._on_success is not None:
            await self._invoke_success_cb(tx_hash)

    async def _invoke_success_cb(self, tx_hash: str) -> None:
        assert self._on_success is not None
        try:
            result = self._on_success(tx_hash)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception(
                "on_success callback raised; daemon continues"
            )
