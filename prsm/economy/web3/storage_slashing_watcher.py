"""StorageSlashing event watcher.

Async daemon polling the on-chain StorageSlashing contract for
HeartbeatRecorded / ProofFailureSlashed / HeartbeatMissingSlashed
events. Closes the operator-monitoring half of
`EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §5.3 (anomalous
slashing detection): without a watcher, the only way to detect
slashes is manual Basescan polling.

Operational use cases:
  - Storage providers monitoring for slashing of their own address
    (real-time alert).
  - Foundation council monitoring fleet-wide slashing rate (anomaly
    detection).
  - Dashboards / forensics tooling.

Mirrors KeyDistributionWatcher shape.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from prsm.economy.web3.storage_slashing import (
    HeartbeatMissingSlashedEvent,
    HeartbeatRecordedEvent,
    ProofFailureSlashedEvent,
)


logger = logging.getLogger(__name__)

# sp1457 — Base public RPC rejects any eth_getLogs range wider than ~10k blocks (-32614). A node that
# restarts >10k blocks behind its persisted baseline must not poll [persisted+1, latest] in ONE call
# (it would hard-fail, get swallowed, and wedge the watcher forever). Cap each poll to a <=9k window
# so a far-behind watcher advances one bounded window per tick and catches up across ticks. Mirrors
# batch_settlement_contract_client / stake_manager _SCAN_MAX_WINDOW.
_SCAN_MAX_WINDOW = 9_000


HeartbeatRecordedCallback = Callable[
    [HeartbeatRecordedEvent], Union[None, Awaitable[None]],
]
ProofFailureSlashedCallback = Callable[
    [ProofFailureSlashedEvent], Union[None, Awaitable[None]],
]
HeartbeatMissingSlashedCallback = Callable[
    [HeartbeatMissingSlashedEvent], Union[None, Awaitable[None]],
]


class StorageSlashingWatcher:
    """Polls a StorageSlashingClient and fires callbacks on each new
    event observed.

    Construction:
        client: StorageSlashingClient instance (must expose
            latest_block / get_heartbeat_recorded_events /
            get_proof_failure_slashed_events /
            get_heartbeat_missing_slashed_events).
        on_heartbeat_recorded / on_proof_failure_slashed /
        on_heartbeat_missing_slashed: optional callbacks. If None
        for a given event type, that event is NOT polled.
        poll_interval_sec: cadence between polls. Default 30.0.

    First-tick semantics: marks current chain tip as baseline; does
    NOT replay history.

    Failure-mode contract: per-event-type RPC failures swallowed;
    last_processed_block does NOT advance on RPC error. Callback
    exceptions swallowed.
    """

    WATCHER_KEY = "storage_slashing"

    KNOWN_EVENT_NAMES = frozenset({
        "HeartbeatRecorded",
        "ProofFailureSlashed",
        "HeartbeatMissingSlashed",
    })

    def __init__(
        self,
        client,
        *,
        on_heartbeat_recorded: Optional[HeartbeatRecordedCallback] = None,
        on_proof_failure_slashed: Optional[ProofFailureSlashedCallback] = None,
        on_heartbeat_missing_slashed: Optional[
            HeartbeatMissingSlashedCallback
        ] = None,
        poll_interval_sec: float = 30.0,
        state_store=None,
        event_filters: Optional[Dict[str, Dict[str, Any]]] = None,
        dedup_store=None,
    ) -> None:
        if poll_interval_sec <= 0:
            raise ValueError(
                f"poll_interval_sec must be > 0, got {poll_interval_sec}"
            )
        if event_filters is not None:
            if not isinstance(event_filters, dict):
                raise TypeError(
                    f"event_filters must be a dict mapping event-name "
                    f"to argument_filters dict, got "
                    f"{type(event_filters).__name__}"
                )
            unknown = set(event_filters.keys()) - self.KNOWN_EVENT_NAMES
            if unknown:
                raise ValueError(
                    f"event_filters contains unknown event name(s): "
                    f"{unknown!r}. Valid names: {self.KNOWN_EVENT_NAMES!r}"
                )
        self._client = client
        self._on_recorded = on_heartbeat_recorded
        self._on_proof = on_proof_failure_slashed
        self._on_missing = on_heartbeat_missing_slashed
        self._poll_interval = float(poll_interval_sec)
        self._state_store = state_store
        # Sprint 551: persistent (watcher_key, tx_hash, log_index)
        # dedup. Without it, restart-catch-up re-dispatches every
        # slash + heartbeat event the previous run handled between
        # callback dispatch and post-loop baseline persist. Final
        # sibling in the sprint-549/550 audit trifecta.
        self._dedup_store = dedup_store
        self._event_filters = event_filters or {}
        self._stop_event = asyncio.Event()
        self.last_processed_block: Optional[int] = None
        # Sprint 401 — tick-age tracking. Bumped only on
        # successful poll completion (latest_block() RPC
        # success + tick body reaching its natural exit).
        # RPC failure leaves it stale → silent-watcher-
        # death surfaces on /health/detailed.
        self.last_tick_at: Optional[datetime] = None

    @property
    def poll_interval_sec(self) -> float:
        return self._poll_interval

    @property
    def interval_seconds(self) -> float:
        """Alias for poll_interval_sec — adopts the sprint-
        400 _daemon_subsystem helper's canonical attr name
        so /health/detailed auto-surfaces tick_status."""
        return self._poll_interval

    @property
    def last_tick_age_seconds(self) -> Optional[float]:
        if self.last_tick_at is None:
            return None
        return (
            datetime.now(timezone.utc) - self.last_tick_at
        ).total_seconds()

    async def run_forever(self) -> None:
        self._stop_event.clear()
        while not self._stop_event.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._poll_interval,
                )
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stop_event.set()

    async def tick(self) -> None:
        try:
            latest = self._client.latest_block()
        except Exception:
            logger.exception(
                "StorageSlashingWatcher: latest_block() RPC failed"
            )
            # Sprint 401 — RPC failure means no forward
            # progress; last_tick_at stays stale.
            return

        if self.last_processed_block is None:
            persisted = None
            if self._state_store is not None:
                try:
                    persisted = self._state_store.load(self.WATCHER_KEY)
                except Exception:
                    logger.exception(
                        "StorageSlashingWatcher: state_store.load() "
                        "raised; falling back to chain-tip baseline",
                    )
            if persisted is not None:
                self.last_processed_block = persisted
            else:
                self.last_processed_block = latest
                self._persist_baseline()
                # Sprint 401 — baseline-established path
                # IS a successful tick.
                self.last_tick_at = datetime.now(timezone.utc)
                return

        if latest <= self.last_processed_block:
            # Sprint 401 — no-new-blocks path IS a
            # successful tick (poll completed, no work).
            self.last_tick_at = datetime.now(timezone.utc)
            return

        from_block = self.last_processed_block + 1
        to_block = min(latest, from_block + _SCAN_MAX_WINDOW - 1)

        all_succeeded = True

        if self._on_recorded is not None:
            if not await self._poll_event_type(
                "heartbeat_recorded", from_block, to_block,
                self._client.get_heartbeat_recorded_events, self._on_recorded,
                argument_filters=self._event_filters.get("HeartbeatRecorded"),
            ):
                all_succeeded = False

        if self._on_proof is not None:
            if not await self._poll_event_type(
                "proof_failure_slashed", from_block, to_block,
                self._client.get_proof_failure_slashed_events, self._on_proof,
                argument_filters=self._event_filters.get("ProofFailureSlashed"),
            ):
                all_succeeded = False

        if self._on_missing is not None:
            if not await self._poll_event_type(
                "heartbeat_missing_slashed", from_block, to_block,
                self._client.get_heartbeat_missing_slashed_events,
                self._on_missing,
                argument_filters=self._event_filters.get("HeartbeatMissingSlashed"),
            ):
                all_succeeded = False

        if all_succeeded:
            self.last_processed_block = to_block
            self._persist_baseline()
            # Sprint 401 — full-poll success path.
            self.last_tick_at = datetime.now(timezone.utc)
        # If all_succeeded=False, tick partially failed —
        # do NOT bump. Operators see stale tick_status until
        # the next clean poll catches up.

    def _persist_baseline(self) -> None:
        if self._state_store is None or self.last_processed_block is None:
            return
        try:
            self._state_store.save(
                self.WATCHER_KEY, self.last_processed_block,
            )
        except Exception:
            logger.exception(
                "StorageSlashingWatcher: state_store.save() raised "
                "for block=%d; will retry on next baseline advance",
                self.last_processed_block,
            )

    async def _poll_event_type(
        self, name: str, from_block: int, to_block: int, getter, callback,
        *, argument_filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            # Pass argument_filters ONLY when set, for backwards-compat
            # with client/stub implementations that predate the kwarg.
            if argument_filters is not None:
                events = getter(
                    from_block, to_block,
                    argument_filters=argument_filters,
                )
            else:
                events = getter(from_block, to_block)
        except Exception:
            logger.exception(
                "StorageSlashingWatcher: get_%s_events RPC failed", name,
            )
            return False
        for event in events:
            # Sprint 551: persistent dedup mirroring sprints 549 + 550.
            # Skip events the previous run already dispatched; mark
            # AFTER successful callback. Fail-soft on SQLite hiccups.
            tx_hash = getattr(event, "tx_hash", None)
            log_index = getattr(event, "log_index", None)
            if (
                self._dedup_store is not None
                and tx_hash is not None
                and log_index is not None
            ):
                try:
                    if self._dedup_store.has_processed_event(
                        self.WATCHER_KEY, tx_hash, log_index,
                    ):
                        continue
                except Exception:
                    logger.exception(
                        "StorageSlashingWatcher: dedup lookup raised; "
                        "dispatching anyway"
                    )
            await self._invoke_cb(callback, event)
            if (
                self._dedup_store is not None
                and tx_hash is not None
                and log_index is not None
            ):
                try:
                    self._dedup_store.mark_processed_event(
                        self.WATCHER_KEY, tx_hash, log_index,
                    )
                except Exception:
                    logger.exception(
                        "StorageSlashingWatcher: dedup mark raised; "
                        "next tick may re-dispatch"
                    )
        return True

    async def _invoke_cb(self, callback, event) -> None:
        try:
            result = callback(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception(
                "StorageSlashingWatcher: callback raised; daemon continues"
            )
