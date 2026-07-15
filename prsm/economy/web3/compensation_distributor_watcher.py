"""CompensationDistributor event watcher.

Async daemon polling the on-chain CompensationDistributor contract
for Distributed events. Operationally smaller surface than the
other two watchers — only one event class is operationally
meaningful (admin-triggered WeightsScheduled / WeightsActivated /
PoolAddressesUpdated events are visible on Basescan and don't
drive operator-side automation; if an operator wants to react to
weight changes they should monitor Basescan directly).

Use cases:
  - Pool operators reconciling on-chain Distributed amounts against
    their internal accounting.
  - Foundation governance monitoring distribution cadence (call-gap
    > 7 days alert per CompensationDistributor.sol §3.5).
  - Public dashboards showing real-time distribution flow.

Mirrors KeyDistributionWatcher / StorageSlashingWatcher shape.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Union

from prsm.economy.web3.compensation_distributor import DistributedEvent


logger = logging.getLogger(__name__)

# sp1457 — Base public RPC rejects any eth_getLogs range wider than ~10k blocks (-32614). A node that
# restarts >10k blocks behind its persisted baseline must not poll [persisted+1, latest] in ONE call
# (it would hard-fail, get swallowed, and wedge the watcher forever). Cap each poll to a <=9k window
# so a far-behind watcher advances one bounded window per tick and catches up across ticks. Mirrors
# batch_settlement_contract_client / stake_manager _SCAN_MAX_WINDOW.
_SCAN_MAX_WINDOW = 9_000


DistributedCallback = Callable[
    [DistributedEvent], Union[None, Awaitable[None]],
]


class CompensationDistributorWatcher:
    """Polls a CompensationDistributorClient and fires callbacks on
    each new Distributed event observed.

    Construction:
        client: CompensationDistributorClient instance (must expose
            latest_block / get_distributed_events).
        on_distributed: optional callback. If None, no polling
            happens (saves RPC).
        poll_interval_sec: cadence between polls. Default 30.0.

    First-tick semantics: marks current chain tip as baseline; does
    NOT replay history.

    Failure-mode contract: RPC failures swallowed; last_processed_block
    does NOT advance on RPC error. Callback exceptions swallowed.
    """

    WATCHER_KEY = "compensation_distributor"

    def __init__(
        self,
        client,
        *,
        on_distributed: Optional[DistributedCallback] = None,
        poll_interval_sec: float = 30.0,
        state_store=None,
        dedup_store=None,
    ) -> None:
        if poll_interval_sec <= 0:
            raise ValueError(
                f"poll_interval_sec must be > 0, got {poll_interval_sec}"
            )
        self._client = client
        self._on_distributed = on_distributed
        self._poll_interval = float(poll_interval_sec)
        self._state_store = state_store
        # Sprint 549: persistent (watcher_key, tx_hash, log_index)
        # dedup. Without it, restart-catch-up re-dispatches every
        # event between the previous run's last successful baseline-
        # persist and the crash — duplicating distribution-log rows
        # + duplicating distribution.distributed webhook fires. None
        # is back-compat (the daemon opts into persistence in
        # node.py wiring; tests + ephemeral runs pass None).
        self._dedup_store = dedup_store
        self._stop_event = asyncio.Event()
        self.last_processed_block: Optional[int] = None
        # Sprint 401 — tick-age tracking. Bumped on each
        # success path; RPC failures leave it stale.
        self.last_tick_at: Optional[datetime] = None

    @property
    def poll_interval_sec(self) -> float:
        return self._poll_interval

    @property
    def interval_seconds(self) -> float:
        """Alias for poll_interval_sec — sprint-400
        _daemon_subsystem helper's canonical attr name."""
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
                "CompensationDistributorWatcher: latest_block() RPC failed"
            )
            return

        if self.last_processed_block is None:
            persisted = None
            if self._state_store is not None:
                try:
                    persisted = self._state_store.load(self.WATCHER_KEY)
                except Exception:
                    logger.exception(
                        "CompensationDistributorWatcher: state_store."
                        "load() raised; falling back to chain-tip baseline",
                    )
            if persisted is not None:
                self.last_processed_block = persisted
            else:
                self.last_processed_block = latest
                self._persist_baseline()
                self.last_tick_at = datetime.now(timezone.utc)
                return

        if latest <= self.last_processed_block:
            self.last_tick_at = datetime.now(timezone.utc)
            return

        if self._on_distributed is None:
            # No subscriber — just advance baseline so we don't waste
            # RPC on subsequent ticks.
            self.last_processed_block = latest
            self._persist_baseline()
            self.last_tick_at = datetime.now(timezone.utc)
            return

        from_block = self.last_processed_block + 1
        to_block = min(latest, from_block + _SCAN_MAX_WINDOW - 1)

        try:
            events = self._client.get_distributed_events(from_block, to_block)
        except Exception:
            logger.exception(
                "CompensationDistributorWatcher: get_distributed_events "
                "RPC failed",
            )
            return  # do NOT advance baseline OR last_tick_at

        for event in events:
            # Sprint 549: persistent dedup. Skip events the previous
            # run already dispatched (the crash-between-callback-and-
            # baseline-persist scenario). Mark only AFTER the callback
            # completes — if the callback raises, _invoke_cb's
            # exception handler swallows it but the lack of mark
            # means the next tick can retry. Dedup is best-effort:
            # both has_processed_event + mark_processed_event are
            # wrapped so a SQLite hiccup degrades to pre-sprint
            # behavior with a warning, not a crash.
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
                        "CompensationDistributorWatcher: dedup lookup "
                        "raised; dispatching anyway"
                    )
            await self._invoke_cb(event)
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
                        "CompensationDistributorWatcher: dedup mark "
                        "raised; next tick may re-dispatch"
                    )
        self.last_processed_block = to_block
        self._persist_baseline()
        # Sprint 401 — full poll-and-dispatch success.
        self.last_tick_at = datetime.now(timezone.utc)

    def _persist_baseline(self) -> None:
        if self._state_store is None or self.last_processed_block is None:
            return
        try:
            self._state_store.save(
                self.WATCHER_KEY, self.last_processed_block,
            )
        except Exception:
            logger.exception(
                "CompensationDistributorWatcher: state_store.save() "
                "raised for block=%d; will retry on next baseline "
                "advance",
                self.last_processed_block,
            )

    async def _invoke_cb(self, event) -> None:
        assert self._on_distributed is not None
        try:
            result = self._on_distributed(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception(
                "CompensationDistributorWatcher: callback raised; "
                "daemon continues"
            )
