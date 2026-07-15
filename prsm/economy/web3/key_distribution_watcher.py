"""KeyDistribution event watcher.

Async daemon that polls the on-chain KeyDistribution contract for
KeyReleased / KeyDeposited / KeyDeauthorized events and fires
user-supplied callbacks. Closes the operationally-meaningful half
of `EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §5.4 detection
scenario: without a watcher, the only way to detect KeyReleased
events is manual Basescan polling — too slow for a P0 surface
where Tier C trust depends on payment-verification correctness.

Mirrors `prsm/emission/watcher.py` (EmissionWatcher) shape:
async asyncio.Event-driven loop, exception-swallowing, optional
callbacks, graceful stop. Only events the operator subscribes to
are polled — saves RPC bandwidth when only one of the three is
of interest.

Activation pattern::

    from prsm.economy.web3.key_distribution import (
        KeyDistributionClient,
    )
    from prsm.economy.web3.key_distribution_watcher import (
        KeyDistributionWatcher,
    )

    client = KeyDistributionClient(rpc_url=..., contract_address=...)

    async def on_release(event):
        # Cross-check against payment-escrow records.
        ...

    watcher = KeyDistributionWatcher(
        client=client,
        on_key_released=on_release,
        poll_interval_sec=30.0,
    )
    asyncio.create_task(watcher.run_forever())
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from prsm.economy.web3.key_distribution import KeyReleasedEvent


logger = logging.getLogger(__name__)

# sp1457 — Base public RPC rejects any eth_getLogs range wider than ~10k blocks (-32614). A node that
# restarts >10k blocks behind its persisted baseline must not poll [persisted+1, latest] in ONE call
# (it would hard-fail, get swallowed, and wedge the watcher forever). Cap each poll to a <=9k window
# so a far-behind watcher advances one bounded window per tick and catches up across ticks. Mirrors
# batch_settlement_contract_client / stake_manager _SCAN_MAX_WINDOW.
_SCAN_MAX_WINDOW = 9_000


# ──────────────────────────────────────────────────────────────────────
# Event dataclasses missing from the client module
# ──────────────────────────────────────────────────────────────────────

# Note: KeyReleasedEvent already lives in
# `prsm/economy/web3/key_distribution.py`. The two below augment it
# so the watcher can decode the full event surface.


def _validate_content_hash(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError(
            f"content_hash must be bytes, got {type(value).__name__}"
        )
    if len(value) != 32:
        raise ValueError(f"content_hash must be 32 bytes, got {len(value)}")
    return bytes(value)


@dataclass(frozen=True)
class KeyDepositedEvent:
    """Decoded ``KeyDeposited(bytes32 indexed contentHash,
    address indexed publisher, address indexed royalty,
    uint256 releaseFeeFtnsWei)``.

    Sprint 550: ``tx_hash`` + ``log_index`` carry on-chain event
    identity for sprint-549's persistent dedup pattern.
    """
    content_hash: bytes
    publisher: str
    royalty: str
    release_fee_ftns_wei: int
    tx_hash: Optional[str] = None
    log_index: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_content_hash(self.content_hash)
        if not isinstance(self.release_fee_ftns_wei, int) or self.release_fee_ftns_wei < 0:
            raise ValueError(
                f"release_fee_ftns_wei must be non-negative int, "
                f"got {self.release_fee_ftns_wei!r}"
            )

    @classmethod
    def from_decoded_args(
        cls,
        args: Dict[str, Any],
        *,
        tx_hash: Optional[str] = None,
        log_index: Optional[int] = None,
    ) -> "KeyDepositedEvent":
        return cls(
            content_hash=bytes(args["contentHash"]),
            publisher=str(args["publisher"]),
            royalty=str(args["royalty"]),
            release_fee_ftns_wei=int(args["releaseFeeFtnsWei"]),
            tx_hash=tx_hash,
            log_index=log_index,
        )


@dataclass(frozen=True)
class KeyDeauthorizedEvent:
    """Decoded ``KeyDeauthorized(bytes32 indexed contentHash,
    address indexed publisher)``.

    Sprint 550: ``tx_hash`` + ``log_index`` carry on-chain event
    identity for sprint-549's persistent dedup pattern.
    """
    content_hash: bytes
    publisher: str
    tx_hash: Optional[str] = None
    log_index: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_content_hash(self.content_hash)

    @classmethod
    def from_decoded_args(
        cls,
        args: Dict[str, Any],
        *,
        tx_hash: Optional[str] = None,
        log_index: Optional[int] = None,
    ) -> "KeyDeauthorizedEvent":
        return cls(
            content_hash=bytes(args["contentHash"]),
            publisher=str(args["publisher"]),
            tx_hash=tx_hash,
            log_index=log_index,
        )


# ──────────────────────────────────────────────────────────────────────
# Callback typing
# ──────────────────────────────────────────────────────────────────────


KeyReleasedCallback = Callable[[KeyReleasedEvent], Union[None, Awaitable[None]]]
KeyDepositedCallback = Callable[[KeyDepositedEvent], Union[None, Awaitable[None]]]
KeyDeauthorizedCallback = Callable[
    [KeyDeauthorizedEvent], Union[None, Awaitable[None]],
]


# ──────────────────────────────────────────────────────────────────────
# Watcher
# ──────────────────────────────────────────────────────────────────────


class KeyDistributionWatcher:
    """Polls a KeyDistributionClient and fires callbacks on each new
    event observed.

    Construction:
        client: KeyDistributionClient instance (must expose
            latest_block / get_key_released_events /
            get_key_deposited_events / get_key_deauthorized_events).
        on_key_released / on_key_deposited / on_key_deauthorized:
            optional async or sync callbacks. If None for a given
            event type, that event is NOT polled (saves RPC).
        poll_interval_sec: cadence between polls. Default 30.0.

    First-tick semantics: marks the current chain tip as the
    baseline; does NOT replay history. Operators wanting historical
    backfill should call `client.get_*_events` directly.

    Failure-mode contract:
      - Per-event-type RPC failures are swallowed; ``last_processed_block``
        does NOT advance on RPC error (so events aren't lost on
        transient failure — next tick retries the same range).
      - Callback exceptions are swallowed; daemon stays alive across
        user-callback bugs.
    """

    WATCHER_KEY = "key_distribution"

    # Valid event names for event_filters validation. If a future
    # event is added to KeyDistribution.sol, extend this set + add
    # a new callback parameter + a new poll-call branch in tick().
    KNOWN_EVENT_NAMES = frozenset(
        {"KeyReleased", "KeyDeposited", "KeyDeauthorized"},
    )

    def __init__(
        self,
        client,
        *,
        on_key_released: Optional[KeyReleasedCallback] = None,
        on_key_deposited: Optional[KeyDepositedCallback] = None,
        on_key_deauthorized: Optional[KeyDeauthorizedCallback] = None,
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
        self._on_released = on_key_released
        self._on_deposited = on_key_deposited
        self._on_deauthorized = on_key_deauthorized
        self._poll_interval = float(poll_interval_sec)
        self._state_store = state_store
        self._event_filters = event_filters or {}
        # Sprint 550: persistent (watcher_key, tx_hash, log_index)
        # dedup. Without it, restart-catch-up re-dispatches every
        # event the previous run handled between callback dispatch
        # and post-loop baseline persist. Sibling primitive to
        # sprint 549's CompensationDistributorWatcher fix.
        self._dedup_store = dedup_store
        self._stop_event = asyncio.Event()
        self.last_processed_block: Optional[int] = None
        # Sprint 401 — tick-age tracking.
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
        """One poll iteration. Always returns; never raises."""
        try:
            latest = self._client.latest_block()
        except Exception:
            logger.exception(
                "KeyDistributionWatcher: latest_block() RPC failed"
            )
            return

        if self.last_processed_block is None:
            # First tick — try state_store load first; if found, use
            # the persisted baseline (restart-resilient: pick up
            # where we left off). If not found, fall back to chain
            # tip + persist.
            persisted = None
            if self._state_store is not None:
                try:
                    persisted = self._state_store.load(self.WATCHER_KEY)
                except Exception:
                    logger.exception(
                        "KeyDistributionWatcher: state_store.load() "
                        "raised; falling back to chain-tip baseline",
                    )
            if persisted is not None:
                self.last_processed_block = persisted
                # Fall through to normal polling — don't return; the
                # downtime-window range gets polled below.
            else:
                self.last_processed_block = latest
                self._persist_baseline()
                self.last_tick_at = datetime.now(timezone.utc)
                return

        if latest <= self.last_processed_block:
            self.last_tick_at = datetime.now(timezone.utc)
            return  # no new blocks

        from_block = self.last_processed_block + 1
        to_block = min(latest, from_block + _SCAN_MAX_WINDOW - 1)

        # Track per-event-type RPC success — we only advance the
        # baseline if ALL subscribed event types succeed for this
        # range. A partial failure rolls back the advance so the next
        # tick retries the full range. Trade-off: re-emits already-
        # seen events for the successful types on retry; documented
        # as honest-scope. Callback idempotency is the operator's
        # contract.
        all_succeeded = True

        if self._on_released is not None:
            if not await self._poll_event_type(
                "released", from_block, to_block,
                self._client.get_key_released_events, self._on_released,
                argument_filters=self._event_filters.get("KeyReleased"),
            ):
                all_succeeded = False

        if self._on_deposited is not None:
            if not await self._poll_event_type(
                "deposited", from_block, to_block,
                self._client.get_key_deposited_events, self._on_deposited,
                argument_filters=self._event_filters.get("KeyDeposited"),
            ):
                all_succeeded = False

        if self._on_deauthorized is not None:
            if not await self._poll_event_type(
                "deauthorized", from_block, to_block,
                self._client.get_key_deauthorized_events, self._on_deauthorized,
                argument_filters=self._event_filters.get("KeyDeauthorized"),
            ):
                all_succeeded = False

        if all_succeeded:
            self.last_processed_block = to_block
            self._persist_baseline()
            # Sprint 401 — full poll success.
            self.last_tick_at = datetime.now(timezone.utc)
        # Partial-failure: do NOT bump. tick_status surfaces
        # stale until next clean poll.

    def _persist_baseline(self) -> None:
        """Save the current baseline to state_store (if wired).
        Logs + continues on save failure; next successful tick
        re-persists."""
        if self._state_store is None or self.last_processed_block is None:
            return
        try:
            self._state_store.save(
                self.WATCHER_KEY, self.last_processed_block,
            )
        except Exception:
            logger.exception(
                "KeyDistributionWatcher: state_store.save() raised "
                "for block=%d; will retry on next baseline advance",
                self.last_processed_block,
            )

    async def _poll_event_type(
        self, name: str, from_block: int, to_block: int, getter, callback,
        *, argument_filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            # Pass argument_filters through ONLY when set, preserving
            # backward-compat with client/stub implementations that
            # predate the argument_filters kwarg. When set, web3.py
            # event.get_logs(argument_filters=...) does RPC-side
            # filtering.
            if argument_filters is not None:
                events = getter(
                    from_block, to_block,
                    argument_filters=argument_filters,
                )
            else:
                events = getter(from_block, to_block)
        except Exception:
            logger.exception(
                "KeyDistributionWatcher: get_%s_events RPC failed", name,
            )
            return False
        for event in events:
            # Sprint 550: persistent dedup mirroring sprint 549.
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
                        "KeyDistributionWatcher: dedup lookup raised; "
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
                        "KeyDistributionWatcher: dedup mark raised; "
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
                "KeyDistributionWatcher: callback raised; daemon continues"
            )
