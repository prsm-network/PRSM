"""Sprint 1382 — background watcher for settlement challenges against this operator's batches.

Turns a landed ``ReceiptChallenged`` into a WARNING alert + an active-challenge count (a metric
source) WITHOUT the operator polling ``prsm node challenge-status``. Reuses sp1381's
``Web3SettlementContractClient.get_challenges_for_provider`` on a fixed cadence.

Baseline semantics (mirrors StorageSlashingWatcher): the FIRST tick seeds the seen-set from any
pre-existing challenges and logs a one-time summary, so the watcher only fires per-challenge alerts
for challenges that land AFTER it starts — no alarm storm on startup, but the operator is still told
that unresolved challenges exist.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class SettlementChallengeWatcher:
    def __init__(
        self,
        client: Any,
        operator_address: str,
        *,
        poll_interval_sec: float = 60.0,
        lookback_blocks: Optional[int] = None,
        on_new_challenge: Optional[Callable[[Any], None]] = None,
    ) -> None:
        if poll_interval_sec <= 0:
            raise ValueError(f"poll_interval_sec must be > 0, got {poll_interval_sec}")
        self._client = client
        self._operator = operator_address
        self._poll_interval = float(poll_interval_sec)
        self._lookback = lookback_blocks
        self._on_new_challenge = on_new_challenge
        self._seen: set = set()
        self._active: List[Any] = []
        self._baseline_done = False
        self._stop_event = asyncio.Event()
        self.last_tick_at: Optional[datetime] = None

    @property
    def active_challenge_count(self) -> int:
        """Challenges seen against the operator's batches as of the last tick (a metric source)."""
        return len(self._active)

    @property
    def poll_interval_sec(self) -> float:
        return self._poll_interval

    def last_tick_age_seconds(self) -> Optional[float]:
        if self.last_tick_at is None:
            return None
        return (datetime.now(timezone.utc) - self.last_tick_at).total_seconds()

    async def run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001 — a failed poll must never kill the loop
                logger.warning("SettlementChallengeWatcher tick failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                pass

    async def stop(self) -> None:
        self._stop_event.set()

    async def tick(self) -> None:
        challenges = await asyncio.to_thread(
            lambda: self._client.get_challenges_for_provider(
                self._operator, lookback_blocks=self._lookback))
        self.last_tick_at = datetime.now(timezone.utc)
        self._active = list(challenges)
        if not self._baseline_done:
            self._baseline_done = True
            self._seen |= {self._key(c) for c in challenges}
            if challenges:
                logger.warning(
                    "SettlementChallengeWatcher: %d pre-existing challenge(s) against operator %s "
                    "at startup (run `prsm node challenge-status` for detail).",
                    len(challenges), self._operator)
            return
        for c in challenges:
            k = self._key(c)
            if k in self._seen:
                continue
            self._seen.add(k)
            self._dispatch(c)

    @staticmethod
    def _key(c: Any) -> tuple:
        return (getattr(c, "batch_id", None), getattr(c, "receipt_leaf_hash", None),
                getattr(c, "tx_hash", None))

    def _dispatch(self, c: Any) -> None:
        logger.warning(
            "SettlementChallengeWatcher: NEW challenge on batch %s (%s, -%s FTNS) by %s "
            "[status=%s] — stake at risk if it succeeds; see `prsm node slash-status`.",
            getattr(c, "batch_id", "?"), getattr(c, "reason", "?"),
            getattr(c, "invalidated_value_ftns", "?"), getattr(c, "challenger", "?"),
            getattr(c, "batch_status", "?"))
        if self._on_new_challenge is not None:
            try:
                self._on_new_challenge(c)
            except Exception as exc:  # noqa: BLE001 — a bad callback must not break the watcher
                logger.warning(
                    "SettlementChallengeWatcher on_new_challenge callback failed: %s", exc)
