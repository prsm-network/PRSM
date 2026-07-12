"""Phase 3.1 Task 4: ReceiptAccumulator.

Off-chain per-counterparty queue of Phase 2 ShardExecutionReceipts
awaiting on-chain batch settlement. Pure data + trigger logic — no
network, no async, no crypto. Pairs naturally with Task 5's Merkle-tree
builder + Task 6's on-chain client: when the accumulator reports a
batch is READY, the client pops it, builds the Merkle tree, and posts
commitBatch to the BatchSettlementRegistry.

Design contract (docs/2026-04-21-phase3.1-batch-settlement-design.md §2.1):
  - Receipts are indexed per (requester_address, provider_address)
    pair. One batch = one pair (Task 2 design refinement).
  - Three trigger thresholds, all configurable:
      count: default 1000 receipts
      time:  default 3600 seconds since first receipt in batch
      value: default 100 FTNS cumulative (in base units, 18 decimals)
  - First trigger to fire wins; the batch becomes READY.
  - Caller (Task 6 client) polls `ready_batches()` and pops each.

The accumulator is intentionally stateless about settlement outcomes —
it does not track "in-flight" batches after pop. Caller owns the
committed-batch lifecycle (via the on-chain registry state + local
PaymentEscrow reconciliation, Task 7).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from prsm.compute.shard_receipt import ShardExecutionReceipt

logger = logging.getLogger(__name__)


class TriggerReason(str, Enum):
    """Which of the three configured thresholds fired for a batch."""
    COUNT = "count"
    TIME = "time"
    VALUE = "value"


@dataclass(frozen=True)
class BatchedReceipt:
    """A Phase 2 ShardExecutionReceipt wrapped with the metadata needed
    for on-chain settlement.

    Constructed by the caller (typically MarketplaceOrchestrator, Task 7)
    immediately after a successful remote dispatch, from the receipt
    returned by RemoteShardDispatcher + the addresses + the quoted
    price that was actually escrowed.
    """
    receipt: ShardExecutionReceipt
    requester_address: str       # 0x-hex Ethereum address paying this receipt
    provider_address: str        # 0x-hex Ethereum address earning this receipt
    value_ftns: int              # quoted price in FTNS base units (wei, 18 dec)
    local_escrow_id: str         # pointer back to local PaymentEscrow entry
    # Phase 7: the provider's bond-time slash-rate snapshot in basis
    # points. 0 = no bonded stake / no slash. Snapshot at commit time
    # means a provider cannot dodge slashing via mid-batch downgrade.
    # Defaults to 0 so pre-Phase-7 callers don't have to specify.
    tier_slash_rate_bps: int = 0
    # Phase 7.1x §8.7: non-zero binds this receipt to a k-of-n consensus
    # dispatch. CONSENSUS_MISMATCH challenges require both the minority
    # and majority batches to share a non-zero group_id AND to come from
    # different providers. Zero means "not a consensus receipt" — batch
    # together with other zero-group receipts from the same (requester,
    # provider) pair. Defaults to all-zero bytes so pre-Phase-7.1x
    # callers don't have to specify.
    consensus_group_id: bytes = b"\x00" * 32

    def __post_init__(self):
        if not (0 <= self.tier_slash_rate_bps <= 10000):
            raise ValueError(
                f"tier_slash_rate_bps must be in [0, 10000] "
                f"(got {self.tier_slash_rate_bps})"
            )
        if len(self.consensus_group_id) != 32:
            raise ValueError(
                f"consensus_group_id must be 32 bytes "
                f"(got {len(self.consensus_group_id)})"
            )


@dataclass
class PendingBatch:
    """Mutable accumulator for one (requester, provider, group_id,
    tier_slash_rate_bps) key's receipts.

    Every receipt in one PendingBatch must share the same group_id and
    slash_rate — those are batch-level invariants (they serialize as
    single fields on the on-chain Batch struct). Receipts that disagree
    on either belong in a different PendingBatch keyed separately in
    the accumulator."""
    receipts: List[BatchedReceipt] = field(default_factory=list)
    started_at_unix: int = 0
    total_value_ftns: int = 0
    # Captured from the first receipt; all subsequent receipts must match.
    tier_slash_rate_bps: int = 0
    consensus_group_id: bytes = b"\x00" * 32
    # sp973 — local_escrow_ids already counted in THIS batch, so a duplicate
    # add() is a no-op WITHIN a single un-committed batch. NOT a cross-batch
    # replay guard: this set is PER-PendingBatch and pop_batch() discards it on
    # commit, so the next add() for the same key creates a fresh PendingBatch
    # with an EMPTY seen_escrow_ids. sp1431 (money-path audit #2): callers MUST
    # NOT rely on this as durable idempotency against a receipt whose batch
    # already committed — the client keeps NO durable committed-escrow-id ledger
    # and the registry has no content dedup, so a re-delivered receipt (crash in
    # the commit→discard window, or an upstream retry) double-settles on-chain.
    # The real fix (deferred, latent behind gated per-stage settlement) is a
    # durable committed-escrow-id/leaf ledger consulted here + at commit.
    seen_escrow_ids: Set[str] = field(default_factory=set)

    @property
    def count(self) -> int:
        return len(self.receipts)

    def is_empty(self) -> bool:
        return not self.receipts

    def append(self, br: BatchedReceipt, at_unix: int) -> None:
        """Add a receipt. Sets started_at_unix + batch-level fields on
        the first append. Subsequent appends must agree on
        tier_slash_rate_bps and consensus_group_id — the caller's
        accumulator keying should guarantee this; we assert defensively.

        sp973 — idempotent on local_escrow_id: a receipt whose escrow id is
        already counted in this batch is a no-op (never double-counts
        value/count)."""
        if br.local_escrow_id and br.local_escrow_id in self.seen_escrow_ids:
            logger.debug(
                "PendingBatch.append: duplicate local_escrow_id %s ignored",
                br.local_escrow_id,
            )
            return
        if self.is_empty():
            self.started_at_unix = at_unix
            self.tier_slash_rate_bps = br.tier_slash_rate_bps
            self.consensus_group_id = br.consensus_group_id
        else:
            if br.tier_slash_rate_bps != self.tier_slash_rate_bps:
                raise ValueError(
                    f"receipt tier_slash_rate_bps {br.tier_slash_rate_bps} "
                    f"disagrees with batch {self.tier_slash_rate_bps}"
                )
            if br.consensus_group_id != self.consensus_group_id:
                raise ValueError(
                    "receipt consensus_group_id disagrees with batch"
                )
        self.receipts.append(br)
        self.total_value_ftns += br.value_ftns
        if br.local_escrow_id:
            self.seen_escrow_ids.add(br.local_escrow_id)


# Phase 7.1x §8.7 + Phase 7: accumulator key extended to separate
# batches that disagree on batch-level slash rate or consensus group.
# A non-zero consensus_group_id binds a receipt to a k-of-n dispatch;
# a given provider in a given group has exactly one batch. Non-
# consensus receipts (group_id = all-zero bytes) still batch together
# by (requester, provider) as in Phase 3.1.
AccumulatorKey = Tuple[str, str, bytes, int]
#                     (requester, provider, group_id, slash_rate_bps)


@dataclass(frozen=True)
class ReadyBatch:
    """A batch that has crossed at least one threshold. The caller pops
    it from the accumulator and commits it on chain."""
    key: AccumulatorKey           # see AccumulatorKey alias above
    batch: PendingBatch
    trigger: TriggerReason        # which threshold fired first


@dataclass(frozen=True)
class AccumulatorConfig:
    """Thresholds and configuration. Defaults per Phase 3.1 design §2.1."""
    count_threshold: int = 1000
    time_threshold_seconds: int = 3600          # 1 hour
    value_threshold_ftns: int = 100 * 10**18    # 100 FTNS in wei

    def __post_init__(self):
        if self.count_threshold <= 0:
            raise ValueError(
                f"count_threshold must be positive (got {self.count_threshold})"
            )
        if self.time_threshold_seconds <= 0:
            raise ValueError(
                f"time_threshold_seconds must be positive "
                f"(got {self.time_threshold_seconds})"
            )
        if self.value_threshold_ftns <= 0:
            raise ValueError(
                f"value_threshold_ftns must be positive "
                f"(got {self.value_threshold_ftns})"
            )


class ReceiptAccumulator:
    """Per-counterparty accumulator of batched receipts.

    Key is a (requester_address, provider_address) tuple — each pair gets
    its own independent batch + independent trigger state.

    Accumulator is mutation-only from the writer's side (add); readers
    inspect readiness via ready_batches() and drain via pop_batch().
    Not thread-safe; callers must serialize access if used from multiple
    coroutines (typical deployment: single MarketplaceOrchestrator owns
    one accumulator instance).
    """

    def __init__(self, config: Optional[AccumulatorConfig] = None):
        self._config = config if config is not None else AccumulatorConfig()
        self._pending: Dict[AccumulatorKey, PendingBatch] = {}

    @property
    def config(self) -> AccumulatorConfig:
        return self._config

    def add(
        self,
        br: BatchedReceipt,
        at_unix: Optional[int] = None,
    ) -> None:
        """Add a receipt to its counterparty batch. Sets started_at_unix
        on the first append if this (requester, provider) pair is new
        or had a previously-drained batch.

        at_unix override is for deterministic testing; production uses
        current wall clock.
        """
        key: AccumulatorKey = (
            br.requester_address,
            br.provider_address,
            br.consensus_group_id,
            br.tier_slash_rate_bps,
        )
        batch = self._pending.setdefault(key, PendingBatch())
        now = at_unix if at_unix is not None else int(time.time())
        batch.append(br, now)

    def _check_triggers(
        self,
        batch: PendingBatch,
        now: int,
    ) -> Optional[TriggerReason]:
        """Return the first threshold that has fired, or None.

        Check order matches the spec's listing order (count, time, value).
        In practice two can fire simultaneously; the order is arbitrary
        but stable for logging."""
        if batch.is_empty():
            return None
        if batch.count >= self._config.count_threshold:
            return TriggerReason.COUNT
        elapsed = now - batch.started_at_unix
        if elapsed >= self._config.time_threshold_seconds:
            return TriggerReason.TIME
        if batch.total_value_ftns >= self._config.value_threshold_ftns:
            return TriggerReason.VALUE
        return None

    def ready_batches(self, at_unix: Optional[int] = None) -> List[ReadyBatch]:
        """List all batches that have hit at least one threshold."""
        now = at_unix if at_unix is not None else int(time.time())
        ready: List[ReadyBatch] = []
        for key, batch in self._pending.items():
            trigger = self._check_triggers(batch, now)
            if trigger is not None:
                ready.append(ReadyBatch(key=key, batch=batch, trigger=trigger))
        return ready

    def pop_batch(
        self,
        key: AccumulatorKey,
    ) -> Optional[PendingBatch]:
        """Remove and return the batch for this key.

        The caller (Task 6 settlement client) is responsible for actually
        committing the popped batch on-chain. If the commit fails, the
        caller can re-add the receipts via add() — the accumulator is
        stateless about commit outcomes."""
        return self._pending.pop(key, None)

    def peek_batch(
        self,
        key: AccumulatorKey,
    ) -> Optional[PendingBatch]:
        """Non-destructive read of a batch's current state."""
        return self._pending.get(key)

    def pending_keys(self) -> List[AccumulatorKey]:
        """List of every accumulator key with a non-empty batch."""
        return [k for k, v in self._pending.items() if not v.is_empty()]

    def total_receipt_count(self) -> int:
        """Sum of receipt counts across all pending batches."""
        return sum(b.count for b in self._pending.values())

    def total_pending_value_ftns(self) -> int:
        """Sum of all pending batches' cumulative value."""
        return sum(b.total_value_ftns for b in self._pending.values())

    # ── Durability seam (sp1409) ─────────────────────────────────

    def pending_items(self) -> List[Tuple[AccumulatorKey, PendingBatch]]:
        """Every non-empty pending batch, for a durable snapshot. The accumulator
        stays JSON-agnostic — BatchSettlementClient owns the serialization."""
        return [(k, v) for k, v in self._pending.items() if not v.is_empty()]

    def restore_pending(
        self,
        key: AccumulatorKey,
        receipts: List[BatchedReceipt],
        started_at_unix: int,
    ) -> None:
        """sp1409 — rehydrate one pending batch from a durable snapshot.

        Replays ``append`` rather than trusting persisted derived state, so the
        batch-level invariants (matching tier_slash_rate_bps / consensus_group_id) are
        re-checked and the sp973 ``seen_escrow_ids`` dedup set is rebuilt — a receipt
        re-delivered after a restart is then still a no-op. ``started_at_unix`` is
        restored exactly: it drives the TIME threshold, so resetting it would push a
        restarting node's commit an hour further out on every restart."""
        batch = PendingBatch()
        for br in receipts:
            batch.append(br, started_at_unix)
        if batch.is_empty():
            return
        batch.started_at_unix = started_at_unix
        self._pending[key] = batch
