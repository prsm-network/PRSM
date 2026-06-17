"""Sprint 1135 — settlement-receipt data plane BRICK A: producer retention store.

The settlement producer DISCARDS the ordered receipt set after commit:
``commit_ready_batches`` pops the batch from the accumulator and the
``CommittedBatch`` record keeps only ``leaf_hashes`` (plus the root). The dispute
modules (challenge_watcher, the assemblers, detect_double_spends) need the
*ordered receipt preimages* to act, and there is no runtime channel that retains
them today (see docs/2026-06-16-settlement-receipt-data-plane-design.md §1, §7
Brick A).

This module is that retention store. After a commit succeeds, the client (when an
optional store is injected) hands the popped batch's ordered ``BatchedReceipt``
list here, keyed by ``batch_id``. The store:

  - persists the full ordered receipt set (incl. nested ``ShardExecutionReceipt``
    signature + tee_attestation) byte-exact, mirroring ``SettlementStateStore``'s
    atomic-write + hex-bytes discipline (no network);
  - is bounded by ``max_batches`` (evict OLDEST on overflow) and GC-able by
    ``prune(retention_seconds)`` (drop entries past commit_timestamp + retention);
  - NEVER raises on a malformed/oversized on-disk entry — it skips + logs, so a
    single bad retained entry can't take down the producer.

CORRECTNESS CRUX (proved by the sp1135 tests): a retained batch, reloaded from
disk, recomputes to the SAME merkle root that was committed —
``build_tree_and_proofs([hash_leaf(batched_receipt_to_leaf(br)) ...]).root`` over
the reloaded ordered receipts equals the on-chain root. Order is the only thing
the merkle build depends on, so order is preserved end-to-end.

This store is read/written only OUTSIDE the money path: a write happens after the
batch is already tracked, wrapped in the client's try/except, and a failure here
must never break a commit. Nothing in this module touches commit/finalize logic.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.state_store import SettlementStateStore

logger = logging.getLogger(__name__)

_STORE_VERSION = 1
# Defensive cap: a single batch is ~1000 receipts (design §5); reject anything
# absurd so a forged/oversized on-disk entry can't blow up memory on reload.
_MAX_RECEIPTS_PER_BATCH = 100_000


# ── Serialization helpers (byte-exact, hex bytes) ─────────────────


def _batched_receipt_to_dict(br: BatchedReceipt) -> Dict[str, Any]:
    """JSON-able dict for a BatchedReceipt. Bytes fields are hex-encoded
    (consensus_group_id); the nested ShardExecutionReceipt reuses its own
    to_dict (signature + tee_attestation are JSON-native strings/dicts)."""
    return {
        "receipt": br.receipt.to_dict(),
        "requester_address": br.requester_address,
        "provider_address": br.provider_address,
        "value_ftns": str(br.value_ftns),  # str: value can exceed JSON safe int
        "local_escrow_id": br.local_escrow_id,
        "tier_slash_rate_bps": br.tier_slash_rate_bps,
        "consensus_group_id": br.consensus_group_id.hex(),
    }


def _batched_receipt_from_dict(d: Dict[str, Any]) -> BatchedReceipt:
    """Inverse of _batched_receipt_to_dict. Raises on a malformed entry (the
    caller catches + skips per the store's never-raise-on-load contract)."""
    return BatchedReceipt(
        receipt=ShardExecutionReceipt.from_dict(d["receipt"]),
        requester_address=d["requester_address"],
        provider_address=d["provider_address"],
        value_ftns=int(d["value_ftns"]),
        local_escrow_id=d["local_escrow_id"],
        tier_slash_rate_bps=d["tier_slash_rate_bps"],
        consensus_group_id=bytes.fromhex(d["consensus_group_id"]),
    )


@dataclass(frozen=True)
class PublishedBatch:
    """A committed batch retained for self-verify + (future bricks) publishing.

    Holds exactly what the dispute modules + the chain-root cross-check need: the
    batch_id, the committed merkle_root, the commit_timestamp (for GC), and the
    FULL ORDERED receipt set. ``receipts`` order == leaf order == the order the
    merkle root was built over."""
    batch_id: bytes
    merkle_root: bytes
    commit_timestamp: int
    receipts: List[BatchedReceipt]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id.hex(),
            "merkle_root": self.merkle_root.hex(),
            "commit_timestamp": self.commit_timestamp,
            "receipts": [_batched_receipt_to_dict(br) for br in self.receipts],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PublishedBatch":
        receipts_raw = d["receipts"]
        if not isinstance(receipts_raw, list):
            raise ValueError("receipts must be a list")
        if len(receipts_raw) > _MAX_RECEIPTS_PER_BATCH:
            raise ValueError(
                f"receipt count {len(receipts_raw)} exceeds cap "
                f"{_MAX_RECEIPTS_PER_BATCH}"
            )
        return cls(
            batch_id=bytes.fromhex(d["batch_id"]),
            merkle_root=bytes.fromhex(d["merkle_root"]),
            commit_timestamp=int(d["commit_timestamp"]),
            receipts=[_batched_receipt_from_dict(r) for r in receipts_raw],
        )


class PublishedBatchStore:
    """Bounded, GC-able, atomically-persisted store of committed receipt sets.

    Keyed by batch_id. Insertion order is the eviction order (oldest-first), so
    ``max_batches`` overflow evicts the oldest retained batch. Persistence mirrors
    ``SettlementStateStore`` (atomic .tmp write + fsync + os.replace; bytes hex).

    Never raises on a malformed/oversized on-disk entry: such an entry is skipped
    + logged on load. A corrupt JSON *file* (unparseable) is treated the same way
    — start empty + log — because retention is best-effort and must never block a
    producer from committing money (contrast SettlementStateStore, which is loud
    because IT guards already-escrowed funds)."""

    def __init__(
        self,
        path: Any,
        *,
        max_batches: int = 10_000,
        now: Callable[[], float] = time.time,
    ):
        if max_batches <= 0:
            raise ValueError(f"max_batches must be positive (got {max_batches})")
        self._path = Path(path)
        self._max_batches = max_batches
        self._now = now
        self._store = SettlementStateStore(self._path)
        # OrderedDict[batch_id_hex -> PublishedBatch], insertion-ordered (oldest
        # first) so popitem(last=False) evicts the oldest.
        self._batches: "OrderedDict[str, PublishedBatch]" = OrderedDict()
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    # ── public API ────────────────────────────────────────────────

    def put(
        self,
        batch_id: bytes,
        receipts: List[BatchedReceipt],
        merkle_root: bytes,
        commit_timestamp: int,
    ) -> None:
        """Retain the ordered receipt set for ``batch_id``. Re-putting an
        existing batch_id refreshes it (moved to newest). Evicts the oldest if
        over ``max_batches``. Persisted atomically."""
        pb = PublishedBatch(
            batch_id=bytes(batch_id),
            merkle_root=bytes(merkle_root),
            commit_timestamp=int(commit_timestamp),
            receipts=list(receipts),  # defensive copy; preserves order
        )
        key = pb.batch_id.hex()
        # Re-put moves to newest (refresh recency).
        if key in self._batches:
            del self._batches[key]
        self._batches[key] = pb
        # Bound: evict oldest until within cap.
        while len(self._batches) > self._max_batches:
            evicted_key, _ = self._batches.popitem(last=False)
            logger.debug(
                "PublishedBatchStore: evicted oldest batch %s (over max_batches=%d)",
                evicted_key, self._max_batches,
            )
        self._persist()

    def get(self, batch_id: bytes) -> Optional[PublishedBatch]:
        return self._batches.get(bytes(batch_id).hex())

    def all_batches(self) -> List[PublishedBatch]:
        """All retained batches, oldest-first."""
        return list(self._batches.values())

    def prune(self, retention_seconds: int) -> int:
        """Drop every entry whose commit_timestamp + retention_seconds < now.
        Returns the number dropped. Persists iff anything was dropped."""
        cutoff = self._now() - retention_seconds
        expired = [
            k for k, pb in self._batches.items() if pb.commit_timestamp < cutoff
        ]
        for k in expired:
            del self._batches[k]
        if expired:
            self._persist()
        return len(expired)

    # ── persistence (mirrors SettlementStateStore discipline) ─────

    def _persist(self) -> None:
        state = {
            "version": _STORE_VERSION,
            # dict preserves insertion order in Py3.7+; the load path also
            # restores order, so eviction order survives a restart.
            "batches": {k: pb.to_dict() for k, pb in self._batches.items()},
        }
        try:
            self._store.save(state)
        except Exception as exc:  # pragma: no cover - disk-failure path
            # Retention is best-effort; a persist failure is logged, never raised
            # (the in-memory set is still usable for this process's lifetime, and
            # the money path that called us must never see an exception).
            logger.warning(
                "PublishedBatchStore: persist to %s failed (%s); retention "
                "in-memory only this run", self._path, exc,
            )

    def _load(self) -> None:
        try:
            raw = self._store.load()
        except Exception as exc:
            # A corrupt retention file must NOT block the producer (it guards no
            # money). Start empty + log loudly so an operator notices.
            logger.warning(
                "PublishedBatchStore: state at %s unreadable (%s); starting "
                "empty — retention only, no funds at risk", self._path, exc,
            )
            return
        if raw is None:
            return
        batches = raw.get("batches")
        if not isinstance(batches, dict):
            logger.warning(
                "PublishedBatchStore: %s has no 'batches' object; starting empty",
                self._path,
            )
            return
        for key, entry in batches.items():
            try:
                pb = PublishedBatch.from_dict(entry)
            except Exception as exc:
                # Skip the single bad entry, keep the rest. Never raise.
                logger.warning(
                    "PublishedBatchStore: skipping malformed entry %s in %s (%s)",
                    key, self._path, exc,
                )
                continue
            # Re-key by the receipt's own batch_id hex (authoritative), not the
            # raw map key, so a tampered key can't shadow a real batch.
            self._batches[pb.batch_id.hex()] = pb
