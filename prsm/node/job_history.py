"""B8 async-dispatch follow-on — JobHistoryRecord + JobHistoryStore.

The synchronous-from-caller-view ``/compute/forge`` pipeline
returns the answer in-band. Once async dispatch lands, callers
will need a way to retrieve a job's result *after* the dispatch
call has returned. ``JobHistoryRecord`` is that surface: a
per-job record capturing the load-bearing state (route,
response, aggregator, participants, timing, error if failed).

``/compute/status/{job_id}`` reads from this store first (richer
data) and falls back to ``PaymentEscrow`` (escrow lifecycle
only) when the job isn't in history.

v2 (2026-05-09): optional filesystem persistence via
``persist_dir`` — node restart no longer wipes job history;
operators can retrospectively look up old jobs. v1 in-memory-only
behavior preserved when ``persist_dir`` is None.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESPONSE_CHARS = 65536   # sp925 — per-record response cap


def _max_response_chars() -> int:
    """sp925 — cap on the persisted response field (bounds per-record memory +
    disk; a single job's response is otherwise unbounded). Env override:
    PRSM_JOB_HISTORY_MAX_RESPONSE_BYTES (chars)."""
    try:
        return max(
            1,
            int(os.environ.get(
                "PRSM_JOB_HISTORY_MAX_RESPONSE_BYTES",
                _DEFAULT_MAX_RESPONSE_CHARS,
            )),
        )
    except (TypeError, ValueError):
        return _DEFAULT_MAX_RESPONSE_CHARS


# ──────────────────────────────────────────────────────────────────────
# Status
# ──────────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    """Lifecycle states for a /compute/forge job.

    Mirrors the escrow lifecycle but with semantics tied to the
    compute pipeline rather than the payment leg:
      - IN_PROGRESS: pipeline started; result not yet available.
      - COMPLETED: pipeline finished; ``response`` populated.
      - FAILED: pipeline raised; ``error`` populated.
      - CANCELLED: operator-initiated abort via /compute/cancel
        (v2, ships 2026-05-09); v1 caveat: in-flight Python
        coroutines are NOT interrupted — cancellation marks
        intent + refunds the budget, but the underlying compute
        may still complete (its release_escrow_split call will
        then race-lose against the now-REFUNDED escrow and raise
        EscrowAlreadyFinalizedError, which is the correct
        race-loss outcome).
    """
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ──────────────────────────────────────────────────────────────────────
# Record
# ──────────────────────────────────────────────────────────────────────


@dataclass
class JobHistoryRecord:
    """One job's compute-pipeline state. Distinct from the escrow
    record (which tracks payment) — together they form the full
    audit picture surfaced via ``/compute/status``.
    """

    job_id: str
    query: str
    status: JobStatus
    started_at: float

    # Lifecycle timing
    completed_at: Optional[float] = None

    # Result fields (populated on COMPLETED)
    route: Optional[str] = None
    response: Optional[str] = None
    aggregator_node_id: Optional[str] = None
    contributing_shards: Tuple[str, ...] = field(default_factory=tuple)
    participants: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    traces_collected: int = 0

    # Error field (populated on FAILED)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        if self.started_at < 0:
            raise ValueError(
                f"started_at must be non-negative, got {self.started_at}"
            )
        if self.completed_at is not None:
            if self.completed_at < self.started_at:
                raise ValueError(
                    f"completed_at ({self.completed_at}) must be >= "
                    f"started_at ({self.started_at})"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "query": self.query,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "route": self.route,
            "response": self.response,
            "aggregator_node_id": self.aggregator_node_id,
            "contributing_shards": list(self.contributing_shards),
            "participants": list(self.participants),
            "traces_collected": self.traces_collected,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobHistoryRecord":
        """Round-trip JSON → record. Tolerates missing optional
        keys (forward-compat with older persisted records)."""
        return cls(
            job_id=data["job_id"],
            query=data["query"],
            status=JobStatus(data["status"]),
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            route=data.get("route"),
            response=data.get("response"),
            aggregator_node_id=data.get("aggregator_node_id"),
            contributing_shards=tuple(
                data.get("contributing_shards") or ()
            ),
            participants=tuple(data.get("participants") or ()),
            traces_collected=data.get("traces_collected", 0),
            error=data.get("error"),
        )


# ──────────────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────────────


_DEFAULT_MAX_ENTRIES = 1024


class JobHistoryStore:
    """In-memory LRU-bounded store. Thread-safety: not enforced —
    /compute/forge handles serialize per-job-id naturally (each
    job has a distinct ``forge-<uuid>`` job_id), and the store's
    only multi-thread surface is concurrent reads via
    /compute/status which OrderedDict's GIL-protected get
    handles safely enough for v1.
    """

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, got {max_entries!r}"
            )
        self._max_entries = max_entries
        # OrderedDict preserves insertion order; move_to_end on
        # get() implements LRU semantics.
        self._records: "OrderedDict[str, JobHistoryRecord]" = OrderedDict()
        # Idempotency-key → job_id index for retry-safe POSTs.
        # Persisted alongside _records when persist_dir is set.
        self._idempotency_index: Dict[str, str] = {}

        # Filesystem persistence (optional). When set, put() writes
        # through to disk + get() falls back to disk on memory miss.
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    @staticmethod
    def _safe_filename(job_id: str) -> str:
        """Hash-based filename — guarantees filesystem-safe chars
        AND prevents path traversal regardless of caller-supplied
        job_id. Collision-resistant via SHA-256 first 16 hex chars
        (64 bits). The original job_id is preserved inside the
        record JSON for retrieval."""
        h = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        return f"{h[:16]}.json"

    def _disk_path(self, job_id: str) -> Path:
        assert self._persist_dir is not None
        return self._persist_dir / self._safe_filename(job_id)

    def _write_to_disk(self, record: JobHistoryRecord) -> None:
        if self._persist_dir is None:
            return
        try:
            path = self._disk_path(record.job_id)
            path.write_text(json.dumps(record.to_dict()))
        except Exception as e:
            logger.warning(
                "JobHistoryStore: disk write failed for job_id=%s "
                "(fail-soft, in-memory record still authoritative): %s",
                record.job_id, e,
            )

    def _load_from_disk(self) -> None:
        """Scan persist_dir on init + populate in-memory LRU.
        Records are inserted in started_at-ascending order so the
        most-recently-started ends up LRU-newest. Corrupt files
        are logged + skipped (fail-soft). Idempotency index is
        also rehydrated from _idempotency_index.json if present."""
        assert self._persist_dir is not None
        loaded: list = []
        for path in self._persist_dir.glob("*.json"):
            # Skip the idempotency-index file — it has a different
            # shape and is loaded separately below.
            if path.name == "_idempotency_index.json":
                continue
            try:
                data = json.loads(path.read_text())
                rec = JobHistoryRecord.from_dict(data)
                loaded.append(rec)
            except Exception as e:
                logger.warning(
                    "JobHistoryStore: skipping corrupt file %s: %s",
                    path, e,
                )
        loaded.sort(key=lambda r: r.started_at)
        for rec in loaded:
            self._records[rec.job_id] = rec
        # Apply LRU bound (in case disk has more than max_entries).
        while len(self._records) > self._max_entries:
            self._records.popitem(last=False)
        # Rehydrate idempotency index.
        idx_path = self._persist_dir / "_idempotency_index.json"
        if idx_path.exists():
            try:
                disk = json.loads(idx_path.read_text())
                if isinstance(disk, dict):
                    self._idempotency_index.update(disk)
            except Exception as e:
                logger.warning(
                    "JobHistoryStore: idempotency index rehydrate "
                    "failed: %s", e,
                )

    def _read_from_disk(self, job_id: str) -> Optional[JobHistoryRecord]:
        if self._persist_dir is None:
            return None
        path = self._disk_path(job_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return JobHistoryRecord.from_dict(data)
        except Exception as e:
            logger.warning(
                "JobHistoryStore: disk read failed for job_id=%s: %s",
                job_id, e,
            )
            return None

    def put(self, record: JobHistoryRecord) -> None:
        """Insert or update a record. Updates promote to most-
        recently-used; new inserts may trigger LRU eviction of the
        oldest entry. When persist_dir is set, the record is also
        written through to disk."""
        # sp925 — bound the response field so a single job's (untrusted,
        # provider/executor-supplied) response can't bloat memory + the .json
        # file unboundedly. Truncate with a marker; the cap is configurable.
        resp = record.response
        if resp is not None:
            cap = _max_response_chars()
            if len(resp) > cap:
                record.response = (
                    resp[:cap]
                    + f"...[truncated {len(resp) - cap} chars — raise "
                    f"PRSM_JOB_HISTORY_MAX_RESPONSE_BYTES to retain more]"
                )
        if record.job_id in self._records:
            # Overwrite + promote.
            self._records[record.job_id] = record
            self._records.move_to_end(record.job_id)
            self._write_to_disk(record)
            return
        self._records[record.job_id] = record
        while len(self._records) > self._max_entries:
            self._records.popitem(last=False)  # evict oldest
        self._write_to_disk(record)

    def get(self, job_id: str) -> Optional[JobHistoryRecord]:
        """Look up a record. Promotes to most-recently-used. When
        persist_dir is set, falls back to disk on in-memory miss
        (e.g., LRU-evicted records remain retrievable until manual
        cleanup of persist_dir)."""
        rec = self._records.get(job_id)
        if rec is not None:
            self._records.move_to_end(job_id)
            return rec
        # In-memory miss; try disk if persistence is enabled.
        rec = self._read_from_disk(job_id)
        if rec is None:
            return None
        # Re-populate the in-memory LRU (subject to eviction).
        self._records[rec.job_id] = rec
        self._records.move_to_end(rec.job_id)
        while len(self._records) > self._max_entries:
            self._records.popitem(last=False)
        return rec

    def size(self) -> int:
        return len(self._records)

    def put_with_idempotency(
        self,
        record: JobHistoryRecord,
        *,
        idempotency_key: str,
    ) -> None:
        """Like ``put()`` but also registers the
        idempotency_key → job_id mapping for retry-safe POST
        flows. Repeat calls with same key + same record are a
        no-op (existing mapping preserved)."""
        if not idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        self.put(record)
        self._idempotency_index[idempotency_key] = record.job_id
        # Persist the index alongside records when configured.
        if self._persist_dir is not None:
            try:
                idx_path = self._persist_dir / "_idempotency_index.json"
                idx_path.write_text(json.dumps(self._idempotency_index))
            except Exception as e:
                logger.warning(
                    "JobHistoryStore: idempotency index persist failed: %s",
                    e,
                )

    def lookup_by_idempotency_key(self, key: str) -> Optional[str]:
        """Return job_id mapped to ``key``, or None when unknown.

        Falls back to disk if persist_dir is set + key not in
        memory (e.g., after restart where _load_from_disk hasn't
        rehydrated the index)."""
        if not key:
            return None
        cached = self._idempotency_index.get(key)
        if cached is not None:
            return cached
        if self._persist_dir is None:
            return None
        idx_path = self._persist_dir / "_idempotency_index.json"
        if not idx_path.exists():
            return None
        try:
            disk = json.loads(idx_path.read_text())
            # Hydrate full index from disk to avoid repeated reads.
            if isinstance(disk, dict):
                self._idempotency_index.update(disk)
                return self._idempotency_index.get(key)
        except Exception as e:
            logger.warning(
                "JobHistoryStore: idempotency index read failed: %s", e,
            )
        return None

    def list(
        self,
        *,
        status_filter: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        route_filter: Optional[str] = None,
    ) -> list:
        """Enumerate records most-recent-first (by started_at DESC),
        with optional status filter + pagination. Used by
        /compute/jobs to surface a paginated job list to operators
        via the ``prsm_jobs_list`` MCP tool.

        Sort by started_at (not LRU touch order) so a get() doesn't
        re-order list output unexpectedly.

        Sprint 260 — ``route_filter`` matches ``record.route``
        exactly. Used by operators to scope queries to a single
        compute path (forge | inference | inference_stream |
        qo_swarm | direct_llm | swarm).
        """
        records = list(self._records.values())
        if status_filter is not None:
            records = [r for r in records if r.status == status_filter]
        if route_filter is not None:
            records = [r for r in records if r.route == route_filter]
        records.sort(key=lambda r: r.started_at, reverse=True)
        if offset:
            records = records[offset:]
        if limit is not None:
            records = records[:limit]
        return records

    def count(
        self,
        *,
        status_filter: Optional[JobStatus] = None,
        route_filter: Optional[str] = None,
    ) -> int:
        """Total number of records matching the filter — for
        pagination's `total` field. Cheaper than ``len(list(...))``
        because it skips sort + slice."""
        if status_filter is None and route_filter is None:
            return len(self._records)
        return sum(
            1 for r in self._records.values()
            if (status_filter is None or r.status == status_filter)
            and (route_filter is None or r.route == route_filter)
        )
