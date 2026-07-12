"""Sprint 1317 (S3b-2) — node-side receiver store + ingest for routed per-stage settlement tasks.

A stage node, on receiving its routed ``PerStageSettlementTask`` (+ the full payee set), must:
  1. FAIL-CLOSED verify the requester authorized paying ITS OWN ``(payee, share)`` as a member of
     the signed set (the sp1316 ``verify_routed_settlement_task`` gate), then
  2. STAGE the authorized task for its own settlement client to accumulate + commit (S3b-3, the
     on-chain money path) — but NOT commit yet.

This module is that receive half. ``ingest_routed_task`` runs the gate then stages on authorize
(rejects + stages NOTHING otherwise). ``PerStageReceiverStore`` retains staged tasks, keyed by the
per-node ``local_escrow_id`` (``{job_id}::stage::{node_id}`` — the SAME idempotency key the
splitter stamps + the accumulator dedups on), so a re-routed / retried delivery can NEVER
double-stage and thus never lead to a double-commit. Pure: no transport, no chain (the commit is
S3b-3). Mirrors ``IssuedAuthorizationStore`` discipline — bounded (evict oldest), GC-able
(``prune``), atomically persisted, and NEVER raises on a malformed on-disk entry.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    PerStageAuthorizationVerdict,
)
from prsm.settlement.per_stage_routing import (
    PerStageSettlementTask,
    verify_routed_settlement_task,
)
from prsm.settlement.state_store import SettlementStateStore

logger = logging.getLogger(__name__)

_STORE_VERSION = 1


@dataclass(frozen=True)
class StagedSettlementTask:
    """A routed task this node ACCEPTED (its membership was authorized), staged pending the
    node's own on-chain commit (S3b-3). Holds the task + the full payee set it was verified
    against (the commit re-verifies the identical set) + ``staged_at`` for GC/eviction order."""

    task: PerStageSettlementTask
    payees: List[Tuple[str, int]]
    staged_at: int

    @property
    def local_escrow_id(self) -> str:
        return self.task.batched_receipt.local_escrow_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            # list-of-pairs → list-of-lists for JSON; share as str (may exceed JSON-safe int)
            "payees": [[p, str(s)] for p, s in self.payees],
            "staged_at": self.staged_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StagedSettlementTask":
        return cls(
            task=PerStageSettlementTask.from_dict(d["task"]),
            payees=[(str(p), int(s)) for p, s in d["payees"]],
            staged_at=int(d["staged_at"]),
        )


class PerStageReceiverStore:
    """Bounded, GC-able, atomically-persisted store of staged (authorized) per-stage tasks.

    Keyed by the per-node ``local_escrow_id`` (``{job_id}::stage::{node_id}``). Insertion order is
    eviction order (oldest-first). Idempotent: re-staging an existing key REFRESHES it (never adds
    a second copy), so a duplicate delivery can't double-stage. Persistence mirrors
    ``SettlementStateStore`` (atomic .tmp write + fsync + os.replace). Never raises on a
    malformed/unparseable on-disk entry (it is skipped + logged) — but note: unlike issued-auth
    retention, a LOST stage means an unpaid stage (a safe failure — no double / wrong pay)."""

    def __init__(
        self,
        path: Any,
        *,
        max_entries: int = 100_000,
        now: Callable[[], float] = time.time,
    ):
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive (got {max_entries})")
        self._path = Path(path)
        self._max_entries = max_entries
        self._now = now
        self._store = SettlementStateStore(self._path)
        self._staged: "OrderedDict[str, StagedSettlementTask]" = OrderedDict()
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    # ── public API ────────────────────────────────────────────────

    def stage(
        self, task: PerStageSettlementTask, payees: List[Tuple[str, int]],
    ) -> StagedSettlementTask:
        """Retain an AUTHORIZED task, keyed by its ``local_escrow_id``. Re-staging an existing
        key refreshes it (moved to newest, NOT duplicated). Evicts the oldest if over
        ``max_entries``. Persisted atomically. Returns the stored record.

        Callers should stage ONLY a task whose membership ``verify_routed_settlement_task``
        authorized — ``ingest_routed_task`` enforces that gate."""
        rec = StagedSettlementTask(
            task=task, payees=list(payees), staged_at=int(self._now()))
        key = rec.local_escrow_id
        if not key:
            raise ValueError(
                "cannot stage a task whose batched_receipt has no local_escrow_id — the "
                "per-node idempotency key ({job_id}::stage::{node_id}) is required")
        if key in self._staged:
            del self._staged[key]
        self._staged[key] = rec
        while len(self._staged) > self._max_entries:
            evicted_key, _ = self._staged.popitem(last=False)
            logger.debug(
                "PerStageReceiverStore: evicted oldest staged task %s (over max_entries=%d)",
                evicted_key, self._max_entries)
        self._persist()
        return rec

    def get(self, local_escrow_id: str) -> Optional[StagedSettlementTask]:
        return self._staged.get(str(local_escrow_id))

    def all_staged(self) -> List[StagedSettlementTask]:
        """All staged tasks, oldest-first (the order the commit loop should drain them)."""
        return list(self._staged.values())

    def discard(self, local_escrow_id: str) -> bool:
        """Drop a staged task once its commit landed (so the loop won't re-attempt it). Returns
        True iff present + removed. Persists iff removed. Never raises."""
        key = str(local_escrow_id)
        if key not in self._staged:
            return False
        del self._staged[key]
        self._persist()
        return True

    def prune(self, retention_seconds: int) -> int:
        """Drop every entry whose ``staged_at + retention_seconds < now``. Returns the number
        dropped. Persists iff anything dropped."""
        cutoff = self._now() - retention_seconds
        expired = [k for k, t in self._staged.items() if t.staged_at < cutoff]
        for k in expired:
            del self._staged[k]
        if expired:
            self._persist()
        return len(expired)

    # ── persistence (mirrors IssuedAuthorizationStore) ────────────

    def _persist(self) -> None:
        state = {
            "version": _STORE_VERSION,
            "staged": {k: t.to_dict() for k, t in self._staged.items()},
        }
        try:
            self._store.save(state)
        except Exception as exc:  # pragma: no cover - disk-failure path
            logger.warning(
                "PerStageReceiverStore: persist to %s failed (%s); staged tasks in-memory "
                "only this run", self._path, exc)

    def _load(self) -> None:
        try:
            raw = self._store.load()
        except Exception as exc:
            logger.warning(
                "PerStageReceiverStore: state at %s unreadable (%s); starting empty",
                self._path, exc)
            return
        if raw is None:
            return
        staged = raw.get("staged")
        if not isinstance(staged, dict):
            logger.warning(
                "PerStageReceiverStore: %s has no 'staged' object; starting empty", self._path)
            return
        for key, entry in staged.items():
            try:
                rec = StagedSettlementTask.from_dict(entry)
            except Exception as exc:
                logger.warning(
                    "PerStageReceiverStore: skipping malformed entry %s in %s (%s)",
                    key, self._path, exc)
                continue
            # Re-key by the record's own local_escrow_id (authoritative), not the raw map key.
            lid = rec.local_escrow_id
            if lid:
                self._staged[lid] = rec


# ── the node-side ingest (gate-then-stage) ───────────────────────────


@dataclass(frozen=True)
class TaskIngestResult:
    """Outcome of ingesting one routed task. ``accepted`` is True ONLY when the sp1316 gate
    authorized the node's membership AND the task was staged; otherwise False with the gate's
    ``reason`` and nothing staged (FAIL-CLOSED)."""

    accepted: bool
    reason: str
    local_escrow_id: Optional[str] = None


def ingest_routed_task(
    store: PerStageReceiverStore,
    task: PerStageSettlementTask,
    *,
    payees: List[Tuple[str, int]],
    my_node_id: Optional[str] = None,
    chain_id: int = DEFAULT_CHAIN_ID,
    now_unix: Optional[float] = None,
) -> TaskIngestResult:
    """Node-side receive: FAIL-CLOSED gate (sp1316) then STAGE on authorize.

    Steps (any failure → ``accepted=False``, NOTHING staged):
      - If ``my_node_id`` is given, confirm the task is addressed to THIS node (a misrouted task
        is rejected — a node never stages another node's share).
      - Run ``verify_routed_settlement_task`` (signer / set-hash / membership / cap / expiry /
        request-binding). Only an authorized verdict proceeds.
      - Stage the authorized task (idempotent on ``local_escrow_id``).

    NEVER raises on a logical reject. Returns a ``TaskIngestResult``; the node's commit loop
    (S3b-3) drains ``store.all_staged()``."""
    if my_node_id is not None and task.node_id != my_node_id:
        return TaskIngestResult(
            False,
            f"routed task is addressed to node {task.node_id}, not this node {my_node_id} "
            f"(misrouted — refusing to stage another node's share)")

    verdict: PerStageAuthorizationVerdict = verify_routed_settlement_task(
        task, payees=payees, chain_id=chain_id, now_unix=now_unix)
    if not verdict.authorized:
        return TaskIngestResult(False, verdict.reason)

    try:
        rec = store.stage(task, payees)
    except Exception as exc:  # noqa: BLE001 — a stage failure rejects (never a silent accept)
        logger.warning(
            "ingest_routed_task: staging authorized task failed (%s: %s) — rejecting "
            "(no commit will follow)", type(exc).__name__, exc)
        return TaskIngestResult(False, f"stage failed: {type(exc).__name__}: {exc}")
    return TaskIngestResult(True, "authorized+staged", local_escrow_id=rec.local_escrow_id)


def parse_delivery_request(
    body: Any,
) -> Tuple[PerStageSettlementTask, List[Tuple[str, int]]]:
    """Parse a ``POST /settlement/per-stage-task`` body ``{"task": {...}, "payees": [[addr, share],
    ...]}`` into a ``(PerStageSettlementTask, payees)`` pair for ``ingest_routed_task``.

    Raises ``ValueError`` on ANY malformed shape (caller maps to HTTP 422) — a delivery the node
    can't parse is rejected, never silently accepted."""
    if not isinstance(body, dict):
        raise ValueError("delivery body must be a JSON object")
    task_raw = body.get("task")
    payees_raw = body.get("payees")
    if not isinstance(task_raw, dict):
        raise ValueError("delivery body missing a 'task' object")
    if not isinstance(payees_raw, list):
        raise ValueError("delivery body missing a 'payees' list")
    try:
        task = PerStageSettlementTask.from_dict(task_raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"malformed task: {type(exc).__name__}: {exc}")
    payees: List[Tuple[str, int]] = []
    for item in payees_raw:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError("each payee must be a [address, share_wei] pair")
        try:
            payees.append((str(item[0]), int(item[1])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed payee {item!r}: {exc}")
    return task, payees


# ── the node-side commit driver (S3b-3a: stage → commit → discard) ────


@dataclass(frozen=True)
class StageCommitResult:
    """Outcome of committing ONE staged task on this node's own settlement client.
    ``committed`` is True iff the client landed an on-chain commit for the task's share-batch this
    pass; ``reason`` carries the gate/commit detail. ``committed_batch`` is the client's record
    (opaque here) when one landed. Money-safe either way — a non-commit leaves the share unsettled
    (retained for retry), never an over/under/double pay."""

    node_id: str
    local_escrow_id: str
    committed: bool
    reason: str
    committed_batch: Optional[Any] = None


async def commit_staged_task(
    staged: StagedSettlementTask,
    *,
    client: Any,
    chain_id: int = DEFAULT_CHAIN_ID,
    now_unix: Optional[float] = None,
) -> StageCommitResult:
    """Commit ONE node's staged share-batch on its OWN ``BatchSettlementClient``
    (``msg.sender == node == provider`` in Design A).

    RE-VERIFIES membership against the STORED FULL payee set (``staged.payees``) at commit time —
    re-running the sp1316 gate (signer / set-hash / membership / cap / EXPIRY / request-binding),
    so an auth that expired between stage + commit is caught. NOTE: this differs from the
    bench-oriented ``per_stage_commit.commit_per_node_share_batches``, which reconstructs the
    payee set from the shares it holds (the all-clients-in-one-process model). A node that holds
    ONLY its own share must verify against the stored full set, else the set-hash check fails.

    Then accumulates the task's node-signed ``BatchedReceipt`` into ``client`` and drives
    ``commit_ready_batches``. FAIL-SOFT: a gate reject / client bind-mismatch / commit error
    returns ``committed=False`` with the reason (NEVER raises, NEVER a partial/unauthorized
    commit)."""
    task = staged.task
    lid = staged.local_escrow_id
    verdict: PerStageAuthorizationVerdict = verify_routed_settlement_task(
        task, payees=staged.payees, chain_id=chain_id, now_unix=now_unix)
    if not verdict.authorized:
        return StageCommitResult(task.node_id, lid, False, f"auth re-check failed: {verdict.reason}")
    try:
        await client.accumulate(task.batched_receipt)
        records = await client.commit_ready_batches()
    except Exception as exc:  # noqa: BLE001 — fail-soft: the share stays unsettled (retryable)
        logger.warning(
            "commit_staged_task: node %s commit error (%s: %s) — share %s NOT settled "
            "(retained for retry; no double-settle)",
            task.node_id, type(exc).__name__, exc, lid)
        return StageCommitResult(task.node_id, lid, False, f"commit error: {type(exc).__name__}: {exc}")
    if records:
        return StageCommitResult(task.node_id, lid, True, "committed", committed_batch=records[0])
    return StageCommitResult(
        task.node_id, lid, False,
        "no batch committed this pass (retained/quarantined by the client)")


async def drain_and_commit_staged(
    store: PerStageReceiverStore,
    *,
    client_for_node: Callable[[str], Any],
    chain_id: int = DEFAULT_CHAIN_ID,
    now_unix: Optional[float] = None,
    discard_committed: bool = True,
) -> List[StageCommitResult]:
    """Drain the staged tasks (oldest-first) and commit each on its node's own client.

    Each task is committed via ``commit_staged_task``; on a landed commit it is DISCARDED from the
    store (``discard_committed=True``) so the next drain won't re-attempt it. A task that did NOT
    commit STAYS staged (retryable next drain). NEVER raises — a per-task failure is isolated in its
    result.

    sp1431 (money-path audit #2) — the discard is the ONLY idempotency for a committed task; the
    accumulator's ``seen_escrow_ids`` is NOT a durable cross-batch backstop (it resets when the
    batch pops). So a crash AFTER the on-chain commit lands but BEFORE this discard's atomic persist
    completes leaves the staged task on disk with no client-side marker, and the next drain
    re-commits the same receipt as a distinct on-chain batchId → double-settle. The durable fix (a
    committed-escrow-id ledger, or an idempotency-token check against the client's _tracked before
    re-committing) is deferred — this path is gated OFF (PRSM_MULTISTAGE_SETTLEMENT + a funded
    per-stage settler key, both currently unset), so the double-settle is latent, not live."""
    results: List[StageCommitResult] = []
    for staged in store.all_staged():
        try:
            client = client_for_node(staged.task.node_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("drain_and_commit_staged: no client for node %s (%s) — skipped",
                           staged.task.node_id, exc)
            results.append(StageCommitResult(
                staged.task.node_id, staged.local_escrow_id, False, f"no client: {exc}"))
            continue
        res = await commit_staged_task(
            staged, client=client, chain_id=chain_id, now_unix=now_unix)
        if res.committed and discard_committed:
            store.discard(res.local_escrow_id)
        results.append(res)
    return results
