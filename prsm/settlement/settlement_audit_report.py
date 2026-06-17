"""Sprint 1149 — operator-facing settlement-audit findings surface.

The settlement fraud-defense system is CODE-COMPLETE (all 5 on-chain reason codes —
DOUBLE_SPEND, INVALID_SIGNATURE, NO_ESCROW, EXPIRED, CONSENSUS_MISMATCH — plus the §7
compute-integrity class — detect -> assemble -> dry-run-gate, NEVER broadcast). But the
detected findings were OPERATIONALLY INVISIBLE: ``SettlementAuditScheduler.run_once_tick``
only logged the ``AuditRunResult`` at DEBUG via a ``%s`` repr, even when fraud WAS found.

This module is the READ-ONLY surface that closes that gap WITHOUT touching the
never-broadcast / user-gated invariant:

  - ``AuditFindingRecord`` (frozen) — one operator record per actionable finding, carrying
    HASHES / IDS / INDICES / VERDICTS ONLY (no plaintext content — consistent with the
    data-plane no-content-leak property).
  - ``AuditFindingsSummary`` (frozen) — per-class counts + total + the records +
    ``render()`` (operator-readable text incl. a clear "submit is USER-GATED" pointer).
  - ``summarize_audit_result(result)`` — PURE map of the 5 finding lists into records. For
    §7, ONLY actionable (bound+proven) records are surfaced — never UNBOUND (the ATTACK-A
    defence: a producer cannot mask/fabricate fraud via a substituted key, so an unbound
    receipt is not trusted and must not be shown as a finding).
  - ``finding_key(record)`` — the DEDUP tuple ``(reason, batch_id_hex, target_index)`` so a
    still-PENDING batch re-detected across many scan windows is not duplicated/spammed.
  - ``SettlementAuditFindingsStore`` — bounded (max_entries eviction), atomically persisted
    (mirrors ``PublishedBatchStore`` / ``SettlementStateStore`` discipline), NEVER-RAISES on
    a persist/load error; ``record_summary`` dedups by ``finding_key`` across ticks.

NEVER broadcasts / signs / slashes: the summary, the store, and the log are READ-ONLY.
The user-gated ``ChallengeSubmitter.submit`` path is UNCHANGED and untouched by this module.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from prsm.settlement.state_store import SettlementStateStore

logger = logging.getLogger(__name__)

__all__ = [
    "AuditFindingRecord",
    "AuditFindingsSummary",
    "SettlementAuditFindingsStore",
    "finding_key",
    "summarize_audit_result",
    "summarize_records",
]

_REPORT_VERSION = 1

# The five operator-visible reason tags (stable; mirror the detectors' tags).
_REASON_DOUBLE_SPEND = "double_spend"
_REASON_INVALID_SIGNATURE = "invalid_signature"
_REASON_CONSENSUS_MISMATCH = "consensus_mismatch"
_REASON_EXPIRED = "expired"
_REASON_INFERENCE_RECEIPT = "inference_receipt"

_ALL_REASONS = (
    _REASON_DOUBLE_SPEND,
    _REASON_INVALID_SIGNATURE,
    _REASON_CONSENSUS_MISMATCH,
    _REASON_EXPIRED,
    _REASON_INFERENCE_RECEIPT,
)

# Canonical BatchSettlementRegistry.ReasonCode per class (mirrors the on-chain enum:
# DOUBLE_SPEND=0, INVALID_SIGNATURE=1, NO_ESCROW=2, EXPIRED=3, CONSENSUS_MISMATCH=5).
# Hardcoded (not imported) to keep this read-only surface dependency-light — matching the
# inline style already used for invalid_signature — so the operator sees the exact reason
# code to challenge with rather than an unknown/None read off a pure finding that lacks it.
# (§7 inference-receipt grounds are off-chain reputation signals, not a single clean on-chain
# reason code, so they intentionally keep whatever the finding carries — usually None.)
_OCR_DOUBLE_SPEND = 0
_OCR_INVALID_SIGNATURE = 1
_OCR_CONSENSUS_MISMATCH = 5
_OCR_EXPIRED = 3

# The explicit, operator-facing user-gated-submit pointer. Render() always includes this so
# an operator reading the findings knows the surface NEVER broadcasts and that disputing is a
# separate, deliberate action they take with their own key.
_SUBMIT_NOTE = (
    "NOTE: this is a READ-ONLY audit surface — it NEVER broadcasts, signs, or slashes. "
    "Submitting a challenge is USER-GATED: broadcast it separately, with your own key, "
    "via the challenge submitter after reviewing the dry-run verdict."
)


def _hex(b: Any) -> Optional[str]:
    """Hex-encode a bytes-like value; None passes through; anything else -> str()."""
    if b is None:
        return None
    if isinstance(b, (bytes, bytearray, memoryview)):
        return bytes(b).hex()
    return str(b)


# ── the operator record ───────────────────────────────────────────────


@dataclass(frozen=True)
class AuditFindingRecord:
    """One operator-visible audit finding.

    Carries ONLY hashes / ids / indices / verdicts — never plaintext content. ``detail`` is
    a dict of the same (hashes/ids/indices/booleans), so the record can be logged + persisted
    without leaking any inference prompt/output content."""

    reason: str
    batch_id_hex: str
    target_index: Optional[int] = None
    on_chain_reason: Optional[int] = None
    dry_run_ok: Optional[bool] = None
    dry_run_reason: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "batch_id_hex": self.batch_id_hex,
            "target_index": self.target_index,
            "on_chain_reason": self.on_chain_reason,
            "dry_run_ok": self.dry_run_ok,
            "dry_run_reason": self.dry_run_reason,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuditFindingRecord":
        reason = str(d["reason"])
        if reason not in _ALL_REASONS:
            raise ValueError(f"unknown reason {reason!r}")
        ti = d.get("target_index")
        ocr = d.get("on_chain_reason")
        detail = d.get("detail") or {}
        if not isinstance(detail, dict):
            raise ValueError("detail must be a dict")
        return cls(
            reason=reason,
            batch_id_hex=str(d["batch_id_hex"]),
            target_index=None if ti is None else int(ti),
            on_chain_reason=None if ocr is None else int(ocr),
            dry_run_ok=d.get("dry_run_ok"),
            dry_run_reason=d.get("dry_run_reason"),
            detail=dict(detail),
        )

    def render_line(self) -> str:
        """A single operator-readable line for this finding."""
        verdict = (
            "unknown"
            if self.dry_run_ok is None
            else ("ok" if self.dry_run_ok else "FAIL")
        )
        bits = [
            f"reason={self.reason}",
            f"batch_id={self.batch_id_hex}",
        ]
        if self.target_index is not None:
            bits.append(f"target_index={self.target_index}")
        if self.on_chain_reason is not None:
            bits.append(f"on_chain_reason={self.on_chain_reason}")
        bits.append(f"dry_run={verdict}")
        if self.dry_run_reason:
            bits.append(f"dry_run_reason={self.dry_run_reason}")
        return "  - " + " ".join(bits)


# ── the dedup key ──────────────────────────────────────────────────────


def finding_key(record: AuditFindingRecord) -> Tuple[str, str, Optional[int]]:
    """The DEDUP tuple: (reason, batch_id_hex, target_index/receipt_index).

    A batch stays PENDING across many scan windows and is re-detected each tick; this key is
    stable across ticks so a re-scan does NOT duplicate/spam the store."""
    return (record.reason, record.batch_id_hex, record.target_index)


# ── the summary ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuditFindingsSummary:
    """Per-class counts + the records + an operator-readable render(). Frozen / pure."""

    records: List[AuditFindingRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.records)

    def _count(self, reason: str) -> int:
        return sum(1 for r in self.records if r.reason == reason)

    @property
    def double_spend_count(self) -> int:
        return self._count(_REASON_DOUBLE_SPEND)

    @property
    def invalid_signature_count(self) -> int:
        return self._count(_REASON_INVALID_SIGNATURE)

    @property
    def consensus_mismatch_count(self) -> int:
        return self._count(_REASON_CONSENSUS_MISMATCH)

    @property
    def expired_count(self) -> int:
        return self._count(_REASON_EXPIRED)

    @property
    def inference_receipt_count(self) -> int:
        return self._count(_REASON_INFERENCE_RECEIPT)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "counts": {
                _REASON_DOUBLE_SPEND: self.double_spend_count,
                _REASON_INVALID_SIGNATURE: self.invalid_signature_count,
                _REASON_CONSENSUS_MISMATCH: self.consensus_mismatch_count,
                _REASON_EXPIRED: self.expired_count,
                _REASON_INFERENCE_RECEIPT: self.inference_receipt_count,
            },
            "records": [r.to_dict() for r in self.records],
        }

    def render(self) -> str:
        """Operator-readable text: a header line with per-class counts, one line per finding
        (batch id, reason, dry-run verdict), and the explicit user-gated-submit note."""
        if self.total == 0:
            return "settlement audit: no dispute candidates"
        header = (
            f"settlement audit found {self.total} dispute candidate(s) "
            f"[double_spend={self.double_spend_count} "
            f"invalid_signature={self.invalid_signature_count} "
            f"consensus_mismatch={self.consensus_mismatch_count} "
            f"expired={self.expired_count} "
            f"inference_receipt={self.inference_receipt_count}]"
        )
        lines = [header]
        for r in self.records:
            lines.append(r.render_line())
        lines.append(_SUBMIT_NOTE)
        return "\n".join(lines)


# ── the pure mapper ────────────────────────────────────────────────────


def _double_spend_record(af) -> AuditFindingRecord:
    f = af.finding
    occ = list(getattr(f, "occurrences", []) or [])
    primary_batch = _hex(occ[0].batch_id) if occ else None
    detail = {
        "leaf_hash": _hex(f.leaf_hash),
        "occurrences": [
            {
                "batch_id": _hex(o.batch_id),
                "leaf_index": int(o.leaf_index),
                "provider_address": str(o.provider_address),
            }
            for o in occ
        ],
    }
    return AuditFindingRecord(
        reason=_REASON_DOUBLE_SPEND,
        batch_id_hex=primary_batch or "",
        target_index=(int(occ[0].leaf_index) if occ else None),
        on_chain_reason=_OCR_DOUBLE_SPEND,
        dry_run_ok=af.dry_run_ok,
        dry_run_reason=af.dry_run_reason,
        detail=detail,
    )


def _invalid_sig_record(af) -> AuditFindingRecord:
    return AuditFindingRecord(
        reason=_REASON_INVALID_SIGNATURE,
        batch_id_hex=_hex(af.batch_id) or "",
        target_index=int(af.receipt_index),
        on_chain_reason=_OCR_INVALID_SIGNATURE,
        dry_run_ok=af.dry_run_ok,
        dry_run_reason=af.dry_run_reason,
        detail={"receipt_index": int(af.receipt_index)},
    )


def _consensus_mismatch_record(af) -> AuditFindingRecord:
    f = af.finding
    detail = {
        "consensus_group_id": _hex(f.consensus_group_id),
        "job_id_hash": _hex(f.job_id_hash),
        "shard_index": int(f.shard_index),
        "minority_batch_id": _hex(f.minority_batch_id),
        "minority_leaf_index": int(f.minority_leaf_index),
        "minority_output_hash": _hex(f.minority_output_hash),
        "majority_batch_id": _hex(f.majority_batch_id),
        "majority_leaf_index": int(f.majority_leaf_index),
        "majority_output_hash": _hex(f.majority_output_hash),
    }
    return AuditFindingRecord(
        reason=_REASON_CONSENSUS_MISMATCH,
        batch_id_hex=_hex(f.minority_batch_id) or "",
        target_index=int(f.minority_leaf_index),
        on_chain_reason=_OCR_CONSENSUS_MISMATCH,
        dry_run_ok=af.dry_run_ok,
        dry_run_reason=af.dry_run_reason,
        detail=detail,
    )


def _expired_record(af) -> AuditFindingRecord:
    f = af.finding
    detail = {
        "target_index": int(f.target_index),
        "age": int(f.age),
        "lookback": int(f.lookback),
    }
    return AuditFindingRecord(
        reason=_REASON_EXPIRED,
        batch_id_hex=_hex(f.batch_id) or "",
        target_index=int(f.target_index),
        on_chain_reason=_OCR_EXPIRED,
        dry_run_ok=af.dry_run_ok,
        dry_run_reason=af.dry_run_reason,
        detail=detail,
    )


def _inference_receipt_record(af) -> Optional[AuditFindingRecord]:
    """§7 record — ONLY surface BOUND+proven (actionable). An UNBOUND record (the ATTACK-A
    defence: not trusted, verdict never run) is NEVER surfaced; an unproven bound ground is
    not an actionable finding either."""
    if not getattr(af, "bound", False) or not getattr(af, "proven", False):
        return None
    reason_obj = getattr(af, "reason", None)
    reason_str = getattr(reason_obj, "value", None) or (
        str(reason_obj) if reason_obj is not None else None
    )
    detail = {
        "leaf_hash": _hex(af.leaf_hash),
        "bound": bool(af.bound),
        "proven": bool(af.proven),
    }
    if reason_str is not None:
        detail["ground"] = reason_str
    return AuditFindingRecord(
        reason=_REASON_INFERENCE_RECEIPT,
        batch_id_hex=_hex(af.batch_id) or "",
        target_index=None,
        on_chain_reason=getattr(af, "on_chain_reason", None),
        dry_run_ok=af.dry_run_ok,
        dry_run_reason=af.dry_run_reason,
        detail=detail,
    )


def summarize_records(records: List[AuditFindingRecord]) -> AuditFindingsSummary:
    """PURE wrap of an EXISTING list of records (fresh-built OR persisted) into a summary.

    ``SettlementAuditFindingsStore.all_findings()`` already returns ``AuditFindingRecord``
    objects; this lets an operator surface them via the SAME per-class counts + ``render()``
    so persisted findings render IDENTICALLY to a fresh ``AuditRunResult``. It NEITHER scans
    NOR maps raw findings (that is ``summarize_audit_result``) and NEVER broadcasts/signs —
    it only re-wraps already-built records. A ``None`` is treated as the empty list."""
    return AuditFindingsSummary(records=list(records or []))


def summarize_audit_result(result: Any) -> AuditFindingsSummary:
    """PURE map of an ``AuditRunResult``'s 5 finding lists into operator records.

    §7 surfaces ONLY actionable (bound+proven) records — never UNBOUND. No content leak
    (hashes / ids / indices / verdicts only)."""
    records: List[AuditFindingRecord] = []
    for af in getattr(result, "double_spend_findings", []) or []:
        records.append(_double_spend_record(af))
    for af in getattr(result, "invalid_sig_findings", []) or []:
        records.append(_invalid_sig_record(af))
    for af in getattr(result, "consensus_mismatch_findings", []) or []:
        records.append(_consensus_mismatch_record(af))
    for af in getattr(result, "expired_findings", []) or []:
        records.append(_expired_record(af))
    for af in getattr(result, "inference_receipt_findings", []) or []:
        rec = _inference_receipt_record(af)
        if rec is not None:
            records.append(rec)
    return AuditFindingsSummary(records=records)


# ── the bounded, atomic, never-raising store ────────────────────────────


class SettlementAuditFindingsStore:
    """Bounded, atomically-persisted, never-raising store of operator audit findings.

    Keyed by ``finding_key`` so a still-PENDING batch re-detected across many scan windows is
    recorded ONCE (no spam). Insertion order is the eviction order (oldest-first), so
    ``max_entries`` overflow evicts the oldest finding. Persistence mirrors
    ``PublishedBatchStore`` (atomic .tmp write + fsync + os.replace via SettlementStateStore;
    bytes already hex). NEVER raises on a persist/load error — this is a best-effort
    observability surface that guards no funds and must never break the audit loop."""

    def __init__(
        self,
        path: Any,
        *,
        max_entries: int = 10_000,
        now: Callable[[], float] = time.time,
    ):
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive (got {max_entries})")
        self._path = Path(path)
        self._max_entries = int(max_entries)
        self._now = now
        self._store = SettlementStateStore(self._path)
        # OrderedDict[finding_key -> (AuditFindingRecord, first_seen_ts)], insertion-ordered
        # (oldest first) so popitem(last=False) evicts the oldest.
        self._findings: "OrderedDict[Tuple, Tuple[AuditFindingRecord, float]]" = (
            OrderedDict()
        )
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    # ── public API ────────────────────────────────────────────────

    def record_summary(self, summary: AuditFindingsSummary) -> int:
        """Record every NEW finding in ``summary``, deduped by ``finding_key``. A finding key
        already present is left untouched (its first-seen timestamp is preserved — a re-scan
        of a still-PENDING batch does NOT duplicate/refresh/spam). Returns the count of NEW
        findings added. Persists iff anything was added or evicted. NEVER raises."""
        added = 0
        ts = self._now()
        for rec in summary.records:
            key = finding_key(rec)
            if key in self._findings:
                continue
            self._findings[key] = (rec, ts)
            added += 1
        evicted = False
        while len(self._findings) > self._max_entries:
            self._findings.popitem(last=False)
            evicted = True
        if added or evicted:
            self._persist()
        return added

    def all_findings(self) -> List[AuditFindingRecord]:
        """All retained findings, oldest-first."""
        return [rec for rec, _ts in self._findings.values()]

    def prune(self, retention_seconds: int) -> int:
        """Drop every finding whose first_seen + retention < now. Returns the number dropped.
        Persists iff anything was dropped. NEVER raises."""
        cutoff = self._now() - retention_seconds
        expired = [k for k, (_rec, ts) in self._findings.items() if ts <= cutoff]
        for k in expired:
            del self._findings[k]
        if expired:
            self._persist()
        return len(expired)

    # ── persistence (mirrors PublishedBatchStore discipline) ──────

    def _persist(self) -> None:
        state = {
            "version": _REPORT_VERSION,
            "findings": [
                {"record": rec.to_dict(), "first_seen": ts}
                for rec, ts in self._findings.values()
            ],
        }
        try:
            self._store.save(state)
        except Exception as exc:  # noqa: BLE001 — best-effort; never raise into the caller
            logger.warning(
                "SettlementAuditFindingsStore: persist to %s failed (%s); "
                "findings in-memory only this run", self._path, exc,
            )

    def _load(self) -> None:
        try:
            raw = self._store.load()
        except Exception as exc:  # noqa: BLE001 — a corrupt findings file must not crash
            logger.warning(
                "SettlementAuditFindingsStore: state at %s unreadable (%s); "
                "starting empty (observability only, no funds at risk)",
                self._path, exc,
            )
            return
        if raw is None:
            return
        findings = raw.get("findings")
        if not isinstance(findings, list):
            logger.warning(
                "SettlementAuditFindingsStore: %s has no 'findings' list; starting empty",
                self._path,
            )
            return
        for entry in findings:
            try:
                rec = AuditFindingRecord.from_dict(entry["record"])
                ts = float(entry.get("first_seen", self._now()))
            except Exception as exc:  # noqa: BLE001 — skip a single bad entry, keep the rest
                logger.warning(
                    "SettlementAuditFindingsStore: skipping malformed entry in %s (%s)",
                    self._path, exc,
                )
                continue
            # Re-key by the record's own finding_key (authoritative) so a tampered map
            # key can't shadow a real finding.
            self._findings[finding_key(rec)] = (rec, ts)
