"""Sprint 957 (sp928 on-chain-slash groundwork) — CONSENSUS_MISMATCH evidence log.

When the ComputeResultSampler (sp928 optimistic verification) catches a single
provider returning a fabricated result — its re-executed output disagrees with a
strict majority of independent re-runs — it builds CONSENSUS_MISMATCH evidence
and routes it to an injected ``challenge_sink``. That sink was ``None``: the
evidence was logged once and lost.

An *autonomous* on-chain slash from a single node's re-execution is unsound and
deliberately NOT built (a single node must never be able to slash an honest
provider): ``StakeBond.slash`` is callable only by the immutable on-chain slasher
(the BatchSettlementRegistry), and the one open path, ``challengeReceipt``,
re-verifies the mismatch on-chain via a Merkle proof against a *committed* batch —
which the single-provider pay path (off-chain local-ledger settlement) never
produces. So instead of an abusable direct slash, this records the evidence in a
bounded, opt-in-persistent, operator-reviewable store. A human with the proper
authority (Foundation / governance) reviews accumulated evidence and decides
whether to escalate to the audited on-chain rail. Bonded-status reads only ROUTE
evidence here; they grant no authority and no code path moves stake.

Mirrors the audited ``SlashEventRing`` (slash_event_log.py): bounded ``deque``,
opt-in filesystem persistence via ``persist_dir`` / ``PRSM_CONSENSUS_MISMATCH_LOG_DIR``,
``append``/``recent``/``count``, plus an async ``record(evidence)`` adapter whose
signature matches the sampler's ``challenge_sink`` (``async (dict) -> None``).
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

_REASON = "CONSENSUS_MISMATCH"
_DEFAULT_MAX_ENTRIES = 256


@dataclass(frozen=True)
class ConsensusMismatchEntry:
    """One recorded compute re-execution mismatch (a bonded provider caught
    returning an output that disagreed with the re-run majority)."""
    timestamp: float
    job_id: str
    accused_provider_id: str
    accused_output_hash: str
    majority_output_hash: str
    witness_provider_ids: Tuple[str, ...] = field(default_factory=tuple)
    accused_bonded: bool = False
    accused_stake_wei: int = 0
    accused_operator_address: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "accused_provider_id": self.accused_provider_id,
            "accused_output_hash": self.accused_output_hash,
            "majority_output_hash": self.majority_output_hash,
            "witness_provider_ids": list(self.witness_provider_ids),
            "accused_bonded": self.accused_bonded,
            "accused_stake_wei": self.accused_stake_wei,
            "accused_operator_address": self.accused_operator_address,
        }


class ConsensusMismatchLog:
    """Bounded in-memory ring of ConsensusMismatchEntry, opt-in persistent."""

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
        self._entries: Deque[ConsensusMismatchEntry] = deque(maxlen=max_entries)
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ── persistence (mirrors SlashEventRing) ─────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        loaded: list = []
        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, dict) or "job_id" not in data:
                    raise ValueError("missing job_id")
                loaded.append(data)
            except Exception as exc:  # noqa: BLE001 — corrupt file → skip
                logger.warning(
                    "ConsensusMismatchLog: skipping corrupt file %s: %s",
                    path, exc,
                )
        loaded.sort(key=lambda d: d.get("timestamp", 0))
        for d in loaded:
            try:
                self._entries.append(ConsensusMismatchEntry(
                    timestamp=float(d.get("timestamp", 0.0)),
                    job_id=str(d["job_id"]),
                    accused_provider_id=str(d.get("accused_provider_id", "")),
                    accused_output_hash=str(d.get("accused_output_hash", "")),
                    majority_output_hash=str(d.get("majority_output_hash", "")),
                    witness_provider_ids=tuple(d.get("witness_provider_ids", []) or []),
                    accused_bonded=bool(d.get("accused_bonded", False)),
                    accused_stake_wei=int(d.get("accused_stake_wei", 0) or 0),
                    accused_operator_address=d.get("accused_operator_address"),
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ConsensusMismatchLog: skipping unreconstructable entry: %s",
                    exc,
                )

    def _write_to_disk(self, entry: ConsensusMismatchEntry) -> None:
        if self._persist_dir is None:
            return
        ts_int = int(entry.timestamp)
        # job_id can contain path separators / odd chars; sanitize for filename.
        job_short = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in entry.job_id
        )[:24]
        path = self._persist_dir / f"{ts_int}_{job_short}.json"
        try:
            path.write_text(json.dumps(entry.to_dict()))
        except Exception as exc:  # noqa: BLE001 — fail-soft
            logger.warning(
                "ConsensusMismatchLog: disk write failed (fail-soft): %s", exc,
            )

    # ── append / query ───────────────────────────────────────────────────

    def append(self, entry: ConsensusMismatchEntry) -> None:
        self._entries.append(entry)
        self._write_to_disk(entry)

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        provider: Optional[str] = None,
    ) -> List[ConsensusMismatchEntry]:
        if limit <= 0 or limit > 1000:
            raise ValueError(f"limit must be in [1, 1000], got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        snap = list(self._entries)
        snap.reverse()  # newest-first
        if provider:
            p = provider.lower()
            snap = [e for e in snap if e.accused_provider_id.lower() == p]
        return snap[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)

    # ── challenge_sink adapter (async (evidence: dict) -> None) ──────────

    async def record(self, evidence: Any) -> None:
        """Adapter matching ComputeResultSampler's ``challenge_sink`` signature.
        Validates + records a CONSENSUS_MISMATCH evidence dict. NEVER raises into
        the caller (the sampler) — a malformed payload or a disk error is logged
        and swallowed, so a bookkeeping failure can't break verification."""
        try:
            if not isinstance(evidence, dict):
                return
            if evidence.get("reason") != _REASON:
                return
            job_id = evidence.get("job_id")
            accused = evidence.get("accused_provider_id")
            if not job_id or not accused:
                logger.warning(
                    "ConsensusMismatchLog.record: evidence missing job_id/"
                    "accused_provider_id — ignored: %r", evidence,
                )
                return
            entry = ConsensusMismatchEntry(
                timestamp=time.time(),
                job_id=str(job_id),
                accused_provider_id=str(accused),
                accused_output_hash=str(evidence.get("accused_output_hash", "")),
                majority_output_hash=str(evidence.get("majority_output_hash", "")),
                witness_provider_ids=tuple(
                    evidence.get("witness_provider_ids", []) or []
                ),
                accused_bonded=bool(evidence.get("accused_bonded", False)),
                accused_stake_wei=int(evidence.get("accused_stake_wei", 0) or 0),
                accused_operator_address=evidence.get("accused_operator_address"),
            )
            self.append(entry)
            logger.warning(
                "CONSENSUS_MISMATCH recorded: job=%s accused=%s bonded=%s "
                "stake_wei=%s — review via GET /admin/consensus-mismatch-evidence",
                entry.job_id, entry.accused_provider_id[:12],
                entry.accused_bonded, entry.accused_stake_wei,
            )
        except Exception as exc:  # noqa: BLE001 — never break the sampler
            logger.warning(
                "ConsensusMismatchLog.record failed (swallowed): %s", exc,
            )
