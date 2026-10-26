"""Sprint 300 — DisclosureIntake + disclosure payout composer.

PRSM does not run a bug bounty program and does not pay for
vulnerability reports. The payout composer and payout bands
below were built for a proposed program (Vision §14
mitigation item 3) that was never launched; their figures are
proposals only, not commitments.

This module gives operators a DIRECT-CONTACT path for
security researchers: sensitive findings, anonymous
submissions, and time-critical disclosures during active
incidents.

Components:
  DisclosureSeverity — severity bands (critical /
                       high / medium / low / informational)
  DisclosureStatus — workflow: received → triaged →
                     confirmed | rejected | duplicate |
                     out_of_scope → awarded (terminal)
  DEFAULT_PAYOUT_BANDS_FTNS — proposed payout per severity
                              (proposal only; never
                              activated)
  DisclosureRecord — persistent record (filesystem-backed,
                     same pattern as sprint 272
                     TakedownNoticeRing)
  DisclosureIntake — submit / update_status / list / get /
                     record_payout_tx
  compose_disclosure_payout_tx — composer-only Safe-uploadable
                                 ERC-20 transfer for awarded
                                 disclosures (reuses sprint 299
                                 encode_erc20_transfer_calldata)

State workflow (one-way; no transitions back to RECEIVED):
  RECEIVED → TRIAGED → CONFIRMED → AWARDED   (terminal)
                    → REJECTED              (terminal)
                    → DUPLICATE             (terminal)
                    → OUT_OF_SCOPE          (terminal)
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from prsm.economy.web3.insurance_fund_tracker import (
    encode_erc20_transfer_calldata,
)

logger = logging.getLogger(__name__)


class DisclosureSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class DisclosureStatus(str, Enum):
    RECEIVED = "received"
    TRIAGED = "triaged"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    AWARDED = "awarded"
    DUPLICATE = "duplicate"
    OUT_OF_SCOPE = "out_of_scope"


_TERMINAL_STATUSES = {
    DisclosureStatus.AWARDED,
    DisclosureStatus.REJECTED,
    DisclosureStatus.DUPLICATE,
    DisclosureStatus.OUT_OF_SCOPE,
}


# Proposed payouts in whole FTNS units, from the Vision §14
# proposal. Proposal only: no payout program was launched and
# PRSM does not pay for vulnerability reports.
DEFAULT_PAYOUT_BANDS_FTNS: Dict[DisclosureSeverity, int] = {
    DisclosureSeverity.CRITICAL: 1_000_000,
    DisclosureSeverity.HIGH: 100_000,
    DisclosureSeverity.MEDIUM: 10_000,
    DisclosureSeverity.LOW: 1_000,
    DisclosureSeverity.INFORMATIONAL: 0,
}


# Size caps. Defense against DoS via large submission
# payloads — the intake endpoint is intentionally permissive
# (security researchers may be anonymous) so payload caps
# are the spam-floor defense.
_MAX_SUMMARY_LEN = 4_000
_MAX_DETAILS_LEN = 256 * 1024  # 256KB
_MAX_AFFECTED_CONTRACTS = 50


@dataclass
class DisclosureRecord:
    disclosure_id: str
    timestamp: float
    severity: DisclosureSeverity
    summary: str
    affected_contracts: List[str]
    researcher_contact: str
    status: DisclosureStatus
    details_b64: str = ""
    triage_notes: str = ""
    payout_ftns: int = 0
    payout_tx_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disclosure_id": self.disclosure_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "summary": self.summary,
            "affected_contracts": list(
                self.affected_contracts,
            ),
            "researcher_contact": self.researcher_contact,
            "status": self.status.value,
            "details_b64": self.details_b64,
            "triage_notes": self.triage_notes,
            "payout_ftns": self.payout_ftns,
            "payout_tx_hash": self.payout_tx_hash,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "DisclosureRecord":
        return cls(
            disclosure_id=d["disclosure_id"],
            timestamp=float(d.get("timestamp", 0.0)),
            severity=DisclosureSeverity(d["severity"]),
            summary=d.get("summary", ""),
            affected_contracts=list(
                d.get("affected_contracts") or [],
            ),
            researcher_contact=d.get(
                "researcher_contact", "",
            ),
            status=DisclosureStatus(d["status"]),
            details_b64=d.get("details_b64", ""),
            triage_notes=d.get("triage_notes", ""),
            payout_ftns=int(d.get("payout_ftns", 0)),
            payout_tx_hash=d.get("payout_tx_hash"),
        )


class DisclosureIntake:
    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        self._records: Dict[str, DisclosureRecord] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    @classmethod
    def from_env(cls) -> "DisclosureIntake":
        raw = os.environ.get("PRSM_DISCLOSURE_INTAKE_DIR")
        persist_dir = Path(raw) if raw else None
        return cls(persist_dir=persist_dir)

    # ── Mutations ────────────────────────────────────────

    def submit(
        self,
        *,
        severity: DisclosureSeverity,
        summary: str,
        affected_contracts: List[str],
        researcher_contact: str,
        details: str = "",
        timestamp: Optional[float] = None,
    ) -> DisclosureRecord:
        if not isinstance(severity, DisclosureSeverity):
            raise ValueError(
                f"severity must be a DisclosureSeverity, "
                f"got {severity!r}"
            )
        if not summary or not isinstance(summary, str):
            raise ValueError("summary must be non-empty")
        if len(summary) > _MAX_SUMMARY_LEN:
            raise ValueError(
                f"summary exceeds {_MAX_SUMMARY_LEN}-char cap"
            )
        if not researcher_contact:
            raise ValueError(
                "researcher_contact must be non-empty "
                "(email / PGP / OnionShare / etc.)"
            )
        if len(details) > _MAX_DETAILS_LEN:
            raise ValueError(
                f"details exceeds {_MAX_DETAILS_LEN}-byte "
                f"cap"
            )
        if len(affected_contracts) > _MAX_AFFECTED_CONTRACTS:
            raise ValueError(
                f"affected_contracts exceeds "
                f"{_MAX_AFFECTED_CONTRACTS}-entry cap"
            )

        import base64
        details_b64 = (
            base64.b64encode(
                details.encode("utf-8"),
            ).decode("ascii")
            if details else ""
        )
        record = DisclosureRecord(
            disclosure_id=str(uuid.uuid4()),
            timestamp=(
                timestamp if timestamp is not None
                else time.time()
            ),
            severity=severity,
            summary=summary,
            affected_contracts=list(affected_contracts),
            researcher_contact=researcher_contact,
            status=DisclosureStatus.RECEIVED,
            details_b64=details_b64,
        )
        self._records[record.disclosure_id] = record
        self._write_to_disk(record)
        return record

    def update_status(
        self,
        disclosure_id: str,
        new_status: DisclosureStatus,
        *,
        triage_notes: Optional[str] = None,
        payout_ftns: Optional[int] = None,
    ) -> DisclosureRecord:
        existing = self._records.get(disclosure_id)
        if existing is None:
            raise ValueError(
                f"disclosure {disclosure_id!r} not found"
            )
        # Terminal states can't change
        if existing.status in _TERMINAL_STATUSES:
            raise ValueError(
                f"disclosure {disclosure_id!r} is in "
                f"terminal status {existing.status.value!r}; "
                f"cannot transition"
            )
        # Can't go back to RECEIVED
        if new_status == DisclosureStatus.RECEIVED:
            raise ValueError(
                f"cannot transition back to RECEIVED "
                f"(workflow integrity)"
            )
        updated = DisclosureRecord(
            disclosure_id=existing.disclosure_id,
            timestamp=existing.timestamp,
            severity=existing.severity,
            summary=existing.summary,
            affected_contracts=existing.affected_contracts,
            researcher_contact=existing.researcher_contact,
            status=new_status,
            details_b64=existing.details_b64,
            triage_notes=(
                triage_notes
                if triage_notes is not None
                else existing.triage_notes
            ),
            payout_ftns=(
                payout_ftns
                if payout_ftns is not None
                else existing.payout_ftns
            ),
            payout_tx_hash=existing.payout_tx_hash,
        )
        self._records[disclosure_id] = updated
        self._write_to_disk(updated)
        return updated

    def record_payout_tx(
        self, disclosure_id: str, *, tx_hash: str,
    ) -> DisclosureRecord:
        """After a Safe-executed payout, record the
        on-chain tx hash for audit trail."""
        existing = self._records.get(disclosure_id)
        if existing is None:
            raise ValueError(
                f"disclosure {disclosure_id!r} not found"
            )
        if not tx_hash:
            raise ValueError("tx_hash must be non-empty")
        updated = DisclosureRecord(
            disclosure_id=existing.disclosure_id,
            timestamp=existing.timestamp,
            severity=existing.severity,
            summary=existing.summary,
            affected_contracts=existing.affected_contracts,
            researcher_contact=existing.researcher_contact,
            status=existing.status,
            details_b64=existing.details_b64,
            triage_notes=existing.triage_notes,
            payout_ftns=existing.payout_ftns,
            payout_tx_hash=tx_hash,
        )
        self._records[disclosure_id] = updated
        self._write_to_disk(updated)
        return updated

    # ── Queries ──────────────────────────────────────────

    def get(
        self, disclosure_id: str,
    ) -> Optional[DisclosureRecord]:
        return self._records.get(disclosure_id)

    def list(
        self,
        *,
        severity: Optional[DisclosureSeverity] = None,
        status: Optional[DisclosureStatus] = None,
    ) -> List[DisclosureRecord]:
        out = list(self._records.values())
        out.sort(key=lambda r: r.timestamp, reverse=True)
        if severity is not None:
            out = [r for r in out if r.severity == severity]
        if status is not None:
            out = [r for r in out if r.status == status]
        return out

    def count(self) -> int:
        return len(self._records)

    # ── Persistence ──────────────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                record = DisclosureRecord.from_dict(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DisclosureIntake: skipping corrupt "
                    "%s: %s", path, exc,
                )
                continue
            self._records[record.disclosure_id] = record

    def _write_to_disk(
        self, record: DisclosureRecord,
    ) -> None:
        if self._persist_dir is None:
            return
        safe = (
            record.disclosure_id
            .replace("/", "_")
            .replace("\\", "_")
        )
        path = self._persist_dir / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(record.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DisclosureIntake: disk write failed for "
                "%s: %s",
                record.disclosure_id, exc,
            )


# ── compose_disclosure_payout_tx (top-level helper) ──────


_PAYOUT_WARNING = (
    "DESTRUCTIVE: this transfer pays FTNS to a disclosure "
    "reporter. "
    "Requires Foundation Safe 2-of-3 hardware multisig "
    "approval. Upload the encoded calldata to the Safe UI; "
    "signers verify the target token address + transfer "
    "selector + recipient + amount before signing. The "
    "disclosure record's payout_ftns must match the amount "
    "in the encoded calldata."
)


def _explorer_url_for_address(
    address: str, chain_id: Optional[int],
) -> Optional[str]:
    if chain_id == 8453:
        return f"https://basescan.org/address/{address}"
    if chain_id == 84532:
        return (
            f"https://sepolia.basescan.org/address/"
            f"{address}"
        )
    return None


def compose_disclosure_payout_tx(
    *,
    intake: DisclosureIntake,
    disclosure_id: str,
    recipient: str,
    ftns_token_address: str,
    chain_id: Optional[int] = None,
) -> Dict[str, Any]:
    record = intake.get(disclosure_id)
    if record is None:
        raise ValueError(
            f"disclosure {disclosure_id!r} not found"
        )
    if record.status != DisclosureStatus.AWARDED:
        raise ValueError(
            f"disclosure {disclosure_id!r} is in status "
            f"{record.status.value!r}; can only compose "
            f"payout for AWARDED disclosures"
        )
    if record.payout_ftns <= 0:
        raise ValueError(
            f"disclosure {disclosure_id!r} has "
            f"payout_ftns=0; nothing to pay out (informational "
            f"disclosures earn recognition, not payment)"
        )

    amount_wei = record.payout_ftns * (10 ** 18)
    # encode_erc20_transfer_calldata validates recipient
    # format + amount > 0; let its errors bubble.
    data = encode_erc20_transfer_calldata(
        recipient, amount_wei,
    )
    return {
        "action": "disclosure_payout",
        "disclosure_id": disclosure_id,
        "to": ftns_token_address,
        "data": data,
        "value": "0",
        "recipient": recipient,
        "amount_wei": str(amount_wei),
        "amount_ftns": record.payout_ftns,
        "severity": record.severity.value,
        "summary": record.summary,
        "chain_id": chain_id,
        "warning": _PAYOUT_WARNING,
        "explorer_url": _explorer_url_for_address(
            ftns_token_address, chain_id,
        ),
        "instructions": (
            "1) Open the Foundation Safe UI; "
            "2) Create a new transaction with the `to` "
            "(FTNS token contract), `data` (ABI-encoded "
            "transfer call), and `value=0` fields above; "
            "3) 2-of-3 hardware signers verify the target "
            "token, recipient address embedded in the "
            "calldata, and amount match the disclosure "
            "record before signing; "
            "4) Execute. After confirmation, call "
            "DisclosureIntake.record_payout_tx() with the "
            "on-chain tx hash to close the audit trail."
        ),
    }
