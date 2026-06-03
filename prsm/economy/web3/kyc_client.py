"""Sprint 280 — KYC vendor adapter scaffold.

KYC is the regulatory gate for fiat on/off-ramps in US + EU
jurisdictions. PRSM doesn't roll its own KYC — instead this
adapter wraps a third-party vendor (Persona, Onfido, Plaid
Identity Verification, etc.) behind a uniform interface so
swapping vendors is a config change, not a code change.

PENDING_COMMISSION pattern (mirrors WaaS + paymaster + onramp):
when KYC_VENDOR_API_KEY is absent or the configured vendor is
unknown to this adapter, initiate() returns
PENDING_COMMISSION records without hitting any external
vendor. Once commissioned, the dependency-injected backend
calls into the real vendor SDK.

Status state machine:
  NOT_STARTED → INITIATED → PENDING → VERIFIED | REJECTED
  VERIFIED → EXPIRED (after vendor-defined window)
  REJECTED/EXPIRED → re-INITIATED creates fresh vendor session
  PENDING_COMMISSION orthogonal — adapter not yet commissioned

Levels:
  basic     — light KYC (selfie + ID); supports small fiat amts
  enhanced  — full KYC (proof of address + source of funds);
              required for higher fiat limits per local regs

Per Vision §14 "Crypto-UX adoption barrier" mitigation: vendor
flows are hosted (Persona modal, Onfido iframe, etc.) so the
user never installs anything. PRSM tracks the verification
record + uses is_verified(user_id) to gate fiat operations.

Operator env:
  - KYC_VENDOR             — persona | onfido | plaid | mock
  - KYC_VENDOR_API_KEY     — vendor API key
  - PRSM_KYC_STORE_DIR     — opt-in JSON persistence dir
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# Status enum values exported as string constants — easier to
# serialize to JSON / read in logs than a Python Enum.
KYC_STATUS_NOT_STARTED = "NOT_STARTED"
KYC_STATUS_INITIATED = "INITIATED"
KYC_STATUS_PENDING = "PENDING"
KYC_STATUS_VERIFIED = "VERIFIED"
KYC_STATUS_REJECTED = "REJECTED"
KYC_STATUS_EXPIRED = "EXPIRED"
KYC_STATUS_PENDING_COMMISSION = "PENDING_COMMISSION"

_VALID_STATUSES = {
    KYC_STATUS_NOT_STARTED, KYC_STATUS_INITIATED,
    KYC_STATUS_PENDING, KYC_STATUS_VERIFIED,
    KYC_STATUS_REJECTED, KYC_STATUS_EXPIRED,
    KYC_STATUS_PENDING_COMMISSION,
}

_REINITIATABLE_STATUSES = {
    KYC_STATUS_NOT_STARTED, KYC_STATUS_REJECTED,
    KYC_STATUS_EXPIRED, KYC_STATUS_PENDING_COMMISSION,
}

KYC_LEVEL_BASIC = "basic"
KYC_LEVEL_ENHANCED = "enhanced"
_VALID_LEVELS = {KYC_LEVEL_BASIC, KYC_LEVEL_ENHANCED}


class _KYCBackend(Protocol):
    """Dependency-injected vendor backend. Each vendor (Persona,
    Onfido, Plaid) wraps its own SDK behind this protocol so the
    client doesn't care which vendor is wired."""

    def initiate_session(
        self, user_id: str, email: str, level: str,
    ) -> Dict[str, Any]: ...


@dataclass
class KYCRecord:
    user_id: str
    email: str
    vendor: Optional[str]
    vendor_ref: Optional[str]
    session_url: Optional[str]
    level: str
    status: str
    created_at: float
    verified_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KYCRecord":
        return cls(
            user_id=d["user_id"],
            email=d.get("email", ""),
            vendor=d.get("vendor"),
            vendor_ref=d.get("vendor_ref"),
            session_url=d.get("session_url"),
            level=d.get("level", KYC_LEVEL_BASIC),
            status=d.get("status", KYC_STATUS_NOT_STARTED),
            created_at=d.get("created_at", 0.0),
            verified_at=d.get("verified_at", 0.0),
        )


class KYCClient:
    """In-process KYC adapter with pluggable vendor backend."""

    # Adapter-recognized vendor names. Backend implementations
    # live in sibling modules (when shipped); for now the v1
    # scaffold uses dependency injection only.
    SUPPORTED_VENDORS: List[str] = ["persona", "onfido", "plaid"]

    def __init__(
        self,
        vendor: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        persist_dir: Optional[Path] = None,
        backend: Optional[_KYCBackend] = None,
    ) -> None:
        self._vendor = vendor
        self._api_key = api_key
        self._backend = backend
        self._records: Dict[str, KYCRecord] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    @classmethod
    def from_env(
        cls, *, backend: Optional[_KYCBackend] = None,
    ) -> "KYCClient":
        vendor = (os.environ.get("KYC_VENDOR") or "").strip().lower()
        api_key = os.environ.get("KYC_VENDOR_API_KEY") or None
        # Sp860 — persist by default so KYC records survive daemon
        # restarts. Operators opt out via env=":memory:".
        persist_raw = os.environ.get("PRSM_KYC_STORE_DIR")
        if persist_raw == ":memory:":
            persist_dir = None
        elif persist_raw:
            persist_dir = Path(persist_raw)
        else:
            persist_dir = Path.home() / ".prsm" / "kyc-records"
        # If the vendor isn't recognized, drop it back to None
        # so is_commissioned() returns False — operator sees a
        # clear "not commissioned" signal rather than weird
        # half-configured state.
        if vendor and vendor not in cls.SUPPORTED_VENDORS:
            logger.warning(
                "KYCClient: unknown KYC_VENDOR=%r (supported: %s) "
                "— adapter starts uncommissioned.",
                vendor, cls.SUPPORTED_VENDORS,
            )
            vendor = None
        # Sp849 — auto-wire vendor HTTP backend when caller didn't
        # supply one + env is fully populated. Keeps the existing
        # injection seam (tests pass backend= explicitly) while
        # closing the "commissioned but adapter_wired=False" gap
        # for production deployments.
        if backend is None and vendor == "persona" and api_key:
            try:
                from prsm.economy.web3.kyc_persona_backend import (
                    from_env as _persona_from_env,
                )
                backend = _persona_from_env()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "KYCClient: persona backend auto-wire failed "
                    "(falling back to PENDING_COMMISSION): %s",
                    exc,
                )
                backend = None
        return cls(
            vendor=vendor or None,
            api_key=api_key,
            persist_dir=persist_dir,
            backend=backend,
        )

    def is_commissioned(self) -> bool:
        """True iff a recognized vendor + API key are present."""
        return bool(
            self._vendor
            and self._vendor in self.SUPPORTED_VENDORS
            and self._api_key
        )

    def adapter_wired(self) -> bool:
        """True iff a vendor SDK backend has been dependency-injected.

        Orthogonal to ``is_commissioned`` — env vars can be present
        without an adapter (PENDING_COMMISSION on initiate), and a
        test harness can wire a backend without env vars. Sp848
        exposes both signals so operators see honest readiness state.
        """
        return self._backend is not None

    def initiate(
        self, user_id: str, email: str, level: str,
    ) -> KYCRecord:
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not email or not isinstance(email, str):
            raise ValueError("email must be a non-empty string")
        if not level or not isinstance(level, str):
            raise ValueError("level must be a non-empty string")
        if level not in _VALID_LEVELS:
            raise ValueError(
                f"level must be one of {sorted(_VALID_LEVELS)}, "
                f"got {level!r}"
            )

        # Idempotency: if there's already an active session for this user
        # (INITIATED / PENDING / VERIFIED) AT THE SAME LEVEL, return it.
        # sp967: a level CHANGE must always re-initiate, even from VERIFIED —
        # otherwise a VERIFIED basic user can never upgrade to enhanced (the
        # documented upgrade path), permanently capping their AML tier. The
        # re-initiated session reflects the requested level; the user re-verifies
        # at the new level via the vendor flow (their prior-level entitlement
        # does not silently persist under the new in-progress record).
        existing = self._records.get(user_id)
        if (
            existing is not None
            and existing.status not in _REINITIATABLE_STATUSES
            and existing.level == level
        ):
            return existing

        if not self.is_commissioned() or self._backend is None:
            record = KYCRecord(
                user_id=user_id, email=email,
                vendor=self._vendor,
                vendor_ref=None, session_url=None,
                level=level,
                status=KYC_STATUS_PENDING_COMMISSION,
                created_at=time.time(),
            )
            self._records[user_id] = record
            self._write_to_disk(record)
            return record

        try:
            payload = self._backend.initiate_session(
                user_id, email, level,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "KYCClient: backend initiate_session raised "
                "for user_id=%s: %s",
                user_id, exc,
            )
            record = KYCRecord(
                user_id=user_id, email=email,
                vendor=self._vendor,
                vendor_ref=None, session_url=None,
                level=level,
                status=KYC_STATUS_REJECTED,
                created_at=time.time(),
            )
            self._records[user_id] = record
            self._write_to_disk(record)
            return record

        record = KYCRecord(
            user_id=user_id, email=email,
            vendor=self._vendor,
            vendor_ref=payload.get("vendor_ref"),
            session_url=payload.get("session_url"),
            level=level,
            status=payload.get("status", KYC_STATUS_INITIATED),
            created_at=time.time(),
        )
        self._records[user_id] = record
        self._write_to_disk(record)
        return record

    def get_status(self, user_id: str) -> Optional[KYCRecord]:
        return self._records.get(user_id)

    def update_status(
        self, user_id: str, new_status: str,
        *, vendor_ref_update: Optional[str] = None,
    ) -> Optional[KYCRecord]:
        """Transition a user's KYC status. Used by vendor
        webhook handlers (Persona's `inquiry.completed`,
        Onfido's `report.completed`, etc.) and by operator-side
        EXPIRED-detection jobs."""
        if new_status not in _VALID_STATUSES:
            raise ValueError(
                f"new_status must be one of "
                f"{sorted(_VALID_STATUSES)}, got {new_status!r}"
            )
        old = self._records.get(user_id)
        if old is None:
            return None
        verified_at = (
            time.time()
            if new_status == KYC_STATUS_VERIFIED
            else old.verified_at
        )
        updated = KYCRecord(
            user_id=old.user_id, email=old.email,
            vendor=old.vendor,
            vendor_ref=vendor_ref_update or old.vendor_ref,
            session_url=old.session_url,
            level=old.level,
            status=new_status,
            created_at=old.created_at,
            verified_at=verified_at,
        )
        self._records[user_id] = updated
        self._write_to_disk(updated)
        return updated

    def is_verified(self, user_id: str) -> bool:
        """Convenience predicate: True iff the user currently
        holds a VERIFIED record. Used by fiat-flow endpoints to
        gate execute paths."""
        rec = self._records.get(user_id)
        return bool(rec and rec.status == KYC_STATUS_VERIFIED)

    def list_records(self) -> List[KYCRecord]:
        return list(self._records.values())

    # ── Persistence ──────────────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                record = KYCRecord.from_dict(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "KYCClient: skipping corrupt %s: %s",
                    path, exc,
                )
                continue
            self._records[record.user_id] = record

    def _write_to_disk(self, record: KYCRecord) -> None:
        if self._persist_dir is None:
            return
        safe = record.user_id.replace("/", "_").replace("\\", "_")
        path = self._persist_dir / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(record.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "KYCClient: disk write failed for %s: %s",
                record.user_id, exc,
            )
