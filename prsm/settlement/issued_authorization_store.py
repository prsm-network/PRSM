"""Sprint 1146 — requester-side issued-authorization store + CONSERVATIVE
NO_ESCROW matcher (the REQUESTER self-dispute foundation).

NO_ESCROW (reason code 2) is NOT a third-party fraud detector. On chain
``_handleNoEscrow`` only checks ``msg.sender == batch.requester``: the requester R
attests they never authorized this batch committed against their escrow. A malicious
provider can ``commitBatch`` naming R as requester WITHOUT a valid PaymentAuthorization
(commitBatch does not verify the auth — that is an off-chain check), and at finalize
it would drain R's escrow. R's defense is a NO_ESCROW challenge, but first R needs a
LOCAL way to recognize an unauthorized batch: compare committed batches where
``requester == me`` against the auths R actually issued.

This module is that foundation. It has three pieces, all PURE (no network, no money):

  1. ``IssuedAuthorizationStore`` — retains the auths R issues (the signed payload +
     an OPTIONAL ``linked_job_id`` + ``issued_at``), keyed by ``job_nonce``. Mirrors
     ``PublishedBatchStore`` discipline: bounded (evict oldest), GC-able (``prune``),
     atomically persisted via ``SettlementStateStore``, and NEVER raises on a
     malformed/unparseable on-disk entry (retention guards no funds).

  2. ``build_and_record_payment_authorization`` — a thin, opt-in wrapper over
     ``build_payment_authorization``. ``store=None`` (default) is a pure pass-through
     (byte-identical auth, nothing recorded); a store also records the issued auth
     best-effort (a record failure never breaks the build). Mirrors sp1141's
     ``adapt_and_retain_inference_receipt``.

  3. ``match_unauthorized_batches`` — the CONSERVATIVE matcher. For each committed
     batch where ``requester == my_address`` it classifies AUTHORIZED vs UNAUTHORIZED
     (a NO_ESCROW candidate).

CONSERVATIVE BIAS (the critical soundness property):
  The matcher must NEVER classify a genuinely-AUTHORIZED batch as UNAUTHORIZED. A false
  NO_ESCROW candidate would lead R to dispute a batch they DID authorize — griefing an
  honest provider and wasting R's gas. So WHEN IN DOUBT, classify AUTHORIZED. It is
  ACCEPTABLE to MISS some unauthorized batches (a false-negative is safe — R simply does
  not dispute that batch); it is NOT acceptable to falsely flag an authorized one. A
  batch is UNAUTHORIZED only when NO plausible auth covers it: no recorded auth to that
  provider, OR every recorded auth to that provider has ``max_spend_wei`` below the
  batch's total value, OR was already expired at the batch's ``commit_timestamp`` (and
  no other recorded auth covers it). A recorded auth whose ``linked_job_id`` keccak
  matches one of the batch's leaf ``job_id_hash`` preimages is an EXACT match (only
  available when leaf job-id hashes are supplied with the batch).

This module touches NO commit/finalize/money path. It only reads what R already has:
the auths R signed and the committed batches R observed on chain.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from eth_utils import keccak

from prsm.settlement.payment_client import (
    build_payment_authorization,
    build_per_stage_payment_authorization,
)
from prsm.settlement.state_store import SettlementStateStore

logger = logging.getLogger(__name__)

_STORE_VERSION = 1

# Classification labels.
AUTHORIZED = "AUTHORIZED"
UNAUTHORIZED = "UNAUTHORIZED"

# match_basis labels.
_BASIS_EXACT = "exact-job-link"
_BASIS_HEURISTIC = "provider-value-heuristic"
_BASIS_NONE = "none"


# ── the retained issued authorization ────────────────────────────────


@dataclass(frozen=True)
class IssuedAuthorization:
    """An authorization the requester issued, retained for NO_ESCROW matching.

    Holds the load-bearing PaymentAuthorization payload fields (so the matcher can
    test provider/value/expiry without re-deriving them) plus an OPTIONAL
    ``linked_job_id`` (best-effort: the job_id the resulting receipt carried, enabling
    an EXACT leaf match) and ``issued_at`` (for GC / eviction ordering)."""
    requester: str
    provider: str
    max_spend_wei: int
    job_nonce: str
    expiry_unix: int
    request_hash: str
    linked_job_id: Optional[str]
    issued_at: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requester": self.requester,
            "provider": self.provider,
            # str: a wei value can exceed the JSON safe-integer range
            "max_spend_wei": str(self.max_spend_wei),
            "job_nonce": self.job_nonce,
            "expiry_unix": self.expiry_unix,
            "request_hash": self.request_hash,
            "linked_job_id": self.linked_job_id,
            "issued_at": self.issued_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IssuedAuthorization":
        """Inverse of to_dict. Raises on a malformed entry (the caller catches +
        skips per the store's never-raise-on-load contract)."""
        linked = d.get("linked_job_id")
        if linked is not None and not isinstance(linked, str):
            raise ValueError("linked_job_id must be a string or null")
        return cls(
            requester=str(d["requester"]),
            provider=str(d["provider"]),
            max_spend_wei=int(d["max_spend_wei"]),
            job_nonce=str(d["job_nonce"]),
            expiry_unix=int(d["expiry_unix"]),
            request_hash=str(d["request_hash"]),
            linked_job_id=linked,
            issued_at=int(d["issued_at"]),
        )


def _issued_from_auth(
    auth: Dict[str, Any], *, linked_job_id: Optional[str], issued_at: int,
) -> IssuedAuthorization:
    """Build an IssuedAuthorization from a ``build_payment_authorization`` result
    (``{"payload": {...}, "signature": "0x.."}``). Raises on a malformed payload."""
    payload = auth["payload"]
    return IssuedAuthorization(
        requester=str(payload["requester"]),
        provider=str(payload["provider"]),
        max_spend_wei=int(payload["max_spend_wei"]),
        job_nonce=str(payload["job_nonce"]),
        expiry_unix=int(payload["expiry_unix"]),
        request_hash=str(payload["request_hash"]),
        linked_job_id=linked_job_id,
        issued_at=int(issued_at),
    )


# ── the store (mirrors PublishedBatchStore) ──────────────────────────


class IssuedAuthorizationStore:
    """Bounded, GC-able, atomically-persisted store of issued authorizations.

    Keyed by ``job_nonce`` (the auth's anti-replay nonce — unique per job). Insertion
    order is the eviction order (oldest-first), so ``max_entries`` overflow evicts the
    oldest issued auth. Persistence mirrors ``SettlementStateStore`` (atomic .tmp write
    + fsync + os.replace) via that store.

    Never raises on a malformed/unparseable on-disk entry: such an entry is skipped +
    logged on load. A corrupt JSON *file* is treated the same way (start empty + log)
    — issued-auth retention guards no money, so it must never block the requester."""

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
        # OrderedDict[job_nonce -> IssuedAuthorization], insertion-ordered (oldest
        # first) so popitem(last=False) evicts the oldest.
        self._auths: "OrderedDict[str, IssuedAuthorization]" = OrderedDict()
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    # ── public API ────────────────────────────────────────────────

    def record(
        self,
        auth_payload: Dict[str, Any],
        linked_job_id: Optional[str] = None,
    ) -> IssuedAuthorization:
        """Retain the issued auth, keyed by its ``job_nonce``. Re-recording an
        existing nonce refreshes it (moved to newest). Evicts the oldest if over
        ``max_entries``. Persisted atomically. Returns the stored record."""
        rec = _issued_from_auth(
            auth_payload, linked_job_id=linked_job_id, issued_at=int(self._now()),
        )
        key = rec.job_nonce
        if key in self._auths:
            del self._auths[key]
        self._auths[key] = rec
        while len(self._auths) > self._max_entries:
            evicted_key, _ = self._auths.popitem(last=False)
            logger.debug(
                "IssuedAuthorizationStore: evicted oldest auth %s (over "
                "max_entries=%d)", evicted_key, self._max_entries,
            )
        self._persist()
        return rec

    def record_per_stage(
        self,
        payload: Dict[str, Any],
        payees_wei: Sequence[tuple],
        linked_job_id: Optional[str] = None,
    ) -> List[IssuedAuthorization]:
        """sp1450 — retain a PER-STAGE authorization for NO_ESCROW matching.

        The requester signs ONE per-stage auth over a SET of ``(payee, share_wei)`` (payee_set_hash),
        and each stage node commits its OWN on-chain batch (``provider == payee``, ``value_ftns ==
        that payee's share``). The single-payee matcher keys per provider + covers a batch only when
        a retained auth to that provider has ``max_spend_wei >= value`` and is unexpired — so we
        record ONE ``IssuedAuthorization`` PER PAYEE (``provider = payee``, ``max_spend_wei = share``)
        under a per-(job, payee) synthetic nonce. The existing CONSERVATIVE matcher then classifies an
        HONEST per-stage batch AUTHORIZED (value == its payee's recorded share) and an INFLATED or
        FOREIGN one UNAUTHORIZED (value > share, or no entry) — closing the requester-side blindness to
        per-stage batches with NO matcher change. Mirrors ``record``'s keying/eviction/persist."""
        base_nonce = str(payload["job_nonce"])
        request_hash = str(payload["request_hash"])
        expiry = int(payload["expiry_unix"])
        requester = str(payload["requester"])
        now = int(self._now())
        recorded: List[IssuedAuthorization] = []
        for addr, share in payees_wei:
            rec = IssuedAuthorization(
                requester=requester,
                provider=str(addr),
                max_spend_wei=int(share),
                # per-(job, payee) synthetic nonce so the N per-payee entries do not collide on the
                # store's job_nonce key (the matcher never keys on job_nonce — it uses provider/value).
                job_nonce=f"{base_nonce}#{str(addr).lower()}",
                expiry_unix=expiry,
                request_hash=request_hash,
                linked_job_id=linked_job_id,
                issued_at=now,
            )
            key = rec.job_nonce
            if key in self._auths:
                del self._auths[key]
            self._auths[key] = rec
            recorded.append(rec)
        while len(self._auths) > self._max_entries:
            evicted_key, _ = self._auths.popitem(last=False)
            logger.debug(
                "IssuedAuthorizationStore: evicted oldest auth %s (over max_entries=%d)",
                evicted_key, self._max_entries,
            )
        self._persist()
        return recorded

    def get(self, job_nonce: str) -> Optional[IssuedAuthorization]:
        return self._auths.get(str(job_nonce))

    def all_authorizations(self) -> List[IssuedAuthorization]:
        """All retained auths, oldest-first."""
        return list(self._auths.values())

    def authorizations_for_provider(
        self, provider: str,
    ) -> List[IssuedAuthorization]:
        """All retained auths whose ``provider`` matches (case-insensitive — an EVM
        address compares case-insensitively), oldest-first."""
        target = str(provider).lower()
        return [a for a in self._auths.values() if a.provider.lower() == target]

    def link_job(self, job_nonce: str, job_id: str) -> bool:
        """Best-effort: record the ``job_id`` the resulting receipt carried for the
        auth at ``job_nonce`` (enabling a later EXACT leaf match). Returns True iff
        the nonce was found + updated, False otherwise (never raises). Persisted iff
        updated."""
        key = str(job_nonce)
        rec = self._auths.get(key)
        if rec is None:
            return False
        updated = IssuedAuthorization(
            requester=rec.requester,
            provider=rec.provider,
            max_spend_wei=rec.max_spend_wei,
            job_nonce=rec.job_nonce,
            expiry_unix=rec.expiry_unix,
            request_hash=rec.request_hash,
            linked_job_id=str(job_id),
            issued_at=rec.issued_at,
        )
        self._auths[key] = updated
        self._persist()
        return True

    def prune(self, retention_seconds: int) -> int:
        """Drop every entry whose ``issued_at + retention_seconds < now``. Returns
        the number dropped. Persists iff anything was dropped."""
        cutoff = self._now() - retention_seconds
        expired = [k for k, a in self._auths.items() if a.issued_at < cutoff]
        for k in expired:
            del self._auths[k]
        if expired:
            self._persist()
        return len(expired)

    # ── persistence (mirrors PublishedBatchStore / SettlementStateStore) ──

    def _persist(self) -> None:
        state = {
            "version": _STORE_VERSION,
            "authorizations": {
                k: a.to_dict() for k, a in self._auths.items()
            },
        }
        try:
            self._store.save(state)
        except Exception as exc:  # pragma: no cover - disk-failure path
            # Retention is best-effort; a persist failure is logged, never raised.
            logger.warning(
                "IssuedAuthorizationStore: persist to %s failed (%s); retention "
                "in-memory only this run", self._path, exc,
            )

    def _load(self) -> None:
        try:
            raw = self._store.load()
        except Exception as exc:
            logger.warning(
                "IssuedAuthorizationStore: state at %s unreadable (%s); starting "
                "empty — retention only, no funds at risk", self._path, exc,
            )
            return
        if raw is None:
            return
        auths = raw.get("authorizations")
        if not isinstance(auths, dict):
            logger.warning(
                "IssuedAuthorizationStore: %s has no 'authorizations' object; "
                "starting empty", self._path,
            )
            return
        for key, entry in auths.items():
            try:
                rec = IssuedAuthorization.from_dict(entry)
            except Exception as exc:
                logger.warning(
                    "IssuedAuthorizationStore: skipping malformed entry %s in %s "
                    "(%s)", key, self._path, exc,
                )
                continue
            # Re-key by the record's own job_nonce (authoritative), not the raw map
            # key, so a tampered key can't shadow a real entry.
            self._auths[rec.job_nonce] = rec


# ── the opt-in build+record wrapper (mirrors sp1141) ─────────────────


def build_and_record_payment_authorization(
    *,
    store: Optional[IssuedAuthorizationStore] = None,
    linked_job_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build + sign a PaymentAuthorization (the EXISTING money-path call, unchanged)
    and — ADDITIVELY, when a ``store`` is provided — record the issued auth keyed by
    its ``job_nonce``.

    Default-off safety: ``store=None`` (the default) makes this a pure pass-through to
    ``build_payment_authorization`` — the returned ``{"payload", "signature"}`` is
    byte-for-byte identical and NOTHING is recorded.

    Best-effort retention: the record write is wrapped in try/except. A record failure
    (store.record raises, anything) is logged and swallowed — the produced auth is
    ALWAYS returned. Issued-auth retention guards no funds; it must never break the
    auth build."""
    auth = build_payment_authorization(**kwargs)
    if store is not None:
        try:
            store.record(auth, linked_job_id=linked_job_id)
        except Exception as exc:  # noqa: BLE001 — retention must NEVER break the build
            logger.warning(
                "build_and_record_payment_authorization: issued-auth retention "
                "failed (non-fatal; auth stands): %s: %s",
                type(exc).__name__, exc,
            )
    return auth


def build_and_record_per_stage_payment_authorization(
    *,
    store: Optional["IssuedAuthorizationStore"] = None,
    linked_job_id: Optional[str] = None,
    payees: Sequence[tuple],
    **kwargs: Any,
) -> Dict[str, Any]:
    """sp1450 — build + sign a PER-STAGE PaymentAuthorization (unchanged money-path call) and —
    ADDITIVELY, when a ``store`` is provided — record it requester-side for NO_ESCROW matching.

    The per-stage sibling of ``build_and_record_payment_authorization``. WITHOUT this, the requester
    (sdk client ``pay_and_infer_multistage``) signed a per-stage auth via
    ``build_per_stage_payment_authorization`` but recorded NOTHING — so its ``IssuedAuthorizationStore``
    stayed EMPTY for per-stage and the NO_ESCROW matcher (``match_unauthorized_batches``) was BLIND to
    every per-stage batch (the exact bug the single-payee path already fixed — see the sp-history note
    on the single-payee wrapper). It would then either grief every honest stage node (no matching auth
    → UNAUTHORIZED → NO_ESCROW challenge) or, if unscanned, miss an unauthorized/inflated per-stage
    batch that over-draws the requester's (fungible) escrow. Recording the per-stage auth as one entry
    per payee closes it.

    Default-off + best-effort, exactly like the single-payee sibling: ``store=None`` → pure
    pass-through (byte-identical auth, nothing recorded); a record failure is logged + swallowed."""
    auth = build_per_stage_payment_authorization(payees=payees, **kwargs)
    if store is not None:
        try:
            from prsm.settlement.payment_client import per_stage_payees_to_wei
            store.record_per_stage(
                auth["payload"], per_stage_payees_to_wei(payees), linked_job_id=linked_job_id)
        except Exception as exc:  # noqa: BLE001 — retention must NEVER break the build
            logger.warning(
                "build_and_record_per_stage_payment_authorization: per-stage issued-auth retention "
                "failed (non-fatal; auth stands): %s: %s",
                type(exc).__name__, exc,
            )
    return auth


# ── the committed-batch view + classification result ─────────────────


@dataclass(frozen=True)
class CommittedBatchView:
    """The minimal view of a committed batch the matcher needs.

    Adapt a settlement ``CommittedBatch`` (prsm.settlement.client) to this with
    ``committed_batch_to_view``. ``leaf_job_id_hashes`` are the per-leaf
    ``job_id_hash`` values (keccak(utf8(job_id)) — see merkle.batched_receipt_to_leaf);
    supply them ONLY when the leaf preimages/receipts are available (e.g. via the
    sp1135 PublishedBatchStore). When None, only the provider-value heuristic runs."""
    batch_id: bytes
    requester: str
    provider: str
    total_value_wei: int
    commit_timestamp: int
    leaf_job_id_hashes: Optional[Sequence[bytes]] = None


@dataclass(frozen=True)
class BatchAuthorizationClassification:
    """Per-batch NO_ESCROW classification result.

    ``classification`` is AUTHORIZED or UNAUTHORIZED (a NO_ESCROW candidate).
    ``match_basis`` records WHY: ``exact-job-link`` (a recorded auth's linked_job_id
    keccak matched a leaf job_id_hash), ``provider-value-heuristic`` (a recorded auth
    to the provider with max_spend >= value and not-expired at commit), or ``none``
    (no plausible auth — the UNAUTHORIZED case)."""
    batch_id: bytes
    requester: str
    provider: str
    total_value_wei: int
    classification: str
    match_basis: str


def committed_batch_to_view(
    batch: Any, *, leaf_job_id_hashes: Optional[Sequence[bytes]] = None,
) -> CommittedBatchView:
    """Adapt a settlement ``CommittedBatch`` to the matcher's ``CommittedBatchView``.

    ``CommittedBatch.total_value_ftns`` is wei (the on-chain integer value), mapped to
    ``total_value_wei`` here. ``leaf_job_id_hashes`` is optional and, when omitted, the
    exact job-link check is skipped (heuristic-only)."""
    return CommittedBatchView(
        batch_id=bytes(batch.batch_id),
        requester=str(batch.requester_address),
        provider=str(batch.provider_address),
        total_value_wei=int(batch.total_value_ftns),
        commit_timestamp=int(batch.commit_timestamp),
        leaf_job_id_hashes=leaf_job_id_hashes,
    )


# ── the CONSERVATIVE matcher ─────────────────────────────────────────


def _exact_job_link_covers(
    batch: CommittedBatchView, auths: Sequence[IssuedAuthorization],
) -> bool:
    """True iff some recorded auth has a ``linked_job_id`` whose keccak(utf8) equals
    one of the batch's leaf ``job_id_hash`` values. Only meaningful when the batch
    carries leaf job-id hashes (else there is nothing to match against)."""
    if not batch.leaf_job_id_hashes:
        return False
    leaf_set = {bytes(h) for h in batch.leaf_job_id_hashes}
    for a in auths:
        if a.linked_job_id is None:
            continue
        if keccak(text=a.linked_job_id) in leaf_set:
            return True
    return False


def _heuristic_covers(
    batch: CommittedBatchView, auths: Sequence[IssuedAuthorization],
) -> bool:
    """True iff some recorded auth to this provider PLAUSIBLY covers the batch:
    ``max_spend_wei >= total_value_wei`` AND not-expired at commit
    (``expiry_unix >= commit_timestamp``). Conservative: any one covering auth
    suffices."""
    for a in auths:
        if a.max_spend_wei < batch.total_value_wei:
            continue
        if a.expiry_unix < batch.commit_timestamp:
            continue
        return True
    return False


def match_unauthorized_batches(
    batches: Sequence[CommittedBatchView],
    store: IssuedAuthorizationStore,
    *,
    my_address: str,
) -> List[BatchAuthorizationClassification]:
    """Classify each committed batch where ``requester == my_address`` as AUTHORIZED
    or UNAUTHORIZED (a NO_ESCROW candidate). Batches for OTHER requesters are skipped
    (not this requester's concern — R can only self-dispute its own escrow).

    CONSERVATIVE: a batch is UNAUTHORIZED only when NO plausible auth covers it. The
    EXACT job-link match is tried first (strongest signal, works even when value
    exceeds a single auth's max_spend — e.g. multi-job batches); then the
    provider-value-and-expiry heuristic. If either covers the batch -> AUTHORIZED.
    Only when neither does -> UNAUTHORIZED. This biases toward NEVER falsely flagging
    an authorized batch (a false NO_ESCROW would grief an honest provider + waste R's
    gas); missing a genuinely-unauthorized batch is the safe failure (R just doesn't
    dispute it)."""
    me = str(my_address).lower()
    results: List[BatchAuthorizationClassification] = []
    for batch in batches:
        if str(batch.requester).lower() != me:
            continue  # not my escrow — skip
        provider_auths = store.authorizations_for_provider(batch.provider)

        if _exact_job_link_covers(batch, provider_auths):
            basis, classification = _BASIS_EXACT, AUTHORIZED
        elif _heuristic_covers(batch, provider_auths):
            basis, classification = _BASIS_HEURISTIC, AUTHORIZED
        else:
            basis, classification = _BASIS_NONE, UNAUTHORIZED

        results.append(BatchAuthorizationClassification(
            batch_id=bytes(batch.batch_id),
            requester=str(batch.requester),
            provider=str(batch.provider),
            total_value_wei=int(batch.total_value_wei),
            classification=classification,
            match_basis=basis,
        ))
    return results
