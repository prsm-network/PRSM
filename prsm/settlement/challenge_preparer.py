"""Sprint 1163 — operator challenge-PREPARATION bridge (the findings -> dispute gap).

The settlement fraud-defense pipeline is CODE-COMPLETE end-to-end (detect -> assemble ->
dry-run-gate -> user-gated submit) and, since sp1149/sp1150, operationally VISIBLE: the
audit loop persists dispute candidates into a ``SettlementAuditFindingsStore`` and the
read-only CLI renders them. But that surface was a DEAD END for the operator: it shows
"here are N dispute candidates" and stops. Turning a persisted ``AuditFindingRecord`` into
the concrete on-chain ``challengeReceipt`` inputs meant hand-wiring the right assembler +
re-sourcing the batch's receipts + the submitter's dry-run — there was no command for it.

This module is the bridge. ``prepare_challenge_from_finding`` takes ONE persisted finding +
an injected ``ReceiptLookup`` (batch_id_hex -> ordered ``BatchedReceipt`` set) and dispatches
it to the RIGHT assembler, returning a ``PreparedChallenge`` whose ``to_call_args()`` are the
exact inputs the operator broadcasts. ``dry_run_prepared`` adds the read-only pre-flight.

HARD SECURITY INVARIANTS (this is the money/security path):
  - NEVER broadcasts / signs / spends gas / slashes. The bridge ASSEMBLES (pure) and
    ``dry_run_prepared`` runs a READ-ONLY static eth_call only — it calls the client's
    ``dry_run`` and NEVER its ``submit``. Broadcasting stays the separate, user-gated step.
  - FAIL-CLOSED. A reason this bridge cannot honestly prepare is REFUSED with a clear
    ``UnsupportedChallengeReason`` rather than mis-encoded:
      * ``consensus_mismatch`` (on-chain reason 5) is assembled + broadcast via the
        dedicated marketplace ``consensus_submitter`` — its auxData layout differs from the
        generalized ``challengeReceipt`` reasons, so this bridge does NOT handle it;
      * ``inference_receipt`` (§7) grounds are off-chain reputation signals with no single
        clean on-chain reason code, so they are not preparable as a ``challengeReceipt``.
    The four it DOES prepare are exactly ``ChallengeSubmitter``'s
    ``SUPPORTED_CHALLENGE_REASONS``: INVALID_SIGNATURE(1), DOUBLE_SPEND(0), NO_ESCROW(2),
    EXPIRED(3). (The AUDIT store only ever produces three of them — double_spend,
    invalid_signature, expired; NO_ESCROW is the requester SELF-dispute from sp1147's
    separate path, supported here so a hand-built/requester-side record can be prepared.)
  - The RECEIPT SOURCE is a SEPARATE, injected concern. A persisted finding carries only
    hashes/ids/indices (no receipt preimages — the no-content-leak property), so the
    assemblers' required ordered ``BatchedReceipt`` set must be re-sourced. The lookup is
    pluggable (PublishedBatchStore for the producer's/requester's own batches; a future
    receipt-set-blob source for observer-audited foreign batches) so the bridge stays pure.
    A missing batch is REFUSED (``MissingBatchReceipts``), never assembled against nothing.

The bridge ORCHESTRATES the existing pure assemblers; it does not reimplement merkle /
leaf / auxData logic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import assemble_invalid_signature_challenge
from prsm.settlement.consensus_mismatch_assembler import (
    REASON_CONSENSUS_MISMATCH,
    assemble_consensus_mismatch_challenge,
)
from prsm.settlement.double_spend_assembler import assemble_double_spend_challenge
from prsm.settlement.expired_assembler import assemble_expired_challenge
from prsm.settlement.invalid_signature_submitter import (
    SUPPORTED_CHALLENGE_REASONS,
    AssembledChallenge,
)
from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
from prsm.settlement.no_escrow_assembler import assemble_no_escrow_challenge

# Reasons this bridge can PREPARE: the generalized submitter's four (challengeReceipt ABI) PLUS
# CONSENSUS_MISMATCH (sp1165 — assembled here, dry-run/broadcast via the dedicated
# ConsensusChallengeSubmitter; same challengeReceipt ABI, distinct auxData layout).
_PREPARABLE_REASONS = frozenset(SUPPORTED_CHALLENGE_REASONS | {REASON_CONSENSUS_MISMATCH})

__all__ = [
    "ReceiptLookup",
    "PreparedChallenge",
    "SkippedFinding",
    "UnsupportedChallengeReason",
    "MissingBatchReceipts",
    "prepare_challenge_from_finding",
    "prepare_findings",
    "dry_run_prepared",
    "published_batch_receipt_lookup",
    "composite_receipt_lookup",
]

# The operator-visible reason tags (mirror settlement_audit_report's stable tags). NO_ESCROW
# is NOT produced by the audit loop (it is the requester self-dispute path), but is preparable
# here so a requester-side / hand-built record can be turned into call args uniformly.
_REASON_DOUBLE_SPEND = "double_spend"
_REASON_INVALID_SIGNATURE = "invalid_signature"
_REASON_CONSENSUS_MISMATCH = "consensus_mismatch"
_REASON_EXPIRED = "expired"
_REASON_INFERENCE_RECEIPT = "inference_receipt"
_REASON_NO_ESCROW = "no_escrow"


# A batch_id_hex -> ordered receipts lookup (None when that batch is not retained). Injected
# so the bridge never hard-wires a receipt source.
ReceiptLookup = Callable[[str], Optional[List[BatchedReceipt]]]


class UnsupportedChallengeReason(ValueError):
    """The finding's reason cannot be honestly prepared by this bridge (routed to a
    different submitter, or off-chain). Fail-closed — never mis-encode."""


class MissingBatchReceipts(ValueError):
    """The receipt source has no retained ordered receipts for a batch the assembly needs.
    Fail-closed — never assemble a proof against nothing."""


def _strip0x(s: str) -> str:
    return str(s).lower().removeprefix("0x")


def _leaf_field_hex(v: Any) -> Any:
    """Render one challengeReceipt leaf-tuple field: bytes -> 0x-hex, ints pass through."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "0x" + bytes(v).hex()
    return v


@dataclass(frozen=True)
class PreparedChallenge:
    """An ASSEMBLED challenge ready for the operator to dry-run + broadcast.

    ``challenge`` is the concrete assembler output (``.reason_code`` + ``.to_call_args()``);
    ``to_dict()`` renders the exact ``challengeReceipt`` call args in hex for review.
    Broadcasting it remains a SEPARATE, user-gated step — preparing it here does not."""

    reason: str
    on_chain_reason: int
    batch_id_hex: str
    target_index: int
    challenge: AssembledChallenge

    def to_call_args_hex(self) -> Dict[str, Any]:
        batch_id, leaf_tuple, proof, reason_code, aux = self.challenge.to_call_args()
        return {
            "batch_id": "0x" + bytes(batch_id).hex(),
            "leaf": [_leaf_field_hex(v) for v in leaf_tuple],
            "merkle_proof": ["0x" + bytes(p).hex() for p in proof],
            "reason": int(reason_code),
            "aux_data": "0x" + bytes(aux).hex(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "on_chain_reason": self.on_chain_reason,
            "batch_id_hex": self.batch_id_hex,
            "target_index": self.target_index,
            "call_args": self.to_call_args_hex(),
        }


def _require_receipts(
    receipt_lookup: ReceiptLookup, batch_id_hex: str,
) -> Tuple[bytes, List[BatchedReceipt]]:
    """Resolve the ordered receipts for a batch, fail-closed if absent/empty."""
    receipts = receipt_lookup(batch_id_hex)
    if not receipts:
        raise MissingBatchReceipts(
            f"no retained receipts for batch {batch_id_hex} — the assembler needs the full "
            "ordered receipt set to rebuild the leaf + merkle proof. Provide a receipt "
            "source that retains this batch (PublishedBatchStore for your own/requester "
            "batches; the receipt-set blob for an observer-audited foreign batch)."
        )
    return bytes.fromhex(_strip0x(batch_id_hex)), list(receipts)


# Each ``_prepare_*`` returns ``(challenge, effective_target_index)`` — the index of the leaf
# the assembled challenge ACTUALLY targets, so the operator review surface (and the dry-run
# verdict map key) reflects the genuinely-challenged leaf rather than a top-level field that
# can diverge for a hand-built record.


def _prepare_invalid_signature(record: Any, receipt_lookup: ReceiptLookup):
    batch_id, receipts = _require_receipts(receipt_lookup, record.batch_id_hex)
    ti = int(record.target_index)
    return assemble_invalid_signature_challenge(
        batch_id=batch_id, batch_receipts=receipts, target_index=ti), ti


def _prepare_no_escrow(record: Any, receipt_lookup: ReceiptLookup):
    batch_id, receipts = _require_receipts(receipt_lookup, record.batch_id_hex)
    ti = int(record.target_index)
    return assemble_no_escrow_challenge(
        batch_id=batch_id, batch_receipts=receipts, target_index=ti), ti


def _prepare_expired(record: Any, receipt_lookup: ReceiptLookup, now: Optional[int]):
    lookback = (record.detail if isinstance(record.detail, dict) else {}).get("lookback")
    if lookback is None:
        raise ValueError(
            "expired finding detail lacks 'lookback' — cannot prove expiry without a "
            "trustworthy per-batch lookback window (fail-closed; never grief an honest "
            "provider with a would-revert ReceiptNotExpired tx)"
        )
    batch_id, receipts = _require_receipts(receipt_lookup, record.batch_id_hex)
    now_unix = int(now) if now is not None else int(time.time())
    ti = int(record.target_index)
    return assemble_expired_challenge(
        batch_id=batch_id, batch_receipts=receipts, target_index=ti,
        now=now_unix, lookback_window_seconds=int(lookback)), ti


def _occ_batch_and_index(o: Any) -> Tuple[str, int]:
    """Pull (batch_id, leaf_index) from one occurrence dict, fail-closed as a ValueError on
    any malformed shape (missing key, non-dict, non-int index) — so the batch partitioner
    catches it as a SkippedFinding instead of a KeyError/TypeError escaping the run."""
    if not isinstance(o, dict) or "batch_id" not in o or "leaf_index" not in o:
        raise ValueError(
            f"malformed double_spend occurrence (need a dict with 'batch_id' + "
            f"'leaf_index'): {o!r}"
        )
    try:
        return str(o["batch_id"]), int(o["leaf_index"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed double_spend occurrence ({exc}): {o!r}") from exc


def _prepare_double_spend(record: Any, receipt_lookup: ReceiptLookup):
    occ = list(
        (record.detail if isinstance(record.detail, dict) else {}).get("occurrences") or [])
    if len(occ) < 2:
        raise ValueError(
            "double_spend finding has fewer than two occurrences — cannot identify a "
            "target + a distinct conflicting batch to prove the double-commit"
        )
    target_batch, target_leaf = _occ_batch_and_index(occ[0])
    conflicting = None
    for o in occ[1:]:
        c_batch, c_leaf = _occ_batch_and_index(o)
        if _strip0x(c_batch) != _strip0x(target_batch):
            conflicting = (c_batch, c_leaf)
            break
    if conflicting is None:
        # All occurrences in ONE batch — the conflicting-batch auxData layout can't express
        # an intra-batch double-commit (mirrors double_spend_assembler's own guard).
        raise UnsupportedChallengeReason(
            "double_spend occurrences are all within ONE batch — an intra-batch "
            "double-commit is not expressible via the conflicting-batch auxData layout"
        )
    conflicting_batch, conflicting_leaf = conflicting

    t_id, t_receipts = _require_receipts(receipt_lookup, target_batch)
    c_id, c_receipts = _require_receipts(receipt_lookup, conflicting_batch)
    # The conflicting side needs only the ordered leaf HASHES; derive them from the
    # conflicting batch's retained receipts (PublishedBatch keeps receipts, not a separate
    # leaf-hash list, and order is preserved end-to-end == the order the root folds over).
    conflicting_leaf_hashes = [
        hash_leaf(batched_receipt_to_leaf(br)) for br in c_receipts
    ]
    challenge = assemble_double_spend_challenge(
        target_batch_id=t_id,
        target_batch_receipts=t_receipts,
        target_index=target_leaf,
        conflicting_batch_id=c_id,
        conflicting_leaf_hashes=conflicting_leaf_hashes,
        conflicting_index=conflicting_leaf,
    )
    # effective target index = the leaf ACTUALLY challenged (the target occurrence's index),
    # which may differ from record.target_index for a hand-built record.
    return challenge, target_leaf


def _prepare_consensus_mismatch(record: Any, receipt_lookup: ReceiptLookup):
    """Assemble a CONSENSUS_MISMATCH challenge from the finding's minority+majority coordinates
    (settlement_audit_report._consensus_mismatch_record's detail shape) + both batches' receipts.
    Fail-closed (ValueError) on missing/malformed detail so prepare_findings partitions it."""
    d = record.detail if isinstance(record.detail, dict) else {}
    # sp1170 money-safety: refuse to auto-prepare a consensus finding unless the off-chain
    # majority determination EXPLICITLY says it is non-ambiguous. FAIL-CLOSED: a finding that is
    # ambiguous (no strict majority — a tie/split, honest side undeterminable) OR that OMITS the
    # determination entirely (``ambiguous`` defaults to True here) is MANUAL-review-only —
    # auto-slashing it could slash an honest provider, and the on-chain check is symmetric so it
    # would not protect us. The detector's _consensus_mismatch_record always sets the flag; only
    # a hand-built / legacy record could omit it, and it must declare ambiguous=False to prepare.
    if d.get("ambiguous", True):
        raise UnsupportedChallengeReason(
            "consensus_mismatch finding is AMBIGUOUS or lacks a majority determination (no "
            "confirmed strict-majority output); the honest side is undeterminable off-chain, so "
            "it cannot be auto-prepared into a slash. Adjudicate it manually (see the vote_tally) "
            "before broadcasting — a record must declare ambiguous=False to be auto-prepared."
        )
    min_bid = d.get("minority_batch_id")
    maj_bid = d.get("majority_batch_id")
    if not min_bid or not maj_bid:
        raise ValueError(
            "consensus_mismatch finding detail lacks minority_batch_id / majority_batch_id — "
            "cannot identify the minority (challenged) + majority (conflicting) batches"
        )
    try:
        min_idx = int(d["minority_leaf_index"])
        maj_idx = int(d["majority_leaf_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"consensus_mismatch finding detail has malformed leaf indices ({exc})"
        ) from exc
    min_id, min_receipts = _require_receipts(receipt_lookup, min_bid)
    maj_id, maj_receipts = _require_receipts(receipt_lookup, maj_bid)
    challenge = assemble_consensus_mismatch_challenge(
        minority_batch_id=min_id, minority_receipts=min_receipts, minority_index=min_idx,
        majority_batch_id=maj_id, majority_receipts=maj_receipts, majority_index=maj_idx,
    )
    # effective target index = the minority (challenged) leaf index.
    return challenge, min_idx


def prepare_challenge_from_finding(
    *,
    record: Any,
    receipt_lookup: ReceiptLookup,
    now: Optional[int] = None,
) -> PreparedChallenge:
    """Turn ONE persisted ``AuditFindingRecord`` into a ``PreparedChallenge``.

    ``record`` is duck-typed (``.reason``, ``.batch_id_hex``, ``.target_index``, ``.detail``)
    so either an ``AuditFindingRecord`` or an equivalent struct works. ``receipt_lookup``
    re-sources the ordered ``BatchedReceipt`` set the chosen assembler needs. ``now`` (unix
    seconds) is used ONLY by the EXPIRED path (defaults to the current time).

    Raises ``UnsupportedChallengeReason`` for a reason this bridge cannot honestly prepare
    (consensus_mismatch / inference_receipt / unknown), ``MissingBatchReceipts`` when the
    source lacks a needed batch, and ``ValueError`` from the underlying assembler's fail-fast
    (e.g. the target signature actually verifies, the receipt is not actually expired).

    NEVER broadcasts/signs — this only assembles. Use ``dry_run_prepared`` for the read-only
    pre-flight; broadcast stays a separate user-gated step."""
    reason = getattr(record, "reason", None)
    if reason == _REASON_INVALID_SIGNATURE:
        challenge, eff_target_index = _prepare_invalid_signature(record, receipt_lookup)
    elif reason == _REASON_DOUBLE_SPEND:
        challenge, eff_target_index = _prepare_double_spend(record, receipt_lookup)
    elif reason == _REASON_EXPIRED:
        challenge, eff_target_index = _prepare_expired(record, receipt_lookup, now)
    elif reason == _REASON_NO_ESCROW:
        challenge, eff_target_index = _prepare_no_escrow(record, receipt_lookup)
    elif reason == _REASON_CONSENSUS_MISMATCH:
        # sp1165 — assembled here (challengeReceipt ABI, reason 5, consensus auxData); the
        # dry-run/broadcast goes through the dedicated ConsensusChallengeSubmitter.
        challenge, eff_target_index = _prepare_consensus_mismatch(record, receipt_lookup)
    elif reason == _REASON_INFERENCE_RECEIPT:
        raise UnsupportedChallengeReason(
            "§7 inference_receipt grounds are off-chain reputation signals (no single clean "
            "on-chain reason code) — not preparable as a challengeReceipt"
        )
    else:
        raise UnsupportedChallengeReason(f"unknown finding reason {reason!r}")

    # Belt-and-suspenders: never hand back a challenge no submitter accepts. (The dispatch
    # above already maps to exactly the preparable reasons; this guards a future assembler
    # drift.) _PREPARABLE_REASONS = the generalized submitter's four + CONSENSUS_MISMATCH.
    if int(challenge.reason_code) not in _PREPARABLE_REASONS:
        raise UnsupportedChallengeReason(
            f"assembled reason {challenge.reason_code} is not preparable "
            f"(supported={sorted(_PREPARABLE_REASONS)})"
        )
    return PreparedChallenge(
        reason=str(reason),
        on_chain_reason=int(challenge.reason_code),
        batch_id_hex=_strip0x(record.batch_id_hex),
        target_index=int(eff_target_index),
        challenge=challenge,
    )


@dataclass(frozen=True)
class SkippedFinding:
    """A finding the bridge could not prepare, recorded WITHOUT aborting the batch (so one
    unsupported/un-sourced finding never hides the preparable ones)."""

    reason: str
    batch_id_hex: str
    target_index: Optional[int]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "batch_id_hex": self.batch_id_hex,
            "target_index": self.target_index,
            "message": self.message,
        }


def prepare_findings(
    records: Any,
    receipt_lookup: ReceiptLookup,
    *,
    now: Optional[int] = None,
) -> Tuple[List[PreparedChallenge], List[SkippedFinding]]:
    """Prepare every record that CAN be prepared, partitioning the rest into skipped (with a
    reason). NEVER raises — a per-record failure (unsupported reason, missing receipts, an
    assembler fail-fast) is recorded as a ``SkippedFinding`` and the rest still prepare. This
    is the operator-facing batch entry point. NEVER broadcasts/signs."""
    prepared: List[PreparedChallenge] = []
    skipped: List[SkippedFinding] = []
    for rec in records:
        try:
            prepared.append(prepare_challenge_from_finding(
                record=rec, receipt_lookup=receipt_lookup, now=now))
        except Exception as exc:  # noqa: BLE001 — this batch entry point's contract is
            # NEVER-raises: ANY per-record failure (a typed UnsupportedChallengeReason /
            # MissingBatchReceipts / ValueError, OR an unanticipated malformed-record
            # KeyError/TypeError) is recorded as a SkippedFinding so one bad finding can
            # never hide the preparable ones or abort the operator's whole run.
            skipped.append(SkippedFinding(
                reason=str(getattr(rec, "reason", "?")),
                batch_id_hex=_strip0x(getattr(rec, "batch_id_hex", "") or ""),
                target_index=getattr(rec, "target_index", None),
                message=f"{type(exc).__name__}: {exc}",
            ))
    return prepared, skipped


def dry_run_prepared(prepared: PreparedChallenge, dry_run_client: Any) -> Any:
    """Read-only pre-flight of a prepared challenge: calls the client's ``dry_run`` (a static
    eth_call — no tx, no gas, no state change) and returns its verdict. NEVER calls
    ``submit`` — broadcasting stays the separate, user-gated step."""
    return dry_run_client.dry_run(prepared.challenge)


def published_batch_receipt_lookup(store: Any) -> ReceiptLookup:
    """A ``ReceiptLookup`` backed by a ``PublishedBatchStore`` (the producer's/requester's own
    retained batches). Returns the batch's ordered receipts, or None when not retained."""

    def _lookup(batch_id_hex: str) -> Optional[List[BatchedReceipt]]:
        pb = store.get(bytes.fromhex(_strip0x(batch_id_hex)))
        return list(pb.receipts) if pb is not None else None

    return _lookup


def composite_receipt_lookup(*lookups: Optional[ReceiptLookup]) -> ReceiptLookup:
    """Compose several ``ReceiptLookup``s into one: try each in order, FIRST non-empty hit
    wins; a lookup that returns None/[] falls through to the next. Lets the prepare CLI source
    receipts from the producer's/requester's OWN PublishedBatchStore AND the observer's
    verified-foreign-batch store (sp1164) — so a finding on a third-party batch the observer
    audited prepares too. None entries are ignored; no lookups -> always None."""
    funcs = [f for f in lookups if f is not None]

    def _lookup(batch_id_hex: str) -> Optional[List[BatchedReceipt]]:
        for f in funcs:
            got = f(batch_id_hex)
            if got:
                return got
        return None

    return _lookup
