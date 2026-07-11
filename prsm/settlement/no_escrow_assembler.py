"""On-chain challenge ASSEMBLY + requester-side flow for a NO_ESCROW dispute
(sprint 1147 — the REQUESTER self-dispute brick).

NO_ESCROW (ReasonCode == 2) closes the requester-self-dispute fraud class: a malicious
provider can ``commitBatch`` naming requester R as ``requester`` WITHOUT a valid
PaymentAuthorization, and at finalize it would drain R's escrow. R's defense is a
NO_ESCROW ``challengeReceipt``.

On chain (contracts/contracts/BatchSettlementRegistry.sol ``challengeReceipt`` +
``_handleNoEscrow``):
  - The batch must be PENDING + within the challenge window;
  - ``challengeReceipt`` computes ``leafHash = _hashLeaf(leaf)`` and REQUIRES
    ``MerkleProof.verify(merkleProof, b.merkleRoot, leafHash)`` (caller-side, line 661)
    BEFORE the reason dispatch — so a VALID leaf + proof is required even for NO_ESCROW;
  - ``reason == NO_ESCROW`` -> ``_handleNoEscrow(b)``: ``if (msg.sender != b.requester)
    revert; return true;``. So the CHALLENGER MUST BE THE REQUESTER (msg.sender ==
    b.requester). ``auxData`` is UNUSED (we pass empty ``b""``).
  - NO_ESCROW does NOT slash (griefing risk); it invalidates the challenged leaf value
    (``b.invalidatedValueFTNS += leaf.valueFtns``) so finalize will not draw it from R's
    escrow. For the MVP 1-leaf batch, one challenge invalidates the whole batch value.

This module mirrors challenge_assembler (INVALID_SIGNATURE) and double_spend_assembler:

  1. ``NoEscrowChallenge`` — the assembled inputs, same shape as ``InvalidSignatureChallenge``
     (``leaf_tuple()`` / ``to_call_args()``). ``reason_code = NO_ESCROW``, ``aux_data = b""``.
  2. ``assemble_no_escrow_challenge`` — builds the leaf + merkle_proof for the target leaf
     from a committed batch's order-preserved receipts. NO signature fail-fast (NO_ESCROW
     does NOT check the leaf sig — the proof is ``msg.sender == requester``), but it DOES
     range-check ``target_index`` + require non-empty receipts.
  3. ``assemble_no_escrow_challenges`` — the requester-side flow. For EACH sp1146
     classification that is UNAUTHORIZED (AUTHORIZED batches are SKIPPED entirely — the
     matcher is the gate), assemble a NO_ESCROW challenge from that batch's receipts and,
     if a ``dry_run_client`` is injected, dry-run it (READ-ONLY). It NEVER submits or
     broadcasts — assembly + dry-run only; the user signs the real tx with their own key.

CRITICAL SAFETY: a NO_ESCROW challenge DENIES the provider payment for the leaf. Disputing
a batch R DID authorize would wrongly deny an HONEST provider, so the flow assembles ONLY
for UNAUTHORIZED classifications — NEVER for AUTHORIZED. Broadcast is requester-keyed +
USER-GATED; this brick only ASSEMBLES + DRY-RUNS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prsm.economy.web3.stake_manager import ReasonCode
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import leaf_to_tuple
from prsm.settlement.issued_authorization_store import (
    UNAUTHORIZED,
    BatchAuthorizationClassification,
    CommittedBatchView,
    IssuedAuthorizationStore,
    match_unauthorized_batches,
)
from prsm.settlement.merkle import (
    ReceiptLeaf,
    batched_receipt_to_leaf,
    build_merkle_proof,
    hash_leaf,
)

logger = logging.getLogger(__name__)

# BatchSettlementRegistry.ReasonCode.NO_ESCROW (mirrors stake_manager.ReasonCode == 2).
REASON_NO_ESCROW = int(ReasonCode.NO_ESCROW)


@dataclass(frozen=True)
class NoEscrowChallenge:
    """Everything a (REQUESTER-keyed) caller needs to invoke (or simulate)
    ``challengeReceipt`` for a NO_ESCROW dispute. Identical shape to
    ``InvalidSignatureChallenge`` / ``DoubleSpendChallenge`` so the generalized
    ``ChallengeSubmitter`` accepts it. ``aux_data`` is empty — NO_ESCROW ignores auxData;
    its proof is ``msg.sender == b.requester``. ``broadcast`` is intentionally NOT here —
    that's the user-gated step."""

    batch_id: bytes
    leaf: ReceiptLeaf
    merkle_proof: List[bytes]
    reason_code: int = REASON_NO_ESCROW
    aux_data: bytes = b""

    def leaf_tuple(self) -> Tuple:
        return leaf_to_tuple(self.leaf)

    def to_call_args(self) -> Tuple:
        """The positional args for registry.functions.challengeReceipt(...)."""
        return (
            self.batch_id,
            self.leaf_tuple(),
            list(self.merkle_proof),
            int(self.reason_code),
            self.aux_data,
        )


def assemble_no_escrow_challenge(
    *,
    batch_id: bytes,
    batch_receipts: List[BatchedReceipt],
    target_index: int,
) -> NoEscrowChallenge:
    """Assemble the ``challengeReceipt`` inputs for the receipt at ``target_index`` in a
    committed batch, for a NO_ESCROW dispute. ``batch_receipts`` MUST be the full,
    ORDER-PRESERVED set of receipts the batch's merkle root was built over (the proof
    depends on the exact leaf order).

    Unlike the INVALID_SIGNATURE assembler there is NO signature fail-fast: NO_ESCROW
    does NOT verify the leaf signature on chain — it proves ``msg.sender == requester``.
    The leaf + proof are still required (the contract verifies the merkle proof caller-
    side BEFORE the reason dispatch), so a VALID leaf + proof is assembled here.

    Raises ValueError if ``batch_receipts`` is empty or ``target_index`` is out of range.
    """
    if not batch_receipts:
        raise ValueError("batch_receipts is empty")
    if not (0 <= target_index < len(batch_receipts)):
        raise ValueError(
            f"target_index {target_index} out of range for batch of "
            f"{len(batch_receipts)} receipts"
        )
    leaves = [batched_receipt_to_leaf(br) for br in batch_receipts]
    leaf_hashes = [hash_leaf(leaf) for leaf in leaves]
    merkle_proof = build_merkle_proof(leaf_hashes, target_index)
    return NoEscrowChallenge(
        batch_id=bytes(batch_id),
        leaf=leaves[target_index],
        merkle_proof=merkle_proof,
        reason_code=REASON_NO_ESCROW,
        aux_data=b"",
    )


@dataclass(frozen=True)
class NoEscrowAssemblyResult:
    """Per-classification outcome of the requester-side flow.

    For an UNAUTHORIZED classification: ``challenge`` is the assembled NoEscrowChallenge
    (or None + ``error`` if assembly failed — e.g. missing receipts, fail-closed), and
    ``dry_run_would_succeed`` is the read-only verdict if a dry_run_client was injected
    (else None). AUTHORIZED classifications are SKIPPED entirely and never produce a
    result with a challenge (the matcher is the gate)."""
    batch_id: bytes
    classification: str
    challenge: Optional[NoEscrowChallenge] = None
    dry_run_would_succeed: Optional[bool] = None
    dry_run_revert_reason: Optional[str] = None
    error: Optional[str] = None


def assemble_no_escrow_challenges(
    classifications: Sequence[BatchAuthorizationClassification],
    receipts_by_batch: Dict[bytes, List[BatchedReceipt]],
    *,
    dry_run_client: Any = None,
) -> List[NoEscrowAssemblyResult]:
    """Requester-side flow: assemble NO_ESCROW challenges for the UNAUTHORIZED batches
    only, and (read-only) dry-run them if a ``dry_run_client`` is injected.

    SAFETY (ONLY-UNAUTHORIZED): a classification is acted on ONLY when
    ``classification.classification == UNAUTHORIZED``. AUTHORIZED batches are SKIPPED
    entirely — never assembled, never disputed — because a NO_ESCROW challenge denies the
    provider payment, and disputing an authorized batch would grief an HONEST provider.
    The sp1146 matcher is the gate; this flow trusts it.

    NO AUTO-SUBMIT: this NEVER calls ``submit``/``broadcast``. When a ``dry_run_client``
    is provided, only its READ-ONLY ``dry_run`` is invoked (a static eth_call — no tx, no
    gas, no state change). The real broadcast stays USER-GATED (the requester signs it
    with their own key as ``msg.sender``).

    Fail-closed per item: a missing-receipts entry, an assembly ValueError, or a dry-run
    error is recorded in that item's ``error`` and the loop CONTINUES — one bad batch
    never aborts the others, and a failed assembly never produces a challenge.

    ``receipts_by_batch`` maps ``batch_id`` (bytes) -> the order-preserved batch receipts.
    Returns one ``NoEscrowAssemblyResult`` per UNAUTHORIZED classification.
    """
    results: List[NoEscrowAssemblyResult] = []
    for c in classifications:
        if c.classification != UNAUTHORIZED:
            # AUTHORIZED (or any non-UNAUTHORIZED) — never assemble/dispute.
            continue
        batch_id = bytes(c.batch_id)
        receipts = receipts_by_batch.get(batch_id)
        if not receipts:
            results.append(NoEscrowAssemblyResult(
                batch_id=batch_id, classification=c.classification,
                error="receipts missing for UNAUTHORIZED batch — cannot assemble "
                      "NO_ESCROW challenge (fail-closed)",
            ))
            continue
        try:
            # MVP: one challenge per leaf; assemble the first leaf (a 1-leaf batch is the
            # common case, and invalidating it denies the whole unauthorized batch value).
            challenge = assemble_no_escrow_challenge(
                batch_id=batch_id, batch_receipts=list(receipts), target_index=0)
        except Exception as exc:  # noqa: BLE001 — fail-closed per item, never abort
            logger.warning(
                "assemble_no_escrow_challenges: assembly failed for batch %s (%s: %s)",
                batch_id.hex(), type(exc).__name__, exc,
            )
            results.append(NoEscrowAssemblyResult(
                batch_id=batch_id, classification=c.classification,
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue

        would_succeed: Optional[bool] = None
        revert_reason: Optional[str] = None
        if dry_run_client is not None:
            try:
                verdict = dry_run_client.dry_run(challenge)
                would_succeed = bool(verdict.would_succeed)
                revert_reason = getattr(verdict, "revert_reason", None)
            except Exception as exc:  # noqa: BLE001 — dry-run is read-only/best-effort
                logger.warning(
                    "assemble_no_escrow_challenges: dry_run failed for batch %s "
                    "(%s: %s)", batch_id.hex(), type(exc).__name__, exc,
                )
                would_succeed = False
                revert_reason = f"{type(exc).__name__}: {exc}"

        results.append(NoEscrowAssemblyResult(
            batch_id=batch_id, classification=c.classification,
            challenge=challenge,
            dry_run_would_succeed=would_succeed,
            dry_run_revert_reason=revert_reason,
        ))
    return results


# ── sp1422: FINDING the batches (the piece that was missing) ──────────────────
#
# sp1146 built the matcher, sp1147 (above) built the challenge assembly, sp1172 built the
# issued-auth store. All correct, all well-tested — and NONE of it was ever reachable, because
# nothing ever went and LOOKED for batches committed against this requester. `match_unauthorized_
# batches` needs a list of committed batches; no production code ever produced one.
#
# It cannot come from the node's own PublishedBatchStore: that holds batches THIS node
# published. An attacker's forged batch exists ONLY on chain. So the requester has to read it
# from chain.
#
# AND THE EVENT DOES NOT HELP. `event BatchCommitted(bytes32 indexed batchId, address indexed
# provider, ...)` indexes batchId + provider and does NOT CARRY THE REQUESTER AT ALL. A victim
# therefore cannot filter for "batches drawn against me" — they must enumerate EVERY batch and
# read each one's requester out of contract storage. That is an on-chain design flaw (the
# requester should be an indexed event field); fixing it needs a contract change + Foundation
# deploy, so until then this scan is O(batches) and we do it the slow way. It is fine at
# current volume and it is a real scaling limit — flagged, not hidden.

# BatchStatus (contracts/contracts/BatchSettlementRegistry.sol)
_STATUS_PENDING = 1
_STATUS_FINALIZED = 2


@dataclass(frozen=True)
class UnauthorizedBatchAlert:
    """An on-chain batch drawn against MY escrow that I never authorized."""

    classification: BatchAuthorizationClassification
    status: int
    #: PENDING -> the challenge window may still be open, so a NO_ESCROW dispute can still
    #: save the funds. FINALIZED -> the escrow is ALREADY GONE; this is a post-mortem.
    challengeable: bool

    @property
    def already_drained(self) -> bool:
        return self.status == _STATUS_FINALIZED


def scan_for_unauthorized_batches(
    *,
    enumerate_batches: Any,
    read_batch: Any,
    store: IssuedAuthorizationStore,
    my_address: str,
    from_block: int,
    to_block: int,
) -> List[UnauthorizedBatchAlert]:
    """Find on-chain batches committed against MY escrow that I never authorized.

    This is the requester's only warning that someone is draining them. On-chain,
    ``commitBatch(address requester, ...)`` accepts the requester as a plain, unverified,
    un-bonded argument, and ``finalizeBatch`` — callable by anyone — then pays
    ``settleFromRequester(requester, provider, value)`` out of that requester's escrow. The
    sole defense is a NO_ESCROW challenge that ONLY the requester may raise
    (``_handleNoEscrow`` reverts unless ``msg.sender == b.requester``). So if the requester
    is not looking, nobody is.

    Injected readers (offline-testable, no web3 import here):
      ``enumerate_batches(from_block, to_block)`` -> objects with ``.batch_id`` +
      ``.requester_address`` (``Web3SettlementContractClient._enumerate_committed_batches_sync``
      with ``provider=None`` already returns exactly this).
      ``read_batch(batch_id)`` -> dict with ``requester`` / ``total_value_ftns`` /
      ``commit_timestamp`` / ``status`` (``_get_batch_sync``).

    FAIL-OPEN ON READ, FAIL-CLOSED ON JUDGEMENT: an RPC error on one batch is logged and that
    batch is SKIPPED rather than aborting the sweep (one flaky read must not blind the whole
    scan), but a batch is only ever reported UNAUTHORIZED when the sp1146 matcher — which is
    deliberately conservative — says so. NEVER auto-submits: raising the challenge is a signed,
    user-gated action.

    Returns one alert per UNAUTHORIZED batch, PENDING ones first (those are still savable).
    """
    me = str(my_address).lower()

    try:
        observed = list(enumerate_batches(int(from_block), int(to_block)))
    except Exception as exc:  # noqa: BLE001 — a dead RPC must not crash the audit loop
        logger.error(
            "sp1422: could not enumerate committed batches (%s). THIS REQUESTER IS NOW BLIND "
            "to an escrow drain for blocks %s-%s.", exc, from_block, to_block,
        )
        return []

    mine: List[Tuple[bytes, dict]] = []
    for ob in observed:
        if str(getattr(ob, "requester_address", "") or "").lower() != me:
            continue  # not drawn against my escrow — not my concern (and not my right to dispute)
        bid = bytes(ob.batch_id)
        try:
            mine.append((bid, read_batch(bid)))
        except Exception as exc:  # noqa: BLE001 — skip the unreadable one, keep scanning
            logger.error(
                "sp1422: batch %s names me as requester but could not be read (%s) — SKIPPED. "
                "If this persists, an unauthorized batch could finalize unseen.",
                bid.hex(), exc,
            )

    if not mine:
        return []

    views = [
        CommittedBatchView(
            batch_id=bid,
            requester=str(b["requester"]),
            provider=str(b["provider"]),
            total_value_wei=int(b["total_value_ftns"]),
            commit_timestamp=int(b["commit_timestamp"]),
        )
        for bid, b in mine
    ]
    status_by_id = {bid: int(b["status"]) for bid, b in mine}

    alerts: List[UnauthorizedBatchAlert] = []
    for c in match_unauthorized_batches(views, store, my_address=my_address):
        if c.classification != UNAUTHORIZED:
            continue
        status = status_by_id.get(bytes(c.batch_id), 0)
        challengeable = status == _STATUS_PENDING
        alerts.append(UnauthorizedBatchAlert(
            classification=c, status=status, challengeable=challengeable,
        ))
        logger.critical(
            "sp1422 ESCROW DRAIN %s: batch %s commits %s wei against MY escrow (%s) naming "
            "provider %s — and I have NO issued PaymentAuthorization covering it. %s",
            "ATTEMPT" if challengeable else "ALREADY FINALIZED",
            bytes(c.batch_id).hex(), c.total_value_wei, my_address, c.provider,
            "Raise a NO_ESCROW challenge NOW (only you can) — see assemble_no_escrow_challenges."
            if challengeable else
            "The challenge window is closed; these funds are GONE. Post-mortem only.",
        )

    alerts.sort(key=lambda a: not a.challengeable)  # savable ones first
    return alerts
