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
