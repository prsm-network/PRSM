"""Sprint 1165 — on-chain CONSENSUS_MISMATCH challenge ASSEMBLY (the 5th reason code).

The other four on-chain reason codes (INVALID_SIGNATURE, DOUBLE_SPEND, NO_ESCROW, EXPIRED)
each have a settlement-side assembler that turns a finding + the batch's retained receipts
into the concrete ``challengeReceipt`` inputs. CONSENSUS_MISMATCH was the exception: its
assembly lived only INSIDE the audit engine (``SettlementAuditEngine._try_assemble_consensus_
mismatch``, coupled to the in-memory VerifiedBatchCache), so the out-of-daemon prepare bridge
(sp1163) could not build it — leaving consensus_mismatch the one fraud class an operator could
detect (sp1149) but not prepare.

This module is the PURE, cache-free assembler. Given the minority + majority batches' ordered
receipts + the two leaf indices, it builds the same ``challengeReceipt(minorityBatchId,
minorityLeaf, minorityProof, CONSENSUS_MISMATCH, aux)`` the engine + the
ConsensusChallengeSubmitter use:
  - the minority leaf + its in-batch merkle proof (the CHALLENGED leaf);
  - auxData = abi.encode(majorityBatchId, majorityProof, majorityLeaf) — the conflicting
    (majority) leaf + its in-batch proof, via the SHARED ``encode_consensus_mismatch_aux`` so
    there is no layout drift vs the submitter.

It builds the leaf tuple via the CANONICAL committed convention
``settlement.merkle.batched_receipt_to_leaf`` + ``challenge_assembler.leaf_to_tuple`` — the
SAME convention the production batch-commit folds the on-chain merkleRoot over and the other
four assemblers use. (It deliberately does NOT use
``consensus_submitter.ReceiptLeafFields.from_python_receipt``, whose pre-sp1165
keccak(utf8(b64)) hashing of pubkey/signature diverged from the committed
keccak(b64decode(...)) convention — that divergence made every consensus challenge revert
``InvalidMerkleProof``; sp1165 fixed ``from_python_receipt`` too, but this assembler stays on
the settlement merkle path for clean layering + parity with the other reason codes.) The aux
layout reuses the shared ``encode_consensus_mismatch_aux`` so there is no drift vs the submitter.

PURE: no web3, no network, no signing. Broadcasting the assembled challenge is the separate,
user-gated step (``ConsensusChallengeSubmitter.submit_one``); a read-only pre-flight is
``ConsensusChallengeSubmitter.dry_run``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from prsm.economy.web3.stake_manager import ReasonCode
from prsm.marketplace.consensus_submitter import encode_consensus_mismatch_aux
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import leaf_to_tuple
from prsm.settlement.merkle import batched_receipt_to_leaf, build_merkle_proof, hash_leaf

REASON_CONSENSUS_MISMATCH: int = int(ReasonCode.CONSENSUS_MISMATCH)  # == 5

__all__ = [
    "REASON_CONSENSUS_MISMATCH",
    "ConsensusMismatchChallenge",
    "assemble_consensus_mismatch_challenge",
]


@dataclass(frozen=True)
class ConsensusMismatchChallenge:
    """A CONSENSUS_MISMATCH challenge in the SAME structural contract as the other assembled
    challenges (``.reason_code`` + ``.to_call_args()`` matching the ``challengeReceipt`` ABI),
    so the prepare bridge + a dry-run client handle it uniformly. ``broadcast`` is intentionally
    NOT here — that's the user-gated ``ConsensusChallengeSubmitter.submit_one`` step."""

    minority_batch_id: bytes
    minority_leaf_tuple: Tuple
    minority_proof: List[bytes]
    aux_data: bytes
    reason_code: int = REASON_CONSENSUS_MISMATCH

    def to_call_args(self) -> Tuple:
        """The positional args for registry.functions.challengeReceipt(...)."""
        return (
            self.minority_batch_id,
            tuple(self.minority_leaf_tuple),
            list(self.minority_proof),
            int(self.reason_code),
            self.aux_data,
        )


def _leaf_hashes(receipts: List[BatchedReceipt]) -> List[bytes]:
    return [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]


def assemble_consensus_mismatch_challenge(
    *,
    minority_batch_id: bytes,
    minority_receipts: List[BatchedReceipt],
    minority_index: int,
    majority_batch_id: bytes,
    majority_receipts: List[BatchedReceipt],
    majority_index: int,
) -> ConsensusMismatchChallenge:
    """Assemble the ``challengeReceipt`` inputs to challenge the MINORITY leaf as a
    CONSENSUS_MISMATCH against the MAJORITY leaf.

    ``minority_receipts`` / ``majority_receipts`` MUST each be the full, ORDER-PRESERVED set
    the respective batch's merkle root was built over (the proofs depend on the exact leaf
    order). The minority side supplies the challenged leaf + its proof; the majority side is
    bound into auxData (conflicting batch id + majority leaf + majority proof).

    Raises ValueError fail-fast (refusing a would-revert / griefing tx) on empty receipts or an
    out-of-range index. Mirrors SettlementAuditEngine._try_assemble_consensus_mismatch (same
    ReceiptLeafFields encoding + merkle proof build + auxData layout)."""
    if not minority_receipts:
        raise ValueError("minority_receipts is empty")
    if not majority_receipts:
        raise ValueError("majority_receipts is empty")
    if not (0 <= minority_index < len(minority_receipts)):
        raise ValueError(
            f"minority_index {minority_index} out of range for minority batch of "
            f"{len(minority_receipts)} receipts"
        )
    if not (0 <= majority_index < len(majority_receipts)):
        raise ValueError(
            f"majority_index {majority_index} out of range for majority batch of "
            f"{len(majority_receipts)} receipts"
        )

    # CANONICAL committed convention (keccak(b64decode(...)) for pubkey/sig) — the SAME leaf
    # the on-chain merkleRoot was folded over, so the contract's re-hash + proof verify.
    minority_leaf_tuple = leaf_to_tuple(
        batched_receipt_to_leaf(minority_receipts[minority_index]))
    majority_leaf_tuple = leaf_to_tuple(
        batched_receipt_to_leaf(majority_receipts[majority_index]))
    minority_proof = build_merkle_proof(_leaf_hashes(minority_receipts), minority_index)
    majority_proof = build_merkle_proof(_leaf_hashes(majority_receipts), majority_index)
    aux = encode_consensus_mismatch_aux(
        majority_batch_id=bytes(majority_batch_id),
        majority_proof=majority_proof,
        majority_leaf_tuple=majority_leaf_tuple,
    )
    return ConsensusMismatchChallenge(
        minority_batch_id=bytes(minority_batch_id),
        minority_leaf_tuple=minority_leaf_tuple,
        minority_proof=minority_proof,
        aux_data=aux,
        reason_code=REASON_CONSENSUS_MISMATCH,
    )
