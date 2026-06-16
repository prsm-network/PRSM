"""On-chain challenge ASSEMBLY for a proven DOUBLE_SPEND dispute (challenge/dispute
brick 7).

The brick-6 detector (double_spend_detector.py) finds a merkle leaf committed at >=2
distinct positions — an unambiguous double-commit. This module turns such a finding (or
two batches + the two leaf coordinates) into the concrete inputs the on-chain
BatchSettlementRegistry.challengeReceipt needs:

    challengeReceipt(batchId, leaf, merkleProof, ReasonCode.DOUBLE_SPEND, auxData)

where:
  - batchId     = the TARGET batch the challenge is filed against;
  - leaf        = the canonical ReceiptLeaf for the double-spent receipt
                  (merkle.batched_receipt_to_leaf), built from the TARGET batch's
                  retained receipt preimage;
  - merkleProof = the OpenZeppelin sorted-pair proof that ``leaf`` is in ``batchId``'s
                  committed root (the contract verifies this caller-side first);
  - auxData     = abi.encode(bytes32 conflictingBatchId, bytes32[] conflictingProof) —
                  the conflicting batch id + the proof that the SAME leafHash is also in
                  the conflicting batch's committed root. The on-chain _handleDoubleSpend
                  loads batches[conflictingBatchId] (must NOT be NONEXISTENT) and returns
                  MerkleProof.verify(conflictingProof, other.merkleRoot, leafHash).
                  (contracts/contracts/BatchSettlementRegistry.sol:751-767.)

So the challenge is proven iff the SAME leafHash is provable in BOTH the target root AND
the conflicting root — and this assembler builds BOTH proofs and binds the conflicting
one into auxData.

DATA USE. The TARGET side needs the receipt PREIMAGES (to rebuild the ReceiptLeaf struct
the contract wants and its proof) — exactly like the brick-3 INVALID_SIGNATURE assembler.
The CONFLICTING side needs only that batch's ordered ``leaf_hashes`` (to build
conflictingProof) — and ``CommittedBatch`` retains precisely those. So this is a PURE
function of its inputs; runtime sourcing of receipts is a separate concern.

FAIL-FAST (soundness — this is the slashing path; a wrongful challenge slashes an HONEST
provider's bond and also wastes the challenger's gas on a revert). We refuse to assemble
unless:
  - both indices are in range;
  - target_batch_id != conflicting_batch_id (two DISTINCT batches — a leaf duplicated
    within ONE batch is a different on-chain story the conflicting-batch auxData layout
    cannot express; the contract would load the SAME batch as "other" and, even if it
    verified, this is not the inter-batch double-spend the reason code describes);
  - hash_leaf(target_leaf) == conflicting_leaf_hashes[conflicting_index] — the SAME leaf
    must GENUINELY appear in both batches, else the on-chain conflictingProof would not
    verify and the challenge would revert ChallengeNotProven.

This module is PURE: no web3, no network. eth_abi.encode for auxData is pure encoding.
Broadcasting/signing/slashing the assembled challenge is the separate, user-gated brick-8
submitter (mirroring invalid_signature_submitter.py)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

from prsm.economy.web3.stake_manager import ReasonCode
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import leaf_to_tuple
from prsm.settlement.double_spend_detector import DoubleSpendFinding, LeafOccurrence
from prsm.settlement.merkle import (
    ReceiptLeaf,
    batched_receipt_to_leaf,
    build_merkle_proof,
    hash_leaf,
)

# BatchSettlementRegistry.ReasonCode.DOUBLE_SPEND (== 0).
REASON_DOUBLE_SPEND: int = int(ReasonCode.DOUBLE_SPEND)


@dataclass(frozen=True)
class DoubleSpendChallenge:
    """Everything a caller needs to invoke (or simulate) challengeReceipt for a
    DOUBLE_SPEND dispute. Mirrors InvalidSignatureChallenge so a submitter can call
    challengeReceipt uniformly. ``broadcast`` is intentionally NOT here — that's the
    user-gated step."""

    batch_id: bytes
    leaf: ReceiptLeaf
    merkle_proof: List[bytes]
    reason_code: int = REASON_DOUBLE_SPEND
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


def _encode_double_spend_aux(
    conflicting_batch_id: bytes, conflicting_proof: Sequence[bytes],
) -> bytes:
    """auxData = abi.encode(bytes32 conflictingBatchId, bytes32[] conflictingProof) —
    the layout _handleDoubleSpend decodes (abi.decode(auxData, (bytes32, bytes32[])))."""
    from eth_abi import encode as abi_encode

    return bytes(abi_encode(
        ["bytes32", "bytes32[]"],
        [bytes(conflicting_batch_id), [bytes(p) for p in conflicting_proof]],
    ))


def assemble_double_spend_challenge(
    *,
    target_batch_id: bytes,
    target_batch_receipts: List[BatchedReceipt],
    target_index: int,
    conflicting_batch_id: bytes,
    conflicting_leaf_hashes: Sequence[bytes],
    conflicting_index: int,
) -> DoubleSpendChallenge:
    """Assemble the challengeReceipt inputs for the receipt at ``target_index`` in the
    target batch, proving it was ALSO committed at ``conflicting_index`` in the
    conflicting batch.

    ``target_batch_receipts`` MUST be the full, ORDER-PRESERVED set of receipts the
    target batch's merkle root was built over (the proof depends on the exact leaf
    order). ``conflicting_leaf_hashes`` MUST be the full, ORDER-PRESERVED leaf hashes the
    conflicting batch's merkle root was built over (CommittedBatch.leaf_hashes).

    Raises ValueError (fail-fast — refusing a challenge that would revert / slash an
    honest provider) if:
      - either index is out of range;
      - target_batch_id == conflicting_batch_id (must be two DISTINCT batches);
      - the target leaf does NOT genuinely appear at conflicting_index in the conflicting
        batch (hash_leaf(target_leaf) != conflicting_leaf_hashes[conflicting_index])."""
    if not target_batch_receipts:
        raise ValueError("target_batch_receipts is empty")
    if not conflicting_leaf_hashes:
        raise ValueError("conflicting_leaf_hashes is empty")
    if not (0 <= target_index < len(target_batch_receipts)):
        raise ValueError(
            f"target_index {target_index} out of range for target batch of "
            f"{len(target_batch_receipts)} receipts"
        )
    if not (0 <= conflicting_index < len(conflicting_leaf_hashes)):
        raise ValueError(
            f"conflicting_index {conflicting_index} out of range for conflicting batch "
            f"of {len(conflicting_leaf_hashes)} leaf hashes"
        )
    if bytes(target_batch_id) == bytes(conflicting_batch_id):
        raise ValueError(
            "refusing to assemble a DOUBLE_SPEND challenge: target_batch_id == "
            "conflicting_batch_id — the two batches must be DISTINCT (a leaf duplicated "
            "within ONE batch is not expressible via the conflicting-batch auxData)"
        )

    # Build the target leaf + its proof from the retained receipt preimages.
    leaves = [batched_receipt_to_leaf(br) for br in target_batch_receipts]
    target_leaf_hashes = [hash_leaf(leaf) for leaf in leaves]
    target_leaf = leaves[target_index]
    target_leaf_hash = target_leaf_hashes[target_index]

    # SOUNDNESS: the SAME leaf must genuinely appear in the conflicting batch, else the
    # on-chain conflictingProof would not verify → ChallengeNotProven revert.
    expected = bytes(conflicting_leaf_hashes[conflicting_index])
    if target_leaf_hash != expected:
        raise ValueError(
            "refusing to assemble a DOUBLE_SPEND challenge: the conflicting batch does "
            f"not contain the target leaf at conflicting_index {conflicting_index} "
            f"(target leafHash {target_leaf_hash.hex()} != conflicting leaf hash "
            f"{expected.hex()}) — not the same leaf, the on-chain challenge would revert"
        )

    target_proof = build_merkle_proof(target_leaf_hashes, target_index)
    conflicting_proof = build_merkle_proof(
        [bytes(h) for h in conflicting_leaf_hashes], conflicting_index)
    aux = _encode_double_spend_aux(conflicting_batch_id, conflicting_proof)

    return DoubleSpendChallenge(
        batch_id=bytes(target_batch_id),
        leaf=target_leaf,
        merkle_proof=target_proof,
        reason_code=REASON_DOUBLE_SPEND,
        aux_data=aux,
    )


def _pick_distinct_batch_occurrences(
    finding: DoubleSpendFinding,
) -> Tuple[LeafOccurrence, LeafOccurrence]:
    """Pick two occurrences from DISTINCT batches (the inter-batch double-spend the
    auxData layout can express). Raises ValueError if the finding has fewer than two
    distinct-batch occurrences (e.g., a purely intra-batch double-commit)."""
    first = finding.occurrences[0]
    for occ in finding.occurrences[1:]:
        if bytes(occ.batch_id) != bytes(first.batch_id):
            return first, occ
    raise ValueError(
        "DoubleSpendFinding has no two occurrences in DISTINCT batches — an intra-batch "
        "double-commit cannot be expressed via the conflicting-batch auxData layout"
    )


def assemble_from_double_spend_finding(
    *,
    finding: DoubleSpendFinding,
    receipts_by_batch: Mapping[bytes, List[BatchedReceipt]],
    leaf_hashes_by_batch: Mapping[bytes, Sequence[bytes]],
) -> DoubleSpendChallenge:
    """Convenience: assemble a DOUBLE_SPEND challenge directly from a brick-6 detector
    finding plus the two batches' retained data.

    Picks two occurrences in DISTINCT batches; uses the FIRST as the TARGET (we need its
    receipt preimages → must be in ``receipts_by_batch``) and the SECOND as the
    CONFLICTING (needs only its ordered leaf_hashes → from ``leaf_hashes_by_batch``).

    Raises ValueError if no two distinct-batch occurrences exist, if the target batch's
    receipts are not provided, or if the conflicting batch's leaf_hashes are not provided
    (plus all the fail-fast checks of assemble_double_spend_challenge)."""
    target_occ, conflicting_occ = _pick_distinct_batch_occurrences(finding)

    target_id = bytes(target_occ.batch_id)
    conflicting_id = bytes(conflicting_occ.batch_id)

    if target_id not in receipts_by_batch:
        raise ValueError(
            f"receipts_by_batch missing target batch {target_id.hex()} — the target side "
            "needs receipt preimages to build the leaf struct + proof"
        )
    if conflicting_id not in leaf_hashes_by_batch:
        raise ValueError(
            f"leaf_hashes_by_batch missing conflicting batch {conflicting_id.hex()}"
        )

    return assemble_double_spend_challenge(
        target_batch_id=target_id,
        target_batch_receipts=list(receipts_by_batch[target_id]),
        target_index=target_occ.leaf_index,
        conflicting_batch_id=conflicting_id,
        conflicting_leaf_hashes=list(leaf_hashes_by_batch[conflicting_id]),
        conflicting_index=conflicting_occ.leaf_index,
    )
