"""On-chain challenge ASSEMBLY for an EXPIRED dispute (sprint 1148 — the 5th and FINAL
on-chain reason code).

EXPIRED (ReasonCode == 3) is protocol HYGIENE: a committed leaf whose ``executed_at_unix``
is older than the batch's per-batch lookback window must not be paid from escrow.

On chain (contracts/contracts/BatchSettlementRegistry.sol ``challengeReceipt`` +
``_handleExpired``):
  - the batch must be PENDING + within the challenge window;
  - ``challengeReceipt`` computes ``leafHash = _hashLeaf(leaf)`` and REQUIRES
    ``MerkleProof.verify(merkleProof, b.merkleRoot, leafHash)`` (caller-side) BEFORE the
    reason dispatch — so a VALID leaf + proof is required even for EXPIRED;
  - ``reason == EXPIRED`` -> ``_handleExpired(leaf, b.lookbackWindowSecondsAtCommit)``: a PURE
    time check ``if (age <= lookback) revert ReceiptNotExpired(...)``; ``age > lookback`` ->
    return true. ``auxData`` is UNUSED (we pass empty ``b""``);
  - EXPIRED does NOT slash (hygiene, not malice); it invalidates the challenged leaf value
    (``b.invalidatedValueFTNS += leaf.valueFtns``) so finalize won't draw it from escrow. ANY
    observer may challenge (no ``msg.sender`` restriction — third-party detectable).

This module mirrors ``no_escrow_assembler`` (the CLOSEST sibling — empty auxData, no-slash
class, range/empty guards) plus the INVALID_SIGNATURE assembler's FAIL-FAST discipline:

  1. ``ExpiredChallenge`` — the assembled inputs, identical shape to ``NoEscrowChallenge``
     (``leaf_tuple()`` / ``to_call_args()``). ``reason_code = EXPIRED``, ``aux_data = b""``.
  2. ``assemble_expired_challenge`` — builds the leaf + sorted-pair merkle_proof for the
     target leaf, AND FAIL-FASTS with a clear ValueError unless ``now - leaf.executed_at_unix
     > lookback_window_seconds`` (STRICT). It will NOT assemble a challenge that would revert
     ``ReceiptNotExpired`` on chain — refusing to build a griefing tx against an honest
     provider is the assembler's safety belt (the detector's strict check + the dry-run gate
     are the others).

CRITICAL SAFETY: an EXPIRED challenge DENIES the provider payment for the leaf. Disputing a
leaf that is NOT provably expired would wrongly deny an HONEST provider, so the assembler
fail-fasts on ``age <= lookback`` (the on-chain revert condition) AND on a non-positive
lookback (no trustworthy window). Broadcast is observer-keyed + USER-GATED; this brick only
ASSEMBLES (the dry-run + the irreversible tx are downstream).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from prsm.economy.web3.stake_manager import ReasonCode
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import leaf_to_tuple
from prsm.settlement.merkle import (
    ReceiptLeaf,
    batched_receipt_to_leaf,
    build_merkle_proof,
    hash_leaf,
)

# BatchSettlementRegistry.ReasonCode.EXPIRED (== 3).
REASON_EXPIRED = int(ReasonCode.EXPIRED)


@dataclass(frozen=True)
class ExpiredChallenge:
    """Everything an (observer-keyed) caller needs to invoke (or simulate)
    ``challengeReceipt`` for an EXPIRED dispute. Identical shape to ``NoEscrowChallenge`` /
    ``DoubleSpendChallenge`` so the generalized ``ChallengeSubmitter`` accepts it.
    ``aux_data`` is empty — EXPIRED ignores auxData; its proof is the on-chain time check.
    ``broadcast`` is intentionally NOT here — that's the user-gated step."""

    batch_id: bytes
    leaf: ReceiptLeaf
    merkle_proof: List[bytes]
    reason_code: int = REASON_EXPIRED
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


def assemble_expired_challenge(
    *,
    batch_id: bytes,
    batch_receipts: List[BatchedReceipt],
    target_index: int,
    now: int,
    lookback_window_seconds: int,
) -> ExpiredChallenge:
    """Assemble the ``challengeReceipt`` inputs for the receipt at ``target_index`` in a
    committed batch, for an EXPIRED dispute. ``batch_receipts`` MUST be the full,
    ORDER-PRESERVED set of receipts the batch's merkle root was built over (the proof depends
    on the exact leaf order).

    FAIL-FAST (refuses to build a griefing / would-revert tx):
      - empty ``batch_receipts`` or out-of-range ``target_index`` -> ValueError;
      - non-positive ``lookback_window_seconds`` -> ValueError (no trustworthy window to prove
        expiry against — fail-closed, never grief an honest provider);
      - ``now - leaf.executed_at_unix <= lookback_window_seconds`` -> ValueError. This is the
        EXACT on-chain ``_handleExpired`` revert condition (``age <= lookback`` is NOT
        expired); assembling here would only build a tx that reverts ``ReceiptNotExpired``.

    On success: ``reason_code = EXPIRED``, ``aux_data = b""``, with the leaf + sorted-pair
    merkle proof for ``target_index``.
    """
    if not batch_receipts:
        raise ValueError("batch_receipts is empty")
    if not (0 <= target_index < len(batch_receipts)):
        raise ValueError(
            f"target_index {target_index} out of range for batch of "
            f"{len(batch_receipts)} receipts"
        )
    lookback = int(lookback_window_seconds)
    if lookback <= 0:
        raise ValueError(
            f"lookback_window_seconds must be positive to prove expiry (got {lookback}); "
            "fail-closed — refusing to build a challenge without a trustworthy window"
        )

    leaves = [batched_receipt_to_leaf(br) for br in batch_receipts]
    target_leaf = leaves[target_index]
    age = int(now) - int(target_leaf.executed_at_unix)
    # STRICT on-chain check: age > lookback. age <= lookback is NOT expired -> would revert
    # ReceiptNotExpired on chain, so refuse to assemble it (anti-griefing fail-fast).
    if age <= lookback:
        raise ValueError(
            f"receipt at index {target_index} is NOT expired "
            f"(age={age} <= lookback={lookback}); refusing to assemble a challenge that "
            "would revert ReceiptNotExpired on chain (would grief an honest provider)"
        )

    leaf_hashes = [hash_leaf(leaf) for leaf in leaves]
    merkle_proof = build_merkle_proof(leaf_hashes, target_index)
    return ExpiredChallenge(
        batch_id=bytes(batch_id),
        leaf=target_leaf,
        merkle_proof=merkle_proof,
        reason_code=REASON_EXPIRED,
        aux_data=b"",
    )


__all__ = [
    "REASON_EXPIRED",
    "ExpiredChallenge",
    "assemble_expired_challenge",
]
