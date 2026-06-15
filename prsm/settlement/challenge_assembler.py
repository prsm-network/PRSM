"""On-chain challenge ASSEMBLY for a proven INVALID_SIGNATURE dispute (challenge/dispute
brick 3).

The off-chain verifier (challenge_verifier) produces a verdict; the on-chain
BatchSettlementRegistry.challengeReceipt needs concrete inputs. This module assembles them
for the INVALID_SIGNATURE reason from a committed batch's receipts:

    challengeReceipt(batchId, leaf, merkleProof, ReasonCode.INVALID_SIGNATURE, auxData)

where:
  - leaf      = the canonical ReceiptLeaf for the target receipt (merkle.batched_receipt
                _to_leaf), whose providerPubkeyHash / signatureHash / signingMessageHash
                are what the on-chain handler binds the challenge to;
  - merkleProof = the OpenZeppelin sorted-pair proof that the leaf is in the committed
                batch (merkle.build_merkle_proof over ALL the batch's leaf hashes);
  - auxData   = abi.encode(bytes publicKey, bytes signature) — the committed pubkey +
                signature that the on-chain ISignatureVerifier will find DO NOT verify
                over the leaf-committed signingMessageHash.

Fail-fast: we assemble a challenge ONLY when the committed shard signature actually does
NOT verify (a valid signature → the on-chain challenge would revert ChallengeNotProven,
wasting gas and looking like griefing). The signed preimage is build_receipt_signing_
payload (== the leaf's signing_message_hash), so the check here matches what the chain
checks. Broadcasting the assembled challenge is a separate, user-gated step (it spends gas
and can slash a provider's bond)."""
from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass, field
from typing import List, Tuple

from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.merkle import (
    ReceiptLeaf,
    batched_receipt_to_leaf,
    build_merkle_proof,
    hash_leaf,
)

# BatchSettlementRegistry.ReasonCode.INVALID_SIGNATURE (mirrors stake_manager.ReasonCode).
REASON_INVALID_SIGNATURE = 1


def _shard_signature_verifies(br: BatchedReceipt) -> bool:
    """True iff the committed shard signature verifies over the canonical signing payload
    (== the leaf's signing_message_hash) under the committed pubkey. Fail-closed: any
    error → False (treated as not-verifying)."""
    from prsm.compute.shard_receipt import build_receipt_signing_payload
    from prsm.node.identity import verify_signature

    r = br.receipt
    try:
        payload = build_receipt_signing_payload(
            job_id=r.job_id, shard_index=r.shard_index,
            output_hash=r.output_hash, executed_at_unix=r.executed_at_unix,
        )
        return verify_signature(r.provider_pubkey_b64, payload, r.signature)
    except Exception:  # noqa: BLE001
        return False


def leaf_to_tuple(leaf: ReceiptLeaf) -> Tuple:
    """The ReceiptLeaf as the ordered tuple the contract's `leaf` struct expects
    (matches the Solidity ReceiptLeaf field order + merkle.encode_leaf)."""
    return (
        leaf.job_id_hash,
        int(leaf.shard_index),
        leaf.provider_id_hash,
        leaf.provider_pubkey_hash,
        leaf.output_hash,
        int(leaf.executed_at_unix),
        int(leaf.value_ftns),
        leaf.signature_hash,
        leaf.signing_message_hash,
    )


@dataclass(frozen=True)
class InvalidSignatureChallenge:
    """Everything a caller needs to invoke (or simulate) challengeReceipt for an
    INVALID_SIGNATURE dispute. ``broadcast`` is intentionally NOT here — that's the
    user-gated step."""

    batch_id: bytes
    leaf: ReceiptLeaf
    merkle_proof: List[bytes]
    reason_code: int = REASON_INVALID_SIGNATURE
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


def _encode_invalid_signature_aux(pubkey_b64: str, signature_b64: str) -> bytes:
    """auxData = abi.encode(bytes publicKey, bytes signature) — the raw committed bytes
    the on-chain _handleInvalidSignature decodes and binds to the leaf via keccak."""
    from eth_abi import encode as abi_encode

    return bytes(abi_encode(
        ["bytes", "bytes"],
        [b64decode(pubkey_b64), b64decode(signature_b64)],
    ))


def assemble_invalid_signature_challenge(
    *,
    batch_id: bytes,
    batch_receipts: List[BatchedReceipt],
    target_index: int,
) -> InvalidSignatureChallenge:
    """Assemble the challengeReceipt inputs for the receipt at ``target_index`` in a
    committed batch. ``batch_receipts`` MUST be the full, ORDER-PRESERVED set of receipts
    the batch's merkle root was built over (the proof depends on the exact leaf order).

    Raises ValueError if the index is out of range, or if the target's committed shard
    signature actually VERIFIES (refusing to assemble a challenge that would revert
    on-chain — fail-fast against wasted gas / griefing)."""
    if not batch_receipts:
        raise ValueError("batch_receipts is empty")
    if not (0 <= target_index < len(batch_receipts)):
        raise ValueError(
            f"target_index {target_index} out of range for batch of "
            f"{len(batch_receipts)} receipts"
        )
    target = batch_receipts[target_index]
    if _shard_signature_verifies(target):
        raise ValueError(
            "refusing to assemble an INVALID_SIGNATURE challenge: the target receipt's "
            "committed shard signature VERIFIES — the on-chain challenge would revert "
            "ChallengeNotProven (no fraud to prove here)"
        )
    leaves = [batched_receipt_to_leaf(br) for br in batch_receipts]
    leaf_hashes = [hash_leaf(leaf) for leaf in leaves]
    merkle_proof = build_merkle_proof(leaf_hashes, target_index)
    aux = _encode_invalid_signature_aux(
        target.receipt.provider_pubkey_b64, target.receipt.signature,
    )
    return InvalidSignatureChallenge(
        batch_id=bytes(batch_id),
        leaf=leaves[target_index],
        merkle_proof=merkle_proof,
        reason_code=REASON_INVALID_SIGNATURE,
        aux_data=aux,
    )
