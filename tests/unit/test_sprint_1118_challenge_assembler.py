"""Sprint 1118 — challenge/dispute brick 3: assemble the on-chain INVALID_SIGNATURE
challengeReceipt inputs (leaf + merkle proof + auxData) from a committed batch.

Fail-fast: refuses to assemble against a receipt whose committed shard signature actually
verifies (that challenge would revert ChallengeNotProven on-chain).
"""
from __future__ import annotations

from base64 import b64decode

import pytest

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.challenge_assembler import (
    REASON_INVALID_SIGNATURE,
    assemble_invalid_signature_challenge,
)
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_merkle_root,
    hash_leaf,
    verify_merkle_proof,
)


def _receipt(identity, *, job="job-1", shard=0, valid=True, out=None):
    out = out or ("ab" * 32)
    payload = build_receipt_signing_payload(
        job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000)
    if valid:
        sig = identity.sign(payload)
    else:
        # sign a DIFFERENT payload → the committed signature won't verify over this leaf
        sig = identity.sign(build_receipt_signing_payload(
            job_id=job, shard_index=shard, output_hash=("cd" * 32), executed_at_unix=1000))
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address="0x" + "1" * 40,
        provider_address="0x" + "2" * 40, value_ftns=10**18, local_escrow_id="e1")


def _batch(specs):
    """specs: list of (identity, valid_bool)."""
    return [_receipt(idn, shard=i, valid=v) for i, (idn, v) in enumerate(specs)]


def test_assembles_for_invalid_signature_receipt():
    a, b, c = (generate_node_identity(n) for n in ("a", "b", "c"))
    batch = _batch([(a, True), (b, False), (c, True)])  # index 1 is fraudulent
    batch_id = b"\x11" * 32
    ch = assemble_invalid_signature_challenge(
        batch_id=batch_id, batch_receipts=batch, target_index=1)

    assert ch.reason_code == REASON_INVALID_SIGNATURE == 1
    assert ch.batch_id == batch_id
    # the leaf is the target's canonical leaf
    assert ch.leaf == batched_receipt_to_leaf(batch[1])


def test_merkle_proof_folds_to_the_batch_root():
    a, b, c, d = (generate_node_identity(n) for n in ("a", "b", "c", "d"))
    batch = _batch([(a, True), (b, True), (c, False), (d, True)])  # index 2 fraudulent
    leaf_hashes = [hash_leaf(batched_receipt_to_leaf(br)) for br in batch]
    root = build_merkle_root(leaf_hashes)
    ch = assemble_invalid_signature_challenge(
        batch_id=b"\x22" * 32, batch_receipts=batch, target_index=2)
    # the assembled proof + the target leaf hash must verify against the batch root
    assert verify_merkle_proof(ch.merkle_proof, root, hash_leaf(ch.leaf))


def test_aux_data_decodes_to_committed_pubkey_and_signature():
    from eth_abi import decode as abi_decode
    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = _batch([(a, True), (b, False)])
    ch = assemble_invalid_signature_challenge(
        batch_id=b"\x33" * 32, batch_receipts=batch, target_index=1)
    pub, sig = abi_decode(["bytes", "bytes"], ch.aux_data)
    assert pub == b64decode(batch[1].receipt.provider_pubkey_b64)
    assert sig == b64decode(batch[1].receipt.signature)


def test_refuses_to_assemble_against_a_valid_signature():
    """Fail-fast: a receipt whose committed signature VERIFIES has no fraud to prove."""
    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = _batch([(a, True), (b, True)])  # both valid
    with pytest.raises(ValueError, match="VERIFIES"):
        assemble_invalid_signature_challenge(
            batch_id=b"\x44" * 32, batch_receipts=batch, target_index=0)


def test_out_of_range_index_rejected():
    a = generate_node_identity("a")
    batch = _batch([(a, False)])
    with pytest.raises(ValueError, match="out of range"):
        assemble_invalid_signature_challenge(
            batch_id=b"\x55" * 32, batch_receipts=batch, target_index=5)


def test_call_args_shape():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = _batch([(a, True), (b, False)])
    ch = assemble_invalid_signature_challenge(
        batch_id=b"\x66" * 32, batch_receipts=batch, target_index=1)
    args = ch.to_call_args()
    assert args[0] == b"\x66" * 32          # batchId
    assert isinstance(args[1], tuple) and len(args[1]) == 9  # leaf struct
    assert isinstance(args[2], list)        # merkleProof
    assert args[3] == 1                     # reason code
    assert isinstance(args[4], bytes)       # auxData


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
