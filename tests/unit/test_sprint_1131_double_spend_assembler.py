"""Sprint 1131 — challenge/dispute brick 7: assemble the on-chain DOUBLE_SPEND
challengeReceipt inputs (leaf + target merkle proof + auxData) from two committed
batches that each commit the SAME receipt leaf.

This mirrors the brick-3 INVALID_SIGNATURE assembler (sprint 1118). The on-chain
_handleDoubleSpend (contracts/contracts/BatchSettlementRegistry.sol:751-767) proves
the challenge iff the SAME leafHash is provable in BOTH batchId's merkleRoot (verified
caller-side) AND conflictingBatchId's merkleRoot (verified via the conflictingProof in
auxData). So the assembler builds BOTH proofs and binds them.

Fail-fast: refuses to assemble a challenge that would revert on-chain — the named leaf
must GENUINELY appear in the conflicting batch (hash_leaf(target_leaf) ==
conflicting_leaf_hashes[conflicting_index]), indices must be in range, and the two
batches must be DISTINCT (a leaf duplicated within ONE batch is not expressible via the
conflicting-batch auxData layout).
"""
from __future__ import annotations

import pytest
from eth_abi import decode as abi_decode

from prsm.compute.shard_receipt import build_receipt_signing_payload
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.client import CommittedBatch, TriggerReason
from prsm.settlement.double_spend_assembler import (
    REASON_DOUBLE_SPEND,
    DoubleSpendChallenge,
    assemble_double_spend_challenge,
    assemble_from_double_spend_finding,
)
from prsm.settlement.double_spend_detector import detect_double_spends
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_merkle_root,
    hash_leaf,
    verify_merkle_proof,
)


def _receipt(identity, *, job="job-1", shard=0, out=None):
    out = out or ("ab" * 32)
    payload = build_receipt_signing_payload(
        job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000)
    sig = identity.sign(payload)
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address="0x" + "1" * 40,
        provider_address="0x" + "2" * 40, value_ftns=10**18, local_escrow_id="e1")


def _leaf_hashes(batch):
    return [hash_leaf(batched_receipt_to_leaf(br)) for br in batch]


def test_assembles_when_leaf_committed_in_two_batches():
    """Happy path: the SAME receipt leaf sits in both batches → assemble succeeds, and
    BOTH the target proof and the auxData conflictingProof verify the SAME leafHash
    against their respective roots (exactly what the contract checks)."""
    a, b, c, d = (generate_node_identity(n) for n in ("a", "b", "c", "d"))
    shared = _receipt(a, job="dup-job", shard=0)

    target_batch = [shared, _receipt(b, shard=1), _receipt(c, shard=2)]
    conflicting_batch = [_receipt(d, shard=3), _receipt(b, shard=4), shared]

    target_id = b"\x11" * 32
    conflicting_id = b"\x22" * 32
    conflicting_hashes = _leaf_hashes(conflicting_batch)

    ch = assemble_double_spend_challenge(
        target_batch_id=target_id,
        target_batch_receipts=target_batch,
        target_index=0,
        conflicting_batch_id=conflicting_id,
        conflicting_leaf_hashes=conflicting_hashes,
        conflicting_index=2,
    )

    assert isinstance(ch, DoubleSpendChallenge)
    assert ch.reason_code == REASON_DOUBLE_SPEND == 0
    assert ch.batch_id == target_id
    assert ch.leaf == batched_receipt_to_leaf(shared)

    leaf_hash = hash_leaf(ch.leaf)
    # 1) target proof verifies leaf against target root
    target_root = build_merkle_root(_leaf_hashes(target_batch))
    assert verify_merkle_proof(ch.merkle_proof, target_root, leaf_hash)
    # 2) conflictingProof (from auxData) verifies the SAME leafHash against conflicting root
    conflicting_id_dec, conflicting_proof = abi_decode(
        ["bytes32", "bytes32[]"], ch.aux_data)
    conflicting_root = build_merkle_root(conflicting_hashes)
    assert verify_merkle_proof(conflicting_proof, conflicting_root, leaf_hash)


def test_failfast_conflicting_batch_does_not_contain_leaf():
    """Fail-fast: the conflicting_index points at a DIFFERENT leaf → ValueError, no
    challenge (the on-chain conflictingProof would not verify → ChallengeNotProven)."""
    a, b, c = (generate_node_identity(n) for n in ("a", "b", "c"))
    shared = _receipt(a, job="dup-job", shard=0)
    target_batch = [shared, _receipt(b, shard=1)]
    conflicting_batch = [_receipt(c, shard=2), _receipt(b, shard=3)]  # no `shared`

    with pytest.raises(ValueError, match="does not contain|same leaf|not.*conflicting"):
        assemble_double_spend_challenge(
            target_batch_id=b"\x11" * 32,
            target_batch_receipts=target_batch,
            target_index=0,
            conflicting_batch_id=b"\x22" * 32,
            conflicting_leaf_hashes=_leaf_hashes(conflicting_batch),
            conflicting_index=0,
        )


def test_failfast_same_batch_id_rejected():
    """Fail-fast: target_batch_id == conflicting_batch_id → ValueError (a leaf
    duplicated within ONE batch is not expressible via the conflicting-batch auxData)."""
    a, b = generate_node_identity("a"), generate_node_identity("b")
    shared = _receipt(a, job="dup-job", shard=0)
    batch = [shared, _receipt(b, shard=1)]
    same_id = b"\x33" * 32

    with pytest.raises(ValueError, match="distinct|same batch|DISTINCT"):
        assemble_double_spend_challenge(
            target_batch_id=same_id,
            target_batch_receipts=batch,
            target_index=0,
            conflicting_batch_id=same_id,
            conflicting_leaf_hashes=_leaf_hashes(batch),
            conflicting_index=0,
        )


def test_failfast_target_index_out_of_range():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    shared = _receipt(a, job="dup-job", shard=0)
    target_batch = [shared, _receipt(b, shard=1)]
    conflicting_batch = [shared]

    with pytest.raises(ValueError, match="out of range"):
        assemble_double_spend_challenge(
            target_batch_id=b"\x11" * 32,
            target_batch_receipts=target_batch,
            target_index=9,
            conflicting_batch_id=b"\x22" * 32,
            conflicting_leaf_hashes=_leaf_hashes(conflicting_batch),
            conflicting_index=0,
        )


def test_failfast_conflicting_index_out_of_range():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    shared = _receipt(a, job="dup-job", shard=0)
    target_batch = [shared, _receipt(b, shard=1)]
    conflicting_batch = [shared]

    with pytest.raises(ValueError, match="out of range"):
        assemble_double_spend_challenge(
            target_batch_id=b"\x11" * 32,
            target_batch_receipts=target_batch,
            target_index=0,
            conflicting_batch_id=b"\x22" * 32,
            conflicting_leaf_hashes=_leaf_hashes(conflicting_batch),
            conflicting_index=9,
        )


def test_aux_data_round_trips():
    """auxData = abi.encode(bytes32 conflictingBatchId, bytes32[] conflictingProof) —
    decode yields exactly the conflicting batch id + the built conflictingProof."""
    a, b, c = (generate_node_identity(n) for n in ("a", "b", "c"))
    shared = _receipt(a, job="dup-job", shard=0)
    target_batch = [shared, _receipt(b, shard=1)]
    conflicting_batch = [_receipt(c, shard=2), shared, _receipt(b, shard=3)]
    conflicting_id = b"\xab" * 32
    conflicting_hashes = _leaf_hashes(conflicting_batch)

    from prsm.settlement.merkle import build_merkle_proof
    expected_proof = build_merkle_proof(conflicting_hashes, 1)

    ch = assemble_double_spend_challenge(
        target_batch_id=b"\x11" * 32,
        target_batch_receipts=target_batch,
        target_index=0,
        conflicting_batch_id=conflicting_id,
        conflicting_leaf_hashes=conflicting_hashes,
        conflicting_index=1,
    )

    dec_id, dec_proof = abi_decode(["bytes32", "bytes32[]"], ch.aux_data)
    assert dec_id == conflicting_id
    assert [bytes(p) for p in dec_proof] == [bytes(p) for p in expected_proof]


def test_call_args_shape():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    shared = _receipt(a, job="dup-job", shard=0)
    target_batch = [shared, _receipt(b, shard=1)]
    conflicting_batch = [shared]
    ch = assemble_double_spend_challenge(
        target_batch_id=b"\x66" * 32,
        target_batch_receipts=target_batch,
        target_index=0,
        conflicting_batch_id=b"\x77" * 32,
        conflicting_leaf_hashes=_leaf_hashes(conflicting_batch),
        conflicting_index=0,
    )
    args = ch.to_call_args()
    assert args[0] == b"\x66" * 32              # batchId
    assert isinstance(args[1], tuple) and len(args[1]) == 9  # leaf struct
    assert isinstance(args[2], list)            # merkleProof
    assert args[3] == 0                         # reason code DOUBLE_SPEND
    assert isinstance(args[4], bytes)           # auxData


def _committed(batch_id, batch, provider="0x" + "2" * 40):
    leaf_hashes = tuple(_leaf_hashes(batch))
    return CommittedBatch(
        batch_id=batch_id, tx_hash="0xdead", provider_address=provider,
        requester_address="0x" + "1" * 40, merkle_root=build_merkle_root(list(leaf_hashes)),
        receipt_count=len(batch), total_value_ftns=sum(br.value_ftns for br in batch),
        commit_timestamp=1000, leaf_hashes=leaf_hashes, trigger_reason=TriggerReason.COUNT)


def test_from_double_spend_finding_helper():
    """The from-finding helper: feed a REAL detector finding (>=2 occurrences in
    distinct batches) + the two batches' data → a valid challenge whose BOTH proofs
    verify."""
    a, b, c, d = (generate_node_identity(n) for n in ("a", "b", "c", "d"))
    shared = _receipt(a, job="dup-job", shard=0)
    target_batch = [shared, _receipt(b, shard=1)]
    conflicting_batch = [_receipt(c, shard=2), _receipt(d, shard=3), shared]

    target_id = b"\x11" * 32
    conflicting_id = b"\x22" * 32
    cb_target = _committed(target_id, target_batch)
    cb_conflicting = _committed(conflicting_id, conflicting_batch)

    findings = detect_double_spends([cb_target, cb_conflicting])
    assert len(findings) == 1
    finding = findings[0]
    assert len(finding.occurrences) >= 2

    # receipts only retained for the TARGET batch; conflicting uses retained leaf_hashes.
    receipts_by_batch = {target_id: target_batch}
    leaf_hashes_by_batch = {
        target_id: cb_target.leaf_hashes,
        conflicting_id: cb_conflicting.leaf_hashes,
    }

    ch = assemble_from_double_spend_finding(
        finding=finding,
        receipts_by_batch=receipts_by_batch,
        leaf_hashes_by_batch=leaf_hashes_by_batch,
    )
    assert ch.reason_code == REASON_DOUBLE_SPEND == 0
    leaf_hash = hash_leaf(ch.leaf)
    assert leaf_hash == finding.leaf_hash

    # both proofs verify
    target_root = build_merkle_root(list(cb_target.leaf_hashes))
    assert verify_merkle_proof(ch.merkle_proof, target_root, leaf_hash)
    dec_id, dec_proof = abi_decode(["bytes32", "bytes32[]"], ch.aux_data)
    other_root = build_merkle_root(list(leaf_hashes_by_batch[dec_id]))
    assert verify_merkle_proof(dec_proof, other_root, leaf_hash)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
