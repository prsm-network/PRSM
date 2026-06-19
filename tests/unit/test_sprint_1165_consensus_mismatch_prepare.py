"""Sprint 1165 — complete prepare coverage to ALL 5 on-chain reason codes: consensus_mismatch.

sp1163's bridge handled 4 of the 5 challengeReceipt reasons (invalid_signature=1,
double_spend=0, no_escrow=2, expired=3) but EXCLUDED consensus_mismatch=5 — it has its own
ConsensusChallengeSubmitter with a different auxData layout (abi.encode(majorityBatchId,
majorityProof, majorityLeaf)). So an operator who DETECTED a consensus disagreement saw it
surfaced (sp1149, on_chain_reason=5) but could not PREPARE it.

sp1165 closes that:
  - prsm/settlement/consensus_mismatch_assembler.py: ConsensusMismatchChallenge (reason_code=5,
    to_call_args() matching the challengeReceipt ABI) + assemble_consensus_mismatch_challenge,
    mirroring the audit engine's proven _try_assemble_consensus_mismatch (ReceiptLeafFields +
    build_merkle_proof + the consensus aux);
  - ConsensusChallengeSubmitter gains a KEYLESS read-only dry_run + an injection seam (the
    verify-before-broadcast pattern the rest of the codebase uses; sp1151 principle);
  - challenge_preparer routes consensus_mismatch through the assembler (no longer
    UnsupportedChallengeReason).

NEVER broadcasts — assemble + read-only dry-run only.
"""
from __future__ import annotations

import pytest

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.economy.web3.stake_manager import ReasonCode
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.merkle import batched_receipt_to_leaf, build_merkle_root, hash_leaf
from prsm.settlement.settlement_audit_report import AuditFindingRecord

from prsm.settlement.consensus_mismatch_assembler import (
    ConsensusMismatchChallenge,
    assemble_consensus_mismatch_challenge,
)
from prsm.settlement.challenge_preparer import (
    PreparedChallenge,
    dry_run_prepared,
    prepare_challenge_from_finding,
)

REASON_CONSENSUS_MISMATCH = int(ReasonCode.CONSENSUS_MISMATCH)  # == 5


def _receipt(identity, *, job="job-1", shard=0, out=None, provider="0x" + "2" * 40):
    out = out or ("ab" * 32)
    sig = identity.sign(build_receipt_signing_payload(
        job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000))
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address="0x" + "1" * 40, provider_address=provider,
        value_ftns=10 ** 18, local_escrow_id="e1")


def _two_provider_disagreement():
    """A minority + majority batch: same job+shard, DIFFERENT providers, DIFFERENT outputs."""
    p_min, p_maj = generate_node_identity("minority"), generate_node_identity("majority")
    minority = [_receipt(p_min, job="J", shard=0, out=("11" * 32),
                         provider="0x" + "a" * 40)]
    majority = [_receipt(p_maj, job="J", shard=0, out=("22" * 32),
                         provider="0x" + "b" * 40)]
    return minority, majority


def _consensus_record(min_bid, maj_bid, *, min_idx=0, maj_idx=0):
    # mirrors settlement_audit_report._consensus_mismatch_record's detail shape
    return AuditFindingRecord(
        reason="consensus_mismatch", batch_id_hex=min_bid, target_index=min_idx,
        on_chain_reason=REASON_CONSENSUS_MISMATCH,
        detail={
            "consensus_group_id": "33" * 32, "job_id_hash": "44" * 32, "shard_index": 0,
            "minority_batch_id": min_bid, "minority_leaf_index": min_idx,
            "minority_output_hash": "11" * 32,
            "majority_batch_id": maj_bid, "majority_leaf_index": maj_idx,
            "majority_output_hash": "22" * 32,
        })


def _lookup_from(mapping):
    norm = {k.lower().removeprefix("0x"): v for k, v in mapping.items()}
    return lambda bid: norm.get(str(bid).lower().removeprefix("0x"))


# ── the assembler ────────────────────────────────────────────────────────────────────


def test_assemble_consensus_mismatch_challenge_shape():
    minority, majority = _two_provider_disagreement()
    min_bid, maj_bid = b"\xaa" * 32, b"\xbb" * 32
    ch = assemble_consensus_mismatch_challenge(
        minority_batch_id=min_bid, minority_receipts=minority, minority_index=0,
        majority_batch_id=maj_bid, majority_receipts=majority, majority_index=0)
    assert isinstance(ch, ConsensusMismatchChallenge)
    assert ch.reason_code == REASON_CONSENSUS_MISMATCH == 5
    # to_call_args matches the challengeReceipt ABI: (batchId, leafTuple, proof, reason, aux)
    args = ch.to_call_args()
    assert len(args) == 5
    assert args[0] == min_bid              # minority batch id
    assert isinstance(args[1], tuple) and len(args[1]) == 9   # the minority leaf tuple
    assert args[3] == REASON_CONSENSUS_MISMATCH
    assert args[4]                          # aux (majority batch id + proof + leaf), non-empty


def test_assembled_minority_leaf_matches_the_committed_minority_leaf():
    minority, majority = _two_provider_disagreement()
    ch = assemble_consensus_mismatch_challenge(
        minority_batch_id=b"\xaa" * 32, minority_receipts=minority, minority_index=0,
        majority_batch_id=b"\xbb" * 32, majority_receipts=majority, majority_index=0)
    # the leaf tuple's outputHash field == the minority receipt's committed output (32 bytes)
    leaf_tuple = ch.to_call_args()[1]
    assert leaf_tuple[4] == bytes.fromhex("11" * 32)  # outputHash field


def test_assembled_minority_leaf_verifies_against_a_real_committed_root():
    # REGRESSION (sp1165 verify, BLOCKER): the assembled minority leaf tuple MUST re-hash into
    # the PRODUCTION committed root (the merkle.batched_receipt_to_leaf convention — keccak(
    # b64decode(pubkey/sig))) and its proof MUST verify. Otherwise the on-chain challengeReceipt
    # reverts InvalidMerkleProof on the top-level leaf BEFORE the consensus reason dispatch, so
    # the challenge could never slash a real disagreement. (The pre-fix consensus convention
    # hashed keccak(utf8(b64)) and failed this cross-check.)
    from eth_abi import encode as abi_encode
    from eth_utils import keccak

    from prsm.settlement.merkle import _LEAF_SIGNATURE, build_merkle_root, verify_merkle_proof

    p0, p1 = generate_node_identity("p0"), generate_node_identity("p1")
    # a 2-leaf minority batch -> challenge leaf at index 1 exercises a NON-trivial proof
    minority = [_receipt(p0, job="J", shard=0, out=("11" * 32), provider="0x" + "a" * 40),
                _receipt(p1, job="J", shard=1, out=("33" * 32), provider="0x" + "a" * 40)]
    majority = [_receipt(generate_node_identity("pm"), job="J", shard=0, out=("22" * 32),
                         provider="0x" + "b" * 40)]
    committed_root = build_merkle_root(
        [hash_leaf(batched_receipt_to_leaf(br)) for br in minority])
    ch = assemble_consensus_mismatch_challenge(
        minority_batch_id=b"\xaa" * 32, minority_receipts=minority, minority_index=1,
        majority_batch_id=b"\xbb" * 32, majority_receipts=majority, majority_index=0)
    _bid, leaf_tuple, proof, _reason, _aux = ch.to_call_args()
    # the on-chain re-hash of the SUBMITTED leaf tuple must be provable in the committed tree
    leaf_hash = keccak(abi_encode([_LEAF_SIGNATURE], [tuple(leaf_tuple)]))
    assert verify_merkle_proof(proof, committed_root, leaf_hash)


def test_assembler_index_out_of_range_fails_closed():
    minority, majority = _two_provider_disagreement()
    with pytest.raises((ValueError, IndexError)):
        assemble_consensus_mismatch_challenge(
            minority_batch_id=b"\xaa" * 32, minority_receipts=minority, minority_index=9,
            majority_batch_id=b"\xbb" * 32, majority_receipts=majority, majority_index=0)


# ── the bridge now prepares consensus_mismatch (was UnsupportedChallengeReason) ──────


def test_prepare_challenge_handles_consensus_mismatch():
    minority, majority = _two_provider_disagreement()
    min_bid, maj_bid = "aa" * 32, "bb" * 32
    rec = _consensus_record(min_bid, maj_bid)
    prepared = prepare_challenge_from_finding(
        record=rec,
        receipt_lookup=_lookup_from({min_bid: minority, maj_bid: majority}))
    assert isinstance(prepared, PreparedChallenge)
    assert prepared.reason == "consensus_mismatch"
    assert prepared.on_chain_reason == REASON_CONSENSUS_MISMATCH == 5
    assert prepared.challenge.reason_code == 5
    # the prepared challenge renders call args uniformly (reason 5)
    d = prepared.to_dict()
    assert d["on_chain_reason"] == 5
    assert d["call_args"]["reason"] == 5
    assert isinstance(d["call_args"]["leaf"], list) and len(d["call_args"]["leaf"]) == 9


def test_consensus_mismatch_missing_majority_batch_fails_closed():
    from prsm.settlement.challenge_preparer import MissingBatchReceipts
    minority, _majority = _two_provider_disagreement()
    min_bid, maj_bid = "aa" * 32, "bb" * 32
    rec = _consensus_record(min_bid, maj_bid)
    with pytest.raises(MissingBatchReceipts):
        prepare_challenge_from_finding(
            record=rec, receipt_lookup=_lookup_from({min_bid: minority}))  # no majority


def test_consensus_mismatch_detail_missing_majority_fields_is_skipped_by_batch_api():
    # a malformed consensus finding (no majority_batch_id) must be SKIPPED by prepare_findings,
    # never raise out (the never-raises batch contract).
    from prsm.settlement.challenge_preparer import prepare_findings
    bad = AuditFindingRecord(reason="consensus_mismatch", batch_id_hex="aa" * 32,
                             target_index=0, on_chain_reason=5,
                             detail={"minority_batch_id": "aa" * 32})  # no majority_*
    prepared, skipped = prepare_findings([bad], _lookup_from({}))
    assert prepared == []
    assert len(skipped) == 1 and skipped[0].reason == "consensus_mismatch"


# ── the keyless, read-only dry-run on the consensus submitter ────────────────────────


class _FakeRegistryFn:
    def __init__(self, recorder, *, revert=None):
        self._recorder = recorder
        self._revert = revert

    def call(self, opts):
        self._recorder["call_opts"] = opts
        if self._revert is not None:
            raise RuntimeError(self._revert)
        return []


class _FakeRegistry:
    def __init__(self, recorder, *, revert=None):
        self._recorder = recorder
        self._revert = revert

        class _Fns:
            def challengeReceipt(_self, *args):
                recorder["call_args"] = args
                return _FakeRegistryFn(recorder, revert=revert)
        self.functions = _Fns()


class _FromOnly:
    address = "0x" + "c" * 40


def test_consensus_submitter_keyless_dry_run_is_read_only():
    from prsm.marketplace.consensus_submitter import ConsensusChallengeSubmitter

    minority, majority = _two_provider_disagreement()
    ch = assemble_consensus_mismatch_challenge(
        minority_batch_id=b"\xaa" * 32, minority_receipts=minority, minority_index=0,
        majority_batch_id=b"\xbb" * 32, majority_receipts=majority, majority_index=0)

    rec = {}
    # injection seam: web3=None acceptable (dry_run only touches registry + account.address)
    submitter = ConsensusChallengeSubmitter(
        web3=None, registry=_FakeRegistry(rec), account=_FromOnly())
    verdict = submitter.dry_run(ch)
    assert verdict.would_succeed is True
    # the static call was made with the consensus call args + from-address (NO broadcast)
    assert rec["call_args"][3] == REASON_CONSENSUS_MISMATCH
    assert rec["call_opts"]["from"] == _FromOnly.address


def test_consensus_submitter_dry_run_surfaces_revert():
    from prsm.marketplace.consensus_submitter import ConsensusChallengeSubmitter
    minority, majority = _two_provider_disagreement()
    ch = assemble_consensus_mismatch_challenge(
        minority_batch_id=b"\xaa" * 32, minority_receipts=minority, minority_index=0,
        majority_batch_id=b"\xbb" * 32, majority_receipts=majority, majority_index=0)
    submitter = ConsensusChallengeSubmitter(
        web3=None, registry=_FakeRegistry({}, revert="ChallengeNotProven"),
        account=_FromOnly())
    verdict = submitter.dry_run(ch)
    assert verdict.would_succeed is False
    assert "ChallengeNotProven" in (verdict.revert_reason or "")


def test_dry_run_prepared_routes_consensus_through_the_submitter():
    # the bridge's dry_run_prepared works uniformly for a consensus PreparedChallenge.
    from prsm.marketplace.consensus_submitter import ConsensusChallengeSubmitter
    minority, majority = _two_provider_disagreement()
    min_bid, maj_bid = "aa" * 32, "bb" * 32
    prepared = prepare_challenge_from_finding(
        record=_consensus_record(min_bid, maj_bid),
        receipt_lookup=_lookup_from({min_bid: minority, maj_bid: majority}))
    submitter = ConsensusChallengeSubmitter(
        web3=None, registry=_FakeRegistry({}), account=_FromOnly())
    verdict = dry_run_prepared(prepared, submitter)
    assert verdict.would_succeed is True
