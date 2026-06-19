"""Sprint 1170 — money-safety fix: the consensus-mismatch detector must pick the slash target
by a real output-hash VOTE, not by batch_id sort order.

THE BUG (caught by the sp1149-1169 holistic review). detect_consensus_mismatches labelled the
lower-sorted side (by batch_id) "minority" (the slash target) for EVERY disagreeing pair, with no
vote count. The on-chain _handleConsensusMismatch is SYMMETRIC (it slashes whichever side is
challenged), so in a genuine k-of-n group the HONEST MAJORITY would be slashed whenever its
batch_ids sorted below the faulty minority's — and the dry-run gate gives no protection (the
on-chain check passes for the wrong-side direction too).

THE FIX. Within each (group, job, shard) slot, tally DISTINCT PROVIDERS per output_hash:
  - a STRICT majority output (> half the providers) is the honest answer -> only NON-majority
    providers are challengeable (ambiguous=False); a majority provider is NEVER targeted;
  - NO strict majority (a tie / split) -> ambiguous=True: surfaced for MANUAL review, and the
    engine + prepare bridge REFUSE to auto-assemble a slash (the honest side is undeterminable).
"""
from __future__ import annotations

import pytest

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.consensus_mismatch_detector import (
    DetectorBatch,
    detect_consensus_mismatches,
)

_G = b"\x11" * 32


def _receipt(identity, *, out, job="J", shard=0):
    sig = identity.sign(build_receipt_signing_payload(
        job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000))
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address="0x" + "1" * 40, provider_address="0x" + "2" * 40,
        value_ftns=10 ** 18, local_escrow_id="e")


def _dbatch(batch_id, provider, identity, out):
    return DetectorBatch(
        batch_id=batch_id, provider_address=provider, consensus_group_id=_G,
        receipts=[_receipt(identity, out=out)])


# ── THE BUG REPRO: an honest majority must NEVER be the slash target ─────────────────


def test_honest_majority_is_never_the_slash_target():
    # 2 honest providers agree on X with LOW batch_ids; 1 faulty on Y with a HIGH batch_id —
    # exactly the ordering that the old positional labelling slashed the honest majority for.
    X, Y = ("aa" * 32), ("bb" * 32)
    a, b, c = (generate_node_identity(n) for n in ("a", "b", "c"))
    batches = [
        _dbatch(b"\x01" * 32, "0x" + "a" * 40, a, X),   # honest, low batch_id
        _dbatch(b"\x02" * 32, "0x" + "b" * 40, b, X),   # honest, low batch_id
        _dbatch(b"\x03" * 32, "0x" + "c" * 40, c, Y),   # FAULTY minority, high batch_id
    ]
    findings = detect_consensus_mismatches(batches)
    assert findings, "expected the faulty minority to be flagged"
    for f in findings:
        assert f.ambiguous is False                         # a strict 2-of-3 majority exists
        assert f.minority_output_hash == bytes.fromhex(Y)   # the FAULTY output is the target
        assert f.minority_provider == "0x" + "c" * 40       # the faulty provider, NOT a/b
        assert f.majority_output_hash == bytes.fromhex(X)   # proven against the honest output
        assert f.majority_provider in ("0x" + "a" * 40, "0x" + "b" * 40)
    # the honest providers are NEVER a minority (slash target)
    assert all(f.minority_provider == "0x" + "c" * 40 for f in findings)
    # vote tally reflects 2 (X) vs 1 (Y)
    assert findings[0].vote_tally == {X: 2, Y: 1}


def test_strict_majority_emits_one_finding_per_faulty_provider():
    # 3 honest on X, 2 faulty providers on different wrong outputs Y, Z -> 2 challengeable.
    X, Y, Z = ("a0" * 32), ("b0" * 32), ("c0" * 32)
    ids = [generate_node_identity(str(i)) for i in range(5)]
    batches = [
        _dbatch(b"\x01" * 32, "0x" + "1" * 40, ids[0], X),
        _dbatch(b"\x02" * 32, "0x" + "2" * 40, ids[1], X),
        _dbatch(b"\x03" * 32, "0x" + "3" * 40, ids[2], X),
        _dbatch(b"\x04" * 32, "0x" + "4" * 40, ids[3], Y),
        _dbatch(b"\x05" * 32, "0x" + "5" * 40, ids[4], Z),
    ]
    findings = detect_consensus_mismatches(batches)
    minority_providers = {f.minority_provider for f in findings}
    assert minority_providers == {"0x" + "4" * 40, "0x" + "5" * 40}  # only the 2 faulty
    assert all(f.ambiguous is False for f in findings)
    assert all(f.majority_output_hash == bytes.fromhex(X) for f in findings)


# ── TIES / SPLITS ARE AMBIGUOUS (not auto-challengeable) ─────────────────────────────


def test_one_vs_one_is_ambiguous():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    batches = [
        _dbatch(b"\x01" * 32, "0x" + "a" * 40, a, "aa" * 32),
        _dbatch(b"\x02" * 32, "0x" + "b" * 40, b, "bb" * 32),
    ]
    findings = detect_consensus_mismatches(batches)
    assert findings, "the disagreement is still SURFACED"
    assert all(f.ambiguous is True for f in findings)   # but ambiguous (no majority)


def test_two_vs_two_tie_is_ambiguous():
    X, Y = ("aa" * 32), ("bb" * 32)
    ids = [generate_node_identity(str(i)) for i in range(4)]
    batches = [
        _dbatch(b"\x01" * 32, "0x" + "1" * 40, ids[0], X),
        _dbatch(b"\x02" * 32, "0x" + "2" * 40, ids[1], X),
        _dbatch(b"\x03" * 32, "0x" + "3" * 40, ids[2], Y),
        _dbatch(b"\x04" * 32, "0x" + "4" * 40, ids[3], Y),
    ]
    findings = detect_consensus_mismatches(batches)
    assert findings
    assert all(f.ambiguous is True for f in findings)   # 2-vs-2 has no strict majority


def test_all_agree_no_finding():
    a, b, c = (generate_node_identity(n) for n in ("a", "b", "c"))
    batches = [
        _dbatch(b"\x01" * 32, "0x" + "1" * 40, a, "aa" * 32),
        _dbatch(b"\x02" * 32, "0x" + "2" * 40, b, "aa" * 32),
        _dbatch(b"\x03" * 32, "0x" + "3" * 40, c, "aa" * 32),
    ]
    assert detect_consensus_mismatches(batches) == []


def test_duplicate_output_by_one_provider_does_not_inflate_the_vote():
    # one provider committing the SAME output in two batches counts as ONE vote, so a 1-(real
    # provider)-vs-1 with a duplicate is still a tie (ambiguous), not a fake majority.
    X, Y = ("aa" * 32), ("bb" * 32)
    a, b = generate_node_identity("a"), generate_node_identity("b")
    pa = "0x" + "a" * 40
    batches = [
        _dbatch(b"\x01" * 32, pa, a, X),
        _dbatch(b"\x02" * 32, pa, a, X),               # SAME provider A, duplicate output X
        _dbatch(b"\x03" * 32, "0x" + "b" * 40, b, Y),
    ]
    findings = detect_consensus_mismatches(batches)
    # provider-count: X -> {A} (1 distinct provider), Y -> {B} (1) -> NO strict majority -> ambiguous
    assert findings
    assert all(f.ambiguous is True for f in findings)
    assert findings[0].vote_tally == {X: 1, Y: 1}


# ── the prepare bridge REFUSES an ambiguous finding ──────────────────────────────────


def test_prepare_refuses_ambiguous_consensus_finding():
    from prsm.settlement.challenge_preparer import (
        UnsupportedChallengeReason,
        prepare_challenge_from_finding,
    )
    from prsm.settlement.settlement_audit_report import AuditFindingRecord

    rec = AuditFindingRecord(
        reason="consensus_mismatch", batch_id_hex="aa" * 32, target_index=0, on_chain_reason=5,
        detail={"ambiguous": True, "vote_tally": {"aa": 1, "bb": 1},
                "minority_batch_id": "aa" * 32, "minority_leaf_index": 0,
                "majority_batch_id": "bb" * 32, "majority_leaf_index": 0})
    with pytest.raises(UnsupportedChallengeReason):
        prepare_challenge_from_finding(record=rec, receipt_lookup=lambda b: None)


def test_prepare_fail_closed_when_consensus_finding_omits_the_majority_determination():
    # sp1170 defense-in-depth: a consensus record that OMITS `ambiguous` entirely (a
    # hand-built/legacy record bypassing the detector) must FAIL CLOSED — refused, not
    # auto-prepared (we can't confirm a strict majority was determined).
    from prsm.settlement.challenge_preparer import (
        UnsupportedChallengeReason,
        prepare_challenge_from_finding,
    )
    from prsm.settlement.settlement_audit_report import AuditFindingRecord

    rec = AuditFindingRecord(
        reason="consensus_mismatch", batch_id_hex="aa" * 32, target_index=0, on_chain_reason=5,
        detail={"minority_batch_id": "aa" * 32, "minority_leaf_index": 0,
                "majority_batch_id": "bb" * 32, "majority_leaf_index": 0})  # NO 'ambiguous'
    with pytest.raises(UnsupportedChallengeReason):
        prepare_challenge_from_finding(record=rec, receipt_lookup=lambda b: None)


def test_prepare_findings_handles_non_dict_detail_without_opaque_error():
    # sp1170 nit fix: a non-dict detail must not raise an opaque AttributeError out of the
    # never-raises batch API — it falls through to a clean SkippedFinding.
    from prsm.settlement.challenge_preparer import prepare_findings
    from prsm.settlement.settlement_audit_report import AuditFindingRecord

    bad = AuditFindingRecord(reason="consensus_mismatch", batch_id_hex="aa" * 32,
                             target_index=0, on_chain_reason=5, detail={})
    # AuditFindingRecord coerces detail to a dict, so force a non-dict via a duck-typed stub:
    class _Rec:
        reason = "consensus_mismatch"
        batch_id_hex = "aa" * 32
        target_index = 0
        detail = "not-a-dict"
    prepared, skipped = prepare_findings([_Rec()], lambda b: None)
    assert prepared == []
    assert len(skipped) == 1
    assert "AttributeError" not in skipped[0].message  # clean ValueError, not opaque
