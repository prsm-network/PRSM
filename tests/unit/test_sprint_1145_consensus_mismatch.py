"""Sprint 1145 — CONSENSUS_MISMATCH detection (the 3rd on-chain-actionable settlement
fraud class, after INVALID_SIGNATURE + DOUBLE_SPEND).

CONSENSUS_MISMATCH catches co-group provider DISAGREEMENT: providers in a k-of-n
consensus group redundantly execute the same work and commit conflicting outputs.

The on-chain proof conditions (BatchSettlementRegistry _handleConsensusMismatch) the
detector must match EXACTLY (else a false mismatch -> wrongful slash):
  - the challenged (minority) batch has a NON-ZERO consensus_group_id;
  - minority & majority leaves share the SAME jobIdHash AND SAME shardIndex;
  - DIFFERENT outputHash (genuine disagreement);
  - both batches share the SAME non-zero group_id AND are committed by DIFFERENT providers;
  - distinct batch ids.

These tests inject everything (no network, no chain, no disk). They assert:
  (a) GENUINE MISMATCH -> exactly one finding naming both sides;
  (b) SOUNDNESS — same-provider, different shard/job, agreeing outputs, zero/absent group
      are NEVER flagged (the on-chain reject conditions; a false flag would wrongly slash);
  (c) scan_consensus_mismatches over a verified cache surfaces a planted co-group mismatch,
      sets dry_run_ok when a dry-run client is injected, and NEVER calls submit_one/broadcast;
  (d) only root-VERIFIED batches feed the scan (a rejected ingest is never scanned);
  (e) FAIL-CLOSED — a batch that makes assemble/detect raise is recorded+skipped, no crash;
  (f) LOOP e2e — run_once surfaces a planted consensus mismatch + counts it; never broadcasts.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_tree_and_proofs,
    hash_leaf,
)
from prsm.settlement.published_batch_store import PublishedBatch
from prsm.settlement.settlement_audit_engine import (
    ObservedBatch,
    VerifiedBatchCache,
    SettlementAuditEngine,
)
from prsm.settlement.consensus_mismatch_detector import (
    ConsensusMismatchFinding,
    DetectorBatch,
    detect_consensus_mismatches,
)
from prsm.marketplace.consensus_submitter import ChallengeAttempt

ZERO_GROUP = b"\x00" * 32
G1 = b"\x11" * 32
PROV_A = "0x" + "a" * 40
PROV_B = "0x" + "b" * 40
PROV_C = "0x" + "c" * 40
REQ = "0x" + "1" * 40


# ── builders ──────────────────────────────────────────────────────────


def _receipt(identity, *, job="job-1", shard=0, out=None,
             requester=REQ, provider=PROV_A, group=ZERO_GROUP, value=10 ** 18):
    out = out or ("ab" * 32)
    payload = build_receipt_signing_payload(
        job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000)
    sig = identity.sign(payload)
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address=requester, provider_address=provider,
        value_ftns=value, local_escrow_id="e1", consensus_group_id=group)


def _detector_batch(batch_id, provider, group, receipts):
    return DetectorBatch(
        batch_id=batch_id, provider_address=provider,
        consensus_group_id=group, receipts=list(receipts))


def _root_for(receipts):
    leaf_hashes = [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
    return build_tree_and_proofs(leaf_hashes).root


def _published(batch_id, receipts):
    return PublishedBatch(
        batch_id=batch_id, merkle_root=_root_for(receipts),
        commit_timestamp=1000, receipts=list(receipts))


def _observed(batch_id, receipts, *, provider=PROV_A, requester=REQ, group=ZERO_GROUP):
    return ObservedBatch(
        batch_id=batch_id, provider_address=provider,
        requester_address=requester, merkle_root=_root_for(receipts),
        consensus_group_id=group)


class _FakeDryRunClient:
    """Injected dry-run-only client. Records dry_run calls; submit_one/broadcast are mocks
    that MUST NEVER be called by the engine."""

    def __init__(self, would_succeed=True):
        self._would_succeed = would_succeed
        self.dry_run_calls = []
        self.submit_one = MagicMock(name="submit_one")
        self.submit_batch = MagicMock(name="submit_batch")
        self.broadcast = AsyncMock(name="broadcast")

    def dry_run(self, attempt):
        self.dry_run_calls.append(attempt)
        return _DryVerdict(self._would_succeed)


class _DryVerdict:
    def __init__(self, would_succeed):
        self.would_succeed = would_succeed
        self.revert_reason = None


# ── (a) GENUINE MISMATCH ──────────────────────────────────────────────


def test_genuine_consensus_mismatch_one_finding_naming_both_sides():
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    # Same non-zero group, DIFFERENT providers, same job+shard, DIFFERENT output.
    ra = _receipt(a, job="J", shard=3, out=("11" * 32), provider=PROV_A, group=G1)
    rb = _receipt(b, job="J", shard=3, out=("22" * 32), provider=PROV_B, group=G1)
    batch_a = _detector_batch(b"\xa1" * 32, PROV_A, G1, [ra])
    batch_b = _detector_batch(b"\xb1" * 32, PROV_B, G1, [rb])

    findings = detect_consensus_mismatches([batch_a, batch_b])
    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, ConsensusMismatchFinding)
    assert f.consensus_group_id == G1
    sides = {(f.minority_batch_id, f.minority_provider),
             (f.majority_batch_id, f.majority_provider)}
    assert sides == {(b"\xa1" * 32, PROV_A), (b"\xb1" * 32, PROV_B)}
    # job/shard identity is the agreed key; outputs differ.
    assert f.minority_receipt.receipt.output_hash != f.majority_receipt.receipt.output_hash


def test_deterministic_ordering_with_multiple_groups():
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    g2 = b"\x22" * 32
    ra1 = _receipt(a, job="J", shard=0, out=("11" * 32), provider=PROV_A, group=G1)
    rb1 = _receipt(b, job="J", shard=0, out=("22" * 32), provider=PROV_B, group=G1)
    ra2 = _receipt(a, job="K", shard=1, out=("33" * 32), provider=PROV_A, group=g2)
    rb2 = _receipt(b, job="K", shard=1, out=("44" * 32), provider=PROV_B, group=g2)
    batches = [
        _detector_batch(b"\xa1" * 32, PROV_A, G1, [ra1]),
        _detector_batch(b"\xb1" * 32, PROV_B, G1, [rb1]),
        _detector_batch(b"\xa2" * 32, PROV_A, g2, [ra2]),
        _detector_batch(b"\xb2" * 32, PROV_B, g2, [rb2]),
    ]
    f1 = detect_consensus_mismatches(batches)
    f2 = detect_consensus_mismatches(list(reversed(batches)))
    assert len(f1) == 2
    # Same findings regardless of input order.
    key = lambda fs: [(x.consensus_group_id, x.job_id_hash, x.shard_index) for x in fs]
    assert key(f1) == key(f2)


# ── (b) SOUNDNESS — the on-chain reject conditions ────────────────────


def test_same_provider_disagreement_not_flagged():
    a = generate_node_identity("a")
    r1 = _receipt(a, job="J", shard=0, out=("11" * 32), provider=PROV_A, group=G1)
    r2 = _receipt(a, job="J", shard=0, out=("22" * 32), provider=PROV_A, group=G1)
    batches = [
        _detector_batch(b"\x01" * 32, PROV_A, G1, [r1]),
        _detector_batch(b"\x02" * 32, PROV_A, G1, [r2]),
    ]
    assert detect_consensus_mismatches(batches) == []


def test_different_shard_not_flagged():
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    ra = _receipt(a, job="J", shard=0, out=("11" * 32), provider=PROV_A, group=G1)
    rb = _receipt(b, job="J", shard=1, out=("22" * 32), provider=PROV_B, group=G1)
    batches = [
        _detector_batch(b"\x01" * 32, PROV_A, G1, [ra]),
        _detector_batch(b"\x02" * 32, PROV_B, G1, [rb]),
    ]
    assert detect_consensus_mismatches(batches) == []


def test_different_job_not_flagged():
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    ra = _receipt(a, job="J", shard=0, out=("11" * 32), provider=PROV_A, group=G1)
    rb = _receipt(b, job="K", shard=0, out=("22" * 32), provider=PROV_B, group=G1)
    batches = [
        _detector_batch(b"\x01" * 32, PROV_A, G1, [ra]),
        _detector_batch(b"\x02" * 32, PROV_B, G1, [rb]),
    ]
    assert detect_consensus_mismatches(batches) == []


def test_agreeing_outputs_not_flagged():
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    same_out = "ee" * 32
    ra = _receipt(a, job="J", shard=0, out=same_out, provider=PROV_A, group=G1)
    rb = _receipt(b, job="J", shard=0, out=same_out, provider=PROV_B, group=G1)
    batches = [
        _detector_batch(b"\x01" * 32, PROV_A, G1, [ra]),
        _detector_batch(b"\x02" * 32, PROV_B, G1, [rb]),
    ]
    assert detect_consensus_mismatches(batches) == []


def test_zero_group_not_flagged():
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    ra = _receipt(a, job="J", shard=0, out=("11" * 32), provider=PROV_A, group=ZERO_GROUP)
    rb = _receipt(b, job="J", shard=0, out=("22" * 32), provider=PROV_B, group=ZERO_GROUP)
    batches = [
        _detector_batch(b"\x01" * 32, PROV_A, ZERO_GROUP, [ra]),
        _detector_batch(b"\x02" * 32, PROV_B, ZERO_GROUP, [rb]),
    ]
    assert detect_consensus_mismatches(batches) == []


def test_different_groups_not_flagged():
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    g2 = b"\x22" * 32
    ra = _receipt(a, job="J", shard=0, out=("11" * 32), provider=PROV_A, group=G1)
    rb = _receipt(b, job="J", shard=0, out=("22" * 32), provider=PROV_B, group=g2)
    batches = [
        _detector_batch(b"\x01" * 32, PROV_A, G1, [ra]),
        _detector_batch(b"\x02" * 32, PROV_B, g2, [rb]),
    ]
    assert detect_consensus_mismatches(batches) == []


# ── (c) scan_consensus_mismatches over the verified cache ─────────────


def _ingest_cogroup_pair(cache, *, out_a="11" * 32, out_b="22" * 32):
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    ra = _receipt(a, job="J", shard=2, out=out_a, provider=PROV_A, group=G1)
    rb = _receipt(b, job="J", shard=2, out=out_b, provider=PROV_B, group=G1)
    ba_id = b"\xa1" * 32
    bb_id = b"\xb1" * 32
    assert cache.ingest(_observed(ba_id, [ra], provider=PROV_A, group=G1),
                        _published(ba_id, [ra]))
    assert cache.ingest(_observed(bb_id, [rb], provider=PROV_B, group=G1),
                        _published(bb_id, [rb]))
    return ba_id, bb_id


def _ingest_cogroup_majority(cache, *, out_majority="11" * 32, out_minority="22" * 32):
    """sp1170 — a 2-of-3 STRICT majority: PROV_A + PROV_B agree on out_majority (the honest
    answer); PROV_C (lone faulty) commits out_minority. Returns (a_id, b_id, c_id). Only C is a
    legitimately-challengeable minority; A + B must NEVER be the slash target."""
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    c = generate_node_identity("c")
    ra = _receipt(a, job="J", shard=2, out=out_majority, provider=PROV_A, group=G1)
    rb = _receipt(b, job="J", shard=2, out=out_majority, provider=PROV_B, group=G1)
    rc = _receipt(c, job="J", shard=2, out=out_minority, provider=PROV_C, group=G1)
    a_id, b_id, c_id = b"\xa1" * 32, b"\xb1" * 32, b"\xc1" * 32
    assert cache.ingest(_observed(a_id, [ra], provider=PROV_A, group=G1), _published(a_id, [ra]))
    assert cache.ingest(_observed(b_id, [rb], provider=PROV_B, group=G1), _published(b_id, [rb]))
    assert cache.ingest(_observed(c_id, [rc], provider=PROV_C, group=G1), _published(c_id, [rc]))
    return a_id, b_id, c_id


def test_scan_surfaces_planted_mismatch_and_dry_runs_without_broadcast():
    # sp1170 — a genuine 2-of-3 STRICT majority: A+B agree, C is the lone faulty minority.
    # ONLY C is challengeable; the honest majority providers are NEVER the slash target.
    cache = VerifiedBatchCache()
    _ingest_cogroup_majority(cache)
    client = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=client)

    findings = engine.scan_consensus_mismatches()
    assert len(findings) == 1
    af = findings[0]
    assert af.finding.consensus_group_id == G1
    assert af.finding.ambiguous is False                  # a strict majority exists
    assert af.finding.minority_provider == PROV_C          # the FAULTY one, NOT honest A/B
    assert af.finding.majority_provider in (PROV_A, PROV_B)
    assert af.dry_run_ok is True
    # A ChallengeAttempt was assembled and passed to dry_run.
    assert len(client.dry_run_calls) == 1
    assert isinstance(client.dry_run_calls[0], ChallengeAttempt)
    # NEVER broadcasts.
    client.submit_one.assert_not_called()
    client.submit_batch.assert_not_called()
    client.broadcast.assert_not_called()


def test_scan_one_vs_one_is_ambiguous_not_auto_challenged():
    # sp1170 — a 1-vs-1 disagreement has NO strict majority (a tie): the honest side is
    # undeterminable off-chain, so the finding is SURFACED but ambiguous + NOT auto-challenged
    # (no ChallengeAttempt assembled, no dry-run verdict). Prevents a wrongful auto-slash.
    cache = VerifiedBatchCache()
    _ingest_cogroup_pair(cache)
    client = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=client)
    findings = engine.scan_consensus_mismatches()
    assert len(findings) >= 1
    for af in findings:
        assert af.finding.ambiguous is True
        assert af.dry_run_ok is None          # never auto-dry-run an ambiguous slash
    assert len(client.dry_run_calls) == 0     # nothing assembled -> nothing dry-run
    client.submit_one.assert_not_called()
    client.broadcast.assert_not_called()


def test_scan_no_dry_run_client_still_surfaces_finding():
    cache = VerifiedBatchCache()
    _ingest_cogroup_pair(cache)
    engine = SettlementAuditEngine(cache)  # no dry-run client
    findings = engine.scan_consensus_mismatches()
    assert len(findings) == 1
    assert findings[0].dry_run_ok is None


def test_scan_agreeing_outputs_no_finding():
    cache = VerifiedBatchCache()
    _ingest_cogroup_pair(cache, out_a="ee" * 32, out_b="ee" * 32)
    engine = SettlementAuditEngine(cache, dry_run_client=_FakeDryRunClient())
    assert engine.scan_consensus_mismatches() == []


# ── (d) only root-VERIFIED batches feed the scan ──────────────────────


def test_rejected_ingest_never_scanned():
    cache = VerifiedBatchCache()
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    ra = _receipt(a, job="J", shard=2, out=("11" * 32), provider=PROV_A, group=G1)
    rb = _receipt(b, job="J", shard=2, out=("22" * 32), provider=PROV_B, group=G1)
    ba_id = b"\xa1" * 32
    bb_id = b"\xb1" * 32
    # batch_a verifies; batch_b is published with a TAMPERED root -> rejected.
    assert cache.ingest(_observed(ba_id, [ra], provider=PROV_A, group=G1),
                        _published(ba_id, [ra]))
    bad_observed = ObservedBatch(
        batch_id=bb_id, provider_address=PROV_B, requester_address=REQ,
        merkle_root=b"\xde" * 32, consensus_group_id=G1)  # wrong root
    assert cache.ingest(bad_observed, _published(bb_id, [rb])) is False

    engine = SettlementAuditEngine(cache, dry_run_client=_FakeDryRunClient())
    # Only one verified batch (batch_a) — no co-group pair -> no finding.
    assert engine.scan_consensus_mismatches() == []


# ── (e) FAIL-CLOSED ───────────────────────────────────────────────────


def test_scan_fail_closed_on_assemble_error(monkeypatch):
    cache = VerifiedBatchCache()
    _ingest_cogroup_pair(cache)
    client = _FakeDryRunClient()
    engine = SettlementAuditEngine(cache, dry_run_client=client)

    # Force assembly of the ChallengeAttempt to raise; the scan must not crash.
    import prsm.settlement.settlement_audit_engine as eng_mod

    def _boom(*a, **k):
        raise RuntimeError("assemble boom")

    monkeypatch.setattr(
        eng_mod.ReceiptLeafFields, "from_python_receipt",
        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    # Scan does not crash. The detector still found the pair, but assembly fails ->
    # the finding is surfaced WITHOUT a dry-run verdict (or skipped), never raises.
    findings = engine.scan_consensus_mismatches()
    # No crash; if any finding is surfaced its dry_run_ok is unknown.
    for af in findings:
        assert af.dry_run_ok is None
    client.submit_one.assert_not_called()


def test_detector_fail_closed_on_bad_receipt():
    """PER-RECEIPT fail-closed: a malformed receipt (whose canonical leaf can't be built)
    is skipped, so it can NOT suppress detection of the genuine co-group mismatch, and the
    detector never crashes. (Before the per-receipt guard this raised and lost ALL findings.)"""
    a = generate_node_identity(display_name="cfa")
    b = generate_node_identity(display_name="cfb")
    # genuine co-group disagreement: same job+shard, different providers, different output.
    ra = _receipt(a, job="job-1", shard=0, out="11" * 32, provider=PROV_A, group=G1)
    rb = _receipt(b, job="job-1", shard=0, out="22" * 32, provider=PROV_B, group=G1)
    batch_a = _detector_batch(b"\xaa" * 32, PROV_A, G1, [ra])
    batch_b = _detector_batch(b"\xbb" * 32, PROV_B, G1, [rb])
    # a third co-group batch carrying a MALFORMED receipt (batched_receipt_to_leaf will raise
    # on a bare object that has no .receipt) — must be skipped, not crash the scan.
    batch_bad = _detector_batch(b"\xcc" * 32, "0x" + "c" * 40, G1, [object()])
    findings = detect_consensus_mismatches([batch_a, batch_bad, batch_b])
    assert len(findings) == 1  # bad receipt skipped; the genuine A-B mismatch survives
    f = findings[0]
    assert {f.minority_batch_id, f.majority_batch_id} == {b"\xaa" * 32, b"\xbb" * 32}


# ── (f) LOOP e2e ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_run_once_surfaces_and_counts_consensus_mismatch():
    from prsm.settlement.settlement_audit_loop import SettlementAuditLoop

    cache = VerifiedBatchCache()
    # sp1170 — a 2-of-3 strict majority (A+B agree on the honest output, C is the lone faulty
    # minority) so the surfaced finding has a genuine, auto-challengeable minority (C).
    a = generate_node_identity("a")
    b = generate_node_identity("b")
    c = generate_node_identity("c")
    ra = _receipt(a, job="J", shard=2, out=("11" * 32), provider=PROV_A, group=G1)
    rb = _receipt(b, job="J", shard=2, out=("11" * 32), provider=PROV_B, group=G1)  # agrees with A
    rc = _receipt(c, job="J", shard=2, out=("22" * 32), provider=PROV_C, group=G1)  # minority
    ba_id = b"\xa1" * 32
    bb_id = b"\xb1" * 32
    bc_id = b"\xc1" * 32
    obs_a = _observed(ba_id, [ra], provider=PROV_A, group=G1)
    obs_b = _observed(bb_id, [rb], provider=PROV_B, group=G1)
    obs_c = _observed(bc_id, [rc], provider=PROV_C, group=G1)
    pb_a = _published(ba_id, [ra])
    pb_b = _published(bb_id, [rb])
    pb_c = _published(bc_id, [rc])

    client = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=client)

    from prsm.settlement.receipt_set_blob import serialize_receipt_set

    blob_by_cid = {
        "cid-a": serialize_receipt_set(pb_a),
        "cid-b": serialize_receipt_set(pb_b),
        "cid-c": serialize_receipt_set(pb_c),
    }
    cid_by_batch = {ba_id: "cid-a", bb_id: "cid-b", bc_id: "cid-c"}

    class _Enum:
        async def enumerate_committed_batches(self, f, t, *, provider=None):
            return [obs_a, obs_b, obs_c]

    class _Sel:
        def select(self, observed):
            return list(observed)

    class _Ad:
        def get(self, batch_id):
            cid = cid_by_batch.get(bytes(batch_id))
            return type("A", (), {"cid": cid})() if cid else None

    class _Fetch:
        async def fetch(self, cid):
            return blob_by_cid[cid]

    loop = SettlementAuditLoop(
        enumerator=_Enum(), selector=_Sel(), ad_index=_Ad(),
        fetcher=_Fetch(), cache=cache, engine=engine)

    result = await loop.run_once(0, 100)
    assert result.verified == 3
    assert result.consensus_mismatch_finding_count == 1
    assert len(result.consensus_mismatch_findings) == 1
    af = result.consensus_mismatch_findings[0]
    assert af.dry_run_ok is True
    assert af.finding.minority_provider == PROV_C  # the faulty minority, not honest A/B
    # NEVER broadcasts.
    client.submit_one.assert_not_called()
    client.broadcast.assert_not_called()
