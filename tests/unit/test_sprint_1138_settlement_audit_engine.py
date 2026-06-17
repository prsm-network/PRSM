"""Sprint 1138 — settlement-receipt data plane BRICK D: the observer AUDIT ENGINE.

Brick D is the PURE/injectable offline audit engine that ties Bricks A/B/C into an
actionable dispute pipeline WITHOUT ever broadcasting/signing/slashing:

  - two incentive-aligned, pluggable SELECTORS (OwnStake, ConsensusGroup) that pick,
    from a stream of on-chain-anchored ``ObservedBatch`` metadata, the batches the
    operator actually cares to audit;
  - the ``VerifiedBatchCache`` — THE TRUST GATE: a fetched (untrusted) receipt set may
    only be cached (and thereby reach detection) if it RECOMPUTES to the trusted
    on-chain ``merkle_root`` via Brick B's ``verify_receipt_set_against_root``. A set
    that fails the cross-check is REJECTED and caches NOTHING;
  - ``SettlementAuditEngine`` — runs ``detect_double_spends`` + the
    ``assemble_invalid_signature_challenge`` fail-fast DETECTOR over the verified
    cache, optionally dry-run-gating each actionable finding. It NEVER broadcasts.

These tests inject everything (no network, no chain, no disk). They assert:
  (a) selector correctness (exact OwnStake/ConsensusGroup membership);
  (b) the TRUST BOUNDARY crux (tampered/forged set rejected, caches nothing);
  (c) double-spend surfacing + dry_run_ok attachment;
  (d) invalid-shard-sig surfacing at exactly the bad index, clean batches surface
      nothing (assembler fail-fast path);
  (e) NO broadcast/submit is EVER called across both scans;
  (f) fail-closed per-item (a raising batch/receipt is skipped, scan never aborts).
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
from prsm.settlement.invalid_signature_submitter import DryRunResult

from prsm.settlement.settlement_audit_engine import (
    ObservedBatch,
    OwnStakeSelector,
    ConsensusGroupSelector,
    VerifiedBatchCache,
    SettlementAuditEngine,
)

ZERO_GROUP = b"\x00" * 32


# ── fixtures / builders ──────────────────────────────────────────────


def _receipt(identity, *, job="job-1", shard=0, valid=True, out=None,
             requester="0x" + "1" * 40, provider="0x" + "2" * 40,
             group=ZERO_GROUP, value=10 ** 18):
    out = out or ("ab" * 32)
    payload = build_receipt_signing_payload(
        job_id=job, shard_index=shard, output_hash=out, executed_at_unix=1000)
    if valid:
        sig = identity.sign(payload)
    else:
        # sign a DIFFERENT payload → the committed signature won't verify over this leaf
        sig = identity.sign(build_receipt_signing_payload(
            job_id=job, shard_index=shard, output_hash=("cd" * 32),
            executed_at_unix=1000))
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address=requester, provider_address=provider,
        value_ftns=value, local_escrow_id="e1", consensus_group_id=group)


def _root_for(receipts):
    leaf_hashes = [hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
    return build_tree_and_proofs(leaf_hashes).root


def _published(batch_id, receipts):
    return PublishedBatch(
        batch_id=batch_id, merkle_root=_root_for(receipts),
        commit_timestamp=1000, receipts=list(receipts))


def _observed(batch_id, receipts, *, provider="0x" + "2" * 40,
              requester="0x" + "1" * 40, group=ZERO_GROUP):
    return ObservedBatch(
        batch_id=batch_id, provider_address=provider,
        requester_address=requester, merkle_root=_root_for(receipts),
        consensus_group_id=group)


class _FakeDryRunClient:
    """Injected dry-run-only client. Records every dry_run call; its submit/broadcast
    are mocks that MUST NEVER be called by the engine."""

    def __init__(self, would_succeed=True):
        self._would_succeed = would_succeed
        self.dry_run_calls = []
        self.submit = MagicMock(name="submit")
        self.broadcast = AsyncMock(name="broadcast")

    def dry_run(self, challenge):
        self.dry_run_calls.append(challenge)
        return DryRunResult(would_succeed=self._would_succeed)


# ── (a) SELECTOR CORRECTNESS ─────────────────────────────────────────


def test_own_stake_selector_keeps_exactly_requester_is_me():
    me = "0x" + "a" * 40
    a = generate_node_identity("a")
    mine = _observed(b"\x01" * 32, [_receipt(a)], requester=me)
    not_mine = _observed(b"\x02" * 32, [_receipt(a)], requester="0x" + "b" * 40)
    # case-insensitive eth-address compare
    mine_upper = _observed(b"\x03" * 32, [_receipt(a)], requester=me.upper())

    sel = OwnStakeSelector(my_address=me)
    kept = sel.select([mine, not_mine, mine_upper])
    assert {b.batch_id for b in kept} == {b"\x01" * 32, b"\x03" * 32}


def test_consensus_group_selector_keeps_exactly_my_group_ignores_zero():
    a = generate_node_identity("a")
    g1 = b"\x11" * 32
    g2 = b"\x22" * 32
    other = b"\x33" * 32
    in_g1 = _observed(b"\x01" * 32, [_receipt(a, group=g1)], group=g1)
    in_g2 = _observed(b"\x02" * 32, [_receipt(a, group=g2)], group=g2)
    in_other = _observed(b"\x03" * 32, [_receipt(a, group=other)], group=other)
    no_group = _observed(b"\x04" * 32, [_receipt(a)], group=ZERO_GROUP)

    sel = ConsensusGroupSelector(my_group_ids={g1, g2})
    kept = sel.select([in_g1, in_g2, in_other, no_group])
    assert {b.batch_id for b in kept} == {b"\x01" * 32, b"\x02" * 32}


def test_consensus_group_selector_empty_groupset_keeps_nothing():
    a = generate_node_identity("a")
    g1 = b"\x11" * 32
    in_g1 = _observed(b"\x01" * 32, [_receipt(a, group=g1)], group=g1)
    sel = ConsensusGroupSelector(my_group_ids=set())
    assert sel.select([in_g1]) == []


# ── (b) TRUST BOUNDARY (the crux) ────────────────────────────────────


def test_cache_ingests_a_set_that_matches_the_onchain_root():
    a = generate_node_identity("a")
    receipts = [_receipt(a, shard=0), _receipt(a, shard=1)]
    obs = _observed(b"\x01" * 32, receipts)
    pb = _published(b"\x01" * 32, receipts)

    cache = VerifiedBatchCache(max_batches=8)
    assert cache.ingest(obs, pb) is True

    verified = cache.verified_batches()
    assert len(verified) == 1
    cb = verified[0]
    assert cb.batch_id == b"\x01" * 32
    assert cb.provider_address == obs.provider_address
    assert cb.merkle_root == obs.merkle_root
    # leaf hashes recomputed from the fetched receipts, in order
    assert list(cb.leaf_hashes) == [
        hash_leaf(batched_receipt_to_leaf(br)) for br in receipts]
    # the order-preserved receipts are retained for the assembler
    assert cache.receipts_for(b"\x01" * 32) == receipts


def test_cache_rejects_a_root_mismatch_and_caches_nothing():
    """A fetched set whose recomputed root != the observed (trusted) on-chain root is
    REJECTED — ingest returns False, NOTHING is cached, nothing reaches detection."""
    a = generate_node_identity("a")
    receipts = [_receipt(a, shard=0), _receipt(a, shard=1)]
    # observed metadata carries a root that does NOT match the receipts
    bad_obs = ObservedBatch(
        batch_id=b"\x01" * 32, provider_address="0x" + "2" * 40,
        requester_address="0x" + "1" * 40,
        merkle_root=b"\xde\xad" * 16, consensus_group_id=ZERO_GROUP)
    pb = _published(b"\x01" * 32, receipts)

    cache = VerifiedBatchCache(max_batches=8)
    assert cache.ingest(bad_obs, pb) is False
    assert cache.verified_batches() == []
    assert cache.receipts_for(b"\x01" * 32) is None


def test_cache_rejects_a_tampered_receipt_set():
    """Tamper a single fetched receipt → recomputed root no longer matches the trusted
    on-chain root → ingest rejects it, caches nothing."""
    a = generate_node_identity("a")
    receipts = [_receipt(a, shard=0), _receipt(a, shard=1)]
    obs = _observed(b"\x01" * 32, receipts)  # root anchors the ORIGINAL set

    # the producer published a TAMPERED set (different output_hash on receipt 0)
    tampered = [_receipt(a, shard=0, out=("ff" * 32)), receipts[1]]
    pb = PublishedBatch(
        batch_id=b"\x01" * 32, merkle_root=obs.merkle_root,
        commit_timestamp=1000, receipts=tampered)

    cache = VerifiedBatchCache(max_batches=8)
    assert cache.ingest(obs, pb) is False
    assert cache.verified_batches() == []


def test_cache_is_bounded_evicts_oldest():
    a = generate_node_identity("a")
    cache = VerifiedBatchCache(max_batches=2)
    for i in range(3):
        bid = bytes([i + 1]) * 32
        receipts = [_receipt(a, shard=0, job=f"job-{i}")]
        assert cache.ingest(_observed(bid, receipts), _published(bid, receipts))
    ids = {cb.batch_id for cb in cache.verified_batches()}
    # the oldest (batch \x01) was evicted
    assert ids == {b"\x02" * 32, b"\x03" * 32}


# ── (c) DOUBLE-SPEND scan ────────────────────────────────────────────


def test_scan_double_spends_surfaces_a_shared_leaf_across_providers():
    a = generate_node_identity("a")
    # the SAME receipt committed in two batches by two DIFFERENT providers
    shared = _receipt(a, shard=0, job="shared-job")
    other1 = _receipt(a, shard=1, job="filler-1")
    other2 = _receipt(a, shard=1, job="filler-2")
    b1_receipts = [shared, other1]
    b2_receipts = [shared, other2]

    cache = VerifiedBatchCache(max_batches=8)
    assert cache.ingest(
        _observed(b"\x01" * 32, b1_receipts, provider="0x" + "a" * 40),
        _published(b"\x01" * 32, b1_receipts))
    assert cache.ingest(
        _observed(b"\x02" * 32, b2_receipts, provider="0x" + "b" * 40),
        _published(b"\x02" * 32, b2_receipts))

    fake = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=fake)
    findings = engine.scan_double_spends()

    assert len(findings) == 1
    f = findings[0]
    # surfaces the shared leaf as a double-spend
    assert f.finding.leaf_hash == hash_leaf(batched_receipt_to_leaf(shared))
    assert {b"\x01" * 32, b"\x02" * 32} == {
        occ.batch_id for occ in f.finding.occurrences}
    # dry-run gated, would-succeed propagated
    assert f.dry_run_ok is True
    assert len(fake.dry_run_calls) == 1
    # NEVER broadcast / submit
    fake.submit.assert_not_called()
    fake.broadcast.assert_not_called()


def test_scan_double_spends_clean_cache_surfaces_nothing():
    a = generate_node_identity("a")
    b1 = [_receipt(a, shard=0, job="j1")]
    b2 = [_receipt(a, shard=0, job="j2")]
    cache = VerifiedBatchCache(max_batches=8)
    cache.ingest(_observed(b"\x01" * 32, b1), _published(b"\x01" * 32, b1))
    cache.ingest(_observed(b"\x02" * 32, b2), _published(b"\x02" * 32, b2))
    engine = SettlementAuditEngine(cache)
    assert engine.scan_double_spends() == []


# ── (d) INVALID-SHARD-SIG scan ───────────────────────────────────────


def test_scan_invalid_signatures_surfaces_exactly_the_bad_index():
    a, b, c = (generate_node_identity(n) for n in ("a", "b", "c"))
    # index 1 has a non-verifying committed signature
    receipts = [_receipt(a, shard=0, valid=True),
                _receipt(b, shard=1, valid=False),
                _receipt(c, shard=2, valid=True)]
    cache = VerifiedBatchCache(max_batches=8)
    cache.ingest(_observed(b"\x01" * 32, receipts), _published(b"\x01" * 32, receipts))

    fake = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=fake)
    findings = engine.scan_invalid_signatures()

    assert len(findings) == 1
    f = findings[0]
    assert f.batch_id == b"\x01" * 32
    assert f.receipt_index == 1
    assert f.dry_run_ok is True
    assert len(fake.dry_run_calls) == 1
    fake.submit.assert_not_called()
    fake.broadcast.assert_not_called()


def test_scan_invalid_signatures_all_good_batch_surfaces_nothing():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    receipts = [_receipt(a, shard=0, valid=True), _receipt(b, shard=1, valid=True)]
    cache = VerifiedBatchCache(max_batches=8)
    cache.ingest(_observed(b"\x01" * 32, receipts), _published(b"\x01" * 32, receipts))
    engine = SettlementAuditEngine(cache)
    # the assembler fail-fast (signature VERIFIES) path → no actionable findings
    assert engine.scan_invalid_signatures() == []


# ── (e) NO-BROADCAST proof (both scans) ──────────────────────────────


def test_no_broadcast_or_submit_across_both_scans():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    # one batch with a double-spent leaf + a bad-sig receipt; another sharing the leaf
    shared = _receipt(a, shard=0, job="shared")
    bad = _receipt(b, shard=1, valid=False, job="bad")
    b1 = [shared, bad]
    b2 = [shared, _receipt(a, shard=1, job="other")]
    cache = VerifiedBatchCache(max_batches=8)
    cache.ingest(_observed(b"\x01" * 32, b1, provider="0x" + "a" * 40),
                 _published(b"\x01" * 32, b1))
    cache.ingest(_observed(b"\x02" * 32, b2, provider="0x" + "c" * 40),
                 _published(b"\x02" * 32, b2))

    fake = _FakeDryRunClient(would_succeed=True)
    engine = SettlementAuditEngine(cache, dry_run_client=fake)
    engine.scan_double_spends()
    engine.scan_invalid_signatures()

    # dry_run was used, but submit/broadcast NEVER
    assert len(fake.dry_run_calls) >= 1
    fake.submit.assert_not_called()
    fake.broadcast.assert_not_called()


# ── (f) FAIL-CLOSED per-item ─────────────────────────────────────────


def test_scan_invalid_signatures_is_fail_closed_per_item():
    """A batch whose assembler raises a NON-fail-fast error is skipped without aborting
    the scan; a subsequent good-finding batch is still surfaced."""
    a, b = generate_node_identity("a"), generate_node_identity("b")
    boom = [_receipt(a, shard=0, valid=False)]            # would normally surface
    good_bad = [_receipt(b, shard=0, valid=False)]        # actionable bad sig
    cache = VerifiedBatchCache(max_batches=8)
    cache.ingest(_observed(b"\x01" * 32, boom), _published(b"\x01" * 32, boom))
    cache.ingest(_observed(b"\x02" * 32, good_bad), _published(b"\x02" * 32, good_bad))

    # make the assembler explode for batch \x01 only
    import prsm.settlement.settlement_audit_engine as eng_mod
    real_assemble = eng_mod.assemble_invalid_signature_challenge

    def flaky(*, batch_id, batch_receipts, target_index):
        if batch_id == b"\x01" * 32:
            raise RuntimeError("synthetic assembler explosion")
        return real_assemble(
            batch_id=batch_id, batch_receipts=batch_receipts,
            target_index=target_index)

    eng_mod.assemble_invalid_signature_challenge = flaky
    try:
        engine = SettlementAuditEngine(cache)
        findings = engine.scan_invalid_signatures()
    finally:
        eng_mod.assemble_invalid_signature_challenge = real_assemble

    # the exploding batch was skipped; the scan did NOT raise; the good one surfaced
    assert {f.batch_id for f in findings} == {b"\x02" * 32}


def test_scan_double_spends_is_fail_closed_per_item():
    """If the detector raises, the scan returns [] without propagating the exception."""
    a = generate_node_identity("a")
    shared = _receipt(a, shard=0, job="shared")
    b1 = [shared]
    b2 = [shared]
    cache = VerifiedBatchCache(max_batches=8)
    cache.ingest(_observed(b"\x01" * 32, b1, provider="0x" + "a" * 40),
                 _published(b"\x01" * 32, b1))
    cache.ingest(_observed(b"\x02" * 32, b2, provider="0x" + "b" * 40),
                 _published(b"\x02" * 32, b2))

    import prsm.settlement.settlement_audit_engine as eng_mod
    real_detect = eng_mod.detect_double_spends
    eng_mod.detect_double_spends = MagicMock(side_effect=RuntimeError("boom"))
    try:
        engine = SettlementAuditEngine(cache)
        assert engine.scan_double_spends() == []
    finally:
        eng_mod.detect_double_spends = real_detect


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
