"""Sprint 1148 — EXPIRED detection (the 5th and FINAL on-chain-actionable settlement
fraud/hygiene class, completing DOUBLE_SPEND=0 / INVALID_SIGNATURE=1 / NO_ESCROW=2 /
EXPIRED=3 / CONSENSUS_MISMATCH=5).

EXPIRED catches a stale receipt: a committed leaf whose ``executed_at_unix`` is older than
the batch's per-batch ``lookbackWindowSecondsAtCommit``. On-chain ``_handleExpired``
(BatchSettlementRegistry.sol 840-850) is a PURE time check:

    age = block.timestamp - leaf.executedAtUnix
    lookback = lookbackWindowSecondsAtCommit
    if (age <= lookback) revert ReceiptNotExpired(...)   # age <= lookback is NOT expired
    return true                                          # age >  lookback IS expired

So a leaf is EXPIRED-challengeable IFF ``age > lookback`` (STRICT). EXPIRED does NOT slash
(protocol-hygiene, not malice) but it DOES invalidate the leaf value
(``b.invalidatedValueFTNS += leaf.valueFtns``) so finalize won't pay it. ANY observer can
challenge (no msg.sender restriction); auxData is UNUSED (empty b"").

CRITICAL SAFETY (anti-griefing): a FALSE EXPIRED finding wrongly DENIES an HONEST provider's
payment. So the detector matches the on-chain check EXACTLY (strict age > lookback), is
FAIL-CLOSED on missing/<=0 lookback + per-receipt errors, and the assembler FAIL-FASTS
unless provably expired (refuses to build a tx that would revert ReceiptNotExpired).

These tests inject everything (NO network, NO chain, NO disk): batches built with a
controllable ``executed_at_unix`` + per-batch lookback + an injected ``now``, and a mocked
dry-run client. They assert:
  (a) detector flags ONLY age>lookback (strict): age==lookback NOT flagged (boundary),
      age<lookback NOT flagged, age>lookback flagged;
  (b) detector FAIL-CLOSED: lookback<=0/missing -> no finding; a bare-object() bad receipt is
      skipped, the good leaf survives;
  (c) assembler builds a valid challenge: to_call_args reason==3, aux_data==b"", merkle_proof
      verifies the leaf against the independently-recomputed batch root;
  (d) assembler FAIL-FAST: a not-expired leaf (age<=lookback) -> ValueError; out-of-range
      index / empty receipts -> ValueError;
  (e) submitter accepts EXPIRED(3) (dry_run+submit forward to_call_args) and STILL rejects
      CONSENSUS_MISMATCH(5) + an undefined reason;
  (f) scan_expired_receipts: only-expired leaves produce dry-run-gated challenges; an injected
      Mock client asserts submit/broadcast NEVER called;
  (g) end-to-end: a batch with a stale leaf + a fresh leaf -> exactly the stale leaf is
      challenged; an honest (all-fresh) batch -> no challenge;
  (h) ObservedBatch lookback default + enumerator idx-10 capture.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.compute.shard_receipt import (
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.merkle import (
    batched_receipt_to_leaf,
    build_merkle_root,
    hash_leaf,
    verify_merkle_proof,
)
from prsm.settlement.published_batch_store import PublishedBatch
from prsm.settlement.settlement_audit_engine import (
    ObservedBatch,
    SettlementAuditEngine,
    VerifiedBatchCache,
)
from prsm.settlement.expired_receipt_detector import (
    ExpiredReceiptFinding,
    detect_expired_receipts,
)
from prsm.settlement.expired_assembler import (
    REASON_EXPIRED,
    ExpiredChallenge,
    assemble_expired_challenge,
)
from prsm.settlement.invalid_signature_submitter import (
    SUPPORTED_CHALLENGE_REASONS,
    ChallengeSubmitter,
)

PROV_A = "0x" + "a" * 40
REQ = "0x" + "1" * 40

# A wall-clock "now" used across tests; receipts choose executed_at_unix relative to it.
NOW = 2_000_000


# ── builders ──────────────────────────────────────────────────────────


def _receipt(identity, *, job="job-1", shard=0, out=None,
             requester=REQ, provider=PROV_A, value=10 ** 18, executed_at_unix=1000):
    out = out or ("ab" * 32)
    payload = build_receipt_signing_payload(
        job_id=job, shard_index=shard, output_hash=out,
        executed_at_unix=executed_at_unix)
    sig = identity.sign(payload)
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=executed_at_unix, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address=requester, provider_address=provider,
        value_ftns=value, local_escrow_id="e1")


def _root_for(receipts):
    return build_merkle_root([hash_leaf(batched_receipt_to_leaf(br)) for br in receipts])


def _published(batch_id, receipts):
    return PublishedBatch(
        batch_id=batch_id, merkle_root=_root_for(receipts),
        commit_timestamp=1000, receipts=list(receipts))


def _observed(batch_id, receipts, *, provider=PROV_A, requester=REQ,
              lookback_window_seconds=0):
    return ObservedBatch(
        batch_id=batch_id, provider_address=provider,
        requester_address=requester, merkle_root=_root_for(receipts),
        lookback_window_seconds=lookback_window_seconds)


# The detector consumes a small batch shape carrying (batch_id, receipts, lookback). We mirror
# what scan_expired_receipts feeds it. The detector accepts any object with these attrs.
class _DetBatch:
    def __init__(self, batch_id, receipts, lookback_window_seconds):
        self.batch_id = batch_id
        self.receipts = list(receipts)
        self.lookback_window_seconds = lookback_window_seconds


@pytest.fixture(scope="module")
def identity():
    return generate_node_identity()


# ── (a) detector: STRICT age > lookback ────────────────────────────────


def test_detector_flags_only_strictly_expired(identity):
    lookback = 1000
    # age == lookback exactly -> NOT expired (boundary, on-chain age <= lookback reverts).
    boundary = _receipt(identity, out="aa" * 32, executed_at_unix=NOW - lookback)
    # age < lookback -> fresh, NOT expired.
    fresh = _receipt(identity, out="bb" * 32, executed_at_unix=NOW - (lookback - 1))
    # age > lookback -> EXPIRED.
    stale = _receipt(identity, out="cc" * 32, executed_at_unix=NOW - (lookback + 1))

    batch = _DetBatch(b"\x01" * 32, [boundary, fresh, stale], lookback)
    findings = detect_expired_receipts([batch], now=NOW)

    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, ExpiredReceiptFinding)
    assert f.target_index == 2  # only the stale leaf
    assert f.batch_id == b"\x01" * 32
    assert f.lookback == lookback
    assert f.age == lookback + 1
    assert f.age > f.lookback  # strict
    assert f.leaf.executed_at_unix == NOW - (lookback + 1)


def test_detector_safety_margin_only_flags_beyond_margin(identity):
    lookback = 1000
    margin = 100
    # age = lookback + 50: expired on chain, but within the conservative margin -> NOT flagged.
    just_over = _receipt(identity, out="dd" * 32, executed_at_unix=NOW - (lookback + 50))
    # age = lookback + margin + 1: beyond margin -> flagged.
    well_over = _receipt(identity, out="ee" * 32,
                         executed_at_unix=NOW - (lookback + margin + 1))
    batch = _DetBatch(b"\x02" * 32, [just_over, well_over], lookback)
    findings = detect_expired_receipts([batch], now=NOW, safety_margin_seconds=margin)
    assert len(findings) == 1
    assert findings[0].target_index == 1


# ── (b) detector FAIL-CLOSED ────────────────────────────────────────────


def test_detector_fail_closed_on_nonpositive_lookback(identity):
    stale = _receipt(identity, executed_at_unix=NOW - 999_999)
    # lookback <= 0 / missing -> never flag (a finding leads to a challenge that would grief).
    assert detect_expired_receipts([_DetBatch(b"\x03" * 32, [stale], 0)], now=NOW) == []
    assert detect_expired_receipts([_DetBatch(b"\x03" * 32, [stale], -5)], now=NOW) == []
    assert detect_expired_receipts([_DetBatch(b"\x03" * 32, [stale], None)], now=NOW) == []


def test_detector_fail_closed_skips_bad_receipt_keeps_good(identity):
    lookback = 1000
    stale = _receipt(identity, out="cc" * 32, executed_at_unix=NOW - (lookback + 1))
    # A bare object() cannot be turned into a leaf -> skipped (logged), not fatal.
    batch = _DetBatch(b"\x04" * 32, [object(), stale], lookback)
    findings = detect_expired_receipts([batch], now=NOW)
    assert len(findings) == 1
    assert findings[0].target_index == 1  # the good (stale) leaf survives


# ── (c) assembler builds a valid challenge ──────────────────────────────


def test_assembler_builds_valid_expired_challenge(identity):
    lookback = 1000
    fresh = _receipt(identity, out="bb" * 32, executed_at_unix=NOW - (lookback - 1))
    stale = _receipt(identity, out="cc" * 32, executed_at_unix=NOW - (lookback + 1))
    receipts = [fresh, stale]

    ch = assemble_expired_challenge(
        batch_id=b"\x05" * 32, batch_receipts=receipts, target_index=1,
        now=NOW, lookback_window_seconds=lookback)
    assert isinstance(ch, ExpiredChallenge)

    call = ch.to_call_args()
    # (batch_id, leaf_tuple, merkle_proof, reason, aux_data)
    assert call[0] == b"\x05" * 32
    assert int(call[3]) == REASON_EXPIRED == 3
    assert call[4] == b""

    # The merkle_proof verifies the challenged leaf against the independently-recomputed root.
    root = _root_for(receipts)
    leaf_hash = hash_leaf(batched_receipt_to_leaf(stale))
    assert verify_merkle_proof(ch.merkle_proof, root, leaf_hash)


# ── (d) assembler FAIL-FAST ─────────────────────────────────────────────


def test_assembler_fail_fast_when_not_expired(identity):
    lookback = 1000
    # age == lookback -> NOT expired -> assembling would revert ReceiptNotExpired on chain.
    boundary = _receipt(identity, executed_at_unix=NOW - lookback)
    with pytest.raises(ValueError):
        assemble_expired_challenge(
            batch_id=b"\x06" * 32, batch_receipts=[boundary], target_index=0,
            now=NOW, lookback_window_seconds=lookback)
    # age < lookback -> fresh.
    fresh = _receipt(identity, executed_at_unix=NOW - (lookback - 1))
    with pytest.raises(ValueError):
        assemble_expired_challenge(
            batch_id=b"\x06" * 32, batch_receipts=[fresh], target_index=0,
            now=NOW, lookback_window_seconds=lookback)


def test_assembler_fail_fast_on_bad_index_and_empty(identity):
    stale = _receipt(identity, executed_at_unix=NOW - 999_999)
    with pytest.raises(ValueError):
        assemble_expired_challenge(
            batch_id=b"\x07" * 32, batch_receipts=[stale], target_index=5,
            now=NOW, lookback_window_seconds=1000)
    with pytest.raises(ValueError):
        assemble_expired_challenge(
            batch_id=b"\x07" * 32, batch_receipts=[], target_index=0,
            now=NOW, lookback_window_seconds=1000)


def test_assembler_fail_fast_on_nonpositive_lookback(identity):
    stale = _receipt(identity, executed_at_unix=NOW - 999_999)
    with pytest.raises(ValueError):
        assemble_expired_challenge(
            batch_id=b"\x07" * 32, batch_receipts=[stale], target_index=0,
            now=NOW, lookback_window_seconds=0)


# ── (e) submitter accepts EXPIRED, rejects others ───────────────────────


def test_submitter_accepts_expired_rejects_consensus_and_undefined(identity):
    assert REASON_EXPIRED in SUPPORTED_CHALLENGE_REASONS
    # CONSENSUS_MISMATCH=5 has its OWN submitter — generic set must still reject it.
    assert 5 not in SUPPORTED_CHALLENGE_REASONS

    lookback = 1000
    stale = _receipt(identity, executed_at_unix=NOW - (lookback + 1))
    ch = assemble_expired_challenge(
        batch_id=b"\x08" * 32, batch_receipts=[stale], target_index=0,
        now=NOW, lookback_window_seconds=lookback)

    registry = MagicMock()
    registry.functions.challengeReceipt.return_value.call.return_value = None
    account = MagicMock()
    account.address = "0x" + "c" * 40
    submitter = ChallengeSubmitter(registry=registry, account=account, web3=MagicMock())

    # EXPIRED(3) -> dry_run forwards to_call_args to the contract.
    res = submitter.dry_run(ch)
    assert res.would_succeed is True
    registry.functions.challengeReceipt.assert_called_with(*ch.to_call_args())

    # An undefined reason is rejected WITHOUT calling the contract / raising.
    bad = ExpiredChallenge(
        batch_id=ch.batch_id, leaf=ch.leaf, merkle_proof=ch.merkle_proof,
        reason_code=99, aux_data=b"")
    registry.functions.challengeReceipt.reset_mock()
    rej = submitter.dry_run(bad)
    assert rej.would_succeed is False
    registry.functions.challengeReceipt.assert_not_called()

    # CONSENSUS_MISMATCH(5) likewise rejected by the generic submitter.
    cm = ExpiredChallenge(
        batch_id=ch.batch_id, leaf=ch.leaf, merkle_proof=ch.merkle_proof,
        reason_code=5, aux_data=b"")
    sub_res = submitter.submit(cm)
    assert sub_res.success is False
    registry.functions.challengeReceipt.assert_not_called()


# ── (f) + (g) scan_expired_receipts over the verified cache ─────────────


def _ingest(cache, observed, receipts):
    assert cache.ingest(observed, _published(observed.batch_id, receipts)) is True


def test_scan_expired_only_stale_dry_run_gated_never_broadcasts(identity):
    lookback = 1000
    fresh = _receipt(identity, out="bb" * 32, executed_at_unix=NOW - (lookback - 1))
    stale = _receipt(identity, out="cc" * 32, executed_at_unix=NOW - (lookback + 1))
    receipts = [fresh, stale]

    bid = b"\x09" * 32
    cache = VerifiedBatchCache()
    _ingest(cache, _observed(bid, receipts, lookback_window_seconds=lookback), receipts)

    dry_client = MagicMock()
    dry_client.dry_run.return_value = MagicMock(would_succeed=True, revert_reason=None)
    engine = SettlementAuditEngine(cache, dry_run_client=dry_client)

    # margin=0 isolates the scan plumbing from the conservative default cushion (the default
    # 60s margin is exercised by the detector-level margin test).
    findings = engine.scan_expired_receipts(now=NOW, safety_margin_seconds=0)
    assert len(findings) == 1
    assert findings[0].finding.target_index == 1  # only the stale leaf
    assert findings[0].dry_run_ok is True

    # NEVER broadcasts — only the read-only dry_run was used.
    dry_client.dry_run.assert_called()
    assert not hasattr(dry_client, "submit") or not dry_client.submit.called
    assert not dry_client.method_calls or all(
        c[0] == "dry_run" for c in dry_client.method_calls)


def test_scan_expired_end_to_end_stale_vs_honest(identity):
    lookback = 1000
    # Stale batch: one fresh + one stale leaf -> exactly the stale leaf challenged.
    fresh = _receipt(identity, out="bb" * 32, executed_at_unix=NOW - (lookback - 1))
    stale = _receipt(identity, out="cc" * 32, executed_at_unix=NOW - (lookback + 1))
    stale_bid = b"\x0a" * 32
    # Honest batch: all leaves fresh -> NO challenge.
    h1 = _receipt(identity, out="dd" * 32, executed_at_unix=NOW - 1)
    h2 = _receipt(identity, out="ee" * 32, executed_at_unix=NOW - 2)
    honest_bid = b"\x0b" * 32

    cache = VerifiedBatchCache()
    _ingest(cache, _observed(stale_bid, [fresh, stale], lookback_window_seconds=lookback),
            [fresh, stale])
    _ingest(cache, _observed(honest_bid, [h1, h2], lookback_window_seconds=lookback),
            [h1, h2])

    dry_client = MagicMock()
    dry_client.dry_run.return_value = MagicMock(would_succeed=True, revert_reason=None)
    engine = SettlementAuditEngine(cache, dry_run_client=dry_client)

    findings = engine.scan_expired_receipts(now=NOW, safety_margin_seconds=0)
    assert len(findings) == 1
    assert findings[0].finding.batch_id == stale_bid
    assert findings[0].finding.target_index == 1


def test_scan_expired_fail_closed_when_lookback_unknown(identity):
    # A batch whose retained lookback is 0 (default / unknown) is NEVER scanned to a finding.
    stale = _receipt(identity, executed_at_unix=NOW - 999_999)
    bid = b"\x0c" * 32
    cache = VerifiedBatchCache()
    # lookback default 0 -> fail-closed in the detector.
    _ingest(cache, _observed(bid, [stale], lookback_window_seconds=0), [stale])
    engine = SettlementAuditEngine(cache, dry_run_client=MagicMock())
    assert engine.scan_expired_receipts(now=NOW) == []


# ── (h) ObservedBatch default + cache retention ─────────────────────────


def test_observed_batch_lookback_default_and_cache_retention(identity):
    # ADDITIVE default: an ObservedBatch built WITHOUT lookback_window_seconds still works.
    ob = ObservedBatch(
        batch_id=b"\x0d" * 32, provider_address=PROV_A, requester_address=REQ,
        merkle_root=b"\x00" * 32)
    assert ob.lookback_window_seconds == 0

    # The cache retains it per batch (mirrors consensus_group_id retention).
    lookback = 1234
    stale = _receipt(identity, executed_at_unix=NOW - 999_999)
    bid = b"\x0e" * 32
    cache = VerifiedBatchCache()
    _ingest(cache, _observed(bid, [stale], lookback_window_seconds=lookback), [stale])
    assert cache.lookback_window_seconds_for(bid) == lookback
    assert cache.lookback_window_seconds_for(b"\xff" * 32) is None


def test_enumerator_captures_lookback_idx_10():
    """The enumerator reads struct idx 10 (lookbackWindowSecondsAtCommit) into the
    ObservedBatch — no new RPC call (the struct is already read for idx 1/2/9)."""
    from prsm.economy.web3.batch_settlement_contract_client import (
        Web3SettlementContractClient,
    )

    bid = b"\x10" * 32
    # A live batch struct dump: idx 0..10. idx10 == lookbackWindowSecondsAtCommit.
    struct = [
        PROV_A,           # 0 provider
        REQ,              # 1 requester
        b"\x22" * 32,     # 2 merkleRoot
        1,                # 3 receiptCount
        10 ** 18,         # 4 totalValueFTNS
        0,                # 5 invalidatedValueFTNS
        1000,             # 6 commitTimestamp
        0,                # 7 status
        500,              # 8 tier_slash_rate_bps
        b"\x00" * 32,     # 9 consensus_group_id
        4321,             # 10 lookbackWindowSecondsAtCommit
    ]

    client = Web3SettlementContractClient.__new__(Web3SettlementContractClient)
    contract = MagicMock()
    # One BatchCommitted log for our batch.
    log = {"args": {"batchId": bid, "provider": PROV_A}}
    flt = MagicMock()
    flt.get_all_entries.return_value = [log]
    contract.events.BatchCommitted.return_value.create_filter.return_value = flt
    contract.functions.batches.return_value.call.return_value = struct
    client.contract = contract

    observed = client._enumerate_committed_batches_sync(0, 0, None)
    assert len(observed) == 1
    assert observed[0].batch_id == bid
    assert observed[0].lookback_window_seconds == 4321
    assert observed[0].consensus_group_id == b"\x00" * 32
