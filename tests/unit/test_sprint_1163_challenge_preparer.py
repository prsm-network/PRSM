"""Sprint 1163 — operator challenge-PREPARATION bridge.

THE GAP this closes: the settlement fraud-defense pipeline DEAD-ENDS at the findings
surface (sp1149 store + sp1150 read-only CLI). An operator sees "here are N dispute
candidates" but has NO command to turn a persisted ``AuditFindingRecord`` into a
PREPARED, dry-run challenge tx ready to broadcast. The assemblers (challenge_assembler,
double_spend_assembler, no_escrow_assembler, expired_assembler) + the ChallengeSubmitter
are pure + correct, but nothing bridges ``finding -> fetch batch receipts -> assemble ->
(optional) dry-run -> emit call args``.

``prepare_challenge_from_finding`` is that bridge. It is PURE (a finding + an injected
``ReceiptLookup``), fail-closed, and dispatches the persisted finding to the RIGHT
assembler. It NEVER broadcasts/signs — assembling + (optionally, in the CLI) a read-only
dry-run is all it does; broadcasting stays the separate user-gated step.

These tests use REAL Ed25519 identities + REAL receipts + REAL assemblers + REAL merkle
(no mocks of the money/crypto path) so the bridge is proven against exactly what the
on-chain ``challengeReceipt`` will be handed.
"""
from __future__ import annotations

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
from prsm.settlement.double_spend_assembler import REASON_DOUBLE_SPEND
from prsm.settlement.expired_assembler import REASON_EXPIRED
from prsm.settlement.merkle import batched_receipt_to_leaf, hash_leaf
from prsm.settlement.no_escrow_assembler import REASON_NO_ESCROW
from prsm.settlement.settlement_audit_report import AuditFindingRecord

from prsm.settlement.challenge_preparer import (
    MissingBatchReceipts,
    PreparedChallenge,
    SkippedFinding,
    UnsupportedChallengeReason,
    dry_run_prepared,
    prepare_challenge_from_finding,
    prepare_findings,
    published_batch_receipt_lookup,
)


# ── fixtures: real receipts (mirrors the sp1118 assembler test builder) ──────────


def _receipt(identity, *, job="job-1", shard=0, valid=True, out=None, executed=1000):
    out = out or ("ab" * 32)
    if valid:
        sig = identity.sign(build_receipt_signing_payload(
            job_id=job, shard_index=shard, output_hash=out, executed_at_unix=executed))
    else:
        # sign a DIFFERENT payload -> the committed signature won't verify over the leaf
        sig = identity.sign(build_receipt_signing_payload(
            job_id=job, shard_index=shard, output_hash=("cd" * 32),
            executed_at_unix=executed))
    rec = ShardExecutionReceipt(
        job_id=job, shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=executed, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address="0x" + "1" * 40,
        provider_address="0x" + "2" * 40, value_ftns=10 ** 18, local_escrow_id="e1")


def _batch(specs):
    return [_receipt(idn, shard=i, valid=v) for i, (idn, v) in enumerate(specs)]


def _lookup_from(mapping):
    """A ReceiptLookup over a plain {batch_id_hex: [BatchedReceipt]} mapping."""
    norm = {k.lower().removeprefix("0x"): v for k, v in mapping.items()}

    def _lookup(batch_id_hex):
        return norm.get(str(batch_id_hex).lower().removeprefix("0x"))

    return _lookup


def _invalid_sig_record(batch_id_hex, target_index):
    return AuditFindingRecord(
        reason="invalid_signature", batch_id_hex=batch_id_hex,
        target_index=target_index, on_chain_reason=1,
        detail={"receipt_index": target_index})


# ── INVALID_SIGNATURE ────────────────────────────────────────────────────────────


def test_prepares_invalid_signature_challenge():
    a, b, c = (generate_node_identity(n) for n in ("a", "b", "c"))
    batch = _batch([(a, True), (b, False), (c, True)])  # index 1 is fraudulent
    bid = "11" * 32
    rec = _invalid_sig_record(bid, 1)

    prepared = prepare_challenge_from_finding(
        record=rec, receipt_lookup=_lookup_from({bid: batch}))

    assert isinstance(prepared, PreparedChallenge)
    assert prepared.reason == "invalid_signature"
    assert prepared.on_chain_reason == REASON_INVALID_SIGNATURE == 1
    assert prepared.target_index == 1
    assert prepared.challenge.reason_code == 1
    # the prepared challenge is BYTE-IDENTICAL to a direct assembler call (no mangling)
    direct = assemble_invalid_signature_challenge(
        batch_id=bytes.fromhex(bid), batch_receipts=batch, target_index=1)
    assert prepared.challenge.to_call_args() == direct.to_call_args()


def test_invalid_signature_fail_fast_propagates_when_target_sig_verifies():
    # the assembler refuses a target whose signature VERIFIES; the bridge surfaces that.
    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = _batch([(a, True), (b, True)])  # index 1 is HONEST -> nothing to challenge
    bid = "22" * 32
    with pytest.raises(ValueError):
        prepare_challenge_from_finding(
            record=_invalid_sig_record(bid, 1),
            receipt_lookup=_lookup_from({bid: batch}))


# ── NO_ESCROW (requester self-dispute path; not in the audit store, but supported) ──


def test_prepares_no_escrow_challenge():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = _batch([(a, True), (b, True)])
    bid = "33" * 32
    rec = AuditFindingRecord(
        reason="no_escrow", batch_id_hex=bid, target_index=0, on_chain_reason=2,
        detail={})
    prepared = prepare_challenge_from_finding(
        record=rec, receipt_lookup=_lookup_from({bid: batch}))
    assert prepared.on_chain_reason == REASON_NO_ESCROW == 2
    assert prepared.challenge.reason_code == 2


# ── EXPIRED ───────────────────────────────────────────────────────────────────────


def test_prepares_expired_challenge_using_lookback_from_detail():
    a = generate_node_identity("a")
    batch = [_receipt(a, shard=0, valid=True, executed=1000)]
    bid = "44" * 32
    rec = AuditFindingRecord(
        reason="expired", batch_id_hex=bid, target_index=0, on_chain_reason=3,
        detail={"target_index": 0, "age": 1000, "lookback": 500})
    # now=2000 -> age = 2000-1000 = 1000 > lookback 500 -> expired (assembler accepts)
    prepared = prepare_challenge_from_finding(
        record=rec, receipt_lookup=_lookup_from({bid: batch}), now=2000)
    assert prepared.on_chain_reason == REASON_EXPIRED == 3


def test_expired_fail_fast_when_not_actually_expired():
    a = generate_node_identity("a")
    batch = [_receipt(a, shard=0, valid=True, executed=1000)]
    bid = "45" * 32
    rec = AuditFindingRecord(
        reason="expired", batch_id_hex=bid, target_index=0, on_chain_reason=3,
        detail={"lookback": 5000})
    # now=2000 -> age=1000 <= lookback 5000 -> NOT expired -> assembler refuses
    with pytest.raises(ValueError):
        prepare_challenge_from_finding(
            record=rec, receipt_lookup=_lookup_from({bid: batch}), now=2000)


def test_expired_fails_closed_when_detail_lacks_lookback():
    a = generate_node_identity("a")
    batch = [_receipt(a, shard=0, valid=True, executed=1000)]
    bid = "46" * 32
    rec = AuditFindingRecord(
        reason="expired", batch_id_hex=bid, target_index=0, on_chain_reason=3,
        detail={})  # no lookback -> cannot prove expiry -> fail-closed
    with pytest.raises(ValueError):
        prepare_challenge_from_finding(
            record=rec, receipt_lookup=_lookup_from({bid: batch}), now=2000)


# ── DOUBLE_SPEND ──────────────────────────────────────────────────────────────────


def test_prepares_double_spend_challenge_across_two_batches():
    a, x, y = (generate_node_identity(n) for n in ("a", "x", "y"))
    dup = _receipt(a, job="dup", shard=0, valid=True)  # the double-committed receipt
    other1 = _receipt(x, job="o1", shard=1, valid=True)
    other2 = _receipt(y, job="o2", shard=1, valid=True)
    target_batch = [dup, other1]       # dup at index 0
    conflicting_batch = [other2, dup]  # SAME dup at index 1
    t_bid, c_bid = "aa" * 32, "bb" * 32
    # the surface's _double_spend_record sets batch_id_hex/target_index to occ[0], and
    # detail.occurrences to the full list of {batch_id, leaf_index, provider_address}.
    rec = AuditFindingRecord(
        reason="double_spend", batch_id_hex=t_bid, target_index=0, on_chain_reason=0,
        detail={
            "leaf_hash": hash_leaf(batched_receipt_to_leaf(dup)).hex(),
            "occurrences": [
                {"batch_id": t_bid, "leaf_index": 0, "provider_address": "0x" + "2" * 40},
                {"batch_id": c_bid, "leaf_index": 1, "provider_address": "0x" + "2" * 40},
            ],
        })
    prepared = prepare_challenge_from_finding(
        record=rec,
        receipt_lookup=_lookup_from({t_bid: target_batch, c_bid: conflicting_batch}))
    assert prepared.on_chain_reason == REASON_DOUBLE_SPEND == 0
    assert prepared.challenge.reason_code == 0
    # auxData encodes the conflicting batch id + proof (non-empty)
    assert prepared.challenge.aux_data


def test_double_spend_intra_batch_only_is_unsupported():
    a = generate_node_identity("a")
    dup = _receipt(a, job="dup", valid=True)
    bid = "cc" * 32
    rec = AuditFindingRecord(
        reason="double_spend", batch_id_hex=bid, target_index=0, on_chain_reason=0,
        detail={"occurrences": [
            {"batch_id": bid, "leaf_index": 0, "provider_address": "0x" + "2" * 40},
            {"batch_id": bid, "leaf_index": 3, "provider_address": "0x" + "2" * 40},
        ]})
    with pytest.raises(UnsupportedChallengeReason):
        prepare_challenge_from_finding(
            record=rec, receipt_lookup=_lookup_from({bid: [dup]}))


def test_double_spend_fewer_than_two_occurrences_is_an_error():
    bid = "cd" * 32
    rec = AuditFindingRecord(
        reason="double_spend", batch_id_hex=bid, target_index=0, on_chain_reason=0,
        detail={"occurrences": [
            {"batch_id": bid, "leaf_index": 0, "provider_address": "0x" + "2" * 40}]})
    with pytest.raises(ValueError):
        prepare_challenge_from_finding(
            record=rec, receipt_lookup=_lookup_from({bid: []}))


# ── unsupported reasons are fail-closed (never mis-assembled) ───────────────────────


def test_consensus_mismatch_is_unsupported_by_this_bridge():
    rec = AuditFindingRecord(
        reason="consensus_mismatch", batch_id_hex="ee" * 32, target_index=0,
        on_chain_reason=5, detail={})
    with pytest.raises(UnsupportedChallengeReason):
        prepare_challenge_from_finding(
            record=rec, receipt_lookup=_lookup_from({}))


def test_inference_receipt_is_unsupported_by_this_bridge():
    rec = AuditFindingRecord(
        reason="inference_receipt", batch_id_hex="ef" * 32, target_index=None,
        on_chain_reason=None, detail={})
    with pytest.raises(UnsupportedChallengeReason):
        prepare_challenge_from_finding(
            record=rec, receipt_lookup=_lookup_from({}))


# ── missing-receipts: fail-closed, never assemble against nothing ───────────────────


def test_missing_target_receipts_fail_closed():
    rec = _invalid_sig_record("12" * 32, 0)
    with pytest.raises(MissingBatchReceipts):
        prepare_challenge_from_finding(
            record=rec, receipt_lookup=_lookup_from({}))  # lookup returns None


def test_double_spend_missing_conflicting_receipts_fail_closed():
    a, x = generate_node_identity("a"), generate_node_identity("x")
    dup = _receipt(a, job="dup", valid=True)
    target_batch = [dup, _receipt(x, shard=1, valid=True)]
    t_bid, c_bid = "a1" * 32, "b1" * 32
    rec = AuditFindingRecord(
        reason="double_spend", batch_id_hex=t_bid, target_index=0, on_chain_reason=0,
        detail={"occurrences": [
            {"batch_id": t_bid, "leaf_index": 0, "provider_address": "0x" + "2" * 40},
            {"batch_id": c_bid, "leaf_index": 1, "provider_address": "0x" + "2" * 40},
        ]})
    with pytest.raises(MissingBatchReceipts):
        prepare_challenge_from_finding(
            record=rec, receipt_lookup=_lookup_from({t_bid: target_batch}))  # no c_bid


# ── to_dict / call-args rendering ───────────────────────────────────────────────────


def test_prepared_to_dict_renders_hex_call_args():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = _batch([(a, True), (b, False)])
    bid = "13" * 32
    prepared = prepare_challenge_from_finding(
        record=_invalid_sig_record(bid, 1), receipt_lookup=_lookup_from({bid: batch}))
    d = prepared.to_dict()
    assert d["reason"] == "invalid_signature"
    assert d["on_chain_reason"] == 1
    assert d["batch_id_hex"] == bid
    ca = d["call_args"]
    assert ca["batch_id"] == "0x" + bid
    assert ca["reason"] == 1
    assert isinstance(ca["merkle_proof"], list)
    assert ca["aux_data"].startswith("0x")
    # the leaf renders as an ordered list of 9 fields (matches the challengeReceipt tuple)
    assert isinstance(ca["leaf"], list) and len(ca["leaf"]) == 9


# ── published_batch_receipt_lookup over a real PublishedBatchStore ──────────────────


def test_published_batch_receipt_lookup(tmp_path):
    from prsm.settlement.merkle import build_merkle_root
    from prsm.settlement.published_batch_store import PublishedBatchStore

    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = _batch([(a, True), (b, False)])
    bid = bytes.fromhex("14" * 32)
    root = build_merkle_root([hash_leaf(batched_receipt_to_leaf(br)) for br in batch])
    store = PublishedBatchStore(tmp_path / "pb.json")
    store.put(bid, batch, root, commit_timestamp=1234)

    lookup = published_batch_receipt_lookup(store)
    got = lookup("14" * 32)
    assert got is not None and len(got) == 2
    assert lookup("0x" + "14" * 32) is not None  # 0x-prefixed accepted
    assert lookup("99" * 32) is None  # not retained -> None

    # end-to-end: the lookup feeds the bridge for a real INVALID_SIGNATURE prepare
    prepared = prepare_challenge_from_finding(
        record=_invalid_sig_record("14" * 32, 1), receipt_lookup=lookup)
    assert prepared.challenge.reason_code == 1


# ── dry-run is READ-ONLY: never calls submit ────────────────────────────────────────


class _DryRunOnlyTripwire:
    """A fake challenge client: dry_run returns a verdict; submit RAISES if ever called
    (the never-broadcast tripwire)."""

    def __init__(self):
        self.dry_run_calls = []

    def dry_run(self, challenge):
        self.dry_run_calls.append(challenge)

        class _V:
            would_succeed = True
            revert_reason = None
        return _V()

    def submit(self, challenge):  # pragma: no cover - must never be reached
        raise AssertionError("dry_run_prepared MUST NOT broadcast (submit was called)")


def test_prepare_findings_partitions_preparable_from_skipped():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    good_batch = _batch([(a, True), (b, False)])  # idx 1 fraudulent -> preparable
    good_bid = "21" * 32
    records = [
        _invalid_sig_record(good_bid, 1),                      # preparable
        _invalid_sig_record("31" * 32, 0),                     # missing receipts -> skipped
        AuditFindingRecord(reason="consensus_mismatch",        # unsupported -> skipped
                           batch_id_hex="41" * 32, target_index=0,
                           on_chain_reason=5, detail={}),
        AuditFindingRecord(reason="inference_receipt",         # unsupported -> skipped
                           batch_id_hex="51" * 32, target_index=None,
                           on_chain_reason=None, detail={}),
    ]
    prepared, skipped = prepare_findings(
        records, _lookup_from({good_bid: good_batch}))
    assert len(prepared) == 1
    assert prepared[0].batch_id_hex == good_bid
    assert len(skipped) == 3
    assert all(isinstance(s, SkippedFinding) for s in skipped)
    msgs = " ".join(s.message for s in skipped)
    assert "MissingBatchReceipts" in msgs
    assert "UnsupportedChallengeReason" in msgs
    # to_dict shape is JSON-able
    assert skipped[0].to_dict()["reason"]


def test_prepare_findings_never_raises_on_empty():
    prepared, skipped = prepare_findings([], _lookup_from({}))
    assert prepared == [] and skipped == []


@pytest.mark.parametrize("bad_occurrences", [
    [{"leaf_index": 0}, {"leaf_index": 1}],                  # occurrences missing batch_id
    [{"batch_id": "aa" * 32}, {"batch_id": "bb" * 32}],      # missing leaf_index
    ["not-a-dict", {"batch_id": "bb" * 32, "leaf_index": 1}],  # non-dict occurrence
    [{"batch_id": "aa" * 32, "leaf_index": "x"},             # non-int leaf_index
     {"batch_id": "bb" * 32, "leaf_index": 1}],
])
def test_prepare_findings_partitions_malformed_double_spend_without_raising(bad_occurrences):
    # REGRESSION (sp1163 verify, major): a malformed double_spend occurrence used to raise an
    # UNCAUGHT KeyError/TypeError that escaped prepare_findings (its "never raises" contract)
    # and aborted the whole CLI run, hiding every preparable finding. It MUST now be skipped,
    # and a sibling preparable invalid_signature finding MUST still prepare.
    a, b = generate_node_identity("a"), generate_node_identity("b")
    good_batch = _batch([(a, True), (b, False)])
    good_bid = "61" * 32
    records = [
        AuditFindingRecord(reason="double_spend", batch_id_hex="62" * 32, target_index=0,
                           on_chain_reason=0, detail={"occurrences": bad_occurrences}),
        _invalid_sig_record(good_bid, 1),  # the preparable sibling that must NOT be hidden
    ]
    # the call itself must NOT raise
    prepared, skipped = prepare_findings(records, _lookup_from({good_bid: good_batch}))
    assert len(prepared) == 1 and prepared[0].reason == "invalid_signature"
    assert len(skipped) == 1 and skipped[0].reason == "double_spend"


def test_double_spend_reported_target_index_is_the_challenged_leaf():
    # MINOR (sp1163 verify): the reported target_index must be the leaf ACTUALLY challenged
    # (the target occurrence's leaf_index), not a top-level record.target_index that can
    # diverge for a hand-built record.
    a, x, y = (generate_node_identity(n) for n in ("a", "x", "y"))
    dup = _receipt(a, job="dup", shard=0, valid=True)
    target_batch = [dup, _receipt(x, shard=1, valid=True)]       # dup at index 0
    conflicting_batch = [_receipt(y, shard=1, valid=True), dup]  # SAME dup at index 1
    t_bid, c_bid = "71" * 32, "72" * 32
    rec = AuditFindingRecord(
        reason="double_spend", batch_id_hex=t_bid,
        target_index=99,  # deliberately divergent top-level value (hand-built record)
        on_chain_reason=0,
        detail={"occurrences": [
            {"batch_id": t_bid, "leaf_index": 0, "provider_address": "0x" + "2" * 40},
            {"batch_id": c_bid, "leaf_index": 1, "provider_address": "0x" + "2" * 40},
        ]})
    prepared = prepare_challenge_from_finding(
        record=rec,
        receipt_lookup=_lookup_from({t_bid: target_batch, c_bid: conflicting_batch}))
    # reported target_index reflects the genuinely-challenged leaf (0), NOT the stray 99
    assert prepared.target_index == 0
    # and the assembled leaf is indeed the dup leaf at index 0
    assert prepared.challenge.leaf == batched_receipt_to_leaf(dup)


def test_dry_run_prepared_is_read_only_and_never_submits():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = _batch([(a, True), (b, False)])
    bid = "15" * 32
    prepared = prepare_challenge_from_finding(
        record=_invalid_sig_record(bid, 1), receipt_lookup=_lookup_from({bid: batch}))
    client = _DryRunOnlyTripwire()
    verdict = dry_run_prepared(prepared, client)
    assert verdict.would_succeed is True
    assert len(client.dry_run_calls) == 1
    # the exact assembled challenge was handed to the read-only dry_run
    assert client.dry_run_calls[0] is prepared.challenge


# ── the operator CLI (offline assemble path, end-to-end) ────────────────────────────


def _load_cli():
    import importlib.util
    from pathlib import Path as _P

    repo = _P(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "settlement_challenge_prepare", repo / "scripts" / "settlement_challenge_prepare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_offline_assembles_and_partitions(tmp_path, capsys):
    import json as _json

    from prsm.settlement.merkle import build_merkle_root
    from prsm.settlement.published_batch_store import PublishedBatchStore
    from prsm.settlement.settlement_audit_report import (
        SettlementAuditFindingsStore,
        summarize_records,
    )

    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = _batch([(a, True), (b, False)])  # idx 1 fraudulent
    bid = bytes.fromhex("16" * 32)
    root = build_merkle_root([hash_leaf(batched_receipt_to_leaf(br)) for br in batch])

    # the receipt source (PublishedBatchStore file)
    pb_path = tmp_path / "pb.json"
    PublishedBatchStore(pb_path).put(bid, batch, root, commit_timestamp=111)

    # the findings store: one preparable invalid_signature + one unsupported consensus_mismatch
    fin_path = tmp_path / "findings.json"
    SettlementAuditFindingsStore(fin_path).record_summary(summarize_records([
        _invalid_sig_record("16" * 32, 1),
        AuditFindingRecord(reason="consensus_mismatch", batch_id_hex="17" * 32,
                           target_index=0, on_chain_reason=5, detail={}),
    ]))

    cli = _load_cli()
    rc = cli.main([
        "--findings-file", str(fin_path),
        "--published-batches-file", str(pb_path),
        "--json",
    ])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    assert len(out["prepared"]) == 1
    assert out["prepared"][0]["reason"] == "invalid_signature"
    assert out["prepared"][0]["on_chain_reason"] == 1
    assert out["prepared"][0]["call_args"]["reason"] == 1
    assert out["prepared"][0]["dry_run"] is None  # no --dry-run -> no chain touched
    # the unsupported consensus_mismatch is SKIPPED, not mis-encoded
    assert len(out["skipped"]) == 1
    assert out["skipped"][0]["reason"] == "consensus_mismatch"
    assert "broadcast" in out["note"].lower()


def test_cli_dry_run_requires_chain_args(tmp_path):
    cli = _load_cli()
    args = cli.build_parser().parse_args([
        "--findings-file", str(tmp_path / "f.json"),
        "--published-batches-file", str(tmp_path / "pb.json"),
        "--dry-run",  # but no --rpc-url/--registry-address/--from-address
    ])
    with pytest.raises(SystemExit) as ei:
        cli._maybe_dry_run([], args)
    msg = str(ei.value)
    assert "--rpc-url" in msg and "--registry-address" in msg and "--from-address" in msg
