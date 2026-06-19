"""Sprint 1166 — user-gated challenge BROADCAST (the capstone of the fraud-defense loop).

The prepare bridge (sp1163-1165) is deliberately KEYLESS + never-broadcast: it assembles +
read-only-dry-runs, then dead-ends at "here are the exact challengeReceipt call args". sp1166
adds the final step — actually FILING a prepared challenge — as an EXPLICITLY user-gated
action (the operator runs it with their own key; the assistant never broadcasts).

  - prsm/settlement/challenge_broadcaster.py: gated_broadcast_prepared(prepared, submitter,
    broadcast_enabled, ...) — ALWAYS dry-runs first; broadcasts ONLY when broadcast_enabled
    AND (the dry-run passed OR force); NEVER calls submit otherwise. Submitter is injected
    (the CLI builds the keyed one); the core never holds a key.
  - ConsensusChallengeSubmitter.submit(challenge): uniform broadcast via to_call_args (so the
    reason-5 path broadcasts symmetrically with the generalized ChallengeSubmitter.submit),
    sharing broadcast mechanics with submit_one.

These tests prove the never-auto-broadcast invariant with a tripwire submitter, the
dry-run-first gate, the dry-run-fail refusal, and the reason-5 uniform submit.
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
from prsm.settlement.challenge_preparer import prepare_challenge_from_finding
from prsm.settlement.settlement_audit_report import AuditFindingRecord

from prsm.settlement.challenge_broadcaster import (
    BroadcastOutcome,
    gated_broadcast_prepared,
)


def _receipt(identity, *, valid=True, out=None, shard=0):
    out = out or ("ab" * 32)
    if valid:
        sig = identity.sign(build_receipt_signing_payload(
            job_id="job-1", shard_index=shard, output_hash=out, executed_at_unix=1000))
    else:
        sig = identity.sign(build_receipt_signing_payload(
            job_id="job-1", shard_index=shard, output_hash=("cd" * 32),
            executed_at_unix=1000))
    rec = ShardExecutionReceipt(
        job_id="job-1", shard_index=shard, provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64, output_hash=out,
        executed_at_unix=1000, signature=sig)
    return BatchedReceipt(
        receipt=rec, requester_address="0x" + "1" * 40, provider_address="0x" + "2" * 40,
        value_ftns=10 ** 18, local_escrow_id="e1")


def _prepared_invalid_sig():
    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = [_receipt(a, valid=True, shard=0), _receipt(b, valid=False, shard=1)]
    bid = "11" * 32
    rec = AuditFindingRecord(reason="invalid_signature", batch_id_hex=bid, target_index=1,
                             on_chain_reason=1, detail={"receipt_index": 1})
    return prepare_challenge_from_finding(
        record=rec, receipt_lookup=lambda x: batch if x == bid else None)


class _Submitter:
    """A fake challenge submitter: dry_run returns a configurable verdict; submit records the
    call + returns a success result. Both are tripwires for assertions."""

    def __init__(self, *, dry_ok=True, dry_reason=None):
        self._dry_ok = dry_ok
        self._dry_reason = dry_reason
        self.dry_run_calls = []
        self.submit_calls = []

    def dry_run(self, challenge):
        self.dry_run_calls.append(challenge)
        return type("V", (), {"would_succeed": self._dry_ok,
                              "revert_reason": self._dry_reason})()

    def submit(self, challenge):
        self.submit_calls.append(challenge)
        return type("R", (), {"success": True, "tx_hash_hex": "0xdeadbeef",
                              "error_type": None, "error_message": None})()


class _NeverSubmit(_Submitter):
    def submit(self, challenge):  # pragma: no cover - must never be reached
        raise AssertionError("gated_broadcast_prepared MUST NOT broadcast here")


# ── the never-auto-broadcast invariant ──────────────────────────────────────────────


def test_disabled_broadcast_dry_runs_but_never_submits():
    prepared = _prepared_invalid_sig()
    sub = _NeverSubmit(dry_ok=True)
    outcome = gated_broadcast_prepared(prepared, submitter=sub, broadcast_enabled=False)
    assert isinstance(outcome, BroadcastOutcome)
    assert outcome.broadcast is False
    assert outcome.dry_run_ok is True            # the read-only dry-run DID run
    assert len(sub.dry_run_calls) == 1
    assert outcome.tx_hash_hex is None
    assert "not enabled" in (outcome.refused_reason or "").lower()


def test_enabled_and_dry_run_ok_broadcasts():
    prepared = _prepared_invalid_sig()
    sub = _Submitter(dry_ok=True)
    outcome = gated_broadcast_prepared(prepared, submitter=sub, broadcast_enabled=True)
    assert outcome.broadcast is True
    assert outcome.tx_hash_hex == "0xdeadbeef"
    assert len(sub.submit_calls) == 1
    assert sub.submit_calls[0] is prepared.challenge


def test_enabled_but_dry_run_fails_refuses_to_broadcast():
    prepared = _prepared_invalid_sig()
    sub = _NeverSubmit(dry_ok=False, dry_reason="ChallengeNotProven")
    outcome = gated_broadcast_prepared(prepared, submitter=sub, broadcast_enabled=True)
    assert outcome.broadcast is False
    assert outcome.dry_run_ok is False
    assert "ChallengeNotProven" in (outcome.dry_run_reason or "")
    assert "dry-run" in (outcome.refused_reason or "").lower()


def test_force_broadcasts_despite_dry_run_failure():
    prepared = _prepared_invalid_sig()
    sub = _Submitter(dry_ok=False, dry_reason="ChallengeNotProven")
    outcome = gated_broadcast_prepared(
        prepared, submitter=sub, broadcast_enabled=True, force=True)
    assert outcome.broadcast is True
    assert len(sub.submit_calls) == 1


def test_core_enforces_never_raise_on_a_nonconforming_submitter():
    # sp1166 verify (nit, fixed): the core ENFORCES never-raise itself, so even a submitter
    # whose submit()/dry_run() raises yields a clean fail-closed outcome (no exception escapes
    # a bond-slashing path).
    prepared = _prepared_invalid_sig()

    class _SubmitRaises:
        def dry_run(self, ch):
            return type("V", (), {"would_succeed": True, "revert_reason": None})()

        def submit(self, ch):
            raise RuntimeError("exploded inside submit")

    outcome = gated_broadcast_prepared(
        prepared, submitter=_SubmitRaises(), broadcast_enabled=True)
    assert outcome.broadcast is False
    assert outcome.error_type == "RuntimeError"
    assert "exploded" in (outcome.error_message or "")

    class _DryRunRaises:
        def dry_run(self, ch):
            raise RuntimeError("dry-run exploded")

        def submit(self, ch):  # pragma: no cover - a failed dry-run must refuse to broadcast
            raise AssertionError("must not broadcast when dry-run errored (fail-closed)")

    # a dry-run that ERRORS fails closed: not ok -> refuses to broadcast (no force)
    outcome2 = gated_broadcast_prepared(
        prepared, submitter=_DryRunRaises(), broadcast_enabled=True)
    assert outcome2.broadcast is False
    assert outcome2.dry_run_ok is False
    assert "dry-run raised" in (outcome2.dry_run_reason or "")


def test_outcome_to_dict_is_json_able():
    prepared = _prepared_invalid_sig()
    sub = _Submitter(dry_ok=True)
    outcome = gated_broadcast_prepared(prepared, submitter=sub, broadcast_enabled=True)
    d = outcome.to_dict()
    assert d["broadcast"] is True and d["tx_hash_hex"] == "0xdeadbeef"


# ── ConsensusChallengeSubmitter.submit broadcasts uniformly via to_call_args ─────────


def test_consensus_submitter_submit_broadcasts_via_to_call_args():
    from prsm.economy.web3.stake_manager import ReasonCode
    from prsm.marketplace.consensus_submitter import ConsensusChallengeSubmitter

    rec = {}

    class _Built:
        def build_transaction(self, overrides):
            rec["overrides"] = overrides
            return {"tx": "built"}

    class _Fns:
        def challengeReceipt(self, *args):
            rec["call_args"] = args
            return _Built()

    class _Registry:
        functions = _Fns()
        address = "0x" + "d" * 40

    class _Acct:
        address = "0x" + "c" * 40
        key = b"\x01" * 32

    w3 = MagicMock()
    w3.eth.account.sign_transaction.return_value = type(
        "S", (), {"raw_transaction": b"\xab"})()
    w3.eth.send_raw_transaction.return_value = b"\x12\x34"
    w3.eth.wait_for_transaction_receipt.return_value = type("Rcpt", (), {"status": 1})()
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1
    w3.eth.chain_id = 84532

    submitter = ConsensusChallengeSubmitter(
        web3=w3, registry=_Registry(), account=_Acct())

    # a minimal consensus challenge with a to_call_args() the submit path broadcasts
    from prsm.settlement.consensus_mismatch_assembler import ConsensusMismatchChallenge
    ch = ConsensusMismatchChallenge(
        minority_batch_id=b"\xaa" * 32,
        minority_leaf_tuple=(b"\x00" * 32, 0, b"\x00" * 32, b"\x00" * 32, b"\x00" * 32,
                             1000, 10 ** 18, b"\x00" * 32, b"\x00" * 32),
        minority_proof=[], aux_data=b"\x01")
    result = submitter.submit(ch)
    assert result.success is True
    assert rec["call_args"][3] == int(ReasonCode.CONSENSUS_MISMATCH)  # broadcast the right reason
    assert rec["call_args"][0] == b"\xaa" * 32


# ── the operator broadcast CLI (gating, default dry-run-only) ────────────────────────


def _load_submit_cli():
    import importlib.util
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "settlement_challenge_submit", repo / "scripts" / "settlement_challenge_submit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_requires_challenger_key(monkeypatch):
    monkeypatch.delenv("PRSM_CHALLENGER_KEY", raising=False)
    cli = _load_submit_cli()
    rc = cli.main(["--batch-id", "11" * 32, "--reason", "invalid_signature",
                   "--target-index", "1", "--rpc-url", "http://x", "--registry-address", "0xR"])
    assert rc == 2  # missing key -> misconfiguration


def test_cli_broadcast_without_ack_fails_fast(monkeypatch):
    monkeypatch.setenv("PRSM_CHALLENGER_KEY", "0x" + "11" * 32)
    cli = _load_submit_cli()
    # --broadcast WITHOUT the ack flag must refuse BEFORE any chain connection (exit 2)
    rc = cli.main(["--batch-id", "11" * 32, "--reason", "invalid_signature",
                   "--target-index", "1", "--rpc-url", "http://x",
                   "--registry-address", "0xR", "--broadcast"])
    assert rc == 2


def test_cli_default_is_dry_run_only_and_never_broadcasts(monkeypatch, tmp_path, capsys):
    import json as _json

    from prsm.settlement.merkle import (
        batched_receipt_to_leaf,
        build_merkle_root,
        hash_leaf,
    )
    from prsm.settlement.published_batch_store import PublishedBatchStore
    from prsm.settlement.settlement_audit_report import (
        SettlementAuditFindingsStore,
        summarize_records,
    )

    a, b = generate_node_identity("a"), generate_node_identity("b")
    batch = [_receipt(a, valid=True, shard=0), _receipt(b, valid=False, shard=1)]
    bid = b"\x11" * 32
    root = build_merkle_root([hash_leaf(batched_receipt_to_leaf(br)) for br in batch])
    pb_path = tmp_path / "pb.json"
    PublishedBatchStore(pb_path).put(bid, batch, root, commit_timestamp=1)
    fin_path = tmp_path / "fin.json"
    SettlementAuditFindingsStore(fin_path).record_summary(summarize_records([
        AuditFindingRecord(reason="invalid_signature", batch_id_hex=bid.hex(),
                           target_index=1, on_chain_reason=1, detail={"receipt_index": 1})]))

    cli = _load_submit_cli()

    # inject a fake keyed submitter (NO chain): dry_run ok, submit is a tripwire.
    class _FakeSub:
        web3 = type("W3", (), {"eth": type("E", (), {"chain_id": 84532})()})()

        def dry_run(self, challenge):
            return type("V", (), {"would_succeed": True, "revert_reason": None})()

        def submit(self, challenge):  # pragma: no cover - must NOT be reached by default
            raise AssertionError("default mode MUST NOT broadcast")

    monkeypatch.setenv("PRSM_CHALLENGER_KEY", "0x" + "11" * 32)
    monkeypatch.setattr(cli, "_build_keyed_submitter",
                        lambda *a, **k: _FakeSub())

    rc = cli.main([
        "--findings-file", str(fin_path), "--published-batches-file", str(pb_path),
        "--verified-batches-file", str(tmp_path / "v.json"),
        "--batch-id", bid.hex(), "--reason", "invalid_signature", "--target-index", "1",
        "--rpc-url", "http://x", "--registry-address", "0xR", "--json",
    ])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    assert out["outcome"]["broadcast"] is False
    assert out["outcome"]["dry_run_ok"] is True
    assert "not enabled" in (out["outcome"]["refused_reason"] or "").lower()


def test_cli_selection_mismatch_is_misconfiguration(monkeypatch, tmp_path):
    from prsm.settlement.settlement_audit_report import (
        SettlementAuditFindingsStore,
        summarize_records,
    )
    fin_path = tmp_path / "fin.json"
    SettlementAuditFindingsStore(fin_path).record_summary(summarize_records([
        AuditFindingRecord(reason="invalid_signature", batch_id_hex="11" * 32,
                           target_index=1, on_chain_reason=1, detail={"receipt_index": 1})]))
    monkeypatch.setenv("PRSM_CHALLENGER_KEY", "0x" + "11" * 32)
    cli = _load_submit_cli()
    # wrong target-index -> 0 matches -> exit 2 (never reaches a chain build)
    rc = cli.main([
        "--findings-file", str(fin_path), "--published-batches-file", str(tmp_path / "pb.json"),
        "--batch-id", "11" * 32, "--reason", "invalid_signature", "--target-index", "9",
        "--rpc-url", "http://x", "--registry-address", "0xR"])
    assert rc == 2
