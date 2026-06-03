"""Sprint 957 (sp928 follow-on) — wire the requester's CONSENSUS_MISMATCH
challenge routing into the persistent evidence log + stake-posture resolution.

sp928 built the ComputeResultSampler but left `challenge_sink=None` and never
passed the original provider's bonded status — so on a live run, EVERY caught
liar looked unbonded and the evidence was logged once and lost. sp957:

  1. `_build_default_sampler` wires `challenge_sink` to an injected
     ConsensusMismatchLog.record (the persistent, operator-reviewable store).
  2. `_resolve_stake_posture(provider_id)` maps a node_id → its CLAIMED
     operator_address (via the peer's hardware_profile) → verified-delegation
     gate (sprint 788) → on-chain stake. A spoofed/unverified operator claim
     degrades to unbonded (stake 0) — it can never ride another operator's bond.
  3. `_run_verification` resolves the original provider's posture and passes
     `original_bonded`/`original_stake_wei`/`original_operator_address` through
     to the sampler, and `_reexec_for_verification` attaches the re-run
     provider's posture to its ReExecResult.

No code path here moves stake — the evidence is captured for a future
authority-gated on-chain bridge.
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

import prsm.node.compute_requester as _requester_mod
import prsm.node.compute_result_sampler as _sampler_mod
import prsm.node.consensus_mismatch_log as _log_mod
from prsm.node.compute_requester import ComputeRequester, JobType, SubmittedJob
from prsm.node.compute_result_sampler import ReExecResult, VerificationVerdict


def _make_requester():
    identity = MagicMock()
    identity.node_id = "requester_node"
    transport = MagicMock()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    ledger = MagicMock()
    discovery = MagicMock()
    discovery.record_job_failure = MagicMock()
    req = ComputeRequester(
        identity=identity, transport=transport, gossip=gossip,
        ledger=ledger, discovery=discovery,
    )
    req.escrow = None
    req.ledger_sync = None
    return req


class _FakeLog:
    def __init__(self):
        self.records = []

    async def record(self, evidence):
        self.records.append(evidence)


# ── challenge_sink wiring ─────────────────────────────────────────────────


def test_default_sampler_wires_mismatch_log_as_challenge_sink():
    req = _make_requester()
    log = _FakeLog()
    req.mismatch_log = log
    sampler = req._build_default_sampler()
    # The sink must be the log's record method (so caught liars are persisted).
    assert sampler._challenge_sink == log.record


def test_default_sampler_without_log_leaves_sink_none():
    req = _make_requester()
    req.mismatch_log = None
    sampler = req._build_default_sampler()
    assert sampler._challenge_sink is None


# ── stake-posture resolution ──────────────────────────────────────────────


def _peer_with_hw(node_id, hw):
    p = MagicMock()
    p.node_id = node_id
    p.hardware_profile = hw
    return p


def test_resolve_posture_verified_delegation_returns_stake(monkeypatch):
    req = _make_requester()
    req.discovery.known_peers = {
        "prov-A": _peer_with_hw("prov-A", {
            "operator_address": "0xOPERATOR",
            "operator_delegation": {"some": "blob"},
        }),
    }
    monkeypatch.setattr(
        "prsm.node.compute_requester.verify_operator_delegation_blob",
        lambda **kw: True,
    )
    req.stake_reader = MagicMock()
    req.stake_reader.stake_amount_for = MagicMock(return_value=5_000_000_000_000_000_000)
    bonded, stake_wei, op = req._resolve_stake_posture("prov-A")
    assert bonded is True
    assert stake_wei == 5_000_000_000_000_000_000
    assert op == "0xOPERATOR"


def test_resolve_posture_spoofed_delegation_degrades_unbonded(monkeypatch):
    """A peer claiming someone else's operator_address with an invalid/missing
    delegation must be treated as unbonded (stake 0) — it cannot ride the bond."""
    req = _make_requester()
    req.discovery.known_peers = {
        "prov-A": _peer_with_hw("prov-A", {
            "operator_address": "0xVICTIM",
            "operator_delegation": None,
        }),
    }
    monkeypatch.setattr(
        "prsm.node.compute_requester.verify_operator_delegation_blob",
        lambda **kw: False,
    )
    req.stake_reader = MagicMock()
    req.stake_reader.stake_amount_for = MagicMock(return_value=9_000_000_000_000_000_000)
    bonded, stake_wei, op = req._resolve_stake_posture("prov-A")
    assert bonded is False
    assert stake_wei == 0
    assert op is None
    # The stake reader must NOT have been consulted on a rejected claim.
    req.stake_reader.stake_amount_for.assert_not_called()


def test_resolve_posture_unknown_provider_unbonded():
    req = _make_requester()
    req.discovery.known_peers = {}
    bonded, stake_wei, op = req._resolve_stake_posture("ghost")
    assert (bonded, stake_wei, op) == (False, 0, None)


def test_resolve_posture_no_stake_reader_unbonded(monkeypatch):
    req = _make_requester()
    req.discovery.known_peers = {
        "prov-A": _peer_with_hw("prov-A", {
            "operator_address": "0xOPERATOR",
            "operator_delegation": {"some": "blob"},
        }),
    }
    monkeypatch.setattr(
        "prsm.node.compute_requester.verify_operator_delegation_blob",
        lambda **kw: True,
    )
    req.stake_reader = None
    bonded, stake_wei, op = req._resolve_stake_posture("prov-A")
    # No reader → can't size the bond → unbonded (safe degrade), op still known.
    assert bonded is False
    assert stake_wei == 0


# ── _run_verification threads posture into the sampler ─────────────────────


@pytest.mark.asyncio
async def test_run_verification_passes_original_posture(monkeypatch):
    req = _make_requester()
    req.sampler = MagicMock()
    req.sampler.verify = AsyncMock(return_value=MagicMock(
        verdict=VerificationVerdict.VERIFIED))
    req._resolve_stake_posture = MagicMock(
        return_value=(True, 4_000_000_000_000_000_000, "0xOP"))
    job = SubmittedJob(
        job_id="job_x", job_type=JobType.INFERENCE,
        payload={"prompt": "hi"}, ftns_budget=0.0,
    )
    await req._run_verification(job, "prov-A", "hash-X")
    _, kwargs = req.sampler.verify.call_args
    assert kwargs["original_bonded"] is True
    assert kwargs["original_stake_wei"] == 4_000_000_000_000_000_000
    assert kwargs["original_operator_address"] == "0xOP"


@pytest.mark.asyncio
async def test_reexec_attaches_posture_to_result(monkeypatch):
    """The re-run provider's ReExecResult must carry its own stake posture so a
    lying re-exec provider's bond is recorded in the evidence."""
    req = _make_requester()
    req._get_capable_peers = MagicMock(return_value=["prov-R"])
    rerun_job = SubmittedJob(
        job_id="rerun_1", job_type=JobType.INFERENCE,
        payload={"prompt": "hi"}, ftns_budget=0.0,
    )
    rerun_job.provider_id = "prov-R"
    req.submit_job = AsyncMock(return_value=rerun_job)
    req.get_result = AsyncMock(return_value={"out": "ok"})
    req.submitted_jobs["rerun_1"] = rerun_job
    req._resolve_stake_posture = MagicMock(
        return_value=(True, 2_000_000_000_000_000_000, "0xRUN"))
    res = await req._reexec_for_verification(JobType.INFERENCE, {"prompt": "hi"}, set())
    assert isinstance(res, ReExecResult)
    assert res.provider_id == "prov-R"
    assert res.bonded is True
    assert res.stake_wei == 2_000_000_000_000_000_000
    assert res.operator_address == "0xRUN"


# ── SOUNDNESS PIN: a single node must NEVER execute an on-chain slash ───────


def test_no_onchain_slash_in_compute_verification_path():
    """The load-bearing soundness invariant: an autonomous slash from one node's
    re-execution is unsound (StakeBond.slash is slasher-only; the open
    challengeReceipt rail re-verifies a Merkle proof the off-chain single-
    provider pay path never produces). The requester / sampler / evidence-log
    modules must therefore contain NO code path that EXECUTES a slash — they
    only CAPTURE evidence. We scan for executable call shapes (a `.slash(` /
    `.challengeReceipt(` invocation, or a slash-execution method definition);
    prose that NAMES the on-chain rail to document why we don't touch it is
    expected and allowed. A future authority-gated bridge must live in a
    separate, explicitly-authorized module (and this pin updated deliberately)."""
    forbidden_calls = (".slash(", ".challengereceipt(", "execute_slash(", "def slash")
    for mod in (_requester_mod, _sampler_mod, _log_mod):
        src = inspect.getsource(mod).lower()
        for needle in forbidden_calls:
            assert needle not in src, (
                f"{mod.__name__} contains '{needle}' — an on-chain slash CALL "
                f"must NOT live in the single-node verification/evidence modules"
            )
