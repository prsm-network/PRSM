"""Sprint 928 — optimistic verification for the single-provider compute pay path
(compute/inference review #2: paid-on-signature-alone mis-pay).

The single-provider pay path (ComputeRequester._on_job_result) pays a provider
on a VALID SIGNATURE ALONE — there is no proof the work was actually done. A
provider can sign a well-formed-but-fabricated result and collect payment. The
marketplace k-of-n path already runs consensus + challenge + slash, but the
single-provider path has zero verification.

sp928 closes the gap with OPTIMISTIC verification (the standard decentralized-
compute model): pay as before, but with probability `sample_rate` re-execute
the job on an INDEPENDENT provider and compare outputs. On a mismatch, escalate
to a tiebreaker so we penalize the OUTLIER (the liar), not whoever happened to
be first — slashing the wrong provider would itself be a critical bug. Penalty
is reputation-always; a bonded provider additionally gets a CONSENSUS_MISMATCH
challenge routed into the existing on-chain slash rails.

The orchestrator is built with every external effect injected (re-executor,
reputation sink, challenge sink, RNG, comparator) so it is fully testable
without a multi-node bench; production wires thin adapters over the live
re-dispatch + reputation + challenge-queue machinery.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional, Set

import pytest

from prsm.node.compute_result_sampler import (
    ComputeResultSampler,
    ReExecResult,
    VerificationVerdict,
)
from prsm.node.compute_provider import JobType


# ── fakes ────────────────────────────────────────────────────────────────


class FakeReExecutor:
    """Returns queued independent re-executions, recording the exclude sets."""

    def __init__(self, queue: List[Optional[ReExecResult]]):
        self._queue = list(queue)
        self.calls: List[Set[str]] = []
        self.raise_on_call = False

    async def __call__(self, job_type, payload, exclude: Set[str]) -> Optional[ReExecResult]:
        self.calls.append(set(exclude))
        if self.raise_on_call:
            raise RuntimeError("re-exec dispatch blew up")
        if not self._queue:
            return None
        return self._queue.pop(0)


class FakeReputationSink:
    def __init__(self):
        self.recorded: List[str] = []

    def __call__(self, provider_id: str, outcome) -> None:
        self.recorded.append(provider_id)


class FakeChallengeSink:
    def __init__(self):
        self.evidence: List[dict] = []

    async def __call__(self, evidence: dict) -> None:
        self.evidence.append(evidence)


def _sampler(re_exec, *, rep=None, chal=None, rate=0.05, comparator=None,
             roll=None, max_tiebreakers=1):
    return ComputeResultSampler(
        re_executor=re_exec,
        reputation_sink=rep,
        challenge_sink=chal,
        sample_rate=rate,
        comparator=comparator,
        roll=roll,
        max_tiebreakers=max_tiebreakers,
    )


async def _verify(sampler, *, original="prov-A", out="hash-X", bonded=False):
    return await sampler.verify(
        job_id="job-1",
        job_type=JobType.INFERENCE,
        payload={"prompt": "hi"},
        original_provider_id=original,
        original_output_hash=out,
        original_bonded=bonded,
    )


# ── should_sample: the unpredictable, rate-respecting decision ───────────


def test_should_sample_respects_roll_below_rate():
    s = _sampler(FakeReExecutor([]), rate=0.1, roll=lambda: 0.05)
    assert s.should_sample() is True


def test_should_sample_respects_roll_at_or_above_rate():
    s = _sampler(FakeReExecutor([]), rate=0.1, roll=lambda: 0.10)
    assert s.should_sample() is False


def test_should_sample_rate_zero_never_samples():
    s = _sampler(FakeReExecutor([]), rate=0.0, roll=lambda: 0.0)
    assert s.should_sample() is False


def test_should_sample_rate_one_always_samples():
    s = _sampler(FakeReExecutor([]), rate=1.0, roll=lambda: 0.999999)
    assert s.should_sample() is True


# ── verify: the core verdicts ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skipped_when_no_independent_provider():
    # Single-node / single-provider: re-executor has nobody else → SKIPPED,
    # no penalty (cannot verify, must not punish).
    re_exec = FakeReExecutor([None])
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    out = await _verify(_sampler(re_exec, rep=rep, chal=chal))
    assert out.verdict == VerificationVerdict.SKIPPED
    assert rep.recorded == []
    assert chal.evidence == []


@pytest.mark.asyncio
async def test_verified_when_reexecution_agrees():
    re_exec = FakeReExecutor([ReExecResult("prov-B", "hash-X")])
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    out = await _verify(_sampler(re_exec, rep=rep, chal=chal), out="hash-X")
    assert out.verdict == VerificationVerdict.VERIFIED
    assert rep.recorded == []          # honest agreement → nobody penalized
    assert chal.evidence == []
    assert len(re_exec.calls) == 1     # no tiebreaker needed on agreement


@pytest.mark.asyncio
async def test_reexecutor_never_asked_to_rerun_on_original():
    re_exec = FakeReExecutor([ReExecResult("prov-B", "hash-X")])
    await _verify(_sampler(re_exec), original="prov-A", out="hash-X")
    assert "prov-A" in re_exec.calls[0]   # original always excluded


# ── verify: mismatch → tiebreaker → penalize the OUTLIER ─────────────────


@pytest.mark.asyncio
async def test_mismatch_majority_against_original_penalizes_original():
    # original=hash-X, two independent re-execs both =hash-Y → original is the
    # 1-of-3 outlier → MISMATCH, original penalized.
    re_exec = FakeReExecutor([
        ReExecResult("prov-B", "hash-Y"),
        ReExecResult("prov-C", "hash-Y"),
    ])
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    out = await _verify(_sampler(re_exec, rep=rep, chal=chal), original="prov-A", out="hash-X")
    assert out.verdict == VerificationVerdict.MISMATCH
    assert out.penalized_provider_ids == ["prov-A"]
    assert rep.recorded == ["prov-A"]
    assert len(re_exec.calls) == 2     # one tiebreaker fetched


@pytest.mark.asyncio
async def test_mismatch_but_original_in_majority_penalizes_the_liar_not_original():
    # original=hash-X agrees with tiebreaker prov-C=hash-X; prov-B=hash-Y is the
    # outlier → original is VERIFIED (honest); the lying re-exec is penalized.
    re_exec = FakeReExecutor([
        ReExecResult("prov-B", "hash-Y"),   # disagrees first
        ReExecResult("prov-C", "hash-X"),   # tiebreaker sides with original
    ])
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    out = await _verify(_sampler(re_exec, rep=rep, chal=chal), original="prov-A", out="hash-X")
    assert out.verdict == VerificationVerdict.VERIFIED
    assert "prov-A" not in out.penalized_provider_ids
    assert out.penalized_provider_ids == ["prov-B"]
    assert rep.recorded == ["prov-B"]


@pytest.mark.asyncio
async def test_inconclusive_when_no_majority_no_penalty():
    # All three disagree → cannot establish truth → INCONCLUSIVE, NOBODY slashed.
    re_exec = FakeReExecutor([
        ReExecResult("prov-B", "hash-Y"),
        ReExecResult("prov-C", "hash-Z"),
    ])
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    out = await _verify(_sampler(re_exec, rep=rep, chal=chal), original="prov-A", out="hash-X")
    assert out.verdict == VerificationVerdict.INCONCLUSIVE
    assert out.penalized_provider_ids == []
    assert rep.recorded == []
    assert chal.evidence == []


@pytest.mark.asyncio
async def test_mismatch_no_tiebreaker_available_is_inconclusive():
    # 2-way mismatch and no third provider to break the tie → INCONCLUSIVE,
    # never slash on a bare 1-vs-1 disagreement.
    re_exec = FakeReExecutor([ReExecResult("prov-B", "hash-Y"), None])
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    out = await _verify(_sampler(re_exec, rep=rep, chal=chal), original="prov-A", out="hash-X")
    assert out.verdict == VerificationVerdict.INCONCLUSIVE
    assert rep.recorded == []
    assert chal.evidence == []


# ── penalty routing: challenge only for bonded providers ─────────────────


@pytest.mark.asyncio
async def test_bonded_outlier_routes_onchain_challenge():
    re_exec = FakeReExecutor([
        ReExecResult("prov-B", "hash-Y"),
        ReExecResult("prov-C", "hash-Y"),
    ])
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    out = await _verify(
        _sampler(re_exec, rep=rep, chal=chal), original="prov-A", out="hash-X", bonded=True
    )
    assert out.verdict == VerificationVerdict.MISMATCH
    assert len(chal.evidence) == 1
    ev = chal.evidence[0]
    assert ev["accused_provider_id"] == "prov-A"
    assert ev["reason"] == "CONSENSUS_MISMATCH"
    assert ev["accused_output_hash"] == "hash-X"
    assert ev["majority_output_hash"] == "hash-Y"


@pytest.mark.asyncio
async def test_unbonded_outlier_reputation_only_no_challenge():
    re_exec = FakeReExecutor([
        ReExecResult("prov-B", "hash-Y"),
        ReExecResult("prov-C", "hash-Y"),
    ])
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    out = await _verify(
        _sampler(re_exec, rep=rep, chal=chal), original="prov-A", out="hash-X", bonded=False
    )
    assert out.verdict == VerificationVerdict.MISMATCH
    assert rep.recorded == ["prov-A"]
    assert chal.evidence == []      # unbonded → reputation deterrent only


# ── robustness ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reexecutor_error_yields_error_verdict_no_penalty():
    re_exec = FakeReExecutor([ReExecResult("prov-B", "hash-Y")])
    re_exec.raise_on_call = True
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    out = await _verify(_sampler(re_exec, rep=rep, chal=chal))
    assert out.verdict == VerificationVerdict.ERROR
    assert rep.recorded == []
    assert chal.evidence == []


@pytest.mark.asyncio
async def test_custom_comparator_tolerates_close_outputs():
    # A tolerance comparator (for non-greedy / heterogeneous-hardware drift)
    # treats "near" outputs as agreeing → VERIFIED, no false-positive slash.
    def tolerant(a, b):
        return a[:4] == b[:4]
    re_exec = FakeReExecutor([ReExecResult("prov-B", "hash-Xdrifted")])
    rep = FakeReputationSink()
    out = await _verify(
        _sampler(re_exec, rep=rep, comparator=tolerant), out="hash-Xoriginal"
    )
    assert out.verdict == VerificationVerdict.VERIFIED
    assert rep.recorded == []


@pytest.mark.asyncio
async def test_stats_counts_outcomes():
    rep, chal = FakeReputationSink(), FakeChallengeSink()
    # one verified
    await _verify(_sampler(FakeReExecutor([ReExecResult("prov-B", "hash-X")]), rep=rep, chal=chal), out="hash-X")
    s = _sampler(
        FakeReExecutor([ReExecResult("prov-B", "hash-Y"), ReExecResult("prov-C", "hash-Y")]),
        rep=rep, chal=chal,
    )
    await _verify(s, original="prov-A", out="hash-X")
    stats = s.get_stats()
    assert stats["mismatch"] == 1
    assert stats["sampled"] >= 1


# ── live integration: the ComputeRequester adapter glue ──────────────────

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

from prsm.node.compute_provider import JobStatus  # noqa: E402
from prsm.node.compute_requester import ComputeRequester, SubmittedJob  # noqa: E402


def _make_requester(capable_peers=None):
    identity = MagicMock()
    identity.node_id = "requester_node"
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=1000.0)
    ledger.transfer = AsyncMock()
    discovery = MagicMock()
    discovery.record_job_success = MagicMock()
    discovery.record_job_failure = MagicMock()
    discovery.find_peers_with_capability = MagicMock(
        return_value=[MagicMock(node_id=p, reliability_score=1.0) for p in (capable_peers or [])]
    )
    discovery.find_peers_with_backend = MagicMock(return_value=[])
    req = ComputeRequester(
        identity=identity, transport=MagicMock(), gossip=gossip,
        ledger=ledger, discovery=discovery,
    )
    req.escrow = None
    req.ledger_sync = None
    req._running = True
    return req


async def _drain(req):
    tasks = list(req._verify_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_adapter_fires_verification_and_penalizes_outlier_on_mismatch():
    # A remote, validly-signed result that gets sampled and fails re-execution
    # → the real reputation sink drops the provider's reliability.
    req = _make_requester()
    re_exec = FakeReExecutor([ReExecResult("prov-B", "hash-Y"), ReExecResult("prov-C", "hash-Y")])
    req.sampler = ComputeResultSampler(
        re_executor=re_exec,
        reputation_sink=req._record_verification_failure,   # real sink → discovery
        sample_rate=1.0, roll=lambda: 0.0,
    )
    req.submitted_jobs["job_x"] = SubmittedJob(
        job_id="job_x", job_type=JobType.INFERENCE, payload={"prompt": "hi"}, ftns_budget=0.0,
    )
    with patch("prsm.node.compute_requester.verify_signature", return_value=True):
        await req._on_job_result("job_result", {
            "job_id": "job_x", "provider_id": "provider_node", "status": "completed",
            "result": {"output": "fabricated"}, "public_key": "pk", "signature": "sig",
        }, "provider_node")
        await _drain(req)
    # provider_node was the 1-of-3 outlier → penalized via reputation.
    req.discovery.record_job_failure.assert_any_call("provider_node")
    assert re_exec.calls, "re-execution should have been attempted"


@pytest.mark.asyncio
async def test_adapter_does_not_sample_a_verification_run():
    # The re-execution job itself must never be re-verified (no regress).
    req = _make_requester()
    re_exec = FakeReExecutor([ReExecResult("prov-B", "hash-Y")])
    req.sampler = ComputeResultSampler(re_executor=re_exec, sample_rate=1.0, roll=lambda: 0.0)
    job = SubmittedJob(
        job_id="job_v", job_type=JobType.INFERENCE, payload={"p": 1}, ftns_budget=0.0,
        verification_run=True,
    )
    req.submitted_jobs["job_v"] = job
    with patch("prsm.node.compute_requester.verify_signature", return_value=True):
        await req._on_job_result("job_result", {
            "job_id": "job_v", "provider_id": "provider_node", "status": "completed",
            "result": {"output": "x"}, "public_key": "pk", "signature": "sig",
        }, "provider_node")
        await _drain(req)
    assert re_exec.calls == [], "a verification run must not itself be sampled"
    assert not req._verify_tasks


@pytest.mark.asyncio
async def test_real_reexec_adapter_skips_when_no_independent_provider():
    # Single-provider network: _get_capable_peers yields nobody else → the real
    # re-executor returns None → SKIPPED → no false penalty.
    req = _make_requester(capable_peers=[])   # discovery finds no capable peers
    req.sampler = ComputeResultSampler(
        re_executor=req._reexec_for_verification,
        reputation_sink=req._record_verification_failure,
        sample_rate=1.0, roll=lambda: 0.0,
    )
    req.submitted_jobs["job_s"] = SubmittedJob(
        job_id="job_s", job_type=JobType.INFERENCE, payload={"prompt": "hi"}, ftns_budget=0.0,
    )
    with patch("prsm.node.compute_requester.verify_signature", return_value=True):
        await req._on_job_result("job_result", {
            "job_id": "job_s", "provider_id": "provider_node", "status": "completed",
            "result": {"output": "z"}, "public_key": "pk", "signature": "sig",
        }, "provider_node")
        await _drain(req)
    req.discovery.record_job_failure.assert_not_called()
    assert req.sampler.get_stats()["skipped"] == 1


@pytest.mark.asyncio
async def test_targeted_job_rejects_accept_from_disallowed_provider():
    # THE CRITICAL BYPASS: a verification re-run must NOT be executable by the
    # original (excluded) provider. allowed_providers must be ENFORCED at accept
    # time — target_peers in the offer is only a hint a malicious provider
    # ignores. Without enforcement, the original grabs its own re-run, returns
    # the same fabricated result, and the comparison falsely VERIFIES.
    req = _make_requester()
    job = SubmittedJob(
        job_id="job_t", job_type=JobType.INFERENCE, payload={"p": 1}, ftns_budget=0.0,
        verification_run=True, allowed_providers={"prov-B", "prov-C"},
    )
    req.submitted_jobs["job_t"] = job
    # original / excluded provider tries to grab its own re-run
    await req._on_job_accept("job_accept", {
        "job_id": "job_t", "provider_id": "prov-A", "public_key": "pk",
    }, "prov-A")
    assert job.provider_id != "prov-A"
    assert job.status == JobStatus.PENDING        # NOT confirmed to the excluded provider
    # an allowed independent provider may take it
    await req._on_job_accept("job_accept", {
        "job_id": "job_t", "provider_id": "prov-B", "public_key": "pk",
    }, "prov-B")
    assert job.provider_id == "prov-B"
    assert job.status == JobStatus.ACCEPTED


@pytest.mark.asyncio
async def test_reexec_marks_allowed_providers_excluding_original():
    # The real adapter must stamp allowed_providers = the independent capable
    # set (original excluded) so accept-time enforcement can bite.
    req = _make_requester(capable_peers=["prov-B", "prov-C"])
    req.get_result = AsyncMock(return_value=None)   # don't block on the re-run
    out = await req._reexec_for_verification(JobType.INFERENCE, {"p": 1}, exclude={"prov-A"})
    assert out is None                              # no result → SKIPPED upstream
    reruns = [j for j in req.submitted_jobs.values() if j.verification_run]
    assert len(reruns) == 1
    assert reruns[0].allowed_providers == {"prov-B", "prov-C"}
    assert "prov-A" not in reruns[0].allowed_providers
    assert reruns[0].ftns_budget == 0.0             # re-runs are always free (no mis-pay regress)


@pytest.mark.asyncio
async def test_self_compute_is_never_sampled():
    # Self-compute (provider == own node) is trusted; no re-execution.
    req = _make_requester(capable_peers=["other"])
    re_exec = FakeReExecutor([ReExecResult("other", "hash-Y")])
    req.sampler = ComputeResultSampler(re_executor=re_exec, sample_rate=1.0, roll=lambda: 0.0)
    req.submitted_jobs["job_self"] = SubmittedJob(
        job_id="job_self", job_type=JobType.INFERENCE, payload={"p": 1}, ftns_budget=0.0,
    )
    await req._on_job_result("job_result", {
        "job_id": "job_self", "provider_id": "requester_node", "status": "completed",
        "result": {"output": "ok"},
    }, "requester_node")
    await _drain(req)
    assert re_exec.calls == []
    assert not req._verify_tasks
