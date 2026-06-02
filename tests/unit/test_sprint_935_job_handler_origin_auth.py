"""Sprint 935 — only the job's requester may confirm/cancel it (P2P transport review).

Building on sp934 (the gossip `origin` is now cryptographically authenticated),
the compute-provider job handlers must actually ENFORCE it. Before this fix:
  * _on_job_confirm executed a pending job for whoever sent a CONFIRM — it never
    checked the confirm came from the job's requester. Any peer could confirm
    (or, combined with the old origin spoof, redirect) a job.
  * _on_job_cancel removed/failed a job for ANY sender — any peer could cancel
    another requester's in-flight job (a free DoS on compute work + payment).

Fix: both handlers reject a confirm/cancel whose authenticated `origin` is not
the job's `requester_id`.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.compute_provider import ComputeJob, ComputeProvider, JobStatus, JobType
from prsm.node.identity import generate_node_identity


def _provider():
    return ComputeProvider(
        identity=generate_node_identity("provider"),
        transport=MagicMock(),
        gossip=MagicMock(),
        ledger=MagicMock(),
        max_concurrent_jobs=4,
    )


def _job(provider, requester="requester-node"):
    return ComputeJob(
        job_id="job-1", job_type=JobType.INFERENCE, requester_id=requester,
        payload={"p": 1}, ftns_budget=1.0,
    )


@pytest.mark.asyncio
async def test_confirm_from_non_requester_is_rejected():
    p = _provider()
    job = _job(p)
    p._pending_confirm = {"job-1": job}
    # An attacker (authenticated as itself by sp934) sends a confirm.
    await p._on_job_confirm("job_confirm", {"job_id": "job-1", "provider_id": p.identity.node_id}, origin="attacker-node")
    # Not executed, not consumed — the legit requester can still confirm.
    assert "job-1" in p._pending_confirm
    assert "job-1" not in p.active_jobs


@pytest.mark.asyncio
async def test_confirm_from_requester_is_accepted():
    p = _provider()
    job = _job(p)
    p._pending_confirm = {"job-1": job}
    # Requester confirms, but picks ANOTHER provider → consumed, not executed here.
    await p._on_job_confirm("job_confirm", {"job_id": "job-1", "provider_id": "other-provider"}, origin="requester-node")
    assert "job-1" not in p._pending_confirm   # auth passed → handled
    assert "job-1" not in p.active_jobs         # another provider won


@pytest.mark.asyncio
async def test_cancel_from_non_requester_is_rejected():
    p = _provider()
    job = _job(p)
    job.status = JobStatus.RUNNING
    p.active_jobs = {"job-1": job}
    await p._on_job_cancel("job_cancel", {"job_id": "job-1"}, origin="attacker-node")
    assert "job-1" in p.active_jobs            # NOT cancelled by a stranger
    assert p.active_jobs["job-1"].status == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_cancel_from_requester_is_accepted():
    p = _provider()
    job = _job(p)
    job.status = JobStatus.RUNNING
    p.active_jobs = {"job-1": job}
    await p._on_job_cancel("job_cancel", {"job_id": "job-1"}, origin="requester-node")
    assert "job-1" not in p.active_jobs
    assert job.status == JobStatus.FAILED


@pytest.mark.asyncio
async def test_cancel_of_pending_job_requires_requester():
    p = _provider()
    job = _job(p)
    p._pending_confirm = {"job-1": job}
    await p._on_job_cancel("job_cancel", {"job_id": "job-1"}, origin="attacker-node")
    assert "job-1" in p._pending_confirm        # stranger cannot drop a pending job
    await p._on_job_cancel("job_cancel", {"job_id": "job-1"}, origin="requester-node")
    assert "job-1" not in p._pending_confirm     # requester can
