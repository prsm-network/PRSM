"""Sprint 1390 — /compute/submit self-computes on a single node + delivers the result directly.

transport.gossip has no loopback, so a self-computed result never reaches the submitting job via the
normal GOSSIP_JOB_RESULT path. deliver_local_result injects it directly. (The endpoint's background
self-compute is validated live; this pins the delivery primitive.)
"""
import pytest

from prsm.node.compute_provider import JobStatus, JobType
from prsm.node.compute_requester import ComputeRequester, SubmittedJob
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from prsm.node.transport import WebSocketTransport


async def _make_requester():
    identity = generate_node_identity("req")
    transport = WebSocketTransport(identity, host="127.0.0.1", port=19366)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    return ComputeRequester(identity=identity, transport=transport, gossip=gossip, ledger=ledger)


@pytest.mark.asyncio
async def test_deliver_local_result_completes_pending_job():
    r = await _make_requester()
    job = SubmittedJob(job_id="j1", job_type=JobType.INFERENCE, payload={"prompt": "hi"}, ftns_budget=0.0)
    r.submitted_jobs["j1"] = job
    ok = r.deliver_local_result("j1", {"response": " Paris", "source": "local_inference"})
    assert ok is True
    assert job.status == JobStatus.COMPLETED
    assert job.result["response"] == " Paris"
    assert job._result_event.is_set()             # get_result() waiters wake up


@pytest.mark.asyncio
async def test_deliver_local_result_idempotent_and_unknown():
    r = await _make_requester()
    job = SubmittedJob(job_id="j1", job_type=JobType.INFERENCE, payload={}, ftns_budget=0.0)
    job._result_event.set()                        # already resolved (e.g. by a peer)
    r.submitted_jobs["j1"] = job
    assert r.deliver_local_result("j1", {"x": 1}) is False    # no double-delivery
    assert r.deliver_local_result("nope", {"x": 1}) is False  # unknown job


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
