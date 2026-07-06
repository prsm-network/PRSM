"""Sprint 1387 — single-node self-compute returns a REAL answer via the local model, not a mock.

_run_inference used to go NWTN-orchestrator -> mock string, never touching the node's loaded local
model. Now it runs node.inference_executor (wired by node.py) before the mock, so a bare-install
single node's compute jobs produce real output. The executor is faked so no model loads here.
"""
import pytest

from prsm.node.compute_provider import ComputeJob, ComputeProvider, JobType
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from prsm.node.transport import WebSocketTransport


class _Res:
    def __init__(self, ok, out, err=None):
        self.success = ok
        self.output = out
        self.error = err


class _FakeExecutor:
    def __init__(self, res):
        self._res = res
        self.seen = []

    def supported_models(self):
        return ["distilgpt2"]

    async def execute(self, req):
        self.seen.append(req)
        return self._res


async def _make_provider():
    identity = generate_node_identity("provider")
    transport = WebSocketTransport(identity, host="127.0.0.1", port=19355)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    return ComputeProvider(
        identity=identity, transport=transport, gossip=gossip, ledger=ledger,
        max_concurrent_jobs=2)


def _job():
    return ComputeJob(
        job_id="j-local-1", job_type=JobType.INFERENCE, requester_id="req",
        payload={"prompt": "The capital of France is", "model": "nwtn", "max_tokens": 8},
        ftns_budget=0.0)


@pytest.mark.asyncio
async def test_self_compute_uses_local_inference():
    p = await _make_provider()
    p.orchestrator = None
    ex = _FakeExecutor(_Res(True, " Paris"))
    p.inference_executor = ex
    res = await p._run_inference(_job())
    assert res["source"] == "local_inference"       # real model, not mock
    assert res["response"] == " Paris"
    assert res["model"] == "distilgpt2"              # unserved "nwtn" remapped to a served model
    assert ex.seen and ex.seen[0].prompt == "The capital of France is"


@pytest.mark.asyncio
async def test_falls_back_to_mock_without_executor():
    p = await _make_provider()
    p.orchestrator = None
    p.inference_executor = None
    res = await p._run_inference(_job())
    assert res["source"] == "mock"                   # unchanged when no local model available


@pytest.mark.asyncio
async def test_falls_back_to_mock_on_executor_failure():
    p = await _make_provider()
    p.orchestrator = None
    p.inference_executor = _FakeExecutor(_Res(False, "", err="boom"))
    res = await p._run_inference(_job())
    assert res["source"] == "mock"                   # executor failure -> mock, never a crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
