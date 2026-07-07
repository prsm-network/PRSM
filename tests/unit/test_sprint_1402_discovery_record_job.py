"""Sprint 1402 — PeerDiscovery has record_job_success/failure; a tracking error can't block payment.

Root cause of the cross-node "settles to no one" bug: compute_requester._on_job_result calls
self.discovery.record_job_success(provider) after marking the job complete but BEFORE releasing the
provider's payment. record_job_success existed only on Libp2pDiscovery — on the WebSocket transport
(PeerDiscovery) it raised AttributeError, killing the handler before payout. Fix: add the methods to
PeerDiscovery (data fields already existed) + make the call best-effort so payment is never blocked.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.compute_provider import JobType
from prsm.node.compute_requester import ComputeRequester, JobStatus, SubmittedJob
from prsm.node.discovery import PeerDiscovery, PeerInfo
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from prsm.node.transport import WebSocketTransport


def _discovery():
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = "n" * 32
    return PeerDiscovery(transport=transport, bootstrap_nodes=[])


def test_peerdiscovery_has_record_job_methods():
    d = _discovery()
    d.known_peers["peerA"] = PeerInfo(node_id="peerA", address="1.2.3.4")
    d.record_job_success("peerA")
    d.record_job_success("peerA")
    d.record_job_failure("peerA")
    assert d.known_peers["peerA"].job_success_count == 2
    assert d.known_peers["peerA"].job_failure_count == 1
    assert d.known_peers["peerA"].last_failure_time > 0
    # unknown peer → no crash
    d.record_job_success("nobody")
    d.record_job_failure("nobody")


async def _make_requester():
    identity = generate_node_identity("requester")
    transport = WebSocketTransport(identity, host="127.0.0.1", port=19402)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    gossip.publish = AsyncMock()
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    return ComputeRequester(identity=identity, transport=transport, gossip=gossip, ledger=ledger)


@pytest.mark.asyncio
async def test_tracking_error_does_not_block_payment(monkeypatch):
    req = await _make_requester()
    provider = "p" * 32
    job = SubmittedJob(job_id="jx", job_type=JobType.INFERENCE, payload={}, ftns_budget=1.0)
    job.provider_id = provider
    job.provider_public_key = "PUBKEY"
    job.escrow_id = "esc-1"
    job.status = JobStatus.ACCEPTED
    req.submitted_jobs["jx"] = job
    # discovery whose record_job_success RAISES (mimics the old AttributeError)
    req.discovery = MagicMock()
    req.discovery.record_job_success = MagicMock(side_effect=AttributeError("boom"))
    req.escrow = MagicMock()
    req.escrow.release_escrow = AsyncMock(return_value=MagicMock())
    req.ledger_sync = MagicMock()
    req.ledger_sync.broadcast_transaction = AsyncMock()
    monkeypatch.setattr("prsm.node.compute_requester.verify_signature", lambda *a, **k: True)

    await req._on_job_result("job_result", {
        "job_id": "jx", "provider_id": provider, "status": "ok",
        "result": {"response": "hi"}, "signature": "SIG", "public_key": "PUBKEY"}, provider)

    req.escrow.release_escrow.assert_awaited_once()   # payment happened DESPITE the tracking error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
