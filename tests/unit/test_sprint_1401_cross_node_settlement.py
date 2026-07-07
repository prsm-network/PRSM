"""Sprint 1401 — cross-node job settlement: the serving PROVIDER gets paid via escrow release.

Live 2026-07-07: a cross-node /compute/submit paid nobody — us locked escrow, the job completed, but
the provider (sfo) earned nothing and the escrow stayed locked (txs_broadcast=0). Two causes:
(1) sp1390 self-compute RACED the remote provider (requester self-computed via deliver_local_result,
    which settles nothing) — fixed in api.py by skipping self-compute when a remote provider accepted;
(2) _on_job_result would have done a SECOND direct transfer on top of the locked escrow (double-pay).
Now, for an escrow job, _on_job_result RELEASES the escrow to the provider (idempotent) and broadcasts
it via ledger_sync so the provider's own ledger credits itself. This test covers (2).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.compute_provider import JobType
from prsm.node.compute_requester import ComputeRequester, JobStatus, SubmittedJob
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from prsm.node.transport import WebSocketTransport


async def _make_requester():
    identity = generate_node_identity("requester")
    transport = WebSocketTransport(identity, host="127.0.0.1", port=19400)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    gossip.publish = AsyncMock()
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    return ComputeRequester(identity=identity, transport=transport, gossip=gossip, ledger=ledger)


def _job(req, provider_id, escrow_id):
    job = SubmittedJob(job_id="jx", job_type=JobType.INFERENCE, payload={}, ftns_budget=1.0)
    job.provider_id = provider_id
    job.provider_public_key = "PUBKEY"
    job.escrow_id = escrow_id
    job.status = JobStatus.ACCEPTED
    req.submitted_jobs[job.job_id] = job
    return job


_RESULT = {"job_id": "jx", "status": "ok", "result": {"response": "hi"},
           "signature": "SIG", "public_key": "PUBKEY"}


@pytest.mark.asyncio
async def test_escrow_job_releases_escrow_not_double_pay(monkeypatch):
    req = await _make_requester()
    provider = "p" * 32
    _job(req, provider, escrow_id="esc-1")
    release_tx = MagicMock()
    req.escrow = MagicMock()
    req.escrow.release_escrow = AsyncMock(return_value=release_tx)
    req.ledger_sync = MagicMock()
    req.ledger_sync.broadcast_transaction = AsyncMock()
    req.ledger.transfer = AsyncMock()                      # must NOT be called (would double-pay)
    monkeypatch.setattr("prsm.node.compute_requester.verify_signature", lambda *a, **k: True)

    await req._on_job_result("job_result", {**_RESULT, "provider_id": provider}, provider)

    req.escrow.release_escrow.assert_awaited_once()        # locked escrow → provider
    assert req.escrow.release_escrow.call_args.kwargs["provider_id"] == provider
    req.ledger.transfer.assert_not_called()                # NO second direct transfer
    req.ledger_sync.broadcast_transaction.assert_awaited_once_with(release_tx)  # cross-ledger credit


@pytest.mark.asyncio
async def test_non_escrow_job_uses_direct_transfer(monkeypatch):
    req = await _make_requester()
    provider = "q" * 32
    _job(req, provider, escrow_id=None)                    # no escrow locked
    req.escrow = MagicMock()
    req.escrow.release_escrow = AsyncMock()
    req.ledger_sync = MagicMock()
    req.ledger_sync.broadcast_transaction = AsyncMock()
    direct_tx = MagicMock()
    req.ledger.transfer = AsyncMock(return_value=direct_tx)
    monkeypatch.setattr("prsm.node.compute_requester.verify_signature", lambda *a, **k: True)

    await req._on_job_result("job_result", {**_RESULT, "provider_id": provider}, provider)

    req.ledger.transfer.assert_awaited_once()              # direct transfer for a non-escrow job
    req.escrow.release_escrow.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
