"""Sprint 1408 — never pay for a self-declared MOCK result.

A provider with no inference backend (a bare ``pip install prsm`` without the ``.[ml]``
extra, or env drift losing ``PRSM_INFERENCE_EXECUTOR``) does not fail the job: it
fabricates a placeholder string and tags it ``source: "mock"``
(``compute_provider._run_inference`` fallback), then signs it with its own node key.

On the requester, ``_on_job_result`` gated payment on the SIGNATURE alone. The signature
verifies — the result genuinely IS from the accepted provider — so the job was marked
COMPLETED and the locked escrow was RELEASED to that provider. The requester paid its full
FTNS budget for the string "[PRSM node abc12345 processed inference]".

A signature proves ORIGIN, not that any compute happened. A result the provider's own
software labels a mock must never be paid for. These tests pin that.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.compute_provider import JobType
from prsm.node.compute_requester import ComputeRequester, JobStatus, SubmittedJob
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from prsm.node.transport import WebSocketTransport


async def _make_requester(port: int):
    identity = generate_node_identity("requester")
    transport = WebSocketTransport(identity, host="127.0.0.1", port=port)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    gossip.publish = AsyncMock()
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    return ComputeRequester(identity=identity, transport=transport, gossip=gossip, ledger=ledger)


def _job(req, provider_id, escrow_id="esc-1"):
    job = SubmittedJob(job_id="jx", job_type=JobType.INFERENCE, payload={}, ftns_budget=1.0)
    job.provider_id = provider_id
    job.provider_public_key = "PUBKEY"
    job.escrow_id = escrow_id
    job.status = JobStatus.ACCEPTED
    req.submitted_jobs[job.job_id] = job
    return job


def _wire_money(req):
    req.escrow = MagicMock()
    req.escrow.release_escrow = AsyncMock(return_value=MagicMock())
    req.ledger_sync = MagicMock()
    req.ledger_sync.broadcast_transaction = AsyncMock()
    req.ledger.transfer = AsyncMock(return_value=MagicMock())
    req.discovery = MagicMock()


# The exact payload compute_provider._run_inference fabricates with no LLM backend.
_MOCK_RESULT = {
    "model": "nwtn",
    "prompt": "hello",
    "response": "[PRSM node abc12345 processed inference]",
    "tokens_used": 1,
    "provider_node": "p" * 32,
    "source": "mock",
    "warning": "No LLM backend configured. This is a mock response.",
}
_REAL_RESULT = {"model": "gpt2", "response": "Paris", "source": "local_inference"}


def _msg(result, provider):
    return {"job_id": "jx", "status": "ok", "result": result,
            "signature": "SIG", "public_key": "PUBKEY", "provider_id": provider}


@pytest.mark.asyncio
async def test_signed_remote_mock_result_is_never_paid(monkeypatch):
    """The money bug: a VALIDLY SIGNED mock from the accepted provider must not be paid."""
    req = await _make_requester(19410)
    provider = "p" * 32
    job = _job(req, provider)
    _wire_money(req)
    monkeypatch.setattr("prsm.node.compute_requester.verify_signature", lambda *a, **k: True)

    await req._on_job_result("job_result", _msg(_MOCK_RESULT, provider), provider)

    req.escrow.release_escrow.assert_not_called()   # escrow stays locked — provider unpaid
    req.ledger.transfer.assert_not_called()         # and no direct transfer either
    assert job.status is not JobStatus.COMPLETED    # a fabricated answer is not a completed job
    assert job.status is JobStatus.FAILED
    assert "mock" in (job.error or "").lower()
    req.discovery.record_job_failure.assert_called_once_with(provider)


@pytest.mark.asyncio
async def test_mock_rejection_unblocks_the_waiter(monkeypatch):
    """The submitter must get a fast, honest failure — not hang until the job times out."""
    req = await _make_requester(19411)
    provider = "p" * 32
    job = _job(req, provider)
    _wire_money(req)
    monkeypatch.setattr("prsm.node.compute_requester.verify_signature", lambda *a, **k: True)

    await req._on_job_result("job_result", _msg(_MOCK_RESULT, provider), provider)

    assert job._result_event.is_set()


@pytest.mark.asyncio
async def test_mock_embedding_result_is_also_rejected(monkeypatch):
    """_run_embedding has the same `source: "mock"` fallback — the gate is on the field."""
    req = await _make_requester(19412)
    provider = "p" * 32
    job = _job(req, provider)
    _wire_money(req)
    monkeypatch.setattr("prsm.node.compute_requester.verify_signature", lambda *a, **k: True)

    mock_emb = {"embedding": [0.0] * 8, "source": "mock"}
    await req._on_job_result("job_result", _msg(mock_emb, provider), provider)

    req.escrow.release_escrow.assert_not_called()
    assert job.status is JobStatus.FAILED


@pytest.mark.asyncio
async def test_real_remote_result_still_pays(monkeypatch):
    """Regression: a genuine result is unaffected — escrow still releases to the provider."""
    req = await _make_requester(19413)
    provider = "p" * 32
    job = _job(req, provider)
    _wire_money(req)
    monkeypatch.setattr("prsm.node.compute_requester.verify_signature", lambda *a, **k: True)

    await req._on_job_result("job_result", _msg(_REAL_RESULT, provider), provider)

    req.escrow.release_escrow.assert_awaited_once()
    assert req.escrow.release_escrow.call_args.kwargs["provider_id"] == provider
    assert job.status is JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_result_without_source_field_still_pays(monkeypatch):
    """Regression: legacy/other job types carry no `source` — must not be swept up by the gate."""
    req = await _make_requester(19414)
    provider = "p" * 32
    job = _job(req, provider)
    _wire_money(req)
    monkeypatch.setattr("prsm.node.compute_requester.verify_signature", lambda *a, **k: True)

    await req._on_job_result("job_result", _msg({"response": "hi"}, provider), provider)

    req.escrow.release_escrow.assert_awaited_once()
    assert job.status is JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_self_compute_mock_is_not_rejected(monkeypatch):
    """Self-compute pays nobody (payer == payee), so sp1390/sp1387's local path must keep working."""
    req = await _make_requester(19415)
    me = req.identity.node_id
    job = _job(req, me, escrow_id=None)
    job.provider_public_key = None
    _wire_money(req)
    monkeypatch.setattr("prsm.node.compute_requester.verify_signature", lambda *a, **k: False)

    await req._on_job_result("job_result", _msg(_MOCK_RESULT, me), me)

    assert job.status is JobStatus.COMPLETED     # unchanged for the same-node path
    req.ledger.transfer.assert_not_called()      # and it never pays itself anyway


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
