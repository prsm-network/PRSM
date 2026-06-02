"""Sprint 924 — compute job-result payment guards (compute/inference review).

The compute/inference adversarial review found two real bugs in
ComputeRequester._on_job_result (the GOSSIP_JOB_RESULT handler that pays the
provider), both in the gossip-money class hardened across sp898/899/909/911/912:

  * CRITICAL WRONG-PAYEE: provider_id + public_key are taken from the UNTRUSTED
    result message. A third party can publish a JOB_RESULT for someone else's
    job, signed with THEIR OWN key + their own payout id — the signature
    "verifies" (against the attacker's key) and payment is redirected to the
    attacker. No check bound the result to the ACCEPTED provider
    (job.provider_id / job.provider_public_key, set at job-accept time).

  * HIGH DOUBLE-PAY: the transport nonce-dedup only covers a 300s window; a
    JOB_RESULT re-delivered after expiry re-ran the handler, which re-paid (no
    idempotency guard — job.status was set COMPLETED but never checked first).

Fix: (1) idempotency — once a job is COMPLETED, ignore further results;
(2) bind the result to the accepted provider — reject a provider_id or
public_key that doesn't match what this requester accepted for the job.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.compute_requester import (
    ComputeRequester,
    SubmittedJob,
    JobStatus,
    JobType,
)
from prsm.node.identity import generate_node_identity


def _requester():
    me = generate_node_identity()
    ledger = MagicMock()
    ledger.transfer = AsyncMock(return_value=MagicMock())
    gossip = MagicMock()
    gossip.publish = AsyncMock()
    cr = ComputeRequester(
        identity=me, transport=MagicMock(), gossip=gossip, ledger=ledger,
    )
    cr.ledger_sync = None
    cr.discovery = None
    return cr


def _accepted_job(cr, provider_identity, *, budget=10.0):
    job = SubmittedJob(
        job_id="job-1", job_type=JobType.INFERENCE, payload={},
        ftns_budget=budget, status=JobStatus.PENDING,
        provider_id=provider_identity.node_id,
        provider_public_key=provider_identity.public_key_b64,
    )
    cr.submitted_jobs["job-1"] = job
    return job


def _result_data(signer_identity, provider_id, result=None):
    result = {"output": "ok"} if result is None else result
    sig = signer_identity.sign(json.dumps(result, sort_keys=True).encode())
    return {
        "job_id": "job-1",
        "provider_id": provider_id,
        "public_key": signer_identity.public_key_b64,
        "signature": sig,
        "result": result,
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_happy_path_pays_accepted_provider_once():
    cr = _requester()
    provider = generate_node_identity()
    _accepted_job(cr, provider)
    await cr._on_job_result(
        "result", _result_data(provider, provider.node_id), provider.node_id,
    )
    cr.ledger.transfer.assert_awaited_once()
    assert cr.ledger.transfer.await_args.kwargs["to_wallet"] == provider.node_id


@pytest.mark.asyncio
async def test_wrong_payee_attacker_result_is_rejected():
    # CRITICAL: attacker B publishes a result for A's job, signed with B's own
    # key + B's payout id. Must NOT pay B.
    cr = _requester()
    provider = generate_node_identity()      # the accepted provider A
    attacker = generate_node_identity()      # B
    _accepted_job(cr, provider)
    await cr._on_job_result(
        "result", _result_data(attacker, attacker.node_id), attacker.node_id,
    )
    cr.ledger.transfer.assert_not_awaited()   # payment NOT redirected


@pytest.mark.asyncio
async def test_wrong_public_key_is_rejected():
    # Even if the attacker spoofs the accepted provider_id but signs with a
    # different key, the pubkey-binding check rejects it.
    cr = _requester()
    provider = generate_node_identity()
    attacker = generate_node_identity()
    _accepted_job(cr, provider)
    data = _result_data(attacker, provider.node_id)   # provider_id spoofed to A
    await cr._on_job_result("result", data, attacker.node_id)
    cr.ledger.transfer.assert_not_awaited()


@pytest.mark.asyncio
async def test_redelivery_does_not_double_pay():
    cr = _requester()
    provider = generate_node_identity()
    _accepted_job(cr, provider)
    data = _result_data(provider, provider.node_id)
    await cr._on_job_result("result", data, provider.node_id)   # first
    await cr._on_job_result("result", data, provider.node_id)   # re-delivery
    assert cr.ledger.transfer.await_count == 1   # paid exactly once
