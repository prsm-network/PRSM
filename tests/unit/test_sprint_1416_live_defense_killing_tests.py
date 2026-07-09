"""Sprint 1416 — killing tests for LIVE money defenses the audit found unguarded.

The sp1415 guard registry is only as strong as the killing tests behind its entries. The 133-agent
defense-liveness audit named several defenses that are on the DEFAULT money path today yet have no
test that fails when the guard is deleted — the sp1412 shape, waiting to recur. This file closes them
one at a time and registers each in prsm/security/guard_registry.py.

--- Guard 1: the result-signature reject gate (compute_requester.py) ---

``_on_job_result`` pays the provider. Before paying, it requires a remote result to carry a signature
that verifies under the accepted provider's key::

    if not verified and provider_id != self.identity.node_id:
        return   # reject unsigned / invalid-signature remote results

Neutralize that line and 177 payment/compute tests still pass (verified: `if False and ...` → all
green). The reason nothing catches it: sp924's wrong-payee and wrong-pubkey guards fire FIRST, so
their tests reject on identity mismatch and never reach the signature check. The untested case — and
the one the gate exists for — is a result from the ACCEPTED provider_id, with the ACCEPTED public
key, but a signature that does not verify. Both binding checks pass; only the signature is bad. With
the gate dead, that result is paid: a provider (or a MITM that kept the right ids) is paid for a
result it never validly signed.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.compute_requester import (
    ComputeRequester,
    JobStatus,
    JobType,
    SubmittedJob,
)
from prsm.node.identity import generate_node_identity
from prsm.node.payment_escrow import PaymentEscrow


def _requester():
    me = generate_node_identity()
    ledger = MagicMock()
    ledger.transfer = AsyncMock(return_value=MagicMock())
    gossip = MagicMock()
    gossip.publish = AsyncMock()
    cr = ComputeRequester(identity=me, transport=MagicMock(), gossip=gossip, ledger=ledger)
    cr.ledger_sync = None
    cr.discovery = None
    cr.escrow = None
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


def _result_from_accepted_provider(provider_identity, *, signature, result=None):
    """A result whose provider_id AND public_key MATCH the accepted provider (so the wrong-payee and
    pubkey-binding guards both PASS) — only the signature is the variable under test."""
    result = {"output": "ok"} if result is None else result
    return {
        "job_id": "job-1",
        "provider_id": provider_identity.node_id,
        "public_key": provider_identity.public_key_b64,
        "signature": signature,
        "result": result,
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_unsigned_result_from_accepted_provider_is_rejected():
    """THE killing test: correct provider + correct key, but NO signature. Only the signature gate
    stands between this and payment. If the gate is deleted, this pays."""
    cr = _requester()
    provider = generate_node_identity()
    job = _accepted_job(cr, provider)

    data = _result_from_accepted_provider(provider, signature="")
    await cr._on_job_result("result", data, provider.node_id)

    cr.ledger.transfer.assert_not_awaited()          # NOT paid for an unsigned result
    assert job.status is not JobStatus.COMPLETED     # and the job is not marked done


@pytest.mark.asyncio
async def test_invalid_signature_from_accepted_provider_is_rejected():
    """Correct provider + correct key, but a signature that does not verify (signed over other
    bytes). The wrong-payee/pubkey guards pass; only the signature gate rejects it."""
    cr = _requester()
    provider = generate_node_identity()
    _accepted_job(cr, provider)

    # a real signature, but over a DIFFERENT payload → verify_signature returns False for this result
    bogus = provider.sign(json.dumps({"output": "something else"}, sort_keys=True).encode())
    data = _result_from_accepted_provider(provider, signature=bogus, result={"output": "ok"})
    await cr._on_job_result("result", data, provider.node_id)

    cr.ledger.transfer.assert_not_awaited()


@pytest.mark.asyncio
async def test_validly_signed_result_from_accepted_provider_is_paid():
    """Regression: the gate must not be over-broad — a correctly signed result IS paid."""
    cr = _requester()
    provider = generate_node_identity()
    _accepted_job(cr, provider)

    result = {"output": "ok"}
    good = provider.sign(json.dumps(result, sort_keys=True).encode())
    data = _result_from_accepted_provider(provider, signature=good, result=result)
    await cr._on_job_result("result", data, provider.node_id)

    cr.ledger.transfer.assert_awaited_once()
    assert cr.ledger.transfer.await_args.kwargs["to_wallet"] == provider.node_id


@pytest.mark.asyncio
async def test_self_compute_result_is_exempt_from_the_signature_gate():
    """Self-compute (provider_id == own node id) is trusted without a signature — the gate's own
    carve-out. Deleting `and provider_id != self.identity.node_id` must not start rejecting these."""
    cr = _requester()
    job = SubmittedJob(
        job_id="job-1", job_type=JobType.INFERENCE, payload={},
        ftns_budget=10.0, status=JobStatus.PENDING,
        provider_id=cr.identity.node_id,          # self
        provider_public_key=None,
    )
    cr.submitted_jobs["job-1"] = job
    data = {
        "job_id": "job-1", "provider_id": cr.identity.node_id, "public_key": "",
        "signature": "", "result": {"output": "ok"}, "status": "completed",
    }
    await cr._on_job_result("result", data, cr.identity.node_id)

    assert job.status is JobStatus.COMPLETED       # self-compute completes without a signature
    cr.ledger.transfer.assert_not_awaited()        # and pays nobody (payer == payee)


# --- Guard 2: the sp907 escrow per-job serialization lock (payment_escrow.py) ---
#
# release/refund/split scan for the PENDING escrow, then `await` the ledger BEFORE writing the
# terminal RELEASED status (payment_escrow.py: PENDING check ~271, transfer ~302, status set ~310).
# Two concurrent releases on the same job both observe PENDING across the awaits and both pay — the
# escrow wallet goes negative (FTNS minted from nothing). The sp907 fix is a per-job asyncio.Lock held
# across the whole body so the loser re-evaluates after the winner commits RELEASED.
#
# sp907 shipped a race test — but it drives the REAL LocalLedger, whose awaits resolve without yielding
# to the sibling coroutine at the critical point, so the two gathered releases never actually
# interleave (coroutine A runs to completion before B starts). Neutralize the lock and that test STILL
# PASSES — it does not kill the guard. This test forces the interleave with a ledger that yields at the
# race point, so removing the lock provably double-pays.


class _RacyLedger:
    """An in-memory ledger whose get_balance/transfer YIELD (await asyncio.sleep(0)) at the exact
    points _release_escrow_locked awaits — forcing two concurrent releases to interleave on the single
    event loop. The real LocalLedger does not yield there, which is why the sp907 test is vacuous."""

    def __init__(self, balances):
        self.balances = dict(balances)
        self.transfers = []            # (from, to, amount)

    async def get_balance(self, wallet):
        await asyncio.sleep(0)         # yield: let the sibling coroutine reach its PENDING check
        return self.balances.get(wallet, 0.0)

    async def transfer(self, from_wallet, to_wallet, amount, description="", **_kw):
        await asyncio.sleep(0)         # yield at the critical point, BEFORE the caller sets RELEASED
        self.balances[from_wallet] = self.balances.get(from_wallet, 0.0) - amount
        self.balances[to_wallet] = self.balances.get(to_wallet, 0.0) + amount
        self.transfers.append((from_wallet, to_wallet, amount))
        return SimpleNamespace(tx_id=f"tx-{len(self.transfers)}")


async def _escrow_with_pending_job():
    ledger = _RacyLedger({"alice": 100.0})
    escrow = PaymentEscrow(ledger=ledger, node_id="alice")
    entry = await escrow.create_escrow(job_id="job-1", amount=10.0, requester_id="alice")
    ledger.transfers.clear()           # ignore the funding transfer; count only release transfers
    return ledger, escrow, f"escrow-{entry.escrow_id}"


@pytest.mark.asyncio
async def test_concurrent_release_pays_the_provider_exactly_once():
    """THE killing test: two concurrent releases, forced to interleave. With the lock, the provider is
    paid once and the escrow wallet drains to 0. Without the lock, both transfer -> 20 to bob, escrow
    wallet -10 (minted from nothing)."""
    ledger, escrow, escrow_wallet = await _escrow_with_pending_job()

    await asyncio.gather(
        escrow.release_escrow("job-1", "bob"),
        escrow.release_escrow("job-1", "bob"),
        return_exceptions=True,
    )

    to_bob = [t for t in ledger.transfers if t[1] == "bob"]
    assert len(to_bob) == 1, f"provider paid {len(to_bob)}x — the escrow lock did not serialize"
    assert ledger.balances["bob"] == 10.0
    assert ledger.balances[escrow_wallet] >= 0.0    # never negative — no mint from nothing


@pytest.mark.asyncio
async def test_concurrent_release_and_refund_have_a_single_winner():
    """Release racing refund on the same job: exactly one terminal outcome, escrow never negative."""
    ledger, escrow, escrow_wallet = await _escrow_with_pending_job()

    await asyncio.gather(
        escrow.release_escrow("job-1", "bob"),
        escrow.refund_escrow("job-1", reason="race"),
        return_exceptions=True,
    )

    paid_bob = ledger.balances.get("bob", 0.0) == 10.0
    refunded_alice = ledger.balances.get("alice", 0.0) == 100.0
    assert paid_bob != refunded_alice, (
        f"both-won race: bob={ledger.balances.get('bob')} alice={ledger.balances.get('alice')}"
    )
    assert ledger.balances[escrow_wallet] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
