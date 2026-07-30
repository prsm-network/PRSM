"""Sprint 1489 — a failed escrow broadcast must not vanish silently.

Both escrow broadcast sites were bare `except Exception: pass`. The NON-ROLLBACK
part is right — the local ledger transfer has already committed, and reversing it
would reopen the TOCTOU that ordering exists to close (and on release, the
provider did honest work). The SILENCE is not: a broadcast that never landed left
no trace anywhere, so the local ledger says paid while the chain does not, and an
operator cannot reconcile a divergence they were never told about.

These tests assert the divergence is reported and that the escrow still succeeds.
"""
from __future__ import annotations

import logging

import pytest

from prsm.node.payment_escrow import PaymentEscrow


class _Boom:
    """A broadcast that always fails, as an unreachable chain/peer would."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, tx):
        self.calls += 1
        raise RuntimeError("rpc unreachable")


@pytest.fixture
async def ledger():
    from prsm.node.local_ledger import LocalLedger, TransactionType
    led = LocalLedger(":memory:")
    await led.initialize()
    await led.create_wallet("requester-1", "Requester")
    await led.create_wallet("provider-9", "Provider")
    await led.credit(wallet_id="requester-1", amount=100.0,
                     tx_type=TransactionType.WELCOME_GRANT, description="seed")
    yield led
    if led._db is not None:
        await led._db.close()


@pytest.mark.asyncio
async def test_create_broadcast_failure_is_LOGGED_not_swallowed(ledger, caplog):
    """★ Was a bare `pass` — the failure left no trace at all."""
    boom = _Boom()
    esc = PaymentEscrow(ledger=ledger, node_id="requester-1", broadcast_transaction=boom)

    with caplog.at_level(logging.ERROR):
        entry = await esc.create_escrow(
            job_id="job-1", requester_id="requester-1", amount=10.0)

    assert entry is not None, "a broadcast failure must not fail the escrow"
    assert boom.calls == 1
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "escrow-create broadcast FAILED" in msgs
    assert "Reconciliation required" in msgs
    assert "job-1"[:8] in msgs


@pytest.mark.asyncio
async def test_release_broadcast_failure_is_LOGGED_not_swallowed(ledger, caplog):
    """★ The payout leg. A swallowed failure here means the provider is paid
    locally and NOT on chain — precisely what an operator must reconcile."""
    boom = _Boom()
    esc = PaymentEscrow(ledger=ledger, node_id="requester-1", broadcast_transaction=boom)
    await esc.create_escrow(
        job_id="job-2", requester_id="requester-1", amount=10.0)
    caplog.clear()

    with caplog.at_level(logging.ERROR):
        tx = await esc.release_escrow("job-2", "provider-9")

    assert tx is not None, "a broadcast failure must not fail the release"
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "escrow-release broadcast FAILED" in msgs
    assert "Reconciliation required" in msgs
    assert "provider-9" in msgs


@pytest.mark.asyncio
async def test_the_local_release_still_credits_the_provider(ledger):
    """Non-rollback: the provider keeps the local credit despite the failure."""
    esc = PaymentEscrow(ledger=ledger, node_id="requester-1", broadcast_transaction=_Boom())
    await esc.create_escrow(
        job_id="job-3", requester_id="requester-1", amount=10.0)
    await esc.release_escrow("job-3", "provider-9")
    assert await ledger.get_balance("provider-9") == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_a_healthy_broadcast_logs_no_error(ledger, caplog):
    """The loud path must stay quiet when nothing is wrong."""
    sent = []

    async def ok(tx):
        sent.append(tx)

    esc = PaymentEscrow(ledger=ledger, node_id="requester-1", broadcast_transaction=ok)
    with caplog.at_level(logging.ERROR):
        await esc.create_escrow(
            job_id="job-4", requester_id="requester-1", amount=10.0)
        await esc.release_escrow("job-4", "provider-9")
    assert len(sent) == 2
    assert not [r for r in caplog.records if "broadcast FAILED" in r.getMessage()]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
