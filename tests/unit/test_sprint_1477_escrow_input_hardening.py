"""Sprint 1477 — PaymentEscrow input hardening (audit wf_ce5c63f0, 2 LOW survivors).

The PaymentEscrow adversarial audit found the core lifecycle SOUND against
double-spend / drain (sp907 locks + sp489 ordering + state machine hold,
conservation preserved). Two LOW latent input-validation gaps survived — both
make the module self-defend regardless of backend/caller:

  LOW-1 create_escrow did not dedup by job_id, and release/refund/split locate
        their escrow by the FIRST-PENDING match — so a duplicate create (an
        idempotency retry whose response was lost) funded a SECOND escrow wallet
        that a later release could pay out separately (a requester pays twice for
        one job; conservation holds, not a drain). Fixed: a same-job_id re-create
        returns the existing live PENDING escrow instead of locking a 2nd hold,
        under the sp907 per-job lock (also serializes concurrent same-job creates).

  LOW-2 release_escrow did not validate partial_amount sign/finiteness (the split
        path does). A negative value bypasses the `escrow_balance < amount` guard;
        NaN poisons every comparison. Fixed: reject non-finite / <= 0.
"""
from __future__ import annotations

import asyncio

import pytest

from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.payment_escrow import PaymentEscrow, EscrowStatus


@pytest.fixture
async def ledger():
    led = LocalLedger(":memory:")
    await led.initialize()
    await led.create_wallet("alice", "Alice")
    await led.create_wallet("bob", "Bob")
    await led.credit(wallet_id="alice", amount=100.0,
                     tx_type=TransactionType.WELCOME_GRANT, description="seed")
    yield led
    if led._db is not None:
        await led._db.close()


@pytest.fixture
async def escrow(ledger):
    return PaymentEscrow(ledger=ledger, node_id="alice")


async def _bal(ledger, w):
    return await ledger.get_balance(w)


# ───────────────────── LOW-1 one-live-escrow-per-job_id ─────────────────────

@pytest.mark.asyncio
async def test_duplicate_create_same_job_returns_existing_no_second_hold(ledger, escrow):
    a = await escrow.create_escrow(job_id="job-1", amount=10.0, requester_id="alice")
    assert await _bal(ledger, "alice") == 90.0     # one hold
    b = await escrow.create_escrow(job_id="job-1", amount=10.0, requester_id="alice")
    # ★ same escrow returned, NO second wallet funded.
    assert b is not None and b.escrow_id == a.escrow_id
    assert await _bal(ledger, "alice") == 90.0     # still one hold, not 80
    live = [e for e in escrow._escrows.values()
            if e.job_id == "job-1" and e.status == EscrowStatus.PENDING]
    assert len(live) == 1


@pytest.mark.asyncio
async def test_concurrent_create_same_job_locks_single_escrow(ledger, escrow):
    # Two concurrent creates for one job_id → serialized by the sp907 lock →
    # exactly one hold (never two wallets funded from the same job).
    results = await asyncio.gather(
        escrow.create_escrow(job_id="job-x", amount=10.0, requester_id="alice"),
        escrow.create_escrow(job_id="job-x", amount=10.0, requester_id="alice"),
    )
    ids = {r.escrow_id for r in results if r is not None}
    assert len(ids) == 1                            # both calls → one escrow
    assert await _bal(ledger, "alice") == 90.0      # debited once, not twice
    live = [e for e in escrow._escrows.values()
            if e.job_id == "job-x" and e.status == EscrowStatus.PENDING]
    assert len(live) == 1


@pytest.mark.asyncio
async def test_release_then_recreate_same_job_is_fresh(ledger, escrow):
    # After the job's escrow is RELEASED (no longer live), a create for the same
    # job_id legitimately opens a fresh escrow (the prior job completed).
    a = await escrow.create_escrow(job_id="job-2", amount=10.0, requester_id="alice")
    await escrow.release_escrow("job-2", "bob")
    assert a.status == EscrowStatus.RELEASED
    c = await escrow.create_escrow(job_id="job-2", amount=5.0, requester_id="alice")
    assert c is not None and c.escrow_id != a.escrow_id
    assert c.status == EscrowStatus.PENDING


# ───────────────────── LOW-2 partial_amount validation ──────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-5.0, 0.0, float("nan"), float("inf"), float("-inf")])
async def test_release_bad_partial_amount_raises_no_payout(ledger, escrow, bad):
    a = await escrow.create_escrow(job_id="job-3", amount=10.0, requester_id="alice")
    bob_before = await _bal(ledger, "bob")
    with pytest.raises(ValueError):
        await escrow.release_escrow("job-3", "bob", partial_amount=bad)
    # No state change, no payout, escrow still PENDING + funded.
    assert a.status == EscrowStatus.PENDING
    assert await _bal(ledger, "bob") == bob_before
    assert await _bal(ledger, f"escrow-{a.escrow_id}") == 10.0


@pytest.mark.asyncio
async def test_release_valid_partial_amount_still_works(ledger, escrow):
    a = await escrow.create_escrow(job_id="job-4", amount=10.0, requester_id="alice")
    tx = await escrow.release_escrow("job-4", "bob", partial_amount=3.0)
    assert tx is not None
    assert await _bal(ledger, "bob") == 3.0
    assert a.status == EscrowStatus.RELEASED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
