"""Sprint 971 — agent-collaboration escrow idempotency (refund/pay at-most-once).

Agent-collab/marketplace review finding #1 (MEDIUM). _refund_task_escrow and
_pay_task_completion lacked the idempotency guard that the reviewer/responder
paths have (paid_reviewers/paid_responders), and local_ledger.credit() is NOT
write-lock-atomic (unlike sp929's debit()) — so credits have no ledger backstop;
the app layer is the only defense. Reachable window: AgentCollaboration.stop()
refunds an OPEN/ASSIGNED task's escrow, and the _cleanup_loop expiry sweep
(_expire_stale_records) refunds the SAME task — at the shutdown boundary both
fire → credit() double-mints the budget back to the requester.

Fix: a per-task `refunded`/`paid` flag (mirroring paid_reviewers), set BEFORE the
credit await so a concurrent refund sees it and no-ops (set-before-await makes the
check-and-set atomic within the coroutine — asyncio doesn't preempt mid-sync).
refund and pay are mutually exclusive (a task is refunded XOR paid). On a credit
failure the flag rolls back so a legitimate retry can still refund.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.agent_collaboration import AgentCollaboration, TaskOffer


def _collab(credit=None):
    ledger = MagicMock()
    ledger.credit = credit if credit is not None else AsyncMock()
    c = AgentCollaboration(gossip=MagicMock(), node_id="me", ledger=ledger)
    return c, ledger


def _task(budget=10.0):
    return TaskOffer(title="t", ftns_budget=budget, requester_node_id="me")


@pytest.mark.asyncio
async def test_refund_is_idempotent_across_repeated_calls():
    c, ledger = _collab()
    task = _task(10.0)
    await c._refund_task_escrow(task)
    await c._refund_task_escrow(task)  # stop() + expiry sweep on the same task
    assert ledger.credit.await_count == 1
    assert task.refunded is True


@pytest.mark.asyncio
async def test_concurrent_refund_credits_once():
    """Two concurrent refunds (stop() racing the expiry sweep) must credit once —
    the set-before-await flag makes the second a no-op."""
    c, ledger = _collab()
    task = _task(10.0)
    await asyncio.gather(
        c._refund_task_escrow(task),
        c._refund_task_escrow(task),
    )
    assert ledger.credit.await_count == 1


@pytest.mark.asyncio
async def test_pay_after_refund_noops():
    """A refunded task must never also be paid (mutual exclusion)."""
    c, ledger = _collab()
    task = _task(10.0)
    task.assigned_agent_id = "agent-x"
    task.bids = [{"bidder_agent_id": "agent-x", "bidder_node_id": "me"}]
    await c._refund_task_escrow(task)
    await c._pay_task_completion(task)
    assert ledger.credit.await_count == 1  # only the refund
    assert task.refunded is True and task.paid is False


@pytest.mark.asyncio
async def test_refund_after_pay_noops():
    """A paid task must never also be refunded."""
    c, ledger = _collab()
    task = _task(10.0)
    task.assigned_agent_id = "agent-x"
    task.bids = [{"bidder_agent_id": "agent-x", "bidder_node_id": "me"}]
    await c._pay_task_completion(task)
    await c._refund_task_escrow(task)
    assert ledger.credit.await_count == 1  # only the payment
    assert task.paid is True and task.refunded is False


@pytest.mark.asyncio
async def test_refund_rolls_back_on_failure_to_allow_retry():
    """A transient credit failure must NOT permanently consume the refund — the
    flag rolls back so a later sweep can retry."""
    calls = {"n": 0}

    async def flaky_credit(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ledger transient error")

    c, ledger = _collab(credit=AsyncMock(side_effect=flaky_credit))
    task = _task(10.0)
    await c._refund_task_escrow(task)   # fails
    assert task.refunded is False       # rolled back
    await c._refund_task_escrow(task)   # retry succeeds
    assert task.refunded is True
    assert ledger.credit.await_count == 2
