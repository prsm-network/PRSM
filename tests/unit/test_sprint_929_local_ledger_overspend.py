"""Sprint 929 — LocalLedger overspend: non-atomic check-then-act (money rail review).

LocalLedger.debit/transfer/agent_debit derived the balance with an `await
get_balance()` and then inserted the debit with a separate `await _insert_tx()`,
with NO lock between. asyncio yields at every await, so two balance-decreasing
coroutines on the same wallet both read the OLD balance, both pass the check,
and both insert — driving the balance NEGATIVE (spend FTNS that doesn't exist).
The DAG ledger guards this with an atomic balance check; LocalLedger did not.

This is the same check-then-act-without-an-atomic-claim class as the gossip-money
bugs (sp898/899/911/912/924). Fix: a per-ledger write lock makes the balance
check + insert atomic across all balance-decreasing paths, and makes the
agent-allowance `spent` update atomic with its check (closing over-allowance too).
"""
from __future__ import annotations

import asyncio

import pytest

from prsm.node.local_ledger import LocalLedger, TransactionType


async def _ledger(balance: float = 100.0, wallet: str = "W"):
    led = LocalLedger(":memory:")
    await led.initialize()
    await led.create_wallet(wallet, "principal")
    if balance:
        await led.credit(wallet, balance, TransactionType.WELCOME_GRANT, "seed")
    return led


@pytest.mark.asyncio
async def test_concurrent_transfers_cannot_overspend():
    # Balance 100; five concurrent transfers of 60 each. Only ONE can fit
    # (100 - 60 = 40 < 60). Without an atomic guard, several read balance=100
    # before any insert commits and all succeed → balance goes negative.
    led = await _ledger(100.0)
    await led.create_wallet("dest", "dest")

    async def _try():
        try:
            await led.transfer("W", "dest", 60.0, TransactionType.TRANSFER, "x")
            return True
        except ValueError:
            return False

    results = await asyncio.gather(*[_try() for _ in range(5)])

    assert sum(results) == 1, f"exactly one transfer should succeed, got {sum(results)}"
    assert await led.get_balance("W") == pytest.approx(40.0)
    assert await led.get_balance("W") >= 0.0          # never negative — no overspend


@pytest.mark.asyncio
async def test_concurrent_debits_cannot_overspend():
    led = await _ledger(100.0)

    async def _try():
        try:
            await led.debit("W", 60.0, TransactionType.TRANSFER, "x")
            return True
        except ValueError:
            return False

    results = await asyncio.gather(*[_try() for _ in range(5)])
    assert sum(results) == 1
    assert await led.get_balance("W") == pytest.approx(40.0)
    assert await led.get_balance("W") >= 0.0


@pytest.mark.asyncio
async def test_concurrent_agent_debits_respect_balance_and_allowance():
    # Principal has 100; agent allowance 100. Five concurrent agent_debits of 60.
    # Only ONE fits the balance AND the allowance — the rest must be rejected
    # (None), and neither the balance nor the allowance may be exceeded.
    led = await _ledger(100.0)
    await led.grant_agent_allowance("W", "agent-1", 100.0)

    async def _try():
        tx = await led.agent_debit("agent-1", 60.0, TransactionType.TRANSFER, "x")
        return tx is not None

    results = await asyncio.gather(*[_try() for _ in range(5)])

    assert sum(results) == 1
    assert await led.get_balance("W") == pytest.approx(40.0)
    assert await led.get_balance("W") >= 0.0
    allowance = await led.get_agent_allowance("agent-1")
    assert allowance["spent"] == pytest.approx(60.0)   # spent never over-counted
    assert allowance["spent"] <= allowance["allowance"]


@pytest.mark.asyncio
async def test_sequential_transfers_still_work():
    # The lock must not break the normal sequential path.
    led = await _ledger(100.0)
    await led.create_wallet("dest", "dest")
    await led.transfer("W", "dest", 30.0, TransactionType.TRANSFER, "a")
    await led.transfer("W", "dest", 30.0, TransactionType.TRANSFER, "b")
    assert await led.get_balance("W") == pytest.approx(40.0)
    assert await led.get_balance("dest") == pytest.approx(60.0)
    with pytest.raises(ValueError):
        await led.transfer("W", "dest", 50.0, TransactionType.TRANSFER, "c")
