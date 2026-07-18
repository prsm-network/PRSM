"""Sprint 1479 — welcome-grant mint integrity: exactly-once per wallet, ever.

The money-in mint-path review found the "one welcome grant per wallet" invariant
was not actually enforced:
  - issue_welcome_grant's guard was a SELECT-then-credit across an await (a
    concurrent double-grant could both pass the SELECT and both mint);
  - node._seed_welcome_grant granted on a raw credit when balance <= 0, BYPASSING
    the guard entirely — so a node granted then drained to <= 0 was re-granted
    ANOTHER 100 FTNS on every restart (a self-mint on the operator's per-node
    ledger).

Fix: issue_welcome_grant credits with a deterministic idempotency_key
(welcome-grant:{wallet}) → exactly-once even under a race / restart; the seed
path now routes through issue_welcome_grant's guard instead of a raw credit.
This tests the ledger invariant both paths now depend on.
"""
from __future__ import annotations

import asyncio

import pytest

from prsm.node.dag_ledger import DAGLedger
from prsm.node.local_ledger import LocalLedger, TransactionType

pytestmark = pytest.mark.asyncio


async def _dag():
    lg = DAGLedger(":memory:", verify_signatures=False)
    await lg.initialize()
    await lg.create_wallet("w1")
    return lg


async def _local():
    lg = LocalLedger(":memory:")
    await lg.initialize()
    await lg.create_wallet("w1")
    return lg


@pytest.mark.parametrize("factory", [_dag, _local])
async def test_second_grant_refused_balance_unchanged(factory):
    lg = await factory()
    await lg.issue_welcome_grant("w1", 100.0)
    assert await lg.get_balance("w1") == 100.0
    with pytest.raises(ValueError):
        await lg.issue_welcome_grant("w1", 100.0)
    assert await lg.get_balance("w1") == 100.0     # not 200


@pytest.mark.parametrize("factory", [_dag, _local])
async def test_concurrent_grant_mints_exactly_once(factory):
    """★ Two concurrent grants for one wallet → EXACTLY one 100-FTNS mint, even
    if both pass the SELECT guard before either commits (the idempotency_key
    makes the second credit a no-op). One may raise ValueError; balance is 100."""
    lg = await factory()
    await asyncio.gather(
        lg.issue_welcome_grant("w1", 100.0),
        lg.issue_welcome_grant("w1", 100.0),
        return_exceptions=True,
    )
    assert await lg.get_balance("w1") == 100.0     # ★ never 200


@pytest.mark.parametrize("factory", [_dag, _local])
async def test_not_reissued_after_drain_to_zero(factory):
    """★ The drain+restart re-mint: a wallet granted then drained to 0 must NOT
    be re-granted — exactly what node._seed_welcome_grant (balance <= 0) used to
    do via a raw credit. Now it routes through this guard → no re-mint."""
    lg = await factory()
    await lg.create_wallet("system")
    await lg.issue_welcome_grant("w1", 100.0)
    assert await lg.get_balance("w1") == 100.0
    # Drain the whole grant away (simulating the node spending its balance).
    await lg.debit("w1", 100.0, TransactionType.TRANSFER, description="drain")
    assert await lg.get_balance("w1") == 0.0
    # A restart's seed path re-invokes issue_welcome_grant — the WELCOME_GRANT
    # record still exists, so it refuses and does NOT mint another 100.
    with pytest.raises(ValueError):
        await lg.issue_welcome_grant("w1", 100.0)
    assert await lg.get_balance("w1") == 0.0       # ★ NOT re-minted


@pytest.mark.parametrize("factory", [_dag, _local])
async def test_credit_with_welcome_key_is_idempotent(factory):
    """The underlying mechanism: two credits sharing the welcome-grant
    idempotency key mint once (models node-init + seed both granting)."""
    lg = await factory()
    key = "welcome-grant:w1"
    await lg.credit("w1", 100.0, TransactionType.WELCOME_GRANT, idempotency_key=key)
    await lg.credit("w1", 100.0, TransactionType.WELCOME_GRANT, idempotency_key=key)
    assert await lg.get_balance("w1") == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
