"""Sprint 1469 — DAGLedger money-audit CRITICAL: sp910 write-lock invariant on every connection write.

The sp910 fix serialized every write of the SHARED aiosqlite connection through one lock, because the
SAVEPOINT balance_check is connection-GLOBAL — a COMMIT from ANY other writer releases an in-flight
submit_transaction's savepoint mid-debit, durably persisting the debit while the withdraw 500s and
skips payout (silent no-refund fund loss). The audit (workflow wf_6ceaaeff) found the retrofit was
applied ONLY to submit_transaction / record_nonce / release_nonce; FIVE other connection-committing
methods (bump_withdraw_nonce, create_wallet, register_wallet_public_key, link_eth_address,
set_requires_user_signature) still committed WITHOUT the lock — and several are reachable from public
HTTP endpoints (/wallet/withdraw, /wallet/deposit/link, /wallet/require-signature).

Fix: each acquires self._get_write_lock(). create_wallet is called from within submit_transaction
(under the lock), so its body is extracted to _create_wallet_impl to avoid a non-reentrant-lock
deadlock; the public create_wallet locks + delegates.
"""
from __future__ import annotations

import asyncio

import pytest

from prsm.node.dag_ledger import DAGLedger
from prsm.node.local_ledger import TransactionType

pytestmark = pytest.mark.asyncio


async def _ledger():
    lg = DAGLedger(":memory:", verify_signatures=False)
    await lg.initialize()
    return lg


async def test_credit_to_brand_new_wallet_does_not_deadlock():
    # submit_transaction holds the write_lock and creates the (new) to_wallet — the reentrancy-safe
    # path must NOT re-acquire the non-reentrant lock (that would deadlock/hang).
    lg = await _ledger()
    await asyncio.wait_for(lg.credit("brand-new-wallet", 10.0, TransactionType.REWARD), timeout=5)
    assert await lg.get_balance("brand-new-wallet") == 10.0


async def test_transfer_creating_both_wallets_does_not_deadlock():
    lg = await _ledger()
    await lg.credit("src", 50.0, TransactionType.REWARD)
    # transfer to a brand-new dst → submit_transaction creates dst under the lock.
    await asyncio.wait_for(lg.transfer("src", "dst-new", 20.0), timeout=5)
    assert await lg.get_balance("dst-new") == 20.0
    assert await lg.get_balance("src") == 30.0


class _SpyLock:
    """Records each `async with` acquisition, delegating to a real asyncio.Lock so behavior + mutual
    exclusion are preserved. Deterministic (no timing) proof that a method takes the write lock."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.acquisitions = 0

    async def __aenter__(self):
        self.acquisitions += 1
        await self._lock.acquire()
        return self

    async def __aexit__(self, *exc):
        self._lock.release()

    async def acquire(self):
        self.acquisitions += 1
        return await self._lock.acquire()

    def release(self):
        return self._lock.release()

    def locked(self):
        return self._lock.locked()


@pytest.mark.parametrize("name,call", [
    ("bump_withdraw_nonce", lambda lg: lg.bump_withdraw_nonce("w")),
    ("create_wallet", lambda lg: lg.create_wallet("fresh-wallet", "fresh")),
    ("register_wallet_public_key", lambda lg: lg.register_wallet_public_key("w", "pubkey-hex")),
    ("link_eth_address", lambda lg: lg.link_eth_address("w", "0x" + "ab" * 20)),
    ("set_requires_user_signature", lambda lg: lg.set_requires_user_signature("w", True)),
])
async def test_connection_write_acquires_write_lock(name, call):
    # Each of these methods commits the SHARED connection; per the sp910 invariant it MUST hold the
    # write_lock so a foreign commit can't release a concurrent debit's balance_check savepoint. Prove
    # deterministically (via a spy lock) that the method acquires it. (Pre-fix: acquisitions == 0.)
    lg = await _ledger()
    await lg.create_wallet("w", "w")                       # pre-create (its own acquisition, then reset)
    spy = _SpyLock()
    lg._write_lock = spy                                   # _get_write_lock() returns self._write_lock
    await call(lg)
    assert spy.acquisitions >= 1, (
        f"{name} committed the shared connection WITHOUT acquiring the write_lock — sp910 invariant "
        f"broken (a concurrent debit's savepoint can be destroyed → silent withdraw fund loss)"
    )


async def test_the_five_methods_still_function_correctly():
    # The lock wrapping must not change behavior.
    lg = await _ledger()
    await lg.create_wallet("w", "w", public_key="pk0")
    old = await lg.bump_withdraw_nonce("w")
    assert await lg.bump_withdraw_nonce("w") == old + 1     # advanced
    await lg.register_wallet_public_key("w", "pk1")
    await lg.link_eth_address("w", "0x" + "cd" * 20)
    await lg.set_requires_user_signature("w", True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
