"""Sprint 1472 — DAGLedger.credit(idempotency_key=...) so bridge deposits work on the default backend.

Bridge-deposit audit (workflow wf_1160e260) HIGH: InboundMonitor._credit_deposit calls
`self._local_ledger.credit(..., idempotency_key="bridge-deposit:{recipient}:{tx_hash}")`
(ftns_onchain.py:805), and node.py:5627 wires _local_ledger = self.ledger = the DEFAULT DAGLedger.
But DAGLedger.credit had NO idempotency_key parameter (only LocalLedger.credit did, sp1101), so every
bridge deposit on the shipped default (config.ledger_type="dag") raised
`TypeError: credit() got an unexpected keyword argument 'idempotency_key'` → the tick swallowed it and
advanced past the block → the deposit was PERMANENTLY stranded on-chain (depositor fund-loss).

Fix: add idempotency_key to DAGLedger.credit, plumbed as a deterministic tx_id + has_transaction
pre-check through submit_transaction (mirroring LocalLedger's exactly-once), so the deposit credit
succeeds AND a restart catch-up re-presenting the same Transfer is a no-op (no double-credit).
"""
from __future__ import annotations

import math

import pytest

from prsm.node.dag_ledger import DAGLedger
from prsm.node.local_ledger import TransactionType

pytestmark = pytest.mark.asyncio


async def _ledger():
    lg = DAGLedger(":memory:", verify_signatures=False)
    await lg.initialize()
    return lg


async def test_credit_accepts_idempotency_key_and_credits():
    lg = await _ledger()
    # Must NOT raise TypeError (the bug), and must credit.
    await lg.credit("wallet-a", 10.0, TransactionType.BRIDGE_DEPOSIT, idempotency_key="dep-1")
    assert await lg.get_balance("wallet-a") == 10.0


async def test_idempotent_replay_applies_once():
    lg = await _ledger()
    key = "bridge-deposit:0xrecip:0xtxhash"
    await lg.credit("wallet-a", 25.0, TransactionType.BRIDGE_DEPOSIT, idempotency_key=key)
    # A restart catch-up re-presents the SAME Transfer → same key → no-op.
    await lg.credit("wallet-a", 25.0, TransactionType.BRIDGE_DEPOSIT, idempotency_key=key)
    bal = await lg.get_balance("wallet-a")
    assert bal == 25.0 and math.isfinite(bal)             # credited exactly once, not 50


async def test_distinct_keys_both_apply():
    lg = await _ledger()
    await lg.credit("w", 5.0, TransactionType.BRIDGE_DEPOSIT, idempotency_key="tx-A")
    await lg.credit("w", 7.0, TransactionType.BRIDGE_DEPOSIT, idempotency_key="tx-B")
    assert await lg.get_balance("w") == 12.0


async def test_credit_without_key_is_unchanged_default_behavior():
    lg = await _ledger()
    await lg.credit("w", 3.0, TransactionType.REWARD)     # no key → fresh uuid tx, credits normally
    await lg.credit("w", 3.0, TransactionType.REWARD)     # distinct uuid → both apply
    assert await lg.get_balance("w") == 6.0


async def test_bridge_deposit_style_credit_end_to_end_on_dagledger():
    # Mirror the exact _credit_deposit call shape (from_wallet=None system credit, BRIDGE_DEPOSIT,
    # the recipient:tx_hash idempotency key) against the DEFAULT ledger.
    lg = await _ledger()
    recipient, tx_hash = "0xoperator", "0xdeadbeef"
    key = f"bridge-deposit:{recipient}:{tx_hash}"
    await lg.credit(
        wallet_id="depositor-wallet", amount=100.0,
        tx_type=TransactionType.BRIDGE_DEPOSIT,
        description=f"bridge deposit tx={tx_hash}", idempotency_key=key)
    assert await lg.get_balance("depositor-wallet") == 100.0
    # the restart catch-up scan re-presenting it must not double-credit
    await lg.credit(
        wallet_id="depositor-wallet", amount=100.0,
        tx_type=TransactionType.BRIDGE_DEPOSIT,
        description=f"bridge deposit tx={tx_hash}", idempotency_key=key)
    assert await lg.get_balance("depositor-wallet") == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
