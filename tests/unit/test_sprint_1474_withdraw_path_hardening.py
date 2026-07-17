"""Sprint 1474 — bridge-WITHDRAW path hardening (audit wf_8d70ed5a).

The bridge-withdraw adversarial audit (off-chain debit → on-chain ERC-20 transfer)
surfaced five real defects across the endpoint, the on-chain transfer, and the
pending-withdraw reconciler. Each test below proves one fix and FAILS on the pre-fix
code (verified RED):

  #1 Nonce check→bump TOCTOU — get_next_withdraw_nonce (read) + bump_withdraw_nonce
     (write) were separate awaits, so two concurrent replays of ONE signature both
     cleared the sp556 gate → one signature drove N on-chain payouts. Fixed with an
     ATOMIC compare_and_bump_withdraw_nonce on both ledger backends.
  #2 Reconciler _is_dropped refunded a CONFIRMED withdraw — it returned
     confirmed_nonce > intent.nonce with NO receipt re-poll, so a tx that confirmed in
     the gap after the earlier receipt read was mis-classified dropped → refund a
     landed tx → double-pay. Fixed with a receipt re-poll AFTER the nonce gate.
  #3 transfer() set tx_hash only AFTER send → a send whose response was lost returned
     None → refund → tx mines → silent double-pay. Fixed: pin the deterministic hash
     BEFORE send + classify the send failure (only a provable rejection → None).
  #4 A failed inline refund only LOGGED → the debit was stranded with no reconciler
     linkage. Fixed: record a durable refund_owed intent the reconciler retries.
  #5 The signed-payload wei used a hardcoded 1e18 vs the transfer's 10**decimals.
     Fixed: derive the signed wei from the ledger's decimals (single source).
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prsm.node.dag_ledger import DAGLedger
from prsm.node.local_ledger import LocalLedger
from prsm.node.pending_withdraw_reconciler import (
    PendingWithdrawReconciler,
    PendingWithdrawStore,
    WithdrawIntent,
    reconcile_pending_withdraws,
)

pytestmark = pytest.mark.asyncio


# ───────────────────────── #1 atomic nonce compare-and-bump ─────────────────

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
async def test_compare_and_bump_matches_and_advances(factory):
    lg = await factory()
    assert await lg.get_next_withdraw_nonce("w1") == 0
    consumed = await lg.compare_and_bump_withdraw_nonce("w1", 0)
    assert consumed == 0
    assert await lg.get_next_withdraw_nonce("w1") == 1


@pytest.mark.parametrize("factory", [_dag, _local])
async def test_compare_and_bump_wrong_expected_returns_none_no_advance(factory):
    lg = await factory()
    # expected != current → None, and the nonce must NOT advance.
    assert await lg.compare_and_bump_withdraw_nonce("w1", 5) is None
    assert await lg.get_next_withdraw_nonce("w1") == 0


@pytest.mark.parametrize("factory", [_dag, _local])
async def test_compare_and_bump_concurrent_same_nonce_only_one_wins(factory):
    """★ The sp556 single-use guarantee: N concurrent callers all presenting the
    SAME nonce (one replayed signature) → EXACTLY ONE consumes it; the rest get
    None. Pre-fix (separate read + unconditional bump) all N would proceed."""
    lg = await factory()
    results = await asyncio.gather(
        *[lg.compare_and_bump_withdraw_nonce("w1", 0) for _ in range(8)]
    )
    winners = [r for r in results if r is not None]
    assert winners == [0], f"exactly one caller may consume nonce 0, got {results}"
    assert await lg.get_next_withdraw_nonce("w1") == 1  # advanced exactly once


# ───────────────────── #2 reconciler is_dropped receipt re-poll ─────────────

def _reconciler_with_chain(*, confirmed_nonce, receipt):
    """Build a reconciler whose chain reads return the given nonce + receipt.
    `receipt` may be a dict (present) or an Exception instance (raised → absent)."""
    def _get_receipt(h):
        if isinstance(receipt, Exception):
            raise receipt
        return receipt

    eth = SimpleNamespace(
        get_transaction_count=lambda addr, block: confirmed_nonce,
        get_transaction_receipt=_get_receipt,
    )
    ftns = SimpleNamespace(w3=SimpleNamespace(eth=eth),
                           _connected_address="0x" + "ab" * 20)
    rec = PendingWithdrawReconciler.__new__(PendingWithdrawReconciler)
    rec._ftns_ledger = ftns
    rec._local_ledger = SimpleNamespace()
    return rec


def _intent(nonce=42):
    return WithdrawIntent(job_id="j1", wallet_id="w1", amount=50.0,
                          to_addr="0x" + "cc" * 20, tx_hash="0x" + "de" * 32,
                          nonce=nonce)


async def test_is_dropped_false_when_tx_confirmed_in_read_gap():
    """★ kills withdraw-is-dropped-receipt-repoll: the nonce advanced past the tx,
    but a receipt now EXISTS (the tx confirmed in the gap after the earlier receipt
    read). is_dropped MUST return False so the reconciler does not refund a landed
    tx (double-pay). Pre-fix it returned True purely on the nonce advance."""
    rec = _reconciler_with_chain(confirmed_nonce=43, receipt={"status": 1, "blockNumber": 9})
    assert await rec._is_dropped(_intent(nonce=42)) is False


async def test_is_dropped_true_when_receipt_absent_and_nonce_advanced():
    """A DIFFERENT tx took the nonce slot → our tx_hash has a permanently-null
    receipt → provably dead → refund is safe (True)."""
    from web3.exceptions import TransactionNotFound
    rec = _reconciler_with_chain(confirmed_nonce=43, receipt=TransactionNotFound("no"))
    assert await rec._is_dropped(_intent(nonce=42)) is True


async def test_is_dropped_false_when_nonce_not_advanced():
    """Nonce not advanced → the tx can still land → never dropped (no re-poll needed)."""
    rec = _reconciler_with_chain(confirmed_nonce=42, receipt={"status": 1})
    assert await rec._is_dropped(_intent(nonce=42)) is False


# ─────────────────── #3 transfer pins tx_hash before send ───────────────────

from prsm.economy import ftns_onchain  # noqa: E402
from prsm.economy.ftns_onchain import OnChainFTNSLedger  # noqa: E402


class _Eth:
    def __init__(self, *, send_exc=None):
        self._send_exc = send_exc
        self.account = SimpleNamespace(
            sign_transaction=lambda tx, key: SimpleNamespace(
                raw_transaction=b"\x01",
                hash=SimpleNamespace(hex=lambda: "0x" + "d" * 64)))

    def get_transaction_count(self, addr, block):
        return 7

    def send_raw_transaction(self, raw):
        raise self._send_exc


def _onchain(send_exc):
    ftns_onchain.estimate_gas_price = lambda w3, **k: 10 ** 9
    led = OnChainFTNSLedger.__new__(OnChainFTNSLedger)
    led._is_initialized = True
    led._decimals = 18
    led._lock = asyncio.Lock()
    led._transactions = []
    led.chain_id = 8453
    led.contract_address = "0x" + "11" * 20
    led._connected_address = "0x" + "22" * 20
    led._account = SimpleNamespace(key=b"\x01" * 32)
    led._token = SimpleNamespace(functions=SimpleNamespace(
        transfer=lambda a, w: SimpleNamespace(_encode_transaction_data=lambda: b"\x00")))
    led.w3 = SimpleNamespace(eth=_Eth(send_exc=send_exc))
    led._record_tx = AsyncMock()
    led._update_tx_status = AsyncMock()
    return led


async def test_lost_response_after_send_is_pending_with_deterministic_hash():
    """★ #3 — an ambiguous/transport send failure is surfaced as PENDING (not None)
    with the deterministic tx_hash + nonce pinned, so a debit whose tx actually
    landed is never refunded → no silent double-pay. Pre-fix (tx_hash set only after
    send) this returned None → refund → double-pay."""
    import requests
    led = _onchain(send_exc=requests.exceptions.ConnectionError("reset"))
    rec = await led.transfer(job_id="j", to_address="0x" + "33" * 20, amount_ftns=1.0)
    assert rec is not None and rec.status == "pending"
    assert rec.tx_hash == "0x" + "d" * 64        # deterministic, set BEFORE send
    assert rec.nonce == 7


async def test_provable_rejection_after_send_returns_none():
    """A provable node rejection (nonce too low) means the tx did NOT enter the
    mempool → None (safe immediate refund)."""
    led = _onchain(send_exc=ValueError("nonce too low"))
    rec = await led.transfer(job_id="j", to_address="0x" + "33" * 20, amount_ftns=1.0)
    assert rec is None


# ─────────────────── #4 refund-owed reconciler linkage ──────────────────────

async def test_refund_owed_intent_is_refunded_by_reconciler():
    """★ #4 — a debit whose broadcast failed AND whose inline refund failed is
    recorded as refund_owed; the reconciler retries the idempotent credit and
    resolves it, instead of stranding it on a log line."""
    store = PendingWithdrawStore(persist_dir=None)
    store.record_refund_owed(job_id="jx", wallet_id="w1", amount=7.0,
                             to_addr="0x" + "cc" * 20)
    credited = []

    async def _refund(intent):
        credited.append((intent.wallet_id, intent.amount))
        return True

    out = await reconcile_pending_withdraws(
        store,
        get_receipt_status=AsyncMock(),   # must NOT be consulted for refund_owed
        refund=_refund,
    )
    assert credited == [("w1", 7.0)]
    assert out["refunded"] == 1
    assert store.unresolved() == []       # resolved


async def test_refund_owed_retries_when_refund_fails_then_succeeds():
    store = PendingWithdrawStore(persist_dir=None)
    store.record_refund_owed(job_id="jx", wallet_id="w1", amount=7.0,
                             to_addr="0x" + "cc" * 20)
    calls = {"n": 0}

    async def _refund(intent):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db busy")
        return True

    out1 = await reconcile_pending_withdraws(
        store, get_receipt_status=AsyncMock(), refund=_refund)
    assert out1["refunded"] == 0 and len(store.unresolved()) == 1  # left for retry
    out2 = await reconcile_pending_withdraws(
        store, get_receipt_status=AsyncMock(), refund=_refund)
    assert out2["refunded"] == 1 and store.unresolved() == []


# ─────────────────── #5 signed wei from ledger decimals ─────────────────────

from eth_account import Account            # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from unittest.mock import MagicMock        # noqa: E402

_USER_PK = "0x" + "ab" * 32


class _DecimalsStub:
    """Withdraw-handler stand-in with a configurable on-chain `_decimals`."""
    class _Debit:
        def __init__(self):
            self.tx_id = "d0"

    class _Broadcast:
        status, tx_hash, block_number = "confirmed", "0xfeed", 1

    def __init__(self, *, wallet_id, linked, decimals):
        self.wallet_id = wallet_id
        self._linked = linked
        self._decimals = decimals   # ← the field the endpoint must read
        self._next_nonce = 0
        self._balance = 100.0
        self.credits = []

    async def eth_address_for_wallet(self, w):
        return self._linked if w == self.wallet_id else None

    async def get_requires_user_signature(self, w):
        return True

    async def get_next_withdraw_nonce(self, w):
        return self._next_nonce

    async def compare_and_bump_withdraw_nonce(self, w, expected):
        if self._next_nonce != int(expected):
            return None
        old = self._next_nonce
        self._next_nonce += 1
        return old

    async def get_balance(self, w):
        return self._balance

    async def debit(self, *, wallet_id, amount, tx_type, description):
        self._balance -= amount
        return _DecimalsStub._Debit()

    async def credit(self, **k):
        self.credits.append(k)

    async def transfer(self, *, job_id, to_address, amount_ftns):
        return _DecimalsStub._Broadcast()


def _signed_body_for_decimals(*, amount, to_addr, decimals, wallet_id="w1"):
    from prsm.economy.withdraw_signature import sign_withdraw_payload
    expiry = int(time.time()) + 300
    payload = {
        "wallet_id": wallet_id,
        "amount_ftns_wei": int(amount * (10 ** decimals)),   # signer uses REAL decimals
        "to_eth_address": to_addr,
        "nonce": 0,
        "expiry_unix": expiry,
    }
    sig = sign_withdraw_payload(payload, _USER_PK)
    return {
        "amount_ftns": amount, "wallet_id": wallet_id, "to_eth_address": to_addr,
        "signature": "0x" + sig.hex(), "nonce": 0, "expiry_unix": expiry,
    }


def test_withdraw_verifies_signature_against_ledger_decimals():
    """★ #5 (endpoint) — a wallet on a 6-decimal token signs its withdraw over
    int(amount * 10**6). The endpoint must derive the signed-payload wei from the
    ledger's `_decimals`, so the recovered signer matches → 200. With the pre-fix
    hardcoded 1e18 the endpoint would compute a different wei, the signature would
    recover to a different digest → signer mismatch → 401."""
    from prsm.node.api import create_api_app
    acct = Account.from_key(_USER_PK)
    ledger = _DecimalsStub(wallet_id="w1", linked=acct.address.lower(), decimals=6)
    node = MagicMock()
    node.identity = MagicMock(node_id="w1")
    node.ledger = ledger
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    resp = client.post("/wallet/withdraw", json=_signed_body_for_decimals(
        amount=2.5, to_addr=acct.address, decimals=6))
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "confirmed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
