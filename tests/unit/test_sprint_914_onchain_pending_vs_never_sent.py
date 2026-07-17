"""Sprint 914 — on-chain transfer must distinguish PENDING from NEVER-SENT.

The money-path review predicted sp911 siblings on the paths that broadcast a
REAL Base-mainnet tx. Found two, both rooted in one primitive bug:

`OnChainFTNSLedger.transfer()` returns `None` in two situations callers can't
tell apart:
  1. the tx was NEVER broadcast (no account / send_raw_transaction raised) —
     SAFE to refund / retry;
  2. the tx WAS broadcast (we have a tx_hash) but `wait_for_transaction_receipt`
     TIMED OUT (60s on a congested Base) — the tx sits in the mempool and will
     likely confirm. Refunding or retrying here DOUBLE-PAYS.

Consequences in the live callers:
  * POST /wallet/withdraw refunds the off-chain debit on `None` → on a receipt
    timeout the user gets the refund AND the on-chain tokens (counterfeit FTNS).
  * BatchSettlementManager.flush clears the queue then, on `None`, silently
    DROPS the owed on-chain payout (no re-queue) — a payee is never paid.

Fix (sp914): the primitive returns the `pending` record (with its tx_hash) when
the broadcast succeeded but the receipt is unconfirmed, and `None` ONLY when the
tx was never broadcast. flush() then re-queues ONLY never-sent transfers (safe
retry) and never re-queues a pending (in-flight) one.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prsm.economy import ftns_onchain
from prsm.economy.ftns_onchain import OnChainFTNSLedger
from prsm.economy.batch_settlement import (
    BatchSettlementManager,
    PendingTransfer,
    SettlementMode,
)

_ADDR_FROM = "0x" + "a" * 40
_ADDR_TO = "0x" + "b" * 40
_TOKEN = "0x" + "c" * 40


# ── primitive: OnChainFTNSLedger.transfer ────────────────────────────────


class _FakeEth:
    def __init__(self, *, send_raises=False, send_exc=None, receipt=None,
                 receipt_raises=False):
        self._send_raises = send_raises
        self._send_exc = send_exc
        self._receipt = receipt
        self._receipt_raises = receipt_raises
        self.account = SimpleNamespace(
            # sp1474 — a signed tx exposes its deterministic .hash before send.
            sign_transaction=lambda tx, key: SimpleNamespace(
                raw_transaction=b"\x01\x02",
                hash=SimpleNamespace(hex=lambda: "0x" + "d" * 64),
            )
        )

    def get_transaction_count(self, addr, block):
        return 7

    def send_raw_transaction(self, raw):
        if self._send_exc is not None:
            raise self._send_exc
        if self._send_raises:
            raise RuntimeError("rpc down — tx never broadcast")
        return SimpleNamespace(hex=lambda: "0x" + "d" * 64)

    def wait_for_transaction_receipt(self, tx_hash, timeout=60):
        if self._receipt_raises:
            raise TimeoutError("timed out waiting for receipt")
        return self._receipt


def _make_ledger(monkeypatch, **eth_kwargs):
    monkeypatch.setattr(ftns_onchain, "estimate_gas_price", lambda w3, **k: 10 ** 9)
    led = OnChainFTNSLedger.__new__(OnChainFTNSLedger)
    led._is_initialized = True
    led._decimals = 18
    led._lock = asyncio.Lock()
    led._transactions = []
    led.chain_id = 8453
    led.contract_address = _TOKEN
    led._connected_address = _ADDR_FROM
    led._account = SimpleNamespace(key=b"\x01" * 32, address=_ADDR_FROM)
    led._token = SimpleNamespace(functions=SimpleNamespace(
        transfer=lambda a, w: SimpleNamespace(_encode_transaction_data=lambda: b"\x00")
    ))
    led.w3 = SimpleNamespace(eth=_FakeEth(**eth_kwargs))
    led._record_tx = AsyncMock()
    led._update_tx_status = AsyncMock()
    return led


@pytest.mark.asyncio
async def test_receipt_timeout_after_broadcast_returns_pending_not_none(monkeypatch):
    # Broadcast succeeds, receipt wait times out → the tx is in the mempool.
    led = _make_ledger(monkeypatch, receipt_raises=True)
    rec = await led.transfer(job_id="j", to_address=_ADDR_TO, amount_ftns=1.0)
    assert rec is not None, "broadcast succeeded → returning None would double-pay on retry"
    assert rec.status == "pending"
    assert rec.tx_hash and rec.tx_hash.startswith("0x")


@pytest.mark.asyncio
async def test_explicit_rejection_returns_none(monkeypatch):
    # sp1474 — an EXPLICIT node rejection (nonce too low) PROVES the tx did not
    # enter the mempool → safe to return None → refund. Only a provable rejection
    # takes this path now (see the ambiguous-failure test below).
    led = _make_ledger(monkeypatch, send_exc=ValueError("nonce too low"))
    rec = await led.transfer(job_id="j", to_address=_ADDR_TO, amount_ftns=1.0)
    assert rec is None


@pytest.mark.asyncio
async def test_ambiguous_send_failure_returns_pending_not_none(monkeypatch):
    # sp1474 — a GENERIC/transport send failure does NOT prove the tx never
    # landed (the node may have accepted it and only the response was lost).
    # Returning None here would refund a debit whose tx later mines → double-pay.
    # It must surface as "pending" so the reconciler refunds ONLY if it proves
    # the tx dead. Regression guard for the bridge-withdraw audit (wf_8d70ed5a).
    led = _make_ledger(monkeypatch, send_raises=True)
    rec = await led.transfer(job_id="j", to_address=_ADDR_TO, amount_ftns=1.0)
    assert rec is not None, "ambiguous send failure must NOT be a clean None (double-pay)"
    assert rec.status == "pending"
    assert rec.tx_hash and rec.tx_hash.startswith("0x")
    assert rec.nonce == 7


@pytest.mark.asyncio
async def test_confirmed_receipt_is_confirmed(monkeypatch):
    led = _make_ledger(monkeypatch, receipt={"status": 1, "blockNumber": 99})
    rec = await led.transfer(job_id="j", to_address=_ADDR_TO, amount_ftns=1.0)
    assert rec is not None and rec.status == "confirmed"
    assert rec.block_number == 99


@pytest.mark.asyncio
async def test_reverted_receipt_is_rejected(monkeypatch):
    led = _make_ledger(monkeypatch, receipt={"status": 0})
    rec = await led.transfer(job_id="j", to_address=_ADDR_TO, amount_ftns=1.0)
    assert rec is not None and rec.status == "rejected"


# ── BatchSettlementManager.flush re-queue safety ─────────────────────────


class _FakeOnChain:
    """Returns a transfer outcome of the configured kind."""

    def __init__(self, kind):
        self.kind = kind   # "none" | "pending" | "confirmed" | "rejected"
        self.calls = 0

    async def transfer(self, *, job_id, to_address, amount_ftns):
        self.calls += 1
        if self.kind == "none":
            return None
        return SimpleNamespace(
            status=self.kind, tx_hash="0x" + "f" * 64, block_number=1,
        )


def _mgr(fake):
    return BatchSettlementManager(
        ftns_ledger=fake, node_id="n1", connected_address=_ADDR_FROM,
        mode=SettlementMode.MANUAL,
    )


def _queue_one(mgr, amount=5.0):
    mgr._queue.append(PendingTransfer(
        tx_id="t1", from_wallet=_ADDR_FROM, to_wallet=_ADDR_TO,
        amount=amount, job_id="j1",
    ))


@pytest.mark.asyncio
async def test_never_sent_transfer_is_requeued():
    fake = _FakeOnChain("none")
    mgr = _mgr(fake)
    _queue_one(mgr, amount=5.0)
    await mgr.flush()
    assert fake.calls == 1
    # The owed payout must NOT be silently dropped — re-queued for retry.
    assert len(mgr._queue) == 1
    assert mgr._queue[0].to_wallet == _ADDR_TO
    assert mgr._queue[0].amount == 5.0


@pytest.mark.asyncio
async def test_pending_transfer_is_not_requeued():
    # In-flight (broadcast OK, unconfirmed) — a retry would DOUBLE-PAY.
    fake = _FakeOnChain("pending")
    mgr = _mgr(fake)
    _queue_one(mgr)
    await mgr.flush()
    assert mgr._queue == []


@pytest.mark.asyncio
async def test_confirmed_transfer_is_cleared():
    fake = _FakeOnChain("confirmed")
    mgr = _mgr(fake)
    _queue_one(mgr)
    res = await mgr.flush()
    assert mgr._queue == []
    assert res.tx_hashes == ["0x" + "f" * 64]


@pytest.mark.asyncio
async def test_rejected_transfer_is_not_requeued():
    # On-chain revert is deterministic — re-queuing would loop forever.
    fake = _FakeOnChain("rejected")
    mgr = _mgr(fake)
    _queue_one(mgr)
    await mgr.flush()
    assert mgr._queue == []
