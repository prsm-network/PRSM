"""sp1439 (withdrawal audit) — the two pending-withdraw reconciler lost-debit bugs.

An adversarial audit of the live crypto money-OUT path found no double-withdraw, no
mint-from-nothing, and no unauthorized drain (the EIP-712 nonce + the under-lock balance
re-check hold), and the fiat off-ramp is scaffold. It found exactly two lost-debit defects,
both in the pending-withdraw reconciler (the sole safety net for a debited-but-undelivered
withdraw):

  #1 A withdraw tx that is broadcast but DROPPED/evicted from the mempool (underpriced gas,
     RBF, nonce-collision) never produces a receipt, so the reconciler mapped it to "pending"
     on every tick FOREVER and never refunded — the off-chain debit was permanently lost.
     Fixed by refunding a tx that is PROVABLY dead (the escrow's confirmed nonce has advanced
     strictly past this tx's nonce → it can never mine) — deterministic, so a tx that could
     still land is never refunded (that would double-pay once it confirms).

  #2 _refund() claimed a record_nonce idempotency marker in a SEPARATE durable commit BEFORE
     crediting; a crash in the gap burned the claim while no credit landed, and on restart the
     burned claim made _refund skip the credit forever. Fixed by folding idempotency into the
     credit itself (idempotency_key), so a crash simply re-runs the exactly-once credit.

Money assertions (CLAUDE.md: never weaken to pass).
"""
from __future__ import annotations

import json

import pytest

from prsm.node.pending_withdraw_reconciler import (
    PendingWithdrawReconciler,
    PendingWithdrawStore,
    WithdrawIntent,
    reconcile_pending_withdraws,
)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


class _FakeLedger:
    """Off-chain ledger: credit is exactly-once by idempotency_key (mirrors the real
    transactions-PRIMARY-KEY dedup); record_nonce is a one-shot claim (for the crash sim)."""

    def __init__(self):
        self._bal = {}
        self._idem = set()
        self._nonces = set()

    async def credit(self, *, wallet_id, amount, tx_type, idempotency_key=None, description=""):
        if idempotency_key is not None:
            if idempotency_key in self._idem:
                return None  # exactly-once no-op on replay
            self._idem.add(idempotency_key)
        self._bal[wallet_id] = self._bal.get(wallet_id, 0.0) + float(amount)
        return object()

    async def record_nonce(self, nonce, who):
        if nonce in self._nonces:
            return False
        self._nonces.add(nonce)
        return True

    def balance(self, wallet_id):
        return self._bal.get(wallet_id, 0.0)


class _FakeEth:
    def __init__(self, confirmed):
        self._c = confirmed

    def get_transaction_count(self, addr, block):
        return self._c


class _FakeW3:
    def __init__(self, confirmed):
        self.eth = _FakeEth(confirmed)


class _FakeFtnsLedger:
    def __init__(self, confirmed=0, connected="0x" + "e" * 40):
        self.w3 = _FakeW3(confirmed)
        self._connected_address = connected


def _reconciler(local, ftns=None):
    return PendingWithdrawReconciler(
        store=PendingWithdrawStore(),
        ftns_ledger=(ftns if ftns is not None else _FakeFtnsLedger()),
        local_ledger=local,
        interval_seconds=60,
    )


def _intent(job_id="j", amount=100.0, nonce=5):
    return WithdrawIntent(job_id=job_id, wallet_id="w", amount=amount,
                          to_addr="0x" + "a" * 40, tx_hash="0x" + "b" * 64, nonce=nonce)


# ── Finding #2 — the refund is crash-safe (exactly-once via the credit) ───────


def test_refund_credits_despite_a_burned_claim_from_a_crashed_prior_run():
    """The money shot for #2: simulate a crash that durably claimed the refund nonce before
    crediting. The refund must STILL credit the wallet (the credit is now the idempotency
    point). Pre-fix, the burned claim made _refund return without ever crediting → lost."""
    led = _FakeLedger()
    _run(led.record_nonce("withdraw-refund:job-1", "reconciler"))  # the burned claim
    rec = _reconciler(led)
    assert _run(rec._refund(_intent(job_id="job-1"))) is True
    assert led.balance("w") == 100.0, "a reverted withdraw was never refunded — lost debit"


def test_refund_is_exactly_once_across_a_double_run():
    led = _FakeLedger()
    rec = _reconciler(led)
    _run(rec._refund(_intent(job_id="job-2", amount=50.0)))
    _run(rec._refund(_intent(job_id="job-2", amount=50.0)))
    assert led.balance("w") == 50.0, "double-run double-credited the refund"


# ── Finding #1 — a provably-dead dropped tx is refunded (determinism) ─────────


def test_is_dropped_true_only_when_confirmed_nonce_advanced_past_tx_nonce():
    led = _FakeLedger()
    # confirmed nonce 7 > tx nonce 5 → a different tx took the slot → this tx can never mine.
    assert _run(_reconciler(led, _FakeFtnsLedger(confirmed=7))._is_dropped(_intent(nonce=5))) is True
    # confirmed nonce == tx nonce → this tx could STILL mine → NOT dropped (never double-pay).
    assert _run(_reconciler(led, _FakeFtnsLedger(confirmed=5))._is_dropped(_intent(nonce=5))) is False
    # confirmed nonce < tx nonce → clearly still live.
    assert _run(_reconciler(led, _FakeFtnsLedger(confirmed=3))._is_dropped(_intent(nonce=5))) is False
    # legacy intent with no recorded nonce → can't be proven dead → stays pending.
    assert _run(_reconciler(led, _FakeFtnsLedger(confirmed=7))._is_dropped(_intent(nonce=None))) is False


def test_dropped_pending_tx_is_refunded_not_stranded_forever():
    store = PendingWithdrawStore()
    store.record(job_id="j", wallet_id="w", amount=100.0, to_addr="0x", tx_hash="0xabc", nonce=5)
    refunded = []

    async def get_status(h):
        return "pending"

    async def is_dropped(i):
        return True

    async def refund(i):
        refunded.append(i.job_id)
        return True

    out = _run(reconcile_pending_withdraws(
        store, get_receipt_status=get_status, refund=refund, is_dropped=is_dropped))
    assert refunded == ["j"]
    assert out["dropped_refunded"] == 1 and out["still_pending"] == 0
    resolved = store.all()[0]
    assert resolved.resolved and resolved.outcome == "refunded_dropped"


def test_live_pending_tx_is_left_pending_and_never_refunded():
    """A tx that could still mine must NEVER be refunded — that would double-pay when it lands."""
    store = PendingWithdrawStore()
    store.record(job_id="j", wallet_id="w", amount=100.0, to_addr="0x", tx_hash="0xabc", nonce=5)

    async def get_status(h):
        return "pending"

    async def is_dropped(i):
        return False

    async def refund(i):
        raise AssertionError("refunded a LIVE pending tx — double-pay risk")

    out = _run(reconcile_pending_withdraws(
        store, get_receipt_status=get_status, refund=refund, is_dropped=is_dropped))
    assert out["still_pending"] == 1 and out["dropped_refunded"] == 0
    assert not store.all()[0].resolved


def test_reconcile_without_is_dropped_is_backward_compatible():
    """Omitting is_dropped preserves the pre-sp1439 behavior: a pending tx stays pending."""
    store = PendingWithdrawStore()
    store.record(job_id="j", wallet_id="w", amount=1.0, to_addr="0x", tx_hash="0xabc", nonce=5)

    async def get_status(h):
        return "pending"

    async def refund(i):
        raise AssertionError("no refund without a dropped signal")

    out = _run(reconcile_pending_withdraws(store, get_receipt_status=get_status, refund=refund))
    assert out["still_pending"] == 1


# ── persistence: the nonce round-trips + legacy files still load ──────────────


def test_nonce_round_trips_through_store_persistence(tmp_path):
    s1 = PendingWithdrawStore(persist_dir=str(tmp_path))
    s1.record(job_id="j", wallet_id="w", amount=1.0, to_addr="0x", tx_hash="0xabc", nonce=42)
    s2 = PendingWithdrawStore(persist_dir=str(tmp_path))  # "restart"
    assert s2.all()[0].nonce == 42


def test_pre_sp1439_intent_without_nonce_key_still_loads(tmp_path):
    p = tmp_path / "pending_withdraws.json"
    p.write_text(json.dumps({"intents": [{
        "job_id": "j", "wallet_id": "w", "amount": 1.0, "to_addr": "0x",
        "tx_hash": "0xabc", "recorded_at": 1.0, "resolved": False, "outcome": None,
    }]}))
    s = PendingWithdrawStore(persist_dir=str(tmp_path))
    assert len(s.all()) == 1 and s.all()[0].nonce is None
