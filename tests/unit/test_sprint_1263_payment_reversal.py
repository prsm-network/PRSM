"""Sprint 1263 — handle payment reversals (refunds / chargebacks / disputes).

The webhook processors only handled SUCCESS events, so a refund or chargeback that lands
AFTER FTNS was credited went entirely undetected — the user keeps the FTNS and gets their
fiat back (a free-money / fraud vector). _handle_payment_reversal now marks the transaction
REFUNDED, flags it (which hard-stops any future credit), and logs CRITICAL when FTNS was
already distributed so an operator can run a manual clawback (auto-clawback is a policy
decision and the tokens may already be spent).

Reversal events are wired into both providers. Stripe maps reliably (the PaymentIntent id —
our transaction_id — is on the charge/dispute object). PayPal capture refunds/reversals map
via resource.supplementary_data.related_ids.order_id; events that don't carry the order id
(e.g. disputes) are logged for manual review rather than silently dropped.
"""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

import prsm.economy.payments.payment_processor as pp_module
from prsm.economy.payments.payment_processor import (
    PaymentProcessor,
    _extract_paypal_reversal_order_id,
)
from prsm.economy.payments.payment_models import PaymentStatus


class _Txn:
    def __init__(self, *, credited=False):
        self.transaction_id = uuid.uuid4()
        self.crypto_currency = "FTNS"
        self.crypto_amount = Decimal("10")
        self.user_id = "user-1"
        self.payment_method = "paypal"
        self.fiat_amount = Decimal("5")
        self.fiat_currency = "USD"
        self.status = PaymentStatus.COMPLETED.value
        self.completed_at = None
        self.additional_data = {"tokens_distributed": True} if credited else {}


class _FakeSession:
    def __init__(self, txn):
        self._txn = txn
        self.commits = 0

    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._txn

    def commit(self):
        self.commits += 1


class _Ctx:
    def __init__(self, s):
        self._s = s

    async def __aenter__(self):
        return self._s

    async def __aexit__(self, *a):
        return False


class _DB:
    def __init__(self, s):
        self._s = s

    def session(self):
        return _Ctx(self._s)


class _Ftns:
    def __init__(self):
        self.calls = 0

    async def transfer_tokens(self, **k):
        self.calls += 1
        return True


def _proc(monkeypatch, txn):
    proc = PaymentProcessor()
    proc.token_distribution_enabled = True
    proc.ftns_service = _Ftns()
    monkeypatch.setattr(pp_module, "db_manager", _DB(_FakeSession(txn)))
    monkeypatch.setattr(pp_module.audit_logger, "log_event", lambda *a, **k: None, raising=False)
    return proc


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# ── order-id extraction ─────────────────────────────────────────────────────────

def test_paypal_reversal_order_id_extracted():
    payload = {"resource": {"supplementary_data": {"related_ids": {"order_id": "ORDER-9"}}}}
    assert _extract_paypal_reversal_order_id(payload) == "ORDER-9"


def test_paypal_reversal_order_id_none_when_absent():
    # a dispute payload with no order_id → None (caller logs for manual review)
    assert _extract_paypal_reversal_order_id({"resource": {"disputed_transactions": [{}]}}) is None
    assert _extract_paypal_reversal_order_id({}) is None


# ── core reversal handler ─────────────────────────────────────────────────────────

def test_reversal_marks_refunded_and_flags(monkeypatch):
    txn = _Txn(credited=False)
    proc = _proc(monkeypatch, txn)
    _run(proc._handle_payment_reversal(txn.transaction_id,
                                       event_type="charge.refunded", reason="requested"))
    assert txn.status == PaymentStatus.REFUNDED.value
    assert txn.additional_data.get("reversed", {}).get("event_type") == "charge.refunded"


def test_reversal_after_credit_logs_critical(monkeypatch):
    txn = _Txn(credited=True)   # FTNS already distributed
    proc = _proc(monkeypatch, txn)
    crit = []
    monkeypatch.setattr(pp_module.logger, "critical",
                        lambda msg, *a, **k: crit.append((msg, k)), raising=False)
    _run(proc._handle_payment_reversal(txn.transaction_id,
                                       event_type="PAYMENT.CAPTURE.REFUNDED", reason="refund"))
    assert txn.status == PaymentStatus.REFUNDED.value
    assert crit, "expected a CRITICAL clawback-review log when FTNS was already credited"


def test_reversal_unknown_transaction_does_not_crash(monkeypatch):
    proc = _proc(monkeypatch, None)   # session.first() → None
    _run(proc._handle_payment_reversal(uuid.uuid4(), event_type="charge.refunded", reason="x"))


# ── reversed transactions can never be credited ───────────────────────────────────

def test_reversed_flag_hard_stops_credit(monkeypatch):
    txn = _Txn(credited=False)
    txn.additional_data = {"reversed": {"event_type": "charge.refunded"}}
    proc = _proc(monkeypatch, txn)
    _run(proc._process_crypto_conversion(txn.transaction_id))
    assert proc.ftns_service.calls == 0   # a reversed transaction is never credited


# ── webhook wiring ────────────────────────────────────────────────────────────────

def test_stripe_refund_event_marks_reversed(monkeypatch):
    txn = _Txn(credited=True)
    proc = _proc(monkeypatch, txn)
    payload = {"type": "charge.refunded",
               "data": {"object": {"payment_intent": str(txn.transaction_id),
                                   "reason": "requested_by_customer"}}}
    assert _run(proc._process_stripe_webhook(payload)) is True
    assert txn.status == PaymentStatus.REFUNDED.value
    assert "reversed" in txn.additional_data


def test_stripe_dispute_event_marks_reversed(monkeypatch):
    txn = _Txn(credited=True)
    proc = _proc(monkeypatch, txn)
    payload = {"type": "charge.dispute.created",
               "data": {"object": {"payment_intent": str(txn.transaction_id), "reason": "fraudulent"}}}
    assert _run(proc._process_stripe_webhook(payload)) is True
    assert txn.status == PaymentStatus.REFUNDED.value


def test_paypal_refund_event_marks_reversed(monkeypatch):
    txn = _Txn(credited=True)
    proc = _proc(monkeypatch, txn)
    payload = {"event_type": "PAYMENT.CAPTURE.REFUNDED",
               "summary": "A payment was refunded",
               "resource": {"supplementary_data": {"related_ids": {"order_id": str(txn.transaction_id)}}}}
    assert _run(proc._process_paypal_webhook(payload)) is True
    assert txn.status == PaymentStatus.REFUNDED.value


def test_paypal_reversal_without_order_id_is_handled_gracefully(monkeypatch):
    txn = _Txn(credited=True)
    proc = _proc(monkeypatch, txn)
    # a dispute event with no resolvable order_id → handler returns True, tx untouched
    payload = {"event_type": "CUSTOMER.DISPUTE.CREATED", "resource": {"disputed_transactions": [{}]}}
    assert _run(proc._process_paypal_webhook(payload)) is True
    assert txn.status == PaymentStatus.COMPLETED.value   # unchanged — couldn't map it


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
