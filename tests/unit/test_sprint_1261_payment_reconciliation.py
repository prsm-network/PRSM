"""Sprint 1261 — server-side payment reconciliation before crediting FTNS.

A signature-verified webhook (sp1250 Stripe / sp1260 PayPal) proves the event is authentic
but NOT that funds were captured at the expected amount: PayPal CHECKOUT.ORDER.APPROVED is
buyer approval, not capture; an amount could differ from what we recorded; a later
refund/void could land. So before minting FTNS, _reconcile_payment re-fetches the
authoritative payment from the provider (reusing fiat_gateway.get_payment_status) and
requires status == COMPLETED with a matching amount + currency.

Opt-in via PRSM_PAYMENT_RECONCILIATION. ENABLED → any mismatch/error fails closed (no
credit). DISABLED (default) → skip with a warning, preserving the prior behavior. The
reconciliation runs inside _process_crypto_conversion, the single credit chokepoint shared
by both providers.
"""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

import prsm.economy.payments.payment_processor as pp_module
from prsm.economy.payments.payment_processor import (
    PaymentProcessor,
    _provider_name_for_payment_method,
)
from prsm.economy.payments.payment_models import FiatCurrency, PaymentStatus


class _Txn:
    def __init__(self, *, amount="5", currency="USD", method="paypal"):
        self.transaction_id = uuid.uuid4()
        self.crypto_currency = "FTNS"
        self.crypto_amount = Decimal("10")
        self.user_id = "user-1"
        self.payment_method = method
        self.fiat_amount = Decimal(amount)
        self.fiat_currency = currency
        self.additional_data = {}


class _Resp:
    """A stand-in for fiat_gateway PaymentResponse (only the fields reconciliation reads)."""
    def __init__(self, *, success=True, status=PaymentStatus.COMPLETED,
                 fiat_amount=Decimal("5"), fiat_currency=FiatCurrency.USD):
        self.success = success
        self.status = status
        self.fiat_amount = fiat_amount
        self.fiat_currency = fiat_currency


class _FakeGateway:
    def __init__(self, resp=None, *, raises=False):
        self._resp = resp
        self._raises = raises
        self.calls = []

    async def get_payment_status(self, transaction_id, provider_name=None):
        self.calls.append((transaction_id, provider_name))
        if self._raises:
            raise RuntimeError("provider down")
        return self._resp


def _proc(monkeypatch, *, gateway=None):
    proc = PaymentProcessor()
    proc.token_distribution_enabled = True
    if gateway is not None:
        proc.fiat_gateway = gateway
    return proc


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("PRSM_PAYMENT_RECONCILIATION", raising=False)


# ── provider mapping ───────────────────────────────────────────────────────────

def test_provider_name_mapping():
    assert _provider_name_for_payment_method("paypal") == "paypal"
    assert _provider_name_for_payment_method("credit_card") == "stripe"
    assert _provider_name_for_payment_method("debit_card") == "stripe"
    assert _provider_name_for_payment_method("STRIPE") == "stripe"
    assert _provider_name_for_payment_method("unknown") == "unknown"  # passthrough, not silent wrong-pick


# ── disabled (default) skips, preserving prior behavior ──────────────────────────

@pytest.mark.asyncio
async def test_disabled_skips_and_returns_true(monkeypatch):
    gw = _FakeGateway(_Resp())
    proc = _proc(monkeypatch, gateway=gw)
    assert await proc._reconcile_payment(_Txn()) is True
    assert gw.calls == []  # disabled → never even queries the provider


# ── enabled: pass / fail-closed branches ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_enabled_completed_matching_passes(monkeypatch):
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    gw = _FakeGateway(_Resp(status=PaymentStatus.COMPLETED,
                            fiat_amount=Decimal("5.00"), fiat_currency=FiatCurrency.USD))
    proc = _proc(monkeypatch, gateway=gw)
    txn = _Txn(amount="5", currency="USD", method="paypal")
    assert await proc._reconcile_payment(txn) is True
    assert len(gw.calls) == 1
    assert gw.calls[0][0] == str(txn.transaction_id)   # queried by the provider's tx id
    assert gw.calls[0][1] == "paypal"                   # provider resolved from payment_method


@pytest.mark.asyncio
async def test_enabled_non_completed_status_fails_closed(monkeypatch):
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    # PayPal APPROVED maps to PROCESSING/PENDING, not COMPLETED → must NOT credit.
    gw = _FakeGateway(_Resp(status=PaymentStatus.PROCESSING))
    proc = _proc(monkeypatch, gateway=gw)
    assert await proc._reconcile_payment(_Txn()) is False


@pytest.mark.asyncio
async def test_enabled_amount_mismatch_fails_closed(monkeypatch):
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    gw = _FakeGateway(_Resp(fiat_amount=Decimal("4.99")))   # provider says 4.99, we expected 5
    proc = _proc(monkeypatch, gateway=gw)
    assert await proc._reconcile_payment(_Txn(amount="5")) is False


@pytest.mark.asyncio
async def test_enabled_currency_mismatch_fails_closed(monkeypatch):
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    gw = _FakeGateway(_Resp(fiat_currency=FiatCurrency.EUR))
    proc = _proc(monkeypatch, gateway=gw)
    assert await proc._reconcile_payment(_Txn(currency="USD")) is False


@pytest.mark.asyncio
async def test_enabled_provider_unsuccessful_fails_closed(monkeypatch):
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    gw = _FakeGateway(_Resp(success=False, status=PaymentStatus.FAILED))
    proc = _proc(monkeypatch, gateway=gw)
    assert await proc._reconcile_payment(_Txn()) is False


@pytest.mark.asyncio
async def test_enabled_gateway_exception_fails_closed(monkeypatch):
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    proc = _proc(monkeypatch, gateway=_FakeGateway(raises=True))
    assert await proc._reconcile_payment(_Txn()) is False


@pytest.mark.asyncio
async def test_enabled_no_gateway_fails_closed(monkeypatch):
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    proc = PaymentProcessor()             # no fiat_gateway wired
    proc.fiat_gateway = None
    assert await proc._reconcile_payment(_Txn()) is False


@pytest.mark.asyncio
async def test_amount_decimal_equality_tolerant(monkeypatch):
    # 5 vs 5.00 vs 5.000 are numerically equal → must pass.
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    gw = _FakeGateway(_Resp(fiat_amount=Decimal("5.000")))
    proc = _proc(monkeypatch, gateway=gw)
    assert await proc._reconcile_payment(_Txn(amount="5")) is True


# ── integration: the credit chokepoint (_process_crypto_conversion) honors the gate ──

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


class _FakeSessionCtx:
    def __init__(self, s):
        self._s = s

    async def __aenter__(self):
        return self._s

    async def __aexit__(self, *a):
        return False


class _FakeDBManager:
    def __init__(self, s):
        self._s = s

    def session(self):
        return _FakeSessionCtx(self._s)


class _FtnsService:
    def __init__(self):
        self.calls = 0

    async def transfer_tokens(self, **kwargs):
        self.calls += 1
        return True


def _wire_credit_path(monkeypatch, proc, txn):
    proc.ftns_service = _FtnsService()
    monkeypatch.setattr(pp_module, "db_manager", _FakeDBManager(_FakeSession(txn)))
    monkeypatch.setattr(pp_module.audit_logger, "log_event", lambda *a, **k: None, raising=False)


@pytest.mark.asyncio
async def test_conversion_blocks_credit_when_reconciliation_fails(monkeypatch):
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    txn = _Txn(amount="5")
    gw = _FakeGateway(_Resp(fiat_amount=Decimal("999")))   # amount mismatch → reconciliation fails
    proc = _proc(monkeypatch, gateway=gw)
    _wire_credit_path(monkeypatch, proc, txn)

    await proc._process_crypto_conversion(txn.transaction_id)
    assert proc.ftns_service.calls == 0                         # credit BLOCKED
    assert "tokens_distributed" not in txn.additional_data      # not marked → can retry


@pytest.mark.asyncio
async def test_conversion_credits_when_reconciliation_passes(monkeypatch):
    monkeypatch.setenv("PRSM_PAYMENT_RECONCILIATION", "1")
    txn = _Txn(amount="5", currency="USD")
    gw = _FakeGateway(_Resp(status=PaymentStatus.COMPLETED,
                            fiat_amount=Decimal("5"), fiat_currency=FiatCurrency.USD))
    proc = _proc(monkeypatch, gateway=gw)
    _wire_credit_path(monkeypatch, proc, txn)

    await proc._process_crypto_conversion(txn.transaction_id)
    assert proc.ftns_service.calls == 1                         # credited
    assert txn.additional_data.get("tokens_distributed") is True


@pytest.mark.asyncio
async def test_conversion_credits_when_reconciliation_disabled(monkeypatch):
    # default (disabled) preserves prior behavior: credit on the verified webhook alone.
    txn = _Txn(amount="5")
    proc = _proc(monkeypatch)                                   # no gateway needed when disabled
    _wire_credit_path(monkeypatch, proc, txn)

    await proc._process_crypto_conversion(txn.transaction_id)
    assert proc.ftns_service.calls == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
