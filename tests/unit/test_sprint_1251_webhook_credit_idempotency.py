"""Sprint 1251 — webhook crediting is idempotent (no double-credit on replay/retry).

Defense-in-depth follow-on to sp1250. Even with signature verification, a VALID
signed webhook can be re-delivered (Stripe/PayPal retry on a slow/failed ack) or
replayed within the window. _process_crypto_conversion called transfer_tokens with NO
check whether tokens were already distributed → double-credit. This guard credits at
most once per transaction via a JSONB additional_data["tokens_distributed"] marker,
persisted only after a successful transfer (so a genuinely-failed credit can retry).
"""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

import prsm.economy.payments.payment_processor as pp_module
from prsm.economy.payments.payment_processor import PaymentProcessor


class _Txn:
    def __init__(self):
        self.transaction_id = uuid.uuid4()
        self.crypto_currency = "FTNS"
        self.crypto_amount = Decimal("10")
        self.user_id = "user-1"
        self.payment_method = "card"
        self.fiat_amount = Decimal("5")
        self.fiat_currency = "USD"
        self.additional_data = {}


class _FakeSession:
    """Sync-query session (matches db_manager.session() usage)."""
    def __init__(self, txn):
        self._txn = txn
        self.commits = 0

    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def with_for_update(self, *a, **k):
        return self

    def first(self):
        return self._txn

    def commit(self):
        self.commits += 1


class _FakeSessionCtx:
    def __init__(self, session):
        self._s = session

    async def __aenter__(self):
        return self._s

    async def __aexit__(self, *a):
        return False


class _FakeDBManager:
    def __init__(self, session):
        self._s = session

    def session(self):
        return _FakeSessionCtx(self._s)


class _FtnsService:
    def __init__(self, ok=True):
        self.calls = 0
        self._ok = ok

    async def transfer_tokens(self, **kwargs):
        self.calls += 1
        return self._ok


def _proc(monkeypatch, txn, *, transfer_ok=True):
    proc = PaymentProcessor()
    proc.token_distribution_enabled = True
    proc.ftns_service = _FtnsService(ok=transfer_ok)
    monkeypatch.setattr(pp_module, "db_manager", _FakeDBManager(_FakeSession(txn)))
    # audit_logger.log_event is sync + fire-and-forget; make it a no-op
    monkeypatch.setattr(pp_module.audit_logger, "log_event", lambda *a, **k: None, raising=False)
    return proc


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_credits_once_then_idempotent_skip_on_replay(monkeypatch):
    txn = _Txn()
    proc = _proc(monkeypatch, txn)

    _run(proc._process_crypto_conversion(txn.transaction_id))   # first delivery
    assert proc.ftns_service.calls == 1
    assert txn.additional_data.get("tokens_distributed") is True

    _run(proc._process_crypto_conversion(txn.transaction_id))   # replay / provider retry
    assert proc.ftns_service.calls == 1                         # NOT credited again

    _run(proc._process_crypto_conversion(txn.transaction_id))   # a third replay
    assert proc.ftns_service.calls == 1


def test_failed_credit_is_not_marked_and_can_retry(monkeypatch):
    # a genuinely-failed transfer must NOT set the idempotency marker, so a later
    # retry can still credit (we only skip when crediting actually succeeded).
    txn = _Txn()
    proc = _proc(monkeypatch, txn, transfer_ok=False)

    _run(proc._process_crypto_conversion(txn.transaction_id))
    assert proc.ftns_service.calls == 1
    assert "tokens_distributed" not in txn.additional_data       # not marked on failure

    # transfer now succeeds on retry
    proc.ftns_service._ok = True
    _run(proc._process_crypto_conversion(txn.transaction_id))
    assert proc.ftns_service.calls == 2                          # retried + credited
    assert txn.additional_data.get("tokens_distributed") is True

    _run(proc._process_crypto_conversion(txn.transaction_id))    # now idempotent
    assert proc.ftns_service.calls == 2


def test_preexisting_marker_blocks_credit(monkeypatch):
    # a transaction already flagged distributed must never re-credit (e.g. a replay
    # arriving after a prior process already credited + persisted the marker).
    txn = _Txn()
    txn.additional_data = {"tokens_distributed": True}
    proc = _proc(monkeypatch, txn)

    _run(proc._process_crypto_conversion(txn.transaction_id))
    assert proc.ftns_service.calls == 0                          # never credited


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
