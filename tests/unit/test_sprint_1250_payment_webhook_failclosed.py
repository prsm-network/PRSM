"""Sprint 1250 — payment webhook signature verification must FAIL CLOSED.

Found by the fail-open security audit (workflow wp0nc72qd). The webhook route
POST /payments/webhooks/{provider} is UNAUTHENTICATED and always mounted — the
signature IS the authentication. But:
  - _verify_stripe_signature returned True when STRIPE_WEBHOOK_SECRET was unset
    (an OPTIONAL secret — the node boots without it; the promised prod guard never
    existed), and
  - _verify_paypal_signature returned True UNCONDITIONALLY (the real PayPal
    verify-webhook-signature call was never implemented).
So an attacker could create a real PENDING FTNS purchase, then POST a forged
"completed/approved" event referencing its UUID with no valid signature → the txn
flips to COMPLETED → ftns_service.transfer_tokens credits real FTNS with no fiat
charged (replayable). CRITICAL mint-without-payment bypass.

Fix: both verifiers FAIL CLOSED (reject) unless a signature is actually verified —
Stripe rejects when the secret is unset (its real HMAC path is unchanged when set);
PayPal rejects until real verify-webhook-signature wiring lands. An explicit
DEV-ONLY opt-in (PRSM_ALLOW_INSECURE_PAYMENT_WEBHOOKS, default OFF) preserves local
testing without weakening the secure default.
"""
from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from prsm.economy.payments.payment_processor import PaymentProcessor

INSECURE = "PRSM_ALLOW_INSECURE_PAYMENT_WEBHOOKS"


@pytest.fixture
def proc():
    return PaymentProcessor()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # default to the SECURE posture: no dev opt-in, no secrets, unless a test sets them
    for k in (INSECURE, "STRIPE_WEBHOOK_SECRET", "PAYPAL_WEBHOOK_ID"):
        monkeypatch.delenv(k, raising=False)


def _stripe_header(secret: str, body: bytes, ts: int | None = None) -> str:
    ts = ts if ts is not None else int(time.time())
    sig = hmac.new(secret.encode(), f"{ts}.{body.decode()}".encode(), hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


# ── Stripe ───────────────────────────────────────────────────────────────────

def test_stripe_no_secret_fails_closed(proc):
    # the headline bug: unset secret used to return True. Now → reject.
    assert proc._verify_stripe_signature(b'{"x":1}', "") is False


def test_stripe_no_secret_dev_optin_allows(proc, monkeypatch):
    monkeypatch.setenv(INSECURE, "1")
    assert proc._verify_stripe_signature(b'{"x":1}', "") is True


def test_stripe_valid_hmac_accepted(proc, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    body = b'{"type":"payment_intent.succeeded"}'
    assert proc._verify_stripe_signature(body, _stripe_header("whsec_test", body)) is True


def test_stripe_invalid_hmac_rejected(proc, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    body = b'{"type":"payment_intent.succeeded"}'
    bad = _stripe_header("WRONG_secret", body)
    assert proc._verify_stripe_signature(body, bad) is False


def test_stripe_replay_old_timestamp_rejected(proc, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    body = b'{"type":"payment_intent.succeeded"}'
    old = _stripe_header("whsec_test", body, ts=int(time.time()) - 4000)
    assert proc._verify_stripe_signature(body, old) is False


# ── PayPal ───────────────────────────────────────────────────────────────────

def test_paypal_fails_closed_by_default(proc):
    # unconditional return True is gone — reject until real verification is wired.
    assert proc._verify_paypal_signature(b'{"event_type":"CHECKOUT.ORDER.APPROVED"}', "sig") is False


def test_paypal_fails_closed_even_with_webhook_id(proc, monkeypatch):
    # a configured PAYPAL_WEBHOOK_ID must NOT be mistaken for verification.
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "WH-123")
    assert proc._verify_paypal_signature(b'{"event_type":"CHECKOUT.ORDER.APPROVED"}', "sig") is False


def test_paypal_dev_optin_allows(proc, monkeypatch):
    monkeypatch.setenv(INSECURE, "1")
    assert proc._verify_paypal_signature(b'{"event_type":"CHECKOUT.ORDER.APPROVED"}', "sig") is True


# ── end-to-end: the forged-webhook exploit is now blocked ────────────────────

def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_process_webhook_forged_stripe_rejected_no_crediting(proc, monkeypatch):
    # the exploit chain: no secret + forged event → must NOT reach the crediting path.
    credited = []
    monkeypatch.setattr(proc, "_process_stripe_webhook",
                        lambda payload: credited.append(True))  # would-be crediting
    forged = {"type": "payment_intent.succeeded",
              "data": {"object": {"id": "00000000-0000-0000-0000-000000000001"}}}
    ok = _run(proc.process_webhook("stripe", forged, b"forged", ""))
    assert ok is False
    assert credited == []          # crediting path never reached


def test_process_webhook_forged_paypal_rejected_no_crediting(proc, monkeypatch):
    credited = []
    monkeypatch.setattr(proc, "_process_paypal_webhook",
                        lambda payload: credited.append(True))
    forged = {"event_type": "CHECKOUT.ORDER.APPROVED",
              "resource": {"id": "00000000-0000-0000-0000-000000000001"}}
    ok = _run(proc.process_webhook("paypal", forged, b"forged", "sig"))
    assert ok is False
    assert credited == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
