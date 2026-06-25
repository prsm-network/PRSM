"""Sprint 1260 — real PayPal webhook signature verification (fail-closed money path).

Before this sprint `_verify_paypal_signature` was a pure fail-closed stub (sp1250): it
ALWAYS rejected because the webhook route forwarded only a single signature string, and the
real PayPal verify-webhook-signature flow (which needs the FULL transmission header set + an
OAuth token + a server-side verify call) was unwired. That made PayPal unusable as a fiat
on-ramp.

This sprint wires it for real: the route threads the 5 PayPal transmission headers through to
the verifier, which obtains an OAuth client-credentials token and POSTs the event to PayPal's
/v1/notifications/verify-webhook-signature, accepting ONLY verification_status == "SUCCESS".
Everything fails closed: missing config, missing headers, OAuth failure, non-SUCCESS status,
or any network exception → reject. A DEV-ONLY opt-in (PRSM_ALLOW_INSECURE_PAYMENT_WEBHOOKS)
still bypasses for local testing (the payment analog of sp888 KYC).

The HTTP flow is exercised against httpx.MockTransport — a faithful wire-level mock that
validates the real request construction (URLs, auth, body fields), not a simplified stub.
"""
from __future__ import annotations

import httpx
import pytest

# Capture the REAL AsyncClient at import time — tests/conftest.py autouse-mocks
# httpx.AsyncClient at RUNTIME (and its mock doesn't even support `async with`), so we bind
# the genuine class now (before any fixture patches the module attribute) and drive it with
# a real MockTransport for a faithful wire-level test of the PayPal verify flow.
from httpx import AsyncClient as _RealAsyncClient

from prsm.economy.payments.payment_processor import (
    PaymentProcessor,
    _paypal_transmission_headers,
)

_PAYPAL_HEADERS = {
    "paypal-transmission-id": "tx-id-123",
    "paypal-transmission-time": "2026-06-25T00:00:00Z",
    "paypal-cert-url": "https://api.paypal.com/cert.pem",
    "paypal-auth-algo": "SHA256withRSA",
    "paypal-transmission-sig": "deadbeefsig==",
}

_EVENT = {"event_type": "CHECKOUT.ORDER.APPROVED", "resource": {"id": "ORDER-1"}}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.delenv("PRSM_ALLOW_INSECURE_PAYMENT_WEBHOOKS", raising=False)
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "WH-TEST")
    monkeypatch.setenv("PAYPAL_CLIENT_ID", "cid")
    monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("PAYPAL_API_BASE", "https://api-m.sandbox.paypal.com")
    return PaymentProcessor({})


def _mock_client(monkeypatch, processor, *, verification_status="SUCCESS",
                 token_status=200, verify_status=200, capture=None):
    """Patch processor._paypal_async_client to return an AsyncClient backed by a
    MockTransport that emulates PayPal's OAuth + verify endpoints."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v1/oauth2/token"):
            # client-credentials with HTTP Basic auth must be present
            assert request.headers.get("authorization", "").startswith("Basic ")
            if token_status != 200:
                return httpx.Response(token_status, json={"error": "boom"})
            return httpx.Response(200, json={"access_token": "tok-abc", "token_type": "Bearer"})
        if path.endswith("/v1/notifications/verify-webhook-signature"):
            import json as _json
            body = _json.loads(request.content.decode())
            if capture is not None:
                capture.update(body)
                capture["_auth"] = request.headers.get("authorization", "")
            if verify_status != 200:
                return httpx.Response(verify_status, json={"error": "boom"})
            return httpx.Response(200, json={"verification_status": verification_status})
        return httpx.Response(404, json={"error": "unexpected path"})

    def _factory():
        return _RealAsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(processor, "_paypal_async_client", _factory)


# ── header extraction ────────────────────────────────────────────────────────

def test_transmission_headers_extracted_case_insensitive():
    upper = {k.upper(): v for k, v in _PAYPAL_HEADERS.items()}
    th = _paypal_transmission_headers(upper)
    assert th is not None
    assert th["transmission_id"] == "tx-id-123"
    assert th["transmission_sig"] == "deadbeefsig=="


@pytest.mark.parametrize("missing", list(_PAYPAL_HEADERS))
def test_transmission_headers_none_when_any_missing(missing):
    partial = {k: v for k, v in _PAYPAL_HEADERS.items() if k != missing}
    assert _paypal_transmission_headers(partial) is None


# ── verifier: happy path + fail-closed modes ──────────────────────────────────

@pytest.mark.asyncio
async def test_verify_success(configured, monkeypatch):
    captured = {}
    _mock_client(monkeypatch, configured, verification_status="SUCCESS", capture=captured)
    ok = await configured._verify_paypal_signature(_EVENT, dict(_PAYPAL_HEADERS))
    assert ok is True
    # the verify request carried the right fields
    assert captured["webhook_id"] == "WH-TEST"
    assert captured["transmission_id"] == "tx-id-123"
    assert captured["webhook_event"] == _EVENT
    assert captured["_auth"] == "Bearer tok-abc"


@pytest.mark.asyncio
async def test_verify_failure_status_rejects(configured, monkeypatch):
    _mock_client(monkeypatch, configured, verification_status="FAILURE")
    assert await configured._verify_paypal_signature(_EVENT, dict(_PAYPAL_HEADERS)) is False


@pytest.mark.asyncio
async def test_oauth_failure_fails_closed(configured, monkeypatch):
    _mock_client(monkeypatch, configured, token_status=401)
    assert await configured._verify_paypal_signature(_EVENT, dict(_PAYPAL_HEADERS)) is False


@pytest.mark.asyncio
async def test_verify_endpoint_error_fails_closed(configured, monkeypatch):
    _mock_client(monkeypatch, configured, verify_status=500)
    assert await configured._verify_paypal_signature(_EVENT, dict(_PAYPAL_HEADERS)) is False


@pytest.mark.asyncio
async def test_network_exception_fails_closed(configured, monkeypatch):
    def _factory():
        def handler(request):
            raise httpx.ConnectError("network down")
        return _RealAsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(configured, "_paypal_async_client", _factory)
    assert await configured._verify_paypal_signature(_EVENT, dict(_PAYPAL_HEADERS)) is False


@pytest.mark.asyncio
async def test_missing_config_fails_closed(monkeypatch):
    monkeypatch.delenv("PRSM_ALLOW_INSECURE_PAYMENT_WEBHOOKS", raising=False)
    for v in ("PAYPAL_WEBHOOK_ID", "PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET"):
        monkeypatch.delenv(v, raising=False)
    proc = PaymentProcessor({})
    # must NOT even attempt a network call → reject on config
    assert await proc._verify_paypal_signature(_EVENT, dict(_PAYPAL_HEADERS)) is False


@pytest.mark.asyncio
async def test_missing_transmission_headers_fails_closed(configured):
    assert await configured._verify_paypal_signature(_EVENT, {}) is False
    assert await configured._verify_paypal_signature(_EVENT, None) is False


@pytest.mark.asyncio
async def test_dev_optin_bypasses_without_network(monkeypatch):
    monkeypatch.setenv("PRSM_ALLOW_INSECURE_PAYMENT_WEBHOOKS", "1")
    proc = PaymentProcessor({})
    # no config, no headers, no client patched — must still return True via the dev opt-in
    assert await proc._verify_paypal_signature(_EVENT, {}) is True


# ── process_webhook threads headers + verifies before processing ───────────────

@pytest.mark.asyncio
async def test_process_webhook_rejects_on_bad_paypal_signature(configured, monkeypatch):
    _mock_client(monkeypatch, configured, verification_status="FAILURE")
    handled = await configured.process_webhook(
        "paypal", _EVENT, b"{}", "", headers=dict(_PAYPAL_HEADERS),
    )
    assert handled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
