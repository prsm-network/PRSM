"""Sprint 183 — NodeAuthMiddleware returns 401/403 not 500 on auth
failure, and `/transactions` + `/content/mine` are now protected.

Pre-fix two security bugs:

1. `dispatch()` raised `HTTPException(401)` and `HTTPException(403)`
   on auth failure. Starlette's BaseHTTPMiddleware does NOT catch
   HTTPException in dispatch() — the raise propagates to the ASGI
   error handler and the client gets 500. Monitoring tools then
   mis-classify auth failures as outages.

2. `/transactions` (operator financial history) and `/content/mine`
   (operator upload history) were not in PROTECTED_PREFIXES — any
   network-adjacent caller could enumerate the operator's
   transactions or uploaded content list.

Live verification on Base Sepolia 2026-05-11:
  pre:
    PRSM_NODE_API_KEY=set
    GET /admin/webhook-history (no auth) → 500 Internal Server Error
    GET /transactions (no auth) → 200 with full tx history
  post:
    GET /admin/webhook-history (no auth) → 401 with auth-required
    GET /transactions (no auth) → 401
"""
from __future__ import annotations

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prsm.api.auth_middleware import (
    NodeAuthMiddleware,
    PROTECTED_PREFIXES,
    hash_api_key,
)


def _app_with_auth(api_key: str):
    app = FastAPI()
    app.add_middleware(
        NodeAuthMiddleware,
        api_key_hash=hash_api_key(api_key),
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/admin/webhook-history")
    def webhook_history():
        return {"entries": []}

    @app.get("/transactions")
    def transactions():
        return {"transactions": []}

    @app.get("/content/mine")
    def content_mine():
        return {"entries": []}

    return TestClient(app)


class TestAuthMiddlewareResponse:
    def test_no_auth_returns_401_not_500(self):
        """Sprint 183 — missing auth returns 401 (was 500 because
        HTTPException raised from middleware dispatch leaked to
        500 via Starlette's error handler)."""
        client = _app_with_auth("real-key")
        resp = client.get("/admin/webhook-history")
        assert resp.status_code == 401
        assert "authentication required" in resp.json()["detail"].lower()

    def test_wrong_auth_returns_403_not_500(self):
        """Sprint 183 — wrong key returns 403 (was 500)."""
        client = _app_with_auth("real-key")
        resp = client.get(
            "/admin/webhook-history",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403
        assert "invalid api key" in resp.json()["detail"].lower()

    def test_right_auth_passes_through(self):
        """Sprint 183 invariant — correct key still grants access."""
        client = _app_with_auth("real-key")
        resp = client.get(
            "/admin/webhook-history",
            headers={"Authorization": "Bearer real-key"},
        )
        assert resp.status_code == 200

    def test_x_api_key_header_also_works(self):
        """Sprint 183 — X-API-Key header is the alternate auth path."""
        client = _app_with_auth("real-key")
        resp = client.get(
            "/admin/webhook-history",
            headers={"X-API-Key": "real-key"},
        )
        assert resp.status_code == 200


class TestConstantTimeKeyCompare:
    """Sprint 1271 (audit round 5) — the API-key check uses hmac.compare_digest. Both
    operands are fixed-length SHA256 hex digests, so compare_digest's equal-length
    requirement always holds even when the SUPPLIED key differs wildly in length."""

    def test_wrong_key_of_very_different_length_rejected(self):
        client = _app_with_auth("real-key")
        for bad in ("x", "y" * 500, ""):
            resp = client.get(
                "/admin/webhook-history",
                headers={"Authorization": f"Bearer {bad}"} if bad else {},
            )
            assert resp.status_code in (401, 403)  # never 500 / never accepted

    def test_uses_constant_time_compare(self):
        # the module imports hmac and the dispatch path calls compare_digest
        import inspect
        import prsm.api.auth_middleware as m
        assert hasattr(m, "hmac")
        assert "compare_digest" in inspect.getsource(m.NodeAuthMiddleware.dispatch)


class TestNewProtectedPrefixes:
    def test_transactions_is_protected(self):
        """Sprint 183 — /transactions added to PROTECTED_PREFIXES
        (leaks operator financial history)."""
        assert any(p == "/transactions" or "/transactions".startswith(p)
                   for p in PROTECTED_PREFIXES)

        client = _app_with_auth("real-key")
        resp = client.get("/transactions")
        assert resp.status_code == 401

    def test_content_mine_is_protected(self):
        """Sprint 183 — /content/mine added to PROTECTED_PREFIXES
        (leaks operator upload history + provenance fingerprint)."""
        assert any(p == "/content/mine" or "/content/mine".startswith(p)
                   for p in PROTECTED_PREFIXES)

        client = _app_with_auth("real-key")
        resp = client.get("/content/mine")
        assert resp.status_code == 401

    def test_health_still_unprotected(self):
        """Public endpoint regression-pin."""
        client = _app_with_auth("real-key")
        resp = client.get("/health")
        assert resp.status_code == 200
