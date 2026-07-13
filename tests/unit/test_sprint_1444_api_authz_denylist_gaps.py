"""sp1444 (API-authorization re-audit) — four deny-list gaps on the highest-value routes.

NodeAuthMiddleware is a DENY-LIST: it enforces the operator API key only for paths that
is_protected_path() flags (a PROTECTED_PREFIXES startswith or a PROTECTED_PATH_PATTERNS regex);
anything not enumerated is served unauthenticated even on a KEYED public node. The re-audit
(4th time the deny-list leaked) found four uncovered sensitive routes:

  A (CRITICAL) — the mounted dashboard sub-app's /api/ftns/transfer → node.ledger_sync.signed_transfer
    is an unkeyed DRAIN of the operator's off-chain FTNS; siblings /api/ftns/stake + the money-PII
    /api/ftns/balance|history were also open. The whole dashboard /api/* surface was invisible.
  B (HIGH) — POST /content/paid/publish spends the operator's publisher-key gas + registers on-chain.
  D (MEDIUM) — GET /balance leaks operator balance + last 20 tx (same PII as the protected /transactions).

Fixed by adding the specific sensitive prefixes. These are money/security assertions — never weaken.
A blanket "/api/" is deliberately NOT used (it would gate the dashboard's own /api/auth/login +
public /api/status|health), so the test also pins that legit-public paths stay OPEN.
"""
from __future__ import annotations

import pytest

from prsm.api.auth_middleware import (
    NodeAuthMiddleware,
    hash_api_key,
    is_protected_path,
)


# ── the newly-closed gaps must now be PROTECTED ──────────────────────────────


@pytest.mark.parametrize("path", [
    "/api/ftns/transfer",        # A — the CRITICAL operator FTNS drain
    "/api/ftns/stake",           # A — mutating
    "/api/ftns/balance",         # A/C — money PII
    "/api/ftns/history",         # C — full operator ledger PII
    "/api/jobs/submit",          # A — mutating job submit
    "/api/distillation/submit",  # A — mutating
    "/api/teacher/create",       # A — mutating teacher create
    "/content/paid/publish",     # B — operator gas + on-chain creator
    "/balance",                  # D — operator balance + tx PII
    "/balance/onchain",          # D — same PII class
])
def test_sensitive_route_is_now_protected(path):
    assert is_protected_path(path) is True, f"{path} is an unauthenticated gap on a keyed node"


# ── legit-public / dashboard-login / free-commons paths must STAY OPEN ────────


@pytest.mark.parametrize("path", [
    "/api/auth/login",     # the dashboard's OWN login — a blanket /api/ would break it
    "/api/auth/me",
    "/api/status",         # dashboard public status read
    "/api/health",         # dashboard health
    "/api/jobs",           # job LIST read (only /api/jobs/submit is gated)
    "/api/jobs/job-123",   # job status read
    "/api/peers",          # dashboard peer read
    "/api/content/search", # public content search
    "/content/abc123def",  # free-commons content read (public by design)
    "/health",
    "/peers",
])
def test_legit_public_route_stays_open(path):
    assert is_protected_path(path) is False, (
        f"{path} was over-gated — the dashboard/commons would break on a keyed node")


# ── end-to-end middleware dispatch: keyed node blocks the unkeyed drain ───────


class _FakeReq:
    def __init__(self, path, headers=None):
        from starlette.datastructures import Headers, URL
        self.url = URL(f"http://n{path}")
        self.headers = Headers(headers or {})


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_keyed_node_rejects_unkeyed_ftns_transfer():
    """The money shot for finding A: on a keyed node, POST /api/ftns/transfer with NO key → 401."""
    mw = NodeAuthMiddleware(app=None, api_key_hash=hash_api_key("secret-operator-key"))
    assert mw.auth_enabled

    called = {"next": False}

    async def _call_next(req):
        called["next"] = True
        return "PASSED"

    resp = _run(mw.dispatch(_FakeReq("/api/ftns/transfer"), _call_next))
    assert not called["next"], "the drain route reached the handler without a key"
    assert getattr(resp, "status_code", None) == 401


def test_keyed_node_allows_ftns_transfer_with_valid_key():
    key = "secret-operator-key"
    mw = NodeAuthMiddleware(app=None, api_key_hash=hash_api_key(key))

    async def _call_next(req):
        return "PASSED"

    resp = _run(mw.dispatch(
        _FakeReq("/api/ftns/transfer", {"x-api-key": key}), _call_next))
    assert resp == "PASSED"


def test_public_status_never_requires_a_key_even_when_enabled():
    mw = NodeAuthMiddleware(app=None, api_key_hash=hash_api_key("secret-operator-key"))

    async def _call_next(req):
        return "PASSED"

    # /api/status is a dashboard read that must stay reachable unauthenticated.
    resp = _run(mw.dispatch(_FakeReq("/api/status"), _call_next))
    assert resp == "PASSED"
