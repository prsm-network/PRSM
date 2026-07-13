"""sp1445 — DEFAULT-DENY inversion of NodeAuthMiddleware.

The auth model was a DENY-LIST: protect only the enumerated PROTECTED_PREFIXES; every unenumerated
path was OPEN even on a keyed node. It leaked a new gap every time a sensitive route shipped
(sp138/183/1012/1103/1444 — the last an unkeyed operator-FTNS drain). sp1445 inverts it: a path is
PROTECTED by default and open ONLY if on the explicit PUBLIC allowlist. A NEW route is therefore
fail-closed (protected) the moment it is added.

This test pins THREE things:
  1. BEHAVIOR-PRESERVING — every route that was reachable-without-key before stays open (so the free
     content commons, health/status, the paid-key self-auth serve, the dashboard reads + login, and
     the KYC webhook don't break on a keyed node).
  2. THE CORE PROPERTY — a hypothetical NEW/unlisted route (and the sp1444-class sensitive routes) is
     protected by default, with no allowlist entry needed.
  3. Path-normalization tricks (//, /../, /./, %2f, %2e, backslash) fail closed.

Security assertions — never weaken to pass.
"""
from __future__ import annotations

import pytest

from prsm.api.auth_middleware import (
    NodeAuthMiddleware,
    hash_api_key,
    is_protected_path,
)


# ── 1. behavior-preserving: every currently-reachable route stays OPEN ────────

# The exact set enumerated from the live @app (api.py) + @self.app (dashboard) decorators, plus the
# FastAPI auto-routes (/docs, /openapi.json, /redoc). If a real public route is missing here, a
# keyed node would start 401'ing it — so this list IS the behavior-preservation contract.
_CURRENTLY_OPEN = [
    # root / health / status / discovery / info
    "/", "/health", "/health/detailed", "/health/ready", "/readyz", "/status", "/metrics",
    "/info", "/api-info", "/node/info", "/node/identity/pubkey", "/peers", "/bootstrap/status",
    "/rings/status", "/privacy/budget", "/auth/verify", "/dashboard", "/agents",
    "/docs", "/openapi.json", "/redoc", "/audit/recent", "/audit/summary",
    # free Tier-A content commons
    "/content/search", "/content/search/semantic", "/content/index/stats", "/content/provider-stats",
    "/content/abc123def",                       # GET /content/{cid}
    "/content/retrieve/abc123def",              # GET /content/retrieve/{cid}
    "/content/recipient-manifest/abc123def",    # GET /content/recipient-manifest/{cid}
    "/content/paid-key/0xdeadbeef",             # GET /content/paid-key/{hash} (SELF_AUTH)
    # storage / marketplace aggregate reads
    "/storage/stats", "/storage/pinned-stats", "/storage/provider-reputations", "/marketplace/reputation",
    # KYC webhook (SELF_AUTH via vendor HMAC)
    "/wallet/kyc/webhook/persona",
    # dashboard sub-app public reads + login
    "/api/auth/login", "/api/auth/logout", "/api/auth/me", "/api/status", "/api/node", "/api/health",
    "/api/peers", "/api/agents", "/api/content/search", "/api/distillation", "/api/jobs",
    "/api/teacher/list", "/api/agents/agent-1", "/api/jobs/job-1", "/api/distillation/job-9",
    "/api/teacher/teacher-1",
]


@pytest.mark.parametrize("path", _CURRENTLY_OPEN)
def test_currently_reachable_route_stays_open(path):
    assert is_protected_path(path) is False, (
        f"{path} was reachable-without-key before; default-deny must not break it")


# ── 2. the core property: unlisted / sensitive routes are protected by default ─


@pytest.mark.parametrize("path", [
    "/some/brand/new/route",          # a hypothetical FUTURE route — the whole point
    "/newfeature/action",
    "/admin/newly-added-trigger",     # a new operator route ships → protected, no allowlist edit
    "/wallet/new-money-move",
    "/api/newmutation",               # a new dashboard route
])
def test_a_new_unlisted_route_is_protected_by_default(path):
    assert is_protected_path(path) is True, (
        f"{path} is unlisted — default-deny must protect it (this is the inversion's whole value)")


@pytest.mark.parametrize("path", [
    # the sp1444 findings — still protected (now by DEFAULT, not an explicit prefix)
    "/api/ftns/transfer", "/api/ftns/history", "/content/paid/publish", "/balance",
    # reserved-word collisions kept protected via the {param} exclusions
    "/content/upload", "/content/mine", "/api/jobs/submit", "/api/distillation/submit",
    "/api/teacher/create",
    # a sampling of clearly-operator routes
    "/transactions", "/admin/heartbeat/trigger", "/wallet/withdraw", "/staking/stake",
    "/ledger/transfer", "/settler/register", "/peers/connect",
])
def test_sensitive_route_is_protected(path):
    assert is_protected_path(path) is True, f"{path} must require the operator key"


# ── 3. path-normalization tricks fail closed ─────────────────────────────────


@pytest.mark.parametrize("path", [
    "//admin/heartbeat/trigger",       # leading double-slash
    "/admin/../wallet/withdraw",       # dot-segment
    "/content/./../wallet/withdraw",
    "/wallet/%2e%2e/withdraw",         # encoded dot
    "/content/%2fadmin",               # encoded slash
    "/admin\\heartbeat",               # backslash
])
def test_normalization_tricks_are_protected(path):
    assert is_protected_path(path) is True, f"normalization bypass not fail-closed: {path}"


# ── end-to-end dispatch: keyed node protects a NEW route without any allowlist edit ─


class _FakeReq:
    def __init__(self, path, headers=None):
        from starlette.datastructures import Headers, URL
        self.url = URL(f"http://n{path}")
        self.headers = Headers(headers or {})


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_keyed_node_rejects_a_brand_new_unlisted_route():
    mw = NodeAuthMiddleware(app=None, api_key_hash=hash_api_key("op-key"))
    reached = {"h": False}

    async def _next(req):
        reached["h"] = True
        return "PASSED"

    resp = _run(mw.dispatch(_FakeReq("/some/future/money/route"), _next))
    assert not reached["h"] and getattr(resp, "status_code", None) == 401


def test_keyed_node_lets_the_free_commons_read_through():
    mw = NodeAuthMiddleware(app=None, api_key_hash=hash_api_key("op-key"))

    async def _next(req):
        return "PASSED"

    assert _run(mw.dispatch(_FakeReq("/content/retrieve/somecid"), _next)) == "PASSED"
