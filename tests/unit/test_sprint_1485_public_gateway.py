"""Sprint 1485 — PRSM_PUBLIC_GATEWAY: a read-only public front door for the commons.

Closing the last gap of the content-durability arc. `NodeConfig.api_host` defaults
to 127.0.0.1, so every content route is loopback-only and a user who is not running
a node has nowhere to fetch from — content can now be published durably (sp1483) and
replicated (sp1484) and still be unreachable by the public.

THE SECURITY PROPERTY: on a publicly-bound node, "protected" must mean DENIED, not
"needs the operator key". If gateway mode merely required a key, one leaked or
reused node key would let anyone on the internet drive the operator's wallet,
uploads and admin surface. So gateway mode hard-refuses every non-allowlisted path
REGARDLESS of credentials — the blast radius of exposing the port is exactly the
public read allowlist and nothing else.
"""
from __future__ import annotations

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prsm.api.auth_middleware import (
    NodeAuthMiddleware,
    _public_gateway_mode,
    is_protected_path,
)

# Routes a gateway EXISTS to serve — the free Tier-A commons + health/discovery.
PUBLIC_READS = [
    "/health", "/readyz", "/status", "/info", "/peers",
    "/content/search", "/content/index/stats", "/content/provider-stats",
    "/content/retrieve/bafyabc123",
    "/storage/stats", "/marketplace/reputation", "/marketplace/providers",
]

# Routes that must NEVER be reachable through a gateway, key or no key.
MUST_BE_REFUSED = [
    "/wallet/withdraw", "/wallet/onramp/execute", "/balance", "/transactions",
    "/content/upload", "/content/upload-stream", "/content/paid/publish",
    "/admin/emissions/trigger", "/ftns/faucet", "/ledger/transfer",
    "/staking/stake", "/compute/inference", "/peers/connect",
    "/onboarding/identity", "/api/ftns/transfer", "/settler/withdraw",
    "/content/mine",
]


@pytest.fixture
def gateway_on(monkeypatch):
    monkeypatch.setenv("PRSM_PUBLIC_GATEWAY", "1")


@pytest.fixture
def gateway_off(monkeypatch):
    monkeypatch.delenv("PRSM_PUBLIC_GATEWAY", raising=False)


def test_gateway_mode_is_off_by_default(gateway_off):
    """An ordinary node must be completely unaffected."""
    assert _public_gateway_mode() is False


@pytest.mark.parametrize("flag,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True),
    ("0", False), ("false", False), ("", False), ("maybe", False),
])
def test_gateway_flag_parsing(monkeypatch, flag, expected):
    monkeypatch.setenv("PRSM_PUBLIC_GATEWAY", flag)
    assert _public_gateway_mode() is expected


@pytest.mark.parametrize("path", PUBLIC_READS)
def test_public_reads_are_not_protected(path):
    """★ A gateway must actually SERVE the commons — otherwise it is useless."""
    assert is_protected_path(path) is False, f"{path} must be publicly readable"


@pytest.mark.parametrize("path", MUST_BE_REFUSED)
def test_sensitive_routes_are_protected(path):
    """★ Every one of these is refused outright in gateway mode (see the
    middleware test below) because is_protected_path classifies it protected."""
    assert is_protected_path(path) is True, f"{path} must NOT be publicly reachable"


# ───────────────── middleware behaviour ─────────────────

def _client(*, api_key_hash=""):
    app = FastAPI()

    @app.get("/health")
    async def _h():
        return {"ok": True}

    @app.get("/content/search")
    async def _s():
        return {"results": []}

    @app.post("/wallet/withdraw")
    async def _w():
        return {"sent": True}

    @app.get("/balance")
    async def _b():
        return {"balance": 1234}

    app.add_middleware(NodeAuthMiddleware, api_key_hash=api_key_hash)
    return TestClient(app, raise_server_exceptions=False)


def test_gateway_serves_public_reads(gateway_on):
    c = _client()
    assert c.get("/health").status_code == 200
    assert c.get("/content/search").status_code == 200


def test_gateway_refuses_protected_routes_with_no_key(gateway_on):
    c = _client()
    assert c.post("/wallet/withdraw").status_code == 403
    assert c.get("/balance").status_code == 403


def test_gateway_refuses_protected_routes_EVEN_WITH_A_VALID_KEY(gateway_on):
    """★ THE property. If a valid key could widen a gateway, one leaked/reused node
    key would hand the internet the operator's wallet. Credentials must not open
    write surfaces on a publicly-bound node."""
    key = "prsm_test_key_value"
    c = _client(api_key_hash=hashlib.sha256(key.encode()).hexdigest())
    r = c.post("/wallet/withdraw", headers={"X-API-Key": key})
    assert r.status_code == 403, "a valid key must NOT widen a public gateway"
    assert "gateway" in r.json()["detail"].lower()


def test_without_gateway_mode_a_valid_key_still_works(gateway_off):
    """Non-gateway nodes are unchanged: the key still authorizes protected routes."""
    key = "prsm_test_key_value"
    c = _client(api_key_hash=hashlib.sha256(key.encode()).hexdigest())
    assert c.post("/wallet/withdraw", headers={"X-API-Key": key}).status_code == 200


def test_without_gateway_mode_and_no_key_is_dev_open(gateway_off):
    """Dev mode (no key configured) keeps its existing open behavior."""
    c = _client()
    assert c.post("/wallet/withdraw").status_code == 200


def test_gateway_refusal_is_explicit_not_a_404(gateway_on):
    """The refusal must be legible — a 404 would send an integrator debugging the
    wrong problem."""
    r = _client().post("/wallet/withdraw")
    assert r.status_code == 403
    body = r.json()
    assert body["error"] == "forbidden"
    assert "run your own node" in body["detail"].lower()


def test_path_normalization_tricks_stay_refused(gateway_on):
    """Encoded-slash / dot-segment tricks must not smuggle a write past a gateway."""
    for sneaky in ["/content/%2e%2e/wallet/withdraw", "/content//../balance",
                   "/content/./mine", "/wallet\\withdraw"]:
        assert is_protected_path(sneaky) is True, f"{sneaky} must stay protected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
