"""Sprint 954 — server-side Origin check on the wallet onboarding endpoints.

The auth review's last residual: /api/v1/auth/wallet/* had no server-side Origin
validation. CORS (PRSM_ALLOWED_ORIGINS, sp 2026-05-09) stops a cross-origin page
from READING wallet responses, but a DNS-rebinding attack makes the victim's
browser send a (state-changing) request to the loopback-bound node that the
browser treats as same-origin to the attacker's page — the Origin header still
carries the attacker's domain. This guard rejects any wallet request whose Origin
is present AND not in the operator's allowlist.

Contract (must not break legitimate clients):
  - No Origin header (CLI / MCP / server-to-server) → ALLOWED.
  - PRSM_ALLOWED_ORIGINS unset / empty (dev / unconfigured) → permissive,
    matching the CORS default + the 127.0.0.1 default bind.
  - Origin present + allowlist set + Origin in allowlist → ALLOWED.
  - Origin present + allowlist set + Origin NOT in allowlist → 403 (rebinding).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


# The 403 detail emitted by the guard carries this marker, so tests can
# distinguish the rebinding rejection from any other coincidental 403.
_REBIND_MARKER = "PRSM_ALLOWED_ORIGINS"
_WALLET_PATH = "/api/v1/auth/wallet/siwe/nonce"  # POST, empty body issues a nonce


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._royalty_distributor_client = None
    return node


def _client(node):
    # raise_server_exceptions=False: the wallet handlers need services wired
    # (set_services), which we intentionally don't do here — the Origin guard is
    # middleware-level (path + Origin + allowlist), so it runs BEFORE the handler.
    # An unwired handler 500s, which for the ALLOW cases just confirms the guard
    # let the request through (it's not the rebinding-403).
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _is_rebind_403(resp) -> bool:
    if resp.status_code != 403:
        return False
    try:
        return _REBIND_MARKER in str(resp.json().get("detail", ""))
    except Exception:
        return False


class TestWalletOriginGuard:
    def test_cross_origin_rejected_when_allowlist_set(self):
        """An Origin not in PRSM_ALLOWED_ORIGINS is rejected (the
        DNS-rebinding case)."""
        with patch.dict(os.environ, {
            "PRSM_ALLOWED_ORIGINS": "https://dash.example.com",
        }):
            resp = _client(_node()).post(
                _WALLET_PATH, json={},
                headers={"Origin": "https://attacker.example.com"},
            )
        assert _is_rebind_403(resp), (
            f"cross-origin wallet request must be rejected; got "
            f"{resp.status_code} {resp.text[:200]}"
        )

    def test_allowlisted_origin_not_rejected(self):
        """An Origin in the allowlist (the legit wallet UI) passes the guard."""
        with patch.dict(os.environ, {
            "PRSM_ALLOWED_ORIGINS": "https://dash.example.com",
        }):
            resp = _client(_node()).post(
                _WALLET_PATH, json={},
                headers={"Origin": "https://dash.example.com"},
            )
        assert not _is_rebind_403(resp), resp.text[:200]

    def test_no_origin_allowed_cli_mcp(self):
        """No Origin header (CLI / MCP / server-to-server) is always allowed,
        even with an allowlist configured."""
        with patch.dict(os.environ, {
            "PRSM_ALLOWED_ORIGINS": "https://dash.example.com",
        }):
            resp = _client(_node()).post(_WALLET_PATH, json={})
        assert not _is_rebind_403(resp), resp.text[:200]

    def test_no_allowlist_is_permissive(self):
        """Unset PRSM_ALLOWED_ORIGINS → permissive (dev), matching CORS default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_ALLOWED_ORIGINS", None)
            resp = _client(_node()).post(
                _WALLET_PATH, json={},
                headers={"Origin": "https://anything.example.com"},
            )
        assert not _is_rebind_403(resp), resp.text[:200]

    def test_whitespace_only_allowlist_is_permissive(self):
        """All-whitespace allowlist is treated as unset (matches CORS)."""
        with patch.dict(os.environ, {"PRSM_ALLOWED_ORIGINS": "   "}):
            resp = _client(_node()).post(
                _WALLET_PATH, json={},
                headers={"Origin": "https://anything.example.com"},
            )
        assert not _is_rebind_403(resp), resp.text[:200]

    def test_non_wallet_path_unaffected(self):
        """The guard is scoped to /api/v1/auth/wallet/* — other paths with a
        cross-origin header are not rejected by it (CORS still applies)."""
        with patch.dict(os.environ, {
            "PRSM_ALLOWED_ORIGINS": "https://dash.example.com",
        }):
            resp = _client(_node()).get(
                "/health",
                headers={"Origin": "https://attacker.example.com"},
            )
        assert not _is_rebind_403(resp)
        assert resp.status_code == 200
