"""Sprint 1103 — fold the Domain-07 review API-surface findings:

- HIGH (onboarding): the /onboarding/* router (config + identity OVERWRITE — POST
  /onboarding/identity import writes config/node_identity.json, /onboarding/launch
  writes node_config.json) is mounted on the daemon app but was in NEITHER
  PROTECTED_PREFIXES nor the loopback gate → unauthenticated node-takeover-on-restart
  even with PRSM_NODE_API_KEY set. Now protected.
- MEDIUM (rate-limit key spoof): _resolve_requester_key trusted X-Forwarded-For /
  X-Real-IP UNCONDITIONALLY, so a directly-exposed node let an attacker rotate XFF to
  get a fresh token bucket every request, defeating PRSM_INFERENCE_MAX_RPS_PER_REQUESTER.
  Now the proxy headers are only trusted when the IMMEDIATE client is loopback (a
  trusted reverse proxy) — mirroring the admin-loopback middleware's precondition.

(The wallet read-IDOR HIGH on /api/v1/auth/wallet/* is closed by the sp1013 session
mechanism whose default-on flip is an explicit frontend-coordinated follow-on — tracked
separately, not folded here.)
"""
from __future__ import annotations

from unittest.mock import MagicMock


# ── HIGH: onboarding routes are in the protected set ─────────────────────────────

def test_onboarding_identity_is_protected():
    from prsm.api.auth_middleware import is_protected_path
    assert is_protected_path("/onboarding/identity") is True


def test_onboarding_launch_is_protected():
    from prsm.api.auth_middleware import is_protected_path
    assert is_protected_path("/onboarding/launch") is True


def test_onboarding_api_keys_is_protected():
    from prsm.api.auth_middleware import is_protected_path
    assert is_protected_path("/onboarding/api-keys") is True


# ── MEDIUM: rate-limit key can't be spoofed from a non-loopback immediate client ─

def test_xff_ignored_when_immediate_client_is_public():
    """A directly-exposed node (public immediate client) must NOT trust X-Forwarded-For
    — else an attacker rotates it to evade the per-requester bucket."""
    from prsm.node.api import _resolve_requester_key
    request = MagicMock()
    request.headers = {"x-forwarded-for": "10.0.0.1, attacker-rotated-value"}
    request.client = MagicMock()
    request.client.host = "203.0.113.7"  # public — NOT a trusted proxy
    # must fall back to the real socket peer, not the spoofable header
    assert _resolve_requester_key(request) == "203.0.113.7"


def test_x_real_ip_ignored_when_immediate_client_is_public():
    from prsm.node.api import _resolve_requester_key
    request = MagicMock()
    request.headers = {"x-real-ip": "attacker-chosen"}
    request.client = MagicMock()
    request.client.host = "203.0.113.7"
    assert _resolve_requester_key(request) == "203.0.113.7"


def test_xff_trusted_from_loopback_proxy():
    """Behind a loopback reverse proxy (the supported deployment), the last XFF hop IS
    the real client — preserve the sprint-741 behavior."""
    from prsm.node.api import _resolve_requester_key
    request = MagicMock()
    request.headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8, 9.9.9.9"}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"  # trusted loopback proxy
    assert _resolve_requester_key(request) == "9.9.9.9"


def test_direct_public_client_uses_socket_peer():
    from prsm.node.api import _resolve_requester_key
    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "203.0.113.42"
    assert _resolve_requester_key(request) == "203.0.113.42"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
