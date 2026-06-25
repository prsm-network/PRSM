"""Sprint 1274 — global rate limiter must not trust spoofable forwarded headers (round 6, HIGH).

RateLimitMiddleware._get_client_id keyed the per-client bucket on x-forwarded-for / x-real-ip
WITHOUT checking the immediate socket peer is a trusted (loopback) reverse proxy, and took the
LEFTMOST (client-controlled) XFF hop. On a directly-exposed node an attacker sends a unique
X-Forwarded-For per request → a fresh bucket each time → the default 100/min cap never trips
(and combined with sp1273, unlimited unmetered GPU inferences). This is the gap sp1103 already
closed for the per-requester inference bucket; sp1274 applies the same to the GLOBAL limiter.
"""
from __future__ import annotations

from types import SimpleNamespace

from prsm.node.api_hardening import RateLimitMiddleware


def _mw():
    return RateLimitMiddleware.__new__(RateLimitMiddleware)  # bypass the ASGI-app constructor


def _req(host, headers=None, user_id=None):
    state = SimpleNamespace()
    if user_id is not None:
        state.user_id = user_id
    return SimpleNamespace(
        client=SimpleNamespace(host=host),
        headers=headers or {},
        state=state,
    )


def test_spoofed_xff_from_untrusted_peer_ignored():
    mw = _mw()
    # attacker on a public peer rotates X-Forwarded-For; must key on the REAL socket host
    k1 = mw._get_client_id(_req("203.0.113.7", {"x-forwarded-for": "1.1.1.1"}))
    k2 = mw._get_client_id(_req("203.0.113.7", {"x-forwarded-for": "2.2.2.2"}))
    assert k1 == k2 == "ip:203.0.113.7"   # same bucket regardless of spoofed header


def test_spoofed_x_real_ip_from_untrusted_peer_ignored():
    mw = _mw()
    k = mw._get_client_id(_req("203.0.113.7", {"x-real-ip": "9.9.9.9"}))
    assert k == "ip:203.0.113.7"


def test_trusted_loopback_proxy_xff_honored():
    mw = _mw()
    # behind a co-located reverse proxy (loopback peer), trust the proxy-appended (rightmost) hop
    k = mw._get_client_id(_req("127.0.0.1", {"x-forwarded-for": "1.1.1.1, 203.0.113.50"}))
    assert k == "ip:203.0.113.50"


def test_authenticated_user_keyed_by_user():
    mw = _mw()
    assert mw._get_client_id(_req("203.0.113.7", user_id="alice")) == "user:alice"


def test_no_headers_keys_on_socket_host():
    mw = _mw()
    assert mw._get_client_id(_req("198.51.100.4")) == "ip:198.51.100.4"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
