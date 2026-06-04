"""Sprint 960 — fail-closed-aware startup check for the public-bind + no-auth posture.

NodeAuthMiddleware only authenticates the protected money prefixes (/wallet/,
/compute/, /transactions/) when PRSM_NODE_API_KEY is set, and listen_host
defaults to 0.0.0.0. So the out-of-the-box posture is "bound to every interface
with no API key" → those money endpoints are reachable UNAUTHENTICATED by anyone
who can route to the host, silently.

This adds a pure posture classifier used at node startup. sp1011 later flipped
the startup response to FAIL-CLOSED BY DEFAULT (the node refuses to start in the
insecure posture unless the operator acks via PRSM_ALLOW_INSECURE_PUBLIC_BIND);
the classifier itself is unchanged and still pinned here.
"""
from __future__ import annotations

import inspect

from prsm.node.node import PRSMNode, assess_public_bind_auth_posture


def test_public_bind_without_key_is_insecure():
    level, msg = assess_public_bind_auth_posture(
        listen_host="0.0.0.0", api_key_present=False)
    assert level == "insecure"
    assert "PRSM_NODE_API_KEY" in msg
    # Names the endpoints actually at risk so the operator knows the blast radius.
    assert "/wallet/" in msg


def test_ipv6_all_interfaces_without_key_is_insecure():
    level, _ = assess_public_bind_auth_posture(
        listen_host="::", api_key_present=False)
    assert level == "insecure"


def test_specific_public_ip_without_key_is_insecure():
    level, _ = assess_public_bind_auth_posture(
        listen_host="203.0.113.10", api_key_present=False)
    assert level == "insecure"


def test_public_bind_with_key_is_ok():
    level, msg = assess_public_bind_auth_posture(
        listen_host="0.0.0.0", api_key_present=True)
    assert level == "ok"
    assert msg == ""


def test_loopback_without_key_is_ok():
    # Local-dev / reverse-proxy-fronted: 127.0.0.1 is not network-exposed.
    for host in ("127.0.0.1", "::1", "localhost"):
        level, _ = assess_public_bind_auth_posture(
            listen_host=host, api_key_present=False)
        assert level == "ok", f"{host} should be treated as loopback"


def test_loopback_case_insensitive_and_whitespace():
    level, _ = assess_public_bind_auth_posture(
        listen_host="  LocalHost ", api_key_present=False)
    assert level == "ok"


def test_start_wires_posture_check_failclosed_by_default():
    """Structural pin: node.start() runs the posture check and (sp1011) refuses
    to start in the insecure posture BY DEFAULT, with the explicit
    PRSM_ALLOW_INSECURE_PUBLIC_BIND ack as the only escape."""
    src = inspect.getsource(PRSMNode.start)
    assert "assess_public_bind_auth_posture(" in src
    assert "should_refuse_insecure_public_bind(" in src
    assert "PRSM_ALLOW_INSECURE_PUBLIC_BIND" in src
    assert "raise RuntimeError(" in src
    # The check must read the configured bind host + the API key presence.
    assert "listen_host=" in src
    assert "PRSM_NODE_API_KEY" in src
