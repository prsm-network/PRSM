"""Sprint 1011 — public-bind + no-auth posture is fail-closed BY DEFAULT.

The API-authz hunt (workflow wt1tb4n3q, finding 5, HIGH) confirmed the systemic
root cause behind the whole unauthenticated-endpoint class: NodeAuthMiddleware
only enforces when PRSM_NODE_API_KEY is set, so a node bound to a non-loopback
interface WITHOUT a key exposes the money + KYC endpoints unauthenticated. sp960
classified this posture but only WARNED by default (refusing only under the
opt-in PRSM_REQUIRE_AUTH_ON_PUBLIC_BIND).

sp1011 flips the default: the node REFUSES to start in the insecure
public-bind/no-auth posture unless the operator explicitly acknowledges running
unauthenticated via PRSM_ALLOW_INSECURE_PUBLIC_BIND (e.g. a firewall-fronted
deployment). This is non-disruptive to the common case — the CLI binds 127.0.0.1
by default (a loopback "ok" posture) — so only a deliberate non-loopback bind
without a key is affected, which is exactly the dangerous configuration.
"""
from __future__ import annotations

import inspect

from prsm.node.node import (
    PRSMNode,
    assess_public_bind_auth_posture,
    should_refuse_insecure_public_bind,
)


def test_ok_posture_never_refuses():
    assert should_refuse_insecure_public_bind("ok", allow_insecure=False) is False
    assert should_refuse_insecure_public_bind("ok", allow_insecure=True) is False


def test_insecure_refuses_by_default():
    assert should_refuse_insecure_public_bind("insecure", allow_insecure=False) is True


def test_insecure_with_explicit_ack_proceeds():
    assert should_refuse_insecure_public_bind("insecure", allow_insecure=True) is False


def test_classifier_still_marks_public_no_key_insecure():
    # The keystone precondition: public bind + no key is insecure...
    level, _ = assess_public_bind_auth_posture(listen_host="0.0.0.0", api_key_present=False)
    assert level == "insecure"
    # ...and now refuses by default.
    assert should_refuse_insecure_public_bind(level, allow_insecure=False) is True


def test_loopback_default_unaffected():
    # The CLI default bind is loopback → "ok" → never refuses (common case safe).
    level, _ = assess_public_bind_auth_posture(listen_host="127.0.0.1", api_key_present=False)
    assert level == "ok"
    assert should_refuse_insecure_public_bind(level, allow_insecure=False) is False


def test_start_is_failclosed_by_default_with_ack_escape():
    """Structural pin: start() refuses on the insecure posture by default and the
    escape is the explicit PRSM_ALLOW_INSECURE_PUBLIC_BIND ack."""
    src = inspect.getsource(PRSMNode.start)
    assert "should_refuse_insecure_public_bind(" in src
    assert "PRSM_ALLOW_INSECURE_PUBLIC_BIND" in src
    assert "raise RuntimeError(" in src
