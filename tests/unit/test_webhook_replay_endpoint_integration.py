"""Sprint 284 — webhook replay protection endpoint integration.

Layers the sprint-284 timestamp + dedup defenses on top of
the sprint-283 signature verification. Confirms:

  - Persona webhook with stale timestamp → 401 (rejected
    before state mutation)
  - Persona webhook with future timestamp → 401
  - Replayed signature (same v1 value twice) → 409 Conflict
  - Distinct signatures pass through unchanged
  - Replay defense bypassed when no secret configured
    (sprint-280 pass-through preserved)
  - PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1 bypasses replay
    defense too (consistent escape hatch)

Webhook-replay state MUST NOT mutate KYC records — the
security property is re-asserted here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from prsm.economy.web3.kyc_client import KYCClient
from prsm.economy.web3.webhook_replay_defense import (
    WebhookReplayRing,
)
from prsm.node.api import create_api_app


class FakeKYCBackend:
    def initiate_session(self, user_id, email, level):
        return {
            "vendor_ref": f"persona-{user_id}",
            "session_url": f"https://persona.example/v/{user_id}",
            "status": "INITIATED",
        }


def _client(kyc=None, replay_ring=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._kyc_client = kyc
    node._fiat_compliance_ring = None
    node._coinbase_waas_client = None
    # Real ring (not MagicMock) so dedup state actually works.
    node._kyc_webhook_replay_ring = (
        replay_ring
        if replay_ring is not None
        else WebhookReplayRing()
    )
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _commissioned_kyc(vendor="persona"):
    return KYCClient(
        vendor=vendor, api_key="k", backend=FakeKYCBackend(),
    )


def _seed_alice(kyc):
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")


def _persona_header(body_bytes, secret, ts):
    payload = f"{ts}.".encode("utf-8") + body_bytes
    sig = hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256,
    ).hexdigest()
    return f"t={ts},v1={sig}"


def _onfido_header(body_bytes, token):
    return hmac.new(
        token.encode("utf-8"), body_bytes, hashlib.sha256,
    ).hexdigest()


# ── Timestamp freshness ──────────────────────────────────


def test_persona_stale_timestamp_rejected(monkeypatch):
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc("persona")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    # Sign with a timestamp 1 hour in the past
    stale_ts = str(int(time.time()) - 3600)
    header = _persona_header(body, "wh_secret", stale_ts)
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": header,
        },
    )
    assert resp.status_code == 401
    # State must not have flipped
    assert kyc.get_status("alice").status == "INITIATED"


def test_persona_future_timestamp_rejected(monkeypatch):
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc("persona")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    future_ts = str(int(time.time()) + 3600)
    header = _persona_header(body, "wh_secret", future_ts)
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": header,
        },
    )
    assert resp.status_code == 401


def test_persona_fresh_timestamp_accepted(monkeypatch):
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc("persona")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    fresh_ts = str(int(time.time()))
    header = _persona_header(body, "wh_secret", fresh_ts)
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": header,
        },
    )
    assert resp.status_code == 200


def test_persona_tolerance_env_var_widens_window(monkeypatch):
    """PRSM_KYC_WEBHOOK_TIMESTAMP_TOLERANCE_SEC tunable for
    operators behind slow networks / batch processors."""
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    monkeypatch.setenv(
        "PRSM_KYC_WEBHOOK_TIMESTAMP_TOLERANCE_SEC", "7200",
    )
    kyc = _commissioned_kyc("persona")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    # 1 hour old → would fail default 300s but pass at 7200s
    old_ts = str(int(time.time()) - 3600)
    header = _persona_header(body, "wh_secret", old_ts)
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": header,
        },
    )
    assert resp.status_code == 200


# ── Replay dedup ─────────────────────────────────────────


def test_persona_replay_same_signature_rejected(monkeypatch):
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc("persona")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    fresh_ts = str(int(time.time()))
    header = _persona_header(body, "wh_secret", fresh_ts)
    cli = _client(kyc)
    # First post succeeds
    r1 = cli.post(
        "/wallet/kyc/webhook/persona",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": header,
        },
    )
    assert r1.status_code == 200
    # Identical replay → 409
    r2 = cli.post(
        "/wallet/kyc/webhook/persona",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": header,
        },
    )
    assert r2.status_code == 409


def test_onfido_replay_same_signature_rejected(monkeypatch):
    monkeypatch.setenv("ONFIDO_WEBHOOK_TOKEN", "wh_tok")
    kyc = _commissioned_kyc("onfido")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    header = _onfido_header(body, "wh_tok")
    cli = _client(kyc)
    r1 = cli.post(
        "/wallet/kyc/webhook/onfido",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-SHA2-Signature": header,
        },
    )
    assert r1.status_code == 200
    r2 = cli.post(
        "/wallet/kyc/webhook/onfido",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-SHA2-Signature": header,
        },
    )
    assert r2.status_code == 409


def _client_no_ring(kyc):
    """Client whose replay ring is None — simulates WebhookReplayRing
    construction failure (node.py:2867-2873 swallows it to None)."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._kyc_client = kyc
    node._fiat_compliance_ring = None
    node._coinbase_waas_client = None
    node._kyc_webhook_replay_ring = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_onfido_valid_webhook_fails_closed_when_replay_ring_none(monkeypatch):
    """sp969 — replay ring None (construction failed) + a secret configured →
    a VALID-signed Onfido webhook must FAIL CLOSED (503), not silently skip
    replay defense. Onfido carries no timestamp, so the ring is its ONLY replay
    defense; fail-open let a captured valid Onfido webhook be replayed at will."""
    monkeypatch.setenv("ONFIDO_WEBHOOK_TOKEN", "wh_tok")
    kyc = _commissioned_kyc("onfido")
    _seed_alice(kyc)
    body = json.dumps({"user_id": "alice", "status": "VERIFIED"}).encode("utf-8")
    header = _onfido_header(body, "wh_tok")
    r = _client_no_ring(kyc).post(
        "/wallet/kyc/webhook/onfido", content=body,
        headers={"Content-Type": "application/json", "X-SHA2-Signature": header},
    )
    assert r.status_code == 503


def test_forged_webhook_still_401_even_when_replay_ring_none(monkeypatch):
    """The fail-closed 503 must come AFTER signature verification: a FORGED
    webhook is still rejected at 401 (the sig check), never reaching the 503 —
    so a bad actor can't probe the 503 without a valid signature."""
    monkeypatch.setenv("ONFIDO_WEBHOOK_TOKEN", "wh_tok")
    kyc = _commissioned_kyc("onfido")
    _seed_alice(kyc)
    body = json.dumps({"user_id": "alice", "status": "VERIFIED"}).encode("utf-8")
    r = _client_no_ring(kyc).post(
        "/wallet/kyc/webhook/onfido", content=body,
        headers={"Content-Type": "application/json",
                 "X-SHA2-Signature": "deadbeef" * 8},  # forged
    )
    assert r.status_code == 401


def test_distinct_signatures_pass_through(monkeypatch):
    """Two webhooks with different signatures (different
    bodies → different HMACs) must both succeed."""
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc("persona")
    kyc.initiate(
        user_id="alice", email="a@x.io", level="basic",
    )
    kyc.initiate(
        user_id="bob", email="b@x.io", level="basic",
    )
    cli = _client(kyc)
    ts = str(int(time.time()))
    body1 = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    body2 = json.dumps({
        "user_id": "bob", "status": "VERIFIED",
    }).encode("utf-8")
    h1 = _persona_header(body1, "wh_secret", ts)
    h2 = _persona_header(body2, "wh_secret", ts)
    r1 = cli.post(
        "/wallet/kyc/webhook/persona",
        content=body1,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": h1,
        },
    )
    r2 = cli.post(
        "/wallet/kyc/webhook/persona",
        content=body2,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": h2,
        },
    )
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_replay_does_not_mutate_state(monkeypatch):
    """Re-assert the security property: a replayed webhook
    MUST NOT mutate state on the second hit (status was set
    to VERIFIED on first hit; subsequent change attempts
    via replay must be blocked even if the first hit
    succeeded)."""
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc("persona")
    _seed_alice(kyc)
    cli = _client(kyc)
    ts = str(int(time.time()))
    # First: status=VERIFIED
    body_verify = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    h_verify = _persona_header(body_verify, "wh_secret", ts)
    cli.post(
        "/wallet/kyc/webhook/persona",
        content=body_verify,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": h_verify,
        },
    )
    assert kyc.get_status("alice").status == "VERIFIED"
    # An attacker replays the same exact payload — should not
    # mutate anything (state is already VERIFIED but the
    # rejection prevents future-attacker scenarios where
    # they replay older webhooks to overwrite newer state).
    resp = cli.post(
        "/wallet/kyc/webhook/persona",
        content=body_verify,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": h_verify,
        },
    )
    assert resp.status_code == 409
    assert kyc.get_status("alice").status == "VERIFIED"


# ── Bypass paths preserved ───────────────────────────────


def test_no_secret_fails_closed(monkeypatch):
    """Sp888 — with no secret AND no explicit dev-bypass, the
    endpoint FAILS CLOSED (503). Pre-sp888 this passed through
    unsigned (200, no replay defense) — the fail-open hole. Now
    an operator must either configure the secret (enabling both
    signature + replay defense) or set the explicit dev-bypass."""
    monkeypatch.delenv("PERSONA_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv(
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED", raising=False,
    )
    kyc = _commissioned_kyc("persona")
    _seed_alice(kyc)
    cli = _client(kyc)
    payload = {"user_id": "alice", "status": "VERIFIED"}
    r1 = cli.post(
        "/wallet/kyc/webhook/persona", json=payload,
    )
    assert r1.status_code == 503
    # Unsigned webhook must not have mutated KYC state.
    assert kyc.is_verified("alice") is False


def test_disable_flag_bypasses_replay_defense(monkeypatch):
    """The verify-disabled escape hatch bypasses replay
    defense too — single consistent switch for all webhook
    security."""
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    monkeypatch.setenv(
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED", "1",
    )
    kyc = _commissioned_kyc("persona")
    _seed_alice(kyc)
    cli = _client(kyc)
    payload = {"user_id": "alice", "status": "VERIFIED"}
    r1 = cli.post(
        "/wallet/kyc/webhook/persona", json=payload,
    )
    r2 = cli.post(
        "/wallet/kyc/webhook/persona", json=payload,
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
