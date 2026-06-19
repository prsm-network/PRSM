"""Sprint 280 — KYC HTTP endpoints.

Operator + LLM-facing surface for the KYCClient. Pre-commission
returns PENDING_COMMISSION records (preview only). Post-
commission delegates to the configured vendor backend.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.kyc_client import (
    KYCClient,
    KYC_STATUS_VERIFIED, KYC_STATUS_REJECTED,
    KYC_STATUS_EXPIRED,
)
from prsm.node.api import create_api_app


class FakeBackend:
    def initiate_session(self, user_id, email, level):
        return {
            "vendor_ref": f"persona-{user_id}",
            "session_url": f"https://persona.example/v/{user_id}",
            "status": "INITIATED",
        }


def _client(kyc=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._kyc_client = kyc
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _commissioned_kyc():
    return KYCClient(
        vendor="persona", api_key="k", backend=FakeBackend(),
    )


# ── POST /wallet/kyc/initiate ────────────────────────────


def test_initiate_503_when_unwired():
    resp = _client(None).post(
        "/wallet/kyc/initiate",
        json={
            "user_id": "alice", "email": "a@x.io",
            "level": "basic",
        },
    )
    assert resp.status_code == 503


def test_initiate_pending_commission_when_uncommissioned():
    resp = _client(KYCClient()).post(
        "/wallet/kyc/initiate",
        json={
            "user_id": "alice", "email": "a@x.io",
            "level": "basic",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PENDING_COMMISSION"
    assert body["vendor_ref"] is None


def test_initiate_happy_path():
    resp = _client(_commissioned_kyc()).post(
        "/wallet/kyc/initiate",
        json={
            "user_id": "alice", "email": "a@x.io",
            "level": "basic",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "INITIATED"
    assert body["vendor"] == "persona"
    assert body["vendor_ref"] == "persona-alice"
    assert "persona.example/v/alice" in body["session_url"]


def test_initiate_422_missing_fields():
    resp = _client(_commissioned_kyc()).post(
        "/wallet/kyc/initiate",
        json={"user_id": "alice"},
    )
    assert resp.status_code == 422


def test_initiate_422_invalid_level():
    resp = _client(_commissioned_kyc()).post(
        "/wallet/kyc/initiate",
        json={
            "user_id": "alice", "email": "a@x.io",
            "level": "ultra",
        },
    )
    assert resp.status_code == 422


def test_initiate_idempotent_when_active():
    cli = _client(_commissioned_kyc())
    r1 = cli.post(
        "/wallet/kyc/initiate",
        json={
            "user_id": "alice", "email": "a@x.io",
            "level": "basic",
        },
    )
    r2 = cli.post(
        "/wallet/kyc/initiate",
        json={
            "user_id": "alice", "email": "a@x.io",
            "level": "basic",
        },
    )
    assert r1.json()["vendor_ref"] == r2.json()["vendor_ref"]


# ── GET /wallet/kyc/{user_id} ────────────────────────────


def test_get_one_503_when_unwired():
    resp = _client(None).get("/wallet/kyc/alice")
    assert resp.status_code == 503


def test_get_one_404_when_missing():
    resp = _client(_commissioned_kyc()).get("/wallet/kyc/ghost")
    assert resp.status_code == 404


def test_get_one_happy_path():
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    resp = _client(kyc).get("/wallet/kyc/alice")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "alice"
    assert body["status"] == "INITIATED"


# ── GET /wallet/kyc ──────────────────────────────────────


def test_list_503_when_unwired():
    resp = _client(None).get("/wallet/kyc")
    assert resp.status_code == 503


def test_list_empty():
    resp = _client(_commissioned_kyc()).get("/wallet/kyc")
    body = resp.json()
    assert body["records"] == []
    assert body["count"] == 0


def test_list_populated():
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    kyc.initiate(user_id="bob", email="b@x.io", level="enhanced")
    resp = _client(kyc).get("/wallet/kyc")
    body = resp.json()
    assert body["count"] == 2


def test_list_invalid_limit():
    resp = _client(_commissioned_kyc()).get("/wallet/kyc?limit=0")
    assert resp.status_code == 422


# ── GET /wallet/kyc/status ───────────────────────────────


def test_status_503_when_unwired():
    resp = _client(None).get("/wallet/kyc/status")
    assert resp.status_code == 503


def test_status_uncommissioned():
    resp = _client(KYCClient()).get("/wallet/kyc/status")
    body = resp.json()
    assert body["commissioned"] is False


def test_status_commissioned():
    resp = _client(_commissioned_kyc()).get("/wallet/kyc/status")
    body = resp.json()
    assert body["commissioned"] is True
    assert body["vendor"] == "persona"


def test_status_supported_vendors_listed():
    resp = _client(KYCClient()).get("/wallet/kyc/status")
    body = resp.json()
    assert "persona" in body["supported_vendors"]
    assert "onfido" in body["supported_vendors"]


# ── POST /wallet/kyc/webhook/{vendor} ────────────────────


@pytest.fixture(autouse=True)
def _kyc_webhook_dev_bypass(monkeypatch):
    """Sp888 — the webhook endpoint now FAILS CLOSED on unsigned
    input. These tests exercise body-parsing / state-transition
    logic (not signature auth), so they use the explicit dev/test
    bypass (the correct pattern: prod requires a secret, dev sets
    this flag). Signature-verification behavior is covered by
    test_kyc_webhook_signature_integration.py."""
    monkeypatch.setenv("PRSM_KYC_WEBHOOK_VERIFY_DISABLED", "1")
    yield


def test_webhook_503_when_unwired():
    resp = _client(None).post(
        "/wallet/kyc/webhook/persona",
        json={"user_id": "alice", "status": "VERIFIED"},
    )
    assert resp.status_code == 503


def test_webhook_happy_path_verified():
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        json={"user_id": "alice", "status": "VERIFIED"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "VERIFIED"
    assert kyc.is_verified("alice") is True


def test_webhook_happy_path_rejected():
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        json={"user_id": "alice", "status": "REJECTED"},
    )
    assert resp.status_code == 200
    assert kyc.is_verified("alice") is False


def test_webhook_404_unknown_user():
    resp = _client(_commissioned_kyc()).post(
        "/wallet/kyc/webhook/persona",
        json={"user_id": "ghost", "status": "VERIFIED"},
    )
    assert resp.status_code == 404


def test_webhook_422_invalid_status():
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        json={"user_id": "alice", "status": "BOGUS"},
    )
    assert resp.status_code == 422


def test_webhook_422_missing_user_id():
    resp = _client(_commissioned_kyc()).post(
        "/wallet/kyc/webhook/persona",
        json={"status": "VERIFIED"},
    )
    assert resp.status_code == 422


def test_webhook_with_vendor_ref_update():
    kyc = _commissioned_kyc()
    rec = kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    # sp1174 — a real Persona inquiry.approved webhook carries the SAME inquiry's id as
    # vendor_ref (== the live record's), so it verifies the record (vendor_ref unchanged).
    # (Pre-sp1174 this posted a FABRICATED mismatched ref 'persona-final-ref' + asserted it
    # overwrote vendor_ref — exactly the stale-inquiry-verifies hole the guard now closes.)
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        json={
            "user_id": "alice", "status": "VERIFIED",
            "vendor_ref": rec.vendor_ref,
        },
    )
    body = resp.json()
    assert body["vendor_ref"] == rec.vendor_ref
    assert kyc.is_verified("alice")
