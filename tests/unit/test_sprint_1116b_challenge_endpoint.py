"""Sprint 1116 brick 2 — POST /settlement/challenge/verify exposes the off-chain
challenge verifier so an auditor can submit a receipt + keys and get a fraud verdict.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.receipt import sign_receipt
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.api import create_api_app
from prsm.node.identity import generate_node_identity


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def _signed_receipt(settler):
    r = InferenceReceipt(
        job_id="job-1", request_id="req-1", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=b"\xab" * 32, duration_seconds=1.0, cost_ftns=Decimal("0.1"),
        settler_node_id="")
    return sign_receipt(r, settler)


def test_clean_receipt_verdict_ok():
    c = _client()
    settler = generate_node_identity("settler")
    r = _signed_receipt(settler)
    resp = c.post("/settlement/challenge/verify", json={
        "receipt": r.to_dict(),
        "settler_public_key_b64": settler.public_key_b64,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["receipt_ok"] is True
    assert body["on_chain_actionable"] == []


def test_wrong_key_verdict_flags_invalid_signature_actionable():
    c = _client()
    settler = generate_node_identity("settler")
    r = _signed_receipt(settler)
    imposter = generate_node_identity("imposter")
    resp = c.post("/settlement/challenge/verify", json={
        "receipt": r.to_dict(),
        "settler_public_key_b64": imposter.public_key_b64,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["receipt_ok"] is False
    # the invalid settler signature is on-chain-actionable (ReasonCode.INVALID_SIGNATURE)
    assert any(f["on_chain_reason"] == 1 for f in body["on_chain_actionable"])


def test_missing_receipt_is_422():
    c = _client()
    resp = c.post("/settlement/challenge/verify", json={"settler_public_key_b64": "x"})
    assert resp.status_code == 422


def test_missing_settler_key_is_422():
    c = _client()
    resp = c.post("/settlement/challenge/verify", json={"receipt": {}})
    assert resp.status_code == 422


def test_malformed_receipt_is_422():
    c = _client()
    settler = generate_node_identity("settler")
    resp = c.post("/settlement/challenge/verify", json={
        "receipt": {"not": "a valid receipt"},
        "settler_public_key_b64": settler.public_key_b64,
    })
    assert resp.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
