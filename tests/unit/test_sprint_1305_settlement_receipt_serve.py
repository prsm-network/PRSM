"""Sprint 1305 — §7 settlement-receipt data plane: serve by leaf hash.

sp1141 retained §7 InferenceReceipts locally (keyed by the committed-batch leaf hash,
identical to what an observer reads on-chain), but exposed NO fetch path — so a
challenger holding a suspicious on-chain leaf could not obtain the receipt to run
verify_inference_receipt_for_challenge(). sp1305 adds GET
/settlement/receipt/leaf/{leaf_hash}, closing the data-plane gap.

The served record is verification metadata only (signatures + hashes + topology — the
receipt carries no plaintext), so the endpoint is read-only + ungated like the other
/settlement/* read routes, gated only by the audit data plane existing.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app

_LEAF = "ab" * 32  # 32 bytes / 64 hex chars


class _FakeRec:
    def to_dict(self):
        return {
            "leaf_hash": _LEAF,
            "inference_receipt": {"job_id": "j1", "prompt_hash": "cd" * 32,
                                  "output_hash": "ef" * 32, "settler_signature": "sig"},
            "settler_public_key_b64": "PUBKEY",
            "stage_public_keys": {"node-a": "k1"},
            "retained_at": 1700000000,
        }


class _FakeStore:
    """Duck-typed: the endpoint only calls .get(bytes) → record with .to_dict()."""
    def __init__(self, has_leaf):
        self._has = has_leaf

    def get(self, leaf: bytes):
        return _FakeRec() if (self._has and leaf == bytes.fromhex(_LEAF)) else None


def _client(store):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._settlement_inference_receipt_store = store
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def test_503_when_store_absent():
    r = _client(None).get(f"/settlement/receipt/leaf/{_LEAF}")
    assert r.status_code == 503
    assert "PRSM_SETTLEMENT_AUDIT" in r.json()["detail"]


def test_422_on_bad_hex():
    r = _client(_FakeStore(has_leaf=True)).get("/settlement/receipt/leaf/zzzz")
    assert r.status_code == 422
    assert "hex" in r.json()["detail"].lower()


def test_422_on_wrong_length():
    # valid hex but only 4 bytes, not 32
    r = _client(_FakeStore(has_leaf=True)).get("/settlement/receipt/leaf/abcd1234")
    assert r.status_code == 422
    assert "32 bytes" in r.json()["detail"]


def test_404_when_leaf_not_retained():
    r = _client(_FakeStore(has_leaf=False)).get(f"/settlement/receipt/leaf/{_LEAF}")
    assert r.status_code == 404
    assert _LEAF in r.json()["detail"]


def test_200_serves_retained_receipt():
    r = _client(_FakeStore(has_leaf=True)).get(f"/settlement/receipt/leaf/{_LEAF}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["leaf_hash"] == _LEAF
    assert body["settler_public_key_b64"] == "PUBKEY"
    assert body["inference_receipt"]["prompt_hash"] == "cd" * 32
    assert body["stage_public_keys"] == {"node-a": "k1"}


def test_0x_prefixed_leaf_accepted():
    r = _client(_FakeStore(has_leaf=True)).get(f"/settlement/receipt/leaf/0x{_LEAF}")
    assert r.status_code == 200, r.text
    assert r.json()["leaf_hash"] == _LEAF


def test_served_record_carries_no_plaintext():
    """Guard the privacy contract: the served receipt exposes hashes + signatures,
    never a plaintext prompt/output field."""
    body = _client(_FakeStore(has_leaf=True)).get(
        f"/settlement/receipt/leaf/{_LEAF}").json()
    rec = body["inference_receipt"]
    assert "prompt" not in rec and "output_text" not in rec and "plaintext" not in rec
    assert rec["prompt_hash"] and rec["output_hash"]  # hashes only


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
