"""Sprint 1306 — challenger-side fetch+verify for the §7 settlement data plane.

Makes the sp1305 serve endpoint actionable end to end: fetch a peer's retained §7
receipt by committed-batch leaf hash and run the §7 verifier on it. Fetch failures are
fail-LOUD (ReceiptFetchError) so a challenger never mistakes "couldn't fetch" for
"verified clean"; the verification result is a ChallengeReport.

The verifier + receipt reconstruction are monkeypatched so these tests isolate the
fetch/orchestration logic (the verifier has its own tests).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import prsm.settlement.receipt_challenge_client as rcc
from prsm.settlement.receipt_challenge_client import (
    ReceiptFetchError,
    _normalize_leaf,
    fetch_and_verify_receipt_for_leaf,
)

_LEAF = "ab" * 32


class _Resp:
    def __init__(self, status, body=None, *, raise_json=False, text=""):
        self.status_code = status
        self._body = body
        self._raise_json = raise_json
        self.text = text

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._body


def _ok_body(leaf=_LEAF, stage=None):
    return {
        "leaf_hash": leaf,
        "inference_receipt": {"job_id": "j1", "prompt_hash": "cd" * 32},
        "settler_public_key_b64": "PUBKEY",
        "stage_public_keys": stage,
        "retained_at": 1700000000,
    }


@pytest.fixture
def _patch_verify(monkeypatch):
    """Isolate from real crypto: receipt reconstruction + verifier are stubbed."""
    seen = {}

    class _FakeIR:
        @staticmethod
        def from_dict(d):
            return SimpleNamespace(settler_node_id="node-x", _d=d)

    def _fake_verify(receipt, *, settler_public_key_b64, stage_public_keys=None):
        seen["settler_pk"] = settler_public_key_b64
        seen["stage_keys"] = stage_public_keys
        seen["receipt"] = receipt
        return SimpleNamespace(receipt_ok=True, findings=[])

    monkeypatch.setattr(rcc, "InferenceReceipt", _FakeIR)
    monkeypatch.setattr(rcc, "verify_inference_receipt_for_challenge", _fake_verify)
    return seen


# ── leaf normalization ───────────────────────────────────────────────────────

def test_normalize_leaf_accepts_0x_and_lowercases():
    assert _normalize_leaf("0x" + ("AB" * 32)) == _LEAF


@pytest.mark.parametrize("bad", ["zz", "abcd", "ab" * 33, "xy" * 32])
def test_normalize_leaf_rejects_bad(bad):
    with pytest.raises(ValueError):
        _normalize_leaf(bad)


# ── happy path ───────────────────────────────────────────────────────────────

def test_fetch_and_verify_happy(_patch_verify):
    res = fetch_and_verify_receipt_for_leaf(
        "http://peer:8000/", _LEAF,
        http_get=lambda url, **k: _Resp(200, _ok_body()))
    assert res.receipt_ok is True
    assert res.leaf_hash == _LEAF
    assert res.settler_node_id == "node-x"
    assert res.retained_at == 1700000000
    assert _patch_verify["settler_pk"] == "PUBKEY"


def test_url_is_built_with_normalized_leaf(_patch_verify):
    seen = {}

    def _get(url, **k):
        seen["url"] = url
        return _Resp(200, _ok_body())

    fetch_and_verify_receipt_for_leaf("http://peer:8000", "0x" + ("AB" * 32),
                                      http_get=_get)
    assert seen["url"] == f"http://peer:8000/settlement/receipt/leaf/{_LEAF}"


def test_stage_keys_passed_through(_patch_verify):
    fetch_and_verify_receipt_for_leaf(
        "http://peer:8000", _LEAF,
        http_get=lambda url, **k: _Resp(200, _ok_body(stage={"n": "k"})))
    assert _patch_verify["stage_keys"] == {"n": "k"}


# ── fail-loud paths ──────────────────────────────────────────────────────────

def test_transport_error_raises(_patch_verify):
    def _boom(url, **k):
        raise OSError("connection refused")
    with pytest.raises(ReceiptFetchError, match="transport error"):
        fetch_and_verify_receipt_for_leaf("http://peer:8000", _LEAF, http_get=_boom)


def test_non_200_raises_with_detail(_patch_verify):
    r = _Resp(503, {"detail": "PRSM_SETTLEMENT_AUDIT off"})
    with pytest.raises(ReceiptFetchError, match="503"):
        fetch_and_verify_receipt_for_leaf("http://peer:8000", _LEAF,
                                          http_get=lambda url, **k: r)


def test_404_not_retained_raises(_patch_verify):
    r = _Resp(404, {"detail": "no retained receipt"})
    with pytest.raises(ReceiptFetchError):
        fetch_and_verify_receipt_for_leaf("http://peer:8000", _LEAF,
                                          http_get=lambda url, **k: r)


def test_non_json_raises(_patch_verify):
    r = _Resp(200, raise_json=True, text="<html>")
    with pytest.raises(ReceiptFetchError, match="non-JSON"):
        fetch_and_verify_receipt_for_leaf("http://peer:8000", _LEAF,
                                          http_get=lambda url, **k: r)


def test_missing_fields_raises(_patch_verify):
    r = _Resp(200, {"leaf_hash": _LEAF})  # no inference_receipt / settler key
    with pytest.raises(ReceiptFetchError, match="missing"):
        fetch_and_verify_receipt_for_leaf("http://peer:8000", _LEAF,
                                          http_get=lambda url, **k: r)


def test_served_leaf_mismatch_raises(_patch_verify):
    r = _Resp(200, _ok_body(leaf="cd" * 32))  # peer answered a DIFFERENT leaf
    with pytest.raises(ReceiptFetchError, match="!="):
        fetch_and_verify_receipt_for_leaf("http://peer:8000", _LEAF,
                                          http_get=lambda url, **k: r)


def test_malformed_receipt_raises(monkeypatch):
    class _BadIR:
        @staticmethod
        def from_dict(d):
            raise ValueError("bad receipt")
    monkeypatch.setattr(rcc, "InferenceReceipt", _BadIR)
    r = _Resp(200, _ok_body())
    with pytest.raises(ReceiptFetchError, match="malformed inference_receipt"):
        fetch_and_verify_receipt_for_leaf("http://peer:8000", _LEAF,
                                          http_get=lambda url, **k: r)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
