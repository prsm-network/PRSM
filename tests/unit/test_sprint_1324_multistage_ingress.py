"""Sprint 1324 (S3b ingress) — paid multi-stage request ingress: carry + authenticate the
per-stage PaymentAuthorization at request time.

A paid big-model (multi-stage) request carries a per-stage auth instead of the single-stage one.
_resolve_paid_requester_or_402 authenticates the requester (signer == payload.requester) + CARRIES
the auth (the FULL money gate runs at each node's commit, sp1316, against the served payee set) —
gated by PRSM_MULTISTAGE_SETTLEMENT (off → ignored, self-pay, proven path unchanged). Plus a unit
test of the public recover_per_stage_signer helper.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from eth_account import Account
from eth_utils import keccak

from prsm.node.api import _resolve_paid_requester_or_402
from prsm.settlement.per_stage_payment_authorization import (
    DEFAULT_CHAIN_ID,
    InvalidSignatureFormat,
    compute_payee_set_hash,
    recover_per_stage_signer,
    sign_per_stage_authorization,
)

_REQ_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
_REQ_ADDR = Account.from_key(_REQ_KEY).address
_PAYEE_A = "0x" + "a1" * 20
_PAYEE_B = "0x" + "b2" * 20
_WEI = 10 ** 18


def _payload(*, requester=_REQ_ADDR, cap=2 * _WEI):
    payees = [(_PAYEE_A, _WEI), (_PAYEE_B, _WEI)]
    return {
        "requester": requester,
        "payee_set_hash": "0x" + compute_payee_set_hash(payees).hex(),
        "total_max_spend_wei": cap,
        "job_nonce": "0x" + keccak(b"n-1324").hex(),
        "expiry_unix": int(time.time()) + 86400,
        "request_hash": "0x" + keccak(b"r-1324").hex(),
    }


def _auth(*, key=_REQ_KEY, **kw):
    p = _payload(**kw)
    sig = sign_per_stage_authorization(p, key, chain_id=DEFAULT_CHAIN_ID)
    return {"payload": p, "signature": "0x" + sig.hex()}


def _body(auth):
    return {"model_id": "qwen2.5-72b", "prompt": "Hi",
            "per_stage_payment_authorization": auth}


# ── recover_per_stage_signer ──────────────────────────────────────────────────

def test_recover_signer_matches_requester():
    auth = _auth()
    assert recover_per_stage_signer(
        auth["payload"], auth["signature"]).lower() == _REQ_ADDR.lower()


def test_recover_signer_malformed_sig_raises():
    auth = _auth()
    with pytest.raises(InvalidSignatureFormat):
        recover_per_stage_signer(auth["payload"], "0xdeadbeef")


# ── ingress resolver: multi-stage branch ──────────────────────────────────────

def _resolve(body, monkeypatch, *, gate_on):
    if gate_on:
        monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    else:
        monkeypatch.delenv("PRSM_MULTISTAGE_SETTLEMENT", raising=False)
    node = MagicMock()
    return asyncio.run(_resolve_paid_requester_or_402(node, body, 2.0))


def test_gate_off_ignores_per_stage_auth(monkeypatch):
    # gate off → per-stage auth ignored, falls through to self-pay (None)
    assert _resolve(_body(_auth()), monkeypatch, gate_on=False) is None


def test_gate_on_carries_authenticated_auth(monkeypatch):
    info = _resolve(_body(_auth()), monkeypatch, gate_on=True)
    assert info is not None
    assert info["multi_stage"] is True
    assert info["requester"].lower() == _REQ_ADDR.lower()
    assert info["max_spend_wei"] == 2 * _WEI
    assert info["per_stage_authorization"]["payload"]["requester"] == _REQ_ADDR


def test_signer_mismatch_rejected_402(monkeypatch):
    # signed by a DIFFERENT key than payload.requester → 402
    other = Account.create().key.hex()
    bad = _auth(key=other)  # payload.requester=_REQ_ADDR but signed by `other`
    with pytest.raises(Exception) as ei:
        _resolve(_body(bad), monkeypatch, gate_on=True)
    assert getattr(ei.value, "status_code", None) == 402


def test_malformed_auth_rejected_402(monkeypatch):
    with pytest.raises(Exception) as ei:
        _resolve({"per_stage_payment_authorization": {"payload": {}, "signature": "0x12"}},
                 monkeypatch, gate_on=True)
    assert getattr(ei.value, "status_code", None) == 402


def test_no_auth_at_all_returns_none(monkeypatch):
    assert _resolve({"model_id": "m", "prompt": "p"}, monkeypatch, gate_on=True) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
