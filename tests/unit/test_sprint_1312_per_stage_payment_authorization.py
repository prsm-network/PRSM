"""Sprint 1312 — build_per_stage_payment_authorization (paid big-model multi-stage).

The client-side builder for PAID cross-host multi-stage inference: a requester authorizes
a SET of stage-node payees (each with a max share) rather than a single provider. The
VERIFIER (verify_per_stage_authorization, sp1172) + the per-node commit gate already exist
and are tested; this closes the missing client builder. These tests round-trip the builder
through the real verifier (so it's provably non-inert).

NOTE: the on-chain multi-stage SETTLEMENT that consumes this in the main inference path is a
separate, deferred money-path build (per-stage accumulation, per-node settler keys, worker
per-stage signatures live) — out of scope here.
"""
from __future__ import annotations

from decimal import Decimal

from eth_account import Account

from prsm.settlement.payment_client import build_per_stage_payment_authorization
from prsm.settlement.per_stage_payment_authorization import (
    verify_per_stage_authorization,
)

_A = "0x" + "a1" * 20
_B = "0x" + "b2" * 20


def _wei(ftns):
    return int(Decimal(str(ftns)) * (Decimal(10) ** 18))


def _build(req, payees_ftns, **kw):
    return build_per_stage_payment_authorization(
        requester_key=req.key.hex(), payees=payees_ftns,
        model_id="qwen2.5-72b", prompt="Hello", max_tokens=8,
        privacy_tier="none", content_tier="A", expiry_unix=9999999999, **kw)


def test_shape_and_fields():
    req = Account.create()
    d = _build(req, [(_A, 0.6), (_B, 0.4)])
    assert set(d) == {"payload", "signature"}
    p = d["payload"]
    assert p["requester"].lower() == req.address.lower()
    assert p["payee_set_hash"].startswith("0x") and len(p["payee_set_hash"]) == 66
    assert p["total_max_spend_wei"] == _wei(0.6) + _wei(0.4)   # defaults to sum of shares
    assert p["request_hash"] and p["job_nonce"].startswith("0x")
    assert d["signature"].startswith("0x")


def test_roundtrips_through_verifier_for_each_payee():
    """The built+signed auth must AUTHORIZE each (payee, share) under the real verifier."""
    req = Account.create()
    payees_ftns = [(_A, 0.6), (_B, 0.4)]
    d = _build(req, payees_ftns)
    payees_wei = [(a, _wei(s)) for a, s in payees_ftns]
    for payee, share in payees_wei:
        v = verify_per_stage_authorization(
            d["payload"], d["signature"], payees=payees_wei,
            payee=payee, share_wei=share, now_unix=1000.0)
        assert v.authorized is True, (payee, v.reason)


def test_verifier_rejects_non_member_payee():
    req = Account.create()
    payees_wei = [(_A, _wei(0.6)), (_B, _wei(0.4))]
    d = _build(req, [(_A, 0.6), (_B, 0.4)])
    stranger = "0x" + "cc" * 20
    v = verify_per_stage_authorization(
        d["payload"], d["signature"], payees=payees_wei,
        payee=stranger, share_wei=_wei(0.6), now_unix=1000.0)
    assert v.authorized is False


def test_verifier_rejects_wrong_share():
    req = Account.create()
    payees_wei = [(_A, _wei(0.6)), (_B, _wei(0.4))]
    d = _build(req, [(_A, 0.6), (_B, 0.4)])
    v = verify_per_stage_authorization(
        d["payload"], d["signature"], payees=payees_wei,
        payee=_A, share_wei=_wei(0.9), now_unix=1000.0)   # A's real share is 0.6
    assert v.authorized is False


def test_chain_id_bound():
    """Signed for 8453 must not verify under 84532 (signer recovery differs)."""
    req = Account.create()
    payees_wei = [(_A, _wei(0.6)), (_B, _wei(0.4))]
    d = _build(req, [(_A, 0.6), (_B, 0.4)], chain_id=8453)
    v = verify_per_stage_authorization(
        d["payload"], d["signature"], payees=payees_wei,
        payee=_A, share_wei=_wei(0.6), chain_id=84532, now_unix=1000.0)
    assert v.authorized is False


def test_order_independent_payee_set():
    """Same set in a different order → same payee_set_hash (so the server's re-derived set,
    in any order, matches the signed commitment)."""
    req = Account.create()
    d1 = _build(req, [(_A, 0.6), (_B, 0.4)], job_nonce="0x" + "11" * 32)
    d2 = _build(req, [(_B, 0.4), (_A, 0.6)], job_nonce="0x" + "11" * 32)
    assert d1["payload"]["payee_set_hash"] == d2["payload"]["payee_set_hash"]


def test_explicit_total_cap_honored():
    req = Account.create()
    d = _build(req, [(_A, 0.6), (_B, 0.4)], total_max_spend_ftns=2.0)
    assert d["payload"]["total_max_spend_wei"] == _wei(2.0)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
