"""Sprint 1094 — relayer ingress wiring (requester-payment relayer, brick 4).

Wires the sp1093 RelayerAuthorizationVerifier into the inference ingress: a request
carrying a ``payment_delegation`` (the funder's signed delegation) alongside the
relayer-signed ``payment_authorization`` routes to the relayer verifier, and the
authenticated FUNDER is threaded as the paying requester. Tests the client_wiring
resolve/build helpers (the load-bearing, node-free pieces).
"""
from __future__ import annotations

import pytest
from eth_account import Account

from prsm.settlement.client_wiring import (
    build_relayer_verifier_or_none,
    resolve_relayer_requester,
)
from prsm.settlement.payment_authorization import (
    canonical_request_hash, inference_request_fields, sign_payment_authorization,
)
from prsm.settlement.payment_authorization_verifier import (
    AuthorizationRejected, RelayerAuthorizationVerifier,
)
from prsm.settlement.payment_delegation import sign_payment_delegation

_CHAIN = 8453
_FUNDER = Account.create()
_RELAYER = Account.create()
_PROVIDER = Account.create()


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _rhash():
    return canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="hi", max_tokens=8, privacy_tier="none", content_tier="A"))


def _signed_pair(rhash):
    d = {"requester": _FUNDER.address, "relayer": _RELAYER.address,
         "max_total_spend_wei": 10**18, "delegation_nonce": "0x" + "cd" * 32,
         "expiry_unix": 9999999999}
    a = {"requester": _FUNDER.address, "provider": _PROVIDER.address,
         "max_spend_wei": 10**17, "job_nonce": "0x" + "ab" * 32,
         "expiry_unix": 9999999999, "request_hash": rhash}
    auth = {"payload": a, "signature": "0x" + sign_payment_authorization(
        a, _RELAYER.key, chain_id=_CHAIN).hex()}
    deleg = {"payload": d, "signature": "0x" + sign_payment_delegation(
        d, _FUNDER.key, chain_id=_CHAIN).hex()}
    return auth, deleg


def _verifier():
    return RelayerAuthorizationVerifier(
        provider_address=_PROVIDER.address, chain_id=_CHAIN, now=lambda: 100.0)


# ── resolve_relayer_requester ────────────────────────────────────────────────────

def test_no_delegation_returns_none():
    out = _run(resolve_relayer_requester(
        verifier=_verifier(), payment_authorization={"payload": {}, "signature": "0x"},
        payment_delegation=None, request_hash=_rhash(), quoted_price_wei=1))
    assert out is None   # caller falls back to the self-signed path


def test_valid_chain_resolves_funder():
    rh = _rhash()
    auth, deleg = _signed_pair(rh)
    funder = _run(resolve_relayer_requester(
        verifier=_verifier(), payment_authorization=auth, payment_delegation=deleg,
        request_hash=rh, quoted_price_wei=10**16))
    assert funder.lower() == _FUNDER.address.lower()


def test_delegation_without_verifier_rejected():
    rh = _rhash()
    auth, deleg = _signed_pair(rh)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(resolve_relayer_requester(
            verifier=None, payment_authorization=auth, payment_delegation=deleg,
            request_hash=rh, quoted_price_wei=1))
    assert ei.value.reason == "verifier-not-wired"


def test_malformed_delegation_rejected():
    rh = _rhash()
    auth, _ = _signed_pair(rh)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(resolve_relayer_requester(
            verifier=_verifier(), payment_authorization=auth,
            payment_delegation={"nope": 1}, request_hash=rh, quoted_price_wei=1))
    assert ei.value.reason == "malformed-delegation"


def test_wrong_request_hash_rejected():
    rh = _rhash()
    auth, deleg = _signed_pair(rh)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(resolve_relayer_requester(
            verifier=_verifier(), payment_authorization=auth, payment_delegation=deleg,
            request_hash="0x" + "99" * 32, quoted_price_wei=10**16))
    assert ei.value.reason == "request-hash-mismatch"


# ── build_relayer_verifier_or_none gating ────────────────────────────────────────

def test_build_off_when_flag_unset():
    assert build_relayer_verifier_or_none(
        {}, provider_address=_PROVIDER.address) is None


def test_build_off_when_no_provider():
    assert build_relayer_verifier_or_none(
        {"PRSM_REQUESTER_PAYMENT": "1"}, provider_address=None) is None


def test_build_on_when_enabled():
    v = build_relayer_verifier_or_none(
        {"PRSM_REQUESTER_PAYMENT": "1", "PRSM_RELAYER_NONCE_FILE": ":memory:",
         "PRSM_DELEGATION_BUDGET_FILE": ":memory:"},
        provider_address=_PROVIDER.address)
    assert isinstance(v, RelayerAuthorizationVerifier)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
