"""Sprint 1091 — relayer delegated-authorization (requester-payment follow-on, brick 1).

The sp1054-1059 self-signed model requires the paying requester to sign EVERY request.
This adds the relayer/gateway primitive: a funding requester signs ONE EIP-712
``PaymentDelegation`` authorizing a relayer to sign per-request ``PaymentAuthorization``s
on its behalf, bounded by a cumulative cap + expiry. End-users then transact through the
gateway (relayer) without each holding a wallet / signing every request — while the
funder keeps custody (escrow) + caps.

This brick: the PaymentDelegation EIP-712 type (sign/verify/hash/expiry) + the PURE
delegated-auth chain verifier (signatures + binding + expiry + per-request cap). Stateful
cumulative-spend tracking + ingress wiring are later bricks.
"""
from __future__ import annotations

import pytest
from eth_account import Account

from prsm.settlement.payment_authorization import (
    sign_payment_authorization,
)
from prsm.settlement.payment_delegation import (
    DelegatedAuthError,
    encode_delegation_typed_data_hash,
    is_delegation_expired,
    sign_payment_delegation,
    verify_delegated_authorization,
    verify_payment_delegation_signature,
)

_CHAIN = 8453
_FUNDER = Account.create()        # the requester who funds escrow + delegates
_RELAYER = Account.create()       # the gateway authorized to sign on the funder's behalf
_PROVIDER = "0x" + "11" * 20
_NONCE = "0x" + "ab" * 32
_DNONCE = "0x" + "cd" * 32
_RHASH = "0x" + "ef" * 32


def _delegation(*, requester=None, relayer=None, provider=None, max_total=10**18, expiry=9999999999):
    return {
        "requester": requester or _FUNDER.address,
        "relayer": relayer or _RELAYER.address,
        # sp1437 — the delegation is now bound to one provider; default to the same provider
        # the _auth() helper names so the chain verifies.
        "provider": provider or _PROVIDER,
        "max_total_spend_wei": max_total,
        "delegation_nonce": _DNONCE,
        "expiry_unix": expiry,
    }


def _auth(*, requester=None, max_spend=10**17, expiry=9999999999):
    return {
        "requester": requester or _FUNDER.address,
        "provider": _PROVIDER,
        "max_spend_wei": max_spend,
        "job_nonce": _NONCE,
        "expiry_unix": expiry,
        "request_hash": _RHASH,
    }


# ── EIP-712 delegation primitive ─────────────────────────────────────────────────

def test_delegation_sign_and_recover_roundtrip():
    d = _delegation()
    sig = sign_payment_delegation(d, _FUNDER.key, chain_id=_CHAIN)
    assert verify_payment_delegation_signature(d, sig, chain_id=_CHAIN).lower() == _FUNDER.address.lower()


def test_delegation_hash_binds_chain_id():
    d = _delegation()
    assert encode_delegation_typed_data_hash(d, chain_id=8453) != \
        encode_delegation_typed_data_hash(d, chain_id=84532)


def test_delegation_tampered_payload_recovers_different_signer():
    d = _delegation()
    sig = sign_payment_delegation(d, _FUNDER.key, chain_id=_CHAIN)
    tampered = dict(d, max_total_spend_wei=10**24)
    assert verify_payment_delegation_signature(tampered, sig, chain_id=_CHAIN).lower() != _FUNDER.address.lower()


def test_is_delegation_expired():
    assert is_delegation_expired(_delegation(expiry=1), now=lambda: 2.0) is True
    assert is_delegation_expired(_delegation(expiry=10), now=lambda: 2.0) is False


# ── delegated-auth chain verification (the relayer primitive) ────────────────────

def test_valid_delegated_chain_returns_funder():
    d = _delegation()
    a = _auth()
    dsig = sign_payment_delegation(d, _FUNDER.key, chain_id=_CHAIN)
    asig = sign_payment_authorization(a, _RELAYER.key, chain_id=_CHAIN)   # RELAYER signs
    funder = verify_delegated_authorization(
        auth_payload=a, auth_signature=asig,
        delegation_payload=d, delegation_signature=dsig,
        chain_id=_CHAIN, now=lambda: 100.0)
    assert funder.lower() == _FUNDER.address.lower()


def test_rejects_auth_not_signed_by_relayer():
    """The per-request auth must be signed by the delegated relayer, not a stranger."""
    d = _delegation()
    a = _auth()
    dsig = sign_payment_delegation(d, _FUNDER.key, chain_id=_CHAIN)
    stranger = Account.create()
    asig = sign_payment_authorization(a, stranger.key, chain_id=_CHAIN)
    with pytest.raises(DelegatedAuthError):
        verify_delegated_authorization(
            auth_payload=a, auth_signature=asig, delegation_payload=d,
            delegation_signature=dsig, chain_id=_CHAIN, now=lambda: 100.0)


def test_rejects_delegation_not_signed_by_requester():
    """The delegation must be signed by the funding requester it names."""
    d = _delegation()
    a = _auth()
    attacker = Account.create()
    dsig = sign_payment_delegation(d, attacker.key, chain_id=_CHAIN)   # not the funder
    asig = sign_payment_authorization(a, _RELAYER.key, chain_id=_CHAIN)
    with pytest.raises(DelegatedAuthError):
        verify_delegated_authorization(
            auth_payload=a, auth_signature=asig, delegation_payload=d,
            delegation_signature=dsig, chain_id=_CHAIN, now=lambda: 100.0)


def test_rejects_auth_requester_mismatch_delegation():
    """The auth's funding requester must equal the delegation's requester."""
    d = _delegation()
    other = Account.create()
    a = _auth(requester=other.address)   # names a different funder than the delegation
    dsig = sign_payment_delegation(d, _FUNDER.key, chain_id=_CHAIN)
    asig = sign_payment_authorization(a, _RELAYER.key, chain_id=_CHAIN)
    with pytest.raises(DelegatedAuthError):
        verify_delegated_authorization(
            auth_payload=a, auth_signature=asig, delegation_payload=d,
            delegation_signature=dsig, chain_id=_CHAIN, now=lambda: 100.0)


def test_rejects_expired_delegation():
    d = _delegation(expiry=50)
    a = _auth()
    dsig = sign_payment_delegation(d, _FUNDER.key, chain_id=_CHAIN)
    asig = sign_payment_authorization(a, _RELAYER.key, chain_id=_CHAIN)
    with pytest.raises(DelegatedAuthError):
        verify_delegated_authorization(
            auth_payload=a, auth_signature=asig, delegation_payload=d,
            delegation_signature=dsig, chain_id=_CHAIN, now=lambda: 100.0)


def test_rejects_per_request_exceeding_delegation_cap():
    d = _delegation(max_total=10**17)
    a = _auth(max_spend=10**18)   # one request exceeds the whole delegation budget
    dsig = sign_payment_delegation(d, _FUNDER.key, chain_id=_CHAIN)
    asig = sign_payment_authorization(a, _RELAYER.key, chain_id=_CHAIN)
    with pytest.raises(DelegatedAuthError):
        verify_delegated_authorization(
            auth_payload=a, auth_signature=asig, delegation_payload=d,
            delegation_signature=dsig, chain_id=_CHAIN, now=lambda: 100.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
