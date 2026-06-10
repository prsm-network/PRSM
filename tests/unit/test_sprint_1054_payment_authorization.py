"""Sprint 1054 — PaymentAuthorization EIP-712 type + sign/verify (brick 1).

The requester-payment model (docs/2026-06-10-requester-payment-model-proposal.md),
self-signed end-user-wallet path. A paying requester signs a per-request
``PaymentAuthorization`` with the eth key that controls their on-chain EscrowPool
balance; the provider recovers the signer and binds the settled receipt to the
authenticated requester address.

This brick is PURE — hermetic tests with generated eth keys, no network, mirroring
the sprint-555 ``withdraw_signature`` primitive. Domain "PRSM-Payment" / "v1" /
Base mainnet (8453).

Payload shape:
  PaymentAuthorization {
    address requester      // EscrowPool depositor; settleFromRequester source
    address provider       // the node being paid; pins the auth to THIS provider
    uint256 max_spend_wei  // ceiling the requester authorizes for this job
    bytes32 job_nonce      // unique per job (anti-replay)
    uint256 expiry_unix    // auth invalid after this (anti-replay / staleness)
    bytes32 request_hash   // keccak of the canonical inference-request params
  }
"""
from __future__ import annotations

import pytest
from eth_account import Account

from prsm.settlement.payment_authorization import (
    DEFAULT_CHAIN_ID,
    DOMAIN_NAME,
    DOMAIN_VERSION,
    InvalidSignatureFormat,
    encode_payment_typed_data_hash,
    is_expired,
    sign_payment_authorization,
    verify_payment_authorization_signature,
)

# Deterministic test keys (NEVER used on any live chain).
REQUESTER_KEY = "0x" + "11" * 32
PROVIDER_KEY = "0x" + "22" * 32
REQUESTER = Account.from_key(REQUESTER_KEY).address
PROVIDER = Account.from_key(PROVIDER_KEY).address
ONE_FTNS = 10**18


def _payload(**over):
    p = {
        "requester": REQUESTER,
        "provider": PROVIDER,
        "max_spend_wei": ONE_FTNS,
        "job_nonce": "0x" + "ab" * 32,
        "expiry_unix": 2_000_000_000,
        "request_hash": "0x" + "cd" * 32,
    }
    p.update(over)
    return p


def test_domain_constants_locked():
    """The signed domain is load-bearing — every signature keys off it."""
    assert DOMAIN_NAME == "PRSM-Payment"
    assert DOMAIN_VERSION == "v1"
    assert DEFAULT_CHAIN_ID == 8453


def test_sign_recover_roundtrip_yields_requester():
    sig = sign_payment_authorization(_payload(), REQUESTER_KEY)
    recovered = verify_payment_authorization_signature(_payload(), sig)
    assert recovered.lower() == REQUESTER.lower()


def test_hash_is_stable_for_identical_payload():
    a = encode_payment_typed_data_hash(_payload())
    b = encode_payment_typed_data_hash(_payload())
    assert a == b
    assert isinstance(a, bytes) and len(a) == 32


def test_chain_id_binds_the_hash():
    """A different chainId must produce a different digest (no cross-chain replay)."""
    base = encode_payment_typed_data_hash(_payload(), chain_id=8453)
    other = encode_payment_typed_data_hash(_payload(), chain_id=1)
    assert base != other


@pytest.mark.parametrize("field,new", [
    ("provider", Account.from_key("0x" + "33" * 32).address),
    ("max_spend_wei", ONE_FTNS * 2),
    ("job_nonce", "0x" + "ef" * 32),
    ("expiry_unix", 2_000_000_001),
    ("request_hash", "0x" + "00" * 32),
])
def test_tampering_any_field_breaks_recovery(field, new):
    """Sign the honest payload; recovering against a tampered payload must NOT
    return the requester (the signature covers every field)."""
    sig = sign_payment_authorization(_payload(), REQUESTER_KEY)
    recovered = verify_payment_authorization_signature(_payload(**{field: new}), sig)
    assert recovered.lower() != REQUESTER.lower()


def test_wrong_signer_recovers_to_a_different_address():
    """A provider can't name a requester who didn't sign: a PROVIDER-key signature
    over a REQUESTER-named payload recovers to the provider, not the requester."""
    sig = sign_payment_authorization(_payload(), PROVIDER_KEY)
    recovered = verify_payment_authorization_signature(_payload(), sig)
    assert recovered.lower() == PROVIDER.lower()
    assert recovered.lower() != REQUESTER.lower()


def test_bad_signature_shape_raises():
    with pytest.raises(InvalidSignatureFormat):
        verify_payment_authorization_signature(_payload(), b"\x00" * 10)
    with pytest.raises(InvalidSignatureFormat):
        verify_payment_authorization_signature(_payload(), "not-hex-zzz")


def test_bytes32_fields_accept_raw_bytes_and_hex_equivalently():
    """job_nonce / request_hash given as 32 raw bytes hash identically to the
    0x-hex form (so a wallet and a python client agree)."""
    hex_form = _payload(job_nonce="0x" + "ab" * 32, request_hash="0x" + "cd" * 32)
    raw_form = _payload(job_nonce=b"\xab" * 32, request_hash=b"\xcd" * 32)
    assert encode_payment_typed_data_hash(hex_form) == encode_payment_typed_data_hash(raw_form)


def test_missing_field_raises_value_error():
    bad = _payload()
    del bad["request_hash"]
    with pytest.raises(ValueError):
        encode_payment_typed_data_hash(bad)


def test_is_expired():
    assert is_expired(_payload(expiry_unix=100), now=lambda: 101.0) is True
    assert is_expired(_payload(expiry_unix=100), now=lambda: 100.0) is False
    assert is_expired(_payload(expiry_unix=100), now=lambda: 99.0) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
