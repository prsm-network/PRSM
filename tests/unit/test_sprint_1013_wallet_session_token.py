"""Sprint 1013 — wallet session-token primitive (API-authz Residual A).

The API-authz hunt (workflow wt1tb4n3q, finding 2) found the wallet-onboarding
read endpoints (/api/v1/auth/wallet/binding|bindings|devices/earnings|balance)
keyed solely on a wallet_address query param with no ownership check — a
read-IDOR. The correct gate is proof the caller controls the wallet, not the
operator's API key (these are wallet-OWNER calls). The SIWE flow already proves
control at /siwe/verify, but there was no SESSION TOKEN to carry that proof to
later reads (a documented future task, wallet_api.py:35).

This ships the primitive: a stateless, HMAC-signed, wallet-bound, TTL'd token
minted on a successful SIWE verify and presented on the read path. It binds the
verified address into the signed payload so a holder can only act for the wallet
it proved control of.
"""
from __future__ import annotations

import pytest

from prsm.interface.onboarding.session_token import (
    SessionTokenExpired,
    SessionTokenInvalid,
    SessionTokenMalformed,
    mint_session_token,
    verify_session_token,
)

_SECRET = b"0123456789abcdef0123456789abcdef"  # 32 bytes
_OTHER = b"ffffffffffffffffffffffffffffffff"
_ADDR = "0x" + "A" * 40


def test_mint_verify_roundtrip_returns_normalized_address():
    tok = mint_session_token(_ADDR, secret=_SECRET, ttl_seconds=3600, now=1000)
    assert verify_session_token(tok, secret=_SECRET, now=1100) == _ADDR.lower()


def test_case_insensitive_address_binding():
    # A checksummed and a lowercased address mint to the same bound wallet.
    t1 = mint_session_token("0xAbCdef" + "0" * 34, secret=_SECRET, now=1000)
    t2 = mint_session_token("0xabcdef" + "0" * 34, secret=_SECRET, now=1000)
    assert verify_session_token(t1, secret=_SECRET, now=1000) == \
        verify_session_token(t2, secret=_SECRET, now=1000)


def test_wrong_secret_rejected():
    tok = mint_session_token(_ADDR, secret=_SECRET, now=1000)
    with pytest.raises(SessionTokenInvalid):
        verify_session_token(tok, secret=_OTHER, now=1000)


def test_tampered_payload_rejected():
    tok = mint_session_token(_ADDR, secret=_SECRET, now=1000)
    version, payload_b64, sig_b64 = tok.split(".")
    # Flip a payload char (keeps the signature) → signature must no longer match.
    bad_payload = ("A" if payload_b64[0] != "A" else "B") + payload_b64[1:]
    forged = f"{version}.{bad_payload}.{sig_b64}"
    with pytest.raises(SessionTokenError_or_subclass()):
        verify_session_token(forged, secret=_SECRET, now=1000)


def test_tampered_signature_rejected():
    tok = mint_session_token(_ADDR, secret=_SECRET, now=1000)
    version, payload_b64, sig_b64 = tok.split(".")
    bad_sig = ("A" if sig_b64[0] != "A" else "B") + sig_b64[1:]
    with pytest.raises(SessionTokenError_or_subclass()):
        verify_session_token(f"{version}.{payload_b64}.{bad_sig}", secret=_SECRET, now=1000)


def test_expired_rejected():
    tok = mint_session_token(_ADDR, secret=_SECRET, ttl_seconds=100, now=1000)
    # exactly at exp and beyond → expired
    with pytest.raises(SessionTokenExpired):
        verify_session_token(tok, secret=_SECRET, now=1100)
    with pytest.raises(SessionTokenExpired):
        verify_session_token(tok, secret=_SECRET, now=2000)


def test_valid_within_ttl():
    tok = mint_session_token(_ADDR, secret=_SECRET, ttl_seconds=100, now=1000)
    assert verify_session_token(tok, secret=_SECRET, now=1099) == _ADDR.lower()


@pytest.mark.parametrize("bad", ["", "x", "v1.only-two", "v2.aaa.bbb", "v1..", "not.a.token"])
def test_malformed_rejected(bad):
    with pytest.raises(SessionTokenMalformed):
        verify_session_token(bad, secret=_SECRET, now=1000)


def test_mint_rejects_short_secret():
    with pytest.raises(Exception):
        mint_session_token(_ADDR, secret=b"tooshort", now=1000)


def test_mint_rejects_empty_wallet():
    with pytest.raises(Exception):
        mint_session_token("", secret=_SECRET, now=1000)


def test_mint_rejects_nonpositive_ttl():
    with pytest.raises(Exception):
        mint_session_token(_ADDR, secret=_SECRET, ttl_seconds=0, now=1000)


# Helper so the tamper tests accept Invalid OR Malformed (both are correct
# rejections depending on whether the tamper corrupts the b64/JSON or just the
# bytes under a valid b64).
def SessionTokenError_or_subclass():
    from prsm.interface.onboarding.session_token import SessionTokenError
    return SessionTokenError
