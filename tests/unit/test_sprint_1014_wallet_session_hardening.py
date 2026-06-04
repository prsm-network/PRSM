"""Sprint 1014 — harden the sp1013 wallet session token (adversarial-review fixes).

The sp1013 adversarial review (workflow wtu8jljgu) confirmed three real defects:

  HIGH — an empty-hex PRSM_WALLET_SESSION_SECRET ("0x" / "0X" / "  0x  ") parses
  to b"" which, being <16 bytes, was SHA-256-stretched to the WORLD-KNOWN
  constant sha256(b"") = e3b0c442... — a universally-forgeable signing key
  (reopens the IDOR sp1013 closed). Realistic misconfig: a deploy template
  emitting `0x${SECRET}` with SECRET unset.

  MEDIUM — the wallet_address query param flowed un-normalized into
  eth_utils.to_checksum_address, which raises ValueError on trailing whitespace
  / non-address input. Uncaught → HTTP 500. Reachable UNAUTHENTICATED on the
  default-off path (any caller appends a space → 500 on all four reads = DoS) and
  a legit-owner false-reject.

  LOW — a token with exp=inf raises OverflowError in int(exp), which is neither
  TypeError nor ValueError nor SessionTokenError, so it escaped the handler →
  500 instead of a clean 401 (secret-holder-only).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from decimal import Decimal
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prsm.interface.api.wallet_api import (
    WalletApiServices,
    WalletApiSettings,
    _ZeroBalanceLookup,
    get_services,
    router,
)
from prsm.interface.display import StaticPriceSource
from prsm.interface.onboarding.session_token import (
    SessionTokenMalformed,
    mint_session_token,
    verify_session_token,
)
from prsm.interface.onboarding.siwe import InMemoryNonceStore
from prsm.interface.onboarding.wallet_binding import (
    InMemoryWalletBindingStore,
    WalletBindingService,
)
from prsm.node.wallet_api_wiring import _wallet_session_config_from_env

SECRET = b"k" * 32
BASE = "/api/v1/auth/wallet"
_SHA256_EMPTY = hashlib.sha256(b"").digest()


# ── HIGH: empty-hex secret must NOT collapse to the world-known constant ────


@pytest.mark.parametrize("val", ["0x", "0X", "  0x  ", "", "   "])
def test_degenerate_secret_falls_back_to_random(monkeypatch, val):
    monkeypatch.setenv("PRSM_WALLET_SESSION_SECRET", val)
    secret, _ = _wallet_session_config_from_env()
    assert secret != _SHA256_EMPTY, "degenerate secret collapsed to sha256(b'')"
    assert len(secret) >= 16


def test_degenerate_secret_is_random_each_call(monkeypatch):
    monkeypatch.setenv("PRSM_WALLET_SESSION_SECRET", "0x")
    s1, _ = _wallet_session_config_from_env()
    s2, _ = _wallet_session_config_from_env()
    assert s1 != s2  # random fallback, not a fixed derived constant


def test_real_hex_secret_used_verbatim(monkeypatch):
    hex_secret = "ab" * 32  # 32 bytes
    monkeypatch.setenv("PRSM_WALLET_SESSION_SECRET", "0x" + hex_secret)
    secret, _ = _wallet_session_config_from_env()
    assert secret == bytes.fromhex(hex_secret)


def test_short_nonempty_secret_still_stretched(monkeypatch):
    monkeypatch.setenv("PRSM_WALLET_SESSION_SECRET", "0xabcd")  # 2 bytes, nonempty
    secret, _ = _wallet_session_config_from_env()
    assert secret == hashlib.sha256(b"\xab\xcd").digest()
    assert len(secret) == 32


# ── LOW: inf-exp token rejected cleanly (no OverflowError) ──────────────────


def _forge_token(payload: dict, secret: bytes) -> str:
    pb = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sig = hmac.new(secret, pb, hashlib.sha256).digest()
    enc = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    return f"v1.{enc(pb)}.{enc(sig)}"


def test_inf_exp_token_raises_malformed_not_overflow():
    tok = _forge_token({"w": "0xabc", "iat": 0, "exp": float("inf")}, SECRET)
    with pytest.raises(SessionTokenMalformed):
        verify_session_token(tok, secret=SECRET, now=1000)


def test_nan_exp_token_rejected():
    tok = _forge_token({"w": "0xabc", "iat": 0, "exp": float("nan")}, SECRET)
    with pytest.raises(SessionTokenMalformed):
        verify_session_token(tok, secret=SECRET, now=1000)


# ── MEDIUM: malformed wallet_address → 400, not 500 ─────────────────────────


def _services(session_required: bool) -> WalletApiServices:
    return WalletApiServices(
        settings=WalletApiSettings(
            expected_domain="app.prsm-network.com",
            expected_chain_id=8453,
            session_secret=SECRET,
            session_required=session_required,
        ),
        nonce_store=InMemoryNonceStore(),
        binding_service=WalletBindingService(InMemoryWalletBindingStore()),
        price_source=StaticPriceSource(price_usd=Decimal("2.00")),
        balance_lookup=_ZeroBalanceLookup(),
    )


def _client(services: WalletApiServices) -> TestClient:
    a = FastAPI()
    a.include_router(router)
    a.dependency_overrides[get_services] = lambda: services
    # raise_server_exceptions=False so an uncaught 500 surfaces as a response.
    return TestClient(a, raise_server_exceptions=False)


@pytest.mark.parametrize("path", ["/binding", "/bindings", "/devices/earnings", "/balance"])
def test_trailing_whitespace_wallet_address_handled_gracefully_default_off(path):
    # The pre-fix bug was an uncaught 500. The fix strips surrounding whitespace
    # so a valid-but-whitespace-padded address succeeds (legit-owner false-reject
    # closed) rather than crashing — never a 500.
    client = _client(_services(session_required=False))
    r = client.get(f"{BASE}{path}", params={"wallet_address": "0x" + "a" * 40 + " "})
    # Valid address (modulo whitespace) → processed, never a crash (500) and
    # never rejected as malformed (400). 200 for the list/binding reads; /balance
    # legitimately 404s on an unbound wallet.
    assert r.status_code not in (400, 500), f"{path} returned {r.status_code}"


@pytest.mark.parametrize("path", ["/binding", "/bindings", "/devices/earnings", "/balance"])
def test_trailing_whitespace_wallet_address_handled_gracefully_gated(path):
    client = _client(_services(session_required=True))
    addr = "0x" + "a" * 40
    token = mint_session_token(addr, secret=SECRET)
    # Trailing whitespace on the query; the gate's strip() still matches the
    # token, and the endpoint normalizes before the store call — 200, not 500.
    r = client.get(
        f"{BASE}{path}",
        params={"wallet_address": addr + " "},
        headers={"X-Wallet-Session": token},
    )
    assert r.status_code not in (400, 500), f"{path} returned {r.status_code}"


def test_clean_wallet_address_still_ok():
    client = _client(_services(session_required=False))
    r = client.get(f"{BASE}/bindings", params={"wallet_address": "0x" + "a" * 40})
    assert r.status_code == 200


def test_non_address_garbage_is_400():
    client = _client(_services(session_required=False))
    r = client.get(f"{BASE}/bindings", params={"wallet_address": "not-an-address"})
    assert r.status_code == 400
