"""Sprint 1013 — wallet session-token enforcement wiring (API-authz Residual A).

Integration tests for the session token wired into wallet_api:
  - /siwe/verify mints a wallet-bound session token (additive; non-breaking).
  - The end-to-end SIWE→token→read flow lets a wallet owner read its own data.
  - When session_required is enabled, the wallet-OWNER reads enforce ownership:
    no token → 401, valid token → 200, token for ANOTHER wallet → 403 (the IDOR
    attempt), garbage token → 401.
  - When session_required is off (default), reads are unchanged (back-compat).
"""
from __future__ import annotations

from decimal import Decimal
from typing import Iterator

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct
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
    mint_session_token,
    verify_session_token,
)
from prsm.interface.onboarding.siwe import InMemoryNonceStore
from prsm.interface.onboarding.wallet_binding import (
    InMemoryWalletBindingStore,
    WalletBindingService,
)

DOMAIN = "app.prsm-network.com"
CHAIN_ID = 8453
SECRET = b"k" * 32
BASE = "/api/v1/auth/wallet"


def _services(session_required: bool) -> WalletApiServices:
    return WalletApiServices(
        settings=WalletApiSettings(
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            session_secret=SECRET,
            session_ttl_seconds=3600,
            session_required=session_required,
        ),
        nonce_store=InMemoryNonceStore(),
        binding_service=WalletBindingService(InMemoryWalletBindingStore()),
        price_source=StaticPriceSource(price_usd=Decimal("2.00")),
        balance_lookup=_ZeroBalanceLookup(),
    )


def _client(services: WalletApiServices) -> Iterator[TestClient]:
    a = FastAPI()
    a.include_router(router)
    a.dependency_overrides[get_services] = lambda: services
    return TestClient(a)


def _siwe_message(address: str, nonce: str) -> str:
    return "\n".join([
        f"{DOMAIN} wants you to sign in with your Ethereum account:",
        address,
        "",
        "Sign in to PRSM.",
        "",
        "URI: https://app.prsm-network.com/login",
        "Version: 1",
        f"Chain ID: {CHAIN_ID}",
        f"Nonce: {nonce}",
        "Issued At: 2026-06-04T12:00:00Z",
    ])


# ── /siwe/verify mints a token (additive) ──────────────────────────────────


def test_verify_returns_session_token_bound_to_signer():
    services = _services(session_required=False)
    client = _client(services)
    acct = Account.from_key("0x" + "ab" * 31 + "cd")
    nonce = client.post(f"{BASE}/siwe/nonce", json={}).json()["nonce"]
    msg = _siwe_message(acct.address, nonce)
    sig = acct.sign_message(encode_defunct(text=msg)).signature.hex()
    if not sig.startswith("0x"):
        sig = "0x" + sig

    r = client.post(f"{BASE}/siwe/verify", json={"message": msg, "signature": sig})
    assert r.status_code == 200, r.text
    token = r.json()["session_token"]
    assert token
    # The minted token verifies under the server secret to the signer's address.
    assert verify_session_token(token, secret=SECRET) == acct.address.lower()


def test_end_to_end_minted_token_authorizes_own_read():
    services = _services(session_required=True)
    client = _client(services)
    acct = Account.from_key("0x" + "12" * 31 + "34")
    nonce = client.post(f"{BASE}/siwe/nonce", json={}).json()["nonce"]
    msg = _siwe_message(acct.address, nonce)
    sig = acct.sign_message(encode_defunct(text=msg)).signature.hex()
    if not sig.startswith("0x"):
        sig = "0x" + sig
    token = client.post(
        f"{BASE}/siwe/verify", json={"message": msg, "signature": sig}
    ).json()["session_token"]

    r = client.get(
        f"{BASE}/bindings",
        params={"wallet_address": acct.address},
        headers={"X-Wallet-Session": token},
    )
    assert r.status_code == 200, r.text
    assert r.json() == []  # bound? no — but the ownership gate passed


# ── enforcement (session_required=True) ─────────────────────────────────────


def test_read_without_token_rejected_when_required():
    client = _client(_services(session_required=True))
    r = client.get(f"{BASE}/bindings", params={"wallet_address": "0x" + "a" * 40})
    assert r.status_code == 401
    assert r.json()["detail"]["error"] == "wallet_session_required"


def test_read_with_valid_token_allowed():
    client = _client(_services(session_required=True))
    addr = "0x" + "a" * 40
    token = mint_session_token(addr, secret=SECRET)
    r = client.get(
        f"{BASE}/bindings",
        params={"wallet_address": addr},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200


def test_read_with_token_for_other_wallet_forbidden():
    client = _client(_services(session_required=True))
    token = mint_session_token("0x" + "b" * 40, secret=SECRET)  # bound to B
    r = client.get(
        f"{BASE}/bindings",
        params={"wallet_address": "0x" + "a" * 40},  # asking for A
        headers={"X-Wallet-Session": token},
    )
    assert r.status_code == 403
    assert r.json()["detail"]["error"] == "wallet_session_address_mismatch"


def test_read_with_garbage_token_rejected():
    client = _client(_services(session_required=True))
    r = client.get(
        f"{BASE}/bindings",
        params={"wallet_address": "0x" + "a" * 40},
        headers={"X-Wallet-Session": "not-a-real-token"},
    )
    assert r.status_code == 401
    assert r.json()["detail"]["error"] == "wallet_session_invalid"


def test_read_with_foreign_secret_token_rejected():
    client = _client(_services(session_required=True))
    addr = "0x" + "a" * 40
    forged = mint_session_token(addr, secret=b"f" * 32)  # wrong secret
    r = client.get(
        f"{BASE}/bindings",
        params={"wallet_address": addr},
        headers={"X-Wallet-Session": forged},
    )
    assert r.status_code == 401
    assert r.json()["detail"]["error"] == "wallet_session_invalid"


@pytest.mark.parametrize("path", ["/binding", "/bindings", "/devices/earnings", "/balance"])
def test_all_owner_reads_gated_when_required(path):
    client = _client(_services(session_required=True))
    r = client.get(f"{BASE}{path}", params={"wallet_address": "0x" + "a" * 40})
    assert r.status_code == 401


# ── back-compat (default off) ───────────────────────────────────────────────


def test_reads_open_when_not_required():
    client = _client(_services(session_required=False))
    r = client.get(f"{BASE}/bindings", params={"wallet_address": "0x" + "a" * 40})
    assert r.status_code == 200  # unchanged behavior — no token needed
