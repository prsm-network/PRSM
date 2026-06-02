"""Unit tests for ``prsm.interface.api.wallet_api``.

Phase 4 Task 3a — HTTP API surface for the shipped onboarding modules
(SIWE verifier, wallet binding, USD display). Real eth_account signers,
real SIWE library, real binding signature recovery — only the FastAPI
dependency injection is overridden to swap services per test.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
from prsm.interface.onboarding.siwe import InMemoryNonceStore
from prsm.interface.onboarding.wallet_binding import (
    InMemoryWalletBindingStore,
    WalletBindingService,
    build_binding_message,
)


# ──────────────────────────────────────────────────────────────────────────
# Constants matching the SIWE message construction in
# tests/integration/test_phase4_wallet_sdk_e2e.py
# ──────────────────────────────────────────────────────────────────────────


DOMAIN = "app.prsm-network.com"
CHAIN_ID = 8453  # Base mainnet
URI = "https://app.prsm-network.com/login"
VERSION = "1"
STATEMENT = "Sign in to PRSM."


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _new_account(seed: str = "ab"):
    return Account.from_key("0x" + seed * 31 + "cd")


def _build_siwe_message(
    address: str,
    nonce: str,
    *,
    domain: str = DOMAIN,
    chain_id: int = CHAIN_ID,
    issued_at: str = "2026-04-27T12:00:00Z",
) -> str:
    lines = [
        f"{domain} wants you to sign in with your Ethereum account:",
        address,
        "",
        STATEMENT,
        "",
        f"URI: {URI}",
        f"Version: {VERSION}",
        f"Chain ID: {chain_id}",
        f"Nonce: {nonce}",
        f"Issued At: {issued_at}",
    ]
    return "\n".join(lines)


def _sign_message(message: str, acct) -> str:
    sig = acct.sign_message(encode_defunct(text=message))
    return "0x" + sig.signature.hex() if not sig.signature.hex().startswith("0x") else sig.signature.hex()


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def services() -> WalletApiServices:
    """Fresh in-memory services per test — no shared state between tests."""
    return WalletApiServices(
        settings=WalletApiSettings(
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_ttl_seconds=300,
        ),
        nonce_store=InMemoryNonceStore(),
        binding_service=WalletBindingService(InMemoryWalletBindingStore()),
        price_source=StaticPriceSource(price_usd=Decimal("2.00")),
        balance_lookup=_ZeroBalanceLookup(),
    )


@pytest.fixture
def app(services) -> FastAPI:
    a = FastAPI()
    a.include_router(router)
    a.dependency_overrides[get_services] = lambda: services
    return a


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def alice():
    return _new_account("ab")


@pytest.fixture
def bob():
    return _new_account("cd")


# ──────────────────────────────────────────────────────────────────────────
# /siwe/nonce
# ──────────────────────────────────────────────────────────────────────────


class TestNonceEndpoint:
    def test_issues_nonce(self, client):
        r = client.post("/api/v1/auth/wallet/siwe/nonce", json={})
        assert r.status_code == 200
        body = r.json()
        assert "nonce" in body
        assert len(body["nonce"]) >= 8  # EIP-4361 alphanumeric, ≥8 chars
        assert body["domain"] == DOMAIN
        assert body["chain_id"] == CHAIN_ID
        assert isinstance(body["expires_at_unix"], int)

    def test_two_nonces_are_distinct(self, client):
        r1 = client.post("/api/v1/auth/wallet/siwe/nonce", json={})
        r2 = client.post("/api/v1/auth/wallet/siwe/nonce", json={})
        assert r1.json()["nonce"] != r2.json()["nonce"]

    def test_chain_id_mismatch_rejected_early(self, client):
        r = client.post(
            "/api/v1/auth/wallet/siwe/nonce", json={"chain_id": 1}
        )
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "siwe_chain_id_mismatch"

    def test_chain_id_match_accepted(self, client):
        r = client.post(
            "/api/v1/auth/wallet/siwe/nonce", json={"chain_id": CHAIN_ID}
        )
        assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────────
# /siwe/verify
# ──────────────────────────────────────────────────────────────────────────


class TestVerifyEndpoint:
    def _get_nonce(self, client) -> str:
        return client.post(
            "/api/v1/auth/wallet/siwe/nonce", json={}
        ).json()["nonce"]

    def test_verify_happy_path(self, client, alice):
        nonce = self._get_nonce(client)
        msg = _build_siwe_message(alice.address, nonce)
        sig = _sign_message(msg, alice)

        r = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["address"].lower() == alice.address.lower()
        assert len(body["node_id_hex"]) == 32
        assert body["is_new_user"] is True
        assert "PRSM Identity Binding" in body["binding_message"]
        assert body["binding_issued_at"].endswith("Z")

    def test_verify_returning_user(self, client, alice, services):
        # Pre-bind alice in the service so sign_in returns her existing node_id.
        nonce = self._get_nonce(client)
        msg = _build_siwe_message(alice.address, nonce)
        sig = _sign_message(msg, alice)
        first = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        ).json()

        # Bind alice's node so the next sign_in returns is_new_user=False.
        binding_sig = _sign_message(first["binding_message"], alice)
        client.post(
            "/api/v1/auth/wallet/bind",
            json={
                "wallet_address": alice.address,
                "node_id_hex": first["node_id_hex"],
                "signature": binding_sig,
                "issued_at": first["binding_issued_at"],
            },
        )

        # Second login.
        nonce2 = self._get_nonce(client)
        msg2 = _build_siwe_message(alice.address, nonce2)
        sig2 = _sign_message(msg2, alice)
        r2 = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg2, "signature": sig2},
        )
        body2 = r2.json()
        assert body2["is_new_user"] is False
        assert body2["node_id_hex"] == first["node_id_hex"]

    def test_verify_bad_signature(self, client, alice, bob):
        # Sign with bob's key but claim alice's address in the message.
        nonce = self._get_nonce(client)
        msg = _build_siwe_message(alice.address, nonce)
        sig = _sign_message(msg, bob)

        r = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "siwe_signature_invalid"

    def test_verify_unknown_nonce(self, client, alice):
        # Don't request a nonce — fabricate one. Server has never issued it.
        msg = _build_siwe_message(alice.address, "fakenonce0000000000")
        sig = _sign_message(msg, alice)
        r = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "siwe_nonce_invalid_or_consumed"

    def test_verify_nonce_replay_rejected(self, client, alice):
        nonce = self._get_nonce(client)
        msg = _build_siwe_message(alice.address, nonce)
        sig = _sign_message(msg, alice)
        # First use succeeds.
        r1 = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        )
        assert r1.status_code == 200
        # Replay rejected — nonce already consumed.
        r2 = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        )
        assert r2.status_code == 400
        assert r2.json()["detail"]["error"] == "siwe_nonce_invalid_or_consumed"

    def test_verify_wrong_chain_id_in_message(self, client, alice):
        nonce = self._get_nonce(client)
        msg = _build_siwe_message(alice.address, nonce, chain_id=1)
        sig = _sign_message(msg, alice)
        r = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "siwe_chain_id_mismatch"

    def test_verify_malformed_message(self, client):
        r = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": "not a SIWE message", "signature": "0xdeadbeef"},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["error"] in (
            "siwe_malformed",
            "siwe_signature_invalid",
        )


# ──────────────────────────────────────────────────────────────────────────
# /wallet/bind
# ──────────────────────────────────────────────────────────────────────────


class TestBindEndpoint:
    def _onboard_to_verify(self, client, acct):
        nonce = client.post(
            "/api/v1/auth/wallet/siwe/nonce", json={}
        ).json()["nonce"]
        msg = _build_siwe_message(acct.address, nonce)
        sig = _sign_message(msg, acct)
        return client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        ).json()

    def test_bind_happy_path(self, client, alice):
        verified = self._onboard_to_verify(client, alice)
        binding_sig = _sign_message(verified["binding_message"], alice)
        r = client.post(
            "/api/v1/auth/wallet/bind",
            json={
                "wallet_address": alice.address,
                "node_id_hex": verified["node_id_hex"],
                "signature": binding_sig,
                "issued_at": verified["binding_issued_at"],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["wallet_address"].lower() == alice.address.lower()
        assert body["node_id_hex"] == verified["node_id_hex"]
        assert body["bound_at_unix"] > 0
        assert body["signing_message_hash"].startswith("0x")

    def test_bind_idempotent(self, client, alice):
        verified = self._onboard_to_verify(client, alice)
        binding_sig = _sign_message(verified["binding_message"], alice)
        payload = {
            "wallet_address": alice.address,
            "node_id_hex": verified["node_id_hex"],
            "signature": binding_sig,
            "issued_at": verified["binding_issued_at"],
        }
        r1 = client.post("/api/v1/auth/wallet/bind", json=payload).json()
        r2 = client.post("/api/v1/auth/wallet/bind", json=payload).json()
        assert r1 == r2

    def test_bind_signature_invalid(self, client, alice, bob):
        verified = self._onboard_to_verify(client, alice)
        # Bob signs Alice's binding message — recovery yields Bob, mismatch.
        bad_sig = _sign_message(verified["binding_message"], bob)
        r = client.post(
            "/api/v1/auth/wallet/bind",
            json={
                "wallet_address": alice.address,
                "node_id_hex": verified["node_id_hex"],
                "signature": bad_sig,
                "issued_at": verified["binding_issued_at"],
            },
        )
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "binding_signature_invalid"

    def test_bind_conflict_returns_409(self, client, alice, bob, services):
        # sp932 — refreshed for the Sprint 786 multi-device contract. A wallet
        # binding to a SECOND node is now allowed (wallet→node is 1:N), so the
        # still-load-bearing 409 conflict is NODE-side: a different wallet may
        # not bind a node_id already claimed by another wallet (otherwise an
        # operator could steal another node's identity).
        # sp946 — Issued-At must be within the binding freshness window; use
        # recent (distinct) timestamps since these flow through the HTTP /bind
        # endpoint, which has no now_unix test hook.
        _base = datetime.now(timezone.utc)
        node_a = "a" * 32
        issued_a = _base.strftime("%Y-%m-%dT%H:%M:%SZ")
        services.binding_service.bind(
            wallet_address=alice.address,
            node_id_hex=node_a,
            signature=_sign_message(build_binding_message(alice.address, node_a, issued_a), alice),
            issued_at_iso=issued_a,
        )

        # Sprint 786: alice binding a DIFFERENT node (multi-device) SUCCEEDS.
        node_b = "b" * 32
        issued_b = (_base + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ok = client.post(
            "/api/v1/auth/wallet/bind",
            json={
                "wallet_address": alice.address,
                "node_id_hex": node_b,
                "signature": _sign_message(build_binding_message(alice.address, node_b, issued_b), alice),
                "issued_at": issued_b,
            },
        )
        assert ok.status_code == 200, ok.text

        # But BOB binding alice's node_a is a node-side conflict → 409.
        issued_c = (_base + timedelta(seconds=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = client.post(
            "/api/v1/auth/wallet/bind",
            json={
                "wallet_address": bob.address,
                "node_id_hex": node_a,
                "signature": _sign_message(build_binding_message(bob.address, node_a, issued_c), bob),
                "issued_at": issued_c,
            },
        )
        assert r.status_code == 409
        assert r.json()["detail"]["error"] == "binding_conflict"


# ──────────────────────────────────────────────────────────────────────────
# /wallet/binding (lookup)
# ──────────────────────────────────────────────────────────────────────────


class TestGetBindingEndpoint:
    def test_returns_null_when_unbound(self, client, alice):
        r = client.get(
            "/api/v1/auth/wallet/binding",
            params={"wallet_address": alice.address},
        )
        assert r.status_code == 200
        assert r.json() is None

    def test_returns_binding_after_bind(self, client, alice):
        # Drive nonce → verify → bind.
        nonce = client.post(
            "/api/v1/auth/wallet/siwe/nonce", json={}
        ).json()["nonce"]
        msg = _build_siwe_message(alice.address, nonce)
        sig = _sign_message(msg, alice)
        verified = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        ).json()
        binding_sig = _sign_message(verified["binding_message"], alice)
        client.post(
            "/api/v1/auth/wallet/bind",
            json={
                "wallet_address": alice.address,
                "node_id_hex": verified["node_id_hex"],
                "signature": binding_sig,
                "issued_at": verified["binding_issued_at"],
            },
        )

        r = client.get(
            "/api/v1/auth/wallet/binding",
            params={"wallet_address": alice.address},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["node_id_hex"] == verified["node_id_hex"]


# ──────────────────────────────────────────────────────────────────────────
# /wallet/balance
# ──────────────────────────────────────────────────────────────────────────


class _FixedBalanceLookup:
    def __init__(self, ftns: Decimal):
        self._ftns = ftns

    def get_ftns_balance(self, wallet_address: str) -> Decimal:
        return self._ftns


class TestBalanceEndpoint:
    def _bind_alice(self, client, alice):
        nonce = client.post(
            "/api/v1/auth/wallet/siwe/nonce", json={}
        ).json()["nonce"]
        msg = _build_siwe_message(alice.address, nonce)
        sig = _sign_message(msg, alice)
        verified = client.post(
            "/api/v1/auth/wallet/siwe/verify",
            json={"message": msg, "signature": sig},
        ).json()
        binding_sig = _sign_message(verified["binding_message"], alice)
        client.post(
            "/api/v1/auth/wallet/bind",
            json={
                "wallet_address": alice.address,
                "node_id_hex": verified["node_id_hex"],
                "signature": binding_sig,
                "issued_at": verified["binding_issued_at"],
            },
        )

    def test_unbound_wallet_returns_404(self, client, alice):
        r = client.get(
            "/api/v1/auth/wallet/balance",
            params={"wallet_address": alice.address},
        )
        assert r.status_code == 404
        assert r.json()["detail"]["error"] == "wallet_not_bound"

    def test_bound_wallet_returns_zero_balance_default(self, client, alice):
        self._bind_alice(client, alice)
        r = client.get(
            "/api/v1/auth/wallet/balance",
            params={"wallet_address": alice.address},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ftns"] == "0"
        assert body["mode"] == "usd"

    def test_bound_wallet_with_balance(self, client, alice, services):
        services.balance_lookup = _FixedBalanceLookup(Decimal("1.5"))
        self._bind_alice(client, alice)
        r = client.get(
            "/api/v1/auth/wallet/balance",
            params={"wallet_address": alice.address},
        )
        body = r.json()
        # 1.5 FTNS @ $2.00 = $3.00
        assert body["ftns"] == "1.5"
        assert body["usd"] == "3.00"
        assert "$3.00" in body["formatted"]

    def test_ftns_mode(self, client, alice, services):
        services.balance_lookup = _FixedBalanceLookup(Decimal("2.0"))
        self._bind_alice(client, alice)
        r = client.get(
            "/api/v1/auth/wallet/balance",
            params={"wallet_address": alice.address, "mode": "ftns"},
        )
        body = r.json()
        assert body["mode"] == "ftns"
        assert body["usd"] is None
        assert "FTNS" in body["formatted"]

    def test_invalid_mode_rejected(self, client, alice):
        self._bind_alice(client, alice)
        r = client.get(
            "/api/v1/auth/wallet/balance",
            params={"wallet_address": alice.address, "mode": "btc"},
        )
        assert r.status_code in (400, 422)


# ──────────────────────────────────────────────────────────────────────────
# Service-injection guard
# ──────────────────────────────────────────────────────────────────────────


class TestSiweCrossLanguageRoundTrip:
    """Round-1 review HIGH-1 + HIGH-2: pin the EIP-4361 layouts the
    JS ``buildSiweMessage`` produces against the Python ``siwe``
    library. If a future JS change drifts from this contract, the
    backend will reject login messages."""

    def test_no_statement_layout_parses(self):
        """JS buildSiweMessage with no statement emits TWO blank lines
        between address and URI. Confirms the Python siwe library
        accepts that form (round-1 review HIGH-1)."""
        from siwe import SiweMessage

        # Exactly the bytes JS buildSiweMessage produces with no statement.
        # Format: header / address / blank / blank / URI / Version / Chain / Nonce / Issued.
        msg = (
            f"{DOMAIN} wants you to sign in with your Ethereum account:\n"
            "0x52908400098527886E0F7030069857D2E4169EE7\n"
            "\n"
            "\n"
            f"URI: {URI}\n"
            "Version: 1\n"
            f"Chain ID: {CHAIN_ID}\n"
            "Nonce: abc12345\n"
            "Issued At: 2026-04-27T12:00:00Z"
        )
        # Must parse without raising.
        SiweMessage.from_message(msg)

    def test_with_statement_layout_parses(self):
        """JS buildSiweMessage with statement emits the standard layout."""
        from siwe import SiweMessage

        msg = (
            f"{DOMAIN} wants you to sign in with your Ethereum account:\n"
            "0x52908400098527886E0F7030069857D2E4169EE7\n"
            "\n"
            "Sign in to PRSM.\n"
            "\n"
            f"URI: {URI}\n"
            "Version: 1\n"
            f"Chain ID: {CHAIN_ID}\n"
            "Nonce: abc12345\n"
            "Issued At: 2026-04-27T12:00:00Z"
        )
        SiweMessage.from_message(msg)

    def test_lowercase_address_rejected_by_siwe(self):
        """Confirms the bug we're defending against: lowercase
        addresses (as some EIP-1193 providers return) are rejected
        by the siwe library. The JS toChecksumAddress normalizes
        before this point (round-1 review HIGH-2)."""
        from siwe import SiweMessage
        from pydantic import ValidationError

        msg = (
            f"{DOMAIN} wants you to sign in with your Ethereum account:\n"
            "0xab5801a7d398351b8be11c439e05c5b3259aec9b\n"
            "\n"
            "stmt\n"
            "\n"
            f"URI: {URI}\n"
            "Version: 1\n"
            f"Chain ID: {CHAIN_ID}\n"
            "Nonce: abc12345\n"
            "Issued At: 2026-04-27T12:00:00Z"
        )
        with pytest.raises((ValidationError, ValueError)):
            SiweMessage.from_message(msg)


class TestServicesGuard:
    def test_unconfigured_services_raise_runtime_error(self):
        # Build app WITHOUT the dependency override — get_services should
        # raise RuntimeError, surfaced as 500.
        a = FastAPI()
        a.include_router(router)
        with TestClient(a, raise_server_exceptions=False) as c:
            r = c.post("/api/v1/auth/wallet/siwe/nonce", json={})
            assert r.status_code == 500
