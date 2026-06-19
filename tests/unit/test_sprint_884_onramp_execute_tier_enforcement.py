"""Sprint 884 — onramp/execute enforces KYC + tier limits.

sp281 surfaced kyc_required + sp285 surfaced tier_limit_exceeded as
ADVISORY flags on the onramp QUOTE. But sp853's onramp EXECUTE
(the real-money entry point that mints a Coinbase Pay session)
never checked either — a caller could skip the quote and execute
any amount on an unverified or over-limit account.

Sp884 makes execute BLOCK (HTTP 403) when, for a
destination_user_id:
  - KYC is not VERIFIED → 403 kyc_required (+ resume session_url)
  - the requested USD would exceed the user's tier limit
    (basic $1k / enhanced $10k, rolling-window aware) → 403
    tier_limit_exceeded, with an upgrade-to-enhanced hint for
    basic-tier users (the teeth behind sp883's enhanced template)

A raw destination_address (no PRSM identity) is NOT gated —
documented operator responsibility, mirroring the quote's bypass.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.economy.web3.kyc_client import KYCClient
from prsm.node.api import create_api_app


class _FakeWaasBackend:
    def create_wallet(self, user_id, email):
        return {
            "wallet_id": f"w-{user_id}",
            "address": "0x" + "11" * 20,
            "network": "base-mainnet",
        }


class _FakeKYCBackend:
    def initiate_session(self, user_id, email, level):
        return {
            "vendor_ref": f"persona-{user_id}",
            "session_url": f"https://persona.example/v/{user_id}",
            "status": "INITIATED",
        }


def _client(*, waas=None, kyc=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._coinbase_waas_client = waas
    node._kyc_client = kyc
    # None ring → _tier_check sees rolling=0.0 cleanly (no MagicMock
    # float() coercion through the except path).
    node._aerodrome_client = None
    node._fiat_compliance_ring = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _waas_with(user_id):
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=_FakeWaasBackend(),
    )
    c.provision_wallet(user_id=user_id, email="a@x.io")
    return c


def _kyc_verified(user_id, level):
    kyc = KYCClient(
        vendor="persona", api_key="k", backend=_FakeKYCBackend(),
    )
    kyc.initiate(user_id=user_id, email="a@x.io", level=level)
    kyc.update_status(user_id, "VERIFIED")
    return kyc


# ── KYC-required enforcement ─────────────────────────────────

def test_execute_blocks_unverified_user():
    """destination_user_id with NO KYC record → 403 kyc_required."""
    waas = _waas_with("alice")
    kyc = KYCClient(
        vendor="persona", api_key="k", backend=_FakeKYCBackend(),
    )  # alice never initiated
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 100.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error"] == "kyc_required"
    assert detail["kyc_status"] == "NOT_STARTED"


def test_execute_blocks_in_progress_user_with_resume_url():
    """INITIATED-but-not-VERIFIED → 403 + resume session_url."""
    waas = _waas_with("alice")
    kyc = KYCClient(
        vendor="persona", api_key="k", backend=_FakeKYCBackend(),
    )
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 100.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error"] == "kyc_required"
    assert detail["kyc_status"] == "INITIATED"
    assert detail["kyc_session_url"] is not None


# ── Tier-limit enforcement ───────────────────────────────────

def test_execute_allows_verified_basic_within_limit():
    """Verified basic user, $500 < $1k limit → NOT 403. (Returns
    200 PENDING_COMMISSION since no CDP env — proving the gate
    passed, not the session mint.)"""
    waas = _waas_with("alice")
    kyc = _kyc_verified("alice", "basic")
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 500.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 200
    # Gate passed; session mint not wired in test env.
    assert resp.json()["status"] in (
        "PENDING_COMMISSION", "SESSION_READY",
    )


def test_execute_blocks_basic_over_limit_with_upgrade_hint():
    """Verified basic user, $1500 > $1k → 403 tier_limit_exceeded
    with an enhanced-KYC upgrade hint (the sp883 teeth)."""
    waas = _waas_with("alice")
    kyc = _kyc_verified("alice", "basic")
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 1500.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error"] == "tier_limit_exceeded"
    assert detail["tier_level"] == "basic"
    assert detail["tier_limit_usd"] == 1000.0
    assert detail["requested_usd"] == 1500.0
    assert "enhanced" in detail["message"].lower()


def test_execute_allows_enhanced_within_higher_limit():
    """Verified enhanced user, $5000 < $10k → NOT 403."""
    waas = _waas_with("bob")
    kyc = _kyc_verified("bob", "enhanced")
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 5000.0, "destination_user_id": "bob"},
    )
    assert resp.status_code == 200


def test_execute_blocks_enhanced_over_limit_no_upgrade_hint():
    """Verified enhanced user, $15000 > $10k → 403; message does
    NOT suggest enhanced (already there) — points to support."""
    waas = _waas_with("bob")
    kyc = _kyc_verified("bob", "enhanced")
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 15000.0, "destination_user_id": "bob"},
    )
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error"] == "tier_limit_exceeded"
    assert detail["tier_level"] == "enhanced"
    assert detail["tier_limit_usd"] == 10000.0
    assert "support" in detail["message"].lower()


# ── Bypass paths ─────────────────────────────────────────────

def test_execute_address_only_not_gated():
    """Raw destination_address (no user_id) → no KYC/tier gate
    (operator responsibility). Proceeds past the gate."""
    resp = _client(waas=None, kyc=_kyc_verified("x", "basic")).post(
        "/wallet/onramp/execute",
        json={
            "usd_amount": 999999.0,
            "destination_address": "0x" + "22" * 20,
        },
    )
    # Not 403 — no per-user identity to gate. Proceeds to session
    # mint (PENDING_COMMISSION without CDP env).
    assert resp.status_code == 200


def test_execute_no_kyc_client_not_gated():
    """When _kyc_client is unwired, no gating (pre-sp884 behavior
    for deployments that haven't wired KYC). destination_user_id
    still resolves via WaaS."""
    waas = _waas_with("alice")
    resp = _client(waas=waas, kyc=None).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 5000.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 200


# ── Sp1175: per-replica enforcement scope + strict-shared lever ──

def test_tier_scope_surfaced_as_process_local(monkeypatch):
    """The tier block exposes tier_limit_scope=process_local so a
    caller/auditor sees the rolling AML total is per-replica, not
    global. Default mode is a pure no-op on the allow path."""
    monkeypatch.delenv("PRSM_FIAT_TIER_LIMIT_MODE", raising=False)
    waas = _waas_with("alice")
    kyc = _kyc_verified("alice", "basic")
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={"usd_amount": 500.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 200
    assert resp.json()["tier_limit_scope"] == "process_local"


def test_strict_shared_mode_fails_closed_within_limit(monkeypatch):
    """PRSM_FIAT_TIER_LIMIT_MODE=strict_shared — a VERIFIED user
    WITHIN their limit is denied 503 (not 403): the cross-replica
    AML limit can't be enforced without a shared backend, so the
    gated fiat surface fails closed rather than under-enforcing."""
    monkeypatch.setenv("PRSM_FIAT_TIER_LIMIT_MODE", "strict_shared")
    waas = _waas_with("alice")
    kyc = _kyc_verified("alice", "basic")
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 100.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["error"] == "tier_limit_enforcement_unavailable"
    assert detail["tier_limit_scope"] == "process_local"


def test_strict_shared_does_not_block_unverified_kyc_still_first(
    monkeypatch,
):
    """Strict mode must not mask the KYC gate: an unverified user
    is still 403 kyc_required (the strict block only applies to the
    tier limit, which only VERIFIED users are subject to)."""
    monkeypatch.setenv("PRSM_FIAT_TIER_LIMIT_MODE", "strict_shared")
    waas = _waas_with("alice")
    kyc = KYCClient(
        vendor="persona", api_key="k", backend=_FakeKYCBackend(),
    )  # alice never initiated
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 100.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "kyc_required"


def test_default_mode_still_allows_within_limit(monkeypatch):
    """Sanity: with the env unset (default process_local) a within-
    limit VERIFIED user is NOT blocked — strict mode is opt-in."""
    monkeypatch.delenv("PRSM_FIAT_TIER_LIMIT_MODE", raising=False)
    waas = _waas_with("alice")
    kyc = _kyc_verified("alice", "basic")
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/execute",
        json={"usd_amount": 500.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 200


# ── Quote still advisory (unchanged) ─────────────────────────

def test_quote_still_advisory_not_403():
    """The QUOTE must remain non-blocking (advisory flags only) —
    sp884 only hardens EXECUTE. A regression that made the quote
    403 would break the preview UX."""
    waas = _waas_with("alice")
    kyc = _kyc_verified("alice", "basic")
    resp = _client(waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={"usd_amount": 1500.0, "destination_user_id": "alice"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier_limit_exceeded"] is True  # flagged, not blocked
