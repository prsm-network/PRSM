"""Phase 4 Wallet SDK end-to-end integration tests.

Backend slice of the plan §7 acceptance criterion:

    "A consumer with zero prior crypto experience can: (a) land on PRSM's
    web onboarding; (b) sign in with email via the embedded wallet; (c)
    bind their wallet to a freshly-created PRSM node identity; (d) see
    their FTNS balance in USD; (e) land on the dashboard ready to earn
    or consume, in under 90 seconds total onboarding time (TTO metric)."

This file exercises (c) + (d) end-to-end through the real shipped
modules (SIWE verifier, wallet binding, USD display). Parts (a), (b),
(e) are frontend concerns covered by Tasks 3/4 + a playwright harness.

Capstone for Phase 4 Tasks 1, 2, 5 — composes them in realistic
onboarding sequences.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import to_checksum_address

from prsm.interface.display import (
    DisplayMode,
    InMemoryDisplayPreferenceStore,
    SqliteDisplayPreferenceStore,
    StaticPriceSource,
    format_balance,
    ftns_to_usd,
)
from prsm.interface.onboarding.siwe import (
    InMemoryNonceStore,
    SiweNonceError,
    SiweSignatureError,
    verify,
)
from prsm.interface.onboarding.wallet_binding import (
    BindingSignatureError,
    InMemoryWalletBindingStore,
    SqliteWalletBindingStore,
    WalletBindingService,
    build_binding_message,
)

# -----------------------------------------------------------------------------
# Shared fixtures / constants
# -----------------------------------------------------------------------------

DOMAIN = "app.prsm-network.com"
CHAIN_ID = 8453  # Base mainnet
URI = "https://app.prsm-network.com/login"
VERSION = "1"
STATEMENT = "Sign in to PRSM."
BINDING_ISSUED_AT = "2026-04-22T12:00:00Z"
# sp946 — unix of BINDING_ISSUED_AT; pass as now_unix so the freshness window
# treats the fixed-timestamp binding signatures as freshly issued (age 0).
BINDING_ISSUED_AT_UNIX = int(
    datetime.fromisoformat(BINDING_ISSUED_AT.replace("Z", "+00:00")).timestamp()
)


@pytest.fixture
def now():
    return datetime(2026, 4, 22, 12, 5, tzinfo=timezone.utc)


@pytest.fixture
def price_source():
    # $2 per FTNS — simple math for assertions.
    return StaticPriceSource(price_usd=Decimal("2.00"))


def _new_account(tag: str = "a"):
    """Two distinct deterministic accounts for multi-wallet scenarios."""
    if tag == "a":
        return Account.from_key("0x" + "ab" * 31 + "cd")
    if tag == "b":
        return Account.from_key("0x" + "de" * 31 + "ad")
    raise ValueError(tag)


def _build_siwe_message(
    address: str,
    nonce: str,
    *,
    domain: str = DOMAIN,
    chain_id: int = CHAIN_ID,
    issued_at: str = "2026-04-22T12:00:00Z",
) -> str:
    lines = [f"{domain} wants you to sign in with your Ethereum account:", address, ""]
    lines.append(STATEMENT)
    lines.append("")
    lines.append(f"URI: {URI}")
    lines.append(f"Version: {VERSION}")
    lines.append(f"Chain ID: {chain_id}")
    lines.append(f"Nonce: {nonce}")
    lines.append(f"Issued At: {issued_at}")
    return "\n".join(lines)


def _sign_siwe(message: str, acct) -> str:
    encoded = encode_defunct(text=message)
    return acct.sign_message(encoded).signature.hex()


def _sign_binding(acct, wallet_address: str, node_id_hex: str) -> str:
    message = build_binding_message(wallet_address, node_id_hex, BINDING_ISSUED_AT)
    return acct.sign_message(encode_defunct(text=message)).signature.hex()


# -----------------------------------------------------------------------------
# Full happy-path onboarding
# -----------------------------------------------------------------------------


def test_full_new_user_onboarding_flow(now, price_source):
    """New user: nonce -> SIWE sign-in -> node_id issued -> bind -> display.

    Pins the primary §7 acceptance backend path.
    """
    acct = _new_account("a")

    # 1. Server issues a single-use nonce.
    nonce_store = InMemoryNonceStore()
    nonce = nonce_store.issue(ttl_seconds=300)

    # 2. Client builds + signs a SIWE message.
    siwe_msg = _build_siwe_message(acct.address, nonce)
    siwe_sig = _sign_siwe(siwe_msg, acct)

    # 3. Server verifies SIWE; consumes nonce.
    siwe_result = verify(
        siwe_msg,
        siwe_sig,
        expected_domain=DOMAIN,
        expected_chain_id=CHAIN_ID,
        nonce_store=nonce_store,
        now=now,
    )
    assert siwe_result.address.lower() == acct.address.lower()

    # 4. New user sign-in → fresh node_id, is_new_user=True.
    binding_svc = WalletBindingService(store=InMemoryWalletBindingStore())
    node_id, is_new_user = binding_svc.sign_in(siwe_result.address)
    assert is_new_user is True
    assert len(node_id) == 32  # 32-char hex sha256 prefix

    # 5. Client signs the binding attestation.
    binding_sig = _sign_binding(acct, acct.address, node_id)

    # 6. Server binds wallet <-> node_id.
    binding = binding_svc.bind(
        acct.address, node_id, binding_sig, BINDING_ISSUED_AT, now_unix=BINDING_ISSUED_AT_UNIX
    )
    assert binding.wallet_address == to_checksum_address(acct.address)
    assert binding.node_id_hex == node_id

    # 7. Dashboard renders balance in USD-default mode.
    balance_ftns = Decimal("0.125")
    displayed = format_balance(balance_ftns, price_source, mode="usd")
    assert displayed == "$0.25 · 0.1250 FTNS"


def test_returning_user_onboarding_reuses_bound_node_id(now, price_source):
    """Second sign-in for the same wallet returns the SAME node_id."""
    acct = _new_account("a")
    binding_svc = WalletBindingService(store=InMemoryWalletBindingStore())

    # First session — bind.
    node_id_1, is_new_user_1 = binding_svc.sign_in(acct.address)
    assert is_new_user_1 is True
    sig = _sign_binding(acct, acct.address, node_id_1)
    binding_svc.bind(acct.address, node_id_1, sig, BINDING_ISSUED_AT, now_unix=BINDING_ISSUED_AT_UNIX)

    # Second session — same wallet, returning user.
    node_id_2, is_new_user_2 = binding_svc.sign_in(acct.address)
    assert is_new_user_2 is False
    assert node_id_2 == node_id_1


# -----------------------------------------------------------------------------
# Replay + conflict protection
# -----------------------------------------------------------------------------


def test_siwe_nonce_replay_is_rejected(now):
    """Second sign-in attempt with same nonce fails — single-use protection."""
    acct = _new_account("a")
    nonce_store = InMemoryNonceStore()
    nonce = nonce_store.issue()
    msg = _build_siwe_message(acct.address, nonce)
    sig = _sign_siwe(msg, acct)

    # First verify succeeds + consumes nonce.
    verify(
        msg, sig,
        expected_domain=DOMAIN, expected_chain_id=CHAIN_ID,
        nonce_store=nonce_store, now=now,
    )

    # Replay attempt.
    with pytest.raises(SiweNonceError):
        verify(
            msg, sig,
            expected_domain=DOMAIN, expected_chain_id=CHAIN_ID,
            nonce_store=nonce_store, now=now,
        )


def test_binding_cross_wallet_conflict_rejected(now):
    """B cannot hijack A's node by signing a binding attestation for it."""
    acct_a = _new_account("a")
    acct_b = _new_account("b")

    binding_svc = WalletBindingService(store=InMemoryWalletBindingStore())

    # A signs in + binds.
    node_id_a, _ = binding_svc.sign_in(acct_a.address)
    sig_a = _sign_binding(acct_a, acct_a.address, node_id_a)
    binding_svc.bind(acct_a.address, node_id_a, sig_a, BINDING_ISSUED_AT, now_unix=BINDING_ISSUED_AT_UNIX)

    # B attempts to bind the SAME node_id to B's wallet.
    sig_b = _sign_binding(acct_b, acct_b.address, node_id_a)
    from prsm.interface.onboarding.wallet_binding import BindingConflictError
    with pytest.raises(BindingConflictError):
        binding_svc.bind(acct_b.address, node_id_a, sig_b, BINDING_ISSUED_AT, now_unix=BINDING_ISSUED_AT_UNIX)


def test_binding_rejects_signature_from_different_wallet(now):
    """A cannot claim ownership of B's wallet -> node binding."""
    acct_a = _new_account("a")
    acct_b = _new_account("b")

    binding_svc = WalletBindingService(store=InMemoryWalletBindingStore())
    node_id, _ = binding_svc.sign_in(acct_a.address)

    # A signs for B's address — signature recovers to A, not B.
    bad_sig = _sign_binding(acct_a, acct_b.address, node_id)
    with pytest.raises(BindingSignatureError):
        binding_svc.bind(acct_b.address, node_id, bad_sig, BINDING_ISSUED_AT, now_unix=BINDING_ISSUED_AT_UNIX)


# -----------------------------------------------------------------------------
# Display preference integration with the bound node_id
# -----------------------------------------------------------------------------


def test_display_preference_keyed_by_bound_node_id(now, price_source):
    """Dashboard toggles USD <-> FTNS using the bound node_id as the key."""
    acct = _new_account("a")
    binding_svc = WalletBindingService(store=InMemoryWalletBindingStore())
    node_id, _ = binding_svc.sign_in(acct.address)
    sig = _sign_binding(acct, acct.address, node_id)
    binding_svc.bind(acct.address, node_id, sig, BINDING_ISSUED_AT, now_unix=BINDING_ISSUED_AT_UNIX)

    pref_store = InMemoryDisplayPreferenceStore(default="usd")

    # Default display (unset).
    assert pref_store.get_mode(node_id) == "usd"
    assert (
        format_balance(Decimal("1.5"), price_source, mode=pref_store.get_mode(node_id))
        == "$3.00 · 1.5000 FTNS"
    )

    # User toggles to FTNS-only.
    pref_store.set_mode(node_id, "ftns")
    assert (
        format_balance(Decimal("1.5"), price_source, mode=pref_store.get_mode(node_id))
        == "1.5000 FTNS"
    )


def test_display_preference_persistence_across_sessions(now, price_source):
    """Plan §6 Task 5 acceptance: FTNS-toggle persists across pages."""
    acct = _new_account("a")
    binding_svc = WalletBindingService(store=InMemoryWalletBindingStore())
    node_id, _ = binding_svc.sign_in(acct.address)
    sig = _sign_binding(acct, acct.address, node_id)
    binding_svc.bind(acct.address, node_id, sig, BINDING_ISSUED_AT, now_unix=BINDING_ISSUED_AT_UNIX)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "prefs.sqlite"

        # Session 1 — set.
        session_1 = SqliteDisplayPreferenceStore(db_path)
        session_1.set_mode(node_id, "ftns")

        # Session 2 — reload against same file.
        session_2 = SqliteDisplayPreferenceStore(db_path)
        assert session_2.get_mode(node_id) == "ftns"


# -----------------------------------------------------------------------------
# Persistence — the full flow survives a process restart
# -----------------------------------------------------------------------------


def test_full_flow_survives_sqlite_restart(now, price_source):
    """Operator restart scenario: bind in session 1; session 2 (fresh
    process, new service instance against same sqlite file) returns the
    same binding."""
    acct = _new_account("a")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bindings.sqlite"

        # Session 1 — bind.
        store_1 = SqliteWalletBindingStore(db_path)
        svc_1 = WalletBindingService(store=store_1)
        node_id, is_new = svc_1.sign_in(acct.address)
        assert is_new is True
        sig = _sign_binding(acct, acct.address, node_id)
        binding_1 = svc_1.bind(acct.address, node_id, sig, BINDING_ISSUED_AT, now_unix=BINDING_ISSUED_AT_UNIX)

        # Session 2 — same sqlite file, new service instance.
        store_2 = SqliteWalletBindingStore(db_path)
        svc_2 = WalletBindingService(store=store_2)

        # Same wallet returns the same node_id.
        node_id_2, is_new_2 = svc_2.sign_in(acct.address)
        assert is_new_2 is False
        assert node_id_2 == node_id

        # Lookup by wallet returns the original binding.
        found = svc_2.get_by_wallet(acct.address)
        assert found == binding_1


# -----------------------------------------------------------------------------
# Price-source failure surface
# -----------------------------------------------------------------------------


def test_display_falls_back_to_ftns_only_when_price_oracle_stale():
    """If the price oracle is known-stale, the UI can flip to FTNS-only
    without the module knowing about oracle health. This test pins the
    integration shape: a stale-indicator fallback path using display
    mode preference, not an automatic oracle-health check.
    """
    price_source = StaticPriceSource(price_usd=Decimal("2.00"))
    balance = Decimal("0.5")

    # Nominal — USD default.
    assert (
        format_balance(balance, price_source, mode="usd")
        == "$1.00 · 0.5000 FTNS"
    )

    # Operator-detected stale oracle → switch user default to FTNS mode
    # (out-of-band, a dashboard-level decision).
    assert format_balance(balance, price_source, mode="ftns") == "0.5000 FTNS"
