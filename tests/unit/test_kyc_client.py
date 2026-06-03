"""Sprint 280 — KYCClient (pluggable KYC vendor adapter).

KYC is the regulatory gate for fiat on/off-ramps in US + EU
jurisdictions. PRSM doesn't roll its own KYC — instead this
adapter wraps a third-party vendor (Persona, Onfido, Plaid
Identity Verification, etc.) behind a uniform interface so
swapping vendors is a config change, not a code change.

PENDING_COMMISSION pattern (mirrors WaaS / paymaster): when
KYC_VENDOR_API_KEY is absent, the adapter returns preview
records without hitting any external vendor. Once
commissioned, delegates to a dependency-injected backend that
wraps the real vendor SDK.

Status transitions:
  NOT_STARTED → INITIATED → PENDING → VERIFIED | REJECTED
  VERIFIED can later become EXPIRED (re-KYC required after
  vendor-defined window).
  PENDING_COMMISSION is orthogonal — shown only when adapter
  has no vendor configured.

Levels:
  basic     — light KYC (selfie + ID); supports small fiat amts
  enhanced  — full KYC (proof of address + source of funds);
              required for higher fiat limits

Per Vision §14 "Crypto-UX adoption barrier" mitigation: the
KYC flow itself is vendor-hosted (Persona's modal etc.), so
the user never installs anything. PRSM's job is just to track
the verification record and gate fiat operations on it.
"""
from __future__ import annotations

import json

import pytest

from prsm.economy.web3.kyc_client import (
    KYCClient, KYCRecord,
    KYC_STATUS_NOT_STARTED, KYC_STATUS_INITIATED,
    KYC_STATUS_PENDING, KYC_STATUS_VERIFIED,
    KYC_STATUS_REJECTED, KYC_STATUS_EXPIRED,
    KYC_STATUS_PENDING_COMMISSION,
    KYC_LEVEL_BASIC, KYC_LEVEL_ENHANCED,
)


class FakeVendorBackend:
    """Test backend mirroring KYC vendor SDK surface.

    Generates deterministic vendor refs so tests can assert
    exact values."""

    def __init__(self, vendor_name="persona"):
        self.vendor_name = vendor_name
        self.initiated = []

    def initiate_session(self, user_id, email, level):
        self.initiated.append((user_id, email, level))
        return {
            "vendor_ref": f"{self.vendor_name}-session-{user_id}",
            "session_url": (
                f"https://{self.vendor_name}.example/verify/"
                f"{user_id}"
            ),
            "status": KYC_STATUS_INITIATED,
        }


# ── PENDING_COMMISSION ───────────────────────────────────


def test_from_env_uncommissioned_when_key_missing(monkeypatch):
    monkeypatch.delenv("KYC_VENDOR_API_KEY", raising=False)
    monkeypatch.delenv("KYC_VENDOR", raising=False)
    c = KYCClient.from_env()
    assert c is not None
    assert c.is_commissioned() is False


def test_initiate_uncommissioned_returns_pending_commission():
    c = KYCClient()
    rec = c.initiate(
        user_id="alice", email="a@x.io",
        level=KYC_LEVEL_BASIC,
    )
    assert rec.user_id == "alice"
    assert rec.status == KYC_STATUS_PENDING_COMMISSION
    assert rec.vendor_ref is None
    assert rec.session_url is None


def test_initiate_uncommissioned_does_not_hit_backend():
    fake = FakeVendorBackend()
    c = KYCClient(backend=fake)  # no API key
    c.initiate(user_id="alice", email="a@x.io", level="basic")
    assert fake.initiated == []


# ── COMMISSIONED ─────────────────────────────────────────


def test_initiate_commissioned_calls_backend():
    fake = FakeVendorBackend()
    c = KYCClient(
        vendor="persona", api_key="k", backend=fake,
    )
    assert c.is_commissioned() is True
    rec = c.initiate(
        user_id="alice", email="a@x.io",
        level=KYC_LEVEL_BASIC,
    )
    assert rec.status == KYC_STATUS_INITIATED
    assert rec.vendor == "persona"
    assert rec.vendor_ref == "persona-session-alice"
    assert "persona.example/verify/alice" in rec.session_url
    assert rec.level == KYC_LEVEL_BASIC
    assert fake.initiated == [("alice", "a@x.io", "basic")]


def test_initiate_idempotent_for_active_session():
    """Once a session is INITIATED for a user, re-calling
    initiate returns the existing record without a new vendor
    session — until status flips to VERIFIED/REJECTED/EXPIRED."""
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    first = c.initiate(
        user_id="alice", email="a@x.io",
        level=KYC_LEVEL_BASIC,
    )
    second = c.initiate(
        user_id="alice", email="a@x.io",
        level=KYC_LEVEL_BASIC,
    )
    assert first.vendor_ref == second.vendor_ref
    assert len(fake.initiated) == 1


def test_verified_user_can_upgrade_level():
    """sp967 — a VERIFIED basic user requesting 'enhanced' must START a new
    enhanced vendor session (the documented upgrade path). Previously the
    idempotency guard returned the unchanged basic record for ANY
    non-reinitiatable status, so VERIFIED users could never upgrade — capping
    their AML tier forever."""
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    c.initiate(user_id="alice", email="a@x.io", level=KYC_LEVEL_BASIC)
    c.update_status("alice", KYC_STATUS_VERIFIED)
    rec = c.initiate(user_id="alice", email="a@x.io", level=KYC_LEVEL_ENHANCED)
    assert rec.level == KYC_LEVEL_ENHANCED
    assert rec.status == KYC_STATUS_INITIATED
    assert len(fake.initiated) == 2
    assert fake.initiated[-1] == ("alice", "a@x.io", "enhanced")


def test_verified_user_same_level_still_idempotent():
    """A VERIFIED user re-requesting the SAME level stays idempotent (no new
    vendor session, record unchanged) — only a level change re-initiates."""
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    c.initiate(user_id="alice", email="a@x.io", level=KYC_LEVEL_BASIC)
    c.update_status("alice", KYC_STATUS_VERIFIED)
    rec = c.initiate(user_id="alice", email="a@x.io", level=KYC_LEVEL_BASIC)
    assert rec.status == KYC_STATUS_VERIFIED  # unchanged
    assert rec.level == KYC_LEVEL_BASIC
    assert len(fake.initiated) == 1  # no new session


def test_uncommissioned_verified_user_can_upgrade_to_pending_commission():
    """Even uncommissioned, a level change must re-initiate (→ PENDING_COMMISSION
    enhanced) rather than returning the stale basic record."""
    c = KYCClient()  # uncommissioned
    c.initiate(user_id="alice", email="a@x.io", level=KYC_LEVEL_BASIC)
    # Force a VERIFIED basic record (simulating a prior commissioned verification).
    c.update_status("alice", KYC_STATUS_VERIFIED)
    rec = c.initiate(user_id="alice", email="a@x.io", level=KYC_LEVEL_ENHANCED)
    assert rec.level == KYC_LEVEL_ENHANCED
    assert rec.status == KYC_STATUS_PENDING_COMMISSION


def test_initiate_re_initiates_after_rejection():
    """REJECTED → re-initiate should start a fresh session."""
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    c.initiate(user_id="alice", email="a@x.io", level="basic")
    c.update_status("alice", KYC_STATUS_REJECTED)
    # Now re-initiate
    rec = c.initiate(
        user_id="alice", email="a@x.io", level="basic",
    )
    assert rec.status == KYC_STATUS_INITIATED
    assert len(fake.initiated) == 2


def test_initiate_re_initiates_after_expiry():
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    c.initiate(user_id="alice", email="a@x.io", level="basic")
    c.update_status("alice", KYC_STATUS_EXPIRED)
    rec = c.initiate(
        user_id="alice", email="a@x.io", level="basic",
    )
    assert rec.status == KYC_STATUS_INITIATED
    assert len(fake.initiated) == 2


def test_initiate_validates_required_fields():
    c = KYCClient(
        vendor="persona", api_key="k", backend=FakeVendorBackend(),
    )
    with pytest.raises(ValueError):
        c.initiate(user_id="", email="a@x.io", level="basic")
    with pytest.raises(ValueError):
        c.initiate(user_id="alice", email="", level="basic")
    with pytest.raises(ValueError):
        c.initiate(user_id="alice", email="a@x.io", level="")


def test_initiate_validates_level():
    c = KYCClient(
        vendor="persona", api_key="k", backend=FakeVendorBackend(),
    )
    with pytest.raises(ValueError):
        c.initiate(
            user_id="alice", email="a@x.io",
            level="ultra-mega",
        )


# ── Status transitions ───────────────────────────────────


def test_get_status_returns_none_for_unknown():
    c = KYCClient()
    assert c.get_status("nobody") is None


def test_update_status_happy_path():
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    c.initiate(user_id="alice", email="a@x.io", level="basic")
    updated = c.update_status(
        "alice", KYC_STATUS_VERIFIED,
        vendor_ref_update="persona-final-ref",
    )
    assert updated is not None
    assert updated.status == KYC_STATUS_VERIFIED
    assert updated.vendor_ref == "persona-final-ref"
    assert updated.verified_at > 0


def test_update_status_validates():
    c = KYCClient()
    with pytest.raises(ValueError):
        c.update_status("alice", "wat")


def test_update_status_unknown_user_returns_none():
    c = KYCClient()
    assert c.update_status("nobody", KYC_STATUS_VERIFIED) is None


# ── is_verified helper ───────────────────────────────────


def test_is_verified_false_for_unknown():
    c = KYCClient()
    assert c.is_verified("alice") is False


def test_is_verified_true_after_verification():
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    c.initiate(user_id="alice", email="a@x.io", level="basic")
    c.update_status("alice", KYC_STATUS_VERIFIED)
    assert c.is_verified("alice") is True


def test_is_verified_false_when_rejected():
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    c.initiate(user_id="alice", email="a@x.io", level="basic")
    c.update_status("alice", KYC_STATUS_REJECTED)
    assert c.is_verified("alice") is False


def test_is_verified_false_when_expired():
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    c.initiate(user_id="alice", email="a@x.io", level="basic")
    c.update_status("alice", KYC_STATUS_VERIFIED)
    c.update_status("alice", KYC_STATUS_EXPIRED)
    assert c.is_verified("alice") is False


# ── Persistence ──────────────────────────────────────────


def test_persistence_round_trip(tmp_path):
    fake = FakeVendorBackend()
    c1 = KYCClient(
        vendor="persona", api_key="k",
        persist_dir=tmp_path, backend=fake,
    )
    c1.initiate(user_id="alice", email="a@x.io", level="basic")
    c1.update_status("alice", KYC_STATUS_VERIFIED)
    c2 = KYCClient(
        vendor="persona", api_key="k",
        persist_dir=tmp_path, backend=FakeVendorBackend(),
    )
    rec = c2.get_status("alice")
    assert rec is not None
    assert rec.status == KYC_STATUS_VERIFIED


def test_persistence_corrupt_file_fail_soft(tmp_path):
    (tmp_path / "garbage.json").write_text("{not valid json")
    c = KYCClient(persist_dir=tmp_path)
    assert c.get_status("alice") is None


# ── KYCRecord round-trip ─────────────────────────────────


def test_record_round_trip():
    r = KYCRecord(
        user_id="alice", email="a@x.io",
        vendor="persona", vendor_ref="ref-1",
        session_url="https://x.io/v/alice",
        level=KYC_LEVEL_ENHANCED,
        status=KYC_STATUS_VERIFIED,
        created_at=100.0, verified_at=200.0,
    )
    d = r.to_dict()
    restored = KYCRecord.from_dict(d)
    assert restored == r


# ── Enhanced level pass-through ──────────────────────────


def test_enhanced_level_passed_to_backend():
    fake = FakeVendorBackend()
    c = KYCClient(vendor="persona", api_key="k", backend=fake)
    c.initiate(
        user_id="alice", email="a@x.io",
        level=KYC_LEVEL_ENHANCED,
    )
    assert fake.initiated[-1][2] == KYC_LEVEL_ENHANCED


# ── Multi-vendor pluggability ────────────────────────────


def test_vendor_name_threaded_through():
    fake = FakeVendorBackend(vendor_name="onfido")
    c = KYCClient(vendor="onfido", api_key="k", backend=fake)
    rec = c.initiate(
        user_id="alice", email="a@x.io", level="basic",
    )
    assert rec.vendor == "onfido"
    assert "onfido" in rec.session_url


def test_supported_vendors_list():
    """The client should advertise which vendors it knows by
    name — useful for operator UX surfacing valid KYC_VENDOR
    env values."""
    assert "persona" in KYCClient.SUPPORTED_VENDORS
    assert "onfido" in KYCClient.SUPPORTED_VENDORS
    assert "plaid" in KYCClient.SUPPORTED_VENDORS


def test_from_env_unknown_vendor_falls_back_to_uncommissioned(
    monkeypatch,
):
    monkeypatch.setenv("KYC_VENDOR", "made-up-vendor")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    c = KYCClient.from_env()
    # Unknown vendor → adapter still constructs but is_commissioned
    # returns False since we don't know how to call this vendor.
    assert c.is_commissioned() is False
