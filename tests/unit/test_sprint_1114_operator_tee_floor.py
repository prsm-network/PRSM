"""Sprint 1114 — Domain-08 review MEDIUM-7: operator-mandated TEE floor.

The dispatch-time TEE-policy gate was requester-opt-in only (absent tee_policy → no
gating), and the node's attestation blob was never populated in production. So an
operator could not require "this node only serves >= hardware-verified work", and even a
requester-supplied hardware policy could never PASS (blob unset → tier none). This brick:
  - adds an operator floor (PRSM_MIN_ATTESTATION_TIER etc.) merged into EVERY request,
  - centralizes loading the node's raw attestation quote so the gate can admit real
    hardware work,
  - makes a misconfigured floor fail CLOSED.
"""
from __future__ import annotations

import base64

import pytest

from prsm.enterprise.tee_policy import (
    AttestationTier,
    TEEPolicy,
    operator_floor_policy_from_env,
)


# ── policy merge (stricter_of) ─────────────────────────────────────────────────────

def test_stricter_of_takes_higher_tier():
    a = TEEPolicy(min_attestation_tier=AttestationTier.SOFTWARE)
    b = TEEPolicy(min_attestation_tier=AttestationTier.HARDWARE_VERIFIED)
    assert a.stricter_of(b).min_attestation_tier == AttestationTier.HARDWARE_VERIFIED
    assert b.stricter_of(a).min_attestation_tier == AttestationTier.HARDWARE_VERIFIED


def test_stricter_of_intersects_vendors_and_ors_chain():
    a = TEEPolicy(allowed_vendors={"intel", "amd"}, require_signature_chain=False)
    b = TEEPolicy(allowed_vendors={"amd", "arm"}, require_signature_chain=True)
    m = a.stricter_of(b)
    assert m.allowed_vendors == {"amd"}
    assert m.require_signature_chain is True


def test_stricter_of_none_vendors_uses_other():
    a = TEEPolicy(allowed_vendors=None)
    b = TEEPolicy(allowed_vendors={"intel"})
    assert a.stricter_of(b).allowed_vendors == {"intel"}


def test_stricter_of_cannot_weaken():
    """The merged policy is never weaker than the requester's on any dimension."""
    requester = TEEPolicy(min_attestation_tier=AttestationTier.HARDWARE_VERIFIED)
    floor = TEEPolicy(min_attestation_tier=AttestationTier.SOFTWARE)
    assert requester.stricter_of(floor).min_attestation_tier == \
        AttestationTier.HARDWARE_VERIFIED


# ── operator floor from env ─────────────────────────────────────────────────────────

def test_no_env_means_no_floor():
    assert operator_floor_policy_from_env({}) is None


def test_tier_none_only_is_no_floor():
    assert operator_floor_policy_from_env({"PRSM_MIN_ATTESTATION_TIER": "none"}) is None


def test_floor_tier_parsed():
    p = operator_floor_policy_from_env(
        {"PRSM_MIN_ATTESTATION_TIER": "hardware_verified"})
    assert p is not None
    assert p.min_attestation_tier == AttestationTier.HARDWARE_VERIFIED


def test_floor_vendors_and_chain():
    p = operator_floor_policy_from_env({
        "PRSM_TEE_ALLOWED_VENDORS": "intel, amd",
        "PRSM_TEE_REQUIRE_SIGNATURE_CHAIN": "true",
    })
    assert p is not None
    assert p.allowed_vendors == {"intel", "amd"}
    assert p.require_signature_chain is True


def test_bad_tier_raises():
    with pytest.raises(ValueError):
        operator_floor_policy_from_env({"PRSM_MIN_ATTESTATION_TIER": "ultra"})


# ── blob loader ──────────────────────────────────────────────────────────────────

def test_loader_reads_b64(monkeypatch):
    from prsm.node.hardware_profile_loader import load_tee_attestation_blob
    quote = b"fake-sgx-quote-bytes"
    monkeypatch.setenv("PRSM_TEE_ATTESTATION_B64", base64.b64encode(quote).decode())
    assert load_tee_attestation_blob() == quote


def test_loader_none_when_unset(monkeypatch):
    from prsm.node.hardware_profile_loader import load_tee_attestation_blob
    monkeypatch.delenv("PRSM_TEE_ATTESTATION_B64", raising=False)
    monkeypatch.delenv("PRSM_TEE_ATTESTATION_FILE", raising=False)
    assert load_tee_attestation_blob() is None


def test_loader_rejects_oversized(monkeypatch):
    from prsm.node.hardware_profile_loader import load_tee_attestation_blob
    big = base64.b64encode(b"x" * (64 * 1024 + 1)).decode()
    monkeypatch.setenv("PRSM_TEE_ATTESTATION_B64", big)
    assert load_tee_attestation_blob() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
