"""Sprint 1046 — activate the Intel DCAP engine: wire it into the attestation
registry (Tier 3 activation).

sp1044 built IntelDCAPBackend (real ECDSA quote verification) but left it un-wired
— a verified component not yet reachable from /compute/receipt/verify. This wires
it: when an operator configures an Intel SGX Root CA
(PRSM_INTEL_SGX_ROOT_CA_PEM / _FILE), build_intel_dcap_backend_or_none constructs
the backend and it is registered at the FRONT of the registry so SGX quotes get
REAL cryptographic verification (vendor_verified can be true). Without the anchor
configured, behavior is unchanged: the structural IntelASPBackend handles SGX with
vendor_verified=False. DCAP uses handles_vendor='intel-sgx' so Intel TDX quotes
still fall through to the structural backend.
"""
from __future__ import annotations

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
# reuse the sp1044 hermetic self-consistent quote builder
from test_sprint_1044_intel_dcap_verify import build_self_consistent_quote


def _sgx_quote_and_root():
    quote, root_pem, _expected, _root_priv = build_self_consistent_quote()
    return quote, root_pem


# ── build_intel_dcap_backend_or_none ──────────────────────────────────────────


def test_build_uses_bundled_root_by_default():
    """sp1067 — with no operator root configured, the build defaults to the bundled
    (fingerprint-pinned) Intel SGX Root CA (real DCAP), removing the bring-up
    ceremony. The opt-out env restores the old structural-fallback behavior."""
    from prsm.compute.inference.intel_dcap import (
        build_intel_dcap_backend_or_none, IntelDCAPBackend)
    assert isinstance(build_intel_dcap_backend_or_none(env={}), IntelDCAPBackend)


def test_build_none_when_bundled_opted_out():
    from prsm.compute.inference.intel_dcap import build_intel_dcap_backend_or_none
    assert build_intel_dcap_backend_or_none(
        env={"PRSM_INTEL_SGX_USE_BUNDLED_ROOT": "0"}) is None


def test_build_from_pem_env():
    from prsm.compute.inference.intel_dcap import (
        build_intel_dcap_backend_or_none, IntelDCAPBackend)
    _quote, root_pem = _sgx_quote_and_root()
    backend = build_intel_dcap_backend_or_none(
        env={"PRSM_INTEL_SGX_ROOT_CA_PEM": root_pem.decode()})
    assert isinstance(backend, IntelDCAPBackend)
    assert backend.handles_vendor == "intel-sgx"


def test_build_from_file_env(tmp_path):
    from prsm.compute.inference.intel_dcap import (
        build_intel_dcap_backend_or_none, IntelDCAPBackend)
    _quote, root_pem = _sgx_quote_and_root()
    p = tmp_path / "intel_root.pem"
    p.write_bytes(root_pem)
    backend = build_intel_dcap_backend_or_none(
        env={"PRSM_INTEL_SGX_ROOT_CA_FILE": str(p)})
    assert isinstance(backend, IntelDCAPBackend)


def test_build_invalid_pem_returns_none_fail_open():
    from prsm.compute.inference.intel_dcap import build_intel_dcap_backend_or_none
    assert build_intel_dcap_backend_or_none(
        env={"PRSM_INTEL_SGX_ROOT_CA_PEM": "not a cert"}) is None


# ── registry activation ───────────────────────────────────────────────────────


def test_registry_does_real_verification_when_configured():
    """THE activation: with an Intel root configured, an SGX quote that chains to
    it gets vendor_verified=True through the registry (not just structural)."""
    from prsm.compute.inference.attestation_backends import (
        AttestationBackendRegistry, register_intel_dcap)
    quote, root_pem = _sgx_quote_and_root()
    reg = AttestationBackendRegistry()
    registered = register_intel_dcap(reg, env={"PRSM_INTEL_SGX_ROOT_CA_PEM": root_pem.decode()})
    assert registered is True
    res = reg.verify(quote)
    assert res.vendor == "intel-sgx"
    assert res.vendor_verified is True       # ← reachable in production now
    assert res.signature_chain_ok is True


def test_registry_structural_fallback_when_bundled_opted_out():
    """With DCAP opted out (PRSM_INTEL_SGX_USE_BUNDLED_ROOT=0) and no operator root,
    SGX falls through to the structural IntelASPBackend (vendor_verified=False)."""
    from prsm.compute.inference.attestation_backends import (
        AttestationBackendRegistry, register_intel_dcap)
    quote, _root = _sgx_quote_and_root()
    reg = AttestationBackendRegistry()
    assert register_intel_dcap(reg, env={"PRSM_INTEL_SGX_USE_BUNDLED_ROOT": "0"}) is False
    res = reg.verify(quote)
    assert res.vendor == "intel-sgx"
    assert res.vendor_verified is False
    assert res.structural_parse_ok is True


def test_dcap_does_not_hijack_tdx():
    """DCAP handles SGX only (handles_vendor=intel-sgx); a TDX v4 quote must still
    fall through to the structural IntelASPBackend, not get DCAP-rejected."""
    from prsm.compute.inference.attestation_backends import (
        AttestationBackendRegistry, register_intel_dcap)
    _quote, root_pem = _sgx_quote_and_root()
    reg = AttestationBackendRegistry()
    register_intel_dcap(reg, env={"PRSM_INTEL_SGX_ROOT_CA_PEM": root_pem.decode()})
    # minimal TDX v4 blob (version=4 + enough bytes for the structural parser)
    tdx = struct.pack("<H", 4) + b"\x00" * 300
    res = reg.verify(tdx)
    assert res.vendor == "intel-tdx"
    assert res.structural_parse_ok is True   # IntelASPBackend handled it
    assert res.vendor_verified is False


def test_register_is_idempotent():
    from prsm.compute.inference.attestation_backends import (
        AttestationBackendRegistry, register_intel_dcap)
    _quote, root_pem = _sgx_quote_and_root()
    reg = AttestationBackendRegistry()
    env = {"PRSM_INTEL_SGX_ROOT_CA_PEM": root_pem.decode()}
    assert register_intel_dcap(reg, env=env) is True
    assert register_intel_dcap(reg, env=env) is False    # already present
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    assert sum(isinstance(b, IntelDCAPBackend) for b in reg.backends) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
