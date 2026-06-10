"""Sprint 1050 — activate the AMD SEV-SNP engine: wire it into the attestation
registry (Tier 3 activation, parallel to sp1046 for Intel).

sp1049 built AMDSEVSNPBackend but left it un-wired. This wires it: when an operator
configures an AMD ARK (PRSM_AMD_SEV_SNP_ARK_PEM/_FILE), the backend is registered at
the FRONT of the registry and the PRSM SEV-SNP envelope (b"PRSMSNP1"…) is routed to
it for REAL verification. Routing detail: the envelope gets a DISTINCT detect_vendor
tag ("amd-sev-snp-envelope") that only AMDSEVSNPBackend claims, so a RAW SEV-SNP
report (version u32==2) still routes to the structural AMDKDSBackend (no hijack).
Without the ARK configured: unchanged behavior.
"""
from __future__ import annotations

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_sprint_1049_amd_sev_snp_verify import build_sev_snp_envelope


def _raw_v2_report() -> bytes:
    r = bytearray(1184)
    struct.pack_into("<I", r, 0, 2)          # version 2 raw report (no envelope)
    r[144:192] = b"\x5a" * 48                # measurement (AMDKDSBackend offsets)
    return bytes(r)


def test_detect_vendor_recognizes_envelope():
    from prsm.compute.inference.attestation_backends import detect_vendor
    env, _ark, _ = build_sev_snp_envelope()
    assert detect_vendor(env) == "amd-sev-snp-envelope"
    # a raw report is still plain amd-sev-snp
    assert detect_vendor(_raw_v2_report()) == "amd-sev-snp"


def test_build_amd_backend_none_without_config():
    from prsm.compute.inference.amd_sev_snp import build_amd_sev_snp_backend_or_none
    assert build_amd_sev_snp_backend_or_none(env={}) is None


def test_register_amd_activates_real_verification():
    from prsm.compute.inference.attestation_backends import (
        AttestationBackendRegistry, register_amd_sev_snp)
    env, ark_pem, expected = build_sev_snp_envelope()
    reg = AttestationBackendRegistry()
    assert register_amd_sev_snp(reg, env={"PRSM_AMD_SEV_SNP_ARK_PEM": ark_pem.decode()}) is True
    res = reg.verify(env)
    assert res.vendor == "amd-sev-snp"
    assert res.vendor_verified is True       # reachable in production now
    assert res.signature_chain_ok is True
    assert res.vendor_data["measurement_hex"] == expected["measurement"]


def test_register_amd_none_without_config():
    from prsm.compute.inference.attestation_backends import (
        AttestationBackendRegistry, register_amd_sev_snp)
    reg = AttestationBackendRegistry()
    assert register_amd_sev_snp(reg, env={}) is False


def test_raw_sev_snp_report_still_structural_when_amd_wired():
    """A RAW SEV-SNP report must NOT be hijacked by the envelope-only real backend;
    it still routes to the structural AMDKDSBackend (vendor_verified=False)."""
    from prsm.compute.inference.attestation_backends import (
        AttestationBackendRegistry, register_amd_sev_snp)
    _env, ark_pem, _ = build_sev_snp_envelope()
    reg = AttestationBackendRegistry()
    register_amd_sev_snp(reg, env={"PRSM_AMD_SEV_SNP_ARK_PEM": ark_pem.decode()})
    res = reg.verify(_raw_v2_report())
    assert res.vendor == "amd-sev-snp"
    assert res.vendor_verified is False
    assert res.structural_parse_ok is True   # structural backend handled it


def test_register_amd_idempotent():
    from prsm.compute.inference.attestation_backends import (
        AttestationBackendRegistry, register_amd_sev_snp)
    _env, ark_pem, _ = build_sev_snp_envelope()
    reg = AttestationBackendRegistry()
    env = {"PRSM_AMD_SEV_SNP_ARK_PEM": ark_pem.decode()}
    assert register_amd_sev_snp(reg, env=env) is True
    assert register_amd_sev_snp(reg, env=env) is False   # already present
    from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend
    assert sum(isinstance(b, AMDSEVSNPBackend) for b in reg.backends) == 1


def test_configure_default_registry_registers_both_vendors():
    """configure_default_registry_from_env activates Intel AND AMD when both anchors
    are configured (the node-startup entrypoint)."""
    from prsm.compute.inference.attestation_backends import (
        AttestationBackendRegistry, configure_attestation_registry)
    from test_sprint_1044_intel_dcap_verify import build_self_consistent_quote
    iq, iroot, _e, _r = build_self_consistent_quote()
    aenv, aark, _ax = build_sev_snp_envelope()
    reg = AttestationBackendRegistry()
    configure_attestation_registry(reg, env={
        "PRSM_INTEL_SGX_ROOT_CA_PEM": iroot.decode(),
        "PRSM_AMD_SEV_SNP_ARK_PEM": aark.decode(),
    })
    assert reg.verify(iq).vendor_verified is True       # Intel SGX real
    assert reg.verify(aenv).vendor_verified is True      # AMD SEV-SNP real


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
