"""Sprint 1333 (TEE Tier-B producer) — build_tee_runtime selector.

The stage server stamps each §7 receipt's tee_attestation from tee_runtime.get_attestation_bytes()
(→ the on-chain attestation commitment, roadmap F), but the runtime was HARDCODED to
SoftwareTEERuntime (dev-only). This selects the SEV-SNP/SGX hardware runtime per
PRSM_TEE_RUNTIME_KIND — node-bound, and FAIL-CLOSED for an explicit hardware kind on a host
without the device (never silently downgrade to a dev-only blob claiming hardware).
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.compute.tee.models import TEEType
from prsm.compute.tee.runtime import (
    SevSnpTEERuntime,
    SgxTEERuntime,
    SoftwareTEERuntime,
    TEEHardwareUnavailableError,
    build_tee_runtime,
)


def _avail(monkeypatch, cls, value):
    monkeypatch.setattr(cls, "available", property(lambda self: value))


# ── default / software (unchanged) ───────────────────────────────────────────

def test_default_unset_is_software():
    assert isinstance(build_tee_runtime(node_id="n", environ={}), SoftwareTEERuntime)


def test_explicit_software():
    assert isinstance(
        build_tee_runtime(node_id="n", environ={"PRSM_TEE_RUNTIME_KIND": "software"}),
        SoftwareTEERuntime)


def test_unknown_kind_falls_back_to_software():
    assert isinstance(
        build_tee_runtime(node_id="n", environ={"PRSM_TEE_RUNTIME_KIND": "bogus"}),
        SoftwareTEERuntime)


# ── explicit hardware kind: FAIL-CLOSED without the device ────────────────────

def test_explicit_sevsnp_fail_closed_when_no_device(monkeypatch):
    _avail(monkeypatch, SevSnpTEERuntime, False)
    with pytest.raises(TEEHardwareUnavailableError, match="explicitly requested"):
        build_tee_runtime(node_id="n", environ={"PRSM_TEE_RUNTIME_KIND": "sev-snp"})


def test_explicit_sgx_fail_closed_when_no_device(monkeypatch):
    _avail(monkeypatch, SgxTEERuntime, False)
    with pytest.raises(TEEHardwareUnavailableError):
        build_tee_runtime(node_id="n", environ={"PRSM_TEE_RUNTIME_KIND": "sgx"})


# ── explicit hardware kind: selected + node-bound when the device is present ──

@pytest.mark.parametrize("kind", ["sev-snp", "sevsnp", "sev"])
def test_explicit_sevsnp_selected_when_available(monkeypatch, kind):
    _avail(monkeypatch, SevSnpTEERuntime, True)
    rt = build_tee_runtime(node_id="node-xyz", environ={"PRSM_TEE_RUNTIME_KIND": kind})
    assert isinstance(rt, SevSnpTEERuntime)
    assert rt.tee_type == TEEType.SEV
    # node binding (sp1083): REPORT_DATA[:32] == sha256(node_id)
    assert rt.report_data()[:32] == hashlib.sha256(b"node-xyz").digest()


# ── auto: opportunistic (hardware if present, else software) ──────────────────

def test_auto_falls_back_to_software_when_no_device(monkeypatch):
    _avail(monkeypatch, SevSnpTEERuntime, False)
    _avail(monkeypatch, SgxTEERuntime, False)
    assert isinstance(
        build_tee_runtime(node_id="n", environ={"PRSM_TEE_RUNTIME_KIND": "auto"}),
        SoftwareTEERuntime)


def test_auto_selects_sevsnp_when_available(monkeypatch):
    _avail(monkeypatch, SevSnpTEERuntime, True)
    rt = build_tee_runtime(node_id="n", environ={"PRSM_TEE_RUNTIME_KIND": "auto"})
    assert isinstance(rt, SevSnpTEERuntime)


def test_auto_prefers_sevsnp_over_sgx(monkeypatch):
    _avail(monkeypatch, SevSnpTEERuntime, True)
    _avail(monkeypatch, SgxTEERuntime, True)
    rt = build_tee_runtime(node_id="n", environ={"PRSM_TEE_RUNTIME_KIND": "auto"})
    assert isinstance(rt, SevSnpTEERuntime)  # SEV-SNP is the validated platform, tried first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
