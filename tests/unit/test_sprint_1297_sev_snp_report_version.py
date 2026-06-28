"""Sprint 1297 — accept SEV-SNP report versions 2–5 (roadmap E hardware finding).

The on-VM validation (GCP N2D Milan confidential VM) generated a real, node-bound,
structurally-valid quote — but AMDSEVSNPBackend rejected it with "unsupported SEV-SNP
report version=5 (this backend parses v2)". Real AMD firmware now emits v5 reports. The
field offsets the verifier reads (report_data @0x50, measurement @0x90, chip_id @0x1A0,
reported_tcb @0x180) and the signed region report[:0x2A0] are identical across v2–v5 —
AMD only appends new fields into previously-reserved bytes — so the v2-only gate was
over-conservative. The gate now accepts v2–v5 (v6+ stays gated pending a layout review).
"""
from __future__ import annotations

from prsm.compute.inference.amd_sev_snp import AMDSEVSNPBackend, _MAX_SNP_REPORT_VERSION
from tests.unit.test_sprint_1049_amd_sev_snp_verify import build_sev_snp_envelope


def _verify(version):
    env, ark_pem, _ = build_sev_snp_envelope(version=version)
    return AMDSEVSNPBackend(trusted_root_pem=ark_pem).verify(env)


def test_v5_report_now_verifies():
    """The exact case from the GCP Milan VM: a v5 report must verify end-to-end."""
    res = _verify(5)
    assert res.structural_parse_ok is True
    assert res.signature_chain_ok is True
    assert res.vendor_verified is True
    assert res.error is None


def test_v2_and_v3_still_verify():
    for v in (2, 3):
        res = _verify(v)
        assert res.vendor_verified is True, f"v{v} should verify"
        assert res.error is None


def test_max_version_is_five():
    assert _MAX_SNP_REPORT_VERSION == 5


def test_v6_still_gated():
    """A version above the validated range is rejected (not silently mis-parsed) so a
    future ABI change must be reviewed before it's trusted."""
    res = _verify(6)
    assert res.vendor_verified is False
    assert res.error is not None and "version=6" in res.error


def test_v1_rejected():
    res = _verify(1)
    assert res.vendor_verified is False
    assert res.error is not None and "version=1" in res.error


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
