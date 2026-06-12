"""Sprint 1083 — verified-tier-attestation bridge (activates the Tier-3 investment).

The §7 engines (sp1044-1082) compute `vendor_verified`, but the parallax tier gate
(sp702 TierGateAdapter) filters on a SELF-ADVERTISED `tier_attestation` string prefix —
`is_hardware_attestation` explicitly does not verify the claim. So a node could advertise
"tier-sgx" with no real attestation and be routed confidential Tier B/C work. This bridge
closes that gap: `verified_tier_attestation(blob)` returns a hardware tier string ONLY when
the blob is a real, cryptographically `vendor_verified` attestation; otherwise "tier-none".
A node's eligibility for confidential work now depends on a VERIFIED attestation.
"""
from __future__ import annotations

import pytest

from prsm.compute.parallax_scheduling.trust_adapter import (
    expected_attestation_report_data,
    is_hardware_attestation,
    verified_tier_attestation,
)
from prsm.compute.parallax_scheduling.prsm_types import TIER_ATTESTATION_NONE

_NODE = "n" * 32
# A REPORT_DATA (64 bytes / 128 hex) committing to _NODE in its first 32 bytes.
_BOUND_RD = expected_attestation_report_data(_NODE) + "00" * 32


class _FakeResult:
    def __init__(self, vendor, vendor_verified, report_data_hex=_BOUND_RD):
        self.vendor = vendor
        self.vendor_verified = vendor_verified
        self.vendor_data = {"report_data_hex": report_data_hex}
        self.error = None


class _FakeRegistry:
    def __init__(self, result):
        self._result = result

    def verify(self, blob):
        return self._result


def _reg(vendor, verified, report_data_hex=_BOUND_RD):
    return _FakeRegistry(_FakeResult(vendor, verified, report_data_hex))


# ── the gate: vendor_verified is REQUIRED for a hardware tier ────────────────────

def test_verified_intel_sgx_maps_to_tier_sgx():
    # bound to _NODE (the report_data commits to it) → eligible
    tier = verified_tier_attestation(b"quote", node_id=_NODE, registry=_reg("intel-sgx", True))
    assert tier == "tier-sgx"
    assert is_hardware_attestation(tier) is True   # now eligible for confidential work


def test_replayed_quote_for_other_node_rejected():
    """The replay/binding guard: a VERIFIED quote whose REPORT_DATA commits to node A
    must NOT grant node B a hardware tier (otherwise a software node replays a genuine
    TEE node's quote and receives confidential work)."""
    # quote bound to _NODE, but evaluated for a DIFFERENT node_id
    tier = verified_tier_attestation(
        b"quote", node_id="b" * 32, registry=_reg("intel-sgx", True))
    assert tier == TIER_ATTESTATION_NONE


def test_missing_report_data_rejected_when_binding():
    tier = verified_tier_attestation(
        b"quote", node_id=_NODE, registry=_reg("intel-sgx", True, report_data_hex=""))
    assert tier == TIER_ATTESTATION_NONE


def test_no_node_id_skips_binding():
    """Without node_id the binding is skipped (callers that bind elsewhere)."""
    assert verified_tier_attestation(b"q", registry=_reg("intel-sgx", True)) == "tier-sgx"


def test_oversized_blob_rejected():
    """An oversized blob (not a real ≤~10KB TEE quote) is refused before verification —
    closes the gossip-relay DoS (review LOW)."""
    big = b"\x00" * (200 * 1024)
    assert verified_tier_attestation(big, registry=_reg("intel-sgx", True)) == TIER_ATTESTATION_NONE
    import base64
    big_b64 = base64.b64encode(big).decode()
    assert verified_tier_attestation(big_b64, registry=_reg("intel-sgx", True)) == TIER_ATTESTATION_NONE


def test_merge_attestation_oversized_b64_skipped(monkeypatch):
    import base64
    from prsm.node.hardware_profile_loader import _merge_attestation
    monkeypatch.setenv("PRSM_TEE_ATTESTATION_B64", base64.b64encode(b"\x00" * (200 * 1024)).decode())
    data = {}
    _merge_attestation(data)
    assert "attestation" not in data


def test_verified_amd_sev_snp_maps_to_tier_sev():
    assert verified_tier_attestation(b"q", registry=_reg("amd-sev-snp", True)) == "tier-sev"


def test_verified_intel_tdx_maps_to_tier_tdx():
    assert verified_tier_attestation(b"q", registry=_reg("intel-tdx", True)) == "tier-tdx"


def test_unverified_hardware_vendor_is_tier_none():
    """A real-vendor quote that did NOT cryptographically verify must NOT grant a
    hardware tier — this is the whole point (a structural/forged claim is rejected)."""
    tier = verified_tier_attestation(b"q", registry=_reg("intel-sgx", False))
    assert tier == TIER_ATTESTATION_NONE
    assert is_hardware_attestation(tier) is False


def test_software_fallback_is_tier_none():
    assert verified_tier_attestation(
        b"q", registry=_reg("software-fallback", False)) == TIER_ATTESTATION_NONE


def test_unknown_vendor_is_tier_none():
    assert verified_tier_attestation(
        b"q", registry=_reg("unknown", True)) == TIER_ATTESTATION_NONE


def test_none_blob_is_tier_none():
    assert verified_tier_attestation(None, registry=_reg("intel-sgx", True)) == TIER_ATTESTATION_NONE
    assert verified_tier_attestation(b"", registry=_reg("intel-sgx", True)) == TIER_ATTESTATION_NONE


def test_registry_raising_is_tier_none():
    class _Boom:
        def verify(self, blob):
            raise RuntimeError("backend exploded")
    # fail-closed: an error during verification must never grant a hardware tier
    assert verified_tier_attestation(b"q", registry=_Boom()) == TIER_ATTESTATION_NONE


def test_base64_string_blob_is_decoded():
    import base64
    b64 = base64.b64encode(b"quotebytes").decode()
    assert verified_tier_attestation(b64, registry=_reg("intel-sgx", True)) == "tier-sgx"


# ── brick 2: the pool provider sets tier_attestation from a VERIFIED blob ────────

_HW = {"tflops_fp16": 4.6, "gpu_vram_gb": 24.0, "ram_total_gb": 64.0,
       "gpu_name": "TEE Box", "gpu_api": "cuda"}


def test_pool_gpu_tier_is_none_without_attestation():
    """No attestation blob → tier-none (the safe default; can't serve confidential)."""
    from prsm.node.dht_backed_pool_provider import _hw_dict_to_parallax_gpu
    gpu = _hw_dict_to_parallax_gpu("n" * 32, dict(_HW), "us-east")
    assert gpu is not None
    assert gpu.tier_attestation == TIER_ATTESTATION_NONE


def test_pool_gpu_tier_set_from_verified_attestation(monkeypatch):
    """A node carrying an attestation blob gets a hardware tier ONLY via the verifying
    bridge — not by self-advertising the string."""
    import prsm.node.dht_backed_pool_provider as mod
    seen = {}

    def fake_verify(blob, *, node_id=None):
        seen["blob"] = blob
        seen["node_id"] = node_id
        return "tier-sgx"
    monkeypatch.setattr(mod, "verified_tier_attestation", fake_verify)
    hw = dict(_HW, attestation="QUOTEB64")
    gpu = mod._hw_dict_to_parallax_gpu("n" * 32, hw, "us-east")
    assert gpu.tier_attestation == "tier-sgx"
    assert seen["blob"] == "QUOTEB64"   # the bridge was actually called with the blob
    assert seen["node_id"] == "n" * 32   # ...and bound to the node's id


def test_pool_gpu_tier_none_when_attestation_unverified(monkeypatch):
    """A self-advertised but unverifiable attestation → tier-none (the closed gap)."""
    import prsm.node.dht_backed_pool_provider as mod
    monkeypatch.setattr(mod, "verified_tier_attestation",
                        lambda blob, *, node_id=None: TIER_ATTESTATION_NONE)
    hw = dict(_HW, attestation="FORGED")
    gpu = mod._hw_dict_to_parallax_gpu("n" * 32, hw, "us-east")
    assert gpu.tier_attestation == TIER_ATTESTATION_NONE


# ── brick 3: the announce side publishes the attestation quote ──────────────────

def test_merge_attestation_from_b64_env(monkeypatch):
    import base64
    from prsm.node.hardware_profile_loader import _merge_attestation
    quote = base64.b64encode(b"a real quote").decode()
    monkeypatch.setenv("PRSM_TEE_ATTESTATION_B64", quote)
    data = {}
    _merge_attestation(data)
    assert data["attestation"] == quote


def test_merge_attestation_from_file_env(monkeypatch, tmp_path):
    import base64
    from prsm.node.hardware_profile_loader import _merge_attestation
    monkeypatch.delenv("PRSM_TEE_ATTESTATION_B64", raising=False)
    qf = tmp_path / "quote.bin"
    qf.write_bytes(b"\x03\x00binary quote bytes")
    monkeypatch.setenv("PRSM_TEE_ATTESTATION_FILE", str(qf))
    data = {}
    _merge_attestation(data)
    assert base64.b64decode(data["attestation"]) == b"\x03\x00binary quote bytes"


def test_merge_attestation_absent_is_noop(monkeypatch):
    from prsm.node.hardware_profile_loader import _merge_attestation
    monkeypatch.delenv("PRSM_TEE_ATTESTATION_B64", raising=False)
    monkeypatch.delenv("PRSM_TEE_ATTESTATION_FILE", raising=False)
    data = {}
    _merge_attestation(data)
    assert "attestation" not in data   # non-TEE node advertises nothing — unchanged


def test_merge_attestation_invalid_b64_skipped(monkeypatch):
    from prsm.node.hardware_profile_loader import _merge_attestation
    monkeypatch.setenv("PRSM_TEE_ATTESTATION_B64", "!!!not base64!!!")
    data = {}
    _merge_attestation(data)
    assert "attestation" not in data   # fail-safe: garbage → no advertise → tier-none


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
