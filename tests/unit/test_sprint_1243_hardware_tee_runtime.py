"""Sprint 1243 (TEE Tier-3 roadmap E) — SgxTEERuntime / SevSnpTEERuntime.

Real hardware TEE runtimes conforming to the get_attestation_bytes contract
(sp1239). DEVICE-GATED (available=False without the platform device, so the
selector never wrongly picks them), node-bound REPORT_DATA (sp1083), execution
delegated to the WASM sandbox. The hardware quote PRODUCTION is
HARDWARE-VALIDATION-PENDING (no TEE box in CI) — it raises loudly rather than
emit anything dev-only or unvalidated. These tests cover everything validatable
WITHOUT hardware; the actual quote-gen awaits a real SGX/SEV VM (user chose
code-now-defer-validation).
"""

from unittest.mock import patch

import pytest

from prsm.compute.inference.attestation_backends import expected_attestation_report_data
from prsm.compute.inference.executor import SOFTWARE_TEE_ATTESTATION_PREFIX
from prsm.compute.tee.models import TEEType
from prsm.compute.tee.runtime import (
    SevSnpTEERuntime,
    SgxTEERuntime,
    SoftwareTEERuntime,
    TEEHardwareUnavailableError,
    TEERuntime,
)


def test_are_teeruntime_subclasses_and_instantiable():
    for cls in (SevSnpTEERuntime, SgxTEERuntime):
        rt = cls(node_id="node-A")
        assert isinstance(rt, TEERuntime)


def test_tee_type_and_name():
    assert SevSnpTEERuntime().tee_type == TEEType.SEV
    assert SevSnpTEERuntime().name == "sev-snp"
    assert SgxTEERuntime().tee_type == TEEType.SGX
    assert SgxTEERuntime().name == "sgx"


def test_available_false_without_device():
    with patch("os.path.exists", return_value=False):
        assert SevSnpTEERuntime().available is False
        assert SgxTEERuntime().available is False


def test_available_true_with_device():
    with patch("os.path.exists", side_effect=lambda p: p == "/dev/sev-guest"):
        assert SevSnpTEERuntime().available is True
    with patch("os.path.exists", side_effect=lambda p: p == "/dev/sgx_enclave"):
        assert SgxTEERuntime().available is True


def test_report_data_binds_node_id():
    rd = SevSnpTEERuntime(node_id="node-A").report_data()
    assert len(rd) == 64                       # 64-byte REPORT_DATA
    assert rd[32:] == b"\x00" * 32             # second half zero-padded
    # first 32 bytes == sha256(node_id) — matches the sp1083 verifier binding
    assert rd[:32].hex() == expected_attestation_report_data("node-A")


def test_get_attestation_raises_loudly_without_device_NO_fallback():
    # THE contract: a hardware tier must never silently emit a dev-only blob.
    with patch("os.path.exists", return_value=False):
        for cls in (SevSnpTEERuntime, SgxTEERuntime):
            with pytest.raises(TEEHardwareUnavailableError):
                cls(node_id="n").get_attestation_bytes()


def test_get_attestation_never_emits_unvalidated_quote_even_with_device():
    # even when the device is "present", quote-gen is hardware-validation-pending
    # → it raises rather than fabricate/dev-only (no silent unvalidated quote).
    with patch("os.path.exists", side_effect=lambda p: p == "/dev/sev-guest"):
        with pytest.raises(NotImplementedError):
            SevSnpTEERuntime(node_id="n").get_attestation_bytes()
    with patch("os.path.exists", side_effect=lambda p: p == "/dev/sgx_enclave"):
        with pytest.raises(NotImplementedError):
            SgxTEERuntime(node_id="n").get_attestation_bytes()


def test_software_runtime_still_returns_dev_only_blob():
    # contrast: SoftwareTEERuntime legitimately returns the DEV-ONLY blob; the
    # hardware runtimes are the ones forbidden from doing so.
    assert SoftwareTEERuntime().get_attestation_bytes().startswith(
        SOFTWARE_TEE_ATTESTATION_PREFIX)
