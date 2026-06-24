"""
TEE Runtime Interface
=====================

Abstract interface for trusted execution environments.
SoftwareTEERuntime wraps the Ring 1 WASM sandbox as a fallback.
"""

import abc
import logging
from typing import Any

from prsm.compute.tee.models import TEEType
from prsm.compute.wasm.models import ExecutionResult, ResourceLimits

logger = logging.getLogger(__name__)


class TEERuntime(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Runtime name."""

    @property
    @abc.abstractmethod
    def tee_type(self) -> TEEType:
        """Type of TEE."""

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """Whether available on current hardware."""

    @abc.abstractmethod
    def load(self, wasm_bytes: bytes) -> Any:
        """Load a WASM module."""

    @abc.abstractmethod
    def execute(self, module: Any, input_data: bytes, resource_limits: ResourceLimits) -> ExecutionResult:
        """Execute in TEE sandbox."""

    @abc.abstractmethod
    def get_attestation_bytes(self) -> bytes:
        """Sprint 1239 (TEE Tier-3 roadmap D) — generate this runtime's
        attestation quote bytes for the current enclave/measurement. The
        quote-GENERATION half of the attestation contract (verification is
        attestation_backends / the sp1236 multi-stage chain). chain_rpc/server.py
        stamps the returned bytes onto each stage's §7 proof; verifiers route
        them through the AttestationBackendRegistry.

        ABSTRACT on purpose: a TEE runtime MUST declare how it attests — a
        hardware runtime that forgot to implement this would otherwise silently
        emit a software/dev-only blob, masking the bug. ``SoftwareTEERuntime``
        returns a DEV-ONLY-marked blob (NO confidentiality — verifiers MUST
        reject it as non-production); real ``SgxTEERuntime``/``SevTEERuntime``
        return a hardware quote (sprint E, needs hardware)."""


class SoftwareTEERuntime(TEERuntime):
    """Software-only TEE using Ring 1 WASM sandbox as fallback."""

    def __init__(self):
        self._wasm_runtime = None

    def _get_wasm_runtime(self):
        if self._wasm_runtime is None:
            from prsm.compute.wasm.runtime import WasmtimeRuntime
            self._wasm_runtime = WasmtimeRuntime()
        return self._wasm_runtime

    @property
    def name(self) -> str:
        return "software"

    @property
    def tee_type(self) -> TEEType:
        return TEEType.SOFTWARE

    @property
    def available(self) -> bool:
        return self._get_wasm_runtime().available

    def load(self, wasm_bytes: bytes) -> Any:
        return self._get_wasm_runtime().load(wasm_bytes)

    def execute(self, module: Any, input_data: bytes, resource_limits: ResourceLimits) -> ExecutionResult:
        return self._get_wasm_runtime().execute(module, input_data, resource_limits)

    def get_attestation_bytes(self) -> bytes:
        """A DEV-ONLY-marked blob (16-byte ``DEV-ONLY-SW-TEE:`` prefix + a
        48-byte sha384 tag = 64 bytes), matching the executor's software-stub
        format. Verifiers (DevOnlyBackend / is_dev_only) treat it as having NO
        confidentiality guarantee — it brands itself non-production so a
        software TEE can never be mistaken for a hardware one. Lazy imports keep
        this low-level runtime module decoupled from the heavier executor."""
        import hashlib
        from prsm.compute.inference.executor import (
            SOFTWARE_TEE_ATTESTATION_PREFIX,
        )
        return SOFTWARE_TEE_ATTESTATION_PREFIX + hashlib.sha384(b"sw-tee").digest()


class TEEHardwareUnavailableError(RuntimeError):
    """Sprint 1243 — raised when a hardware TEE runtime is asked to generate a
    quote on a host without the platform device. A hardware tier must produce a
    REAL quote or fail loudly — it MUST NOT fall back to a dev-only blob (which
    would let a software host masquerade as hardware-attested)."""


# Sprint 1243 — AMD SEV-SNP guest ioctl structure (Linux uapi
# include/uapi/linux/sev-guest.h). Captured as constants so the quote-gen path
# is concrete + reviewable; the actual ioctl call is hardware-validation-pending.
#   struct snp_report_req  { u8 user_data[64]; u32 vmpl; u8 rsvd[28]; }   // 96 B
#   struct snp_report_resp { u8 data[4000]; }   // resp_offset 32 → ATTESTATION_REPORT
#   struct snp_guest_request_ioctl { u8 msg_version; u64 req_data; u64 resp_data; u64 fw_err; }
#   SNP_GET_REPORT = _IOWR('S', 0x0, struct snp_guest_request_ioctl)
_SNP_REPORT_REQ_USER_DATA_LEN = 64
_SNP_REPORT_RESP_DATA_OFFSET = 32     # report body starts after the 32-byte resp header
_SNP_GET_REPORT_IOCTL = 0xC0205300    # _IOWR('S', 0x0, 32-byte ioctl struct)


class _HardwareTEERuntime(TEERuntime):
    """Sprint 1243 — shared base for REAL hardware TEE runtimes (SGX / SEV-SNP).

    Execution delegates to the same Ring-1 WASM sandbox as SoftwareTEERuntime
    (enclave-isolated execution is a separate future hardware concern); the
    TEE-specific value here is ``get_attestation_bytes`` — a real platform quote
    BOUND to the node_id (REPORT_DATA[:32] == sha256(node_id), the sp1083
    binding the verifiers check).

    ``available`` is DEVICE-GATED: with no platform device the runtime reports
    unavailable so the selector never picks it (it falls back to software).
    Quote generation is HARDWARE-VALIDATION-PENDING (sprint E — no TEE box in
    CI): the platform sequence is documented + the request structure is captured,
    but the actual syscall/SDK/KDS flow is unverified until run on real hardware,
    so it raises a precise error rather than emit anything dev-only or unvalidated.
    """

    _DEVICE_PATHS: tuple = ()
    _TEE_TYPE = TEEType.NONE
    _NAME = "hardware"

    def __init__(self, node_id: str = "") -> None:
        self._node_id = node_id or ""
        self._wasm_runtime = None

    def _get_wasm_runtime(self):
        if self._wasm_runtime is None:
            from prsm.compute.wasm.runtime import WasmtimeRuntime
            self._wasm_runtime = WasmtimeRuntime()
        return self._wasm_runtime

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def tee_type(self) -> TEEType:
        return self._TEE_TYPE

    @property
    def available(self) -> bool:
        import os
        return any(os.path.exists(p) for p in self._DEVICE_PATHS)

    def load(self, wasm_bytes: bytes) -> Any:
        return self._get_wasm_runtime().load(wasm_bytes)

    def execute(self, module: Any, input_data: bytes, resource_limits: ResourceLimits) -> ExecutionResult:
        return self._get_wasm_runtime().execute(module, input_data, resource_limits)

    def report_data(self) -> bytes:
        """The 64-byte REPORT_DATA to embed in the quote: sha256(node_id) in the
        first 32 bytes (the sp1083 node binding the verifiers check), zero-padded.
        Empty when no node_id is bound."""
        import hashlib
        commitment = hashlib.sha256((self._node_id or "").encode()).digest()  # 32 B
        return commitment + b"\x00" * 32  # 64-byte REPORT_DATA

    def get_attestation_bytes(self) -> bytes:
        if not self.available:
            raise TEEHardwareUnavailableError(
                f"{self._NAME} quote generation requires a platform device "
                f"({' or '.join(self._DEVICE_PATHS)}); none present on this host. "
                f"A hardware TEE tier must NOT fall back to a dev-only blob."
            )
        return self._generate_quote(self.report_data())

    def _generate_quote(self, report_data: bytes) -> bytes:  # pragma: no cover - hardware
        raise NotImplementedError


class SevSnpTEERuntime(_HardwareTEERuntime):
    """AMD SEV-SNP. ``get_attestation_bytes`` returns the ``PRSMSNP1`` envelope
    the ``AMDSEVSNPBackend`` verifies. HARDWARE-VALIDATION-PENDING (sprint E)."""

    _DEVICE_PATHS = ("/dev/sev-guest", "/dev/sev")
    _TEE_TYPE = TEEType.SEV
    _NAME = "sev-snp"

    def _generate_quote(self, report_data: bytes) -> bytes:  # pragma: no cover - hardware
        # Platform sequence (run on a real AMD SEV-SNP confidential VM):
        #  1. SNP_GET_REPORT ioctl(/dev/sev-guest) with snp_report_req.user_data =
        #     report_data[:64] → snp_report_resp; the ATTESTATION_REPORT begins at
        #     _SNP_REPORT_RESP_DATA_OFFSET (it carries report_data, measurement,
        #     chip_id, reported TCB, and the VCEK signature at offset 0x2A0).
        #  2. Read chip_id + reported TCB from the report.
        #  3. Fetch VCEK (+ ASK/ARK) from the AMD KDS for that chip_id/TCB.
        #  4. Assemble: b"PRSMSNP1" | uint32_le(len(report)) | report | (VCEK||ASK PEM).
        # The verifier (prsm.compute.inference.amd_sev_snp.AMDSEVSNPBackend) is
        # complete + tested; only this generation half awaits hardware validation.
        raise NotImplementedError(
            "SEV-SNP quote generation (SNP_GET_REPORT ioctl + AMD KDS VCEK fetch "
            "+ PRSMSNP1 envelope) is hardware-validation-pending — must be run + "
            "verified on a real AMD SEV-SNP confidential VM (sprint E). The "
            "verifier side (AMDSEVSNPBackend) is already complete + tested."
        )


class SgxTEERuntime(_HardwareTEERuntime):
    """Intel SGX (DCAP). ``get_attestation_bytes`` returns the SGX v3 quote the
    ``IntelDCAPBackend`` verifies. HARDWARE+SDK-VALIDATION-PENDING (sprint E)."""

    _DEVICE_PATHS = ("/dev/sgx_enclave", "/dev/sgx/enclave")
    _TEE_TYPE = TEEType.SGX
    _NAME = "sgx"

    def _generate_quote(self, report_data: bytes) -> bytes:  # pragma: no cover - hardware
        # Platform sequence (run on an Intel SGX DCAP host with the SDK + aesmd):
        #  1. Create the enclave; produce an EREPORT with the 64-byte report_data
        #     (REPORT_DATA[:32] = sha256(node_id) — the node binding).
        #  2. The Quoting Enclave signs it via the Attestation Key; sgx_qe_get_quote
        #     wraps it as a DCAP SGX v3 quote (ISV report + AK + QE report + PCK chain).
        # The verifier (prsm.compute.inference.intel_dcap.IntelDCAPBackend) is
        # complete + tested; only this generation half awaits hardware + the SDK.
        raise NotImplementedError(
            "SGX DCAP quote generation (enclave EREPORT + Quoting Enclave + "
            "sgx_qe_get_quote, report_data-bound) requires the Intel SGX DCAP SDK "
            "+ hardware — hardware-validation-pending (sprint E). The verifier "
            "side (IntelDCAPBackend) is already complete + tested."
        )
