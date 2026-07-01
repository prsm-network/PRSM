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
# Sprint 1296 — ATTESTATION_REPORT (v2) layout + AMD KDS, for SevSnpTEERuntime quote-gen.
_SNP_REPORT_LEN = 0x4A0               # 1184 — fixed-size report (signed region ends at 0x2A0)
_SNP_REPORTED_TCB_OFF = 0x180         # 8-byte REPORTED_TCB: [bl, tee, rsvd*4, snp, ucode]
_SNP_CHIPID_OFF, _SNP_CHIPID_LEN = 0x1A0, 64
# AMD Key Distribution Service — VCEK leaf (DER) + ASK/ARK cert_chain (PEM) for the
# report's signing key. {product} = EPYC gen (Milan/Genoa/Turin); the 4 SPLs come from
# the report's REPORTED_TCB.
_AMD_KDS_BASE = "https://kdsintf.amd.com/vcek/v1"


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
    """AMD SEV-SNP. ``get_attestation_bytes`` returns the ``PRSMSNP1`` envelope the
    ``AMDSEVSNPBackend`` verifies.

    Sprint 1296 — quote generation is now IMPLEMENTED (SNP_GET_REPORT ioctl + AMD KDS
    VCEK/ASK fetch + envelope). The pure pieces (report parse, KDS URL build, envelope
    assembly) are unit-tested round-tripping through the real verifier; the two impure
    primitives — the ``/dev/sev-guest`` ioctl (``_snp_get_report``) and the KDS HTTP
    fetch (``_http_get``) — are HARDWARE/NETWORK-validation-pending until run on a real
    AMD SEV-SNP confidential VM (sprint E)."""

    _DEVICE_PATHS = ("/dev/sev-guest", "/dev/sev")
    _TEE_TYPE = TEEType.SEV
    _NAME = "sev-snp"

    def _generate_quote(self, report_data: bytes) -> bytes:
        # 1. ATTESTATION_REPORT via the SNP guest ioctl, REPORT_DATA = node binding.
        report = self._snp_get_report(report_data)
        # 2. The report's REPORTED_TCB + chip_id select the VCEK at the AMD KDS.
        tcb = self._parse_reported_tcb(report)
        chip_id = report[_SNP_CHIPID_OFF:_SNP_CHIPID_OFF + _SNP_CHIPID_LEN]
        product = self._kds_product()
        vcek_pem = self._fetch_vcek_pem(product, chip_id, tcb)
        ask_pem = self._fetch_ask_pem(product)
        # 3. Bundle exactly as AMDSEVSNPBackend.verify expects.
        return self._assemble_envelope(report, vcek_pem, ask_pem)

    def _snp_get_report(self, report_data: bytes) -> bytes:  # pragma: no cover - hardware
        """SNP_GET_REPORT ioctl(/dev/sev-guest) → the 1184-byte ATTESTATION_REPORT, with
        report_data[:64] embedded as REPORT_DATA[:64] (the sp1083 node binding)."""
        import ctypes
        import fcntl
        import os

        class _Req(ctypes.Structure):       # struct snp_report_req (96 B)
            _fields_ = [("user_data", ctypes.c_uint8 * 64),
                        ("vmpl", ctypes.c_uint32),
                        ("rsvd", ctypes.c_uint8 * 28)]

        class _Resp(ctypes.Structure):      # struct snp_report_resp
            _fields_ = [("data", ctypes.c_uint8 * 4000)]

        class _Ioctl(ctypes.Structure):     # snp_guest_request_ioctl — natural align = 32 B (_IOWR size 0x20)
            _fields_ = [("msg_version", ctypes.c_uint8),
                        ("req_data", ctypes.c_uint64),
                        ("resp_data", ctypes.c_uint64),
                        ("fw_err", ctypes.c_uint64)]

        req = _Req()
        ctypes.memmove(req.user_data, bytes(report_data[:64]).ljust(64, b"\x00"), 64)
        resp = _Resp()
        call = _Ioctl()
        call.msg_version = 1
        call.req_data = ctypes.addressof(req)
        call.resp_data = ctypes.addressof(resp)
        dev = next((p for p in self._DEVICE_PATHS if os.path.exists(p)), self._DEVICE_PATHS[0])
        fd = os.open(dev, os.O_RDWR | os.O_CLOEXEC)
        try:
            fcntl.ioctl(fd, _SNP_GET_REPORT_IOCTL, call)
        finally:
            os.close(fd)
        if call.fw_err != 0:
            raise TEEHardwareUnavailableError(
                f"SNP_GET_REPORT returned firmware error 0x{call.fw_err:x}")
        body = bytes(resp.data)[_SNP_REPORT_RESP_DATA_OFFSET:
                                _SNP_REPORT_RESP_DATA_OFFSET + _SNP_REPORT_LEN]
        if len(body) != _SNP_REPORT_LEN:
            raise TEEHardwareUnavailableError("short SNP report from ioctl")
        return body

    @staticmethod
    def _parse_reported_tcb(report: bytes) -> dict:
        """REPORTED_TCB (8 B @0x180): byte0=bootloader, byte1=tee, [2..5] reserved,
        byte6=snp, byte7=microcode — the SPLs the KDS VCEK URL needs."""
        tcb = report[_SNP_REPORTED_TCB_OFF:_SNP_REPORTED_TCB_OFF + 8]
        return {"blSPL": tcb[0], "teeSPL": tcb[1], "snpSPL": tcb[6], "ucodeSPL": tcb[7]}

    @staticmethod
    def _kds_product() -> str:
        """EPYC generation for the KDS path. Overridable via PRSM_SEV_SNP_PRODUCT;
        defaults to Milan (Azure DCas/ECas_v5 = 3rd-gen EPYC). Genoa/Turin for v6+."""
        import os
        return (os.environ.get("PRSM_SEV_SNP_PRODUCT", "").strip() or "Milan")

    @classmethod
    def _vcek_url(cls, product: str, chip_id: bytes, tcb: dict) -> str:
        return (f"{_AMD_KDS_BASE}/{product}/{chip_id.hex()}"
                f"?blSPL={tcb['blSPL']}&teeSPL={tcb['teeSPL']}"
                f"&snpSPL={tcb['snpSPL']}&ucodeSPL={tcb['ucodeSPL']}")

    @classmethod
    def _chain_url(cls, product: str) -> str:
        return f"{_AMD_KDS_BASE}/{product}/cert_chain"

    @staticmethod
    def _http_get(url: str) -> bytes:  # pragma: no cover - network
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - fixed AMD KDS host
            return r.read()

    @classmethod
    def _fetch_vcek_pem(cls, product: str, chip_id: bytes, tcb: dict) -> bytes:
        """KDS serves the VCEK as DER; convert to PEM for the envelope."""
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        der = cls._http_get(cls._vcek_url(product, chip_id, tcb))
        return x509.load_der_x509_certificate(der).public_bytes(serialization.Encoding.PEM)

    @classmethod
    def _fetch_ask_pem(cls, product: str) -> bytes:
        """KDS cert_chain returns ASK then ARK (PEM); the envelope carries the ASK (the
        VCEK's issuer) — the ARK is the verifier's CONFIGURED trust root, not bundled."""
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        certs = x509.load_pem_x509_certificates(cls._http_get(cls._chain_url(product)))
        if not certs:
            raise TEEHardwareUnavailableError("KDS cert_chain returned no certificates")
        return certs[0].public_bytes(serialization.Encoding.PEM)  # ASK

    @staticmethod
    def _assemble_envelope(report: bytes, vcek_pem: bytes, ask_pem: bytes) -> bytes:
        """b"PRSMSNP1" | report_len(u32 LE) | report | (VCEK PEM || ASK PEM) — the exact
        shape AMDSEVSNPBackend.verify parses."""
        import struct
        return b"PRSMSNP1" + struct.pack("<I", len(report)) + report + vcek_pem + ask_pem


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


def build_tee_runtime(node_id: str = "", environ=None) -> "TEERuntime":
    """Sprint 1333 (TEE Tier-B producer) — select the TEE runtime the stage server attests with,
    per ``PRSM_TEE_RUNTIME_KIND``.

    The stage server (``LayerStageServer``) already calls ``tee_runtime.get_attestation_bytes()``
    and stamps each §7 receipt's ``tee_attestation`` + ``tee_type`` from it, which flows into the
    on-chain attestation commitment (roadmap F). Until now that runtime was hardcoded to
    ``SoftwareTEERuntime`` (a DEV-ONLY blob), so a node on real SEV-SNP hardware could never emit a
    real quote. This is the missing selection:

      - ``""`` / ``"software"`` (DEFAULT): ``SoftwareTEERuntime`` — the dev-only attestation
        (Tier A). Byte-for-byte the prior behavior.
      - ``"auto"``: the first hardware runtime whose device is present (SEV-SNP, then SGX), else
        software. Logs which was selected — a silent downgrade to software IS possible here, so
        operators who require hardware attestation should pin the explicit kind, not ``auto``.
      - ``"sev-snp"`` / ``"sgx"``: that hardware runtime, node-bound (REPORT_DATA[:32] ==
        sha256(node_id), sp1083). FAIL-CLOSED — if the device is absent it RAISES
        ``TEEHardwareUnavailableError`` rather than fall back to software: an operator who
        explicitly asked for hardware attestation must never be silently handed the dev-only blob
        (that would emit receipts CLAIMING hardware without it — a Tier-B integrity violation).
    """
    import os
    env = environ if environ is not None else os.environ
    kind = str(env.get("PRSM_TEE_RUNTIME_KIND", "") or "").strip().lower()

    if kind in ("", "software"):
        return SoftwareTEERuntime()

    if kind == "auto":
        for cls in (SevSnpTEERuntime, SgxTEERuntime):
            rt = cls(node_id=node_id)
            if rt.available:
                logger.info(
                    "build_tee_runtime: auto-selected hardware TEE runtime %s "
                    "(node-bound attestation).", rt.name)
                return rt
        logger.info(
            "build_tee_runtime: PRSM_TEE_RUNTIME_KIND=auto but no TEE device present; "
            "using SoftwareTEERuntime (dev-only attestation).")
        return SoftwareTEERuntime()

    _KINDS = {
        "sev-snp": SevSnpTEERuntime, "sevsnp": SevSnpTEERuntime, "sev": SevSnpTEERuntime,
        "sgx": SgxTEERuntime,
    }
    cls = _KINDS.get(kind)
    if cls is None:
        logger.warning(
            "build_tee_runtime: unknown PRSM_TEE_RUNTIME_KIND=%r; using SoftwareTEERuntime "
            "(dev-only). Valid: software | auto | sev-snp | sgx.", kind)
        return SoftwareTEERuntime()

    rt = cls(node_id=node_id)
    if not rt.available:
        raise TEEHardwareUnavailableError(
            f"PRSM_TEE_RUNTIME_KIND={kind!r} explicitly requested hardware TEE attestation, but "
            f"the {rt.name} device is not present on this host. Refusing to fall back to the "
            f"dev-only software runtime — that would emit §7 receipts CLAIMING hardware "
            f"attestation without it. Provision the TEE device, or set "
            f"PRSM_TEE_RUNTIME_KIND=auto (opportunistic) / software (dev-only).")
    logger.info(
        "build_tee_runtime: selected hardware TEE runtime %s (node-bound attestation) — "
        "receipts will carry real quotes (Tier B).", rt.name)
    return rt
