"""Sprint 293 — hardware attestation backend interface +
Intel ASP / dev-only backends.

Sprint 292 surfaced the truth that every local-executor
receipt today carries a `DEV-ONLY-SW-TEE:` stub attestation.
This module builds the BACKEND INTERFACE that real hardware
attestation backends (Intel SGX/TDX via ASP, AMD SEV-SNP via
KDS, Apple SEP) plug behind.

Scope this sprint:

  AttestationBackend  Protocol
    .handles_vendor: str   — vendor identifier for routing
    .verify(blob)  → AttestationVerificationResult

  AttestationVerificationResult  dataclass
    vendor / vendor_verified / vendor_data / signature_chain_ok
    / error / structural_parse_ok

  AttestationBackendRegistry  vendor-detection dispatcher
    Default backends: IntelASPBackend + DevOnlyBackend
    Operators .register() their own to handle proprietary TEEs.

  IntelASPBackend
    Parses SGX (v3) + TDX (v4) quote headers + measurement
    fields (MRENCLAVE / MRSIGNER for SGX, MRTD / RTMR0 for
    TDX). v1 returns structurally-parsed result with
    vendor_verified=False — the real Intel DCAP cryptographic
    verification (Intel ASP signing-key chain) wires behind
    this same interface in a future sprint (sprint 297 E2E).

  DevOnlyBackend
    First-class handler for the sprint-292 software-fallback
    prefix; returns vendor="software-fallback" so receipts
    carrying this blob get a CLEAR vendor label (not just
    "hardware_attested=False" without an explanation).

Real Intel SGX/TDX quote format references:
  Intel SGX DCAP "Quote 3" structure (versioned). v3 magic
  bytes occupy the first two bytes as a little-endian uint16.
  MRENCLAVE at offset 48, MRSIGNER at offset 176 in the
  enclave report body that follows the quote header. TDX v4
  uses MRTD + RTMR0 at the equivalent offsets in the TD
  report body. v1 of this module parses those offsets
  structurally; production verification (signature chain to
  Intel's signing key + TCB level recency check + revocation
  list lookup) wires behind the same interface.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from prsm.compute.inference.executor import (
    SOFTWARE_TEE_ATTESTATION_PREFIX,
)

logger = logging.getLogger(__name__)


@dataclass
class AttestationVerificationResult:
    """Result of running an attestation through a backend.

    Distinct fields for distinct concerns:
      vendor              — which hardware vendor this quote
                            claims to be from (intel-sgx,
                            intel-tdx, amd-sev-snp, apple-sep,
                            software-fallback, unknown)
      vendor_verified     — backend cryptographically verified
                            the signature chain to the vendor's
                            attestation key. False in v1 (no
                            DCAP integration); True post-sprint
                            297 with real backend.
      vendor_data         — backend-specific parsed fields
                            (MRENCLAVE_hex, MRSIGNER_hex, TCB
                            level, etc.) for callers to pin
                            against expected values.
      signature_chain_ok  — finer-grained signature-chain
                            check; False in v1 stubs.
      error               — human-readable failure reason
                            (None on success).
      structural_parse_ok — backend recognized + parsed the
                            quote structure (independent of
                            cryptographic verification).
    """

    vendor: str
    vendor_verified: bool = False
    vendor_data: Dict[str, Any] = field(default_factory=dict)
    signature_chain_ok: bool = False
    error: Optional[str] = None
    structural_parse_ok: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor,
            "vendor_verified": self.vendor_verified,
            "vendor_data": dict(self.vendor_data),
            "signature_chain_ok": self.signature_chain_ok,
            "error": self.error,
            "structural_parse_ok": self.structural_parse_ok,
        }


class AttestationBackend(Protocol):
    """Backend interface for hardware-attestation verification.

    Production backends wrap vendor SDKs (Intel DCAP, AMD VLEK
    library, Apple SEP framework). Test backends operate on
    structural byte parsing only — the registry routes by
    vendor-detection from quote bytes so callers don't need
    to know which backend will run.
    """

    handles_vendor: str

    def verify(
        self, blob: Optional[bytes],
    ) -> AttestationVerificationResult: ...


def detect_vendor(blob: Optional[bytes]) -> str:
    """Identify which attestation vendor a quote blob is
    from, by inspecting the magic bytes. Used by the
    registry to dispatch + by callers that want vendor
    info without running verification."""
    if blob is None or not isinstance(blob, bytes):
        return "unknown"
    if len(blob) < 2:
        return "unknown"
    if blob.startswith(SOFTWARE_TEE_ATTESTATION_PREFIX):
        return "software-fallback"
    # Read version (little-endian uint16)
    version = struct.unpack("<H", blob[:2])[0]
    if version == 3:
        return "intel-sgx"
    if version == 4:
        return "intel-tdx"
    # Sprint 294 — AMD SEV-SNP attestation reports use a
    # uint32 version field (bytes 0-3); current spec is
    # version=2. Disambiguate from Intel uint16 version=2
    # (which doesn't exist in SGX/TDX) by requiring the
    # upper 16 bits of the uint32 to be zero.
    if len(blob) >= 4:
        version_u32 = struct.unpack("<I", blob[:4])[0]
        if version_u32 == 2:
            return "amd-sev-snp"
    # Future: vendor-specific magic bytes for Apple SEP, etc.
    return "unknown"


class IntelASPBackend:
    """Intel SGX/TDX attestation backend.

    v1 parses SGX (v3) + TDX (v4) quote headers + measurement
    fields. Cryptographic verification (signature chain to
    Intel's ASP signing key + TCB recency + revocation list)
    is deferred to the production wiring in sprint 297 —
    structural parsing is sufficient for callers that want to
    PIN against expected MRENCLAVE / MRSIGNER values out-of-
    band, which is the dominant use case for development +
    pre-commission environments.
    """

    handles_vendor: str = "intel"

    # SGX v3 quote header is 48 bytes; report body follows
    # with MRENCLAVE at offset 0 (= absolute offset 48) and
    # MRSIGNER at offset 128 (= absolute offset 176).
    _SGX_MIN_LEN = 48 + 32 + 96 + 32  # header + mrenclave + filler + mrsigner
    _TDX_MIN_LEN = 48 + 48 + 80 + 48

    def verify(
        self, blob: Optional[bytes],
    ) -> AttestationVerificationResult:
        if blob is None or not isinstance(blob, bytes):
            return AttestationVerificationResult(
                vendor="unknown",
                error="attestation blob is not bytes",
            )
        if len(blob) < 2:
            return AttestationVerificationResult(
                vendor="unknown",
                error=(
                    "attestation blob too short to "
                    "identify Intel vendor"
                ),
            )
        version = struct.unpack("<H", blob[:2])[0]
        if version == 3:
            return self._parse_sgx(blob)
        if version == 4:
            return self._parse_tdx(blob)
        return AttestationVerificationResult(
            vendor="unknown",
            error=(
                f"unrecognized version={version} for "
                f"Intel ASP backend (expected 3 for SGX, "
                f"4 for TDX)"
            ),
        )

    def _parse_sgx(
        self, blob: bytes,
    ) -> AttestationVerificationResult:
        if len(blob) < self._SGX_MIN_LEN:
            return AttestationVerificationResult(
                vendor="intel-sgx",
                error=(
                    f"SGX quote truncated: "
                    f"got {len(blob)} bytes, need "
                    f">= {self._SGX_MIN_LEN}"
                ),
            )
        mrenclave = blob[48:80]
        mrsigner = blob[176:208]
        return AttestationVerificationResult(
            vendor="intel-sgx",
            vendor_verified=False,  # DCAP not wired yet
            vendor_data={
                "version": 3,
                "mrenclave_hex": mrenclave.hex(),
                "mrsigner_hex": mrsigner.hex(),
                "structural_only": True,
            },
            signature_chain_ok=False,
            structural_parse_ok=True,
        )

    def _parse_tdx(
        self, blob: bytes,
    ) -> AttestationVerificationResult:
        if len(blob) < self._TDX_MIN_LEN:
            return AttestationVerificationResult(
                vendor="intel-tdx",
                error=(
                    f"TDX quote truncated: "
                    f"got {len(blob)} bytes, need "
                    f">= {self._TDX_MIN_LEN}"
                ),
            )
        mrtd = blob[48:96]
        rtmr0 = blob[176:224]
        return AttestationVerificationResult(
            vendor="intel-tdx",
            vendor_verified=False,
            vendor_data={
                "version": 4,
                "mrtd_hex": mrtd.hex(),
                "rtmr0_hex": rtmr0.hex(),
                "structural_only": True,
            },
            signature_chain_ok=False,
            structural_parse_ok=True,
        )


class AMDKDSBackend:
    """AMD SEV-SNP attestation backend.

    v1 parses SEV-SNP attestation reports structurally
    (version, guest_svn, MEASUREMENT, REPORT_DATA, chip_id).
    Cryptographic verification (signing-chain to AMD KDS
    endorsement key + VEK/VCEK lookup + TCB version recency)
    is deferred to the sprint-297 production wiring.

    SEV-SNP attestation report layout (subset per AMD's "SEV
    Secure Nested Paging Firmware ABI Specification"):
      bytes 0-3:    version (uint32 LE) = 2
      bytes 4-7:    guest_svn
      bytes 8-15:   policy
      bytes 16-31:  family_id
      bytes 32-47:  image_id
      bytes 48-51:  vmpl
      bytes 144-191: MEASUREMENT (48 bytes; sha384 of guest)
      bytes 320-383: REPORT_DATA (64 bytes; user nonce)
      bytes 416-479: chip_id (64 bytes)

    Real reports are ~1184 bytes total including the 512-byte
    ECDSA-P384 signature; v1 structural parsing only needs
    the report body up through chip_id (offset 480).
    """

    handles_vendor: str = "amd"

    _VERSION_OFFSET = 0
    _GUEST_SVN_OFFSET = 4
    _MEASUREMENT_OFFSET = 144
    _MEASUREMENT_LEN = 48
    _REPORT_DATA_OFFSET = 320
    _REPORT_DATA_LEN = 64
    _CHIP_ID_OFFSET = 416
    _CHIP_ID_LEN = 64
    _MIN_LEN = _CHIP_ID_OFFSET + _CHIP_ID_LEN

    def verify(
        self, blob: Optional[bytes],
    ) -> AttestationVerificationResult:
        if blob is None or not isinstance(blob, bytes):
            return AttestationVerificationResult(
                vendor="unknown",
                error="attestation blob is not bytes",
            )
        if len(blob) < 4:
            return AttestationVerificationResult(
                vendor="unknown",
                error=(
                    "attestation blob too short to "
                    "identify AMD vendor"
                ),
            )
        version = struct.unpack(
            "<I", blob[:4],
        )[0]
        if version != 2:
            return AttestationVerificationResult(
                vendor="unknown",
                error=(
                    f"unrecognized SEV-SNP version="
                    f"{version} (expected 2)"
                ),
            )
        if len(blob) < self._MIN_LEN:
            return AttestationVerificationResult(
                vendor="amd-sev-snp",
                error=(
                    f"SEV-SNP report truncated: "
                    f"got {len(blob)} bytes, need "
                    f">= {self._MIN_LEN}"
                ),
            )
        guest_svn = struct.unpack(
            "<I", blob[
                self._GUEST_SVN_OFFSET:
                self._GUEST_SVN_OFFSET + 4
            ],
        )[0]
        measurement = blob[
            self._MEASUREMENT_OFFSET:
            self._MEASUREMENT_OFFSET + self._MEASUREMENT_LEN
        ]
        report_data = blob[
            self._REPORT_DATA_OFFSET:
            self._REPORT_DATA_OFFSET + self._REPORT_DATA_LEN
        ]
        chip_id = blob[
            self._CHIP_ID_OFFSET:
            self._CHIP_ID_OFFSET + self._CHIP_ID_LEN
        ]
        return AttestationVerificationResult(
            vendor="amd-sev-snp",
            vendor_verified=False,  # KDS not wired yet
            vendor_data={
                "version": version,
                "guest_svn": guest_svn,
                "measurement_hex": measurement.hex(),
                "report_data_hex": report_data.hex(),
                "chip_id_hex": chip_id.hex(),
                "structural_only": True,
            },
            signature_chain_ok=False,
            structural_parse_ok=True,
        )


class DevOnlyBackend:
    """First-class handler for the sprint-292 software-
    fallback attestation prefix. Returns
    vendor="software-fallback" so receipts carrying this
    blob get a clear vendor label rather than the ambiguous
    "unknown."""

    handles_vendor: str = "software-fallback"

    def verify(
        self, blob: Optional[bytes],
    ) -> AttestationVerificationResult:
        if blob is None or not isinstance(blob, bytes):
            return AttestationVerificationResult(
                vendor="unknown",
                error="attestation blob is not bytes",
            )
        if blob.startswith(SOFTWARE_TEE_ATTESTATION_PREFIX):
            return AttestationVerificationResult(
                vendor="software-fallback",
                vendor_verified=False,
                vendor_data={
                    "note": (
                        "DEV-ONLY software-fallback "
                        "attestation; not a production "
                        "confidentiality proof. Hardware TEE "
                        "backends (Intel ASP, AMD KDS, "
                        "Apple SEP) ship in future sprints."
                    ),
                    "prefix_match": True,
                },
                signature_chain_ok=False,
                structural_parse_ok=True,
            )
        return AttestationVerificationResult(
            vendor="unknown",
            error=(
                "blob does not carry the DEV-ONLY "
                "software-fallback prefix"
            ),
        )


class AttestationBackendRegistry:
    """Vendor-detection dispatcher for attestation backends.

    Default backends: IntelASPBackend (handles SGX + TDX) +
    DevOnlyBackend (handles software-fallback). Operators
    .register() additional backends for proprietary TEEs;
    custom backends are tried before defaults so they can
    override default vendor handling.
    """

    def __init__(
        self,
        backends: Optional[List[AttestationBackend]] = None,
    ) -> None:
        if backends is None:
            self.backends: List[AttestationBackend] = [
                IntelASPBackend(),
                AMDKDSBackend(),
                DevOnlyBackend(),
            ]
        else:
            self.backends = list(backends)

    def register(
        self, backend: AttestationBackend,
    ) -> None:
        """Add a backend at the FRONT of the chain (custom
        backends override defaults). Vendor uniqueness not
        enforced — last-registered wins for the same
        handles_vendor key."""
        self.backends.insert(0, backend)

    def verify(
        self, blob: Optional[bytes],
    ) -> AttestationVerificationResult:
        # Try detect_vendor for the well-known prefixes
        # first (SGX/TDX magic bytes, software-fallback
        # prefix). If detection produces a known vendor,
        # route to a matching backend.
        vendor = detect_vendor(blob)
        vendor_prefix = (
            vendor.split("-")[0] if vendor != "unknown" else ""
        )
        if vendor != "unknown":
            for backend in self.backends:
                if backend.handles_vendor in (
                    vendor, vendor_prefix,
                ):
                    try:
                        return backend.verify(blob)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "AttestationBackend %s "
                            "raised: %s",
                            backend.__class__.__name__,
                            exc,
                        )
                        return (
                            AttestationVerificationResult(
                                vendor=vendor,
                                error=(
                                    f"backend raised: {exc}"
                                ),
                            )
                        )

        # Unknown vendor by built-in detection → give every
        # registered backend a chance to claim the blob.
        # Custom backends (registered for proprietary TEEs)
        # are at the FRONT of the list (per .register()
        # semantics), so this tries customs first. First
        # backend that returns a structurally-parsed result
        # wins.
        for backend in self.backends:
            try:
                result = backend.verify(blob)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AttestationBackend %s raised "
                    "during fallback: %s",
                    backend.__class__.__name__, exc,
                )
                continue
            if result.structural_parse_ok:
                return result

        # Truly unknown — no backend wanted it.
        return AttestationVerificationResult(
            vendor="unknown",
            error=(
                "no backend recognized the attestation "
                "(unknown magic bytes / version + no "
                "custom backend claimed it)"
            ),
        )


# Module-level default registry + convenience helper.
_DEFAULT_REGISTRY = AttestationBackendRegistry()


def verify_attestation(
    blob: Optional[bytes],
) -> AttestationVerificationResult:
    """Convenience helper backed by the default registry.
    Callers that don't need custom backend registration can
    use this directly."""
    return _DEFAULT_REGISTRY.verify(blob)


def register_intel_dcap(registry, env=None) -> bool:
    """Sprint 1046 — register a REAL Intel SGX DCAP verifier on ``registry`` when
    an Intel SGX Root CA is configured (PRSM_INTEL_SGX_ROOT_CA_PEM / _FILE), so SGX
    quotes get cryptographic ``vendor_verified``. Front-of-chain registration means
    it overrides the structural IntelASPBackend for SGX (and only SGX —
    handles_vendor='intel-sgx' — so TDX still falls through). Idempotent: returns
    False if nothing is configured OR a DCAP backend is already present. Lazy import
    avoids the circular dependency (intel_dcap imports from this module)."""
    from prsm.compute.inference.intel_dcap import (
        IntelDCAPBackend, build_intel_dcap_backend_or_none,
    )
    if any(isinstance(b, IntelDCAPBackend) for b in registry.backends):
        return False
    backend = build_intel_dcap_backend_or_none(env)
    if backend is None:
        return False
    registry.register(backend)
    return True


def configure_default_registry_from_env(env=None) -> bool:
    """Activate real Intel SGX DCAP verification on the DEFAULT registry (the one
    backing ``verify_attestation`` + /compute/receipt/verify) if an Intel SGX Root
    CA is configured. Call once at node startup; fail-open (no anchor → unchanged
    structural behavior). Returns True iff DCAP was activated."""
    return register_intel_dcap(_DEFAULT_REGISTRY, env)
