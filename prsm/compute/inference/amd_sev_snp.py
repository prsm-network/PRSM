"""Sprint 1049 — real AMD SEV-SNP attestation verification (Tier 3).

Closes the §7 ``attestation_vendor_verified=false`` deferral for AMD SEV-SNP, the
parallel to the Intel SGX DCAP engine (sp1044). ``AMDSEVSNPBackend`` cryptographically
verifies a SEV-SNP attestation report (no SEV hardware, no AMD KDS network — pure
``cryptography``):

  1. parse the ATTESTATION_REPORT (v2): measurement (0x90,48), report_data
     (0x50,64), chip_id (0x1A0,64), signature (0x2A0,512);
  2. verify the report's ECDSA-P384 / SHA-384 signature over report[:0x2A0] using
     the VCEK public key — AMD stores r,s LITTLE-ENDIAN in 72-byte slots (NOT the
     big-endian r||s Intel uses, the key encoding gotcha);
  3. verify the VCEK → ASK → ARK X.509 chain up to a CONFIGURED ARK root via the
     shared hardened path-validator (prsm.compute.inference.x509_path) — same
     CA-flag/validity/name-chaining/pathLen rules as Intel, so the critical
     non-CA-issuer forge-bypass is closed here too.

``vendor_verified`` is True only when the report signature AND the chain both pass.
``trusted_root_pem`` (AMD ARK) is REQUIRED — vendor_verified is only as strong as
that anchor. Out of scope (deferred): TCB/reported_tcb recency, policy/VMPL pinning,
and the VLEK variant.

Transport: a SEV-SNP report carries no embedded certs (unlike Intel's quote), so
PRSM wraps report + VCEK/ASK chain in an envelope:
  b"PRSMSNP1" | report_len(uint32 LE) | report | chain_pem(VCEK then ASK)
The ARK is the configured trust anchor (not in the envelope).
"""
from __future__ import annotations

import logging
import struct
from typing import List, Optional

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from prsm.compute.inference.attestation_backends import AttestationVerificationResult
from prsm.compute.inference.x509_path import verify_cert_chain

logger = logging.getLogger(__name__)

_MAGIC = b"PRSMSNP1"
_SIG_OFF = 0x2A0           # signature field offset; report[:0x2A0] is signed
_COMPONENT = 72           # AMD stores each of r,s in a 72-byte LE slot
_MEAS_OFF, _MEAS_LEN = 0x90, 48
_RDATA_OFF, _RDATA_LEN = 0x50, 64
_CHIPID_OFF, _CHIPID_LEN = 0x1A0, 64
_MIN_REPORT_LEN = _SIG_OFF + 2 * _COMPONENT   # need the signed region + r + s


def _err(msg: str, structural: bool = False) -> AttestationVerificationResult:
    return AttestationVerificationResult(
        vendor="amd-sev-snp", vendor_verified=False, signature_chain_ok=False,
        structural_parse_ok=structural, error=msg)


def _load_chain(cert_pem: bytes) -> List[x509.Certificate]:
    certs: List[x509.Certificate] = []
    marker = b"-----BEGIN CERTIFICATE-----"
    for part in cert_pem.split(marker)[1:]:
        try:
            certs.append(x509.load_pem_x509_certificate(marker + part))
        except Exception:  # noqa: BLE001 - skip unparseable block
            continue
    return certs


class AMDSEVSNPBackend:
    """Cryptographic verifier for AMD SEV-SNP attestation reports (envelope form)."""

    handles_vendor: str = "amd-sev-snp"

    def __init__(self, trusted_root_pem: bytes):
        if not trusted_root_pem:
            raise ValueError(
                "AMDSEVSNPBackend requires trusted_root_pem (AMD ARK in production) "
                "— vendor_verified is only as strong as this anchor")
        try:
            self._root = x509.load_pem_x509_certificate(trusted_root_pem)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"trusted_root_pem is not a valid certificate: {exc}")

    def verify(self, blob: Optional[bytes]) -> AttestationVerificationResult:
        if not isinstance(blob, (bytes, bytearray)):
            return _err("attestation blob is not bytes")
        blob = bytes(blob)
        try:
            return self._verify(blob)
        except Exception as exc:  # noqa: BLE001 - never crash the verify path
            logger.debug("SEV-SNP verify error: %s: %s", type(exc).__name__, exc)
            return _err(f"SEV-SNP parse/verify error: {type(exc).__name__}: {exc}")

    def _verify(self, blob: bytes) -> AttestationVerificationResult:
        if len(blob) < len(_MAGIC) + 4 or blob[:len(_MAGIC)] != _MAGIC:
            return _err("not a PRSM SEV-SNP envelope (missing PRSMSNP1 magic) — the "
                        "report must be bundled with its VCEK/ASK chain")
        off = len(_MAGIC)
        (report_len,) = struct.unpack("<I", blob[off:off + 4])
        off += 4
        if report_len < _MIN_REPORT_LEN or len(blob) < off + report_len:
            return _err("SEV-SNP envelope truncated (report length)")
        report = blob[off:off + report_len]
        chain_pem = blob[off + report_len:]

        # Version-gate (parity with Intel's v3 gate): the field offsets below are
        # the SEV-SNP v2 ABI. A v3+ report can shift previously-reserved fields, so
        # reject rather than silently mis-parse measurement/report_data/chip_id.
        (version,) = struct.unpack("<I", report[:4])
        if version != 2:
            return _err(f"unsupported SEV-SNP report version={version} (this backend "
                        f"parses v2; v3 layout is a follow-on)", structural=True)

        measurement = report[_MEAS_OFF:_MEAS_OFF + _MEAS_LEN]
        report_data = report[_RDATA_OFF:_RDATA_OFF + _RDATA_LEN]
        chip_id = report[_CHIPID_OFF:_CHIPID_OFF + _CHIPID_LEN]
        vendor_data = {
            "version": version,
            "measurement_hex": measurement.hex(),
            "report_data_hex": report_data.hex(),
            "chip_id_hex": chip_id.hex(),
        }

        def fail(msg: str, chain_ok: bool = False) -> AttestationVerificationResult:
            return AttestationVerificationResult(
                vendor="amd-sev-snp", vendor_verified=False, signature_chain_ok=chain_ok,
                structural_parse_ok=True, vendor_data=vendor_data, error=msg)

        chain = _load_chain(chain_pem)
        if not chain:
            return fail("VCEK/ASK certificate chain missing/unparseable")
        vcek = chain[0]
        vcek_pub = vcek.public_key()
        if not isinstance(vcek_pub, ec.EllipticCurvePublicKey):
            return fail("VCEK key is not EC (expected ECDSA-P384)")

        # report signature: ECDSA-P384/SHA-384 over report[:0x2A0]; r,s are LE 72B slots
        r = int.from_bytes(report[_SIG_OFF:_SIG_OFF + _COMPONENT], "little")
        s = int.from_bytes(report[_SIG_OFF + _COMPONENT:_SIG_OFF + 2 * _COMPONENT], "little")
        try:
            vcek_pub.verify(encode_dss_signature(r, s), report[:_SIG_OFF],
                            ec.ECDSA(hashes.SHA384()))
        except InvalidSignature:
            return fail("SEV-SNP report signature invalid (VCEK did not sign the report)")
        except Exception as exc:  # noqa: BLE001
            return fail(f"SEV-SNP report signature check error: {exc}")

        chain_ok, chain_err = verify_cert_chain(chain, self._root)
        if not chain_ok:
            return fail(f"VCEK chain does not verify to ARK root: {chain_err}")

        return AttestationVerificationResult(
            vendor="amd-sev-snp", vendor_verified=True, signature_chain_ok=True,
            structural_parse_ok=True, vendor_data=vendor_data, error=None)


def build_amd_sev_snp_backend_or_none(env: Optional[dict] = None):
    """sp1049 — construct an AMDSEVSNPBackend from operator config, or None.
    Reads the AMD ARK trust anchor from PRSM_AMD_SEV_SNP_ARK_PEM (inline) or
    PRSM_AMD_SEV_SNP_ARK_FILE (path). Fail-open: missing → None (structural
    fallback); unreadable/invalid → None + log."""
    import os
    environ = env if env is not None else os.environ
    pem = (environ.get("PRSM_AMD_SEV_SNP_ARK_PEM", "") or "").strip()
    root_bytes = pem.encode() if pem else None
    if not root_bytes:
        path = (environ.get("PRSM_AMD_SEV_SNP_ARK_FILE", "") or "").strip()
        if path:
            try:
                with open(path, "rb") as f:
                    root_bytes = f.read()
            except OSError as exc:
                logger.warning("AMD ARK file unreadable (%s) — SEV-SNP verification "
                               "OFF, structural fallback", exc)
                return None
    if not root_bytes:
        return None
    try:
        return AMDSEVSNPBackend(trusted_root_pem=root_bytes)
    except ValueError as exc:
        logger.warning("AMD ARK invalid (%s) — SEV-SNP verification OFF, structural "
                       "fallback", exc)
        return None
