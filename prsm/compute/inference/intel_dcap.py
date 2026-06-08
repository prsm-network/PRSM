"""Sprint 1044 — real Intel SGX DCAP (ECDSA-P256) attestation verification (Tier 3).

Closes the §7 ``attestation_vendor_verified=false`` honest-scope deferral for Intel
SGX. ``IntelDCAPBackend`` cryptographically verifies a v3 ECDSA quote's full
signature chain (no TEE hardware, no Intel PCS network — pure ``cryptography``):

  1. ISV enclave report signature — by the embedded Attestation Key (AK).
  2. AK→QE binding — SHA256(ak_pub || qe_auth_data) == QE report_data[:32]
     (proves the AK was certified by the Quoting Enclave, not forged).
  3. QE report signature — by the PCK leaf certificate's key.
  4. PCK X.509 chain (PCK → intermediate(s) → root) verified up to a CONFIGURED
     trusted root. In production the operator pins Intel's SGX Root CA; the
     vendor_verified=True claim is exactly as strong as that configured anchor.

``vendor_verified`` is True only when ALL four pass. Out of scope (explicitly
deferred, documented): TCB recency, QE-identity pinning, and CRL revocation — those
require live Intel PCS collateral and are a production-hardening follow-on. This
backend establishes the cryptographic verification interface; the deferred checks
wire behind the same result type.

Quote layout (Intel SGX ECDSA Quote v3):
  header(48) | isv_report(384) | sig_data_len(uint32) | sig_data
  sig_data = isv_sig(64 raw r||s) | ak_pub(64 x||y) | qe_report(384) |
             qe_sig(64) | qe_auth_len(uint16) | qe_auth | cert_len(uint32) | cert_pem
"""
from __future__ import annotations

import hashlib
import logging
import struct
from typing import List, Optional

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from prsm.compute.inference.attestation_backends import AttestationVerificationResult

logger = logging.getLogger(__name__)

_HEADER_LEN = 48
_REPORT_LEN = 384
_SIG_RAW_LEN = 64
_AK_PUB_LEN = 64
# offsets within a 384-byte SGX report body
_MRENCLAVE_OFF = 64
_MRSIGNER_OFF = 128
_REPORT_DATA_OFF = 320


def _err(msg: str) -> AttestationVerificationResult:
    return AttestationVerificationResult(
        vendor="intel-sgx", vendor_verified=False, signature_chain_ok=False,
        structural_parse_ok=False, error=msg)


def _raw_to_der(raw: bytes) -> bytes:
    """SGX ECDSA sigs are raw r||s (64 bytes); cryptography verifies DER."""
    r = int.from_bytes(raw[:32], "big")
    s = int.from_bytes(raw[32:], "big")
    return encode_dss_signature(r, s)


def _verify_ecdsa(pub: ec.EllipticCurvePublicKey, raw_sig: bytes, msg: bytes) -> bool:
    try:
        pub.verify(_raw_to_der(raw_sig), msg, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False
    except Exception:  # noqa: BLE001 - malformed sig/key → not verified
        return False


class IntelDCAPBackend:
    """Cryptographic DCAP verifier for Intel SGX v3 ECDSA quotes.

    ``trusted_root_pem`` is the trust anchor the PCK chain must verify up to —
    REQUIRED (no default): in production pass Intel's SGX Root CA PEM; the strength
    of ``vendor_verified`` is exactly the strength of this anchor."""

    handles_vendor: str = "intel"

    def __init__(self, trusted_root_pem: bytes):
        if not trusted_root_pem:
            raise ValueError(
                "IntelDCAPBackend requires trusted_root_pem (Intel SGX Root CA in "
                "production) — vendor_verified is only as strong as this anchor")
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
            logger.debug("DCAP verify error: %s: %s", type(exc).__name__, exc)
            return _err(f"DCAP parse/verify error: {type(exc).__name__}: {exc}")

    def _verify(self, blob: bytes) -> AttestationVerificationResult:
        if len(blob) < _HEADER_LEN + _REPORT_LEN + 4:
            return _err(f"quote too short ({len(blob)} bytes) for header+report+siglen")
        version = struct.unpack("<H", blob[:2])[0]
        if version != 3:
            return _err(f"unsupported quote version={version} (DCAP backend handles SGX v3)")

        isv_report = blob[_HEADER_LEN:_HEADER_LEN + _REPORT_LEN]
        off = _HEADER_LEN + _REPORT_LEN
        (sig_len,) = struct.unpack("<I", blob[off:off + 4])
        off += 4
        sig_data = blob[off:off + sig_len]
        if len(sig_data) < _SIG_RAW_LEN + _AK_PUB_LEN + _REPORT_LEN + _SIG_RAW_LEN + 2:
            return _err("signature_data section truncated")

        p = 0
        isv_sig = sig_data[p:p + _SIG_RAW_LEN]; p += _SIG_RAW_LEN
        ak_pub_raw = sig_data[p:p + _AK_PUB_LEN]; p += _AK_PUB_LEN
        qe_report = sig_data[p:p + _REPORT_LEN]; p += _REPORT_LEN
        qe_sig = sig_data[p:p + _SIG_RAW_LEN]; p += _SIG_RAW_LEN
        (qe_auth_len,) = struct.unpack("<H", sig_data[p:p + 2]); p += 2
        qe_auth = sig_data[p:p + qe_auth_len]; p += qe_auth_len
        if len(sig_data) < p + 4:
            return _err("signature_data section missing cert-data length")
        (cert_len,) = struct.unpack("<I", sig_data[p:p + 4]); p += 4
        cert_pem = sig_data[p:p + cert_len]

        # parsed enough — structural_parse_ok from here on
        mrenclave = isv_report[_MRENCLAVE_OFF:_MRENCLAVE_OFF + 32]
        mrsigner = isv_report[_MRSIGNER_OFF:_MRSIGNER_OFF + 32]
        vendor_data = {
            "mrenclave_hex": mrenclave.hex(),
            "mrsigner_hex": mrsigner.hex(),
            "report_data_hex": isv_report[_REPORT_DATA_OFF:_REPORT_DATA_OFF + 64].hex(),
        }

        def fail(msg: str, chain_ok: bool = False) -> AttestationVerificationResult:
            return AttestationVerificationResult(
                vendor="intel-sgx", vendor_verified=False, signature_chain_ok=chain_ok,
                structural_parse_ok=True, vendor_data=vendor_data, error=msg)

        # 1. ISV report signed by the Attestation Key
        try:
            ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(), b"\x04" + ak_pub_raw)
        except Exception as exc:  # noqa: BLE001
            return fail(f"attestation key malformed: {exc}")
        if not _verify_ecdsa(ak_pub, isv_sig, isv_report):
            return fail("ISV report signature invalid (AK did not sign the report)")

        # 2. AK bound to the QE: SHA256(ak_pub || qe_auth) == QE report_data[:32]
        binding = hashlib.sha256(ak_pub_raw + qe_auth).digest()
        qe_report_data = qe_report[_REPORT_DATA_OFF:_REPORT_DATA_OFF + 64]
        if binding != qe_report_data[:32]:
            return fail("attestation-key binding invalid "
                        "(SHA256(AK||auth) != QE report_data)")

        # 3. + 4. PCK chain verifies to the trusted root, and its leaf signed the QE report
        chain = self._load_chain(cert_pem)
        if not chain:
            return fail("PCK certificate chain missing/unparseable")
        pck_leaf = chain[0]
        leaf_pub = pck_leaf.public_key()
        if not isinstance(leaf_pub, ec.EllipticCurvePublicKey):
            return fail("PCK leaf key is not EC (expected ECDSA-P256)")
        if not _verify_ecdsa(leaf_pub, qe_sig, qe_report):
            return fail("QE report signature invalid (PCK leaf did not sign QE report)")

        chain_ok, chain_err = self._verify_chain(chain)
        if not chain_ok:
            return fail(f"PCK chain does not verify to trusted root: {chain_err}",
                        chain_ok=False)

        return AttestationVerificationResult(
            vendor="intel-sgx", vendor_verified=True, signature_chain_ok=True,
            structural_parse_ok=True, vendor_data=vendor_data, error=None)

    @staticmethod
    def _load_chain(cert_pem: bytes) -> List[x509.Certificate]:
        certs: List[x509.Certificate] = []
        marker = b"-----BEGIN CERTIFICATE-----"
        parts = cert_pem.split(marker)
        for part in parts[1:]:
            try:
                certs.append(x509.load_pem_x509_certificate(marker + part))
            except Exception:  # noqa: BLE001 - skip unparseable block
                continue
        return certs

    def _verify_chain(self, chain: List[x509.Certificate]):
        """X.509 path validation of the PCK chain up to the configured trusted
        root. ``chain`` is [end-entity(PCK leaf), intermediate(s)...]; the trust
        anchor is appended. For each (child, issuer) pair this enforces — beyond
        the signature — the path-validation rules WITHOUT which a signature-only
        walk is forgeable (sp1044 review, CRITICAL):

          - the ISSUER is a CA: BasicConstraints present + ca=True. This is the
            load-bearing check: a real but non-CA PCK leaf can NOT be used to
            "issue" an attacker's forged cert (the reproduced critical bypass).
          - the ISSUER's KeyUsage (if present) permits keyCertSign.
          - the ISSUER's pathLenConstraint (if set) allows the depth below it.
          - name chaining: child.issuer == issuer.subject.
          - validity: every cert is within [not_valid_before, not_valid_after] now.

        Only the end-entity (chain[0], which signs the QE report) may be non-CA."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        full = list(chain) + [self._root]

        def _within_validity(cert) -> bool:
            try:
                nvb, nva = cert.not_valid_before_utc, cert.not_valid_after_utc
            except AttributeError:  # older cryptography: naive UTC datetimes
                nvb = cert.not_valid_before.replace(tzinfo=timezone.utc)
                nva = cert.not_valid_after.replace(tzinfo=timezone.utc)
            return nvb <= now <= nva

        def _is_ca(cert) -> bool:
            try:
                bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
            except x509.ExtensionNotFound:
                return False
            return bool(bc.ca)

        def _ca_path_len(cert):
            try:
                bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
            except x509.ExtensionNotFound:
                return None
            return bc.path_length

        def _allows_cert_sign(cert) -> bool:
            try:
                ku = cert.extensions.get_extension_for_class(x509.KeyUsage).value
            except x509.ExtensionNotFound:
                return True  # KeyUsage absent → unconstrained (RFC 5280)
            return bool(ku.key_cert_sign)

        # every cert must be within its validity window (incl. the leaf + root)
        for cert in full:
            if not _within_validity(cert):
                return False, f"{cert.subject.rfc4514_string()} outside its validity window"

        # walk child -> issuer; count CA hops below each issuer for pathLen
        for depth, (child, issuer) in enumerate(zip(full, full[1:])):
            if child.issuer != issuer.subject:
                return False, "issuer/subject name mismatch in chain"
            if not _is_ca(issuer):
                return False, (
                    f"{issuer.subject.rfc4514_string()} is not a CA "
                    f"(BasicConstraints ca!=True) — cannot issue certificates")
            if not _allows_cert_sign(issuer):
                return False, f"{issuer.subject.rfc4514_string()} KeyUsage forbids keyCertSign"
            plen = _ca_path_len(issuer)
            # number of CA certs strictly between this issuer and the end-entity
            cas_below = sum(1 for c in full[1:depth + 1] if _is_ca(c))
            if plen is not None and cas_below > plen:
                return False, (
                    f"{issuer.subject.rfc4514_string()} pathLenConstraint={plen} "
                    f"exceeded ({cas_below} CA(s) below)")
            try:
                issuer.public_key().verify(
                    child.signature, child.tbs_certificate_bytes,
                    ec.ECDSA(child.signature_hash_algorithm),
                )
            except InvalidSignature:
                return False, f"{child.subject.rfc4514_string()} not signed by its issuer"
            except Exception as exc:  # noqa: BLE001
                return False, f"chain verify error: {exc}"

        # the trust anchor must be a self-signed CA (it is the configured root)
        if not _is_ca(self._root):
            return False, "configured trusted root is not a CA"
        try:
            self._root.public_key().verify(
                self._root.signature, self._root.tbs_certificate_bytes,
                ec.ECDSA(self._root.signature_hash_algorithm),
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"trusted root is not self-consistent: {exc}"
        return True, None
