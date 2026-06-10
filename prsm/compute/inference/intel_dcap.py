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

    # SGX v3 only — so Intel TDX (v4) quotes still fall through to the structural
    # IntelASPBackend (whose handles_vendor='intel') rather than being DCAP-rejected.
    handles_vendor: str = "intel-sgx"

    def __init__(self, trusted_root_pem: bytes, crls_pem: Optional[bytes] = None,
                 tcb_info_json: Optional[bytes] = None,
                 tcb_signer_cert_pem: Optional[bytes] = None,
                 tcb_policy=None):
        if not trusted_root_pem:
            raise ValueError(
                "IntelDCAPBackend requires trusted_root_pem (Intel SGX Root CA in "
                "production) — vendor_verified is only as strong as this anchor")
        try:
            self._root = x509.load_pem_x509_certificate(trusted_root_pem)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"trusted_root_pem is not a valid certificate: {exc}")
        # sp1060 — optional operator-supplied CRLs (Intel Root-CA CRL revoking
        # intermediates + PCK-CA CRL revoking PCK leaves). When present, a revoked
        # PCK chain cert fails verification. None → no revocation check (unchanged).
        self._crls = self._load_crls(crls_pem) if crls_pem else None
        # sp1061-1063 — optional TCB-recency enforcement. When BOTH the signed
        # TCB-Info and its signer cert are supplied, a validly-signed quote from a
        # DOWNLEVEL (out-of-date) TEE is rejected per the policy. Absent → no TCB
        # check (signature-chain only, unchanged).
        self._tcb_info_json = bytes(tcb_info_json) if tcb_info_json else None
        self._tcb_signer = None
        if tcb_signer_cert_pem:
            try:
                self._tcb_signer = x509.load_pem_x509_certificate(bytes(tcb_signer_cert_pem))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"tcb_signer_cert_pem is not a valid certificate: {exc}")
        if tcb_policy is None:
            from prsm.compute.inference.sgx_tcb import default_tcb_policy
            tcb_policy = default_tcb_policy()
        self._tcb_policy = tcb_policy

    @staticmethod
    def _load_crls(crls_pem: bytes) -> Optional[list]:
        crls = []
        marker = b"-----BEGIN X509 CRL-----"
        for part in crls_pem.split(marker)[1:]:
            try:
                crls.append(x509.load_pem_x509_crl(marker + part))
            except Exception:  # noqa: BLE001 - skip unparseable block
                continue
        return crls or None

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

        # sp1061-1063 — TCB-recency: the chain proves AUTHENTICITY; this proves the
        # platform isn't running a known-vulnerable (downlevel/revoked) TCB. Only
        # enforced when the operator configured a signed TCB-Info + its signer.
        if self._tcb_info_json is not None and self._tcb_signer is not None:
            tcb_status, tcb_err = self._evaluate_tcb(pck_leaf)
            if tcb_status is not None:
                vendor_data["tcb_status"] = tcb_status
            if tcb_err is not None:
                # the chain verified, so signature_chain_ok=True, but vendor_verified
                # is False — the TEE is authentic yet not recency-acceptable.
                return AttestationVerificationResult(
                    vendor="intel-sgx", vendor_verified=False, signature_chain_ok=True,
                    structural_parse_ok=True, vendor_data=vendor_data, error=tcb_err)

        return AttestationVerificationResult(
            vendor="intel-sgx", vendor_verified=True, signature_chain_ok=True,
            structural_parse_ok=True, vendor_data=vendor_data, error=None)

    def _evaluate_tcb(self, pck_leaf):
        """Return (tcb_status, error). error is None when the platform's TCB status is
        policy-acceptable; otherwise a message (and vendor_verified must be False).
        A parse/verify/FMSPC failure is fail-closed (error set, status maybe None)."""
        from prsm.compute.inference.sgx_tcb import (
            TcbInfoError, evaluate_tcb_status, parse_and_verify_tcb_info,
            parse_pck_sgx_extension,
        )
        try:
            pck_tcb = parse_pck_sgx_extension(pck_leaf)
            tcb_info = parse_and_verify_tcb_info(self._tcb_info_json, self._tcb_signer)
            status = evaluate_tcb_status(pck_tcb, tcb_info)
        except TcbInfoError as exc:
            return None, f"TCB-Info check failed (fail-closed): {exc}"
        except ValueError as exc:
            return None, f"TCB level unreadable (fail-closed): {exc}"
        if not self._tcb_policy.is_acceptable(status):
            return status, f"TCB status '{status}' is not acceptable (downlevel/revoked TEE)"
        return status, None

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
        """X.509 path validation of the PCK chain (end-entity PCK leaf →
        intermediate(s)) up to the configured trusted root, via the shared
        hardened validator (sp1049 — same path-validation Intel + AMD use, with the
        load-bearing CA-flag check from the sp1044 review)."""
        from prsm.compute.inference.x509_path import verify_cert_chain
        # require_crl=False: revocation is enforced for any cert whose issuer CRL is
        # supplied, but a chain whose issuer has no configured CRL still verifies —
        # so configuring one of Intel's two CRLs can't brick the other path. An
        # operator wanting strict revocation passes both CRLs.
        return verify_cert_chain(chain, self._root, crls=self._crls, require_crl=False)


def build_intel_dcap_backend_or_none(env: Optional[dict] = None):
    """sp1046 — construct an IntelDCAPBackend from operator config, or None.

    Reads the Intel SGX Root CA trust anchor from (in order):
      - ``PRSM_INTEL_SGX_ROOT_CA_PEM``  — inline PEM, or
      - ``PRSM_INTEL_SGX_ROOT_CA_FILE`` — path to a PEM file.
    Returns None when neither is set (→ structural-only fallback, unchanged
    behavior) OR when the configured anchor is unreadable/invalid (fail-open: a
    misconfigured root must not crash startup or silently weaken trust; it logs
    and falls back to structural, which reports vendor_verified=False)."""
    import os
    environ = env if env is not None else os.environ
    pem = (environ.get("PRSM_INTEL_SGX_ROOT_CA_PEM", "") or "").strip()
    root_bytes = None
    if pem:
        root_bytes = pem.encode() if isinstance(pem, str) else pem
    else:
        path = (environ.get("PRSM_INTEL_SGX_ROOT_CA_FILE", "") or "").strip()
        if path:
            try:
                with open(path, "rb") as f:
                    root_bytes = f.read()
            except OSError as exc:
                logger.warning("Intel SGX Root CA file unreadable (%s) — DCAP "
                               "verification OFF, structural fallback", exc)
                return None
    if not root_bytes:
        # sp1067 — default to the bundled (fingerprint-pinned) Intel SGX Root CA so a
        # daemon does real DCAP without the operator sourcing the PEM. Opt out with
        # PRSM_INTEL_SGX_USE_BUNDLED_ROOT=0 (→ structural fallback, the old default).
        if str(environ.get("PRSM_INTEL_SGX_USE_BUNDLED_ROOT", "")).strip().lower() in (
                "0", "false", "no", "off"):
            return None
        try:
            from prsm.compute.inference.vendor_anchors import bundled_intel_sgx_root_ca
            root_bytes = bundled_intel_sgx_root_ca()
            logger.info("Using bundled Intel SGX Root CA for DCAP verification "
                        "(set PRSM_INTEL_SGX_USE_BUNDLED_ROOT=0 to disable)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("bundled Intel SGX Root CA unusable (%s) — DCAP OFF, "
                           "structural fallback", exc)
            return None
    # sp1060 — optional CRLs (revocation). PEM inline or file; concatenate multiple
    # (Root-CA CRL + PCK-CA CRL). Absent → no revocation check (unchanged).
    crls_bytes = None
    crl_pem = (environ.get("PRSM_INTEL_SGX_CRL_PEM", "") or "").strip()
    if crl_pem:
        crls_bytes = crl_pem.encode() if isinstance(crl_pem, str) else crl_pem
    else:
        crl_path = (environ.get("PRSM_INTEL_SGX_CRL_FILE", "") or "").strip()
        if crl_path:
            try:
                with open(crl_path, "rb") as f:
                    crls_bytes = f.read()
            except OSError as exc:
                logger.warning("Intel SGX CRL file unreadable (%s) — proceeding "
                               "WITHOUT revocation checking", exc)
    # sp1061-1063 — optional TCB-recency enforcement: a signed TCB-Info + its signer
    # cert. Both required to enable; one without the other is a likely misconfig
    # (log + leave TCB checking OFF rather than fail). Absent → signature-chain only.
    def _read_cfg(pem_var, file_var, label):
        v = (environ.get(pem_var, "") or "").strip()
        if v:
            return v.encode() if isinstance(v, str) else v
        path = (environ.get(file_var, "") or "").strip()
        if path:
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError as exc:
                logger.warning("%s file unreadable (%s) — TCB enforcement OFF", label, exc)
        return None

    tcb_info = _read_cfg("PRSM_INTEL_SGX_TCB_INFO_JSON",
                         "PRSM_INTEL_SGX_TCB_INFO_FILE", "Intel SGX TCB-Info")
    tcb_signer = _read_cfg("PRSM_INTEL_SGX_TCB_SIGNING_PEM",
                           "PRSM_INTEL_SGX_TCB_SIGNING_FILE", "Intel SGX TCB Signing cert")
    if bool(tcb_info) != bool(tcb_signer):
        logger.warning("Intel SGX TCB enforcement needs BOTH the TCB-Info and the "
                       "TCB Signing cert — only one configured; TCB checking OFF")
        tcb_info = tcb_signer = None
    from prsm.compute.inference.sgx_tcb import tcb_policy_from_env
    try:
        return IntelDCAPBackend(
            trusted_root_pem=root_bytes, crls_pem=crls_bytes,
            tcb_info_json=tcb_info, tcb_signer_cert_pem=tcb_signer,
            tcb_policy=tcb_policy_from_env(environ))
    except ValueError as exc:
        logger.warning("Intel SGX Root CA / TCB config invalid (%s) — DCAP "
                       "verification OFF, structural fallback", exc)
        return None
