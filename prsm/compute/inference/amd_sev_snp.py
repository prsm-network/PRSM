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

    # Registry routing key for the PRSM SEV-SNP envelope (sp1050). Distinct from a
    # raw report's "amd-sev-snp" so this real backend claims ONLY the envelope and a
    # raw report still routes to the structural AMDKDSBackend. The result.vendor
    # this backend returns is still "amd-sev-snp" (the actual vendor).
    handles_vendor: str = "amd-sev-snp-envelope"

    def __init__(self, trusted_root_pem: Optional[bytes] = None,
                 crls_pem: Optional[bytes] = None, tcb_policy=None,
                 trusted_root_pems: Optional[List[bytes]] = None,
                 vmpl_policy=None):
        # sp1067 — AMD has per-product ARKs (Milan/Genoa/Turin), so accept a SET of
        # roots; a VCEK chain selects its product's ARK by name. A single
        # trusted_root_pem still works (back-compat). At least one is required.
        pems = list(trusted_root_pems or [])
        if trusted_root_pem:
            pems.append(trusted_root_pem)
        if not pems:
            raise ValueError(
                "AMDSEVSNPBackend requires at least one trusted root (AMD ARK) — "
                "vendor_verified is only as strong as these anchors")
        try:
            self._roots = [x509.load_pem_x509_certificate(p) for p in pems]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"a trusted root is not a valid certificate: {exc}")
        # sp1060 — optional AMD KDS CRLs (ARK CRL revoking ASK, ASK CRL revoking
        # VCEK). Present → a revoked VCEK chain cert fails; None → no check (unchanged).
        self._crls = self._load_crls(crls_pem) if crls_pem else None
        # sp1064-1065 — optional TCB-recency enforcement. When a policy is set, a
        # report whose REPORTED_TCB doesn't bind to its VCEK's issued TCB, or falls
        # below the floor, is rejected. None → no TCB check (signature-chain only).
        self._tcb_policy = tcb_policy
        # sp1080 — optional VMPL / guest-policy enforcement. When set, a report from a
        # less-privileged VMPL or a DEBUG-enabled guest is rejected. None → no check.
        self._vmpl_policy = vmpl_policy

    def _select_root(self, chain):
        """Pick the trusted ARK whose subject issued this chain's top cert (the ASK).
        With a single configured root, use it (verify_cert_chain then reports any
        name mismatch). With multiple (the bundled per-product ARKs), select by the
        ASK's issuer name so the right product's ARK is used."""
        if len(self._roots) == 1:
            return self._roots[0]
        issuer = chain[-1].issuer
        for r in self._roots:
            if r.subject == issuer:
                return r
        return None

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

        root = self._select_root(chain)
        if root is None:
            return fail(f"no configured AMD ARK matches the chain's issuer "
                        f"{chain[-1].issuer.rfc4514_string()} (unknown product?)")
        chain_ok, chain_err = verify_cert_chain(
            chain, root, crls=self._crls, require_crl=False)
        if not chain_ok:
            return fail(f"VCEK chain does not verify to ARK root: {chain_err}")

        # sp1064-1065 — TCB-recency: the chain proves AUTHENTICITY; this proves the
        # report's TCB binds to the VCEK's issued TCB AND meets the floor. Only when
        # the operator configured a policy.
        if self._tcb_policy is not None:
            tcb_err = self._evaluate_tcb(report, vcek, vendor_data)
            if tcb_err is not None:
                return AttestationVerificationResult(
                    vendor="amd-sev-snp", vendor_verified=False, signature_chain_ok=True,
                    structural_parse_ok=True, vendor_data=vendor_data, error=tcb_err)

        # sp1080 — VMPL / guest-policy: reject a less-privileged VMPL or a DEBUG-enabled
        # (inspectable, non-confidential) guest. Only when the operator set a policy.
        if self._vmpl_policy is not None:
            vmpl_err = self._evaluate_vmpl(report, vendor_data)
            if vmpl_err is not None:
                return AttestationVerificationResult(
                    vendor="amd-sev-snp", vendor_verified=False, signature_chain_ok=True,
                    structural_parse_ok=True, vendor_data=vendor_data, error=vmpl_err)

        return AttestationVerificationResult(
            vendor="amd-sev-snp", vendor_verified=True, signature_chain_ok=True,
            structural_parse_ok=True, vendor_data=vendor_data, error=None)

    def _evaluate_vmpl(self, report: bytes, vendor_data: dict):
        """Return an error message if the report's VMPL/guest-policy violates the
        configured policy; None if acceptable. Adds vmpl/debug to vendor_data.
        Fail-closed on any parse error."""
        from prsm.compute.inference.amd_vmpl import (
            SnpPolicyError, parse_snp_report_flags, verify_snp_vmpl_policy,
        )
        try:
            flags = parse_snp_report_flags(report)
            vendor_data["vmpl"] = flags.vmpl
            vendor_data["debug"] = flags.debug
            verify_snp_vmpl_policy(flags, self._vmpl_policy)
        except SnpPolicyError as exc:
            return f"VMPL/guest-policy check failed (fail-closed): {exc}"
        return None

    def _evaluate_tcb(self, report: bytes, vcek, vendor_data: dict):
        """Return an error message if the report's TCB doesn't bind to the VCEK or
        falls below the policy floor; None if acceptable. Adds reported_tcb to
        vendor_data. Fail-closed on any parse error."""
        from prsm.compute.inference.amd_tcb import (
            SnpTcbError, check_tcb_binding, parse_reported_tcb, parse_vcek_tcb,
        )
        try:
            reported = parse_reported_tcb(report)
            issued = parse_vcek_tcb(vcek)
            vendor_data["reported_tcb"] = {
                "bootloader": reported.bootloader, "tee": reported.tee,
                "snp": reported.snp, "microcode": reported.microcode}
            check_tcb_binding(reported, issued)
        except SnpTcbError as exc:
            return f"TCB binding/parse failed (fail-closed): {exc}"
        if not self._tcb_policy.is_acceptable(reported):
            return (f"TCB below the configured floor (downlevel TEE): reported "
                    f"{vendor_data['reported_tcb']}")
        return None


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
    bundled_arks = None
    if not root_bytes:
        # sp1067 — default to the bundled (fingerprint-pinned) AMD product ARKs so a
        # daemon does real SEV-SNP verification without sourcing the ARK PEMs. Opt out
        # with PRSM_AMD_SEV_SNP_USE_BUNDLED_ROOT=0 (→ structural fallback).
        if str(environ.get("PRSM_AMD_SEV_SNP_USE_BUNDLED_ROOT", "")).strip().lower() in (
                "0", "false", "no", "off"):
            return None
        try:
            from prsm.compute.inference.vendor_anchors import bundled_amd_arks
            bundled_arks = list(bundled_amd_arks().values())
            logger.info("Using bundled AMD ARKs (Milan/Genoa/Turin) for SEV-SNP "
                        "verification (set PRSM_AMD_SEV_SNP_USE_BUNDLED_ROOT=0 to disable)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("bundled AMD ARKs unusable (%s) — SEV-SNP OFF, structural "
                           "fallback", exc)
            return None
    # sp1060 — optional AMD KDS CRLs (PEM inline / file; concatenate ARK + ASK CRLs).
    crls_bytes = None
    crl_pem = (environ.get("PRSM_AMD_SEV_SNP_CRL_PEM", "") or "").strip()
    if crl_pem:
        crls_bytes = crl_pem.encode() if isinstance(crl_pem, str) else crl_pem
    else:
        crl_path = (environ.get("PRSM_AMD_SEV_SNP_CRL_FILE", "") or "").strip()
        if crl_path:
            try:
                with open(crl_path, "rb") as f:
                    crls_bytes = f.read()
            except OSError as exc:
                logger.warning("AMD KDS CRL file unreadable (%s) — proceeding "
                               "WITHOUT revocation checking", exc)
    # sp1064-1065 — optional TCB-recency policy (PRSM_AMD_SEV_SNP_MIN_TCB =
    # "bootloader,tee,snp,microcode"). Absent → no TCB check (unchanged).
    from prsm.compute.inference.amd_tcb import snp_tcb_policy_from_env
    try:
        tcb_policy = snp_tcb_policy_from_env(environ)
    except ValueError as exc:
        # sp1065 review #5 — the operator ASKED for TCB enforcement but the value is
        # malformed. Building the backend with enforcement silently OFF would let a
        # downlevel TEE pass (vendor_verified=true). FAIL CLOSED: refuse to build the
        # real backend → structural fallback returns vendor_verified=false.
        logger.error("PRSM_AMD_SEV_SNP_MIN_TCB invalid (%s) — REFUSING to build the "
                     "SEV-SNP backend (would skip the TCB enforcement you configured); "
                     "structural fallback (vendor_verified=false)", exc)
        return None
    # sp1080 — optional VMPL / guest-policy enforcement. Malformed → refuse to build
    # (same fail-closed reasoning as the TCB policy: silently skipping a policy the
    # operator asked for would let a debug/less-privileged guest pass).
    from prsm.compute.inference.amd_vmpl import snp_vmpl_policy_from_env
    try:
        vmpl_policy = snp_vmpl_policy_from_env(environ)
    except ValueError as exc:
        logger.error("PRSM_AMD_SEV_SNP_VMPL_POLICY config invalid (%s) — REFUSING to "
                     "build the SEV-SNP backend; structural fallback", exc)
        return None
    try:
        return AMDSEVSNPBackend(trusted_root_pem=root_bytes,
                                trusted_root_pems=bundled_arks,
                                crls_pem=crls_bytes, tcb_policy=tcb_policy,
                                vmpl_policy=vmpl_policy)
    except ValueError as exc:
        logger.warning("AMD ARK invalid (%s) — SEV-SNP verification OFF, structural "
                       "fallback", exc)
        return None
