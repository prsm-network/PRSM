"""Sprint 1049 — shared X.509 certificate-chain path validation for TEE attestation.

Extracted from the sp1044 Intel DCAP verifier (where an adversarial review found a
CRITICAL forge-bypass in a signature-only chain walk) so both the Intel SGX (PCK →
intermediate → Intel root) and AMD SEV-SNP (VCEK → ASK → ARK) verifiers share ONE
hardened path-validator. Curve-agnostic: each link is checked against its issuer's
public key using the child's own ``signature_hash_algorithm`` (works for P-256 and
P-384 alike).

The load-bearing rule (the bypass the Intel review caught): every cert that acts as
an ISSUER must be a CA (BasicConstraints ca=True). Without it, a genuine but non-CA
leaf (a real Intel PCK leaf / AMD VCEK) could be used by its holder to "issue" a
forged signer cert and attest an attacker-chosen measurement.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ec


def _within_validity(cert, now) -> bool:
    try:
        nvb, nva = cert.not_valid_before_utc, cert.not_valid_after_utc
    except AttributeError:  # pragma: no cover - cryptography < 42
        nvb = cert.not_valid_before.replace(tzinfo=timezone.utc)
        nva = cert.not_valid_after.replace(tzinfo=timezone.utc)
    return nvb <= now <= nva


def _is_ca(cert) -> bool:
    try:
        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    except x509.ExtensionNotFound:
        return False
    return bool(bc.ca)


def _ca_path_len(cert) -> Optional[int]:
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


def verify_cert_chain(
    chain: List[x509.Certificate],
    trusted_root: x509.Certificate,
    now: Optional[datetime] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate ``chain`` = [end-entity, intermediate(s)...] up to ``trusted_root``.

    Beyond verifying each cert's signature against its issuer's key, enforce the
    path-validation rules that a signature-only walk omits (the Intel-review
    CRITICAL): the ISSUER of every link must be a validity-current, keyCertSign-
    capable CA whose pathLenConstraint allows the depth below it, with proper name
    chaining; only the end-entity (chain[0], which signs the attestation report)
    may be non-CA. Returns (ok, error_message)."""
    if not chain:
        return False, "empty certificate chain"
    if now is None:
        now = datetime.now(timezone.utc)
    full = list(chain) + [trusted_root]

    for cert in full:
        if not _within_validity(cert, now):
            return False, f"{cert.subject.rfc4514_string()} outside its validity window"

    for depth, (child, issuer) in enumerate(zip(full, full[1:])):
        if child.issuer != issuer.subject:
            return False, "issuer/subject name mismatch in chain"
        if not _is_ca(issuer):
            return False, (f"{issuer.subject.rfc4514_string()} is not a CA "
                           f"(BasicConstraints ca!=True) — cannot issue certificates")
        if not _allows_cert_sign(issuer):
            return False, f"{issuer.subject.rfc4514_string()} KeyUsage forbids keyCertSign"
        plen = _ca_path_len(issuer)
        cas_below = sum(1 for c in full[1:depth + 1] if _is_ca(c))
        if plen is not None and cas_below > plen:
            return False, (f"{issuer.subject.rfc4514_string()} pathLenConstraint={plen} "
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

    if not _is_ca(trusted_root):
        return False, "configured trusted root is not a CA"
    try:
        trusted_root.public_key().verify(
            trusted_root.signature, trusted_root.tbs_certificate_bytes,
            ec.ECDSA(trusted_root.signature_hash_algorithm),
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"trusted root is not self-consistent: {exc}"
    return True, None
