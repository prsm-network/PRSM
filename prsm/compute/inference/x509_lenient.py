"""Sprint 1336 — lenient X.509 loading tolerant of AMD's non-positive certificate serials.

AMD's KDS-issued VCEK/ARK certificates carry serial numbers that violate RFC 5280's
"serial number MUST be a positive integer" rule. ``cryptography`` currently only warns
(``CryptographyDeprecationWarning: Parsed a serial number which wasn't positive``) but a
FUTURE release will RAISE on load — which would break SEV-SNP attestation verification
outright (the sp1335 hardware run surfaced this on a real Milan VCEK).

This wraps the loader so the verifier keeps working across that upgrade WITHOUT
hand-rolling security-critical X.509 verification:

  * FAST PATH (today): ``cryptography``'s own loader, with the deprecation warning
    suppressed for clean logs. Returns a real ``cryptography.x509.Certificate``.
  * FALLBACK (future raise): ``asn1crypto`` re-encodes the serial to a positive
    placeholder so ``cryptography`` will parse the cert — giving us its extensions,
    validity, issuer/subject, public key, and all the chain-path logic for free — while
    the returned ``_LenientCertificate`` OVERRIDES ``tbs_certificate_bytes`` +
    ``signature`` + ``serial_number`` to the ORIGINAL values. So chain SIGNATURE
    verification uses the exact bytes AMD signed, and CRL lookups see the real serial.
    No signed bytes are altered; nothing about the trust decision changes.
"""
from __future__ import annotations

import base64
import warnings
from typing import Any, List

__all__ = [
    "load_der_x509_certificate_lenient",
    "load_pem_x509_certificate_lenient",
]


class _LenientCertificate:
    """Duck-types ``cryptography.x509.Certificate``. Delegates every attribute to a
    serial-normalized real certificate (so extensions / validity / issuer / subject /
    public_key / key-usage all come straight from ``cryptography``), but returns the
    ORIGINAL tbs bytes, signature, and serial so signature verification + revocation
    lookups operate on the authentic, AMD-signed values."""

    def __init__(self, normalized: Any, tbs: bytes, signature: bytes, serial: int) -> None:
        self._c = normalized
        self._tbs = tbs
        self._sig = signature
        self._serial = serial

    @property
    def tbs_certificate_bytes(self) -> bytes:
        return self._tbs

    @property
    def signature(self) -> bytes:
        return self._sig

    @property
    def serial_number(self) -> int:
        return self._serial

    def public_key(self):
        return self._c.public_key()

    def public_bytes(self, encoding):
        return self._c.public_bytes(encoding)

    def __getattr__(self, name):
        # Only reached for attributes NOT defined above (issuer, subject, extensions,
        # not_valid_before_utc, signature_hash_algorithm, …) → the normalized cert is
        # byte-identical to the original for all of these (only the serial differs).
        return getattr(self._c, name)


def _pem_to_der(pem: bytes) -> bytes:
    b64 = b"".join(ln.strip() for ln in pem.splitlines() if b"-----" not in ln)
    return base64.b64decode(b64)


def _lenient_from_der(der: bytes):
    """Build a ``_LenientCertificate`` from a DER cert ``cryptography`` refused to load."""
    from cryptography import x509
    from asn1crypto import x509 as a_x509

    original = a_x509.Certificate.load(der)
    tbs = original["tbs_certificate"].dump()
    sig = original["signature_value"].native
    serial = original.serial_number

    normalized = a_x509.Certificate.load(der)
    normalized["tbs_certificate"]["serial_number"] = abs(serial) if serial else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm_cert = x509.load_der_x509_certificate(normalized.dump())
    return _LenientCertificate(norm_cert, tbs, sig, serial)


def load_der_x509_certificate_lenient(der: bytes):
    """Load a DER cert, tolerating a non-positive serial (see module docstring)."""
    from cryptography import x509
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return x509.load_der_x509_certificate(der)
    except Exception:
        return _lenient_from_der(der)


def load_pem_x509_certificate_lenient(pem: bytes):
    """Load a single PEM cert, tolerating a non-positive serial (see module docstring)."""
    from cryptography import x509
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return x509.load_pem_x509_certificate(pem)
    except Exception:
        return _lenient_from_der(_pem_to_der(pem))
