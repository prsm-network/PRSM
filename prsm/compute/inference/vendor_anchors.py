"""Sprint 1067 — bundled real vendor root anchors for Tier-3 attestation defaults.

Ships the genuine Intel SGX Root CA and AMD product ARKs (Milan / Genoa / Turin) as
package data so a daemon defaults to REAL attestation verification without each
operator having to source, verify, and configure the vendor root PEM (a real bring-up
ceremony). These are the SAME anchors the sp1066 golden vectors verify real Intel
PCS / AMD KDS chains against.

SECURITY: bundling trust material means a repo compromise could swap a root. Each
anchor is therefore PINNED by its SHA-256 (DER) fingerprint and verified on load —
a mismatch raises ``AnchorIntegrityError`` (fail-closed) instead of silently trusting
an attacker-substituted root. The fingerprints below are independently checkable
against Intel's and AMD's published certificates.

These defaults only ever make verification STRONGER or fail closed: a forged quote
still fails the chain to the real root; the worst case of a stale bundled root is a
genuine quote failing (vendor_verified=false), never a bad one passing. Operators can
opt out (PRSM_INTEL_SGX_USE_BUNDLED_ROOT=0 / PRSM_AMD_SEV_SNP_USE_BUNDLED_ROOT=0) or
override with their own PEM.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

from cryptography import x509
from cryptography.hazmat.primitives import serialization

_DIR = Path(__file__).parent / "vendor_anchors"

# SHA-256 of each anchor's DER. Independently verifiable against the vendors'
# published roots; a bundled-file swap is caught on load.
_FINGERPRINTS = {
    "intel_sgx_root_ca.pem": "44a0196b2b99f889b8e149e95b807a350e7424964399e885a7cbb8ccfab674d3",
    "amd_ark_milan.pem": "69d063b45344d26a2e94e1f4210de49ef555308287d4c174445c95639a540bcd",
    "amd_ark_genoa.pem": "4c6598d19c18719c5dfd4a7d335f674e5bfe1d8f800cea2cf270c10d103db2f1",
    "amd_ark_turin.pem": "1f084161a44bb6d93778a904877d4819cafa5d05ef4193b2ded9dd9c73dd3f6a",
}

_AMD_PRODUCTS = ("milan", "genoa", "turin")


class AnchorIntegrityError(RuntimeError):
    """A bundled vendor anchor's fingerprint did not match the pin — refuse to use it
    (a swapped root must never be silently trusted)."""


def _load_pinned(name: str) -> bytes:
    pem = (_DIR / name).read_bytes()
    der = x509.load_pem_x509_certificate(pem).public_bytes(serialization.Encoding.DER)
    actual = hashlib.sha256(der).hexdigest()
    expected = _FINGERPRINTS.get(name)
    if expected is None:
        raise AnchorIntegrityError(f"no pinned fingerprint for bundled anchor {name}")
    if actual != expected:
        raise AnchorIntegrityError(
            f"bundled anchor {name} fingerprint mismatch: expected {expected}, got "
            f"{actual} — refusing to use a possibly-swapped trust root")
    return pem


def bundled_intel_sgx_root_ca() -> bytes:
    """The genuine Intel SGX Root CA PEM (fingerprint-verified)."""
    return _load_pinned("intel_sgx_root_ca.pem")


def bundled_amd_arks() -> Dict[str, bytes]:
    """The genuine AMD product ARKs keyed by product ('milan'/'genoa'/'turin'),
    each fingerprint-verified. A SEV-SNP VCEK chain selects its product's ARK."""
    return {p: _load_pinned(f"amd_ark_{p}.pem") for p in _AMD_PRODUCTS}
