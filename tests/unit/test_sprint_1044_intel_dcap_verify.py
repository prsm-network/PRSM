"""Sprint 1044 — real Intel SGX DCAP (ECDSA-P256) attestation verification (Tier 3).

Closes the §7 attestation_vendor_verified=false gap for Intel SGX: IntelDCAPBackend
cryptographically verifies a v3 ECDSA quote's full signature chain —
  1. the ISV enclave report signature, by the embedded Attestation Key (AK);
  2. the AK is bound to the Quoting Enclave (SHA256(AK || qe_auth_data) ==
     QE report_data[:32]);
  3. the QE report signature, by the PCK leaf certificate;
  4. the PCK X.509 chain (PCK → intermediate → root) up to a CONFIGURED trusted
     root (Intel's SGX Root CA in production).
vendor_verified=True only when all pass. TCB recency / QE-identity / revocation are
explicitly deferred (documented honest scope).

Hermetic: the test builds a SELF-CONSISTENT quote (its own root/intermediate/PCK
chain + AK), so verification runs fully offline, no TEE hardware, no Intel PCS.
Production pins Intel's real SGX Root CA as the trusted root; this test proves the
verification LOGIC by injecting a test root. Tamper/wrong-root tests prove it
REJECTS correctly.
"""
from __future__ import annotations

import struct
import datetime as _dt

import pytest

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    encode_dss_signature, decode_dss_signature,
)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography import x509
from cryptography.x509.oid import NameOID
import hashlib


# ── self-consistent SGX v3 ECDSA quote builder (test fixture) ─────────────────

def _raw64(der_sig: bytes) -> bytes:
    r, s = decode_dss_signature(der_sig)
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _sign_raw(priv: ec.EllipticCurvePrivateKey, data: bytes) -> bytes:
    return _raw64(priv.sign(data, ec.ECDSA(hashes.SHA256())))


def _mk_cert(subject_name, issuer_name, subject_pub, issuer_priv, *, ca: bool,
             not_before=None, not_after=None):
    nvb = not_before or _dt.datetime(2025, 1, 1)
    nva = not_after or (nvb + _dt.timedelta(days=3650))
    b = (x509.CertificateBuilder()
         .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_name)]))
         .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_name)]))
         .public_key(subject_pub)
         .serial_number(x509.random_serial_number())
         .not_valid_before(nvb)
         .not_valid_after(nva)
         .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
    return b.sign(issuer_priv, hashes.SHA256())


def _assemble_quote(*, mrenclave, mrsigner, ak_priv, qe_signer_priv, chain_certs):
    """Low-level quote assembler used by the adversarial tests to inject an
    arbitrary AK / QE-signer / cert chain."""
    ak_pub_raw = ak_priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)[1:]
    isv_report = _build_sgx_report(mrenclave, mrsigner, b"\x11" * 64)
    isv_sig = _sign_raw(ak_priv, isv_report)
    qe_auth = b"qe-auth-blob"
    binding = hashlib.sha256(ak_pub_raw + qe_auth).digest()
    qe_report = _build_sgx_report(b"\xcc" * 32, b"\xdd" * 32, binding + b"\x00" * 32)
    qe_sig = _sign_raw(qe_signer_priv, qe_report)
    chain_pem = b"".join(c.public_bytes(serialization.Encoding.PEM) for c in chain_certs)
    sig_data = (isv_sig + ak_pub_raw + qe_report + qe_sig
                + struct.pack("<H", len(qe_auth)) + qe_auth
                + struct.pack("<I", len(chain_pem)) + chain_pem)
    header = struct.pack("<H", 3) + struct.pack("<H", 2) + b"\x00" * 44
    return header + isv_report + struct.pack("<I", len(sig_data)) + sig_data


def _build_sgx_report(mrenclave: bytes, mrsigner: bytes, report_data: bytes) -> bytes:
    """384-byte SGX report body (real offsets): mrenclave@64, mrsigner@128,
    report_data@320."""
    assert len(mrenclave) == 32 and len(mrsigner) == 32 and len(report_data) == 64
    r = bytearray(384)
    r[64:96] = mrenclave
    r[128:160] = mrsigner
    r[320:384] = report_data
    return bytes(r)


def build_self_consistent_quote(*, root_priv=None):
    """Returns (quote_bytes, root_cert_pem, expected). If root_priv given, the
    chain roots at that key (else a fresh root)."""
    # key hierarchy
    if root_priv is None:
        root_priv = ec.generate_private_key(ec.SECP256R1())
    inter_priv = ec.generate_private_key(ec.SECP256R1())
    pck_priv = ec.generate_private_key(ec.SECP256R1())
    ak_priv = ec.generate_private_key(ec.SECP256R1())

    root_cert = _mk_cert("Test SGX Root CA", "Test SGX Root CA",
                         root_priv.public_key(), root_priv, ca=True)
    inter_cert = _mk_cert("Test Intermediate CA", "Test SGX Root CA",
                          inter_priv.public_key(), root_priv, ca=True)
    pck_cert = _mk_cert("Test PCK Leaf", "Test Intermediate CA",
                        pck_priv.public_key(), inter_priv, ca=False)

    # AK public key as raw 64-byte (x||y) uncompressed-minus-prefix
    ak_pub_raw = ak_priv.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint)[1:]   # drop 0x04
    assert len(ak_pub_raw) == 64

    mrenclave = b"\xaa" * 32
    mrsigner = b"\xbb" * 32
    isv_report = _build_sgx_report(mrenclave, mrsigner, b"\x11" * 64)

    # AK signs the ISV report
    isv_sig = _sign_raw(ak_priv, isv_report)

    # QE report_data[:32] binds the AK: SHA256(ak_pub_raw || qe_auth_data)
    qe_auth_data = b"qe-auth-blob"
    binding = hashlib.sha256(ak_pub_raw + qe_auth_data).digest()
    qe_report = _build_sgx_report(b"\xcc" * 32, b"\xdd" * 32, binding + b"\x00" * 32)

    # PCK signs the QE report
    qe_sig = _sign_raw(pck_priv, qe_report)

    # cert chain PEM (PCK || intermediate; root is the configured trust anchor)
    chain_pem = (pck_cert.public_bytes(serialization.Encoding.PEM)
                 + inter_cert.public_bytes(serialization.Encoding.PEM))

    # assemble signature_data
    sig_data = (
        isv_sig                                  # 64
        + ak_pub_raw                             # 64
        + qe_report                              # 384
        + qe_sig                                 # 64
        + struct.pack("<H", len(qe_auth_data)) + qe_auth_data
        + struct.pack("<I", len(chain_pem)) + chain_pem
    )
    header = struct.pack("<H", 3) + struct.pack("<H", 2) + b"\x00" * 44   # 48 bytes
    quote = header + isv_report + struct.pack("<I", len(sig_data)) + sig_data

    root_pem = root_cert.public_bytes(serialization.Encoding.PEM)
    expected = {"mrenclave": mrenclave.hex(), "mrsigner": mrsigner.hex()}
    return quote, root_pem, expected, root_priv


# ── tests ─────────────────────────────────────────────────────────────────────

def test_dcap_verifies_self_consistent_quote():
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    quote, root_pem, expected, _ = build_self_consistent_quote()
    backend = IntelDCAPBackend(trusted_root_pem=root_pem)
    res = backend.verify(quote)
    assert res.vendor == "intel-sgx"
    assert res.structural_parse_ok is True
    assert res.vendor_verified is True          # ← the gap this closes
    assert res.signature_chain_ok is True
    assert res.error is None
    assert res.vendor_data["mrenclave_hex"] == expected["mrenclave"]
    assert res.vendor_data["mrsigner_hex"] == expected["mrsigner"]


def test_dcap_rejects_tampered_isv_report():
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    quote, root_pem, _, _ = build_self_consistent_quote()
    q = bytearray(quote)
    q[112] ^= 0xFF          # flip a byte inside the ISV report (mrenclave region)
    res = IntelDCAPBackend(trusted_root_pem=root_pem).verify(bytes(q))
    assert res.vendor_verified is False
    assert res.error and "signature" in res.error.lower()


def test_dcap_rejects_broken_ak_binding():
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    quote, root_pem, _, _ = build_self_consistent_quote()
    # corrupt the AK pubkey region (offsets: 48 hdr + 384 isv + 4 len + 64 isv_sig
    # = 500 is where ak_pub starts) → breaks the SHA256(AK||auth)==report_data bind
    q = bytearray(quote)
    q[500] ^= 0xFF
    res = IntelDCAPBackend(trusted_root_pem=root_pem).verify(bytes(q))
    assert res.vendor_verified is False


def test_dcap_rejects_wrong_trusted_root():
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    quote, _good_root, _, _ = build_self_consistent_quote()
    # a DIFFERENT root: the PCK chain won't verify up to it
    _, other_root_pem, _, _ = build_self_consistent_quote()
    res = IntelDCAPBackend(trusted_root_pem=other_root_pem).verify(quote)
    assert res.vendor_verified is False
    assert res.error and ("chain" in res.error.lower() or "root" in res.error.lower())


def test_dcap_requires_trusted_root():
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    with pytest.raises(ValueError):
        IntelDCAPBackend(trusted_root_pem=b"")


def test_dcap_handles_truncated_quote_gracefully():
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    quote, root_pem, _, _ = build_self_consistent_quote()
    res = IntelDCAPBackend(trusted_root_pem=root_pem).verify(quote[:100])
    assert res.vendor_verified is False
    assert res.error is not None                # no crash on a short blob


def test_dcap_rejects_forged_cert_issued_by_non_ca_leaf():
    """sp1044 review CRITICAL regression — an attacker holding a genuine Intel PCK
    LEAF (ca=False) must NOT be able to use it to 'issue' a forged cert that signs
    a forged QE report attesting an attacker-chosen MRENCLAVE. The non-CA leaf
    cannot be a valid issuer → vendor_verified MUST be False."""
    root_priv = ec.generate_private_key(ec.SECP256R1())
    inter_priv = ec.generate_private_key(ec.SECP256R1())
    pck_priv = ec.generate_private_key(ec.SECP256R1())   # genuine PCK leaf (ca=False)
    forged_priv = ec.generate_private_key(ec.SECP256R1())  # attacker-controlled
    ak_priv = ec.generate_private_key(ec.SECP256R1())      # attacker AK

    root_cert = _mk_cert("Root", "Root", root_priv.public_key(), root_priv, ca=True)
    inter_cert = _mk_cert("Inter", "Root", inter_priv.public_key(), root_priv, ca=True)
    pck_leaf = _mk_cert("PCK Leaf", "Inter", pck_priv.public_key(), inter_priv, ca=False)
    # attacker mints a child cert SIGNED BY the non-CA PCK leaf's key
    forged = _mk_cert("Forged", "PCK Leaf", forged_priv.public_key(), pck_priv, ca=False)

    quote = _assemble_quote(
        mrenclave=b"\x99" * 32, mrsigner=b"\x99" * 32,   # attacker-chosen measurement
        ak_priv=ak_priv, qe_signer_priv=forged_priv,
        chain_certs=[forged, pck_leaf, inter_cert])

    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    res = IntelDCAPBackend(
        trusted_root_pem=root_cert.public_bytes(serialization.Encoding.PEM)).verify(quote)
    assert res.vendor_verified is False           # bypass closed
    assert res.error and ("CA" in res.error or "issue" in res.error.lower())


def test_dcap_rejects_expired_chain_cert():
    """sp1044 review #2 — a cert outside its validity window must fail."""
    root_priv = ec.generate_private_key(ec.SECP256R1())
    inter_priv = ec.generate_private_key(ec.SECP256R1())
    pck_priv = ec.generate_private_key(ec.SECP256R1())
    ak_priv = ec.generate_private_key(ec.SECP256R1())
    root_cert = _mk_cert("Root", "Root", root_priv.public_key(), root_priv, ca=True)
    inter_cert = _mk_cert("Inter", "Root", inter_priv.public_key(), root_priv, ca=True)
    # PCK leaf EXPIRED in 2001
    pck_leaf = _mk_cert("PCK Leaf", "Inter", pck_priv.public_key(), inter_priv, ca=False,
                        not_before=_dt.datetime(2000, 1, 1),
                        not_after=_dt.datetime(2001, 1, 1))
    quote = _assemble_quote(mrenclave=b"\xaa" * 32, mrsigner=b"\xbb" * 32,
                            ak_priv=ak_priv, qe_signer_priv=pck_priv,
                            chain_certs=[pck_leaf, inter_cert])
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    res = IntelDCAPBackend(
        trusted_root_pem=root_cert.public_bytes(serialization.Encoding.PEM)).verify(quote)
    assert res.vendor_verified is False
    assert res.error and "validity" in res.error.lower()


def test_dcap_non_sgx_version_rejected():
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend
    quote, root_pem, _, _ = build_self_consistent_quote()
    q = bytearray(quote)
    q[0:2] = struct.pack("<H", 4)               # claim TDX v4
    res = IntelDCAPBackend(trusted_root_pem=root_pem).verify(bytes(q))
    assert res.vendor_verified is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
