"""Sprint 1060 — CRL revocation in the shared TEE X.509 path validator (Tier 3).

The sp1049 shared validator (used by BOTH Intel SGX PCK chains and AMD SEV-SNP
VCEK chains) verifies signatures + CA-flag/pathLen/validity/name-chaining but does
NOT check revocation: a PCK/VCEK cert that Intel/AMD has REVOKED (a compromised or
decommissioned TEE) still verifies → vendor_verified=true for an untrusted part.
This closes that hole. Backward-compatible: crls=None keeps the current behavior;
passing crls turns on revocation enforcement, fail-closed (a missing/forged/expired
CRL can't be used to prove non-revocation under require_crl).
"""
from __future__ import annotations

import datetime as _dt

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from prsm.compute.inference.x509_path import verify_cert_chain

_NOW = _dt.datetime(2025, 6, 1, tzinfo=_dt.timezone.utc)


def _mk_cert(subject, issuer, subject_pub, issuer_priv, *, ca):
    nvb = _dt.datetime(2025, 1, 1)
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
            .public_key(subject_pub)
            .serial_number(x509.random_serial_number())
            .not_valid_before(nvb)
            .not_valid_after(nvb + _dt.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
            .sign(issuer_priv, hashes.SHA256()))


def _mk_crl(issuer_name, issuer_priv, revoked_serials, *, this_update=None, next_update=None):
    tu = this_update or (_NOW - _dt.timedelta(days=1))
    nu = next_update or (_NOW + _dt.timedelta(days=7))
    b = (x509.CertificateRevocationListBuilder()
         .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_name)]))
         .last_update(tu).next_update(nu))
    for s in revoked_serials:
        b = b.add_revoked_certificate(
            x509.RevokedCertificateBuilder().serial_number(s)
            .revocation_date(tu).build())
    return b.sign(issuer_priv, hashes.SHA256())


def _chain():
    """A 2-cert hierarchy: leaf issued directly by the root. Returns
    (leaf, root, root_priv)."""
    root_priv = ec.generate_private_key(ec.SECP256R1())
    leaf_priv = ec.generate_private_key(ec.SECP256R1())
    root = _mk_cert("Test Root CA", "Test Root CA", root_priv.public_key(), root_priv, ca=True)
    leaf = _mk_cert("Test Leaf", "Test Root CA", leaf_priv.public_key(), root_priv, ca=False)
    return leaf, root, root_priv


def test_crls_none_is_backward_compatible():
    leaf, root, _ = _chain()
    ok, err = verify_cert_chain([leaf], root, now=_NOW)
    assert ok is True and err is None


def test_revoked_leaf_is_rejected():
    leaf, root, root_priv = _chain()
    crl = _mk_crl("Test Root CA", root_priv, [leaf.serial_number])
    ok, err = verify_cert_chain([leaf], root, now=_NOW, crls=[crl])
    assert ok is False
    assert "revoked" in err.lower()


def test_non_revoked_leaf_with_valid_crl_passes():
    leaf, root, root_priv = _chain()
    other_serial = x509.random_serial_number()
    crl = _mk_crl("Test Root CA", root_priv, [other_serial])
    ok, err = verify_cert_chain([leaf], root, now=_NOW, crls=[crl])
    assert ok is True and err is None


def test_missing_crl_fails_closed_when_required():
    leaf, root, _ = _chain()
    ok, err = verify_cert_chain([leaf], root, now=_NOW, crls=[], require_crl=True)
    assert ok is False
    assert "crl" in err.lower()


def test_missing_crl_allowed_when_not_required():
    leaf, root, _ = _chain()
    ok, err = verify_cert_chain([leaf], root, now=_NOW, crls=[], require_crl=False)
    assert ok is True and err is None


def test_forged_crl_signed_by_wrong_key_is_not_trusted():
    """A 'clean' CRL signed by an attacker key (not the issuer) must NOT be accepted
    as proof of non-revocation — under require_crl that's a missing CRL → reject."""
    leaf, root, _ = _chain()
    attacker = ec.generate_private_key(ec.SECP256R1())
    forged = _mk_crl("Test Root CA", attacker, [])  # claims leaf not revoked
    ok, err = verify_cert_chain([leaf], root, now=_NOW, crls=[forged], require_crl=True)
    assert ok is False
    assert "crl" in err.lower()


def test_expired_crl_is_not_trusted():
    leaf, root, root_priv = _chain()
    stale = _mk_crl("Test Root CA", root_priv, [],
                    this_update=_NOW - _dt.timedelta(days=30),
                    next_update=_NOW - _dt.timedelta(days=1))  # nextUpdate in the past
    ok, err = verify_cert_chain([leaf], root, now=_NOW, crls=[stale], require_crl=True)
    assert ok is False
    assert "crl" in err.lower()


def test_revoked_leaf_rejected_even_with_a_valid_clean_crl_present():
    """Defense: a revoking CRL must win even if a stale/irrelevant CRL is also in
    the bundle (the matching valid CRL is the one that revokes)."""
    leaf, root, root_priv = _chain()
    revoking = _mk_crl("Test Root CA", root_priv, [leaf.serial_number])
    ok, err = verify_cert_chain([leaf], root, now=_NOW, crls=[revoking])
    assert ok is False and "revoked" in err.lower()


# ── end-to-end through the Intel DCAP backend (revocation flips vendor_verified) ──

def test_intel_dcap_revoked_pck_leaf_flips_vendor_verified_false():
    """A revoked PCK leaf must make the full IntelDCAPBackend report
    vendor_verified=False when its issuer's CRL is configured — while the SAME quote
    verifies True without the CRL (proving the CRL is what rejects it)."""
    import os
    import sys
    import struct
    from cryptography.hazmat.primitives import serialization
    sys.path.insert(0, os.path.dirname(__file__))
    from test_sprint_1044_intel_dcap_verify import _mk_cert, _assemble_quote
    from prsm.compute.inference.intel_dcap import IntelDCAPBackend

    root_priv = ec.generate_private_key(ec.SECP256R1())
    inter_priv = ec.generate_private_key(ec.SECP256R1())
    pck_priv = ec.generate_private_key(ec.SECP256R1())
    ak_priv = ec.generate_private_key(ec.SECP256R1())
    root_c = _mk_cert("Test SGX Root CA", "Test SGX Root CA", root_priv.public_key(), root_priv, ca=True)
    inter_c = _mk_cert("Test Intermediate CA", "Test SGX Root CA", inter_priv.public_key(), root_priv, ca=True)
    pck_c = _mk_cert("Test PCK Leaf", "Test Intermediate CA", pck_priv.public_key(), inter_priv, ca=False)

    quote = _assemble_quote(mrenclave=b"\xaa" * 32, mrsigner=b"\xbb" * 32,
                            ak_priv=ak_priv, qe_signer_priv=pck_priv,
                            chain_certs=[pck_c, inter_c])
    root_pem = root_c.public_bytes(serialization.Encoding.PEM)

    # sanity: verifies True without revocation
    assert IntelDCAPBackend(trusted_root_pem=root_pem).verify(quote).vendor_verified is True

    # the intermediate (PCK leaf's issuer) revokes the PCK leaf's serial
    crl = (x509.CertificateRevocationListBuilder()
           .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Intermediate CA")]))
           .last_update(_dt.datetime(2025, 1, 2, tzinfo=_dt.timezone.utc))
           .next_update(_dt.datetime(2030, 1, 1, tzinfo=_dt.timezone.utc))
           .add_revoked_certificate(
               x509.RevokedCertificateBuilder().serial_number(pck_c.serial_number)
               .revocation_date(_dt.datetime(2025, 1, 2, tzinfo=_dt.timezone.utc)).build())
           .sign(inter_priv, hashes.SHA256()))
    crl_pem = crl.public_bytes(serialization.Encoding.PEM)

    res = IntelDCAPBackend(trusted_root_pem=root_pem, crls_pem=crl_pem).verify(quote)
    assert res.vendor_verified is False
    assert "revoked" in (res.error or "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
