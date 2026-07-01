"""Sprint 1336 — lenient X.509 loading for AMD's non-positive certificate serials.

The sp1335 SEV-SNP hardware run surfaced a CryptographyDeprecationWarning ("Parsed a serial
number which wasn't positive") on the real Milan VCEK — a FUTURE cryptography release will
RAISE, breaking attestation verification. The lenient loader keeps cryptography's own parsing
but sidesteps the serial rejection, returning the ORIGINAL tbs/signature/serial so the trust
decision is byte-identical. These tests craft a validly-signed serial-0 CA cert and prove the
fallback preserves the signed bytes AND still chain-verifies.
"""
from __future__ import annotations

import datetime
import warnings

import pytest
from asn1crypto import x509 as a_x509
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from prsm.compute.inference.x509_lenient import (
    _lenient_from_der,
    _pem_to_der,
    load_der_x509_certificate_lenient,
    load_pem_x509_certificate_lenient,
)


def _ca_cert_with_serial(serial: int):
    """A validly self-signed EC-P384 CA cert whose serial is forced to ``serial`` (0 →
    non-positive, the AMD case) by re-signing the patched TBS with the same key."""
    key = ec.generate_private_key(ec.SECP384R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "amd-ark-test")])
    good = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(1)
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2040, 1, 1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=False, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=True,
            crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
        .sign(key, hashes.SHA384()))
    c = a_x509.Certificate.load(good.public_bytes(serialization.Encoding.DER))
    c["tbs_certificate"]["serial_number"] = serial
    c["signature_value"] = key.sign(c["tbs_certificate"].dump(), ec.ECDSA(hashes.SHA384()))
    return c.dump(), key


def _normal_der():
    der, _ = _ca_cert_with_serial(12345)
    return der


# ── fast path (today) ─────────────────────────────────────────────────────────

def test_fast_path_positive_serial_is_real_cert():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a positive serial emits NO warning
        cert = load_der_x509_certificate_lenient(_normal_der())
    assert isinstance(cert, x509.Certificate)
    assert cert.serial_number == 12345


def test_fast_path_suppresses_the_deprecation_warning():
    """On today's cryptography a serial-0 cert only WARNS; the loader must swallow it so
    it never escapes to logs (or to a warnings-as-error test/CI config)."""
    bad_der, _ = _ca_cert_with_serial(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # if the loader let the warning out, this raises
        cert = load_der_x509_certificate_lenient(bad_der)
    assert cert.serial_number == 0  # loaded, no warning escaped


# ── fallback (future cryptography raises) ─────────────────────────────────────

def test_fallback_shim_preserves_original_signed_bytes():
    bad_der, _ = _ca_cert_with_serial(0)
    orig = a_x509.Certificate.load(bad_der)
    shim = _lenient_from_der(bad_der)
    assert shim.serial_number == 0
    assert shim.tbs_certificate_bytes == orig["tbs_certificate"].dump()
    assert shim.signature == orig["signature_value"].native
    # delegated attributes come straight from cryptography's parse of the normalized cert
    assert shim.issuer == shim.subject
    assert shim.public_key() is not None
    bc = shim.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert bc.ca is True


def test_fallback_bad_serial_cert_still_chain_verifies():
    """THE property: a non-positive-serial CA cert that cryptography refuses to load still
    verifies to itself through the shim — the fallback changes nothing about trust."""
    from prsm.compute.inference.x509_path import verify_cert_chain
    bad_der, _ = _ca_cert_with_serial(0)
    shim = _lenient_from_der(bad_der)
    ok, err = verify_cert_chain([shim], shim, crls=None, require_crl=False)
    assert ok, err


def test_fallback_detects_tampered_signature():
    """The shim must NOT be a rubber stamp: tampering the signed report/cert bytes fails."""
    from prsm.compute.inference.x509_path import verify_cert_chain
    bad_der, _ = _ca_cert_with_serial(0)
    shim = _lenient_from_der(bad_der)
    shim._sig = b"\x00" * len(shim._sig)  # corrupt the signature
    ok, _err = verify_cert_chain([shim], shim, crls=None, require_crl=False)
    assert ok is False


# ── producer transcode (runtime.py) ───────────────────────────────────────────

def test_der_to_pem_roundtrips_without_parsing():
    from prsm.compute.tee.runtime import SevSnpTEERuntime
    der = _normal_der()
    pem = SevSnpTEERuntime._der_to_pem(der)
    assert pem.startswith(b"-----BEGIN CERTIFICATE-----")
    assert pem.rstrip().endswith(b"-----END CERTIFICATE-----")
    assert _pem_to_der(pem) == der  # byte-exact, no parse needed


def test_verifier_load_chain_accepts_bad_serial():
    from prsm.compute.inference.amd_sev_snp import _load_chain
    bad_der, _ = _ca_cert_with_serial(0)
    pem = SevSnpTEERuntime_der_to_pem(bad_der)
    chain = _load_chain(pem)
    assert len(chain) == 1
    assert chain[0].serial_number == 0


def SevSnpTEERuntime_der_to_pem(der):
    from prsm.compute.tee.runtime import SevSnpTEERuntime
    return SevSnpTEERuntime._der_to_pem(der)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
