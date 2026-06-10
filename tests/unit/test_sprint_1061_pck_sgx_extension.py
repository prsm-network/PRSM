"""Sprint 1061 — parse the PCK certificate's Intel SGX extension (Tier 3, brick 1).

The first brick of TCB-recency enforcement. A real Intel PCK leaf carries a custom
X.509 extension (OID 1.2.840.113741.1.13.1) holding the platform's TCB level: 16
SGX TCB component SVNs + the PCESVN + the FMSPC (the platform family the TCB-Info
is keyed by). To decide whether a validly-SIGNED quote comes from an up-to-date
(non-vulnerable) TEE, we must first read this level off the PCK cert, then compare
it against Intel's signed TCB-Info (later bricks).

Structure (Intel PCK cert spec): the extension value is a SEQUENCE of (OID, value)
pairs; the TCB pair's value is itself a SEQUENCE of (OID, value) pairs —
.2.1..2.16 = sgxTcbCompNNSvn (INTEGER 0-255), .2.17 = pcesvn (INTEGER),
.2.18 = cpusvn (OCTET STRING); .4 = fmspc (OCTET STRING, 6 bytes).
"""
from __future__ import annotations

import datetime as _dt

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID, ObjectIdentifier
from pyasn1.type import univ
from pyasn1.codec.der.encoder import encode

from prsm.compute.inference.sgx_tcb import parse_pck_sgx_extension

_SGX_EXT_OID = "1.2.840.113741.1.13.1"


def _pair(oid, val):
    s = univ.Sequence()
    s.setComponentByPosition(0, univ.ObjectIdentifier(oid))
    s.setComponentByPosition(1, val)
    return s


def _sgx_ext_der(comp_svns, pcesvn, fmspc_hex, *, include_fmspc=True):
    """Build a faithful SGX-extension DER value."""
    tcb = univ.Sequence()
    pos = 0
    for i, svn in enumerate(comp_svns, start=1):
        tcb.setComponentByPosition(pos, _pair(f"{_SGX_EXT_OID}.2.{i}", univ.Integer(svn))); pos += 1
    tcb.setComponentByPosition(pos, _pair(f"{_SGX_EXT_OID}.2.17", univ.Integer(pcesvn))); pos += 1
    tcb.setComponentByPosition(pos, _pair(f"{_SGX_EXT_OID}.2.18",
                                          univ.OctetString(hexValue="aa" * 16))); pos += 1
    top = univ.Sequence()
    tpos = 0
    top.setComponentByPosition(tpos, _pair(f"{_SGX_EXT_OID}.1",
                                           univ.OctetString(hexValue="bb" * 16))); tpos += 1  # ppid
    top.setComponentByPosition(tpos, _pair(f"{_SGX_EXT_OID}.2", tcb)); tpos += 1
    top.setComponentByPosition(tpos, _pair(f"{_SGX_EXT_OID}.3",
                                           univ.OctetString(hexValue="0000"))); tpos += 1  # pceid
    if include_fmspc:
        top.setComponentByPosition(tpos, _pair(f"{_SGX_EXT_OID}.4",
                                               univ.OctetString(hexValue=fmspc_hex))); tpos += 1
    return encode(top)


def _cert_with_sgx_ext(der_value):
    priv = ec.generate_private_key(ec.SECP256R1())
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test PCK")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test PCK CA")]))
            .public_key(priv.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_dt.datetime(2025, 1, 1))
            .not_valid_after(_dt.datetime(2035, 1, 1))
            .add_extension(x509.UnrecognizedExtension(
                ObjectIdentifier(_SGX_EXT_OID), der_value), critical=False)
            .sign(priv, hashes.SHA256()))


def test_parses_components_pcesvn_and_fmspc():
    svns = list(range(1, 17))  # 1..16
    cert = _cert_with_sgx_ext(_sgx_ext_der(svns, 11, "00906ed50000"))
    tcb = parse_pck_sgx_extension(cert)
    assert tcb.comp_svns == svns
    assert tcb.pcesvn == 11
    assert tcb.fmspc.hex() == "00906ed50000"


def test_exactly_sixteen_components():
    cert = _cert_with_sgx_ext(_sgx_ext_der([7] * 16, 3, "112233445566"))
    tcb = parse_pck_sgx_extension(cert)
    assert len(tcb.comp_svns) == 16
    assert all(c == 7 for c in tcb.comp_svns)


def test_missing_extension_raises():
    priv = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "No SGX ext")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "No SGX ext")]))
            .public_key(priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1))
            .not_valid_after(_dt.datetime(2035, 1, 1))
            .sign(priv, hashes.SHA256()))
    with pytest.raises(ValueError):
        parse_pck_sgx_extension(cert)


def test_missing_fmspc_raises():
    cert = _cert_with_sgx_ext(_sgx_ext_der(list(range(1, 17)), 11, "", include_fmspc=False))
    with pytest.raises(ValueError):
        parse_pck_sgx_extension(cert)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
