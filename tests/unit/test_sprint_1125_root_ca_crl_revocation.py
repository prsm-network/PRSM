"""Sprint 1125 — Domain-01 review B1: revocation-check the collateral issuer chain
against the Intel Root-CA CRL.

The collateral refresh validated its issuer chain to the bundled Intel root but did NO
revocation check (require_crl=False, crls=None) — a REVOKED intermediate CA would still
validate. Now the refresher fetches + caches the Intel Root-CA CRL and passes it so a
revoked intermediate is rejected (best-effort: enforced when the root CRL is cached).
"""
from __future__ import annotations

import datetime as _dt
import urllib.parse

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from prsm.compute.inference.collateral_refresh import (
    CollateralRefresher,
    read_cached_intel_root_ca_crl,
)

_NOW = _dt.datetime(2030, 6, 1, tzinfo=_dt.timezone.utc)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _root_and_intermediate():
    root_priv = ec.generate_private_key(ec.SECP256R1())
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel-like Root")])
    ku = x509.KeyUsage(
        digital_signature=False, content_commitment=False, key_encipherment=False,
        data_encipherment=False, key_agreement=False, key_cert_sign=True,
        crl_sign=True, encipher_only=False, decipher_only=False)
    root = (x509.CertificateBuilder()
            .subject_name(root_name).issuer_name(root_name)
            .public_key(root_priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(ku, critical=True)
            .sign(root_priv, hashes.SHA256()))
    int_priv = ec.generate_private_key(ec.SECP256R1())
    int_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel-like PCK CA")])
    intermediate = (x509.CertificateBuilder()
                    .subject_name(int_name).issuer_name(root_name)
                    .public_key(int_priv.public_key()).serial_number(2)
                    .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2034, 1, 1))
                    .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                    .add_extension(ku, critical=True)
                    .sign(root_priv, hashes.SHA256()))
    return root_priv, root, int_priv, intermediate


def _root_crl(root_priv, *, revoked_serials=()):
    b = (x509.CertificateRevocationListBuilder()
         .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel-like Root")]))
         .last_update(_NOW - _dt.timedelta(days=1))
         .next_update(_NOW + _dt.timedelta(days=20)))
    for s in revoked_serials:
        b = b.add_revoked_certificate(
            x509.RevokedCertificateBuilder().serial_number(s)
            .revocation_date(_NOW - _dt.timedelta(days=1)).build())
    return b.sign(root_priv, hashes.SHA256()).public_bytes(serialization.Encoding.PEM)


def _pck_crl(int_priv):
    crl = (x509.CertificateRevocationListBuilder()
           .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel-like PCK CA")]))
           .last_update(_NOW - _dt.timedelta(days=1)).next_update(_NOW + _dt.timedelta(days=20))
           .sign(int_priv, hashes.SHA256()))
    return crl.public_bytes(serialization.Encoding.PEM)


def _chain_header(*certs):
    pem = b"".join(c.public_bytes(serialization.Encoding.PEM) for c in certs)
    return urllib.parse.quote(pem.decode())


def _refresher(tmp_path, root, *, fetch):
    return CollateralRefresher(
        tmp_path, fetch=fetch,
        intel_root_pem=root.public_bytes(serialization.Encoding.PEM),
        now_fn=lambda: _NOW)


def _router(root_crl_pem, pck_crl_pem, chain_header):
    async def fetch(url):
        if "rootcacrl" in url:
            return root_crl_pem, {}
        if "pckcrl" in url:
            return pck_crl_pem, {"SGX-PCK-CRL-Issuer-Chain": chain_header}
        raise AssertionError(f"unexpected url {url}")
    return fetch


def test_refresh_root_ca_crl_caches(tmp_path):
    root_priv, root, _ip, _i = _root_and_intermediate()
    root_crl = _root_crl(root_priv)
    fetch = _router(root_crl, b"", "")
    r = _refresher(tmp_path, root, fetch=fetch)
    res = _run(r.refresh_intel_root_ca_crl())
    assert res.ok is True
    # the refresher wrote the cache file directly...
    assert (tmp_path / "intel_root_ca.crl.pem").read_bytes() == root_crl
    # ...and the env-driven read helper finds it
    assert read_cached_intel_root_ca_crl(
        {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path)}) == root_crl


def test_revoked_intermediate_in_issuer_chain_is_rejected(tmp_path):
    root_priv, root, int_priv, intermediate = _root_and_intermediate()
    # the Root-CA CRL REVOKES the intermediate (serial 2)
    root_crl = _root_crl(root_priv, revoked_serials=(2,))
    pck_crl = _pck_crl(int_priv)
    fetch = _router(root_crl, pck_crl, _chain_header(intermediate, root))
    r = _refresher(tmp_path, root, fetch=fetch)
    assert _run(r.refresh_intel_root_ca_crl()).ok is True  # cache the revoking root CRL
    # now the PCK CRL refresh's issuer chain contains the REVOKED intermediate → refused
    res = _run(r.refresh_intel_pck_crl("platform"))
    assert res.ok is False


def test_non_revoked_intermediate_accepted_with_root_crl(tmp_path):
    root_priv, root, int_priv, intermediate = _root_and_intermediate()
    root_crl = _root_crl(root_priv, revoked_serials=(999,))  # revokes some OTHER serial
    pck_crl = _pck_crl(int_priv)
    fetch = _router(root_crl, pck_crl, _chain_header(intermediate, root))
    r = _refresher(tmp_path, root, fetch=fetch)
    assert _run(r.refresh_intel_root_ca_crl()).ok is True
    res = _run(r.refresh_intel_pck_crl("platform"))
    assert res.ok is True  # intermediate not revoked → refresh proceeds


def test_best_effort_when_no_root_crl_cached(tmp_path):
    """Back-compat: without a cached Root-CA CRL, the chain still validates (require_crl=
    False) — revocation is enforced only once the root CRL is present."""
    root_priv, root, int_priv, intermediate = _root_and_intermediate()
    pck_crl = _pck_crl(int_priv)
    fetch = _router(b"", pck_crl, _chain_header(intermediate, root))
    r = _refresher(tmp_path, root, fetch=fetch)
    # do NOT refresh the root CRL first
    res = _run(r.refresh_intel_pck_crl("platform"))
    assert res.ok is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
