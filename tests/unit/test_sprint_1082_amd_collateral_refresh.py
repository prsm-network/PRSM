"""Sprint 1082 — AMD KDS CRL auto-refresh (Tier-3 ops completion; AMD parallel to sp1081).

The AMD parallel to the Intel collateral refresh: a static/bundled AMD VCEK CRL carries
a nextUpdate, and when it expires sp1060's revocation check treats it as "no CRL" → a
revoked VCEK (decommissioned/compromised SEV-SNP platform) silently passes. This refreshes
the AMD VCEK CRL from AMD KDS, validated under the ASK (fetched from KDS cert_chain, bound
to the BUNDLED, fingerprint-pinned ARK — never a KDS-supplied root), and atomically caches
it as PEM for AMDSEVSNPBackend.

Two differences from the Intel path: the CRL is DER (KDS serves DER), and the issuer chain
comes from a SEPARATE cert_chain endpoint (not a response header).
"""
from __future__ import annotations

import datetime as _dt

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from prsm.compute.inference.collateral_refresh import (
    CollateralRefresher,
    crl_is_valid_and_current,
)

_NOW = _dt.datetime(2026, 6, 15, tzinfo=_dt.timezone.utc)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _ca(cn, *, signer_priv=None, signer_cn=None):
    """A keyCertSign+crlSign CA cert. Self-signed unless a signer is given."""
    priv = ec.generate_private_key(ec.SECP256R1())
    issuer_cn = signer_cn or cn
    sk = signer_priv or priv
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_cn)]))
            .public_key(priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=False, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True,
                crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
            .sign(sk, hashes.SHA256()))
    return priv, cert


def _amd_chain():
    """Bundled-ARK-like root + an ASK signed by it (KDS returns [ASK, ARK])."""
    ark_priv, ark = _ca("AMD ARK")
    ask_priv, ask = _ca("AMD ASK", signer_priv=ark_priv, signer_cn="AMD ARK")
    return ark_priv, ark, ask_priv, ask


def _vcek_crl_der(ask_priv, *, next_update=None):
    nu = next_update or (_NOW + _dt.timedelta(days=20))
    crl = (x509.CertificateRevocationListBuilder()
           .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AMD ASK")]))
           .last_update(_NOW - _dt.timedelta(days=1)).next_update(nu)
           .sign(ask_priv, hashes.SHA256()))
    return crl.public_bytes(serialization.Encoding.DER)   # KDS serves DER


def _pem(cert):
    return cert.public_bytes(serialization.Encoding.PEM)


# ── DER CRL support (the KDS serves DER) ────────────────────────────────────────

def test_crl_valid_and_current_accepts_der():
    ask_priv, ask = _ca("AMD ASK")
    der = _vcek_crl_der(ask_priv)
    assert crl_is_valid_and_current(der, ask, _NOW) is True


def test_crl_der_expired_rejected():
    ask_priv, ask = _ca("AMD ASK")
    der = _vcek_crl_der(ask_priv, next_update=_NOW - _dt.timedelta(days=1))
    assert crl_is_valid_and_current(der, ask, _NOW) is False


# ── AMD VCEK CRL refresh: ASK bound to the BUNDLED ARK ──────────────────────────

def _amd_refresher(tmp_path, fetch, ark):
    return CollateralRefresher(
        tmp_path, fetch=fetch, now_fn=lambda: _NOW,
        amd_ark_pems={"milan": _pem(ark)})


def test_amd_crl_refresh_validates_to_bundled_ark_and_caches(tmp_path):
    ark_priv, ark, ask_priv, ask = _amd_chain()
    crl_der = _vcek_crl_der(ask_priv)
    chain_pem = _pem(ask) + _pem(ark)

    async def fetch(url):
        if url.endswith("/crl"):
            return crl_der, {}
        if url.endswith("/cert_chain"):
            return chain_pem, {}
        raise AssertionError(url)

    res = _run(_amd_refresher(tmp_path, fetch, ark).refresh_amd_crl("milan"))
    assert res.ok is True
    cached = (tmp_path / "amd_milan.crl.pem").read_bytes()
    # cached as PEM so the backend's PEM-split _load_crls reads it
    assert b"BEGIN X509 CRL" in cached
    assert crl_is_valid_and_current(cached, ask, _NOW) is True


def test_amd_crl_refresh_rejects_ask_not_chaining_to_bundled_ark(tmp_path):
    """MITM defense: an ASK that does not chain to OUR pinned ARK is refused."""
    _ap, _ark, ask_priv, ask = _amd_chain()
    other_ark_priv, other_ark = _ca("AMD ARK")   # the ARK we actually trust
    crl_der = _vcek_crl_der(ask_priv)
    chain_pem = _pem(ask)   # chains to _ark, not other_ark

    async def fetch(url):
        if url.endswith("/crl"):
            return crl_der, {}
        return chain_pem, {}

    cache = tmp_path / "amd_milan.crl.pem"
    cache.write_bytes(b"-----BEGIN X509 CRL-----\nexisting\n-----END X509 CRL-----\n")
    r = CollateralRefresher(tmp_path, fetch=fetch, now_fn=lambda: _NOW,
                            amd_ark_pems={"milan": _pem(other_ark)})
    res = _run(r.refresh_amd_crl("milan"))
    assert res.ok is False
    assert b"existing" in cache.read_bytes()   # good copy not downgraded


def test_amd_crl_refresh_fetch_failure_keeps_cache(tmp_path):
    _ap, ark, _askp, _ask = _amd_chain()

    async def boom(url):
        raise RuntimeError("KDS down")

    res = _run(_amd_refresher(tmp_path, boom, ark).refresh_amd_crl("milan"))
    assert res.ok is False and "fetch" in (res.reason or "").lower()


def test_read_cached_amd_crls(tmp_path):
    from prsm.compute.inference.collateral_refresh import read_cached_amd_crls
    (tmp_path / "amd_milan.crl.pem").write_bytes(
        b"-----BEGIN X509 CRL-----\nm\n-----END X509 CRL-----\n")
    env = {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path)}
    out = read_cached_amd_crls(env)
    assert out is not None and b"BEGIN X509 CRL" in out


# ── RSA-4096 / RSA-PSS CRL (the REAL AMD ARK/ASK reality — review M5) ────────────
# Every test above uses EC CAs, but real AMD ARK/ASK are RSA-4096 RSA-PSS. sp1066
# caught exactly this class of gap for verify_cert_chain (hardcoded ECDSA rejected the
# genuine AMD chain on hardware). crl.is_signature_valid is a SEPARATE code path from
# the cert-chain signature check, so it gets its own RSA-PSS exercise here.

def _rsa_pss_crl(*, der, next_update=None):
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    priv = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AMD ASK (RSA)")])
    issuer = (x509.CertificateBuilder()
              .subject_name(name).issuer_name(name)
              .public_key(priv.public_key()).serial_number(1)
              .not_valid_before(_dt.datetime(2025, 1, 1))
              .not_valid_after(_dt.datetime(2035, 1, 1))
              .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
              .sign(priv, hashes.SHA384(), rsa_padding=padding.PSS(
                  mgf=padding.MGF1(hashes.SHA384()),
                  salt_length=padding.PSS.DIGEST_LENGTH)))
    nu = next_update or (_NOW + _dt.timedelta(days=20))
    crl = (x509.CertificateRevocationListBuilder()
           .issuer_name(name)
           .last_update(_NOW - _dt.timedelta(days=1)).next_update(nu)
           .sign(priv, hashes.SHA384(), rsa_padding=padding.PSS(
               mgf=padding.MGF1(hashes.SHA384()),
               salt_length=padding.PSS.DIGEST_LENGTH)))
    enc = serialization.Encoding.DER if der else serialization.Encoding.PEM
    return crl.public_bytes(enc), issuer


def test_rsa_pss_der_crl_validates():
    """A real-shape AMD CRL: RSA-4096 issuer, RSA-PSS/SHA-384 signature, DER encoding."""
    der, issuer = _rsa_pss_crl(der=True)
    assert crl_is_valid_and_current(der, issuer, _NOW) is True


def test_rsa_pss_crl_survives_der_to_pem_normalization():
    """_crl_to_pem must preserve the RSA-PSS signature through DER→PEM re-encode."""
    from prsm.compute.inference.collateral_refresh import _crl_to_pem
    der, issuer = _rsa_pss_crl(der=True)
    pem = _crl_to_pem(der)
    assert pem is not None and b"BEGIN X509 CRL" in pem
    assert crl_is_valid_and_current(pem, issuer, _NOW) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
