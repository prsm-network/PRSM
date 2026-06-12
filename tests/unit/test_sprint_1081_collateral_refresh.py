"""Sprint 1081 — attestation collateral auto-refresh (Tier 3 ops completion).

The bundled/configured Intel & AMD CRLs (and TCB-Info / QE-Identity) carry a
nextUpdate (~monthly). When a CRL goes stale, sp1060's _crl_current treats it as "no
CRL" → revocation is SILENTLY not enforced (a revoked TEE passes). This refreshes the
collateral from Intel PCS / AMD KDS on a schedule. The load-bearing safety property:
a refresh must NEVER replace good collateral with stale/invalid/garbage data — validate
BEFORE the atomic swap, keep the existing copy on any bad fetch.
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
    refresh_collateral_item,
)

_NOW = _dt.datetime(2026, 6, 15, tzinfo=_dt.timezone.utc)


def _ca():
    priv = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")]))
            .public_key(priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(priv, hashes.SHA256()))
    return priv, cert


def _crl(priv, issuer_cn="Test CA", *, this_update=None, next_update=None):
    tu = this_update or (_NOW - _dt.timedelta(days=1))
    nu = next_update or (_NOW + _dt.timedelta(days=20))
    crl = (x509.CertificateRevocationListBuilder()
           .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_cn)]))
           .last_update(tu).next_update(nu)
           .sign(priv, hashes.SHA256()))
    return crl.public_bytes(serialization.Encoding.PEM)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# A small Intel-PCS-like CA hierarchy: a self-signed root, and a keyCertSign-capable
# intermediate CA signed by it (verify_cert_chain requires CA basicConstraints +
# keyCertSign keyUsage on issuers).
def _root_and_intermediate():
    root_priv = ec.generate_private_key(ec.SECP256R1())
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel-like Root")])
    root = (x509.CertificateBuilder()
            .subject_name(root_name).issuer_name(root_name)
            .public_key(root_priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=False, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True,
                crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
            .sign(root_priv, hashes.SHA256()))

    int_priv = ec.generate_private_key(ec.SECP256R1())
    int_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel-like PCK CA")])
    intermediate = (x509.CertificateBuilder()
                    .subject_name(int_name).issuer_name(root_name)
                    .public_key(int_priv.public_key()).serial_number(2)
                    .not_valid_before(_dt.datetime(2025, 1, 1))
                    .not_valid_after(_dt.datetime(2034, 1, 1))
                    .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                    .add_extension(x509.KeyUsage(
                        digital_signature=False, content_commitment=False,
                        key_encipherment=False, data_encipherment=False, key_agreement=False,
                        key_cert_sign=True, crl_sign=True, encipher_only=False,
                        decipher_only=False), critical=True)
                    .sign(root_priv, hashes.SHA256()))
    return root_priv, root, int_priv, intermediate


def _pck_crl(int_priv, *, next_update=None):
    nu = next_update or (_NOW + _dt.timedelta(days=20))
    crl = (x509.CertificateRevocationListBuilder()
           .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel-like PCK CA")]))
           .last_update(_NOW - _dt.timedelta(days=1)).next_update(nu)
           .sign(int_priv, hashes.SHA256()))
    return crl.public_bytes(serialization.Encoding.PEM)


def _issuer_chain_header(*certs):
    import urllib.parse
    pem = b"".join(c.public_bytes(serialization.Encoding.PEM) for c in certs)
    return urllib.parse.quote(pem.decode())


# ── validation ──────────────────────────────────────────────────────────────────

def test_valid_current_crl_accepted():
    priv, cert = _ca()
    assert crl_is_valid_and_current(_crl(priv), cert, _NOW) is True


def test_expired_crl_rejected():
    priv, cert = _ca()
    stale = _crl(priv, next_update=_NOW - _dt.timedelta(days=1))
    assert crl_is_valid_and_current(stale, cert, _NOW) is False


def test_wrong_issuer_signature_rejected():
    priv, cert = _ca()
    other_priv, _ = _ca()
    forged = _crl(other_priv)   # signed by a different key
    assert crl_is_valid_and_current(forged, cert, _NOW) is False


def test_garbage_rejected():
    _priv, cert = _ca()
    assert crl_is_valid_and_current(b"not a crl", cert, _NOW) is False
    assert crl_is_valid_and_current(b"", cert, _NOW) is False


# ── refresh item: validate-before-swap, never cache invalid ─────────────────────

def test_valid_fetch_is_cached(tmp_path):
    priv, cert = _ca()
    path = tmp_path / "test.crl.pem"
    fresh = _crl(priv)

    async def fetch(url):
        return fresh

    res = _run(refresh_collateral_item(
        fetch=fetch, url="http://x", validate=lambda b: crl_is_valid_and_current(b, cert, _NOW),
        cache_path=path))
    assert res.ok is True and res.cached is True
    assert path.read_bytes() == fresh


def test_invalid_fetch_does_not_overwrite_existing(tmp_path):
    """The decisive safety test: a bad refresh must NOT clobber the good cached copy."""
    priv, cert = _ca()
    path = tmp_path / "test.crl.pem"
    good = _crl(priv)
    path.write_bytes(good)   # an existing good cached CRL

    async def fetch_garbage(url):
        return b"corrupt or MITM'd response"

    res = _run(refresh_collateral_item(
        fetch=fetch_garbage, url="http://x",
        validate=lambda b: crl_is_valid_and_current(b, cert, _NOW), cache_path=path))
    assert res.ok is False and res.cached is False
    assert path.read_bytes() == good   # untouched — old good copy survives


def test_stale_fetch_does_not_overwrite(tmp_path):
    priv, cert = _ca()
    path = tmp_path / "test.crl.pem"
    good = _crl(priv)
    path.write_bytes(good)
    stale = _crl(priv, next_update=_NOW - _dt.timedelta(days=1))

    async def fetch(url):
        return stale

    res = _run(refresh_collateral_item(
        fetch=fetch, url="http://x",
        validate=lambda b: crl_is_valid_and_current(b, cert, _NOW), cache_path=path))
    assert res.ok is False
    assert path.read_bytes() == good   # stale refresh rejected, good copy survives


def test_fetch_exception_is_caught(tmp_path):
    priv, cert = _ca()
    path = tmp_path / "test.crl.pem"

    async def boom(url):
        raise RuntimeError("network down")

    res = _run(refresh_collateral_item(
        fetch=boom, url="http://x",
        validate=lambda b: crl_is_valid_and_current(b, cert, _NOW), cache_path=path))
    assert res.ok is False and res.reason and "fetch" in res.reason.lower()
    assert not path.exists()   # nothing written


# ── orchestration: Intel PCK CRL refresh, issuer chain validated to the root ────

def test_pck_crl_refresh_validates_chain_and_caches(tmp_path):
    root_priv, root, int_priv, intermediate = _root_and_intermediate()
    crl_pem = _pck_crl(int_priv)
    header = _issuer_chain_header(intermediate, root)

    async def fetch(url):
        return crl_pem, {"SGX-PCK-CRL-Issuer-Chain": header}

    r = CollateralRefresher(tmp_path, fetch=fetch,
                            intel_root_pem=root.public_bytes(serialization.Encoding.PEM),
                            now_fn=lambda: _NOW)
    res = _run(r.refresh_intel_pck_crl("platform"))
    assert res.ok is True
    assert (tmp_path / "intel_pck_platform.crl.pem").read_bytes() == crl_pem


def test_pck_crl_refresh_rejects_chain_not_to_root(tmp_path):
    """A CRL whose issuer chain does NOT chain to our pinned root is refused — this is
    the MITM defense: a valid-looking CRL signed by an attacker CA must not be cached."""
    _rp, _root, int_priv, intermediate = _root_and_intermediate()
    # a DIFFERENT, unrelated root the refresher is configured to trust
    other_root_priv, other_root = _ca()
    crl_pem = _pck_crl(int_priv)
    header = _issuer_chain_header(intermediate)   # chains to _root, not other_root

    async def fetch(url):
        return crl_pem, {"SGX-PCK-CRL-Issuer-Chain": header}

    cache = tmp_path / "intel_pck_platform.crl.pem"
    cache.write_bytes(b"existing good crl")
    r = CollateralRefresher(
        tmp_path, fetch=fetch,
        intel_root_pem=other_root.public_bytes(serialization.Encoding.PEM),
        now_fn=lambda: _NOW)
    res = _run(r.refresh_intel_pck_crl("platform"))
    assert res.ok is False
    assert cache.read_bytes() == b"existing good crl"   # not downgraded


def test_pck_crl_refresh_rejects_missing_issuer_header(tmp_path):
    root_priv, root, int_priv, _intermediate = _root_and_intermediate()
    crl_pem = _pck_crl(int_priv)

    async def fetch(url):
        return crl_pem, {}   # no issuer chain header

    r = CollateralRefresher(tmp_path, fetch=fetch,
                            intel_root_pem=root.public_bytes(serialization.Encoding.PEM),
                            now_fn=lambda: _NOW)
    res = _run(r.refresh_intel_pck_crl("platform"))
    assert res.ok is False
    assert not (tmp_path / "intel_pck_platform.crl.pem").exists()


def test_refresh_all_runs_each_item_and_never_raises(tmp_path):
    root_priv, root, int_priv, intermediate = _root_and_intermediate()
    crl_pem = _pck_crl(int_priv)
    header = _issuer_chain_header(intermediate, root)

    async def fetch(url):
        if "pckcrl" in url:
            return crl_pem, {"SGX-PCK-CRL-Issuer-Chain": header}
        raise RuntimeError("qe identity endpoint down")   # exercise the never-raise path

    r = CollateralRefresher(tmp_path, fetch=fetch,
                            intel_root_pem=root.public_bytes(serialization.Encoding.PEM),
                            now_fn=lambda: _NOW)
    results = _run(r.refresh_all())
    assert results["intel_pck_platform_crl"].ok is True
    assert results["intel_pck_processor_crl"].ok is True
    assert results["intel_qe_identity"].ok is False   # fetch raised, handled


# ── QE-Identity freshness (review M2) ───────────────────────────────────────────

def _qe_signer(priv):
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "QE Signer")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "QE Signer")]))
            .public_key(priv.public_key()).serial_number(1)
            .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1))
            .sign(priv, hashes.SHA256()))


def _signed_qe(priv, *, issue_date=None, next_update=None, omit_next=False):
    import json
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    ei = {"id": "QE", "version": 2, "mrsigner": ("11" * 32),
          "isvprodid": 1,
          "tcbLevels": [{"tcb": {"isvsvn": 8}, "tcbStatus": "UpToDate"}]}
    ei["issueDate"] = issue_date or (_NOW - _dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not omit_next:
        nu = next_update or (_NOW + _dt.timedelta(days=20))
        ei["nextUpdate"] = nu.strftime("%Y-%m-%dT%H:%M:%SZ")
    b = json.dumps(ei, separators=(",", ":")).encode()
    r, s = decode_dss_signature(priv.sign(b, ec.ECDSA(hashes.SHA256())))
    sig = (r.to_bytes(32, "big") + s.to_bytes(32, "big")).hex()
    return b'{"enclaveIdentity":' + b + b',"signature":"' + sig.encode() + b'"}'


def test_qe_identity_fresh_accepted():
    from prsm.compute.inference.collateral_refresh import qe_identity_is_valid
    priv = ec.generate_private_key(ec.SECP256R1())
    blob = _signed_qe(priv)
    assert qe_identity_is_valid(blob, _qe_signer(priv), _NOW) is True


def test_qe_identity_stale_rejected_despite_valid_signature():
    """A correctly-signed but replayed-stale QE-Identity must be rejected (M2)."""
    from prsm.compute.inference.collateral_refresh import qe_identity_is_valid
    priv = ec.generate_private_key(ec.SECP256R1())
    stale = _signed_qe(priv, next_update=_NOW - _dt.timedelta(days=1))
    assert qe_identity_is_valid(stale, _qe_signer(priv), _NOW) is False


def test_qe_identity_missing_next_update_rejected():
    from prsm.compute.inference.collateral_refresh import qe_identity_is_valid
    priv = ec.generate_private_key(ec.SECP256R1())
    no_horizon = _signed_qe(priv, omit_next=True)
    assert qe_identity_is_valid(no_horizon, _qe_signer(priv), _NOW) is False


def test_qe_identity_wrong_signer_rejected():
    from prsm.compute.inference.collateral_refresh import qe_identity_is_valid
    priv = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    blob = _signed_qe(priv)
    assert qe_identity_is_valid(blob, _qe_signer(other), _NOW) is False


# ── partial CRL cache surfaces a warning (review H1) ────────────────────────────

def test_read_cached_crls_warns_on_partial_set(tmp_path, caplog):
    import logging
    from prsm.compute.inference.collateral_refresh import (
        read_cached_intel_crls, INTEL_PCK_PLATFORM_CRL_FILE)
    (tmp_path / INTEL_PCK_PLATFORM_CRL_FILE).write_bytes(b"-----BEGIN X509 CRL-----\nx\n-----END X509 CRL-----\n")
    env = {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path)}
    with caplog.at_level(logging.WARNING):
        out = read_cached_intel_crls(env)
    assert out is not None   # the one present CRL is still returned
    assert any("INCOMPLETE" in r.message for r in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
