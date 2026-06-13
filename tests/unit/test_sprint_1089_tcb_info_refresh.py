"""Sprint 1089 — per-FMSPC Intel TCB-Info auto-refresh (completes the Intel collateral
refresh trifecta: CRL [sp1081] + QE-Identity [sp1081] + TCB-Info [sp1089]).

A static/configured TCB-Info carries a ~monthly nextUpdate. When it goes stale, the
sp1061-1063 TCB-recency enforcement keeps evaluating against the OLD tcbLevels — so a
TEE that Intel has SINCE marked OutOfDate (a newly-disclosed vuln) still evaluates as
UpToDate and passes. Same silent-lapse class as the CRL gap sp1081 closed. This refreshes
the per-FMSPC TCB-Info from Intel PCS with the proven validate-before-swap machinery.
"""
from __future__ import annotations

import datetime as _dt
import json

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.x509.oid import NameOID

from prsm.compute.inference.collateral_refresh import (
    CollateralRefresher,
    tcb_info_is_valid_and_current,
    read_cached_intel_tcb_info,
    intel_tcb_info_cache_file,
)

_NOW = _dt.datetime(2026, 6, 15, tzinfo=_dt.timezone.utc)
_FMSPC = "00906ed50000"


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _raw_sig(priv, msg):
    r, s = decode_dss_signature(priv.sign(msg, ec.ECDSA(hashes.SHA256())))
    return (r.to_bytes(32, "big") + s.to_bytes(32, "big")).hex()


def _signer_cert(priv, cn="Intel SGX TCB Signing", *, issuer_priv=None, issuer_cn=None,
                 ca=False):
    ipriv = issuer_priv or priv
    icn = issuer_cn or cn
    b = (x509.CertificateBuilder()
         .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)]))
         .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, icn)]))
         .public_key(priv.public_key()).serial_number(1)
         .not_valid_before(_dt.datetime(2025, 1, 1)).not_valid_after(_dt.datetime(2035, 1, 1)))
    if ca:
        b = (b.add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
              .add_extension(x509.KeyUsage(
                  digital_signature=False, content_commitment=False, key_encipherment=False,
                  data_encipherment=False, key_agreement=False, key_cert_sign=True,
                  crl_sign=True, encipher_only=False, decipher_only=False), critical=True))
    return b.sign(ipriv, hashes.SHA256())


def _signed_tcb_info(priv, *, fmspc=_FMSPC, issue=None, nxt=None, omit_next=False):
    obj = {"id": "SGX", "version": 3, "fmspc": fmspc, "pceId": "0000", "tcbType": 0,
           "tcbEvaluationDataNumber": 15,
           "tcbLevels": [{"tcb": {"sgxtcbcomponents": [{"svn": 10}] * 16, "pcesvn": 12},
                          "tcbStatus": "UpToDate"}]}
    obj["issueDate"] = issue or (_NOW - _dt.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not omit_next:
        obj["nextUpdate"] = (nxt or (_NOW + _dt.timedelta(days=25))).strftime("%Y-%m-%dT%H:%M:%SZ")
    b = json.dumps(obj, separators=(",", ":")).encode()
    return b'{"tcbInfo":' + b + b',"signature":"' + _raw_sig(priv, b).encode() + b'"}'


# ── freshness gate ───────────────────────────────────────────────────────────────

def test_current_tcb_info_accepted():
    priv = ec.generate_private_key(ec.SECP256R1())
    assert tcb_info_is_valid_and_current(_signed_tcb_info(priv), _signer_cert(priv), _NOW) is True


def test_stale_tcb_info_rejected_despite_valid_signature():
    priv = ec.generate_private_key(ec.SECP256R1())
    stale = _signed_tcb_info(priv, nxt=_NOW - _dt.timedelta(days=1))
    assert tcb_info_is_valid_and_current(stale, _signer_cert(priv), _NOW) is False


def test_missing_next_update_rejected():
    priv = ec.generate_private_key(ec.SECP256R1())
    assert tcb_info_is_valid_and_current(
        _signed_tcb_info(priv, omit_next=True), _signer_cert(priv), _NOW) is False


def test_wrong_signer_rejected():
    priv = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    assert tcb_info_is_valid_and_current(
        _signed_tcb_info(priv), _signer_cert(other), _NOW) is False


def test_real_fixture_current_at_now():
    """The committed real Intel TCB-Info (valid 2026-06-10 → 2026-07-10) is current at _NOW."""
    import pathlib
    base = pathlib.Path("tests/fixtures/attestation")
    blob = (base / "intel_tcb_info_00906ed50000.json").read_bytes()
    signer = x509.load_pem_x509_certificate((base / "intel_tcb_signing.pem").read_bytes())
    assert tcb_info_is_valid_and_current(blob, signer, _NOW) is True
    # ...but stale a year later
    assert tcb_info_is_valid_and_current(
        blob, signer, _NOW.replace(year=2027)) is False


# ── refresh orchestration: issuer chain → bundled root, then TCB-Info under leaf ──

def _root_and_signer():
    root_priv = ec.generate_private_key(ec.SECP256R1())
    root = _signer_cert(root_priv, cn="Intel-like Root", ca=True)
    signer_priv = ec.generate_private_key(ec.SECP256R1())
    signer = _signer_cert(signer_priv, cn="TCB Signing", issuer_priv=root_priv,
                          issuer_cn="Intel-like Root")
    return root_priv, root, signer_priv, signer


def _issuer_header(*certs):
    import urllib.parse
    pem = b"".join(c.public_bytes(serialization.Encoding.PEM) for c in certs)
    return urllib.parse.quote(pem.decode())


def test_tcb_info_refresh_validates_chain_and_caches(tmp_path):
    root_priv, root, signer_priv, signer = _root_and_signer()
    blob = _signed_tcb_info(signer_priv)
    header = _issuer_header(signer, root)

    async def fetch(url):
        assert f"fmspc={_FMSPC}" in url
        return blob, {"TCB-Info-Issuer-Chain": header}

    r = CollateralRefresher(tmp_path, fetch=fetch, now_fn=lambda: _NOW,
                            intel_root_pem=root.public_bytes(serialization.Encoding.PEM),
                            intel_fmspc=_FMSPC)
    res = _run(r.refresh_intel_tcb_info(_FMSPC))
    assert res.ok is True
    assert (tmp_path / intel_tcb_info_cache_file(_FMSPC)).read_bytes() == blob


def test_tcb_info_refresh_rejects_chain_not_to_root(tmp_path):
    _rp, _root, signer_priv, signer = _root_and_signer()
    other_root_priv = ec.generate_private_key(ec.SECP256R1())
    other_root = _signer_cert(other_root_priv, cn="Other Root", ca=True)
    blob = _signed_tcb_info(signer_priv)
    header = _issuer_header(signer)   # chains to _root, not other_root

    async def fetch(url):
        return blob, {"TCB-Info-Issuer-Chain": header}

    cache = tmp_path / intel_tcb_info_cache_file(_FMSPC)
    cache.write_bytes(b'{"tcbInfo":"existing"}')
    r = CollateralRefresher(tmp_path, fetch=fetch, now_fn=lambda: _NOW,
                            intel_root_pem=other_root.public_bytes(serialization.Encoding.PEM),
                            intel_fmspc=_FMSPC)
    res = _run(r.refresh_intel_tcb_info(_FMSPC))
    assert res.ok is False
    assert cache.read_bytes() == b'{"tcbInfo":"existing"}'   # good copy not downgraded


def test_refresh_all_includes_tcb_info_when_fmspc_set(tmp_path):
    root_priv, root, signer_priv, signer = _root_and_signer()
    crl = None  # not exercised here; QE/CRL fetches raise → handled

    async def fetch(url):
        if "tcb?fmspc=" in url:
            return _signed_tcb_info(signer_priv), {"TCB-Info-Issuer-Chain": _issuer_header(signer, root)}
        raise RuntimeError("only TCB-Info exercised here")

    r = CollateralRefresher(tmp_path, fetch=fetch, now_fn=lambda: _NOW,
                            intel_root_pem=root.public_bytes(serialization.Encoding.PEM),
                            intel_fmspc=_FMSPC)
    results = _run(r.refresh_all())
    assert results.get(f"intel_tcb_info_{_FMSPC}") is not None
    assert results[f"intel_tcb_info_{_FMSPC}"].ok is True


def test_read_cached_tcb_info(tmp_path):
    (tmp_path / intel_tcb_info_cache_file(_FMSPC)).write_bytes(b'{"tcbInfo":"x"}')
    env = {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path), "PRSM_INTEL_SGX_FMSPC": _FMSPC}
    assert read_cached_intel_tcb_info(env) == b'{"tcbInfo":"x"}'
    # no fmspc configured → None
    assert read_cached_intel_tcb_info({"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path)}) is None


# ── review #4: fmspc must be 12-hex (no URL/filename metacharacter surface) ──────

def test_invalid_fmspc_rejected(tmp_path):
    async def fetch(url):
        raise AssertionError("must not fetch with an invalid fmspc")

    r = CollateralRefresher(tmp_path, fetch=fetch, now_fn=lambda: _NOW)
    for bad in ("../etc", "00906ed50000&x=1", "ZZZ", "00906ed5000", "00906ed500000"):
        res = _run(r.refresh_intel_tcb_info(bad))
        assert res.ok is False and res.reason == "invalid-fmspc"


def test_configured_fmspc_rejects_non_hex():
    from prsm.compute.inference.collateral_refresh import _configured_fmspc
    assert _configured_fmspc({"PRSM_INTEL_SGX_FMSPC": "00906ed50000"}) == "00906ed50000"
    assert _configured_fmspc({"PRSM_INTEL_SGX_FMSPC": "00906ed50000&evil"}) is None
    assert _configured_fmspc({"PRSM_INTEL_SGX_FMSPC": "../../x"}) is None


# ── review #3b: read_cached returns a stale cache (build logs it; doesn't hard-drop) ─

def test_read_cached_returns_stale_cache_for_build_to_warn(tmp_path):
    """The cache reader returns whatever is cached (stale or not); the freshness call is
    the build's to make (it warns but still enforces vs the old baseline)."""
    priv = ec.generate_private_key(ec.SECP256R1())
    stale = _signed_tcb_info(priv, nxt=_NOW - _dt.timedelta(days=1))
    (tmp_path / intel_tcb_info_cache_file(_FMSPC)).write_bytes(stale)
    env = {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path), "PRSM_INTEL_SGX_FMSPC": _FMSPC}
    assert read_cached_intel_tcb_info(env) == stale
    from prsm.compute.inference.collateral_refresh import _tcb_info_is_current
    assert _tcb_info_is_current(stale, _NOW) is False   # build will warn on this


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
