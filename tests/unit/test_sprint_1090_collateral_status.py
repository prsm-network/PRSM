"""Sprint 1090 — collateral-refresh observability (operational capstone of sp1081-1089).

The auto-refresh keeps the Intel/AMD attestation collateral fresh, but an operator had no
way to SEE whether it's working — making the whole revocation/recency enforcement
unverifiable in production. `collateral_refresh_status` surfaces, per cached item:
present? fresh (within nextUpdate)? age? — so /health/detailed can show "revocation +
recency enforcement is fresh". Mirrors the sp1051 settlement-status pattern.
"""
from __future__ import annotations

import datetime as _dt
import json

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from prsm.compute.inference.collateral_refresh import (
    collateral_refresh_status,
    INTEL_PCK_PLATFORM_CRL_FILE,
    INTEL_QE_IDENTITY_FILE,
)

_NOW = _dt.datetime(2026, 6, 15, tzinfo=_dt.timezone.utc)


def _crl_pem(*, next_update):
    priv = ec.generate_private_key(ec.SECP256R1())
    crl = (x509.CertificateRevocationListBuilder()
           .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CA")]))
           .last_update(_NOW - _dt.timedelta(days=1)).next_update(next_update)
           .sign(priv, hashes.SHA256()))
    return crl.public_bytes(serialization.Encoding.PEM)


def test_status_disabled_and_no_cache():
    st = collateral_refresh_status({})
    assert st["enabled"] is False
    assert st["cache_dir"] is None
    assert st["items"] == {}


def test_status_enabled_flag_reads_env(tmp_path):
    env = {"PRSM_COLLATERAL_AUTO_REFRESH": "1",
           "PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path)}
    st = collateral_refresh_status(env)
    assert st["enabled"] is True
    assert st["cache_dir"] == str(tmp_path)


def test_status_reports_fresh_crl(tmp_path):
    (tmp_path / INTEL_PCK_PLATFORM_CRL_FILE).write_bytes(
        _crl_pem(next_update=_NOW + _dt.timedelta(days=20)))
    env = {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path)}
    st = collateral_refresh_status(env, now=_NOW)
    item = st["items"]["intel_pck_platform_crl"]
    assert item["present"] is True
    assert item["fresh"] is True
    assert item["next_update"] is not None
    assert item["age_seconds"] is not None and item["age_seconds"] >= 0


def test_status_reports_stale_crl(tmp_path):
    (tmp_path / INTEL_PCK_PLATFORM_CRL_FILE).write_bytes(
        _crl_pem(next_update=_NOW - _dt.timedelta(days=1)))
    env = {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path)}
    st = collateral_refresh_status(env, now=_NOW)
    assert st["items"]["intel_pck_platform_crl"]["fresh"] is False


def test_status_reports_qe_identity_freshness(tmp_path):
    blob = json.dumps({"enclaveIdentity": {
        "id": "QE", "nextUpdate": (_NOW + _dt.timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "signature": "ab"}).encode()
    (tmp_path / INTEL_QE_IDENTITY_FILE).write_bytes(blob)
    env = {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path)}
    st = collateral_refresh_status(env, now=_NOW)
    assert st["items"]["intel_qe_identity"]["present"] is True
    assert st["items"]["intel_qe_identity"]["fresh"] is True


def test_status_includes_tcb_info_when_fmspc_configured(tmp_path):
    from prsm.compute.inference.collateral_refresh import intel_tcb_info_cache_file
    fmspc = "00906ed50000"
    blob = json.dumps({"tcbInfo": {
        "fmspc": fmspc, "nextUpdate": (_NOW + _dt.timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "signature": "ab"}).encode()
    (tmp_path / intel_tcb_info_cache_file(fmspc)).write_bytes(blob)
    env = {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path), "PRSM_INTEL_SGX_FMSPC": fmspc}
    st = collateral_refresh_status(env, now=_NOW)
    key = f"intel_tcb_info_{fmspc}"
    assert st["items"][key]["present"] is True
    assert st["items"][key]["fresh"] is True


def test_status_never_raises_on_corrupt_file(tmp_path):
    (tmp_path / INTEL_PCK_PLATFORM_CRL_FILE).write_bytes(b"not a crl")
    env = {"PRSM_ATTESTATION_COLLATERAL_DIR": str(tmp_path)}
    st = collateral_refresh_status(env, now=_NOW)
    item = st["items"]["intel_pck_platform_crl"]
    assert item["present"] is True
    assert item["fresh"] is None   # unparseable → unknown freshness, not a crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
