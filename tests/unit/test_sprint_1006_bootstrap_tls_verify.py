"""Sprint 1006 — bootstrap WSS client verifies TLS by default (MITM fix).

The P2P-substrate integrity hunt (workflow wbu7u2ftm, finding 9, HIGH) confirmed
that BootstrapClient.connect built its SSL context with
``check_hostname = False`` + ``verify_mode = ssl.CERT_NONE`` unconditionally on
the live ``wss://`` discovery path. That makes the bootstrap connection
MITM-able: an on-path attacker terminates the TLS, impersonates the bootstrap
server, and feeds the joining node an all-attacker peer list — a cold-start
ECLIPSE of every new node.

The pre-fix rationale ("bootstrap servers may use self-signed certs in dev") is
real but was solved the wrong way (disable verification for everyone). The live
fleet (bootstrap-us/eu/apac.prsm-network.com:8765) in fact presents valid
Let's Encrypt certificates that pass full verification including hostname — so
verifying by default is non-breaking for production.

Fix: ``_build_bootstrap_ssl_context`` verifies by default (CERT_REQUIRED +
check_hostname, via ssl.create_default_context). Dev/self-signed deployments
opt in explicitly: PRSM_BOOTSTRAP_TLS_CA_FILE pins a custom CA, or
PRSM_BOOTSTRAP_TLS_INSECURE=1 disables verification (loudly warned). Plain
ws:// returns no context (unchanged).
"""
from __future__ import annotations

import ssl

import pytest

from prsm.bootstrap.client import _build_bootstrap_ssl_context


def test_wss_verifies_by_default(monkeypatch):
    monkeypatch.delenv("PRSM_BOOTSTRAP_TLS_INSECURE", raising=False)
    monkeypatch.delenv("PRSM_BOOTSTRAP_TLS_CA_FILE", raising=False)
    ctx = _build_bootstrap_ssl_context("wss://bootstrap-us.prsm-network.com:8765")
    assert ctx is not None
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def test_plain_ws_has_no_ssl_context():
    assert _build_bootstrap_ssl_context("ws://localhost:8765") is None


def test_insecure_env_opts_out(monkeypatch):
    monkeypatch.setenv("PRSM_BOOTSTRAP_TLS_INSECURE", "1")
    monkeypatch.delenv("PRSM_BOOTSTRAP_TLS_CA_FILE", raising=False)
    ctx = _build_bootstrap_ssl_context("wss://self-signed.local:8765")
    assert ctx is not None
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.check_hostname is False


def test_insecure_env_false_value_stays_secure(monkeypatch):
    monkeypatch.setenv("PRSM_BOOTSTRAP_TLS_INSECURE", "0")
    monkeypatch.delenv("PRSM_BOOTSTRAP_TLS_CA_FILE", raising=False)
    ctx = _build_bootstrap_ssl_context("wss://bootstrap-us.prsm-network.com:8765")
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def test_ca_file_env_loads_custom_anchor(monkeypatch, tmp_path):
    # A real (self-signed) PEM so load_verify_locations succeeds.
    pem = _self_signed_pem()
    ca = tmp_path / "ca.pem"
    ca.write_text(pem)
    monkeypatch.delenv("PRSM_BOOTSTRAP_TLS_INSECURE", raising=False)
    monkeypatch.setenv("PRSM_BOOTSTRAP_TLS_CA_FILE", str(ca))
    ctx = _build_bootstrap_ssl_context("wss://self-signed.local:8765")
    assert ctx is not None
    # Still verifying — the custom CA is the trust anchor, not "no verification".
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    # The pinned CA is now a trusted root.
    der_loaded = {c["subject"] for c in ctx.get_ca_certs()}
    assert len(der_loaded) >= 1


def _self_signed_pem() -> str:
    """Generate a throwaway self-signed cert PEM for the CA-file test."""
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "self-signed.local")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode()
