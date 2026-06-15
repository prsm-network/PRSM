"""Sprint 1122 — Domain-08 review HIGH-4 (final remainder): a missing PRSM_MASTER_KEY
must FAIL CLOSED in production.

The environment master-key path silently minted a per-process-random key when
PRSM_MASTER_KEY was unset — fine for dev, but in production that means every stored
encrypted key becomes undecryptable after a restart (silent data loss) and there's no
real at-rest protection. Now: PRSM_ENV=production + no key → refuse to start. Dev (the
default env) keeps the convenient ephemeral key. Mirrors the schemas.py jwt-secret gate.
"""
from __future__ import annotations

import base64

import pytest
from cryptography.fernet import Fernet

from prsm.core.cryptography.key_management import KeyManager


def test_production_without_master_key_fails_closed(monkeypatch):
    monkeypatch.setenv("PRSM_ENV", "production")
    monkeypatch.delenv("PRSM_MASTER_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PRSM_MASTER_KEY is required in production"):
        KeyManager(config={"master_key_source": "environment"})


def test_production_with_master_key_starts(monkeypatch):
    monkeypatch.setenv("PRSM_ENV", "production")
    monkeypatch.setenv("PRSM_MASTER_KEY", base64.b64encode(Fernet.generate_key()).decode())
    km = KeyManager(config={"master_key_source": "environment"})
    assert km.master_key is not None


def test_development_without_master_key_still_starts(monkeypatch):
    """Dev convenience preserved: default env → ephemeral key + warning, no hard fail."""
    monkeypatch.setenv("PRSM_ENV", "development")
    monkeypatch.delenv("PRSM_MASTER_KEY", raising=False)
    km = KeyManager(config={"master_key_source": "environment"})
    assert km.master_key is not None


def test_unset_env_defaults_to_development(monkeypatch):
    monkeypatch.delenv("PRSM_ENV", raising=False)
    monkeypatch.delenv("PRSM_MASTER_KEY", raising=False)
    km = KeyManager(config={"master_key_source": "environment"})  # must not raise
    assert km.master_key is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
