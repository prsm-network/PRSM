"""Sprint 1123 (Domain-08 review LOW — CORS): with no PRSM_ALLOWED_ORIGINS, the CORS
default is environment-aware — dev keeps permissive `*`, production DEFAULT-DENIES
cross-origin (empty allowlist). An explicit allowlist always wins.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.api import create_api_app


def _cors_allow_origins(app):
    """Pull the configured allow_origins from the mounted CORSMiddleware."""
    from fastapi.middleware.cors import CORSMiddleware
    for mw in app.user_middleware:
        if mw.cls is CORSMiddleware:
            # Starlette Middleware stores kwargs in .kwargs (newer) or .options.
            opts = getattr(mw, "kwargs", None) or getattr(mw, "options", {})
            return opts.get("allow_origins")
    return None


def _app():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    return create_api_app(node, enable_security=False)


def test_dev_unset_origins_is_permissive(monkeypatch):
    monkeypatch.setenv("PRSM_ENV", "development")
    monkeypatch.delenv("PRSM_ALLOWED_ORIGINS", raising=False)
    assert _cors_allow_origins(_app()) == ["*"]


def test_production_unset_origins_denies_all(monkeypatch):
    monkeypatch.setenv("PRSM_ENV", "production")
    monkeypatch.delenv("PRSM_ALLOWED_ORIGINS", raising=False)
    assert _cors_allow_origins(_app()) == []  # deny-all, not "*"


def test_explicit_allowlist_wins_in_production(monkeypatch):
    monkeypatch.setenv("PRSM_ENV", "production")
    monkeypatch.setenv("PRSM_ALLOWED_ORIGINS", "https://app.example.com, https://b.example")
    assert _cors_allow_origins(_app()) == [
        "https://app.example.com", "https://b.example"]


def test_unset_env_defaults_to_permissive(monkeypatch):
    monkeypatch.delenv("PRSM_ENV", raising=False)
    monkeypatch.delenv("PRSM_ALLOWED_ORIGINS", raising=False)
    assert _cors_allow_origins(_app()) == ["*"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
