"""Sprint 1017 — posture check assesses the API bind, not the P2P bind.

The sp1016 live verify surfaced a false-positive in sp1011: `prsm node start`
refused to boot by default. Root cause — sp1011's posture check read
`config.listen_host` (the P2P transport bind, default 0.0.0.0) to judge whether
the *management API's* money/KYC endpoints are publicly exposed. But the API is
served by `_run_api`, which bound `host="127.0.0.1"` (loopback) unconditionally
and ignored configuration. So the API was never publicly exposed, yet sp1011
refused startup citing exposed endpoints — blocking every default node start.

Fix: add a configurable `NodeConfig.api_host` (default 127.0.0.1, override via
PRSM_API_HOST); `_run_api` binds it; and the posture check assesses `api_host`
(the ACTUAL API bind), not `listen_host`. Net: the default (loopback API) just
works without a key, and the fail-closed guard fires only when an operator
*intentionally* exposes the API (api_host=0.0.0.0) without a key.
"""
from __future__ import annotations

import inspect

import pytest

from prsm.node.config import NodeConfig
from prsm.node.node import (
    PRSMNode,
    assess_public_bind_auth_posture,
    should_refuse_insecure_public_bind,
)


def test_api_host_defaults_to_loopback():
    cfg = NodeConfig()
    assert cfg.api_host == "127.0.0.1"
    # P2P stays 0.0.0.0 (peers must reach it); only the API defaults to loopback.
    assert cfg.listen_host == "0.0.0.0"


def test_api_host_env_override(monkeypatch):
    monkeypatch.setenv("PRSM_API_HOST", "0.0.0.0")
    assert NodeConfig().api_host == "0.0.0.0"
    monkeypatch.delenv("PRSM_API_HOST", raising=False)
    assert NodeConfig().api_host == "127.0.0.1"


def test_default_posture_does_not_refuse():
    """The default config (api_host loopback, no key) must NOT refuse — the API
    is on loopback, so there is nothing exposed to fail-close on."""
    cfg = NodeConfig()  # api_host=127.0.0.1, listen_host=0.0.0.0
    level, _ = assess_public_bind_auth_posture(
        listen_host=cfg.api_host, api_key_present=False,
    )
    assert level == "ok"
    assert should_refuse_insecure_public_bind(level, allow_insecure=False) is False


def test_public_api_host_without_key_refuses():
    """An operator who INTENTIONALLY exposes the API (api_host=0.0.0.0) with no
    key is still correctly fail-closed."""
    level, _ = assess_public_bind_auth_posture(
        listen_host="0.0.0.0", api_key_present=False,
    )
    assert level == "insecure"
    assert should_refuse_insecure_public_bind(level, allow_insecure=False) is True


def test_run_api_binds_configured_api_host():
    """Structural pin: _run_api binds config.api_host (not a hardcoded host)."""
    src = inspect.getsource(PRSMNode._run_api)
    assert "api_host" in src
    assert 'host="127.0.0.1"' not in src  # no longer hardcoded


def test_start_posture_check_reads_api_host():
    """Structural pin: the startup posture check assesses api_host (the API
    bind), not listen_host (the P2P bind)."""
    src = inspect.getsource(PRSMNode.start)
    assert "api_host" in src
    assert "assess_public_bind_auth_posture(" in src


def test_save_roundtrips_api_host(tmp_path):
    cfg = NodeConfig(data_dir=str(tmp_path), api_host="0.0.0.0")
    cfg.save()
    import json
    saved = json.loads((tmp_path / "node_config.json").read_text())
    assert saved.get("api_host") == "0.0.0.0"
