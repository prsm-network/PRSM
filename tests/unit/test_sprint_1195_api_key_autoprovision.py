"""Sprint 1195 — self-provision / reuse the node API key for a public bind.

Day-one operator UX: today a public bind with no PRSM_NODE_API_KEY hits the
sp1011 fail-closed wall (refuse to start) with no key + no guidance on making
one. This adds a secure-by-default provisioning path that runs BEFORE the
posture gate:
  - an explicit PRSM_NODE_API_KEY always wins;
  - a public bind reuses a PERSISTED key (stable across restarts so clients keep
    working);
  - opt-in PRSM_AUTO_PROVISION_API_KEY mints + persists (0600) + logs a key;
  - loopback dev posture is untouched, and a public bind with neither still falls
    through to the reviewed fail-closed gate.
"""
from __future__ import annotations

import inspect

from prsm.node.node import (
    PRSMNode,
    decide_api_key_provisioning,
    ensure_public_bind_api_key,
)


# ── decide_api_key_provisioning: pure decision ───────────────────────────────────────

def test_env_key_always_wins():
    action, _ = decide_api_key_provisioning(
        api_host="0.0.0.0", env_key="secret", persisted_key=None, auto_provision=True)
    assert action == "use_env"


def test_loopback_is_none_even_without_key():
    for host in ("127.0.0.1", "::1", "localhost"):
        action, _ = decide_api_key_provisioning(
            api_host=host, env_key="", persisted_key=None, auto_provision=True)
        assert action == "none", host


def test_public_reuses_persisted_key():
    action, _ = decide_api_key_provisioning(
        api_host="0.0.0.0", env_key="", persisted_key="persisted-key",
        auto_provision=False)
    assert action == "use_persisted"


def test_public_generates_only_when_opted_in():
    action, _ = decide_api_key_provisioning(
        api_host="0.0.0.0", env_key="", persisted_key=None, auto_provision=True)
    assert action == "generate"


def test_public_no_key_no_optin_falls_through_to_failclosed():
    action, _ = decide_api_key_provisioning(
        api_host="0.0.0.0", env_key="", persisted_key=None, auto_provision=False)
    assert action == "none"  # the sp1011 gate then refuses startup


# ── ensure_public_bind_api_key: IO behavior (tmp key file + dict env) ────────────────

def test_generate_persists_and_sets_env(tmp_path):
    key_file = tmp_path / "node_api_key"
    env = {"PRSM_AUTO_PROVISION_API_KEY": "1"}
    key = ensure_public_bind_api_key(api_host="0.0.0.0", key_path=key_file, env=env)
    assert key and len(key) > 20
    assert env["PRSM_NODE_API_KEY"] == key
    assert key_file.read_text(encoding="utf-8").strip() == key
    # 0600 owner-only perms
    import stat
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_reuse_persisted_is_stable_across_restart(tmp_path):
    key_file = tmp_path / "node_api_key"
    env1 = {"PRSM_AUTO_PROVISION_API_KEY": "1"}
    first = ensure_public_bind_api_key(api_host="0.0.0.0", key_path=key_file, env=env1)
    # a later "restart": no opt-in this time, but the file exists → reuse it
    env2 = {}
    second = ensure_public_bind_api_key(api_host="0.0.0.0", key_path=key_file, env=env2)
    assert second == first
    assert env2["PRSM_NODE_API_KEY"] == first


def test_public_no_optin_no_file_returns_none(tmp_path):
    key_file = tmp_path / "node_api_key"
    env = {}
    key = ensure_public_bind_api_key(api_host="0.0.0.0", key_path=key_file, env=env)
    assert key is None
    assert "PRSM_NODE_API_KEY" not in env
    assert not key_file.exists()  # nothing written → fail-closed gate applies


def test_loopback_untouched(tmp_path):
    key_file = tmp_path / "node_api_key"
    env = {"PRSM_AUTO_PROVISION_API_KEY": "1"}  # even opted-in, loopback stays dev
    key = ensure_public_bind_api_key(api_host="127.0.0.1", key_path=key_file, env=env)
    assert key is None
    assert "PRSM_NODE_API_KEY" not in env
    assert not key_file.exists()


def test_explicit_env_key_not_overwritten(tmp_path):
    key_file = tmp_path / "node_api_key"
    env = {"PRSM_NODE_API_KEY": "operator-key", "PRSM_AUTO_PROVISION_API_KEY": "1"}
    key = ensure_public_bind_api_key(api_host="0.0.0.0", key_path=key_file, env=env)
    assert key == "operator-key"
    assert not key_file.exists()  # never overwrote / persisted the operator's key


def test_persist_failure_still_authenticates_this_process(tmp_path):
    # key_path points into a non-existent, uncreatable location → write fails, but the
    # in-process env key is still set so the node comes up authenticated this run.
    bad_path = tmp_path / "nope"
    bad_path.write_text("i am a file, not a dir")  # parent of key would be a file
    key_file = bad_path / "node_api_key"
    env = {"PRSM_AUTO_PROVISION_API_KEY": "1"}
    key = ensure_public_bind_api_key(api_host="0.0.0.0", key_path=key_file, env=env)
    assert key and env["PRSM_NODE_API_KEY"] == key  # never raised; authenticated anyway


# ── structural pin: start() provisions BEFORE the posture gate ───────────────────────

def test_start_provisions_before_posture_gate():
    src = inspect.getsource(PRSMNode.start)
    assert "ensure_public_bind_api_key(" in src
    assert src.index("ensure_public_bind_api_key(") < src.index(
        "assess_public_bind_auth_posture("), "must provision before the gate reads the key"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
