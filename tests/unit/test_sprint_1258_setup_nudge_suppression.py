"""Sprint 1258 — suppress the "PRSM is not configured yet. Run: prsm setup" nudge for
users who are actually configured.

Found by the live front-door validation: the nudge fired on EVERY working command
(faucet/deposit/pay-infer/infer) because it keyed solely on ~/.prsm/config.yaml, which
env-driven + SSH-driven node runs never write — undermining user confidence even though
everything worked. The nudge now suppresses once the user is past the brand-new state:
a node identity (~/.prsm/identity.json) or explicit PRSM_*/wallet env. Genuinely-fresh
users still get nudged.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from prsm.cli import _prsm_appears_configured, main

_ENV_MARKERS = (
    "PRSM_NETWORK", "PRSM_INFERENCE_EXECUTOR", "FTNS_WALLET_PRIVATE_KEY",
    "PRIVATE_KEY", "PRSM_NODE_API_KEY", "PRSM_OPERATOR_ADDRESS",
)


@pytest.fixture
def fresh_home(monkeypatch, tmp_path):
    """A pristine HOME (no ~/.prsm) with all PRSM markers cleared."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for v in _ENV_MARKERS:
        monkeypatch.delenv(v, raising=False)
    return tmp_path


def test_genuinely_fresh_user_is_unconfigured(fresh_home):
    assert _prsm_appears_configured() is False


def test_config_yaml_marks_configured(fresh_home):
    (fresh_home / ".prsm").mkdir()
    (fresh_home / ".prsm" / "config.yaml").write_text("x: 1")
    assert _prsm_appears_configured() is True


def test_node_identity_marks_configured(fresh_home):
    # a node that has run (env-driven, no config.yaml) has an identity → not fresh.
    (fresh_home / ".prsm").mkdir()
    (fresh_home / ".prsm" / "identity.json").write_text('{"node_id":"x"}')
    assert _prsm_appears_configured() is True


@pytest.mark.parametrize("var", _ENV_MARKERS)
def test_env_marker_marks_configured(fresh_home, monkeypatch, var):
    monkeypatch.setenv(var, "something")
    assert _prsm_appears_configured() is True


def test_blank_env_marker_does_not_count(fresh_home, monkeypatch):
    monkeypatch.setenv("PRSM_NETWORK", "   ")   # whitespace-only must not count
    assert _prsm_appears_configured() is False


# ── end-to-end via the CLI group ─────────────────────────────────────────────

def test_nudge_shown_for_fresh_user(fresh_home):
    res = CliRunner().invoke(main, ["version"])
    assert "not configured yet" in res.output


def test_nudge_suppressed_when_configured_via_env(fresh_home, monkeypatch):
    monkeypatch.setenv("PRSM_NETWORK", "testnet")
    res = CliRunner().invoke(main, ["version"])
    assert "not configured yet" not in res.output


def test_nudge_still_skipped_on_setup_and_bare(fresh_home):
    # bare invocation (help) never nudges, configured or not.
    assert "not configured yet" not in CliRunner().invoke(main, []).output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
