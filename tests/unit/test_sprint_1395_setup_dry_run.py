"""Sprint 1395 — `setup --dry-run` is a non-interactive preview (F6).

Pre-fix, when PRSM was already configured, --dry-run still fired the interactive "Re-run setup
wizard? [y/N]" prompt instead of previewing. Now dry-run (and --minimal) skip it and run the steps
non-interactively.
"""
from unittest.mock import MagicMock

import prsm.cli_modules.setup_wizard as sw

_STEPS = ("_step_welcome", "_step_role", "_step_resources", "_step_api_keys",
          "_step_network", "_step_ai_integration", "_step_review")


def _wire(monkeypatch, calls):
    monkeypatch.setattr(sw, "prompt_confirm", lambda *a, **k: calls.append(a) or False)
    cfg = MagicMock()
    cfg.setup_completed = True
    fake_cls = MagicMock()
    fake_cls.exists.return_value = True          # already configured
    fake_cls.load.return_value = cfg
    fake_cls.return_value = MagicMock()          # PRSMConfig() -> fresh config
    monkeypatch.setattr(sw, "PRSMConfig", fake_cls)
    monkeypatch.setattr(sw, "_detect_system", lambda: MagicMock())
    for step in _STEPS:
        monkeypatch.setattr(sw, step, lambda *a, **k: None)


def test_dry_run_skips_rerun_prompt(monkeypatch):
    calls = []
    _wire(monkeypatch, calls)
    sw.run_setup_wizard(dry_run=True)
    assert not any("Re-run" in str(a) for a in calls)   # no interactive gate in dry-run


def test_minimal_skips_rerun_prompt(monkeypatch):
    calls = []
    _wire(monkeypatch, calls)
    sw.run_setup_wizard(minimal=True)
    assert not any("Re-run" in str(a) for a in calls)


def test_interactive_still_prompts(monkeypatch):
    calls = []
    _wire(monkeypatch, calls)
    sw.run_setup_wizard()                                # plain interactive
    assert any("Re-run" in str(a) for a in calls)       # the gate still fires


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


def test_ftns_history_no_login_points_to_alternatives(monkeypatch):
    """sp1395 (F11) — `ftns history` without login used to bare-fail "log in"; now it explains why
    (full history/search needs login) and points to the login-free `ftns balance` / `--onchain`."""
    from click.testing import CliRunner
    monkeypatch.setattr("prsm.cli._auth_headers", lambda: None)
    from prsm.cli import ftns as _ftns_group
    r = CliRunner().invoke(_ftns_group, ["history"])
    assert r.exit_code == 1
    assert "prsm ftns balance" in r.output
    assert "--onchain" in r.output
