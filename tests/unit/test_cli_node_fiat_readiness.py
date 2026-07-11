"""Sprint 422 — `prsm node fiat-readiness` CLI subcommand.

Operator-facing pre-commission probe for the Phase 5 fiat
surface. Wraps sprint-285's `check_fiat_surface_health()`
function with a color-coded status table + JSON output
mode for ops automation.

Use case: operator runs this BEFORE attempting Phase 5
activation (per `docs/operations/phase-5-fiat-surface-
activation-runbook.md` Step 4) to verify their env is
ready. Same probe core as the runbook's Python one-liner,
just wrapped for CLI discoverability.
"""
from __future__ import annotations

import json

# sp1418 — parse .stdout (the DATA channel), not .output. In click >=8.2 Result.output is
# stdout+stderr MIXED (what a terminal shows), so ANY advisory on stderr corrupts a JSON
# parse. The CLI's first-run 'not configured' nudge now correctly goes to stderr; these
# assertions must therefore read stdout, which is what a real `| jq` consumer receives.
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from prsm.cli import main as cli


@pytest.fixture
def runner():
    return CliRunner()


# ── Healthy path: empty env (un-commissioned, no findings) ─


def test_un_commissioned_env_returns_ok(runner):
    """An empty env (KYC_VENDOR unset) is the
    pre-commission state. `check_fiat_surface_health()`
    returns no findings → CLI reports OK."""
    with patch.dict("os.environ", {}, clear=True):
        result = runner.invoke(
            cli, ["node", "fiat-readiness"],
        )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output or "ready" in result.output.lower()


# ── ERROR findings ───────────────────────────────────────


def test_commissioned_kyc_without_webhook_secret_errors(runner):
    """Sprint-282 KYC commissioned (KYC_VENDOR +
    KYC_VENDOR_API_KEY set) but no webhook secret →
    ERROR finding, exit code non-zero."""
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "test-key",
        # PERSONA_WEBHOOK_SECRET intentionally unset
    }
    with patch.dict("os.environ", env, clear=True):
        result = runner.invoke(
            cli, ["node", "fiat-readiness"],
        )
    assert result.exit_code != 0, result.output
    assert "ERROR" in result.output.upper() or "✗" in result.output
    # The actual cause text from fiat_surface_health surfaces
    assert "webhook" in result.output.lower()


def test_webhook_verify_disabled_in_commissioned_env_errors(runner):
    """Sprint-283 webhook signature verification cannot
    legally be disabled in a commissioned env."""
    env = {
        "KYC_VENDOR": "onfido",
        "KYC_VENDOR_API_KEY": "test-key",
        "ONFIDO_WEBHOOK_TOKEN": "secret",
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED": "1",
    }
    with patch.dict("os.environ", env, clear=True):
        result = runner.invoke(
            cli, ["node", "fiat-readiness"],
        )
    assert result.exit_code != 0
    assert "verify_disabled" in result.output.lower() or (
        "webhook" in result.output.lower()
    )


# ── WARN findings ────────────────────────────────────────


def test_compliance_log_dir_unset_is_warning_only(runner):
    """Sprint-285 audit-retention requirement is WARN
    (not ERROR) — the surface still works, just without
    on-disk audit trail. Exit code stays 0 for WARN-only
    findings; failures are ERROR-only."""
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "test-key",
        "PERSONA_WEBHOOK_SECRET": "secret",
        # PRSM_FIAT_COMPLIANCE_LOG_DIR intentionally unset
    }
    with patch.dict("os.environ", env, clear=True):
        result = runner.invoke(
            cli, ["node", "fiat-readiness"],
        )
    # WARN findings present but no ERROR — exit code 0
    assert result.exit_code == 0, result.output
    assert "WARN" in result.output.upper() or "⚠" in result.output


# ── JSON output ──────────────────────────────────────────


def test_json_output_shape(runner):
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "test-key",
        # Missing webhook secret → ERROR
    }
    with patch.dict("os.environ", env, clear=True):
        result = runner.invoke(
            cli, ["node", "fiat-readiness", "--format", "json"],
        )
    payload = json.loads(result.stdout)
    assert "findings" in payload
    assert "overall_status" in payload
    assert payload["overall_status"] in ("ok", "warn", "error")
    # At least one ERROR finding for this env
    assert any(
        f["severity"].upper() == "ERROR"
        for f in payload["findings"]
    )


def test_json_output_un_commissioned_clean(runner):
    with patch.dict("os.environ", {}, clear=True):
        result = runner.invoke(
            cli, ["node", "fiat-readiness", "--format", "json"],
        )
    payload = json.loads(result.stdout)
    assert payload["overall_status"] == "ok"
    assert payload["findings"] == []


def test_json_failure_still_emits_valid_json(runner):
    """Even when exit-code is non-zero (ERROR findings),
    `--format json` must emit parseable JSON so ops scripts
    can consume the structured output."""
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "test-key",
    }
    with patch.dict("os.environ", env, clear=True):
        result = runner.invoke(
            cli, ["node", "fiat-readiness", "--format", "json"],
        )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)  # must parse
    assert payload["overall_status"] == "error"


# ── Finding rendering ────────────────────────────────────


def test_text_output_includes_remediation_hints(runner):
    """Each finding must surface its remediation hint
    inline — operators triaging via the CLI shouldn't need
    to read the runbook to fix common issues."""
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "test-key",
    }
    with patch.dict("os.environ", env, clear=True):
        result = runner.invoke(
            cli, ["node", "fiat-readiness"],
        )
    # Remediation text from fiat_surface_health.py surfaces
    # (contains "Set " + the var name)
    assert "Set " in result.output or "set " in result.output
    assert "PERSONA_WEBHOOK_SECRET" in result.output


def test_each_finding_has_cause_label(runner):
    env = {
        "KYC_VENDOR": "plaid",
        "KYC_VENDOR_API_KEY": "test-key",
    }
    with patch.dict("os.environ", env, clear=True):
        result = runner.invoke(
            cli, ["node", "fiat-readiness"],
        )
    # Cause string from fiat_surface_health (vendor-named)
    assert "plaid" in result.output.lower()
