"""Sprint 390 — `prsm bootstrap-server status` CLI subcommand.

Operator-trifecta third corner. Renders the
BootstrapServerProbe.to_dict() shape as a clean ops
summary table. Defaults to localhost:8000 (the canonical
api_port from BootstrapConfig). --host + --port overrides
for remote probes. --format json for ops automation.
"""
from __future__ import annotations

import json

# sp1418 — parse .stdout (the DATA channel), not .output. In click >=8.2 Result.output is
# stdout+stderr MIXED (what a terminal shows), so ANY advisory on stderr corrupts a JSON
# parse. The CLI's first-run 'not configured' nudge now correctly goes to stderr; these
# assertions must therefore read stdout, which is what a real `| jq` consumer receives.
from unittest.mock import patch, AsyncMock

import pytest
from click.testing import CliRunner

from prsm.cli import main as cli
from prsm.cli_helpers.bootstrap_server_probe import (
    BootstrapServerProbe,
    ProbeStatus,
)


@pytest.fixture
def runner():
    return CliRunner()


def _ok_probe() -> BootstrapServerProbe:
    return BootstrapServerProbe(
        host="bootstrap-eu.prsm-network.com", port=8000,
        status=ProbeStatus.OK,
        health={
            "healthy": True,
            "uptime_seconds": 3600.0,
            "active_connections": 5,
        },
        metrics={
            "active_connections": 5,
            "total_connections": 100,
            "rejected_connections": 2,
            "total_peers_served": 42,
            "messages_processed": 1500,
            "errors_count": 0,
            "uptime_seconds": 3600.0,
            "peers_by_region": {
                "us-east-1": 2, "eu-west-1": 3,
            },
        },
    )


def _refused_probe() -> BootstrapServerProbe:
    return BootstrapServerProbe(
        host="127.0.0.1", port=8000,
        status=ProbeStatus.CONNECT_FAIL,
        error="connection error: connection refused",
    )


# ── Healthy path ─────────────────────────────────────────


def test_status_renders_summary_on_healthy(runner):
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=_ok_probe()),
    ):
        result = runner.invoke(
            cli,
            ["bootstrap-server", "status",
             "--host", "bootstrap-eu.prsm-network.com"],
        )
    assert result.exit_code == 0, result.output
    # Status marker present
    out = result.output
    assert "healthy" in out.lower() or "ok" in out.lower()
    # Key metrics rendered
    assert "active_connections" in out or "Active" in out
    # Host visible
    assert "bootstrap-eu.prsm-network.com" in out


def test_status_json_output_shape(runner):
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=_ok_probe()),
    ):
        result = runner.invoke(
            cli,
            ["bootstrap-server", "status",
             "--host", "x", "--format", "json"],
        )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["health"]["healthy"] is True
    assert payload["metrics"]["total_connections"] == 100


# ── Unreachable path ─────────────────────────────────────


def test_status_connect_refused_exit_code_nonzero(runner):
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=_refused_probe()),
    ):
        result = runner.invoke(
            cli,
            ["bootstrap-server", "status"],
        )
    # Exit non-zero so ops tooling can detect the failure
    assert result.exit_code != 0
    assert (
        "connect" in result.output.lower()
        or "refused" in result.output.lower()
    )


def test_status_connect_refused_json_still_renders(runner):
    """Even failure paths emit valid json under --format json
    so wrappers can parse the error programmatically."""
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=_refused_probe()),
    ):
        result = runner.invoke(
            cli,
            ["bootstrap-server", "status",
             "--format", "json"],
        )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "connect_fail"
    assert payload["error"]


# ── Defaults ─────────────────────────────────────────────


def test_status_defaults_to_localhost_8000(runner):
    captured = {}

    async def fake_probe(host, port, **kw):
        captured["host"] = host
        captured["port"] = port
        return _ok_probe()

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        result = runner.invoke(
            cli, ["bootstrap-server", "status"],
        )
    assert result.exit_code == 0, result.output
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000


def test_status_host_and_port_override(runner):
    captured = {}

    async def fake_probe(host, port, **kw):
        captured["host"] = host
        captured["port"] = port
        return _ok_probe()

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        result = runner.invoke(
            cli,
            ["bootstrap-server", "status",
             "--host", "remote.example", "--port", "9000"],
        )
    assert result.exit_code == 0, result.output
    assert captured["host"] == "remote.example"
    assert captured["port"] == 9000


def test_status_timeout_param_propagated(runner):
    captured = {}

    async def fake_probe(host, port, *, timeout_seconds, **kw):
        captured["timeout"] = timeout_seconds
        return _ok_probe()

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        result = runner.invoke(
            cli,
            ["bootstrap-server", "status", "--timeout", "20"],
        )
    assert result.exit_code == 0, result.output
    assert captured["timeout"] == 20.0


# ── Partial-state rendering ──────────────────────────────


def test_status_detailed_flag_propagates_include_subsystems(runner):
    """Sprint 393 — `--detailed` flag flips
    include_subsystems=True on the probe call."""
    captured = {}

    async def fake_probe(host, port, *, timeout_seconds, include_subsystems=False, **kw):
        captured["include_subsystems"] = include_subsystems
        return _ok_probe()

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        result = runner.invoke(
            cli,
            ["bootstrap-server", "status", "--detailed"],
        )
    assert result.exit_code == 0, result.output
    assert captured["include_subsystems"] is True


def test_status_default_does_not_pass_include_subsystems_true(runner):
    captured = {}

    async def fake_probe(host, port, *, timeout_seconds, include_subsystems=False, **kw):
        captured["include_subsystems"] = include_subsystems
        return _ok_probe()

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        result = runner.invoke(
            cli, ["bootstrap-server", "status"],
        )
    assert result.exit_code == 0, result.output
    assert captured["include_subsystems"] is False


def test_status_detailed_renders_subsystems_table(runner):
    probe = BootstrapServerProbe(
        host="x", port=8000, status=ProbeStatus.OK,
        health={"healthy": True, "uptime_seconds": 1.0},
        metrics={"total_connections": 1},
        health_detailed={
            "status": "degraded",
            "subsystems": {
                "peer_cleanup": {
                    "alive": True, "status": "healthy",
                    "last_heartbeat_age_seconds": 1.0,
                    "expected_interval_seconds": 60.0,
                },
                "peer_backup": {
                    "alive": True, "status": "degraded",
                    "last_heartbeat_age_seconds": 700.0,
                    "expected_interval_seconds": 300.0,
                },
            },
        },
    )
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=probe),
    ):
        result = runner.invoke(
            cli,
            ["bootstrap-server", "status", "--detailed"],
        )
    assert result.exit_code == 0, result.output
    assert "peer_cleanup" in result.output
    assert "peer_backup" in result.output
    assert "degraded" in result.output.lower()


def test_status_partial_marker_when_metrics_unavailable(runner):
    partial = BootstrapServerProbe(
        host="x", port=8000, status=ProbeStatus.PARTIAL,
        health={"healthy": True, "uptime_seconds": 1.0},
        metrics=None,
        error="/metrics fetch failed: timeout",
    )
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=partial),
    ):
        result = runner.invoke(
            cli, ["bootstrap-server", "status"],
        )
    # PARTIAL → exit non-zero so monitors flag the
    # degraded-observability case, but health is still
    # surfaced for liveness verification
    assert result.exit_code != 0
    assert (
        "partial" in result.output.lower()
        or "degraded" in result.output.lower()
    )
    assert "metrics" in result.output.lower()
