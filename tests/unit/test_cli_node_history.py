"""prsm node slash-history / heartbeats / distributions CLI."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node


@pytest.fixture
def runner():
    return CliRunner()


def _ok(payload):
    r = MagicMock()
    r.status_code = 200
    r.json = MagicMock(return_value=payload)
    return r


class TestSlashHistory:
    def test_renders_entries(self, runner):
        payload = {
            "entries": [
                {
                    "timestamp": 1700000000.0,
                    "kind": "proof_failure_slashed",
                    "provider": "0xPROV",
                    "challenger": "0xCHAL",
                    "slash_id": "0x" + "ab" * 32,
                    "extras": {},
                },
            ],
            "total": 1,
        }
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(node, ["slash-history"])
        assert result.exit_code == 0
        assert "Slash Events" in result.output

    def test_503_friendly_exit_0(self, runner):
        bad = MagicMock()
        bad.status_code = 503
        bad.json = MagicMock(return_value={"detail": "not wired"})
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=bad)
            result = runner.invoke(node, ["slash-history"])
        # 503 = not wired = exit 0 (informational, not error)
        assert result.exit_code == 0
        assert "not configured" in result.output.lower()

    def test_json_format(self, runner):
        payload = {"entries": [], "total": 0}
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(
                node, ["slash-history", "--format", "json"],
            )
        assert result.exit_code == 0
        assert json.loads(result.output) == payload

    def test_node_unreachable_exit_2(self, runner):
        import httpx
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(
                side_effect=httpx.RequestError("conn"),
            )
            result = runner.invoke(node, ["slash-history"])
        assert result.exit_code == 2

    def test_limit_passthrough(self, runner):
        payload = {"entries": [], "total": 0}
        captured_url = {}

        def capture_get(url):
            captured_url["url"] = url
            return _ok(payload)

        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(side_effect=capture_get)
            runner.invoke(node, ["slash-history", "--limit", "5"])
        assert "limit=5" in captured_url["url"]


class TestHeartbeats:
    def test_command_registered(self, runner):
        # Clean smoke test: command is callable end-to-end
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(
                return_value=_ok({"entries": [], "total": 0}),
            )
            result = runner.invoke(node, ["heartbeats"])
        assert result.exit_code == 0
        assert "Heartbeats" in result.output


class TestDistributions:
    def test_command_registered(self, runner):
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(
                return_value=_ok({"entries": [], "total": 0}),
            )
            result = runner.invoke(node, ["distributions"])
        assert result.exit_code == 0
        assert "Distributions" in result.output


class TestWebhooks:
    def test_command_registered(self, runner):
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(
                return_value=_ok({"entries": [], "total": 0}),
            )
            result = runner.invoke(node, ["webhooks"])
        assert result.exit_code == 0
        assert "Webhooks" in result.output


class TestConsensusMismatch:
    """sp957 — `prsm node consensus-mismatch list` wraps
    GET /admin/consensus-mismatch-evidence (read-only operator triage)."""

    def test_renders_entries(self, runner):
        payload = {
            "entries": [
                {
                    "timestamp": 1700000000.0,
                    "job_id": "job-99",
                    "accused_provider_id": "0xPROV",
                    "accused_output_hash": "0xBAD",
                    "majority_output_hash": "0xGOOD",
                    "witness_provider_ids": ["w1", "w2"],
                    "accused_bonded": True,
                    "accused_stake_wei": 5_000_000_000_000_000_000,
                    "accused_operator_address": "0xOP",
                },
            ],
            "total": 1,
        }
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(node, ["consensus-mismatch", "list"])
        assert result.exit_code == 0
        assert "Consensus Mismatch" in result.output
        assert "job-99" in result.output

    def test_503_friendly_exit_0(self, runner):
        bad = MagicMock()
        bad.status_code = 503
        bad.json = MagicMock(return_value={"detail": "not wired"})
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=bad)
            result = runner.invoke(node, ["consensus-mismatch", "list"])
        assert result.exit_code == 0
        assert "not configured" in result.output.lower()

    def test_json_format(self, runner):
        payload = {"entries": [], "total": 0}
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(
                node, ["consensus-mismatch", "list", "--format", "json"],
            )
        assert result.exit_code == 0
        assert json.loads(result.output) == payload

    def test_url_targets_evidence_endpoint(self, runner):
        payload = {"entries": [], "total": 0}
        captured = {}

        def capture_get(url):
            captured["url"] = url
            return _ok(payload)

        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(side_effect=capture_get)
            runner.invoke(
                node, ["consensus-mismatch", "list", "--limit", "5"])
        assert "/admin/consensus-mismatch-evidence" in captured["url"]
        assert "limit=5" in captured["url"]

    def test_provider_filter_passthrough(self, runner):
        payload = {"entries": [], "total": 0}
        captured = {}

        def capture_get(url):
            captured["url"] = url
            return _ok(payload)

        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(side_effect=capture_get)
            runner.invoke(
                node, ["consensus-mismatch", "list", "--provider", "0xPROV"])
        assert "provider=0xPROV" in captured["url"]

    def test_summary_renders_excluded_providers(self, runner):
        payload = {
            "threshold": 2, "window_sec": 0.0,
            "providers": [
                {"provider_id": "0xLIAR", "mismatch_count": 3, "excluded": True},
                {"provider_id": "0xONCE", "mismatch_count": 1, "excluded": False},
            ],
        }
        captured = {}

        def capture_get(url):
            captured["url"] = url
            return _ok(payload)

        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(side_effect=capture_get)
            result = runner.invoke(node, ["consensus-mismatch", "summary"])
        assert result.exit_code == 0
        assert "/admin/consensus-mismatch-evidence/summary" in captured["url"]
        assert "0xLIAR" in result.output
        assert "EXCLUDED" in result.output.upper()

    def test_summary_503_friendly_exit_0(self, runner):
        bad = MagicMock()
        bad.status_code = 503
        bad.json = MagicMock(return_value={"detail": "not wired"})
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=bad)
            result = runner.invoke(node, ["consensus-mismatch", "summary"])
        assert result.exit_code == 0
        assert "not" in result.output.lower()

    def test_summary_json_format(self, runner):
        payload = {"threshold": 2, "window_sec": 0.0, "providers": []}
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(
                node, ["consensus-mismatch", "summary", "--format", "json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == payload
