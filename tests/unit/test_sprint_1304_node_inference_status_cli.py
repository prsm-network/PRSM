"""Sprint 1304 — `prsm node inference-status` CLI capstone.

Operator-facing one-command answer to "is real inference ready, and is the model
actually loaded yet?" Reads the daemon readiness probe (/readyz, which sp1303
enriched with inference_detail) and renders it. Exit codes:
  0 — ready to serve inference
  1 — NOT ready (no executor / inference disabled)
  2 — daemon unreachable at /readyz

Closes the last default-user-inference micro-gap (sp1302 pre-warm + sp1303 readiness
surface) with an operator command, instead of "curl /readyz | jq".
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _invoke(args=None):
    from click.testing import CliRunner
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["inference-status"] + (args or []))


def _resp(status, body):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    return r


def _route(resp):
    def _get(url, *a, **k):
        if isinstance(resp, Exception):
            raise resp
        return resp
    return _get


def test_command_registered():
    from prsm.cli import node as _node_group
    assert "inference-status" in [c.name for c in _node_group.commands.values()]


def test_ready_local_loaded_exit_0():
    body = {"ready": True, "subsystems": {"inference": True},
            "inference_detail": {"enabled": True, "kind": "local",
                                 "model_id": "distilgpt2", "loaded": True,
                                 "device": "cpu", "offline": False,
                                 "max_tokens": 32}}
    with patch("httpx.get", side_effect=_route(_resp(200, body))):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "ready" in out and "distilgpt2" in out
    assert "loaded" in out and "true" in out
    assert "cpu" in out


def test_ready_local_warming_exit_0_shows_prewarm_hint():
    body = {"ready": True, "subsystems": {"inference": True},
            "inference_detail": {"enabled": True, "kind": "local",
                                 "model_id": "distilgpt2", "loaded": False,
                                 "device": None, "offline": False,
                                 "max_tokens": 32}}
    with patch("httpx.get", side_effect=_route(_resp(200, body))):
        result = _invoke()
    assert result.exit_code == 0, result.output
    assert "pre-warming" in result.output.lower()


def test_not_ready_exit_1_shows_reason():
    body = {"ready": False, "subsystems": {"inference": False},
            "reason": "inference_executor_unavailable — install the .[ml] extra",
            "inference_detail": {"enabled": False, "kind": None}}
    with patch("httpx.get", side_effect=_route(_resp(503, body))):
        result = _invoke()
    assert result.exit_code == 1, result.output
    out = result.output.lower()
    assert "not ready" in out
    assert ".[ml]" in result.output


def test_non_local_executor_ready():
    body = {"ready": True, "subsystems": {"inference": True},
            "inference_detail": {"enabled": False, "kind": None}}
    with patch("httpx.get", side_effect=_route(_resp(200, body))):
        result = _invoke()
    assert result.exit_code == 0, result.output
    assert "ready" in result.output.lower()


def test_daemon_unreachable_exit_2():
    with patch("httpx.get", side_effect=_route(RuntimeError("connection refused"))):
        result = _invoke()
    assert result.exit_code == 2, result.output
    assert "unreachable" in result.output.lower()


def test_json_format_passes_through_detail():
    import json
    body = {"ready": True, "subsystems": {"inference": True},
            "inference_detail": {"enabled": True, "kind": "local",
                                 "model_id": "gpt2", "loaded": True}}
    with patch("httpx.get", side_effect=_route(_resp(200, body))):
        result = _invoke(["--format", "json"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["ready"] is True
    assert parsed["inference_detail"]["model_id"] == "gpt2"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
