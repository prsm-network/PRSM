"""Sprint 1391 — `prsm compute submit`/`status`/`result` actually call the daemon.

Pre-1391 they used a local jobs_store and never contacted the node (submit wrote a pending record
nothing processed). Now submit POSTs /compute/submit and status/result GET /compute/job/{id}. httpx
is mocked so no node is needed.
"""
from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner


def _invoke(args, tmp_path, monkeypatch):
    monkeypatch.setattr("prsm.cli._api_url_from_creds", lambda o: "http://x")
    monkeypatch.setattr("prsm.cli._node_api_key_headers", lambda: {})
    import prsm.compute.jobs_store as js
    monkeypatch.setattr(js, "JOBS_FILE", tmp_path / "jobs.json")
    from prsm.cli import compute as _compute_group
    return CliRunner().invoke(_compute_group, args)


def _resp(status_code, body):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = body
    r.headers = {"content-type": "application/json"}
    return r


def test_submit_posts_to_daemon(tmp_path, monkeypatch):
    with patch("httpx.post", return_value=_resp(200, {"job_id": "abc123", "status": "pending"})) as mp:
        r = _invoke(["submit", "--prompt", "hi", "--budget", "1.0"], tmp_path, monkeypatch)
    assert r.exit_code == 0, r.output
    called_url = mp.call_args.args[0] if mp.call_args.args else mp.call_args.kwargs.get("url", "")
    assert called_url.endswith("/compute/submit")
    body = mp.call_args.kwargs.get("json") or {}
    assert body["job_type"] == "inference"
    assert body["payload"]["prompt"] == "hi"
    assert body["ftns_budget"] == 1.0
    assert "abc123" in r.output                       # daemon's job_id surfaced


def test_status_queries_daemon(tmp_path, monkeypatch):
    daemon = {"job_id": "abc123", "status": "completed",
              "result": {"response": " Paris", "model": "distilgpt2", "prompt": "hi"}, "error": None}
    with patch("httpx.get", return_value=_resp(200, daemon)):
        r = _invoke(["status", "abc123"], tmp_path, monkeypatch)
    assert r.exit_code == 0, r.output
    assert "completed" in r.output and "Paris" in r.output


def test_result_queries_daemon(tmp_path, monkeypatch):
    daemon = {"job_id": "abc123", "status": "completed",
              "result": {"response": " Paris"}, "error": None}
    with patch("httpx.get", return_value=_resp(200, daemon)):
        r = _invoke(["result", "abc123"], tmp_path, monkeypatch)
    assert r.exit_code == 0 and "Paris" in r.output


def test_submit_connect_error_exits_2(tmp_path, monkeypatch):
    with patch("httpx.post", side_effect=httpx.ConnectError("no conn")):
        r = _invoke(["submit", "--prompt", "hi"], tmp_path, monkeypatch)
    assert r.exit_code == 2 and "Cannot connect" in r.output


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
