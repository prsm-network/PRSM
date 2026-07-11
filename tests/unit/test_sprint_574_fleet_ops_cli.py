"""Sprint 574 — fleet-ops CLI quartet.

Multi-host bench (sprints 564-573) made cross-host content xfer
work end-to-end + auto-dial. Operator-facing gap remaining:
inspecting fleet state + moving content between operators requires
hand-crafted curl/python invocations. Sprint 574 closes the gap
with four CLI commands wrapping the daemon's HTTP API:

  prsm node peers            (enhanced — shows connected + known)
  prsm node dial <addr>      (wraps POST /peers/connect)
  prsm node fetch <cid>      (wraps GET /content/retrieve)
  prsm node share <file>     (wraps POST /content/upload)

Invariants:
- Each command queries the LOCAL daemon (127.0.0.1:api_port);
  no remote-API guesswork.
- Each surfaces actionable error text when the daemon is down
  (not a stack trace).
- `fetch` base64-decodes the response into bytes + writes to
  stdout (or to --output) — no operator-facing base64 gymnastics.
- `share` accepts a file path (or `-` for stdin) and prints the
  CID back so operators can compose share + dial workflows.
- `dial` accepts host:port or ws://host:port (the same forms the
  endpoint already accepts per sprint-569 invariant).
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner


def _runner():
    return CliRunner()


# ── prsm node peers (enhanced) ───────────────────────────────────


def test_peers_shows_known_but_unconnected(prsm_home_with_identity):
    """Enhanced `prsm node peers` displays known-but-unconnected
    peers in addition to connected ones — operators can see the
    gap between bootstrap-discovered and actually-connected.

    sp1418 — takes prsm_home_with_identity: the command loads the node
    identity, so without an isolated $HOME this silently read the
    developer's real ~/.prsm and failed on a clean CI runner.
    """
    from prsm.cli import node

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "connected": [
            {
                "peer_id": "a" * 32,
                "address": "1.1.1.1:9001",
                "display_name": "alice",
                "outbound": True,
            },
        ],
        "known": [
            {
                "node_id": "b" * 32,
                "address": "2.2.2.2:9001",
                "display_name": "bob",
                "capabilities": ["compute"],
            },
            {
                "node_id": "a" * 32,  # also in connected — should not duplicate
                "address": "1.1.1.1:9001",
                "display_name": "alice",
                "capabilities": [],
            },
        ],
        "connected_count": 1,
        "known_count": 2,
    }
    with patch("httpx.get", return_value=fake_resp):
        result = _runner().invoke(node, ["peers"])
    assert result.exit_code == 0, result.output
    assert "alice" in result.output  # connected entry rendered
    assert "bob" in result.output    # known-only entry rendered
    # `bob` is known-only — should appear in the known-only table
    # somewhere distinct from the connected-table section.


# ── prsm node dial ───────────────────────────────────────────────


def test_dial_calls_peers_connect_with_address():
    """`prsm node dial 1.2.3.4:9001` POSTs the address to
    /peers/connect on the local daemon.
    """
    from prsm.cli import node

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "connected": True,
        "peer_id": "remote-peer",
        "address": "1.2.3.4:9001",
    }
    with patch("httpx.post", return_value=fake_resp) as mock_post:
        result = _runner().invoke(node, ["dial", "1.2.3.4:9001"])
    assert result.exit_code == 0, result.output
    call_kwargs = mock_post.call_args
    # URL is /peers/connect on the local daemon
    url = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("url", "")
    assert "/peers/connect" in url
    # Body carries the address
    body = call_kwargs.kwargs.get("json", {})
    assert body.get("address") == "1.2.3.4:9001"
    # Output surfaces the peer_id
    assert "remote-peer" in result.output


def test_dial_502_surfaces_actionable_error():
    """When /peers/connect returns 502 (transport.connect returned
    None), the CLI prints a clear "could not connect" message
    rather than crashing.
    """
    from prsm.cli import node

    fake_resp = MagicMock()
    fake_resp.status_code = 502
    fake_resp.json.return_value = {
        "detail": "transport.connect_to_peer returned None for '1.2.3.4:9001'",
    }
    fake_resp.text = "transport.connect_to_peer returned None"
    with patch("httpx.post", return_value=fake_resp):
        result = _runner().invoke(node, ["dial", "1.2.3.4:9001"])
    # Don't crash; print an error
    assert result.exit_code != 0
    assert (
        "could not connect" in result.output.lower()
        or "failed" in result.output.lower()
        or "502" in result.output
    )


# ── prsm node fetch ──────────────────────────────────────────────


def test_fetch_decodes_base64_to_stdout():
    """`prsm node fetch <cid>` GETs /content/retrieve, base64-decodes
    the data, and writes to stdout by default.
    """
    from prsm.cli import node
    import base64

    payload = b"hello world from fetch"
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "cid": "abc123",
        "status": "success",
        "data": base64.b64encode(payload).decode(),
        "size_bytes": len(payload),
        "filename": "doc.txt",
        "content_hash": "sha256-fake",
    }
    with patch("httpx.get", return_value=fake_resp):
        result = _runner().invoke(node, ["fetch", "abc123"])
    assert result.exit_code == 0, result.output
    assert "hello world from fetch" in result.output


def test_fetch_not_found_surfaces_clean_error():
    from prsm.cli import node

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "cid": "missing",
        "status": "not_found",
        "error": "Content not found on any available provider",
    }
    with patch("httpx.get", return_value=fake_resp):
        result = _runner().invoke(node, ["fetch", "missing"])
    assert result.exit_code != 0
    assert "not_found" in result.output or "not found" in result.output.lower()


# ── prsm node share ──────────────────────────────────────────────


def test_share_posts_file_content_and_prints_cid(tmp_path):
    """`prsm node share <file>` reads the file, POSTs its text to
    /content/upload, and prints the returned CID.
    """
    from prsm.cli import node

    f = tmp_path / "to_share.txt"
    f.write_text("payload to share")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "cid": "deadbeef" * 5,
        "filename": "to_share.txt",
        "size_bytes": 16,
        "content_hash": "sha256-fake",
        "creator_id": "self-node",
    }
    with patch("httpx.post", return_value=fake_resp) as mock_post:
        result = _runner().invoke(node, ["share", str(f)])
    assert result.exit_code == 0, result.output
    body = mock_post.call_args.kwargs.get("json", {})
    assert body.get("text") == "payload to share"
    # CID printed for downstream pipelining
    assert "deadbeef" in result.output


# ── PRSM_API_PORT env override ───────────────────────────────────


def test_dial_honors_prsm_api_port_env_var():
    """When PRSM_API_PORT=8002 is set, dial targets 127.0.0.1:8002
    instead of the NodeConfig default. Required for operators
    running daemons on non-default ports (e.g., bootstrap-us
    droplet co-located with bootstrap-server-v2).
    """
    import os
    from prsm.cli import node

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "connected": True, "peer_id": "x", "address": "y",
    }
    with patch.dict(os.environ, {"PRSM_API_PORT": "8002"}, clear=False):
        with patch("httpx.post", return_value=fake_resp) as mock_post:
            result = _runner().invoke(node, ["dial", "1.2.3.4:9001"])
    assert result.exit_code == 0, result.output
    url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url", "")
    assert "127.0.0.1:8002" in url, (
        f"PRSM_API_PORT=8002 must override; got {url!r}"
    )


# ── daemon-down behavior ─────────────────────────────────────────


def test_dial_when_daemon_down_prints_friendly_error():
    """When httpx raises (daemon not running), `prsm node dial`
    must NOT show a Python stack trace — only an actionable hint.
    """
    from prsm.cli import node
    import httpx as _httpx

    with patch(
        "httpx.post",
        side_effect=_httpx.ConnectError("Connection refused"),
    ):
        result = _runner().invoke(node, ["dial", "1.2.3.4:9001"])
    assert result.exit_code != 0
    # No raw exception class
    assert "Traceback" not in result.output
    # Actionable hint
    assert (
        "node start" in result.output.lower()
        or "daemon" in result.output.lower()
        or "not running" in result.output.lower()
    )
