"""Sprint 805 — `prsm content fetch <cid>` user-facing retrieval CLI.

Pre-805 the content surface had `prsm content mine` (list MY
uploads) but no command to RETRIEVE content by CID. Users had
to curl /content/retrieve/{cid} themselves + base64-decode the
data + write to a file.

Sprint 805 ships:

  prsm content fetch <cid>
                    [--output PATH]
                    [--timeout 30.0]
                    [--no-verify-hash]
                    [--api-url <url>]
                    [--format text|json]

Wraps GET /content/retrieve/{cid}. base64-decodes data, writes
to `--output` (or stdout binary if not set + content is small).
Surfaces filename + size_bytes + content_hash + providers_tried.

Exit codes:
  0 — success
  1 — not_found / error from server
  2 — daemon unreachable

Pin tests:
- Command registered under `content` group.
- GET hits /content/retrieve/{cid}.
- Successful response writes decoded data to --output file.
- File contents match the original bytes (round-trip).
- size_bytes + content_hash surfaced in text mode.
- status="not_found" → exit 1 with actionable hint.
- status="error" → exit 1 + error surfaced.
- Unreachable daemon → exit 2.
- JSON mode returns full payload (without --output writes,
  base64 data passes through).
- --timeout flag threaded into query param.
- --no-verify-hash sets verify_hash=false query param.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args):
    from prsm.cli import content as _content_group
    return CliRunner().invoke(
        _content_group, ["fetch"] + list(args),
    )


def _success_response(data_bytes: bytes, **overrides):
    r = MagicMock()
    r.status_code = 200
    body = {
        "cid": "QmTestCid",
        "status": "success",
        "data": base64.b64encode(data_bytes).decode("utf-8"),
        "size_bytes": len(data_bytes),
        "content_hash": "deadbeef",
        "filename": "test.txt",
        "providers_tried": 1,
        "error": None,
    }
    body.update(overrides)
    r.json.return_value = body
    return r


# ---- Command registration ---------------------------------------


def test_fetch_command_registered():
    from prsm.cli import content as _content_group
    cmd_names = [c.name for c in _content_group.commands.values()]
    assert "fetch" in cmd_names


# ---- Happy path: file written --------------------------------


def test_successful_fetch_writes_file(tmp_path: Path):
    payload = b"Hello, PRSM content world!\n"
    target = tmp_path / "fetched.bin"
    with patch(
        "httpx.get", return_value=_success_response(payload),
    ):
        result = _invoke([
            "QmTestCid",
            "--output", str(target),
            "--format", "text",
        ])
    assert result.exit_code == 0, result.output
    assert target.exists()
    assert target.read_bytes() == payload


def test_endpoint_url_uses_cid(tmp_path: Path):
    target = tmp_path / "x.bin"
    with patch(
        "httpx.get", return_value=_success_response(b"x"),
    ) as mg:
        _invoke([
            "QmMyContent42",
            "--output", str(target),
            "--format", "text",
        ])
    url = mg.call_args.args[0] if mg.call_args.args else (
        mg.call_args.kwargs.get("url")
    )
    assert "/content/retrieve/QmMyContent42" in url


def test_size_and_hash_surfaced_text(tmp_path: Path):
    payload = b"x" * 42
    target = tmp_path / "x.bin"
    with patch(
        "httpx.get", return_value=_success_response(payload),
    ):
        result = _invoke([
            "QmTestCid",
            "--output", str(target),
            "--format", "text",
        ])
    assert result.exit_code == 0
    # Both fields visible to the operator
    assert "42" in result.output
    assert "deadbeef" in result.output


# ---- Error paths ----------------------------------------------


def test_not_found_exits_1():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "cid": "QmMissing",
        "status": "not_found",
        "data": None,
        "providers_tried": 3,
        "error": None,
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke(["QmMissing", "--format", "text"])
    assert result.exit_code == 1
    assert "not_found" in result.output.lower() or (
        "not found" in result.output.lower()
    )


def test_error_status_exits_1():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "cid": "QmBad",
        "status": "error",
        "data": None,
        "providers_tried": 1,
        "error": "hash mismatch",
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke(["QmBad", "--format", "text"])
    assert result.exit_code == 1
    assert "hash mismatch" in result.output or (
        "error" in result.output.lower()
    )


def test_unreachable_daemon_exits_2():
    with patch(
        "httpx.get",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke(["QmX", "--format", "text"])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()


# ---- JSON mode -------------------------------------------------


def test_json_mode_returns_full_payload():
    with patch(
        "httpx.get", return_value=_success_response(b"abc"),
    ):
        result = _invoke(["QmTestCid", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "success"
    assert data["size_bytes"] == 3
    # Base64 data passes through unchanged
    assert data["data"] == base64.b64encode(b"abc").decode()


# ---- Query-param flags ---------------------------------------


def test_timeout_passed_as_query_param():
    with patch(
        "httpx.get", return_value=_success_response(b"x"),
    ) as mg:
        _invoke([
            "QmTestCid", "--timeout", "60",
            "--format", "json",
        ])
    params = mg.call_args.kwargs.get("params") or {}
    assert params.get("timeout") in (60, 60.0, "60", "60.0")


def test_no_verify_hash_flag_threaded():
    with patch(
        "httpx.get", return_value=_success_response(b"x"),
    ) as mg:
        _invoke([
            "QmTestCid", "--no-verify-hash",
            "--format", "json",
        ])
    params = mg.call_args.kwargs.get("params") or {}
    # FastAPI accepts 0/false/no for bool query params; we want
    # something that maps to verify_hash=False on the server.
    assert params.get("verify_hash") in (
        False, "false", "False", "0", 0,
    )


# ---- sp1457: --stream (large-content streaming to file) ---------------------

class _FakeStream:
    """A context-manager mock mirroring httpx.Client.stream()'s response object."""
    def __init__(self, status_code=200, chunks=(b"",), body=b""):
        self.status_code = status_code
        self._chunks = list(chunks)
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self):
        for c in self._chunks:
            yield c

    def read(self):
        return self._body


def test_stream_writes_large_body_to_output(tmp_path: Path):
    payload = b"streamed-large-content-" * 100_000  # ~2.3 MB, chunked
    chunks = [payload[i:i + 65536] for i in range(0, len(payload), 65536)]
    target = tmp_path / "big.bin"
    with patch("httpx.stream", return_value=_FakeStream(200, chunks=chunks)) as ms:
        result = _invoke(["QmBig", "--stream", "--output", str(target)])
    assert result.exit_code == 0, result.output
    assert target.read_bytes() == payload
    # hit the STREAMING endpoint, not the JSON one.
    assert ms.call_args.args[1].endswith("/content/retrieve-stream/QmBig")


def test_stream_requires_output():
    with patch("httpx.stream") as ms:
        result = _invoke(["QmBig", "--stream"])
    assert result.exit_code == 1
    ms.assert_not_called()  # rejected before any request


def test_stream_not_found_exit_1(tmp_path: Path):
    target = tmp_path / "x.bin"
    with patch("httpx.stream",
               return_value=_FakeStream(404, body=b'{"detail":"not found"}')):
        result = _invoke(["QmMissing", "--stream", "--output", str(target)])
    assert result.exit_code == 1


def test_stream_daemon_unreachable_exit_2(tmp_path: Path):
    import httpx as _hx
    target = tmp_path / "x.bin"
    with patch("httpx.stream", side_effect=_hx.ConnectError("refused")):
        result = _invoke(["QmX", "--stream", "--output", str(target)])
    assert result.exit_code == 2
