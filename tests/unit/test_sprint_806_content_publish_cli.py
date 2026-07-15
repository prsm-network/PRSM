"""Sprint 806 — `prsm content publish FILE` upload CLI.

Closes the content surface: pre-806 we had list (`mine`, sprint
pre-771) + retrieve (`fetch`, sprint 805) but no first-class
publish command. Operators had to curl /content/upload with the
right JSON shape.

  prsm content publish FILE
                      [--filename NAME]
                      [--replicas N]
                      [--royalty-rate R]
                      [--parent-cid CID ...]
                      [--api-url URL]
                      [--format text|json]

Reads FILE bytes, decodes as UTF-8 (the /content/upload endpoint
takes a `text` field), POSTs to /content/upload, renders the
returned CID + size + status. Binary files are out of scope for
this endpoint — use /content/upload/shard for those.

Exit codes:
  0 — uploaded
  1 — file read error / server error / non-UTF-8 input
  2 — daemon unreachable

Pin tests:
- Command registered under `content` group.
- POSTs to /content/upload.
- Body shape matches ContentUploadRequest (text, filename,
  replicas, royalty_rate, parent_cids).
- File text passed verbatim as `text` field.
- --filename overrides the path-derived default.
- --parent-cid (repeatable) collected into a list.
- Happy path text renders CID + filename.
- Non-existent file → exit 1.
- Non-UTF-8 file → exit 1 with clear message (don't crash).
- 503 from server → exit 1.
- Unreachable → exit 2.
- JSON mode returns full server payload.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args):
    from prsm.cli import content as _content_group
    return CliRunner().invoke(
        _content_group, ["publish"] + list(args),
    )


def _success_response(cid="QmNewCid", **overrides):
    r = MagicMock()
    r.status_code = 200
    body = {
        "cid": cid,
        "filename": "test.txt",
        "size_bytes": 42,
        "replicas": 3,
        "status": "uploaded",
    }
    body.update(overrides)
    r.json.return_value = body
    return r


# ---- Command registration --------------------------------------


def test_publish_command_registered():
    from prsm.cli import content as _content_group
    cmd_names = [c.name for c in _content_group.commands.values()]
    assert "publish" in cmd_names


# ---- POSTs to /content/upload + body shape -------------------


def test_post_url_is_content_upload(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    with patch("httpx.post", return_value=_success_response()) as mp:
        result = _invoke([
            str(f), "--format", "json",
        ])
    assert result.exit_code == 0, result.output
    url = mp.call_args.args[0] if mp.call_args.args else (
        mp.call_args.kwargs.get("url")
    )
    assert url.endswith("/content/upload")


def test_body_shape_matches_endpoint(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    with patch("httpx.post", return_value=_success_response()) as mp:
        _invoke([str(f), "--format", "json"])
    body = mp.call_args.kwargs.get("json") or {}
    assert "text" in body
    assert "filename" in body
    assert "replicas" in body
    # royalty_rate is optional — present when --royalty-rate set,
    # absent (or None) when not. parent_cids should be a list.
    assert "parent_cids" in body
    assert isinstance(body["parent_cids"], list)


def test_file_text_passed_verbatim(tmp_path: Path):
    payload = "PRSM rocks\nLine 2\n"
    f = tmp_path / "doc.txt"
    f.write_text(payload)
    with patch("httpx.post", return_value=_success_response()) as mp:
        _invoke([str(f), "--format", "json"])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["text"] == payload


def test_filename_default_from_path(tmp_path: Path):
    f = tmp_path / "subfile.md"
    f.write_text("x")
    with patch("httpx.post", return_value=_success_response()) as mp:
        _invoke([str(f), "--format", "json"])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["filename"] == "subfile.md"


def test_filename_override(tmp_path: Path):
    f = tmp_path / "subfile.md"
    f.write_text("x")
    with patch("httpx.post", return_value=_success_response()) as mp:
        _invoke([
            str(f), "--filename", "rename.md",
            "--format", "json",
        ])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["filename"] == "rename.md"


def test_parent_cid_repeatable(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("x")
    with patch("httpx.post", return_value=_success_response()) as mp:
        _invoke([
            str(f),
            "--parent-cid", "QmA",
            "--parent-cid", "QmB",
            "--format", "json",
        ])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["parent_cids"] == ["QmA", "QmB"]


def test_replicas_and_royalty_threaded(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("x")
    with patch("httpx.post", return_value=_success_response()) as mp:
        _invoke([
            str(f),
            "--replicas", "5",
            "--royalty-rate", "0.05",
            "--format", "json",
        ])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["replicas"] == 5
    assert body["royalty_rate"] == 0.05


# ---- Output -----------------------------------------------------


def test_happy_text_shows_cid(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("hi")
    with patch(
        "httpx.post",
        return_value=_success_response(cid="QmAbc123"),
    ):
        result = _invoke([str(f), "--format", "text"])
    assert result.exit_code == 0
    assert "QmAbc123" in result.output


def test_json_mode_returns_full_payload(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("hi")
    with patch(
        "httpx.post",
        return_value=_success_response(cid="QmJson"),
    ):
        result = _invoke([str(f), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["cid"] == "QmJson"
    assert data["status"] == "uploaded"


# ---- Error paths ----------------------------------------------


def test_missing_file_exits_1(tmp_path: Path):
    nonexistent = tmp_path / "nope.txt"
    result = _invoke([str(nonexistent), "--format", "text"])
    assert result.exit_code != 0
    assert "nope.txt" in result.output


def test_non_utf8_file_exits_1(tmp_path: Path):
    f = tmp_path / "binary.bin"
    f.write_bytes(b"\xff\xfe\x00\x01garbage")
    result = _invoke([str(f), "--format", "text"])
    assert result.exit_code == 1
    out = result.output.lower()
    assert "utf-8" in out or "decode" in out


def test_server_503_exits_1(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("x")
    fake = MagicMock()
    fake.status_code = 503
    fake.text = '{"detail":"ContentPublisher not wired"}'
    with patch("httpx.post", return_value=fake):
        result = _invoke([str(f), "--format", "text"])
    assert result.exit_code == 1
    assert "503" in result.output or (
        "ContentPublisher" in result.output
    )


def test_unreachable_exits_2(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("x")
    with patch(
        "httpx.post",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke([str(f), "--format", "text"])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()


# ---- sp1457: --stream (large/binary file → /content/upload-stream) ---------

def test_publish_stream_hits_upload_stream_endpoint(tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(b"large binary dataset " * 100_000)
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"cid": "QmBig", "content_hash": "abc",
                           "size_bytes": f.stat().st_size, "status": "published"}
    with patch("httpx.post", return_value=r) as mp:
        result = _invoke([str(f), "--stream"])
    assert result.exit_code == 0, result.output
    assert mp.call_args.args[0].endswith("/content/upload-stream")


def test_publish_stream_server_error_exit_1(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"data")
    r = MagicMock()
    r.status_code = 500
    r.text = "boom"
    with patch("httpx.post", return_value=r):
        result = _invoke([str(f), "--stream"])
    assert result.exit_code == 1


def test_publish_stream_unreachable_exit_2(tmp_path):
    import httpx as _hx
    f = tmp_path / "x.bin"
    f.write_bytes(b"data")
    with patch("httpx.post", side_effect=_hx.ConnectError("refused")):
        result = _invoke([str(f), "--stream"])
    assert result.exit_code == 2
