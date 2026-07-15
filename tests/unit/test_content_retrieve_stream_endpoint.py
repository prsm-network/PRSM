"""Sprint 1457 — GET /content/retrieve-stream/{cid}: the streaming retrieval endpoint.

Closes the sp1290 receive-side wiring gap. sp1290 built the >64 MiB streaming SEND
(_send_chunked_from_path, auto-routed for a large staged file) AND the streaming RECEIVE
(request_content_to_file → _StreamingSink, 2 GiB ceiling), but nothing in a live retrieval
flow called request_content_to_file — the JSON /content/retrieve buffers in memory and
rejects a response above the in-memory chunked ceiling. So large content could be SERVED
but not FETCHED. This endpoint fetches to a temp file via request_content_to_file and
streams it back as a FileResponse (CID-verified, temp removed after the response).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app

_BODY = b"a large streamed content payload " * 4096  # ~128 KiB — the size is irrelevant to wiring


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.content_index = None
    node._content_filter_store = None
    node.content_provider = MagicMock()
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def _wire_to_file(node, *, body=_BODY, returns_path=True):
    """Make request_content_to_file write `body` to the endpoint-supplied dest_path and
    return it (or None). Records the dest_path so cleanup can be asserted."""
    captured = {}

    async def _fake(cid, dest_path, timeout=None, verify_hash=True, preferred_peer=None):
        captured["dest_path"] = Path(dest_path)
        if returns_path:
            Path(dest_path).write_bytes(body)
            return Path(dest_path)
        return None

    node.content_provider.request_content_to_file = AsyncMock(side_effect=_fake)
    return captured


def test_streams_content_and_returns_exact_bytes():
    node = _node()
    captured = _wire_to_file(node)
    resp = _client(node).get("/content/retrieve-stream/somecid")
    assert resp.status_code == 200
    assert resp.content == _BODY
    # verify_hash defaults True → the CID anchor (re)verifies the streamed file before return.
    assert node.content_provider.request_content_to_file.call_args.kwargs["verify_hash"] is True


def test_temp_file_removed_after_response():
    node = _node()
    captured = _wire_to_file(node)
    resp = _client(node).get("/content/retrieve-stream/somecid")
    assert resp.status_code == 200
    # The FileResponse background task unlinks the temp file after streaming completes.
    assert not captured["dest_path"].exists()


def test_not_found_returns_404_and_cleans_up():
    node = _node()
    captured = _wire_to_file(node, returns_path=False)
    resp = _client(node).get("/content/retrieve-stream/missingcid")
    assert resp.status_code == 404
    assert not captured["dest_path"].exists()


def test_content_filter_block_returns_451_before_fetch():
    node = _node()
    node._content_filter_store = MagicMock()
    node._content_filter_store.is_cid_blocked = MagicMock(return_value=True)
    node.content_provider.request_content_to_file = AsyncMock()
    resp = _client(node).get("/content/retrieve-stream/blockedcid")
    assert resp.status_code == 451
    node.content_provider.request_content_to_file.assert_not_called()  # refused before fetch


def test_provider_not_initialized_returns_503():
    node = _node()
    node.content_provider = None
    resp = _client(node).get("/content/retrieve-stream/somecid")
    assert resp.status_code == 503


def test_infinite_timeout_rejected_422():
    node = _node()
    node.content_provider.request_content_to_file = AsyncMock()
    resp = _client(node).get("/content/retrieve-stream/somecid",
                             params={"timeout": "Infinity"})
    assert resp.status_code == 422
    node.content_provider.request_content_to_file.assert_not_called()


def test_download_filename_from_index():
    node = _node()
    node.content_index = MagicMock()
    rec = MagicMock()
    rec.filename = "dataset.parquet"
    node.content_index.lookup = MagicMock(return_value=rec)
    _wire_to_file(node)
    resp = _client(node).get("/content/retrieve-stream/somecid")
    assert resp.status_code == 200
    assert "dataset.parquet" in resp.headers.get("content-disposition", "")
