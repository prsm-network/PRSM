"""Sprint 821 — PRSMClient content methods: publish + fetch.

Mirrors sprint 806 (`content publish`) + sprint 805
(`content fetch`) CLI commands on the SDK side. Closes the
Python integration surface for content operations.

  async def publish_content(
      self, text: str, *,
      filename: str = "document.txt",
      replicas: int = 3,
      royalty_rate: Optional[float] = None,
      parent_cids: Optional[List[str]] = None,
  ) -> Dict[str, Any]:
      \"\"\"POST /content/upload; return server payload with cid.\"\"\"

  async def fetch_content(
      self, cid: str, *,
      timeout: float = 30.0,
      verify_hash: bool = True,
  ) -> Dict[str, Any]:
      \"\"\"GET /content/retrieve/{cid}; return parsed payload.

      Caller can base64-decode `data` field for binary content.
      Returns {cid, status, data, size_bytes, content_hash,
               filename, providers_tried, error}.
      \"\"\"

Pin tests:
- publish_content method exists.
- publish_content POSTs to /content/upload.
- publish_content body has canonical 5 fields.
- publish_content defaults match sprint-806 CLI.
- publish_content overrides pass through.
- fetch_content method exists.
- fetch_content GETs /content/retrieve/{cid}.
- fetch_content threads timeout + verify_hash query params.
- fetch_content returns parsed payload.
- Round-trip via the SDK methods (publish text → fetch CID).
"""
from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


# ---- publish_content --------------------------------------------


async def test_publish_content_method_exists():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    assert hasattr(client, "publish_content")
    assert callable(client.publish_content)


async def test_publish_content_posts_to_upload():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={
            "cid": "QmTestCid", "status": "uploaded",
        }),
    ) as mp:
        result = await client.publish_content("hello world")
    path, body = mp.call_args.args[0], mp.call_args.args[1]
    assert path == "/content/upload"
    assert result["cid"] == "QmTestCid"


async def test_publish_content_body_canonical_shape():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={"cid": "Qm", "status": "uploaded"}),
    ) as mp:
        await client.publish_content("hi")
    body = mp.call_args.args[1]
    assert body["text"] == "hi"
    assert "filename" in body
    assert "replicas" in body
    assert "parent_cids" in body
    assert isinstance(body["parent_cids"], list)


async def test_publish_content_defaults_match_cli():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={"cid": "Qm", "status": "uploaded"}),
    ) as mp:
        await client.publish_content("hi")
    body = mp.call_args.args[1]
    # Defaults match server-side defaults + sprint-806 CLI
    assert body["filename"] == "document.txt"
    assert body["replicas"] == 3
    assert body["parent_cids"] == []


async def test_publish_content_overrides_pass_through():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={"cid": "Qm", "status": "uploaded"}),
    ) as mp:
        await client.publish_content(
            "payload",
            filename="rename.md",
            replicas=5,
            royalty_rate=0.05,
            parent_cids=["QmA", "QmB"],
        )
    body = mp.call_args.args[1]
    assert body["text"] == "payload"
    assert body["filename"] == "rename.md"
    assert body["replicas"] == 5
    assert body["royalty_rate"] == 0.05
    assert body["parent_cids"] == ["QmA", "QmB"]


async def test_publish_content_omits_royalty_when_none():
    """When royalty_rate is None (default), the field is OMITTED
    from the body so the server's default applies. Including it
    as `None` would 422 against the ge=0.001 constraint."""
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={"cid": "Qm", "status": "uploaded"}),
    ) as mp:
        await client.publish_content("hi")
    body = mp.call_args.args[1]
    assert "royalty_rate" not in body


# ---- fetch_content ----------------------------------------------


async def test_fetch_content_method_exists():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    assert hasattr(client, "fetch_content")
    assert callable(client.fetch_content)


async def test_fetch_content_uses_get_with_cid_path():
    """fetch_content does an HTTP GET to /content/retrieve/{cid}
    with timeout + verify_hash query params. Uses an internal
    _get_params helper (or aiohttp directly) — pin via patching
    the session's .get method."""
    from prsm.sdk.client import PRSMClient

    # Build a fake response object with .json() returning the
    # expected payload.
    class _FakeResp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {
                "cid": "QmTest",
                "status": "success",
                "data": "aGVsbG8=",  # base64("hello")
                "size_bytes": 5,
                "content_hash": "deadbeef",
                "filename": "doc.txt",
                "providers_tried": 1,
                "error": None,
            }

    captured = {}

    def _fake_get(url, headers=None, timeout=None, params=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResp()

    class _FakeSession:
        def get(self, url, **kwargs):
            return _fake_get(url, **kwargs)

        async def close(self):
            pass

    client = PRSMClient("http://node:8000")
    client._session = _FakeSession()

    result = await client.fetch_content("QmTest")
    assert "/content/retrieve/QmTest" in captured["url"]
    assert result["status"] == "success"
    assert result["data"] == "aGVsbG8="


async def test_fetch_content_threads_query_params():
    from prsm.sdk.client import PRSMClient

    class _FakeResp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {"cid": "X", "status": "success"}

    captured = {}

    def _fake_get(url, headers=None, timeout=None, params=None):
        captured["params"] = params
        return _FakeResp()

    class _FakeSession:
        def get(self, url, **kwargs):
            return _fake_get(url, **kwargs)

        async def close(self):
            pass

    client = PRSMClient("http://node:8000")
    client._session = _FakeSession()

    await client.fetch_content(
        "QmX", timeout=60.0, verify_hash=False,
    )
    params = captured.get("params") or {}
    # Allow either True/False or "true"/"false" depending on
    # how the SDK serializes booleans
    assert params.get("timeout") in (60.0, 60, "60", "60.0")
    assert params.get("verify_hash") in (False, "false", "False", 0, "0")


# ---- Round-trip via SDK methods --------------------------------


async def test_round_trip_publish_then_fetch():
    """End-to-end SDK round-trip: mock both calls + verify
    cid returned from publish flows into fetch."""
    from prsm.sdk.client import PRSMClient

    client = PRSMClient("http://node:8000")

    # First call: publish returns CID
    publish_resp = {
        "cid": "QmNewCid", "status": "uploaded",
        "filename": "doc.txt", "size_bytes": 11,
    }
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value=publish_resp),
    ):
        publish_result = await client.publish_content("hello world")

    new_cid = publish_result["cid"]
    assert new_cid == "QmNewCid"

    # Second call: fetch by that CID
    class _FakeResp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {
                "cid": new_cid, "status": "success",
                "data": "aGVsbG8gd29ybGQ=",  # base64("hello world")
                "size_bytes": 11,
            }

    captured_url = {}

    def _fake_get(url, headers=None, timeout=None, params=None):
        captured_url["v"] = url
        return _FakeResp()

    class _FakeSession:
        def get(self, url, **kwargs):
            return _fake_get(url, **kwargs)

        async def close(self):
            pass

    client._session = _FakeSession()
    fetch_result = await client.fetch_content(new_cid)
    assert fetch_result["cid"] == new_cid
    assert fetch_result["status"] == "success"
    # CID appears in the GET URL
    assert new_cid in captured_url["v"]


# ---- sp1457: fetch_content_to_file (streaming large content to disk) --------

async def test_fetch_content_to_file_streams_to_disk(tmp_path):
    from prsm.sdk.client import PRSMClient
    payload = b"large-streamed-sdk-body-" * 50_000
    chunks = [payload[i:i + 65536] for i in range(0, len(payload), 65536)]

    class _Content:
        async def iter_chunked(self, n):
            for c in chunks:
                yield c

    class _FakeResp:
        status = 200
        content = _Content()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def text(self):
            return ""

    captured = {}

    class _FakeSession:
        def get(self, url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs.get("params")
            return _FakeResp()

        async def close(self):
            pass

    client = PRSMClient("http://node:8000")
    client._session = _FakeSession()
    dest = tmp_path / "big.bin"
    result = await client.fetch_content_to_file("QmBig", dest)
    assert "/content/retrieve-stream/QmBig" in captured["url"]
    assert dest.read_bytes() == payload           # byte-exact, streamed
    assert result == str(dest)


async def test_fetch_content_to_file_raises_on_non_200(tmp_path):
    import pytest
    from prsm.sdk.client import PRSMClient

    class _FakeResp:
        status = 404

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def text(self):
            return "not found on any provider"

    class _FakeSession:
        def get(self, url, **kwargs):
            return _FakeResp()

        async def close(self):
            pass

    client = PRSMClient("http://node:8000")
    client._session = _FakeSession()
    with pytest.raises(ValueError, match="404"):
        await client.fetch_content_to_file("QmMissing", tmp_path / "x.bin")


# ---- sp1457: publish_content_from_file (streaming large publish) ------------

async def test_publish_content_from_file_streams(tmp_path):
    from prsm.sdk.client import PRSMClient
    src = tmp_path / "big.bin"
    src.write_bytes(b"binary dataset " * 50_000)

    class _FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {"cid": "QmBig", "size_bytes": src.stat().st_size, "status": "published"}

        async def text(self):
            return ""

    captured = {}

    class _FakeSession:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs.get("params")
            return _FakeResp()

        async def close(self):
            pass

    client = PRSMClient("http://node:8000")
    client._session = _FakeSession()
    result = await client.publish_content_from_file(src, filename="ds.bin")
    assert "/content/upload-stream" in captured["url"]
    assert result["cid"] == "QmBig"
    assert captured["params"]["filename"] == "ds.bin"


async def test_publish_content_from_file_raises_non_200(tmp_path):
    import pytest
    from prsm.sdk.client import PRSMClient
    src = tmp_path / "x.bin"
    src.write_bytes(b"d")

    class _FakeResp:
        status = 501

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def text(self):
            return "no streaming publisher"

    class _FakeSession:
        def post(self, url, **kwargs):
            return _FakeResp()

        async def close(self):
            pass

    client = PRSMClient("http://node:8000")
    client._session = _FakeSession()
    with pytest.raises(ValueError, match="501"):
        await client.publish_content_from_file(src)


# ── sp1458: publish_paid_content_from_file (streaming large paid publish) ─────

async def test_publish_paid_content_from_file_streams(tmp_path):
    from prsm.sdk.client import PRSMClient
    src = tmp_path / "big.bin"
    src.write_bytes(b"paid dataset " * 50_000)

    class _FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {"content_hash": "0xab", "cid": "QmX", "commitment": "0xcd",
                    "deposit_tx": "0xdep"}

        async def text(self):
            return ""

    captured = {}

    class _FakeSession:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs.get("params")
            return _FakeResp()

        async def close(self):
            pass

    client = PRSMClient("http://node:8000")
    client._session = _FakeSession()
    result = await client.publish_paid_content_from_file(
        src, buyer_x25519_pubkeys=["k1", "k2"], fee_wei=10 ** 18, filename="ds.bin")
    assert "/content/paid/publish-stream" in captured["url"]
    assert result["content_hash"] == "0xab"
    assert captured["params"]["buyer_x25519_pubkeys"] == "k1,k2"
    assert captured["params"]["fee_wei"] == 10 ** 18


async def test_publish_paid_content_from_file_raises_non_200(tmp_path):
    import pytest
    from prsm.sdk.client import PRSMClient
    src = tmp_path / "x.bin"
    src.write_bytes(b"d")

    class _FakeResp:
        status = 503

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def text(self):
            return "not wired"

    class _FakeSession:
        def post(self, url, **kwargs):
            return _FakeResp()

        async def close(self):
            pass

    client = PRSMClient("http://node:8000")
    client._session = _FakeSession()
    with pytest.raises(ValueError, match="503"):
        await client.publish_paid_content_from_file(
            src, buyer_x25519_pubkeys=["k"], fee_wei=10 ** 18)
