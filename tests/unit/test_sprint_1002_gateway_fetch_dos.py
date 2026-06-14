"""Sprint 1002 — GATEWAY content-fetch DoS hardening.

The content-data-plane integrity hunt (workflow w8gvkhelm) confirmed two
retrieval-resource-exhaustion bugs in ``ContentProvider._fetch_from_url``,
the path that fetches content when a peer responds with
``TransferMode.GATEWAY`` (used for content larger than ``MAX_INLINE_SIZE``):

  Finding 7 (HIGH) — no byte cap. ``await resp.read()`` slurps the entire
  HTTP body into memory. A malicious peer advertises a CID, answers the
  request with a GATEWAY url it controls, and serves an unbounded /
  much-larger-than-advertised body → the *retrieving* node OOMs. The
  gateway_url is attacker-supplied (ContentResponseMessage.gateway_url,
  parsed verbatim from the peer's response payload).

  Finding 8 (MEDIUM) — the fetch ran with a hard-coded 120s ClientTimeout,
  OUTSIDE the caller's retrieve-timeout budget (the ``asyncio.wait_for`` at
  _request_from_provider only wraps waiting for the RESPONSE message, not
  the subsequent gateway HTTP fetch). A slow-loris gateway ties up a worker
  for 120s regardless of the caller's much-smaller timeout.

Fix: ``_fetch_from_url`` now takes ``timeout`` + ``max_bytes`` and streams
the body with a hard byte cap (abort + return None when exceeded; reject
early when the advertised Content-Length already exceeds the cap), and uses
the caller's remaining timeout instead of a fixed 120s. ``_request_from_provider``
computes the cap from the advertised size (bounded by a hard ceiling) and
threads the remaining timeout through.

These are non-regressing: honest content (served within its advertised
size, in the caller's time budget) flows through unchanged; only oversized
/ slow malicious gateways are refused.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.content_provider import (
    ContentProvider,
    ContentResponseMessage,
    ContentStatus,
    TransferMode,
)


def _make_provider():
    identity = MagicMock()
    identity.node_id = "test-node-id"
    transport = MagicMock()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    return ContentProvider(identity=identity, transport=transport, gossip=gossip)


class _FakeStreamReader:
    """Mimics aiohttp StreamReader.iter_chunked for the fetch body."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.consumed = False

    async def iter_chunked(self, _n):
        self.consumed = True
        for c in self._chunks:
            yield c


class _FakeResponse:
    def __init__(self, status=200, headers=None, chunks=None):
        self.status = status
        self.headers = headers or {}
        self.content = _FakeStreamReader(chunks or [])
        self._full = b"".join(chunks or [])

    async def read(self):  # the pre-fix code path
        return self._full

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False


class _FakeSession:
    def __init__(self, response):
        self._response = response
        self.captured_timeout = None
        self.captured_allow_redirects = None
        self.get_calls = 0

    def get(self, _url, timeout=None, allow_redirects=None):
        # sp1102 — the production fetch now passes allow_redirects=False (SSRF: a
        # public URL must not 302 to an internal one). Mirror the real signature.
        self.get_calls += 1
        self.captured_timeout = timeout
        self.captured_allow_redirects = allow_redirects
        return self._response

    # sp1102 — the fetch now opens a per-request session via _open_gateway_session;
    # the fake doubles as that async context manager.
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False


def _install_fake_session(provider, session):
    """sp1102 — route _fetch_from_url's session through the fake (was p._http_session;
    the SSRF fix opens a per-fetch pinned session via _open_gateway_session instead)."""
    provider._open_gateway_session = lambda _pinned_ips: session


@pytest.mark.asyncio
async def test_fetch_from_url_aborts_when_streamed_body_exceeds_cap():
    """A gateway that streams more bytes than the cap must be aborted
    mid-stream and return None — never accumulate the full body."""
    chunks = [b"x" * 50 for _ in range(20)]  # 1000 bytes total
    resp = _FakeResponse(status=200, headers={}, chunks=chunks)
    p = _make_provider()
    _install_fake_session(p, _FakeSession(resp))

    # public numeric IP → passes the sp1102 SSRF guard, so the cap logic is exercised
    out = await p._fetch_from_url("http://93.184.216.34/gw", timeout=5.0, max_bytes=100)

    assert out is None


@pytest.mark.asyncio
async def test_fetch_from_url_rejects_oversized_content_length_early():
    """When the advertised Content-Length already exceeds the cap, refuse
    BEFORE reading the body (fast-fail, no allocation)."""
    resp = _FakeResponse(
        status=200,
        headers={"Content-Length": "999999999"},
        chunks=[b"x" * 10],
    )
    p = _make_provider()
    _install_fake_session(p, _FakeSession(resp))

    out = await p._fetch_from_url("http://93.184.216.34/gw", timeout=5.0, max_bytes=1000)

    assert out is None
    # body must NOT have been streamed
    assert resp.content.consumed is False


@pytest.mark.asyncio
async def test_fetch_from_url_returns_bytes_within_cap():
    """Honest content within the cap flows through unchanged."""
    body = b"legitimate content payload"
    resp = _FakeResponse(
        status=200,
        headers={"Content-Length": str(len(body))},
        chunks=[body],
    )
    p = _make_provider()
    _install_fake_session(p, _FakeSession(resp))

    out = await p._fetch_from_url("http://93.184.216.34/ok", timeout=5.0, max_bytes=10_000)

    assert out == body


@pytest.mark.asyncio
async def test_fetch_from_url_honors_caller_timeout_not_hardcoded_120():
    """The HTTP fetch must run under the caller's remaining timeout budget,
    not a fixed 120s (finding 8 — slow-loris worker tie-up)."""
    resp = _FakeResponse(status=200, headers={}, chunks=[b"ok"])
    sess = _FakeSession(resp)
    p = _make_provider()
    _install_fake_session(p, sess)

    await p._fetch_from_url("http://93.184.216.34/ok", timeout=7.0, max_bytes=10_000)

    assert sess.captured_timeout is not None
    # aiohttp.ClientTimeout exposes .total
    assert getattr(sess.captured_timeout, "total", None) == pytest.approx(7.0)
    # sp1102 — redirects must be disabled on the untrusted fetch
    assert sess.captured_allow_redirects is False


@pytest.mark.asyncio
async def test_fetch_from_url_skips_non_http_scheme():
    """Non-HTTP gateway URLs (e.g. prsm://) are not fetchable here —
    preserve the pre-fix early return."""
    p = _make_provider()
    p._http_session = _FakeSession(_FakeResponse())

    out = await p._fetch_from_url("prsm://node/x", timeout=5.0, max_bytes=10_000)

    assert out is None


@pytest.mark.asyncio
async def test_request_from_provider_caps_gateway_by_advertised_size():
    """_request_from_provider must derive a byte cap from the advertised
    size (bounded by the hard ceiling) and thread the remaining timeout
    into _fetch_from_url, so a peer advertising a tiny size cannot then
    serve an unbounded gateway body."""
    p = _make_provider()
    record = MagicMock()
    record.content_hash = ""
    record.size_bytes = 10  # advertised tiny
    record.providers = {"prov"}
    p.content_index = MagicMock()
    p.content_index.lookup = MagicMock(return_value=record)

    captured = {}

    async def fake_fetch(url, timeout=None, max_bytes=None):
        captured["url"] = url
        captured["timeout"] = timeout
        captured["max_bytes"] = max_bytes
        return None

    p._fetch_from_url = fake_fetch

    resp = ContentResponseMessage(
        request_id="ignored",
        cid="somecid",
        status=ContentStatus.FOUND,
        transfer_mode=TransferMode.GATEWAY,
        gateway_url="http://evil/gw",
    )

    async def _send(_peer_id, _msg):
        # exactly one pending request — resolve it with the gateway response
        for _rid, fut in list(p._pending_requests.items()):
            if not fut.done():
                fut.set_result(resp)
        return True

    p.transport.send_to_peer = _send

    out = await p._request_from_provider(
        "somecid", "prov", timeout=5.0, expected_hash=None,
    )

    assert out is None
    # cap derived from advertised 10 bytes (with slack), well under the ceiling
    assert captured["max_bytes"] == 10 * 2 + 4096
    assert captured["timeout"] is not None
    assert 0 < captured["timeout"] <= 5.0


@pytest.mark.asyncio
async def test_request_from_provider_uses_hard_ceiling_when_size_unknown():
    """When the index has no positive advertised size, the cap falls back
    to the hard ceiling (still bounded — never unbounded)."""
    from prsm.node import content_provider as cp

    p = _make_provider()
    record = MagicMock()
    record.content_hash = ""
    record.size_bytes = 0  # unknown
    record.providers = {"prov"}
    p.content_index = MagicMock()
    p.content_index.lookup = MagicMock(return_value=record)

    captured = {}

    async def fake_fetch(url, timeout=None, max_bytes=None):
        captured["max_bytes"] = max_bytes
        return None

    p._fetch_from_url = fake_fetch

    resp = ContentResponseMessage(
        request_id="ignored",
        cid="c",
        status=ContentStatus.FOUND,
        transfer_mode=TransferMode.GATEWAY,
        gateway_url="http://gw",
    )

    async def _send(_peer_id, _msg):
        for _rid, fut in list(p._pending_requests.items()):
            if not fut.done():
                fut.set_result(resp)
        return True

    p.transport.send_to_peer = _send

    await p._request_from_provider("c", "prov", timeout=5.0, expected_hash=None)

    assert captured["max_bytes"] == cp._gateway_fetch_hard_ceiling_bytes()
