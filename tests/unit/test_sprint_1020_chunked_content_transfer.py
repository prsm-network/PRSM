"""Sprint 1020 — TransferMode.CHUNKED: kill the >1 MiB cross-host content cliff.

Tier-1 readiness audit (2026-06-05) gap 3: large content (> MAX_INLINE_SIZE = 1 MiB)
could NOT transfer cross-host at all. The provider returned a GATEWAY
``prsm://content/{cid}`` URL that the requester's ``_fetch_from_url`` (http(s)-only)
can never fetch, and libtorrent is not a default dependency, so the BitTorrent swarm
path no-ops. Net: every file over 1 MiB was undeliverable peer-to-peer.

Fix: a new ``TransferMode.CHUNKED`` that splits content into ``GATEWAY_FETCH_CHUNK_BYTES``
(256 KiB) frames sent over the EXISTING WebSocket P2P substrate (the same path the
INLINE mode uses), reassembled + integrity-re-verified by the requester. No
libtorrent, no HTTP gateway.

These tests also close the audit's gap 6 (no in-repo evidence of a live cross-process
content fetch — all prior P2P tests mock the transport away): the integration tests
here wire two real ``ContentProvider`` instances with a genuine bidirectional
in-process link that routes each provider's ``send_to_peer`` into the other's real
``_handle_direct_message`` dispatch, so the actual serve / chunk / reassemble /
verify code runs end to end.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.node.content_provider import (
    ContentAnnouncement,
    ContentProvider,
    ContentRequestMessage,
    ContentResponseMessage,
    ContentStatus,
    TransferMode,
    MAX_INLINE_SIZE,
    GATEWAY_FETCH_CHUNK_BYTES,
)
from prsm.node.transport import MSG_DIRECT, P2PMessage


# ── Harness ─────────────────────────────────────────────────────────────────

def _make_provider(node_id: str) -> ContentProvider:
    identity = MagicMock()
    identity.node_id = node_id
    identity.public_key_b64 = "pk-" + node_id
    identity.sign = MagicMock(return_value="sig-" + node_id)
    transport = MagicMock()
    transport.identity = identity
    transport.on_message = MagicMock()
    transport.send_to_peer = AsyncMock(return_value=True)
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    return ContentProvider(identity=identity, transport=transport, gossip=gossip)


class _InProcessLink:
    """A genuine bidirectional P2P link between two providers: each provider's
    ``send_to_peer`` is routed into the OTHER provider's real
    ``_handle_direct_message`` (the same subtype-dispatch the live transport
    uses). Records every content_response frame for assertions."""

    def __init__(self, a: ContentProvider, b: ContentProvider) -> None:
        self.a, self.b = a, b
        self.frames_from_a: list[ContentResponseMessage] = []
        self.frames_from_b: list[ContentResponseMessage] = []
        a.transport.send_to_peer = self._make_sender(a, b, self.frames_from_a)
        b.transport.send_to_peer = self._make_sender(b, a, self.frames_from_b)

    @staticmethod
    def _make_sender(sender: ContentProvider, receiver: ContentProvider, sink: list):
        async def _send(target_id: str, msg: P2PMessage) -> bool:
            if msg.payload.get("subtype") == "content_response":
                sink.append(ContentResponseMessage.from_payload(msg.payload))
            peer = MagicMock()
            peer.peer_id = sender.identity.node_id
            await receiver._handle_direct_message(msg, peer)
            return True
        return _send


def _serve_request(provider: ContentProvider, cid: str, requester_id: str = "node_b"):
    peer = MagicMock()
    peer.peer_id = requester_id
    msg = P2PMessage(
        msg_type=MSG_DIRECT,
        sender_id=requester_id,
        payload=ContentRequestMessage(cid=cid, request_id="req-" + uuid.uuid4().hex[:8]).to_payload(),
    )
    return provider._handle_content_request(msg, peer)


# ── Message-encoding round-trip (CHUNKED) ─────────────────────────────────────

def test_chunked_response_message_roundtrip():
    """A CHUNKED frame must carry its chunk payload (base64) + chunk_index +
    total_chunks through to_payload/from_payload intact."""
    chunk = os.urandom(GATEWAY_FETCH_CHUNK_BYTES)
    original = ContentResponseMessage(
        request_id="req1",
        cid="QmChunkTest",
        status=ContentStatus.FOUND,
        data=chunk,
        size=GATEWAY_FETCH_CHUNK_BYTES * 5,
        transfer_mode=TransferMode.CHUNKED,
        content_hash="abc123",
        filename="big.bin",
        chunk_index=2,
        total_chunks=5,
    )
    payload = original.to_payload()
    assert payload["transfer_mode"] == "chunked"
    assert payload["chunk_index"] == 2
    assert payload["total_chunks"] == 5
    assert "data_b64" in payload  # chunk bytes carried inline like INLINE

    restored = ContentResponseMessage.from_payload(payload)
    assert restored.transfer_mode == TransferMode.CHUNKED
    assert restored.data == chunk
    assert restored.chunk_index == 2
    assert restored.total_chunks == 5
    assert restored.size == original.size


# ── Serve side: large content splits into CHUNKED frames ──────────────────────

@pytest.mark.asyncio
async def test_serve_chunks_large_content():
    """Content larger than MAX_INLINE_SIZE (but within the chunked ceiling) is
    served as N>=2 ordered CHUNKED frames that reassemble to the exact bytes."""
    payload = os.urandom(MAX_INLINE_SIZE + 300_000)  # ~1.3 MiB → multiple 256 KiB chunks
    chash = hashlib.sha256(payload).hexdigest()
    provider = _make_provider("node_a")
    provider.start()

    sent: list[P2PMessage] = []

    async def _capture(peer_id, msg):
        sent.append(msg)
        return True

    provider.transport.send_to_peer = _capture
    provider.register_local_content(cid="QmBig", size_bytes=len(payload), content_hash=chash, filename="big.bin")

    with patch.object(provider, "_fetch_local", new_callable=AsyncMock) as fl:
        fl.return_value = payload
        await _serve_request(provider, "QmBig")

    responses = [ContentResponseMessage.from_payload(m.payload) for m in sent]
    assert len(responses) >= 2, "large content should be split into multiple chunks"
    assert all(r.transfer_mode == TransferMode.CHUNKED for r in responses)
    assert all(r.total_chunks == len(responses) for r in responses)
    ordered = sorted(responses, key=lambda r: r.chunk_index)
    assert [r.chunk_index for r in ordered] == list(range(len(responses)))
    assert b"".join(r.data for r in ordered) == payload, "chunks must reassemble to exact bytes"


# ── Cross-node round-trips (real serve + reassemble over the in-process link) ──

@pytest.mark.asyncio
async def test_cross_node_inline_roundtrip():
    """Gap-6 evidence: a <=1 MiB upload on A is retrieved byte-identical on B over
    the real WebSocket-dispatch path (proves the EXISTING inline path live, not a
    local shortcut — B has no local copy)."""
    payload = os.urandom(64 * 1024)  # 64 KiB → inline
    chash = hashlib.sha256(payload).hexdigest()
    cid = "QmInlineRoundtrip"
    a, b = _make_provider("node_a"), _make_provider("node_b")
    a.start(); b.start()
    _InProcessLink(a, b)
    a.register_local_content(cid=cid, size_bytes=len(payload), content_hash=chash, filename="small.bin")
    b.content_discovery.announce_content(
        cid, "node_a",
        ContentAnnouncement(cid=cid, size=len(payload), content_type="application/octet-stream",
                            content_hash=chash, provider_id="node_a", filename="small.bin"),
    )
    assert not b.has_local_content(cid)  # B must fetch from A, not locally
    with patch.object(a, "_fetch_local", new_callable=AsyncMock) as fl:
        fl.return_value = payload
        result = await b.request_content(cid, timeout=10, verify_hash=True)
    assert result == payload


@pytest.mark.asyncio
async def test_cross_node_chunked_roundtrip():
    """Headline: a >1 MiB upload on A is retrieved byte-identical on B via CHUNKED
    transfer over the real P2P dispatch path. Pre-fix this returned None (GATEWAY
    prsm:// dead path)."""
    payload = os.urandom(MAX_INLINE_SIZE + 300_000)  # ~1.3 MiB
    chash = hashlib.sha256(payload).hexdigest()
    cid = "QmChunkedRoundtrip"
    a, b = _make_provider("node_a"), _make_provider("node_b")
    a.start(); b.start()
    link = _InProcessLink(a, b)
    a.register_local_content(cid=cid, size_bytes=len(payload), content_hash=chash, filename="big.bin")
    b.content_discovery.announce_content(
        cid, "node_a",
        ContentAnnouncement(cid=cid, size=len(payload), content_type="application/octet-stream",
                            content_hash=chash, provider_id="node_a", filename="big.bin"),
    )
    assert not b.has_local_content(cid)
    with patch.object(a, "_fetch_local", new_callable=AsyncMock) as fl:
        fl.return_value = payload
        result = await b.request_content(cid, timeout=10, verify_hash=True)
    assert result == payload, "B must reassemble the exact >1 MiB bytes from A"
    # A genuinely served multiple CHUNKED frames (not inline, not the dead gateway)
    assert len(link.frames_from_a) >= 2
    assert all(f.transfer_mode == TransferMode.CHUNKED for f in link.frames_from_a)


@pytest.mark.asyncio
async def test_cross_node_chunked_corrupted_chunk_rejected():
    """Integrity: if a chunk is tampered in flight, the reassembled bytes fail the
    hash/CID check and retrieval returns None (no silently-corrupt content)."""
    payload = os.urandom(MAX_INLINE_SIZE + 300_000)
    chash = hashlib.sha256(payload).hexdigest()
    cid = "QmCorruptChunk"
    a, b = _make_provider("node_a"), _make_provider("node_b")
    a.start(); b.start()

    # Link that corrupts the first chunk A sends to B.
    corrupted = {"done": False}

    async def _a_send(target_id, msg):
        if msg.payload.get("subtype") == "content_response" and not corrupted["done"]:
            db64 = msg.payload.get("data_b64")
            if db64 and msg.payload.get("transfer_mode") == "chunked":
                import base64
                raw = bytearray(base64.b64decode(db64))
                raw[0] ^= 0xFF  # flip a bit
                msg.payload["data_b64"] = base64.b64encode(bytes(raw)).decode()
                corrupted["done"] = True
        peer = MagicMock(); peer.peer_id = a.identity.node_id
        await b._handle_direct_message(msg, peer)
        return True

    async def _b_send(target_id, msg):
        peer = MagicMock(); peer.peer_id = b.identity.node_id
        await a._handle_direct_message(msg, peer)
        return True

    a.transport.send_to_peer = _a_send
    b.transport.send_to_peer = _b_send

    a.register_local_content(cid=cid, size_bytes=len(payload), content_hash=chash, filename="big.bin")
    b.content_discovery.announce_content(
        cid, "node_a",
        ContentAnnouncement(cid=cid, size=len(payload), content_type="application/octet-stream",
                            content_hash=chash, provider_id="node_a", filename="big.bin"),
    )
    with patch.object(a, "_fetch_local", new_callable=AsyncMock) as fl:
        fl.return_value = payload
        result = await b.request_content(cid, timeout=10, verify_hash=True)
    assert result is None, "corrupted reassembly must be rejected, not returned"


# ── Memory-DoS defenses (adversarial review, 2026-06-05) ──────────────────────

def _pending_chunk_frame(provider, request_id):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    provider._pending_requests[request_id] = fut
    return fut


@pytest.mark.asyncio
async def test_oversized_chunk_frame_refused_without_buffering():
    """Memory-amplification defense: a frame larger than the protocol chunk size
    (a provider declares a small total_chunks to pass the count guard, then sends
    a giant frame) must be dropped WITHOUT buffering — never held until timeout."""
    p = _make_provider("node_b")
    rid = "req-oversized"
    fut = _pending_chunk_frame(p, rid)
    oversized = ContentResponseMessage(
        request_id=rid, cid="QmX", status=ContentStatus.FOUND,
        data=b"x" * (GATEWAY_FETCH_CHUNK_BYTES + 1),  # one byte over the chunk size
        size=GATEWAY_FETCH_CHUNK_BYTES * 2,
        transfer_mode=TransferMode.CHUNKED, chunk_index=0, total_chunks=2,
    )
    p._handle_content_response(P2PMessage(
        msg_type=MSG_DIRECT, sender_id="node_a", payload=oversized.to_payload()))
    assert rid not in p._pending_chunks, "oversized frame must not be buffered"
    assert not fut.done(), "request must not resolve on a rejected frame (it times out → None)"


@pytest.mark.asyncio
async def test_running_total_over_ceiling_refused():
    """Defense-in-depth: even with each frame at/under the chunk size, the running
    buffered total must not exceed the chunked ceiling."""
    p = _make_provider("node_b")
    rid = "req-ceiling"
    fut = _pending_chunk_frame(p, rid)
    # Force a tiny ceiling so two legitimate-sized frames already exceed it.
    os.environ["PRSM_MAX_CHUNKED_TRANSFER_BYTES"] = str(GATEWAY_FETCH_CHUNK_BYTES)
    try:
        total = 4  # 4 * 256KiB > 256KiB ceiling → count guard already trips
        frame = ContentResponseMessage(
            request_id=rid, cid="QmX", status=ContentStatus.FOUND,
            data=b"y" * GATEWAY_FETCH_CHUNK_BYTES, size=GATEWAY_FETCH_CHUNK_BYTES * total,
            transfer_mode=TransferMode.CHUNKED, chunk_index=0, total_chunks=total,
        )
        p._handle_content_response(P2PMessage(
            msg_type=MSG_DIRECT, sender_id="node_a", payload=frame.to_payload()))
        assert rid not in p._pending_chunks, "over-ceiling transfer must be refused"
        assert not fut.done()
    finally:
        os.environ.pop("PRSM_MAX_CHUNKED_TRANSFER_BYTES", None)


@pytest.mark.asyncio
async def test_inconsistent_total_chunks_refused():
    """A provider that changes total_chunks across frames for one request is
    malformed — fail fast (drop the buffer) rather than re-keying and stalling
    to the full request timeout."""
    p = _make_provider("node_b")
    rid = "req-incon"
    fut = _pending_chunk_frame(p, rid)
    f0 = ContentResponseMessage(
        request_id=rid, cid="QmY", status=ContentStatus.FOUND, data=b"a" * 100,
        size=500, transfer_mode=TransferMode.CHUNKED, chunk_index=0, total_chunks=5)
    f1 = ContentResponseMessage(
        request_id=rid, cid="QmY", status=ContentStatus.FOUND, data=b"b" * 100,
        size=600, transfer_mode=TransferMode.CHUNKED, chunk_index=1, total_chunks=6)
    p._handle_content_response(P2PMessage(
        msg_type=MSG_DIRECT, sender_id="node_a", payload=f0.to_payload()))
    assert rid in p._pending_chunks  # first frame buffered
    p._handle_content_response(P2PMessage(
        msg_type=MSG_DIRECT, sender_id="node_a", payload=f1.to_payload()))
    assert rid not in p._pending_chunks, "inconsistent total_chunks must drop the buffer"
    assert not fut.done()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
