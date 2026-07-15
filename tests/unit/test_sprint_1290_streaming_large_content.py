"""Sprint 1290 — streaming large-content transfer (lift the dead-GATEWAY ceiling).

Before this, content above the in-memory CHUNKED ceiling (64 MiB) returned a
``prsm://content/{cid}`` GATEWAY URL that nothing actually served (HTTP-only
``_fetch_from_url`` + no default libtorrent) — so large encrypted content (e.g. a Tier
B/C artifact bundle over 64 MiB) had NO decentralized serve path.

The fix streams such content over the EXISTING P2P substrate with no full-content
materialization on either side:
  * send: ``_send_chunked_from_path`` reads the provider's on-disk staged file
    frame-by-frame (256 KiB) — only one frame resident at a time.
  * receive: ``request_content_to_file`` writes incoming frames straight to a
    destination file via ``_StreamingSink`` (bounded by the streaming ceiling), then
    CID/hash-verifies by a streaming read.
The common path (<= 64 MiB, or no on-disk staged copy) is unchanged — INLINE/CHUNKED
behave byte-identically; this only replaces the previously-dead >64 MiB GATEWAY case.
"""
from __future__ import annotations

import hashlib
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.node.content_provider import (
    ContentAnnouncement,
    ContentProvider,
    ContentResponseMessage,
    ContentStatus,
    TransferMode,
    MAX_INLINE_SIZE,
    GATEWAY_FETCH_CHUNK_BYTES,
    _StreamingSink,
    _max_streaming_transfer_bytes,
)
from prsm.node.transport import MSG_DIRECT, P2PMessage


# ── Harness (mirrors test_sprint_1020) ───────────────────────────────────────

def _make_provider(node_id: str) -> ContentProvider:
    identity = MagicMock()
    identity.node_id = node_id
    identity.public_key_b64 = "pk-" + node_id
    identity.sign = MagicMock(return_value="sig-" + node_id)
    transport = MagicMock()
    transport.identity = identity
    transport.send_to_peer = AsyncMock(return_value=True)
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    return ContentProvider(identity=identity, transport=transport, gossip=gossip)


class _InProcessLink:
    """Bidirectional P2P link: each provider's send_to_peer routes into the OTHER's
    real _handle_direct_message. Records every content_response frame."""

    def __init__(self, a: ContentProvider, b: ContentProvider) -> None:
        self.frames_from_a: list = []
        self.frames_from_b: list = []
        a.transport.send_to_peer = self._sender(a, b, self.frames_from_a)
        b.transport.send_to_peer = self._sender(b, a, self.frames_from_b)

    @staticmethod
    def _sender(sender, receiver, sink):
        async def _send(target_id, msg):
            if msg.payload.get("subtype") == "content_response":
                sink.append(ContentResponseMessage.from_payload(msg.payload))
            peer = MagicMock()
            peer.peer_id = sender.identity.node_id
            await receiver._handle_direct_message(msg, peer)
            return True
        return _send


def _staged_publisher(path):
    pub = MagicMock()
    pub.local_publish_path = MagicMock(return_value=path)
    return pub


# ── _StreamingSink unit tests ─────────────────────────────────────────────────

def _frame(cid, data, index, total):
    return ContentResponseMessage(
        request_id="r", cid=cid, status=ContentStatus.FOUND, data=data,
        size=total * GATEWAY_FETCH_CHUNK_BYTES, transfer_mode=TransferMode.CHUNKED,
        chunk_index=index, total_chunks=total,
    )


def test_streaming_sink_in_order(tmp_path):
    dest = tmp_path / "out.bin"
    body = os.urandom(GATEWAY_FETCH_CHUNK_BYTES * 2 + 100)
    cs = GATEWAY_FETCH_CHUNK_BYTES
    frames = [body[i * cs:(i + 1) * cs] for i in range((len(body) + cs - 1) // cs)]
    sink = _StreamingSink(dest, cs, _max_streaming_transfer_bytes())
    statuses = [sink.add_frame(_frame("c", f, i, len(frames))) for i, f in enumerate(frames)]
    assert statuses[-1] == "complete"
    assert statuses[:-1] == ["pending"] * (len(frames) - 1)
    assert dest.read_bytes() == body


def test_streaming_sink_out_of_order(tmp_path):
    dest = tmp_path / "out.bin"
    body = os.urandom(GATEWAY_FETCH_CHUNK_BYTES * 3 + 7)
    cs = GATEWAY_FETCH_CHUNK_BYTES
    frames = [body[i * cs:(i + 1) * cs] for i in range((len(body) + cs - 1) // cs)]
    sink = _StreamingSink(dest, cs, _max_streaming_transfer_bytes())
    order = [2, 0, 3, 1]  # deliberately scrambled
    final = "pending"
    for i in order:
        final = sink.add_frame(_frame("c", frames[i], i, len(frames)))
    assert final == "complete"
    assert dest.read_bytes() == body  # frames written at their offsets


def test_streaming_sink_rejects_oversized_frame(tmp_path):
    dest = tmp_path / "out.bin"
    cs = GATEWAY_FETCH_CHUNK_BYTES
    sink = _StreamingSink(dest, cs, _max_streaming_transfer_bytes())
    # a frame larger than the protocol chunk size is a disk-amplification attempt
    status = sink.add_frame(_frame("c", os.urandom(cs + 1), 0, 4))
    assert status == "error"
    assert not dest.exists()  # partial file removed on failure


def test_streaming_sink_rejects_over_ceiling_count(tmp_path):
    dest = tmp_path / "out.bin"
    cs = GATEWAY_FETCH_CHUNK_BYTES
    ceiling = cs * 4  # tiny ceiling
    sink = _StreamingSink(dest, cs, ceiling)
    # declared total implies a body far over the ceiling → reject up front
    status = sink.add_frame(_frame("c", os.urandom(cs), 0, 1000))
    assert status == "error"
    assert not dest.exists()


# ── Send side: stream from disk, byte-identical frames ────────────────────────

@pytest.mark.asyncio
async def test_send_chunked_from_path_reassembles_and_matches_inmemory(tmp_path):
    body = os.urandom(GATEWAY_FETCH_CHUNK_BYTES * 3 + 500)
    src = tmp_path / "big.bin"
    src.write_bytes(body)
    provider = _make_provider("node_a")
    sent: list = []

    async def _capture(peer_id, msg):
        sent.append(ContentResponseMessage.from_payload(msg.payload))
        return True

    provider.transport.send_to_peer = _capture
    info = {"content_hash": hashlib.sha256(body).hexdigest(), "filename": "big.bin"}
    await provider._send_chunked_from_path("node_b", "req1", "cid", src, len(body), info)

    assert all(r.transfer_mode == TransferMode.CHUNKED for r in sent)
    ordered = sorted(sent, key=lambda r: r.chunk_index)
    assert [r.chunk_index for r in ordered] == list(range(len(sent)))
    # streamed-from-disk frames reassemble to the exact file bytes
    assert b"".join(r.data for r in ordered) == body
    # ... and are byte-identical to what the in-memory _send_chunked_response produces
    inmem: list = []

    async def _capture2(peer_id, msg):
        inmem.append(ContentResponseMessage.from_payload(msg.payload))
        return True

    provider.transport.send_to_peer = _capture2
    await provider._send_chunked_response("node_b", "req1", "cid", body, info)
    assert [r.data for r in sorted(inmem, key=lambda r: r.chunk_index)] == [r.data for r in ordered]


# ── Serve-decision routing ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_serve_routes_large_staged_content_to_stream(tmp_path, monkeypatch):
    """Content above the (shrunk) chunked ceiling WITH an on-disk staged copy streams
    from disk as CHUNKED frames — NOT the dead GATEWAY URL."""
    monkeypatch.setenv("PRSM_MAX_CHUNKED_TRANSFER_BYTES", str(GATEWAY_FETCH_CHUNK_BYTES))
    body = os.urandom(MAX_INLINE_SIZE + 300_000)  # > MAX_INLINE and > the shrunk ceiling
    src = tmp_path / "staged.bin"
    src.write_bytes(body)
    a = _make_provider("node_a")
    a.content_publisher = _staged_publisher(src)
    a.register_local_content(cid="QmBigStaged", size_bytes=len(body),
                             content_hash=hashlib.sha256(body).hexdigest(), filename="staged.bin")
    sent: list = []

    async def _capture(peer_id, msg):
        sent.append(ContentResponseMessage.from_payload(msg.payload))
        return True

    a.transport.send_to_peer = _capture
    peer = MagicMock(); peer.peer_id = "node_b"
    from prsm.node.content_provider import ContentRequestMessage
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="node_b",
                     payload=ContentRequestMessage(cid="QmBigStaged").to_payload())
    await a._handle_content_request(msg, peer)

    assert sent, "must have served something"
    assert all(r.transfer_mode == TransferMode.CHUNKED for r in sent), "should stream, not GATEWAY"
    assert not any(r.transfer_mode == TransferMode.GATEWAY for r in sent)
    assert b"".join(r.data for r in sorted(sent, key=lambda r: r.chunk_index)) == body


@pytest.mark.asyncio
async def test_common_path_unchanged_without_publisher(tmp_path, monkeypatch):
    """No staged on-disk copy → _local_stream_info is None → the in-memory path runs
    unchanged (content materialized via _fetch_local, served as ordinary CHUNKED)."""
    body = os.urandom(MAX_INLINE_SIZE + 200_000)
    a = _make_provider("node_a")
    assert a._local_stream_info("QmNoPub") is None  # no publisher wired
    a.register_local_content(cid="QmNoPub", size_bytes=len(body),
                             content_hash=hashlib.sha256(body).hexdigest(), filename="x.bin")
    sent: list = []

    async def _capture(peer_id, msg):
        sent.append(ContentResponseMessage.from_payload(msg.payload))
        return True

    a.transport.send_to_peer = _capture
    peer = MagicMock(); peer.peer_id = "node_b"
    from prsm.node.content_provider import ContentRequestMessage
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="node_b",
                     payload=ContentRequestMessage(cid="QmNoPub").to_payload())
    with patch.object(a, "_fetch_local", new_callable=AsyncMock) as fl:
        fl.return_value = body
        await a._handle_content_request(msg, peer)
        fl.assert_awaited()  # common path materialized via _fetch_local (unchanged)
    assert all(r.transfer_mode == TransferMode.CHUNKED for r in sent)
    assert b"".join(r.data for r in sorted(sent, key=lambda r: r.chunk_index)) == body


# ── End-to-end: request_content_to_file over a real in-process link ───────────

@pytest.mark.asyncio
async def test_cross_node_request_content_to_file_roundtrip(tmp_path, monkeypatch):
    """Headline: a large staged file on A is streamed to a destination file on B via
    the real serve→stream→sink→verify path, with NEITHER side materializing the body.
    Pre-fix this was the dead prsm:// GATEWAY (undeliverable)."""
    monkeypatch.setenv("PRSM_MAX_CHUNKED_TRANSFER_BYTES", str(GATEWAY_FETCH_CHUNK_BYTES))
    body = os.urandom(MAX_INLINE_SIZE + 400_000)
    chash = hashlib.sha256(body).hexdigest()
    cid = "QmStreamToFile"
    src = tmp_path / "a_staged.bin"
    src.write_bytes(body)
    a, b = _make_provider("node_a"), _make_provider("node_b")
    a.content_publisher = _staged_publisher(src)
    a.register_local_content(cid=cid, size_bytes=len(body), content_hash=chash, filename="big.bin")
    link = _InProcessLink(a, b)
    b.content_discovery.announce_content(
        cid, "node_a",
        ContentAnnouncement(cid=cid, size=len(body), content_type="application/octet-stream",
                            content_hash=chash, provider_id="node_a", filename="big.bin"),
    )
    assert not b.has_local_content(cid)

    dest = tmp_path / "b_received.bin"
    result = await b.request_content_to_file(cid, dest, timeout=10, verify_hash=True)

    assert result is not None, "B must receive the streamed file"
    assert dest.read_bytes() == body, "received file must be byte-identical"
    assert len(link.frames_from_a) >= 2, "A streamed multiple CHUNKED frames"
    assert all(f.transfer_mode == TransferMode.CHUNKED for f in link.frames_from_a)


@pytest.mark.asyncio
async def test_request_content_to_file_rejects_corrupted_stream(tmp_path, monkeypatch):
    """A tampered frame → reassembled file fails the streaming hash check → None + the
    partial file is removed (no silently-corrupt large content)."""
    monkeypatch.setenv("PRSM_MAX_CHUNKED_TRANSFER_BYTES", str(GATEWAY_FETCH_CHUNK_BYTES))
    body = os.urandom(MAX_INLINE_SIZE + 400_000)
    chash = hashlib.sha256(body).hexdigest()
    cid = "QmStreamCorrupt"
    src = tmp_path / "a_staged.bin"
    src.write_bytes(body)
    a, b = _make_provider("node_a"), _make_provider("node_b")
    a.content_publisher = _staged_publisher(src)
    a.register_local_content(cid=cid, size_bytes=len(body), content_hash=chash, filename="big.bin")

    # Link that corrupts the first content_response frame A sends.
    corrupted = {"done": False}

    async def _a_send(target_id, msg):
        if msg.payload.get("subtype") == "content_response" and not corrupted["done"]:
            import base64
            raw = bytearray(base64.b64decode(msg.payload["data_b64"]))
            raw[0] ^= 0xFF
            msg.payload["data_b64"] = base64.b64encode(bytes(raw)).decode()
            corrupted["done"] = True
        peer = MagicMock(); peer.peer_id = "node_a"
        await b._handle_direct_message(msg, peer)
        return True

    a.transport.send_to_peer = _a_send
    b.content_discovery.announce_content(
        cid, "node_a",
        ContentAnnouncement(cid=cid, size=len(body), content_type="application/octet-stream",
                            content_hash=chash, provider_id="node_a", filename="big.bin"),
    )
    dest = tmp_path / "b_received.bin"
    result = await b.request_content_to_file(cid, dest, timeout=10, verify_hash=True)
    assert result is None, "corrupted stream must be rejected"
    assert not dest.exists(), "partial/corrupt file must be removed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── sp1457: _verify_file_cid must anchor a canonical ContentHash CID (regression) ──
# The streaming receive path's CID anchor read a dead attribute (`algorithm` vs the real
# `algorithm_id`), so the branch was inert: substituted bytes fell through to the weaker
# gossip `expected_hash` check (which a malicious provider also controls) or to accept-all.
# These exercise the ContentHash-CID branch the Qm…-CID tests above never reach.

@pytest.mark.asyncio
async def test_verify_file_cid_rejects_substituted_bytes_for_contenthash_cid(tmp_path):
    from prsm.storage import ContentHash
    provider = _make_provider("node_v")
    real = b"the genuine paid dataset bytes " * 200
    cid = ContentHash.from_data(real).hex()          # canonical sha256 ContentHash CID
    evil = b"attacker-substituted content!! " * 200
    got = tmp_path / "got.bin"
    got.write_bytes(evil)
    # The attacker also advertises a gossip hash matching the evil bytes.
    evil_gossip = hashlib.sha256(evil).hexdigest()
    ok = await provider._verify_file_cid(cid, got, True, expected_hash=evil_gossip)
    assert ok is False                               # RED before the fix (accepted via gossip)


@pytest.mark.asyncio
async def test_verify_file_cid_accepts_matching_bytes_for_contenthash_cid(tmp_path):
    from prsm.storage import ContentHash
    provider = _make_provider("node_v")
    body = b"genuine addressed content " * 500
    cid = ContentHash.from_data(body).hex()
    got = tmp_path / "got.bin"
    got.write_bytes(body)
    assert await provider._verify_file_cid(cid, got, True) is True


@pytest.mark.asyncio
async def test_verify_file_cid_anchors_on_the_cids_own_algorithm_sha3(tmp_path):
    # The anchor must hash with the CID's OWN algorithm — a SHA3_256 CID anchors on sha3_256,
    # not sha256 (the old code only ever computed sha256, so a SHA3 CID could never anchor).
    from prsm.storage import ContentHash
    from prsm.storage.models import AlgorithmID
    provider = _make_provider("node_v")
    body = b"sha3-256 addressed content " * 500
    cid = ContentHash.from_data(body, AlgorithmID.SHA3_256).hex()
    good = tmp_path / "good.bin"
    good.write_bytes(body)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"different bytes " * 500)
    assert await provider._verify_file_cid(cid, good, True) is True
    assert await provider._verify_file_cid(cid, bad, True) is False


@pytest.mark.asyncio
async def test_verify_file_cid_anchors_bittorrent_infohash_cid(tmp_path):
    # sp1457 — a 40-hex BitTorrent-infohash CID is now anchored on the streaming path by
    # re-deriving the file's infohash (previously a "documented follow-on" → fell through to
    # the weaker gossip check, so substituted bytes with a matching gossip hash were accepted).
    from prsm.core.torrent_infohash import compute_v1_infohash_single_file_from_path
    provider = _make_provider("node_v")
    real = b"tier-a public dataset bytes " * 500
    good = tmp_path / "good.bin"
    good.write_bytes(real)
    cid = compute_v1_infohash_single_file_from_path(good)     # 40-hex infohash CID
    assert len(cid) == 40 and all(c in "0123456789abcdef" for c in cid)
    assert await provider._verify_file_cid(cid, good, True) is True    # matching bytes accepted

    evil = b"attacker-substituted bytes!! " * 500
    bad = tmp_path / "bad.bin"
    bad.write_bytes(evil)
    evil_gossip = hashlib.sha256(evil).hexdigest()
    # Substituted file REJECTED even though the gossip hash matches the evil bytes.
    assert await provider._verify_file_cid(cid, bad, True, expected_hash=evil_gossip) is False


@pytest.mark.asyncio
async def test_publish_from_path_then_fetch_roundtrip_end_to_end(tmp_path, monkeypatch):
    """★ sp1457 CAPSTONE — the full large-content pipeline composes end to end: publish a large
    file via LocalContentPublisher.publish_from_path on A → fetch it on B via
    request_content_to_file by the REAL infohash CID, byte-identical and CID-verified (this
    exercises the sp1457 streaming infohash anchor on a LIVE fetch, not just in isolation).
    Proves the seam between publish_from_path's staging and the provider serve/fetch path."""
    monkeypatch.setenv("PRSM_MAX_CHUNKED_TRANSFER_BYTES", str(GATEWAY_FETCH_CHUNK_BYTES))
    from prsm.node.local_content_publisher import LocalContentPublisher

    body = os.urandom(MAX_INLINE_SIZE + 500_000)   # > inline, forces CHUNKED streaming
    src = tmp_path / "dataset.bin"
    src.write_bytes(body)

    a, b = _make_provider("node_a"), _make_provider("node_b")
    publisher = LocalContentPublisher(tmp_path / "stage_a", node_id="node_a")
    a.content_publisher = publisher
    published = await publisher.publish_from_path(src, provenance_id="prov-1")
    cid = published.torrent_infohash                # the REAL v1 infohash CID
    assert len(cid) == 40 and all(c in "0123456789abcdef" for c in cid)

    a.register_local_content(
        cid=cid, size_bytes=published.manifest.total_size,
        content_hash=published.staged_path.name, filename="dataset.bin")

    link = _InProcessLink(a, b)
    b.content_discovery.announce_content(
        cid, "node_a",
        ContentAnnouncement(cid=cid, size=len(body), content_type="application/octet-stream",
                            content_hash=published.staged_path.name, provider_id="node_a",
                            filename="dataset.bin"))
    assert not b.has_local_content(cid)

    dest = tmp_path / "b_received.bin"
    result = await b.request_content_to_file(cid, dest, timeout=10, verify_hash=True)

    assert result is not None, "B must receive + CID-verify the streamed file"
    assert dest.read_bytes() == body, "round-tripped file must be byte-identical"
    assert all(f.transfer_mode == TransferMode.CHUNKED for f in link.frames_from_a)
