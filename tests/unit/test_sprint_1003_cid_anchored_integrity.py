"""Sprint 1003 — CID-anchored content-substitution defense.

The content-data-plane integrity hunt (workflow w8gvkhelm) confirmed that
the remote INLINE/GATEWAY fetch lane verified returned bytes against
``expected_hash`` = ``ContentRecord.content_hash``, which is populated from
an UNAUTHENTICATED ``GOSSIP_CONTENT_ADVERTISE`` (findings 1, 2, 4):

  - Finding 2/4 (HIGH): an attacker advertises a CID with
    ``content_hash = sha256(evil_bytes)`` (first-writer-wins locks it in),
    becomes a routed provider, and serves ``evil_bytes``. The check
    ``sha256(evil_bytes) == expected_hash`` PASSES → the victim accepts
    attacker-chosen bytes under a CID it trusts.
  - Finding 1 (HIGH): a replica advertises ``content_hash=""`` →
    ``if expected_hash:`` is falsy → the hash check is SKIPPED entirely →
    bytes accepted unverified.

Root cause: ``expected_hash`` is attacker-controllable, so it is not a
trustworthy integrity anchor. The CID itself, however, IS trustworthy — the
requester chose it. For a ``ContentHash``-shaped CID (algorithm-prefixed
content address, e.g. 66-char SHA-256) the served bytes can be re-hashed and
checked against the CID directly, immune to gossip poisoning. The
ContentStore is content-addressed, so legit content always re-hashes to its
CID (cipher or plaintext — the CID addresses whatever bytes were stored).

Fix (``_request_from_provider``): when the CID is an unambiguous
``ContentHash`` (round-trips through ``from_hex`` AND has the algorithm's
canonical digest length), verify the returned bytes against the CID and
reject on mismatch — regardless of the gossip ``content_hash``. A 40/64-char
BitTorrent infohash is NOT a recomputable content address from inline bytes;
that residual gap is documented in
``docs/2026-06-04-content-data-plane-trust-anchors.md`` and falls back to the
(weaker) gossip-hash check.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from prsm.node.content_provider import (
    ContentProvider,
    ContentResponseMessage,
    ContentStatus,
    TransferMode,
)
from prsm.storage import ContentHash
from prsm.storage.models import AlgorithmID


def _make_provider():
    identity = MagicMock()
    identity.node_id = "test-node-id"
    transport = MagicMock()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    return ContentProvider(identity=identity, transport=transport, gossip=gossip)


def _wire_inline_response(p, cid, served_bytes):
    """Make the provider's next request resolve to an INLINE response that
    serves ``served_bytes`` for ``cid``."""
    resp = ContentResponseMessage(
        request_id="ignored",
        cid=cid,
        status=ContentStatus.FOUND,
        data=served_bytes,
        size=len(served_bytes),
        transfer_mode=TransferMode.INLINE,
    )

    async def _send(_peer_id, _msg):
        for _rid, fut in list(p._pending_requests.items()):
            if not fut.done():
                fut.set_result(resp)
        return True

    p.transport.send_to_peer = _send


@pytest.mark.asyncio
async def test_contenthash_cid_accepts_matching_bytes_even_without_gossip_hash():
    """A ContentHash CID + the genuine bytes verify against the CID itself,
    so retrieval succeeds even when the gossip content_hash is empty
    (closes finding 1's empty-hash skip for content-addressed CIDs)."""
    content = b"the genuine content payload"
    cid = ContentHash.from_data(content, AlgorithmID.SHA256).hex()
    p = _make_provider()
    _wire_inline_response(p, cid, content)

    out = await p._request_from_provider(
        cid, "prov", timeout=5.0, expected_hash=None,
    )
    assert out == content


@pytest.mark.asyncio
async def test_contenthash_cid_rejects_substituted_bytes_despite_matching_gossip_hash():
    """THE substitution test. Attacker advertises content_hash=sha256(evil)
    and serves evil bytes; the legacy gossip-hash check passes. The
    CID-anchor must catch it: evil bytes do not hash to the trusted CID."""
    genuine = b"the genuine content payload"
    cid = ContentHash.from_data(genuine, AlgorithmID.SHA256).hex()
    evil = b"attacker-chosen substituted bytes"
    poisoned_gossip_hash = hashlib.sha256(evil).hexdigest()  # matches evil

    p = _make_provider()
    _wire_inline_response(p, cid, evil)

    out = await p._request_from_provider(
        cid, "prov", timeout=5.0, expected_hash=poisoned_gossip_hash,
    )
    # Pre-fix: gossip-hash check passes (sha256(evil)==poisoned) → returns evil.
    # Post-fix: CID-anchor rejects (evil does not hash to cid) → None.
    assert out is None


@pytest.mark.asyncio
async def test_contenthash_cid_rejects_substituted_bytes_with_empty_gossip_hash():
    """Substitution under an empty gossip hash (the replica-ad case) is also
    caught by the CID-anchor."""
    genuine = b"genuine bytes here"
    cid = ContentHash.from_data(genuine, AlgorithmID.SHA256).hex()
    evil = b"totally different bytes"

    p = _make_provider()
    _wire_inline_response(p, cid, evil)

    out = await p._request_from_provider(
        cid, "prov", timeout=5.0, expected_hash="",
    )
    assert out is None


@pytest.mark.asyncio
async def test_sha3_contenthash_cid_anchored():
    """The anchor is algorithm-agile: a SHA3-256 ContentHash CID is also
    verified against the genuine bytes."""
    content = b"sha3 addressed content"
    cid = ContentHash.from_data(content, AlgorithmID.SHA3_256).hex()
    p = _make_provider()
    _wire_inline_response(p, cid, content)

    out = await p._request_from_provider(
        cid, "prov", timeout=5.0, expected_hash=None,
    )
    assert out == content


@pytest.mark.asyncio
async def test_bt_infohash_cid_authentic_bytes_served_no_false_reject():
    """sp1076 — a 40-char BT v1 infohash IS recomputable from the bytes now, so the
    CID-anchor verifies it. AUTHENTIC content (whose infohash == the CID) must still
    be served (no false reject)."""
    from prsm.core.torrent_infohash import compute_v1_infohash_single_file
    body = b"legit BT-published content"
    bt_infohash = compute_v1_infohash_single_file(
        body, hashlib.sha256(body).hexdigest())
    gossip_hash = hashlib.sha256(body).hexdigest()

    p = _make_provider()
    _wire_inline_response(p, bt_infohash, body)

    out = await p._request_from_provider(
        bt_infohash, "prov", timeout=5.0, expected_hash=gossip_hash,
    )
    assert out == body


@pytest.mark.asyncio
async def test_bt_infohash_substitution_now_rejected():
    """sp1076 — the sp1003 residual gap is now CLOSED. Bytes that do NOT re-derive to
    the BT v1 infohash CID are rejected, even with an empty/attacker-controlled gossip
    hash (previously these were served unverified)."""
    from prsm.core.torrent_infohash import compute_v1_infohash_single_file
    real = b"the authentic BT-published content"
    bt_infohash = compute_v1_infohash_single_file(
        real, hashlib.sha256(real).hexdigest())
    evil = b"substituted bytes advertised under the same CID"

    p = _make_provider()
    _wire_inline_response(p, bt_infohash, evil)

    out = await p._request_from_provider(
        bt_infohash, "prov", timeout=5.0, expected_hash="",
    )
    assert out is None  # gap CLOSED — substituted bytes rejected by the infohash anchor
