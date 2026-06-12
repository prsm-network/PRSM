"""Sprint 1076 — bind fetched content to a BitTorrent-infohash CID (content-substitution hardening).

The sp1003 ContentProvider._cid_anchor_rejects re-hashes fetched bytes against a
ContentHash (sha256-family) CID, but DEFERS for a 40-hex BitTorrent v1 infohash CID
(documented residual gap) — leaving only the attacker-controllable gossip
expected_hash. Since the v1 infohash IS recomputable from the bytes (sp1070), this
adds a v1-infohash anchor so a 40-hex CID is bound to its bytes too: a malicious
provider can no longer return substituted bytes for an infohash CID. Both INLINE and
CHUNKED fetches funnel through _cid_anchor_rejects (the chunked reassembly re-verifies
via _request_from_provider), so this closes the gap on both.
"""
from __future__ import annotations

import hashlib

from prsm.core.torrent_infohash import compute_v1_infohash_single_file
from prsm.node.content_provider import ContentProvider


def _infohash_cid(data: bytes) -> str:
    return compute_v1_infohash_single_file(data, hashlib.sha256(data).hexdigest())


def test_infohash_anchor_accepts_authentic_bytes():
    data = b"authentic public content bytes"
    cid = _infohash_cid(data)
    assert ContentProvider._cid_anchor_rejects(cid, data) is False   # matches → don't reject


def test_infohash_anchor_rejects_substituted_bytes():
    data = b"authentic public content bytes"
    cid = _infohash_cid(data)
    # a malicious provider returns different bytes for the same infohash CID
    assert ContentProvider._cid_anchor_rejects(cid, b"EVIL substituted payload") is True


def test_infohash_anchor_rejects_truncated_or_extended():
    data = b"x" * 5000
    cid = _infohash_cid(data)
    assert ContentProvider._cid_anchor_rejects(cid, data[:-1]) is True   # truncated
    assert ContentProvider._cid_anchor_rejects(cid, data + b"!") is True  # extended


def test_infohash_anchor_works_for_a_tier_bc_bundle_blob():
    """A Tier-B/C bundle's CID is the infohash of the BUNDLE; the anchor binds the
    fetched bundle bytes to it."""
    from prsm.node.artifact_bundle import bundle_artifacts
    blob = bundle_artifacts(b"manifest", b"[]", [b"shard0", b"shard1"])
    cid = _infohash_cid(blob)
    assert ContentProvider._cid_anchor_rejects(cid, blob) is False
    assert ContentProvider._cid_anchor_rejects(cid, blob + b"x") is True


def test_non_recomputable_cid_still_defers():
    """A CID that is neither a canonical ContentHash nor a 40-hex infohash → no
    opinion (defer to the gossip-hash check), unchanged from sp1003."""
    assert ContentProvider._cid_anchor_rejects("not-a-real-cid", b"anything") is False
    assert ContentProvider._cid_anchor_rejects("abc123", b"anything") is False
    # 40 chars but not hex → not an infohash shape → defer
    assert ContentProvider._cid_anchor_rejects("z" * 40, b"anything") is False


def test_contenthash_anchor_unchanged():
    """The sp1003 ContentHash path must still reject substitution + accept authentic."""
    from prsm.storage import ContentHash
    from prsm.storage.models import AlgorithmID
    h = ContentHash.from_data(b"the original bytes", AlgorithmID.SHA256)
    cid = h.hex()
    assert ContentProvider._cid_anchor_rejects(cid, b"the original bytes") is False
    assert ContentProvider._cid_anchor_rejects(cid, b"tampered bytes") is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
