"""Sprint 1097 — folds the Domain-02 cross-cutting review findings (libp2p hardening):
- F1 (HIGH): port the WS replay defense (sp941 announce_time) to libp2p capability ingest.
- F2 (HIGH): port the WS table cap (_max_known_peers) + add a per-CID shard-providers cap.
- F5 (MEDIUM): port the sp1026 _join_bootstrap_address double-port fix to the libp2p hydrate.
- F4 (MEDIUM): receiver-side reassembly ceiling in ShardAssembler.
"""
from __future__ import annotations

import hashlib
import time
import types
from unittest.mock import MagicMock

import pytest

from prsm.node.discovery import _attest_announce_payload
from prsm.node.identity import generate_node_identity
from prsm.node.libp2p_discovery import Libp2pDiscovery
from prsm.node.shard_streaming import ShardAssembler, ShardChunk, ShardManifest, StreamingError


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _transport(node_id="local"):
    t = MagicMock()
    t.identity = generate_node_identity(node_id)
    t.port = 9001
    return t


def _cap_announce(identity, *, announce_time, caps=None, nonce="n"):
    data = {"node_id": identity.node_id, "capabilities": caps or ["inference"],
            "supported_backends": [], "gpu_available": True, "startup_timestamp": 1.0,
            "announce_time": announce_time, "nonce": nonce}
    _attest_announce_payload(identity, data, nonce)
    return data


# ── F1: replay defense on capability ingest ─────────────────────────────────────

def test_stale_replay_does_not_refresh_capability_entry():
    d = Libp2pDiscovery(_transport())
    peer = generate_node_identity("peer")
    # accept a fresh announce (t=100) advertising GPU
    _run(d._on_capability("capability_announce",
                          _cap_announce(peer, announce_time=100.0, caps=["inference", "gpu"], nonce="a"),
                          peer.node_id))
    entry = d._capability_index[peer.node_id]
    last_seen_1 = entry.last_seen
    assert entry.last_announce_time == 100.0
    time.sleep(0.01)
    # replay an OLDER captured announce (t=50) with different caps → must be rejected
    _run(d._on_capability("capability_announce",
                          _cap_announce(peer, announce_time=50.0, caps=["storage"], nonce="b"),
                          peer.node_id))
    entry = d._capability_index[peer.node_id]
    assert entry.last_announce_time == 100.0          # unchanged — replay rejected
    assert "gpu" in [c.lower() for c in entry.capabilities]  # caps NOT overwritten
    assert entry.last_seen == last_seen_1             # last_seen NOT refreshed


def test_newer_announce_is_accepted():
    d = Libp2pDiscovery(_transport())
    peer = generate_node_identity("peer")
    _run(d._on_capability("capability_announce",
                          _cap_announce(peer, announce_time=100.0, nonce="a"), peer.node_id))
    _run(d._on_capability("capability_announce",
                          _cap_announce(peer, announce_time=200.0, caps=["storage"], nonce="b"),
                          peer.node_id))
    assert d._capability_index[peer.node_id].last_announce_time == 200.0


# ── F2: capability index cap + shard providers cap ───────────────────────────────

def test_capability_index_bounded_under_flood(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_KNOWN_PEERS", "3")
    d = Libp2pDiscovery(_transport())
    for i in range(6):   # six distinct (freshly-keyed) authenticated announces
        p = generate_node_identity(f"peer{i}")
        _run(d._on_capability("capability_announce",
                              _cap_announce(p, announce_time=100.0 + i, nonce=f"n{i}"),
                              p.node_id))
    assert len(d._capability_index) == 3   # capped — honest table not flooded unbounded


def test_shard_providers_bounded(monkeypatch):
    import prsm.node.libp2p_discovery as mod
    monkeypatch.setattr(mod, "_MAX_SHARD_PROVIDERS", 2)
    d = Libp2pDiscovery(_transport())
    cid = "QmFlood"
    for i in range(5):
        p = generate_node_identity(f"sp{i}")
        data = {"cid": cid, "node_id": p.node_id, "announce_time": 100.0 + i, "nonce": f"s{i}"}
        _attest_announce_payload(p, data, f"s{i}")
        _run(d._on_shard_available("shard_available", data, p.node_id))
    assert len(d._shard_cache[cid]) == 2   # capped


# ── F5: bootstrap-relay address join (no double-port) ────────────────────────────

def test_hydrate_no_double_port():
    d = Libp2pDiscovery(_transport())
    bp = types.SimpleNamespace(
        peer_id="p" * 32, address="1.2.3.4:9001", port=9001,
        capabilities=["inference"], hardware_profile=None, announce_credential=None)
    d._hydrate_peers_from_bootstrap([bp])
    addr = d._capability_index["p" * 32].address
    assert addr == "1.2.3.4:9001"          # NOT "1.2.3.4:9001:9001"


def test_hydrate_bare_host_gets_port():
    d = Libp2pDiscovery(_transport())
    bp = types.SimpleNamespace(
        peer_id="q" * 32, address="5.6.7.8", port=9001,
        capabilities=[], hardware_profile=None, announce_credential=None)
    d._hydrate_peers_from_bootstrap([bp])
    assert d._capability_index["q" * 32].address == "5.6.7.8:9001"


# ── F4: shard reassembly ceiling ─────────────────────────────────────────────────

def _manifest(payload: bytes, chunk_bytes=4):
    n = max(1, (len(payload) + chunk_bytes - 1) // chunk_bytes) if payload else 0
    return ShardManifest(shard_id="s1", payload_sha256=hashlib.sha256(payload).hexdigest(),
                         payload_bytes=len(payload), total_chunks=n, chunk_bytes=chunk_bytes)


def test_oversized_manifest_refused_before_buffering(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_SHARD_REASSEMBLY_BYTES", "100")
    m = ShardManifest(shard_id="s1", payload_sha256="00", payload_bytes=10_000,
                      total_chunks=1, chunk_bytes=10_000)
    with pytest.raises(StreamingError):
        ShardAssembler(m)


def test_sender_exceeding_declared_payload_rejected_midstream(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_SHARD_REASSEMBLY_BYTES", "1000000")
    # manifest DECLARES 4 bytes but the sender streams more
    m = ShardManifest(shard_id="s1", payload_sha256="00", payload_bytes=4,
                      total_chunks=10, chunk_bytes=4)
    a = ShardAssembler(m)
    data = b"AAAA"
    a.feed(ShardChunk(shard_id="s1", sequence=0, data=data,
                      chunk_sha256=hashlib.sha256(data).hexdigest()))
    with pytest.raises(StreamingError):   # second chunk pushes past the declared 4 bytes
        a.feed(ShardChunk(shard_id="s1", sequence=1, data=data,
                          chunk_sha256=hashlib.sha256(data).hexdigest()))


def test_normal_reassembly_still_works():
    payload = b"hello world payload"
    m = _manifest(payload, chunk_bytes=4)
    a = ShardAssembler(m)
    for seq in range(m.total_chunks):
        d = payload[seq * 4:(seq + 1) * 4]
        a.feed(ShardChunk(shard_id="s1", sequence=seq, data=d,
                          chunk_sha256=hashlib.sha256(d).hexdigest()))
    assert a.finalize() == payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
