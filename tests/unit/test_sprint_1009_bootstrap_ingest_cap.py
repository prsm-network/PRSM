"""Sprint 1009 — cap bootstrap-client peer ingestion (finding 14).

The P2P-substrate integrity hunt (workflow wbu7u2ftm, finding 14) noted that the
bootstrap-client hydration loop fed every peer the bootstrap returned into
known_peers with NO length cap, bypassing the sp1005 cap that guards the
announce + PEX paths. sp1006 closed the MITM angle (the bootstrap is now
TLS-verified), but a COMPROMISED bootstrap server could still flood known_peers
without bound (memory DoS) on the cold-start path.

Fix: route bootstrap ingestion through _ingest_bootstrap_peers, which applies the
same PRSM_MAX_KNOWN_PEERS cap (an existing peer always refreshes; new ids beyond
the cap are dropped). Behavior is otherwise unchanged (same PeerInfo fields,
including the sp838 relayed hardware_profile).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from prsm.node.discovery import PeerDiscovery
from prsm.node.identity import generate_node_identity


def _discovery():
    t = MagicMock()
    t.identity = generate_node_identity("self")
    t.on_message = MagicMock()
    return PeerDiscovery(t)


def _bp(peer_id, hardware_profile=None):
    return SimpleNamespace(
        peer_id=peer_id, address="5.5.5.5", port=9001,
        capabilities=["inference"], hardware_profile=hardware_profile,
    )


def test_bootstrap_ingest_respects_cap(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_KNOWN_PEERS", "3")
    pd = _discovery()
    pd._ingest_bootstrap_peers([_bp(f"peer{i}") for i in range(20)])
    assert len(pd.known_peers) <= 3


def test_bootstrap_ingest_keeps_legit_peers(monkeypatch):
    monkeypatch.delenv("PRSM_MAX_KNOWN_PEERS", raising=False)
    pd = _discovery()
    n = pd._ingest_bootstrap_peers([_bp(f"peer{i}") for i in range(5)])
    assert n == 5
    assert len(pd.known_peers) == 5


def test_bootstrap_ingest_propagates_hardware_profile():
    pd = _discovery()
    hw = {"tflops_fp16": 7.5, "memory_gb": 12.0}
    pd._ingest_bootstrap_peers([_bp("p1", hardware_profile=hw)])
    assert pd.known_peers["p1"].hardware_profile == hw
    assert pd.known_peers["p1"].address == "5.5.5.5:9001"


def test_bootstrap_ingest_skips_self():
    pd = _discovery()
    self_id = pd.transport.identity.node_id
    pd._ingest_bootstrap_peers([_bp(self_id), _bp("other")])
    assert self_id not in pd.known_peers
    assert "other" in pd.known_peers
