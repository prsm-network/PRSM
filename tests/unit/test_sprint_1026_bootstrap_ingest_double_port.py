"""Sprint 1026 — bootstrap peer ingestion must not double-append the port.

Tier-1 bench bug (live-caught): an operator sets PRSM_ADVERTISE_ADDRESS=<ip>:9001
(the documented sp566 host:port format). The bootstrap server stores that verbatim
as the peer's `address`. _ingest_bootstrap_peers (sp1009) then did
`f"{bp.address}:{bp.port}"`, producing "ip:9001:9001" — an undialable address that
broke cross-host discovery (the remote operator showed up in `known` but could never
be dialed). The fix joins host+port only when the address doesn't already carry one.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.discovery import PeerDiscovery, _join_bootstrap_address


def _pd() -> PeerDiscovery:
    transport = MagicMock()
    transport.identity.node_id = "self-node"
    transport.port = 9001
    return PeerDiscovery(transport=transport, bootstrap_nodes=["wss://x:8765"])


def _bp(peer_id, address, port, caps=("compute",)):
    bp = MagicMock()
    bp.peer_id = peer_id
    bp.address = address
    bp.port = port
    bp.capabilities = list(caps)
    bp.hardware_profile = None
    return bp


# ── The join helper (load-bearing logic) ──────────────────────────────────────

@pytest.mark.parametrize("address,port,expected", [
    ("170.9.22.192:9001", 9001, "170.9.22.192:9001"),   # already host:port → unchanged
    ("159.203.129.218", 9001, "159.203.129.218:9001"),  # host-only → append
    ("bootstrap-us.prsm-network.com", 8765, "bootstrap-us.prsm-network.com:8765"),
    ("bootstrap-us.prsm-network.com:8765", 9001, "bootstrap-us.prsm-network.com:8765"),
    ("[::1]", 9001, "[::1]:9001"),                       # bracketed IPv6, no port → append
    ("[::1]:9001", 9001, "[::1]:9001"),                  # bracketed IPv6 w/ port → unchanged
])
def test_join_bootstrap_address(address, port, expected):
    assert _join_bootstrap_address(address, port) == expected


# ── Integration through _ingest_bootstrap_peers ───────────────────────────────

def test_ingest_does_not_double_append_port_when_address_has_one():
    """The live bug: PRSM_ADVERTISE_ADDRESS=host:port stored on the bootstrap →
    must NOT become host:port:port."""
    pd = _pd()
    pd._ingest_bootstrap_peers([_bp("peerA", "170.9.22.192:9001", 9001)])
    assert pd.known_peers["peerA"].address == "170.9.22.192:9001"


def test_ingest_appends_port_when_address_is_host_only():
    """Back-compat: a host-only address (e.g. the bootstrap client_ip fallback)
    still gets the port appended exactly once."""
    pd = _pd()
    pd._ingest_bootstrap_peers([_bp("peerB", "159.203.129.218", 9001)])
    assert pd.known_peers["peerB"].address == "159.203.129.218:9001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
