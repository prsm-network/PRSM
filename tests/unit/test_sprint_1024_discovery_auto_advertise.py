"""Sprint 1024 — discovery auto-advertise from the bootstrap-observed address.

Completes Tier-1 gap-1. The F14 co-location loopback rewrite (sp781) + the gossip
announce only worked when PRSM_ADVERTISE_ADDRESS was set manually. sp1023 made the
bootstrap server report each node its observed address; this sprint captures that in
PeerDiscovery and uses it as the own-advertise FALLBACK so co-located / NAT'd nodes
get a routable advertise value with ZERO manual config. The env var still wins.

The single source of the node's own-advertise value is PeerDiscovery._own_advertise();
all three consumption sites (announce_self + the two dial-site rewrites) route through
it. These tests pin its precedence (env > observed > None) and that the sites use it.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from prsm.node.discovery import PeerDiscovery


def _pd() -> PeerDiscovery:
    transport = MagicMock()
    transport.identity.node_id = "self-node"
    transport.port = 9001
    return PeerDiscovery(transport=transport, bootstrap_nodes=["wss://x:8765"])


def test_own_advertise_none_when_nothing_set(monkeypatch):
    monkeypatch.delenv("PRSM_ADVERTISE_ADDRESS", raising=False)
    pd = _pd()
    assert pd._own_advertise() is None


def test_own_advertise_uses_observed_when_env_unset(monkeypatch):
    monkeypatch.delenv("PRSM_ADVERTISE_ADDRESS", raising=False)
    pd = _pd()
    pd._observed_advertise = "203.0.113.5:9001"
    assert pd._own_advertise() == "203.0.113.5:9001"


def test_own_advertise_env_takes_precedence_over_observed(monkeypatch):
    monkeypatch.setenv("PRSM_ADVERTISE_ADDRESS", "198.51.100.7")
    pd = _pd()
    pd._observed_advertise = "203.0.113.5:9001"
    assert pd._own_advertise() == "198.51.100.7"


def test_observed_advertise_defaults_none(monkeypatch):
    monkeypatch.delenv("PRSM_ADVERTISE_ADDRESS", raising=False)
    assert _pd()._observed_advertise is None


def test_all_advertise_sites_route_through_unified_helper():
    """announce_self + both dial-site rewrites must take their own-advertise value
    from _own_advertise() (so the observed-address fallback reaches every site), not
    from the raw env-only resolver."""
    for meth in ("announce_self", "_auto_dial_sweep", "maintain_connections"):
        src = inspect.getsource(getattr(PeerDiscovery, meth))
        assert "_own_advertise(" in src, (
            f"{meth} must route own-advertise through _own_advertise()"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
