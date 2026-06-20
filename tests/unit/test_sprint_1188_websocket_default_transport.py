"""Sprint 1188 — default the transport to the hardened WebSocket path (day-one-live #7).

config defaulted transport_backend to "libp2p", but the LIVE fleet runs
PRSM_TRANSPORT_BACKEND=websocket (reference_live_fleet_ips), the bootstrap server is
WebSocket, and the libp2p data-plane carries none of the WS path's origin-auth (sp1010
Residual A) — so a fresh user who didn't set the env got the WRONG transport (doesn't
match the fleet) on the unauthenticated path. Flipping the default to "websocket" aligns
with what operators already run + the hardened path; libp2p stays available via explicit
PRSM_TRANSPORT_BACKEND=libp2p.
"""
from __future__ import annotations

import pytest

from prsm.node.config import NodeConfig


def test_default_transport_is_websocket(monkeypatch):
    monkeypatch.delenv("PRSM_TRANSPORT_BACKEND", raising=False)
    assert NodeConfig().transport_backend == "websocket"


def test_explicit_libp2p_env_override_still_honored(monkeypatch):
    monkeypatch.setenv("PRSM_TRANSPORT_BACKEND", "libp2p")
    assert NodeConfig().transport_backend == "libp2p"


def test_explicit_websocket_env_override(monkeypatch):
    monkeypatch.setenv("PRSM_TRANSPORT_BACKEND", "websocket")
    assert NodeConfig().transport_backend == "websocket"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
