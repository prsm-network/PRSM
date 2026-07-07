"""Sprint 1399 — bootstrap server stores a peer's advertise "host:port" as host-only + port.

A compute node with PRSM_ADVERTISE_ADDRESS="host:port" (the documented form) sends BOTH that address
and a port field. The server stored the whole "host:port" in PeerInfo.address (meant to be host-only)
alongside the port field, so /peers rendered "host:port:port" (live: us as 159.203.129.218:9002:9002).
Now the port is split off — address is host-only, port authoritative (and the advertise port wins for
NAT/port-forward). Dialing already deduped (sp1026); this fixes the stored/displayed value.
"""
from unittest.mock import AsyncMock

import pytest

from prsm.bootstrap.config import BootstrapConfig
from prsm.bootstrap.server import BootstrapServer, _split_host_port


def test_split_host_port_variants():
    assert _split_host_port("1.2.3.4:9002") == ("1.2.3.4", 9002)
    assert _split_host_port("host.example.com:8765") == ("host.example.com", 8765)
    assert _split_host_port("[::1]:9002") == ("[::1]", 9002)          # bracketed IPv6 + port
    assert _split_host_port("1.2.3.4") == ("1.2.3.4", None)           # no port
    assert _split_host_port("::1") == ("::1", None)                    # bare IPv6, multiple colons
    assert _split_host_port("[2001:db8::1]") == ("[2001:db8::1]", None)
    assert _split_host_port("host:notaport") == ("host:notaport", None)  # non-numeric tail


async def _register(peer_id, port, address, ip):
    server = BootstrapServer(BootstrapConfig())
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.remote_address = (ip, 54321)
    await server._handle_register(
        ws,
        {"type": "register", "peer_id": peer_id, "port": port,
         "address": address, "capabilities": ["compute"], "version": "t"},
        ip)
    return server.peers[peer_id]


@pytest.mark.asyncio
async def test_advertise_host_port_stored_host_only():
    peer = await _register("us", 9002, "159.203.129.218:9002", "159.203.129.218")
    assert peer.address == "159.203.129.218"        # host-only, NOT host:port
    assert peer.port == 9002
    assert f"{peer.address}:{peer.port}" == "159.203.129.218:9002"   # no doubling on render


@pytest.mark.asyncio
async def test_host_only_advertise_unchanged():
    peer = await _register("sfo", 9001, "146.190.175.239", "146.190.175.239")
    assert peer.address == "146.190.175.239"
    assert peer.port == 9001


@pytest.mark.asyncio
async def test_advertise_port_wins_for_nat():
    # public 203.0.113.5:30000 advertised, internal listen port 9002 → dial the public port
    peer = await _register("nat", 9002, "203.0.113.5:30000", "203.0.113.5")
    assert peer.address == "203.0.113.5" and peer.port == 30000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
