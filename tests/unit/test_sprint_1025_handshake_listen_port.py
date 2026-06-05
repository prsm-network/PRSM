"""Sprint 1025 — carry the declared listen port in the P2P handshake (Tier-1 gap-b).

Root cause of gap-b: when a peer dials IN to us, the transport built its
PeerConnection.address from websocket.remote_address — the OS-assigned ephemeral
SOURCE port, not the peer's listen port. That address is fine for the live inbound
connection but UNDIALABLE for a later fresh outbound dial, so a gossip-learned (or
relayed) peer that falls back to peer.address (sp570) could never be re-dialed.

Fix (transport.py, leaves discovery.py + sp570 byte-identical → cannot regress):
the dialer now includes "listen_port" in its signed MSG_HANDSHAKE payload, and the
listener builds the inbound peer's address from the OBSERVED source host (NAT-safe —
never a peer-asserted host) joined to that DECLARED listen port, validated, with a
fallback to the ephemeral source port for pre-1025 / mixed-version peers.
"""
from __future__ import annotations

import inspect

import pytest

from prsm.node.transport import (
    P2PMessage,
    MSG_HANDSHAKE,
    WebSocketTransport,
    _peer_address_from_handshake,
)
from prsm.node.identity import generate_node_identity, verify_signature


# ── The address helper (load-bearing logic) ──────────────────────────────────

def test_uses_declared_listen_port_over_ephemeral_source_port():
    addr = _peer_address_from_handshake(("203.0.113.9", 54123), {"listen_port": 9001})
    assert addr == "203.0.113.9:9001"


def test_falls_back_to_source_port_when_listen_port_absent():
    # pre-1025 / mixed-version peer omits the field → keep working (sp570 discipline)
    addr = _peer_address_from_handshake(("203.0.113.9", 54123), {})
    assert addr == "203.0.113.9:54123"


@pytest.mark.parametrize("bad", [0, 70000, -1, "9001", None, True, 1.5])
def test_invalid_listen_port_falls_back_to_source_port(bad):
    addr = _peer_address_from_handshake(("203.0.113.9", 54123), {"listen_port": bad})
    assert addr == "203.0.113.9:54123", f"invalid listen_port {bad!r} must not reach the dial path"


def test_host_is_always_the_observed_source_never_peer_asserted():
    # NAT-safe: a malicious peer cannot make us store an arbitrary host. We only
    # ever take the host from the observed TCP source; payload host/address keys
    # are ignored.
    addr = _peer_address_from_handshake(
        ("203.0.113.9", 54123),
        {"listen_port": 9001, "host": "10.0.0.1", "address": "8.8.8.8:9001"},
    )
    assert addr == "203.0.113.9:9001"


# ── Signing coverage: listen_port is inside the Ed25519-signed envelope ───────

def test_listen_port_is_covered_by_the_handshake_signature():
    ident = generate_node_identity("dialer")
    msg = P2PMessage(
        msg_type=MSG_HANDSHAKE,
        sender_id=ident.node_id,
        payload={"public_key": ident.public_key_b64, "listen_port": 9001},
    )
    msg.sign(ident)
    assert verify_signature(ident.public_key_b64, msg.to_bytes(), msg.signature)

    # Tampering the listen_port in flight must invalidate the signature.
    msg.payload["listen_port"] = 6666
    assert not verify_signature(ident.public_key_b64, msg.to_bytes(), msg.signature)


# ── The dialer advertises its listen port in the handshake ────────────────────

def test_outbound_handshake_includes_listen_port():
    """The connecting node must put its own listen port in the handshake payload so
    the listener can build a dialable address for it."""
    src = inspect.getsource(WebSocketTransport.connect_to_peer)
    assert '"listen_port": self.port' in src, (
        "connect_to_peer must include 'listen_port': self.port in the handshake payload"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
