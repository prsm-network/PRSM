"""Sprint 593 (Phase 2B) — address-resolver helper.

Phase 2 RPC chain executor wiring requires an
``AddressResolver = Callable[[str], str]`` that maps a chain
stage's ``node_id`` to a transport address.

Sprint 593 ships ``build_address_resolver(node)`` which queries
``node.transport.peers[node_id].address``. Raises typed
``PeerNotFound`` when the node isn't currently in transport.peers.

Phase 2A added the SendMessage scaffolding. Phase 2C wires the
async-to-sync bridge. Phase 2D wires everything into the executor.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_module_exposes_build_address_resolver_and_peer_not_found():
    import prsm.node.chain_executor_adapters as m
    assert hasattr(m, "build_address_resolver")
    assert hasattr(m, "PeerNotFound")
    assert issubclass(m.PeerNotFound, RuntimeError)


def test_resolver_returns_peer_id_for_known_node_id():
    """Known node_id in transport.peers → resolves successfully.

    sp695 F45 / sp596: the resolver returns the node_id (peer_id) itself, NOT
    peer.address — the send_message adapter calls transport.send_to_peer(addr)
    which looks the peer up by peer_id, not network address. The peer-existence
    check still gates it (unknown node_id raises PeerNotFound, below)."""
    from prsm.node.chain_executor_adapters import build_address_resolver

    peer = MagicMock()
    peer.address = "1.2.3.4:9001"
    node = MagicMock()
    node.transport.peers = {"remote-node-id": peer}

    resolve = build_address_resolver(node)
    assert resolve("remote-node-id") == "remote-node-id"


def test_resolver_raises_peer_not_found_for_unknown_node_id():
    from prsm.node.chain_executor_adapters import (
        build_address_resolver, PeerNotFound,
    )
    node = MagicMock()
    node.transport.peers = {}

    resolve = build_address_resolver(node)
    with pytest.raises(PeerNotFound):
        resolve("nonexistent-node-id")


def test_resolver_error_message_includes_node_id():
    """Operators triaging chain-executor failures need to see
    WHICH node_id couldn't be resolved.
    """
    from prsm.node.chain_executor_adapters import (
        build_address_resolver, PeerNotFound,
    )
    node = MagicMock()
    node.transport.peers = {}

    resolve = build_address_resolver(node)
    try:
        resolve("abcd1234-missing")
    except PeerNotFound as exc:
        assert "abcd1234-missing" in str(exc)


def test_resolver_handles_self_node_id():
    """sp687 F34 — a stage routed to the LOCAL node resolves to its own node_id
    (the self-dispatch sentinel the send adapter detects to short-circuit to the
    local StageExecutor), NOT a PeerNotFound. Self is never in transport.peers
    (only remote connections are), so without the sentinel the whole chain would
    abort even though local execution needs no peer connection.
    """
    from prsm.node.chain_executor_adapters import build_address_resolver

    node = MagicMock()
    node.transport.peers = {}
    node.identity.node_id = "self-node-id"

    resolve = build_address_resolver(node)
    assert resolve("self-node-id") == "self-node-id"
