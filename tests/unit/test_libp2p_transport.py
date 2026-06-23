"""
Unit tests for prsm.node.libp2p_transport
==========================================

All tests run WITHOUT the Go shared library — ctypes is mocked so the
test suite is pure-Python and CI-friendly.
"""

import ctypes
import types
from unittest.mock import MagicMock, patch

import pytest

from prsm.node.libp2p_transport import Libp2pTransport, Libp2pTransportError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_identity():
    """Return a lightweight mock NodeIdentity."""
    ident = MagicMock()
    ident.node_id = "abcdef1234567890abcdef1234567890"
    ident.private_key_bytes = b"\x01" * 32
    ident.public_key_bytes = b"\x02" * 32
    ident.public_key_b64 = "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI="
    ident.display_name = "test-node"
    ident.sign.return_value = "fakesig"
    return ident


def _make_mock_cdll():
    """Return a mock ctypes.CDLL with stubs for all 13 exports."""
    lib = MagicMock()
    # Each exported function needs argtypes / restype attributes that
    # _setup_ctypes will set — MagicMock allows arbitrary attribute writes.
    for name in [
        "PrsmStart", "PrsmStop", "PrsmPeerCount", "PrsmConnect",
        "PrsmFree", "PrsmPublish", "PrsmSubscribe", "PrsmUnsubscribe",
        "PrsmSend", "PrsmDHTProvide", "PrsmDHTFindProviders",
        "PrsmGetNATStatus", "PrsmPeerList",
    ]:
        getattr(lib, name)  # ensure attribute exists on mock
    return lib


@pytest.fixture
def mock_transport():
    """Yield a Libp2pTransport with a fully mocked shared library."""
    identity = _make_mock_identity()
    lib = _make_mock_cdll()
    with patch.object(Libp2pTransport, "_load_library", return_value=lib):
        transport = Libp2pTransport(identity=identity, port=9001)
    return transport


# ---------------------------------------------------------------------------
# Multiaddr translation
# ---------------------------------------------------------------------------

class TestMultiaddrTranslation:
    def test_multiaddr_translation_quic(self):
        result = Libp2pTransport._to_multiaddr("1.2.3.4:9001")
        assert result == "/ip4/1.2.3.4/udp/9001/quic-v1"

    def test_multiaddr_translation_ws(self):
        # Hostnames now use /dns4/ per libp2p multiaddr spec
        # (sprint 120 fix). /ip4/ requires dotted-quad literal.
        result = Libp2pTransport._to_multiaddr("wss://host:8765")
        assert result == "/dns4/host/tcp/8765/ws"

    def test_multiaddr_translation_ws_plain(self):
        # Same — hostname → /dns4/
        result = Libp2pTransport._to_multiaddr("ws://myhost:4000")
        assert result == "/dns4/myhost/tcp/4000/ws"

    def test_multiaddr_translation_ws_ipv4(self):
        # IPv4 literal → /ip4/ (sprint 120 distinction)
        result = Libp2pTransport._to_multiaddr("wss://1.2.3.4:8765")
        assert result == "/ip4/1.2.3.4/tcp/8765/ws"

    def test_multiaddr_passthrough(self):
        addr = "/ip4/10.0.0.1/tcp/4001/p2p/QmPeer"
        assert Libp2pTransport._to_multiaddr(addr) == addr

    def test_multiaddr_passthrough_ip6(self):
        addr = "/ip6/::1/tcp/4001"
        assert Libp2pTransport._to_multiaddr(addr) == addr


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

class TestLibraryLoading:
    def test_missing_library_raises(self):
        """Loading a nonexistent path raises Libp2pTransportError (wrapping OSError)."""
        with pytest.raises(Libp2pTransportError):
            Libp2pTransport._load_library("/nonexistent/path/libfoo.so")

    def test_missing_library_no_path_raises(self):
        """Auto-detect with no library on disk raises Libp2pTransportError."""
        with patch("prsm.node.libp2p_transport.Path") as MockPath:
            # Make the auto-detected path point nowhere
            mock_path = MagicMock()
            MockPath.return_value.resolve.return_value.parent.parent.parent = mock_path
            mock_path.__truediv__ = lambda self, x: mock_path
            mock_path.__str__ = lambda self: "/fake/libprsm_p2p.so"

            with patch("ctypes.CDLL", side_effect=OSError("not found")):
                with pytest.raises(Libp2pTransportError):
                    Libp2pTransport._load_library("")


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

class TestHandlerRegistration:
    def test_on_message_registers_handler(self, mock_transport):
        async def my_handler(msg, peer):
            pass

        mock_transport.on_message("gossip", my_handler)
        assert "gossip" in mock_transport._handlers
        assert my_handler in mock_transport._handlers["gossip"]

    def test_on_message_multiple_handlers(self, mock_transport):
        async def h1(msg, peer):
            pass

        async def h2(msg, peer):
            pass

        mock_transport.on_message("direct", h1)
        mock_transport.on_message("direct", h2)
        assert len(mock_transport._handlers["direct"]) == 2


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_telemetry_snapshot(self, mock_transport):
        snap = mock_transport.get_telemetry_snapshot()

        expected_keys = {
            "messages_sent",
            "messages_received",
            "publish_count",
            "connect_count",
            "error_count",
            "dispatch_success_total",
            "dispatch_failure_total",
            "dispatch_failure_reasons",
            "keepalive_pings_sent",  # sprint 1229
        }
        assert set(snap.keys()) == expected_keys

    def test_telemetry_initial_zeros(self, mock_transport):
        snap = mock_transport.get_telemetry_snapshot()
        for key in ("messages_sent", "messages_received", "publish_count",
                     "connect_count", "error_count"):
            assert snap[key] == 0

    def test_telemetry_dispatch_failure_reasons_is_dict(self, mock_transport):
        snap = mock_transport.get_telemetry_snapshot()
        assert isinstance(snap["dispatch_failure_reasons"], dict)


# ---------------------------------------------------------------------------
# Properties with mocked library
# ---------------------------------------------------------------------------

class TestProperties:
    def test_peer_count_no_handle(self, mock_transport):
        """peer_count returns 0 when handle is -1 (not started)."""
        assert mock_transport._handle == -1
        assert mock_transport.peer_count == 0

    def test_peer_addresses_no_handle(self, mock_transport):
        assert mock_transport._handle == -1
        assert mock_transport.peer_addresses == []

    def test_identity_property(self, mock_transport):
        assert mock_transport.identity.node_id == "abcdef1234567890abcdef1234567890"
