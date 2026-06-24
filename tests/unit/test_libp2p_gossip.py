"""
Unit tests for prsm.node.libp2p_gossip.Libp2pGossip.

All tests use mocked transport objects — no Go shared library required.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.node.libp2p_gossip import Libp2pGossip
from prsm.node.transport import MSG_GOSSIP, P2PMessage


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_transport(node_id: str = "test-node-001") -> MagicMock:
    """Return a minimal mock that looks like Libp2pTransport."""
    transport = MagicMock()
    # sp1246 — publish() now signs an sp934 origin attestation, so the transport
    # needs a REAL identity (public_key_b64 + sign) that JSON-serializes, not a
    # MagicMock. node_id is used as the display name (the derived node_id differs
    # but no test asserts a specific value).
    from prsm.node.identity import generate_node_identity
    transport.identity = generate_node_identity(node_id)
    transport._handle = 0
    transport._lib = MagicMock()
    transport._lib.PrsmSubscribe.return_value = 0
    transport._lib.PrsmPublish.return_value = 0
    # on_message should just record calls (not async)
    transport.on_message = MagicMock()
    return transport


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestTopicName:
    def test_topic_name(self) -> None:
        assert Libp2pGossip._topic_name("job_offer") == "prsm/job_offer"

    def test_topic_name_with_slash(self) -> None:
        """Subtypes that already contain a slash should still be wrapped."""
        assert Libp2pGossip._topic_name("a/b") == "prsm/a/b"

    def test_topic_name_empty(self) -> None:
        assert Libp2pGossip._topic_name("") == "prsm/"


class TestLazySubscription:
    """PrsmSubscribe is called exactly once for the first callback on a subtype."""

    def test_lazy_subscription(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        cb = AsyncMock()
        gossip.subscribe("job_offer", cb)

        transport._lib.PrsmSubscribe.assert_called_once_with(
            transport._handle,
            b"prsm/job_offer",
        )

    def test_second_callback_no_extra_subscribe(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        cb1 = AsyncMock()
        cb2 = AsyncMock()
        gossip.subscribe("job_offer", cb1)
        gossip.subscribe("job_offer", cb2)

        # PrsmSubscribe must have been called exactly once despite two callbacks
        assert transport._lib.PrsmSubscribe.call_count == 1

    def test_lazy_subscribe_topic_tracked(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        gossip.subscribe("heartbeat", AsyncMock())

        assert "prsm/heartbeat" in gossip._subscribed_topics


class TestMultipleSubtypes:
    """Subscribing to different subtypes each triggers PrsmSubscribe."""

    def test_multiple_subtypes(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        gossip.subscribe("job_offer", AsyncMock())
        gossip.subscribe("job_result", AsyncMock())
        gossip.subscribe("heartbeat", AsyncMock())

        assert transport._lib.PrsmSubscribe.call_count == 3
        called_topics = {
            call.args[1] for call in transport._lib.PrsmSubscribe.call_args_list
        }
        assert b"prsm/job_offer" in called_topics
        assert b"prsm/job_result" in called_topics
        assert b"prsm/heartbeat" in called_topics


class TestTelemetrySnapshot:
    """get_telemetry_snapshot returns the expected structure."""

    def test_telemetry_snapshot_initial(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        snap = gossip.get_telemetry_snapshot()

        assert "publish_total" in snap
        assert "deliver_total" in snap
        assert "error_total" in snap
        assert "subscribed_topics" in snap
        assert snap["publish_total"] == 0
        assert snap["deliver_total"] == 0
        assert snap["error_total"] == 0
        assert isinstance(snap["subscribed_topics"], list)

    def test_telemetry_snapshot_after_subscribe(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        gossip.subscribe("capability_announce", AsyncMock())
        gossip.subscribe("job_offer", AsyncMock())

        snap = gossip.get_telemetry_snapshot()
        assert "prsm/capability_announce" in snap["subscribed_topics"]
        assert "prsm/job_offer" in snap["subscribed_topics"]

    @pytest.mark.asyncio
    async def test_telemetry_publish_increments(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        await gossip.publish("job_offer", {"fee": 1})
        await gossip.publish("job_offer", {"fee": 2})

        snap = gossip.get_telemetry_snapshot()
        assert snap["publish_total"] == 2


class TestHandleGossip:
    """_handle_gossip dispatches to registered callbacks correctly."""

    @pytest.mark.asyncio
    async def test_handle_dispatches_to_callback(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        received: list = []

        async def cb(subtype, data, sender_id):
            received.append((subtype, data, sender_id))

        gossip.subscribe("job_offer", cb)

        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="peer-abc",
            payload={
                "subtype": "job_offer",
                "data": {"fee": 42},
                "sender_id": "peer-abc",
                "timestamp": 1234567890.0,
            },
        )
        peer_stub = MagicMock()

        await gossip._handle_gossip(msg, peer_stub)

        assert len(received) == 1
        subtype, data, sender_id = received[0]
        assert subtype == "job_offer"
        assert data == {"fee": 42}
        assert sender_id == "peer-abc"

    @pytest.mark.asyncio
    async def test_handle_unknown_subtype_no_error(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id="peer-xyz",
            payload={
                "subtype": "unknown_subtype",
                "data": {},
                "sender_id": "peer-xyz",
                "timestamp": 0.0,
            },
        )
        # Should not raise
        await gossip._handle_gossip(msg, MagicMock())

    @pytest.mark.asyncio
    async def test_start_registers_handler(self) -> None:
        transport = _make_transport()
        gossip = Libp2pGossip(transport)

        await gossip.start()

        transport.on_message.assert_called_once_with(MSG_GOSSIP, gossip._handle_gossip)
