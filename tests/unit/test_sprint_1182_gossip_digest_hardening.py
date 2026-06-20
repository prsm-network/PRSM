"""Sprint 1182 — harden the gossip digest-request handler + origin fallback.

A P2P/gossip review (live WebSocket substrate) confirmed a digest-request DoS
and surfaced two attacker-payload-trust footguns in the gossip handlers:

1. **Digest-request CPU/DB DoS** — `_handle_digest_request` looped over the
   requester's `timestamps` dict (one ledger query per entry) with NO cap, and
   passed the attacker's `last_seen` straight through (0 forces a full-window
   scan). A single rate-limited 16MB request packs ~1e6 entries → ~1e6 ledger
   queries. Fix: cap the processed subtype count + floor `last_seen` to the
   retention window.
2. **Digest-response reflection** — the response was sent to the attacker-
   controlled `data["requester_id"]` payload field (set it to a victim →
   reflect a digest at them). Fix: route to the handshake-authenticated
   `peer.peer_id`.
3. **Origin fallback to a spoofable frame field** — on failed origin
   attestation, `_handle_gossip` fell back to the raw `msg.sender_id` (a peer
   can set it to a victim's id). Fix: fall back to the authenticated
   `peer.peer_id` (defense-in-depth; the valid-attestation path is unchanged).
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.gossip import (
    GossipProtocol,
    GOSSIP_DIGEST_REQUEST,
    _MAX_DIGEST_REQUEST_SUBTYPES,
    _DIGEST_REQUEST_MAX_LOOKBACK_SEC,
)
from prsm.node.identity import generate_node_identity
from prsm.node.transport import MSG_GOSSIP, P2PMessage


class _FakeLedger:
    def __init__(self, return_msgs=None):
        self.calls = []  # (since, subtypes) per get_recent_gossip
        self._return = return_msgs or []

    async def get_recent_gossip(self, since, subtypes):
        self.calls.append((since, subtypes))
        return list(self._return)

    async def log_gossip(self, **kwargs):
        return None


def _gossip(ledger=None):
    transport = MagicMock()
    transport.identity = generate_node_identity("receiver")
    transport.gossip = AsyncMock(return_value=0)
    transport.send_to_peer = AsyncMock()
    gp = GossipProtocol(transport)
    gp.ledger = ledger
    return gp


def _digest_req(timestamps, requester_id=None):
    data = {"timestamps": timestamps}
    if requester_id is not None:
        data["requester_id"] = requester_id
    return P2PMessage(
        msg_type=MSG_GOSSIP, sender_id="sender",
        payload={"subtype": GOSSIP_DIGEST_REQUEST, "data": data},
        ttl=1, nonce="dn",
    )


class TestDigestDoS:
    async def test_subtype_count_is_capped(self):
        """A 1000-entry timestamps dict must NOT drive 1000 ledger
        queries — capped to _MAX_DIGEST_REQUEST_SUBTYPES (+ a few fixed
        catch-up subtypes)."""
        ledger = _FakeLedger()
        gp = _gossip(ledger)
        big = {f"sub{i}": time.time() for i in range(1000)}
        await gp._handle_digest_request(
            _digest_req(big), MagicMock(peer_id="peerA"),
        )
        assert len(ledger.calls) <= _MAX_DIGEST_REQUEST_SUBTYPES + 8, (
            f"{len(ledger.calls)} ledger queries for one request "
            "(DoS: should be capped near _MAX_DIGEST_REQUEST_SUBTYPES)"
        )
        assert len(ledger.calls) < 1000

    async def test_last_seen_is_clamped_to_retention_floor(self):
        """An attacker's last_seen=0 (force a full-window scan) is
        clamped to >= now - retention."""
        ledger = _FakeLedger()
        gp = _gossip(ledger)
        await gp._handle_digest_request(
            _digest_req({"x": 0}), MagicMock(peer_id="peerA"),
        )
        assert ledger.calls, "expected a ledger query"
        since_used = ledger.calls[0][0]
        floor = time.time() - _DIGEST_REQUEST_MAX_LOOKBACK_SEC
        assert since_used >= floor - 5
        assert since_used > 1000.0  # definitely not the attacker's 0

    async def test_non_numeric_last_seen_does_not_crash(self):
        ledger = _FakeLedger()
        gp = _gossip(ledger)
        await gp._handle_digest_request(
            _digest_req({"x": "not-a-number"}), MagicMock(peer_id="p"),
        )
        assert ledger.calls  # handled (clamped to floor), no exception


class TestDigestReflection:
    async def test_response_goes_to_authed_peer_not_payload_requester(self):
        """The digest response must go to the handshake-authenticated
        peer.peer_id, NEVER the attacker-controlled requester_id payload
        (which would reflect the response at a victim)."""
        ledger = _FakeLedger(return_msgs=[{"received_at": 1.0, "subtype": "x"}])
        gp = _gossip(ledger)
        msg = _digest_req({"x": time.time()}, requester_id="VICTIM")
        await gp._handle_digest_request(msg, MagicMock(peer_id="ATTACKER"))
        gp.transport.send_to_peer.assert_called_once()
        target = gp.transport.send_to_peer.call_args[0][0]
        assert target == "ATTACKER", (
            f"digest response reflected to {target} (must be the authed peer)"
        )


class TestOriginFallback:
    async def test_unattested_origin_falls_back_to_authed_peer(self):
        """NON-VACUITY: an unattested gossip message with a spoofed
        sender_id is attributed to the authenticated peer.peer_id, NOT
        the spoofed msg.sender_id. (Under the old code origin would be
        the spoofed sender_id.)"""
        gp = _gossip(_FakeLedger())
        seen = []

        async def cb(subtype, data, origin):
            seen.append(origin)

        gp._subscribers["test_sub"] = [cb]
        msg = P2PMessage(
            msg_type=MSG_GOSSIP, sender_id="VICTIM",  # spoofed frame field
            payload={"subtype": "test_sub", "data": {}},
            ttl=1, nonce="nz",
        )
        await gp._handle_gossip(msg, MagicMock(peer_id="ATTACKER"))
        assert seen == ["ATTACKER"], (
            f"origin={seen} — must be the authed peer, not the spoofed sender"
        )
