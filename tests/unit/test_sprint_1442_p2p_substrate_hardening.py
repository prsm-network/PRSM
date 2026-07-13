"""sp1442 (P2P substrate audit) — the two confirmed substrate defects.

An adversarial audit of the P2P/transport/discovery substrate (already hardened by sp934/937/941
auth, sp1414 known_peers cap, sp936 rate limit, sp1326 WS frame cap) found its AUTH posture sound
but a systemic DoS blind spot: every existing cap guards a DIFFERENT axis than the one an attacker
controls. Two confirmed defects:

  Finding A (HIGH) — no ceiling on CONCURRENT inbound connections. node_id = sha256(pubkey)[:32] is
    free to mint, so one host opens unlimited authenticated WebSocket slots: fd/memory exhaustion,
    AND — because gossip() fans out by random-sampling self.peers — attacker-dominated slots
    black-hole honest publishes (eclipse). The sp936 bucket caps messages-per-peer not peer COUNT;
    sp1414 caps a DIFFERENT dict (discovery.known_peers); sp1326 only collapses duplicate ids.
    Fixed with a global _max_peers ceiling + a per-source-IP cap (_try_admit_peer), atomic with the
    install; reconnects (existing peer_id) are always admitted (sp1326 preserved).

  Finding B (MEDIUM+replay) — an unsolicited gossip digest_response is fully processed with no check
    that we ever sent a matching request, and each of up to 200 entries drove a full-window ledger
    range-scan — a ~200x CPU/DB amplifier past the sp936 per-frame limit. Fixed by gating on an
    outstanding request (covered in test_sprint_1442_digest_response_solicitation.py sibling if
    split; here we assert the connection caps).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import time

import pytest

from prsm.node.gossip import (
    GossipProtocol,
    GOSSIP_DIGEST_RESPONSE,
    _DIGEST_REQUEST_TTL_SEC,
)
from prsm.node.identity import generate_node_identity
from prsm.node.transport import MSG_GOSSIP, P2PMessage, PeerConnection, WebSocketTransport


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _transport(max_peers=256, max_conns_per_ip=16):
    return WebSocketTransport(
        generate_node_identity("test-node"),
        max_peers=max_peers, max_conns_per_ip=max_conns_per_ip)


def _peer(pid):
    return PeerConnection(peer_id=pid, address="1.2.3.4:9001", websocket=None)


# ── Finding A — the global concurrent-connection ceiling ─────────────────────


def test_global_max_peers_rejects_a_new_peer_when_full():
    t = _transport(max_peers=2)
    _run(t._try_admit_peer("p1", _peer("p1"), "1.1.1.1"))
    _run(t._try_admit_peer("p2", _peer("p2"), "2.2.2.2"))
    assert len(t.peers) == 2
    old, reject, counted = _run(t._try_admit_peer("p3", _peer("p3"), "3.3.3.3"))
    assert reject == "peer limit reached"
    assert counted is None
    assert "p3" not in t.peers and len(t.peers) == 2, "a rejected peer must NOT be installed"


def test_per_ip_cap_rejects_a_single_source_flood_but_not_other_ips():
    t = _transport(max_peers=100, max_conns_per_ip=2)
    _run(t._try_admit_peer("a1", _peer("a1"), "9.9.9.9"))
    _run(t._try_admit_peer("a2", _peer("a2"), "9.9.9.9"))
    # third distinct id from the SAME ip → rejected (the Sybil-flood cost floor).
    _, reject, _ = _run(t._try_admit_peer("a3", _peer("a3"), "9.9.9.9"))
    assert reject == "per-ip connection limit reached"
    assert "a3" not in t.peers
    # a DIFFERENT source ip is unaffected — the cap is per-source, not global-by-ip.
    _, reject2, _ = _run(t._try_admit_peer("b1", _peer("b1"), "8.8.8.8"))
    assert reject2 is None and "b1" in t.peers


def test_reconnect_of_an_existing_peer_id_is_always_admitted_even_at_cap():
    """sp1326 must survive: a peer re-dialing (existing id) replaces its slot and is NEVER
    rejected by the caps — else an asymmetric-firewall peer could be locked out."""
    t = _transport(max_peers=1, max_conns_per_ip=1)
    first = _peer("p1")
    _run(t._try_admit_peer("p1", first, "1.1.1.1"))
    assert len(t.peers) == 1
    replacement = _peer("p1")
    old, reject, counted = _run(t._try_admit_peer("p1", replacement, "1.1.1.1"))
    assert reject is None, "a reconnect must not be rejected at the cap"
    assert old is first and t.peers["p1"] is replacement, "the reconnect replaced the old slot"


def test_ip_charge_is_refunded_on_teardown_so_a_freed_slot_reopens():
    t = _transport(max_peers=100, max_conns_per_ip=1)
    _run(t._try_admit_peer("p1", _peer("p1"), "5.5.5.5"))
    assert t._ip_conn_counts["5.5.5.5"] == 1
    # at the per-ip cap: a second new peer from 5.5.5.5 is rejected.
    _, reject, _ = _run(t._try_admit_peer("p2", _peer("p2"), "5.5.5.5"))
    assert reject == "per-ip connection limit reached"
    # p1 disconnects → the finally refunds its charge.
    _run(t._release_ip("5.5.5.5"))
    assert "5.5.5.5" not in t._ip_conn_counts, "the counter dict must pop at 0 (stay bounded)"
    # now a fresh connection from 5.5.5.5 is admitted again.
    _, reject2, _ = _run(t._try_admit_peer("p2", _peer("p2"), "5.5.5.5"))
    assert reject2 is None and "p2" in t.peers


def test_caps_are_env_tunable_and_floored():
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {"PRSM_MAX_PEERS": "64", "PRSM_MAX_CONNS_PER_IP": "3"}):
        t = WebSocketTransport(generate_node_identity("n"))
        assert t._max_peers == 64 and t._max_conns_per_ip == 3
    # a 0/negative override clamps to the floor (1) rather than bricking inbound entirely.
    with patch.dict(os.environ, {"PRSM_MAX_PEERS": "0", "PRSM_MAX_CONNS_PER_IP": "0"}):
        t2 = WebSocketTransport(generate_node_identity("n"))
        assert t2._max_peers >= 1 and t2._max_conns_per_ip >= 1


# ── Finding B — the digest_response solicitation gate + O(1) dedup ────────────


class _GateFakeLedger:
    async def get_recent_gossip(self, since, subtypes=None, limit=500):
        return []

    async def gossip_nonce_exists(self, nonce):
        return False

    async def log_gossip(self, **kw):
        return None


def _gossip(ledger=None):
    transport = MagicMock()
    transport.identity = generate_node_identity("receiver")
    transport.send_to_peer = AsyncMock()
    gp = GossipProtocol(transport)
    gp.ledger = ledger
    return gp


def _peer(pid):
    m = MagicMock()
    m.peer_id = pid
    return m


def _digest_response(nonces, sender):
    return P2PMessage(
        msg_type=MSG_GOSSIP, sender_id=sender,
        payload={"subtype": GOSSIP_DIGEST_RESPONSE, "data": {"messages": [
            {"subtype": "content_ad", "payload": {"n": n}, "nonce": n} for n in nonces]}},
        ttl=1, nonce="dr")


async def test_unsolicited_digest_response_is_dropped_before_processing():
    """The money shot for Finding B: a digest_response from a peer we never sent a request to is
    dropped BEFORE the per-entry auth+dedup loop — closing the ~200x ledger-scan amplifier AND the
    signed-frame replay-injection (an inbound attacker never receives a request, so never a slot)."""
    gp = _gossip(_GateFakeLedger())
    delivered = []

    async def cb(st, pl, org):
        delivered.append(pl)

    gp._subscribers["content_ad"] = [cb]
    # No request was ever sent to "attacker" → _pending_digest empty.
    await gp._handle_digest_response(_digest_response(["n1", "n2"], sender="attacker"), _peer("attacker"))
    assert delivered == [], "an UNSOLICITED digest_response was processed — amplifier/replay open"


async def test_solicited_digest_response_is_processed_once_then_single_use():
    gp = _gossip(_GateFakeLedger())
    delivered = []

    async def cb(st, pl, org):
        delivered.append(pl)

    gp._subscribers["content_ad"] = [cb]
    # Simulate that WE dialed this peer and sent it a digest request.
    gp._pending_digest["goodpeer"] = time.time() + _DIGEST_REQUEST_TTL_SEC
    await gp._handle_digest_response(_digest_response(["n1"], sender="goodpeer"), _peer("goodpeer"))
    assert len(delivered) == 1, "a solicited response must be processed"
    # A second (replayed) response reuses no live request — single-use consumed it.
    await gp._handle_digest_response(_digest_response(["n2"], sender="goodpeer"), _peer("goodpeer"))
    assert len(delivered) == 1, "single-use pending request must admit only ONE response"


async def test_gate_keys_on_authenticated_peer_not_spoofable_sender_id():
    """The pending entry is keyed on the handshake-authenticated peer.peer_id; a frame whose
    sender_id claims a solicited peer but arrives on a DIFFERENT connection is still dropped."""
    gp = _gossip(_GateFakeLedger())
    delivered = []

    async def cb(st, pl, org):
        delivered.append(pl)

    gp._subscribers["content_ad"] = [cb]
    gp._pending_digest["goodpeer"] = time.time() + _DIGEST_REQUEST_TTL_SEC
    # attacker frames sender_id="goodpeer" but the authenticated peer.peer_id is "attacker".
    await gp._handle_digest_response(_digest_response(["n1"], sender="goodpeer"), _peer("attacker"))
    assert delivered == [], "gate must key on authenticated peer.peer_id, not the frame sender_id"


async def test_request_digest_records_a_pending_entry():
    gp = _gossip(_GateFakeLedger())
    gp._get_last_seen_timestamps = AsyncMock(return_value={})
    await gp.request_digest("dialed-peer")
    assert "dialed-peer" in gp._pending_digest


async def test_gossip_nonce_exists_is_o1_and_used_by_is_duplicate(tmp_path):
    from prsm.node.local_ledger import LocalLedger
    led = LocalLedger(db_path=str(tmp_path / "l.db"))
    await led.initialize()
    assert await led.gossip_nonce_exists("nope") is False
    await led.log_gossip(nonce="abc", subtype="content_ad", origin="o", payload={"x": 1})
    assert await led.gossip_nonce_exists("abc") is True
    # _is_duplicate prefers the O(1) helper.
    gp = _gossip(led)
    assert await gp._is_duplicate("abc") is True
    assert await gp._is_duplicate("nope") is False
