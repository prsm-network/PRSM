"""
Gossip Protocol
===============

Epidemic gossip for propagating messages across the PRSM network.
Handles job offers, storage requests, transaction confirmations,
and other network-wide announcements with deduplication and TTL.
"""

import asyncio
import base64
import collections
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional

from prsm.node.identity import verify_signature
from prsm.node.transport import MSG_GOSSIP, MSG_PEER_CONNECTED, P2PMessage, PeerConnection, WebSocketTransport

logger = logging.getLogger(__name__)


# ── sp934: authenticated gossip origin ────────────────────────────────────
# The transport authenticates the immediate sender_id (sp731 F64), but the
# `origin` (original author, preserved across multi-hop relay) was an untrusted
# payload field that application handlers used for authorization. The author now
# signs an attestation over the message's identifying fields and ships its
# pubkey; a receiver trusts `origin` ONLY if the pubkey hashes to it (node_id ==
# sha256(pubkey)[:32]) AND the signature verifies — otherwise it falls back to
# the authenticated sender_id. Self-verifying (no pubkey registry needed) and
# relay-safe (payload + nonce are preserved when forwarding).

def _gossip_origin_signing_bytes(
    subtype: str, data: Any, origin: str, origin_time: Any, nonce: str,
) -> bytes:
    """Canonical bytes the origin author signs / the receiver verifies."""
    return json.dumps(
        {
            "subtype": subtype,
            "data": data,
            "origin": origin,
            "origin_time": origin_time,
            "nonce": nonce,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def build_gossip_origin_fields(
    identity, subtype: str, data: Any, origin_time: float, nonce: str,
) -> Dict[str, Any]:
    """Payload fields that let a receiver authenticate `identity` as the origin."""
    signed = _gossip_origin_signing_bytes(subtype, data, identity.node_id, origin_time, nonce)
    return {
        "origin": identity.node_id,
        "origin_time": origin_time,
        "origin_pubkey": identity.public_key_b64,
        "origin_sig": identity.sign(signed),
    }


def _authenticate_origin(payload: Dict[str, Any], nonce: str, fallback_sender_id: str) -> str:
    """Return the AUTHENTICATED origin for a gossip message.

    Trust `payload['origin']` iff its attestation (pubkey + signature) is present,
    the pubkey hashes to the claimed node_id, and the signature verifies. Else
    return the transport-authenticated `fallback_sender_id` — never the bare,
    unsigned payload origin.
    """
    claimed = payload.get("origin")
    pubkey = payload.get("origin_pubkey", "")
    sig = payload.get("origin_sig", "")
    if not (claimed and pubkey and sig):
        return fallback_sender_id
    try:
        derived = hashlib.sha256(base64.b64decode(pubkey)).hexdigest()[:32]
    except Exception:
        return fallback_sender_id
    if derived != claimed:
        return fallback_sender_id   # pubkey does not bind to the claimed node_id
    signed = _gossip_origin_signing_bytes(
        payload.get("subtype", ""), payload.get("data", {}),
        claimed, payload.get("origin_time"), nonce,
    )
    if verify_signature(pubkey, signed, sig):
        return claimed
    return fallback_sender_id


def _attestation_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """sp961 — extract the persistable origin attestation from a live gossip
    payload (the fields a later catch-up needs to RE-verify authorship). Returns
    None when the message carries no attestation (legacy/unsigned)."""
    pubkey = payload.get("origin_pubkey", "")
    sig = payload.get("origin_sig", "")
    if not (pubkey and sig):
        return None
    return {
        "origin_time": payload.get("origin_time"),
        "origin_pubkey": pubkey,
        "origin_sig": sig,
    }


def _authenticate_catchup_origin(
    *, subtype: str, data: Any, claimed_origin: str,
    attestation: Optional[Dict[str, Any]], nonce: str, fallback_sender_id: str,
) -> str:
    """sp961 — authenticate the origin of a CAUGHT-UP (digest-replayed) message.

    Reconstructs the attested payload from the stored entry + attestation and
    reuses :func:`_authenticate_origin`. Without a valid attestation the claimed
    origin is NOT trusted — a relaying peer cannot forge authorship — so it falls
    back to the transport-authenticated relaying sender."""
    att = attestation or {}
    verify_payload = {
        "subtype": subtype,
        "data": data,
        "origin": claimed_origin,
        "origin_time": att.get("origin_time"),
        "origin_pubkey": att.get("origin_pubkey", ""),
        "origin_sig": att.get("origin_sig", ""),
    }
    return _authenticate_origin(verify_payload, nonce, fallback_sender_id)

_BOUNDED_GOSSIP_LABELS = {
    "heartbeat",
    "agent_task_offer",
    "agent_task_bid",
    "agent_task_assign",
    "agent_task_complete",
    "agent_task_cancel",
    "agent_review_request",
    "agent_review_submit",
    "agent_knowledge_query",
    "agent_knowledge_response",
}

# Gossip subtypes for the compute/storage marketplace
GOSSIP_JOB_OFFER = "job_offer"
GOSSIP_JOB_ACCEPT = "job_accept"
GOSSIP_JOB_CONFIRM = "job_confirm"          # Requester confirms which provider won
GOSSIP_JOB_CANCEL = "job_cancel"            # Cancel a job (timeout, requester abort)
GOSSIP_JOB_RESULT = "job_result"
GOSSIP_PAYMENT_CONFIRM = "payment_confirm"
GOSSIP_ESCROW_CREATE = "escrow_create"       # FTNS locked for a job
GOSSIP_ESCROW_RELEASE = "escrow_release"     # Payment distributed
GOSSIP_ESCROW_REFUND = "escrow_refund"       # Refund to requester
GOSSIP_STORAGE_REQUEST = "storage_request"
GOSSIP_STORAGE_CONFIRM = "storage_confirm"
GOSSIP_PROOF_OF_STORAGE = "proof_of_storage"
GOSSIP_PROVENANCE_REGISTER = "provenance_register"
GOSSIP_CONTENT_ADVERTISE = "content_advertise"
GOSSIP_CONTENT_ACCESS = "content_access"
GOSSIP_FTNS_TRANSACTION = "ftns_transaction"
GOSSIP_AGENT_ADVERTISE = "agent_advertise"
GOSSIP_AGENT_DEREGISTER = "agent_deregister"
GOSSIP_PROVENANCE_QUERY = "provenance_query"
GOSSIP_PROVENANCE_RESPONSE = "provenance_response"
GOSSIP_CAPABILITY_ANNOUNCE = "capability_announce"
GOSSIP_MARKETPLACE_LISTING = "marketplace_listing"     # Phase 3: provider capacity + price advertisement
GOSSIP_HARDWARE_PROFILE = "hardware_profile"
GOSSIP_TEE_CAPABILITY = "tee_capability"

# Gossip subtypes for mobile agent dispatch (Ring 2)
GOSSIP_AGENT_DISPATCH = "agent_dispatch"
GOSSIP_AGENT_ACCEPT = "agent_accept"
GOSSIP_AGENT_RESULT = "agent_result"

# Gossip subtypes for agent collaboration protocols
GOSSIP_TASK_ASSIGN = "agent_task_assign"
GOSSIP_TASK_COMPLETE = "agent_task_complete"
GOSSIP_TASK_CANCEL = "agent_task_cancel"
GOSSIP_REVIEW_SUBMIT = "agent_review_submit"
GOSSIP_KNOWLEDGE_RESPONSE = "agent_knowledge_response"

# Gossip subtypes for digest exchange (late-joining node catch-up)
GOSSIP_DIGEST_REQUEST = "digest_request"
GOSSIP_DIGEST_RESPONSE = "digest_response"

# Sp1182 — bound the attacker-controlled digest REQUEST. The requester's
# `timestamps` dict drives one ledger query per entry; a 16MB frame packs
# ~1e6 entries, and a `last_seen` of 0 forces each query to the bottom of
# the retention window — a CPU/DB DoS amplified far past the per-peer
# message rate limit (which counts messages, not subtypes-per-message).
# The legitimate requester only ever emits the handful of catch-up
# subtypes, so cap the processed count and floor `last_seen` to the
# retention window. Generous cap (legit ~4-7); env-tunable for headroom.
_MAX_DIGEST_REQUEST_SUBTYPES = 32
_DIGEST_REQUEST_MAX_LOOKBACK_SEC = 86400.0  # 24h — the gossip-log retention
# sp1270 — cap the catch-up messages we'll process from a single digest RESPONSE. A legit
# responder caps its list at 100 (the request-path max_messages); each message triggers a
# ledger dedup scan, so an unbounded peer-controlled list is a DoS amplifier. 200 = generous
# slack over the legit 100.
_MAX_DIGEST_RESPONSE_MESSAGES = 200


# sp1008 — gossip-layer replay barrier. The transport dedups nonces only within
# its ~300s window, but the gossip log retains messages 1h–24h, so a captured
# signed frame replayed after 300s would be re-delivered to subscribers and
# re-fanned into the mesh. A gossip-layer seen-nonce set with a TTL >= the
# retention closes this; an LRU cap bounds its memory. Legit gossip is
# unaffected (every publish() uses a fresh nonce — only exact-nonce replays,
# which mesh redundancy already tolerates being dropped, are caught).
_DEFAULT_GOSSIP_DEDUP_WINDOW_SEC = 86400.0  # 24h — covers the max log retention
_DEFAULT_GOSSIP_DEDUP_MAX = 100_000


def _gossip_dedup_window() -> float:
    raw = os.environ.get("PRSM_GOSSIP_DEDUP_WINDOW_SEC", "").strip()
    if not raw:
        return _DEFAULT_GOSSIP_DEDUP_WINDOW_SEC
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return _DEFAULT_GOSSIP_DEDUP_WINDOW_SEC
    return val if val > 0 else _DEFAULT_GOSSIP_DEDUP_WINDOW_SEC


def _gossip_dedup_max() -> int:
    raw = os.environ.get("PRSM_GOSSIP_DEDUP_MAX", "").strip()
    if not raw:
        return _DEFAULT_GOSSIP_DEDUP_MAX
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return _DEFAULT_GOSSIP_DEDUP_MAX
    return val if val > 0 else _DEFAULT_GOSSIP_DEDUP_MAX

# Gossip subtypes for BitTorrent integration
GOSSIP_BITTORRENT_ANNOUNCE = "bittorrent_announce"
GOSSIP_BITTORRENT_WITHDRAW = "bittorrent_withdraw"
GOSSIP_BITTORRENT_STATS = "bittorrent_stats"
GOSSIP_BITTORRENT_REQUEST = "bittorrent_request"

# sp1137 — settlement-receipt data plane BRICK C. A producer ANNOUNCES that a
# committed batch's ordered receipt set is available + where to fetch it (a tiny
# untrusted pointer: batch_id, provider, merkle_root, CID, optional leaf_hashes).
# Observers index the pointer (pull-on-demand, NOT flood) and verify the fetched
# blob LATER against the on-chain root (Brick B/D). Origin-auth (sp934) only proves
# WHO gossiped it — for spam/replay accounting + dedup — not that the ad is correct.
GOSSIP_SETTLEMENT_BATCH_AVAILABLE = "settlement_batch_available"

# Retention configuration per gossip subtype (in seconds)
# Messages older than these values are pruned from the gossip log
GOSSIP_RETENTION_SECONDS: Dict[str, float] = {
    # Task-related messages: 1 hour retention
    "job_offer": 3600,
    "job_accept": 3600,
    "job_result": 3600,
    "payment_confirm": 3600,
    "agent_task_offer": 3600,
    "agent_task_bid": 3600,
    "agent_task_assign": 3600,
    "agent_task_complete": 3600,
    "agent_task_cancel": 3600,
    "agent_review_request": 3600,
    "agent_review_submit": 3600,
    "agent_knowledge_query": 3600,
    "agent_knowledge_response": 3600,
    # Content-related messages: 24 hour retention
    "content_advertise": 86400,
    "content_access": 86400,
    "storage_request": 86400,
    "storage_confirm": 86400,
    "provenance_register": 86400,
    "provenance_query": 3600,
    "provenance_response": 3600,
    "proof_of_storage": 86400,
    # Agent registration: 24 hour retention
    "agent_advertise": 86400,
    "agent_deregister": 86400,
    "capability_announce": 86400,
    "hardware_profile": 86400,  # 24 hours
    "tee_capability": 86400,  # 24 hours
    # Mobile agent dispatch (Ring 2): 1 hour retention
    "agent_dispatch": 3600,
    "agent_accept": 3600,
    "agent_result": 3600,
    # FTNS transactions: 24 hour retention for audit trail
    "ftns_transaction": 86400,
    # BitTorrent messages: 24 hour retention for announces, 1 hour for withdrawals
    "bittorrent_announce": 86400,
    "bittorrent_withdraw": 3600,
    "bittorrent_stats": 1800,      # 30 minutes — stats decay quickly
    "bittorrent_request": 300,     # 5 minutes — short-lived requests
    # sp1137 — settlement batch-availability ad: retain past the dispute challenge
    # window + a margin so a late-joining observer can still learn a batch exists in
    # time to fetch + cross-check it before finalization closes the dispute window.
    "settlement_batch_available": 2 * 86400,  # 48h (challenge window + margin)
    # Heartbeat: very short retention (not stored anyway)
    "heartbeat": 60,
    # Digest exchange: not stored
    "digest_request": 0,
    "digest_response": 0,
}

# Callback type for gossip subscribers
GossipCallback = Callable[[str, Dict[str, Any], str], Coroutine[Any, Any, None]]
# (subtype, payload, sender_id) -> None


class GossipProtocol:
    """Epidemic gossip with fanout, TTL, and deduplication.

    Messages are forwarded to a random subset of peers (fanout),
    with decreasing TTL. Nonce-based dedup prevents infinite loops.
    """

    def __init__(
        self,
        transport: WebSocketTransport,
        fanout: int = 3,
        default_ttl: int = 5,
        heartbeat_interval: float = 30.0,
        gossip_log_retention: float = 3600.0,
    ):
        self.transport = transport
        self.fanout = fanout
        self.default_ttl = default_ttl
        self.heartbeat_interval = heartbeat_interval
        self.gossip_log_retention = gossip_log_retention

        self._subscribers: Dict[str, List[GossipCallback]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # sp1008 — gossip-layer replay barrier (nonce -> first-seen time).
        # OrderedDict so the oldest entries evict first (TTL prune + LRU cap).
        self._seen_gossip_nonces: "collections.OrderedDict[str, float]" = (
            collections.OrderedDict()
        )

        # Ledger for gossip persistence (set post-construction by node.py)
        self.ledger: Optional[Any] = None

        # Additive observability counters (must never change gossip behavior)
        self._telemetry: Dict[str, Any] = {
            "publish_total": 0,
            "publish_by_subtype": collections.Counter(),
            "forward_total": 0,
            "forward_by_subtype": collections.Counter(),
            "drop_total": 0,
            "drop_by_subtype": collections.Counter(),
            "drop_by_reason": collections.Counter(),
        }

        # Register as handler for all gossip messages
        self.transport.on_message(MSG_GOSSIP, self._handle_gossip)
        
        # Register handler for peer connection events (for digest exchange)
        self.transport.on_message(MSG_PEER_CONNECTED, self._on_peer_connected)

    @staticmethod
    def _telemetry_subtype_label(subtype: str) -> str:
        """Map raw subtypes to a bounded cardinality label set."""
        if subtype in _BOUNDED_GOSSIP_LABELS:
            return subtype
        return "other"

    def _record_publish(self, subtype: str) -> None:
        try:
            label = self._telemetry_subtype_label(subtype)
            self._telemetry["publish_total"] += 1
            self._telemetry["publish_by_subtype"][label] += 1
        except Exception:
            pass

    def _record_forward(self, subtype: str) -> None:
        try:
            label = self._telemetry_subtype_label(subtype)
            self._telemetry["forward_total"] += 1
            self._telemetry["forward_by_subtype"][label] += 1
        except Exception:
            pass

    def _record_drop(self, subtype: str, reason: str) -> None:
        try:
            label = self._telemetry_subtype_label(subtype)
            self._telemetry["drop_total"] += 1
            self._telemetry["drop_by_subtype"][label] += 1
            self._telemetry["drop_by_reason"][reason] += 1
        except Exception:
            pass

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return a stable copy of gossip telemetry counters for tests/debugging."""
        return {
            "publish_total": int(self._telemetry["publish_total"]),
            "publish_by_subtype": dict(self._telemetry["publish_by_subtype"]),
            "forward_total": int(self._telemetry["forward_total"]),
            "forward_by_subtype": dict(self._telemetry["forward_by_subtype"]),
            "drop_total": int(self._telemetry["drop_total"]),
            "drop_by_subtype": dict(self._telemetry["drop_by_subtype"]),
            "drop_by_reason": dict(self._telemetry["drop_by_reason"]),
        }

    def subscribe(self, subtype: str, callback: GossipCallback) -> None:
        """Subscribe to a specific gossip subtype."""
        self._subscribers.setdefault(subtype, []).append(callback)

    async def publish(self, subtype: str, data: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Publish a gossip message to the network.

        Returns number of peers the message was sent to.
        
        In single-node mode (no peers), also delivers to local subscribers
        to enable self-compute and other local operations.
        """
        self._record_publish(subtype)
        # sp934 — sign an origin attestation so receivers can authenticate this
        # node as the author across multi-hop relay. The nonce is generated here
        # (rather than letting P2PMessage default it) so it can be bound into the
        # signed payload.
        nonce = uuid.uuid4().hex[:16]
        origin_fields = build_gossip_origin_fields(
            self.transport.identity, subtype, data, time.time(), nonce,
        )
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload={
                "subtype": subtype,
                "data": data,
                **origin_fields,
            },
            ttl=ttl if ttl is not None else self.default_ttl,
            nonce=nonce,
        )
        
        # Send to peers
        sent = await self.transport.gossip(msg, fanout=self.fanout)
        
        # In single-node mode (no peers), deliver to local subscribers
        # This enables self-compute and other local operations
        if sent == 0:
            callbacks = self._subscribers.get(subtype, [])
            for cb in callbacks:
                try:
                    await cb(subtype, data, self.transport.identity.node_id)
                except Exception as e:
                    logger.error(f"Gossip local subscriber error ({subtype}): {e}")
        
        return sent

    async def start(self) -> None:
        """Start heartbeat loop."""
        self._running = True
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        logger.info("Gossip protocol started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    # ── Peer Connection Handler ─────────────────────────────────────

    async def _on_peer_connected(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle new peer connection by requesting digest exchange.

        When a new peer connects, we send a digest request to catch up on
        any messages we may have missed while we were disconnected.
        """
        peer_id = msg.sender_id
        direction = msg.payload.get("direction", "unknown")
        
        logger.info(f"Peer connected: {peer_id[:8]}... ({direction})")
        
        # Only request digest from outbound connections (we initiated)
        # This prevents both peers from sending digest requests simultaneously
        if direction == "outbound":
            try:
                await self.request_digest(peer_id)
            except Exception as e:
                logger.debug(f"Failed to send digest request to {peer_id[:8]}...: {e}")

    # ── Internal ─────────────────────────────────────────────────

    def _is_replayed_gossip(self, nonce: str, now: float) -> bool:
        """sp1008 — gossip-layer replay barrier. Returns True if ``nonce`` was
        already seen within the dedup window (a replay to drop); otherwise
        records it and returns False. Lazily prunes entries older than the
        window and enforces an LRU cap, so the set is bounded in memory."""
        window = _gossip_dedup_window()
        seen = self._seen_gossip_nonces
        # TTL prune from the oldest (insertion-ordered).
        while seen:
            oldest_nonce = next(iter(seen))
            if now - seen[oldest_nonce] > window:
                seen.popitem(last=False)
            else:
                break
        if nonce in seen:
            return True
        seen[nonce] = now
        # LRU cap — evict oldest beyond the bound.
        cap = _gossip_dedup_max()
        while len(seen) > cap:
            seen.popitem(last=False)
        return False

    async def _handle_gossip(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Process incoming gossip and optionally re-propagate."""
        subtype = msg.payload.get("subtype", "")
        data = msg.payload.get("data", {})
        # sp934 — authenticate the origin: trust payload['origin'] only if its
        # attestation verifies, else fall back to the relayer's identity.
        # A peer can no longer impersonate another node by setting payload origin.
        # Sp1182 — the fallback is now the HANDSHAKE-AUTHENTICATED peer.peer_id,
        # not the raw (spoofable) msg.sender_id frame field: for a legitimately
        # relayed message they coincide (the forwarder stamps its own id), but
        # for an unattested message a peer could otherwise set sender_id=<victim>
        # and have the fallback origin attribute the content to that victim.
        origin = _authenticate_origin(
            msg.payload, msg.nonce,
            getattr(peer, "peer_id", None) or msg.sender_id,
        )

        if not subtype:
            self._record_drop("", "missing_subtype")
            return

        # Handle digest exchange messages specially
        if subtype == GOSSIP_DIGEST_REQUEST:
            await self._handle_digest_request(msg, peer)
            return
        
        if subtype == GOSSIP_DIGEST_RESPONSE:
            await self._handle_digest_response(msg, peer)
            return

        # sp1008 — drop gossip-layer replays before delivering or re-fanning.
        # Placed AFTER the digest special-casing (the catch-up/sync mechanism
        # must keep working) and BEFORE subscriber delivery + re-propagation, so
        # a frame replayed past the transport's ~300s window can neither
        # re-trigger handlers nor re-amplify into the mesh.
        if msg.nonce and self._is_replayed_gossip(msg.nonce, time.time()):
            self._record_drop(subtype, "replayed_nonce")
            return

        # Deliver to local subscribers
        callbacks = self._subscribers.get(subtype, [])
        for cb in callbacks:
            try:
                await cb(subtype, data, origin)
            except Exception as e:
                logger.error(f"Gossip subscriber error ({subtype}): {e}")

        # Persist to gossip log (skip heartbeats — too frequent, low value)
        if self.ledger and subtype != "heartbeat":
            try:
                await self.ledger.log_gossip(
                    nonce=msg.nonce,
                    subtype=subtype,
                    origin=origin,
                    payload=data,
                    ttl=msg.ttl,
                    # sp961 — persist the origin attestation so a later digest
                    # catch-up can RE-verify authorship relayer-independently.
                    attestation=_attestation_from_payload(msg.payload),
                )
            except Exception:
                pass  # Fire-and-forget; don't break gossip on log failure

        # Re-propagate with decremented TTL
        if msg.ttl > 1:
            self._record_forward(subtype)
            fwd = P2PMessage(
                msg_type=MSG_GOSSIP,
                sender_id=self.transport.identity.node_id,
                payload=msg.payload,
                ttl=msg.ttl - 1,
                nonce=msg.nonce,  # preserve nonce for dedup
            )
            await self.transport.gossip(fwd, fanout=self.fanout)
        else:
            self._record_drop(subtype, "ttl_exhausted")

    async def get_catchup_messages(
        self,
        since: float,
        subtypes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return persisted gossip messages received after *since*.

        Used by newly-connected peers to catch up on missed state changes.
        """
        if not self.ledger:
            return []
        return await self.ledger.get_recent_gossip(since, subtypes)

    # ── Digest Exchange for Late-Joining Nodes ───────────────────────────

    async def request_digest(self, peer_id: str) -> None:
        """Send a digest request to a peer to catch up on missed messages.

        Called when a new peer connection is established. The peer will
        respond with any messages we've missed based on our last-seen timestamps.
        """
        # Build digest request with last-seen timestamps per subtype
        timestamps = await self._get_last_seen_timestamps()
        
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload={
                "subtype": GOSSIP_DIGEST_REQUEST,
                "data": {
                    "timestamps": timestamps,
                    "requester_id": self.transport.identity.node_id,
                },
                "origin": self.transport.identity.node_id,
                "origin_time": time.time(),
            },
            ttl=1,  # Direct message, don't re-propagate
        )
        
        await self.transport.send_to_peer(peer_id, msg)
        logger.debug(f"Sent digest request to {peer_id[:8]}... with {len(timestamps)} subtype timestamps")

    async def _get_last_seen_timestamps(self) -> Dict[str, float]:
        """Get the last-seen timestamp for each gossip subtype.

        Returns a dict mapping subtype -> last received timestamp.
        Used to request only messages we haven't seen.
        """
        if not self.ledger:
            return {}
        
        try:
            # Query the ledger for last-seen timestamps
            # This is a simplified implementation - could be optimized with a dedicated query
            timestamps = {}
            current_time = time.time()
            
            # Get recent messages for each subtype we care about
            catchup_subtypes = [
                GOSSIP_JOB_OFFER,
                GOSSIP_CONTENT_ADVERTISE,
                GOSSIP_AGENT_ADVERTISE,
                GOSSIP_PROVENANCE_REGISTER,
                GOSSIP_STORAGE_REQUEST,
                "agent_task_offer",
                "agent_task_assign",
            ]
            
            for subtype in catchup_subtypes:
                # Get the most recent message of this subtype
                messages = await self.ledger.get_recent_gossip(
                    since=current_time - 86400,  # Look back up to 24 hours
                    subtypes=[subtype]
                )
                if messages:
                    # Get the timestamp of the most recent message
                    timestamps[subtype] = max(m["received_at"] for m in messages)
            
            return timestamps
        except Exception as e:
            logger.debug(f"Error getting last-seen timestamps: {e}")
            return {}

    async def _handle_digest_request(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle incoming digest request from a peer.

        Queries the local gossip log for messages after the requested timestamps
        and sends them back in a digest response.
        """
        data = msg.payload.get("data", {})
        timestamps = data.get("timestamps", {})
        # Sp1182 — route the response to the HANDSHAKE-AUTHENTICATED peer
        # (the connection this request arrived on), NOT the attacker-
        # controlled data["requester_id"] payload field. The latter let a
        # peer set requester_id=<victim> and reflect a digest response at
        # an arbitrary node. peer.peer_id is the only trustworthy target.
        requester_id = getattr(peer, "peer_id", None) or msg.sender_id

        if not self.ledger:
            logger.debug(f"No ledger available for digest request from {requester_id[:8]}...")
            return

        # Sp1182 — bound the attacker-controlled request before any work:
        # cap the subtype count (one ledger query per entry) and floor
        # last_seen to the retention window (an unvalidated 0 forces a
        # full-window scan). Without this a single rate-limited 16MB
        # request drives up to ~1e6 ledger queries (a CPU/DB DoS).
        if not isinstance(timestamps, dict):
            timestamps = {}
        _capped_items = list(timestamps.items())[:_MAX_DIGEST_REQUEST_SUBTYPES]
        _since_floor = time.time() - _DIGEST_REQUEST_MAX_LOOKBACK_SEC

        # Collect messages that the requester hasn't seen
        missing_messages: List[Dict[str, Any]] = []

        for subtype, last_seen in _capped_items:
            try:
                # Clamp last_seen to a numeric value no older than the
                # retention floor (reject an attacker's 0 / junk that
                # would force a max-depth scan).
                try:
                    _since = max(float(last_seen), _since_floor)
                except (TypeError, ValueError):
                    _since = _since_floor
                messages = await self.ledger.get_recent_gossip(
                    since=_since,
                    subtypes=[subtype]
                )
                missing_messages.extend(messages)
            except Exception as e:
                logger.debug(f"Error fetching messages for {subtype}: {e}")
        
        # Also include messages for subtypes the requester didn't mention
        # (they may be new to the network)
        catchup_subtypes = [
            GOSSIP_JOB_OFFER,
            GOSSIP_CONTENT_ADVERTISE,
            GOSSIP_AGENT_ADVERTISE,
            GOSSIP_PROVENANCE_REGISTER,
        ]
        
        for subtype in catchup_subtypes:
            if subtype not in timestamps:
                try:
                    # Get messages from the last hour for new subtypes
                    messages = await self.ledger.get_recent_gossip(
                        since=time.time() - 3600,
                        subtypes=[subtype]
                    )
                    missing_messages.extend(messages)
                except Exception as e:
                    logger.debug(f"Error fetching messages for new subtype {subtype}: {e}")
        
        # Send response if we have messages to share
        if missing_messages:
            # Limit response size to avoid overwhelming the peer
            max_messages = 100
            if len(missing_messages) > max_messages:
                # Sort by timestamp and take most recent
                missing_messages.sort(key=lambda m: m.get("received_at", 0), reverse=True)
                missing_messages = missing_messages[:max_messages]
            
            response = P2PMessage(
                msg_type=MSG_GOSSIP,
                sender_id=self.transport.identity.node_id,
                payload={
                    "subtype": GOSSIP_DIGEST_RESPONSE,
                    "data": {
                        "messages": missing_messages,
                        "total_count": len(missing_messages),
                    },
                    "origin": self.transport.identity.node_id,
                    "origin_time": time.time(),
                },
                ttl=1,  # Direct message, don't re-propagate
            )
            
            await self.transport.send_to_peer(requester_id, response)
            logger.debug(f"Sent digest response with {len(missing_messages)} messages to {requester_id[:8]}...")

    async def _handle_digest_response(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle incoming digest response with catch-up messages.

        Processes each message as if it were a new gossip message,
        updating local timestamps and storing in local gossip log.
        """
        data = msg.payload.get("data", {})
        messages = data.get("messages", [])

        if not messages:
            logger.debug(f"Received empty digest response from {msg.sender_id[:8]}...")
            return

        # sp1270 — bound the peer-controlled list BEFORE the per-message dedup/auth loop. Each
        # message triggers a ledger scan via _is_duplicate; an unbounded response would be a
        # DoS amplifier (one ledger scan per attacker-supplied entry). Legit responses are
        # capped at 100 by the request path, so truncating to 200 never drops honest data.
        if not isinstance(messages, list):
            logger.warning(f"Digest response from {msg.sender_id[:8]}... had a non-list "
                           f"'messages' field — ignoring")
            return
        if len(messages) > _MAX_DIGEST_RESPONSE_MESSAGES:
            logger.warning(f"Digest response from {msg.sender_id[:8]}... carried "
                           f"{len(messages)} messages (> cap {_MAX_DIGEST_RESPONSE_MESSAGES}) "
                           f"— truncating to the cap")
            messages = messages[:_MAX_DIGEST_RESPONSE_MESSAGES]

        logger.info(f"Processing {len(messages)} catch-up messages from {msg.sender_id[:8]}...")
        
        processed = 0
        for message_data in messages:
            try:
                subtype = message_data.get("subtype", "")
                payload = message_data.get("payload", {})
                nonce = message_data.get("nonce", "")
                attestation = message_data.get("attestation")
                # sp961 — RE-authenticate the origin of the caught-up message
                # rather than trusting the relaying peer's claim. Without a valid
                # attestation, attribute it to the transport-authenticated relayer
                # (never the bare claimed origin) — a relayer cannot forge
                # authorship of replayed content/job/provenance/agent ads.
                origin = _authenticate_catchup_origin(
                    subtype=subtype, data=payload,
                    claimed_origin=message_data.get("origin", msg.sender_id),
                    attestation=attestation, nonce=nonce,
                    fallback_sender_id=msg.sender_id,
                )

                if not subtype or not payload:
                    continue

                # Skip if we've already seen this message (dedup)
                if nonce and await self._is_duplicate(nonce):
                    continue

                # Deliver to local subscribers
                callbacks = self._subscribers.get(subtype, [])
                for cb in callbacks:
                    try:
                        await cb(subtype, payload, origin)
                    except Exception as e:
                        logger.error(f"Error in catch-up subscriber callback ({subtype}): {e}")

                # Store in local gossip log (origin is the AUTHENTICATED value;
                # carry the attestation forward so this node can re-serve it).
                if self.ledger and subtype not in ("heartbeat", GOSSIP_DIGEST_REQUEST, GOSSIP_DIGEST_RESPONSE):
                    try:
                        await self.ledger.log_gossip(
                            nonce=nonce,
                            subtype=subtype,
                            origin=origin,
                            payload=payload,
                            ttl=1,  # Already propagated, just storing locally
                            attestation=attestation if origin == message_data.get("origin") else None,
                        )
                    except Exception:
                        pass  # Don't break on log failure
                
                processed += 1
                
            except Exception as e:
                logger.debug(f"Error processing catch-up message: {e}")
        
        logger.info(f"Processed {processed}/{len(messages)} catch-up messages from {msg.sender_id[:8]}...")

    async def _is_duplicate(self, nonce: str) -> bool:
        """Check if a message with this nonce has already been processed."""
        # The transport handles nonce dedup for regular messages
        # For catch-up messages, we check the gossip log
        if not self.ledger:
            return False
        
        try:
            # Check if this nonce exists in our log
            messages = await self.ledger.get_recent_gossip(
                since=time.time() - 86400,  # Check last 24 hours
            )
            return any(m.get("nonce") == nonce for m in messages)
        except Exception:
            return False

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat to maintain network liveness info."""
        prune_counter = 0
        while self._running:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await self.publish(
                    "heartbeat",
                    {
                        "peer_count": self.transport.peer_count,
                        "uptime": time.time(),
                    },
                    ttl=2,  # heartbeats don't need to travel far
                )
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")

            # Prune old gossip log entries every ~10 heartbeats
            prune_counter += 1
            if prune_counter >= 10 and self.ledger:
                prune_counter = 0
                try:
                    pruned = await self._prune_gossip_log_by_retention()
                    if pruned:
                        logger.debug(f"Pruned {pruned} old gossip log entries")
                except Exception as e:
                    logger.debug(f"Error pruning gossip log: {e}")

    async def _prune_gossip_log_by_retention(self) -> int:
        """Prune gossip log entries based on per-subtype retention policy.

        Uses GOSSIP_RETENTION_SECONDS to determine how long to keep
        messages of each type.
        """
        if not self.ledger:
            return 0
        
        total_pruned = 0
        current_time = time.time()
        
        # Get all subtypes with retention policies
        for subtype, retention_seconds in GOSSIP_RETENTION_SECONDS.items():
            if retention_seconds <= 0:
                continue  # Skip subtypes that shouldn't be stored
            
            cutoff = current_time - retention_seconds
            
            try:
                # Delete messages older than retention window
                # The ledger's prune_gossip_log uses a single max_age parameter,
                # so we use the minimum retention across all subtypes for now
                # A more sophisticated implementation would add subtype-specific pruning
                pass  # Handled by the ledger's prune_gossip_log with default retention
            except Exception as e:
                logger.debug(f"Error pruning {subtype}: {e}")
        
        # Use the ledger's built-in pruning with default retention
        try:
            total_pruned = await self.ledger.prune_gossip_log(self.gossip_log_retention)
        except Exception:
            pass
        
        return total_pruned
