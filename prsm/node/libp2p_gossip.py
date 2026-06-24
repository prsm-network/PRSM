"""
libp2p GossipSub Wrapper
========================

Thin wrapper around ``Libp2pTransport`` that exposes the same public API
as ``GossipProtocol`` (prsm/node/gossip.py).

The key difference from the WebSocket-based GossipProtocol is that
GossipSub topic subscription is *lazy*: ``PrsmSubscribe`` is only called
once per unique subtype, the first time a callback is registered for it.
Subsequent callbacks for the same subtype reuse the existing subscription.

Message envelope format (JSON, published to GossipSub topic):
    {
        "subtype":      str,    # e.g. "job_offer"
        "data":         dict,
        "sender_id":    str,
        "timestamp":    float,
        # sp1246 — sp934 origin attestation (lets a receiver authenticate the
        # author across multi-hop relay) + the nonce bound into its signature:
        "nonce":        str,
        "origin":       str,    # author node_id
        "origin_time":  float,
        "origin_pubkey": str,   # base64
        "origin_sig":   str,    # base64
    }
"""

import json
import logging
import time
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional

from prsm.node.gossip import (
    _attestation_from_payload,
    _authenticate_origin,
    build_gossip_origin_fields,
)
from prsm.node.transport import MSG_GOSSIP, P2PMessage

logger = logging.getLogger(__name__)

# Re-export the same callback type as gossip.py
GossipCallback = Callable[[str, Dict[str, Any], str], Coroutine[Any, Any, None]]
# (subtype, data, origin) -> None  — sp1246: the 3rd arg is the AUTHENTICATED
# origin (sp934), not the raw relay sender_id.


class Libp2pGossip:
    """GossipSub-backed gossip layer for Libp2pTransport.

    Provides the same public interface as ``GossipProtocol`` so higher-level
    code (node.py, tests) can swap implementations without changes.
    """

    def __init__(self, transport: Any, **kwargs: Any) -> None:
        """
        Args:
            transport: A ``Libp2pTransport`` instance.
            **kwargs:  Ignored (for drop-in compatibility with GossipProtocol).
        """
        self.transport = transport

        # Set post-construction by node.py (same pattern as GossipProtocol)
        self.ledger: Optional[Any] = None

        # subtype -> list of callbacks
        self._callbacks: Dict[str, List[GossipCallback]] = {}

        # Subtypes for which PrsmSubscribe has already been called
        self._subscribed_topics: set = set()
        # Topics queued during __init__-time subscribes (before
        # transport.start()). Flushed on first start().
        self._pending_topics: set = set()

        # Additive telemetry counters (never change gossip behaviour)
        self._telemetry: Dict[str, int] = {
            "publish_total": 0,
            "deliver_total": 0,
            "error_total": 0,
        }

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Register handler for inbound gossip messages from the transport.

        Flushes pending subscriptions queued during __init__ paths
        that ran before the libp2p host was ready.
        """
        self.transport.on_message(MSG_GOSSIP, self._handle_gossip)
        # Flush deferred subscriptions now that the host is up
        if self._pending_topics:
            for topic in list(self._pending_topics):
                if self.transport._handle < 0:
                    break  # transport still not ready (shouldn't happen)
                try:
                    self.transport._lib.PrsmSubscribe(
                        self.transport._handle,
                        topic.encode("utf-8"),
                    )
                    self._subscribed_topics.add(topic)
                except Exception as exc:
                    logger.warning(
                        "PrsmSubscribe failed for topic %s during "
                        "start() flush: %s", topic, exc,
                    )
                self._pending_topics.discard(topic)
        logger.info("Libp2pGossip started")

    async def stop(self) -> None:
        """No-op — GossipSub shuts down with the libp2p host."""

    # ── Public API (mirrors GossipProtocol) ──────────────────────

    def subscribe(self, subtype: str, callback: GossipCallback) -> None:
        """Register a callback for *subtype*.

        Lazily calls ``PrsmSubscribe`` the first time a subtype is registered.

        Guards against subscribing before the libp2p host has
        started (handle < 0). When called pre-start, the topic
        is recorded but the C-side subscription is deferred —
        callers register subscriptions during __init__ paths
        that run before transport.start(). Without the guard,
        the C bridge logs a noisy `handle -1 not found` and
        the topic gets recorded as "subscribed" anyway,
        leaving the GossipSub layer permanently desynced.
        """
        topic = self._topic_name(subtype)

        # Always register the Python-side callback so messages
        # are dispatched correctly once the topic IS subscribed.
        self._callbacks.setdefault(subtype, []).append(callback)

        # Defer the C-bridge call when handle isn't ready yet.
        # Track as pending so transport.start() can flush.
        if self.transport._handle < 0:
            if topic not in self._subscribed_topics:
                self._pending_topics.add(topic)
            return

        # Handle is ready — proceed with subscription
        if topic not in self._subscribed_topics:
            try:
                self.transport._lib.PrsmSubscribe(
                    self.transport._handle,
                    topic.encode("utf-8"),
                )
                self._subscribed_topics.add(topic)
            except Exception as exc:
                logger.warning(
                    "PrsmSubscribe failed for topic %s: %s",
                    topic, exc,
                )
                # NOT adding to _subscribed_topics — let next
                # call retry rather than silently dropping

    async def publish(
        self,
        subtype: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> int:
        """Publish a gossip message over GossipSub.

        Args:
            subtype: Message subtype label (e.g. ``"job_offer"``).
            data:    Arbitrary payload dict.
            ttl:     Ignored in GossipSub mode (kept for API compatibility).

        Returns:
            1 on success, 0 on failure.
        """
        topic = self._topic_name(subtype)
        sender_id = self.transport.identity.node_id

        # sp1246 (Residual A) — attach the sp934 origin attestation + a fresh nonce
        # (bound into the signed bytes) so a receiver can authenticate THIS node as
        # the author across multi-hop GossipSub relay, exactly as the WebSocket
        # GossipProtocol.publish does. A discovery `data` payload may ALSO carry its
        # own sp937/sp1086 attestation — the two layers coexist (envelope vs data)
        # and must not be conflated.
        nonce = uuid.uuid4().hex[:16]
        origin_fields = build_gossip_origin_fields(
            self.transport.identity, subtype, data, time.time(), nonce,
        )
        envelope = {
            "subtype": subtype,
            "data": data,
            "sender_id": sender_id,
            "timestamp": time.time(),
            "nonce": nonce,
            **origin_fields,
        }

        payload_bytes = json.dumps(envelope).encode("utf-8")

        rc = 0
        try:
            rc = self.transport._lib.PrsmPublish(
                self.transport._handle,
                topic.encode("utf-8"),
                payload_bytes,
                len(payload_bytes),
            )
        except Exception as exc:
            logger.error("PrsmPublish error (%s): %s", subtype, exc)
            self._telemetry["error_total"] += 1
            return 0

        if rc == 0:
            self._telemetry["publish_total"] += 1

            # Optional ledger persistence (fire-and-forget)
            if self.ledger and subtype != "heartbeat":
                try:
                    await self.ledger.log_gossip(
                        nonce=nonce,
                        subtype=subtype,
                        origin=sender_id,
                        payload=data,
                        ttl=1,
                    )
                except Exception:
                    pass

            return 1

        self._telemetry["error_total"] += 1
        return 0

    async def get_catchup_messages(
        self,
        since: float,
        subtypes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Delegate to the ledger when available (same as GossipProtocol)."""
        if not self.ledger:
            return []
        return await self.ledger.get_recent_gossip(since, subtypes)

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return a stable copy of gossip telemetry counters."""
        return {
            "publish_total": int(self._telemetry["publish_total"]),
            "deliver_total": int(self._telemetry["deliver_total"]),
            "error_total": int(self._telemetry["error_total"]),
            "subscribed_topics": list(self._subscribed_topics),
        }

    # ── Internal ─────────────────────────────────────────────────

    async def _handle_gossip(
        self, msg: P2PMessage, peer_info: Any
    ) -> None:
        """Dispatch an inbound gossip message to registered callbacks."""
        try:
            payload = msg.payload
            if isinstance(payload, (str, bytes)):
                payload = json.loads(payload)

            subtype = payload.get("subtype", "")
            data = payload.get("data", {})
            sender_id = payload.get("sender_id", msg.sender_id)
        except Exception as exc:
            logger.debug("Failed to parse gossip envelope: %s", exc)
            return

        if not subtype:
            return

        # sp1246 (Residual A) — authenticate the ORIGIN, exactly as the WebSocket
        # GossipProtocol does: trust payload['origin'] only if its attestation
        # verifies (pubkey hashes to the claimed node_id AND the signature checks
        # out), else fall back to the transport-relayed sender_id — NEVER the bare,
        # unsigned payload origin. Security-critical subscribers (settlement _on_ad:
        # admit iff origin == announcer_id) are thus secure-by-design, not merely
        # fail-safe-by-omission. Discovery subscribers self-verify their `data`
        # (sp1086/1087/1097) and ignore this arg, so they are unaffected.
        origin = _authenticate_origin(payload, payload.get("nonce", ""), sender_id)

        callbacks = self._callbacks.get(subtype, [])
        for cb in callbacks:
            try:
                await cb(subtype, data, origin)
                self._telemetry["deliver_total"] += 1
            except Exception as exc:
                logger.error("Gossip callback error (%s): %s", subtype, exc)
                self._telemetry["error_total"] += 1

        # Optional ledger persistence. sp1246 — mirror the WebSocket GossipProtocol
        # (gossip.py): persist the AUTHENTICATED origin (not the relay sender_id),
        # the signature-bound envelope nonce (not msg.nonce, which the transport may
        # regenerate), and the sp961 attestation so a future digest catch-up consumer
        # can RE-verify authorship relayer-independently (_authenticate_catchup_origin).
        if self.ledger and subtype != "heartbeat":
            try:
                await self.ledger.log_gossip(
                    nonce=payload.get("nonce") or msg.nonce,
                    subtype=subtype,
                    origin=origin,
                    payload=data,
                    ttl=msg.ttl,
                    attestation=_attestation_from_payload(payload),
                )
            except Exception:
                pass

    @staticmethod
    def _topic_name(subtype: str) -> str:
        """Map a gossip subtype to a GossipSub topic string."""
        return f"prsm/{subtype}"
