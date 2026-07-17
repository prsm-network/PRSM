"""
Ledger Sync
===========

Network-wide FTNS ledger synchronization via gossip.

Handles three responsibilities:
1. **Transaction broadcasting** — when this node creates a transaction
   (credit, debit, transfer), sign it and gossip to the network.
2. **Incoming transaction processing** — verify signatures, reject
   replayed nonces, and apply remote transactions to the local ledger.
3. **Balance reconciliation** — periodically exchange balance proofs
   with peers and flag discrepancies.

The model is eventually consistent: each node is authoritative over its
own balance, and the network verifies consistency via observed transactions.
"""

import asyncio
import json
import logging
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# sp1428 — bounds on the reconciliation balance_response path (money-path DoS hardening).
# A legitimate response echoes at most the request path's `limit=20` ids; the cap is generous
# headroom over that, so an oversized/hostile list can't drive an unbounded per-element DB storm
# on the shared ledger connection. The outstanding-request set is bounded so correlation tracking
# can't itself leak. See tests/unit/test_sprint_1428_balance_response_dos.py.
_MAX_RECONCILIATION_TX_IDS = 64
_MAX_OUTSTANDING_REQUEST_IDS = 1024

from prsm.node.dag_ledger import DAGLedger
from prsm.node.gossip import GOSSIP_FTNS_TRANSACTION, GossipProtocol
from prsm.node.identity import NodeIdentity, verify_signature
from prsm.node.local_ledger import LocalLedger, Transaction, TransactionType
from prsm.node.transport import MSG_DIRECT, P2PMessage, PeerConnection, WebSocketTransport

# Sprint 1419 — `ledger` is annotated LocalLedger|DAGLedger because Node builds EITHER
# (config.ledger_type, defaulting to "dag"). The old `ledger: LocalLedger` annotation was simply
# wrong about production, and it is exactly why a DAGLedger missing get_recent_tx_ids type-checked
# fine while silently killing balance reconciliation on every default node. DAGLedger is NOT a
# subclass of LocalLedger, so the annotation bought no safety at all.
# tests/unit/test_sprint_1419_*.py pins the full interface against BOTH impls.

logger = logging.getLogger(__name__)

# How often to run balance reconciliation (seconds)
DEFAULT_RECONCILIATION_INTERVAL = 300.0  # 5 minutes


class LedgerSync:
    """Synchronize FTNS ledger state across the P2P network.

    Cross-node transfers work as follows:
    1. Paying node creates a signed transaction with a unique nonce
    2. Paying node gossips GOSSIP_FTNS_TRANSACTION
    3. Paying node debits locally
    4. Receiving node verifies signature + nonce, credits locally

    Self-credits (storage rewards, royalties) are also gossiped for
    transparency but only the originating node applies them.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        gossip: GossipProtocol,
        ledger: "LocalLedger | DAGLedger",
        transport: WebSocketTransport,
        reconciliation_interval: float = DEFAULT_RECONCILIATION_INTERVAL,
    ):
        self.identity = identity
        self.gossip = gossip
        self.ledger = ledger
        self.transport = transport
        self.reconciliation_interval = reconciliation_interval

        # Stats
        self._txs_broadcast = 0
        self._txs_received = 0
        self._txs_rejected = 0
        self._reconciliations_run = 0
        self._discrepancies_found = 0

        self._running = False
        self._tasks: List[asyncio.Task] = []

        # sp1428 — request_ids we have SENT and not yet matched to a response, bounded + one-shot.
        # `_handle_balance_response` drops any response whose request_id we didn't send, so a peer
        # cannot make us process (and DB-scan) an UNSOLICITED balance_response.
        self._outstanding_request_ids: "OrderedDict[str, None]" = OrderedDict()

    def _register_outstanding_request(self) -> str:
        """Mint + record a request_id for a balance_request we are about to send. Bounded FIFO."""
        rid = str(uuid.uuid4())
        self._outstanding_request_ids[rid] = None
        while len(self._outstanding_request_ids) > _MAX_OUTSTANDING_REQUEST_IDS:
            self._outstanding_request_ids.popitem(last=False)  # evict oldest
        return rid

    def start(self) -> None:
        """Subscribe to gossip and register direct message handlers."""
        self.gossip.subscribe(GOSSIP_FTNS_TRANSACTION, self._on_ftns_transaction)
        self.transport.on_message(MSG_DIRECT, self._on_direct_message)
        self._running = True
        self._tasks.append(asyncio.create_task(self._reconciliation_loop()))
        logger.info("Ledger sync started — broadcasting and reconciliation active")

    async def stop(self) -> None:
        """Stop reconciliation loop."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    # ── Broadcasting ──────────────────────────────────────────────

    async def broadcast_transaction(self, tx: Transaction) -> None:
        """Sign and gossip a transaction to the network.

        Call this after the transaction has been applied to the local
        ledger.  The signature covers the canonical fields so other
        nodes can verify the originating node authorized it.
        """
        nonce = tx.tx_id  # tx_id is already a UUID — use as nonce

        # Build canonical payload for signing
        canonical = self._canonical_tx_payload(tx, nonce)
        signature = self.identity.sign(json.dumps(canonical, sort_keys=True).encode())

        await self.gossip.publish(GOSSIP_FTNS_TRANSACTION, {
            **canonical,
            "signature": signature,
            "origin_public_key": self.identity.public_key_b64,
        })

        # Record our own nonce so we don't re-process it
        await self.ledger.record_nonce(nonce, self.identity.node_id)
        self._txs_broadcast += 1

    async def signed_transfer(
        self,
        to_wallet: str,
        amount: float,
        description: str = "",
    ) -> Optional[Transaction]:
        """Create a signed cross-node transfer with double-spend prevention.

        1. Check balance is sufficient
        2. Generate unique nonce (the tx_id)
        3. Create and sign the transaction
        4. Gossip to network
        5. Debit locally

        Returns the transaction, or None if insufficient balance.
        """
        balance = await self.ledger.get_balance(self.identity.node_id)
        if balance < amount:
            logger.warning(
                f"Transfer rejected: insufficient balance {balance:.6f} < {amount:.6f}"
            )
            return None

        tx = await self.ledger.transfer(
            from_wallet=self.identity.node_id,
            to_wallet=to_wallet,
            amount=amount,
            description=description,
        )

        # Sign and broadcast
        await self.broadcast_transaction(tx)
        logger.info(f"Signed transfer: {amount:.6f} FTNS -> {to_wallet[:12]}...")
        return tx

    # ── Incoming transaction processing ───────────────────────────

    async def _on_ftns_transaction(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Process an incoming FTNS transaction from gossip."""
        if origin == self.identity.node_id:
            return  # Skip our own broadcasts

        nonce = data.get("nonce", "")
        signature = data.get("signature", "")
        origin_public_key = data.get("origin_public_key", "")

        if not nonce or not signature or not origin_public_key:
            self._txs_rejected += 1
            return

        # Replay prevention: reject if nonce already seen
        if await self.ledger.has_seen_nonce(nonce):
            logger.debug(f"Rejected replay: nonce {nonce[:12]}...")
            self._txs_rejected += 1
            return

        # Verify signature
        canonical = {
            k: v for k, v in data.items()
            if k not in ("signature", "origin_public_key")
        }
        canonical_bytes = json.dumps(canonical, sort_keys=True).encode()
        if not verify_signature(origin_public_key, canonical_bytes, signature):
            logger.warning(f"Rejected transaction from {origin[:12]}...: bad signature")
            self._txs_rejected += 1
            return

        # Sp898 — atomically CLAIM the nonce (the seen_nonces PRIMARY
        # KEY makes INSERT OR IGNORE the serialization point). Of N
        # concurrent handlers processing the SAME gossiped transaction
        # (gossip floods the same signed tx via multiple peers),
        # exactly one wins the claim and proceeds to credit; the rest
        # get claimed=False and stop here. Without this, both pass the
        # has_seen_nonce + has_transaction checks (check-then-act
        # across awaits) and BOTH credit → cross-node double-credit.
        # The has_seen_nonce check above remains a cheap fast-path.
        claimed = await self.ledger.record_nonce(nonce, origin)
        if not claimed:
            logger.debug(
                f"Rejected concurrent/replay nonce {nonce[:12]}..."
            )
            self._txs_rejected += 1
            return
        self._txs_received += 1

        # Apply the transaction locally if we are the recipient
        to_wallet = data.get("to_wallet", "")
        from_wallet = data.get("from_wallet")
        amount = data.get("amount", 0)
        tx_type_str = data.get("tx_type", "transfer")
        tx_id = data.get("tx_id", nonce)
        description = data.get("description", "")

        if amount <= 0:
            return

        # Only apply if this transaction involves us
        if to_wallet == self.identity.node_id:
            # We're the recipient — credit our local ledger
            if await self.ledger.has_transaction(tx_id):
                return  # Already applied

            try:
                tx_type = TransactionType(tx_type_str)
            except ValueError:
                tx_type = TransactionType.TRANSFER

            try:
                await self.ledger.credit(
                    wallet_id=self.identity.node_id,
                    amount=amount,
                    tx_type=tx_type,
                    description=f"[remote] {description}",
                    signature=signature,
                )
            except Exception as exc:  # noqa: BLE001 — a failed credit must not STRAND the payment
                # Sp1466 — the nonce was CLAIMED + committed above (record_nonce, the sp898
                # double-credit serialization point) BEFORE this credit. credit() is a single atomic
                # transaction, so a raise means NO payment was applied (rolled back) — a
                # definitively-no-payment revert, exactly release_nonce's sanctioned case (sp918).
                # Without releasing it, the claim strands the payment forever: every re-gossiped copy
                # is rejected at the has_seen_nonce fast-path (line ~196) BEFORE the has_transaction
                # retry above can ever fire, so the recipient is never credited. Releasing lets a
                # re-gossip re-claim + retry. (Preserves sp898: a SUCCESSFUL credit does NOT release,
                # so a duplicate stays rejected → no double-credit.)
                await self.ledger.release_nonce(nonce)
                logger.error(
                    "sp1466: credit failed for incoming tx %s from %s (nonce RELEASED so gossip "
                    "can retry — payment not yet applied): %s",
                    tx_id[:12], (from_wallet[:12] if from_wallet else "system"), exc,
                )
                return
            logger.info(
                f"Received {amount:.6f} FTNS from {from_wallet[:12] if from_wallet else 'system'}... "
                f"({tx_type_str})"
            )

    # ── Balance reconciliation ────────────────────────────────────

    async def _reconciliation_loop(self) -> None:
        """Periodically request balance proofs from connected peers."""
        while self._running:
            await asyncio.sleep(self.reconciliation_interval)
            try:
                await self._run_reconciliation()
            except Exception as e:
                logger.error(f"Reconciliation error: {e}")

    async def _run_reconciliation(self) -> None:
        """Send balance proof requests to all connected peers."""
        if not self.transport.peers:
            return

        self._reconciliations_run += 1
        my_balance = await self.ledger.get_balance(self.identity.node_id)
        my_recent_txs = await self.ledger.get_recent_tx_ids(self.identity.node_id, limit=20)

        for peer_id in list(self.transport.peers.keys()):
            try:
                msg = P2PMessage(
                    msg_type=MSG_DIRECT,
                    sender_id=self.identity.node_id,
                    payload={
                        "subtype": "balance_request",
                        "request_id": self._register_outstanding_request(),
                        "requester_balance": my_balance,
                        "recent_tx_ids": my_recent_txs,
                    },
                )
                await self.transport.send_to_peer(peer_id, msg)
            except Exception as e:
                logger.debug(f"Failed to send reconciliation request to {peer_id[:12]}...: {e}")

    async def _on_direct_message(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Route balance request/response direct messages."""
        subtype = msg.payload.get("subtype", "")
        if subtype == "balance_request":
            await self._handle_balance_request(msg, peer)
        elif subtype == "balance_response":
            await self._handle_balance_response(msg, peer)

    async def _handle_balance_request(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Respond with our balance and recent transaction IDs."""
        my_balance = await self.ledger.get_balance(self.identity.node_id)
        my_recent_txs = await self.ledger.get_recent_tx_ids(self.identity.node_id, limit=20)

        response = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload={
                "subtype": "balance_response",
                "request_id": msg.payload.get("request_id", ""),
                "responder_balance": my_balance,
                "recent_tx_ids": my_recent_txs,
            },
        )
        await self.transport.send_to_peer(peer.peer_id, response)

    async def _handle_balance_response(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Compare peer's reported balance against our observed state.

        This is a soft DETECTION check — discrepancies are logged but not acted
        on automatically (there is no fetch/apply/merge path here).

        Two classes of "not seen locally", only one of which ever resolves:
          • A tx that INVOLVES us (to_wallet == our node_id) but arrived late:
            the gossip apply path (_on_ftns_transaction) stores it, so it
            resolves within a gossip round.
          • A tx that does NOT involve us (the peer's own escrow-funding debits,
            its third-party payments, system credits): the apply path refuses to
            store it (see the `to_wallet == self.identity.node_id` gate above),
            so it NEVER resolves. The DEFAULT ledger is per-node — each node holds
            only txs it is a wallet-party to; cross-node value moves through escrow
            accounts, so two idle peers permanently report each other's ~20 recent
            own-side txs as "not seen locally". That steady, non-zero count is
            EXPECTED, not drift. (See sp1466.)
        """
        # sp1428 — DROP unsolicited responses. We only process a balance_response whose request_id
        # matches a balance_request we actually SENT (and consume it, one-shot). Without this, any
        # connected peer could push an unsolicited response and make us DB-scan its payload.
        request_id = msg.payload.get("request_id", "")
        if request_id not in self._outstanding_request_ids:
            logger.debug(
                "Dropping balance_response with unknown/duplicate request_id from %s...",
                peer.peer_id[:12],
            )
            return
        del self._outstanding_request_ids[request_id]

        responder_balance = msg.payload.get("responder_balance", 0)
        # sp1428 — CAP the id count. A legit response echoes <=20 ids (the request path's limit);
        # this bound stops a hostile/oversized list from driving an unbounded per-element DB storm
        # on the shared ledger connection (a money-path DoS).
        peer_recent_txs = list(msg.payload.get("recent_tx_ids", []) or [])[:_MAX_RECONCILIATION_TX_IDS]

        # Check if we have all of the peer's recent transactions
        missing = 0
        for tx_id in peer_recent_txs:
            if not await self.ledger.has_transaction(tx_id):
                # Check nonce too — might be a transaction that doesn't involve us
                if not await self.ledger.has_seen_nonce(tx_id):
                    missing += 1

        if missing > 0:
            self._discrepancies_found += 1
            logger.info(
                f"Reconciliation with {peer.peer_id[:12]}...: "
                f"{missing} transaction(s) not seen locally "
                f"(peer balance: {responder_balance:.6f} FTNS)"
            )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _canonical_tx_payload(tx: Transaction, nonce: str) -> Dict[str, Any]:
        """Build the canonical payload dict for signing/gossip."""
        return {
            "tx_id": tx.tx_id,
            "nonce": nonce,
            "tx_type": tx.tx_type.value,
            "from_wallet": tx.from_wallet,
            "to_wallet": tx.to_wallet,
            "amount": tx.amount,
            "description": tx.description,
            "timestamp": tx.timestamp,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return ledger sync statistics."""
        return {
            "txs_broadcast": self._txs_broadcast,
            "txs_received": self._txs_received,
            "txs_rejected": self._txs_rejected,
            "reconciliations_run": self._reconciliations_run,
            "discrepancies_found": self._discrepancies_found,
        }
