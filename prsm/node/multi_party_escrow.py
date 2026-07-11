"""
Multi-Party On-Chain Escrow for Content Royalties
==================================================

Batches royalty payments for gas-efficient on-chain settlement.
Instead of sending individual FTNS transfers for each content access,
this system:
1. Accumulates royalties in a local ledger
2. Batches them into single on-chain transactions
3. Distributes to multiple creators in one transaction

This reduces gas costs significantly when paying multiple creators.

Usage:
    from prsm.node.multi_party_escrow import MultiPartyEscrow, EscrowConfig
    
    escrow = MultiPartyEscrow(ftns_ledger, config)
    await escrow.initialize()
    
    # Accumulate royalties
    await escrow.accumulate("creator-a", 0.05, "QmCID1")
    await escrow.accumulate("creator-b", 0.03, "QmCID2")
    
    # Batch settle on-chain
    result = await escrow.settle_batch()
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass
class EscrowConfig:
    """Configuration for multi-party escrow."""
    # Settlement thresholds
    min_batch_size: int = 5  # Minimum creators per batch
    min_batch_value: float = 0.1  # Minimum total FTNS per batch
    max_batch_size: int = 50  # Maximum creators per batch (gas limit)
    
    # Timing
    settlement_interval: float = 300.0  # Seconds between auto-settlements
    max_pending_age: float = 3600.0  # Max seconds before forced settlement
    
    # Gas optimization
    gas_price_multiplier: float = 1.0  # For EIP-1559
    max_gas_per_tx: int = 500_000  # Approximate gas limit
    
    # Creator resolution
    require_onchain_address: bool = False  # Skip creators without addresses


@dataclass
class PendingRoyalty:
    """A pending royalty payment."""
    creator_id: str
    amount: float
    source_cid: str
    accessor_id: str
    timestamp: float
    creator_address: Optional[str] = None  # On-chain address if known


@dataclass
class CreatorAccumulator:
    """Accumulated royalties for a single creator."""
    creator_id: str
    total_amount: float
    royalties: List[PendingRoyalty]
    onchain_address: Optional[str] = None
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    
    def add(self, royalty: PendingRoyalty) -> None:
        self.total_amount += royalty.amount
        self.royalties.append(royalty)
        if self.first_timestamp == 0:
            self.first_timestamp = royalty.timestamp
        self.last_timestamp = royalty.timestamp


@dataclass
class SettlementBatch:
    """A batch of royalties ready for on-chain settlement."""
    batch_id: str
    creators: Dict[str, CreatorAccumulator]  # creator_id -> accumulator
    total_amount: float
    created_at: float
    settled_at: Optional[float] = None
    tx_hash: Optional[str] = None
    gas_used: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "creator_count": len(self.creators),
            "total_amount": self.total_amount,
            "created_at": self.created_at,
            "settled_at": self.settled_at,
            "tx_hash": self.tx_hash,
            "gas_used": self.gas_used,
            "error": self.error,
        }


# ── Multi-Party Escrow ─────────────────────────────────────────────────────

class MultiPartyEscrow:
    """Manages batched royalty payments for gas-efficient settlement."""
    
    def __init__(
        self,
        ftns_ledger: Any,  # OnChainFTNSLedger
        config: Optional[EscrowConfig] = None,
        creator_registry: Optional[Any] = None,  # Optional address lookup
    ):
        self.ftns_ledger = ftns_ledger
        self.config = config or EscrowConfig()
        self.creator_registry = creator_registry
        
        # Pending royalties by creator
        self._pending: Dict[str, CreatorAccumulator] = {}
        
        # Settlement history
        self._batches: List[SettlementBatch] = []
        self._total_settled: float = 0.0
        self._total_gas_used: int = 0
        
        # Background task
        self._running = False
        self._settlement_task: Optional[asyncio.Task] = None
        
        # Callback for settlement events
        self._on_settlement: Optional[callable] = None
    
    async def initialize(self) -> bool:
        """Initialize the escrow system."""
        if not self.ftns_ledger:
            logger.warning("No FTNS ledger configured, escrow will run in simulation mode")
            return True
        
        # Check if ledger is connected
        if hasattr(self.ftns_ledger, "_connected_address"):
            if not self.ftns_ledger._connected_address:
                logger.warning("FTNS ledger not connected to on-chain")
        
        logger.info(
            f"Multi-party escrow initialized "
            f"(min_batch_size={self.config.min_batch_size}, "
            f"min_batch_value={self.config.min_batch_value} FTNS)"
        )
        return True
    
    def start(self) -> None:
        """Start background settlement loop."""
        if self._running:
            return
        
        self._running = True
        self._settlement_task = asyncio.create_task(self._settlement_loop())
        logger.info("Multi-party escrow background loop started")
    
    def stop(self) -> None:
        """Stop background settlement loop."""
        self._running = False
        if self._settlement_task:
            self._settlement_task.cancel()
            self._settlement_task = None
    
    # ── Royalty Accumulation ────────────────────────────────────────────────
    
    async def accumulate(
        self,
        creator_id: str,
        amount: float,
        source_cid: str,
        accessor_id: str,
        creator_address: Optional[str] = None,
    ) -> None:
        """Accumulate a royalty payment for later batch settlement.
        
        Args:
            creator_id: Creator's node ID
            amount: Royalty amount in FTNS
            source_cid: Content CID that generated the royalty
            accessor_id: Node/user that accessed the content
            creator_address: Creator's on-chain address (if known)
        """
        royalty = PendingRoyalty(
            creator_id=creator_id,
            amount=amount,
            source_cid=source_cid,
            accessor_id=accessor_id,
            timestamp=time.time(),
            creator_address=creator_address,
        )
        
        # Get or create accumulator
        if creator_id not in self._pending:
            # Try to resolve on-chain address
            if not creator_address and self.creator_registry:
                creator_address = await self._resolve_address(creator_id)
            
            self._pending[creator_id] = CreatorAccumulator(
                creator_id=creator_id,
                total_amount=0.0,
                royalties=[],
                onchain_address=creator_address,
            )
        
        # Add to accumulator
        self._pending[creator_id].add(royalty)
        
        logger.debug(
            f"Accumulated royalty: {amount:.6f} FTNS for {creator_id[:8]} "
            f"(total now: {self._pending[creator_id].total_amount:.6f})"
        )
    
    async def _resolve_address(self, creator_id: str) -> Optional[str]:
        """Resolve on-chain address for a creator."""
        if not self.creator_registry:
            return None
        
        try:
            # Check if registry has address lookup
            if hasattr(self.creator_registry, "get_address"):
                return await self.creator_registry.get_address(creator_id)
        except Exception as e:
            logger.debug(f"Could not resolve address for {creator_id[:8]}: {e}")
        
        return None
    
    # ── Settlement ──────────────────────────────────────────────────────────
    
    async def settle_batch(
        self,
        force: bool = False,
    ) -> Optional[SettlementBatch]:
        """Create and settle a batch of accumulated royalties.
        
        Args:
            force: Force settlement even if thresholds not met
            
        Returns:
            SettlementBatch if settled, None if nothing to settle
        """
        if not self._pending:
            return None
        
        # Check thresholds
        total_pending = sum(acc.total_amount for acc in self._pending.values())
        creator_count = len(self._pending)
        
        if not force:
            if creator_count < self.config.min_batch_size:
                if total_pending < self.config.min_batch_value:
                    # Check if any pending is too old
                    oldest = min(acc.first_timestamp for acc in self._pending.values())
                    if time.time() - oldest < self.config.max_pending_age:
                        logger.debug(
                            f"Skipping settlement: {creator_count} creators, "
                            f"{total_pending:.4f} FTNS (below thresholds)"
                        )
                        return None
        
        # Create batch
        batch_id = f"batch-{int(time.time() * 1000)}"
        
        # Limit batch size for gas
        creators_to_settle = dict(list(self._pending.items())[:self.config.max_batch_size])
        
        batch = SettlementBatch(
            batch_id=batch_id,
            creators=creators_to_settle,
            total_amount=sum(acc.total_amount for acc in creators_to_settle.values()),
            created_at=time.time(),
        )
        
        logger.info(
            f"Settling batch {batch_id}: {len(creators_to_settle)} creators, "
            f"{batch.total_amount:.6f} FTNS total"
        )
        
        # Attempt on-chain settlement
        try:
            result = await self._execute_onchain_settlement(batch)
            
            if result.get("success"):
                batch.tx_hash = result.get("tx_hash")
                batch.gas_used = result.get("gas_used")
                batch.settled_at = time.time()

                # Determine which creators to clear from pending.
                # For partial settlements (individual transfer fallback),
                # only remove creators whose transfers actually succeeded.
                if result.get("partial"):
                    settled_ids = set(result.get("settled_creators", []))
                    failed_ids = result.get("failed_creators", [])
                    settled_amount = 0.0
                    for creator_id in settled_ids:
                        if creator_id in self._pending:
                            settled_amount += self._pending[creator_id].total_amount
                            del self._pending[creator_id]
                    if failed_ids:
                        logger.warning(
                            f"Batch {batch_id}: {len(failed_ids)} creators failed "
                            f"settlement, keeping in pending for retry"
                        )
                    self._total_settled += settled_amount
                else:
                    # Atomic settlement (batch/multicall/simulation) — all or nothing FOR THE
                    # RECIPIENTS. sp1426: but the recipients are ONLY the creators with a
                    # resolvable on-chain address — _execute_onchain_settlement silently drops
                    # address-less creators from recipients/amounts (require_onchain_address is
                    # False by default). So clear + book ONLY the creators that were actually
                    # paid; a creator with no address was NOT in the transfer and MUST stay
                    # pending (so it can be retried once an address resolves). Booking it as
                    # settled would mark its on-chain claim satisfied without paying it — a fund
                    # loss — and delete the only record that could recover it. This mirrors the
                    # `partial` branch above, which already clears only what actually settled.
                    settled_amount = 0.0
                    for creator_id, acc in creators_to_settle.items():
                        if acc.onchain_address and creator_id in self._pending:
                            settled_amount += self._pending[creator_id].total_amount
                            del self._pending[creator_id]
                    self._total_settled += settled_amount

                if batch.gas_used:
                    self._total_gas_used += batch.gas_used

                logger.info(
                    f"Batch {batch_id} settled on-chain: "
                    f"{batch.tx_hash[:16] if batch.tx_hash else 'N/A'}... "
                    f"(gas: {batch.gas_used})"
                )

            else:
                batch.error = result.get("error", "Unknown error")
                logger.error(f"Batch {batch_id} settlement failed: {batch.error}")
                
        except Exception as e:
            batch.error = str(e)
            logger.error(f"Batch {batch_id} settlement exception: {e}")
        
        # Record batch
        self._batches.append(batch)
        
        # Callback
        if self._on_settlement:
            try:
                await self._on_settlement(batch)
            except Exception as e:
                logger.warning(f"Settlement callback error: {e}")
        
        return batch
    
    async def _execute_onchain_settlement(
        self,
        batch: SettlementBatch,
    ) -> Dict[str, Any]:
        """Execute the on-chain settlement transaction.
        
        For contracts that support batch transfers:
        - FTNS.transferBatch(recipients[], amounts[])
        
        For contracts without batch support:
        - Execute multiple individual transfers
        - Or use multicall pattern
        """
        if not self.ftns_ledger:
            # Simulation mode
            return {
                "success": True,
                "tx_hash": f"sim-{batch.batch_id}",
                "gas_used": 0,
            }
        
        # Build recipients, amounts, and map addresses back to creator_ids
        recipients = []
        amounts = []
        address_to_creator: Dict[str, str] = {}

        for creator_id, acc in batch.creators.items():
            # Skip creators without on-chain address
            if not acc.onchain_address:
                if self.config.require_onchain_address:
                    logger.debug(f"Skipping {creator_id[:8]}: no on-chain address")
                    continue

            if acc.onchain_address:
                recipients.append(acc.onchain_address)
                amounts.append(acc.total_amount)
                address_to_creator[acc.onchain_address] = creator_id

        if not recipients:
            return {"success": False, "error": "No valid recipients"}

        try:
            # Check if FTNS contract has batch transfer method
            if hasattr(self.ftns_ledger, "transfer_batch"):
                # Use batch transfer (most gas efficient) — atomic, all or nothing
                result = await self.ftns_ledger.transfer_batch(
                    recipients=recipients,
                    amounts=amounts,
                )
                return result

            elif hasattr(self.ftns_ledger, "multicall"):
                # Use multicall pattern — atomic, all or nothing
                calls = []
                for recipient, amount in zip(recipients, amounts):
                    calls.append({
                        "method": "transfer",
                        "args": [recipient, int(amount * 1e18)],  # Convert to wei
                    })

                result = await self.ftns_ledger.multicall(calls)
                return result

            else:
                # Fallback: individual transfers
                # Non-atomic — track per-creator success to avoid data loss
                tx_hashes = []
                total_gas = 0
                settled_creators = []
                failed_creators = []

                for recipient, amount in zip(recipients, amounts):
                    creator_id = address_to_creator.get(recipient, "")
                    result = await self.ftns_ledger.transfer(
                        recipient=recipient,
                        amount=amount,
                    )

                    if result.get("success"):
                        tx_hashes.append(result.get("tx_hash"))
                        total_gas += result.get("gas_used", 0)
                        settled_creators.append(creator_id)
                    else:
                        logger.warning(
                            f"Individual transfer failed for {recipient[:10]}...: "
                            f"{result.get('error')}"
                        )
                        failed_creators.append(creator_id)

                if tx_hashes:
                    return {
                        "success": True,
                        "tx_hash": tx_hashes[0],  # Primary tx
                        "gas_used": total_gas,
                        "additional_txs": tx_hashes[1:],
                        "settled_creators": settled_creators,
                        "failed_creators": failed_creators,
                        "partial": len(failed_creators) > 0,
                    }
                else:
                    return {"success": False, "error": "All transfers failed"}

        except Exception as e:
            logger.error(f"On-chain settlement error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _settlement_loop(self) -> None:
        """Background loop for periodic settlement."""
        while self._running:
            await asyncio.sleep(self.config.settlement_interval)
            
            if self._pending:
                try:
                    await self.settle_batch()
                except Exception as e:
                    logger.error(f"Auto-settlement error: {e}")
    
    # ── Query Methods ───────────────────────────────────────────────────────
    
    def get_pending_stats(self) -> Dict[str, Any]:
        """Get statistics about pending royalties."""
        if not self._pending:
            return {
                "creator_count": 0,
                "total_amount": 0.0,
                "oldest_pending_age": 0.0,
            }
        
        total = sum(acc.total_amount for acc in self._pending.values())
        oldest_ts = min(acc.first_timestamp for acc in self._pending.values())
        oldest_age = time.time() - oldest_ts
        
        return {
            "creator_count": len(self._pending),
            "total_amount": total,
            "oldest_pending_age": oldest_age,
            "largest_creator": max(
                self._pending.items(),
                key=lambda x: x[1].total_amount,
            )[0][:8] if self._pending else None,
        }
    
    def get_settlement_history(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent settlement history."""
        return [b.to_dict() for b in self._batches[-limit:]]
    
    def get_total_stats(self) -> Dict[str, Any]:
        """Get total settlement statistics."""
        return {
            "total_batches": len(self._batches),
            "successful_batches": sum(1 for b in self._batches if b.tx_hash),
            "total_settled_ftns": self._total_settled,
            "total_gas_used": self._total_gas_used,
            "pending": self.get_pending_stats(),
        }
    
    def get_creator_pending(
        self,
        creator_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get pending royalties for a specific creator."""
        acc = self._pending.get(creator_id)
        if not acc:
            return None
        
        return {
            "creator_id": creator_id,
            "total_amount": acc.total_amount,
            "royalty_count": len(acc.royalties),
            "first_pending": acc.first_timestamp,
            "last_pending": acc.last_timestamp,
            "onchain_address": acc.onchain_address,
        }
    
    # ── Callback Registration ───────────────────────────────────────────────
    
    def on_settlement(self, callback: callable) -> None:
        """Register a callback for settlement events.
        
        Callback receives SettlementBatch as argument.
        """
        self._on_settlement = callback


# ── Integration Helper ──────────────────────────────────────────────────────

class ContentEconomyEscrowBridge:
    """Bridge between ContentEconomy and MultiPartyEscrow.
    
    This class integrates the multi-party escrow into the content
    economy payment flow, automatically batching royalties for
    on-chain settlement.
    """
    
    def __init__(
        self,
        content_economy: Any,  # ContentEconomy
        escrow: MultiPartyEscrow,
    ):
        self.content_economy = content_economy
        self.escrow = escrow
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the bridge and start escrow background loop."""
        # Register callback to accumulate royalties
        self.escrow.on_settlement(self._on_settlement_callback)
        
        # Start escrow background loop
        self.escrow.start()
        
        self._initialized = True
        logger.info("Content economy escrow bridge initialized")
        return True
    
    async def accumulate_from_payment(
        self,
        payment: Any,  # ContentAccessPayment
    ) -> None:
        """Accumulate royalties from a content access payment.
        
        Called by ContentEconomy after processing a payment.
        """
        if not self._initialized:
            return
        
        for dist in payment.royalty_distributions:
            recipient_id = dist.get("recipient_id")
            amount = dist.get("amount", 0)
            dist_type = dist.get("type")
            
            # Skip network fees and system payments
            if recipient_id == "system":
                continue
            
            # Accumulate for batch settlement
            await self.escrow.accumulate(
                creator_id=recipient_id,
                amount=amount,
                source_cid=payment.content_id,
                accessor_id=payment.accessor_id,
            )
    
    async def _on_settlement_callback(
        self,
        batch: SettlementBatch,
    ) -> None:
        """Called when a batch is settled."""
        # Could trigger additional actions like:
        # - Notify creators
        # - Update local ledger
        # - Record on-chain proofs
        
        logger.info(
            f"Settlement callback: {batch.batch_id} - "
            f"{len(batch.creators)} creators, {batch.total_amount:.6f} FTNS"
        )
    
    async def close(self) -> None:
        """Stop the escrow and flush pending royalties."""
        # Force settle any remaining
        if self.escrow._pending:
            await self.escrow.settle_batch(force=True)
        
        self.escrow.stop()
