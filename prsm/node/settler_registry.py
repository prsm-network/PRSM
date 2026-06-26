"""
Settler Registry — Phase 6 Governance & Staking
================================================

Simple L2-style staking for batch settlement security.

Architecture:
  1. THE BOND: Settlers must stake FTNS to earn the right to settle batches
  2. THE CHALLENGE: Export local ledger for public audit
  3. THE MULTI-SIG: Require 3-of-N settler signatures before settlement

This is a pragmatic alternative to complex fraud-proof systems.
The local ledger remains authoritative; on-chain settlement is a mirror.

Security Model:
  - Bond ensures settlers have "skin in the game"
  - Multi-sig prevents single-point-of-failure
  - Public ledger export enables community audit
  - Slashing via governance vote for proven fraud
"""

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


def verify_settler_signature(public_key_b64: str, message: bytes, signature_b64: str) -> bool:
    """Verify a settler's Ed25519 signature."""
    if not signature_b64:
        return False
    try:
        from prsm.node.identity import verify_signature
        return verify_signature(public_key_b64, message, signature_b64)
    except Exception:
        return False


def _allow_unsigned_settler_batch() -> bool:
    """sp1277 — DEV-ONLY opt-in to accept a batch signature from a settler that registered no
    public key (the pre-fix forgeable behavior). Default OFF → production fails closed: a
    settler MUST register a key and produce a real signature to approve a batch."""
    return os.getenv("PRSM_ALLOW_UNSIGNED_SETTLER_BATCH", "").strip().lower() in (
        "1", "true", "yes", "on")


# === Configuration ===

DEFAULT_MIN_SETTLER_BOND = 10_000.0       # 10K FTNS to become a settler
DEFAULT_SETTLEMENT_THRESHOLD = 3          # 3 signatures required
DEFAULT_SETTLER_MAX = 10                  # Max 10 active settlers
DEFAULT_BOND_LOCK_PERIOD_DAYS = 30        # 30 days to unbond
DEFAULT_CHALLENGE_PERIOD_SECONDS = 86400  # 24 hours to challenge a batch


class SettlerStatus(str, Enum):
    """Status of a registered settler."""
    ACTIVE = "active"                 # Can sign and settle
    UNBONDING = "unbonding"           # Waiting out lock period
    SLASHED = "slashed"               # Slashed for misconduct
    INACTIVE = "inactive"             # Unbonded or removed


@dataclass
class Settler:
    """A registered batch settler with staked bond."""
    settler_id: str
    address: str                      # 0x Ethereum address
    bond_amount: float
    staked_at: datetime
    # sp1277 — Ed25519 public key (base64) the settler signs batch approvals with. Without it,
    # sign_batch can't verify a signature and fails closed (the multi-sig was forgeable while
    # this field didn't exist and the verify branch was dead code).
    public_key_b64: Optional[str] = None
    status: SettlerStatus = SettlerStatus.ACTIVE
    unbonding_at: Optional[datetime] = None
    total_settled: int = 0
    total_volume: float = 0.0
    slashed_amount: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def can_settle(self) -> bool:
        return self.status == SettlerStatus.ACTIVE and self.bond_amount > 0


@dataclass
class BatchSignature:
    """A settler's signature on a batch."""
    settler_id: str
    batch_hash: str
    signature: str                    # Simulated signature (real: ECDSA)
    signed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "settler_id": self.settler_id,
            "batch_hash": self.batch_hash,
            "signature": self.signature[:16] + "..." if len(self.signature) > 16 else self.signature,
            "signed_at": self.signed_at.isoformat(),
        }


@dataclass
class PendingBatch:
    """A batch awaiting multi-sig approval."""
    batch_id: str
    batch_hash: str                   # Hash of all pending transfers
    transfers: List[Dict[str, Any]]
    total_amount: float
    created_at: datetime
    signatures: List[BatchSignature] = field(default_factory=list)
    settled: bool = False
    settlement_tx: Optional[str] = None
    challenged: bool = False
    challenge_reason: Optional[str] = None
    
    @property
    def signature_count(self) -> int:
        return len(self.signatures)
    
    @property
    def signer_ids(self) -> Set[str]:
        return {s.settler_id for s in self.signatures}


@dataclass
class SlashProposal:
    """Governance proposal to slash a settler."""
    proposal_id: str
    settler_id: str
    slash_amount: float
    reason: str
    evidence: Dict[str, Any]
    created_at: datetime
    votes_for: int = 0
    votes_against: int = 0
    total_voting_power: float = 0.0
    approved: bool = False
    executed: bool = False


class SettlerRegistry:
    """
    Manages registered batch settlers with staked bonds.
    
    Simple L2-style security:
    - Settlers stake FTNS to earn settlement rights
    - Multi-sig requirement (3-of-N) for batch approval
    - Public ledger export for transparency
    - Governance-based slashing for fraud
    """
    
    def __init__(
        self,
        min_settler_bond: float = DEFAULT_MIN_SETTLER_BOND,
        settlement_threshold: int = DEFAULT_SETTLEMENT_THRESHOLD,
        max_settlers: int = DEFAULT_SETTLER_MAX,
        bond_lock_period_days: int = DEFAULT_BOND_LOCK_PERIOD_DAYS,
        challenge_period_seconds: int = DEFAULT_CHALLENGE_PERIOD_SECONDS,
        ftns_service=None,                     # FTNS service for bond locking
        staking_manager=None,                  # Optional: use existing StakingManager
    ):
        self.min_settler_bond = min_settler_bond
        self.settlement_threshold = settlement_threshold
        self.max_settlers = max_settlers
        self.bond_lock_period = timedelta(days=bond_lock_period_days)
        self.challenge_period = timedelta(seconds=challenge_period_seconds)
        
        self.ftns_service = ftns_service
        self.staking_manager = staking_manager
        
        # Registry state
        self._settlers: Dict[str, Settler] = {}           # settler_id -> Settler
        self._address_to_settler: Dict[str, str] = {}     # address -> settler_id
        self._pending_batches: Dict[str, PendingBatch] = {}  # batch_id -> PendingBatch
        self._slash_proposals: Dict[str, SlashProposal] = {}
        
        # Ledger export (for challenge/audit)
        self._ledger_export: Dict[str, Any] = {}
        self._last_export_at: Optional[datetime] = None
        
        # Callbacks
        self._on_settlement_ready: Optional[Callable] = None
        
        self._lock = asyncio.Lock()
        
        logger.info(
            "SettlerRegistry initialized",
            min_bond=min_settler_bond,
            threshold=settlement_threshold,
            max_settlers=max_settlers,
        )
    
    # ── Settler Registration ─────────────────────────────────────
    
    async def register_settler(
        self,
        settler_id: str,
        address: str,
        bond_amount: float,
        public_key_b64: Optional[str] = None,
    ) -> Settler:
        """
        Register a new settler with staked bond.
        
        Args:
            settler_id: Unique identifier for the settler
            address: Ethereum address (0x...) for on-chain operations
            bond_amount: Amount of FTNS to stake as bond
            
        Returns:
            Settler: The registered settler
            
        Raises:
            ValueError: If registration fails validation
        """
        async with self._lock:
            # Validate bond amount
            if bond_amount < self.min_settler_bond:
                raise ValueError(
                    f"Bond {bond_amount} below minimum {self.min_settler_bond} FTNS"
                )
            
            # Check max settlers
            active_count = sum(1 for s in self._settlers.values() if s.can_settle)
            if active_count >= self.max_settlers:
                raise ValueError(
                    f"Maximum settlers ({self.max_settlers}) reached"
                )
            
            # Check for duplicate address
            if address.lower() in self._address_to_settler:
                raise ValueError(f"Address {address[:12]}... already registered")
            
            # Lock the bond (via FTNS service or staking manager).
            # Sprint 492 (F32 fix) — check the return value AND
            # let exceptions propagate. Pre-fix, lock_tokens
            # silently returned False on insufficient balance
            # and the registry created the settler record anyway
            # → adversary could claim a HUGE bond without
            # actually staking the FTNS (anti-Sybil broken).
            if self.ftns_service:
                locked = await self.ftns_service.lock_tokens(
                    settler_id,
                    Decimal(str(bond_amount)),
                    reason="settler_bond"
                )
                if not locked:
                    raise ValueError(
                        f"FTNS lock failed for settler {settler_id} "
                        f"bond {bond_amount} — likely insufficient "
                        f"available balance. Registration aborted."
                    )
            elif self.staking_manager:
                from prsm.economy.tokenomics.staking_manager import StakeType
                await self.staking_manager.stake(
                    user_id=settler_id,
                    amount=Decimal(str(bond_amount)),
                    stake_type=StakeType.VALIDATION,
                    metadata={"role": "settler", "address": address},
                )
            
            # Create settler record
            settler = Settler(
                settler_id=settler_id,
                address=address.lower(),
                bond_amount=bond_amount,
                staked_at=datetime.now(timezone.utc),
                public_key_b64=public_key_b64,
                status=SettlerStatus.ACTIVE,
            )
            
            self._settlers[settler_id] = settler
            self._address_to_settler[address.lower()] = settler_id
            
            logger.info(
                "Settler registered",
                settler_id=settler_id,
                address=address[:12],
                bond=bond_amount,
            )
            
            return settler
    
    async def unbond_settler(self, settler_id: str) -> Optional[datetime]:
        """
        Initiate unbonding for a settler.
        
        The settler cannot participate in settlements during unbonding period.
        After the lock period, the settler can withdraw their bond.
        
        Returns:
            datetime: When the bond can be withdrawn
        """
        async with self._lock:
            settler = self._settlers.get(settler_id)
            if not settler:
                raise ValueError(f"Settler {settler_id} not found")
            
            if settler.status != SettlerStatus.ACTIVE:
                raise ValueError(f"Settler is not active: {settler.status}")
            
            # Mark as unbonding
            settler.status = SettlerStatus.UNBONDING
            settler.unbonding_at = datetime.now(timezone.utc) + self.bond_lock_period
            
            # Remove from address index
            self._address_to_settler.pop(settler.address, None)
            
            logger.info(
                "Settler unbonding initiated",
                settler_id=settler_id,
                unbond_at=settler.unbonding_at.isoformat(),
            )
            
            return settler.unbonding_at
    
    async def withdraw_bond(self, settler_id: str) -> float:
        """
        Withdraw bond after unbonding period.
        
        Returns:
            float: Amount withdrawn
        """
        async with self._lock:
            settler = self._settlers.get(settler_id)
            if not settler:
                raise ValueError(f"Settler {settler_id} not found")
            
            if settler.status != SettlerStatus.UNBONDING:
                raise ValueError(f"Settler is not unbonding: {settler.status}")
            
            if datetime.now(timezone.utc) < settler.unbonding_at:
                raise ValueError(
                    f"Unbonding period not complete. Wait until {settler.unbonding_at}"
                )
            
            # Release the bond
            bond = settler.bond_amount
            settler.bond_amount = 0
            settler.status = SettlerStatus.INACTIVE
            
            if self.ftns_service:
                await self.ftns_service.unlock_tokens(
                    settler_id,
                    Decimal(str(bond)),
                    reason="settler_unbond"
                )
            elif self.staking_manager:
                # Would need to implement partial unstake
                pass
            
            logger.info(
                "Settler bond withdrawn",
                settler_id=settler_id,
                amount=bond,
            )
            
            return bond
    
    def get_settler(self, settler_id: str) -> Optional[Settler]:
        """Get a settler by ID."""
        return self._settlers.get(settler_id)
    
    def get_settler_by_address(self, address: str) -> Optional[Settler]:
        """Get a settler by Ethereum address."""
        settler_id = self._address_to_settler.get(address.lower())
        if settler_id:
            return self._settlers.get(settler_id)
        return None
    
    def list_active_settlers(self) -> List[Settler]:
        """List all active settlers."""
        return [s for s in self._settlers.values() if s.can_settle]
    
    # ── Multi-Sig Batch Approval ─────────────────────────────────
    
    async def propose_batch(
        self,
        transfers: List[Dict[str, Any]],
    ) -> PendingBatch:
        """
        Create a new batch pending multi-sig approval.
        
        Args:
            transfers: List of transfer dicts with 'to', 'amount', 'tx_id'
            
        Returns:
            PendingBatch: The created batch awaiting signatures
        """
        # Compute batch hash
        batch_data = json.dumps(transfers, sort_keys=True)
        batch_hash = hashlib.sha256(batch_data.encode()).hexdigest()[:32]
        
        batch_id = str(uuid4())[:12]
        total_amount = sum(t.get("amount", 0) for t in transfers)
        
        batch = PendingBatch(
            batch_id=batch_id,
            batch_hash=batch_hash,
            transfers=transfers,
            total_amount=total_amount,
            created_at=datetime.now(timezone.utc),
        )
        
        async with self._lock:
            self._pending_batches[batch_id] = batch
        
        logger.info(
            "Batch proposed",
            batch_id=batch_id,
            batch_hash=batch_hash,
            transfer_count=len(transfers),
            total_amount=total_amount,
        )
        
        return batch
    
    async def sign_batch(
        self,
        batch_id: str,
        settler_id: str,
        signature: str,
    ) -> BatchSignature:
        """
        Add a settler's signature to a batch.
        
        Args:
            batch_id: ID of the batch to sign
            settler_id: ID of the signing settler
            signature: Cryptographic signature (simulated)
            
        Returns:
            BatchSignature: The created signature
            
        Raises:
            ValueError: If batch or settler invalid, or already signed
        """
        async with self._lock:
            batch = self._pending_batches.get(batch_id)
            if not batch:
                raise ValueError(f"Batch {batch_id} not found")
            
            if batch.settled:
                raise ValueError(f"Batch {batch_id} already settled")
            
            settler = self._settlers.get(settler_id)
            if not settler or not settler.can_settle:
                raise ValueError(f"Settler {settler_id} not authorized")
            
            # Check for duplicate signature
            if settler_id in batch.signer_ids:
                raise ValueError(f"Settler {settler_id} already signed this batch")
            
            # sp1277 — MANDATORY signature verification. Previously this only verified when
            # `public_key_b64` was set, but that field didn't exist on Settler → the branch was
            # dead and every signature was accepted, making the 3-of-N multi-sig forgeable by
            # any caller. Now: with a key, verify (fail closed on a bad sig); WITHOUT a key,
            # REJECT — a settler that registered no key cannot approve a batch — unless the
            # DEV-ONLY PRSM_ALLOW_UNSIGNED_SETTLER_BATCH opt-in is set (legacy/local testing).
            settler = self._settlers.get(settler_id)
            pubkey = getattr(settler, "public_key_b64", None) if settler else None
            if pubkey:
                expected_msg = f"PRSM:{batch.batch_hash}:{settler_id}".encode()
                if not verify_settler_signature(pubkey, expected_msg, signature):
                    raise ValueError(f"Invalid signature from settler {settler_id}")
            elif not _allow_unsigned_settler_batch():
                raise ValueError(
                    f"Settler {settler_id} has no registered public key; its batch signature "
                    f"cannot be verified (set PRSM_ALLOW_UNSIGNED_SETTLER_BATCH for DEV-ONLY "
                    f"legacy behavior)"
                )
            
            batch_sig = BatchSignature(
                settler_id=settler_id,
                batch_hash=batch.batch_hash,
                signature=signature,
            )
            
            batch.signatures.append(batch_sig)
            
            logger.info(
                "Batch signed",
                batch_id=batch_id,
                settler_id=settler_id,
                signature_count=batch.signature_count,
                threshold=self.settlement_threshold,
            )
            
            # Check if threshold reached
            if batch.signature_count >= self.settlement_threshold:
                logger.info(
                    "Batch approved - threshold reached",
                    batch_id=batch_id,
                    signatures=batch.signature_count,
                )
                if self._on_settlement_ready:
                    await self._on_settlement_ready(batch)
            
            return batch_sig
    
    def is_batch_approved(self, batch_id: str) -> bool:
        """Check if a batch has enough signatures."""
        batch = self._pending_batches.get(batch_id)
        if not batch:
            return False
        return batch.signature_count >= self.settlement_threshold
    
    def get_pending_batch(self, batch_id: str) -> Optional[PendingBatch]:
        """Get a pending batch by ID."""
        return self._pending_batches.get(batch_id)
    
    def list_pending_batches(self) -> List[PendingBatch]:
        """List all pending batches."""
        return list(self._pending_batches.values())
    
    # ── Ledger Export (Challenge System) ──────────────────────────
    
    async def export_ledger(
        self,
        ledger_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Export local ledger state for public audit.
        
        This enables the "Challenge" mechanism where anyone can
        compare the local ledger against on-chain settlement.
        
        Args:
            ledger_data: Full ledger state (balances, transactions)
            
        Returns:
            Dict with export info and hash
        """
        export = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "settlers": [
                {
                    "id": s.settler_id,
                    "address": s.address,
                    "bond": s.bond_amount,
                    "status": s.status.value,
                    "settled_count": s.total_settled,
                }
                for s in self.list_active_settlers()
            ],
            "pending_batches": [
                {
                    "batch_id": b.batch_id,
                    "batch_hash": b.batch_hash,
                    "signature_count": b.signature_count,
                    "total_amount": b.total_amount,
                }
                for b in self._pending_batches.values()
                if not b.settled
            ],
            **ledger_data,
        }
        
        # Compute integrity hash
        export_json = json.dumps(export, sort_keys=True, default=str)
        export["integrity_hash"] = hashlib.sha256(export_json.encode()).hexdigest()[:32]
        
        async with self._lock:
            self._ledger_export = export
            self._last_export_at = datetime.now(timezone.utc)
        
        logger.info(
            "Ledger exported",
            integrity_hash=export["integrity_hash"],
            batch_count=len(export["pending_batches"]),
        )
        
        return export
    
    def get_ledger_export(self) -> Optional[Dict[str, Any]]:
        """Get the latest ledger export."""
        return self._ledger_export if self._ledger_export else None
    
    # ── Slashing (Governance) ─────────────────────────────────────
    
    async def propose_slash(
        self,
        settler_id: str,
        slash_amount: float,
        reason: str,
        evidence: Dict[str, Any],
        proposer_id: str,
    ) -> SlashProposal:
        """
        Create a governance proposal to slash a settler.
        
        In production, this would integrate with the governance voting system.
        For now, it creates a record that can be voted on.
        
        Args:
            settler_id: ID of settler to slash
            slash_amount: Amount of bond to slash
            reason: Human-readable reason
            evidence: Evidence dict (e.g., fraudulent batch details)
            proposer_id: ID of the proposer
            
        Returns:
            SlashProposal: The created proposal
        """
        async with self._lock:
            settler = self._settlers.get(settler_id)
            if not settler:
                raise ValueError(f"Settler {settler_id} not found")
            
            if slash_amount > settler.bond_amount:
                slash_amount = settler.bond_amount  # Cap at available bond
            
            proposal = SlashProposal(
                proposal_id=str(uuid4())[:12],
                settler_id=settler_id,
                slash_amount=slash_amount,
                reason=reason,
                evidence=evidence,
                created_at=datetime.now(timezone.utc),
            )
            
            self._slash_proposals[proposal.proposal_id] = proposal
            
            logger.warning(
                "Slash proposal created",
                proposal_id=proposal.proposal_id,
                settler_id=settler_id,
                amount=slash_amount,
                reason=reason,
            )
            
            return proposal
    
    async def execute_slash(
        self,
        proposal_id: str,
    ) -> float:
        """
        Execute an approved slash proposal.
        
        In production, this would be called after governance voting confirms.
        
        Returns:
            float: Amount slashed
        """
        async with self._lock:
            proposal = self._slash_proposals.get(proposal_id)
            if not proposal:
                raise ValueError(f"Proposal {proposal_id} not found")
            
            if proposal.executed:
                raise ValueError(f"Proposal {proposal_id} already executed")
            
            settler = self._settlers.get(proposal.settler_id)
            if not settler:
                raise ValueError(f"Settler {proposal.settler_id} not found")
            
            # Execute slash
            slash_amount = min(proposal.slash_amount, settler.bond_amount)
            settler.bond_amount -= slash_amount
            settler.slashed_amount += slash_amount
            
            if settler.bond_amount <= 0:
                settler.status = SettlerStatus.SLASHED
                self._address_to_settler.pop(settler.address, None)
            
            proposal.executed = True
            proposal.approved = True
            
            logger.warning(
                "Slash executed",
                proposal_id=proposal_id,
                settler_id=settler.settler_id,
                amount=slash_amount,
                remaining_bond=settler.bond_amount,
            )
            
            return slash_amount
    
    # ── Callbacks ────────────────────────────────────────────────
    
    def on_settlement_ready(self, callback: Callable):
        """Register callback for when a batch reaches approval threshold."""
        self._on_settlement_ready = callback
    
    # ── Stats ─────────────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        active_settlers = self.list_active_settlers()
        total_bond = sum(s.bond_amount for s in active_settlers)
        
        return {
            "active_settlers": len(active_settlers),
            "max_settlers": self.max_settlers,
            "settlement_threshold": self.settlement_threshold,
            "min_settler_bond": self.min_settler_bond,
            "total_bonded": total_bond,
            "pending_batches": len([b for b in self._pending_batches.values() if not b.settled]),
            "slash_proposals": len(self._slash_proposals),
            "last_ledger_export": self._last_export_at.isoformat() if self._last_export_at else None,
        }
