"""
FTNS Staking Manager
Implements staking, unstaking, slashing, and reward distribution

This module provides a comprehensive staking system for FTNS tokens, enabling:
- Token staking with configurable minimums and maximums
- Unstaking with time-locked withdrawal periods
- Slashing for misbehavior penalties
- Staking rewards with configurable rates
- Governance integration for parameter changes

Core Features:
- Secure staking with balance locking
- Time-delayed unstaking for security
- Attributable slashing with evidence tracking
- Automatic reward calculation and distribution
- Comprehensive audit trails

Security Considerations:
- All operations are atomic with proper rollback
- Slashing requires governance approval or automated triggers
- Unstaking has mandatory delay periods
- Rate limits prevent rapid stake manipulation
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from prsm.core.database import get_async_session, StakeModel, UnstakeRequestModel, SlashEventModel

logger = structlog.get_logger(__name__)


# === Enums ===

class StakeStatus(str, Enum):
    """Status of a stake"""
    ACTIVE = "active"              # Currently staked and earning rewards
    UNSTAKING = "unstaking"         # Unstake requested, waiting for period
    WITHDRAWN = "withdrawn"         # Fully withdrawn
    SLASHED = "slashed"             # Partially or fully slashed
    LOCKED = "locked"               # Locked due to dispute or investigation


class UnstakeRequestStatus(str, Enum):
    """Status of an unstake request"""
    PENDING = "pending"            # Waiting for unstaking period
    AVAILABLE = "available"         # Ready for withdrawal
    COMPLETED = "completed"         # Withdrawn successfully
    CANCELLED = "cancelled"         # Request cancelled
    SLASHED = "slashed"             # Slashed during unstaking


class SlashReason(str, Enum):
    """Reasons for slashing"""
    MISCONDUCT = "misconduct"                    # General misbehavior
    VALIDATION_FAILURE = "validation_failure"    # Failed to validate properly
    DOUBLE_SIGNING = "double_signing"           # Signed conflicting blocks
    DOWNTIME = "downtime"                        # Excessive offline time
    GOVERNANCE_VIOLATION = "governance_violation"  # Violated governance rules
    FRAUD = "fraud"                              # Fraudulent activity
    COLLUSION = "collusion"                       # Collusion with other nodes
    DATA_MANIPULATION = "data_manipulation"      # Manipulated data
    SECURITY_BREACH = "security_breach"          # Security violation
    APPEAL_REJECTED = "appeal_rejected"           # Appeal was rejected


class StakeType(str, Enum):
    """Types of staking"""
    GOVERNANCE = "governance"        # Staking for governance participation
    VALIDATION = "validation"        # Staking for network validation
    COMPUTE = "compute"              # Staking for compute provision
    STORAGE = "storage"              # Staking for storage provision
    LIQUIDITY = "liquidity"          # Staking for liquidity provision
    GENERAL = "general"              # General purpose staking


# === Staking utility benefits (sp906) ===
# PRSM_Tokenomics.md §5.3: staking confers UTILITY benefits — lock-based
# service discounts + priority access — and NO token yield (sp904). The
# discount applies to the network-fee (treasury) portion of a payment
# only (the foundation's take), never the operator/creator share, so it
# can't shortchange providers. Priority biases dispatch ordering toward
# faster / higher-quality providers.
#
# Schedule, high→low (first match wins):
#   (min_lock_days, discount_fraction, priority_boost, tier_label)
_DEFAULT_BENEFIT_TIERS: Tuple[Tuple[int, float, float, str], ...] = (
    (365, 0.10, 0.50, "365d"),
    (90, 0.05, 0.25, "90d"),
    (30, 0.02, 0.10, "30d"),
)

# Lock commitments a staker may choose (days). 0/None = no lock, no benefit.
ALLOWED_LOCK_DAYS: Tuple[int, ...] = (30, 90, 365)


@dataclass(frozen=True)
class StakingBenefits:
    """The utility benefits a staker's active lock confers."""
    lock_period_days: int
    tier_label: str
    discount_fraction: float   # network-fee discount, in [0, 1)
    priority_boost: float      # dispatch-priority multiplier addend (>= 0)

    @classmethod
    def none(cls) -> "StakingBenefits":
        """No qualifying lock — inert benefits."""
        return cls(0, "none", 0.0, 0.0)

    @property
    def is_active(self) -> bool:
        return self.discount_fraction > 0 or self.priority_boost > 0

    def discounted_fee(self, fee: Decimal) -> Decimal:
        """Apply the network-fee discount to a treasury-fee amount."""
        if self.discount_fraction <= 0:
            return fee
        return fee * (Decimal("1") - Decimal(str(self.discount_fraction)))

    def boosted_priority(self, base: float) -> float:
        """Scale a base dispatch-priority score by the priority boost."""
        return base * (1.0 + self.priority_boost)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lock_period_days": self.lock_period_days,
            "tier_label": self.tier_label,
            "discount_fraction": self.discount_fraction,
            "priority_boost": self.priority_boost,
            "is_active": self.is_active,
        }


# === Dataclasses ===

@dataclass
class StakingConfig:
    """Configuration for staking mechanism"""
    minimum_stake: int = 1000                    # Minimum FTNS to stake
    unstaking_period_seconds: int = 7 * 24 * 3600  # 7 days in seconds
    # Sprint 904 — staking yields ELIMINATED (PRSM_Tokenomics.md §10 #7
    # + the utility-token posture). A mint-funded staking return is a
    # Howey "expectation of profit from the efforts of others" flag, and
    # the deployed v1 emission split has no staker-yield pool to fund it
    # from. Staking confers utility benefits (lock-based service
    # discounts / priority access — see §5.3) but NO token yield. Default
    # is 0.0; calculate_rewards() returns no accruals regardless.
    reward_rate_annual: float = 0.0               # NO staking yield (sp904)
    slashing_rate_base: float = 0.1               # 10% base slash rate
    max_stake_per_user: int = 10_000_000          # Maximum stake per user
    max_total_stake: int = 1_000_000_000           # Maximum total network stake
    reward_compounding: bool = False              # Whether rewards compound
    min_reward_claim_interval_seconds: int = 24 * 3600  # 1 day minimum between claims
    governance_slash_multiplier: float = 2.0       # Multiplier for governance-approved slashes
    appeal_period_seconds: int = 3 * 24 * 3600    # 3 days to appeal slashes
    
    # Rate limits
    max_stake_operations_per_day: int = 5         # Max stake operations per day
    max_unstake_operations_per_day: int = 3       # Max unstake operations per day
    
    # Reward parameters
    reward_calculation_interval_seconds: int = 3600  # Calculate rewards hourly
    min_stake_age_for_rewards_seconds: int = 24 * 3600  # 1 day minimum stake age
    
    # Slashing thresholds
    downtime_slash_threshold_seconds: int = 24 * 3600  # 24 hours offline triggers slash
    max_slash_per_event: float = 0.5            # Max 50% slash per event

    # Staking utility-benefit schedule (sp906) — lock_days → (discount, priority)
    benefit_tiers: Tuple[Tuple[int, float, float, str], ...] = _DEFAULT_BENEFIT_TIERS

    def benefits_for_lock_days(
        self, lock_period_days: Optional[int]
    ) -> StakingBenefits:
        """Map a lock period (days) to its utility benefits.

        Returns the highest tier whose ``min_lock_days`` threshold the
        lock meets; ``StakingBenefits.none()`` if the lock is None/0 or
        below the lowest tier. Tiers are stored high→low so the first
        match wins."""
        if not lock_period_days or lock_period_days <= 0:
            return StakingBenefits.none()
        for min_days, discount, priority, label in self.benefit_tiers:
            if lock_period_days >= min_days:
                return StakingBenefits(
                    lock_period_days=lock_period_days,
                    tier_label=label,
                    discount_fraction=discount,
                    priority_boost=priority,
                )
        return StakingBenefits.none()


@dataclass
class StakeRecord:
    """Record of a stake"""
    stake_id: str
    user_id: str
    amount: Decimal
    stake_type: StakeType
    staked_at: datetime
    rewards_earned: Decimal = Decimal('0')
    rewards_claimed: Decimal = Decimal('0')
    last_reward_calculation: datetime = None
    status: StakeStatus = StakeStatus.ACTIVE
    lock_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.last_reward_calculation is None:
            self.last_reward_calculation = self.staked_at
        if isinstance(self.stake_type, str):
            self.stake_type = StakeType(self.stake_type)
        if isinstance(self.status, str):
            self.status = StakeStatus(self.status)


@dataclass
class UnstakeRequest:
    """Request to unstake tokens"""
    request_id: str
    stake_id: str
    user_id: str
    amount: Decimal
    requested_at: datetime
    available_at: datetime
    status: UnstakeRequestStatus = UnstakeRequestStatus.PENDING
    completed_at: Optional[datetime] = None
    cancellation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = UnstakeRequestStatus(self.status)
    
    @property
    def is_available(self) -> bool:
        """Check if unstake is available for withdrawal"""
        return (
            self.status == UnstakeRequestStatus.AVAILABLE and
            datetime.now(timezone.utc) >= self.available_at
        )


@dataclass
class SlashRecord:
    """Record of a slashing event"""
    slash_id: str
    stake_id: str
    user_id: str
    amount_slashed: Decimal
    reason: SlashReason
    slash_rate: float
    evidence: Dict[str, Any]
    slashed_at: datetime
    slashed_by: str  # Governance proposal ID or automated system
    appeal_deadline: Optional[datetime] = None
    appeal_status: Optional[str] = None  # "pending", "approved", "rejected"
    appeal_evidence: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.reason, str):
            self.reason = SlashReason(self.reason)
        if self.appeal_deadline is None:
            self.appeal_deadline = self.slashed_at + timedelta(days=3)


@dataclass
class RewardCalculation:
    """Result of reward calculation"""
    stake_id: str
    user_id: str
    principal: Decimal
    reward_amount: Decimal
    annual_rate: float
    days_staked: float
    calculation_timestamp: datetime
    next_calculation: datetime



class StakingManager:
    """
    Manages FTNS staking operations
    
    The StakingManager handles all staking-related operations including:
    - Creating and managing stakes
    - Processing unstake requests with time delays
    - Applying slashing penalties
    - Calculating and distributing rewards
    - Managing stake status transitions
    
    All operations are designed to be atomic and maintain consistency
    with the underlying FTNS balance system.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        ftns_service,
        config: Optional[StakingConfig] = None
    ):
        """
        Initialize the staking manager
        
        Args:
            db_session: Database session for persistence (deprecated - not used, sessions are created per-operation)
            ftns_service: FTNS service for balance operations
            config: Staking configuration (uses defaults if not provided)
        """
        self.db = db_session
        self.ftns = ftns_service
        self.config = config or StakingConfig()
        
        # Rate limiting (stays in-memory for performance)
        self._daily_operations: Dict[str, List[datetime]] = {}  # user_id -> [timestamps]

        # sp933 — per-entity locks. The balance-mutating operations
        # (withdraw/cancel/unstake/slash/resolve_appeal) each open their OWN
        # session, read a row's status, then commit + apply a money side-effect
        # (unlock/burn/mint) — with nothing held between. Two concurrent or
        # replayed calls on the same entity both pass the stale-read check and
        # both apply the side-effect (double-withdraw / double-burn / double-
        # mint). Serializing per entity closes that (same class as sp929).
        self._entity_locks: Dict[str, asyncio.Lock] = {}
        
        logger.info(
            "StakingManager initialized",
            minimum_stake=self.config.minimum_stake,
            unstaking_period_days=self.config.unstaking_period_seconds / 86400,
            reward_rate=self.config.reward_rate_annual
        )
    
    def _lock_for(self, key: str) -> asyncio.Lock:
        """sp933 — return the process-local lock for an entity key, creating it
        on first use. Sync + await-free, so the get-or-create is atomic under
        asyncio (no interleaving). Operations sharing a key serialize: withdraw
        and cancel share the request key; unstake and slash share the stake key.
        """
        lock = self._entity_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._entity_locks[key] = lock
        return lock

    # === Conversion Helper Methods ===

    @staticmethod
    def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
        """Sprint 432 (F11 fix): normalize naive datetimes from the
        DB to tz-aware UTC.

        SQLite (the default backend) drops tz info when persisting
        datetimes, so a value stored as ``datetime.now(timezone.utc)``
        comes back as a naive datetime on load. The reward
        calculation does ``datetime.now(timezone.utc) - last_calc``,
        which raises ``TypeError: can't subtract offset-naive and
        offset-aware datetimes`` — silently breaking every call to
        ``claim_rewards`` until the first claim returns 0 rewards
        AND throws a 500. Production-blocking bug surfaced during
        sprint 432 stake → claim verification.

        Stored values are UTC by construction (all writers use
        ``datetime.now(timezone.utc)``), so re-tagging the loaded
        value as UTC is sound — no clock arithmetic. Postgres
        TIMESTAMPTZ would round-trip the tz cleanly; this helper
        is the SQLite-compatibility shim.
        """
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def _stake_row_to_record(self, row: StakeModel) -> StakeRecord:
        """Convert StakeModel database row to StakeRecord dataclass."""
        return StakeRecord(
            stake_id=str(row.stake_id),
            user_id=row.user_id,
            amount=Decimal(str(row.amount)),
            stake_type=StakeType(row.stake_type),
            staked_at=self._ensure_utc(row.staked_at),
            rewards_earned=Decimal(str(row.rewards_earned)),
            rewards_claimed=Decimal(str(row.rewards_claimed)),
            last_reward_calculation=self._ensure_utc(
                row.last_reward_calculation,
            ),
            status=StakeStatus(row.status),
            lock_reason=row.lock_reason,
            metadata=row.stake_metadata or {},
        )
    
    def _unstake_row_to_record(self, row: UnstakeRequestModel) -> UnstakeRequest:
        """Convert UnstakeRequestModel database row to UnstakeRequest dataclass."""
        return UnstakeRequest(
            request_id=str(row.request_id),
            stake_id=str(row.stake_id),
            user_id=row.user_id,
            amount=Decimal(str(row.amount)),
            requested_at=row.requested_at,
            available_at=row.available_at,
            status=UnstakeRequestStatus(row.status),
            completed_at=row.completed_at,
            cancellation_reason=row.cancellation_reason,
            metadata=row.request_metadata or {},
        )
    
    def _slash_row_to_record(self, row: SlashEventModel) -> SlashRecord:
        """Convert SlashEventModel database row to SlashRecord dataclass."""
        return SlashRecord(
            slash_id=str(row.slash_id),
            stake_id=str(row.stake_id),
            user_id=row.user_id,
            amount_slashed=Decimal(str(row.amount_slashed)),
            reason=SlashReason(row.reason),
            slash_rate=Decimal(str(row.slash_rate)),
            slashed_by=row.slashed_by,
            evidence=row.evidence or {},
            slashed_at=row.slashed_at,
            appeal_deadline=row.appeal_deadline,
            appeal_status=row.appeal_status,
            appeal_evidence=row.appeal_evidence,
            metadata=row.slash_metadata or {},
        )
    
    # === Staking Operations ===
    
    async def stake(
        self,
        user_id: str,
        amount: Decimal,
        stake_type: StakeType = StakeType.GENERAL,
        metadata: Optional[Dict[str, Any]] = None,
        lock_period_days: Optional[int] = None,
    ) -> StakeRecord:
        """
        Stake FTNS tokens

        Args:
            user_id: User making the stake
            amount: Amount to stake
            stake_type: Type of staking (governance, validation, etc.)
            metadata: Additional metadata for the stake
            lock_period_days: sp906 — optional utility-benefit lock
                commitment (one of ALLOWED_LOCK_DAYS = 30/90/365). While
                the lock is in effect the stake cannot be unstaked and the
                staker earns the tier's service discount + priority access
                (§5.3). None/0 = no lock, no benefit, 7-day unstake as
                before.

        Returns:
            StakeRecord: The created stake record

        Raises:
            ValueError: If stake validation fails
        """

        # Validate amount
        if amount < Decimal(self.config.minimum_stake):
            raise ValueError(
                f"Stake amount {amount} below minimum {self.config.minimum_stake}"
            )

        # Validate lock commitment (sp906).
        if lock_period_days is not None and lock_period_days != 0:
            if lock_period_days not in ALLOWED_LOCK_DAYS:
                raise ValueError(
                    f"lock_period_days must be one of {ALLOWED_LOCK_DAYS} "
                    f"(or None); got {lock_period_days}"
                )
        
        # Check user's current total stake
        user_total_stake = await self.get_user_total_stake(user_id)
        if user_total_stake + amount > Decimal(self.config.max_stake_per_user):
            raise ValueError(
                f"Stake would exceed user maximum of {self.config.max_stake_per_user}"
            )
        
        # Check network total stake
        network_total = await self.get_network_total_stake()
        if network_total + amount > Decimal(self.config.max_total_stake):
            raise ValueError(
                f"Stake would exceed network maximum of {self.config.max_total_stake}"
            )
        
        # Check rate limit
        await self._check_rate_limit(user_id, "stake")
        
        # Verify user has sufficient balance
        available_balance = await self.ftns.get_available_balance(user_id)
        if available_balance < amount:
            raise ValueError(
                f"Insufficient balance. Available: {available_balance}, Required: {amount}"
            )
        
        # Lock tokens from user's balance
        await self.ftns.lock_tokens(user_id, amount, reason="staking")
        
        # Create stake record
        stake_id = uuid4()
        now = datetime.now(timezone.utc)

        # sp906 — record the lock commitment + expiry in metadata so
        # get_user_benefits() can compute the tier and unstake() can
        # enforce the lock. No DB-schema change (uses stake_metadata JSON).
        stake_metadata = dict(metadata or {})
        if lock_period_days:
            stake_metadata["lock_period_days"] = int(lock_period_days)
            stake_metadata["lock_expires_at"] = (
                now + timedelta(days=int(lock_period_days))
            ).isoformat()

        async with get_async_session() as db:
            model = StakeModel(
                stake_id=stake_id,
                user_id=user_id,
                amount=float(amount),
                stake_type=stake_type.value,
                status=StakeStatus.ACTIVE.value,
                rewards_earned=0.0,
                rewards_claimed=0.0,
                last_reward_calculation=now,
                staked_at=now,
                stake_metadata=stake_metadata,
            )
            db.add(model)
            await db.commit()
            await db.refresh(model)
            
            # Record operation for rate limiting
            self._record_operation(user_id, "stake")
            
            await logger.ainfo(
                "Stake created",
                stake_id=str(stake_id),
                user_id=user_id,
                amount=float(amount),
                stake_type=stake_type.value
            )
            
            return self._stake_row_to_record(model)
    
    async def unstake(
        self,
        user_id: str,
        stake_id: str,
        amount: Optional[Decimal] = None
    ) -> UnstakeRequest:
        # sp933 — serialize ops on this stake (unstake + slash) so concurrent
        # partial-unstakes can't each pass on the same stale balance.
        async with self._lock_for(f"stake:{stake_id}"):
            return await self._unstake_locked(user_id, stake_id, amount)

    async def _unstake_locked(
        self,
        user_id: str,
        stake_id: str,
        amount: Optional[Decimal] = None
    ) -> UnstakeRequest:
        """
        Request to unstake tokens
        
        Args:
            user_id: User making the request
            stake_id: ID of the stake to unstake
            amount: Amount to unstake (None = full stake)
            
        Returns:
            UnstakeRequest: The created unstake request
            
        Raises:
            ValueError: If unstake validation fails
        """
        
        # Get stake from database
        async with get_async_session() as db:
            query = select(StakeModel).where(StakeModel.stake_id == UUID(stake_id))
            result = await db.execute(query)
            stake_row = result.scalar_one_or_none()
            
            if not stake_row:
                raise ValueError(f"Stake {stake_id} not found")
            
            if stake_row.user_id != user_id:
                raise ValueError("Stake does not belong to user")
            
            if stake_row.status != StakeStatus.ACTIVE.value:
                raise ValueError(f"Stake is not active: {stake_row.status}")

            # sp906 — enforce the utility-benefit lock. A staker who
            # committed to a 30/90/365-day lock (to earn discounts +
            # priority) cannot unstake until it expires; otherwise the
            # benefit would be trivially gameable (stake, get the
            # discount, immediately unstake).
            lock_expires = self._lock_expires_at(stake_row.stake_metadata)
            if lock_expires is not None and (
                datetime.now(timezone.utc) < lock_expires
            ):
                raise ValueError(
                    f"Stake is locked until {lock_expires.isoformat()} "
                    f"(utility-benefit lock commitment); cannot unstake yet"
                )

            stake = self._stake_row_to_record(stake_row)
            
            # Determine amount
            unstake_amount = amount if amount else stake.amount
            if unstake_amount > stake.amount:
                raise ValueError(f"Cannot unstake more than staked amount")
            
            if unstake_amount <= 0:
                raise ValueError("Unstake amount must be positive")
            
            # Check rate limit
            await self._check_rate_limit(user_id, "unstake")
            
            # Create unstake request
            request_id = uuid4()
            now = datetime.now(timezone.utc)
            available_at = now + timedelta(seconds=self.config.unstaking_period_seconds)
            
            # Create unstake request model
            request_model = UnstakeRequestModel(
                request_id=request_id,
                stake_id=UUID(stake_id),
                user_id=user_id,
                amount=float(unstake_amount),
                status=UnstakeRequestStatus.PENDING.value,
                requested_at=now,
                available_at=available_at,
            )
            db.add(request_model)
            
            # Update stake status
            if unstake_amount == stake.amount:
                stake_row.status = StakeStatus.UNSTAKING.value
            else:
                # Partial unstake - reduce stake amount
                stake_row.amount = float(stake.amount - unstake_amount)
            
            await db.commit()
            await db.refresh(request_model)
            
            # Record operation
            self._record_operation(user_id, "unstake")
            
            await logger.ainfo(
                "Unstake request created",
                request_id=str(request_id),
                stake_id=stake_id,
                user_id=user_id,
                amount=float(unstake_amount),
                available_at=available_at.isoformat()
            )
            
            return self._unstake_row_to_record(request_model)
    
    async def withdraw(
        self,
        user_id: str,
        request_id: str
    ) -> Tuple[bool, Decimal]:
        # sp933 — serialize all ops on this unstake request (withdraw + cancel)
        # so two concurrent/replayed calls can't both unlock the tokens.
        async with self._lock_for(f"req:{request_id}"):
            return await self._withdraw_locked(user_id, request_id)

    async def _withdraw_locked(
        self,
        user_id: str,
        request_id: str
    ) -> Tuple[bool, Decimal]:
        """
        Withdraw unstaked tokens after period
        
        Args:
            user_id: User making the withdrawal
            request_id: ID of the unstake request
            
        Returns:
            Tuple[bool, Decimal]: (success, amount withdrawn)
            
        Raises:
            ValueError: If withdrawal validation fails
        """
        
        async with get_async_session() as db:
            # Get unstake request from database
            query = select(UnstakeRequestModel).where(UnstakeRequestModel.request_id == UUID(request_id))
            result = await db.execute(query)
            request_row = result.scalar_one_or_none()
            
            if not request_row:
                raise ValueError(f"Unstake request {request_id} not found")
            
            if request_row.user_id != user_id:
                raise ValueError("Request does not belong to user")
            
            if request_row.status not in (UnstakeRequestStatus.PENDING.value, UnstakeRequestStatus.AVAILABLE.value):
                raise ValueError(f"Request status invalid: {request_row.status}")
            
            # Check if available
            now = datetime.now(timezone.utc)
            if now < request_row.available_at:
                wait_seconds = (request_row.available_at - now).total_seconds()
                raise ValueError(
                    f"Unstaking period not complete. Wait {wait_seconds:.0f} more seconds"
                )
            
            # Convert to dataclass for amount access
            request = self._unstake_row_to_record(request_row)
            
            # Update request status
            request_row.status = UnstakeRequestStatus.COMPLETED.value
            request_row.completed_at = now
            
            # Get stake and update if full unstake
            stake_query = select(StakeModel).where(StakeModel.stake_id == request_row.stake_id)
            stake_result = await db.execute(stake_query)
            stake_row = stake_result.scalar_one_or_none()
            
            if stake_row and stake_row.status == StakeStatus.UNSTAKING.value:
                stake_row.status = StakeStatus.WITHDRAWN.value
            
            await db.commit()
            
            # Unlock tokens back to user
            await self.ftns.unlock_tokens(user_id, request.amount, reason="unstake_withdrawal")
            
            await logger.ainfo(
                "Withdrawal completed",
                request_id=request_id,
                user_id=user_id,
                amount=float(request.amount)
            )
            
            return True, request.amount
    
    async def cancel_unstake(
        self,
        user_id: str,
        request_id: str,
        reason: Optional[str] = None
    ) -> bool:
        # sp933 — same request lock as withdraw: a withdraw and a cancel (or two
        # cancels) on the same request must not both restore/unlock.
        async with self._lock_for(f"req:{request_id}"):
            return await self._cancel_unstake_locked(user_id, request_id, reason)

    async def _cancel_unstake_locked(
        self,
        user_id: str,
        request_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Cancel an unstake request
        
        Args:
            user_id: User cancelling the request
            request_id: ID of the unstake request
            reason: Reason for cancellation
            
        Returns:
            bool: True if cancelled successfully
            
        Raises:
            ValueError: If cancellation validation fails
        """
        
        async with get_async_session() as db:
            # Get unstake request from database
            query = select(UnstakeRequestModel).where(UnstakeRequestModel.request_id == UUID(request_id))
            result = await db.execute(query)
            request_row = result.scalar_one_or_none()
            
            if not request_row:
                raise ValueError(f"Unstake request {request_id} not found")
            
            if request_row.user_id != user_id:
                raise ValueError("Request does not belong to user")
            
            if request_row.status != UnstakeRequestStatus.PENDING.value:
                raise ValueError(f"Cannot cancel request in status: {request_row.status}")
            
            # Convert to dataclass for amount access
            request = self._unstake_row_to_record(request_row)
            
            # Cancel request
            request_row.status = UnstakeRequestStatus.CANCELLED.value
            request_row.cancellation_reason = reason or "User cancelled"
            
            # Restore stake
            stake_query = select(StakeModel).where(StakeModel.stake_id == request_row.stake_id)
            stake_result = await db.execute(stake_query)
            stake_row = stake_result.scalar_one_or_none()
            
            if stake_row:
                if stake_row.status == StakeStatus.UNSTAKING.value:
                    stake_row.status = StakeStatus.ACTIVE.value
                else:
                    # Partial unstake - restore amount
                    stake_row.amount = float(Decimal(str(stake_row.amount)) + request.amount)
            
            await db.commit()
            
            await logger.ainfo(
                "Unstake request cancelled",
                request_id=request_id,
                user_id=user_id,
                reason=reason
            )
            
            return True
    
    # === Slashing Operations ===
    
    async def slash(
        self,
        user_id: str,
        stake_id: str,
        reason: SlashReason,
        evidence: Dict[str, Any],
        slash_rate: Optional[float] = None,
        slashed_by: str = "system"
    ) -> SlashRecord:
        # sp933 — same stake lock as unstake: concurrent slashes must serialize
        # so each burn matches a stake decrement (no double-burn / divergence).
        async with self._lock_for(f"stake:{stake_id}"):
            return await self._slash_locked(
                user_id, stake_id, reason, evidence, slash_rate, slashed_by
            )

    async def _slash_locked(
        self,
        user_id: str,
        stake_id: str,
        reason: SlashReason,
        evidence: Dict[str, Any],
        slash_rate: Optional[float] = None,
        slashed_by: str = "system"
    ) -> SlashRecord:
        """
        Slash staked tokens for misbehavior
        
        Args:
            user_id: User being slashed
            stake_id: ID of the stake to slash
            reason: Reason for slashing
            evidence: Evidence supporting the slash
            slash_rate: Rate to slash (uses default if not provided)
            slashed_by: Entity that initiated the slash
            
        Returns:
            SlashRecord: Record of the slash event
            
        Raises:
            ValueError: If slash validation fails
        """
        
        async with get_async_session() as db:
            # Get stake from database
            query = select(StakeModel).where(StakeModel.stake_id == UUID(stake_id))
            result = await db.execute(query)
            stake_row = result.scalar_one_or_none()
            
            if not stake_row:
                raise ValueError(f"Stake {stake_id} not found")
            
            if stake_row.user_id != user_id:
                raise ValueError("Stake does not belong to user")
            
            if stake_row.status not in (StakeStatus.ACTIVE.value, StakeStatus.UNSTAKING.value):
                raise ValueError(f"Cannot slash stake in status: {stake_row.status}")
            
            stake = self._stake_row_to_record(stake_row)
            
            # Calculate slash rate
            effective_rate = slash_rate if slash_rate else self.config.slashing_rate_base
            
            # Cap at maximum
            effective_rate = min(effective_rate, self.config.max_slash_per_event)
            
            # Calculate slash amount
            slash_amount = stake.amount * Decimal(str(effective_rate))
            slash_amount = slash_amount.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
            
            # Create slash record
            slash_id = uuid4()
            now = datetime.now(timezone.utc)
            appeal_deadline = now + timedelta(seconds=self.config.appeal_period_seconds)
            
            # Create slash model
            slash_model = SlashEventModel(
                slash_id=slash_id,
                stake_id=UUID(stake_id),
                user_id=user_id,
                amount_slashed=float(slash_amount),
                reason=reason.value,
                slash_rate=effective_rate,
                evidence=evidence,
                slashed_at=now,
                slashed_by=slashed_by,
                appeal_deadline=appeal_deadline,
            )
            db.add(slash_model)
            
            # Apply slash to stake
            new_amount = float(stake.amount - slash_amount)
            stake_row.amount = new_amount
            stake_row.rewards_earned = max(0.0, float(stake.rewards_earned - slash_amount))
            
            # If stake is depleted, mark as slashed
            if new_amount <= 0:
                stake_row.status = StakeStatus.SLASHED.value
            
            await db.commit()
            await db.refresh(slash_model)
            
            # Burn slashed tokens (or send to treasury)
            await self.ftns.burn_tokens(user_id, slash_amount, reason=f"slash:{reason.value}")
            
            await logger.awarning(
                "Stake slashed",
                slash_id=str(slash_id),
                stake_id=stake_id,
                user_id=user_id,
                amount=float(slash_amount),
                reason=reason.value,
                slashed_by=slashed_by
            )
            
            return self._slash_row_to_record(slash_model)
    
    async def appeal_slash(
        self,
        user_id: str,
        slash_id: str,
        appeal_evidence: Dict[str, Any]
    ) -> bool:
        """
        Submit an appeal for a slash
        
        Args:
            user_id: User submitting the appeal
            slash_id: ID of the slash to appeal
            appeal_evidence: Evidence supporting the appeal
            
        Returns:
            bool: True if appeal submitted successfully
            
        Raises:
            ValueError: If appeal validation fails
        """
        
        async with get_async_session() as db:
            # Get slash record from database
            query = select(SlashEventModel).where(SlashEventModel.slash_id == UUID(slash_id))
            result = await db.execute(query)
            slash_row = result.scalar_one_or_none()
            
            if not slash_row:
                raise ValueError(f"Slash record {slash_id} not found")
            
            if slash_row.user_id != user_id:
                raise ValueError("Slash does not belong to user")
            
            if slash_row.appeal_status:
                raise ValueError(f"Slash already has appeal status: {slash_row.appeal_status}")
            
            now = datetime.now(timezone.utc)
            if now > slash_row.appeal_deadline:
                raise ValueError("Appeal deadline has passed")
            
            # Set appeal status
            slash_row.appeal_status = "pending"
            slash_row.appeal_evidence = appeal_evidence
            
            await db.commit()
            
            await logger.ainfo(
                "Slash appeal submitted",
                slash_id=slash_id,
                user_id=user_id
            )
            
            return True
    
    async def _stake_id_for_slash(self, slash_id: str) -> Optional[str]:
        """sp945 — the stake a slash belongs to, used by resolve_appeal to take
        the stake-level lock. Returns None for an unknown/malformed slash_id so
        the caller can defer to the inner method's canonical "not found" error
        without holding an irrelevant lock."""
        try:
            sid = UUID(slash_id)
        except (ValueError, TypeError, AttributeError):
            return None
        async with get_async_session() as db:
            row = (await db.execute(
                select(SlashEventModel.stake_id).where(
                    SlashEventModel.slash_id == sid
                )
            )).scalar_one_or_none()
            return str(row) if row is not None else None

    async def resolve_appeal(
        self,
        slash_id: str,
        approved: bool,
        resolution_note: str
    ) -> bool:
        # sp945 — serialize on the STAKE, not just the slash. The approved path
        # refunds stake_row.amount and flips SLASHED→ACTIVE; unstake / slash /
        # withdraw mutate that SAME row under stake:{stake_id}. sp933 locked only
        # slash:{slash_id}, leaving resolve_appeal cross-entity with those: an
        # appeal resolution could interleave between an unstake's read and commit,
        # overwriting the decrement (lost update) and reviving an UNSTAKING/SLASHED
        # stake to ACTIVE while its unstake request still stands (double-withdraw).
        # The stake lock STILL serializes same-slash double-resolve (same stake →
        # second call sees appeal_status != "pending" and raises → mint once) and
        # now also serializes concurrent refunds of MULTIPLE slashes on the same
        # stake, which the slash-keyed lock did not.
        stake_id = await self._stake_id_for_slash(slash_id)
        if stake_id is None:
            # Unknown slash — no stake state to mutate; let _resolve_appeal_locked
            # raise the canonical "not found" error without an irrelevant lock.
            return await self._resolve_appeal_locked(slash_id, approved, resolution_note)
        async with self._lock_for(f"stake:{stake_id}"):
            return await self._resolve_appeal_locked(slash_id, approved, resolution_note)

    async def _resolve_appeal_locked(
        self,
        slash_id: str,
        approved: bool,
        resolution_note: str
    ) -> bool:
        """
        Resolve a slash appeal (governance operation)
        
        Args:
            slash_id: ID of the slash
            approved: Whether to approve the appeal
            resolution_note: Note explaining the resolution
            
        Returns:
            bool: True if resolved successfully
        """
        
        async with get_async_session() as db:
            # Get slash record from database
            query = select(SlashEventModel).where(SlashEventModel.slash_id == UUID(slash_id))
            result = await db.execute(query)
            slash_row = result.scalar_one_or_none()
            
            if not slash_row:
                raise ValueError(f"Slash record {slash_id} not found")
            
            if slash_row.appeal_status != "pending":
                raise ValueError(f"Slash appeal not pending: {slash_row.appeal_status}")
            
            slash_row.appeal_status = "approved" if approved else "rejected"
            
            if approved:
                # Convert to dataclass for amount access
                slash_record = self._slash_row_to_record(slash_row)
                
                # Refund the slashed amount
                stake_query = select(StakeModel).where(StakeModel.stake_id == slash_row.stake_id)
                stake_result = await db.execute(stake_query)
                stake_row = stake_result.scalar_one_or_none()
                
                if stake_row:
                    stake_row.amount = float(Decimal(str(stake_row.amount)) + slash_record.amount_slashed)
                    if stake_row.status == StakeStatus.SLASHED.value:
                        stake_row.status = StakeStatus.ACTIVE.value
                
                await db.commit()
                
                # Credit tokens back
                await self.ftns.mint_tokens(
                    slash_record.user_id,
                    slash_record.amount_slashed,
                    reason=f"appeal_refund:{slash_id}"
                )
                
                await logger.ainfo(
                    "Slash appeal approved - refunding",
                    slash_id=slash_id,
                    user_id=slash_record.user_id,
                    amount=float(slash_record.amount_slashed)
                )
            else:
                await db.commit()
                
                await logger.ainfo(
                    "Slash appeal rejected",
                    slash_id=slash_id,
                    user_id=slash_row.user_id,
                    resolution_note=resolution_note
                )
            
            return True
    
    # === Reward Operations ===
    
    async def calculate_rewards(
        self,
        user_id: str,
        stake_id: Optional[str] = None
    ) -> List[RewardCalculation]:
        """
        Calculate pending rewards for a user's stakes
        
        Args:
            user_id: User to calculate rewards for
            stake_id: Specific stake (None = all user's stakes)
            
        Returns:
            List[RewardCalculation]: empty — staking accrues NO token yield.
        """
        # Sprint 904 — staking yields ELIMINATED. There is no
        # inflationary, mint-funded staking return (PRSM_Tokenomics.md
        # §10 #7: "no discretionary inflation-based rewards"; the
        # deployed v1 emission split has no staker-yield pool). Staking
        # confers utility benefits (lock-based service discounts /
        # priority access), not a token yield. No reward accrues, so
        # claim_rewards() mints nothing. Stake/unstake/slash mechanics
        # are unaffected.
        return []
    
    async def claim_rewards(
        self,
        user_id: str,
        stake_id: Optional[str] = None
    ) -> Decimal:
        """
        Claim accumulated staking rewards
        
        Args:
            user_id: User claiming rewards
            stake_id: Specific stake (None = all stakes)
            
        Returns:
            Decimal: Total rewards claimed — always 0 (sp904: staking
            accrues no token yield).
        """
        # Sprint 904 — staking yields ELIMINATED. calculate_rewards()
        # returns no accruals, so there is nothing to claim and NOTHING
        # IS MINTED. The previous mint-funded `reason="staking_rewards"`
        # path is removed: it was a discretionary inflationary reward
        # (Howey "expectation of profit" flag) with no funding source in
        # the deployed v1 emission split (PRSM_Tokenomics.md §10 #7).
        # Surface retained as an inert no-op so callers (auto-claim
        # worker / CLI / MCP) don't break; staking gives utility, not
        # yield. (Slash appeal-refund minting is a separate, legitimate
        # path and is unaffected.)
        calculations = await self.calculate_rewards(user_id, stake_id)
        if not calculations:
            return Decimal('0')
        # Defense-in-depth: even if a future change re-introduces
        # accruals, this method must never mint inflationary yield.
        await logger.awarning(
            "claim_rewards: staking yields are eliminated (sp904); "
            "ignoring %d stale accrual(s), minting nothing",
            len(calculations),
        )
        return Decimal('0')
    
    # === Query Operations ===
    
    async def get_stake(self, stake_id: str) -> Optional[StakeRecord]:
        """Get a specific stake by ID"""
        async with get_async_session() as db:
            query = select(StakeModel).where(StakeModel.stake_id == UUID(stake_id))
            result = await db.execute(query)
            row = result.scalar_one_or_none()
            if row:
                return self._stake_row_to_record(row)
            return None
    
    async def get_user_stakes(
        self,
        user_id: str,
        status: Optional[StakeStatus] = None
    ) -> List[StakeRecord]:
        """
        Get all stakes for a user
        
        Args:
            user_id: User to get stakes for
            status: Filter by status (None = all statuses)
            
        Returns:
            List[StakeRecord]: User's stakes
        """
        
        async with get_async_session() as db:
            query = select(StakeModel).where(StakeModel.user_id == user_id)
            if status:
                query = query.where(StakeModel.status == status.value)
            
            result = await db.execute(query)
            rows = result.scalars().all()

            return [self._stake_row_to_record(row) for row in rows]

    @staticmethod
    def _lock_expires_at(stake_metadata: Any) -> Optional[datetime]:
        """sp906 — parse the lock expiry from a stake's metadata, if any.

        Returns a tz-aware UTC datetime, or None when the stake carries
        no lock commitment (legacy stakes / no lock chosen). Defensive
        against malformed metadata (SQLite stores JSON as text)."""
        if not stake_metadata:
            return None
        meta = stake_metadata
        if isinstance(meta, str):
            try:
                import json
                meta = json.loads(meta)
            except (ValueError, TypeError):
                return None
        if not isinstance(meta, dict):
            return None
        raw = meta.get("lock_expires_at")
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    async def get_user_benefits(self, user_id: str) -> StakingBenefits:
        """sp906 — the staking utility benefits currently in effect for a
        user.

        Scans the user's ACTIVE stakes and returns the BEST (highest
        discount) tier among those that (a) meet the minimum stake and
        (b) are still within their lock period. A staker can hold several
        locks; they get the strongest one's benefits. Returns
        ``StakingBenefits.none()`` when no active, sufficiently-funded,
        still-locked stake exists.

        This is the single source of truth the pricing layer
        (network-fee discount) and dispatch layer (priority access)
        consume."""
        now = datetime.now(timezone.utc)
        best = StakingBenefits.none()
        async with get_async_session() as db:
            query = select(StakeModel).where(
                StakeModel.user_id == user_id,
                StakeModel.status == StakeStatus.ACTIVE.value,
            )
            result = await db.execute(query)
            rows = result.scalars().all()

        for row in rows:
            if Decimal(str(row.amount)) < Decimal(self.config.minimum_stake):
                continue
            lock_expires = self._lock_expires_at(row.stake_metadata)
            if lock_expires is None or now >= lock_expires:
                continue  # no lock, or lock already expired → no benefit
            meta = row.stake_metadata
            if isinstance(meta, str):
                try:
                    import json
                    meta = json.loads(meta)
                except (ValueError, TypeError):
                    meta = {}
            lock_days = (meta or {}).get("lock_period_days")
            candidate = self.config.benefits_for_lock_days(lock_days)
            if candidate.discount_fraction > best.discount_fraction:
                best = candidate
        return best

    async def get_user_total_stake(self, user_id: str) -> Decimal:
        """Get total staked amount for a user"""
        
        async with get_async_session() as db:
            query = select(StakeModel).where(
                StakeModel.user_id == user_id,
                StakeModel.status == StakeStatus.ACTIVE.value
            )
            result = await db.execute(query)
            rows = result.scalars().all()
            
            return sum(Decimal(str(row.amount)) for row in rows)
    
    async def get_network_total_stake(self) -> Decimal:
        """Get total staked amount across all users"""
        
        async with get_async_session() as db:
            query = select(StakeModel).where(StakeModel.status == StakeStatus.ACTIVE.value)
            result = await db.execute(query)
            rows = result.scalars().all()
            
            return sum(Decimal(str(row.amount)) for row in rows)
    
    async def get_unstake_request(self, request_id: str) -> Optional[UnstakeRequest]:
        """Get a specific unstake request by ID"""
        async with get_async_session() as db:
            query = select(UnstakeRequestModel).where(UnstakeRequestModel.request_id == UUID(request_id))
            result = await db.execute(query)
            row = result.scalar_one_or_none()
            if row:
                return self._unstake_row_to_record(row)
            return None
    
    async def get_pending_unstake_requests(
        self,
        user_id: Optional[str] = None
    ) -> List[UnstakeRequest]:
        """
        Get pending unstake requests
        
        Args:
            user_id: Filter by user (None = all users)
            
        Returns:
            List[UnstakeRequest]: Pending requests
        """
        
        async with get_async_session() as db:
            query = select(UnstakeRequestModel).where(
                UnstakeRequestModel.status.in_([
                    UnstakeRequestStatus.PENDING.value,
                    UnstakeRequestStatus.AVAILABLE.value
                ])
            )
            if user_id:
                query = query.where(UnstakeRequestModel.user_id == user_id)
            
            result = await db.execute(query)
            rows = result.scalars().all()
            
            return [self._unstake_row_to_record(row) for row in rows]
    
    async def get_available_withdrawals(self, user_id: str) -> List[UnstakeRequest]:
        """Get unstake requests ready for withdrawal"""
        
        now = datetime.now(timezone.utc)
        
        async with get_async_session() as db:
            query = select(UnstakeRequestModel).where(
                UnstakeRequestModel.user_id == user_id,
                UnstakeRequestModel.status == UnstakeRequestStatus.AVAILABLE.value,
                UnstakeRequestModel.available_at <= now
            )
            result = await db.execute(query)
            rows = result.scalars().all()
            
            return [self._unstake_row_to_record(row) for row in rows]
    
    async def get_slash_history(
        self,
        user_id: Optional[str] = None,
        stake_id: Optional[str] = None
    ) -> List[SlashRecord]:
        """
        Get slash history
        
        Args:
            user_id: Filter by user (None = all users)
            stake_id: Filter by stake (None = all stakes)
            
        Returns:
            List[SlashRecord]: Slash records
        """
        
        async with get_async_session() as db:
            query = select(SlashEventModel)
            
            if user_id:
                query = query.where(SlashEventModel.user_id == user_id)
            if stake_id:
                query = query.where(SlashEventModel.stake_id == UUID(stake_id))
            
            query = query.order_by(SlashEventModel.slashed_at.desc())
            result = await db.execute(query)
            rows = result.scalars().all()
            
            return [self._slash_row_to_record(row) for row in rows]
    
    # === Internal Methods ===
    
    async def _check_rate_limit(self, user_id: str, operation: str) -> None:
        """Check if user has exceeded rate limits"""
        
        now = datetime.now(timezone.utc)
        key = f"{user_id}:{operation}"
        
        if key not in self._daily_operations:
            self._daily_operations[key] = []
        
        # Clean old operations
        day_ago = now - timedelta(days=1)
        self._daily_operations[key] = [
            ts for ts in self._daily_operations[key]
            if ts > day_ago
        ]
        
        # Check limit
        max_ops = (
            self.config.max_stake_operations_per_day
            if operation == "stake"
            else self.config.max_unstake_operations_per_day
        )
        
        if len(self._daily_operations[key]) >= max_ops:
            raise ValueError(
                f"Rate limit exceeded for {operation}. Max {max_ops} per day."
            )
    
    def _record_operation(self, user_id: str, operation: str) -> None:
        """Record an operation for rate limiting"""
        
        key = f"{user_id}:{operation}"
        if key not in self._daily_operations:
            self._daily_operations[key] = []
        
        self._daily_operations[key].append(datetime.now(timezone.utc))
    
    # === Maintenance Operations ===
    
    async def process_matured_unstakes(self) -> List[UnstakeRequest]:
        """
        Process all unstake requests that have matured
        
        This should be called periodically by a background task.
        
        Returns:
            List[UnstakeRequest]: Requests that are now available
        """
        async with get_async_session() as db:
            now = datetime.now(timezone.utc)
            matured = []
            
            # Query pending unstake requests that have matured
            query = select(UnstakeRequestModel).where(
                UnstakeRequestModel.status == UnstakeRequestStatus.PENDING.value,
                UnstakeRequestModel.available_at <= now
            )
            result = await db.execute(query)
            matured_rows = result.scalars().all()
            
            # Update each matured request
            for row in matured_rows:
                row.status = UnstakeRequestStatus.AVAILABLE.value
                matured.append(self._unstake_row_to_record(row))
                
                await logger.ainfo(
                    "Unstake request matured",
                    request_id=str(row.request_id),
                    user_id=row.user_id
                )
            
            await db.commit()
            return matured
    
    async def get_staking_stats(self) -> Dict[str, Any]:
        """
        Get staking statistics
        
        Returns:
            Dict with staking statistics
        """
        async with get_async_session() as db:
            # Get total stakes count
            total_stakes_query = select(func.count(StakeModel.stake_id))
            total_stakes_result = await db.execute(total_stakes_query)
            total_stakes = total_stakes_result.scalar() or 0
            
            # Get active stakes count and total
            active_stakes_query = select(StakeModel).where(
                StakeModel.status == StakeStatus.ACTIVE.value
            )
            active_result = await db.execute(active_stakes_query)
            active_rows = active_result.scalars().all()
            active_stakes_count = len(active_rows)
            total_staked = sum(Decimal(str(s.amount)) for s in active_rows)
            
            # Get unstaking stakes total
            unstaking_query = select(StakeModel).where(
                StakeModel.status == StakeStatus.UNSTAKING.value
            )
            unstaking_result = await db.execute(unstaking_query)
            unstaking_rows = unstaking_result.scalars().all()
            total_unstaking = sum(Decimal(str(s.amount)) for s in unstaking_rows)
            
            # Get pending unstake requests count
            pending_query = select(func.count(UnstakeRequestModel.request_id)).where(
                UnstakeRequestModel.status == UnstakeRequestStatus.PENDING.value
            )
            pending_result = await db.execute(pending_query)
            pending_count = pending_result.scalar() or 0
            
            # Get total slashes count
            slashes_query = select(func.count(SlashEventModel.slash_id))
            slashes_result = await db.execute(slashes_query)
            total_slashes = slashes_result.scalar() or 0
            
            # Get unique stakers count
            unique_stakers_query = select(func.count(func.distinct(StakeModel.user_id)))
            unique_stakers_result = await db.execute(unique_stakers_query)
            unique_stakers = unique_stakers_result.scalar() or 0
            
            return {
                "total_stakes": total_stakes,
                "active_stakes": active_stakes_count,
                "total_staked": float(total_staked),
                "total_unstaking": float(total_unstaking),
                "pending_unstake_requests": pending_count,
                "total_slashes": total_slashes,
                "unique_stakers": unique_stakers,
                "config": {
                    "minimum_stake": self.config.minimum_stake,
                    "unstaking_period_days": self.config.unstaking_period_seconds / 86400,
                    "reward_rate_annual": self.config.reward_rate_annual,
                    "slashing_rate_base": self.config.slashing_rate_base
                }
            }


# === Factory Function ===

_staking_manager_instance: Optional[StakingManager] = None


async def get_staking_manager(
    db_session: AsyncSession = None,   # kept for API compat, ignored
    ftns_service=None,
    config: Optional[StakingConfig] = None
) -> StakingManager:
    """Get or create the global StakingManager singleton.
    
    Note: db_session is kept for API compatibility but is no longer used.
    StakingManager manages its own database sessions internally via get_async_session().
    """
    global _staking_manager_instance
    if _staking_manager_instance is None:
        _staking_manager_instance = StakingManager(
            db_session=None,
            ftns_service=ftns_service,
            config=config,
        )
    return _staking_manager_instance