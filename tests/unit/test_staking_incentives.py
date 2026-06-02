"""
Tests for Staking and Incentive Mechanisms (Phase 3.2)

This module tests the staking system including:
- Staking operations (stake, unstake, withdraw)
- Slashing mechanism for misbehavior
- Reward calculations and distribution
- Supply adjustments
- Anti-hoarding decay integration
"""

import pytest
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text as sa_text, TypeDecorator, DateTime

from prsm.core.database import Base


class TZDateTime(TypeDecorator):
    """SQLite-compatible timezone-aware datetime type.
    
    Stores datetime as ISO format strings and parses them back as timezone-aware.
    """
    impl = DateTime
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            # Convert to UTC and format as ISO string
            if value.tzinfo is not None:
                value = value.astimezone(timezone.utc)
            return value.isoformat()
        return value
    
    def process_result_value(self, value, dialect):
        if value is not None:
            # Parse ISO string back to timezone-aware datetime
            if isinstance(value, str):
                # Handle both with and without timezone
                if '+' in value or value.endswith('Z'):
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                else:
                    # Naive datetime - assume UTC
                    value = datetime.fromisoformat(value)
                    value = value.replace(tzinfo=timezone.utc)
            elif value.tzinfo is None:
                # Naive datetime from SQLite - assume UTC
                value = value.replace(tzinfo=timezone.utc)
            return value
        return value
from prsm.economy.tokenomics.staking_manager import (
    StakingManager,
    StakingConfig,
    StakeRecord,
    UnstakeRequest,
    SlashRecord,
    RewardCalculation,
    StakeStatus,
    UnstakeRequestStatus,
    SlashReason,
    StakeType,
    get_staking_manager
)


# === Fixtures ===

@pytest.fixture
def staking_config():
    """Create a test staking configuration"""
    return StakingConfig(
        minimum_stake=1000,
        unstaking_period_seconds=60,  # 1 minute for testing
        reward_rate_annual=0.05,
        slashing_rate_base=0.1,
        max_stake_per_user=10_000_000,
        max_total_stake=1_000_000_000,
        reward_compounding=False,
        min_reward_claim_interval_seconds=60,
        governance_slash_multiplier=2.0,
        appeal_period_seconds=180,  # 3 minutes for testing
        max_stake_operations_per_day=5,
        max_unstake_operations_per_day=3,
        reward_calculation_interval_seconds=60,
        min_stake_age_for_rewards_seconds=60,
        downtime_slash_threshold_seconds=300,
        max_slash_per_event=0.5
    )


@pytest.fixture
def mock_ftns_service():
    """Create a mock FTNS service"""
    service = AsyncMock()
    service.get_available_balance = AsyncMock(return_value=Decimal('100000'))
    service.lock_tokens = AsyncMock(return_value=True)
    service.unlock_tokens = AsyncMock(return_value=True)
    service.burn_tokens = AsyncMock(return_value=True)
    service.mint_tokens = AsyncMock(return_value=True)
    return service


@pytest.fixture
async def async_db_session():
    """Create an in-memory SQLite database for testing staking operations."""
    from sqlalchemy import event
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.orm import Session
    
    # Use StaticPool with a single connection so all sessions see the same data
    # This is critical for tests where one session updates data that another session reads
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    # Helper to convert naive datetime to timezone-aware
    def make_tz_aware(dt):
        if dt is None:
            return None
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        # Handle ISO format strings from SQLite
        if isinstance(dt, str):
            try:
                # Try parsing ISO format with timezone
                parsed = datetime.fromisoformat(dt)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                pass
        return dt
    
    # Helper to process datetime fields in loaded objects
    def process_datetime_fields(obj):
        """Convert naive datetime fields to timezone-aware for SQLite compatibility."""
        datetime_fields = [
            'last_reward_calculation', 'staked_at', 'unstake_requested_at', 'withdrawn_at',
            'requested_at', 'available_at', 'completed_at',
            'slashed_at', 'appeal_deadline'
        ]
        for field in datetime_fields:
            if hasattr(obj, field):
                val = getattr(obj, field)
                if val is not None:
                    setattr(obj, field, make_tz_aware(val))
    
    # Use the loaded_as_persistent event on the Session class
    # This event fires when an object is loaded from the database
    @event.listens_for(Session, "loaded_as_persistent")
    def receive_loaded_as_persistent(session, instance):
        process_datetime_fields(instance)
    
    # Only create the staking-related tables (not all Base tables, some use PostgreSQL-specific types)
    from prsm.core.database import StakeModel, UnstakeRequestModel, SlashEventModel
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: None)  # Placeholder
        # Create tables individually using raw SQL for SQLite compatibility
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS ftns_stakes (
                stake_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                stake_type TEXT NOT NULL DEFAULT 'general',
                status TEXT NOT NULL DEFAULT 'active',
                rewards_earned REAL NOT NULL DEFAULT 0.0,
                rewards_claimed REAL NOT NULL DEFAULT 0.0,
                last_reward_calculation TEXT NOT NULL,
                staked_at TEXT NOT NULL,
                unstake_requested_at TEXT,
                withdrawn_at TEXT,
                lock_reason TEXT,
                stake_metadata TEXT
            )
        """))
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS ftns_unstake_requests (
                request_id TEXT PRIMARY KEY,
                stake_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                requested_at TEXT NOT NULL,
                available_at TEXT NOT NULL,
                completed_at TEXT,
                cancellation_reason TEXT,
                request_metadata TEXT
            )
        """))
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS ftns_slash_events (
                slash_id TEXT PRIMARY KEY,
                stake_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                amount_slashed REAL NOT NULL,
                reason TEXT NOT NULL,
                slash_rate REAL NOT NULL,
                slashed_by TEXT NOT NULL,
                evidence TEXT,
                slashed_at TEXT NOT NULL,
                appeal_deadline TEXT,
                appeal_status TEXT,
                appeal_evidence TEXT,
                slash_metadata TEXT
            )
        """))
        await conn.commit()
    
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def _fake_session():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # Patch both the staking_manager import and the core database import
    with patch("prsm.economy.tokenomics.staking_manager.get_async_session", _fake_session), \
         patch("prsm.core.database.get_async_session", _fake_session):
        yield _fake_session  # Yield the session factory for tests to use directly
    
    # Cleanup: remove the event listener
    event.remove(Session, "loaded_as_persistent", receive_loaded_as_persistent)


@pytest.fixture
def mock_db_session():
    """Create a mock database session - kept for backward compatibility with some tests."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
async def staking_manager(async_db_session, mock_db_session, mock_ftns_service, staking_config):
    """Create a staking manager instance with real SQLite database."""
    return StakingManager(
        db_session=mock_db_session,  # Required by constructor but not used (get_async_session is patched)
        ftns_service=mock_ftns_service,
        config=staking_config
    )


# === StakingConfig Tests ===

class TestStakingConfig:
    """Tests for StakingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = StakingConfig()
        
        assert config.minimum_stake == 1000
        assert config.unstaking_period_seconds == 7 * 24 * 3600  # 7 days
        assert config.reward_rate_annual == 0.0  # sp904: staking yields eliminated
        assert config.slashing_rate_base == 0.1  # 10%
        assert config.max_stake_per_user == 10_000_000
        assert config.max_total_stake == 1_000_000_000
        assert config.reward_compounding is False
    
    def test_custom_config(self, staking_config):
        """Test custom configuration values"""
        assert staking_config.minimum_stake == 1000
        assert staking_config.unstaking_period_seconds == 60
        assert staking_config.reward_rate_annual == 0.05
        assert staking_config.slashing_rate_base == 0.1


# === StakeRecord Tests ===

class TestStakeRecord:
    """Tests for StakeRecord dataclass"""
    
    def test_stake_record_creation(self):
        """Test creating a stake record"""
        now = datetime.now(timezone.utc)
        stake = StakeRecord(
            stake_id="test-stake-1",
            user_id="user-123",
            amount=Decimal('5000'),
            stake_type=StakeType.GOVERNANCE,
            staked_at=now,
            status=StakeStatus.ACTIVE
        )
        
        assert stake.stake_id == "test-stake-1"
        assert stake.user_id == "user-123"
        assert stake.amount == Decimal('5000')
        assert stake.stake_type == StakeType.GOVERNANCE
        assert stake.status == StakeStatus.ACTIVE
        assert stake.rewards_earned == Decimal('0')
    
    def test_stake_record_string_conversion(self):
        """Test stake record with string status/type"""
        stake = StakeRecord(
            stake_id="test-stake-1",
            user_id="user-123",
            amount=Decimal('5000'),
            stake_type="governance",  # String instead of enum
            staked_at=datetime.now(timezone.utc),
            status="active"  # String instead of enum
        )
        
        assert stake.stake_type == StakeType.GOVERNANCE
        assert stake.status == StakeStatus.ACTIVE


# === UnstakeRequest Tests ===

class TestUnstakeRequest:
    """Tests for UnstakeRequest dataclass"""
    
    def test_unstake_request_creation(self):
        """Test creating an unstake request"""
        now = datetime.now(timezone.utc)
        available = now + timedelta(hours=1)
        
        request = UnstakeRequest(
            request_id="request-1",
            stake_id="stake-1",
            user_id="user-123",
            amount=Decimal('5000'),
            requested_at=now,
            available_at=available,
            status=UnstakeRequestStatus.PENDING
        )
        
        assert request.request_id == "request-1"
        assert request.status == UnstakeRequestStatus.PENDING
        assert request.is_available is False
    
    def test_is_available_property(self):
        """Test is_available property"""
        # Request not yet available
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        request = UnstakeRequest(
            request_id="request-1",
            stake_id="stake-1",
            user_id="user-123",
            amount=Decimal('5000'),
            requested_at=datetime.now(timezone.utc),
            available_at=future,
            status=UnstakeRequestStatus.AVAILABLE
        )
        assert request.is_available is False
        
        # Request available
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        request.available_at = past
        assert request.is_available is True


# === SlashRecord Tests ===

class TestSlashRecord:
    """Tests for SlashRecord dataclass"""
    
    def test_slash_record_creation(self):
        """Test creating a slash record"""
        now = datetime.now(timezone.utc)
        
        slash = SlashRecord(
            slash_id="slash-1",
            stake_id="stake-1",
            user_id="user-123",
            amount_slashed=Decimal('500'),
            reason=SlashReason.MISCONDUCT,
            slash_rate=0.1,
            evidence={"description": "Test evidence"},
            slashed_at=now,
            slashed_by="governance-proposal-1"
        )
        
        assert slash.slash_id == "slash-1"
        assert slash.reason == SlashReason.MISCONDUCT
        assert slash.slash_rate == 0.1
        assert slash.appeal_deadline is not None
        assert slash.appeal_status is None
    
    def test_slash_record_appeal_deadline(self):
        """Test appeal deadline is set correctly"""
        now = datetime.now(timezone.utc)
        slash = SlashRecord(
            slash_id="slash-1",
            stake_id="stake-1",
            user_id="user-123",
            amount_slashed=Decimal('500'),
            reason=SlashReason.MISCONDUCT,
            slash_rate=0.1,
            evidence={},
            slashed_at=now,
            slashed_by="system"
        )
        
        # Default appeal period is 3 days
        expected_deadline = now + timedelta(days=3)
        assert abs((slash.appeal_deadline - expected_deadline).total_seconds()) < 1


# === StakingManager Tests ===

class TestStakingManager:
    """Tests for StakingManager class"""
    
    @pytest.mark.asyncio
    async def test_stake_creation(self, staking_manager, mock_ftns_service):
        """Test creating a stake"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000'),
            stake_type=StakeType.GOVERNANCE
        )
        
        assert stake is not None
        assert stake.user_id == "user-123"
        assert stake.amount == Decimal('5000')
        assert stake.stake_type == StakeType.GOVERNANCE
        assert stake.status == StakeStatus.ACTIVE
        
        # Verify FTNS service was called
        mock_ftns_service.lock_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stake_below_minimum(self, staking_manager):
        """Test staking below minimum amount"""
        with pytest.raises(ValueError, match="below minimum"):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('100')  # Below minimum of 1000
            )
    
    @pytest.mark.asyncio
    async def test_stake_exceeds_user_max(self, staking_manager, mock_ftns_service):
        """Test staking exceeding user maximum"""
        mock_ftns_service.get_available_balance = AsyncMock(
            return_value=Decimal('20000000')
        )
        
        with pytest.raises(ValueError, match="exceed user maximum"):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('15000000')  # Exceeds max of 10M
            )
    
    @pytest.mark.asyncio
    async def test_stake_insufficient_balance(self, staking_manager, mock_ftns_service):
        """Test staking with insufficient balance"""
        mock_ftns_service.get_available_balance = AsyncMock(
            return_value=Decimal('100')  # Not enough
        )
        
        with pytest.raises(ValueError, match="Insufficient balance"):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('5000')
            )
    
    @pytest.mark.asyncio
    async def test_unstake_creation(self, staking_manager, mock_ftns_service):
        """Test creating an unstake request"""
        # First create a stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Then unstake
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        assert request is not None
        assert request.user_id == "user-123"
        assert request.amount == Decimal('5000')
        assert request.status == UnstakeRequestStatus.PENDING
        # Re-fetch stake to get updated status from database
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_stake.status == StakeStatus.UNSTAKING
    
    @pytest.mark.asyncio
    async def test_partial_unstake(self, staking_manager, mock_ftns_service):
        """Test partial unstake"""
        # Create stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Partial unstake
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id,
            amount=Decimal('2000')
        )
        
        assert request.amount == Decimal('2000')
        # Re-fetch stake to get updated status from database
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_stake.status == StakeStatus.ACTIVE  # Still active
        assert updated_stake.amount == Decimal('3000')  # Reduced
    
    @pytest.mark.asyncio
    async def test_unstake_nonexistent_stake(self, staking_manager):
        """Test unstaking a non-existent stake"""
        # Use a valid UUID format that doesn't exist in the database
        nonexistent_uuid = str(uuid4())
        with pytest.raises(ValueError, match="not found"):
            await staking_manager.unstake(
                user_id="user-123",
                stake_id=nonexistent_uuid
            )
    
    @pytest.mark.asyncio
    async def test_unstake_wrong_user(self, staking_manager, mock_ftns_service):
        """Test unstaking another user's stake"""
        # Create stake for user-123
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Try to unstake as user-456
        with pytest.raises(ValueError, match="does not belong to user"):
            await staking_manager.unstake(
                user_id="user-456",
                stake_id=stake.stake_id
            )
    
    @pytest.mark.asyncio
    async def test_withdraw_success(self, staking_manager, mock_ftns_service):
        """Test successful withdrawal"""
        # Create and unstake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        # Make it available by updating database using ORM update
        from prsm.core.database import get_async_session, UnstakeRequestModel
        from sqlalchemy import update
        from uuid import UUID
        async with get_async_session() as session:
            past_time = datetime.now(timezone.utc) - timedelta(minutes=1)
            await session.execute(
                update(UnstakeRequestModel)
                .where(UnstakeRequestModel.request_id == UUID(request.request_id))
                .values(available_at=past_time, status=UnstakeRequestStatus.AVAILABLE.value)
            )
            await session.commit()
        
        # Withdraw
        success, amount = await staking_manager.withdraw(
            user_id="user-123",
            request_id=request.request_id
        )
        
        assert success is True
        assert amount == Decimal('5000')
        # Re-fetch to get updated status from database
        updated_request = await staking_manager.get_unstake_request(request.request_id)
        assert updated_request.status == UnstakeRequestStatus.COMPLETED
        mock_ftns_service.unlock_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_withdraw_too_early(self, staking_manager, mock_ftns_service):
        """Test withdrawing before period ends"""
        # Create and unstake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        # Try to withdraw immediately (period not over)
        with pytest.raises(ValueError, match="not complete"):
            await staking_manager.withdraw(
                user_id="user-123",
                request_id=request.request_id
            )
    
    @pytest.mark.asyncio
    async def test_cancel_unstake(self, staking_manager, mock_ftns_service):
        """Test cancelling an unstake request"""
        # Create and unstake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        # Cancel
        result = await staking_manager.cancel_unstake(
            user_id="user-123",
            request_id=request.request_id,
            reason="Changed mind"
        )
        
        assert result is True
        # Re-fetch to get updated status from database
        updated_request = await staking_manager.get_unstake_request(request.request_id)
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_request.status == UnstakeRequestStatus.CANCELLED
        assert updated_stake.status == StakeStatus.ACTIVE  # Restored


# === Slashing Tests ===

class TestSlashing:
    """Tests for slashing mechanism"""
    
    @pytest.mark.asyncio
    async def test_slash_creation(self, staking_manager, mock_ftns_service):
        """Test creating a slash"""
        # Create stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Slash
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={"description": "Test violation"},
            slashed_by="governance-proposal-1"
        )
        
        assert slash is not None
        assert slash.reason == SlashReason.MISCONDUCT
        assert slash.slash_rate == Decimal('0.1')  # Default 10%
        assert slash.amount_slashed == Decimal('500')  # 10% of 5000
        # Re-fetch stake to get updated amount from database
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_stake.amount == Decimal('4500')  # Reduced
        
        mock_ftns_service.burn_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_slash_custom_rate(self, staking_manager, mock_ftns_service):
        """Test slashing with custom rate"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.FRAUD,
            evidence={"description": "Fraud detected"},
            slash_rate=Decimal('0.3'),  # 30%
            slashed_by="system"
        )
        
        assert slash.slash_rate == Decimal('0.3')
        assert slash.amount_slashed == Decimal('1500')  # 30% of 5000
        # Re-fetch stake to get updated amount from database
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_stake.amount == Decimal('3500')
    
    @pytest.mark.asyncio
    async def test_slash_max_cap(self, staking_manager, mock_ftns_service):
        """Test slash is capped at maximum"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Try to slash 80% (over max of 50%)
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.SECURITY_BREACH,
            evidence={"description": "Security breach"},
            slash_rate=0.8,  # 80%
            slashed_by="system"
        )
        
        # Should be capped at 50%
        assert slash.slash_rate == 0.5
        assert slash.amount_slashed == Decimal('2500')  # 50% of 5000
    
    @pytest.mark.asyncio
    async def test_slash_depletes_stake(self, staking_manager, mock_ftns_service):
        """Test that slashing can deplete a stake"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('1000')
        )
        
        # Slash 50% (max)
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.FRAUD,
            evidence={"description": "Major fraud"},
            slash_rate=0.5,
            slashed_by="governance"
        )
        
        # Re-fetch stake to get updated amount from database
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_stake.amount == Decimal('500')
        assert updated_stake.status == StakeStatus.ACTIVE  # Still active
        
        # Slash again to deplete
        slash2 = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.FRAUD,
            evidence={"description": "More fraud"},
            slash_rate=0.5,
            slashed_by="governance"
        )
        
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_stake.amount == Decimal('250')
        
        # Slash to depletion
        slash3 = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.FRAUD,
            evidence={"description": "Final fraud"},
            slash_rate=1.0,  # 100% - but capped at 50%
            slashed_by="governance"
        )
        
        # Should be slashed but not fully depleted due to cap
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_stake.status == StakeStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_appeal_slash(self, staking_manager, mock_ftns_service):
        """Test submitting a slash appeal"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={"description": "Test"},
            slashed_by="system"
        )
        
        # Submit appeal
        result = await staking_manager.appeal_slash(
            user_id="user-123",
            slash_id=slash.slash_id,
            appeal_evidence={"description": "False accusation"}
        )
        
        assert result is True
        # Re-fetch slash to get updated appeal_status from database
        updated_slash = await staking_manager.get_slash_history("user-123")
        assert len(updated_slash) > 0
        assert updated_slash[0].appeal_status == "pending"
    
    @pytest.mark.asyncio
    async def test_resolve_appeal_approved(self, staking_manager, mock_ftns_service):
        """Test resolving an appeal (approved)"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={"description": "Test"},
            slashed_by="system"
        )
        
        # Submit appeal
        await staking_manager.appeal_slash(
            user_id="user-123",
            slash_id=slash.slash_id,
            appeal_evidence={"description": "False accusation"}
        )
        
        # Resolve appeal (approved)
        result = await staking_manager.resolve_appeal(
            slash_id=slash.slash_id,
            approved=True,
            resolution_note="Evidence was incorrect"
        )
        
        assert result is True
        # Re-fetch to get updated status from database
        updated_slash = await staking_manager.get_slash_history("user-123")
        assert updated_slash[0].appeal_status == "approved"
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_stake.amount == Decimal('5000')  # Refunded
        mock_ftns_service.mint_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_resolve_appeal_rejected(self, staking_manager, mock_ftns_service):
        """Test resolving an appeal (rejected)"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={"description": "Test"},
            slashed_by="system"
        )
        
        await staking_manager.appeal_slash(
            user_id="user-123",
            slash_id=slash.slash_id,
            appeal_evidence={"description": "Appeal"}
        )
        
        # Reject appeal
        result = await staking_manager.resolve_appeal(
            slash_id=slash.slash_id,
            approved=False,
            resolution_note="Evidence is valid"
        )
        
        assert result is True
        # Re-fetch to get updated status from database
        updated_slash = await staking_manager.get_slash_history("user-123")
        assert updated_slash[0].appeal_status == "rejected"
        updated_stake = await staking_manager.get_stake(stake.stake_id)
        assert updated_stake.amount == Decimal('4500')  # Not refunded


# === Reward Tests ===

class TestRewards:
    """Tests for reward calculations"""
    
    @pytest.mark.asyncio
    async def test_calculate_rewards_yields_nothing(self, staking_manager, mock_ftns_service):
        """Sprint 904 — staking yields eliminated: calculate_rewards
        accrues NOTHING even for a well-aged active stake."""
        # Create stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )

        # Age the stake well past the minimum reward age.
        from prsm.core.database import get_async_session, StakeModel
        from sqlalchemy import update
        from uuid import UUID
        async with get_async_session() as session:
            past_staked = datetime.now(timezone.utc) - timedelta(hours=2)
            past_reward = datetime.now(timezone.utc) - timedelta(hours=1)
            await session.execute(
                update(StakeModel)
                .where(StakeModel.stake_id == UUID(stake.stake_id))
                .values(staked_at=past_staked, last_reward_calculation=past_reward)
            )
            await session.commit()

        calculations = await staking_manager.calculate_rewards("user-123")
        # No token yield accrues — staking confers utility, not yield.
        assert calculations == []
    
    @pytest.mark.asyncio
    async def test_calculate_rewards_too_new(self, staking_manager, mock_ftns_service):
        """Test calculating rewards for new stake"""
        # Create stake (too new)
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )
        
        # Try to calculate rewards immediately
        calculations = await staking_manager.calculate_rewards("user-123")
        
        # Should be empty because stake is too new
        assert len(calculations) == 0
    
    @pytest.mark.asyncio
    async def test_claim_rewards(self, staking_manager, mock_ftns_service):
        """Test claiming rewards"""
        # Create stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )
        
        # Update stake timestamps in database
        from prsm.core.database import get_async_session, StakeModel
        from sqlalchemy import update
        from uuid import UUID
        async with get_async_session() as session:
            past_staked = datetime.now(timezone.utc) - timedelta(hours=2)
            past_reward = datetime.now(timezone.utc) - timedelta(hours=1)
            await session.execute(
                update(StakeModel)
                .where(StakeModel.stake_id == UUID(stake.stake_id))
                .values(staked_at=past_staked, last_reward_calculation=past_reward)
            )
            await session.commit()
        
        # Claim rewards — sp904: staking accrues no yield, so this is an
        # inert no-op that returns 0 and MINTS NOTHING (the Howey-flag
        # inflationary mint is removed).
        total_rewards = await staking_manager.claim_rewards("user-123")

        assert total_rewards == Decimal('0')
        mock_ftns_service.mint_tokens.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_reward_compounding(self, async_db_session, mock_db_session, mock_ftns_service):
        """Test reward compounding"""
        config = StakingConfig(
            minimum_stake=1000,
            reward_compounding=True,  # Enable compounding
            min_stake_age_for_rewards_seconds=60  # 1 minute for testing
        )
        manager = StakingManager(
            db_session=mock_db_session,  # Required by constructor but not used (get_async_session is patched)
            ftns_service=mock_ftns_service,
            config=config
        )
        
        # Create stake
        stake = await manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )
        
        # Update stake timestamps in database
        from prsm.core.database import get_async_session, StakeModel
        from sqlalchemy import update
        from uuid import UUID
        async with get_async_session() as session:
            past_staked = datetime.now(timezone.utc) - timedelta(hours=2)
            past_reward = datetime.now(timezone.utc) - timedelta(hours=1)
            await session.execute(
                update(StakeModel)
                .where(StakeModel.stake_id == UUID(stake.stake_id))
                .values(staked_at=past_staked, last_reward_calculation=past_reward)
            )
            await session.commit()
        
        # Claim rewards — sp904: no yield accrues, so compounding is a
        # no-op. The stake principal is unchanged.
        total_rewards = await manager.claim_rewards("user-123")

        assert total_rewards == Decimal('0')
        updated_stake = await manager.get_stake(stake.stake_id)
        assert updated_stake.amount == Decimal('10000')  # unchanged — no yield to compound


# === Query Tests ===

class TestQueries:
    """Tests for query operations"""
    
    @pytest.mark.asyncio
    async def test_get_user_stakes(self, staking_manager, mock_ftns_service):
        """Test getting user stakes"""
        # Create multiple stakes
        await staking_manager.stake(user_id="user-123", amount=Decimal('5000'))
        await staking_manager.stake(user_id="user-123", amount=Decimal('3000'))
        await staking_manager.stake(user_id="user-456", amount=Decimal('2000'))
        
        # Get user-123's stakes
        stakes = await staking_manager.get_user_stakes("user-123")
        
        assert len(stakes) == 2
        for stake in stakes:
            assert stake.user_id == "user-123"
    
    @pytest.mark.asyncio
    async def test_get_user_total_stake(self, staking_manager, mock_ftns_service):
        """Test getting user total stake"""
        await staking_manager.stake(user_id="user-123", amount=Decimal('5000'))
        await staking_manager.stake(user_id="user-123", amount=Decimal('3000'))
        
        total = await staking_manager.get_user_total_stake("user-123")
        
        assert total == Decimal('8000')
    
    @pytest.mark.asyncio
    async def test_get_network_total_stake(self, staking_manager, mock_ftns_service):
        """Test getting network total stake"""
        await staking_manager.stake(user_id="user-1", amount=Decimal('5000'))
        await staking_manager.stake(user_id="user-2", amount=Decimal('3000'))
        await staking_manager.stake(user_id="user-3", amount=Decimal('2000'))
        
        total = await staking_manager.get_network_total_stake()
        
        assert total == Decimal('10000')
    
    @pytest.mark.asyncio
    async def test_get_pending_unstake_requests(self, staking_manager, mock_ftns_service):
        """Test getting pending unstake requests"""
        # Create and unstake
        stake1 = await staking_manager.stake(user_id="user-1", amount=Decimal('5000'))
        stake2 = await staking_manager.stake(user_id="user-2", amount=Decimal('3000'))
        
        await staking_manager.unstake(user_id="user-1", stake_id=stake1.stake_id)
        await staking_manager.unstake(user_id="user-2", stake_id=stake2.stake_id)
        
        # Get pending requests
        requests = await staking_manager.get_pending_unstake_requests()
        
        assert len(requests) == 2
    
    @pytest.mark.asyncio
    async def test_get_slash_history(self, staking_manager, mock_ftns_service):
        """Test getting slash history"""
        # Create stakes and slash them
        stake1 = await staking_manager.stake(user_id="user-1", amount=Decimal('5000'))
        stake2 = await staking_manager.stake(user_id="user-2", amount=Decimal('3000'))
        
        await staking_manager.slash(
            user_id="user-1",
            stake_id=stake1.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={},
            slashed_by="system"
        )
        await staking_manager.slash(
            user_id="user-2",
            stake_id=stake2.stake_id,
            reason=SlashReason.DOWNTIME,
            evidence={},
            slashed_by="system"
        )
        
        # Get history
        history = await staking_manager.get_slash_history()
        
        assert len(history) == 2
        
        # Get user-specific history
        user_history = await staking_manager.get_slash_history(user_id="user-1")
        assert len(user_history) == 1


# === Rate Limiting Tests ===

class TestRateLimiting:
    """Tests for rate limiting"""
    
    @pytest.mark.asyncio
    async def test_stake_rate_limit(self, staking_manager, mock_ftns_service):
        """Test stake rate limiting"""
        mock_ftns_service.get_available_balance = AsyncMock(
            return_value=Decimal('100000000')
        )
        
        # Make max stakes (5 per day)
        for i in range(5):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('1000')
            )
        
        # Next should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('1000')
            )
    
    @pytest.mark.asyncio
    async def test_unstake_rate_limit(self, staking_manager, mock_ftns_service):
        """Test unstake rate limiting"""
        # Create stakes
        stakes = []
        for i in range(5):
            stake = await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('1000')
            )
            stakes.append(stake)
        
        # Make max unstakes (3 per day)
        for i in range(3):
            await staking_manager.unstake(
                user_id="user-123",
                stake_id=stakes[i].stake_id
            )
        
        # Next should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await staking_manager.unstake(
                user_id="user-123",
                stake_id=stakes[3].stake_id
            )


# === Maintenance Tests ===

class TestMaintenance:
    """Tests for maintenance operations"""
    
    @pytest.mark.asyncio
    async def test_process_matured_unstakes(self, staking_manager, mock_ftns_service):
        """Test processing matured unstakes"""
        # Create and unstake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        # Update request in database to be available
        from prsm.core.database import get_async_session, UnstakeRequestModel
        from sqlalchemy import update
        from uuid import UUID
        async with get_async_session() as session:
            past_time = datetime.now(timezone.utc) - timedelta(minutes=1)
            await session.execute(
                update(UnstakeRequestModel)
                .where(UnstakeRequestModel.request_id == UUID(request.request_id))
                .values(available_at=past_time)
            )
            await session.commit()
        
        # Process matured
        matured = await staking_manager.process_matured_unstakes()
        
        assert len(matured) == 1
        assert matured[0].request_id == request.request_id
        assert matured[0].status == UnstakeRequestStatus.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_get_staking_stats(self, staking_manager, mock_ftns_service):
        """Test getting staking statistics"""
        # Create some stakes
        await staking_manager.stake(user_id="user-1", amount=Decimal('5000'))
        await staking_manager.stake(user_id="user-2", amount=Decimal('3000'))
        
        stats = await staking_manager.get_staking_stats()
        
        assert stats["total_stakes"] == 2
        assert stats["active_stakes"] == 2
        assert stats["total_staked"] == 8000.0
        assert stats["unique_stakers"] == 2
        assert "config" in stats


# === Factory Function Tests ===

class TestFactoryFunction:
    """Tests for get_staking_manager factory"""
    
    @pytest.mark.asyncio
    async def test_get_staking_manager_singleton(self, async_db_session, mock_ftns_service):
        """Test that get_staking_manager returns singleton"""
        manager1 = await get_staking_manager(
            ftns_service=mock_ftns_service
        )
        manager2 = await get_staking_manager(
            ftns_service=mock_ftns_service
        )
        
        assert manager1 is manager2


# === Integration Tests ===

class TestIntegration:
    """Integration tests for staking system"""
    
    @pytest.mark.asyncio
    async def test_full_staking_lifecycle(self, staking_manager, mock_ftns_service):
        """Test complete staking lifecycle"""
        # 1. Stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000'),
            stake_type=StakeType.GOVERNANCE
        )
        assert stake.status == StakeStatus.ACTIVE
        
        # 2. Check balance
        total = await staking_manager.get_user_total_stake("user-123")
        assert total == Decimal('10000')
        
        # 3. Request unstake
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id,
            amount=Decimal('5000')
        )
        assert request.status == UnstakeRequestStatus.PENDING
        # Re-fetch stake to get updated amount
        stake = await staking_manager.get_stake(stake.stake_id)
        assert stake.amount == Decimal('5000')  # Partial unstake
        
        # 4. Cancel unstake
        await staking_manager.cancel_unstake(
            user_id="user-123",
            request_id=request.request_id
        )
        stake = await staking_manager.get_stake(stake.stake_id)
        assert stake.amount == Decimal('10000')  # Restored
        
        # 5. Full unstake
        request2 = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        stake = await staking_manager.get_stake(stake.stake_id)
        assert stake.status == StakeStatus.UNSTAKING
        
        # 6. Make available and withdraw
        from prsm.core.database import get_async_session, UnstakeRequestModel
        from sqlalchemy import update
        from uuid import UUID
        async with get_async_session() as session:
            past_time = datetime.now(timezone.utc) - timedelta(minutes=1)
            await session.execute(
                update(UnstakeRequestModel)
                .where(UnstakeRequestModel.request_id == UUID(request2.request_id))
                .values(available_at=past_time, status=UnstakeRequestStatus.AVAILABLE.value)
            )
            await session.commit()
        
        success, amount = await staking_manager.withdraw(
            user_id="user-123",
            request_id=request2.request_id
        )
        assert success is True
        assert amount == Decimal('10000')
    
    @pytest.mark.asyncio
    async def test_stake_slash_and_appeal(self, staking_manager, mock_ftns_service):
        """Test stake, slash, and appeal flow"""
        # 1. Stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )
        
        # 2. Slash
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.VALIDATION_FAILURE,
            evidence={"description": "Failed validation"},
            slashed_by="validator"
        )
        # Re-fetch stake to get updated amount
        stake = await staking_manager.get_stake(stake.stake_id)
        assert stake.amount == Decimal('9000')  # 10% slashed
        
        # 3. Appeal
        await staking_manager.appeal_slash(
            user_id="user-123",
            slash_id=slash.slash_id,
            appeal_evidence={"description": "False positive"}
        )
        
        # 4. Resolve (approved)
        await staking_manager.resolve_appeal(
            slash_id=slash.slash_id,
            approved=True,
            resolution_note="Appeal approved"
        )
        
        # Re-fetch stake to get restored amount
        stake = await staking_manager.get_stake(stake.stake_id)
        assert stake.amount == Decimal('10000')


# === Sprint 906 — staking utility benefits (lock discounts + priority) ===

class TestStakingBenefits:
    """Lock-based discounts + priority access (PRSM_Tokenomics.md §5.3)."""

    @pytest.mark.asyncio
    async def test_stake_with_lock_records_benefit_tier(
        self, staking_manager, mock_ftns_service
    ):
        await staking_manager.stake(
            user_id="staker-90",
            amount=Decimal('5000'),
            lock_period_days=90,
        )
        benefits = await staking_manager.get_user_benefits("staker-90")
        assert benefits.tier_label == "90d"
        assert benefits.discount_fraction == 0.05
        assert benefits.priority_boost == 0.25
        assert benefits.is_active is True

    @pytest.mark.asyncio
    async def test_stake_rejects_invalid_lock_period(self, staking_manager):
        with pytest.raises(ValueError, match="lock_period_days must be one of"):
            await staking_manager.stake(
                user_id="staker-x",
                amount=Decimal('5000'),
                lock_period_days=45,  # not a valid tier threshold
            )

    @pytest.mark.asyncio
    async def test_no_lock_confers_no_benefits(
        self, staking_manager, mock_ftns_service
    ):
        await staking_manager.stake(
            user_id="staker-none",
            amount=Decimal('5000'),
        )
        benefits = await staking_manager.get_user_benefits("staker-none")
        assert benefits.is_active is False
        assert benefits.tier_label == "none"

    @pytest.mark.asyncio
    async def test_unstake_blocked_while_locked(
        self, staking_manager, mock_ftns_service
    ):
        stake = await staking_manager.stake(
            user_id="staker-locked",
            amount=Decimal('5000'),
            lock_period_days=365,
        )
        with pytest.raises(ValueError, match="locked until"):
            await staking_manager.unstake(
                user_id="staker-locked",
                stake_id=stake.stake_id,
            )

    @pytest.mark.asyncio
    async def test_unstake_allowed_and_benefit_lapses_after_lock_expires(
        self, staking_manager, mock_ftns_service, async_db_session
    ):
        from prsm.core.database import StakeModel
        from sqlalchemy import select as _select
        from uuid import UUID as _UUID

        stake = await staking_manager.stake(
            user_id="staker-expire",
            amount=Decimal('5000'),
            lock_period_days=30,
        )
        # Fast-forward: rewrite the lock expiry to the past.
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        async with async_db_session() as db:
            row = (await db.execute(
                _select(StakeModel).where(
                    StakeModel.stake_id == _UUID(stake.stake_id)
                )
            )).scalar_one()
            md = dict(row.stake_metadata or {})
            md["lock_expires_at"] = past
            row.stake_metadata = md
            await db.commit()

        # Benefit has lapsed (lock expired) ...
        benefits = await staking_manager.get_user_benefits("staker-expire")
        assert benefits.is_active is False
        # ... and unstaking is now permitted.
        req = await staking_manager.unstake(
            user_id="staker-expire",
            stake_id=stake.stake_id,
        )
        assert req is not None

    @pytest.mark.asyncio
    async def test_best_tier_wins_across_multiple_locks(
        self, staking_manager, mock_ftns_service
    ):
        await staking_manager.stake(
            user_id="staker-multi", amount=Decimal('2000'), lock_period_days=30,
        )
        await staking_manager.stake(
            user_id="staker-multi", amount=Decimal('2000'), lock_period_days=365,
        )
        benefits = await staking_manager.get_user_benefits("staker-multi")
        # Highest tier (365d → 10%) wins.
        assert benefits.tier_label == "365d"
        assert benefits.discount_fraction == 0.10

    @pytest.mark.asyncio
    async def test_below_minimum_stake_confers_no_benefit(
        self, staking_manager, mock_ftns_service
    ):
        # minimum_stake is 1000 in the test config; a 1500 stake qualifies,
        # but verify the guard path by checking a sub-minimum stake can't
        # even be created (so it can never confer a benefit).
        with pytest.raises(ValueError, match="below minimum"):
            await staking_manager.stake(
                user_id="staker-tiny", amount=Decimal('500'), lock_period_days=90,
            )

# === Sprint 933 — concurrency / double-spend tests ===

class TestStakingConcurrency:
    """sp933 — staking mutations did check-then-act with NO per-entity lock and
    a money side-effect (unlock/burn/mint) AFTER commit, so two concurrent (or
    replayed) calls on the same entity both pass the stale-read status check and
    both apply the side-effect → double-withdraw / double-burn / double-mint.
    A per-entity lock must serialize them (same sp929 class)."""

    @pytest.mark.asyncio
    async def test_concurrent_withdraw_unlocks_tokens_once(self, staking_manager, mock_ftns_service):
        import asyncio
        from prsm.core.database import get_async_session, UnstakeRequestModel
        from sqlalchemy import update
        from uuid import UUID

        stake = await staking_manager.stake(user_id="u1", amount=Decimal('5000'))
        request = await staking_manager.unstake(user_id="u1", stake_id=stake.stake_id)
        async with get_async_session() as session:
            await session.execute(
                update(UnstakeRequestModel)
                .where(UnstakeRequestModel.request_id == UUID(request.request_id))
                .values(available_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                        status=UnstakeRequestStatus.AVAILABLE.value)
            )
            await session.commit()

        results = await asyncio.gather(
            staking_manager.withdraw(user_id="u1", request_id=request.request_id),
            staking_manager.withdraw(user_id="u1", request_id=request.request_id),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 1, f"exactly one withdraw should succeed, got {results}"
        mock_ftns_service.unlock_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_slash_burn_matches_stake_decrement(self, staking_manager, mock_ftns_service):
        import asyncio
        stake = await staking_manager.stake(user_id="u2", amount=Decimal('5000'))
        await asyncio.gather(
            staking_manager.slash(user_id="u2", stake_id=stake.stake_id,
                                  reason=SlashReason.MISCONDUCT, evidence={"d": "x"}),
            staking_manager.slash(user_id="u2", stake_id=stake.stake_id,
                                  reason=SlashReason.MISCONDUCT, evidence={"d": "y"}),
            return_exceptions=True,
        )
        final = await staking_manager.get_stake(stake.stake_id)
        burned = sum(Decimal(str(c.args[1])) for c in mock_ftns_service.burn_tokens.call_args_list)
        decrement = Decimal('5000') - Decimal(str(final.amount))
        assert abs(burned - decrement) < Decimal('0.0001'), (
            f"burned {burned} != stake decrement {decrement} (double-burn inconsistency)"
        )

    @pytest.mark.asyncio
    async def test_concurrent_resolve_appeal_mints_once(self, staking_manager, mock_ftns_service):
        import asyncio
        from prsm.core.database import get_async_session, SlashEventModel
        from sqlalchemy import update
        from uuid import UUID

        stake = await staking_manager.stake(user_id="u3", amount=Decimal('5000'))
        slash = await staking_manager.slash(user_id="u3", stake_id=stake.stake_id,
                                            reason=SlashReason.MISCONDUCT, evidence={"d": "x"})
        async with get_async_session() as session:
            await session.execute(
                update(SlashEventModel)
                .where(SlashEventModel.slash_id == UUID(slash.slash_id))
                .values(appeal_status="pending")
            )
            await session.commit()

        results = await asyncio.gather(
            staking_manager.resolve_appeal(slash_id=slash.slash_id, approved=True, resolution_note="ok"),
            staking_manager.resolve_appeal(slash_id=slash.slash_id, approved=True, resolution_note="ok"),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 1, f"exactly one appeal resolution should succeed, got {results}"
        mock_ftns_service.mint_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_appeal_serializes_against_stake_mutations(self, staking_manager, mock_ftns_service):
        """sp945 — CROSS-ENTITY race. resolve_appeal's approved path refunds
        stake_row.amount and flips SLASHED→ACTIVE; unstake / slash / withdraw
        mutate the SAME stake_row.amount/status. sp933 locked resolve_appeal on
        ``slash:{slash_id}`` while those lock ``stake:{stake_id}`` — different
        keys, so an appeal resolution could interleave between an unstake's read
        and commit, overwriting the decrement with a stale-base + refund value
        (lost update) and flipping an UNSTAKING/SLASHED stake back to ACTIVE
        while its unstake request still stands (double-withdraw path).

        resolve_appeal must serialize behind the SAME stake lock. Deterministic
        check: hold ``stake:{stake_id}`` exactly as unstake/slash/withdraw do,
        then resolve_appeal MUST block until it frees (pre-fix it only takes the
        slash lock, so it runs straight through)."""
        import asyncio
        from prsm.core.database import get_async_session, SlashEventModel
        from sqlalchemy import update
        from uuid import UUID

        stake = await staking_manager.stake(user_id="u4", amount=Decimal('5000'))
        slash = await staking_manager.slash(user_id="u4", stake_id=stake.stake_id,
                                            reason=SlashReason.MISCONDUCT, evidence={"d": "x"})
        async with get_async_session() as session:
            await session.execute(
                update(SlashEventModel)
                .where(SlashEventModel.slash_id == UUID(slash.slash_id))
                .values(appeal_status="pending")
            )
            await session.commit()

        # Natural runtime of resolve_appeal in this harness is ~2ms; we poll with
        # many short yields (a single long sleep does not pump the aiosqlite-backed
        # awaits) so an UNblocked resolve_appeal would finish well within the window.
        async def _poll_until_done(task, max_iters=50):
            for _ in range(max_iters):
                await asyncio.sleep(0.01)
                if task.done():
                    return True
            return task.done()

        # A stake-level op (unstake/slash/withdraw) is mid-critical-section.
        stake_lock = staking_manager._lock_for(f"stake:{stake.stake_id}")
        await stake_lock.acquire()
        try:
            task = asyncio.create_task(
                staking_manager.resolve_appeal(
                    slash_id=slash.slash_id, approved=True, resolution_note="ok"
                )
            )
            # ~0.5s of polling >> the ~2ms it takes when NOT serialized.
            done = await _poll_until_done(task)
            assert not done, (
                "resolve_appeal completed while a stake:{id} lock was held — it is "
                "NOT mutually exclusive with unstake/slash/withdraw on the same "
                "stake (cross-entity lost-update / orphaned-unstake race)"
            )
        finally:
            stake_lock.release()
        # Lock freed → resolve_appeal proceeds, refunds once.
        result = await asyncio.wait_for(task, timeout=2.0)
        assert result is True
        mock_ftns_service.mint_tokens.assert_called_once()
