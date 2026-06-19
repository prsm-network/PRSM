"""
Atomic FTNS Service
===================

Production-grade FTNS service with atomic operations to prevent double-spend attacks.

This service wraps database operations with proper locking, idempotency, and
optimistic concurrency control to ensure financial integrity.

Addresses Critical Audit Finding:
- Double-spend vulnerability (TOCTOU race condition)
- Lack of idempotency for transaction requests
- Missing optimistic concurrency control

Security Features:
1. SELECT FOR UPDATE row-level locking
2. Optimistic concurrency control via version columns
3. Idempotency keys to prevent duplicate operations
4. Atomic database transactions
5. Automatic rollback on failures
"""

# =============================================================================
# DEPRECATION NOTICE
# =============================================================================
# This module provides atomic transaction handling. The core ftns_service
# now handles atomicity internally. This module is kept for specialized
# use cases requiring explicit transaction atomicity.
# =============================================================================


import structlog
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any
from uuid import uuid4
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from prsm.core.config import get_settings
from prsm.core.database import get_async_session

# Set precision for financial calculations
getcontext().prec = 28

logger = structlog.get_logger(__name__)
settings = get_settings()

# SQLite does not support SELECT FOR UPDATE / ANY() — skip those clauses when running
# on a sqlite:// connection (development / CI).  Wrapped in a helper to avoid
# module-load failures in PyTorch DataLoader worker subprocesses.
def _is_sqlite() -> bool:
    try:
        return get_settings().database_url.startswith("sqlite")
    except Exception:
        return True  # Default to SQLite-safe mode on any error

_SQLITE = _is_sqlite()


class AtomicOperationError(Exception):
    """Base exception for atomic operation failures"""
    pass


class InsufficientBalanceError(AtomicOperationError):
    """Raised when user has insufficient balance"""
    pass


class ConcurrentModificationError(AtomicOperationError):
    """Raised when concurrent modification is detected"""
    pass


class IdempotencyViolationError(AtomicOperationError):
    """Raised when duplicate operation is attempted"""
    pass


class AccountNotFoundError(AtomicOperationError):
    """Raised when account does not exist"""
    pass


@dataclass
class TransactionResult:
    """Result of an atomic transaction"""
    success: bool
    transaction_id: Optional[str] = None
    new_balance: Optional[Decimal] = None
    error_message: Optional[str] = None
    idempotent_replay: bool = False  # True if this was a duplicate request


@dataclass
class BalanceInfo:
    """User balance information"""
    user_id: str
    balance: Decimal
    locked_balance: Decimal
    available_balance: Decimal
    total_earned: Decimal
    total_spent: Decimal
    version: int
    last_updated: datetime


class AtomicFTNSService:
    """
    Atomic FTNS token service with double-spend prevention.

    This service provides thread-safe, atomic operations for FTNS tokens
    using PostgreSQL row-level locking and optimistic concurrency control.
    """

    def __init__(self, database_service=None):
        """
        Initialize atomic FTNS service.

        Args:
            database_service: Database service for session management.
                            If None, will be initialized lazily.
        """
        self._db_service = database_service
        self._initialized = False

        # System accounts
        self.SYSTEM_MINT = "system_mint"
        self.SYSTEM_BURN = "system_burn"
        self.SYSTEM_REWARDS = "system_rewards"
        self.SYSTEM_FEES = "system_fees"

        # Configuration
        self.idempotency_ttl = timedelta(hours=24)
        self.max_retries = 3
        self.retry_delay_ms = 100

    async def initialize(self):
        """Initialize the service."""
        if self._initialized:
            return
        self._initialized = True
        # When using the module-level engine (no injected db_service), ensure all
        # ORM-defined tables exist.  This is idempotent — SQLAlchemy only creates
        # tables that are missing, so it is safe to call on every startup.
        if self._db_service is None:
            try:
                from prsm.core.database import get_async_engine, Base
                engine = await get_async_engine()
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
            except Exception as e:
                logger.warning("AtomicFTNSService could not auto-create tables", error=str(e))
        logger.info("AtomicFTNSService initialized")

    async def _get_session(self):
        """Get database session context manager."""
        if not self._initialized:
            await self.initialize()
        if self._db_service is not None:
            return self._db_service.get_session()
        return get_async_session()

    # =========================================================================
    # Core Atomic Operations
    # =========================================================================

    async def get_balance(self, user_id: str) -> BalanceInfo:
        """
        Get current balance for a user.

        Args:
            user_id: User identifier

        Returns:
            BalanceInfo with current balance state
        """
        async with await self._get_session() as session:
            query = text("""
                SELECT
                    user_id, balance, locked_balance,
                    total_earned, total_spent, version, updated_at
                FROM ftns_balances
                WHERE user_id = :user_id
            """)

            result = await session.execute(query, {"user_id": user_id})
            row = result.fetchone()

            if not row:
                # Auto-create account with zero balance
                await self.ensure_account_exists(user_id)
                return BalanceInfo(
                    user_id=user_id,
                    balance=Decimal("0"),
                    locked_balance=Decimal("0"),
                    available_balance=Decimal("0"),
                    total_earned=Decimal("0"),
                    total_spent=Decimal("0"),
                    version=1,
                    last_updated=datetime.now(timezone.utc)
                )

            balance = Decimal(str(row.balance))
            locked = Decimal(str(row.locked_balance))

            return BalanceInfo(
                user_id=user_id,
                balance=balance,
                locked_balance=locked,
                available_balance=balance - locked,
                total_earned=Decimal(str(row.total_earned)),
                total_spent=Decimal(str(row.total_spent)),
                version=row.version,
                last_updated=row.updated_at
            )

    async def ensure_account_exists(
        self,
        user_id: str,
        initial_balance: Decimal = Decimal("0"),
        account_type: str = "user"
    ) -> bool:
        """
        Ensure user account exists, creating if necessary.

        Args:
            user_id: User identifier
            initial_balance: Initial balance (for system accounts)
            account_type: "user" or "system"

        Returns:
            True if account exists or was created
        """
        async with await self._get_session() as session:
            try:
                # Use INSERT ... ON CONFLICT for atomic upsert
                # Note: account_type is not in the ORM model; excluded here.
                query = text("""
                    INSERT INTO ftns_balances
                    (user_id, balance, locked_balance, total_earned, total_spent,
                     version, created_at, updated_at)
                    VALUES
                    (:user_id, :balance, 0, :total_earned, 0,
                     1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) DO NOTHING
                    RETURNING user_id
                """)

                result = await session.execute(query, {
                    "user_id": user_id,
                    "balance": str(initial_balance),
                    "total_earned": str(initial_balance),
                })

                await session.commit()

                created = result.fetchone() is not None
                if created:
                    logger.info(f"Created FTNS account for {user_id}")

                return True

            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to ensure account exists: {e}")
                return False

    async def deduct_tokens_atomic(
        self,
        user_id: str,
        amount: Decimal,
        idempotency_key: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TransactionResult:
        """
        Atomically deduct tokens with double-spend prevention.

        This method uses:
        1. Idempotency key to prevent duplicate operations
        2. SELECT FOR UPDATE for row-level locking
        3. Optimistic concurrency control via version column

        Args:
            user_id: User to deduct from
            amount: Amount to deduct
            idempotency_key: Unique key for this operation
            description: Transaction description
            metadata: Additional metadata

        Returns:
            TransactionResult with success status and details

        Raises:
            InsufficientBalanceError: If balance is too low
            ConcurrentModificationError: If concurrent modification detected
            IdempotencyViolationError: If duplicate operation attempted
        """
        if amount <= 0:
            return TransactionResult(
                success=False,
                error_message="Amount must be positive"
            )

        async with await self._get_session() as session:
            try:
                # Step 1: Check idempotency
                idempotency_result = await self._check_idempotency(
                    session, idempotency_key
                )

                if idempotency_result is not None:
                    # Duplicate request - return cached result
                    return TransactionResult(
                        success=True,
                        transaction_id=idempotency_result,
                        idempotent_replay=True
                    )

                # Step 2: Acquire row lock and get current balance
                # SQLite has no row-level locking; rely on OCC version check instead.
                if _SQLITE:
                    balance_query = text("""
                        SELECT balance, locked_balance, version
                        FROM ftns_balances
                        WHERE user_id = :user_id
                    """)
                else:
                    balance_query = text("""
                        SELECT balance, locked_balance, version
                        FROM ftns_balances
                        WHERE user_id = :user_id
                        FOR UPDATE NOWAIT
                    """)

                try:
                    result = await session.execute(
                        balance_query, {"user_id": user_id}
                    )
                    row = result.fetchone()
                except Exception as lock_error:
                    # Could not acquire lock - concurrent operation in progress
                    logger.warning(f"Lock acquisition failed for {user_id}: {lock_error}")
                    return TransactionResult(
                        success=False,
                        error_message="Account is locked by another operation. Please retry."
                    )

                if not row:
                    return TransactionResult(
                        success=False,
                        error_message=f"Account not found: {user_id}"
                    )

                current_balance = Decimal(str(row.balance))
                locked_balance = Decimal(str(row.locked_balance))
                current_version = row.version

                # Step 3: Validate available balance
                available = current_balance - locked_balance
                if available < amount:
                    return TransactionResult(
                        success=False,
                        error_message=f"Insufficient balance: {available} < {amount}"
                    )

                # Step 4: Generate transaction ID
                transaction_id = f"ftns_{uuid4().hex[:12]}"
                new_balance = current_balance - amount
                new_version = current_version + 1

                # Step 5: Update balance with OCC check
                update_query = text("""
                    UPDATE ftns_balances
                    SET
                        balance = :new_balance,
                        total_spent = total_spent + :amount,
                        version = :new_version,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id
                    AND version = :current_version
                """)

                update_result = await session.execute(update_query, {
                    "new_balance": str(new_balance),
                    "amount": str(amount),
                    "new_version": new_version,
                    "user_id": user_id,
                    "current_version": current_version
                })

                if update_result.rowcount == 0:
                    # Concurrent modification detected
                    await session.rollback()
                    raise ConcurrentModificationError(
                        "Balance was modified by another transaction"
                    )

                # Step 6: Create transaction record
                tx_query = text("""
                    INSERT INTO ftns_transactions
                    (transaction_id, from_user, to_user, amount, transaction_type,
                     description, status, idempotency_key,
                     balance_before_sender, balance_after_sender,
                     created_at)
                    VALUES
                    (:transaction_id, :from_user, :to_user, :amount, 'deduction',
                     :description, 'completed', :idempotency_key,
                     :balance_before, :balance_after,
                     CURRENT_TIMESTAMP)
                """)

                await session.execute(tx_query, {
                    "transaction_id": transaction_id,
                    "from_user": user_id,
                    "to_user": self.SYSTEM_FEES,  # Required NOT NULL — fees go to system
                    "amount": str(amount),
                    "description": description,
                    "idempotency_key": idempotency_key,
                    "balance_before": str(current_balance),
                    "balance_after": str(new_balance),
                })

                # Step 7: Record idempotency key
                await self._record_idempotency(
                    session, idempotency_key, transaction_id,
                    user_id, "deduction", amount
                )

                # Step 8: Commit
                await session.commit()

                logger.info(
                    "Atomic deduction completed",
                    user_id=user_id,
                    amount=float(amount),
                    new_balance=float(new_balance),
                    transaction_id=transaction_id
                )

                return TransactionResult(
                    success=True,
                    transaction_id=transaction_id,
                    new_balance=new_balance
                )

            except ConcurrentModificationError:
                await session.rollback()
                raise
            except Exception as e:
                await session.rollback()
                logger.error(f"Atomic deduction failed: {e}")
                return TransactionResult(
                    success=False,
                    error_message=str(e)
                )

    async def transfer_tokens_atomic(
        self,
        from_user_id: str,
        to_user_id: str,
        amount: Decimal,
        idempotency_key: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TransactionResult:
        """
        Atomically transfer tokens between users with double-spend prevention.

        Uses consistent lock ordering to prevent deadlocks.

        Args:
            from_user_id: Sender user ID
            to_user_id: Recipient user ID
            amount: Amount to transfer
            idempotency_key: Unique key for this operation
            description: Transaction description
            metadata: Additional metadata

        Returns:
            TransactionResult with success status
        """
        if amount <= 0:
            return TransactionResult(
                success=False,
                error_message="Amount must be positive"
            )

        if from_user_id == to_user_id:
            return TransactionResult(
                success=False,
                error_message="Cannot transfer to same account"
            )

        # Ensure both accounts exist
        await self.ensure_account_exists(from_user_id)
        await self.ensure_account_exists(to_user_id)

        async with await self._get_session() as session:
            try:
                # Check idempotency
                existing_tx = await self._check_idempotency(session, idempotency_key)
                if existing_tx:
                    return TransactionResult(
                        success=True,
                        transaction_id=existing_tx,
                        idempotent_replay=True
                    )

                # Lock accounts in consistent order to prevent deadlocks
                first_user, second_user = sorted([from_user_id, to_user_id])

                # SQLite: no FOR UPDATE and no ANY() — use OR instead.
                if _SQLITE:
                    lock_query = text("""
                        SELECT user_id, balance, locked_balance, version
                        FROM ftns_balances
                        WHERE user_id = :user1 OR user_id = :user2
                        ORDER BY user_id
                    """)
                    result = await session.execute(
                        lock_query, {"user1": first_user, "user2": second_user}
                    )
                else:
                    lock_query = text("""
                        SELECT user_id, balance, locked_balance, version
                        FROM ftns_balances
                        WHERE user_id = ANY(:user_ids)
                        ORDER BY user_id
                        FOR UPDATE
                    """)
                    result = await session.execute(
                        lock_query, {"user_ids": [first_user, second_user]}
                    )
                rows = {row.user_id: row for row in result.fetchall()}

                if from_user_id not in rows:
                    return TransactionResult(
                        success=False,
                        error_message=f"Sender account not found: {from_user_id}"
                    )

                if to_user_id not in rows:
                    return TransactionResult(
                        success=False,
                        error_message=f"Recipient account not found: {to_user_id}"
                    )

                sender = rows[from_user_id]
                receiver = rows[to_user_id]

                sender_balance = Decimal(str(sender.balance))
                sender_locked = Decimal(str(sender.locked_balance))
                sender_available = sender_balance - sender_locked

                if sender_available < amount:
                    return TransactionResult(
                        success=False,
                        error_message=f"Insufficient balance: {sender_available} < {amount}"
                    )

                transaction_id = f"ftns_{uuid4().hex[:12]}"

                # Update sender. sp1173 OCC GUARD (money-safety): the optimistic-concurrency
                # WHERE version=:version matches 0 rows iff a concurrent committed write staled
                # our captured version (reachable on SQLite/non-Postgres, where the locked SELECT
                # degrades to a plain SELECT). Mirror deduct_tokens_atomic: rowcount==0 ->
                # rollback + raise, so we NEVER commit a partial/incorrect transfer (which would
                # destroy or duplicate value + write a false 'completed' ledger row).
                sender_result = await session.execute(text("""
                    UPDATE ftns_balances
                    SET balance = balance - :amount,
                        total_spent = total_spent + :amount,
                        version = version + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id AND version = :version
                """), {
                    "amount": str(amount),
                    "user_id": from_user_id,
                    "version": sender.version
                })
                if sender_result.rowcount == 0:
                    await session.rollback()
                    raise ConcurrentModificationError(
                        "Sender balance was modified by another transaction "
                        "(transfer aborted — no debit, no credit)"
                    )

                # Update receiver (same OCC guard — both UPDATEs must land or the transfer
                # rolls back atomically; a 0-row receiver UPDATE would otherwise debit the
                # sender while never crediting the receiver = value destroyed).
                receiver_result = await session.execute(text("""
                    UPDATE ftns_balances
                    SET balance = balance + :amount,
                        total_earned = total_earned + :amount,
                        version = version + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id AND version = :version
                """), {
                    "amount": str(amount),
                    "user_id": to_user_id,
                    "version": receiver.version
                })
                if receiver_result.rowcount == 0:
                    await session.rollback()
                    raise ConcurrentModificationError(
                        "Receiver balance was modified by another transaction "
                        "(transfer aborted — no debit, no credit)"
                    )

                # Record transaction
                await session.execute(text("""
                    INSERT INTO ftns_transactions
                    (transaction_id, from_user, to_user, amount, transaction_type,
                     description, status, idempotency_key,
                     balance_before_sender, balance_after_sender,
                     balance_before_receiver, balance_after_receiver,
                     created_at)
                    VALUES
                    (:transaction_id, :from_user, :to_user, :amount, 'transfer',
                     :description, 'completed', :idempotency_key,
                     :balance_before_s, :balance_after_s,
                     :balance_before_r, :balance_after_r,
                     CURRENT_TIMESTAMP)
                """), {
                    "transaction_id": transaction_id,
                    "from_user": from_user_id,
                    "to_user": to_user_id,
                    "amount": str(amount),
                    "description": description,
                    "idempotency_key": idempotency_key,
                    "balance_before_s": str(sender_balance),
                    "balance_after_s": str(sender_balance - amount),
                    "balance_before_r": str(receiver.balance),
                    "balance_after_r": str(Decimal(str(receiver.balance)) + amount),
                })

                # Record idempotency
                await self._record_idempotency(
                    session, idempotency_key, transaction_id,
                    from_user_id, "transfer", amount
                )

                await session.commit()

                logger.info(
                    "Atomic transfer completed",
                    from_user=from_user_id,
                    to_user=to_user_id,
                    amount=float(amount),
                    transaction_id=transaction_id
                )

                return TransactionResult(
                    success=True,
                    transaction_id=transaction_id,
                    new_balance=sender_balance - amount
                )

            except Exception as e:
                await session.rollback()
                logger.error(f"Atomic transfer failed: {e}")
                return TransactionResult(
                    success=False,
                    error_message=str(e)
                )

    async def mint_tokens_atomic(
        self,
        to_user_id: str,
        amount: Decimal,
        idempotency_key: str,
        description: str = "Token mint",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TransactionResult:
        """
        Atomically mint new tokens to a user.

        Args:
            to_user_id: Recipient user ID
            amount: Amount to mint
            idempotency_key: Unique key for this operation
            description: Mint description
            metadata: Additional metadata

        Returns:
            TransactionResult with success status
        """
        await self.ensure_account_exists(to_user_id)

        async with await self._get_session() as session:
            try:
                # Check idempotency
                existing_tx = await self._check_idempotency(session, idempotency_key)
                if existing_tx:
                    return TransactionResult(
                        success=True,
                        transaction_id=existing_tx,
                        idempotent_replay=True
                    )

                # Lock and update receiver (SQLite has no FOR UPDATE)
                if _SQLITE:
                    _recv_lock_sql = "SELECT balance, version FROM ftns_balances WHERE user_id = :user_id"
                else:
                    _recv_lock_sql = "SELECT balance, version FROM ftns_balances WHERE user_id = :user_id FOR UPDATE"
                result = await session.execute(text(_recv_lock_sql), {"user_id": to_user_id})

                row = result.fetchone()
                if not row:
                    return TransactionResult(
                        success=False,
                        error_message=f"Account not found: {to_user_id}"
                    )

                current_balance = Decimal(str(row.balance))
                new_balance = current_balance + amount
                transaction_id = f"ftns_{uuid4().hex[:12]}"

                await session.execute(text("""
                    UPDATE ftns_balances
                    SET balance = balance + :amount,
                        total_earned = total_earned + :amount,
                        version = version + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id AND version = :version
                """), {
                    "amount": str(amount),
                    "user_id": to_user_id,
                    "version": row.version
                })

                # Record transaction
                await session.execute(text("""
                    INSERT INTO ftns_transactions
                    (transaction_id, from_user, to_user, amount, transaction_type,
                     description, status, idempotency_key,
                     balance_before_receiver, balance_after_receiver,
                     created_at)
                    VALUES
                    (:transaction_id, :system, :to_user, :amount, 'mint',
                     :description, 'completed', :idempotency_key,
                     :balance_before, :balance_after,
                     CURRENT_TIMESTAMP)
                """), {
                    "transaction_id": transaction_id,
                    "system": self.SYSTEM_MINT,
                    "to_user": to_user_id,
                    "amount": str(amount),
                    "description": description,
                    "idempotency_key": idempotency_key,
                    "balance_before": str(current_balance),
                    "balance_after": str(new_balance),
                })

                await self._record_idempotency(
                    session, idempotency_key, transaction_id,
                    self.SYSTEM_MINT, "mint", amount
                )

                await session.commit()

                logger.info(
                    "Atomic mint completed",
                    to_user=to_user_id,
                    amount=float(amount),
                    new_balance=float(new_balance),
                    transaction_id=transaction_id
                )

                return TransactionResult(
                    success=True,
                    transaction_id=transaction_id,
                    new_balance=new_balance
                )

            except Exception as e:
                await session.rollback()
                logger.error(f"Atomic mint failed: {e}")
                return TransactionResult(
                    success=False,
                    error_message=str(e)
                )

    # =========================================================================
    # Idempotency Helpers
    # =========================================================================

    async def _check_idempotency(
        self,
        session: AsyncSession,
        idempotency_key: str
    ) -> Optional[str]:
        """
        Check if idempotency key has been used.

        Returns:
            Transaction ID if key was used, None otherwise
        """
        result = await session.execute(text("""
            SELECT transaction_id
            FROM ftns_idempotency_keys
            WHERE idempotency_key = :key
            AND expires_at > CURRENT_TIMESTAMP
        """), {"key": idempotency_key})

        row = result.fetchone()
        return row.transaction_id if row else None

    async def _record_idempotency(
        self,
        session: AsyncSession,
        idempotency_key: str,
        transaction_id: str,
        user_id: str,
        operation_type: str,
        amount: Decimal
    ):
        """Record idempotency key for duplicate detection."""
        await session.execute(text("""
            INSERT INTO ftns_idempotency_keys
            (idempotency_key, transaction_id, user_id, operation_type, amount,
             status, created_at, expires_at)
            VALUES
            (:key, :tx_id, :user_id, :op_type, :amount,
             'completed', CURRENT_TIMESTAMP, datetime('now', '+24 hours'))
            ON CONFLICT (idempotency_key) DO NOTHING
        """), {
            "key": idempotency_key,
            "tx_id": transaction_id,
            "user_id": user_id,
            "op_type": operation_type,
            "amount": str(amount)
        })

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def get_transaction_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get transaction history for a user."""
        async with await self._get_session() as session:
            result = await session.execute(text("""
                SELECT
                    id, from_user_id, to_user_id, amount,
                    transaction_type, description, status,
                    balance_before_sender, balance_after_sender,
                    balance_before_receiver, balance_after_receiver,
                    created_at
                FROM ftns_transactions
                WHERE from_user_id = :user_id OR to_user_id = :user_id
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """), {
                "user_id": user_id,
                "limit": limit,
                "offset": offset
            })

            transactions = []
            for row in result.fetchall():
                transactions.append({
                    "id": str(row.id),
                    "from_user_id": row.from_user_id,
                    "to_user_id": row.to_user_id,
                    "amount": str(row.amount),
                    "transaction_type": row.transaction_type,
                    "description": row.description,
                    "status": row.status,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "direction": "incoming" if row.to_user_id == user_id else "outgoing"
                })

            return transactions

    async def get_ledger_stats(self) -> Dict[str, Any]:
        """Get overall ledger statistics."""
        async with await self._get_session() as session:
            result = await session.execute(text("""
                SELECT
                    SUM(balance) as total_supply,
                    SUM(CASE WHEN account_type = 'user' THEN balance ELSE 0 END) as circulating,
                    SUM(locked_balance) as total_locked,
                    COUNT(*) as total_accounts
                FROM ftns_balances
            """))

            row = result.fetchone()

            return {
                "total_supply": str(row.total_supply or 0),
                "circulating_supply": str(row.circulating or 0),
                "locked_supply": str(row.total_locked or 0),
                "total_accounts": row.total_accounts or 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global instance
_atomic_ftns_service: Optional[AtomicFTNSService] = None


async def get_atomic_ftns_service() -> AtomicFTNSService:
    """Get or create global atomic FTNS service instance."""
    global _atomic_ftns_service
    if _atomic_ftns_service is None:
        _atomic_ftns_service = AtomicFTNSService()
        await _atomic_ftns_service.initialize()
    return _atomic_ftns_service


def reset_atomic_ftns_service():
    """Reset global instance (for testing)."""
    global _atomic_ftns_service
    _atomic_ftns_service = None
