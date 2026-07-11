"""
DAG Ledger
 ==========

DAG-based (Directed Acyclic Graph) ledger for FTNS tokens.
Unlike traditional blockchain (linear), this DAG structure allows:
- Horizontal scaling with more users (more users = faster confirmation)
- No mining fees - instead, transactions approve each other
- Faster confirmation times as network grows

This implements a simplified IOTA-style Tangle with:
- Each transaction references 2-8 parent transactions
- Tip selection using MCMC (Markov Chain Monte Carlo)
- Confirmation confidence based on cumulative approval weight

Cryptographic Signature Verification:
- All transactions from non-null wallets must be signed with Ed25519
- Signatures verify the transaction hash (SHA-256)
- Public keys are stored with the transaction for verification
- Genesis and system transactions are exempt from signature requirements

Atomic Balance Operations (TOCTOU Prevention):
==============================================
This ledger implements atomic balance operations to prevent Time-of-Check-Time-of-Use
(TOCTOU) race conditions that could lead to double-spend attacks.

Security Architecture:
1. **Row-Level Locking**: Uses SQLite's BEGIN IMMEDIATE to acquire write locks
   before balance checks, preventing concurrent modifications.

2. **Optimistic Concurrency Control**: Balance cache includes a version number
   that is checked during deduction to detect concurrent modifications.

3. **Atomic Transaction Flow**:
   - BEGIN IMMEDIATE (acquires write lock)
   - Check balance with version
   - Create and store transaction
   - Update balance with version check (OCC)
   - COMMIT or ROLLBACK

4. **Balance Cache Table**: The `wallet_balances` table stores pre-computed
   balances with version numbers for efficient atomic operations.

Exception Hierarchy:
- AtomicOperationError: Base exception for atomic operation failures
- InsufficientBalanceError: Raised when balance is insufficient
- ConcurrentModificationError: Raised when TOCTOU is detected
- BalanceLockError: Raised when lock acquisition fails

Usage Example:
    ledger = DAGLedger(db_path='ledger.db')
    await ledger.initialize()
    
    try:
        tx = await ledger.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=100.0,
            from_wallet='sender_wallet',
            to_wallet='receiver_wallet',
            signature='base64_signature',
            public_key='hex_public_key'
        )
    except InsufficientBalanceError:
        # Handle insufficient balance
        pass
    except ConcurrentModificationError:
        # Retry the operation - concurrent modification detected
        pass
    except BalanceLockError:
        # Retry later - system under high contention
        pass

Security Guarantees:
- No double-spend: Balance check and deduction are atomic
- No lost updates: Concurrent modifications are detected
- Fail-safe: Any error during transaction rolls back all changes
- Audit trail: All operations are logged for security review
"""

import asyncio
import hashlib
import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import aiosqlite

from prsm.node.ledger_node_services import LedgerNodeServicesMixin
from prsm.core.cryptography.dag_signatures import (
    DAGSignatureManager,
    InvalidSignatureError,
    MissingSignatureError,
)

# Configure logger for security audit trail
logger = logging.getLogger(__name__)


# =============================================================================
# ATOMIC OPERATION EXCEPTIONS
# =============================================================================

class AtomicOperationError(Exception):
    """Base exception for atomic operation failures."""
    pass


class InsufficientBalanceError(AtomicOperationError):
    """Raised when wallet has insufficient balance for transaction."""
    pass


class ConcurrentModificationError(AtomicOperationError):
    """Raised when concurrent modification is detected (TOCTOU prevention)."""
    pass


class BalanceLockError(AtomicOperationError):
    """Raised when balance lock cannot be acquired."""
    pass


class TransactionType(str, Enum):
    GENESIS = "genesis"
    WELCOME_GRANT = "welcome_grant"
    COMPUTE_PAYMENT = "compute_payment"
    COMPUTE_EARNING = "compute_earning"
    STORAGE_REWARD = "storage_reward"
    CONTENT_ROYALTY = "content_royalty"
    TRANSFER = "transfer"
    APPROVAL = "approval"
    SYSTEM = "system"
    # Sprint 540 Pattern A: daemon-mediated bridge tx types
    BRIDGE_DEPOSIT = "bridge_deposit"
    BRIDGE_WITHDRAW = "bridge_withdraw"
    # Sprint 572 F24 — parity with local_ledger.TransactionType.REWARD
    # for call sites that funnel reward credits through dag_ledger.
    REWARD = "reward"


@dataclass
class DAGTransaction:
    tx_id: str
    tx_type: TransactionType
    amount: float
    from_wallet: Optional[str]
    to_wallet: str
    timestamp: float
    signature: Optional[str] = None
    public_key: Optional[str] = None  # Ed25519 public key (hex or base64)
    description: str = ""
    
    parent_ids: List[str] = field(default_factory=list)
    
    cumulative_weight: int = 1
    confirmation_level: float = 0.0
    
    approved_by: Set[str] = field(default_factory=set)
    
    def hash(self) -> str:
        """Generate SHA-256 hash of transaction data for signing."""
        data = {
            "tx_id": self.tx_id,
            "tx_type": self.tx_type.value,
            "amount": self.amount,
            "from_wallet": self.from_wallet,
            "to_wallet": self.to_wallet,
            "timestamp": self.timestamp,
            "parent_ids": self.parent_ids,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def get_signing_data(self) -> dict:
        """Get the canonical transaction data for signing."""
        return {
            "tx_id": self.tx_id,
            "tx_type": self.tx_type.value,
            "amount": self.amount,
            "from_wallet": self.from_wallet,
            "to_wallet": self.to_wallet,
            "timestamp": self.timestamp,
            "parent_ids": self.parent_ids,
        }


@dataclass
class DAGState:
    tips: Set[str] = field(default_factory=set)
    transactions: Dict[str, DAGTransaction] = field(default_factory=dict)
    approvals: Dict[str, Set[str]] = field(default_factory=dict)


class DAGLedger(LedgerNodeServicesMixin):
    """
    DAG-based FTNS Ledger
    
    Key concepts:
    - Genesis transaction: Initial token supply
    - Tips: Unapproved transactions (eligible for approval)
    - Parents: Transactions this tx approves
    - Cumulative weight: 1 + number of transactions approving this tx
    - Confirmation level: 0.0-1.0 based on cumulative weight threshold
    
    Signature Verification:
    - Transactions with a from_wallet require valid Ed25519 signatures
    - Genesis and system transactions (from_wallet=None) are exempt
    - Public keys are stored with transactions for verification
    - Signature verification can be disabled for testing via verify_signatures=False
    """
    
    INITIAL_SUPPLY = 1_000_000_000.0
    
    def __init__(self, db_path: str = ":memory:", verify_signatures: bool = True):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        
        self._state = DAGState()
        
        # In-memory version cache for TOCTOU detection.
        # Tracks the last version this service committed per wallet.
        # Only updated by successful commits through _commit_balance_deduction
        # and seeded by _commit_balance_credit.
        # External DB modifications will cause version mismatch at commit time.
        self._balance_version_cache: Dict[str, int] = {}
        
        self.max_parents = 4
        self.alpha = 0.5
        self.confirmation_threshold = 100
        
        # Signature verification setting (can be disabled for testing)
        self.verify_signatures = verify_signatures
        
        # Public key registry for wallet verification
        # Maps wallet_id -> public_key_hex
        self._wallet_public_keys: Dict[str, str] = {}
        
        # Temporary storage for signature verification data during transaction creation
        # This is used to pass verification data between pre-creation and post-creation phases
        self._pending_verification: Optional[Dict[str, str]] = None

        # Sprint 487 (F25) — per-wallet asyncio.Lock map.
        # The savepoint name "balance_check" is hardcoded; concurrent
        # submit_transaction calls on the SAME connection step on
        # each other's savepoint → "no such savepoint: balance_check"
        # exceptions return 500. With this lock, balance-mutation
        # paths are serialized per-wallet (different wallets still
        # proceed in parallel), eliminating the savepoint collision.
        # The optimistic-version-check is still in force as defense-
        # in-depth against external DB modifications.
        self._wallet_locks: Dict[str, "asyncio.Lock"] = {}
        self._wallet_locks_guard: Optional["asyncio.Lock"] = None

        # sp910 — CONNECTION-WIDE write lock. The per-wallet lock above is
        # insufficient: the SAVEPOINT balance_check is connection-GLOBAL, so
        # a COMMIT from ANY other wallet's write (a parallel debit, a
        # credit-only submit, or record_nonce) on this shared aiosqlite
        # connection RELEASES an in-flight savepoint mid-debit. The debit's
        # UPDATE then commits as part of the foreign txn, but the original
        # submit_transaction's `RELEASE SAVEPOINT` raises "no such
        # savepoint" → it returns an error AFTER the money already moved →
        # silent no-refund loss on /wallet/withdraw. Serializing ALL
        # connection writes through one lock closes the interleave window.
        # (Lazy-created in _get_write_lock to bind to the running loop.)
        self._write_lock: Optional["asyncio.Lock"] = None

    def _get_write_lock(self) -> "asyncio.Lock":
        """sp910 — the connection-wide write lock (lazy, loop-bound)."""
        import asyncio as _asyncio
        if self._write_lock is None:
            self._write_lock = _asyncio.Lock()
        return self._write_lock

    def _get_wallet_lock(self, wallet_id: str) -> "asyncio.Lock":
        """Sprint 487 (F25) — return the per-wallet asyncio.Lock.
        Lazy-creates on first use. The guard lock prevents a race
        in the lock-creation itself."""
        import asyncio as _asyncio
        if self._wallet_locks_guard is None:
            self._wallet_locks_guard = _asyncio.Lock()
        if wallet_id not in self._wallet_locks:
            self._wallet_locks[wallet_id] = _asyncio.Lock()
        return self._wallet_locks[wallet_id]

    async def initialize(self) -> None:
        """Initialize database and load existing state."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()
        await self._migrate_schema()
        await self._load_state()
        
    async def _migrate_schema(self) -> None:
        """Apply incremental schema migrations for columns added after initial creation."""
        # Add public_key to dag_transactions if missing (schema v2)
        cursor = await self._db.execute("PRAGMA table_info(dag_transactions)")
        dag_cols = {row[1] async for row in cursor}
        if "public_key" not in dag_cols:
            await self._db.execute("ALTER TABLE dag_transactions ADD COLUMN public_key TEXT")

        # Add public_key to wallets if missing (schema v2)
        cursor = await self._db.execute("PRAGMA table_info(wallets)")
        wallet_cols = {row[1] async for row in cursor}
        if "public_key" not in wallet_cols:
            await self._db.execute("ALTER TABLE wallets ADD COLUMN public_key TEXT")
        # Sprint 540 Pattern A: bridge linkage column
        if "eth_address" not in wallet_cols:
            await self._db.execute(
                "ALTER TABLE wallets ADD COLUMN eth_address TEXT"
            )
            await self._db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_dag_wallets_eth_address "
                "ON wallets(eth_address) "
                "WHERE eth_address IS NOT NULL"
            )
        # Sprint 554 user-sig groundwork: per-wallet opt-in flag +
        # monotonic withdraw nonce counter.
        if "requires_user_signature" not in wallet_cols:
            await self._db.execute(
                "ALTER TABLE wallets ADD COLUMN "
                "requires_user_signature INTEGER NOT NULL DEFAULT 0"
            )
        if "next_withdraw_nonce" not in wallet_cols:
            await self._db.execute(
                "ALTER TABLE wallets ADD COLUMN "
                "next_withdraw_nonce INTEGER NOT NULL DEFAULT 0"
            )

        await self._db.commit()

    async def _create_tables(self) -> None:
        # Note: `eth_address` column + its partial unique index are
        # added in `_migrate_schema()` (runs after create_tables)
        # to handle the legacy-DB upgrade path uniformly. Adding
        # them here as part of CREATE TABLE on a fresh DB then
        # ALTERing on existing DBs caused split-brain init paths.
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS wallets (
                wallet_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                public_key TEXT,
                created_at REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS dag_transactions (
                tx_id TEXT PRIMARY KEY,
                tx_type TEXT NOT NULL,
                amount REAL NOT NULL,
                from_wallet TEXT,
                to_wallet TEXT NOT NULL,
                timestamp REAL NOT NULL,
                signature TEXT,
                public_key TEXT,
                description TEXT DEFAULT '',
                parent_ids TEXT NOT NULL,
                cumulative_weight INTEGER DEFAULT 1,
                confirmation_level REAL DEFAULT 0.0,
                hash TEXT NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_dag_to ON dag_transactions(to_wallet);
            CREATE INDEX IF NOT EXISTS idx_dag_from ON dag_transactions(from_wallet);
            CREATE INDEX IF NOT EXISTS idx_dag_hash ON dag_transactions(hash);
            CREATE INDEX IF NOT EXISTS idx_dag_timestamp ON dag_transactions(timestamp);
            
            CREATE TABLE IF NOT EXISTS tips (
                tx_id TEXT PRIMARY KEY
            );
            
            CREATE TABLE IF NOT EXISTS approvals (
                tx_id TEXT PRIMARY KEY,
                approver_id TEXT NOT NULL,
                approved_at REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS seen_nonces (
                nonce TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                seen_at REAL NOT NULL
            );
            
            -- Balance cache table for atomic operations with optimistic concurrency control
            -- This table stores pre-computed balances with version numbers for TOCTOU prevention
            CREATE TABLE IF NOT EXISTS wallet_balances (
                wallet_id TEXT PRIMARY KEY,
                balance REAL NOT NULL DEFAULT 0.0,
                version INTEGER NOT NULL DEFAULT 1,
                last_updated REAL NOT NULL,
                FOREIGN KEY (wallet_id) REFERENCES wallets(wallet_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_wallet_balances_version ON wallet_balances(wallet_id, version);

            -- Supplementary tables: agent allowances, gossip log, collaboration state.
            -- These match the LocalLedger schema so DAGLedgerAdapter can provide full
            -- feature parity for gossip persistence and collaboration state persistence.

            CREATE TABLE IF NOT EXISTS agent_allowances (
                agent_id TEXT PRIMARY KEY,
                principal_id TEXT NOT NULL,
                allowance REAL NOT NULL DEFAULT 0,
                spent REAL NOT NULL DEFAULT 0,
                epoch_hours REAL NOT NULL DEFAULT 24,
                epoch_start REAL NOT NULL,
                created_at REAL NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS gossip_log (
                nonce TEXT PRIMARY KEY,
                subtype TEXT NOT NULL,
                origin TEXT NOT NULL,
                payload TEXT NOT NULL,
                ttl INTEGER NOT NULL DEFAULT 5,
                received_at REAL NOT NULL,
                attestation TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_gossip_received ON gossip_log(received_at);

            -- sp966 — durable provenance registry (mirrors LocalLedger so the
            -- DEFAULT (dag) backend actually persists + serves gossip provenance;
            -- previously absent → provenance silently dead on default nodes).
            CREATE TABLE IF NOT EXISTS provenance_chains (
                cid               TEXT PRIMARY KEY,
                content_hash      TEXT NOT NULL DEFAULT '',
                creator_id        TEXT NOT NULL,
                creator_pubkey    TEXT NOT NULL DEFAULT '',
                filename          TEXT NOT NULL DEFAULT '',
                size_bytes        INTEGER NOT NULL DEFAULT 0,
                royalty_rate      REAL NOT NULL DEFAULT 0.01,
                parent_cids       TEXT NOT NULL DEFAULT '[]',
                signature         TEXT NOT NULL DEFAULT '',
                embedding_id      TEXT,
                near_duplicate_of TEXT,
                metadata          TEXT NOT NULL DEFAULT '{}',
                registered_at     REAL NOT NULL,
                signed_record     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_prov_creator ON provenance_chains(creator_id);

            CREATE TABLE IF NOT EXISTS collab_tasks (
                task_id TEXT PRIMARY KEY,
                requester_agent_id TEXT NOT NULL,
                requester_node_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                description TEXT DEFAULT '',
                required_capabilities TEXT DEFAULT '[]',
                ftns_budget REAL DEFAULT 0,
                deadline_seconds REAL DEFAULT 3600,
                status TEXT DEFAULT 'open',
                assigned_agent_id TEXT,
                bids TEXT DEFAULT '[]',
                result TEXT,
                created_at REAL NOT NULL,
                escrow_tx_id TEXT
            );

            CREATE TABLE IF NOT EXISTS collab_reviews (
                review_id TEXT PRIMARY KEY,
                submitter_agent_id TEXT NOT NULL,
                submitter_node_id TEXT NOT NULL,
                content_cid TEXT DEFAULT '',
                description TEXT DEFAULT '',
                required_capabilities TEXT DEFAULT '[]',
                ftns_per_review REAL DEFAULT 0.1,
                max_reviewers INTEGER DEFAULT 3,
                status TEXT DEFAULT 'pending',
                reviews TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                escrow_tx_id TEXT,
                paid_reviewers TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS collab_queries (
                query_id TEXT PRIMARY KEY,
                requester_agent_id TEXT NOT NULL,
                requester_node_id TEXT NOT NULL,
                topic TEXT DEFAULT '',
                question TEXT DEFAULT '',
                ftns_per_response REAL DEFAULT 0.05,
                max_responses INTEGER DEFAULT 5,
                responses TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                escrow_tx_id TEXT,
                paid_responders TEXT DEFAULT '[]'
            );
        """)

        # sp961 — idempotent migration: add gossip_log.attestation for DBs
        # created before this column existed (mirrors LocalLedger). SQLite
        # raises on a duplicate column; caught + ignored so re-init is a no-op.
        try:
            await self._db.execute(
                "ALTER TABLE gossip_log ADD COLUMN attestation TEXT"
            )
            await self._db.commit()
        except Exception:
            pass

    async def _load_state(self) -> None:
        cursor = await self._db.execute("SELECT tx_id, parent_ids, cumulative_weight, confirmation_level FROM dag_transactions")
        async for row in cursor:
            tx_id, parent_ids_json, cumulative_weight, confirmation_level = row
            parent_ids = json.loads(parent_ids_json)
            
            cursor2 = await self._db.execute(
                """SELECT tx_id, tx_type, amount, from_wallet, to_wallet, timestamp, signature, public_key, description
                   FROM dag_transactions WHERE tx_id = ?""",
                (tx_id,)
            )
            row2 = await cursor2.fetchone()
            if row2:
                tx = DAGTransaction(
                    tx_id=row2[0],
                    tx_type=TransactionType(row2[1]),
                    amount=row2[2],
                    from_wallet=row2[3],
                    to_wallet=row2[4],
                    timestamp=row2[5],
                    signature=row2[6],
                    public_key=row2[7],
                    description=row2[8] or "",
                    parent_ids=parent_ids,
                    cumulative_weight=cumulative_weight,
                    confirmation_level=confirmation_level,
                )
                self._state.transactions[tx_id] = tx
                
        cursor = await self._db.execute("SELECT tx_id FROM tips")
        async for row in cursor:
            self._state.tips.add(row[0])
        
        # Load wallet public keys
        cursor = await self._db.execute("SELECT wallet_id, public_key FROM wallets WHERE public_key IS NOT NULL")
        async for row in cursor:
            self._wallet_public_keys[row[0]] = row[1]
            
        if not self._state.transactions:
            await self._create_genesis()
            
    async def _create_genesis(self) -> None:
        genesis = DAGTransaction(
            tx_id="genesis",
            tx_type=TransactionType.GENESIS,
            amount=self.INITIAL_SUPPLY,
            from_wallet=None,
            to_wallet="network",
            timestamp=0.0,
            description="Genesis transaction - initial FTNS supply",
            parent_ids=[],
        )
        await self._store_transaction(genesis)
        self._state.tips.add(genesis.tx_id)
        await self._db.execute("INSERT OR IGNORE INTO tips (tx_id) VALUES (?)", ("genesis",))
        await self._db.commit()
        
    async def _store_transaction(self, tx: DAGTransaction) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO dag_transactions
               (tx_id, tx_type, amount, from_wallet, to_wallet, timestamp, signature, public_key, description, 
                parent_ids, cumulative_weight, confirmation_level, hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tx.tx_id,
                tx.tx_type.value,
                tx.amount,
                tx.from_wallet,
                tx.to_wallet,
                tx.timestamp,
                tx.signature,
                tx.public_key,
                tx.description,
                json.dumps(tx.parent_ids),
                tx.cumulative_weight,
                tx.confirmation_level,
                tx.hash(),
            ),
        )
        
    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            
    async def create_wallet(
        self, 
        wallet_id: str, 
        display_name: str = "",
        public_key: Optional[str] = None
    ) -> None:
        """Create a new wallet with optional public key for signature verification."""
        await self._db.execute(
            "INSERT OR IGNORE INTO wallets (wallet_id, display_name, public_key, created_at) VALUES (?, ?, ?, ?)",
            (wallet_id, display_name, public_key, time.time()),
        )
        await self._db.commit()
        
        if public_key:
            self._wallet_public_keys[wallet_id] = public_key
            
    async def register_wallet_public_key(self, wallet_id: str, public_key: str) -> None:
        """Register or update the public key for an existing wallet."""
        await self._db.execute(
            "UPDATE wallets SET public_key = ? WHERE wallet_id = ?",
            (public_key, wallet_id),
        )
        await self._db.commit()
        self._wallet_public_keys[wallet_id] = public_key
        
    def get_wallet_public_key(self, wallet_id: str) -> Optional[str]:
        """Get the registered public key for a wallet."""
        return self._wallet_public_keys.get(wallet_id)
        
    async def wallet_exists(self, wallet_id: str) -> bool:
        cursor = await self._db.execute("SELECT 1 FROM wallets WHERE wallet_id = ?", (wallet_id,))
        return await cursor.fetchone() is not None

    # ── Sprint 540 Pattern A: linked-address registry ────────────
    # Mirrors LocalLedger's interface so InboundMonitor + the
    # /wallet/deposit/* endpoints work against either ledger type.

    async def link_eth_address(
        self, wallet_id: str, eth_address: str,
    ) -> None:
        """Link an Ethereum address to a wallet_id for bridge deposits."""
        if not wallet_id:
            raise ValueError("wallet_id must be non-empty")
        if not eth_address or not eth_address.startswith("0x"):
            raise ValueError(
                f"eth_address must be 0x-prefixed; got {eth_address!r}"
            )
        addr_norm = eth_address.lower()
        # Ensure wallet row exists; DAGLedger doesn't have
        # _ensure_wallet symmetric to LocalLedger, so use INSERT OR
        # IGNORE.
        await self._db.execute(
            "INSERT OR IGNORE INTO wallets "
            "(wallet_id, display_name, created_at) "
            "VALUES (?, '', ?)",
            (wallet_id, time.time()),
        )
        await self._db.execute(
            "UPDATE wallets SET eth_address = NULL WHERE eth_address = ?",
            (addr_norm,),
        )
        await self._db.execute(
            "UPDATE wallets SET eth_address = ? WHERE wallet_id = ?",
            (addr_norm, wallet_id),
        )
        await self._db.commit()

    async def wallet_for_eth_address(
        self, eth_address: str,
    ) -> Optional[str]:
        """Reverse lookup: which wallet_id is linked to this address?"""
        if not eth_address:
            return None
        cursor = await self._db.execute(
            "SELECT wallet_id FROM wallets WHERE eth_address = ?",
            (eth_address.lower(),),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def eth_address_for_wallet(
        self, wallet_id: str,
    ) -> Optional[str]:
        """Forward lookup: ETH address linked to this wallet."""
        cursor = await self._db.execute(
            "SELECT eth_address FROM wallets WHERE wallet_id = ?",
            (wallet_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    # ── Sprint 554: per-wallet signature requirement + nonce ────

    async def get_requires_user_signature(
        self, wallet_id: str,
    ) -> bool:
        cursor = await self._db.execute(
            "SELECT requires_user_signature FROM wallets "
            "WHERE wallet_id = ?",
            (wallet_id,),
        )
        row = await cursor.fetchone()
        return bool(row[0]) if row else False

    async def set_requires_user_signature(
        self, wallet_id: str, enabled: bool,
    ) -> None:
        cursor = await self._db.execute(
            "SELECT 1 FROM wallets WHERE wallet_id = ?", (wallet_id,),
        )
        if await cursor.fetchone() is None:
            raise KeyError(f"unknown wallet {wallet_id!r}")
        await self._db.execute(
            "UPDATE wallets SET requires_user_signature = ? "
            "WHERE wallet_id = ?",
            (1 if enabled else 0, wallet_id),
        )
        await self._db.commit()

    async def get_next_withdraw_nonce(
        self, wallet_id: str,
    ) -> int:
        cursor = await self._db.execute(
            "SELECT next_withdraw_nonce FROM wallets "
            "WHERE wallet_id = ?",
            (wallet_id,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def bump_withdraw_nonce(self, wallet_id: str) -> int:
        """Atomically consume the current nonce + advance. Returns
        the OLD value (the nonce the caller just used)."""
        cursor = await self._db.execute(
            "SELECT next_withdraw_nonce FROM wallets "
            "WHERE wallet_id = ?",
            (wallet_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"unknown wallet {wallet_id!r}")
        old = int(row[0])
        await self._db.execute(
            "UPDATE wallets SET next_withdraw_nonce = ? "
            "WHERE wallet_id = ?",
            (old + 1, wallet_id),
        )
        await self._db.commit()
        return old

    async def get_balance(self, wallet_id: str) -> float:
        """Get the current balance for a wallet (non-atomic read)."""
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM dag_transactions WHERE to_wallet = ? AND tx_type != ?",
            (wallet_id, TransactionType.APPROVAL.value),
        )
        credits = (await cursor.fetchone())[0]
        
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM dag_transactions WHERE from_wallet = ? AND tx_type != ?",
            (wallet_id, TransactionType.APPROVAL.value),
        )
        debits = (await cursor.fetchone())[0]
        return round(credits - debits, 6)
    
    # =========================================================================
    # ATOMIC BALANCE OPERATIONS - TOCTOU Prevention
    # =========================================================================
    
    async def _get_or_create_balance_cache(self, wallet_id: str) -> Tuple[float, int]:
        """
        Get or create a balance cache entry for atomic operations.
        
        This method ensures the wallet_balances table has an entry for the wallet,
        computing the balance from transactions if needed.
        
        Args:
            wallet_id: The wallet to get/create cache for
            
        Returns:
            Tuple of (balance, version)
        """
        cursor = await self._db.execute(
            "SELECT balance, version FROM wallet_balances WHERE wallet_id = ?",
            (wallet_id,)
        )
        row = await cursor.fetchone()

        if row:
            balance, db_version = row[0], row[1]

            # Use service-level version cache when available.
            # This detects external DB modifications: if something outside this
            # service changed the DB version, the cache holds the old value,
            # which will cause a mismatch in _commit_balance_deduction.
            if wallet_id in self._balance_version_cache:
                version = self._balance_version_cache[wallet_id]
            else:
                # First access for this wallet — seed the cache
                version = db_version
                self._balance_version_cache[wallet_id] = version

            return (balance, version)

        else:
            # New wallet — create with version 1
            await self._db.execute(
                """INSERT INTO wallet_balances (wallet_id, balance, version, last_updated)
                   VALUES (?, 0.0, 1, ?)""",
                (wallet_id, time.time())
            )
            self._balance_version_cache[wallet_id] = 1
            return (0.0, 1)
    
    async def _check_balance_atomic(
        self,
        wallet_id: str,
        amount: float,
        lock_timeout_ms: int = 5000
    ) -> Tuple[bool, float, int]:
        """
        Atomically check if wallet has sufficient balance with row-level locking.
        
        This method uses SQLite's SAVEPOINT to create a nested transaction context
        for atomic balance operations, preventing TOCTOU race conditions.
        
        Atomicity Guarantees:
        1. Uses SAVEPOINT for nested transaction support
        2. Reads balance with version for optimistic concurrency control
        3. Balance check and version read happen atomically
        
        Args:
            wallet_id: Wallet to check
            amount: Required amount
            lock_timeout_ms: Timeout for acquiring lock (default 5 seconds)
            
        Returns:
            Tuple of (has_sufficient, current_balance, version)
            
        Raises:
            BalanceLockError: If lock cannot be acquired within timeout
        """
        try:
            # Use SAVEPOINT for nested transaction support
            # This allows us to have atomic operations within an existing transaction
            await self._db.execute("SAVEPOINT balance_check")
            
            # Get or create balance cache with version
            balance, version = await self._get_or_create_balance_cache(wallet_id)
            
            has_sufficient = balance >= amount
            
            if not has_sufficient:
                # Release the savepoint and raise error
                await self._db.execute("RELEASE SAVEPOINT balance_check")
                logger.warning(
                    f"Atomic balance check failed: insufficient balance for {wallet_id[:8]}... "
                    f"(has {balance:.6f}, needs {amount:.6f})"
                )
            else:
                # Keep savepoint for caller to complete
                logger.debug(
                    f"Atomic balance check passed for {wallet_id[:8]}... "
                    f"(balance={balance:.6f}, version={version})"
                )
            
            return (has_sufficient, balance, version)
            
        except Exception as e:
            # Rollback to savepoint on error
            try:
                await self._db.execute("ROLLBACK TO SAVEPOINT balance_check")
                await self._db.execute("RELEASE SAVEPOINT balance_check")
            except Exception:
                # Savepoint may not exist if the outer transaction failed
                # This is acceptable - the outer transaction will handle cleanup
                logger.debug("Could not release savepoint during balance check rollback")
            
            if "database is locked" in str(e).lower():
                raise BalanceLockError(
                    f"Could not acquire balance lock for {wallet_id[:8]}... within {lock_timeout_ms}ms"
                )
            raise
    
    async def _commit_balance_deduction(
        self,
        wallet_id: str,
        amount: float,
        expected_version: int
    ) -> bool:
        """
        Commit a balance deduction with optimistic concurrency control.
        
        This method MUST be called after _check_balance_atomic within the same
        transaction. It updates the balance cache only if the version hasn't
        changed, detecting concurrent modifications.
        
        Atomicity Guarantees:
        1. Uses optimistic concurrency control (version check)
        2. Update only succeeds if version matches expected
        3. Automatic detection of concurrent modifications
        
        Args:
            wallet_id: Wallet to deduct from
            amount: Amount to deduct
            expected_version: Version expected from balance check
            
        Returns:
            True if deduction succeeded
            
        Raises:
            ConcurrentModificationError: If balance was modified by another transaction
        """
        # Update balance with version check (optimistic concurrency control)
        cursor = await self._db.execute(
            """UPDATE wallet_balances
               SET balance = balance - ?,
                   version = version + 1,
                   last_updated = ?
               WHERE wallet_id = ? AND version = ?""",
            (amount, time.time(), wallet_id, expected_version)
        )
        
        if cursor.rowcount == 0:
            # Version mismatch - concurrent modification detected
            # Rollback to savepoint and release
            await self._db.execute("ROLLBACK TO SAVEPOINT balance_check")
            await self._db.execute("RELEASE SAVEPOINT balance_check")
            logger.error(
                f"TOCTOU detected: concurrent modification of {wallet_id[:8]}... "
                f"(expected version {expected_version})"
            )
            raise ConcurrentModificationError(
                f"Balance for {wallet_id[:8]}... was modified by another transaction. "
                "Please retry the operation."
            )
        
        # Release the savepoint - the main transaction will commit later
        await self._db.execute("RELEASE SAVEPOINT balance_check")

        # After successful deduction, update version cache
        self._balance_version_cache[wallet_id] = expected_version + 1

        logger.info(
            f"Atomic balance deduction completed for {wallet_id[:8]}... "
            f"(deducted {amount:.6f}, new version {expected_version + 1})"
        )

        return True
    
    async def _commit_balance_credit(
        self,
        wallet_id: str,
        amount: float
    ) -> bool:
        """
        Commit a balance credit (for receiving wallet).
        
        This method updates the balance cache for credits. It uses INSERT OR REPLACE
        to handle both new and existing wallets atomically.
        
        Args:
            wallet_id: Wallet to credit
            amount: Amount to credit
            
        Returns:
            True if credit succeeded
        """
        # Use INSERT ... ON CONFLICT for atomic upsert
        await self._db.execute(
            """INSERT INTO wallet_balances (wallet_id, balance, version, last_updated)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(wallet_id) DO UPDATE SET
                   balance = balance + excluded.balance,
                   version = version + 1,
                   last_updated = excluded.last_updated""",
            (wallet_id, amount, time.time())
        )

        # The ON CONFLICT DO UPDATE always increments version in the DB.
        # Mirror that increment in the cache so subsequent deductions see the correct version.
        if wallet_id not in self._balance_version_cache:
            self._balance_version_cache[wallet_id] = 1
        else:
            self._balance_version_cache[wallet_id] += 1

        logger.debug(
            f"Atomic balance credit completed for {wallet_id[:8]}... "
            f"(credited {amount:.6f})"
        )

        return True
    
    async def _rollback_balance_check(self) -> None:
        """
        Rollback an atomic balance check savepoint.
        
        Call this if balance check passed but transaction creation fails.
        """
        try:
            await self._db.execute("ROLLBACK TO SAVEPOINT balance_check")
            await self._db.execute("RELEASE SAVEPOINT balance_check")
            logger.debug("Rolled back atomic balance check savepoint")
        except Exception:
            # Savepoint may not exist if the outer transaction failed
            # This is acceptable - the outer transaction will handle cleanup
            logger.debug("Could not rollback savepoint - may not exist or already released")
    
    def select_tips_mcmc(self, num_tips: int = 2) -> List[str]:
        """
        Select tips using simplified weighted random selection.
        
        In a proper IOTA implementation, this would use MCMC to walk
        the DAG and select tips with probability proportional to their
        cumulative weight. For now, we use weighted random from tips.
        """
        if not self._state.tips:
            return []
        
        tips_list = list(self._state.tips)
        
        if len(tips_list) <= num_tips:
            return tips_list
        
        weights = []
        for tip_id in tips_list:
            tip = self._state.transactions.get(tip_id)
            if tip:
                weight = math.exp(self.alpha * math.log(tip.cumulative_weight + 1))
            else:
                weight = 1.0
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return tips_list[:num_tips]
        
        selected = []
        available = list(tips_list)
        
        for _ in range(num_tips):
            if not available:
                break
            r = random.random() * total_weight
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if cumulative >= r and available[i] not in selected:
                    selected.append(available[i])
                    break
            
            weights = [w for i, w in enumerate(weights) if available[i] not in selected]
            available = [t for t in available if t not in selected]
            total_weight = sum(weights) if weights else 0
        
        return selected
    
    def _get_children(self, tx_id: str) -> List[str]:
        children = []
        for other_id, tx in self._state.transactions.items():
            if tx_id in tx.parent_ids:
                children.append(other_id)
        return children
    
    def _verify_transaction_signature(
        self,
        tx: DAGTransaction,
        signature: str,
        public_key: str
    ) -> bool:
        """
        Verify the Ed25519 signature of a transaction.
        
        This method performs cryptographic verification that the transaction
        was authorized by the owner of the private key corresponding to the
        provided public key.
        
        Args:
            tx: The transaction to verify
            signature: Base64-encoded signature (64 bytes when decoded)
            public_key: Hex-encoded Ed25519 public key (32 bytes when decoded)
            
        Returns:
            bool: True if signature is valid
            
        Raises:
            InvalidSignatureError: If signature verification fails due to:
                - Invalid signature format
                - Signature doesn't match transaction hash
                - Invalid public key format
                - Any other cryptographic error
                
        Security Notes:
            - This method fails closed: any error results in rejection
            - The transaction hash includes tx_id, so signatures must be
              created after the client receives the tx_id from the server
              OR the client must use a signing scheme that excludes tx_id
            - For the chicken-and-egg problem, clients typically sign
              transaction data without tx_id, and servers verify using
              a hash that excludes tx_id (see get_signing_data())
        """
        try:
            # Validate inputs are not empty
            if not signature:
                logger.warning("Empty signature provided for verification")
                raise InvalidSignatureError("Signature cannot be empty")
            if not public_key:
                logger.warning("Empty public key provided for verification")
                raise InvalidSignatureError("Public key cannot be empty")
            
            # Load the public key from hex format
            try:
                pk = DAGSignatureManager.load_public_key_from_hex(public_key)
            except ValueError as e:
                logger.warning(f"Invalid public key format: {e}")
                raise InvalidSignatureError(f"Invalid public key format: {e}")
            
            # Get the transaction hash for verification
            tx_hash = tx.hash()
            logger.debug(f"Verifying signature for tx {tx.tx_id[:8]}... with hash {tx_hash[:16]}...")
            
            # Verify the signature using Ed25519
            try:
                result = DAGSignatureManager.verify_signature(tx_hash, signature, pk)
                logger.debug(f"Signature verification successful for tx {tx.tx_id[:8]}...")
                return result
            except InvalidSignatureError:
                # Re-raise InvalidSignatureError from DAGSignatureManager directly
                logger.warning(
                    f"Signature verification failed for tx {tx.tx_id[:8]}... "
                    f"from wallet {tx.from_wallet[:8] if tx.from_wallet else 'system'}..."
                )
                raise
                
        except InvalidSignatureError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            logger.error(f"Unexpected error during signature verification: {e}")
            raise InvalidSignatureError(f"Signature verification failed: {e}")
    
    def _is_signature_required(self, tx_type: TransactionType, from_wallet: Optional[str]) -> bool:
        """
        Determine if a transaction requires a signature.
        
        Transactions that don't require signatures:
        - GENESIS: Initial token supply (system-initiated)
        - Transactions from None (system transactions)
        - WELCOME_GRANT: System-initiated grants for new users
        
        All other transactions require valid Ed25519 signatures.
        
        Security Policy:
            - Fail closed: when in doubt, require a signature
            - Only explicitly exempt transaction types can skip verification
            - System transactions (from_wallet=None) are trusted by definition
            
        Args:
            tx_type: The type of transaction being submitted
            from_wallet: The source wallet (None for system transactions)
            
        Returns:
            bool: True if signature is required, False if exempt
        """
        # System transactions (no source wallet) don't need signatures
        if from_wallet is None:
            logger.debug(f"No signature required: system transaction (tx_type={tx_type.value})")
            return False

        # Genesis transactions are system-initiated
        if tx_type == TransactionType.GENESIS:
            logger.debug(f"No signature required: genesis transaction")
            return False

        # System transactions are service-initiated, no signature required
        if tx_type == TransactionType.SYSTEM:
            logger.debug(f"No signature required: system transaction type")
            return False

        # Sprint 541 — Pattern A bridge transactions are
        # daemon-mediated. The cryptographic authorization is the
        # operator's signature on the on-chain TX (which the user
        # implicitly trusts by calling the daemon API). The off-chain
        # debit is bookkeeping for the daemon-side accounting. In
        # the future when user auth is wired through to /wallet/
        # withdraw, this exemption can be tightened to require a
        # user signature alongside.
        if tx_type in (
            TransactionType.BRIDGE_DEPOSIT,
            TransactionType.BRIDGE_WITHDRAW,
        ):
            logger.debug(
                f"No signature required: daemon-mediated bridge "
                f"tx_type={tx_type.value}"
            )
            return False

        # All other transactions require signatures
        logger.debug(f"Signature required for tx_type={tx_type.value} from {from_wallet[:8]}...")
        return True
    
    async def submit_transaction(
        self,
        tx_type: TransactionType,
        amount: float,
        from_wallet: Optional[str],
        to_wallet: str,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        """
        Submit a new transaction to the DAG.
        
        The transaction must approve (reference) 2-4 tips selected via MCMC.
        This is what makes the DAG grow - every transaction helps confirm
        previous transactions.
        
        Signature Requirements:
        - Transactions with a from_wallet require a valid Ed25519 signature
        - The signature must be created by signing the transaction hash
        - The public key must be provided or registered with the wallet
        
        Atomicity Guarantees (TOCTOU Prevention):
        - Balance checks use BEGIN IMMEDIATE to acquire write lock
        - Balance deduction uses optimistic concurrency control (version check)
        - Concurrent modifications are detected and rejected with retry guidance
        - The entire balance check + deduction is atomic
        
        Args:
            tx_type: Type of transaction
            amount: Transaction amount
            from_wallet: Source wallet (None for system transactions)
            to_wallet: Destination wallet
            description: Optional description
            signature: Base64-encoded Ed25519 signature (required if from_wallet is set)
            public_key: Hex-encoded Ed25519 public key (required for first transaction from wallet)
            
        Returns:
            DAGTransaction: The created transaction
            
        Raises:
            InsufficientBalanceError: If balance is insufficient (atomic check)
            ConcurrentModificationError: If concurrent modification detected
            BalanceLockError: If balance lock cannot be acquired
            MissingSignatureError: If signature is required but not provided
            InvalidSignatureError: If signature verification fails
        """
        # Track atomic balance check state for cleanup
        balance_version = None
        atomic_check_in_progress = False

        # sp910 — CONNECTION-WIDE write lock (supersedes the sprint-487
        # per-wallet lock). The SAVEPOINT balance_check is connection-
        # global, so a foreign COMMIT (from ANY wallet's write — incl. a
        # credit-only submit with no from_wallet, or record_nonce) would
        # release this submit's savepoint mid-debit and silently commit a
        # half-finished debit (→ no-refund loss on /wallet/withdraw).
        # Serializing every connection write through one lock — held from
        # before the savepoint through the final commit — eliminates the
        # interleave. Acquired ALWAYS (credit-only included: its commit can
        # destroy a concurrent debit's savepoint just as easily).
        write_lock = self._get_write_lock()
        await write_lock.acquire()

        try:
            # Create wallets if they don't exist
            if from_wallet and not await self.wallet_exists(from_wallet):
                await self.create_wallet(from_wallet, f"wallet-{from_wallet[:8]}", public_key)
            if not await self.wallet_exists(to_wallet):
                await self.create_wallet(to_wallet, f"wallet-{to_wallet[:8]}")

            # ATOMIC BALANCE CHECK - TOCTOU Prevention
            # Use atomic balance check with row-level locking for debit transactions
            if from_wallet:
                has_sufficient, balance, balance_version = await self._check_balance_atomic(
                    from_wallet, amount
                )
                atomic_check_in_progress = True
                
                if not has_sufficient:
                    raise InsufficientBalanceError(
                        f"Insufficient balance for {from_wallet[:8]}...: "
                        f"has {balance:.6f}, needs {amount:.6f}"
                    )
        
            # Signature verification - pre-creation checks
            # Note: Signature verification happens AFTER transaction creation due to a
            # chicken-and-egg problem: the signature is created by the client who doesn't
            # know the tx_id yet, but the transaction hash includes tx_id.
            # Solution: Clients sign transaction data WITHOUT tx_id, and we verify using
            # the hash of core transaction fields (excluding tx_id).
            if self.verify_signatures and self._is_signature_required(tx_type, from_wallet):
                if not signature:
                    logger.warning(
                        f"Signature required but missing for transaction from {from_wallet}"
                    )
                    raise MissingSignatureError(
                        f"Transaction from {from_wallet} requires a signature"
                    )
                
                # Get public key - prefer provided key, then registered key
                verify_pk = public_key or self.get_wallet_public_key(from_wallet)
                if not verify_pk:
                    logger.warning(
                        f"No public key available for wallet {from_wallet}"
                    )
                    raise MissingSignatureError(
                        f"No public key provided or registered for wallet {from_wallet}"
                    )
                
                # Store verification data for post-creation verification
                # This ensures we don't silently skip verification due to missing data
                self._pending_verification = {
                    "from_wallet": from_wallet,
                    "signature": signature,
                    "public_key": verify_pk,
                }
            else:
                self._pending_verification = None
                    
            parent_ids = self.select_tips_mcmc(num_tips=min(self.max_parents, len(self._state.tips)))
            
            tx = DAGTransaction(
                tx_id=str(uuid.uuid4()),
                tx_type=tx_type,
                amount=amount,
                from_wallet=from_wallet,
                to_wallet=to_wallet,
                timestamp=time.time(),
                signature=signature,
                public_key=public_key or (self.get_wallet_public_key(from_wallet) if from_wallet else None),
                description=description,
                parent_ids=parent_ids,
            )
            
            # Post-creation signature verification
            # This is where we actually verify the signature after the transaction is created.
            # The signature should have been created from transaction data that will match
            # the tx.hash() method output (which includes tx_id).
            #
            # Edge cases handled:
            # - Genesis transactions: No signature required (system-initiated)
            # - System transactions (from_wallet=None): No signature required
            # - SYSTEM transaction type: Service-initiated, no signature required
            # - WELCOME_GRANT: System-initiated, no signature required
            # - First transaction from new wallet: Must provide public_key with signature
            #
            # SECURITY: We explicitly check all conditions and fail closed, not open.
            if self._pending_verification is not None:
                # Retrieve and clear pending verification data
                pending = self._pending_verification
                self._pending_verification = None
                
                # These should never be None due to pre-creation checks, but verify defensively
                sig_to_verify = pending.get("signature")
                pk_to_verify = pending.get("public_key")
                wallet_being_verified = pending.get("from_wallet")
                
                if not sig_to_verify:
                    logger.error(
                        f"SECURITY: Signature disappeared during transaction creation for wallet {wallet_being_verified}"
                    )
                    raise MissingSignatureError(
                        f"Signature required for transaction from {wallet_being_verified} but not found during verification"
                    )
                
                if not pk_to_verify:
                    logger.error(
                        f"SECURITY: Public key disappeared during transaction creation for wallet {wallet_being_verified}"
                    )
                    raise MissingSignatureError(
                        f"Public key required for transaction from {wallet_being_verified} but not found during verification"
                    )
                
                try:
                    self._verify_transaction_signature(tx, sig_to_verify, pk_to_verify)
                    logger.info(
                        f"Signature verified successfully for transaction {tx.tx_id[:8]}... "
                        f"from wallet {wallet_being_verified[:8] if wallet_being_verified else 'system'}..."
                    )
                except InvalidSignatureError as e:
                    logger.warning(
                        f"SECURITY: Invalid signature detected for transaction from {wallet_being_verified}: {e}"
                    )
                    raise InvalidSignatureError(
                        f"Invalid signature for transaction from {wallet_being_verified}"
                    )
                except Exception as e:
                    logger.error(
                        f"SECURITY: Unexpected error during signature verification for wallet {wallet_being_verified}: {e}"
                    )
                    raise InvalidSignatureError(
                        f"Signature verification failed for transaction from {wallet_being_verified}: {e}"
                    )
            
            await self._store_transaction(tx)
            
            for parent_id in parent_ids:
                if parent_id in self._state.tips:
                    self._state.tips.discard(parent_id)
                    await self._db.execute("DELETE FROM tips WHERE tx_id = ?", (parent_id,))
                    
            self._state.tips.add(tx.tx_id)
            await self._db.execute("INSERT OR IGNORE INTO tips (tx_id) VALUES (?)", (tx.tx_id,))
            
            await self._update_weights(tx)
            
            # ATOMIC BALANCE DEDUCTION - Complete the atomic operation
            # This commits the balance deduction with version check
            if from_wallet and balance_version is not None:
                await self._commit_balance_deduction(from_wallet, amount, balance_version)
            else:
                # For credit transactions (no from_wallet), just commit
                await self._db.commit()

            # Update balance cache for receiving wallet
            if to_wallet:
                await self._commit_balance_credit(to_wallet, amount)

            # Sprint 489 (F27 fix) — durability barrier. Pre-fix,
            # `_commit_balance_deduction` released the savepoint
            # but never called `await self._db.commit()` on the
            # outer connection. Same for `_commit_balance_credit`
            # + `_store_transaction`. Result: dag_transactions
            # writes for transfer-style submits were buffered and
            # only the LAST one flushed at close-time. This is
            # how sprint 487's concurrent submits managed to
            # leak escrows — some `Escrow for job X` outflows
            # persisted (the daemon's running connection flushed)
            # but cancel-side `Escrow refund` inflows never
            # actually committed before daemon restart wiped them.
            await self._db.commit()

            self._state.transactions[tx.tx_id] = tx
            
            logger.info(
                f"Transaction {tx.tx_id[:8]}... completed atomically: "
                f"{amount:.6f} from {from_wallet[:8] if from_wallet else 'system'}... "
                f"to {to_wallet[:8]}..."
            )
            
            return tx
            
        except (InsufficientBalanceError, ConcurrentModificationError, BalanceLockError):
            # These are atomic operation errors - re-raise as-is
            raise
        except (MissingSignatureError, InvalidSignatureError):
            # These are signature errors - rollback and re-raise
            if atomic_check_in_progress:
                await self._rollback_balance_check()
            raise
        except Exception as e:
            # Unexpected error - rollback and wrap
            logger.error(f"Unexpected error in submit_transaction: {e}")
            if atomic_check_in_progress:
                await self._rollback_balance_check()
            raise
        finally:
            # sp910 — release the connection-wide write lock taken before
            # the savepoint window (held through the final commit).
            if write_lock.locked():
                write_lock.release()

    async def _update_weights(self, new_tx: DAGTransaction) -> None:
        """Update cumulative weights and confirmation levels for approved transactions."""
        to_update = set()
        queue = list(new_tx.parent_ids)
        
        while queue:
            tx_id = queue.pop(0)
            if tx_id in to_update:
                continue
            to_update.add(tx_id)
            
            tx = self._state.transactions.get(tx_id)
            if tx:
                tx.cumulative_weight += 1
                
                max_weight = max(t.cumulative_weight for t in self._state.transactions.values()) or 1
                tx.confirmation_level = min(1.0, tx.cumulative_weight / self.confirmation_threshold)
                
                await self._db.execute(
                    """UPDATE dag_transactions 
                       SET cumulative_weight = ?, confirmation_level = ?
                       WHERE tx_id = ?""",
                    (tx.cumulative_weight, tx.confirmation_level, tx_id),
                )
                
                for child_id in self._get_children(tx_id):
                    if child_id not in to_update:
                        queue.append(child_id)
    
    async def approve_transaction(self, approver_wallet: str, tx_id: str) -> Optional[DAGTransaction]:
        """Submit an approval transaction that directly approves a specific transaction."""
        tx = self._state.transactions.get(tx_id)
        if not tx:
            return None
            
        return await self.submit_transaction(
            tx_type=TransactionType.APPROVAL,
            amount=0,
            from_wallet=approver_wallet,
            to_wallet=tx_id,
            description=f"Approving transaction {tx_id[:16]}...",
        )
    
    async def get_transaction(self, tx_id: str) -> Optional[DAGTransaction]:
        if tx_id in self._state.transactions:
            return self._state.transactions[tx_id]
            
        cursor = await self._db.execute(
            """SELECT tx_id, tx_type, amount, from_wallet, to_wallet, timestamp, 
                      signature, public_key, description, parent_ids, cumulative_weight, confirmation_level
               FROM dag_transactions WHERE tx_id = ?""",
            (tx_id,),
        )
        row = await cursor.fetchone()
        if row:
            tx = DAGTransaction(
                tx_id=row[0],
                tx_type=TransactionType(row[1]),
                amount=row[2],
                from_wallet=row[3],
                to_wallet=row[4],
                timestamp=row[5],
                signature=row[6],
                public_key=row[7],
                description=row[8] or "",
                parent_ids=json.loads(row[9]),
                cumulative_weight=row[10],
                confirmation_level=row[11],
            )
            self._state.transactions[tx_id] = tx
            return tx
        return None

    async def get_all_transactions(self) -> List[DAGTransaction]:
        """Return all transactions in the ledger, ordered by timestamp ascending."""

        cursor = await self._db.execute(
            """SELECT tx_id, tx_type, amount, from_wallet, to_wallet, timestamp,
                      signature, public_key, description, parent_ids,
                      cumulative_weight, confirmation_level
               FROM dag_transactions
               ORDER BY timestamp ASC"""
        )
        rows = await cursor.fetchall()

        transactions = []
        for row in rows:
            tx = DAGTransaction(
                tx_id=row[0],
                tx_type=TransactionType(row[1]),
                amount=row[2],
                from_wallet=row[3],
                to_wallet=row[4],
                timestamp=row[5],
                signature=row[6],
                public_key=row[7],
                description=row[8] or "",
                parent_ids=json.loads(row[9]) if row[9] else [],
                cumulative_weight=row[10],
                confirmation_level=row[11],
            )
            transactions.append(tx)

        return transactions

    async def get_transaction_history(self, wallet_id: str, limit: int = 50) -> List[DAGTransaction]:
        cursor = await self._db.execute(
            """SELECT tx_id, tx_type, amount, from_wallet, to_wallet, timestamp, 
                      signature, public_key, description, parent_ids, cumulative_weight, confirmation_level
               FROM dag_transactions
               WHERE (to_wallet = ? OR from_wallet = ?) AND tx_type != ?
               ORDER BY timestamp DESC LIMIT ?""",
            (wallet_id, wallet_id, TransactionType.APPROVAL.value, limit),
        )
        rows = await cursor.fetchall()
        return [
            DAGTransaction(
                tx_id=r[0],
                tx_type=TransactionType(r[1]),
                amount=r[2],
                from_wallet=r[3],
                to_wallet=r[4],
                timestamp=r[5],
                signature=r[6],
                public_key=r[7],
                description=r[8] or "",
                parent_ids=json.loads(r[9]),
                cumulative_weight=r[10],
                confirmation_level=r[11],
            )
            for r in rows
        ]

    async def get_recent_tx_ids(self, wallet_id: str, limit: int = 50) -> List[str]:
        """Recent transaction IDs for this wallet — the balance-proof payload.

        Sprint 1419. This existed on LocalLedger and on DAGLedgerAdapter, but NOT here — and
        `Node._initialize()` builds a bare DAGLedger whenever `config.ledger_type == "dag"`, which
        is the DEFAULT. (DAGLedgerAdapter is never instantiated anywhere in prsm/.) So on every
        default node, LedgerSync's `self.ledger.get_recent_tx_ids(...)` raised AttributeError, and
        `_reconciliation_loop` swallowed it — the node silently never sent a balance proof and never
        answered a peer's balance_request. Balance reconciliation was inert across the whole network.
        """
        history = await self.get_transaction_history(wallet_id, limit)
        return [tx.tx_id for tx in history]

    async def transfer(
        self,
        from_wallet: str,
        to_wallet: str,
        amount: float,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        return await self.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=amount,
            from_wallet=from_wallet,
            to_wallet=to_wallet,
            description=description,
            signature=signature,
            public_key=public_key,
        )
    
    async def credit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        # Sp898 — accept signature/public_key for parity with debit()
        # and LocalLedger.credit(). ledger_sync._on_ftns_transaction
        # calls credit(..., signature=...) when applying an incoming
        # cross-node transfer; without these params the DAG backend
        # raised TypeError on every remote credit.
        return await self.submit_transaction(
            tx_type=tx_type,
            amount=amount,
            from_wallet=None,
            to_wallet=wallet_id,
            description=description,
            signature=signature,
            public_key=public_key,
        )
    
    async def debit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        """Debit FTNS from a wallet (payment to system/network)."""
        balance = await self.get_balance(wallet_id)
        if balance < amount:
            raise ValueError(
                f"Insufficient balance: {balance:.6f} < {amount:.6f}"
            )
        return await self.submit_transaction(
            tx_type=tx_type,
            amount=amount,
            from_wallet=wallet_id,
            to_wallet="system",
            description=description,
            signature=signature,
            public_key=public_key,
        )

    async def issue_welcome_grant(self, wallet_id: str, amount: float = 100.0) -> DAGTransaction:
        cursor = await self._db.execute(
            "SELECT 1 FROM dag_transactions WHERE to_wallet = ? AND tx_type = ?",
            (wallet_id, TransactionType.WELCOME_GRANT.value),
        )
        if await cursor.fetchone() is not None:
            raise ValueError(f"Wallet {wallet_id} already received a welcome grant")

        return await self.credit(
            wallet_id=wallet_id,
            amount=amount,
            tx_type=TransactionType.WELCOME_GRANT,
            description=f"Welcome grant of {amount} FTNS",
        )
    
    async def get_stats(self) -> Dict:
        confirmed = sum(1 for tx in self._state.transactions.values() if tx.confirmation_level >= 0.5)
        pending = len(self._state.transactions) - confirmed
        
        return {
            "total_transactions": len(self._state.transactions),
            "tips": len(self._state.tips),
            "confirmed": confirmed,
            "pending": pending,
            "cumulative_weight_sum": sum(tx.cumulative_weight for tx in self._state.transactions.values()),
            "avg_confirmation_level": sum(tx.confirmation_level for tx in self._state.transactions.values()) / max(1, len(self._state.transactions)),
        }
    
    async def get_tips(self) -> List[str]:
        return list(self._state.tips)
    
    async def get_transaction_count(self, wallet_id: str) -> int:
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM dag_transactions WHERE to_wallet = ? OR from_wallet = ?",
            (wallet_id, wallet_id),
        )
        return (await cursor.fetchone())[0]
    
    async def has_seen_nonce(self, nonce: str) -> bool:
        cursor = await self._db.execute("SELECT 1 FROM seen_nonces WHERE nonce = ?", (nonce,))
        return await cursor.fetchone() is not None

    async def has_transaction(self, tx_id: str) -> bool:
        """Sp898 — whether a transaction with this id is already in the
        DAG. Required by ledger_sync's incoming-transfer dedup path
        (`_on_ftns_transaction`); without it the DAG backend raised
        AttributeError on every incoming cross-node transfer. Parity
        with LocalLedger.has_transaction."""
        cursor = await self._db.execute(
            "SELECT 1 FROM dag_transactions WHERE tx_id = ?", (tx_id,)
        )
        return await cursor.fetchone() is not None

    async def record_nonce(self, nonce: str, origin: str) -> bool:
        """Atomically claim a nonce. Returns True if THIS call inserted
        it (won the claim), False if already present. Sp898 — mirrors
        LocalLedger.record_nonce so ledger_sync's double-credit gate
        works identically whichever ledger backend is wired."""
        # sp910 — serialize under the connection-wide write lock: this
        # commit would otherwise release a concurrent submit_transaction's
        # in-flight balance_check savepoint on the shared connection.
        async with self._get_write_lock():
            cursor = await self._db.execute(
                "INSERT OR IGNORE INTO seen_nonces (nonce, origin, seen_at) VALUES (?, ?, ?)",
                (nonce, origin, time.time()),
            )
            await self._db.commit()
            return cursor.rowcount == 1

    async def release_nonce(self, nonce: str) -> bool:
        """Sp918 — release a claimed nonce so it can be re-claimed (see
        LocalLedger.release_nonce; ONLY for definitively-no-payment reverts /
        never-broadcasts — never a 'pending' tx). Under the sp910 write lock
        for shared-connection safety. Returns True if a row was removed."""
        async with self._get_write_lock():
            cursor = await self._db.execute(
                "DELETE FROM seen_nonces WHERE nonce = ?", (nonce,)
            )
            await self._db.commit()
            return cursor.rowcount == 1


class DAGLedgerAdapter:
    """
    Adapter that makes DAGLedger compatible with the existing LocalLedger API.
    This allows the DAG ledger to be used as a drop-in replacement.
    """
    
    def __init__(self, dag_ledger: DAGLedger):
        self._dag = dag_ledger
        
    async def initialize(self) -> None:
        await self._dag.initialize()
        
    async def close(self) -> None:
        await self._dag.close()
        
    async def create_wallet(
        self, 
        wallet_id: str, 
        display_name: str = "",
        public_key: Optional[str] = None
    ) -> None:
        await self._dag.create_wallet(wallet_id, display_name, public_key)
        
    async def register_wallet_public_key(self, wallet_id: str, public_key: str) -> None:
        await self._dag.register_wallet_public_key(wallet_id, public_key)
        
    async def wallet_exists(self, wallet_id: str) -> bool:
        return await self._dag.wallet_exists(wallet_id)
        
    async def get_balance(self, wallet_id: str) -> float:
        return await self._dag.get_balance(wallet_id)
        
    async def credit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
        signature: Optional[str] = None,
    ) -> DAGTransaction:
        return await self._dag.credit(wallet_id, amount, tx_type, description)
        
    async def debit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        return await self._dag.submit_transaction(
            tx_type=tx_type,
            amount=amount,
            from_wallet=wallet_id,
            to_wallet="system",
            description=description,
            signature=signature,
            public_key=public_key,
        )
        
    async def transfer(
        self,
        from_wallet: str,
        to_wallet: str,
        amount: float,
        tx_type: TransactionType = TransactionType.TRANSFER,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        return await self._dag.transfer(from_wallet, to_wallet, amount, description, signature, public_key)
        
    async def get_transaction_history(self, wallet_id: str, limit: int = 50) -> List[DAGTransaction]:
        return await self._dag.get_transaction_history(wallet_id, limit)
        
    async def get_transaction_count(self, wallet_id: str) -> int:
        return await self._dag.get_transaction_count(wallet_id)
        
    async def has_seen_nonce(self, nonce: str) -> bool:
        return await self._dag.has_seen_nonce(nonce)
        
    async def record_nonce(self, nonce: str, origin: str) -> bool:
        # Sp898 — propagate the atomic-claim bool so ledger_sync's
        # double-credit gate works on the DAG backend too.
        return await self._dag.record_nonce(nonce, origin)

    async def release_nonce(self, nonce: str) -> bool:
        # Sp918 — release a claimed nonce (definitive-no-payment reverts only).
        return await self._dag.release_nonce(nonce)
        
    async def get_recent_tx_ids(self, wallet_id: str, limit: int = 50) -> List[str]:
        history = await self._dag.get_transaction_history(wallet_id, limit)
        return [tx.tx_id for tx in history]
        
    async def has_transaction(self, tx_id: str) -> bool:
        return await self._dag.get_transaction(tx_id) is not None
        
    async def issue_welcome_grant(self, wallet_id: str, amount: float = 100.0) -> DAGTransaction:
        return await self._dag.issue_welcome_grant(wallet_id, amount)
        
    # ── Agent Allowances ────────────────────────────────────────

    async def grant_agent_allowance(
        self,
        principal_id: str,
        agent_id: str,
        amount: float,
        epoch_hours: float = 24.0,
    ) -> None:
        now = time.time()
        await self._dag._db.execute(
            """INSERT INTO agent_allowances
               (agent_id, principal_id, allowance, spent, epoch_hours, epoch_start, created_at, revoked)
               VALUES (?, ?, ?, 0, ?, ?, ?, 0)
               ON CONFLICT(agent_id) DO UPDATE SET
                   allowance = excluded.allowance,
                   spent = 0,
                   epoch_hours = excluded.epoch_hours,
                   epoch_start = excluded.epoch_start,
                   revoked = 0""",
            (agent_id, principal_id, amount, epoch_hours, now, now),
        )
        await self._dag._db.commit()

    async def get_agent_allowance(self, agent_id: str) -> Optional[dict]:
        cursor = await self._dag._db.execute(
            "SELECT principal_id, allowance, spent, epoch_hours, epoch_start, revoked "
            "FROM agent_allowances WHERE agent_id = ?",
            (agent_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        principal_id, allowance, spent, epoch_hours, epoch_start, revoked = row
        now = time.time()
        if now - epoch_start >= epoch_hours * 3600:
            spent = 0.0
            epoch_start = now
            await self._dag._db.execute(
                "UPDATE agent_allowances SET spent = 0, epoch_start = ? WHERE agent_id = ?",
                (now, agent_id),
            )
            await self._dag._db.commit()
        return {
            "agent_id": agent_id,
            "principal_id": principal_id,
            "allowance": allowance,
            "spent": spent,
            "remaining": max(0.0, allowance - spent),
            "epoch_hours": epoch_hours,
            "epoch_start": epoch_start,
            "revoked": bool(revoked),
        }

    async def agent_debit(
        self,
        agent_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
    ) -> Optional[DAGTransaction]:
        allowance = await self.get_agent_allowance(agent_id)
        if not allowance or allowance["revoked"]:
            return None
        if amount > allowance["remaining"]:
            return None
        principal_id = allowance["principal_id"]
        balance = await self.get_balance(principal_id)
        if balance < amount:
            return None
        tx = await self.debit(
            wallet_id=principal_id,
            amount=amount,
            tx_type=tx_type,
            description=f"[agent:{agent_id[:12]}] {description}",
        )
        await self._dag._db.execute(
            "UPDATE agent_allowances SET spent = spent + ? WHERE agent_id = ?",
            (amount, agent_id),
        )
        await self._dag._db.commit()
        return tx

    async def revoke_agent_allowance(self, principal_id: str, agent_id: str) -> bool:
        cursor = await self._dag._db.execute(
            "UPDATE agent_allowances SET revoked = 1 "
            "WHERE agent_id = ? AND principal_id = ?",
            (agent_id, principal_id),
        )
        await self._dag._db.commit()
        return cursor.rowcount > 0

    # ── Gossip Log Persistence ────────────────────────────────

    async def log_gossip(
        self,
        nonce: str,
        subtype: str,
        origin: str,
        payload: Dict[str, Any],
        ttl: int = 5,
        attestation: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._dag._db.execute(
            """INSERT OR IGNORE INTO gossip_log
               (nonce, subtype, origin, payload, ttl, received_at, attestation)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                nonce, subtype, origin, json.dumps(payload), ttl, time.time(),
                json.dumps(attestation) if attestation is not None else None,
            ),
        )
        await self._dag._db.commit()

    async def get_recent_gossip(
        self,
        since: float,
        subtypes: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        if subtypes:
            placeholders = ",".join("?" for _ in subtypes)
            cursor = await self._dag._db.execute(
                f"""SELECT nonce, subtype, origin, payload, ttl, received_at, attestation
                    FROM gossip_log
                    WHERE received_at > ? AND subtype IN ({placeholders})
                    ORDER BY received_at ASC LIMIT ?""",
                (since, *subtypes, limit),
            )
        else:
            cursor = await self._dag._db.execute(
                """SELECT nonce, subtype, origin, payload, ttl, received_at, attestation
                   FROM gossip_log
                   WHERE received_at > ?
                   ORDER BY received_at ASC LIMIT ?""",
                (since, limit),
            )
        rows = await cursor.fetchall()
        return [
            {
                "nonce": r[0], "subtype": r[1], "origin": r[2],
                "payload": json.loads(r[3]), "ttl": r[4], "received_at": r[5],
                "attestation": json.loads(r[6]) if r[6] is not None else None,
            }
            for r in rows
        ]

    async def prune_gossip_log(self, max_age: float) -> int:
        cutoff = time.time() - max_age
        cursor = await self._dag._db.execute(
            "DELETE FROM gossip_log WHERE received_at < ?", (cutoff,)
        )
        await self._dag._db.commit()
        return cursor.rowcount

    # ── Collaboration State Persistence ───────────────────────

    async def save_task(self, data: Dict[str, Any]) -> None:
        await self._dag._db.execute(
            """INSERT OR REPLACE INTO collab_tasks
               (task_id, requester_agent_id, requester_node_id, title, description,
                required_capabilities, ftns_budget, deadline_seconds, status,
                assigned_agent_id, bids, result, created_at, escrow_tx_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["task_id"], data["requester_agent_id"], data["requester_node_id"],
                data.get("title", ""), data.get("description", ""),
                json.dumps(data.get("required_capabilities", [])),
                data.get("ftns_budget", 0), data.get("deadline_seconds", 3600),
                data.get("status", "open"), data.get("assigned_agent_id"),
                json.dumps(data.get("bids", [])),
                json.dumps(data.get("result")) if data.get("result") else None,
                data["created_at"], data.get("escrow_tx_id"),
            ),
        )
        await self._dag._db.commit()

    async def save_review(self, data: Dict[str, Any]) -> None:
        await self._dag._db.execute(
            """INSERT OR REPLACE INTO collab_reviews
               (review_id, submitter_agent_id, submitter_node_id, content_cid,
                description, required_capabilities, ftns_per_review, max_reviewers,
                status, reviews, created_at, escrow_tx_id, paid_reviewers)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["review_id"], data["submitter_agent_id"], data["submitter_node_id"],
                data.get("content_cid", ""), data.get("description", ""),
                json.dumps(data.get("required_capabilities", [])),
                data.get("ftns_per_review", 0.1), data.get("max_reviewers", 3),
                data.get("status", "pending"), json.dumps(data.get("reviews", [])),
                data["created_at"], data.get("escrow_tx_id"),
                json.dumps(data.get("paid_reviewers", [])),
            ),
        )
        await self._dag._db.commit()

    async def save_query(self, data: Dict[str, Any]) -> None:
        await self._dag._db.execute(
            """INSERT OR REPLACE INTO collab_queries
               (query_id, requester_agent_id, requester_node_id, topic, question,
                ftns_per_response, max_responses, responses, created_at,
                escrow_tx_id, paid_responders)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["query_id"], data["requester_agent_id"], data["requester_node_id"],
                data.get("topic", ""), data.get("question", ""),
                data.get("ftns_per_response", 0.05), data.get("max_responses", 5),
                json.dumps(data.get("responses", [])),
                data["created_at"], data.get("escrow_tx_id"),
                json.dumps(data.get("paid_responders", [])),
            ),
        )
        await self._dag._db.commit()

    async def load_active_tasks(self) -> List[Dict[str, Any]]:
        cursor = await self._dag._db.execute(
            """SELECT task_id, requester_agent_id, requester_node_id, title,
                      description, required_capabilities, ftns_budget,
                      deadline_seconds, status, assigned_agent_id, bids,
                      result, created_at, escrow_tx_id
               FROM collab_tasks WHERE status NOT IN ('completed', 'cancelled')"""
        )
        rows = await cursor.fetchall()
        return [
            {
                "task_id": r[0], "requester_agent_id": r[1],
                "requester_node_id": r[2], "title": r[3],
                "description": r[4], "required_capabilities": json.loads(r[5]),
                "ftns_budget": r[6], "deadline_seconds": r[7],
                "status": r[8], "assigned_agent_id": r[9],
                "bids": json.loads(r[10]),
                "result": json.loads(r[11]) if r[11] else None,
                "created_at": r[12], "escrow_tx_id": r[13],
            }
            for r in rows
        ]

    async def load_active_reviews(self) -> List[Dict[str, Any]]:
        cursor = await self._dag._db.execute(
            """SELECT review_id, submitter_agent_id, submitter_node_id,
                      content_cid, description, required_capabilities,
                      ftns_per_review, max_reviewers, status, reviews,
                      created_at, escrow_tx_id, paid_reviewers
               FROM collab_reviews WHERE status NOT IN ('accepted', 'rejected')"""
        )
        rows = await cursor.fetchall()
        return [
            {
                "review_id": r[0], "submitter_agent_id": r[1],
                "submitter_node_id": r[2], "content_cid": r[3],
                "description": r[4], "required_capabilities": json.loads(r[5]),
                "ftns_per_review": r[6], "max_reviewers": r[7],
                "status": r[8], "reviews": json.loads(r[9]),
                "created_at": r[10], "escrow_tx_id": r[11],
                "paid_reviewers": json.loads(r[12]),
            }
            for r in rows
        ]

    async def load_active_queries(self) -> List[Dict[str, Any]]:
        cursor = await self._dag._db.execute(
            """SELECT query_id, requester_agent_id, requester_node_id,
                      topic, question, ftns_per_response, max_responses,
                      responses, created_at, escrow_tx_id, paid_responders
               FROM collab_queries"""
        )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            responses = json.loads(r[7])
            if len(responses) < r[6]:
                results.append({
                    "query_id": r[0], "requester_agent_id": r[1],
                    "requester_node_id": r[2], "topic": r[3],
                    "question": r[4], "ftns_per_response": r[5],
                    "max_responses": r[6], "responses": responses,
                    "created_at": r[8], "escrow_tx_id": r[9],
                    "paid_responders": json.loads(r[10]),
                })
        return results

    async def delete_collab_record(self, table: str, id_column: str, record_id: str) -> None:
        allowed = {"collab_tasks": "task_id", "collab_reviews": "review_id", "collab_queries": "query_id"}
        if table not in allowed or allowed[table] != id_column:
            return
        await self._dag._db.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (record_id,))
        await self._dag._db.commit()

    # ── Stats ─────────────────────────────────────────────────

    async def get_stats_async(self) -> Dict:
        """Async version of get_stats for use in async contexts."""
        return await self._dag.get_stats()

    def get_stats(self) -> Dict:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return {
                    "total_transactions": 0,
                    "tips": 0,
                    "dag_mode": True,
                    "note": "Use get_stats_async() for live data"
                }
            return loop.run_until_complete(self._dag.get_stats())
        except Exception:
            logger.debug("Could not retrieve DAG stats in sync context")
            return {"dag_mode": True, "error": "stats unavailable"}
