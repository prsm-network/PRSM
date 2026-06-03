"""
Local FTNS Ledger
=================

SQLite-backed ledger for FTNS token accounting.
Zero-config — no PostgreSQL required. Each node maintains its own ledger
which is reconciled via gossip when connected to the network.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from prsm.node.ledger_node_services import LedgerNodeServicesMixin

import aiosqlite


class TransactionType(str, Enum):
    WELCOME_GRANT = "welcome_grant"
    COMPUTE_PAYMENT = "compute_payment"
    COMPUTE_EARNING = "compute_earning"
    STORAGE_REWARD = "storage_reward"
    CONTENT_ROYALTY = "content_royalty"
    TRANSFER = "transfer"
    # Sprint 540 Pattern A — daemon-mediated bridge (no separate
    # contract). DEPOSIT credits off-chain balance after detected
    # on-chain inbound to escrow address; WITHDRAW debits off-chain
    # before signing on-chain transfer out.
    BRIDGE_DEPOSIT = "bridge_deposit"
    BRIDGE_WITHDRAW = "bridge_withdraw"
    # Sprint 572 F24 — generic reward credit used by:
    # - node.mint_tokens (staking rewards, appeal refunds)
    # - bittorrent_provider._reward_loop (seeder per-GB rewards)
    # - bittorrent_proofs.award_seeder (storage-proof verification)
    # Distinct from STORAGE_REWARD which encodes "storage providers
    # paid for capacity"; REWARD is the catch-all for other emissions.
    REWARD = "reward"


@dataclass
class Transaction:
    tx_id: str
    tx_type: TransactionType
    from_wallet: Optional[str]     # None for grants/rewards
    to_wallet: str
    amount: float
    description: str
    timestamp: float
    signature: Optional[str] = None


class LocalLedger(LedgerNodeServicesMixin):
    """SQLite-backed FTNS ledger for a single node.

    The ledger is append-only for auditability. Balances are derived from
    the transaction log.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        # sp929 — serialize balance-DECREASING operations. Balances are derived
        # from the append-only log, so debit/transfer/agent_debit read the
        # balance (await) and then insert the debit (await) — asyncio yields at
        # each await, so two concurrent decreasers on the same wallet would both
        # read the old balance, both pass the check, and both insert, driving
        # the balance negative (spend FTNS that doesn't exist). This lock makes
        # the check+insert atomic. Credits are additive and never overspend, so
        # they do not take the lock.
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Open database and create tables."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()
        # Sprint 540 Pattern A migration: existing DBs predate the
        # `eth_address` column. ALTER wrapped in try/except so
        # post-migration init stays idempotent (same pattern as
        # sprint-501 onchain_transactions + sprint-528 F43 fix).
        try:
            await self._db.execute(
                "ALTER TABLE wallets ADD COLUMN eth_address TEXT"
            )
            await self._db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_wallets_eth_address ON wallets(eth_address) "
                "WHERE eth_address IS NOT NULL"
            )
            await self._db.commit()
        except Exception:
            pass  # column already exists
        # Sprint 554 user-sig groundwork: per-wallet opt-in flag +
        # monotonic nonce counter. Same idempotent-ALTER pattern as
        # the sprint-540 eth_address column above.
        try:
            await self._db.execute(
                "ALTER TABLE wallets ADD COLUMN "
                "requires_user_signature INTEGER NOT NULL DEFAULT 0"
            )
            await self._db.commit()
        except Exception:
            pass
        try:
            await self._db.execute(
                "ALTER TABLE wallets ADD COLUMN "
                "next_withdraw_nonce INTEGER NOT NULL DEFAULT 0"
            )
            await self._db.commit()
        except Exception:
            pass

    async def _create_tables(self) -> None:
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS wallets (
                wallet_id    TEXT PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                created_at   REAL NOT NULL,
                -- Sprint 540 Pattern A: link to operator's on-chain
                -- wallet for bridge deposit/withdraw flows. Nullable
                -- — wallets without linkage stay off-chain-only.
                -- Unique constraint enforced via index below (SQLite
                -- doesn't allow inline UNIQUE on nullable columns
                -- without extra tooling, so we use partial index).
                eth_address  TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_wallets_eth_address
                ON wallets(eth_address)
                WHERE eth_address IS NOT NULL;

            CREATE TABLE IF NOT EXISTS transactions (
                tx_id       TEXT PRIMARY KEY,
                tx_type     TEXT NOT NULL,
                from_wallet TEXT,
                to_wallet   TEXT NOT NULL,
                amount      REAL NOT NULL CHECK(amount > 0),
                description TEXT NOT NULL DEFAULT '',
                timestamp   REAL NOT NULL,
                signature   TEXT,
                FOREIGN KEY (to_wallet) REFERENCES wallets(wallet_id)
            );

            CREATE INDEX IF NOT EXISTS idx_tx_to ON transactions(to_wallet);
            CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_wallet);
            CREATE INDEX IF NOT EXISTS idx_tx_time ON transactions(timestamp);

            CREATE TABLE IF NOT EXISTS seen_nonces (
                nonce       TEXT PRIMARY KEY,
                origin      TEXT NOT NULL,
                seen_at     REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_allowances (
                agent_id        TEXT PRIMARY KEY,
                principal_id    TEXT NOT NULL,
                allowance       REAL NOT NULL CHECK(allowance >= 0),
                spent           REAL NOT NULL DEFAULT 0 CHECK(spent >= 0),
                epoch_hours     REAL NOT NULL DEFAULT 24,
                epoch_start     REAL NOT NULL,
                created_at      REAL NOT NULL,
                revoked         INTEGER NOT NULL DEFAULT 0
            );

            -- Gossip log for persistence and catch-up
            -- sp961: `attestation` (nullable JSON) persists the origin's signed
            -- attestation (origin_time/origin_pubkey/origin_sig) so the digest
            -- catch-up path can RE-verify authorship relayer-independently. NULL
            -- for pre-sp961 rows / unsigned messages (→ catch-up falls back to
            -- the relaying sender). `payload` still stores just `data`.
            CREATE TABLE IF NOT EXISTS gossip_log (
                nonce       TEXT PRIMARY KEY,
                subtype     TEXT NOT NULL,
                origin      TEXT NOT NULL,
                payload     TEXT NOT NULL,
                ttl         INTEGER NOT NULL DEFAULT 5,
                received_at REAL NOT NULL,
                attestation TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_gossip_log_time ON gossip_log(received_at);
            CREATE INDEX IF NOT EXISTS idx_gossip_log_subtype ON gossip_log(subtype);

            -- Durable provenance registry (populated from GOSSIP_PROVENANCE_REGISTER)
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
            CREATE INDEX IF NOT EXISTS idx_prov_hash ON provenance_chains(content_hash);

            -- Collaboration persistence tables
            CREATE TABLE IF NOT EXISTS collab_tasks (
                task_id                 TEXT PRIMARY KEY,
                requester_agent_id      TEXT NOT NULL,
                requester_node_id       TEXT NOT NULL,
                title                   TEXT NOT NULL DEFAULT '',
                description             TEXT NOT NULL DEFAULT '',
                required_capabilities   TEXT NOT NULL DEFAULT '[]',
                ftns_budget             REAL NOT NULL DEFAULT 0,
                deadline_seconds        REAL NOT NULL DEFAULT 3600,
                status                  TEXT NOT NULL DEFAULT 'open',
                assigned_agent_id       TEXT,
                bids                    TEXT NOT NULL DEFAULT '[]',
                result                  TEXT,
                created_at              REAL NOT NULL,
                escrow_tx_id            TEXT
            );

            CREATE TABLE IF NOT EXISTS collab_reviews (
                review_id               TEXT PRIMARY KEY,
                submitter_agent_id      TEXT NOT NULL,
                submitter_node_id       TEXT NOT NULL,
                content_cid             TEXT NOT NULL DEFAULT '',
                description             TEXT NOT NULL DEFAULT '',
                required_capabilities   TEXT NOT NULL DEFAULT '[]',
                ftns_per_review         REAL NOT NULL DEFAULT 0.1,
                max_reviewers           INTEGER NOT NULL DEFAULT 3,
                status                  TEXT NOT NULL DEFAULT 'pending',
                reviews                 TEXT NOT NULL DEFAULT '[]',
                created_at              REAL NOT NULL,
                escrow_tx_id            TEXT,
                paid_reviewers          TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS collab_queries (
                query_id                TEXT PRIMARY KEY,
                requester_agent_id      TEXT NOT NULL,
                requester_node_id       TEXT NOT NULL,
                topic                   TEXT NOT NULL DEFAULT '',
                question                TEXT NOT NULL DEFAULT '',
                ftns_per_response       REAL NOT NULL DEFAULT 0.05,
                max_responses           INTEGER NOT NULL DEFAULT 5,
                responses               TEXT NOT NULL DEFAULT '[]',
                created_at              REAL NOT NULL,
                escrow_tx_id            TEXT,
                paid_responders         TEXT NOT NULL DEFAULT '[]'
            );
        """)

        # sp961 — idempotent migration: add gossip_log.attestation to DBs
        # created before this column existed. `CREATE TABLE IF NOT EXISTS`
        # above only covers fresh DBs; existing ones need the ALTER. SQLite
        # raises OperationalError ("duplicate column name") when the column is
        # already present — caught + ignored so re-init is a no-op.
        try:
            await self._db.execute(
                "ALTER TABLE gossip_log ADD COLUMN attestation TEXT"
            )
            await self._db.commit()
        except Exception:
            pass

        # sp965 — idempotent migration: add provenance_chains.signed_record (the
        # verbatim signed record) so the cross-node provenance RESPONSE path can
        # re-serve a cryptographically verifiable form. NULL for pre-sp965 rows.
        try:
            await self._db.execute(
                "ALTER TABLE provenance_chains ADD COLUMN signed_record TEXT"
            )
            await self._db.commit()
        except Exception:
            pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Wallet management ────────────────────────────────────────

    async def create_wallet(self, wallet_id: str, display_name: str = "") -> None:
        """Create a wallet if it doesn't already exist."""
        await self._db.execute(
            "INSERT OR IGNORE INTO wallets (wallet_id, display_name, created_at) VALUES (?, ?, ?)",
            (wallet_id, display_name, time.time()),
        )
        await self._db.commit()

    async def wallet_exists(self, wallet_id: str) -> bool:
        cursor = await self._db.execute(
            "SELECT 1 FROM wallets WHERE wallet_id = ?", (wallet_id,)
        )
        return await cursor.fetchone() is not None

    # ── Sprint 540 Pattern A: linked-address registry ────────────

    async def link_eth_address(
        self, wallet_id: str, eth_address: str,
    ) -> None:
        """Link an Ethereum-format address to a PRSM wallet_id.

        Bridge deposits route inbound on-chain transfers from this
        address into the wallet's off-chain balance. Each address
        can be linked to at most ONE wallet_id (UNIQUE index).
        Re-linking the same address to a different wallet replaces
        the prior link.
        """
        if not wallet_id:
            raise ValueError("wallet_id must be non-empty")
        if not eth_address or not eth_address.startswith("0x"):
            raise ValueError(
                f"eth_address must be 0x-prefixed; got {eth_address!r}"
            )
        # Normalize: lowercase (we'll checksum on read paths)
        addr_norm = eth_address.lower()
        await self._ensure_wallet(wallet_id)
        # Clear any existing link for this eth_address (move semantics)
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
        """Reverse lookup: which wallet_id is linked to this ETH addr?

        Returns None if no wallet is linked.
        """
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
        """Return True iff withdraws from this wallet require a
        user-signed EIP-712 payload (sprint-556 enforcement)."""
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
        """Toggle the per-wallet user-signature requirement."""
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
        """Return the nonce expected on the NEXT signed withdraw.
        Sprint-556 verification rejects payloads whose nonce doesn't
        match this value."""
        cursor = await self._db.execute(
            "SELECT next_withdraw_nonce FROM wallets WHERE wallet_id = ?",
            (wallet_id,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def bump_withdraw_nonce(self, wallet_id: str) -> int:
        """Atomically consume the current nonce + advance the counter.
        Returns the OLD value (the nonce the caller just used)."""
        cursor = await self._db.execute(
            "SELECT next_withdraw_nonce FROM wallets WHERE wallet_id = ?",
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

    # ── Balance ──────────────────────────────────────────────────

    async def get_balance(self, wallet_id: str) -> float:
        """Derive balance from transaction log."""
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE to_wallet = ?",
            (wallet_id,),
        )
        credits = (await cursor.fetchone())[0]

        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE from_wallet = ?",
            (wallet_id,),
        )
        debits = (await cursor.fetchone())[0]
        return round(credits - debits, 6)

    # ── Transactions ─────────────────────────────────────────────

    async def _ensure_wallet(self, wallet_id: str) -> None:
        """Create wallet if it doesn't exist (for remote node wallets)."""
        if not await self.wallet_exists(wallet_id):
            await self.create_wallet(wallet_id, f"remote-{wallet_id[:8]}")

    async def credit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
        signature: Optional[str] = None,
    ) -> Transaction:
        """Credit FTNS to a wallet (from_wallet is None for system grants)."""
        if not wallet_id:
            # Phase 1.3 Task 3g pass-6: refuse empty wallet credits.
            # An empty wallet_id creates a phantom row that can't be
            # traced to a real creator. Any code path that tries to
            # credit "" is a bug; fail loudly rather than silently
            # locking FTNS to an unrecoverable key.
            raise ValueError(
                "local_ledger.credit: wallet_id must be non-empty "
                f"(attempted tx_type={tx_type}, amount={amount}, "
                f"description={description!r})"
            )
        await self._ensure_wallet(wallet_id)
        tx = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type=tx_type,
            from_wallet=None,
            to_wallet=wallet_id,
            amount=amount,
            description=description,
            timestamp=time.time(),
            signature=signature,
        )
        await self._insert_tx(tx)
        return tx

    async def debit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
    ) -> Transaction:
        """Debit FTNS from a wallet (payment to system/network)."""
        # sp929 — hold the write lock across the check+insert so two concurrent
        # debits on the same wallet can't both pass on a stale balance read.
        async with self._write_lock:
            return await self._debit_locked(wallet_id, amount, tx_type, description)

    async def _debit_locked(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
    ) -> Transaction:
        """Balance-checked debit. The caller MUST hold self._write_lock."""
        balance = await self.get_balance(wallet_id)
        if balance < amount:
            raise ValueError(
                f"Insufficient balance: {balance:.6f} < {amount:.6f}"
            )
        # The transactions table has FOREIGN KEY(to_wallet) -> wallets, so the
        # "system" sink must exist or the INSERT fails (transfer() already does
        # this for its destination via _ensure_wallet).
        await self._ensure_wallet("system")
        # Record as transfer from wallet to a system sink
        tx = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type=tx_type,
            from_wallet=wallet_id,
            to_wallet="system",
            amount=amount,
            description=description,
            timestamp=time.time(),
        )
        await self._insert_tx(tx)
        return tx

    async def transfer(
        self,
        from_wallet: str,
        to_wallet: str,
        amount: float,
        tx_type: TransactionType = TransactionType.TRANSFER,
        description: str = "",
        signature: Optional[str] = None,
    ) -> Transaction:
        """Transfer FTNS between wallets."""
        await self._ensure_wallet(to_wallet)
        # sp929 — atomic check+insert (see debit()).
        async with self._write_lock:
            balance = await self.get_balance(from_wallet)
            if balance < amount:
                raise ValueError(
                    f"Insufficient balance: {balance:.6f} < {amount:.6f}"
                )
            tx = Transaction(
                tx_id=str(uuid.uuid4()),
                tx_type=tx_type,
                from_wallet=from_wallet,
                to_wallet=to_wallet,
                amount=amount,
                description=description,
                timestamp=time.time(),
                signature=signature,
            )
            await self._insert_tx(tx)
            return tx

    async def get_transaction_history(
        self, wallet_id: str, limit: int = 50
    ) -> List[Transaction]:
        """Get recent transactions involving a wallet."""
        cursor = await self._db.execute(
            """SELECT tx_id, tx_type, from_wallet, to_wallet, amount,
                      description, timestamp, signature
               FROM transactions
               WHERE to_wallet = ? OR from_wallet = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (wallet_id, wallet_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            Transaction(
                tx_id=r[0],
                tx_type=TransactionType(r[1]),
                from_wallet=r[2],
                to_wallet=r[3],
                amount=r[4],
                description=r[5],
                timestamp=r[6],
                signature=r[7],
            )
            for r in rows
        ]

    async def get_transaction_count(self, wallet_id: str) -> int:
        """Count total transactions for a wallet."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM transactions WHERE to_wallet = ? OR from_wallet = ?",
            (wallet_id, wallet_id),
        )
        return (await cursor.fetchone())[0]

    # ── Nonce tracking (double-spend prevention) ────────────────

    async def has_seen_nonce(self, nonce: str) -> bool:
        """Check if a transaction nonce has already been processed."""
        cursor = await self._db.execute(
            "SELECT 1 FROM seen_nonces WHERE nonce = ?", (nonce,)
        )
        return await cursor.fetchone() is not None

    async def record_nonce(self, nonce: str, origin: str) -> bool:
        """Atomically CLAIM a transaction nonce. Returns True if THIS
        call inserted the nonce (i.e. won the claim), False if it was
        already present (replay or a concurrent duplicate).

        Sp898 — `seen_nonces.nonce` is a PRIMARY KEY, so this
        `INSERT OR IGNORE` is the serialization point: of N concurrent
        handlers processing the same gossiped transaction, exactly one
        gets rowcount==1 and may proceed to credit; the rest get False.
        This is the gate that closes the cross-node double-credit
        window (callers that ignore the return value are unaffected)."""
        cursor = await self._db.execute(
            "INSERT OR IGNORE INTO seen_nonces (nonce, origin, seen_at) VALUES (?, ?, ?)",
            (nonce, origin, time.time()),
        )
        await self._db.commit()
        return cursor.rowcount == 1

    async def release_nonce(self, nonce: str) -> bool:
        """Sp918 — RELEASE a previously-claimed nonce so it can be re-claimed.

        Used ONLY to undo an idempotency claim whose on-chain tx definitively
        did NOT pay (OnChainRevertedError = atomic rollback, or
        BroadcastFailedError = never reached the network), so a legitimate
        retry can re-dispatch instead of being permanently
        skipped_already_dispatched. Returns True if a row was removed (was
        claimed), False if absent (harmless no-op → idempotent).

        SAFETY: the caller MUST NOT release a claim whose tx may still confirm
        (a 'pending' broadcast) — that re-opens the double-pay window the claim
        was guarding. See onchain_content_royalty.keys_to_release."""
        cursor = await self._db.execute(
            "DELETE FROM seen_nonces WHERE nonce = ?", (nonce,)
        )
        await self._db.commit()
        return cursor.rowcount == 1

    async def get_recent_tx_ids(self, wallet_id: str, limit: int = 50) -> List[str]:
        """Get recent transaction IDs for reconciliation."""
        cursor = await self._db.execute(
            """SELECT tx_id FROM transactions
               WHERE to_wallet = ? OR from_wallet = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (wallet_id, wallet_id, limit),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def has_transaction(self, tx_id: str) -> bool:
        """Check if a transaction already exists in the ledger."""
        cursor = await self._db.execute(
            "SELECT 1 FROM transactions WHERE tx_id = ?", (tx_id,)
        )
        return await cursor.fetchone() is not None

    # ── Agent allowances (delegated payments) ───────────────────

    async def grant_agent_allowance(
        self,
        principal_id: str,
        agent_id: str,
        amount: float,
        epoch_hours: float = 24.0,
    ) -> None:
        """Grant an agent a spending allowance from the principal's wallet.

        The allowance resets at each epoch boundary (epoch_hours interval).
        """
        now = time.time()
        await self._db.execute(
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
        await self._db.commit()

    async def get_agent_allowance(self, agent_id: str) -> Optional[dict]:
        """Get an agent's current allowance state. Returns None if not found."""
        cursor = await self._db.execute(
            "SELECT principal_id, allowance, spent, epoch_hours, epoch_start, revoked "
            "FROM agent_allowances WHERE agent_id = ?",
            (agent_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        principal_id, allowance, spent, epoch_hours, epoch_start, revoked = row

        # Check if epoch has rolled over — if so, reset spent
        now = time.time()
        epoch_seconds = epoch_hours * 3600
        if now - epoch_start >= epoch_seconds:
            spent = 0.0
            epoch_start = now
            await self._db.execute(
                "UPDATE agent_allowances SET spent = 0, epoch_start = ? WHERE agent_id = ?",
                (now, agent_id),
            )
            await self._db.commit()

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
    ) -> Optional[Transaction]:
        """Debit from the principal's wallet on behalf of an agent.

        Checks the agent's remaining allowance and the principal's balance.
        Returns the transaction, or None if rejected.
        """
        # sp929 — hold the write lock across the WHOLE sequence: the allowance
        # read, the balance-checked debit, and the spent update. Otherwise two
        # concurrent agent_debits both read the same remaining-allowance and the
        # same principal balance, both pass, and both debit + bump spent —
        # overspending the principal AND over-running the allowance. _debit_locked
        # is called (not debit) because the lock is not reentrant.
        async with self._write_lock:
            allowance = await self.get_agent_allowance(agent_id)
            if not allowance or allowance["revoked"]:
                return None

            if amount > allowance["remaining"]:
                return None  # Exceeds allowance

            principal_id = allowance["principal_id"]

            # Debit from principal (raises ValueError on insufficient balance).
            try:
                tx = await self._debit_locked(
                    wallet_id=principal_id,
                    amount=amount,
                    tx_type=tx_type,
                    description=f"[agent:{agent_id[:12]}] {description}",
                )
            except ValueError:
                return None  # Insufficient principal balance

            # Update spent (under the same lock, so the next agent_debit sees it).
            await self._db.execute(
                "UPDATE agent_allowances SET spent = spent + ? WHERE agent_id = ?",
                (amount, agent_id),
            )
            await self._db.commit()
            return tx

    async def revoke_agent_allowance(self, principal_id: str, agent_id: str) -> bool:
        """Revoke an agent's spending authority. Returns True if found."""
        cursor = await self._db.execute(
            "UPDATE agent_allowances SET revoked = 1 "
            "WHERE agent_id = ? AND principal_id = ?",
            (agent_id, principal_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # ── Provenance Registry ──────────────────────────────────────

    async def list_provenance_by_creator(
        self, creator_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return all provenance records for a given creator, newest first."""
        cursor = await self._db.execute(
            """SELECT cid, content_hash, creator_id, creator_pubkey, filename,
                      size_bytes, royalty_rate, parent_cids, signature,
                      embedding_id, near_duplicate_of, metadata, registered_at
               FROM provenance_chains WHERE creator_id = ?
               ORDER BY registered_at DESC LIMIT ?""",
            (creator_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "cid": r[0], "content_hash": r[1], "creator_id": r[2],
                "creator_public_key": r[3], "filename": r[4],
                "size_bytes": r[5], "royalty_rate": r[6],
                "parent_cids": json.loads(r[7]), "signature": r[8],
                "embedding_id": r[9], "near_duplicate_of": r[10],
                "metadata": json.loads(r[11]), "registered_at": r[12],
            }
            for r in rows
        ]

    # ── Collaboration Persistence ────────────────────────────────

    async def save_task(self, data: Dict[str, Any]) -> None:
        """Upsert a collaboration task record."""
        await self._db.execute(
            """INSERT OR REPLACE INTO collab_tasks
               (task_id, requester_agent_id, requester_node_id, title, description,
                required_capabilities, ftns_budget, deadline_seconds, status,
                assigned_agent_id, bids, result, created_at, escrow_tx_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["task_id"],
                data["requester_agent_id"],
                data["requester_node_id"],
                data.get("title", ""),
                data.get("description", ""),
                json.dumps(data.get("required_capabilities", [])),
                data.get("ftns_budget", 0),
                data.get("deadline_seconds", 3600),
                data.get("status", "open"),
                data.get("assigned_agent_id"),
                json.dumps(data.get("bids", [])),
                json.dumps(data.get("result")) if data.get("result") else None,
                data["created_at"],
                data.get("escrow_tx_id"),
            ),
        )
        await self._db.commit()

    async def save_review(self, data: Dict[str, Any]) -> None:
        """Upsert a collaboration review record."""
        await self._db.execute(
            """INSERT OR REPLACE INTO collab_reviews
               (review_id, submitter_agent_id, submitter_node_id, content_cid,
                description, required_capabilities, ftns_per_review, max_reviewers,
                status, reviews, created_at, escrow_tx_id, paid_reviewers)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["review_id"],
                data["submitter_agent_id"],
                data["submitter_node_id"],
                data.get("content_cid", ""),
                data.get("description", ""),
                json.dumps(data.get("required_capabilities", [])),
                data.get("ftns_per_review", 0.1),
                data.get("max_reviewers", 3),
                data.get("status", "pending"),
                json.dumps(data.get("reviews", [])),
                data["created_at"],
                data.get("escrow_tx_id"),
                json.dumps(data.get("paid_reviewers", [])),
            ),
        )
        await self._db.commit()

    async def save_query(self, data: Dict[str, Any]) -> None:
        """Upsert a collaboration query record."""
        await self._db.execute(
            """INSERT OR REPLACE INTO collab_queries
               (query_id, requester_agent_id, requester_node_id, topic, question,
                ftns_per_response, max_responses, responses, created_at,
                escrow_tx_id, paid_responders)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["query_id"],
                data["requester_agent_id"],
                data["requester_node_id"],
                data.get("topic", ""),
                data.get("question", ""),
                data.get("ftns_per_response", 0.05),
                data.get("max_responses", 5),
                json.dumps(data.get("responses", [])),
                data["created_at"],
                data.get("escrow_tx_id"),
                json.dumps(data.get("paid_responders", [])),
            ),
        )
        await self._db.commit()

    async def load_active_tasks(self) -> List[Dict[str, Any]]:
        """Load non-terminal collaboration tasks."""
        cursor = await self._db.execute(
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
                "description": r[4],
                "required_capabilities": json.loads(r[5]),
                "ftns_budget": r[6], "deadline_seconds": r[7],
                "status": r[8], "assigned_agent_id": r[9],
                "bids": json.loads(r[10]),
                "result": json.loads(r[11]) if r[11] else None,
                "created_at": r[12], "escrow_tx_id": r[13],
            }
            for r in rows
        ]

    async def load_active_reviews(self) -> List[Dict[str, Any]]:
        """Load non-terminal collaboration reviews."""
        cursor = await self._db.execute(
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
                "description": r[4],
                "required_capabilities": json.loads(r[5]),
                "ftns_per_review": r[6], "max_reviewers": r[7],
                "status": r[8], "reviews": json.loads(r[9]),
                "created_at": r[10], "escrow_tx_id": r[11],
                "paid_reviewers": json.loads(r[12]),
            }
            for r in rows
        ]

    async def load_active_queries(self) -> List[Dict[str, Any]]:
        """Load active collaboration queries (not yet at max responses)."""
        cursor = await self._db.execute(
            """SELECT query_id, requester_agent_id, requester_node_id,
                      topic, question, ftns_per_response, max_responses,
                      responses, created_at, escrow_tx_id, paid_responders
               FROM collab_queries"""
        )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            responses = json.loads(r[7])
            max_responses = r[6]
            if len(responses) < max_responses:
                results.append({
                    "query_id": r[0], "requester_agent_id": r[1],
                    "requester_node_id": r[2], "topic": r[3],
                    "question": r[4], "ftns_per_response": r[5],
                    "max_responses": max_responses,
                    "responses": responses,
                    "created_at": r[8], "escrow_tx_id": r[9],
                    "paid_responders": json.loads(r[10]),
                })
        return results

    async def delete_collab_record(self, table: str, id_column: str, record_id: str) -> None:
        """Delete a completed/archived collaboration record from SQLite."""
        allowed = {"collab_tasks": "task_id", "collab_reviews": "review_id", "collab_queries": "query_id"}
        if table not in allowed or allowed[table] != id_column:
            return
        await self._db.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (record_id,))
        await self._db.commit()

    # ── Internal ─────────────────────────────────────────────────

    async def _insert_tx(self, tx: Transaction) -> None:
        await self._db.execute(
            """INSERT INTO transactions
               (tx_id, tx_type, from_wallet, to_wallet, amount, description, timestamp, signature)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tx.tx_id,
                tx.tx_type.value,
                tx.from_wallet,
                tx.to_wallet,
                tx.amount,
                tx.description,
                tx.timestamp,
                tx.signature,
            ),
        )
        await self._db.commit()

    async def issue_welcome_grant(self, wallet_id: str, amount: float = 100.0) -> Transaction:
        """Issue a one-time welcome grant to a new node."""
        # Check if this wallet already received a welcome grant
        cursor = await self._db.execute(
            "SELECT 1 FROM transactions WHERE to_wallet = ? AND tx_type = ?",
            (wallet_id, TransactionType.WELCOME_GRANT.value),
        )
        if await cursor.fetchone() is not None:
            raise ValueError(f"Wallet {wallet_id} already received a welcome grant")

        return await self.credit(
            wallet_id=wallet_id,
            amount=amount,
            tx_type=TransactionType.WELCOME_GRANT,
            description=f"Welcome grant of {amount} FTNS for new node",
        )
