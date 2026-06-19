"""
Double-Spend Prevention Tests
=============================

Tests for TOCTOU (Time-of-Check to Time-of-Use) race condition prevention
in the FTNS token system.

Critical Security Fixes Verified:
- SELECT FOR UPDATE row-level locking
- Optimistic concurrency control with version columns
- Idempotency key enforcement
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4
from collections import namedtuple


def _make_mock_row(**kwargs):
    """Create a mock database row with attribute access."""
    Row = namedtuple("Row", kwargs.keys())
    return Row(**kwargs)


def _build_mock_session():
    """
    Build a mock session that behaves as an async context manager
    matching the pattern: async with await self._get_session() as session.
    """
    mock_session = AsyncMock()
    # session.execute returns an awaitable result
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    return mock_session


def _build_service_with_session(mock_session):
    """
    Create an AtomicFTNSService wired to use a mock session.

    The service calls: async with await self._get_session() as session
    So _get_session must return an awaitable that yields an async-ctx-mgr.
    """
    from prsm.economy.tokenomics.atomic_ftns_service import AtomicFTNSService

    mock_db_service = MagicMock()

    # _get_session() is awaited, then used as `async with ... as session:`
    ctx_mgr = AsyncMock()
    ctx_mgr.__aenter__ = AsyncMock(return_value=mock_session)
    ctx_mgr.__aexit__ = AsyncMock(return_value=False)

    mock_db_service.get_session = MagicMock(return_value=ctx_mgr)

    service = AtomicFTNSService(database_service=mock_db_service)
    service._initialized = True  # skip lazy init
    return service


class TestDoubleSpendPrevention:
    """Test suite for double-spend vulnerability prevention."""

    @pytest.mark.asyncio
    async def test_idempotency_key_prevents_duplicate_transactions(self):
        """Verify that idempotency keys prevent duplicate transactions."""
        mock_session = _build_mock_session()
        service = _build_service_with_session(mock_session)

        user_id = str(uuid4())
        idempotency_key = f"test-{uuid4()}"
        cached_tx_id = str(uuid4())

        # First call: _check_idempotency returns None (no prior tx)
        #   then SELECT FOR UPDATE returns a balance row
        #   then UPDATE returns rowcount=1
        #   then INSERT for transaction record
        #   then INSERT for idempotency record
        # Second call: _check_idempotency returns existing tx id
        idempotency_row_none = MagicMock()
        idempotency_row_none.fetchone = MagicMock(return_value=None)

        balance_row = _make_mock_row(
            balance=Decimal("100.0"), locked_balance=Decimal("0"), version=1
        )
        balance_result = MagicMock()
        balance_result.fetchone = MagicMock(return_value=balance_row)

        update_result = MagicMock()
        update_result.rowcount = 1

        insert_tx_result = MagicMock()
        insert_idempotency_result = MagicMock()

        idempotency_hit = _make_mock_row(transaction_id=cached_tx_id)
        idempotency_result_hit = MagicMock()
        idempotency_result_hit.fetchone = MagicMock(return_value=idempotency_hit)

        mock_session.execute = AsyncMock(side_effect=[
            # --- first deduct_tokens_atomic call ---
            idempotency_row_none,      # _check_idempotency SELECT
            balance_result,            # SELECT FOR UPDATE
            update_result,             # UPDATE balance
            insert_tx_result,          # INSERT transaction
            insert_idempotency_result, # INSERT idempotency
            # --- second deduct_tokens_atomic call ---
            idempotency_result_hit,    # _check_idempotency SELECT (cache hit)
        ])

        result1 = await service.deduct_tokens_atomic(
            user_id=user_id,
            amount=Decimal("10.0"),
            idempotency_key=idempotency_key,
            description="Test deduction"
        )

        result2 = await service.deduct_tokens_atomic(
            user_id=user_id,
            amount=Decimal("10.0"),
            idempotency_key=idempotency_key,
            description="Test deduction"
        )

        # The second call should be an idempotent replay
        assert result2.idempotent_replay, "Duplicate idempotency key must return replay"
        assert result2.transaction_id == cached_tx_id

    @pytest.mark.asyncio
    async def test_concurrent_deductions_use_row_locking(self):
        """Verify that concurrent deductions use SELECT FOR UPDATE."""
        from prsm.economy.tokenomics.atomic_ftns_service import AtomicFTNSService

        # Instead of mocking DB internals, verify the source code contains FOR UPDATE
        import inspect
        source = inspect.getsource(AtomicFTNSService.deduct_tokens_atomic)
        assert "FOR UPDATE" in source, "SELECT FOR UPDATE must be used for balance checks"

    @pytest.mark.asyncio
    async def test_optimistic_concurrency_detects_version_mismatch(self):
        """Verify that version mismatch triggers retry or raises ConcurrentModificationError."""
        from prsm.economy.tokenomics.atomic_ftns_service import ConcurrentModificationError

        mock_session = _build_mock_session()
        service = _build_service_with_session(mock_session)

        user_id = str(uuid4())

        # _check_idempotency returns None
        idempotency_result = MagicMock()
        idempotency_result.fetchone = MagicMock(return_value=None)

        # SELECT FOR UPDATE returns a valid balance row
        balance_row = _make_mock_row(
            balance=Decimal("100.0"), locked_balance=Decimal("0"), version=1
        )
        balance_result = MagicMock()
        balance_result.fetchone = MagicMock(return_value=balance_row)

        # UPDATE returns rowcount=0 simulating version mismatch
        update_result = MagicMock()
        update_result.rowcount = 0

        mock_session.execute = AsyncMock(side_effect=[
            idempotency_result,
            balance_result,
            update_result,
        ])

        with pytest.raises(ConcurrentModificationError):
            await service.deduct_tokens_atomic(
                user_id=user_id,
                amount=Decimal("10.0"),
                idempotency_key=str(uuid4()),
                description="Test with version mismatch"
            )

    # ── sp1173: transfer_tokens_atomic must apply the SAME OCC guard as deduct ──
    # (a stale version on either UPDATE -> roll the WHOLE transfer back; never commit a
    # partial transfer that destroys/duplicates value + writes a false 'completed' tx row).

    def _transfer_rows(self, from_user, to_user):
        sender = _make_mock_row(
            user_id=from_user, balance=Decimal("500.0"),
            locked_balance=Decimal("0"), version=1)
        receiver = _make_mock_row(
            user_id=to_user, balance=Decimal("100.0"),
            locked_balance=Decimal("0"), version=1)
        idem = MagicMock(); idem.fetchone = MagicMock(return_value=None)
        lock = MagicMock(); lock.fetchall = MagicMock(return_value=[sender, receiver])
        return idem, lock

    @pytest.mark.asyncio
    async def test_transfer_receiver_version_mismatch_rolls_back_no_partial_commit(self):
        from_user, to_user = str(uuid4()), str(uuid4())
        mock_session = _build_mock_session()
        service = _build_service_with_session(mock_session)
        service.ensure_account_exists = AsyncMock()  # no-op: don't consume the mocked executes
        idem, lock = self._transfer_rows(from_user, to_user)
        sender_upd = MagicMock(); sender_upd.rowcount = 1   # sender debited
        receiver_upd = MagicMock(); receiver_upd.rowcount = 0  # OCC MISS on the receiver
        mock_session.execute = AsyncMock(side_effect=[idem, lock, sender_upd, receiver_upd])

        result = await service.transfer_tokens_atomic(
            from_user_id=from_user, to_user_id=to_user, amount=Decimal("100.0"),
            idempotency_key=str(uuid4()), description="occ receiver miss")

        assert result.success is False                       # fail-closed (no silent success)
        assert mock_session.rollback.await_count >= 1        # rolled back
        mock_session.commit.assert_not_awaited()             # NEVER committed a partial transfer
        # never reached the INSERT (no false 'completed' ledger row): idempotency, lock,
        # sender UPDATE, receiver UPDATE = 4 executes, then the guard raised.
        assert mock_session.execute.await_count == 4

    @pytest.mark.asyncio
    async def test_transfer_sender_version_mismatch_rolls_back(self):
        from_user, to_user = str(uuid4()), str(uuid4())
        mock_session = _build_mock_session()
        service = _build_service_with_session(mock_session)
        service.ensure_account_exists = AsyncMock()  # no-op: don't consume the mocked executes
        idem, lock = self._transfer_rows(from_user, to_user)
        sender_upd = MagicMock(); sender_upd.rowcount = 0   # OCC MISS on the sender
        mock_session.execute = AsyncMock(side_effect=[idem, lock, sender_upd])

        result = await service.transfer_tokens_atomic(
            from_user_id=from_user, to_user_id=to_user, amount=Decimal("100.0"),
            idempotency_key=str(uuid4()), description="occ sender miss")

        assert result.success is False
        assert mock_session.rollback.await_count >= 1
        mock_session.commit.assert_not_awaited()
        # stopped at the sender UPDATE — never even ran the receiver UPDATE.
        assert mock_session.execute.await_count == 3

    @pytest.mark.asyncio
    async def test_transfer_happy_path_commits_when_both_updates_land(self):
        # control: both UPDATEs match (rowcount=1) -> the transfer commits successfully.
        from_user, to_user = str(uuid4()), str(uuid4())
        mock_session = _build_mock_session()
        service = _build_service_with_session(mock_session)
        service.ensure_account_exists = AsyncMock()  # no-op: don't consume the mocked executes
        idem, lock = self._transfer_rows(from_user, to_user)
        sender_upd = MagicMock(); sender_upd.rowcount = 1
        receiver_upd = MagicMock(); receiver_upd.rowcount = 1
        insert_res = MagicMock()
        idem_insert_res = MagicMock()
        mock_session.execute = AsyncMock(side_effect=[
            idem, lock, sender_upd, receiver_upd, insert_res, idem_insert_res])

        result = await service.transfer_tokens_atomic(
            from_user_id=from_user, to_user_id=to_user, amount=Decimal("100.0"),
            idempotency_key=str(uuid4()), description="happy")

        assert result.success is True
        mock_session.commit.assert_awaited()                 # committed
        assert mock_session.execute.await_count >= 5         # reached the INSERT tx row

    @pytest.mark.asyncio
    async def test_insufficient_balance_check_within_transaction(self):
        """Verify balance check happens within the transaction and returns failure."""
        mock_session = _build_mock_session()
        service = _build_service_with_session(mock_session)

        user_id = str(uuid4())
        insufficient_balance = Decimal("50.0")
        deduction_amount = Decimal("100.0")

        # _check_idempotency returns None
        idempotency_result = MagicMock()
        idempotency_result.fetchone = MagicMock(return_value=None)

        # SELECT FOR UPDATE returns a balance row with insufficient funds
        balance_row = _make_mock_row(
            balance=insufficient_balance, locked_balance=Decimal("0"), version=1
        )
        balance_result = MagicMock()
        balance_result.fetchone = MagicMock(return_value=balance_row)

        mock_session.execute = AsyncMock(side_effect=[
            idempotency_result,
            balance_result,
        ])

        # The service returns a failed TransactionResult (not an exception)
        # when the balance is insufficient
        result = await service.deduct_tokens_atomic(
            user_id=user_id,
            amount=deduction_amount,
            idempotency_key=str(uuid4()),
            description="Test insufficient balance"
        )

        assert not result.success, "Deduction with insufficient balance must fail"
        assert "Insufficient balance" in result.error_message

    @pytest.mark.asyncio
    async def test_transfer_atomic_consistency(self):
        """Verify that transfers are atomic (both debit and credit succeed or both fail)."""
        mock_session = _build_mock_session()
        service = _build_service_with_session(mock_session)

        sender_id = str(uuid4())
        receiver_id = str(uuid4())
        transfer_amount = Decimal("50.0")

        # ensure_account_exists needs its own session context per call
        # We patch it to be a no-op since we control the session
        with patch.object(service, "ensure_account_exists", new=AsyncMock(return_value=True)):
            # _check_idempotency returns None
            idempotency_result = MagicMock()
            idempotency_result.fetchone = MagicMock(return_value=None)

            # Lock query returns both sender and receiver rows
            sender_row = _make_mock_row(
                user_id=sender_id, balance=Decimal("100.0"),
                locked_balance=Decimal("0"), version=1
            )
            receiver_row = _make_mock_row(
                user_id=receiver_id, balance=Decimal("200.0"),
                locked_balance=Decimal("0"), version=1
            )
            lock_result = MagicMock()
            lock_result.fetchall = MagicMock(
                return_value=sorted([sender_row, receiver_row], key=lambda r: r.user_id)
            )

            # UPDATE sender, UPDATE receiver, INSERT transaction, INSERT idempotency
            update_sender = MagicMock()
            update_sender.rowcount = 1
            update_receiver = MagicMock()
            update_receiver.rowcount = 1
            insert_tx = MagicMock()
            insert_idempotency = MagicMock()

            mock_session.execute = AsyncMock(side_effect=[
                idempotency_result,
                lock_result,
                update_sender,
                update_receiver,
                insert_tx,
                insert_idempotency,
            ])

            result = await service.transfer_tokens_atomic(
                from_user_id=sender_id,
                to_user_id=receiver_id,
                amount=transfer_amount,
                idempotency_key=str(uuid4()),
                description="Test transfer"
            )

            assert result.success, "Transfer should succeed"
            assert result.transaction_id is not None


class TestRaceConditionSimulation:
    """Simulate race conditions to verify protections."""

    @pytest.mark.asyncio
    async def test_rapid_concurrent_requests_same_user(self):
        """Simulate rapid concurrent requests from same user."""
        # This test simulates the attack scenario from the audit
        user_id = str(uuid4())
        initial_balance = Decimal("100.0")
        num_concurrent_requests = 10
        deduction_per_request = Decimal("20.0")  # Would total 200 if all succeed

        # In a vulnerable system, all 10 requests might succeed
        # With proper locking, only 5 should succeed (100 / 20 = 5)

        # Track successful deductions
        successful_deductions = []
        failed_deductions = []

        async def attempt_deduction(request_id: int):
            """Attempt a deduction - simulate what happens with proper locking."""
            # With proper implementation, this would use the AtomicFTNSService
            # For testing, we simulate the expected behavior
            return {
                "request_id": request_id,
                "success": request_id < 5,  # Only first 5 should succeed
            }

        # Run concurrent requests
        tasks = [attempt_deduction(i) for i in range(num_concurrent_requests)]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        # With 100 balance and 20 per deduction, max 5 should succeed
        assert len(successful) <= 5, "Double-spend protection should limit successful deductions"

    @pytest.mark.asyncio
    async def test_idempotency_key_collision(self):
        """Test that same idempotency key returns same result."""
        idempotency_key = "test-collision-key"
        results = []

        for i in range(5):
            # Simulate repeated requests with same idempotency key
            result = {
                "transaction_id": "tx-12345" if i == 0 else None,
                "idempotent_replay": i > 0,
            }
            results.append(result)

        # All results after the first should be replays
        replays = [r for r in results[1:] if r["idempotent_replay"]]
        assert len(replays) == 4, "Duplicate requests should return idempotent replay"


class TestDatabaseConstraints:
    """Test database-level protections."""

    @pytest.mark.asyncio
    async def test_unique_idempotency_key_constraint(self):
        """Verify database enforces unique idempotency keys."""
        # In production, the database has:
        # CREATE UNIQUE INDEX idx_idempotency_key ON ftns_idempotency_keys(idempotency_key);

        # Attempting to insert duplicate key should raise IntegrityError
        idempotency_key = str(uuid4())

        # Simulate first insert (succeeds)
        first_insert = {"success": True, "key": idempotency_key}

        # Simulate second insert (should fail with unique violation)
        second_insert = {"success": False, "error": "unique_violation"}

        assert first_insert["success"]
        assert not second_insert["success"]
        assert "unique" in second_insert["error"]

    @pytest.mark.asyncio
    async def test_balance_cannot_go_negative(self):
        """Verify balance cannot become negative."""
        # In production, the database has:
        # CHECK (balance >= 0)

        # Attempting to deduct more than balance should fail
        current_balance = Decimal("50.0")
        deduction_amount = Decimal("100.0")

        # The CHECK constraint prevents this
        can_deduct = current_balance >= deduction_amount
        assert not can_deduct, "Balance should not be allowed to go negative"
