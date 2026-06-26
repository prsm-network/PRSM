"""
Tests for Settler Registry — Phase 6 Governance & Staking
=========================================================

Tests the simple L2-style staking system for batch settlement security:
- The Bond: Settlers stake FTNS to earn settlement rights
- The Multi-Sig: Require 3-of-N signatures before settlement
- The Challenge: Export ledger for public audit
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from prsm.node.settler_registry import (
    SettlerRegistry,
    Settler,
    SettlerStatus,
    PendingBatch,
    BatchSignature,
    DEFAULT_MIN_SETTLER_BOND,
    DEFAULT_SETTLEMENT_THRESHOLD,
)


class MockFTNSService:
    """Mock FTNS service for testing."""
    
    def __init__(self):
        self.locked = {}
        self.balances = {}
    
    async def lock_tokens(self, user_id: str, amount: Decimal, reason: str = ""):
        self.locked[user_id] = self.locked.get(user_id, Decimal(0)) + amount
        return True
    
    async def unlock_tokens(self, user_id: str, amount: Decimal, reason: str = ""):
        self.locked[user_id] = self.locked.get(user_id, Decimal(0)) - amount
        return True
    
    async def get_available_balance(self, user_id: str) -> Decimal:
        return self.balances.get(user_id, Decimal(100000))


@pytest.fixture
def registry():
    """Create a settler registry for testing."""
    return SettlerRegistry(
        min_settler_bond=1000.0,
        settlement_threshold=2,
        max_settlers=5,
    )


@pytest.fixture
def registry_with_ftns():
    """Create a registry with mock FTNS service."""
    ftns = MockFTNSService()
    return SettlerRegistry(
        min_settler_bond=1000.0,
        settlement_threshold=2,
        max_settlers=5,
        ftns_service=ftns,
    )


class TestSettlerRegistration:
    """Tests for settler registration and bonding."""
    
    @pytest.mark.asyncio
    async def test_register_settler(self, registry):
        """Test basic settler registration."""
        settler = await registry.register_settler(
            settler_id="node-1",
            address="0xABC123DEF456789ABC123DEF456789ABC123DEF",
            bond_amount=1000.0,
        )
        
        assert settler.settler_id == "node-1"
        assert settler.address == "0xabc123def456789abc123def456789abc123def"
        assert settler.bond_amount == 1000.0
        assert settler.status == SettlerStatus.ACTIVE
        assert settler.can_settle is True
    
    @pytest.mark.asyncio
    async def test_register_below_minimum(self, registry):
        """Test that registration fails below minimum bond."""
        with pytest.raises(ValueError, match="below minimum"):
            await registry.register_settler(
                settler_id="node-1",
                address="0xABC",
                bond_amount=500.0,
            )
    
    @pytest.mark.asyncio
    async def test_register_duplicate_address(self, registry):
        """Test that duplicate address is rejected."""
        await registry.register_settler(
            settler_id="node-1",
            address="0xAAA",
            bond_amount=1000.0,
        )
        
        with pytest.raises(ValueError, match="already registered"):
            await registry.register_settler(
                settler_id="node-2",
                address="0xaaa",  # Same address, different case
                bond_amount=1000.0,
            )
    
    @pytest.mark.asyncio
    async def test_max_settlers_limit(self, registry):
        """Test that max settlers limit is enforced."""
        for i in range(5):
            await registry.register_settler(
                settler_id=f"node-{i}",
                address=f"0x{i:040x}",
                bond_amount=1000.0,
            )
        
        with pytest.raises(ValueError, match="Maximum settlers"):
            await registry.register_settler(
                settler_id="node-6",
                address="0x" + "F" * 40,
                bond_amount=1000.0,
            )
    
    @pytest.mark.asyncio
    async def test_ftns_lock_on_register(self, registry_with_ftns):
        """Test that FTNS are locked on registration."""
        registry = registry_with_ftns
        
        await registry.register_settler(
            settler_id="node-1",
            address="0xABC",
            bond_amount=1500.0,
        )
        
        assert registry.ftns_service.locked.get("node-1") == Decimal("1500.0")


class TestSettlerUnbonding:
    """Tests for unbonding and withdrawal."""
    
    @pytest.mark.asyncio
    async def test_unbond_settler(self, registry):
        """Test unbonding initiation."""
        await registry.register_settler(
            settler_id="node-1",
            address="0xABC",
            bond_amount=1000.0,
        )
        
        unbond_at = await registry.unbond_settler("node-1")
        
        settler = registry.get_settler("node-1")
        assert settler.status == SettlerStatus.UNBONDING
        assert unbond_at is not None
        assert settler.can_settle is False
    
    @pytest.mark.asyncio
    async def test_cannot_unbond_nonexistent(self, registry):
        """Test that unbonding nonexistent settler fails."""
        with pytest.raises(ValueError, match="not found"):
            await registry.unbond_settler("nonexistent")


class TestMultiSigBatchApproval:
    """Tests for multi-signature batch approval."""

    @pytest.fixture(autouse=True)
    def _allow_unsigned(self, monkeypatch):
        # sp1277 — these tests exercise the QUORUM mechanics (counts, threshold, duplicate,
        # settled), not signature verification, using placeholder signatures. Opt into the
        # DEV-ONLY unsigned path so they keep testing quorum; the real signature-verification
        # contract is covered by test_sprint_1277_settler_signature_verification.
        monkeypatch.setenv("PRSM_ALLOW_UNSIGNED_SETTLER_BATCH", "1")

    @pytest.mark.asyncio
    async def test_propose_batch(self, registry):
        """Test batch proposal."""
        transfers = [
            {"to": "0xA", "amount": 1.0, "tx_id": "tx1"},
            {"to": "0xB", "amount": 2.0, "tx_id": "tx2"},
        ]
        
        batch = await registry.propose_batch(transfers)
        
        assert batch.batch_id
        assert batch.batch_hash
        assert len(batch.transfers) == 2
        assert batch.total_amount == 3.0
        assert batch.signature_count == 0
    
    @pytest.mark.asyncio
    async def test_sign_batch(self, registry):
        """Test batch signing."""
        # Register settlers
        await registry.register_settler("s1", "0x1" + "0" * 39, 1000.0)
        await registry.register_settler("s2", "0x2" + "0" * 39, 1000.0)
        
        # Propose batch
        batch = await registry.propose_batch([{"to": "0xA", "amount": 1.0}])
        
        # Sign
        sig = await registry.sign_batch(batch.batch_id, "s1", "sig1")
        
        assert sig.settler_id == "s1"
        assert batch.signature_count == 1
        assert registry.is_batch_approved(batch.batch_id) is False
    
    @pytest.mark.asyncio
    async def test_threshold_approval(self, registry):
        """Test that threshold approval triggers callback."""
        registry.settlement_threshold = 2
        
        # Register settlers
        await registry.register_settler("s1", "0x1" + "0" * 39, 1000.0)
        await registry.register_settler("s2", "0x2" + "0" * 39, 1000.0)
        
        # Propose batch
        batch = await registry.propose_batch([{"to": "0xA", "amount": 1.0}])
        
        # Track callback
        callback_called = []
        async def on_ready(b):
            callback_called.append(b.batch_id)
        registry.on_settlement_ready(on_ready)
        
        # First signature
        await registry.sign_batch(batch.batch_id, "s1", "sig1")
        assert len(callback_called) == 0
        
        # Second signature (threshold reached)
        await registry.sign_batch(batch.batch_id, "s2", "sig2")
        assert registry.is_batch_approved(batch.batch_id) is True
        assert callback_called == [batch.batch_id]
    
    @pytest.mark.asyncio
    async def test_cannot_sign_twice(self, registry):
        """Test that same settler cannot sign twice."""
        await registry.register_settler("s1", "0x1" + "0" * 39, 1000.0)
        batch = await registry.propose_batch([{"to": "0xA", "amount": 1.0}])
        
        await registry.sign_batch(batch.batch_id, "s1", "sig1")
        
        with pytest.raises(ValueError, match="already signed"):
            await registry.sign_batch(batch.batch_id, "s1", "sig2")
    
    @pytest.mark.asyncio
    async def test_inactive_settler_cannot_sign(self, registry):
        """Test that inactive settler cannot sign."""
        await registry.register_settler("s1", "0x1" + "0" * 39, 1000.0)
        batch = await registry.propose_batch([{"to": "0xA", "amount": 1.0}])
        
        await registry.unbond_settler("s1")
        
        with pytest.raises(ValueError, match="not authorized"):
            await registry.sign_batch(batch.batch_id, "s1", "sig1")


class TestLedgerExport:
    """Tests for ledger export (Challenge system)."""
    
    @pytest.mark.asyncio
    async def test_export_ledger(self, registry):
        """Test ledger export."""
        await registry.register_settler("s1", "0x1" + "0" * 39, 1000.0)
        
        export = await registry.export_ledger({"test": "data"})
        
        assert "exported_at" in export
        assert "integrity_hash" in export
        assert "settlers" in export
        assert len(export["settlers"]) == 1
        assert export["test"] == "data"


class TestSlashing:
    """Tests for settler slashing."""
    
    @pytest.mark.asyncio
    async def test_propose_slash(self, registry):
        """Test slash proposal creation."""
        await registry.register_settler("s1", "0x1" + "0" * 39, 1000.0)
        
        proposal = await registry.propose_slash(
            settler_id="s1",
            slash_amount=500.0,
            reason="fraud",
            evidence={"batch_id": "abc"},
            proposer_id="governance",
        )
        
        assert proposal.settler_id == "s1"
        assert proposal.slash_amount == 500.0
        assert proposal.reason == "fraud"
        assert proposal.executed is False
    
    @pytest.mark.asyncio
    async def test_execute_slash(self, registry):
        """Test slash execution."""
        await registry.register_settler("s1", "0x1" + "0" * 39, 1000.0)
        
        proposal = await registry.propose_slash(
            settler_id="s1",
            slash_amount=500.0,
            reason="fraud",
            evidence={},
            proposer_id="governance",
        )
        
        slashed = await registry.execute_slash(proposal.proposal_id)
        
        assert slashed == 500.0
        settler = registry.get_settler("s1")
        assert settler.bond_amount == 500.0
        assert settler.slashed_amount == 500.0
    
    @pytest.mark.asyncio
    async def test_full_slash_removes_settler(self, registry):
        """Test that full slash removes settler from active list."""
        await registry.register_settler("s1", "0x1" + "0" * 39, 1000.0)
        
        proposal = await registry.propose_slash(
            settler_id="s1",
            slash_amount=2000.0,  # More than bond
            reason="major fraud",
            evidence={},
            proposer_id="governance",
        )
        
        await registry.execute_slash(proposal.proposal_id)
        
        settler = registry.get_settler("s1")
        assert settler.status == SettlerStatus.SLASHED
        assert settler.can_settle is False


class TestStats:
    """Tests for registry statistics."""
    
    @pytest.mark.asyncio
    async def test_stats(self, registry):
        """Test stats endpoint."""
        await registry.register_settler("s1", "0x1" + "0" * 39, 1000.0)
        await registry.register_settler("s2", "0x2" + "0" * 39, 2000.0)
        
        stats = registry.get_stats()
        
        assert stats["active_settlers"] == 2
        assert stats["max_settlers"] == 5
        assert stats["settlement_threshold"] == 2
        assert stats["total_bonded"] == 3000.0
