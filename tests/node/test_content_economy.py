"""
Tests for Content Economy - Phase 4 FTNS Payment Integration
=============================================================

Tests cover:
1. FTNS payment on content access
2. Royalty distribution (8% original, 1% derivative)
3. Replication tracking and enforcement
4. Content retrieval marketplace
5. Vector DB integration
"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.node.content_economy import (
    ContentEconomy,
    ContentAccessPayment,
    ReplicationStatus,
    ProviderBid,
    RetrievalRequest,
    ProvenanceChain,
    RoyaltyModel,
    PaymentStatus,
    ORIGINAL_CREATOR_ROYALTY_RATE,
    DERIVATIVE_CREATOR_ROYALTY_RATE,
    NETWORK_FEE_RATE,
)
from prsm.node.local_ledger import LocalLedger, TransactionType


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_identity():
    """Mock node identity."""
    identity = MagicMock()
    identity.node_id = "test-node-abc123"
    identity.sign = MagicMock(return_value="signature")
    return identity


@pytest.fixture
async def ledger():
    """Create an in-memory ledger for testing."""
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    await ledger.create_wallet("test-node-abc123", "Test Node")
    await ledger.create_wallet("creator-xyz", "Creator")
    await ledger.create_wallet("original-creator", "Original")
    await ledger.create_wallet("derivative-creator", "Derivative")
    await ledger.create_wallet("system", "System")
    # Seed accessor wallet with FTNS so payments can be debited
    await ledger.credit(
        wallet_id="test-node-abc123",
        amount=100.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="Test welcome grant",
    )
    yield ledger
    await ledger.close()


@pytest.fixture
def mock_gossip():
    """Mock gossip protocol."""
    gossip = AsyncMock()
    gossip.publish = AsyncMock()
    gossip.subscribe = MagicMock()
    return gossip


@pytest.fixture
def mock_content_index():
    """Mock content index."""
    index = MagicMock()
    
    # Create some test records.
    # Phase 1.3 Task 3g pass-6: the ContentRecord field is parent_cids,
    # NOT parent_cids. The old fixture attribute name was masked
    # by MagicMock's auto-attribute behavior (it returned a truthy child
    # mock for any attribute), which hid a production AttributeError in
    # _resolve_provenance_chain. Using the real field name here keeps
    # the mocks honest.
    original_record = MagicMock(spec=["content_id", "creator_id", "parent_cids", "royalty_rate", "providers"])
    original_record.content_id = "original-cid-123"
    original_record.creator_id = "original-creator"
    original_record.parent_cids = []
    original_record.royalty_rate = 0.08
    original_record.providers = {"provider-1"}

    derivative_record = MagicMock(spec=["content_id", "creator_id", "parent_cids", "royalty_rate", "providers"])
    derivative_record.content_id = "derivative-cid-456"
    derivative_record.creator_id = "derivative-creator"
    derivative_record.parent_cids = ["original-cid-123"]
    derivative_record.royalty_rate = 0.01
    derivative_record.providers = {"provider-2"}

    # test_process_content_access_basic uses this CID with creator_id
    # "creator-xyz" and empty parents. Without a content_index record
    # for it, _resolve_provenance_chain returns None for original_creator
    # and no creator-xyz distribution fires.
    basic_record = MagicMock(spec=["content_id", "creator_id", "parent_cids", "royalty_rate", "providers"])
    basic_record.content_id = "test-cid-123"
    basic_record.creator_id = "creator-xyz"
    basic_record.parent_cids = []
    basic_record.royalty_rate = 0.01
    basic_record.providers = {"test-node-abc123"}

    def lookup(cid):
        if cid == "original-cid-123":
            return original_record
        elif cid == "derivative-cid-456":
            return derivative_record
        elif cid == "test-cid-123":
            return basic_record
        return None
    
    index.lookup = lookup
    return index


@pytest.fixture
async def content_economy(mock_identity, ledger, mock_gossip, mock_content_index):
    """Create a ContentEconomy instance for testing."""
    economy = ContentEconomy(
        identity=mock_identity,
        ledger=ledger,
        gossip=mock_gossip,
        content_index=mock_content_index,
        royalty_model=RoyaltyModel.PHASE4,
        min_replicas=3,
    )
    yield economy


# ── Payment Processing Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_content_access_basic(content_economy, ledger):
    """Test basic content access payment."""
    payment = await content_economy.process_content_access(
        content_id="test-cid-123",
        accessor_id="test-node-abc123",
        content_metadata={
            "royalty_rate": 0.01,
            "creator_id": "creator-xyz",
            "parent_cids": [],
        },
    )
    
    assert payment.status == PaymentStatus.COMPLETED
    assert payment.amount == Decimal("0.01")
    assert len(payment.royalty_distributions) > 0
    
    # Check that creator received royalty
    creator_dist = [d for d in payment.royalty_distributions if d["recipient_id"] == "creator-xyz"]
    assert len(creator_dist) > 0
    
    # Check accessor balance decreased
    balance = await ledger.get_balance("test-node-abc123")
    # Initial welcome grant minus payment
    assert balance < 100.0  # Welcome grant default


@pytest.mark.asyncio
async def test_process_content_access_uses_authenticated_rate(content_economy):
    """sp1077 (Gap B) — a forged advertise-lane royalty_rate is overridden by the
    on-chain REGISTERED rate for content with a registered provenance_hash. The
    payment amount reflects the authentic rate, not the attacker's."""
    from prsm.economy.web3.provenance_registry import ContentRecord
    client = MagicMock()
    client.get_content.return_value = ContentRecord(
        creator="0x" + "11" * 20, royalty_rate_bps=200,   # registered = 2%
        registered_at=1, metadata_uri="")
    content_economy._get_provenance_client = lambda: client

    payment = await content_economy.process_content_access(
        content_id="reg-cid",
        accessor_id="test-node-abc123",
        content_metadata={
            "royalty_rate": 0.50,                  # forged advertise rate (50%)
            "creator_id": "creator-xyz",
            "parent_cids": [],
            "provenance_hash": "0x" + "ab" * 32,
        },
    )
    assert payment.status == PaymentStatus.COMPLETED
    assert payment.amount == Decimal("0.02")        # registered rate, NOT the forged 0.50


@pytest.mark.asyncio
async def test_process_content_access_unregistered_keeps_advertise_rate(content_economy):
    """Unregistered content has no authenticated source → the (sp1004-bounded)
    advertise rate stands."""
    client = MagicMock()
    client.get_content.return_value = None   # not registered on-chain
    content_economy._get_provenance_client = lambda: client
    payment = await content_economy.process_content_access(
        content_id="unreg-cid",
        accessor_id="test-node-abc123",
        content_metadata={
            "royalty_rate": 0.03,
            "creator_id": "creator-xyz",
            "parent_cids": [],
            "provenance_hash": "0x" + "cd" * 32,
        },
    )
    assert payment.amount == Decimal("0.03")


@pytest.mark.asyncio
async def test_royalty_distribution_phase4_model(content_economy, ledger, mock_content_index):
    """Test royalty distribution with Phase4 model (8% original, 1% derivative)."""
    # Process payment for derivative content
    payment = await content_economy.process_content_access(
        content_id="derivative-cid-456",
        accessor_id="test-node-abc123",
        content_metadata={
            "royalty_rate": 0.10,  # 0.10 FTNS access fee
            "creator_id": "derivative-creator",
            "parent_cids": ["original-cid-123"],
        },
    )
    
    assert payment.status == PaymentStatus.COMPLETED
    assert payment.royalty_model == RoyaltyModel.PHASE4
    
    # Calculate expected distributions
    total = Decimal("0.10")
    expected_original = float(total * Decimal(str(ORIGINAL_CREATOR_ROYALTY_RATE)))  # 8%
    expected_network = float(total * Decimal(str(NETWORK_FEE_RATE)))  # 2%
    
    # Find distributions
    original_dist = [d for d in payment.royalty_distributions if d["type"] == "original_creator"]
    network_dist = [d for d in payment.royalty_distributions if d["type"] == "network_fee"]
    
    if original_dist:
        assert abs(original_dist[0]["amount"] - expected_original) < 0.001
    assert len(network_dist) > 0
    assert abs(network_dist[0]["amount"] - expected_network) < 0.001


@pytest.mark.asyncio
async def test_royalty_distribution_legacy_model(mock_identity, ledger, mock_gossip, mock_content_index):
    """Test royalty distribution with legacy model (70/25/5 split)."""
    economy = ContentEconomy(
        identity=mock_identity,
        ledger=ledger,
        gossip=mock_gossip,
        content_index=mock_content_index,
        royalty_model=RoyaltyModel.LEGACY,
    )
    
    payment = await economy.process_content_access(
        content_id="derivative-cid-456",
        accessor_id="test-node-abc123",
        content_metadata={
            "royalty_rate": 0.10,
            "creator_id": "derivative-creator",
            "parent_cids": ["original-cid-123"],
        },
    )
    
    assert payment.status == PaymentStatus.COMPLETED
    assert payment.royalty_model == RoyaltyModel.LEGACY
    
    # Legacy model: 70% to derivative creator
    derivative_dist = [d for d in payment.royalty_distributions if d["type"] == "derivative_creator"]
    assert len(derivative_dist) > 0
    expected_derivative = 0.10 * 0.70  # 70%
    assert abs(derivative_dist[0]["amount"] - expected_derivative) < 0.001


@pytest.mark.asyncio
async def test_payment_insufficient_balance(mock_identity, mock_gossip, mock_content_index):
    """Test payment fails gracefully when balance is insufficient."""
    # Create a fresh ledger with a wallet that has NO balance
    poor_ledger = LocalLedger(":memory:")
    await poor_ledger.initialize()
    await poor_ledger.create_wallet("test-node-abc123", "Test Node")
    await poor_ledger.create_wallet("creator-xyz", "Creator")
    await poor_ledger.create_wallet("system", "System")
    # Deliberately no credit — wallet has 0 balance

    economy = ContentEconomy(
        identity=mock_identity,
        ledger=poor_ledger,
        gossip=mock_gossip,
        content_index=mock_content_index,
        royalty_model=RoyaltyModel.PHASE4,
    )

    payment = await economy.process_content_access(
        content_id="test-cid-123",
        accessor_id="test-node-abc123",  # Must match identity.node_id to trigger debit
        content_metadata={
            "royalty_rate": 100.0,  # Expensive — exceeds 0 balance
            "creator_id": "creator-xyz",
            "parent_cids": [],
        },
    )

    assert payment.status == PaymentStatus.FAILED
    assert "insufficient" in payment.error.lower()
    await poor_ledger.close()


# ── Replication Tracking Tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_track_content_upload(content_economy):
    """Test replication tracking starts on upload."""
    status = await content_economy.track_content_upload(
        content_id="new-content-123",
        size_bytes=1024 * 1024,  # 1MB
        replicas_requested=3,
    )
    
    assert status.content_id == "new-content-123"
    assert status.min_replicas == 3
    assert status.current_replicas == 1  # Just us
    assert "test-node-abc123" in status.providers


@pytest.mark.asyncio
async def test_update_replication_status(content_economy):
    """Test replication status updates correctly."""
    # Start tracking
    await content_economy.track_content_upload(
        content_id="content-456",
        size_bytes=1024,
        replicas_requested=2,
    )
    
    # Simulate provider announcing content
    await content_economy.update_replication_status(
        content_id="content-456",
        provider_id="provider-node-1",
        has_content=True,
    )
    
    status = content_economy._replication_status.get("content-456")
    assert status is not None
    assert status.current_replicas == 2
    assert "provider-node-1" in status.providers
    
    # Simulate provider removing content
    await content_economy.update_replication_status(
        content_id="content-456",
        provider_id="provider-node-1",
        has_content=False,
    )
    
    assert status.current_replicas == 1
    assert "provider-node-1" not in status.providers


@pytest.mark.asyncio
async def test_replication_request_when_below_minimum(content_economy, mock_gossip):
    """Test that additional replicas are requested when below minimum."""
    # Start with just us (1 replica, min=3)
    await content_economy.track_content_upload(
        content_id="under-replicated",
        size_bytes=1024,
        replicas_requested=3,
    )
    
    # Should trigger a storage request via gossip
    await content_economy._check_replication_needs(
        content_id="under-replicated",
        status=content_economy._replication_status["under-replicated"],
    )
    
    # Verify gossip was called
    mock_gossip.publish.assert_called()
    call_args = mock_gossip.publish.call_args
    assert call_args[0][0] == "storage_request"


# ── Retrieval Marketplace Tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieval_request_bidding(content_economy, mock_gossip):
    """Test content retrieval with marketplace bidding."""
    # Start retrieval request
    request_task = asyncio.create_task(
        content_economy.request_content_retrieval(
            content_id="marketplace-cid",
            max_price_ftns=Decimal("0.05"),
            timeout=5.0,
        )
    )
    
    # Give time for request to be published
    await asyncio.sleep(0.1)
    
    # Verify request was broadcast
    mock_gossip.publish.assert_called()
    
    # Simulate a bid arriving
    request_id = None
    for call in mock_gossip.publish.call_args_list:
        if call[0][0] == "retrieval_request":
            request_id = call[0][1].get("request_id")
            break
    
    assert request_id is not None
    
    # Simulate bid
    await content_economy._on_retrieval_bid(
        subtype="retrieval_bid",
        data={
            "request_id": request_id,
            "price_ftns": 0.03,
            "estimated_latency_ms": 50.0,
            "available_bandwidth_mbps": 100.0,
        },
        origin="provider-bidder",
    )
    
    # Cancel the task (we're not testing actual content retrieval)
    request_task.cancel()
    try:
        await request_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_bid_selection(content_economy):
    """Test best bid selection based on price, reputation, latency."""
    bids = [
        ProviderBid(
            provider_id="cheap-slow",
            content_id="test-cid",
            price_ftns=Decimal("0.01"),
            estimated_latency_ms=500.0,
            available_bandwidth_mbps=10.0,
            reputation_score=0.5,
        ),
        ProviderBid(
            provider_id="expensive-fast",
            content_id="test-cid",
            price_ftns=Decimal("0.05"),
            estimated_latency_ms=10.0,
            available_bandwidth_mbps=100.0,
            reputation_score=0.9,
        ),
        ProviderBid(
            provider_id="balanced",
            content_id="test-cid",
            price_ftns=Decimal("0.03"),
            estimated_latency_ms=50.0,
            available_bandwidth_mbps=50.0,
            reputation_score=0.7,
        ),
    ]
    
    # Select best bid with max price 0.05
    best = content_economy._select_best_bid(bids, Decimal("0.05"))
    
    assert best is not None
    # Scoring: price_score*0.5 + rep*0.3 + latency*0.2
    # cheap-slow wins (0.65) because price weight (50%) dominates
    assert best.provider_id == "cheap-slow"


@pytest.mark.asyncio
async def test_bid_selection_exceeds_max_price(content_economy):
    """Test that bids exceeding max price are rejected."""
    bids = [
        ProviderBid(
            provider_id="expensive",
            content_id="test-cid",
            price_ftns=Decimal("0.10"),
            estimated_latency_ms=10.0,
            available_bandwidth_mbps=100.0,
            reputation_score=0.9,
        ),
    ]
    
    best = content_economy._select_best_bid(bids, Decimal("0.05"))
    assert best is None


# ── Vector DB Integration Tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_index_content_embedding(content_economy):
    """Test content indexing into vector store."""
    # Mock vector store and embedding function
    content_economy.vector_store = AsyncMock()
    content_economy.vector_store.upsert = AsyncMock()
    content_economy.embedding_fn = AsyncMock(return_value=[0.1] * 1536)  # Mock embedding
    
    result = await content_economy.index_content_embedding(
        content_id="embed-test-cid",
        content=b"This is test content for semantic indexing. " * 10,
        metadata={
            "creator_id": "creator-xyz",
            "royalty_rate": 0.05,
        },
    )
    
    assert result is True
    content_economy.vector_store.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_semantic_search(content_economy):
    """Test semantic search functionality."""
    # Mock vector store and embedding function
    content_economy.vector_store = AsyncMock()
    content_economy.embedding_fn = AsyncMock(return_value=[0.1] * 1536)
    
    # Mock search results
    mock_result = MagicMock()
    mock_result.content_cid = "search-result-cid"
    mock_result.similarity_score = 0.85
    mock_result.creator_id = "creator-xyz"
    mock_result.royalty_rate = 0.05
    mock_result.metadata = {}
    
    content_economy.vector_store.search = AsyncMock(return_value=[mock_result])
    
    results = await content_economy.semantic_search(
        query="test query",
        limit=5,
        min_similarity=0.7,
    )
    
    assert len(results) == 1
    assert results[0]["cid"] == "search-result-cid"
    assert results[0]["similarity"] == 0.85


# ── Provenance Chain Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_provenance_chain(content_economy, mock_content_index):
    """Test provenance chain resolution for royalty distribution."""
    chain = await content_economy._resolve_provenance_chain(
        content_id="derivative-cid-456",
        parent_cids=["original-cid-123"],
    )
    
    assert chain.original_creator == "original-creator"
    assert chain.original_content_id == "original-cid-123"


@pytest.mark.asyncio
async def test_resolve_parent_creators(content_economy):
    """Test resolving creator IDs from parent CIDs."""
    creators = await content_economy._resolve_parent_creators(
        parent_cids=["original-cid-123", "unknown-cid"],
    )
    
    assert "original-creator" in creators
    assert len(creators) == 1  # Unknown CID not resolved


# ── Statistics Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_stats(content_economy):
    """Test content economy statistics."""
    # Add some test data
    await content_economy.track_content_upload("cid-1", 1024, 3)
    await content_economy.track_content_upload("cid-2", 2048, 2)
    
    stats = content_economy.get_stats()
    
    assert stats["tracked_content"] == 2
    assert stats["royalty_model"] == "phase4"
    assert stats["min_replicas"] == 3
    assert stats["vector_store_enabled"] is False


# ── Edge Cases ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_parent_cids(content_economy, ledger):
    """Test royalty distribution when no parents (original content)."""
    payment = await content_economy.process_content_access(
        content_id="original-content-cid",
        accessor_id="test-node-abc123",
        content_metadata={
            "royalty_rate": 0.10,
            "creator_id": "creator-xyz",
            "parent_cids": [],  # Original content
        },
    )
    
    assert payment.status == PaymentStatus.COMPLETED
    
    # For original content with Phase4 model, creator should get most of the payment
    # (after network fee if applicable)
    total_distributed = sum(d["amount"] for d in payment.royalty_distributions)
    assert total_distributed > 0


@pytest.mark.asyncio
async def test_concurrent_payments(content_economy, ledger):
    """Test handling multiple concurrent payment requests."""
    # Process multiple payments concurrently
    tasks = [
        content_economy.process_content_access(
            content_id=f"concurrent-cid-{i}",
            accessor_id="test-node-abc123",
            content_metadata={
                "royalty_rate": 0.01,
                "creator_id": "creator-xyz",
                "parent_cids": [],
            },
        )
        for i in range(5)
    ]
    
    payments = await asyncio.gather(*tasks)
    
    assert all(p.status == PaymentStatus.COMPLETED for p in payments)
    assert len(set(p.payment_id for p in payments)) == 5  # Unique IDs
