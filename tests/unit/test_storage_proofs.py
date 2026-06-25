"""
Tests for Storage Proof System
==============================

Comprehensive tests for the challenge-response proof-of-storage system.
"""

import asyncio
import hashlib
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.node.storage_proofs import (
    # Dataclasses
    StorageChallenge,
    StorageProof,
    MerkleTree,
    MerkleProof,
    ChallengeRecord,
    # Enums
    ProofType,
    ChallengeStatus,
    # Classes
    MerkleProofGenerator,
    StorageProofVerifier,
    StorageProver,
    StorageRewardIntegration,
    # Constants
    DEFAULT_CHALLENGE_DIFFICULTY,
    DEFAULT_CHALLENGE_TIMEOUT_MINUTES,
    # Functions
    create_storage_challenge,
    verify_storage_proof,
)
from prsm.node.identity import NodeIdentity, generate_node_identity


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_content():
    """Sample content for testing."""
    return b"This is test content for storage proof verification. " * 100


@pytest.fixture
def sample_cid():
    """Sample CID for testing."""
    return "QmTestCID123456789"


@pytest.fixture
def challenger_identity():
    """Generate a challenger identity for testing."""
    return generate_node_identity("test-challenger")


@pytest.fixture
def provider_identity():
    """Generate a provider identity for testing."""
    return generate_node_identity("test-provider")


@pytest.fixture
def merkle_generator():
    """Create a Merkle proof generator."""
    return MerkleProofGenerator()


@pytest.fixture
def proof_verifier():
    """Create a storage proof verifier."""
    return StorageProofVerifier()


@pytest.fixture
def storage_prover(provider_identity):
    """Create a storage prover."""
    return StorageProver(identity=provider_identity)


# =============================================================================
# Test StorageChallenge
# =============================================================================

class TestStorageChallenge:
    """Tests for StorageChallenge dataclass."""
    
    def test_create_challenge(self, sample_cid, challenger_identity):
        """Test creating a storage challenge."""
        challenge = StorageChallenge(
            challenge_id="test_challenge_1",
            shard_hash=sample_cid,
            nonce="test_nonce_123",
            difficulty=1024,
            deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
            created_at=datetime.now(timezone.utc),
            challenger_id=challenger_identity.node_id,
        )
        
        assert challenge.challenge_id == "test_challenge_1"
        assert challenge.shard_hash == sample_cid
        assert challenge.nonce == "test_nonce_123"
        assert challenge.difficulty == 1024
        assert challenge.proof_type == ProofType.MERKLE  # Default
    
    def test_challenge_serialization(self, sample_cid, challenger_identity):
        """Test challenge serialization and deserialization."""
        challenge = StorageChallenge(
            challenge_id="test_challenge_2",
            shard_hash=sample_cid,
            nonce="test_nonce_456",
            difficulty=2048,
            deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
            created_at=datetime.now(timezone.utc),
            challenger_id=challenger_identity.node_id,
            proof_type=ProofType.RANGE,
        )
        
        # Serialize
        data = challenge.to_dict()
        assert data["challenge_id"] == "test_challenge_2"
        assert data["proof_type"] == "range"
        
        # Deserialize
        restored = StorageChallenge.from_dict(data)
        assert restored.challenge_id == challenge.challenge_id
        assert restored.shard_hash == challenge.shard_hash
        assert restored.nonce == challenge.nonce
        assert restored.difficulty == challenge.difficulty
        assert restored.proof_type == ProofType.RANGE
    
    def test_challenge_expiration(self, sample_cid, challenger_identity):
        """Test challenge expiration check."""
        # Create expired challenge
        expired_challenge = StorageChallenge(
            challenge_id="expired_challenge",
            shard_hash=sample_cid,
            nonce="nonce",
            difficulty=1024,
            deadline=datetime.now(timezone.utc) - timedelta(minutes=1),  # In the past
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            challenger_id=challenger_identity.node_id,
        )
        
        assert expired_challenge.is_expired() is True
        
        # Create valid challenge
        valid_challenge = StorageChallenge(
            challenge_id="valid_challenge",
            shard_hash=sample_cid,
            nonce="nonce",
            difficulty=1024,
            deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
            created_at=datetime.now(timezone.utc),
            challenger_id=challenger_identity.node_id,
        )
        
        assert valid_challenge.is_expired() is False


# =============================================================================
# Test MerkleProofGenerator
# =============================================================================

class TestMerkleProofGenerator:
    """Tests for Merkle proof generation and verification."""
    
    def test_build_merkle_tree_empty(self, merkle_generator):
        """Test building a Merkle tree from empty content."""
        tree = merkle_generator.build_merkle_tree(b"")
        
        assert tree.content_size == 0
        assert len(tree.leaves) == 0
        # Root should be hash of empty string
        assert tree.root_hash == hashlib.sha256(b"").hexdigest()
    
    def test_build_merkle_tree_single_chunk(self, merkle_generator, sample_content):
        """Test building a Merkle tree from single chunk."""
        # Use content smaller than chunk size
        small_content = b"Small content"
        tree = merkle_generator.build_merkle_tree(small_content)
        
        assert tree.content_size == len(small_content)
        assert len(tree.leaves) == 1
        assert tree.root_hash == tree.leaves[0]
    
    def test_build_merkle_tree_multiple_chunks(self, merkle_generator):
        """Test building a Merkle tree from multiple chunks."""
        # Create content that spans multiple chunks
        chunk_size = 100  # Use smaller chunks for testing
        generator = MerkleProofGenerator(chunk_size=chunk_size)
        
        content = b"x" * (chunk_size * 3 + 50)  # 3.5 chunks
        tree = generator.build_merkle_tree(content)
        
        assert tree.content_size == len(content)
        assert len(tree.leaves) == 4  # 4 chunks
        assert tree.chunk_size == chunk_size
    
    def test_generate_proof(self, merkle_generator, sample_content):
        """Test generating a Merkle proof."""
        tree = merkle_generator.build_merkle_tree(sample_content)
        
        # Generate proof for first leaf
        proof = merkle_generator.generate_proof(tree, 0)
        
        assert proof.leaf_index == 0
        assert proof.leaf_hash == tree.leaves[0]
        assert proof.root_hash == tree.root_hash
        assert len(proof.siblings) > 0
    
    def test_verify_valid_proof(self, merkle_generator, sample_content):
        """Test verifying a valid Merkle proof."""
        tree = merkle_generator.build_merkle_tree(sample_content)
        proof = merkle_generator.generate_proof(tree, 0)
        
        assert merkle_generator.verify_proof(proof) is True
    
    def test_verify_invalid_proof_wrong_root(self, merkle_generator, sample_content):
        """Test verifying a proof with wrong root hash."""
        tree = merkle_generator.build_merkle_tree(sample_content)
        proof = merkle_generator.generate_proof(tree, 0)
        
        # Tamper with root hash
        proof.root_hash = "0" * 64
        
        assert merkle_generator.verify_proof(proof) is False
    
    def test_verify_invalid_proof_wrong_leaf(self, merkle_generator, sample_content):
        """Test verifying a proof with wrong leaf hash."""
        tree = merkle_generator.build_merkle_tree(sample_content)
        proof = merkle_generator.generate_proof(tree, 0)
        
        # Tamper with leaf hash
        proof.leaf_hash = "0" * 64
        
        assert merkle_generator.verify_proof(proof) is False
    
    def test_generate_challenge_proof(self, merkle_generator, sample_content):
        """Test generating a complete challenge proof."""
        tree, proof, chunk_data = merkle_generator.generate_challenge_proof(
            content=sample_content,
            nonce="test_nonce",
            difficulty=1024,
        )
        
        assert tree.content_size == len(sample_content)
        assert proof.root_hash == tree.root_hash
        assert len(chunk_data) <= merkle_generator.chunk_size
        
        # Verify the chunk hashes to the leaf
        chunk_hash = hashlib.sha256(chunk_data).hexdigest()
        assert chunk_hash == proof.leaf_hash
    
    def test_proof_for_different_leaves(self, merkle_generator):
        """Test generating proofs for different leaves."""
        chunk_size = 100
        generator = MerkleProofGenerator(chunk_size=chunk_size)
        content = b"x" * (chunk_size * 4)
        
        tree = generator.build_merkle_tree(content)
        
        # Generate and verify proofs for all leaves
        for i in range(len(tree.leaves)):
            proof = generator.generate_proof(tree, i)
            assert generator.verify_proof(proof) is True
    
    def test_invalid_leaf_index(self, merkle_generator, sample_content):
        """Test generating proof with invalid leaf index."""
        tree = merkle_generator.build_merkle_tree(sample_content)
        
        with pytest.raises(ValueError):
            merkle_generator.generate_proof(tree, -1)
        
        with pytest.raises(ValueError):
            merkle_generator.generate_proof(tree, len(tree.leaves) + 100)


# =============================================================================
# Test StorageProofVerifier
# =============================================================================

class TestStorageProofVerifier:
    """Tests for storage proof verification."""
    
    def test_generate_challenge(self, proof_verifier, sample_cid, challenger_identity):
        """Test generating a storage challenge."""
        challenge = proof_verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        assert challenge.shard_hash == sample_cid
        assert challenge.challenger_id == challenger_identity.node_id
        assert challenge.difficulty == DEFAULT_CHALLENGE_DIFFICULTY
        assert challenge.proof_type == ProofType.MERKLE
        assert not challenge.is_expired()
        
        # Challenge should be tracked
        assert challenge.challenge_id in proof_verifier._pending_challenges
    
    def test_generate_challenge_custom_params(self, proof_verifier, sample_cid, challenger_identity):
        """Test generating a challenge with custom parameters."""
        challenge = proof_verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
            difficulty=2048,
            proof_type=ProofType.RANGE,
        )
        
        assert challenge.difficulty == 2048
        assert challenge.proof_type == ProofType.RANGE
    
    def test_rate_limiting(self, proof_verifier, sample_cid, challenger_identity):
        """Test challenge rate limiting."""
        provider_id = "test_provider"
        
        # Should be able to challenge initially
        assert proof_verifier.can_challenge(provider_id) is True
        
        # Record a challenge time
        proof_verifier._last_challenge_time[provider_id] = time.time()
        
        # Should be rate limited immediately after
        assert proof_verifier.can_challenge(provider_id) is False
        
        # Should be allowed after interval
        proof_verifier._last_challenge_time[provider_id] = time.time() - 2
        assert proof_verifier.can_challenge(provider_id) is True
    
    def test_get_challenge(self, proof_verifier, sample_cid, challenger_identity):
        """Test retrieving a challenge by ID."""
        challenge = proof_verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        retrieved = proof_verifier.get_challenge(challenge.challenge_id)
        assert retrieved is not None
        assert retrieved.challenge_id == challenge.challenge_id
        
        # Non-existent challenge
        assert proof_verifier.get_challenge("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_verify_merkle_proof(
        self,
        proof_verifier,
        sample_content,
        sample_cid,
        challenger_identity,
        provider_identity,
    ):
        """Test verifying a valid Merkle proof."""
        # Generate challenge
        challenge = proof_verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        # Generate proof
        merkle = MerkleProofGenerator()
        tree, merkle_proof, chunk_data = merkle.generate_challenge_proof(
            content=sample_content,
            nonce=challenge.nonce,
            difficulty=challenge.difficulty,
        )
        
        proof = StorageProof(
            challenge_id=challenge.challenge_id,
            provider_id=provider_identity.node_id,
            shard_hash=sample_cid,
            proof_type=ProofType.MERKLE,
            proof_data=chunk_data,
            timestamp=datetime.now(timezone.utc),
            signature="test_signature",
            merkle_proof=merkle_proof,
        )
        
        # Verify proof (sp1253 — bound to the independently-trusted root)
        is_valid, error = await proof_verifier.verify_proof(
            proof, challenge, expected_merkle_root=tree.root_hash)

        assert is_valid is True
        assert error == ""
    
    @pytest.mark.asyncio
    async def test_verify_expired_challenge(
        self,
        proof_verifier,
        sample_cid,
        challenger_identity,
        provider_identity,
    ):
        """Test verifying a proof for an expired challenge."""
        # Create an expired challenge
        challenge = StorageChallenge(
            challenge_id="expired_test",
            shard_hash=sample_cid,
            nonce="nonce",
            difficulty=1024,
            deadline=datetime.now(timezone.utc) - timedelta(minutes=1),
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            challenger_id=challenger_identity.node_id,
        )
        
        proof = StorageProof(
            challenge_id=challenge.challenge_id,
            provider_id=provider_identity.node_id,
            shard_hash=sample_cid,
            proof_type=ProofType.MERKLE,
            proof_data=b"test data",
            timestamp=datetime.now(timezone.utc),
            signature="sig",
        )
        
        is_valid, error = await proof_verifier.verify_proof(proof, challenge)
        
        assert is_valid is False
        assert "expired" in error.lower()
    
    @pytest.mark.asyncio
    async def test_verify_wrong_cid(
        self,
        proof_verifier,
        sample_cid,
        challenger_identity,
        provider_identity,
    ):
        """Test verifying a proof with wrong CID."""
        challenge = proof_verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        proof = StorageProof(
            challenge_id=challenge.challenge_id,
            provider_id=provider_identity.node_id,
            shard_hash="WrongCID123",  # Wrong shard hash
            proof_type=ProofType.MERKLE,
            proof_data=b"test data",
            timestamp=datetime.now(timezone.utc),
            signature="sig",
        )
        
        is_valid, error = await proof_verifier.verify_proof(proof, challenge)
        
        assert is_valid is False
        assert "mismatch" in error.lower()
    
    @pytest.mark.asyncio
    async def test_verify_range_proof(
        self,
        proof_verifier,
        sample_content,
        sample_cid,
        challenger_identity,
        provider_identity,
    ):
        """Test verifying a range proof."""
        challenge = proof_verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
            proof_type=ProofType.RANGE,
        )
        
        # Generate range proof data
        nonce_hash = hashlib.sha256(challenge.nonce.encode()).hexdigest()
        start = int(nonce_hash, 16) % max(1, len(sample_content) - challenge.difficulty)
        end = min(start + challenge.difficulty, len(sample_content))
        proof_data = sample_content[start:end]
        
        proof = StorageProof(
            challenge_id=challenge.challenge_id,
            provider_id=provider_identity.node_id,
            shard_hash=sample_cid,
            proof_type=ProofType.RANGE,
            proof_data=proof_data,
            timestamp=datetime.now(timezone.utc),
            signature="sig",
        )
        
        is_valid, error = await proof_verifier.verify_proof(proof, challenge)

        # sp1252 — range proofs now FAIL CLOSED. The prior no-op verifier accepted any
        # sufficient-length bytes with NO content binding (a forgeable fail-open); real
        # range verification is unimplemented, so even a genuine byte-slice is rejected
        # until it lands. (The live system only issues MERKLE challenges.)
        assert is_valid is False
        assert "not implemented" in error.lower()

    @pytest.mark.asyncio
    async def test_verify_range_proof_too_small(
        self,
        proof_verifier,
        sample_cid,
        challenger_identity,
        provider_identity,
    ):
        """Test verifying a range proof with insufficient data."""
        challenge = proof_verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
            difficulty=1024,
            proof_type=ProofType.RANGE,
        )
        
        proof = StorageProof(
            challenge_id=challenge.challenge_id,
            provider_id=provider_identity.node_id,
            shard_hash=sample_cid,
            proof_type=ProofType.RANGE,
            proof_data=b"small",  # Too small
            timestamp=datetime.now(timezone.utc),
            signature="sig",
        )
        
        is_valid, error = await proof_verifier.verify_proof(proof, challenge)

        # sp1252 — still rejected (the security property holds), now for a STRONGER
        # reason: range verification fails closed before the size check is reached.
        assert is_valid is False
        assert "not implemented" in error.lower()
    
    def test_cleanup_expired_challenges(self, proof_verifier, sample_cid, challenger_identity):
        """Test cleaning up expired challenges."""
        # Create an expired challenge
        expired_challenge = StorageChallenge(
            challenge_id="expired_cleanup_test",
            shard_hash=sample_cid,
            nonce="nonce",
            difficulty=1024,
            deadline=datetime.now(timezone.utc) - timedelta(minutes=1),
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            challenger_id=challenger_identity.node_id,
        )
        
        proof_verifier._pending_challenges[expired_challenge.challenge_id] = ChallengeRecord(
            challenge=expired_challenge
        )
        
        # Create a valid challenge
        valid_challenge = proof_verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        # Cleanup
        removed = proof_verifier.cleanup_expired_challenges()
        
        assert removed >= 1
        assert proof_verifier._pending_challenges[expired_challenge.challenge_id].status == ChallengeStatus.EXPIRED
    
    def test_provider_stats(self, proof_verifier):
        """Test provider statistics tracking."""
        provider_id = "test_provider_stats"
        
        # Initial stats
        stats = proof_verifier.get_provider_stats(provider_id)
        assert stats["total_challenges"] == 0
        
        # Record some results
        proof_verifier.record_provider_result(provider_id, success=True)
        proof_verifier.record_provider_result(provider_id, success=True)
        proof_verifier.record_provider_result(provider_id, success=False)
        proof_verifier.record_provider_result(provider_id, success=False, expired=True)
        
        stats = proof_verifier.get_provider_stats(provider_id)
        assert stats["total_challenges"] == 4
        assert stats["successful_proofs"] == 2
        assert stats["failed_proofs"] == 1
        assert stats["expired_challenges"] == 1


# =============================================================================
# Test StorageProver
# =============================================================================

class TestStorageProver:
    """Tests for storage prover (provider side)."""
    
    def test_prover_initialization(self, provider_identity):
        """Test storage prover initialization."""
        prover = StorageProver(identity=provider_identity)

        assert prover.identity == provider_identity
        # Post-v1.5.0: StorageProver uses content_client abstraction, not ipfs_api_url
        assert prover.content_client is None
    
    @pytest.mark.asyncio
    async def test_answer_challenge_merkle(
        self,
        storage_prover,
        sample_content,
        sample_cid,
        challenger_identity,
    ):
        """Test answering a Merkle challenge."""
        # Create challenge
        verifier = StorageProofVerifier()
        challenge = verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        # Answer challenge
        proof = await storage_prover.answer_challenge(
            challenge=challenge,
            content=sample_content,
        )
        
        assert proof is not None
        assert proof.challenge_id == challenge.challenge_id
        assert proof.provider_id == storage_prover.identity.node_id
        assert proof.shard_hash == sample_cid
        assert proof.proof_type == ProofType.MERKLE
        assert proof.merkle_proof is not None
        assert proof.signature != ""
    
    @pytest.mark.asyncio
    async def test_answer_challenge_range(
        self,
        provider_identity,
        sample_content,
        sample_cid,
        challenger_identity,
    ):
        """Test answering a range challenge."""
        prover = StorageProver(identity=provider_identity)
        verifier = StorageProofVerifier()
        
        challenge = verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
            proof_type=ProofType.RANGE,
        )
        
        proof = await prover.answer_challenge(
            challenge=challenge,
            content=sample_content,
        )
        
        assert proof is not None
        assert proof.proof_type == ProofType.RANGE
        assert len(proof.proof_data) >= challenge.difficulty
    
    @pytest.mark.asyncio
    async def test_answer_challenge_full(
        self,
        provider_identity,
        sample_content,
        sample_cid,
        challenger_identity,
    ):
        """Test answering a full content challenge."""
        prover = StorageProver(identity=provider_identity)
        verifier = StorageProofVerifier()
        
        challenge = verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
            proof_type=ProofType.FULL,
        )
        
        proof = await prover.answer_challenge(
            challenge=challenge,
            content=sample_content,
        )
        
        assert proof is not None
        assert proof.proof_type == ProofType.FULL
        assert proof.proof_data == sample_content
    
    @pytest.mark.asyncio
    async def test_answer_expired_challenge(
        self,
        storage_prover,
        sample_content,
        sample_cid,
        challenger_identity,
    ):
        """Test answering an expired challenge."""
        # Create expired challenge
        challenge = StorageChallenge(
            challenge_id="expired_answer_test",
            shard_hash=sample_cid,
            nonce="nonce",
            difficulty=1024,
            deadline=datetime.now(timezone.utc) - timedelta(minutes=1),
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            challenger_id=challenger_identity.node_id,
        )
        
        proof = await storage_prover.answer_challenge(
            challenge=challenge,
            content=sample_content,
        )
        
        assert proof is None
    
    @pytest.mark.asyncio
    async def test_signature_verification(
        self,
        provider_identity,
        sample_content,
        sample_cid,
        challenger_identity,
    ):
        """Test that proof signatures can be verified."""
        prover = StorageProver(identity=provider_identity)
        verifier = StorageProofVerifier()
        
        challenge = verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        proof = await prover.answer_challenge(
            challenge=challenge,
            content=sample_content,
        )
        
        assert proof is not None
        
        # Verify signature using provider's public key
        signed_data = f"{proof.challenge_id}:{proof.shard_hash}:{proof.proof_data.hex()}:{proof.timestamp.isoformat()}"
        is_valid = provider_identity.verify(signed_data.encode(), proof.signature)
        assert is_valid is True


# =============================================================================
# Test StorageRewardIntegration
# =============================================================================

class TestStorageRewardIntegration:
    """Tests for storage reward integration."""
    
    def test_initialization(self):
        """Test reward integration initialization."""
        integration = StorageRewardIntegration()
        
        assert integration.base_reward_per_gb == 0.1
        assert integration.penalty_rate == 0.05
    
    def test_track_storage(self, sample_cid):
        """Test tracking provider storage."""
        integration = StorageRewardIntegration()
        provider_id = "test_provider"
        
        # Use exact 1 GB for cleaner test
        integration.track_storage(provider_id, sample_cid, 1024 * 1024 * 1024)  # 1 GB
        
        assert sample_cid in integration._provider_storage[provider_id]
        assert integration.get_provider_storage_gb(provider_id) == pytest.approx(1.0, rel=0.01)
    
    def test_untrack_storage(self, sample_cid):
        """Test untracking provider storage."""
        integration = StorageRewardIntegration()
        provider_id = "test_provider"
        
        integration.track_storage(provider_id, sample_cid, 1024 * 1024 * 100)
        integration.untrack_storage(provider_id, sample_cid)
        
        assert sample_cid not in integration._provider_storage.get(provider_id, {})
    
    @pytest.mark.asyncio
    async def test_process_successful_proof(self, sample_cid, provider_identity):
        """Test processing a successful proof."""
        integration = StorageRewardIntegration()
        
        proof = StorageProof(
            challenge_id="test_challenge",
            provider_id=provider_identity.node_id,
            shard_hash=sample_cid,
            proof_type=ProofType.MERKLE,
            proof_data=b"test data",
            timestamp=datetime.now(timezone.utc),
            signature="sig",
        )
        
        success, reward = await integration.process_successful_proof(
            proof=proof,
            content_size_bytes=1024 * 1024 * 1024,  # 1 GB
        )
        
        assert success is True
        assert reward == pytest.approx(0.1, rel=0.01)  # 0.1 FTNS per GB
        assert integration.get_provider_rewards(provider_identity.node_id) == reward
    
    @pytest.mark.asyncio
    async def test_process_successful_proof_with_ftns_service(
        self,
        sample_cid,
        provider_identity,
    ):
        """Test processing a successful proof with FTNS service."""
        # Mock FTNS service with AsyncMock for credit method
        # The implementation checks for mint_tokens_atomic first, then credit
        mock_ftns = MagicMock()
        mock_ftns.credit = AsyncMock(return_value=None)
        # Make sure mint_tokens_atomic is not present so credit is used
        del mock_ftns.mint_tokens_atomic
        
        integration = StorageRewardIntegration(ftns_service=mock_ftns)
        
        proof = StorageProof(
            challenge_id="test_challenge_ftns",
            provider_id=provider_identity.node_id,
            shard_hash=sample_cid,
            proof_type=ProofType.MERKLE,
            proof_data=b"test data",
            timestamp=datetime.now(timezone.utc),
            signature="sig",
        )
        
        success, reward = await integration.process_successful_proof(
            proof=proof,
            content_size_bytes=1024 * 1024 * 1024,  # 1 GB
        )
        
        assert success is True
        mock_ftns.credit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_failed_challenge(self, sample_cid, challenger_identity):
        """Test processing a failed challenge."""
        integration = StorageRewardIntegration()
        provider_id = "failed_provider"
        
        challenge = StorageChallenge(
            challenge_id="failed_challenge",
            shard_hash=sample_cid,
            nonce="nonce",
            difficulty=1024,
            deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
            created_at=datetime.now(timezone.utc),
            challenger_id=challenger_identity.node_id,
        )
        
        success, penalty = await integration.process_failed_challenge(
            challenge=challenge,
            provider_id=provider_id,
            reason="Provider did not respond",
        )
        
        assert success is True
        
        # Check stats were updated
        stats = integration.proof_verifier.get_provider_stats(provider_id)
        assert stats["failed_proofs"] == 1
    
    @pytest.mark.asyncio
    async def test_process_expired_challenge(self, sample_cid, challenger_identity):
        """Test processing an expired challenge."""
        integration = StorageRewardIntegration()
        provider_id = "expired_provider"
        
        challenge = StorageChallenge(
            challenge_id="expired_challenge",
            shard_hash=sample_cid,
            nonce="nonce",
            difficulty=1024,
            deadline=datetime.now(timezone.utc) - timedelta(minutes=1),
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            challenger_id=challenger_identity.node_id,
        )
        
        await integration.process_expired_challenge(
            challenge=challenge,
            provider_id=provider_id,
        )
        
        stats = integration.proof_verifier.get_provider_stats(provider_id)
        assert stats["expired_challenges"] == 1
    
    def test_provider_reputation(self):
        """Test provider reputation calculation."""
        integration = StorageRewardIntegration()
        provider_id = "reputation_provider"
        
        # New provider has full reputation
        assert integration.get_provider_reputation(provider_id) == 1.0
        
        # Record some results
        integration.proof_verifier.record_provider_result(provider_id, success=True)
        integration.proof_verifier.record_provider_result(provider_id, success=True)
        integration.proof_verifier.record_provider_result(provider_id, success=True)
        integration.proof_verifier.record_provider_result(provider_id, success=False)
        
        # Reputation should be 0.75 (3/4 successful)
        reputation = integration.get_provider_reputation(provider_id)
        assert reputation == pytest.approx(0.75, rel=0.01)
        
        # Record an expired challenge (worse)
        integration.proof_verifier.record_provider_result(provider_id, success=False, expired=True)
        
        reputation = integration.get_provider_reputation(provider_id)
        assert reputation < 0.75  # Should be lower


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_storage_challenge(self, sample_cid, challenger_identity):
        """Test create_storage_challenge function."""
        challenge = create_storage_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        assert challenge.shard_hash == sample_cid
        assert challenge.challenger_id == challenger_identity.node_id
    
    @pytest.mark.asyncio
    async def test_verify_storage_proof(
        self,
        sample_content,
        sample_cid,
        challenger_identity,
        provider_identity,
    ):
        """Test verify_storage_proof function."""
        # Create challenge
        challenge = create_storage_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        # Generate proof
        merkle = MerkleProofGenerator()
        tree, merkle_proof, chunk_data = merkle.generate_challenge_proof(
            content=sample_content,
            nonce=challenge.nonce,
            difficulty=challenge.difficulty,
        )
        
        proof = StorageProof(
            challenge_id=challenge.challenge_id,
            provider_id=provider_identity.node_id,
            shard_hash=sample_cid,
            proof_type=ProofType.MERKLE,
            proof_data=chunk_data,
            timestamp=datetime.now(timezone.utc),
            signature="sig",
            merkle_proof=merkle_proof,
        )
        
        # Verify (sp1253 — bound to the trusted root)
        is_valid, error = await verify_storage_proof(
            proof=proof,
            challenge=challenge,
            expected_merkle_root=tree.root_hash,
        )

        assert is_valid is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete storage proof flow."""
    
    @pytest.mark.asyncio
    async def test_complete_challenge_response_flow(
        self,
        sample_content,
        sample_cid,
        challenger_identity,
        provider_identity,
    ):
        """Test the complete challenge-response flow."""
        # Setup
        verifier = StorageProofVerifier()
        prover = StorageProver(identity=provider_identity)
        integration = StorageRewardIntegration(proof_verifier=verifier)
        
        # 1. Challenger generates challenge
        challenge = verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        # 2. Provider answers challenge
        proof = await prover.answer_challenge(
            challenge=challenge,
            content=sample_content,
        )
        
        assert proof is not None
        
        # 3. Challenger verifies proof (sp1253 — against the root recomputed from the
        # content it holds)
        is_valid, error = await verifier.verify_proof(
            proof=proof,
            challenge=challenge,
            provider_public_key=provider_identity.public_key_bytes,
            expected_merkle_root=MerkleProofGenerator().build_merkle_tree(
                sample_content).root_hash,
        )

        assert is_valid is True
        
        # 4. Process reward
        success, reward = await integration.process_successful_proof(
            proof=proof,
            content_size_bytes=len(sample_content),
        )
        
        assert success is True
        assert reward > 0
    
    @pytest.mark.asyncio
    async def test_multiple_challenges_same_content(
        self,
        sample_content,
        sample_cid,
        challenger_identity,
        provider_identity,
    ):
        """Test multiple challenges for the same content."""
        verifier = StorageProofVerifier()
        prover = StorageProver(identity=provider_identity)
        
        # Generate multiple challenges
        challenges = [
            verifier.generate_challenge(
                shard_hash=sample_cid,
                challenger_id=challenger_identity.node_id,
            )
            for _ in range(5)
        ]
        
        # Answer all challenges
        proofs = []
        for challenge in challenges:
            proof = await prover.answer_challenge(
                challenge=challenge,
                content=sample_content,
            )
            proofs.append(proof)
        
        # Verify all proofs (sp1253 — bound to the content's trusted root)
        trusted_root = MerkleProofGenerator().build_merkle_tree(sample_content).root_hash
        for challenge, proof in zip(challenges, proofs):
            is_valid, _ = await verifier.verify_proof(
                proof, challenge, expected_merkle_root=trusted_root)
            assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_different_proof_types(
        self,
        sample_content,
        sample_cid,
        challenger_identity,
        provider_identity,
    ):
        """Test different proof types for the same content."""
        verifier = StorageProofVerifier()
        prover = StorageProver(identity=provider_identity)
        
        for proof_type in [ProofType.MERKLE, ProofType.RANGE, ProofType.FULL]:
            challenge = verifier.generate_challenge(
                shard_hash=sample_cid,
                challenger_id=challenger_identity.node_id,
                proof_type=proof_type,
            )
            
            proof = await prover.answer_challenge(
                challenge=challenge,
                content=sample_content,
            )
            
            assert proof is not None
            assert proof.proof_type == proof_type

            # sp1252/1253 — the prover can ANSWER all three types, but only MERKLE
            # verifies (and only when bound to the content's trusted root); RANGE/FULL
            # fail closed (no-op verifiers were forgeable fail-opens).
            expected_root = (
                MerkleProofGenerator().build_merkle_tree(sample_content).root_hash
                if proof_type == ProofType.MERKLE else None)
            is_valid, _ = await verifier.verify_proof(
                proof, challenge, expected_merkle_root=expected_root)
            assert is_valid is (proof_type == ProofType.MERKLE)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_content_merkle(self):
        """Test Merkle tree with empty content."""
        generator = MerkleProofGenerator()
        tree = generator.build_merkle_tree(b"")
        
        assert tree.content_size == 0
        assert len(tree.leaves) == 0
    
    @pytest.mark.asyncio
    async def test_empty_content_challenge(self, sample_cid, challenger_identity, provider_identity):
        """Test challenge with empty content."""
        verifier = StorageProofVerifier()
        prover = StorageProver(identity=provider_identity)
        
        challenge = verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
        )
        
        # Should fail gracefully with empty content
        proof = await prover.answer_challenge(
            challenge=challenge,
            content=b"",
        )
        
        # Empty content should return None (can't generate proof)
        assert proof is None
    
    def test_very_large_difficulty(self, sample_cid, challenger_identity):
        """Test challenge with very large difficulty."""
        verifier = StorageProofVerifier()
        
        challenge = verifier.generate_challenge(
            shard_hash=sample_cid,
            challenger_id=challenger_identity.node_id,
            difficulty=10 * 1024 * 1024 * 1024,  # 10 GB
        )
        
        assert challenge.difficulty == 10 * 1024 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_unknown_challenge_id(self, sample_cid, provider_identity):
        """Test verifying proof with unknown challenge ID."""
        verifier = StorageProofVerifier()
        
        proof = StorageProof(
            challenge_id="unknown_challenge_id",
            provider_id=provider_identity.node_id,
            shard_hash=sample_cid,
            proof_type=ProofType.MERKLE,
            proof_data=b"test",
            timestamp=datetime.now(timezone.utc),
            signature="sig",
        )
        
        is_valid, error = await verifier.verify_proof(proof)
        
        assert is_valid is False
        assert "unknown" in error.lower()
    
    def test_challenge_uniqueness(self, sample_cid, challenger_identity):
        """Test that challenges have unique IDs and nonces."""
        verifier = StorageProofVerifier()
        
        challenges = [
            verifier.generate_challenge(
                shard_hash=sample_cid,
                challenger_id=challenger_identity.node_id,
            )
            for _ in range(10)
        ]
        
        ids = [c.challenge_id for c in challenges]
        nonces = [c.nonce for c in challenges]
        
        # All IDs should be unique
        assert len(set(ids)) == len(ids)
        
        # All nonces should be unique
        assert len(set(nonces)) == len(nonces)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])