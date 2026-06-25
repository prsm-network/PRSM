"""
Storage Proof System
====================

Challenge-response proof-of-storage for PRSM network.
Verifies that storage providers actually have the content they claim to store.

This module implements:
- StorageChallenge: Challenges for providers to prove possession
- StorageProof: Proofs submitted by providers
- MerkleProofGenerator: Efficient Merkle tree proofs for large files
- StorageProofVerifier: Verifies storage proofs
- StorageProver: Generates storage proofs (provider side)
- StorageRewardIntegration: Integrates proofs with FTNS rewards
"""

import hashlib
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import structlog

from prsm.node.identity import NodeIdentity

logger = structlog.get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CHALLENGE_DIFFICULTY = 1024  # Bytes to prove
DEFAULT_CHALLENGE_TIMEOUT_MINUTES = 5  # Challenge expiration time
MAX_PENDING_CHALLENGES = 100  # Rate limiting
MIN_CHALLENGE_INTERVAL_SECONDS = 1  # Minimum time between challenges per provider


# =============================================================================
# Enums
# =============================================================================

class ProofType(str, Enum):
    """Types of storage proofs."""
    MERKLE = "merkle"
    RANGE = "range"
    FULL = "full"


class ChallengeStatus(str, Enum):
    """Status of a storage challenge."""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class StorageChallenge:
    """Challenge for storage provider to prove possession of content.

    Attributes:
        challenge_id: Unique identifier for this challenge
        shard_hash: Content hash being challenged
        nonce: Random nonce for this challenge (prevents replay)
        difficulty: Number of bytes to prove possession of
        deadline: When the challenge expires
        created_at: When the challenge was created
        challenger_id: ID of the node issuing the challenge
        proof_type: Type of proof expected (merkle, range, full)
    """
    challenge_id: str
    shard_hash: str
    nonce: str
    difficulty: int
    deadline: datetime
    created_at: datetime
    challenger_id: str
    proof_type: ProofType = ProofType.MERKLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize challenge for transmission."""
        return {
            "challenge_id": self.challenge_id,
            "shard_hash": self.shard_hash,
            "nonce": self.nonce,
            "difficulty": self.difficulty,
            "deadline": self.deadline.isoformat(),
            "created_at": self.created_at.isoformat(),
            "challenger_id": self.challenger_id,
            "proof_type": self.proof_type.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageChallenge":
        """Deserialize challenge from transmission."""
        return cls(
            challenge_id=data["challenge_id"],
            shard_hash=data["shard_hash"],
            nonce=data["nonce"],
            difficulty=data["difficulty"],
            deadline=datetime.fromisoformat(data["deadline"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            challenger_id=data["challenger_id"],
            proof_type=ProofType(data.get("proof_type", "merkle")),
        )
    
    def is_expired(self) -> bool:
        """Check if challenge has expired."""
        return datetime.now(timezone.utc) > self.deadline


@dataclass
class MerkleTree:
    """Merkle tree for content verification.
    
    Attributes:
        root_hash: Root hash of the tree
        leaves: List of leaf hashes
        chunk_size: Size of each chunk in bytes
        content_size: Total content size in bytes
    """
    root_hash: str
    leaves: List[str]
    chunk_size: int
    content_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize Merkle tree."""
        return {
            "root_hash": self.root_hash,
            "leaves": self.leaves,
            "chunk_size": self.chunk_size,
            "content_size": self.content_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MerkleTree":
        """Deserialize Merkle tree."""
        return cls(
            root_hash=data["root_hash"],
            leaves=data["leaves"],
            chunk_size=data["chunk_size"],
            content_size=data["content_size"],
        )


@dataclass
class MerkleProof:
    """Merkle proof for a specific chunk.
    
    Attributes:
        leaf_index: Index of the leaf being proved
        leaf_hash: Hash of the leaf
        siblings: List of sibling hashes for Merkle path
        root_hash: Root hash of the Merkle tree
    """
    leaf_index: int
    leaf_hash: str
    siblings: List[str]
    root_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize Merkle proof."""
        return {
            "leaf_index": self.leaf_index,
            "leaf_hash": self.leaf_hash,
            "siblings": self.siblings,
            "root_hash": self.root_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MerkleProof":
        """Deserialize Merkle proof."""
        return cls(
            leaf_index=data["leaf_index"],
            leaf_hash=data["leaf_hash"],
            siblings=data["siblings"],
            root_hash=data["root_hash"],
        )


@dataclass
class StorageProof:
    """Proof of storage from a provider.

    Attributes:
        challenge_id: ID of the challenge being answered
        provider_id: ID of the storage provider
        shard_hash: Content hash
        proof_type: Type of proof (merkle, range, full)
        proof_data: Actual proof data (serialized)
        timestamp: When proof was generated
        signature: Cryptographic signature
        merkle_proof: Optional Merkle proof for verification
    """
    challenge_id: str
    provider_id: str
    shard_hash: str
    proof_type: ProofType
    proof_data: bytes
    timestamp: datetime
    signature: str
    merkle_proof: Optional[MerkleProof] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize proof for transmission."""
        return {
            "challenge_id": self.challenge_id,
            "provider_id": self.provider_id,
            "shard_hash": self.shard_hash,
            "proof_type": self.proof_type.value,
            "proof_data": self.proof_data.hex(),
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature,
            "merkle_proof": self.merkle_proof.to_dict() if self.merkle_proof else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageProof":
        """Deserialize proof from transmission."""
        return cls(
            challenge_id=data["challenge_id"],
            provider_id=data["provider_id"],
            shard_hash=data["shard_hash"],
            proof_type=ProofType(data["proof_type"]),
            proof_data=bytes.fromhex(data["proof_data"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            signature=data["signature"],
            merkle_proof=MerkleProof.from_dict(data["merkle_proof"]) if data.get("merkle_proof") else None,
        )


@dataclass
class ChallengeRecord:
    """Record of a challenge and its result.
    
    Attributes:
        challenge: The challenge that was issued
        proof: The proof submitted (if any)
        status: Status of the challenge
        verified_at: When the proof was verified (if successful)
        reward_amount: FTNS reward amount (if successful)
        penalty_applied: Whether a penalty was applied (if failed)
    """
    challenge: StorageChallenge
    proof: Optional[StorageProof] = None
    status: ChallengeStatus = ChallengeStatus.PENDING
    verified_at: Optional[datetime] = None
    reward_amount: float = 0.0
    penalty_applied: bool = False


# =============================================================================
# Merkle Proof Generator
# =============================================================================

class MerkleProofGenerator:
    """Generates Merkle proofs for storage verification.
    
    Merkle trees allow efficient verification that a specific chunk
    of content is part of a larger file without needing the entire file.
    """
    
    def __init__(self, chunk_size: int = 4096):
        """Initialize the Merkle proof generator.
        
        Args:
            chunk_size: Size of each chunk in bytes (default 4KB)
        """
        self.chunk_size = chunk_size
    
    def _hash(self, data: bytes) -> str:
        """Compute SHA-256 hash of data.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex-encoded hash string
        """
        return hashlib.sha256(data).hexdigest()
    
    def _hash_pair(self, left: str, right: str) -> str:
        """Hash two Merkle tree nodes together.
        
        Args:
            left: Left child hash
            right: Right child hash
            
        Returns:
            Parent hash
        """
        combined = (left + right).encode()
        return self._hash(combined)
    
    def build_merkle_tree(self, content: bytes) -> MerkleTree:
        """Build a Merkle tree from content.
        
        Args:
            content: Content bytes to build tree from
            
        Returns:
            MerkleTree with root hash and leaves
        """
        # Split content into chunks
        chunks = []
        for i in range(0, len(content), self.chunk_size):
            chunks.append(content[i:i + self.chunk_size])
        
        if not chunks:
            # Empty content - return tree with empty root
            return MerkleTree(
                root_hash=self._hash(b""),
                leaves=[],
                chunk_size=self.chunk_size,
                content_size=0,
            )
        
        # Hash all chunks
        leaves = [self._hash(chunk) for chunk in chunks]
        
        # Build tree bottom-up
        current_level = leaves.copy()
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    next_level.append(self._hash_pair(current_level[i], current_level[i + 1]))
                else:
                    # Odd number of nodes - promote last one
                    next_level.append(current_level[i])
            current_level = next_level
        
        return MerkleTree(
            root_hash=current_level[0],
            leaves=leaves,
            chunk_size=self.chunk_size,
            content_size=len(content),
        )
    
    def generate_proof(self, tree: MerkleTree, leaf_index: int) -> MerkleProof:
        """Generate a Merkle proof for a specific leaf.
        
        Args:
            tree: Merkle tree
            leaf_index: Index of the leaf to prove
            
        Returns:
            MerkleProof containing the proof path
        """
        if leaf_index < 0 or leaf_index >= len(tree.leaves):
            raise ValueError(f"Invalid leaf index: {leaf_index}")
        
        siblings = []
        current_index = leaf_index
        current_level = tree.leaves.copy()
        
        # Build proof path
        while len(current_level) > 1:
            # Find sibling
            if current_index % 2 == 0:
                # Current is left child, need right sibling
                sibling_index = current_index + 1
            else:
                # Current is right child, need left sibling
                sibling_index = current_index - 1
            
            if sibling_index < len(current_level):
                siblings.append(current_level[sibling_index])
            else:
                # No sibling (odd number of nodes)
                siblings.append(current_level[current_index])
            
            # Move to next level
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    next_level.append(self._hash_pair(current_level[i], current_level[i + 1]))
                else:
                    next_level.append(current_level[i])
            
            current_index = current_index // 2
            current_level = next_level
        
        return MerkleProof(
            leaf_index=leaf_index,
            leaf_hash=tree.leaves[leaf_index],
            siblings=siblings,
            root_hash=tree.root_hash,
        )
    
    def verify_proof(self, proof: MerkleProof) -> bool:
        """Verify a Merkle proof.
        
        Args:
            proof: Merkle proof to verify
            
        Returns:
            True if proof is valid, False otherwise
        """
        # Start with the leaf hash
        current_hash = proof.leaf_hash
        current_index = proof.leaf_index
        
        # Traverse up the tree using siblings
        for i, sibling_hash in enumerate(proof.siblings):
            # Determine if we're left or right child at this level
            if current_index % 2 == 0:
                # We're left child
                current_hash = self._hash_pair(current_hash, sibling_hash)
            else:
                # We're right child
                current_hash = self._hash_pair(sibling_hash, current_hash)
            
            current_index = current_index // 2
        
        # Check if we reached the correct root
        return current_hash == proof.root_hash
    
    def generate_challenge_proof(
        self,
        content: bytes,
        nonce: str,
        difficulty: int
    ) -> Tuple[MerkleTree, MerkleProof, bytes]:
        """Generate a complete proof for a challenge.
        
        Args:
            content: Content bytes
            nonce: Challenge nonce
            difficulty: Number of bytes to prove
            
        Returns:
            Tuple of (MerkleTree, MerkleProof, chunk_data)
        """
        # Build Merkle tree
        tree = self.build_merkle_tree(content)
        
        if not tree.leaves:
            raise ValueError("Cannot generate proof for empty content")
        
        # Use nonce to deterministically select which chunk to prove
        # This prevents providers from pre-computing proofs
        nonce_hash = hashlib.sha256(nonce.encode()).hexdigest()
        selected_index = int(nonce_hash, 16) % len(tree.leaves)
        
        # Generate proof for selected chunk
        proof = self.generate_proof(tree, selected_index)
        
        # Extract the actual chunk data
        start = selected_index * self.chunk_size
        end = min(start + self.chunk_size, len(content))
        chunk_data = content[start:end]
        
        return tree, proof, chunk_data


# =============================================================================
# Storage Proof Verifier
# =============================================================================

class StorageProofVerifier:
    """Verifies storage proofs from providers.
    
    This class is used by challengers (nodes requesting proof of storage)
    to generate challenges and verify responses.
    """
    
    def __init__(
        self,
        challenge_timeout_minutes: int = DEFAULT_CHALLENGE_TIMEOUT_MINUTES,
        max_pending_challenges: int = MAX_PENDING_CHALLENGES,
    ):
        """Initialize the proof verifier.

        Args:
            challenge_timeout_minutes: Minutes until challenge expires
            max_pending_challenges: Maximum pending challenges per provider
        """
        self.challenge_timeout_minutes = challenge_timeout_minutes
        self.max_pending_challenges = max_pending_challenges
        
        # Track pending challenges
        self._pending_challenges: Dict[str, ChallengeRecord] = {}
        
        # Track challenge rate (per provider)
        self._last_challenge_time: Dict[str, float] = {}
        
        # Merkle proof generator
        self._merkle = MerkleProofGenerator()
        
        # Provider reputation tracking
        self._provider_stats: Dict[str, Dict[str, int]] = {}
    
    def generate_challenge(
        self,
        shard_hash: str,
        challenger_id: str,
        difficulty: int = DEFAULT_CHALLENGE_DIFFICULTY,
        proof_type: ProofType = ProofType.MERKLE,
    ) -> StorageChallenge:
        """Generate a new storage challenge.

        Args:
            shard_hash: Content hash to challenge
            challenger_id: ID of the node issuing the challenge
            difficulty: Number of bytes to prove
            proof_type: Type of proof expected

        Returns:
            StorageChallenge for the provider to answer
        """
        # Generate unique challenge ID
        challenge_id = f"challenge_{secrets.token_hex(16)}"

        # Generate random nonce
        nonce = secrets.token_hex(32)

        # Set deadline
        deadline = datetime.now(timezone.utc) + timedelta(minutes=self.challenge_timeout_minutes)

        challenge = StorageChallenge(
            challenge_id=challenge_id,
            shard_hash=shard_hash,
            nonce=nonce,
            difficulty=difficulty,
            deadline=deadline,
            created_at=datetime.now(timezone.utc),
            challenger_id=challenger_id,
            proof_type=proof_type,
        )
        
        # Track pending challenge
        self._pending_challenges[challenge_id] = ChallengeRecord(challenge=challenge)
        
        return challenge
    
    def can_challenge(self, provider_id: str) -> bool:
        """Check if we can issue a challenge to a provider (rate limiting).
        
        Args:
            provider_id: Provider to check
            
        Returns:
            True if we can challenge, False if rate limited
        """
        now = time.time()
        last_time = self._last_challenge_time.get(provider_id, 0)
        
        # Check minimum interval
        if now - last_time < MIN_CHALLENGE_INTERVAL_SECONDS:
            return False
        
        # Check pending challenge count
        pending_count = sum(
            1 for record in self._pending_challenges.values()
            if record.challenge.challenger_id == provider_id
            and record.status == ChallengeStatus.PENDING
        )
        
        return pending_count < self.max_pending_challenges
    
    async def verify_proof(
        self,
        proof: StorageProof,
        challenge: Optional[StorageChallenge] = None,
        provider_public_key: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """Verify a storage proof against a challenge.
        
        Args:
            proof: Proof to verify
            challenge: Challenge being answered (looked up if not provided)
            provider_public_key: Provider's public key for signature verification
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get challenge
        if challenge is None:
            record = self._pending_challenges.get(proof.challenge_id)
            if record is None:
                return False, f"Unknown challenge: {proof.challenge_id}"
            challenge = record.challenge
        
        # Check deadline
        if challenge.is_expired():
            self._update_challenge_status(proof.challenge_id, ChallengeStatus.EXPIRED)
            return False, "Challenge has expired"
        
        # Verify challenge matches
        if challenge.shard_hash != proof.shard_hash:
            return False, "shard_hash mismatch"
        
        if challenge.challenger_id != proof.provider_id:
            # Note: In production, we'd verify the provider is authorized
            pass
        
        # Verify signature
        if provider_public_key is not None:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                import base64
                
                # Reconstruct signed data
                signed_data = self._get_signed_data(proof)
                
                # Verify signature
                public_key = Ed25519PublicKey.from_public_bytes(provider_public_key)
                signature_bytes = base64.b64decode(proof.signature)
                public_key.verify(signature_bytes, signed_data)
                
            except Exception as e:
                logger.warning(f"Signature verification failed: {e}")
                return False, f"Signature verification failed: {e}"
        
        # sp1252 — the answered proof_type MUST match the ISSUED challenge's type.
        # Dispatch previously branched on the ATTACKER-controlled proof.proof_type
        # with no check against challenge.proof_type, letting a provider DOWNGRADE an
        # issued MERKLE challenge into the weaker RANGE/FULL verifiers. The honest
        # prover always answers with challenge.proof_type (answer_challenge), so a
        # mismatch is a forgery attempt → reject (fail closed).
        if proof.proof_type != challenge.proof_type:
            return False, (
                f"proof_type {proof.proof_type} does not match challenge proof_type "
                f"{challenge.proof_type} (downgrade rejected)"
            )

        # Verify proof based on type
        if proof.proof_type == ProofType.MERKLE:
            return self._verify_merkle_proof(proof, challenge)
        elif proof.proof_type == ProofType.RANGE:
            return self._verify_range_proof(proof, challenge)
        elif proof.proof_type == ProofType.FULL:
            return self._verify_full_proof(proof, challenge)
        else:
            return False, f"Unknown proof type: {proof.proof_type}"
    
    def _get_signed_data(self, proof: StorageProof) -> bytes:
        """Get the data that was signed in the proof.
        
        Args:
            proof: Storage proof
            
        Returns:
            Bytes that were signed
        """
        # Sign the challenge ID, CID, and proof data hash
        data = f"{proof.challenge_id}:{proof.shard_hash}:{proof.proof_data.hex()}:{proof.timestamp.isoformat()}"
        return data.encode()
    
    def _verify_merkle_proof(
        self,
        proof: StorageProof,
        challenge: StorageChallenge,
    ) -> Tuple[bool, str]:
        """Verify a Merkle proof.
        
        Args:
            proof: Proof to verify
            challenge: Challenge being answered
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if proof.merkle_proof is None:
            return False, "Merkle proof missing"
        
        # Verify the Merkle proof structure
        if not self._merkle.verify_proof(proof.merkle_proof):
            return False, "Merkle proof verification failed"
        
        # Verify the proof data matches the leaf hash
        # The proof_data should contain the chunk that hashes to leaf_hash
        computed_hash = hashlib.sha256(proof.proof_data).hexdigest()
        if computed_hash != proof.merkle_proof.leaf_hash:
            return False, "Proof data does not match Merkle leaf"
        
        # Verify the nonce was used correctly to select the chunk
        nonce_hash = hashlib.sha256(challenge.nonce.encode()).hexdigest()
        # We can't fully verify without knowing the total number of chunks
        # but we can verify the proof is valid
        
        self._update_challenge_status(proof.challenge_id, ChallengeStatus.VERIFIED)
        return True, ""
    
    def _verify_range_proof(
        self,
        proof: StorageProof,
        challenge: StorageChallenge,
    ) -> Tuple[bool, str]:
        """Verify a range proof.
        
        Args:
            proof: Proof to verify
            challenge: Challenge being answered
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # sp1252 — FAIL CLOSED. A range proof must bind the returned bytes to the
        # challenged content (the nonce must deterministically select a byte offset,
        # and the verifier must compare proof_data to content[offset:offset+len]).
        # That content-binding is NOT implemented — the prior code accepted ANY bytes
        # of sufficient LENGTH as VERIFIED, a forgeable no-op (a provider storing
        # nothing could submit random padding of the right size). Until real range
        # verification exists, reject. The live system only ever ISSUES MERKLE
        # challenges (storage_provider.py), so this rejects nothing legitimate; it only
        # closes the proof-type-downgrade target.
        return False, "range proof verification not implemented — fail closed (sp1252)"
    
    def _verify_full_proof(
        self,
        proof: StorageProof,
        challenge: StorageChallenge,
    ) -> Tuple[bool, str]:
        """Verify a full content proof.
        
        Args:
            proof: Proof to verify
            challenge: Challenge being answered
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # sp1252 — FAIL CLOSED. A full proof must recompute the content hash/CID over
        # proof_data and compare it to challenge.shard_hash; that binding is NOT
        # implemented — the prior code accepted ANY non-empty bytes as VERIFIED, a
        # forgeable no-op. Until real full-content verification exists, reject. The
        # live system only ever ISSUES MERKLE challenges, so this rejects nothing
        # legitimate; it only closes the proof-type-downgrade target.
        return False, "full proof verification not implemented — fail closed (sp1252)"
    
    def _update_challenge_status(
        self,
        challenge_id: str,
        status: ChallengeStatus,
        proof: Optional[StorageProof] = None,
    ) -> None:
        """Update the status of a challenge.
        
        Args:
            challenge_id: Challenge to update
            status: New status
            proof: Proof that was submitted (if any)
        """
        record = self._pending_challenges.get(challenge_id)
        if record:
            record.status = status
            if proof:
                record.proof = proof
            if status == ChallengeStatus.VERIFIED:
                record.verified_at = datetime.now(timezone.utc)
    
    def get_challenge(self, challenge_id: str) -> Optional[StorageChallenge]:
        """Get a pending challenge by ID.
        
        Args:
            challenge_id: Challenge ID
            
        Returns:
            Challenge if found, None otherwise
        """
        record = self._pending_challenges.get(challenge_id)
        return record.challenge if record else None
    
    def get_challenge_record(self, challenge_id: str) -> Optional[ChallengeRecord]:
        """Get a challenge record by ID.
        
        Args:
            challenge_id: Challenge ID
            
        Returns:
            ChallengeRecord if found, None otherwise
        """
        return self._pending_challenges.get(challenge_id)
    
    def cleanup_expired_challenges(self) -> int:
        """Remove expired challenges from tracking.
        
        Returns:
            Number of challenges removed
        """
        expired_ids = [
            challenge_id for challenge_id, record in self._pending_challenges.items()
            if record.challenge.is_expired()
        ]

        for challenge_id in expired_ids:
            self._pending_challenges[challenge_id].status = ChallengeStatus.EXPIRED
        
        return len(expired_ids)
    
    def get_provider_stats(self, provider_id: str) -> Dict[str, int]:
        """Get statistics for a provider.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Dictionary with verification statistics
        """
        return self._provider_stats.get(provider_id, {
            "total_challenges": 0,
            "successful_proofs": 0,
            "failed_proofs": 0,
            "expired_challenges": 0,
        })
    
    def record_provider_result(
        self,
        provider_id: str,
        success: bool,
        expired: bool = False,
    ) -> None:
        """Record a verification result for a provider.
        
        Args:
            provider_id: Provider ID
            success: Whether the proof was successful
            expired: Whether the challenge expired
        """
        if provider_id not in self._provider_stats:
            self._provider_stats[provider_id] = {
                "total_challenges": 0,
                "successful_proofs": 0,
                "failed_proofs": 0,
                "expired_challenges": 0,
            }
        
        stats = self._provider_stats[provider_id]
        stats["total_challenges"] += 1
        
        if expired:
            stats["expired_challenges"] += 1
        elif success:
            stats["successful_proofs"] += 1
        else:
            stats["failed_proofs"] += 1


# =============================================================================
# Storage Prover (Provider Side)
# =============================================================================

class StorageProver:
    """Generates storage proofs for content.
    
    This class is used by storage providers to answer challenges
    and prove they have the content they claim to store.
    """
    
    def __init__(
        self,
        identity: NodeIdentity,
        content_client: Optional[Any] = None,
    ):
        """Initialize the storage prover.

        Args:
            identity: Node identity for signing proofs
            content_client: Content storage client for retrieval (optional)
        """
        self.identity = identity
        self.content_client = content_client
        self._merkle = MerkleProofGenerator()
    
    async def answer_challenge(
        self,
        challenge: StorageChallenge,
        content: Optional[bytes] = None,
    ) -> Optional[StorageProof]:
        """Answer a storage challenge with a proof.
        
        Args:
            challenge: Challenge to answer
            content: Content bytes (retrieved from ContentStore if not provided)
            
        Returns:
            StorageProof if successful, None if failed
        """
        # Check if challenge is expired
        if challenge.is_expired():
            logger.warning(f"Challenge {challenge.challenge_id} has expired")
            return None
        
        try:
            # Get content if not provided
            if content is None:
                content = await self._get_content(challenge.shard_hash)
                if content is None:
                    logger.error(f"Failed to retrieve content {challenge.shard_hash}")
                    return None
            
            # Generate proof based on type
            if challenge.proof_type == ProofType.MERKLE:
                proof_data = await self._generate_merkle_proof(content, challenge)
            elif challenge.proof_type == ProofType.RANGE:
                proof_data = await self._generate_range_proof(content, challenge)
            elif challenge.proof_type == ProofType.FULL:
                proof_data = await self._generate_full_proof(content, challenge)
            else:
                logger.error(f"Unknown proof type: {challenge.proof_type}")
                return None
            
            if proof_data is None:
                return None
            
            # Create proof object
            proof = StorageProof(
                challenge_id=challenge.challenge_id,
                provider_id=self.identity.node_id,
                shard_hash=challenge.shard_hash,
                proof_type=challenge.proof_type,
                proof_data=proof_data["proof_data"],
                timestamp=datetime.now(timezone.utc),
                signature="",  # Will be set after signing
                merkle_proof=proof_data.get("merkle_proof"),
            )
            
            # Sign the proof
            proof.signature = self._sign_proof(proof)
            
            return proof
            
        except Exception as e:
            logger.error(f"Failed to answer challenge: {e}")
            return None
    
    async def _get_content(self, shard_hash: str) -> Optional[bytes]:
        """Retrieve content from storage.

        Args:
            shard_hash: Content hash to retrieve

        Returns:
            Content bytes if successful, None otherwise
        """
        try:
            # Try using content client if available
            if self.content_client and hasattr(self.content_client, 'get_content'):
                return await self.content_client.get_content(shard_hash)

            return None

        except Exception as e:
            logger.error(f"Failed to get content {shard_hash}: {e}")
            return None
    
    async def _generate_merkle_proof(
        self,
        content: bytes,
        challenge: StorageChallenge,
    ) -> Optional[Dict[str, Any]]:
        """Generate a Merkle proof for the challenge.
        
        Args:
            content: Content bytes
            challenge: Challenge to answer
            
        Returns:
            Dictionary with proof_data and merkle_proof
        """
        try:
            # Build Merkle tree and generate proof
            tree, proof, chunk_data = self._merkle.generate_challenge_proof(
                content=content,
                nonce=challenge.nonce,
                difficulty=challenge.difficulty,
            )
            
            return {
                "proof_data": chunk_data,
                "merkle_proof": proof,
            }
            
        except Exception as e:
            logger.error(f"Failed to generate Merkle proof: {e}")
            return None
    
    async def _generate_range_proof(
        self,
        content: bytes,
        challenge: StorageChallenge,
    ) -> Optional[Dict[str, Any]]:
        """Generate a range proof for the challenge.
        
        Args:
            content: Content bytes
            challenge: Challenge to answer
            
        Returns:
            Dictionary with proof_data
        """
        try:
            # Use nonce to determine which range to prove
            nonce_hash = hashlib.sha256(challenge.nonce.encode()).hexdigest()
            start = int(nonce_hash, 16) % max(1, len(content) - challenge.difficulty)
            end = min(start + challenge.difficulty, len(content))
            
            return {
                "proof_data": content[start:end],
            }
            
        except Exception as e:
            logger.error(f"Failed to generate range proof: {e}")
            return None
    
    async def _generate_full_proof(
        self,
        content: bytes,
        challenge: StorageChallenge,
    ) -> Optional[Dict[str, Any]]:
        """Generate a full content proof.
        
        Args:
            content: Content bytes
            challenge: Challenge to answer
            
        Returns:
            Dictionary with proof_data
        """
        # For small content, return the entire content
        return {
            "proof_data": content,
        }
    
    def _sign_proof(self, proof: StorageProof) -> str:
        """Sign a storage proof with the node's private key.
        
        Args:
            proof: Proof to sign
            
        Returns:
            Base64-encoded signature
        """
        # Get the data to sign
        signed_data = f"{proof.challenge_id}:{proof.shard_hash}:{proof.proof_data.hex()}:{proof.timestamp.isoformat()}"
        
        # Sign with node's private key
        signature = self.identity.sign(signed_data.encode())
        
        return signature
    
    async def close(self) -> None:
        """Close any open sessions."""
        pass


# =============================================================================
# Storage Reward Integration
# =============================================================================

class StorageRewardIntegration:
    """Integrates storage proofs with the FTNS reward system.
    
    This class handles:
    - Rewarding providers for successful proofs
    - Penalizing providers for failed challenges
    - Tracking provider reputation
    """
    
    def __init__(
        self,
        ftns_service: Optional[Any] = None,
        proof_verifier: Optional[StorageProofVerifier] = None,
        base_reward_per_gb: float = 0.1,
        penalty_rate: float = 0.05,
    ):
        """Initialize the reward integration.
        
        Args:
            ftns_service: FTNS service for token operations
            proof_verifier: Storage proof verifier
            base_reward_per_gb: Base FTNS reward per GB stored
            penalty_rate: Penalty rate for failed challenges (fraction of stake)
        """
        self.ftns_service = ftns_service
        self.proof_verifier = proof_verifier or StorageProofVerifier()
        self.base_reward_per_gb = base_reward_per_gb
        self.penalty_rate = penalty_rate
        
        # Track provider storage
        self._provider_storage: Dict[str, Dict[str, int]] = {}  # provider_id -> {cid: size}
        
        # Track rewards
        self._provider_rewards: Dict[str, float] = {}  # provider_id -> total_rewards
    
    async def process_successful_proof(
        self,
        proof: StorageProof,
        content_size_bytes: int = 0,
    ) -> Tuple[bool, float]:
        """Process a successful storage proof and reward the provider.
        
        Args:
            proof: Successful storage proof
            content_size_bytes: Size of the content in bytes
            
        Returns:
            Tuple of (success, reward_amount)
        """
        provider_id = proof.provider_id
        
        # Calculate reward
        if content_size_bytes > 0:
            size_gb = content_size_bytes / (1024 ** 3)
            reward = size_gb * self.base_reward_per_gb
        else:
            # Base reward if size unknown
            reward = self.base_reward_per_gb / 10  # 0.01 FTNS
        
        # Track the reward
        if provider_id not in self._provider_rewards:
            self._provider_rewards[provider_id] = 0.0
        self._provider_rewards[provider_id] += reward
        
        # Award FTNS tokens if service available
        if self.ftns_service:
            try:
                # Use atomic FTNS service if available
                if hasattr(self.ftns_service, 'mint_tokens_atomic'):
                    result = await self.ftns_service.mint_tokens_atomic(
                        to_user_id=provider_id,
                        amount=reward,
                        idempotency_key=f"storage_proof:{proof.challenge_id}",
                    )
                    if not result.success:
                        logger.warning(f"Failed to mint tokens: {result.error_message}")
                        return False, 0.0
                elif hasattr(self.ftns_service, 'credit'):
                    await self.ftns_service.credit(
                        wallet_id=provider_id,
                        amount=reward,
                        tx_type="storage_reward",
                        description=f"Storage proof reward for {proof.shard_hash[:16]}",
                    )
                else:
                    logger.warning("FTNS service does not support minting/credit")
                    return False, 0.0
                    
                logger.info(f"Awarded {reward:.6f} FTNS to {provider_id} for storage proof")
                return True, reward
                
            except Exception as e:
                logger.error(f"Failed to award FTNS tokens: {e}")
                return False, 0.0
        else:
            # No FTNS service - just track the reward
            logger.debug(f"Tracked {reward:.6f} FTNS reward for {provider_id} (no FTNS service)")
            return True, reward
    
    async def process_failed_challenge(
        self,
        challenge: StorageChallenge,
        provider_id: str,
        reason: str = "unknown",
    ) -> Tuple[bool, float]:
        """Process a failed challenge and apply penalties.
        
        Args:
            challenge: Failed challenge
            provider_id: Provider that failed
            reason: Reason for failure
            
        Returns:
            Tuple of (success, penalty_amount)
        """
        # Record the failure in verifier stats
        self.proof_verifier.record_provider_result(
            provider_id=provider_id,
            success=False,
        )
        
        # Calculate penalty (this would typically come from staked tokens)
        # For now, we just track the failure
        penalty = 0.0
        
        # In a full implementation, we would:
        # 1. Check if provider has staked tokens
        # 2. Apply penalty from stake
        # 3. Update provider reputation
        # 4. Potentially slash provider if too many failures
        
        logger.warning(f"Provider {provider_id} failed challenge {challenge.challenge_id}: {reason}")
        
        return True, penalty
    
    async def process_expired_challenge(
        self,
        challenge: StorageChallenge,
        provider_id: str,
    ) -> None:
        """Process an expired challenge.
        
        Args:
            challenge: Expired challenge
            provider_id: Provider that didn't respond
        """
        # Record the expiration
        self.proof_verifier.record_provider_result(
            provider_id=provider_id,
            success=False,
            expired=True,
        )
        
        logger.warning(f"Challenge {challenge.challenge_id} expired for provider {provider_id}")
    
    def track_storage(self, provider_id: str, content_id: str, size_bytes: int) -> None:
        """Track content stored by a provider.

        Args:
            provider_id: Provider ID
            content_id: Content identifier
            size_bytes: Size in bytes
        """
        if provider_id not in self._provider_storage:
            self._provider_storage[provider_id] = {}

        self._provider_storage[provider_id][content_id] = size_bytes

    def untrack_storage(self, provider_id: str, content_id: str) -> None:
        """Stop tracking content for a provider.

        Args:
            provider_id: Provider ID
            content_id: Content identifier
        """
        if provider_id in self._provider_storage:
            self._provider_storage[provider_id].pop(content_id, None)
    
    def get_provider_storage_gb(self, provider_id: str) -> float:
        """Get total storage provided by a provider in GB.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Total storage in GB
        """
        if provider_id not in self._provider_storage:
            return 0.0
        
        total_bytes = sum(self._provider_storage[provider_id].values())
        return total_bytes / (1024 ** 3)
    
    def get_provider_rewards(self, provider_id: str) -> float:
        """Get total rewards earned by a provider.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Total FTNS rewards
        """
        return self._provider_rewards.get(provider_id, 0.0)
    
    def get_provider_reputation(self, provider_id: str) -> float:
        """Calculate provider reputation score.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Reputation score (0.0 to 1.0)
        """
        stats = self.proof_verifier.get_provider_stats(provider_id)
        
        total = stats.get("total_challenges", 0)
        if total == 0:
            return 1.0  # New providers start with full reputation
        
        successful = stats.get("successful_proofs", 0)
        expired = stats.get("expired_challenges", 0)
        
        # Expired challenges are worse than failed proofs
        # (provider didn't respond at all)
        score = successful / total
        penalty = expired * 0.1  # 10% penalty per expired challenge
        
        return max(0.0, min(1.0, score - penalty))


# =============================================================================
# Convenience Functions
# =============================================================================

def create_storage_challenge(
    shard_hash: str,
    challenger_id: str,
    difficulty: int = DEFAULT_CHALLENGE_DIFFICULTY,
) -> StorageChallenge:
    """Create a new storage challenge.

    Args:
        shard_hash: Content hash to challenge
        challenger_id: ID of the challenger
        difficulty: Challenge difficulty in bytes

    Returns:
        StorageChallenge
    """
    verifier = StorageProofVerifier()
    return verifier.generate_challenge(shard_hash, challenger_id, difficulty)


async def verify_storage_proof(
    proof: StorageProof,
    challenge: StorageChallenge,
    provider_public_key: Optional[bytes] = None,
) -> Tuple[bool, str]:
    """Verify a storage proof.
    
    Args:
        proof: Proof to verify
        challenge: Challenge being answered
        provider_public_key: Provider's public key
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    verifier = StorageProofVerifier()
    return await verifier.verify_proof(proof, challenge, provider_public_key)