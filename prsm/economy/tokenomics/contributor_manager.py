"""
FTNS Contributor Manager
Manages contributor status and proof-of-contribution verification

This module implements the core contributor management system for PRSM's
proof-of-contribution tokenomics model. Users must actively contribute to
the network to earn FTNS tokens, preventing pure speculation.

Core Features:
- Cryptographic proof verification for multiple contribution types
- Dynamic contributor status tracking (None, Basic, Active, Power User)
- Peer validation and quality scoring systems
- Integration with the PRSM ContentStore for storage proofs
- Governance participation tracking
- Grace period management for temporary disconnections

Integration Points:
- FTNSService: For reward distribution based on contributor status
- Storage Verifier: For storage contribution verification (e.g. StorageProvider)
- Governance System: For voting participation tracking
- Database: For persistent storage of proofs and status
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from prsm.economy.tokenomics.models import (
    FTNSContributorStatus, FTNSContributionProof, ContributorTier, ContributionType, ProofStatus
)

logger = structlog.get_logger(__name__)


@dataclass
class VerificationResult:
    """Result of contribution proof verification"""
    hash: str
    value: Decimal
    quality: float
    confidence: float
    metadata: Dict[str, Any] = None


@dataclass
class ContributionProofRequest:
    """Request structure for submitting contribution proofs"""
    contribution_type: str
    proof_data: Dict[str, Any]
    expected_value: Optional[Decimal] = None
    quality_claim: Optional[float] = None


class ContributorManager:
    """
    Manages contributor status and proof-of-contribution verification
    
    The ContributorManager is responsible for:
    1. Verifying cryptographic proofs of contribution
    2. Maintaining contributor status tiers
    3. Calculating contribution scores and quality metrics
    4. Managing grace periods and status transitions
    5. Providing earning eligibility determinations
    """
    
    def __init__(self, db_session: AsyncSession, storage_verifier=None, governance_service=None):
        self.db = db_session
        self.storage_verifier = storage_verifier
        self.governance = governance_service
        self.verification_cache = {}
        
        # Contribution scoring parameters (governance configurable)
        self.scoring_weights = {
            ContributionType.STORAGE: 1.0,
            ContributionType.COMPUTE: 1.2,
            ContributionType.DATA: 2.0,
            ContributionType.GOVERNANCE: 1.5,
            ContributionType.DOCUMENTATION: 1.3,
            ContributionType.MODEL: 3.0,
            ContributionType.RESEARCH: 4.0,
            ContributionType.TEACHING: 2.5
        }
        
        # Tier thresholds (governance configurable)
        self.tier_thresholds = {
            ContributorTier.BASIC: 10.0,      # 10 points minimum
            ContributorTier.ACTIVE: 50.0,     # 50 points for active
            ContributorTier.POWER_USER: 150.0  # 150 points for power user
        }
        
        # Grace period settings
        self.grace_period_days = 90
        self.status_update_cooldown_hours = 24
    
    # === PROOF VERIFICATION ===
    
    async def verify_contribution_proof(self, user_id: str, proof_request: ContributionProofRequest) -> FTNSContributionProof:
        """
        Verify a contribution proof cryptographically
        
        Args:
            user_id: The user submitting the proof
            proof_request: The proof data and metadata
            
        Returns:
            FTNSContributionProof: The verified and stored proof record
            
        Raises:
            ValueError: If proof is invalid or verification fails
        """
        
        contribution_type = proof_request.contribution_type
        proof_data = proof_request.proof_data
        
        # Validate contribution type
        if contribution_type not in [ct.value for ct in ContributionType]:
            raise ValueError(f"Unknown contribution type: {contribution_type}")
        
        # Verify the proof based on type
        verification_methods = {
            ContributionType.STORAGE.value: self._verify_storage_proof,
            ContributionType.COMPUTE.value: self._verify_compute_proof,
            ContributionType.DATA.value: self._verify_data_proof,
            ContributionType.GOVERNANCE.value: self._verify_governance_proof,
            ContributionType.DOCUMENTATION.value: self._verify_documentation_proof,
            ContributionType.MODEL.value: self._verify_model_proof,
            ContributionType.RESEARCH.value: self._verify_research_proof,
            ContributionType.TEACHING.value: self._verify_teaching_proof
        }
        
        verification_result = await verification_methods[contribution_type](user_id, proof_data)
        
        # Create proof record
        proof = FTNSContributionProof(
            user_id=user_id,
            contribution_type=contribution_type,
            proof_hash=verification_result.hash,
            contribution_value=verification_result.value,
            quality_score=verification_result.quality,
            verification_confidence=verification_result.confidence,
            proof_data=proof_data,
            verification_status=ProofStatus.VERIFIED.value,
            verification_timestamp=datetime.now(timezone.utc),
            validation_criteria=self._get_validation_criteria(contribution_type),
            validation_results=verification_result.metadata or {}
        )
        
        # Store in database
        self.db.add(proof)
        await self.db.commit()
        await self.db.refresh(proof)
        
        # Note: Status update is now handled manually by caller to avoid
        # session conflicts when submitting multiple proofs in sequence
        
        await self._log_proof_verification(user_id, contribution_type, verification_result)
        
        return proof
    
    async def _verify_storage_proof(self, user_id: str, proof_data: Dict) -> VerificationResult:
        """Verify storage contribution using ContentStore pinning proofs."""

        required_fields = ["content_hashes", "storage_duration_hours", "redundancy_factor"]
        if not all(field in proof_data for field in required_fields):
            raise ValueError("Missing required storage proof fields")

        if not self.storage_verifier:
            raise ValueError("Storage verifier not available for storage verification")

        verified_storage = 0
        verified_hashes = []

        for content_hash in proof_data["content_hashes"]:
            try:
                pin_status = await self.storage_verifier.verify_pin_status(content_hash, user_id)
                if pin_status:
                    file_size = await self.storage_verifier.get_file_size(content_hash)
                    verified_storage += file_size
                    verified_hashes.append(content_hash)

            except Exception as e:
                logger.warning(f"Failed to verify content hash {content_hash}: {e}")
                continue

        if not verified_hashes:
            raise ValueError("No valid content hashes could be verified")

        storage_gb = verified_storage / (1024**3)
        duration_hours = proof_data["storage_duration_hours"]
        redundancy = proof_data["redundancy_factor"]

        # Value calculation: 0.01 FTNS per GB-hour with redundancy bonus
        base_value = Decimal(str(storage_gb)) * Decimal(str(duration_hours)) * Decimal("0.01")
        redundancy_bonus = min(redundancy, 3.0) / 3.0  # Cap redundancy bonus at 3x
        value = base_value * Decimal(str(1.0 + redundancy_bonus * 0.5))

        # Quality score based on storage amount and duration
        quality = min(1.0, (storage_gb / 100) * 0.7 + (duration_hours / (24 * 30)) * 0.3)

        # Confidence based on verification success rate
        confidence = len(verified_hashes) / len(proof_data["content_hashes"])

        # Generate proof hash
        proof_content = {
            "user_id": user_id,
            "verified_hashes": sorted(verified_hashes),
            "verified_storage_gb": float(storage_gb),
            "duration_hours": duration_hours,
            "redundancy_factor": redundancy
        }
        proof_hash = hashlib.sha256(json.dumps(proof_content, sort_keys=True).encode()).hexdigest()
        
        return VerificationResult(
            hash=proof_hash,
            value=value,
            quality=quality,
            confidence=confidence,
            metadata={
                "verified_hashes": verified_hashes,
                "verified_storage_gb": float(storage_gb),
                "redundancy_applied": redundancy_bonus
            }
        )
    
    async def _verify_compute_proof(self, user_id: str, proof_data: Dict) -> VerificationResult:
        """Verify compute contribution using work proofs"""
        
        required_fields = ["work_units_completed", "computation_hash", "benchmark_score"]
        if not all(field in proof_data for field in required_fields):
            raise ValueError("Missing required compute proof fields")
        
        # Verify computation hash against known work units
        work_verified = await self._verify_computation_hash(
            proof_data["computation_hash"],
            proof_data["work_units_completed"]
        )
        
        if not work_verified:
            raise ValueError("Invalid computation proof - hash verification failed")
        
        # Calculate contribution value
        work_units = proof_data["work_units_completed"]
        benchmark_score = min(proof_data["benchmark_score"], 2.0)  # Cap at 2x baseline
        
        # Value: 0.05 FTNS per work unit with performance bonus
        base_value = Decimal(str(work_units)) * Decimal("0.05")
        performance_bonus = benchmark_score
        value = base_value * Decimal(str(performance_bonus))
        
        # Quality based on benchmark performance
        quality = min(1.0, benchmark_score / 1.5)  # Perfect at 1.5x benchmark
        
        # High confidence for cryptographic proofs
        confidence = 0.95
        
        proof_hash = proof_data["computation_hash"]
        
        return VerificationResult(
            hash=proof_hash,
            value=value,
            quality=quality,
            confidence=confidence,
            metadata={
                "work_units": work_units,
                "benchmark_score": benchmark_score,
                "performance_bonus": performance_bonus
            }
        )
    
    async def _verify_data_proof(self, user_id: str, proof_data: Dict) -> VerificationResult:
        """Verify data contribution quality and originality"""
        
        required_fields = ["dataset_hash", "peer_reviews", "quality_metrics"]
        if not all(field in proof_data for field in required_fields):
            raise ValueError("Missing required data proof fields")
        
        # Verify peer reviews. sp1265 — bind each review to the submitter context so a
        # submitter can't fabricate their own "peer reviews" (self-authored or duplicated) to
        # clear the min-2 gate; `seen` enforces DISTINCT reviewers within this submission.
        verified_reviews = []
        seen_reviewers: set = set()
        for review in proof_data["peer_reviews"]:
            if await self._verify_peer_review(review, submitter_id=user_id, seen=seen_reviewers):
                verified_reviews.append(review)

        if len(verified_reviews) < 2:
            raise ValueError("Insufficient verified peer reviews (minimum 2 required)")
        
        # Validate quality metrics
        quality_metrics = proof_data["quality_metrics"]
        required_metrics = ["completeness", "accuracy", "uniqueness", "usefulness"]
        
        if not all(metric in quality_metrics for metric in required_metrics):
            raise ValueError("Missing required quality metrics")
        
        # Calculate contribution value based on quality metrics
        base_value = Decimal("10.0")  # Base value for data contribution
        
        quality_multiplier = (
            quality_metrics.get("completeness", 0.5) * 0.25 +
            quality_metrics.get("accuracy", 0.5) * 0.35 +
            quality_metrics.get("uniqueness", 0.5) * 0.25 +
            quality_metrics.get("usefulness", 0.5) * 0.15
        )
        
        # Peer review bonus
        review_bonus = min(len(verified_reviews) / 5, 1.0)  # Up to 100% bonus for 5+ reviews
        
        value = base_value * Decimal(str(quality_multiplier)) * Decimal(str(1.0 + review_bonus))
        quality = quality_multiplier
        confidence = min(1.0, len(verified_reviews) / 3)  # Max confidence with 3+ reviews
        
        return VerificationResult(
            hash=proof_data["dataset_hash"],
            value=value,
            quality=quality,
            confidence=confidence,
            metadata={
                "verified_reviews": len(verified_reviews),
                "quality_breakdown": quality_metrics,
                "review_bonus": review_bonus
            }
        )
    
    async def _verify_governance_proof(self, user_id: str, proof_data: Dict) -> VerificationResult:
        """Verify governance participation"""
        
        required_fields = ["votes_cast", "proposals_reviewed", "participation_period"]
        if not all(field in proof_data for field in required_fields):
            raise ValueError("Missing required governance proof fields")
        
        # Verify with governance service if available
        if self.governance:
            verified_votes = await self.governance.verify_user_votes(
                user_id, proof_data["votes_cast"]
            )
            verified_reviews = await self.governance.verify_user_proposal_reviews(
                user_id, proof_data["proposals_reviewed"]
            )
        else:
            # Fallback verification (for testing)
            verified_votes = len(proof_data.get("votes_cast", []))
            verified_reviews = len(proof_data.get("proposals_reviewed", []))
        
        if verified_votes == 0 and verified_reviews == 0:
            raise ValueError("No verifiable governance participation found")
        
        # Calculate value based on participation
        vote_value = verified_votes * Decimal("2.0")  # 2 FTNS per vote
        review_value = verified_reviews * Decimal("5.0")  # 5 FTNS per proposal review
        value = vote_value + review_value
        
        # Quality based on participation diversity
        participation_score = min(1.0, (verified_votes / 10) * 0.6 + (verified_reviews / 5) * 0.4)
        quality = participation_score
        
        confidence = 0.9  # High confidence for governance records
        
        # Generate proof hash
        proof_content = {
            "user_id": user_id,
            "verified_votes": verified_votes,
            "verified_reviews": verified_reviews,
            "period": proof_data["participation_period"]
        }
        proof_hash = hashlib.sha256(json.dumps(proof_content, sort_keys=True).encode()).hexdigest()
        
        return VerificationResult(
            hash=proof_hash,
            value=value,
            quality=quality,
            confidence=confidence,
            metadata={
                "votes_verified": verified_votes,
                "reviews_verified": verified_reviews,
                "participation_score": participation_score
            }
        )
    
    async def _verify_documentation_proof(self, user_id: str, proof_data: Dict) -> VerificationResult:
        """Verify documentation contribution"""
        
        required_fields = ["documents_contributed", "peer_approvals", "content_hash"]
        if not all(field in proof_data for field in required_fields):
            raise ValueError("Missing required documentation proof fields")
        
        # Verify peer approvals
        verified_approvals = 0
        for approval in proof_data["peer_approvals"]:
            if await self._verify_peer_approval(approval):
                verified_approvals += 1
        
        if verified_approvals < 1:
            raise ValueError("No verified peer approvals found")
        
        # Calculate value
        doc_count = len(proof_data["documents_contributed"])
        base_value = Decimal("5.0") * doc_count  # 5 FTNS per document
        
        # Approval bonus
        approval_multiplier = min(verified_approvals / 3, 2.0)  # Up to 2x for 3+ approvals
        value = base_value * Decimal(str(approval_multiplier))
        
        # Quality based on approval ratio
        quality = min(1.0, verified_approvals / len(proof_data["peer_approvals"]))
        confidence = 0.85
        
        return VerificationResult(
            hash=proof_data["content_hash"],
            value=value,
            quality=quality,
            confidence=confidence,
            metadata={
                "documents_count": doc_count,
                "verified_approvals": verified_approvals,
                "approval_multiplier": approval_multiplier
            }
        )
    
    async def _verify_model_proof(self, user_id: str, proof_data: Dict) -> VerificationResult:
        """Verify AI model contribution"""
        
        required_fields = ["model_hash", "performance_metrics", "validation_results"]
        if not all(field in proof_data for field in required_fields):
            raise ValueError("Missing required model proof fields")
        
        # Verify model performance
        performance_metrics = proof_data["performance_metrics"]
        validation_results = proof_data["validation_results"]
        
        # Calculate value based on model performance and novelty
        base_value = Decimal("100.0")  # Base value for model contribution
        
        # Performance multiplier
        accuracy = performance_metrics.get("accuracy", 0.5)
        efficiency = performance_metrics.get("efficiency", 0.5)
        novelty = performance_metrics.get("novelty", 0.5)
        
        performance_score = (accuracy * 0.4 + efficiency * 0.3 + novelty * 0.3)
        performance_multiplier = max(0.5, performance_score * 2)  # 0.5x to 2x multiplier
        
        value = base_value * Decimal(str(performance_multiplier))
        quality = performance_score
        confidence = 0.8
        
        return VerificationResult(
            hash=proof_data["model_hash"],
            value=value,
            quality=quality,
            confidence=confidence,
            metadata={
                "performance_metrics": performance_metrics,
                "performance_multiplier": performance_multiplier
            }
        )
    
    async def _verify_research_proof(self, user_id: str, proof_data: Dict) -> VerificationResult:
        """Verify research publication contribution"""
        
        required_fields = ["publication_hash", "citations", "peer_review_status"]
        if not all(field in proof_data for field in required_fields):
            raise ValueError("Missing required research proof fields")
        
        # Calculate value based on publication impact
        base_value = Decimal("500.0")  # High base value for research
        
        citation_count = len(proof_data.get("citations", []))
        peer_reviewed = proof_data.get("peer_review_status", False)
        
        # Impact multiplier
        citation_multiplier = min(citation_count / 10, 2.0)  # Up to 2x for 10+ citations
        peer_review_bonus = 1.5 if peer_reviewed else 1.0
        
        impact_multiplier = citation_multiplier * peer_review_bonus
        value = base_value * Decimal(str(impact_multiplier))
        
        # Quality based on peer review and citations
        quality = min(1.0, 0.6 + (citation_count / 20) * 0.4)
        if peer_reviewed:
            quality = min(1.0, quality + 0.2)
        
        confidence = 0.9
        
        return VerificationResult(
            hash=proof_data["publication_hash"],
            value=value,
            quality=quality,
            confidence=confidence,
            metadata={
                "citation_count": citation_count,
                "peer_reviewed": peer_reviewed,
                "impact_multiplier": impact_multiplier
            }
        )
    
    async def _verify_teaching_proof(self, user_id: str, proof_data: Dict) -> VerificationResult:
        """Verify teaching/educational contribution"""
        
        required_fields = ["content_hash", "student_feedback", "completion_rates"]
        if not all(field in proof_data for field in required_fields):
            raise ValueError("Missing required teaching proof fields")
        
        # Calculate value based on teaching effectiveness
        base_value = Decimal("20.0")  # Base value per teaching session
        
        feedback_scores = proof_data.get("student_feedback", [])
        completion_rate = proof_data.get("completion_rates", 0.5)
        student_count = len(feedback_scores)
        
        # Effectiveness multiplier
        avg_feedback = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0.5
        effectiveness_score = (avg_feedback * 0.6 + completion_rate * 0.4)
        
        # Scale value by student count and effectiveness
        scale_multiplier = min(student_count / 10, 3.0)  # Up to 3x for 10+ students
        value = base_value * Decimal(str(scale_multiplier * effectiveness_score))
        
        quality = effectiveness_score
        confidence = min(1.0, student_count / 5)  # Max confidence with 5+ students
        
        return VerificationResult(
            hash=proof_data["content_hash"],
            value=value,
            quality=quality,
            confidence=confidence,
            metadata={
                "student_count": student_count,
                "avg_feedback": avg_feedback,
                "completion_rate": completion_rate,
                "effectiveness_score": effectiveness_score
            }
        )
    
    # === STATUS MANAGEMENT ===
    
    async def update_contributor_status(self, user_id: str) -> ContributorTier:
        """Update user's contributor status based on recent proofs"""
        
        # Check cooldown period
        status_record = await self._get_contributor_status_record(user_id)
        if status_record and status_record.last_status_update:
            # Ensure timezone compatibility
            now = datetime.now(timezone.utc)
            last_update = status_record.last_status_update
            if last_update.tzinfo is None:
                last_update = last_update.replace(tzinfo=timezone.utc)
            
            hours_since_update = (now - last_update).total_seconds() / 3600
            if hours_since_update < self.status_update_cooldown_hours:
                return ContributorTier(status_record.status)
        
        # Get recent proofs (last 30 days)
        recent_proofs = await self._get_recent_proofs(user_id, days=30)
        
        if not recent_proofs:
            new_tier = ContributorTier.NONE
            await self._set_contributor_status(user_id, new_tier, 0.0)
            return new_tier
        
        # Calculate comprehensive contribution score
        total_score = 0.0
        quality_sum = 0.0
        contribution_types = set()
        
        for proof in recent_proofs:
            # Weight by contribution type
            type_weight = self.scoring_weights.get(ContributionType(proof.contribution_type), 1.0)
            
            # Calculate weighted contribution score
            contribution_score = (
                float(proof.contribution_value) * 
                proof.quality_score * 
                proof.verification_confidence * 
                type_weight
            )
            
            total_score += contribution_score
            quality_sum += proof.quality_score
            contribution_types.add(proof.contribution_type)
        
        # Diversity bonus for multiple contribution types (1.0x base, up to 1.5x for 4+ types)
        diversity_multiplier = 1.0 + min(len(contribution_types) - 1, 3) * 0.125  # +12.5% per additional type, capped at 37.5%
        final_score = total_score * diversity_multiplier
        
        average_quality = quality_sum / len(recent_proofs) if recent_proofs else 0
        
        # Determine tier based on score and quality
        await logger.adebug(
            "Calculating contributor tier",
            user_id=user_id,
            final_score=final_score,
            average_quality=average_quality,
            total_score=total_score,
            diversity_multiplier=diversity_multiplier,
            contribution_types=list(contribution_types)
        )
        
        if final_score >= self.tier_thresholds[ContributorTier.POWER_USER] and average_quality >= 0.8:
            new_tier = ContributorTier.POWER_USER
        elif final_score >= self.tier_thresholds[ContributorTier.ACTIVE] and average_quality >= 0.6:
            new_tier = ContributorTier.ACTIVE
        elif final_score >= self.tier_thresholds[ContributorTier.BASIC] and average_quality >= 0.3:
            new_tier = ContributorTier.BASIC
        else:
            new_tier = ContributorTier.NONE
        
        await self._set_contributor_status(user_id, new_tier, final_score)
        return new_tier
    
    async def can_earn_ftns(self, user_id: str) -> bool:
        """Check if user is eligible to earn new FTNS"""
        
        status = await self.get_contributor_status(user_id)
        if not status:
            return False
            
        return status.status != ContributorTier.NONE.value
    
    async def get_contribution_multiplier(self, user_id: str) -> float:
        """Get earning multiplier based on contribution tier"""
        
        status = await self.get_contributor_status(user_id)
        if not status:
            return 0.0
        
        multipliers = {
            ContributorTier.NONE.value: 0.0,         # Cannot earn
            ContributorTier.BASIC.value: 1.0,        # Standard rate
            ContributorTier.ACTIVE.value: 1.3,       # 30% bonus
            ContributorTier.POWER_USER.value: 1.6    # 60% bonus
        }
        
        return multipliers.get(status.status, 0.0)
    
    async def get_contributor_status(self, user_id: str) -> Optional[FTNSContributorStatus]:
        """Get current contributor status record"""
        
        return await self._get_contributor_status_record(user_id)
    
    # === HELPER METHODS ===
    
    async def _get_contributor_status_record(self, user_id: str) -> Optional[FTNSContributorStatus]:
        """Get contributor status record from database"""
        
        result = await self.db.execute(
            select(FTNSContributorStatus).where(FTNSContributorStatus.user_id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def _set_contributor_status(self, user_id: str, tier: ContributorTier, score: float):
        """Set or update contributor status"""
        
        status_record = await self._get_contributor_status_record(user_id)
        now = datetime.now(timezone.utc)
        
        if status_record:
            # Update existing record
            status_record.status = tier.value
            status_record.contribution_score = Decimal(str(score))
            status_record.last_status_update = now
            status_record.updated_at = now
            
            if tier != ContributorTier.NONE:
                status_record.last_contribution_date = now
        else:
            # Create new record
            status_record = FTNSContributorStatus(
                user_id=user_id,
                status=tier.value,
                contribution_score=Decimal(str(score)),
                last_contribution_date=now if tier != ContributorTier.NONE else None,
                last_status_update=now
            )
            self.db.add(status_record)
        
        await self.db.commit()
    
    async def _get_recent_proofs(self, user_id: str, days: int = 30) -> List[FTNSContributionProof]:
        """Get recent verified contribution proofs for a user"""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Ensure we see all committed changes by refreshing the session
        await self.db.commit()
        
        result = await self.db.execute(
            select(FTNSContributionProof)
            .where(
                and_(
                    FTNSContributionProof.user_id == user_id,
                    FTNSContributionProof.verification_status == ProofStatus.VERIFIED.value,
                    FTNSContributionProof.verification_timestamp >= cutoff_date
                )
            )
            .order_by(FTNSContributionProof.verification_timestamp.desc())
        )
        
        proofs = result.scalars().all()
        return proofs
    
    async def _verify_computation_hash(self, computation_hash: str, work_units: int) -> bool:
        """Verify a compute-contribution proof hash.

        sp1265: shape checks (SHA256 length, positive work_units) gate obvious junk, but they
        do NOT bind the hash to work the system actually witnessed — the rubber-stamp
        ("assume valid") let a forged 64-hex string mint contributor value. Real binding to
        the node's signed §7 inference/compute receipts is a larger follow-on; until it
        lands, PRSM_CONTRIBUTOR_PROOF_STRICT lets a money-live operator FAIL CLOSED (reject
        any compute proof we cannot cryptographically bind) rather than accept unverifiable
        work. Default keeps the prior behavior but warns loudly.
        """
        if not computation_hash or len(computation_hash) != 64:  # SHA256 length
            return False

        if work_units <= 0:
            return False

        if os.getenv("PRSM_CONTRIBUTOR_PROOF_STRICT", "").strip().lower() in ("1", "true", "yes", "on"):
            logger.error("compute proof rejected — PRSM_CONTRIBUTOR_PROOF_STRICT is on and "
                         "the hash cannot be cryptographically bound to witnessed work "
                         "(receipt-binding not yet wired)", computation_hash=computation_hash[:12])
            return False

        logger.warning("compute proof accepted on SHAPE ONLY — NOT cryptographically verified "
                       "against witnessed work; do not gate real rewards on this until "
                       "receipt-binding lands (or set PRSM_CONTRIBUTOR_PROOF_STRICT)",
                       computation_hash=computation_hash[:12], work_units=work_units)
        return True

    async def _verify_peer_review(self, review: Dict, *, submitter_id: str = None,
                                  seen: set = None) -> bool:
        """Verify a peer review submission.

        sp1265: authenticate the review against the submitter context so a contributor can't
        fabricate their own reviews. Reject self-reviews (reviewer is the submitter) and
        duplicate reviewers within a submission (`seen`), and require a non-empty review hash.
        (Full reviewer authentication — signature + a distinct registered/KYC'd identity — is
        a follow-on; these checks close the concrete self-fabrication / padding vectors now.)
        """
        required_fields = ["reviewer_id", "review_hash", "rating"]
        if not all(field in review for field in required_fields):
            return False

        reviewer_id = review.get("reviewer_id")
        if not reviewer_id or not str(review.get("review_hash") or "").strip():
            return False

        # A submitter cannot peer-review their own contribution.
        if submitter_id is not None and str(reviewer_id) == str(submitter_id):
            return False

        # Distinct reviewers only — no padding the count with one repeated reviewer.
        if seen is not None:
            if str(reviewer_id) in seen:
                return False
            seen.add(str(reviewer_id))

        rating = review.get("rating", 0)
        if not (1 <= rating <= 5):
            return False

        return True
    
    async def _verify_peer_approval(self, approval: Dict) -> bool:
        """Verify a peer approval for documentation"""
        
        required_fields = ["approver_id", "approval_hash", "approved"]
        if not all(field in approval for field in required_fields):
            return False
            
        return approval.get("approved", False)
    
    def _get_validation_criteria(self, contribution_type: str) -> Dict[str, Any]:
        """Get validation criteria for a contribution type"""
        
        criteria = {
            ContributionType.STORAGE.value: {
                "min_duration_hours": 24,
                "min_storage_gb": 1.0,
                "max_redundancy_factor": 3.0
            },
            ContributionType.COMPUTE.value: {
                "min_work_units": 1,
                "min_benchmark_score": 0.1,
                "max_benchmark_score": 2.0
            },
            ContributionType.DATA.value: {
                "min_peer_reviews": 2,
                "required_quality_metrics": ["completeness", "accuracy", "uniqueness", "usefulness"]
            },
            ContributionType.GOVERNANCE.value: {
                "min_participation_actions": 1
            },
            ContributionType.DOCUMENTATION.value: {
                "min_peer_approvals": 1
            },
            ContributionType.MODEL.value: {
                "required_metrics": ["accuracy", "efficiency"],
                "min_accuracy": 0.5
            },
            ContributionType.RESEARCH.value: {
                "peer_review_preferred": True
            },
            ContributionType.TEACHING.value: {
                "min_students": 1,
                "min_completion_rate": 0.3
            }
        }
        
        return criteria.get(contribution_type, {})
    
    async def _log_proof_verification(self, user_id: str, contribution_type: str, result: VerificationResult):
        """Log proof verification for audit trail"""
        
        logger.info(
            "Contribution proof verified",
            user_id=user_id,
            contribution_type=contribution_type,
            value=float(result.value),
            quality=result.quality,
            confidence=result.confidence,
            proof_hash=result.hash
        )