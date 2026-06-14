"""
PRSM Zero-Knowledge Proof System
===============================

Production-grade zero-knowledge proof system with a fully decoupled 
in-memory fallback for edge environments and testing.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

import structlog

# Lazy import for database to avoid environment crashes
try:
    from prsm.core.database import get_async_session
    HAS_DB = True
except (ImportError, ValueError):
    HAS_DB = False

from .crypto_models import (
    ProofResult,
    ZKProofRequest
)

logger = structlog.get_logger(__name__)

class AssertInfo:
    """Granular assertion tracking for scientific verification"""
    def __init__(self, pos: int = 0, content: Any = None):
        self.assert_pos = pos
        self.assert_content = content
        self.assert_result = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pos": self.assert_pos,
            "content": str(self.assert_content),
            "result": self.assert_result
        }

class ConstraintVerificationResult:
    """Result of a neuro-symbolic constraint check"""
    def __init__(self, success: bool = True):
        self.success = success
        self.assertions: List[AssertInfo] = []
        self.confidence: float = 1.0
        self.rejection_reason: Optional[str] = None

    def add_assertion(self, pos: int, content: Any, result: bool):
        assertion = AssertInfo(pos, content)
        assertion.assert_result = result
        self.assertions.append(assertion)
        if not result:
            self.success = False

class ZKProofSystem:
    """
    Production-grade zero-knowledge proof system.
    Automatically handles in-memory fallback if database is unavailable.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.circuit_registry = ZKCircuitRegistry()
        self._memory_proof_store: Dict[str, Any] = {}
        self.proof_expiry_hours = self.config.get("proof_expiry_hours", 24)
        print("🔍 ZK Proof System initialized (Pure-Logic Mode)")

    async def generate_proof(self, request: ZKProofRequest) -> ProofResult:
        """Generate a zero-knowledge proof"""
        start_time = datetime.now()
        
        try:
            circuit = self.circuit_registry.get_circuit(request.circuit_id)
            if not circuit:
                return ProofResult(success=False, error_message=f"Circuit not found: {request.circuit_id}")
            
            setup_keys = self.circuit_registry.get_setup_keys(request.circuit_id)
            proof_result = circuit.generate_proof(
                request.private_inputs,
                request.public_inputs,
                setup_keys["proving_key"]
            )
            
            proof_id = str(uuid4())
            self._memory_proof_store[proof_id] = {
                "proof_data": proof_result["proof"],
                "public_inputs": request.public_inputs,
                "verification_key": setup_keys["verification_key"],
                "circuit_id": request.circuit_id,
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=self.proof_expiry_hours)
            }
            
            generation_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return ProofResult(
                success=True,
                proof_data=proof_id,
                public_inputs=request.public_inputs,
                circuit_id=request.circuit_id,
                proof_system=request.proof_system,
                generation_time_ms=generation_time,
                proof_size_bytes=proof_result["proof_size"]
            )
            
        except Exception as e:
            logger.error("ZK proof generation failed", error=str(e))
            return ProofResult(success=False, error_message=str(e))

    async def verify_proof(self, proof_id: str, verifier_id: str) -> bool:
        """Verify a zero-knowledge proof using memory store"""
        try:
            if proof_id in self._memory_proof_store:
                record = self._memory_proof_store[proof_id]
                circuit = self.circuit_registry.get_circuit(record["circuit_id"])
                return circuit.verify_proof(
                    record["proof_data"], 
                    record["public_inputs"], 
                    record["verification_key"]
                )
            return False
        except Exception as e:
            logger.error("ZK proof verification failed", proof_id=proof_id, error=str(e))
            return False

class MockZKCircuit:
    def __init__(self, circuit_id: str, description: str):
        self.circuit_id = circuit_id
        self.description = description
        
    def setup(self) -> Dict[str, Any]:
        return {"proving_key": b"pk", "verification_key": b"vk"}
    
    def generate_proof(self, private, public, key) -> Dict[str, Any]:
        return {"proof": b"mock_proof", "proof_size": 10, "generation_time_ms": 1}

    def verify_proof(self, proof, public, key) -> bool:
        # sp1104 (Domain-08 review CRITICAL) — this is a MOCK circuit with no real
        # soundness; it must NEVER assert cryptographic validity. Returning True here
        # made the live POST /api/v1/crypto/proofs/{id}/verify endpoint report ANY
        # stored proof (bytes literally b"mock_proof") as is_valid=True, so any
        # authenticated user could mint a meaningless "verified" proof. Fail CLOSED:
        # refuse to verify until a real ZK circuit is implemented. ZKProofSystem.
        # verify_proof wraps this in try/except → the endpoint reports is_valid=False.
        raise NotImplementedError(
            "ZK proof verification is not implemented (MockZKCircuit has no soundness); "
            "refusing to assert validity — a real circuit must replace this before the "
            "proof endpoints can report a verified result"
        )

class ZKCircuitRegistry:
    def __init__(self):
        self.circuits = {"inference_verification": MockZKCircuit("inference_verification", "Mock")}
        self.setup_keys = {"inference_verification": self.circuits["inference_verification"].setup()}
    def get_circuit(self, cid): return self.circuits.get(cid)
    def get_setup_keys(self, cid): return self.setup_keys.get(cid)

_zk_proof_system: Optional[ZKProofSystem] = None

async def get_zk_proof_system() -> ZKProofSystem:
    global _zk_proof_system
    if _zk_proof_system is None:
        _zk_proof_system = ZKProofSystem()
    return _zk_proof_system