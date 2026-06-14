"""Sprint 1104 — fold the Domain-08 review CRITICAL-1: the always-true ZK verifier.

`MockZKCircuit.verify_proof` returned ``True`` for ANY input, and it is the only
registered circuit. It is reached by the LIVE, mounted endpoint
``POST /api/v1/crypto/proofs/{proof_id}/verify`` (cryptography_api.verify_zk_proof →
ZKProofSystem.verify_proof → MockZKCircuit.verify_proof), which advertises
"Cryptographic proof verification" and reported is_valid=True for a proof whose bytes
are literally b"mock_proof". Any authenticated user could obtain a "verified" proof
that proves nothing — zero soundness.

A mock / not-yet-implemented ZK system must FAIL CLOSED: it must never assert
cryptographic validity. The fix makes the mock verifier refuse (raise), which the
ZKProofSystem wrapper turns into is_valid=False — so the endpoint can no longer be
tricked into reporting a forged proof as valid.
"""
from __future__ import annotations

import pytest

from prsm.core.cryptography.crypto_models import ZKProofRequest
from prsm.core.cryptography.zk_proofs import MockZKCircuit, ZKProofSystem


def test_mock_circuit_verify_does_not_return_true():
    """The mock circuit must NOT assert validity (it proves nothing). It must either
    raise or return False — never True."""
    circuit = MockZKCircuit("inference_verification", "mock")
    keys = circuit.setup()
    proof = circuit.generate_proof({"x": 1}, [1], keys["proving_key"])
    result = None
    try:
        result = circuit.verify_proof(proof["proof"], [1], keys["verification_key"])
    except NotImplementedError:
        return  # fail-closed by raising is acceptable
    assert result is not True, "mock ZK verifier must never report a proof valid"


@pytest.mark.asyncio
async def test_generated_proof_is_not_reported_valid_end_to_end():
    """The live path: generate a proof, then verify it. A mock system must report
    is_valid=False (fail-closed), NOT True — otherwise any user gets a meaningless
    'verified' proof."""
    system = ZKProofSystem()
    req = ZKProofRequest(
        circuit_id="inference_verification",
        private_inputs={"secret": 42},
        public_inputs=[1, 2, 3],
        statement="I know x",
        purpose="test",
    )
    gen = await system.generate_proof(req)
    assert gen.success is True
    proof_id = gen.proof_data

    is_valid = await system.verify_proof(proof_id, verifier_id="some-user")
    assert is_valid is False, (
        "a mock/unimplemented ZK system must fail closed — never report is_valid=True"
    )


@pytest.mark.asyncio
async def test_unknown_proof_id_is_false():
    """Regression: an unknown proof id stays False (was already the case)."""
    system = ZKProofSystem()
    assert await system.verify_proof("does-not-exist", "u") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
