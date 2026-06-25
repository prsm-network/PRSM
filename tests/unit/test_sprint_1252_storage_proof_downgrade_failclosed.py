"""Sprint 1252 — storage proof-of-storage verifier: close the proof-type downgrade
and fail-close the no-op RANGE/FULL verifiers.

Found by fail-open audit round 2 (workflow w2a1ewusm). The remote proof handler
(storage_provider._on_storage_proof_response, gossip + direct-P2P reachable) calls
verify_proof on attacker-submitted proofs. Two fail-opens:
  - verify_proof dispatched on the ATTACKER-controlled proof.proof_type with NO check
    against challenge.proof_type → a provider could DOWNGRADE an issued MERKLE
    challenge into the weaker RANGE/FULL paths;
  - _verify_range_proof / _verify_full_proof were no-ops that accepted ANY
    sufficient-length / non-empty bytes as VERIFIED (no content binding) — a provider
    storing nothing could forge a "proof of storage".
Impact is currently latent (the FTNS-mint reward path is dead code, slashing is 0.0,
the reputation written is observability-only), but it's a forged-trusted-state
primitive that turns critical the moment storage rewards / reputation-gated
selection / slashing wire up.

Fix (sp1252, synthesis priorities #1+#2): reject proof_type != challenge.proof_type,
and fail-close the unimplemented RANGE/FULL verifiers. The live system only ever
issues MERKLE challenges, so legitimate proofs are unaffected. (Deeper follow-on
sp1253: bind the MERKLE root to trusted challenge-derived state + require the provider
signature on the remote path + gate any future reward/slashing on content-binding.)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from prsm.node.storage_proofs import (
    MerkleProofGenerator,
    ProofType,
    StorageChallenge,
    StorageProof,
    StorageProofVerifier,
)

CONTENT = b"the quick brown fox jumps over the lazy dog" * 64
CID = "cid-sample-0001"


def _verifier():
    return StorageProofVerifier()


def _challenge(verifier, proof_type=ProofType.MERKLE):
    return verifier.generate_challenge(
        shard_hash=CID, challenger_id="challenger-1", proof_type=proof_type)


def _proof(challenge, *, proof_type, proof_data=b"x" * 4096, merkle_proof=None):
    return StorageProof(
        challenge_id=challenge.challenge_id,
        provider_id="provider-1",
        shard_hash=CID,
        proof_type=proof_type,
        proof_data=proof_data,
        timestamp=datetime.now(timezone.utc),
        signature="sig",
        merkle_proof=merkle_proof,
    )


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_proof_type_downgrade_rejected():
    # MERKLE challenge issued; attacker answers with proof_type=RANGE to route around
    # the merkle check into the (former) no-op path. Must be rejected on the mismatch.
    v = _verifier()
    ch = _challenge(v, ProofType.MERKLE)
    forged = _proof(ch, proof_type=ProofType.RANGE, proof_data=b"A" * 100_000)
    ok, err = _run(v.verify_proof(forged, ch))
    assert ok is False
    assert "does not match" in err.lower() or "downgrade" in err.lower()


def test_full_downgrade_rejected():
    v = _verifier()
    ch = _challenge(v, ProofType.MERKLE)
    forged = _proof(ch, proof_type=ProofType.FULL, proof_data=b"junk")
    ok, err = _run(v.verify_proof(forged, ch))
    assert ok is False


def test_range_proof_fails_closed_even_when_type_matches():
    # even a RANGE proof answering a RANGE challenge must now fail closed (the no-op
    # size-only verifier is gone) — no content binding ⇒ not verifiable.
    v = _verifier()
    ch = _challenge(v, ProofType.RANGE)
    p = _proof(ch, proof_type=ProofType.RANGE, proof_data=b"B" * 1_000_000)
    ok, err = _run(v.verify_proof(p, ch))
    assert ok is False
    assert "not implemented" in err.lower()


def test_full_proof_fails_closed_even_when_type_matches():
    v = _verifier()
    ch = _challenge(v, ProofType.FULL)
    p = _proof(ch, proof_type=ProofType.FULL, proof_data=b"whole content bytes")
    ok, err = _run(v.verify_proof(p, ch))
    assert ok is False
    assert "not implemented" in err.lower()


def test_legit_merkle_proof_still_verifies():
    # NO REGRESSION: a genuine MERKLE proof for a MERKLE challenge must still pass.
    v = _verifier()
    ch = _challenge(v, ProofType.MERKLE)
    merkle = MerkleProofGenerator()
    _tree, merkle_proof, chunk_data = merkle.generate_challenge_proof(
        content=CONTENT, nonce=ch.nonce, difficulty=ch.difficulty)
    p = _proof(ch, proof_type=ProofType.MERKLE, proof_data=chunk_data, merkle_proof=merkle_proof)
    ok, err = _run(v.verify_proof(p, ch))
    assert ok is True, err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
