"""Sprint 1253 — storage MERKLE proof must be bound to a TRUSTED root (close the
forge-your-own-root fail-open; the latent→critical tripwire from audit round 2).

sp1252 closed the proof-type downgrade + no-op RANGE/FULL verifiers. The MERKLE path
remained forgeable: _verify_merkle_proof checked internal leaf→root consistency up to
proof.merkle_proof.root_hash — an ATTACKER-SUPPLIED field never compared to a trusted
root — so a provider storing nothing could fabricate a self-consistent tree with its
own root and pass.

Fix: verify_proof now takes a TRUSTED expected_merkle_root; _verify_merkle_proof binds
proof.merkle_proof.root_hash to it and FAILS CLOSED when none is supplied. The caller
recomputes the trusted root from content it actually holds (storage_provider.
_compute_trusted_merkle_root via the content client); the remote path passes None when
it doesn't hold the CID → fail closed (it cannot independently verify storage of content
it has no trusted root for).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from prsm.node.storage_proofs import (
    MerkleProofGenerator,
    ProofType,
    StorageChallenge,
    StorageProof,
    StorageProofVerifier,
)

CONTENT = b"genuine stored content block " * 200
OTHER_CONTENT = b"completely different content " * 200
CID = "cid-1253"


def _verifier():
    return StorageProofVerifier()


def _challenge(v):
    return v.generate_challenge(shard_hash=CID, challenger_id="challenger-1",
                                proof_type=ProofType.MERKLE)


def _merkle_proof_for(content, challenge):
    merkle = MerkleProofGenerator()
    tree, merkle_proof, chunk_data = merkle.generate_challenge_proof(
        content=content, nonce=challenge.nonce, difficulty=challenge.difficulty)
    proof = StorageProof(
        challenge_id=challenge.challenge_id,
        provider_id="provider-1",
        shard_hash=CID,
        proof_type=ProofType.MERKLE,
        proof_data=chunk_data,
        timestamp=datetime.now(timezone.utc),
        signature="sig",
        merkle_proof=merkle_proof,
    )
    return proof, tree.root_hash


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_no_trusted_root_fails_closed():
    # the headline: without a trusted root, a MERKLE proof is unverifiable → reject.
    v = _verifier()
    ch = _challenge(v)
    proof, _root = _merkle_proof_for(CONTENT, ch)
    ok, err = _run(v.verify_proof(proof, ch))  # expected_merkle_root defaults to None
    assert ok is False
    assert "trusted root" in err.lower()


def test_forged_self_consistent_proof_rejected_against_trusted_root():
    # THE forgery: a provider that does NOT store CONTENT fabricates a self-consistent
    # proof from OTHER_CONTENT (internally valid, its own root). The verifier binds to
    # the TRUSTED root of the real CONTENT → the forged root mismatches → rejected.
    v = _verifier()
    ch = _challenge(v)
    forged, _forged_root = _merkle_proof_for(OTHER_CONTENT, ch)
    trusted_root = MerkleProofGenerator().build_merkle_tree(CONTENT).root_hash
    ok, err = _run(v.verify_proof(forged, ch, expected_merkle_root=trusted_root))
    assert ok is False
    assert "does not match the trusted root" in err.lower()


def test_legit_proof_with_matching_trusted_root_verifies():
    # a genuine proof for CONTENT, bound to CONTENT's real root, verifies.
    v = _verifier()
    ch = _challenge(v)
    proof, root = _merkle_proof_for(CONTENT, ch)
    # the independently-recomputed trusted root equals the proof's root for real content
    trusted_root = MerkleProofGenerator().build_merkle_tree(CONTENT).root_hash
    assert root == trusted_root
    ok, err = _run(v.verify_proof(proof, ch, expected_merkle_root=trusted_root))
    assert ok is True, err


def test_tampered_proof_data_still_rejected_with_trusted_root():
    # even with the right trusted root, proof_data not matching the leaf is rejected.
    v = _verifier()
    ch = _challenge(v)
    proof, root = _merkle_proof_for(CONTENT, ch)
    proof.proof_data = b"tampered chunk bytes"
    ok, err = _run(v.verify_proof(proof, ch, expected_merkle_root=root))
    assert ok is False
    assert "leaf" in err.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
