"""Sprint 1108 — Domain-03 F1/F2 brick 2: verify the per-stage signed activation chain.

Brick 1 gave the receptacle + continuity check. This brick adds the AUTHENTICITY half:
each stage proof's signature must verify under the node_id's anchor-published key, and the
orchestrator's asserted topology must be consistent with the signed stages (F2). A fully
verified chain means: this prompt produced this output through exactly these stages, each
run by the node that cryptographically signed for it.
"""
from __future__ import annotations

import pytest

from prsm.compute.inference.stage_activation_proof import (
    StageActivationChain,
    StageActivationProof,
    activation_hash,
    chain_supports_topology,
    verify_stage_activation_chain,
    verify_stage_proof,
)
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.node.identity import generate_node_identity


class _Anchor:
    def __init__(self, mapping):
        self._m = mapping

    def lookup(self, node_id):
        return self._m.get(node_id)


def _signed_proof(identity, request_id, stage_index, hin, hout):
    unsigned = StageActivationProof(
        stage_index=stage_index, stage_node_id=identity.node_id,
        input_activation_hash=hin, output_activation_hash=hout)
    sig = identity.sign(unsigned.signing_bytes(request_id))
    return StageActivationProof(
        stage_index=stage_index, stage_node_id=identity.node_id,
        input_activation_hash=hin, output_activation_hash=hout,
        stage_signature_b64=sig)


def _two_stage_chain(req="req-1"):
    a = generate_node_identity("alice")
    b = generate_node_identity("bob")
    h0, h1, h2 = activation_hash(b"prompt"), activation_hash(b"mid"), activation_hash(b"out")
    chain = StageActivationChain(
        request_id=req,
        proofs=[
            _signed_proof(a, req, 0, h0, h1),
            _signed_proof(b, req, 1, h1, h2),
        ],
    )
    anchor = _Anchor({a.node_id: a.public_key_b64, b.node_id: b.public_key_b64})
    return chain, anchor, h0, h2, a, b


# ── per-proof signature ────────────────────────────────────────────────────────────

def test_valid_proof_verifies():
    chain, anchor, *_ = _two_stage_chain()
    assert verify_stage_proof(chain.proofs[0], chain.request_id, anchor=anchor) is True


def test_unsigned_proof_rejected():
    chain, anchor, *_ = _two_stage_chain()
    bad = StageActivationProof(0, chain.proofs[0].stage_node_id, "h", "h")  # no sig
    assert verify_stage_proof(bad, chain.request_id, anchor=anchor) is False


def test_proof_replayed_to_different_request_rejected():
    """A proof signed for req-1 must NOT verify when presented under req-2."""
    chain, anchor, *_ = _two_stage_chain(req="req-1")
    assert verify_stage_proof(chain.proofs[0], "req-2", anchor=anchor) is False


def test_unknown_node_rejected():
    chain, _, *_ = _two_stage_chain()
    empty_anchor = _Anchor({})
    assert verify_stage_proof(chain.proofs[0], chain.request_id, anchor=empty_anchor) is False


# ── full chain ──────────────────────────────────────────────────────────────────────

def test_full_chain_verifies():
    chain, anchor, h0, h2, *_ = _two_stage_chain()
    ok, why = verify_stage_activation_chain(
        chain, prompt_hash_hex=h0, output_hash_hex=h2, anchor=anchor)
    assert ok, why


def test_chain_with_one_forged_signature_rejected():
    """An attacker swaps stage 1's node to one whose key it doesn't control — the
    signature no longer verifies."""
    chain, anchor, h0, h2, a, b = _two_stage_chain()
    # forge: keep bob's hashes but claim a different (unknown-key) node signed
    forged = StageActivationProof(
        1, "z" * 32, chain.proofs[1].input_activation_hash,
        chain.proofs[1].output_activation_hash,
        stage_signature_b64=chain.proofs[1].stage_signature_b64)
    bad_chain = StageActivationChain(chain.request_id, [chain.proofs[0], forged])
    ok, why = verify_stage_activation_chain(
        bad_chain, prompt_hash_hex=h0, output_hash_hex=h2, anchor=anchor)
    assert not ok


def test_spliced_chain_rejected_before_signature_check():
    chain, anchor, h0, h2, *_ = _two_stage_chain()
    ok, why = verify_stage_activation_chain(
        chain, prompt_hash_hex="wrong-prompt", output_hash_hex=h2, anchor=anchor)
    assert not ok and "prompt entry" in why


# ── topology cross-check (F2) ─────────────────────────────────────────────────────

def test_topology_consistent_with_signed_stages():
    chain, _, _, _, a, b = _two_stage_chain()
    topo = TopologyAssignment(
        positions={(0, 0): a.node_id, (1, 0): b.node_id},
        stage_count=2, slots_per_stage=1)
    ok, why = chain_supports_topology(chain, topo)
    assert ok, why


def test_topology_that_lies_about_who_ran_a_stage_rejected():
    """F2: the orchestrator asserts stage 1 was run by some other node than the one that
    actually signed → inconsistency detected."""
    chain, _, _, _, a, b = _two_stage_chain()
    topo = TopologyAssignment(
        positions={(0, 0): a.node_id, (1, 0): "someone-else"},
        stage_count=2, slots_per_stage=1)
    ok, why = chain_supports_topology(chain, topo)
    assert not ok and "topology does not assign" in why


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
