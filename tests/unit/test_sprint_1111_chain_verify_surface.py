"""Sprint 1111 — Domain-03 F1/F2 brick 5: surface the per-stage signed activation-chain
verification in /compute/receipt/verify's response.

summarize_stage_activation_chain() is the self-contained verifier the endpoint calls:
internal links + derived topology always; per-stage signatures when the caller supplies
the stage keys; topology cross-check when a topology_assignment is present.
"""
from __future__ import annotations

import pytest

from prsm.compute.inference.stage_activation_proof import (
    StageActivationChain,
    StageActivationProof,
    activation_hash,
    summarize_stage_activation_chain,
)
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.node.identity import generate_node_identity


def _signed(identity, req, idx, hin, hout):
    base = StageActivationProof(idx, identity.node_id, hin, hout)
    return StageActivationProof(idx, identity.node_id, hin, hout,
                                stage_signature_b64=identity.sign(base.signing_bytes(req)))


def _chain(req="req-1"):
    a, b = generate_node_identity("a"), generate_node_identity("b")
    h0, h1, h2 = activation_hash(b"x"), activation_hash(b"y"), activation_hash(b"z")
    chain = StageActivationChain(req, [_signed(a, req, 0, h0, h1), _signed(b, req, 1, h1, h2)])
    keys = {a.node_id: a.public_key_b64, b.node_id: b.public_key_b64}
    return chain, keys, a, b


# ── internal links (no keys, no endpoints) ─────────────────────────────────────────

def test_internal_links_ok_for_good_chain():
    chain, _, _, _ = _chain()
    ok, _why = chain.verify_internal_links()
    assert ok


def test_internal_links_detect_break():
    h0, h1, h2 = activation_hash(b"x"), activation_hash(b"y"), activation_hash(b"z")
    bad = StageActivationChain("r", [
        StageActivationProof(0, "a", h0, h1),
        StageActivationProof(1, "b", "MISMATCH", h2)])
    ok, why = bad.verify_internal_links()
    assert not ok and "between stage" in why


# ── summary helper ──────────────────────────────────────────────────────────────────

def test_summary_without_keys_reports_structure_only():
    chain, _, _, _ = _chain()
    s = summarize_stage_activation_chain(chain)
    assert s["present"] is True
    assert s["stage_count"] == 2
    assert s["internal_links_ok"] is True
    assert s["signatures_checked"] is False
    assert "node_ids" in s


def test_summary_with_keys_verifies_signatures():
    chain, keys, _, _ = _chain()
    s = summarize_stage_activation_chain(chain, stage_public_keys=keys)
    assert s["signatures_checked"] is True
    assert s["signatures_valid"] is True


def test_summary_with_wrong_keys_fails_signatures():
    chain, _, _, _ = _chain()
    bogus = generate_node_identity("z")
    keys = {nid: bogus.public_key_b64 for nid in chain.topology_node_ids()}
    s = summarize_stage_activation_chain(chain, stage_public_keys=keys)
    assert s["signatures_valid"] is False


def test_summary_topology_cross_check():
    chain, keys, a, b = _chain()
    good = TopologyAssignment(
        positions={(0, 0): a.node_id, (1, 0): b.node_id}, stage_count=2, slots_per_stage=1)
    s = summarize_stage_activation_chain(chain, topology_assignment=good)
    assert s["topology_consistent"] is True

    lying = TopologyAssignment(
        positions={(0, 0): a.node_id, (1, 0): "imposter"}, stage_count=2, slots_per_stage=1)
    s2 = summarize_stage_activation_chain(chain, topology_assignment=lying)
    assert s2["topology_consistent"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
