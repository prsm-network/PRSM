"""Sprint 1112 — Domain-03 F1/F2 brick 6: bind the activation chain to the receipt's
request_id (cross-inference splice defense).

Each StageActivationProof is signed over chain.request_id, and the receipt binds
chain.stable_hash() — but nothing checked that chain.request_id equals the RECEIPT's
request_id. So a malicious head could splice a fully-valid chain from inference R1 (real
workers, valid signatures, intact links, consistent topology) into a receipt for R2 and
verify-time would pass. This brick binds the chain to the expected request_id at
verification.
"""
from __future__ import annotations

import pytest

from prsm.compute.inference.stage_activation_proof import (
    StageActivationChain,
    StageActivationProof,
    activation_hash,
    summarize_stage_activation_chain,
    verify_stage_activation_chain,
)
from prsm.node.identity import generate_node_identity


class _Anchor:
    def __init__(self, m):
        self._m = m

    def lookup(self, nid):
        return self._m.get(nid)


def _signed(identity, req, idx, hin, hout):
    base = StageActivationProof(idx, identity.node_id, hin, hout)
    return StageActivationProof(idx, identity.node_id, hin, hout,
                                stage_signature_b64=identity.sign(base.signing_bytes(req)))


def _chain(req):
    a, b = generate_node_identity("a"), generate_node_identity("b")
    h0, h1, h2 = activation_hash(b"x"), activation_hash(b"y"), activation_hash(b"z")
    chain = StageActivationChain(req, [_signed(a, req, 0, h0, h1), _signed(b, req, 1, h1, h2)])
    keys = {a.node_id: a.public_key_b64, b.node_id: b.public_key_b64}
    return chain, keys, h0, h2


def test_matching_request_id_is_bound():
    chain, keys, _, _ = _chain("req-MATCH")
    s = summarize_stage_activation_chain(
        chain, stage_public_keys=keys, expected_request_id="req-MATCH")
    assert s["request_id_bound"] is True
    assert s["signatures_valid"] is True  # genuinely valid for this request


def test_spliced_chain_from_other_inference_detected():
    """A VALID chain for req-R1 (signatures verify!) presented for a receipt whose
    request_id is req-R2 must be flagged unbound."""
    chain, keys, _, _ = _chain("req-R1")
    s = summarize_stage_activation_chain(
        chain, stage_public_keys=keys, expected_request_id="req-R2")
    # the per-stage signatures are still valid (genuine for R1)...
    assert s["signatures_valid"] is True
    # ...but the chain is NOT bound to this receipt — the splice is caught here
    assert s["request_id_bound"] is False


def test_full_verifier_rejects_request_id_mismatch():
    chain, keys, h0, h2 = _chain("req-R1")
    anchor = _Anchor(keys)
    ok, why = verify_stage_activation_chain(
        chain, prompt_hash_hex=h0, output_hash_hex=h2, anchor=anchor,
        expected_request_id="req-R2")
    assert not ok and "different inference" in why


def test_full_verifier_accepts_matching_request_id():
    chain, keys, h0, h2 = _chain("req-OK")
    anchor = _Anchor(keys)
    ok, why = verify_stage_activation_chain(
        chain, prompt_hash_hex=h0, output_hash_hex=h2, anchor=anchor,
        expected_request_id="req-OK")
    assert ok, why


def test_request_id_check_skipped_when_not_supplied():
    """Back-compat: omitting expected_request_id leaves the prior behavior."""
    chain, _, _, _ = _chain("whatever")
    s = summarize_stage_activation_chain(chain)
    assert "request_id_bound" not in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
