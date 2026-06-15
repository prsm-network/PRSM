"""Sprint 1107 — Domain-03 F1/F2 brick 1: the signed per-stage activation-chain
receptacle in the production InferenceReceipt.

This brick adds the StageActivationChain primitive and binds it into the receipt's
signing payload (conditional-append, byte-equivalent when absent — the sprint-297/1099
pattern). Brick 2 has workers produce the per-stage proofs, brick 3 threads them, brick 4
verifies signatures + derives topology from the signed stages.
"""
from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.stage_activation_proof import (
    StageActivationChain,
    StageActivationProof,
    activation_hash,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType


def _proof(idx, node, hin, hout, sig="sig"):
    return StageActivationProof(
        stage_index=idx, stage_node_id=node,
        input_activation_hash=hin, output_activation_hash=hout,
        stage_signature_b64=sig)


def _chain():
    h0 = activation_hash(b"prompt")
    h1 = activation_hash(b"mid")
    h2 = activation_hash(b"out")
    return StageActivationChain(
        request_id="req-1",
        proofs=[_proof(0, "a" * 32, h0, h1), _proof(1, "b" * 32, h1, h2)],
    ), h0, h2


def _receipt(**over):
    base = dict(
        job_id="job-1", request_id="req-1", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att", output_hash=b"\xab" * 32,
        duration_seconds=1.0, cost_ftns=Decimal("0.12"), settler_node_id="s" * 32,
    )
    base.update(over)
    return InferenceReceipt(**base)


# ── the primitive ────────────────────────────────────────────────────────────────

def test_proof_signing_bytes_bind_stage_request_and_hashes():
    p = _proof(0, "node-x", "hin", "hout")
    b = p.signing_bytes("req-9")
    # binds request, stage, node, and both activation hashes
    for token in (b"req-9", b"node-x", b"hin", b"hout", b"prsm-stage-activation-proof-v1"):
        assert token in b
    # a different request id must change the signed bytes (no cross-request replay)
    assert p.signing_bytes("req-9") != p.signing_bytes("req-8")


def test_chain_continuity_holds_for_a_well_formed_chain():
    chain, h0, h2 = _chain()
    ok, why = chain.verify_continuity(prompt_hash_hex=h0, output_hash_hex=h2)
    assert ok, why


def test_chain_continuity_rejects_a_spliced_intermediate():
    h0 = activation_hash(b"prompt")
    h1 = activation_hash(b"mid")
    h2 = activation_hash(b"out")
    forged = activation_hash(b"attacker-substituted")
    # stage 1 claims a DIFFERENT input than stage 0 produced
    chain = StageActivationChain(
        request_id="r", proofs=[_proof(0, "a", h0, h1), _proof(1, "b", forged, h2)])
    ok, why = chain.verify_continuity(prompt_hash_hex=h0, output_hash_hex=h2)
    assert not ok and "between stage" in why


def test_chain_continuity_rejects_prompt_mismatch():
    chain, h0, h2 = _chain()
    ok, why = chain.verify_continuity(prompt_hash_hex="not-the-prompt", output_hash_hex=h2)
    assert not ok and "prompt entry" in why


def test_topology_derived_from_signed_stages():
    chain, _, _ = _chain()
    assert chain.topology_node_ids() == ["a" * 32, "b" * 32]


def test_chain_to_from_dict_round_trips():
    chain, _, _ = _chain()
    rebuilt = StageActivationChain.from_dict(chain.to_dict())
    assert rebuilt.stable_hash() == chain.stable_hash()


# ── binding into the receipt ───────────────────────────────────────────────────────

def test_chain_changes_receipt_signing_payload():
    chain, _, _ = _chain()
    with_chain = _receipt(stage_activation_chain=chain)
    without = _receipt()
    assert with_chain.signing_payload() != without.signing_payload()


def test_chain_absent_is_byte_identical_to_pre_1107():
    payload = _receipt().signing_payload()
    assert b"stage_activation_chain" not in payload


def test_tampering_a_stage_hash_flips_receipt_payload():
    chain, h0, h2 = _chain()
    r1 = _receipt(stage_activation_chain=chain)
    tampered = StageActivationChain(
        request_id="req-1",
        proofs=[_proof(0, "a" * 32, h0, "TAMPERED"), chain.proofs[1]])
    r2 = _receipt(stage_activation_chain=tampered)
    assert r1.signing_payload() != r2.signing_payload()


def test_receipt_chain_round_trips_through_dict():
    chain, _, _ = _chain()
    r = _receipt(stage_activation_chain=chain)
    rebuilt = InferenceReceipt.from_dict(r.to_dict())
    assert rebuilt.stage_activation_chain is not None
    assert rebuilt.signing_payload() == r.signing_payload()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
