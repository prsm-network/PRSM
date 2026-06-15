"""Sprint 1116 — challenge/dispute mechanism, brick 1: the off-chain challenge verifier.

Composes the settler-signature primitive + the per-stage activation-chain verifier
(sp1107-1113) into an actionable fraud verdict for a §7 InferenceReceipt: is there
provable fraud, on what grounds, and is it on-chain-actionable (maps to a
BatchSettlementRegistry ReasonCode)?
"""
from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.receipt import sign_receipt
from prsm.compute.inference.stage_activation_proof import (
    StageActivationChain,
    StageActivationProof,
    activation_hash,
)
from prsm.compute.inference.topology_rotation import TopologyAssignment
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity
from prsm.settlement.challenge_verifier import (
    ChallengeReason,
    verify_inference_receipt_for_challenge,
)


def _signed_proof(identity, req, idx, hin, hout):
    base = StageActivationProof(idx, identity.node_id, hin, hout)
    return StageActivationProof(idx, identity.node_id, hin, hout,
                                stage_signature_b64=identity.sign(base.signing_bytes(req)))


def _receipt(settler, *, chain=None, topo=None, request_id="req-1"):
    r = InferenceReceipt(
        job_id="job-1", request_id=request_id, model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE, tee_attestation=b"att",
        output_hash=b"\xab" * 32, duration_seconds=1.0, cost_ftns=Decimal("0.1"),
        settler_node_id="", stage_activation_chain=chain, topology_assignment=topo)
    return sign_receipt(r, settler)


def _two_stage(req="req-1"):
    a, b = generate_node_identity("a"), generate_node_identity("b")
    h0, h1, h2 = activation_hash(b"x"), activation_hash(b"y"), activation_hash(b"z")
    chain = StageActivationChain(req, [_signed_proof(a, req, 0, h0, h1),
                                       _signed_proof(b, req, 1, h1, h2)])
    keys = {a.node_id: a.public_key_b64, b.node_id: b.public_key_b64}
    return chain, keys, a, b


# ── settler signature ground ─────────────────────────────────────────────────────

def test_clean_receipt_no_chain_is_ok():
    settler = generate_node_identity("settler")
    r = _receipt(settler)
    report = verify_inference_receipt_for_challenge(
        r, settler_public_key_b64=settler.public_key_b64)
    assert report.receipt_ok is True
    assert report.proven_findings() == []


def test_wrong_settler_key_proves_invalid_signature():
    settler = generate_node_identity("settler")
    r = _receipt(settler)
    other = generate_node_identity("imposter")
    report = verify_inference_receipt_for_challenge(
        r, settler_public_key_b64=other.public_key_b64)
    assert report.receipt_ok is False
    proven = report.proven_findings()
    assert any(f.reason == ChallengeReason.INVALID_SETTLER_SIGNATURE for f in proven)
    # this ground is on-chain-actionable (maps to ReasonCode.INVALID_SIGNATURE == 1)
    actionable = report.on_chain_actionable()
    assert any(f.on_chain_reason == 1 for f in actionable)


def test_tampered_receipt_proves_invalid_signature():
    settler = generate_node_identity("settler")
    r = _receipt(settler)
    tampered = dataclasses.replace(r, output_hash=b"\x00" * 32)  # breaks the signature
    report = verify_inference_receipt_for_challenge(
        tampered, settler_public_key_b64=settler.public_key_b64)
    assert report.receipt_ok is False


# ── activation-chain grounds ──────────────────────────────────────────────────────

def test_valid_chain_with_keys_is_ok():
    settler = generate_node_identity("settler")
    chain, keys, a, b = _two_stage()
    topo = TopologyAssignment(
        positions={(0, 0): a.node_id, (1, 0): b.node_id}, stage_count=2, slots_per_stage=1)
    r = _receipt(settler, chain=chain, topo=topo)
    report = verify_inference_receipt_for_challenge(
        r, settler_public_key_b64=settler.public_key_b64, stage_public_keys=keys)
    assert report.receipt_ok is True, report.to_dict()


def test_forged_stage_signature_proves_chain_invalid():
    settler = generate_node_identity("settler")
    chain, keys, a, b = _two_stage()
    r = _receipt(settler, chain=chain)
    # supply the WRONG key for stage a → its signature won't verify
    bogus = generate_node_identity("z")
    bad_keys = dict(keys)
    bad_keys[a.node_id] = bogus.public_key_b64
    report = verify_inference_receipt_for_challenge(
        r, settler_public_key_b64=settler.public_key_b64, stage_public_keys=bad_keys)
    assert report.receipt_ok is False
    assert any(f.reason == ChallengeReason.INVALID_ACTIVATION_CHAIN
               for f in report.proven_findings())


def test_topology_lie_proven():
    settler = generate_node_identity("settler")
    chain, keys, a, b = _two_stage()
    lying = TopologyAssignment(
        positions={(0, 0): a.node_id, (1, 0): "imposter"}, stage_count=2, slots_per_stage=1)
    r = _receipt(settler, chain=chain, topo=lying)
    report = verify_inference_receipt_for_challenge(
        r, settler_public_key_b64=settler.public_key_b64, stage_public_keys=keys)
    assert report.receipt_ok is False
    assert any(f.reason == ChallengeReason.ACTIVATION_TOPOLOGY_MISMATCH
               for f in report.proven_findings())


def test_spliced_chain_from_other_request_proven():
    settler = generate_node_identity("settler")
    # chain signed for req-OTHER, but the receipt's request_id is req-1
    chain, keys, a, b = _two_stage(req="req-OTHER")
    r = _receipt(settler, chain=chain, request_id="req-1")
    report = verify_inference_receipt_for_challenge(
        r, settler_public_key_b64=settler.public_key_b64, stage_public_keys=keys)
    assert report.receipt_ok is False
    assert any(f.reason == ChallengeReason.ACTIVATION_CHAIN_WRONG_REQUEST
               for f in report.proven_findings())


def test_chain_without_keys_notes_unverified_signatures():
    settler = generate_node_identity("settler")
    chain, keys, a, b = _two_stage()
    r = _receipt(settler, chain=chain)
    report = verify_inference_receipt_for_challenge(
        r, settler_public_key_b64=settler.public_key_b64)  # no stage keys
    # internal links + request_id still verify → receipt_ok (no PROVEN fraud)
    assert report.receipt_ok is True
    assert any("not supplied" in n for n in report.notes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
