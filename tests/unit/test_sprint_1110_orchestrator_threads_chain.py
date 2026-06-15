"""Sprint 1110 — Domain-03 F1/F2 brick 4: the orchestrator threads the per-stage signed
proofs into the production InferenceReceipt.

- _build_stage_activation_chain assembles a StageActivationChain from the per-stage
  outcomes, but ONLY when every stage supplied a complete signed proof (all-or-nothing).
- ParallaxScheduledExecutor._build_signed_receipt carries the chain into the signed
  receipt, and the result verifies (the chain is bound into signing_payload).
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from prsm.compute.chain_rpc.client import StageOutcome, _build_stage_activation_chain
from prsm.compute.inference.models import InferenceRequest
from prsm.compute.inference.parallax_executor import (
    ChainExecutionResult,
    ParallaxScheduledExecutor,
)
from prsm.compute.inference.receipt import verify_receipt
from prsm.compute.inference.stage_activation_proof import activation_hash
from prsm.compute.tee.models import TEEType
from prsm.node.identity import generate_node_identity


def _outcome(idx, node, hin, hout, sig="sig"):
    return StageOutcome(
        stage_index=idx, stage_node_id=node, duration_seconds=0.1,
        tee_attestation=b"a", tee_type=TEEType.SOFTWARE, epsilon_spent=0.0,
        stage_input_activation_hash=hin, stage_output_activation_hash=hout,
        stage_activation_proof_b64=sig)


def test_chain_assembled_when_all_stages_have_proofs():
    h0, h1, h2 = activation_hash(b"a"), activation_hash(b"b"), activation_hash(b"c")
    outcomes = [_outcome(0, "n0", h0, h1), _outcome(1, "n1", h1, h2)]
    chain = _build_stage_activation_chain("req-1", outcomes)
    assert chain is not None
    assert chain.request_id == "req-1"
    assert chain.topology_node_ids() == ["n0", "n1"]


def test_chain_omitted_when_any_stage_lacks_a_proof():
    """All-or-nothing: a missing proof anywhere → no chain (absent must mean 'not
    proven', never 'silently dropped')."""
    h0, h1 = activation_hash(b"a"), activation_hash(b"b")
    outcomes = [
        _outcome(0, "n0", h0, h1),
        StageOutcome(1, "n1", 0.1, b"a", TEEType.SOFTWARE, 0.0),  # no proof
    ]
    assert _build_stage_activation_chain("req-1", outcomes) is None


def test_build_signed_receipt_carries_and_signs_the_chain():
    h0, h1, h2 = activation_hash(b"a"), activation_hash(b"b"), activation_hash(b"c")
    chain = _build_stage_activation_chain(
        "req-9", [_outcome(0, "n0", h0, h1), _outcome(1, "n1", h1, h2)])

    identity = generate_node_identity("settler")
    executor = ParallaxScheduledExecutor.__new__(ParallaxScheduledExecutor)
    executor._identity = identity

    request = InferenceRequest(prompt="hi", model_id="gpt2", budget_ftns=Decimal("1"),
                               request_id="req-9")
    outcome = ChainExecutionResult(
        output=" out", duration_seconds=0.2, tee_attestation=b"att",
        tee_type=TEEType.SOFTWARE, epsilon_spent=0.0, stage_activation_chain=chain)
    receipt = executor._build_signed_receipt(
        request=request, cost=Decimal("0.1"), outcome=outcome)

    assert receipt.stage_activation_chain is not None
    assert receipt.stage_activation_chain.request_id == "req-9"
    # the chain is bound into the signed payload → the receipt verifies as-is
    assert verify_receipt(receipt, identity=identity)


def test_single_node_path_without_chain_still_signs():
    """A ChainExecutionResult with no chain (single-node / streaming) yields a receipt
    that verifies and carries no chain — byte-compatible with pre-1110."""
    identity = generate_node_identity("settler")
    executor = ParallaxScheduledExecutor.__new__(ParallaxScheduledExecutor)
    executor._identity = identity
    request = InferenceRequest(prompt="hi", model_id="gpt2", budget_ftns=Decimal("1"))
    outcome = ChainExecutionResult(
        output=" out", duration_seconds=0.2, tee_attestation=b"att",
        tee_type=TEEType.SOFTWARE, epsilon_spent=0.0)
    receipt = executor._build_signed_receipt(
        request=request, cost=Decimal("0.1"), outcome=outcome)
    assert receipt.stage_activation_chain is None
    assert verify_receipt(receipt, identity=identity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
