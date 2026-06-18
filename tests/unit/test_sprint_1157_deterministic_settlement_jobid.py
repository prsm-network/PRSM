"""Sprint 1157 — deterministic settlement job_id (on-chain per-stage payee, brick 2.5).

Brick 2 (worker stage settlement-signature MECHANISM, sp1156) fail-closed on a
REAL gap: the worker signs its per-stage settlement leaf at STAGE-execution time
knowing only ``request.request_id``, but the receipt ``job_id`` was minted
POST-HOC + RANDOM (``f"{prefix}-{uuid4().hex[:12]}"``) at receipt-build time, so
the worker could not know it. A leaf the worker signed over ``request_id`` would
be rebuilt by the splitter over ``receipt.job_id`` (a DIFFERENT value) → the leaf
fails an on-chain INVALID_SIGNATURE challenge and the honest worker is wrongly
slashed.

THIS brick closes that gap: ``settlement_job_id(request_id, *, streamed)`` is a
PURE, deterministic, prefix-preserving, worker-derivable function of
``request_id``; the receipt builder now mints its ``job_id`` through it instead
of a random uuid4. The worker (next brick) imports the SAME helper, so worker +
receipt agree on the identical challenge-defensible leaf.

These tests pin: (a) determinism, (b) uniqueness == request_id uniqueness +
streamed/unary distinct prefix, (c) the existing ``parallax-job-`` /
``parallax-stream-job-`` prefix invariant, (d) worker-derivability (derive the
id from ONLY request_id + streamed and match the receipt), (e) the single-payee
settlement leaf stays challenge-defensible over the deterministic job_id, and
(f) empty/None request_id is rejected.
"""
from __future__ import annotations

import hashlib
import types
from decimal import Decimal

from prsm.compute.shard_receipt import (
    build_receipt_signing_payload,
    settlement_job_id,
)


def _ident(name="settler-1157"):
    from prsm.node.identity import generate_node_identity
    return generate_node_identity(display_name=name)


def _request(request_id="infer-deadbeef0001", prompt="hello", model_id="gpt2"):
    from prsm.compute.inference.models import InferenceRequest
    return InferenceRequest(
        prompt=prompt,
        model_id=model_id,
        budget_ftns=Decimal("10"),
        request_id=request_id,
    )


def _outcome(output="the output"):
    from prsm.compute.inference.parallax_executor import ChainExecutionResult
    from prsm.compute.tee.models import TEEType
    return ChainExecutionResult(
        output=output,
        duration_seconds=0.05,
        tee_attestation=b"\x01" * 32,
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )


def _build_receipt(identity, request, *, streamed=False, output="the output"):
    """Drive the REAL receipt-build path (_build_signed_receipt) without spinning
    up the full scheduler/trust-stack. The method only reads ``self._identity``,
    so a shim namespace carrying it is sufficient + faithful."""
    from prsm.compute.inference.parallax_executor import ParallaxScheduledExecutor
    shim = types.SimpleNamespace(_identity=identity)
    return ParallaxScheduledExecutor._build_signed_receipt(
        shim,
        request=request,
        cost=Decimal("1.0"),
        outcome=_outcome(output),
        streamed=streamed,
    )


# ── (a) DETERMINISM ────────────────────────────────────────────────────────

def test_helper_is_deterministic():
    rid = "infer-aaaa11112222"
    assert settlement_job_id(rid, streamed=False) == settlement_job_id(
        rid, streamed=False
    )
    assert settlement_job_id(rid, streamed=True) == settlement_job_id(
        rid, streamed=True
    )


def test_receipt_job_id_is_deterministic_not_random():
    ident = _ident()
    req = _request("infer-fixed-rid-001")
    r1 = _build_receipt(ident, req)
    r2 = _build_receipt(ident, req)
    # Pre-1157 this was uuid4 → r1 != r2. Now both derive from request_id.
    assert r1.job_id == r2.job_id
    assert r1.job_id == settlement_job_id(req.request_id, streamed=False)


# ── (b) UNIQUENESS ─────────────────────────────────────────────────────────

def test_distinct_request_ids_yield_distinct_job_ids():
    a = settlement_job_id("infer-rid-A", streamed=False)
    b = settlement_job_id("infer-rid-B", streamed=False)
    assert a != b


def test_streamed_vs_unary_distinct_prefix_same_request_id():
    rid = "infer-shared-rid"
    unary = settlement_job_id(rid, streamed=False)
    streamed = settlement_job_id(rid, streamed=True)
    assert unary != streamed
    assert unary.startswith("parallax-job-")
    assert streamed.startswith("parallax-stream-job-")


# ── (c) PREFIX (existing invariant) ─────────────────────────────────────────

def test_unary_prefix_preserved():
    jid = settlement_job_id("infer-x", streamed=False)
    assert jid.startswith("parallax-job-")
    assert not jid.startswith("parallax-stream-job-")


def test_streamed_prefix_preserved():
    jid = settlement_job_id("infer-x", streamed=True)
    assert jid.startswith("parallax-stream-job-")


def test_receipt_keeps_startswith_invariant():
    ident = _ident()
    unary = _build_receipt(ident, _request("infer-u-1"), streamed=False)
    streamed = _build_receipt(ident, _request("infer-s-1"), streamed=True)
    assert unary.job_id.startswith("parallax-job-")
    assert not unary.job_id.startswith("parallax-stream-job-")
    assert streamed.job_id.startswith("parallax-stream-job-")


# ── (d) WORKER-DERIVABLE ─────────────────────────────────────────────────────

def test_worker_derives_same_job_id_from_request_id_only():
    """The worker/orchestrator has ONLY request_id + the streamed flag at
    stage-execution time. It must reproduce the receipt's job_id exactly."""
    ident = _ident()
    for streamed in (False, True):
        req = _request("infer-worker-derive-77")
        receipt = _build_receipt(ident, req, streamed=streamed)
        # Independently derive from what the worker knows — request_id + streamed.
        derived = settlement_job_id(req.request_id, streamed=streamed)
        assert derived == receipt.job_id


# ── (e) SINGLE-PAYEE STILL CHALLENGE-DEFENSIBLE ─────────────────────────────

def test_single_payee_leaf_defensible_over_deterministic_job_id():
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )
    from prsm.settlement.merkle import batched_receipt_to_leaf
    from prsm.node.identity import verify_signature

    ident = _ident()
    req = _request("infer-defensible-001")
    receipt = _build_receipt(ident, req, output="settle me")

    # The settler (== the receipt's settler) re-signs the shard payload.
    executed_at = 1_700_000_000
    shard_index = 0
    br = inference_receipt_to_batched_receipt(
        receipt=receipt,
        identity=ident,
        requester_address="0x" + "11" * 20,
        provider_address="0x" + "22" * 20,
        value_ftns=10 ** 18,
        local_escrow_id=req.request_id,
        executed_at_unix=executed_at,
        shard_index=shard_index,
    )

    # The leaf's signing-message hash binds the DETERMINISTIC job_id.
    leaf = batched_receipt_to_leaf(br)
    expected = build_receipt_signing_payload(
        job_id=receipt.job_id,                 # deterministic, == derivable id
        shard_index=shard_index,
        output_hash=receipt.output_hash.hex(),
        executed_at_unix=executed_at,
    )
    assert leaf.signing_message_hash == expected
    # And the job_id bound is exactly what the worker would derive.
    assert receipt.job_id == settlement_job_id(req.request_id, streamed=False)

    # The settler's signature over that payload verifies → on-chain
    # INVALID_SIGNATURE challenge would FAIL to slash (leaf is defensible).
    assert verify_signature(
        ident.public_key_b64, expected, br.receipt.signature
    )


# ── (f) reject empty/None request_id ────────────────────────────────────────

def test_empty_request_id_rejected():
    import pytest
    with pytest.raises(ValueError):
        settlement_job_id("", streamed=False)


def test_none_request_id_rejected():
    import pytest
    with pytest.raises(ValueError):
        settlement_job_id(None, streamed=False)  # type: ignore[arg-type]


def test_non_str_request_id_rejected():
    import pytest
    with pytest.raises(ValueError):
        settlement_job_id(12345, streamed=True)  # type: ignore[arg-type]
