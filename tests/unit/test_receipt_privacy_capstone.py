"""Sprint 297 — §7 capstone: extend InferenceReceipt schema
with activation_noise_trace (sprint 295) + topology_assignment
(sprint 296), and extend verify_receipt_privacy_claim to
consume them.

This sprint locks down the verifiable receipt schema. Real
`RpcChainExecutor` wiring (so executor populates these fields
during a live inference) stays out of scope — the executor is
delicate and the schema must settle first.

Back-compat invariant: pre-sprint-297 receipts (without the
new fields) round-trip cleanly and verify identically. New
fields are Optional with default None; conditional signing-
payload encoding mirrors Phase 3.x.8's streamed_output pattern.

Verification posture (additive to sprints 292-294):
  - require_activation_dp=True  → must have a valid trace
                                  matching the tier
  - require_topology_rotation=True → topology must be
                                  distinct from supplied
                                  history within window
"""
from __future__ import annotations

import base64
import hashlib
from decimal import Decimal

import pytest

from prsm.compute.inference.activation_dp import (
    ActivationDPInjector,
    ActivationNoiseTrace,
    StageNoisePolicy,
)
from prsm.compute.inference.executor import (
    SOFTWARE_TEE_ATTESTATION_PREFIX,
)
from prsm.compute.inference.models import (
    ContentTier, InferenceReceipt,
)
from prsm.compute.inference.privacy_verification import (
    verify_receipt_privacy_claim,
)
from prsm.compute.inference.receipt import (
    sign_receipt, verify_receipt,
)
from prsm.compute.inference.topology_rotation import (
    TopologyAssignment, TopologyHistory,
    TopologyRotationPolicy,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ── Receipt schema extension: back-compat ───────────────


def test_pre_sprint297_receipt_still_round_trips():
    """A receipt without activation_noise_trace or
    topology_assignment fields parses cleanly + back-serializes
    without those keys present."""
    receipt = InferenceReceipt(
        job_id="j-1",
        request_id="r-1",
        model_id="mock-llama-3-8b",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01" * 64,
        output_hash=b"\x02" * 32,
        duration_seconds=0.1,
        cost_ftns=Decimal("0.01"),
    )
    d = receipt.to_dict()
    # Defaults: new fields absent or null
    assert (
        d.get("activation_noise_trace") is None
        or "activation_noise_trace" not in d
    )
    restored = InferenceReceipt.from_dict(d)
    assert restored.activation_noise_trace is None
    assert restored.topology_assignment is None


def test_pre_sprint297_signing_bytes_unchanged():
    """Critical back-compat: a receipt without the new
    fields produces byte-identical signing_payload to what
    pre-sprint-297 callers + verifiers would produce. This
    means existing on-chain signed receipts stay valid."""
    receipt = InferenceReceipt(
        job_id="j", request_id="r",
        model_id="m",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01" * 64,
        output_hash=b"\x02" * 32,
        duration_seconds=0.1,
        cost_ftns=Decimal("0.01"),
    )
    # Pre-sprint-297 expected canonical bytes (mirrors
    # the existing signing_payload format).
    expected = (
        "j\nr\nm\nA\nstandard\n8.0000000000\n"
        "software\n"
        + (b"\x01" * 64).hex() + "\n"
        + (b"\x02" * 32).hex() + "\n"
        + "0.100000\n0.01\n"
    ).encode("utf-8")
    assert receipt.signing_payload() == expected


# ── Receipt with both new fields ─────────────────────────


def _make_trace(
    stages: int = 4,
    tier: PrivacyLevel = PrivacyLevel.STANDARD,
) -> ActivationNoiseTrace:
    """Build a valid trace by actually running the injector
    — matches what RpcChainExecutor will produce."""
    import numpy as np
    policy = StageNoisePolicy.for_tier(
        tier, stage_count=stages,
    )
    injector = ActivationDPInjector(policy=policy)
    for i in range(stages):
        injector.inject_stage(
            np.ones(10, dtype=np.float64), stage_index=i,
        )
    return injector.trace()


def _make_topology(nodes_offset: int = 0) -> TopologyAssignment:
    """Build a valid 2x2 topology."""
    return TopologyAssignment(
        positions={
            (0, 0): f"node-{nodes_offset+0}",
            (0, 1): f"node-{nodes_offset+1}",
            (1, 0): f"node-{nodes_offset+2}",
            (1, 1): f"node-{nodes_offset+3}",
        },
        stage_count=2,
        slots_per_stage=2,
    )


def test_receipt_round_trip_with_new_fields():
    trace = _make_trace()
    topo = _make_topology()
    receipt = InferenceReceipt(
        job_id="j-2",
        request_id="r-2",
        model_id="m",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01" * 64,
        output_hash=b"\x02" * 32,
        duration_seconds=0.1,
        cost_ftns=Decimal("0.01"),
        activation_noise_trace=trace,
        topology_assignment=topo,
    )
    d = receipt.to_dict()
    assert "activation_noise_trace" in d
    assert "topology_assignment" in d
    restored = InferenceReceipt.from_dict(d)
    assert restored.activation_noise_trace == trace
    assert restored.topology_assignment == topo


def test_signing_payload_includes_new_fields_when_present():
    """When either new field is present, signing_payload
    appends a canonical hash — tampering the field flips
    bytes → signature fails."""
    trace = _make_trace()
    receipt_with = InferenceReceipt(
        job_id="j", request_id="r", model_id="m",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01" * 64,
        output_hash=b"\x02" * 32,
        duration_seconds=0.1,
        cost_ftns=Decimal("0.01"),
        activation_noise_trace=trace,
    )
    receipt_without = InferenceReceipt(
        job_id="j", request_id="r", model_id="m",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01" * 64,
        output_hash=b"\x02" * 32,
        duration_seconds=0.1,
        cost_ftns=Decimal("0.01"),
    )
    # Sprint-297 receipt with trace must NOT match the
    # bare receipt's bytes
    assert (
        receipt_with.signing_payload()
        != receipt_without.signing_payload()
    )


def test_signed_receipt_with_new_fields_verifies():
    """Full end-to-end: sign a receipt with both new fields,
    verify it cryptographically."""
    identity = generate_node_identity("capstone-test")
    trace = _make_trace()
    topo = _make_topology()
    receipt = InferenceReceipt(
        job_id="j", request_id="r", model_id="m",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01" * 64,
        output_hash=b"\x02" * 32,
        duration_seconds=0.1,
        cost_ftns=Decimal("0.01"),
        activation_noise_trace=trace,
        topology_assignment=topo,
    )
    signed = sign_receipt(receipt, identity)
    assert verify_receipt(signed, identity=identity) is True


def test_tampered_trace_breaks_signature():
    """Forging the activation_noise_trace must invalidate
    the signature — defense against operator claiming better
    privacy posture than actually applied."""
    import dataclasses as _dc
    identity = generate_node_identity("tamper-test")
    real_trace = _make_trace(tier=PrivacyLevel.STANDARD)
    receipt = InferenceReceipt(
        job_id="j", request_id="r", model_id="m",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01" * 64,
        output_hash=b"\x02" * 32,
        duration_seconds=0.1,
        cost_ftns=Decimal("0.01"),
        activation_noise_trace=real_trace,
    )
    signed = sign_receipt(receipt, identity)
    # Forge a trace claiming MAXIMUM privacy (tighter ε)
    forged_trace = _make_trace(tier=PrivacyLevel.MAXIMUM)
    tampered = _dc.replace(
        signed, activation_noise_trace=forged_trace,
    )
    assert verify_receipt(tampered, identity=identity) is False


def test_tampered_topology_breaks_signature():
    import dataclasses as _dc
    identity = generate_node_identity("tamper-test-2")
    real_topo = _make_topology(nodes_offset=0)
    receipt = InferenceReceipt(
        job_id="j", request_id="r", model_id="m",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01" * 64,
        output_hash=b"\x02" * 32,
        duration_seconds=0.1,
        cost_ftns=Decimal("0.01"),
        topology_assignment=real_topo,
    )
    signed = sign_receipt(receipt, identity)
    forged_topo = _make_topology(nodes_offset=100)
    tampered = _dc.replace(
        signed, topology_assignment=forged_topo,
    )
    assert verify_receipt(tampered, identity=identity) is False


# ── verify_receipt_privacy_claim integration ────────────


def _build_signed_receipt(
    *,
    trace=None,
    topo=None,
    tier=PrivacyLevel.STANDARD,
):
    identity = generate_node_identity("verify-test")
    receipt = InferenceReceipt(
        job_id="j", request_id="r", model_id="m",
        content_tier=ContentTier.A,
        privacy_tier=tier,
        epsilon_spent=PrivacyLevel.config_for_level(
            tier,
        ).epsilon if tier != PrivacyLevel.NONE else 0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=(
            SOFTWARE_TEE_ATTESTATION_PREFIX
            + hashlib.sha384(b"sw-tee:test").digest()
        ),
        output_hash=hashlib.sha256(b"out").digest(),
        duration_seconds=0.1,
        cost_ftns=Decimal("0.01"),
        activation_noise_trace=trace,
        topology_assignment=topo,
    )
    return sign_receipt(receipt, identity), identity


def test_verify_with_trace_present_default_permissive():
    """Default posture: trace presence flips an optional
    diagnostic field; ok stays True."""
    trace = _make_trace()
    signed, identity = _build_signed_receipt(trace=trace)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
    )
    assert result.ok is True
    assert result.activation_noise_trace_valid is True


def test_verify_with_present_invalid_trace_is_not_ok():
    """sp1001 — a PRESENT but detectably-INVALID activation_noise_trace must make
    ok=False on EVERY path, including the default (require_activation_dp_trace=
    False) used by /compute/receipt/verify, the MCP tool, and the SDK.

    This test previously asserted ok=True ("default permissive") — pinning the §7
    privacy-OVER-CLAIM bug: a receipt whose own signed trace contradicted its
    privacy claim still verified as VALID. An HONEST trace always validates, so
    honest receipts are unaffected; only a detectably-broken privacy proof is now
    rejected. The signature itself is still valid — the CLAIM is what fails."""
    # Trace claiming STANDARD but with an internally inconsistent sum
    # (per-stage sums to 2, not the claimed total 8) → detectably invalid.
    bad_trace = ActivationNoiseTrace(
        per_stage_epsilon=[1.0, 1.0],  # sum=2 != claimed total 8
        total_epsilon_spent=8.0,
        clip_norm=1.0,
        stage_count=2,
        tier="standard",
    )
    signed, identity = _build_signed_receipt(trace=bad_trace)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
    )
    assert result.signature_valid is True   # signature is fine
    assert result.activation_noise_trace_valid is False
    assert result.ok is False               # sp1001: invalid proof must NOT be ok


def test_overclaim_tier_epsilon_desync_with_present_trace_rejected():
    """sp1001 (over-claim bug #1) — a receipt claiming MAXIMUM tier but charging
    STANDARD ε (8.0), carrying an honest MAXIMUM trace (total 1.0), must verify
    ok=False on the default path: the charged ε (8.0) disagrees with the trace
    total (1.0). Pre-fix this returned ok=True (tier↔ε desync was non-fatal)."""
    import dataclasses as _dc
    max_trace = _make_trace(tier=PrivacyLevel.MAXIMUM)  # honest MAXIMUM → total 1.0
    signed, identity = _build_signed_receipt(
        trace=max_trace, tier=PrivacyLevel.MAXIMUM,
    )
    # The honest builder set epsilon_spent to the MAXIMUM ceiling (1.0); forge the
    # over-claim by charging the STANDARD ε (8.0) instead, then re-sign so the
    # signature is valid (operator self-signing a contradictory pair).
    forged = sign_receipt(_dc.replace(signed, epsilon_spent=8.0), identity)
    result = verify_receipt_privacy_claim(forged, identity=identity)
    assert result.signature_valid is True
    assert result.ok is False


def test_overclaim_zero_noise_trace_on_private_tier_rejected():
    """sp1001 (over-claim bug #2) — a receipt charging full STANDARD ε (8.0) with a
    signed activation_noise_trace recording ZERO noise must verify ok=False even
    under the strict flags. Pre-fix this returned ok=True (the trace validator
    allowed arbitrary under-spend down to 0.0, and dp_noise_applied was derived
    from epsilon_spent>0, not the trace)."""
    zero_trace = ActivationNoiseTrace(
        per_stage_epsilon=[0.0, 0.0],   # NO noise injected
        total_epsilon_spent=0.0,
        clip_norm=1.0,
        stage_count=2,
        tier="standard",
    )
    signed, identity = _build_signed_receipt(trace=zero_trace)  # tier STANDARD, ε=8.0
    # Default path:
    result = verify_receipt_privacy_claim(signed, identity=identity)
    assert result.ok is False
    # And under the strictest flags:
    strict = verify_receipt_privacy_claim(
        signed, identity=identity,
        require_dp_noise=True, require_activation_dp_trace=True,
    )
    assert strict.ok is False


def test_honest_receipt_with_matching_trace_still_ok():
    """sp1001 regression — an HONEST receipt (tier ε == trace total, full noise on
    every stage) must still verify ok=True. The over-claim gates only reject
    detectably-inconsistent proofs, never honest ones."""
    trace = _make_trace(tier=PrivacyLevel.STANDARD)  # total == STANDARD ε (8.0)
    signed, identity = _build_signed_receipt(trace=trace)  # epsilon_spent=8.0
    result = verify_receipt_privacy_claim(signed, identity=identity)
    assert result.ok is True
    assert result.activation_noise_trace_valid is True


def test_verify_require_activation_dp_passes_with_valid_trace():
    trace = _make_trace()
    signed, identity = _build_signed_receipt(trace=trace)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
        require_activation_dp_trace=True,
    )
    assert result.ok is True
    assert result.activation_noise_trace_valid is True


def test_verify_require_activation_dp_fails_when_missing():
    signed, identity = _build_signed_receipt(trace=None)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
        require_activation_dp_trace=True,
    )
    assert result.ok is False
    assert result.activation_noise_trace_valid is False
    assert any(
        "activation" in r.lower() and "missing" in r.lower()
        for r in result.reasons
    )


def test_verify_require_activation_dp_fails_with_invalid_trace():
    bad_trace = ActivationNoiseTrace(
        per_stage_epsilon=[1.0, 1.0],
        total_epsilon_spent=8.0,  # lies — sum is 2
        clip_norm=1.0,
        stage_count=2,
        tier="standard",
    )
    signed, identity = _build_signed_receipt(trace=bad_trace)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
        require_activation_dp_trace=True,
    )
    assert result.ok is False
    assert result.activation_noise_trace_valid is False


def test_verify_trace_tier_mismatch_with_receipt_caught():
    """Trace claims MAXIMUM tier (ε=1) but receipt's
    privacy_tier is STANDARD — surface the mismatch."""
    forged_trace = _make_trace(tier=PrivacyLevel.MAXIMUM)
    signed, identity = _build_signed_receipt(
        trace=forged_trace, tier=PrivacyLevel.STANDARD,
    )
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
        require_activation_dp_trace=True,
    )
    assert result.ok is False
    assert result.activation_noise_trace_valid is False


# ── Topology integration ─────────────────────────────────


def test_verify_with_topology_present_default():
    topo = _make_topology()
    signed, identity = _build_signed_receipt(topo=topo)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
    )
    # Without history supplied: distinctness check is N/A
    assert result.ok is True
    # Structural check still runs
    assert result.topology_structurally_valid is True


def test_verify_topology_rotation_distinct_from_history():
    topo = _make_topology()
    signed, identity = _build_signed_receipt(topo=topo)
    # Empty history → topology is automatically distinct
    history = TopologyHistory(max_entries=5)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
        topology_history=history,
        require_topology_rotation=True,
    )
    assert result.ok is True
    assert result.topology_distinct_from_history is True


def test_verify_topology_repeats_history_caught():
    topo = _make_topology()
    signed, identity = _build_signed_receipt(topo=topo)
    history = TopologyHistory(max_entries=5)
    history.record(topo)  # Same topology already in history
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
        topology_history=history,
        require_topology_rotation=True,
    )
    assert result.ok is False
    assert result.topology_distinct_from_history is False
    assert any(
        "topology" in r.lower()
        and ("repeat" in r.lower() or "history" in r.lower())
        for r in result.reasons
    )


def test_verify_topology_with_duplicate_node_caught():
    """Structurally invalid topology (same node in two
    positions) caught regardless of history."""
    bad_topo = TopologyAssignment(
        positions={
            (0, 0): "node-a",
            (0, 1): "node-a",  # dup
            (1, 0): "node-b",
            (1, 1): "node-c",
        },
        stage_count=2,
        slots_per_stage=2,
    )
    signed, identity = _build_signed_receipt(topo=bad_topo)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
    )
    assert result.topology_structurally_valid is False


def test_verify_require_topology_rotation_fails_when_missing():
    signed, identity = _build_signed_receipt(topo=None)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
        topology_history=TopologyHistory(max_entries=5),
        require_topology_rotation=True,
    )
    assert result.ok is False
    assert any(
        "topology" in r.lower() and "missing" in r.lower()
        for r in result.reasons
    )


# ── End-to-end fully-loaded receipt ──────────────────────


def test_end_to_end_fully_loaded_receipt():
    """The capstone integration test: build a receipt with
    every §7 field populated, sign, verify cryptographically,
    verify privacy claims with all require_* flags on,
    confirm all integrity fields hold."""
    trace = _make_trace(stages=4, tier=PrivacyLevel.STANDARD)
    topo = _make_topology()
    signed, identity = _build_signed_receipt(
        trace=trace, topo=topo,
        tier=PrivacyLevel.STANDARD,
    )
    history = TopologyHistory(max_entries=5)
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
        require_dp_noise=True,
        require_activation_dp_trace=True,
        require_topology_rotation=True,
        topology_history=history,
        # Software fallback attestation — leave hardware
        # check off (default) since we're proving the rest
        # of the receipt schema, not the attestation backend
    )
    assert result.signature_valid is True
    assert result.dp_noise_applied is True
    assert result.activation_noise_trace_valid is True
    assert result.topology_structurally_valid is True
    assert result.topology_distinct_from_history is True
    assert result.ok is True


def test_to_dict_includes_new_fields():
    """PrivacyVerification.to_dict round-trips the new
    sprint-297 booleans for MCP-side rendering."""
    trace = _make_trace()
    topo = _make_topology()
    signed, identity = _build_signed_receipt(
        trace=trace, topo=topo,
    )
    result = verify_receipt_privacy_claim(
        signed, identity=identity,
    )
    d = result.to_dict()
    assert "activation_noise_trace_valid" in d
    assert "topology_structurally_valid" in d
    assert "topology_distinct_from_history" in d
