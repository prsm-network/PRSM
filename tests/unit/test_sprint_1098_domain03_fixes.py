"""Sprint 1098 — folds the clean Domain-03 cross-cutting review findings:
- F3 (HIGH): the `local` executor advertised tier-sgx on a SOFTWARE runtime → confidential
  tiers served in software silently. Now it advertises tier-none; the gate refuses them.
- F5 (MEDIUM): /compute/receipt/verify dropped streamed_output + partial_completion on
  reconstruction (both fold into signing_payload) → honest streamed/partial receipts
  falsely failed verification. Now restored.

(F1 CRITICAL + F2 HIGH — the production receipt not carrying the activation hash chain /
topology not signature-bound — are an architectural change tracked as a dedicated sprint,
not folded here.)
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.tee.models import PrivacyLevel, TEEType


def _receipt(**over):
    base = dict(
        job_id="job-1", request_id="req-1", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE,
        tee_attestation=b"PRSM-MS-ATT-V1:{}", output_hash=b"\xab" * 32,
        duration_seconds=1.0, cost_ftns=Decimal("0.12"), settler_node_id="a" * 32,
    )
    base.update(over)
    return InferenceReceipt(**base)


# ── F5: streamed_output / partial_completion are signed → must round-trip ─────────

def test_streamed_output_changes_signing_payload():
    """The flag IS in the signed bytes, so dropping it on reconstruction changes the
    payload → signature would fail. (Proves the bug the handler had.)"""
    streamed = _receipt(streamed_output=True)
    dropped = _receipt(streamed_output=False)   # what the buggy handler reconstructed
    assert streamed.signing_payload() != dropped.signing_payload()


def test_streamed_receipt_round_trips_when_field_restored():
    """to_dict → from_dict (which restores streamed_output, as the fixed handler now does)
    reproduces the EXACT signed bytes → an honest streamed receipt verifies."""
    streamed = _receipt(streamed_output=True)
    rebuilt = InferenceReceipt.from_dict(streamed.to_dict())
    assert rebuilt.streamed_output is True
    assert rebuilt.signing_payload() == streamed.signing_payload()


def test_partial_completion_round_trips_when_restored():
    from prsm.compute.inference.partial_completion import PartialCompletionInfo
    pc = PartialCompletionInfo(
        reason="error", tokens_completed=3, tokens_requested=8,
        timestamp="2026-05-23T12:00:00Z")
    r = _receipt(partial_completion=pc)
    # dropping it (the old handler bug) changes the signed bytes
    assert r.signing_payload() != _receipt().signing_payload()
    # restoring it (from_dict / the fixed handler) reproduces them
    rebuilt = InferenceReceipt.from_dict(r.to_dict())
    assert rebuilt.partial_completion is not None
    assert rebuilt.signing_payload() == r.signing_payload()


def test_partial_completion_to_dict_from_dict_symmetric_under_nonstr_timestamp():
    """Hardening: to_dict and from_dict must be SYMMETRIC even if a non-str timestamp
    leaks in. Before the fix to_dict left a float as-is while from_dict str()-coerced it,
    so to_dict(from_dict(d)) != d → the signing payload diverged on round-trip and an
    honest partial receipt failed verification."""
    from prsm.compute.inference.partial_completion import PartialCompletionInfo
    leaked = {"reason": "error", "tokens_completed": 3,
              "tokens_requested": 8, "timestamp": 123.0}  # float, not the typed str
    rebuilt = PartialCompletionInfo.from_dict(leaked)
    # round-trip is now a fixed point
    assert PartialCompletionInfo.from_dict(rebuilt.to_dict()).to_dict() == rebuilt.to_dict()
    assert rebuilt.to_dict()["timestamp"] == "123.0"  # coerced once, then stable


# ── F3: a software (tier-none) node is refused confidential work ─────────────────

def test_local_executor_advertises_software_tier_not_sgx():
    """The local builder must NOT hard-code a hardware tier on a software runtime."""
    import inspect
    import prsm.compute.inference.local_inference as li
    src = inspect.getsource(li.build_local_inference_executor)
    assert 'tier_attestation="tier-sgx"' not in src
    assert "TIER_ATTESTATION_NONE" in src


def test_tier_gate_refuses_confidential_for_software_node():
    """The property the F3 fix relies on: a tier-none (software) GPU is refused for a
    confidential privacy tier."""
    from prsm.compute.parallax_scheduling import (
        ParallaxGPU, TierGateAdapter, TierGateRejected,
    )
    from prsm.compute.parallax_scheduling.prsm_types import TIER_ATTESTATION_NONE
    from prsm.compute.tee.models import PrivacyLevel as _PL
    gpu = ParallaxGPU(node_id="n" * 32, region="local-region", layer_capacity=16,
                      stake_amount=1, tier_attestation=TIER_ATTESTATION_NONE,
                      tflops_fp16=1.0, memory_gb=8.0, memory_bandwidth_gbps=100.0)
    with pytest.raises(TierGateRejected):
        TierGateAdapter().filter([gpu], _PL.HIGH)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
