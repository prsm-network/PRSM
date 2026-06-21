"""Sprint 1212 — the mock parallax trust stack honors the advisory bypasses.

The 2-A10 multi-host bench (2026-06-21) hit "no GPU passed pool gating" with
PRSM_PARALLAX_TRUST_STACK_KIND=mock: the mock anchor accepts every node, but its
zero-stake lookup + tier gate still ENFORCED, filtering an un-staked / software-tier
pool to empty. The advisory bypasses (sp686 stake / sp702 tier) were only applied in
the PRODUCTION trust stack. Now the mock stack applies them too, so a dev/bench
operator can run a real off-chain multi-host pool with
PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory + PRSM_PARALLAX_TIER_GATE=advisory.
Default (no advisory) stays enforced.
"""
from __future__ import annotations

import pytest

from prsm.node.inference_wiring import _build_mock_trust_stack
from prsm.compute.parallax_scheduling.prsm_types import ParallaxGPU


def _gpu(node_id: str) -> ParallaxGPU:
    return ParallaxGPU(
        node_id=node_id, region="default", layer_capacity=6, stake_amount=0,
        tier_attestation="tier-none", tflops_fp16=33.9, memory_gb=24.0,
        memory_bandwidth_gbps=600.0, gpu_name="NVIDIA A10", device="cuda",
    )


def test_mock_stack_rejects_unstaked_pool_when_enforced(monkeypatch):
    # Default posture (no advisory) — zero-stake nodes are filtered out.
    monkeypatch.delenv("PRSM_PARALLAX_STAKE_ELIGIBILITY", raising=False)
    monkeypatch.delenv("PRSM_PARALLAX_TIER_GATE", raising=False)
    stack = _build_mock_trust_stack()
    kept = stack.filter_pool([_gpu("a" * 32), _gpu("b" * 32)])
    assert list(kept) == []  # enforced: zero stake → rejected


def test_mock_stack_admits_pool_under_advisory(monkeypatch):
    # The bench posture — advisory bypass makes the mock stack permissive.
    monkeypatch.setenv("PRSM_PARALLAX_STAKE_ELIGIBILITY", "advisory")
    monkeypatch.setenv("PRSM_PARALLAX_TIER_GATE", "advisory")
    stack = _build_mock_trust_stack()
    pool = [_gpu("a" * 32), _gpu("b" * 32)]
    kept = stack.filter_pool(pool)
    assert len(list(kept)) == 2  # both un-staked A10s pass


def test_mock_stack_tier_gate_advisory_passes_standard_tier(monkeypatch):
    # Adapter B (tier) advisory: a software-tier pool serves a standard-tier request.
    monkeypatch.setenv("PRSM_PARALLAX_STAKE_ELIGIBILITY", "advisory")
    monkeypatch.setenv("PRSM_PARALLAX_TIER_GATE", "advisory")
    stack = _build_mock_trust_stack()
    pool = stack.filter_pool([_gpu("a" * 32)])
    tier_kept = stack.filter_for_request(list(pool), "standard")
    assert len(list(tier_kept)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
