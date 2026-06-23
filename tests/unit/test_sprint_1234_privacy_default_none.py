"""Sprint 1234 — honest privacy default: privacy_tier defaults to NONE.

GPU sweep (sp1233 Phase 2) proved activation-DP can't give strong privacy AND
usable quality on high-dim activations (SNR ∝ ε²/D), and the shipped STANDARD
default produced GARBAGE at every ε. So a real user's inference defaulted to
broken output. Per the product decision (2026-06-23): default to NONE (usable
output), keep STANDARD/HIGH/MAXIMUM available but relabeled as experimental /
best-effort obfuscation (NOT strong privacy — real privacy = TEE Tier 3).
"""

from decimal import Decimal

from prsm.compute.inference.models import InferenceRequest
from prsm.compute.tee.models import PrivacyLevel


def test_default_privacy_tier_is_none():
    req = InferenceRequest(
        prompt="The capital of France is",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        budget_ftns=Decimal("1.0"),
    )
    # honest default: a user who doesn't opt into DP gets USABLE output
    assert req.privacy_tier == PrivacyLevel.NONE


def test_dp_tiers_still_selectable_and_mapped():
    # the experimental tiers are NOT removed — operators can still opt in
    for tier, eps in [
        (PrivacyLevel.STANDARD, 8.0),
        (PrivacyLevel.HIGH, 4.0),
        (PrivacyLevel.MAXIMUM, 1.0),
    ]:
        req = InferenceRequest(
            prompt="x", model_id="m", budget_ftns=Decimal("1.0"),
            privacy_tier=tier,
        )
        assert req.privacy_tier == tier
        assert PrivacyLevel.config_for_level(tier).epsilon == eps


def test_none_tier_has_no_dp_noise():
    import math
    cfg = PrivacyLevel.config_for_level(PrivacyLevel.NONE)
    assert math.isinf(cfg.epsilon)  # ε=inf → no noise on the NONE path
