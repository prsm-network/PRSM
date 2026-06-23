"""Sprint 1233 — configurable activation-DP + per-token clipping (experiment infra).

The privacy tier produces garbage because per-element Gaussian noise on a
high-dim activation has SNR ∝ ε²/D (curse of dimensionality). To empirically
characterize the privacy/quality curve and test candidate mitigations on GPU,
the DP mechanism's knobs must be sweepable WITHOUT per-run code edits:

  PRSM_ACTIVATION_DP_EPSILON     override the tier's total ε (sweep 8 → 1e4)
  PRSM_ACTIVATION_DP_CLIP_NORM   override clip_norm (default 1.0)
  PRSM_ACTIVATION_DP_CLIP_MODE   "tensor" (default) | "token" (per-token L2 clip
                                  — bounds the per-token DP unit instead of the
                                  whole [seq×hidden] tensor)

Default (no env) is byte-identical to the pre-sp1233 mechanism. This is
privacy-sensitive: the new modes are opt-in and the noise math is unit-tested.
"""

import numpy as np
import pytest

from prsm.compute.inference.activation_dp import (
    ActivationDPInjector,
    StageNoisePolicy,
    _clip_activation,
)
from prsm.compute.tee.models import PrivacyLevel


# ── env-driven config resolution ────────────────────────────────────

def test_for_tier_defaults_unchanged(monkeypatch):
    for v in ("PRSM_ACTIVATION_DP_EPSILON", "PRSM_ACTIVATION_DP_CLIP_NORM",
              "PRSM_ACTIVATION_DP_CLIP_MODE"):
        monkeypatch.delenv(v, raising=False)
    p = StageNoisePolicy.for_tier(PrivacyLevel.STANDARD, stage_count=2)
    assert p.total_expected_epsilon == pytest.approx(8.0)   # tier default
    assert p.per_stage_epsilon == pytest.approx(4.0)        # 8 / 2
    assert p.clip_norm == pytest.approx(1.0)
    assert p.clip_mode == "tensor"


def test_for_tier_epsilon_override(monkeypatch):
    monkeypatch.setenv("PRSM_ACTIVATION_DP_EPSILON", "1000")
    p = StageNoisePolicy.for_tier(PrivacyLevel.STANDARD, stage_count=4)
    assert p.total_expected_epsilon == pytest.approx(1000.0)
    assert p.per_stage_epsilon == pytest.approx(250.0)


def test_for_tier_clip_and_mode_override(monkeypatch):
    monkeypatch.setenv("PRSM_ACTIVATION_DP_CLIP_NORM", "50")
    monkeypatch.setenv("PRSM_ACTIVATION_DP_CLIP_MODE", "token")
    p = StageNoisePolicy.for_tier(PrivacyLevel.HIGH, stage_count=2)
    assert p.clip_norm == pytest.approx(50.0)
    assert p.clip_mode == "token"


def test_for_tier_garbage_overrides_fall_back(monkeypatch):
    monkeypatch.setenv("PRSM_ACTIVATION_DP_EPSILON", "not-a-number")
    monkeypatch.setenv("PRSM_ACTIVATION_DP_CLIP_NORM", "xyz")
    monkeypatch.setenv("PRSM_ACTIVATION_DP_CLIP_MODE", "bogus")
    p = StageNoisePolicy.for_tier(PrivacyLevel.STANDARD, stage_count=2)
    assert p.total_expected_epsilon == pytest.approx(8.0)
    assert p.clip_norm == pytest.approx(1.0)
    assert p.clip_mode == "tensor"  # unknown mode → safe default


# ── clipping geometry: tensor vs token ──────────────────────────────

def test_clip_tensor_mode_bounds_whole_tensor_norm():
    act = np.array([[3.0, 4.0], [6.0, 8.0]])  # row norms 5, 10; whole 13.6
    clipped = _clip_activation(act, clip_norm=1.0, clip_mode="tensor")
    assert np.linalg.norm(clipped) == pytest.approx(1.0, rel=1e-6)
    # direction preserved
    assert np.allclose(clipped / np.linalg.norm(clipped), act / np.linalg.norm(act))


def test_clip_token_mode_bounds_each_row_norm():
    act = np.array([[3.0, 4.0], [6.0, 8.0], [0.1, 0.0]])  # norms 5, 10, 0.1
    clipped = _clip_activation(act, clip_norm=1.0, clip_mode="token")
    row_norms = np.linalg.norm(clipped, axis=-1)
    # rows that exceeded clip are scaled to 1.0; the small row is left alone
    assert row_norms[0] == pytest.approx(1.0, rel=1e-6)
    assert row_norms[1] == pytest.approx(1.0, rel=1e-6)
    assert row_norms[2] == pytest.approx(0.1, rel=1e-6)


def test_clip_token_mode_preserves_more_signal_than_tensor():
    """The whole point: per-token clipping does NOT shrink a high-norm activation
    by its (large) global norm, so each token keeps unit scale."""
    act = np.ones((10, 100))  # global norm ~31.6; each row norm 10
    tensor_clipped = _clip_activation(act, clip_norm=1.0, clip_mode="tensor")
    token_clipped = _clip_activation(act, clip_norm=1.0, clip_mode="token")
    # tensor: every element ~ 1/31.6; token: every element ~ 1/10 → larger
    assert np.abs(token_clipped).mean() > np.abs(tensor_clipped).mean()


# ── injector honors clip_mode + records ε ───────────────────────────

def test_injector_token_mode_clips_per_token_no_noise_path():
    # NONE tier → clip applied, no noise → assert per-token geometry
    pol = StageNoisePolicy(
        per_stage_epsilon=float("inf"), total_expected_epsilon=float("inf"),
        stage_count=1, tier="none", clip_norm=1.0, clip_mode="token", enabled=False)
    inj = ActivationDPInjector(pol)
    act = np.array([[3.0, 4.0], [6.0, 8.0]])
    out = inj.inject_stage(act, 0)
    assert np.linalg.norm(out[0], axis=-1) == pytest.approx(1.0, rel=1e-6)
    assert np.linalg.norm(out[1], axis=-1) == pytest.approx(1.0, rel=1e-6)


def test_injector_noise_added_and_epsilon_recorded():
    np.random.seed(0)
    pol = StageNoisePolicy.for_tier(PrivacyLevel.STANDARD, stage_count=2)
    inj = ActivationDPInjector(pol)
    act = np.zeros((4, 4))  # zero signal → output is pure noise
    out = inj.inject_stage(act, 0)
    assert not np.allclose(out, 0.0)        # noise was added
    assert inj.trace().per_stage_epsilon == [pytest.approx(4.0)]  # 8/2 recorded
