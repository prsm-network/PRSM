"""Sprint 295 — activation-layer DP noise primitive.

Vision §7 honest-limits names activation-inversion attacks
(Zhu et al. 2019 + follow-up literature) as a real concern
that the current Phase-2 DP injection on aggregated output
does NOT defend against — because the inversion attack
operates on intermediate per-stage activations, not the
final output.

This module ships the primitive that the streaming-inference
subsystem (Phase 3.x.7+ RpcChainExecutor) will use to inject
calibrated Gaussian noise on EACH inter-stage activation:

  StageNoisePolicy
    Pure function mapping (privacy_tier, stage_count) →
    per-stage ε under basic composition. The composition
    theorem under DP says the total ε across k sequential
    mechanisms is bounded by Σ ε_i (basic) or by
    sqrt(2k·ln(1/δ))·max ε_i (advanced). v1 uses basic
    composition (conservative, simpler, easier to verify):
    per_stage_ε = tier_ε / stage_count.

  ActivationDPInjector
    Per-stage Gaussian noise injector. Internal state tracks
    which stage_index has already been injected (prevents
    duplicate noise allocation that would leak privacy) and
    accumulates the trace of per-stage ε spend.

  ActivationNoiseTrace
    Serializable per-stage ε record. Sprint 297 will surface
    this as a field on InferenceReceipt so verifiers can
    confirm activation noise was actually applied as
    promised.

  verify_activation_noise_trace
    Verifier predicate. Checks:
      - tier label matches expected
      - per-stage ε values are all non-negative
      - sum of per-stage ε matches reported total
      - total matches the tier's promised total
        within tolerance
      - stage_count matches the length of per-stage list
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prsm.compute.tee.models import DPConfig, PrivacyLevel


_EPSILON_TOLERANCE = 1e-6


@dataclass
class StageNoisePolicy:
    """Per-stage noise configuration for an inference run.

    Attributes:
      per_stage_epsilon  ε to spend per stage (basic
                         composition; sum across all stages
                         = total_expected_epsilon)
      total_expected_epsilon  ε total that the tier promises
      stage_count       expected number of stages
      tier              str label (matches PrivacyLevel.value)
      clip_norm         L2 clip applied to activations
                        before noise (sensitivity bound)
      delta             δ parameter for Gaussian DP
      enabled           False iff tier == NONE
    """

    per_stage_epsilon: float
    total_expected_epsilon: float
    stage_count: int
    tier: str
    clip_norm: float = 1.0
    delta: float = 1e-5
    enabled: bool = True

    @classmethod
    def for_tier(
        cls,
        tier: PrivacyLevel,
        stage_count: int,
        *,
        clip_norm: float = 1.0,
        delta: float = 1e-5,
    ) -> "StageNoisePolicy":
        if not isinstance(stage_count, int) or stage_count <= 0:
            raise ValueError(
                f"stage_count must be a positive integer, "
                f"got {stage_count!r}"
            )
        total = PrivacyLevel.config_for_level(tier).epsilon
        if math.isinf(total) or tier == PrivacyLevel.NONE:
            return cls(
                per_stage_epsilon=float("inf"),
                total_expected_epsilon=float("inf"),
                stage_count=stage_count,
                tier=tier.value,
                clip_norm=clip_norm,
                delta=delta,
                enabled=False,
            )
        per_stage = total / stage_count
        return cls(
            per_stage_epsilon=per_stage,
            total_expected_epsilon=total,
            stage_count=stage_count,
            tier=tier.value,
            clip_norm=clip_norm,
            delta=delta,
            enabled=True,
        )


@dataclass
class ActivationNoiseTrace:
    """Per-stage ε spend record. Lives on the InferenceReceipt
    (sprint 297 wiring) so verifiers can confirm activation
    noise was applied as promised."""

    per_stage_epsilon: List[float]
    total_epsilon_spent: float
    clip_norm: float
    stage_count: int
    tier: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_stage_epsilon": list(self.per_stage_epsilon),
            "total_epsilon_spent": self.total_epsilon_spent,
            "clip_norm": self.clip_norm,
            "stage_count": self.stage_count,
            "tier": self.tier,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "ActivationNoiseTrace":
        return cls(
            per_stage_epsilon=list(
                d.get("per_stage_epsilon") or [],
            ),
            total_epsilon_spent=float(
                d.get("total_epsilon_spent", 0.0),
            ),
            clip_norm=float(d.get("clip_norm", 1.0)),
            stage_count=int(d.get("stage_count", 0)),
            tier=str(d.get("tier", "")),
        )


class ActivationDPInjector:
    """Per-stage Gaussian noise injector with composition
    tracking.

    Calling code (RpcChainExecutor in sprint 296+) holds a
    single injector for an entire inference run and calls
    .inject_stage(activation, stage_index) for each stage.
    Trying to inject the same stage_index twice raises
    ValueError — defends against the noise-double-spend
    attack where a malicious operator allocates the budget
    once but uses the activation in multiple downstream
    paths.
    """

    def __init__(self, policy: StageNoisePolicy) -> None:
        self._policy = policy
        self._seen_stages: set = set()
        self._per_stage_eps: Dict[int, float] = {}

    @property
    def policy(self) -> StageNoisePolicy:
        return self._policy

    def inject_stage(
        self, activation: np.ndarray, stage_index: int,
    ) -> np.ndarray:
        """Inject calibrated Gaussian noise into ``activation``
        for the given ``stage_index``. Returns the noised
        tensor. Records the per-stage ε spend on the trace."""
        if not isinstance(stage_index, int):
            raise ValueError("stage_index must be an int")
        if stage_index < 0 or stage_index >= self._policy.stage_count:
            raise ValueError(
                f"stage_index {stage_index} out of range "
                f"[0, {self._policy.stage_count})"
            )
        if stage_index in self._seen_stages:
            raise ValueError(
                f"stage_index {stage_index} already "
                f"injected — duplicate inject_stage call "
                f"would leak privacy (same noise budget "
                f"used twice for the same stage)"
            )

        # Clip activation to bound sensitivity. Operates on a
        # copy so caller's array isn't mutated.
        clipped = activation.copy()
        norm = np.linalg.norm(clipped)
        if norm > self._policy.clip_norm:
            clipped = clipped * (self._policy.clip_norm / norm)

        # When disabled (NONE tier), return clipped result
        # without noise + record ε=0.
        if not self._policy.enabled or math.isinf(
            self._policy.per_stage_epsilon,
        ):
            self._seen_stages.add(stage_index)
            self._per_stage_eps[stage_index] = 0.0
            return clipped

        # Gaussian σ from the DPConfig formula:
        #   σ = clip * sqrt(2 ln(1.25/δ)) / ε
        config = DPConfig(
            epsilon=self._policy.per_stage_epsilon,
            delta=self._policy.delta,
            clip_norm=self._policy.clip_norm,
        )
        sigma = config.noise_scale
        noise = np.random.normal(
            loc=0.0, scale=sigma, size=clipped.shape,
        )
        self._seen_stages.add(stage_index)
        self._per_stage_eps[stage_index] = (
            self._policy.per_stage_epsilon
        )
        return clipped + noise

    def trace(self) -> ActivationNoiseTrace:
        """Snapshot the per-stage ε record. Stages are sorted
        by stage_index. Missing stages (never injected) are
        NOT included — caller is expected to verify
        len(trace.per_stage_epsilon) == policy.stage_count
        before trusting the receipt."""
        sorted_indices = sorted(self._per_stage_eps.keys())
        per_stage = [
            self._per_stage_eps[i] for i in sorted_indices
        ]
        total = sum(per_stage)
        return ActivationNoiseTrace(
            per_stage_epsilon=per_stage,
            total_epsilon_spent=total,
            clip_norm=self._policy.clip_norm,
            stage_count=self._policy.stage_count,
            tier=self._policy.tier,
        )


def verify_activation_noise_trace(
    trace: ActivationNoiseTrace,
    *,
    expected_tier: PrivacyLevel,
) -> Tuple[bool, str]:
    """Verifier predicate. Returns (ok, reason). reason is
    "" on success; on failure it's a short human-readable
    explanation for logging (NOT for return to the vendor —
    info leak protection).

    Checks performed:
      1. tier label matches expected
      2. per_stage_epsilon entries are non-negative
      3. len(per_stage_epsilon) == stage_count (no skipped
         stages)
      4. sum(per_stage_epsilon) == total_epsilon_spent
      5. total within tolerance of tier's expected total
    """
    # 1. Tier match
    expected_tier_str = expected_tier.value
    if trace.tier != expected_tier_str:
        return (
            False,
            f"tier mismatch: trace claims tier={trace.tier!r} "
            f"but verifier was told expected_tier="
            f"{expected_tier_str!r}",
        )

    # 2. Non-negative per-stage values
    for i, eps in enumerate(trace.per_stage_epsilon):
        if eps < 0:
            return (
                False,
                f"invalid negative epsilon at stage {i}: "
                f"ε={eps} (would credit-back budget)",
            )

    # 3. Stage count matches list length
    if len(trace.per_stage_epsilon) != trace.stage_count:
        return (
            False,
            f"stage count mismatch: stage_count="
            f"{trace.stage_count} but per_stage_epsilon "
            f"has {len(trace.per_stage_epsilon)} entries "
            f"(missing stages indicate noise-skip attack)",
        )

    # 4. Sum consistency
    actual_sum = sum(trace.per_stage_epsilon)
    if abs(actual_sum - trace.total_epsilon_spent) > _EPSILON_TOLERANCE:
        return (
            False,
            f"total mismatch with sum: claimed total="
            f"{trace.total_epsilon_spent} but sum of "
            f"per-stage values is {actual_sum}",
        )

    # 5. Total ε within tier's expected ceiling. For NONE
    # tier we expect 0; for others we expect at most the
    # tier's published total (under basic composition;
    # advanced composition would allow a tighter bound but
    # v1 takes the conservative basic interpretation).
    expected_total = PrivacyLevel.config_for_level(
        expected_tier,
    ).epsilon
    if math.isinf(expected_total):
        # NONE tier — total must be 0
        if (
            trace.total_epsilon_spent
            > _EPSILON_TOLERANCE
        ):
            return (
                False,
                f"NONE tier should have ε=0, got "
                f"{trace.total_epsilon_spent}",
            )
        return (True, "")

    # sp1001 — a non-NONE tier with a ~0 total ε means NO DP noise was actually
    # applied. The injector records exactly 0.0 only on its disabled/no-noise path
    # (activation_dp.py inject path), so a 0.0 total for STANDARD/HIGH/MAXIMUM is
    # never an honest under-spend — it is a privacy over-claim (the receipt claims
    # a private tier but the signed trace proves no noise was injected). Reject.
    # Genuine stronger-privacy under-spend uses a small POSITIVE ε and still passes.
    if trace.total_epsilon_spent <= _EPSILON_TOLERANCE:
        return (
            False,
            f"tier {expected_tier_str!r} promises DP noise but the trace "
            f"records total ε={trace.total_epsilon_spent} — no activation "
            f"noise was applied (privacy over-claim)",
        )

    # For non-NONE tier, total may not EXCEED the tier's
    # promised ceiling. (Under-spending is acceptable —
    # callers can run with smaller per-stage ε; over-spending
    # breaches the tier promise.)
    if (
        trace.total_epsilon_spent
        > expected_total + _EPSILON_TOLERANCE
    ):
        return (
            False,
            f"total ε spent ({trace.total_epsilon_spent}) "
            f"exceeds tier {expected_tier_str!r}'s "
            f"promised ceiling ({expected_total})",
        )
    return (True, "")
