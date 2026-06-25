"""Sprint 1273 — cap max_tokens on the distributed inference path (audit round 6, HIGH).

The single-node local executor clamps generation to _MAX_TOKENS_CEILING (256), but the
distributed/parallax runners (the ones serving 7B/14B/72B in production) threaded a
caller-supplied max_tokens straight into the generation loop with NO upper bound, and the
escrow/budget cost is independent of max_tokens — so one cheap request with
max_tokens=100_000_000 pins the GPU for hours (unmetered compute exhaustion / amplification).

Fix: SamplingDefaults gains an operator-controlled max_tokens_ceiling (default 4096,
PRSM_INFERENCE_MAX_TOKENS_CEILING-overridable) and both runners' _effective_max_tokens clamp
to it.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

import prsm.compute.inference.autoregressive_runner as ar
from prsm.compute.inference.autoregressive_runner import (
    AutoregressiveStreamingRunner,
    EmbedderBackedStreamingRunner,
    SamplingDefaults,
)

RUNNERS = [AutoregressiveStreamingRunner, EmbedderBackedStreamingRunner]


def _runner(cls, defaults=None):
    r = cls.__new__(cls)            # bypass the heavy constructor
    r._defaults = defaults or SamplingDefaults()
    return r


def test_sampling_defaults_has_ceiling():
    assert SamplingDefaults().max_tokens_ceiling >= 1
    assert SamplingDefaults().max_tokens <= SamplingDefaults().max_tokens_ceiling


@pytest.mark.parametrize("cls", RUNNERS)
def test_huge_max_tokens_clamped_to_ceiling(cls):
    r = _runner(cls)
    ceiling = r._defaults.max_tokens_ceiling
    assert r._effective_max_tokens(SimpleNamespace(max_tokens=100_000_000)) == ceiling


@pytest.mark.parametrize("cls", RUNNERS)
def test_small_max_tokens_unchanged(cls):
    r = _runner(cls)
    assert r._effective_max_tokens(SimpleNamespace(max_tokens=16)) == 16


@pytest.mark.parametrize("cls", RUNNERS)
def test_none_falls_back_to_default_within_ceiling(cls):
    r = _runner(cls)
    eff = r._effective_max_tokens(SimpleNamespace(max_tokens=None))
    assert eff == min(r._defaults.max_tokens, r._defaults.max_tokens_ceiling)


@pytest.mark.parametrize("cls", RUNNERS)
def test_negative_floored_to_at_least_one(cls):
    r = _runner(cls)
    assert r._effective_max_tokens(SimpleNamespace(max_tokens=-5)) >= 1


def test_env_override_lowers_ceiling(monkeypatch):
    monkeypatch.setenv("PRSM_INFERENCE_MAX_TOKENS_CEILING", "100")
    importlib.reload(ar)
    r = ar.AutoregressiveStreamingRunner.__new__(ar.AutoregressiveStreamingRunner)
    r._defaults = ar.SamplingDefaults()
    assert r._defaults.max_tokens_ceiling == 100
    assert r._effective_max_tokens(SimpleNamespace(max_tokens=100_000_000)) == 100
    monkeypatch.delenv("PRSM_INFERENCE_MAX_TOKENS_CEILING", raising=False)
    importlib.reload(ar)   # restore default for other tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
