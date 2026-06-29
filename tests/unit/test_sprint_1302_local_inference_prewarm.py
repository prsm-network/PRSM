"""Sprint 1302 — local inference startup pre-warm + readiness.

Closes the default-user first-request friction: with `.[ml]` installed a bare
install already defaults to real local inference (sp1184), but the model loads
LAZILY on the first /compute/inference (~350MB download + load = a 10–30s hang that
looks like a timeout). sp1302 pre-warms the model OFF the request path at startup
(background daemon thread, fail-soft, PRSM_LOCAL_INFERENCE_PREWARM-gated) and adds a
readiness snapshot for status/health surfaces.

These tests never require torch/transformers: construction is cheap (lazy load), and
warm() is exercised in its fail-soft path or with a stubbed `_model`.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from prsm.compute.inference.local_inference import (
    LocalHuggingFaceChainExecutor,
    _find_local_chain_executor,
    _prewarm_enabled,
    local_inference_readiness,
    start_local_prewarm,
)


# ── pre-warm flag resolution ─────────────────────────────────────────────────

def test_prewarm_enabled_default_on(monkeypatch):
    monkeypatch.delenv("PRSM_LOCAL_INFERENCE_PREWARM", raising=False)
    assert _prewarm_enabled(None) is True


@pytest.mark.parametrize("val,expected", [
    ("0", False), ("off", False), ("false", False), ("no", False),
    ("1", True), ("true", True), ("yes", True), ("on", True),
])
def test_prewarm_enabled_env(monkeypatch, val, expected):
    monkeypatch.setenv("PRSM_LOCAL_INFERENCE_PREWARM", val)
    assert _prewarm_enabled(None) is expected


def test_prewarm_enabled_explicit_wins(monkeypatch):
    monkeypatch.setenv("PRSM_LOCAL_INFERENCE_PREWARM", "0")
    assert _prewarm_enabled(True) is True   # explicit arg overrides env
    monkeypatch.setenv("PRSM_LOCAL_INFERENCE_PREWARM", "1")
    assert _prewarm_enabled(False) is False


# ── executor readiness methods (no torch needed) ─────────────────────────────

def test_is_loaded_and_describe_before_load():
    ex = LocalHuggingFaceChainExecutor(model_id="distilgpt2", max_tokens=8)
    assert ex.is_loaded is False
    d = ex.describe()
    assert d["model_id"] == "distilgpt2"
    assert d["loaded"] is False
    assert d["device"] is None          # not exposed until loaded
    assert d["max_tokens"] == 8


def test_warm_idempotent_when_already_loaded():
    ex = LocalHuggingFaceChainExecutor(model_id="distilgpt2")
    ex._model = object()                 # simulate loaded (no torch)
    assert ex.is_loaded is True
    assert ex.warm() is True             # short-circuits, no load attempt
    assert ex.describe()["loaded"] is True


def test_warm_is_fail_soft():
    """warm() on an offline + nonexistent model must return False WITHOUT raising
    (so a fresh install with no network / no [ml] never crashes startup)."""
    ex = LocalHuggingFaceChainExecutor(
        model_id="__prsm_nonexistent_model__", offline=True)
    assert ex.warm() is False
    assert ex.is_loaded is False         # unchanged; lazy path still available


# ── start_local_prewarm ──────────────────────────────────────────────────────

def test_start_prewarm_none_when_disabled():
    ex = LocalHuggingFaceChainExecutor(model_id="distilgpt2")
    assert start_local_prewarm(ex, enabled=False) is None


def test_start_prewarm_none_for_non_local_executor():
    assert start_local_prewarm(SimpleNamespace(foo=1), enabled=True) is None


def test_start_prewarm_none_when_already_loaded():
    ex = LocalHuggingFaceChainExecutor(model_id="distilgpt2")
    ex._model = object()
    assert start_local_prewarm(ex, enabled=True) is None


def test_start_prewarm_spawns_thread_and_completes():
    """enabled + not loaded → a daemon thread runs warm() (fail-soft) to completion."""
    ex = LocalHuggingFaceChainExecutor(
        model_id="__prsm_nonexistent_model__", offline=True)
    t = start_local_prewarm(ex, enabled=True)
    assert isinstance(t, threading.Thread)
    assert t.daemon is True
    t.join(timeout=15)
    assert not t.is_alive()              # warm() returned (False, fail-soft)


def test_start_prewarm_finds_inner_chain_executor():
    """A ParallaxScheduledExecutor-like wrapper exposes the inner executor at
    ._chain_executor; the pre-warm reaches through it."""
    inner = LocalHuggingFaceChainExecutor(model_id="distilgpt2")
    inner._model = object()              # already loaded → start returns None
    wrapper = SimpleNamespace(_chain_executor=inner)
    assert _find_local_chain_executor(wrapper) is inner
    assert start_local_prewarm(wrapper, enabled=True) is None  # loaded → no thread


# ── readiness snapshot ───────────────────────────────────────────────────────

def test_readiness_for_local_executor():
    ex = LocalHuggingFaceChainExecutor(model_id="gpt2", max_tokens=16)
    r = local_inference_readiness(ex)
    assert r == {"enabled": True, "kind": "local", "model_id": "gpt2",
                 "loaded": False, "device": None, "offline": ex._offline,
                 "max_tokens": 16}


def test_readiness_via_wrapper():
    inner = LocalHuggingFaceChainExecutor(model_id="distilgpt2")
    wrapper = SimpleNamespace(_chain_executor=inner)
    r = local_inference_readiness(wrapper)
    assert r["enabled"] is True and r["kind"] == "local"
    assert r["model_id"] == "distilgpt2"


def test_readiness_for_non_local_executor():
    assert local_inference_readiness(SimpleNamespace()) == {
        "enabled": False, "kind": None}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
