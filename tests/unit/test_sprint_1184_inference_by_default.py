"""Sprint 1184 — real inference by DEFAULT (day-one-live blocker #2).

Out of the box PRSM_INFERENCE_EXECUTOR is unset → executor None → POST /compute/inference
503. This makes a properly-installed node (with the [ml] extra) serve real single-node
inference with NO env var: when unset, the kind resolves to 'local' iff torch+transformers
are importable; otherwise it stays off with an actionable install message. Adds an
explicit none/off opt-out. And the local model load now allows download by default
(PRSM_LOCAL_INFERENCE_OFFLINE=1 restores offline-only) so the FIRST request actually
succeeds on a fresh install instead of failing on local_files_only with an empty cache.
"""
from __future__ import annotations

import pytest

from prsm.node.inference_wiring import (
    diagnose_inference_executor_unavailable,
    resolve_inference_executor_kind,
)


# ── resolve_inference_executor_kind: the default-selection logic ─────────────────────

def test_unset_defaults_to_local_when_ml_available():
    assert resolve_inference_executor_kind("", ml_available=True) == "local"


def test_unset_stays_off_when_ml_unavailable():
    assert resolve_inference_executor_kind("", ml_available=False) == ""


def test_whitespace_is_unset():
    assert resolve_inference_executor_kind("   ", ml_available=True) == "local"


@pytest.mark.parametrize("off", ["none", "off", "disabled", "NONE", " Off "])
def test_explicit_off_disables_even_with_ml(off):
    assert resolve_inference_executor_kind(off, ml_available=True) == ""


@pytest.mark.parametrize("kind", ["mock", "local", "parallax", "mock-streaming"])
def test_explicit_kinds_pass_through(kind):
    assert resolve_inference_executor_kind(kind.upper(), ml_available=True) == kind


def test_explicit_local_honored_even_without_ml():
    # An explicit request is NOT silently overridden — the build then fails loudly and
    # diagnose explains, rather than us second-guessing the operator.
    assert resolve_inference_executor_kind("local", ml_available=False) == "local"


def test_unknown_value_passes_through_for_diagnose():
    # Unknown values flow through unchanged so the existing 'unrecognized' diagnostic fires.
    assert resolve_inference_executor_kind("banana", ml_available=True) == "banana"


# ── diagnose: the unset message now points at the [ml] extra (+ keeps sp1033 tokens) ─

def test_unset_diagnostic_mentions_ml_extra_and_keeps_tokens():
    msg = diagnose_inference_executor_unavailable(
        env={}, module_check=lambda n: False)
    assert "PRSM_INFERENCE_EXECUTOR" in msg   # sp1033 contract preserved
    assert "local" in msg and "parallax" in msg
    assert "[ml]" in msg                      # new: how to enable the default


# ── local inference offline flag: allow download by default ──────────────────────────

def test_offline_resolves_from_env_default_false(monkeypatch):
    from prsm.compute.inference.local_inference import _resolve_offline
    monkeypatch.delenv("PRSM_LOCAL_INFERENCE_OFFLINE", raising=False)
    assert _resolve_offline(None) is False  # default: allow download (just-works)


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE"])
def test_offline_env_truthy(monkeypatch, val):
    from prsm.compute.inference.local_inference import _resolve_offline
    monkeypatch.setenv("PRSM_LOCAL_INFERENCE_OFFLINE", val)
    assert _resolve_offline(None) is True


def test_offline_explicit_arg_wins(monkeypatch):
    from prsm.compute.inference.local_inference import _resolve_offline
    monkeypatch.setenv("PRSM_LOCAL_INFERENCE_OFFLINE", "1")
    assert _resolve_offline(False) is False  # explicit arg overrides env


def test_ensure_loaded_passes_local_files_only_from_offline(monkeypatch):
    """The offline flag threads into transformers from_pretrained(local_files_only=...)."""
    import transformers
    from prsm.compute.inference.local_inference import LocalHuggingFaceChainExecutor

    seen = {}

    class _FakeTok:
        pad_token_id = 0
        eos_token = "<eos>"
        pad_token = None

    class _FakeModel:
        def eval(self):
            return self

        def to(self, *args, **kwargs):  # _ensure_loaded calls model.to(device)
            return self

    def fake_tok(model_id, **kw):
        seen["tok_local_files_only"] = kw.get("local_files_only")
        return _FakeTok()

    def fake_model(model_id, **kw):
        seen["model_local_files_only"] = kw.get("local_files_only")
        return _FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tok))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_model))

    # offline=False → allow download
    ex = LocalHuggingFaceChainExecutor(model_id="distilgpt2", offline=False)
    ex._ensure_loaded()
    assert seen["tok_local_files_only"] is False
    assert seen["model_local_files_only"] is False

    # offline=True → local-only
    seen.clear()
    ex2 = LocalHuggingFaceChainExecutor(model_id="distilgpt2", offline=True)
    ex2._ensure_loaded()
    assert seen["tok_local_files_only"] is True
    assert seen["model_local_files_only"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
