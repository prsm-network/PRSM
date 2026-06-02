"""Tests for SentenceTransformerEmbedder — production-side Embedder
implementation that satisfies the Protocol declared in
`prsm.compute.query_orchestrator.semantic_index_adapter`.

The Embedder bridges query strings to the same vector space used by
the content-upload path (`prsm/node/content_uploader.py`) so that
queries and stored shard embeddings are mutually comparable. The
default model name pinned here MUST match the default model name
content_uploader documents (`sentence-transformers/all-MiniLM-L6-v2`).

These tests use the real sentence_transformers library when present;
a missing-dep environment is signalled via `pytest.importorskip`.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

# Skip the entire module if sentence_transformers isn't available —
# requirements.txt pins it, so this only fires in a stripped env.
pytest.importorskip("sentence_transformers")

# Sprint 141 — skip when torch is in a state where strided
# tensors fail. Some upstream test in the full-suite run pollutes
# torch._dynamo / PrimTorch state such that fresh model
# instantiation hits "PrimTorch doesn't support layout=
# torch.strided". The tests pass when run in isolation and the
# embedder code itself is correct; the failure is purely a
# test-isolation hygiene issue with no clean upstream fix.
# Skip-on-detection keeps CI green without papering over real
# embedder bugs.
try:
    import torch as _torch_check
    _torch_check.empty(1, layout=_torch_check.strided)
    _TORCH_STRIDED_OK = True
except Exception:
    _TORCH_STRIDED_OK = False

if not _TORCH_STRIDED_OK:
    pytest.skip(
        "torch.strided layout broken in this process — likely "
        "test-isolation pollution from an earlier test. Tests "
        "pass when this module is run standalone.",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _force_hf_offline(monkeypatch):
    """Force HF Hub offline mode for tests. The model is pre-cached in
    `~/.cache/huggingface`; no network round-trip needed. Without this
    fixture the test environment's mocked httpx layer (set up in
    `tests/conftest.py`) clashes with transformers' redirect handling
    and produces spurious failures unrelated to the embedder under
    test.

    `huggingface_hub.constants.HF_HUB_OFFLINE` is read once at import
    time, so setting only the env var is too late by the time pytest
    has loaded the package — we monkeypatch the resolved constant
    directly. We set the env vars too as belt-and-suspenders for any
    code path that re-reads the env."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    try:
        from huggingface_hub import constants as _hf_constants

        monkeypatch.setattr(_hf_constants, "HF_HUB_OFFLINE", True)
    except ImportError:
        pass
    yield

from prsm.compute.query_orchestrator.semantic_index_adapter import Embedder
from prsm.compute.query_orchestrator.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)


# ──────────────────────────────────────────────────────────────────────
# Empty-input guards
# ──────────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_empty_query_raises(self):
        embedder = SentenceTransformerEmbedder()
        with pytest.raises(ValueError, match="query is empty"):
            embedder.encode("")

    def test_whitespace_query_raises(self):
        embedder = SentenceTransformerEmbedder()
        with pytest.raises(ValueError, match="query is empty"):
            embedder.encode("   \t\n  ")


# ──────────────────────────────────────────────────────────────────────
# Output shape + dtype
# ──────────────────────────────────────────────────────────────────────


class TestEncodeOutput:
    def test_returns_ndarray(self):
        embedder = SentenceTransformerEmbedder()
        out = embedder.encode("hello world")
        assert isinstance(out, np.ndarray)

    def test_returns_float32(self):
        embedder = SentenceTransformerEmbedder()
        out = embedder.encode("hello world")
        assert out.dtype == np.float32

    def test_returns_1d_vector(self):
        embedder = SentenceTransformerEmbedder()
        out = embedder.encode("hello world")
        # all-MiniLM-L6-v2 → 384-dim. Don't pin the exact dim because
        # the model name is a parameter; pin "1-D, non-trivial-length".
        assert out.ndim == 1
        assert out.shape[0] > 0


# ──────────────────────────────────────────────────────────────────────
# Determinism + meaningful semantics
# ──────────────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_query_same_output(self):
        embedder = SentenceTransformerEmbedder()
        out1 = embedder.encode("the quick brown fox")
        out2 = embedder.encode("the quick brown fox")
        assert np.allclose(out1, out2)

    def test_different_queries_different_outputs(self):
        embedder = SentenceTransformerEmbedder()
        out1 = embedder.encode("the quick brown fox")
        out2 = embedder.encode("a totally unrelated sentence about cats")
        # Don't pin a similarity threshold — just confirm not byte-equal.
        assert not np.allclose(out1, out2)


# ──────────────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────────────


class TestProtocol:
    def test_satisfies_embedder_protocol(self):
        embedder = SentenceTransformerEmbedder()
        assert isinstance(embedder, Embedder)


# ──────────────────────────────────────────────────────────────────────
# Lazy loading
# ──────────────────────────────────────────────────────────────────────


class TestLazyLoading:
    """Constructor must not load weights — first encode() does. This
    is load-bearing for node startup time: the Embedder is constructed
    in node.py wiring even if no query ever arrives, so paying the
    model-load cost up-front would slow every cold start."""

    def test_constructor_does_not_load_model(self):
        # Sprint 460 (F18 fix, commit becc2e4e) made the
        # `sentence_transformers` import lazy — it now lives inside
        # `_ensure_loaded()`, not at module top-level — so there is no
        # `sentence_transformer_embedder.SentenceTransformer` attribute
        # to patch. Patch the real import target (the
        # sentence_transformers module's class) so the assertion still
        # proves the load-bearing invariant: the constructor must not
        # instantiate the model. We additionally assert `_model is None`
        # post-construction, which is the lazy-state invariant that
        # survives the refactor.
        with patch(
            "sentence_transformers.SentenceTransformer"
        ) as mock_st:
            embedder = SentenceTransformerEmbedder()
            assert mock_st.call_count == 0, (
                "Constructor must not instantiate SentenceTransformer; "
                "model load is deferred to first encode()."
            )
            assert embedder._model is None, (
                "Constructor must leave _model unmaterialized; the "
                "model object only exists after first encode()."
            )

    def test_first_encode_loads_model_once(self):
        # We can't mock the real encode call without breaking type
        # contracts, so instead we use the real model and check that
        # repeat encodes don't re-instantiate it.
        embedder = SentenceTransformerEmbedder()
        # First encode triggers load.
        embedder.encode("first")
        first_model_obj = embedder._model
        assert first_model_obj is not None
        # Second encode reuses it.
        embedder.encode("second")
        assert embedder._model is first_model_obj
