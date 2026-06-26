"""Sprint 1285 — pin HuggingFace model artifacts to an immutable commit (audit round 7 #10, MED).

local_inference._ensure_loaded and the `prsm compute infer` CLI loaded weights via
from_pretrained with NO revision — so they fetched whatever the model repo's mutable default
branch currently points at; a hijacked / retagged HF repo could serve poisoned weights. Fix:
resolve_model_revision reads a PRSM_MODEL_REVISIONS JSON map {model_id: revision} and threads
the pinned revision into every from_pretrained call. Default (unset) → None → the default
branch (prior behavior).

(Sibling supply-chain findings #7 AWS-bootstrap binary integrity, #8 dependency
lockfile/hash-pinning, #9 GitHub Action @master→SHA need exact published checksums/SHAs or a
clean lockfile rebuild — operator/network-dependent, documented as ops action items.)
"""
from __future__ import annotations

from prsm.compute.inference.local_inference import resolve_model_revision


def test_unset_returns_none(monkeypatch):
    monkeypatch.delenv("PRSM_MODEL_REVISIONS", raising=False)
    assert resolve_model_revision("Qwen/Qwen2.5-7B-Instruct") is None


def test_pinned_model_returns_revision(monkeypatch):
    monkeypatch.setenv(
        "PRSM_MODEL_REVISIONS",
        '{"Qwen/Qwen2.5-7B-Instruct": "abc123def", "gpt2": "deadbeef"}',
    )
    assert resolve_model_revision("Qwen/Qwen2.5-7B-Instruct") == "abc123def"
    assert resolve_model_revision("gpt2") == "deadbeef"


def test_unpinned_model_in_map_returns_none(monkeypatch):
    monkeypatch.setenv("PRSM_MODEL_REVISIONS", '{"gpt2": "deadbeef"}')
    assert resolve_model_revision("Qwen/Qwen2.5-7B-Instruct") is None


def test_invalid_json_returns_none(monkeypatch):
    monkeypatch.setenv("PRSM_MODEL_REVISIONS", "not-json{")
    assert resolve_model_revision("gpt2") is None


def test_non_string_revision_ignored(monkeypatch):
    monkeypatch.setenv("PRSM_MODEL_REVISIONS", '{"gpt2": 123}')
    assert resolve_model_revision("gpt2") is None


def test_blank_revision_ignored(monkeypatch):
    monkeypatch.setenv("PRSM_MODEL_REVISIONS", '{"gpt2": "   "}')
    assert resolve_model_revision("gpt2") is None


def test_load_path_threads_revision():
    # source-pin: both from_pretrained sites pass revision=
    import inspect
    import prsm.compute.inference.local_inference as li
    src = inspect.getsource(li._LocalHFExecutor) if hasattr(li, "_LocalHFExecutor") else inspect.getsource(li)
    assert "resolve_model_revision(self._model_id)" in src
    assert src.count("revision=_revision") >= 3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
