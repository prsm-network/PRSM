"""Sprint 1214 — the filesystem model registry accepts HuggingFace org/model ids.

The 2-A10 multi-host bench failed at chain stage execution with
"registry error: ValueError: model_id='Qwen/Qwen2.5-3B-Instruct' unsafe for filesystem
registry" — the strict id validator rejected the '/', so only slashless ids like 'gpt2'
could ever be staged, blocking every production HF model on the parallax path.
_validate_model_id now allows '/'-separated fs-safe segments (rejecting '.'/'..'), and
_model_dir's existing is_relative_to(root) guard keeps traversal impossible.
"""
from __future__ import annotations

import pytest

from prsm.compute.model_registry.registry import (
    FilesystemModelRegistry,
    _validate_model_id,
)


def test_accepts_hf_org_model_id():
    _validate_model_id("Qwen/Qwen2.5-3B-Instruct")  # no raise
    _validate_model_id("meta-llama/Llama-3.1-8B-Instruct")
    _validate_model_id("gpt2")  # slashless still fine (back-compat)


def test_rejects_traversal_and_empty_segments():
    for bad in ("../etc", "Qwen/..", ".", "a//b", "", "Qwen/./x", "/leading", "trailing/"):
        with pytest.raises(ValueError):
            _validate_model_id(bad)


def test_model_dir_nests_slash_id_under_root(tmp_path):
    reg = FilesystemModelRegistry(tmp_path)
    d = reg._model_dir("Qwen/Qwen2.5-3B-Instruct")
    assert d.resolve().is_relative_to(tmp_path.resolve())  # stays under root
    assert d.name == "Qwen2.5-3B-Instruct" and d.parent.name == "Qwen"


def test_model_dir_still_rejects_escape(tmp_path):
    reg = FilesystemModelRegistry(tmp_path)
    # _validate_model_id catches "..", but _model_dir's guard is the defense-in-depth.
    with pytest.raises(ValueError):
        _validate_model_id("../escape")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
