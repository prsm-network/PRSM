"""Sprint 1205 — generate a parallax model catalog from HF configs (multi-host).

The multi-host parallax scheduler needs a v1 catalog mapping model_id → ModelInfo.
catalog_entry_for_config derives that from a HF config via the same _derive_model_info
the local path uses, so a production instruct model (Qwen/Llama/Mistral, incl. GQA) is
correctly sliced across hosts — no hand-authoring. The generated entries must satisfy the
real catalog parser's required fields and round-trip through ModelInfo(**entry).
"""
from __future__ import annotations

import pytest

from prsm.compute.inference.local_inference import (
    build_parallax_catalog_v1,
    catalog_entry_for_config,
)
from prsm.node.inference_wiring import _REQUIRED_MODEL_INFO_FIELDS
from prsm.compute.parallax_scheduling.model_info import ModelInfo


class _QwenLikeConfig:
    # GQA modern config (Qwen2.5-7B-Instruct shape)
    num_hidden_layers = 28
    hidden_size = 3584
    num_attention_heads = 28
    num_key_value_heads = 4
    intermediate_size = 18944
    vocab_size = 152064


class _Gpt2LikeConfig:
    n_layer = 12
    n_embd = 768
    n_head = 12
    n_inner = 3072
    vocab_size = 50257


def test_entry_from_modern_gqa_config():
    e = catalog_entry_for_config(_QwenLikeConfig(), "Qwen/Qwen2.5-7B-Instruct")
    assert e["num_layers"] == 28
    assert e["hidden_dim"] == 3584
    assert e["num_attention_heads"] == 28
    assert e["num_kv_heads"] == 4          # GQA: kv-heads < attention-heads
    assert e["head_size"] == 3584 // 28    # 128
    assert e["intermediate_dim"] == 18944
    assert e["vocab_size"] == 152064
    assert e["model_name"] == "Qwen/Qwen2.5-7B-Instruct"


def test_entry_from_gpt2_style_config():
    e = catalog_entry_for_config(_Gpt2LikeConfig(), "gpt2")
    assert e["num_layers"] == 12
    assert e["num_kv_heads"] == 12         # MHA: kv-heads default to attention-heads
    assert e["head_size"] == 64


def test_entry_satisfies_catalog_required_fields():
    e = catalog_entry_for_config(_QwenLikeConfig(), "Qwen/Qwen2.5-7B-Instruct")
    missing = [f for f in _REQUIRED_MODEL_INFO_FIELDS if f not in e]
    assert missing == [], f"missing required catalog fields: {missing}"


def test_entry_round_trips_through_model_info():
    e = catalog_entry_for_config(_QwenLikeConfig(), "Qwen/Qwen2.5-7B-Instruct")
    mi = ModelInfo(**e)  # the parser does exactly this — must not raise
    assert mi.num_layers == 28 and mi.num_kv_heads == 4


def test_build_catalog_v1_shape():
    cat = build_parallax_catalog_v1({
        "Qwen/Qwen2.5-7B-Instruct": _QwenLikeConfig(),
        "gpt2": _Gpt2LikeConfig(),
    })
    assert cat["schema_version"] == "v1"
    assert set(cat["models"].keys()) == {"Qwen/Qwen2.5-7B-Instruct", "gpt2"}
    # every entry is a valid ModelInfo + has the required fields
    for mid, kw in cat["models"].items():
        assert all(f in kw for f in _REQUIRED_MODEL_INFO_FIELDS)
        ModelInfo(**kw)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
