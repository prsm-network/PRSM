"""Sprint 1221 (Brick 2/6) — pure slice-key resolver + straddling-shard
safetensors reader for the per-stage layer-slice weight load.

These are PURE functions (no model construction): given a model snapshot's
weight_map, compute the exact set of checkpoint keys a stage OWNS (its decoder
layers [start,end) + on the final stage the norm + lm_head/embed), then read
ONLY those tensors from the sharded safetensors — grouping by shard so a layer
whose tensors STRADDLE two shards is read correctly, and RELATIVE-remapping
``model.layers.{start+i}`` → ``model.layers.{i}`` so the loaded slice fits a
reduced-config skeleton (Brick 3). Covers the single-file (no index.json) case
flagged by the adversarial review (CO1).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")
from safetensors.torch import save_file  # noqa: E402

from prsm.node.chain_executor_adapters import (  # noqa: E402
    _layer_index_of,
    _read_owned_state_dict,
    _remap_layer_key,
    _resolve_slice_owned_keys,
    _resolve_weight_map,
)

LP = "model.layers"
NORM = "model.norm.weight"
LMH = "lm_head.weight"
EMB = "model.embed_tokens.weight"


def _t(*shape):
    return torch.zeros(*shape, dtype=torch.float32)


def _build_sharded_snapshot(tmp_path) -> Path:
    """3-layer Qwen2-style model split across 2 shards, with layer 1 STRADDLING
    both shards (q_proj in shard1, mlp/norm in shard2)."""
    d = tmp_path / "sharded"
    d.mkdir()
    shard1 = {
        EMB: _t(8, 4),
        f"{LP}.0.self_attn.q_proj.weight": _t(4, 4),
        f"{LP}.0.self_attn.q_proj.bias": _t(4),
        f"{LP}.0.input_layernorm.weight": _t(4),
        f"{LP}.1.self_attn.q_proj.weight": _t(4, 4),   # layer 1 part A
        f"{LP}.1.self_attn.q_proj.bias": _t(4),        # layer 1 part A
    }
    shard2 = {
        f"{LP}.1.mlp.down_proj.weight": _t(4, 8),      # layer 1 part B (straddle)
        f"{LP}.1.post_attention_layernorm.weight": _t(4),  # layer 1 part B
        f"{LP}.2.self_attn.k_proj.weight": _t(2, 4),
        f"{LP}.2.mlp.up_proj.weight": _t(8, 4),
        NORM: _t(4),
        LMH: _t(8, 4),
    }
    s1, s2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    save_file(shard1, str(d / s1))
    save_file(shard2, str(d / s2))
    weight_map = {**{k: s1 for k in shard1}, **{k: s2 for k in shard2}}
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return d


def _build_single_file_snapshot(tmp_path) -> Path:
    d = tmp_path / "single"
    d.mkdir()
    save_file({EMB: _t(8, 4), f"{LP}.0.self_attn.q_proj.weight": _t(4, 4), NORM: _t(4)},
              str(d / "model.safetensors"))
    return d  # NO index.json


# ── weight_map resolution ────────────────────────────────────────────────────

def test_resolve_weight_map_sharded(tmp_path):
    d = _build_sharded_snapshot(tmp_path)
    wm = _resolve_weight_map(str(d))
    assert wm[EMB] == "model-00001-of-00002.safetensors"
    assert wm[f"{LP}.1.mlp.down_proj.weight"] == "model-00002-of-00002.safetensors"


def test_resolve_weight_map_single_file_no_index(tmp_path):
    """CO1 — a model small enough to ship a lone model.safetensors has NO
    index.json; every key must map to that one file."""
    d = _build_single_file_snapshot(tmp_path)
    wm = _resolve_weight_map(str(d))
    assert set(wm.values()) == {"model.safetensors"}
    assert EMB in wm and NORM in wm


# ── index parsing / remap ────────────────────────────────────────────────────

def test_layer_index_of():
    assert _layer_index_of(f"{LP}.7.self_attn.q_proj.weight", LP) == 7
    assert _layer_index_of(EMB, LP) is None
    assert _layer_index_of(NORM, LP) is None


def test_remap_layer_key():
    assert _remap_layer_key(f"{LP}.14.mlp.up_proj.weight", LP, 14) == f"{LP}.0.mlp.up_proj.weight"
    assert _remap_layer_key(f"{LP}.15.self_attn.k_proj.bias", LP, 14) == f"{LP}.1.self_attn.k_proj.bias"
    # non-layer keys unchanged
    assert _remap_layer_key(NORM, LP, 14) == NORM
    assert _remap_layer_key(EMB, LP, 14) == EMB


# ── owned-key resolution ─────────────────────────────────────────────────────

def _all_keys(tmp_path):
    return set(_resolve_weight_map(str(_build_sharded_snapshot(tmp_path))).keys())


def test_owned_keys_middle_stage_layers_only(tmp_path):
    keys = _all_keys(tmp_path)
    owned = _resolve_slice_owned_keys(
        keys, layers_prefix=LP, norm_key=NORM, lm_head_key=LMH, embed_key=EMB,
        start=1, end=2, is_final_stage=False, tie_word_embeddings=False,
    )
    # exactly layer 1's keys (both straddled halves), nothing else
    assert owned == {
        f"{LP}.1.self_attn.q_proj.weight", f"{LP}.1.self_attn.q_proj.bias",
        f"{LP}.1.mlp.down_proj.weight", f"{LP}.1.post_attention_layernorm.weight",
    }
    assert NORM not in owned and LMH not in owned and EMB not in owned


def test_owned_keys_final_untied_includes_norm_and_lm_head(tmp_path):
    keys = _all_keys(tmp_path)
    owned = _resolve_slice_owned_keys(
        keys, layers_prefix=LP, norm_key=NORM, lm_head_key=LMH, embed_key=EMB,
        start=2, end=3, is_final_stage=True, tie_word_embeddings=False,
    )
    assert NORM in owned and LMH in owned
    assert EMB not in owned  # untied → lm_head, not embed
    assert f"{LP}.2.self_attn.k_proj.weight" in owned


def test_owned_keys_final_tied_picks_embed_not_lm_head(tmp_path):
    keys = _all_keys(tmp_path)
    owned = _resolve_slice_owned_keys(
        keys, layers_prefix=LP, norm_key=NORM, lm_head_key=LMH, embed_key=EMB,
        start=2, end=3, is_final_stage=True, tie_word_embeddings=True,
    )
    assert NORM in owned and EMB in owned
    assert LMH not in owned  # tied → load embed_tokens; tie_weights wires lm_head


# ── straddling-shard read + relative remap ───────────────────────────────────

def test_read_owned_state_dict_straddles_shards_and_remaps(tmp_path):
    d = _build_sharded_snapshot(tmp_path)
    wm = _resolve_weight_map(str(d))
    owned = _resolve_slice_owned_keys(
        set(wm), layers_prefix=LP, norm_key=NORM, lm_head_key=LMH, embed_key=EMB,
        start=1, end=2, is_final_stage=False, tie_word_embeddings=False,
    )
    sd = _read_owned_state_dict(str(d), wm, owned, layers_prefix=LP, start=1, dtype="float32")
    # layer 1 → relative layer 0; tensors from BOTH shards present
    assert f"{LP}.0.self_attn.q_proj.weight" in sd       # from shard1
    assert f"{LP}.0.mlp.down_proj.weight" in sd           # from shard2 (straddle)
    assert f"{LP}.0.post_attention_layernorm.weight" in sd
    # no original absolute-index keys leaked through
    assert not any(".1." in k for k in sd)
    assert sd[f"{LP}.0.self_attn.q_proj.weight"].shape == (4, 4)


def test_read_owned_state_dict_dtype_cast(tmp_path):
    d = _build_sharded_snapshot(tmp_path)
    wm = _resolve_weight_map(str(d))
    owned = _resolve_slice_owned_keys(
        set(wm), layers_prefix=LP, norm_key=NORM, lm_head_key=LMH, embed_key=EMB,
        start=0, end=1, is_final_stage=False, tie_word_embeddings=False,
    )
    sd = _read_owned_state_dict(str(d), wm, owned, layers_prefix=LP, start=0, dtype="float16")
    assert all(v.dtype == torch.float16 for v in sd.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
