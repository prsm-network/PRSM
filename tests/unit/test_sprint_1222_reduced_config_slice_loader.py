"""Sprint 1222 (Brick 3/6) — reduced-config slice loader.

build_reduced_config_slice_model builds a from_config skeleton with exactly the
[start,end) decoder layers and materializes ONLY this stage's owned weights from
the on-disk safetensors. The decisive test is the §7 BIT-EXACT gate: a
slice-loaded middle/final stage, driven through the sp1216 runner convention
(model-level rotary position_embeddings + position_ids, no_grad), must produce
output torch.equal to the FULL model's layers[start:end] — at fp32 on CPU.
Also covers the adversarial-review fixes: C2 (layer_types truncation so
from_config doesn't ValueError on v5), C3 (tie_weights → lm_head IS embed,
fail-loud assertion runs after), M1 (middle/stage0 embed deleted), and the
fail-loud silent-corruption guard (off-by-one / dropped owned key → raise).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
_tf = pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import prsm.node.chain_executor_adapters as cea  # noqa: E402
from prsm.node.chain_executor_adapters import (  # noqa: E402
    StageExecutionError,
    build_reduced_config_slice_model,
)


def _save_full_model(tmp_path, *, tie: bool):
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=tie,
    )
    torch.manual_seed(123)
    full = Qwen2ForCausalLM(cfg).eval()
    d = tmp_path / ("tied" if tie else "untied")
    full.save_pretrained(str(d))
    return full, str(d)


def _forward_through(model, layers, hidden):
    """Mimic run_layer_range's forward exactly: rotary position_embeddings from
    the MODEL-LEVEL rotary_emb (sp1216) + position_ids=arange, no_grad."""
    S = hidden.shape[-2]
    pos = torch.arange(S).unsqueeze(0)
    pe = model.model.rotary_emb(hidden, pos)
    with torch.no_grad():
        for layer in layers:
            out = layer(hidden, position_ids=pos, position_embeddings=pe)
            hidden = out[0] if isinstance(out, tuple) else out
    return hidden


# ── skeleton shape + rotary present + C2 ─────────────────────────────────────

def test_skeleton_has_slice_layers_and_rotary(tmp_path):
    _full, d = _save_full_model(tmp_path, tie=False)
    sliced, depth = build_reduced_config_slice_model(
        d, start=1, end=3, is_final_stage=False, device="cpu", dtype="float32",
    )
    assert depth == 4
    assert len(sliced.model.layers) == 2          # C2 didn't crash, n_slice=2
    assert sliced.model.rotary_emb is not None     # sp1216 rotary auto-built
    # M1 — middle stage dropped the full-vocab modules
    assert sliced.model.embed_tokens is None
    assert sliced.lm_head is None


# ── the §7 bit-exact gate ────────────────────────────────────────────────────

def test_middle_slice_forward_bit_exact_vs_full(tmp_path):
    full, d = _save_full_model(tmp_path, tie=False)
    sliced, _ = build_reduced_config_slice_model(
        d, start=1, end=3, is_final_stage=False, device="cpu", dtype="float32",
    )
    torch.manual_seed(7)
    hidden = torch.randn(1, 5, 32)
    out_full = _forward_through(full, list(full.model.layers)[1:3], hidden.clone())
    out_slice = _forward_through(sliced, list(sliced.model.layers)[0:2], hidden.clone())
    assert torch.equal(out_full, out_slice), (
        "sliced middle stage diverged from the full model — §7 chain would break"
    )


def test_final_slice_norm_and_lm_head_bit_exact(tmp_path):
    full, d = _save_full_model(tmp_path, tie=False)
    sliced, _ = build_reduced_config_slice_model(
        d, start=2, end=4, is_final_stage=True, device="cpu", dtype="float32",
    )
    assert sliced.lm_head is not None and sliced.model.norm is not None
    torch.manual_seed(9)
    hidden = torch.randn(1, 4, 32)

    def _final(model, layers):
        h = _forward_through(model, layers, hidden.clone())
        with torch.no_grad():
            h = model.model.norm(h)
            return model.lm_head(h)

    logits_full = _final(full, list(full.model.layers)[2:4])
    logits_slice = _final(sliced, list(sliced.model.layers)[0:2])
    assert torch.equal(logits_full, logits_slice)


# ── C3: tied head ────────────────────────────────────────────────────────────

def test_tied_final_lm_head_is_embed_after_tie(tmp_path):
    _full, d = _save_full_model(tmp_path, tie=True)
    sliced, _ = build_reduced_config_slice_model(
        d, start=2, end=4, is_final_stage=True, device="cpu", dtype="float32",
    )
    # tied: lm_head.weight must BE embed_tokens.weight (tie_weights wired it),
    # and the assertion must NOT have false-failed (we got here = no raise).
    assert sliced.lm_head.weight is sliced.model.embed_tokens.weight


# ── fail-loud silent-corruption guard ────────────────────────────────────────

def test_unexpected_key_raises(tmp_path, monkeypatch):
    _full, d = _save_full_model(tmp_path, tie=False)
    real = cea._read_owned_state_dict

    def _inject_bogus(*a, **k):
        sd = real(*a, **k)
        sd["model.layers.99.bogus.weight"] = torch.zeros(1)  # not in skeleton
        return sd

    monkeypatch.setattr(cea, "_read_owned_state_dict", _inject_bogus)
    with pytest.raises(StageExecutionError, match="UNEXPECTED"):
        build_reduced_config_slice_model(
            d, start=1, end=3, is_final_stage=False, device="cpu", dtype="float32",
        )


def test_dropped_owned_key_raises(tmp_path, monkeypatch):
    _full, d = _save_full_model(tmp_path, tie=False)
    real = cea._read_owned_state_dict

    def _drop_one(*a, **k):
        sd = real(*a, **k)
        # drop a genuinely-owned layer weight → load reports it missing
        for key in list(sd):
            if key.endswith("self_attn.q_proj.weight"):
                del sd[key]
                break
        return sd

    monkeypatch.setattr(cea, "_read_owned_state_dict", _drop_one)
    with pytest.raises(StageExecutionError, match="silent-corruption|OWNED"):
        build_reduced_config_slice_model(
            d, start=1, end=3, is_final_stage=False, device="cpu", dtype="float32",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
