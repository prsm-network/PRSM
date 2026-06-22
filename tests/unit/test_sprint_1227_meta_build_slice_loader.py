"""Sprint 1227 — the slice loader builds its skeleton on the META device.

Found live on the 2-A10 14B bench: build_reduced_config_slice_model's
from_config random-initialized ~billions of params on CPU (minutes for a 14B's
24-layer slices) that load_state_dict immediately overwrote — and that
CPU-bound init starved the node's P2P heartbeat, dropping the worker mid-load
(pool churn → slice rebuild cascade that blocked the 14B chain). transformers
5.2 removed no_init_weights, so the loader now builds on `torch.device("meta")`
(instant, no init) and materializes only the owned weights, rebuilds the rotary
buffer, and asserts no meta tensor survives. Bit-exactness is gated by sp1222 /
sp1225 (still green); these tests lock the meta-build invariants.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from prsm.node.chain_executor_adapters import (  # noqa: E402
    build_reduced_config_slice_model,
)


def _save(tmp_path, *, tie=False):
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=tie,
    )
    torch.manual_seed(99)
    m = Qwen2ForCausalLM(cfg).eval()
    d = tmp_path / "m"
    m.save_pretrained(str(d))
    return str(d)


def _assert_no_meta(model):
    metas = [n for n, t in model.named_parameters() if t.is_meta]
    metas += [n for n, t in model.named_buffers() if t.is_meta]
    assert metas == [], f"meta tensors survived: {metas}"


def test_middle_stage_no_meta_and_rotary_real(tmp_path):
    d = _save(tmp_path)
    model, _ = build_reduced_config_slice_model(
        d, start=1, end=3, is_final_stage=False, device="cpu", dtype="float32",
    )
    _assert_no_meta(model)
    # rotary rebuilt on real device with a finite inv_freq (not meta)
    inv = model.model.rotary_emb.inv_freq
    assert not inv.is_meta and bool(torch.isfinite(inv).all())
    # the loaded layer weights are real (not meta), so a forward is possible
    assert not next(model.model.layers[0].parameters()).is_meta
    # non-final: norm deleted (would otherwise survive as an unloaded meta)
    assert model.model.norm is None


def test_final_untied_no_meta(tmp_path):
    d = _save(tmp_path, tie=False)
    model, _ = build_reduced_config_slice_model(
        d, start=2, end=4, is_final_stage=True, device="cpu", dtype="float32",
    )
    _assert_no_meta(model)
    assert model.model.norm is not None      # final stage keeps + loaded norm
    assert not model.model.norm.weight.is_meta
    assert not model.lm_head.weight.is_meta   # untied → loaded lm_head, real


def test_final_tied_no_meta(tmp_path):
    d = _save(tmp_path, tie=True)
    model, _ = build_reduced_config_slice_model(
        d, start=2, end=4, is_final_stage=True, device="cpu", dtype="float32",
    )
    _assert_no_meta(model)
    # tied: lm_head shares the (real, loaded) embed weight
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not model.lm_head.weight.is_meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
