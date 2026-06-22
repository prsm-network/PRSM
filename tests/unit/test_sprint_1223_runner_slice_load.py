"""Sprint 1223 (Brick 4/6) — wire the slice loader into the runner with
relative-index forward, the per-(start,end,is_final) cache, and the full-load
fallback.

Decisive properties:
  - PRSM_PARALLAX_SLICE_LOAD OFF → the runner uses the proven ABSOLUTE-index
    full-load path, byte-for-byte (the guard self._slice_loaded gates the one
    bit-exact-loop edit).
  - ON + a supported arch → loads only the [start,end) slice, iterates
    RELATIVELY (no IndexError for start>0), validates against the FULL depth,
    and the output is bit-exact vs the full model's layers[start:end].
  - any slice-load error / unsupported arch → graceful fallback to full-load.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import prsm.node.chain_executor_adapters as cea  # noqa: E402
from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("PRSM_PARALLAX_SLICE_LOAD", raising=False)
    cea._HF_MODEL_CACHE.clear()
    yield
    cea._HF_MODEL_CACHE.clear()


def _save_full(tmp_path, *, tie=False):
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=tie,
    )
    torch.manual_seed(321)
    full = Qwen2ForCausalLM(cfg).eval()
    d = tmp_path / "m"
    full.save_pretrained(str(d))
    return full, str(d)


def _forward_through(model, layers, hidden):
    S = hidden.shape[-2]
    pos = torch.arange(S).unsqueeze(0)
    pe = model.model.rotary_emb(hidden, pos)
    with torch.no_grad():
        for layer in layers:
            out = layer(hidden, position_ids=pos, position_embeddings=pe)
            hidden = out[0] if isinstance(out, tuple) else out
    return hidden


# ── recording fake for the slice-OFF absolute-path test ──────────────────────

class _RecLayer:
    def __init__(self, tag):
        self.tag = tag
        self.calls = 0

    def __call__(self, hidden, **kw):
        self.calls += 1
        return (hidden,)


def _fake_model(n):
    layers = [_RecLayer(i) for i in range(n)]
    inner = types.SimpleNamespace(layers=layers, rotary_emb=lambda x, p: ("C", "S"),
                                  norm=lambda h: h)
    m = types.SimpleNamespace(model=inner, lm_head=lambda h: h)
    m.parameters = lambda: iter([torch.zeros(1, dtype=torch.float32)])
    return m, layers


# ── slice OFF → absolute path, unchanged ─────────────────────────────────────

def test_slice_off_uses_absolute_indices(monkeypatch):
    # PRSM_PARALLAX_SLICE_LOAD unset (fixture) → full-load path, absolute idx.
    model, layers = _fake_model(4)
    r = HuggingFaceLayerSliceRunner(model_id="x", device="cpu")
    r._hf_model = model
    r.run_layer_range(model=None, layer_range=(2, 4),
                      activation=np.zeros((1, 3, 4), dtype=np.float32),
                      privacy_tier=None, is_final_stage=False)
    # absolute: layers[2] and layers[3] ran; layers[0],[1] did NOT
    assert layers[2].calls == 1 and layers[3].calls == 1
    assert layers[0].calls == 0 and layers[1].calls == 0


def test_resolve_model_for_range_off_reports_not_sliced(monkeypatch):
    model, layers = _fake_model(4)
    r = HuggingFaceLayerSliceRunner(model_id="x", device="cpu")
    r._hf_model = model
    m, lyrs, sliced, depth = r._resolve_model_for_range(1, 3, False)
    assert sliced is False and depth == 4 and m is model


# ── slice ON → relative index, bit-exact, no IndexError ──────────────────────

def test_slice_on_middle_relative_and_bit_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PARALLAX_SLICE_LOAD", "1")
    full, d = _save_full(tmp_path)
    r = HuggingFaceLayerSliceRunner(model_id=d, device="cpu")

    hidden = torch.randn(1, 5, 32)
    # range (2,4): a full-length-absolute loop would IndexError on a 2-layer
    # slice; relative indexing must iterate layers[0],[1].
    res = r.run_layer_range(
        model=None, layer_range=(2, 4),
        activation=hidden.numpy().astype(np.float32),
        privacy_tier=None, is_final_stage=False,
    )
    ref = _forward_through(full, list(full.model.layers)[2:4], hidden.clone())
    assert torch.equal(torch.from_numpy(res.output), ref)
    # the slice model was cached per range
    assert (2, 4, False) in r._slice_models


def test_slice_on_validation_uses_full_depth(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PARALLAX_SLICE_LOAD", "1")
    _full, d = _save_full(tmp_path)
    r = HuggingFaceLayerSliceRunner(model_id=d, device="cpu")
    # end=9 > full depth 4 → must raise (validated against full_depth, not the
    # slice length)
    with pytest.raises(cea.StageExecutionError, match="invalid layer_range|4 layers"):
        r.run_layer_range(model=None, layer_range=(0, 9),
                          activation=np.zeros((1, 3, 32), dtype=np.float32),
                          privacy_tier=None, is_final_stage=False)


def test_slice_on_final_stage_emits_logits(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PARALLAX_SLICE_LOAD", "1")
    _full, d = _save_full(tmp_path)
    r = HuggingFaceLayerSliceRunner(model_id=d, device="cpu")
    res = r.run_layer_range(model=None, layer_range=(2, 4),
                            activation=np.zeros((1, 4, 32), dtype=np.float32),
                            privacy_tier=None, is_final_stage=True)
    # final stage applies norm + lm_head → vocab logits (vocab_size 64)
    assert res.output.shape == (1, 4, 64)


def test_two_ranges_cached_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PARALLAX_SLICE_LOAD", "1")
    _full, d = _save_full(tmp_path)
    r = HuggingFaceLayerSliceRunner(model_id=d, device="cpu")
    act = np.zeros((1, 3, 32), dtype=np.float32)
    r.run_layer_range(model=None, layer_range=(0, 2), activation=act,
                      privacy_tier=None, is_final_stage=False)
    r.run_layer_range(model=None, layer_range=(2, 4), activation=act,
                      privacy_tier=None, is_final_stage=True)
    assert (0, 2, False) in r._slice_models
    assert (2, 4, True) in r._slice_models


# ── graceful fallback ────────────────────────────────────────────────────────

def test_slice_load_error_falls_back_to_full_load(monkeypatch):
    monkeypatch.setenv("PRSM_PARALLAX_SLICE_LOAD", "1")
    # bogus model_id → _resolve_snapshot_dir raises inside the slice path;
    # pre-set _hf_model so the full-load fallback returns it without loading.
    model, layers = _fake_model(4)
    r = HuggingFaceLayerSliceRunner(model_id="/nonexistent/model/path", device="cpu")
    r._hf_model = model
    m, lyrs, sliced, depth = r._resolve_model_for_range(1, 3, False)
    assert sliced is False and m is model  # fell back cleanly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
