"""Sprint 1224 (Brick 5/6) — embedding-only prompt encoder under slice-load.

A slice-load HEAD runs the prompt encoder AND its stage-0 layer slice; if the
encoder loads the FULL model (Brick 1) just to embed, a genuinely-too-big model
can't even fit the head. Brick 5 materializes ONLY model.embed_tokens as a
standalone nn.Embedding under PRSM_PARALLAX_SLICE_LOAD. The §7 chain INPUT must
not change: the embed-only output is bit-identical to the full model's
get_input_embeddings() (same weight; rotary adds no wpe). gpt2/wpe + any
embed-only failure fall back to the full-model embed.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import prsm.node.chain_executor_adapters as cea  # noqa: E402
from prsm.node.chain_executor_adapters import (  # noqa: E402
    StageExecutionError,
    _build_embedding_only,
    build_hf_prompt_encoder,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("PRSM_PARALLAX_SLICE_LOAD", raising=False)
    cea._HF_MODEL_CACHE.clear()
    yield
    cea._HF_MODEL_CACHE.clear()


def _save_full(tmp_path):
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(55)
    full = Qwen2ForCausalLM(cfg).eval()
    d = tmp_path / "m"
    full.save_pretrained(str(d))
    return full, str(d)


def test_embedding_only_bit_exact_vs_full(tmp_path):
    full, d = _save_full(tmp_path)
    emb = _build_embedding_only(d, "cpu", "float32")
    ids = torch.tensor([[1, 5, 9, 13, 2]])
    with torch.no_grad():
        got = emb(ids)
        ref = full.get_input_embeddings()(ids)
    assert torch.equal(got, ref)
    # only the embedding was materialized (vocab×hidden), nothing else
    assert emb.weight.shape == (64, 32)


def test_embedding_only_raises_without_embed_key(tmp_path, monkeypatch):
    _full, d = _save_full(tmp_path)
    real = cea._resolve_weight_map
    monkeypatch.setattr(
        cea, "_resolve_weight_map",
        lambda sd: {k: v for k, v in real(sd).items() if "embed_tokens" not in k},
    )
    with pytest.raises(StageExecutionError, match="embed-only|embed_tokens"):
        _build_embedding_only(d, "cpu", "float32")


def _fake_tokenizer(ids):
    tok = MagicMock()
    tok.encode = MagicMock(return_value=torch.tensor([ids]))
    return tok


def test_encoder_slice_load_embed_only_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PARALLAX_SLICE_LOAD", "1")
    full, d = _save_full(tmp_path)
    ids = [1, 5, 9, 13, 2]

    import transformers
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *a, **k: _fake_tokenizer(ids)),
    )
    enc = build_hf_prompt_encoder(model_id=d, device="cpu")
    out = enc("ignored — fake tokenizer returns fixed ids")

    # bit-identical to the full model's embedding lookup (chain INPUT unchanged)
    with torch.no_grad():
        ref = full.get_input_embeddings()(torch.tensor([ids])).cpu().numpy()
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, ref)


def test_encoder_slice_load_falls_back_to_full_on_embed_only_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PARALLAX_SLICE_LOAD", "1")
    full, d = _save_full(tmp_path)
    ids = [1, 5, 9, 13, 2]
    # force the embed-only path to fail → encoder must fall back to full-load
    monkeypatch.setattr(
        cea, "_build_embedding_only",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    import transformers
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *a, **k: _fake_tokenizer(ids)),
    )
    enc = build_hf_prompt_encoder(model_id=d, device="cpu")
    out = enc("x")
    with torch.no_grad():
        ref = full.get_input_embeddings()(torch.tensor([ids])).cpu().numpy()
    assert np.array_equal(out, ref)  # full-load embed → same result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
