"""Sprint 614 (Phase 2F-5e) — build_hf_prompt_encoder factory.

Client-side helper. The RpcChainExecutor factory takes an optional
``prompt_encoder: Callable[[str], np.ndarray]`` that converts the
raw prompt into the first stage's activation tensor. The default
``utf8_prompt_encoder`` encodes raw UTF-8 bytes — not real
embeddings; just useful for wire-format tests.

Sprint 614 ships ``build_hf_prompt_encoder(model_id, device)`` —
returns a closure that:
  1. Lazy-loads HuggingFace AutoTokenizer + AutoModelForCausalLM
     (cached on first call)
  2. Tokenizes the prompt: tokenizer.encode(prompt, return_tensors="pt")
  3. Looks up embeddings: model.get_input_embeddings()(token_ids)
  4. Returns embeddings as a numpy ndarray with shape
     [batch, seq_len, hidden_dim]

Sprint 615 wires it into _build_chain_executor's rpc kind so
operators can opt in via env var.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _install_fake_tokenizer_and_model():
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = MagicMock()

    fake_embed_layer = MagicMock()
    fake_embed_output = MagicMock()
    fake_embed_output.cpu = MagicMock()
    fake_embed_output.cpu.return_value.numpy = MagicMock(
        return_value=np.zeros((1, 4, 16), dtype=np.float32),
    )
    fake_embed_layer.return_value = fake_embed_output

    fake_model = MagicMock()
    fake_model.get_input_embeddings = MagicMock(return_value=fake_embed_layer)
    fake_model.eval = MagicMock(return_value=fake_model)
    fake_model.to = MagicMock(return_value=fake_model)
    # sp948 — model a LLaMA/rotary-style architecture (NO learned positional-
    # embedding matrix): no `.wpe` and not GPT-NeoX. A bare MagicMock auto-
    # vivifies `transformer.wpe`, sending production down the sprint-688
    # position-embedding ADD branch where `embeddings + pos_embeds` yields an
    # unconfigured mock instead of the pre-set ndarray. Pinning the wte-only
    # path is exactly what this sprint-614 test validates.
    fake_model.transformer = MagicMock(spec=[])  # no `.wpe` attribute
    del fake_model.gpt_neox                       # hasattr(...,'gpt_neox') -> False

    fake_tf = MagicMock()
    fake_tf.AutoTokenizer = MagicMock()
    fake_tf.AutoTokenizer.from_pretrained = MagicMock(
        return_value=fake_tokenizer,
    )
    fake_tf.AutoModelForCausalLM = MagicMock()
    fake_tf.AutoModelForCausalLM.from_pretrained = MagicMock(
        return_value=fake_model,
    )

    fake_torch = MagicMock()
    fake_torch.float32 = "f32"
    fake_torch.no_grad = MagicMock()
    fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)

    return fake_tf, fake_torch, fake_tokenizer, fake_model, fake_embed_layer


def test_factory_function_exists():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "build_hf_prompt_encoder")


def test_factory_returns_callable():
    from prsm.node.chain_executor_adapters import build_hf_prompt_encoder
    fake_tf, fake_torch, *_ = _install_fake_tokenizer_and_model()
    with patch.dict(sys.modules, {"transformers": fake_tf, "torch": fake_torch}):
        encoder = build_hf_prompt_encoder(model_id="fake/model")
    assert callable(encoder)


def test_encoder_returns_numpy_array_on_call():
    from prsm.node.chain_executor_adapters import build_hf_prompt_encoder
    fake_tf, fake_torch, fake_tokenizer, fake_model, fake_embed = (
        _install_fake_tokenizer_and_model()
    )
    with patch.dict(sys.modules, {"transformers": fake_tf, "torch": fake_torch}):
        encoder = build_hf_prompt_encoder(model_id="fake/model")
        result = encoder("hello world")
    assert isinstance(result, np.ndarray)
    fake_tokenizer.encode.assert_called_once_with(
        "hello world", return_tensors="pt",
    )
    fake_embed.assert_called_once()


def test_encoder_lazy_loads_tokenizer_and_model():
    """Tokenizer + model NOT loaded until first encoder() call."""
    from prsm.node.chain_executor_adapters import build_hf_prompt_encoder
    fake_tf, fake_torch, *_ = _install_fake_tokenizer_and_model()
    with patch.dict(sys.modules, {"transformers": fake_tf, "torch": fake_torch}):
        encoder = build_hf_prompt_encoder(model_id="fake/model")
        # Build encoder but DON'T call it
        fake_tf.AutoTokenizer.from_pretrained.assert_not_called()
        fake_tf.AutoModelForCausalLM.from_pretrained.assert_not_called()
        # First call triggers load
        encoder("first call")
    fake_tf.AutoTokenizer.from_pretrained.assert_called_once_with(
        "fake/model",
    )


def test_encoder_caches_loaded_artifacts():
    """Subsequent encoder() calls reuse the loaded model+tokenizer."""
    from prsm.node.chain_executor_adapters import build_hf_prompt_encoder
    fake_tf, fake_torch, *_ = _install_fake_tokenizer_and_model()
    with patch.dict(sys.modules, {"transformers": fake_tf, "torch": fake_torch}):
        encoder = build_hf_prompt_encoder(model_id="fake/model")
        encoder("a")
        encoder("b")
        encoder("c")
    # Load count must be 1 (cached)
    assert fake_tf.AutoTokenizer.from_pretrained.call_count == 1
    assert fake_tf.AutoModelForCausalLM.from_pretrained.call_count == 1


def test_encoder_raises_stage_execution_error_on_load_failure():
    from prsm.node.chain_executor_adapters import (
        build_hf_prompt_encoder, StageExecutionError,
    )
    fake_tf = MagicMock()
    fake_tf.AutoTokenizer = MagicMock()
    fake_tf.AutoTokenizer.from_pretrained = MagicMock(
        side_effect=RuntimeError("model 404 on HF Hub"),
    )
    fake_torch = MagicMock()
    fake_torch.no_grad = MagicMock()
    fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)
    with patch.dict(sys.modules, {"transformers": fake_tf, "torch": fake_torch}):
        encoder = build_hf_prompt_encoder(model_id="nonexistent/model")
        with pytest.raises(StageExecutionError, match="404"):
            encoder("test prompt")
