"""Sprint 1204 — real inference quality: GPU device + dtype + instruct chat template.

The mainnet proof returned distilgpt2 gibberish because the local executor was
CPU-only, float32, and raw-encoded the prompt (no chat template). These changes let
it serve a coherent INSTRUCT model on a GPU with deterministic greedy decoding
(§7-safe — quality comes from the model + template, not sampling):
  - resolve_device: cuda > mps > cpu (env override) → a Lambda A10 is used auto.
  - resolve_dtype: float16 on CUDA (a 7-8B model fits a 24GB A10), float32 else.
  - should_use_chat_template / encode_for_generation: wrap the prompt in the model's
    chat template for instruct models (coherent), raw-encode base models (gpt2).
All helpers are pure → testable without a GPU or a real model.
"""
from __future__ import annotations

import pytest

from prsm.compute.inference.local_inference import (
    LocalHuggingFaceChainExecutor,
    _chat_template_enabled,
    dtype_kwarg_for_transformers,
    encode_for_generation,
    resolve_device,
    resolve_dtype,
    should_use_chat_template,
)


# ── sp1206: transformers v4/v5 dtype kwarg ────────────────────────────────────────────

def test_dtype_kwarg_v5_uses_dtype():
    assert dtype_kwarg_for_transformers("5.12.1") == "dtype"
    assert dtype_kwarg_for_transformers("5.0.0") == "dtype"


def test_dtype_kwarg_v4_uses_torch_dtype():
    assert dtype_kwarg_for_transformers("4.35.0") == "torch_dtype"
    assert dtype_kwarg_for_transformers("4.56.0") == "torch_dtype"


def test_dtype_kwarg_bad_version_defaults_safe():
    assert dtype_kwarg_for_transformers("") == "torch_dtype"
    assert dtype_kwarg_for_transformers("weird") == "torch_dtype"


# ── device selection ──────────────────────────────────────────────────────────────────

def test_device_prefers_cuda_then_mps_then_cpu():
    assert resolve_device(None, cuda=True, mps=True) == "cuda"
    assert resolve_device(None, cuda=False, mps=True) == "mps"
    assert resolve_device(None, cuda=False, mps=False) == "cpu"


def test_device_explicit_env_wins():
    assert resolve_device("cpu", cuda=True, mps=False) == "cpu"
    assert resolve_device("CUDA", cuda=False, mps=False) == "cuda"  # case-insensitive


def test_device_invalid_env_falls_back_to_auto():
    assert resolve_device("gpu", cuda=True, mps=False) == "cuda"
    assert resolve_device("", cuda=False, mps=False) == "cpu"


# ── dtype selection ───────────────────────────────────────────────────────────────────

def test_dtype_float16_on_cuda_float32_elsewhere():
    assert resolve_dtype(None, "cuda") == "float16"
    assert resolve_dtype(None, "cpu") == "float32"
    assert resolve_dtype(None, "mps") == "float32"


def test_dtype_explicit_wins():
    assert resolve_dtype("bfloat16", "cuda") == "bfloat16"
    assert resolve_dtype("float32", "cuda") == "float32"
    assert resolve_dtype("garbage", "cuda") == "float16"  # invalid → auto


# ── chat template gating (the quality lever) ─────────────────────────────────────────

class _TokWithTemplate:
    chat_template = "{{ messages }}"

    def __init__(self):
        self.applied = None

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors,
                            return_dict=False):
        self.applied = {"messages": messages, "agp": add_generation_prompt,
                        "rt": return_tensors, "return_dict": return_dict}
        if return_dict:  # transformers 4.41+/v5: BatchEncoding-style dict
            return {"input_ids": "CHAT_IDS", "attention_mask": "MASK"}
        return "CHAT_IDS"

    def __call__(self, prompt, return_tensors):  # pragma: no cover
        raise AssertionError("instruct model must use the chat template, not raw encode")


class _TokOldTemplate:
    """Older transformers: apply_chat_template has no return_dict kwarg → TypeError on it
    → encode_for_generation falls back to the bare-tensor call."""
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        return "BARE_IDS"

    def __call__(self, prompt, return_tensors):  # pragma: no cover
        raise AssertionError("should use the chat template")


class _TokRaw:
    chat_template = None

    def __call__(self, prompt, return_tensors):
        return {"input_ids": "RAW_IDS:" + prompt}

    def apply_chat_template(self, *a, **k):  # pragma: no cover
        raise AssertionError("base model has no chat template")


def test_should_use_chat_template_only_for_instruct_and_when_enabled():
    assert should_use_chat_template(_TokWithTemplate(), enabled=True) is True
    assert should_use_chat_template(_TokWithTemplate(), enabled=False) is False  # opt-out
    assert should_use_chat_template(_TokRaw(), enabled=True) is False  # base model


def test_encode_applies_chat_template_for_instruct():
    tok = _TokWithTemplate()
    enc = encode_for_generation(tok, "Hello", use_chat_template=True)
    # sp1208: returns the BatchEncoding-style dict directly (input_ids + attention_mask),
    # NOT {"input_ids": <BatchEncoding>} which broke torch.ones_like.
    assert enc["input_ids"] == "CHAT_IDS"
    assert "attention_mask" in enc
    assert tok.applied["messages"] == [{"role": "user", "content": "Hello"}]
    assert tok.applied["agp"] is True  # add_generation_prompt
    assert tok.applied["return_dict"] is True


def test_encode_chat_template_old_transformers_wraps_bare_tensor():
    # sp1208: a transformers without the return_dict kwarg → TypeError → bare-tensor
    # fallback → wrapped under input_ids so the caller's ones_like gets a real tensor.
    enc = encode_for_generation(_TokOldTemplate(), "Hi", use_chat_template=True)
    assert enc == {"input_ids": "BARE_IDS"}


def test_encode_raw_for_base_model():
    enc = encode_for_generation(_TokRaw(), "Hello", use_chat_template=False)
    assert enc == {"input_ids": "RAW_IDS:Hello"}


# ── chat-template env gate ────────────────────────────────────────────────────────────

def test_chat_template_enabled_default_and_opt_out():
    assert _chat_template_enabled({}) is True
    assert _chat_template_enabled({"PRSM_LOCAL_INFERENCE_CHAT_TEMPLATE": "0"}) is False
    assert _chat_template_enabled({"PRSM_LOCAL_INFERENCE_CHAT_TEMPLATE": "false"}) is False
    assert _chat_template_enabled({"PRSM_LOCAL_INFERENCE_CHAT_TEMPLATE": "1"}) is True


# ── executor wiring (no model load) ──────────────────────────────────────────────────

def test_executor_defaults_cpu_and_reads_chat_template_env(monkeypatch):
    monkeypatch.delenv("PRSM_LOCAL_INFERENCE_CHAT_TEMPLATE", raising=False)
    ex = LocalHuggingFaceChainExecutor(model_id="distilgpt2", offline=True)
    assert ex._device == "cpu"  # before load
    assert ex._use_chat_template is True


def test_executor_chat_template_opt_out(monkeypatch):
    monkeypatch.setenv("PRSM_LOCAL_INFERENCE_CHAT_TEMPLATE", "0")
    ex = LocalHuggingFaceChainExecutor(model_id="distilgpt2", offline=True)
    assert ex._use_chat_template is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
