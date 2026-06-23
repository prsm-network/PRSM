"""Sprint 1230 — apply the chat template on the PARALLAX (distributed) encode path.

The single-node path got chat-template wrapping in sp1208 (local_inference.py), but
the parallax head's prompt encoder (build_hf_prompt_encoder._encode) called raw
tokenizer.encode(prompt). So instruct/chat models served across nodes — the very
Qwen-*-Instruct models proven at 7B/14B/72B — were tokenized completion-style, not
in their native instruction framing. This sprint reuses the tested sp1208 helpers
(should_use_chat_template / encode_for_generation), gated by PRSM_PARALLAX_CHAT_TEMPLATE
(default ON), and must be byte-identical for base models (no chat_template).
"""

import pytest

from prsm.node.chain_executor_adapters import (
    _encode_prompt_tokens,
    _parallax_chat_template_enabled,
)


class _FakeTok:
    """Minimal tokenizer stand-in exercising both encode paths."""

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.applied_msgs = None
        self.raw_prompt = None

    def apply_chat_template(self, msgs, add_generation_prompt, return_tensors,
                            return_dict=None):
        self.applied_msgs = msgs
        assert add_generation_prompt is True
        return {"input_ids": ["TEMPLATED", msgs[0]["content"]]}

    def __call__(self, prompt, return_tensors=None):
        self.raw_prompt = prompt
        return {"input_ids": ["RAW", prompt]}

    def encode(self, prompt, return_tensors=None):  # legacy path (must NOT be used now)
        raise AssertionError("raw .encode() should not be called by _encode_prompt_tokens")


# ── env flag resolution ─────────────────────────────────────────────

def test_parallax_chat_template_enabled_default_on(monkeypatch):
    monkeypatch.delenv("PRSM_PARALLAX_CHAT_TEMPLATE", raising=False)
    assert _parallax_chat_template_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "FALSE", "Off"])
def test_parallax_chat_template_disabled_values(monkeypatch, val):
    monkeypatch.setenv("PRSM_PARALLAX_CHAT_TEMPLATE", val)
    assert _parallax_chat_template_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", ""])
def test_parallax_chat_template_enabled_values(monkeypatch, val):
    monkeypatch.setenv("PRSM_PARALLAX_CHAT_TEMPLATE", val)
    assert _parallax_chat_template_enabled() is True


# ── encode behaviour ────────────────────────────────────────────────

def test_instruct_model_gets_chat_template(monkeypatch):
    monkeypatch.delenv("PRSM_PARALLAX_CHAT_TEMPLATE", raising=False)
    tok = _FakeTok(chat_template="{{ messages }}")  # instruct model
    out = _encode_prompt_tokens(tok, "The capital of France is")
    # template applied; messages wrap the user prompt
    assert tok.applied_msgs == [{"role": "user", "content": "The capital of France is"}]
    assert out == ["TEMPLATED", "The capital of France is"]
    assert tok.raw_prompt is None  # raw path not taken


def test_base_model_uses_raw_encode_byte_identical(monkeypatch):
    monkeypatch.delenv("PRSM_PARALLAX_CHAT_TEMPLATE", raising=False)
    tok = _FakeTok(chat_template=None)  # base model (e.g. gpt2)
    out = _encode_prompt_tokens(tok, "hello")
    # no template; raw encode path; input_ids extracted from the mapping
    assert tok.applied_msgs is None
    assert out == ["RAW", "hello"]


def test_disabled_flag_forces_raw_even_for_instruct(monkeypatch):
    monkeypatch.setenv("PRSM_PARALLAX_CHAT_TEMPLATE", "0")
    tok = _FakeTok(chat_template="{{ messages }}")  # instruct, but disabled
    out = _encode_prompt_tokens(tok, "hi")
    assert tok.applied_msgs is None       # template NOT applied
    assert out == ["RAW", "hi"]


def test_returns_input_ids_not_mapping(monkeypatch):
    """_encode_prompt_tokens must return the bare input_ids (what the embed
    lookup expects), never the BatchEncoding/dict wrapper."""
    monkeypatch.delenv("PRSM_PARALLAX_CHAT_TEMPLATE", raising=False)
    tok = _FakeTok(chat_template="{{ messages }}")
    out = _encode_prompt_tokens(tok, "x")
    assert not isinstance(out, dict)
