"""Sprint 1231 — chat template on the parallax STREAMING encode path.

sp1230 applied the chat template on the unary parallax encoder; this completes
coverage for the streaming path (autoregressive_runner.execute_chain_streaming),
which drives the chat-style streaming UX (prsm_inference MCP tool). The streaming
path consumes a flat List[int] of token ids (len() for prompt-skip + a manual
tensor wrap), so it needs a list-returning encoder distinct from the unary
tensor one. Single source of truth for the gating + encode lives in
local_inference (shared by both paths).
"""

import pytest

from prsm.compute.inference.local_inference import (
    encode_ids_for_generation,
    parallax_chat_template_enabled,
    should_use_chat_template,
)
from prsm.node.chain_executor_adapters import _parallax_chat_template_enabled


class _FakeTok:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.applied_msgs = None
        self.encoded = None

    def apply_chat_template(self, msgs, add_generation_prompt):
        self.applied_msgs = msgs
        assert add_generation_prompt is True
        return [101, 102, 103]  # token-id list (no return_tensors)

    def encode(self, prompt):
        self.encoded = prompt
        return [7, 8]


# ── shared parallax flag (single source) ────────────────────────────

def test_parallax_flag_default_on(monkeypatch):
    monkeypatch.delenv("PRSM_PARALLAX_CHAT_TEMPLATE", raising=False)
    assert parallax_chat_template_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF"])
def test_parallax_flag_disabled(monkeypatch, val):
    monkeypatch.setenv("PRSM_PARALLAX_CHAT_TEMPLATE", val)
    assert parallax_chat_template_enabled() is False


def test_chain_executor_helper_delegates_to_single_source(monkeypatch):
    """sp1230's _parallax_chat_template_enabled must reflect the shared flag."""
    monkeypatch.setenv("PRSM_PARALLAX_CHAT_TEMPLATE", "off")
    assert _parallax_chat_template_enabled() is False
    monkeypatch.setenv("PRSM_PARALLAX_CHAT_TEMPLATE", "1")
    assert _parallax_chat_template_enabled() is True


# ── list-returning encoder ──────────────────────────────────────────

def test_encode_ids_instruct_applies_template_returns_list():
    tok = _FakeTok(chat_template="{{ messages }}")
    use_ct = should_use_chat_template(tok, enabled=True)
    ids = encode_ids_for_generation(tok, "hi there", use_chat_template=use_ct)
    assert tok.applied_msgs == [{"role": "user", "content": "hi there"}]
    assert ids == [101, 102, 103]
    assert isinstance(ids, list)
    assert tok.encoded is None  # raw encode NOT used


def test_encode_ids_base_model_uses_raw_encode_list():
    tok = _FakeTok(chat_template=None)
    use_ct = should_use_chat_template(tok, enabled=True)
    ids = encode_ids_for_generation(tok, "plain", use_chat_template=use_ct)
    assert tok.applied_msgs is None
    assert ids == [7, 8]
    assert tok.encoded == "plain"


def test_encode_ids_disabled_uses_raw_even_for_instruct():
    tok = _FakeTok(chat_template="{{ messages }}")
    use_ct = should_use_chat_template(tok, enabled=False)  # disabled
    ids = encode_ids_for_generation(tok, "x", use_chat_template=use_ct)
    assert tok.applied_msgs is None
    assert ids == [7, 8]


def test_encode_ids_returns_len_able_for_prompt_skip():
    """The streaming runner calls len(input_ids) for prompt-token skip; the
    result must be a sized sequence in both paths."""
    instruct = encode_ids_for_generation(
        _FakeTok(chat_template="t"), "p", use_chat_template=True)
    base = encode_ids_for_generation(
        _FakeTok(chat_template=None), "p", use_chat_template=False)
    assert len(instruct) == 3 and len(base) == 2


class _DictTok(_FakeTok):
    def apply_chat_template(self, msgs, add_generation_prompt):
        self.applied_msgs = msgs
        return {"input_ids": [201, 202]}  # BatchEncoding-like


def test_encode_ids_normalizes_dict_to_input_ids_list():
    tok = _DictTok(chat_template="t")
    ids = encode_ids_for_generation(tok, "p", use_chat_template=True)
    assert ids == [201, 202]
