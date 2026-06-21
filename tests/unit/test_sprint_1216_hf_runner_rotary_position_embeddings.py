"""Sprint 1216 — HuggingFaceLayerSliceRunner threads model-level rotary
position embeddings into each decoder layer.

The 2-A10 multi-host bench ran gpt2 (6+6) fine but failed on
Qwen2.5-1.5B-Instruct with:

    HuggingFaceLayerSliceRunner forward pass failed in
    'Qwen/Qwen2.5-1.5B-Instruct': TypeError: cannot unpack
    non-iterable NoneType object

Root cause: transformers v4.4x+/v5 moved rotary-embedding computation OUT
of each attention sub-layer UP to the model level. ``Qwen2Model.forward``
computes ``position_embeddings = self.rotary_emb(hidden, position_ids)``
ONCE and threads the ``(cos, sin)`` tuple to every decoder layer; the
layer then does ``cos, sin = position_embeddings`` internally. The
layer-slice runner drives the decoder layers DIRECTLY and only passed
``position_ids`` → ``position_embeddings`` defaulted to ``None`` → the
internal unpack raised. GPT-2 uses learned absolute positions (no
model-level ``rotary_emb``) so its layers never unpack and the old call
worked — which is exactly why gpt2 passed and Qwen did not.

Fix: ``_resolve_hf_rotary_position_embeddings`` computes the rotary
``(cos, sin)`` once at the model level (when ``.model.rotary_emb`` is
present) and ``run_layer_range`` / ``run_layer_range_incremental`` pass it
as ``position_embeddings`` to each layer. GPT-2 path is unchanged
(no rotary → position_ids-only call, back-compat).
"""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prsm.node.chain_executor_adapters import (  # noqa: E402
    HuggingFaceLayerSliceRunner,
    _resolve_hf_rotary_position_embeddings,
)


class _RecordingLayer:
    """Fake decoder layer that records the kwargs it was called with and
    returns the (unchanged) hidden state as a 1-tuple (LLaMA/Qwen layer
    return shape)."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, hidden, **kwargs):
        self.calls.append(kwargs)
        return (hidden,)


class _FakeRotary:
    """Fake model-level rotary embedding: returns a sentinel (cos, sin)
    and records what it was called with."""

    def __init__(self, sentinel) -> None:
        self.sentinel = sentinel
        self.called_with = None

    def __call__(self, x, position_ids):
        self.called_with = (x, position_ids)
        return self.sentinel


def _rotary_style_model(n_layers: int, sentinel):
    layers = [_RecordingLayer() for _ in range(n_layers)]
    rotary = _FakeRotary(sentinel)
    inner = types.SimpleNamespace(
        layers=layers,
        rotary_emb=rotary,
        norm=lambda h: h,  # identity final norm (unused unless final stage)
    )
    hf_model = types.SimpleNamespace(model=inner, lm_head=lambda h: h)
    return hf_model, layers, rotary


def _gpt2_style_model(n_layers: int):
    layers = [_RecordingLayer() for _ in range(n_layers)]
    inner = types.SimpleNamespace(h=layers, ln_f=lambda h: h)
    # NOTE: intentionally no ``.model`` attribute → not rotary.
    hf_model = types.SimpleNamespace(transformer=inner, lm_head=lambda h: h)
    return hf_model, layers


# ---------------------------------------------------------------------------
# helper unit tests
# ---------------------------------------------------------------------------

def test_helper_returns_rotary_tuple_for_rotary_model():
    sentinel = ("COS", "SIN")
    hf_model, _layers, rotary = _rotary_style_model(2, sentinel)
    hidden = torch.zeros((1, 3, 4))
    position_ids = torch.arange(3).unsqueeze(0)

    out = _resolve_hf_rotary_position_embeddings(
        hf_model, hidden, position_ids,
    )
    assert out == sentinel
    assert rotary.called_with is not None
    # rotary called with (hidden, position_ids)
    assert rotary.called_with[1] is position_ids


def test_helper_returns_none_for_gpt2_style_model():
    hf_model, _layers = _gpt2_style_model(2)
    hidden = torch.zeros((1, 3, 4))
    position_ids = torch.arange(3).unsqueeze(0)
    assert _resolve_hf_rotary_position_embeddings(
        hf_model, hidden, position_ids,
    ) is None


# ---------------------------------------------------------------------------
# run_layer_range integration (with fake model — no real weights)
# ---------------------------------------------------------------------------

def test_run_layer_range_passes_position_embeddings_for_rotary():
    sentinel = ("COS", "SIN")
    hf_model, layers, _rotary = _rotary_style_model(2, sentinel)
    runner = HuggingFaceLayerSliceRunner(model_id="Qwen/fake", device="cpu")
    runner._hf_model = hf_model  # bypass real load

    activation = np.zeros((1, 3, 4), dtype=np.float32)
    runner.run_layer_range(
        model=None, layer_range=(0, 2), activation=activation,
        privacy_tier=None, is_final_stage=False,
    )

    for layer in layers:
        assert len(layer.calls) == 1
        kw = layer.calls[0]
        assert "position_embeddings" in kw, (
            "rotary model layer must receive position_embeddings "
            "(the cos/sin tuple) or it raises 'cannot unpack NoneType'"
        )
        assert kw["position_embeddings"] == sentinel
        assert "position_ids" in kw


def test_run_layer_range_no_position_embeddings_for_gpt2():
    hf_model, layers = _gpt2_style_model(2)
    runner = HuggingFaceLayerSliceRunner(model_id="gpt2", device="cpu")
    runner._hf_model = hf_model

    activation = np.zeros((1, 3, 4), dtype=np.float32)
    runner.run_layer_range(
        model=None, layer_range=(0, 2), activation=activation,
        privacy_tier=None, is_final_stage=False,
    )

    for layer in layers:
        assert len(layer.calls) == 1
        kw = layer.calls[0]
        assert "position_embeddings" not in kw, (
            "gpt2 (absolute positions) must keep the back-compat "
            "position_ids-only call — no rotary tuple"
        )
        assert "position_ids" in kw


def test_run_layer_range_incremental_passes_position_embeddings_for_rotary():
    from transformers.cache_utils import DynamicCache  # noqa: F401

    sentinel = ("COS", "SIN")
    hf_model, layers, _rotary = _rotary_style_model(2, sentinel)
    runner = HuggingFaceLayerSliceRunner(model_id="Qwen/fake", device="cpu")
    runner._hf_model = hf_model

    activation = np.zeros((1, 3, 4), dtype=np.float32)
    runner.run_layer_range_incremental(
        model=None, layer_range=(0, 2), activation=activation,
        privacy_tier=None, is_final_stage=False, prev_kv_state=None,
    )

    for layer in layers:
        assert len(layer.calls) == 1
        kw = layer.calls[0]
        assert kw.get("position_embeddings") == sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
