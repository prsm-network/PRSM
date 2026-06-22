"""Sprint 1220 (Brick 1 of the big-model slice-load arc) — process-shared HF
model cache + the C1 dtype-cast fix.

The layer-slice runner + the prompt encoder each from_pretrained'd the FULL
model separately (2-4 full copies on the head) AND reloaded per request, and
hardcoded fp32 (torch_dtype is IGNORED on transformers v5, so a 7B loaded fp32
and OOM'd a 24GB A10). This brick:
  - loads each (model_id, device, dtype) ONCE per process via _HF_MODEL_CACHE;
  - resolves dtype correctly (fp16 on CUDA → 7B fits 24GB; fp32 on CPU);
  - C1 (adversarial-verify FATAL, reproduced): the activation-boundary cast now
    matches the MODEL's dtype instead of hardcoded fp32 — feeding an fp32
    tensor into an fp16 nn.Linear raises "mat1 and mat2 must have the same
    dtype". The proven gpt2/CPU path stays fp32 (resolve_dtype → float32 off
    CUDA), so it is unaffected.
NO slicing yet — this is the de-risk that makes 7B multi-host RUN (1× fp16) and
yields a green fp16 reference to diff the slice path against.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import prsm.node.chain_executor_adapters as cea
from prsm.node.chain_executor_adapters import (
    HuggingFaceLayerSliceRunner,
    _load_hf_model_cached,
    _model_compute_dtype,
    _resolve_model_dtype,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    cea._HF_MODEL_CACHE.clear()
    yield
    cea._HF_MODEL_CACHE.clear()


class _FakeModel:
    def __init__(self, dtype):
        self._p = torch.zeros(1, dtype=dtype)

    def to(self, *a, **k):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([self._p])


def _patch_from_pretrained(monkeypatch, captured):
    import transformers

    def _fake(model_id, **kwargs):
        captured.append({"model_id": model_id, "kwargs": kwargs})
        dt = kwargs.get("dtype") or kwargs.get("torch_dtype") or torch.float32
        return _FakeModel(dt)

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(_fake),
    )


# ── dtype resolution ─────────────────────────────────────────────────────────

def test_resolve_model_dtype_cuda_vs_cpu():
    assert _resolve_model_dtype("cuda") == "float16"
    assert _resolve_model_dtype("cpu") == "float32"
    assert _resolve_model_dtype("mps") == "float32"


# ── process-shared cache (load once) ─────────────────────────────────────────

def test_cache_loads_once_per_key(monkeypatch):
    captured: list = []
    _patch_from_pretrained(monkeypatch, captured)

    a = _load_hf_model_cached("m", "cpu", "float32")
    b = _load_hf_model_cached("m", "cpu", "float32")
    assert a is b  # identity — one load
    assert len(captured) == 1

    # a different key reloads
    _load_hf_model_cached("m", "cuda", "float16")
    assert len(captured) == 2


def test_cuda_loads_fp16_cpu_loads_fp32(monkeypatch):
    captured: list = []
    _patch_from_pretrained(monkeypatch, captured)

    _load_hf_model_cached("m", "cuda", "float16")
    _load_hf_model_cached("m", "cpu", "float32")
    # the dtype kwarg value passed to from_pretrained (key is dtype on v5,
    # torch_dtype on v4 — assert on the VALUE, robust across versions)
    def _dt(entry):
        kw = entry["kwargs"]
        return kw.get("dtype") or kw.get("torch_dtype")
    assert _dt(captured[0]) == torch.float16
    assert _dt(captured[1]) == torch.float32


def test_runner_ensure_model_loaded_uses_cache(monkeypatch):
    captured: list = []
    _patch_from_pretrained(monkeypatch, captured)

    r1 = HuggingFaceLayerSliceRunner(model_id="m", device="cpu")
    r2 = HuggingFaceLayerSliceRunner(model_id="m", device="cpu")
    m1 = r1._ensure_model_loaded()
    m2 = r2._ensure_model_loaded()
    assert m1 is m2  # two runner instances share ONE cached model
    assert len(captured) == 1


# ── C1: activation cast matches model dtype (not hardcoded fp32) ─────────────

class _RecordingLayer:
    def __init__(self):
        self.dtypes = []

    def __call__(self, hidden, **kwargs):
        self.dtypes.append(hidden.dtype)
        return (hidden,)


class _FakeRotary:
    def __call__(self, x, position_ids):
        return ("COS", "SIN")


def _fp16_fake_model(n_layers):
    import types
    layers = [_RecordingLayer() for _ in range(n_layers)]
    inner = types.SimpleNamespace(
        layers=layers, rotary_emb=_FakeRotary(), norm=lambda h: h,
    )
    model = types.SimpleNamespace(model=inner, lm_head=lambda h: h)
    # give it fp16 parameters so _model_compute_dtype → float16
    model.parameters = lambda: iter([torch.zeros(1, dtype=torch.float16)])
    return model, layers


def test_model_compute_dtype_reads_params():
    m, _ = _fp16_fake_model(1)
    assert _model_compute_dtype(m) == torch.float16


def test_run_layer_range_casts_activation_to_model_dtype():
    model, layers = _fp16_fake_model(2)
    runner = HuggingFaceLayerSliceRunner(model_id="m", device="cpu")
    runner._hf_model = model  # bypass load

    activation = np.zeros((1, 3, 4), dtype=np.float32)  # fp32 input
    runner.run_layer_range(
        model=None, layer_range=(0, 2), activation=activation,
        privacy_tier=None, is_final_stage=False,
    )
    # C1: every layer must have received the activation in the MODEL's dtype
    # (fp16) — NOT the old hardcoded fp32 (which crashes a real fp16 Linear).
    for layer in layers:
        assert layer.dtypes == [torch.float16]


def test_incremental_casts_activation_to_model_dtype():
    model, layers = _fp16_fake_model(2)
    runner = HuggingFaceLayerSliceRunner(model_id="m", device="cpu")
    runner._hf_model = model

    activation = np.zeros((1, 3, 4), dtype=np.float32)
    runner.run_layer_range_incremental(
        model=None, layer_range=(0, 2), activation=activation,
        privacy_tier=None, is_final_stage=False, prev_kv_state=None,
    )
    for layer in layers:
        assert layer.dtypes == [torch.float16]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
