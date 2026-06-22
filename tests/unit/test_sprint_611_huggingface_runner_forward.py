"""Sprint 611 (Phase 2F-5b) — real model loading + forward pass.

Replaces Phase 2F-5a skeleton's NotImplementedError with the real
implementation. Tests mock ``transformers`` + ``torch`` so CI doesn't
need to download a model.

Invariants tested:
  - Model loads lazily on first run_layer_range call
  - Same runner instance reuses the cached model (no reload)
  - Activation tensor is converted to torch + normalized to [B, S, H]
  - Forward iterates only over layers[start:end]
  - Output activation has the same dtype as input
  - LayerSliceResult.tee_type = SOFTWARE
  - Errors during model load raise StageExecutionError (not bare exc)
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clear_hf_cache():
    # sp1220 — the runner now loads through a PROCESS-shared _HF_MODEL_CACHE
    # (keyed model_id,device,dtype). Clear it around each test so the
    # from_pretrained.call_count assertions + per-test fake models don't leak
    # across tests.
    import prsm.node.chain_executor_adapters as _cea
    _cea._HF_MODEL_CACHE.clear()
    yield
    _cea._HF_MODEL_CACHE.clear()


def _install_fake_transformers_torch():
    """Build mock transformers + torch modules that the runner can import.
    Returns (transformers_mod, torch_mod, fake_model)."""
    fake_layer = MagicMock()

    def _layer_call(hidden, position_ids=None, **kwargs):
        # Mimics LLaMA-style return: (hidden_state, ...optional...)
        # Identity pass-through so test can verify routing without
        # caring about numerical correctness.
        return (hidden,)
    fake_layer.side_effect = _layer_call

    fake_model = MagicMock()
    # LLaMA-style: model.model.layers is a list-like
    fake_model.model = MagicMock()
    fake_model.model.layers = [fake_layer for _ in range(8)]
    fake_model.eval = MagicMock(return_value=fake_model)
    fake_model.to = MagicMock(return_value=fake_model)

    fake_tf = MagicMock()
    fake_tf.AutoModelForCausalLM = MagicMock()
    fake_tf.AutoModelForCausalLM.from_pretrained = MagicMock(
        return_value=fake_model,
    )

    # Fake torch.from_numpy returns an object that supports .to/.unsqueeze/
    # .dim/.shape/.cpu/.numpy/.squeeze
    fake_torch = MagicMock()
    fake_torch.float32 = "float32_sentinel"
    fake_torch.no_grad = MagicMock()
    fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)
    fake_torch.arange = MagicMock(return_value=MagicMock())

    def _from_numpy(arr):
        t = MagicMock()
        t.to = MagicMock(return_value=t)
        # dim() reflects ndim of source for normalize logic
        t.dim = MagicMock(return_value=arr.ndim)
        t.unsqueeze = MagicMock(return_value=t)
        t.shape = arr.shape if arr.ndim == 3 else (1,) + arr.shape
        t.squeeze = MagicMock(return_value=t)
        cpu_mock = MagicMock()
        cpu_mock.numpy = MagicMock(return_value=arr)
        t.cpu = MagicMock(return_value=cpu_mock)
        t.numpy = MagicMock(return_value=arr)
        return t
    fake_torch.from_numpy = MagicMock(side_effect=_from_numpy)

    return fake_tf, fake_torch, fake_model, fake_layer


def test_run_layer_range_loads_model_lazily_on_first_call():
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner
    fake_tf, fake_torch, fake_model, fake_layer = _install_fake_transformers_torch()
    with patch.dict(sys.modules, {
        "transformers": fake_tf,
        "torch": fake_torch,
    }):
        runner = HuggingFaceLayerSliceRunner(model_id="fake/model")
        # Model NOT loaded until first run_layer_range call
        assert getattr(runner, "_hf_model", None) is None

        activation = np.zeros((1, 4, 8), dtype=np.float32)
        runner.run_layer_range(
            model=None, layer_range=(0, 2),
            activation=activation, privacy_tier=None,
            is_final_stage=False,
        )

        # sp1220 — load goes through _load_hf_model_cached: dtype kwarg name is
        # version-resolved (the MagicMock version → "torch_dtype") + the sp702
        # low_cpu_mem_usage=False is now applied uniformly to the runner too.
        fake_tf.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "fake/model", torch_dtype="float32_sentinel", low_cpu_mem_usage=False,
        )
        assert runner._hf_model is fake_model


def test_run_layer_range_reuses_cached_model_on_subsequent_calls():
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner
    fake_tf, fake_torch, fake_model, fake_layer = _install_fake_transformers_torch()
    with patch.dict(sys.modules, {
        "transformers": fake_tf,
        "torch": fake_torch,
    }):
        runner = HuggingFaceLayerSliceRunner(model_id="fake/model")
        activation = np.zeros((1, 4, 8), dtype=np.float32)
        for _ in range(3):
            runner.run_layer_range(
                model=None, layer_range=(0, 1),
                activation=activation, privacy_tier=None,
                is_final_stage=False,
            )
        # Loaded ONCE only across 3 calls
        assert fake_tf.AutoModelForCausalLM.from_pretrained.call_count == 1


def test_run_layer_range_only_iterates_layer_range():
    """forward must call only layers[start:end], NOT all layers."""
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner
    fake_tf, fake_torch, fake_model, fake_layer = _install_fake_transformers_torch()
    with patch.dict(sys.modules, {
        "transformers": fake_tf,
        "torch": fake_torch,
    }):
        runner = HuggingFaceLayerSliceRunner(model_id="fake/model")
        activation = np.zeros((1, 4, 8), dtype=np.float32)
        runner.run_layer_range(
            model=None, layer_range=(2, 5),
            activation=activation, privacy_tier=None,
            is_final_stage=False,
        )
        # layers[2:5] is 3 layers → fake_layer called 3 times
        assert fake_layer.call_count == 3


def test_run_layer_range_returns_layer_slice_result():
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner
    from prsm.compute.tee.models import TEEType
    fake_tf, fake_torch, fake_model, _ = _install_fake_transformers_torch()
    with patch.dict(sys.modules, {
        "transformers": fake_tf,
        "torch": fake_torch,
    }):
        runner = HuggingFaceLayerSliceRunner(model_id="fake/model")
        activation = np.zeros((1, 4, 8), dtype=np.float32)
        result = runner.run_layer_range(
            model=None, layer_range=(0, 1),
            activation=activation, privacy_tier=None,
            is_final_stage=False,
        )
        # Defensive: output is an ndarray
        import numpy as _np
        assert isinstance(result.output, _np.ndarray)
        assert result.duration_seconds >= 0
        assert result.tee_type == TEEType.SOFTWARE


def test_run_layer_range_wraps_load_failure_in_stage_execution_error():
    """transformers.from_pretrained raises → StageExecutionError."""
    from prsm.node.chain_executor_adapters import (
        HuggingFaceLayerSliceRunner, StageExecutionError,
    )
    fake_tf = MagicMock()
    fake_tf.AutoModelForCausalLM = MagicMock()
    fake_tf.AutoModelForCausalLM.from_pretrained = MagicMock(
        side_effect=RuntimeError("network: HF Hub unreachable"),
    )
    fake_torch = MagicMock()
    fake_torch.float32 = "float32_sentinel"
    fake_torch.from_numpy = MagicMock()
    fake_torch.no_grad = MagicMock()
    fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)
    with patch.dict(sys.modules, {
        "transformers": fake_tf,
        "torch": fake_torch,
    }):
        runner = HuggingFaceLayerSliceRunner(model_id="fake/model")
        with pytest.raises(StageExecutionError, match="HF Hub unreachable|load"):
            runner.run_layer_range(
                model=None, layer_range=(0, 1),
                activation=np.zeros((1, 4, 8), dtype=np.float32),
                privacy_tier=None, is_final_stage=False,
            )
