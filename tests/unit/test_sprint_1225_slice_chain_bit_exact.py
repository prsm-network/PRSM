"""Sprint 1225 (Brick 6/6) — chain-level bit-exact gate for the slice-load arc.

The per-stage bit-exactness is proven in sp1222/1223; this is the CAPSTONE
gate that composes the WHOLE multi-stage sliced pipeline and asserts it is
token- AND logit-identical to a single-node FULL-model forward, at fp32 on CPU.
It's the CI safety net + the go/no-go contract for the live GPU 7B run.

The live GPU validation (deferred, on the 2 A10s) — documented for the operator:
  1. Bit-exact middle-stage proof: full fp16 ref vs slice-loaded (14,21) of a
     28-layer 7B, seeded with the full model's post-layer-13 hidden, torch.equal.
  2. Full-chain token-identity at fp16 (this test, but on CUDA/fp16).
  3. 7B fits 14+14: nvidia-smi resident ≈ slice (~6.5GB/side), NOT ~14GB.
  4. End-to-end §7 signed chain across two A10s; output == single-node ref.
  5. Streaming guard: slice-load streaming request cleanly refused, no crash.
  6. Fallback safety: PRSM_PARALLAX_SLICE_LOAD unset → byte-identical full-load.
"""
from __future__ import annotations

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


def _save_full(tmp_path, *, tie):
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=tie,
    )
    torch.manual_seed(2024)
    full = Qwen2ForCausalLM(cfg).eval()
    d = tmp_path / "m"
    full.save_pretrained(str(d))
    return full, str(d)


def _embed(full, ids):
    with torch.no_grad():
        return full.get_input_embeddings()(ids).detach().numpy().astype(np.float32)


@pytest.mark.parametrize("tie", [False, True])
def test_sliced_two_stage_chain_matches_single_node_full(tmp_path, monkeypatch, tie):
    full, d = _save_full(tmp_path, tie=tie)
    ids = torch.tensor([[1, 5, 9, 13, 2]])
    embed = _embed(full, ids)

    # ── single-node FULL-model reference (slice OFF): one stage, all 4 layers,
    # is_final → norm + lm_head → logits.
    runner_full = HuggingFaceLayerSliceRunner(model_id=d, device="cpu")
    logits_full = runner_full.run_layer_range(
        model=None, layer_range=(0, 4), activation=embed,
        privacy_tier=None, is_final_stage=True,
    ).output

    # ── sliced TWO-STAGE chain (slice ON): stage0=(0,2) middle, stage1=(2,4)
    # final. Stage 1 consumes stage 0's activation — exactly the cross-host §7
    # handoff, but in-process.
    monkeypatch.setenv("PRSM_PARALLAX_SLICE_LOAD", "1")
    runner = HuggingFaceLayerSliceRunner(model_id=d, device="cpu")
    act0 = runner.run_layer_range(
        model=None, layer_range=(0, 2), activation=embed,
        privacy_tier=None, is_final_stage=False,
    ).output
    logits_slice = runner.run_layer_range(
        model=None, layer_range=(2, 4), activation=act0,
        privacy_tier=None, is_final_stage=True,
    ).output

    # both stages were genuinely slice-loaded (not silent full-load fallback)
    assert (0, 2, False) in runner._slice_models
    assert (2, 4, True) in runner._slice_models

    # logit-identical (fp32) → the sliced chain is bit-for-bit the full model
    assert logits_slice.shape == logits_full.shape == (1, 5, 64)
    assert np.array_equal(logits_slice, logits_full), (
        "sliced multi-stage chain diverged from the single-node full model — "
        "the §7 chain would not verify"
    )
    # and the greedy next-token (what /compute/inference returns) is identical
    assert int(logits_slice[0, -1].argmax()) == int(logits_full[0, -1].argmax())


def test_three_stage_chain_matches_full(tmp_path, monkeypatch):
    """A 1+1+2 split (every stage type: first, middle, final) still composes
    bit-exactly to the full model."""
    full, d = _save_full(tmp_path, tie=False)
    ids = torch.tensor([[3, 7, 11, 1]])
    embed = _embed(full, ids)

    runner_full = HuggingFaceLayerSliceRunner(model_id=d, device="cpu")
    logits_full = runner_full.run_layer_range(
        model=None, layer_range=(0, 4), activation=embed,
        privacy_tier=None, is_final_stage=True,
    ).output

    monkeypatch.setenv("PRSM_PARALLAX_SLICE_LOAD", "1")
    runner = HuggingFaceLayerSliceRunner(model_id=d, device="cpu")
    a = runner.run_layer_range(model=None, layer_range=(0, 1), activation=embed,
                               privacy_tier=None, is_final_stage=False).output
    a = runner.run_layer_range(model=None, layer_range=(1, 2), activation=a,
                               privacy_tier=None, is_final_stage=False).output
    logits = runner.run_layer_range(model=None, layer_range=(2, 4), activation=a,
                                    privacy_tier=None, is_final_stage=True).output
    assert np.array_equal(logits, logits_full)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
