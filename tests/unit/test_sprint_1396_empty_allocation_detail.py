"""Sprint 1396 — single-node parallax serving: actionable empty-allocation error.

A node whose memory-derived capacity can't hold the model produced a bare "AllocationResult has no
pipelines" (hit live on the seed node: memory_gb=0.8 → capacity 11 < gpt2's 12). Now the error states
the exact shortfall + concrete fixes. Single-node serving works when the node fits (memory_gb ≥ ~1GB
for gpt2) — this is the diagnosability fix for when it doesn't.
"""
import pytest

from prsm.compute.parallax_scheduling.model_info import ModelInfo
from prsm.compute.parallax_scheduling.prsm_request_router import (
    EmptyAllocationError,
    RequestRouter,
    _empty_allocation_detail,
)
from prsm.compute.parallax_scheduling.prsm_types import (
    ParallaxGPU,
    allocate_across_regions,
)

_GPT2 = ModelInfo(head_size=64, hidden_dim=768, intermediate_dim=3072, num_attention_heads=12,
                  num_kv_heads=12, vocab_size=50257, num_layers=12, param_bytes_per_element=4)


def _gpu(memory_gb):
    return ParallaxGPU(node_id="d4" * 16, region="default", layer_capacity=16, stake_amount=10 ** 18,
                       tflops_fp16=30.0, memory_gb=memory_gb, memory_bandwidth_gbps=50.0,
                       tier_attestation="tier-none")


def test_undersized_node_gives_actionable_shortfall():
    gpu = _gpu(0.8)                                    # capacity 11 < gpt2's 12
    alloc = allocate_across_regions([gpu], _GPT2)
    assert alloc.total_pipeline_count() == 0          # confirms the empty case
    with pytest.raises(EmptyAllocationError) as ei:
        RequestRouter(alloc, None, _GPT2, [gpu])
    msg = str(ei.value)
    assert "needs 12" in msg and "decoder layer" in msg     # the exact shortfall
    assert "PRSM_PARALLAX_MEMORY_GB_OVERRIDE" in msg         # actionable fixes
    assert "PRSM_INFERENCE_EXECUTOR=local" in msg


def test_fits_when_node_has_memory():
    # memory_gb >= ~1GB holds gpt2 → single-node parallax SERVES (no error).
    alloc = allocate_across_regions([_gpu(8.0)], _GPT2)
    assert alloc.total_pipeline_count() >= 1
    RequestRouter(alloc, None, _GPT2, [_gpu(8.0)])     # constructs fine


def test_detail_helper_empty_pool_and_failsoft():
    assert "pool is empty" in _empty_allocation_detail(_GPT2, [])
    # fail-soft: garbage gpus don't crash → falls back to the base message
    assert "no pipelines" in _empty_allocation_detail(_GPT2, [object()])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
