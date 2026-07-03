"""Sprint 1371 — static-file GPU pool provider.

The mainnet multi-stage settlement canary (2026-07-03) was blocked because the head's parallax pool
had only itself (`gpu_count == 1`): a `/peers/connect` transport link doesn't propagate the peer's
hardware_profile into the pool (that's libp2p-only). This provider lets an operator PIN the pool from
a JSON file, so a controlled deployment can force a real cross-node split — both nodes listed, same
region, `layer_capacity` below the model's layer count.
"""
from __future__ import annotations

import json

from prsm.node.static_file_pool_provider import build_static_file_pool_provider

_A = "a" * 32
_B = "b" * 32


def _write(tmp_path, obj):
    f = tmp_path / "pool.json"
    f.write_text(json.dumps(obj))
    return str(f)


def test_pins_two_nodes_same_region_low_capacity(tmp_path, monkeypatch):
    # THE fix scenario: two nodes, same region, capacity < model layers → planner can split.
    path = _write(tmp_path, {"gpus": [
        {"node_id": _A, "region": "canary", "layer_capacity": 6, "memory_gb": 16.0},
        {"node_id": _B, "region": "canary", "layer_capacity": 6, "memory_gb": 16.0},
    ]})
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_FILE", path)
    gpus = build_static_file_pool_provider()()
    assert {g.node_id for g in gpus} == {_A, _B}
    assert all(g.layer_capacity == 6 for g in gpus)
    assert all(g.region == "canary" for g in gpus)          # same region → one allocation group


def test_defaults_optional_fields(tmp_path, monkeypatch):
    path = _write(tmp_path, [{"node_id": _A, "region": "r", "layer_capacity": 4, "memory_gb": 8.0}])
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_FILE", path)
    (g,) = build_static_file_pool_provider()()
    assert g.tflops_fp16 > 0 and g.memory_bandwidth_gbps > 0    # valid ParallaxGPU from minimal entry
    assert g.stake_amount == 0 and g.num_gpus == 1
    assert g.tier_attestation == "tier-none"


def test_accepts_bare_list(tmp_path, monkeypatch):
    path = _write(tmp_path, [{"node_id": _A, "region": "r", "layer_capacity": 3, "memory_gb": 4.0}])
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_FILE", path)
    assert len(build_static_file_pool_provider()()) == 1


def test_fail_soft_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_FILE", str(tmp_path / "nope.json"))
    assert build_static_file_pool_provider()() == []


def test_fail_soft_unset_env(monkeypatch):
    monkeypatch.delenv("PRSM_PARALLAX_GPU_POOL_FILE", raising=False)
    assert build_static_file_pool_provider()() == []


def test_fail_soft_bad_json(tmp_path, monkeypatch):
    f = tmp_path / "pool.json"
    f.write_text("{not json")
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_FILE", str(f))
    assert build_static_file_pool_provider()() == []


def test_skips_malformed_entry_keeps_valid(tmp_path, monkeypatch):
    path = _write(tmp_path, {"gpus": [
        {"node_id": _A, "region": "r", "layer_capacity": 6, "memory_gb": 16.0},   # good
        {"region": "r", "layer_capacity": 6, "memory_gb": 16.0},                   # no node_id
        {"node_id": _B, "region": "r", "layer_capacity": -1, "memory_gb": 16.0},   # bad capacity
    ]})
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_FILE", path)
    gpus = build_static_file_pool_provider()()
    assert [g.node_id for g in gpus] == [_A]                 # only the valid one survives


def test_rereads_fresh_each_call(tmp_path, monkeypatch):
    f = tmp_path / "pool.json"
    f.write_text(json.dumps([{"node_id": _A, "region": "r", "layer_capacity": 6, "memory_gb": 8.0}]))
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_FILE", str(f))
    prov = build_static_file_pool_provider()
    assert len(prov()) == 1
    f.write_text(json.dumps([
        {"node_id": _A, "region": "r", "layer_capacity": 6, "memory_gb": 8.0},
        {"node_id": _B, "region": "r", "layer_capacity": 6, "memory_gb": 8.0}]))
    assert len(prov()) == 2                                  # edit visible without rebuilding provider


def test_provider_pool_yields_a_real_2stage_split(tmp_path, monkeypatch):
    """END-TO-END: the pinned pool actually makes the allocator SHARD a model across both nodes —
    the exact thing the mainnet canary needed. Two nodes, same region, memory_gb tuned so the model
    can't fit one node → a 2-stage pipeline (contiguous layer ranges covering the whole model),
    not two replicas and not InsufficientCapacity."""
    from prsm.compute.parallax_scheduling.model_info import ModelInfo
    from prsm.compute.parallax_scheduling.prsm_types import allocate_across_regions

    path = _write(tmp_path, {"gpus": [
        {"node_id": _A, "region": "canary", "layer_capacity": 6, "memory_gb": 0.5,
         "stake_amount": 10 ** 18},
        {"node_id": _B, "region": "canary", "layer_capacity": 6, "memory_gb": 0.5,
         "stake_amount": 10 ** 18},
    ]})
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_FILE", path)
    gpus = build_static_file_pool_provider()()

    # gpt2: 12 layers — with memory_gb=0.5 the model can't sit on one node, so it must split.
    gpt2 = ModelInfo(head_size=64, hidden_dim=768, intermediate_dim=3072, num_attention_heads=12,
                     num_kv_heads=12, vocab_size=50257, num_layers=12, param_bytes_per_element=4)
    pipelines = allocate_across_regions(gpus, gpt2).all_pipelines()

    split = [p for p in pipelines if len(p.stages) >= 2]
    assert split, f"expected a 2+-stage pipeline, got {[(p.stages, p.layer_ranges) for p in pipelines]}"
    p = split[0]
    assert len(set(p.stages)) == 2                          # two DISTINCT nodes in the pipeline
    # contiguous ranges covering the whole model (0..12) with no gap/overlap
    ranges = sorted(p.layer_ranges)
    assert ranges[0][0] == 0 and ranges[-1][1] == gpt2.num_layers
    for (_, end_prev), (start_next, _) in zip(ranges, ranges[1:]):
        assert end_prev == start_next                       # no gap, no overlap → conservation


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
