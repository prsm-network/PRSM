"""Sprint 1228 — per-node shard-only fetch: a node downloads ONLY the
safetensors shards holding its assigned layers, not the whole model.

The slice loader already READS only its owned shards (sp1221); this closes the
FETCH side so a node DOWNLOADS + STORES ~its slice, not the full model — the
storage half of the distributed-model end-state (a frontier model is ~hundreds
of GB; a node should pull only its ~tens of GB). OPT-IN via
PRSM_PARALLAX_SHARD_FETCH; default OFF + local dirs keep the full-snapshot path.

These tests build a real multi-shard tiny model, mock hf_hub_download to serve
from it into a fresh cache, and assert ONLY the owned shards (+ config + index)
are fetched — a shard with no owned layer is never pulled.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
import huggingface_hub  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import prsm.node.chain_executor_adapters as cea  # noqa: E402
from prsm.node.chain_executor_adapters import (  # noqa: E402
    _fetch_only_files,
    build_reduced_config_slice_model,
)


def _sharded_repo(tmp_path):
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(5)
    m = Qwen2ForCausalLM(cfg).eval()
    d = tmp_path / "repo"
    m.save_pretrained(str(d), max_shard_size="30KB")  # force multiple shards
    assert (d / "model.safetensors.index.json").exists(), "model did not shard"
    return str(d)


def _mock_hub(monkeypatch, repo_dir, cache_dir, fetched):
    os.makedirs(cache_dir, exist_ok=True)

    def fake(repo_id, filename, **kw):  # noqa: ARG001
        fetched.append(filename)
        src = os.path.join(repo_dir, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(filename)  # mimic hub 404 (e.g. no index)
        shutil.copy(src, os.path.join(cache_dir, filename))
        return os.path.join(cache_dir, filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("PRSM_PARALLAX_SHARD_FETCH", raising=False)
    yield


# ── _fetch_only_files: only config + index + given shards ─────────────────────

def test_fetch_only_files_requests_only_given(tmp_path, monkeypatch):
    repo = _sharded_repo(tmp_path)
    wm = json.load(open(os.path.join(repo, "model.safetensors.index.json")))["weight_map"]
    a_shard = sorted(set(wm.values()))[0]
    fetched = []
    _mock_hub(monkeypatch, repo, str(tmp_path / "cache"), fetched)
    _fetch_only_files("fake/repo", {a_shard})
    assert "config.json" in fetched
    assert "model.safetensors.index.json" in fetched
    assert a_shard in fetched
    other = set(wm.values()) - {a_shard}
    assert not (other & set(fetched)), f"fetched un-asked shards: {other & set(fetched)}"


# ── build: fetches ONLY the owned shards ─────────────────────────────────────

def test_build_shard_fetch_pulls_only_owned_shards(tmp_path, monkeypatch):
    repo = _sharded_repo(tmp_path)
    wm = json.load(open(os.path.join(repo, "model.safetensors.index.json")))["weight_map"]
    layer0_shards = {wm[k] for k in wm if k.startswith("model.layers.0.")}
    all_shards = set(wm.values())
    not_owned = all_shards - layer0_shards
    assert not_owned, "test needs a shard with no layer-0 weights to prove selectivity"

    fetched = []
    _mock_hub(monkeypatch, repo, str(tmp_path / "cache"), fetched)
    monkeypatch.setenv("PRSM_PARALLAX_SHARD_FETCH", "1")
    # model_id is NOT a local dir → shard-fetch path
    model, depth = build_reduced_config_slice_model(
        "fake/repo", start=0, end=1, is_final_stage=False, device="cpu", dtype="float32",
    )
    fetched_shards = {f for f in fetched if f.endswith(".safetensors")}
    assert layer0_shards.issubset(fetched_shards), "owned (layer-0) shard not fetched"
    assert not (not_owned & fetched_shards), (
        f"fetched non-owned shard(s): {not_owned & fetched_shards}"
    )
    assert depth == 4 and len(model.model.layers) == 1  # slice built from owned shards


# ── back-compat: local dir → never fetches ───────────────────────────────────

def test_local_dir_never_fetches_even_when_enabled(tmp_path, monkeypatch):
    # a local model dir must use the on-disk files, never hf_hub_download
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(6)
    d = tmp_path / "local"
    Qwen2ForCausalLM(cfg).eval().save_pretrained(str(d))
    fetched = []
    _mock_hub(monkeypatch, str(d), str(tmp_path / "cache"), fetched)
    monkeypatch.setenv("PRSM_PARALLAX_SHARD_FETCH", "1")
    model, _ = build_reduced_config_slice_model(
        str(d), start=1, end=3, is_final_stage=False, device="cpu", dtype="float32",
    )
    assert fetched == [], "local dir must not trigger any hub fetch"
    assert len(model.model.layers) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
