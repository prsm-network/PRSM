"""Sprint 1133 — config-driven ModelInfo for the local inference executor.

The local executor (``prsm/compute/inference/local_inference.py``) loads ANY
HuggingFace causal-LM generically for *generation*, but its ModelInfo METADATA
was hardcoded to the gpt2 family. That metadata feeds the allocator catalog
which drives chain-stage LAYER SLICING across nodes — so for any non-gpt2 model
(Llama/Mistral/Qwen: different layer count, hidden size, GQA kv-heads,
intermediate size) the hardcoded gpt2 dims produced WRONG slicing.

These tests pin a PURE ``_derive_model_info(config, model_id)`` helper that maps
a HF-config-like object (anything exposing the attributes — SimpleNamespace, so
NO network) to a ModelInfo, handling BOTH gpt2-style and modern naming + GQA
num_kv_heads + head_size + intermediate fallback. They also pin the thin
``_model_info_for`` loader's resilience: on AutoConfig load failure it falls back
to the proven gpt2 hardcode path without crashing.

CRITICAL invariant: gpt2/distilgpt2 dims must remain BYTE-IDENTICAL to the
historical hardcoded values (the proven, in-production dims).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prsm.compute.inference import local_inference as li
from prsm.compute.parallax_scheduling.model_info import ModelInfo


# ---------------------------------------------------------------------------
# (a) gpt2 PARITY — derivation must reproduce the historical hardcoded values
# ---------------------------------------------------------------------------
def _gpt2_style_config(n_layer: int) -> SimpleNamespace:
    """A gpt2-style PretrainedConfig: n_layer / n_embd / n_head / n_inner / vocab_size.

    Modern attributes are absent (getattr -> None), so the helper must rely on
    the gpt2-naming fallbacks. gpt2 is MHA (no num_key_value_heads).
    """
    return SimpleNamespace(
        n_layer=n_layer,
        n_embd=768,
        n_head=12,
        n_inner=3072,
        vocab_size=50257,
    )


def test_gpt2_parity_full_model_info():
    """gpt2-style config -> the exact historical hardcoded ModelInfo dims."""
    info = li._derive_model_info(_gpt2_style_config(12), "gpt2")
    assert isinstance(info, ModelInfo)
    assert info.model_name == "gpt2"
    assert info.mlx_model_name == "gpt2-mlx"
    assert info.num_layers == 12
    assert info.head_size == 64
    assert info.hidden_dim == 768
    assert info.intermediate_dim == 3072
    assert info.num_attention_heads == 12
    assert info.num_kv_heads == 12  # gpt2 is MHA -> kv == attention heads
    assert info.vocab_size == 50257


def test_gpt2_parity_matches_legacy_hardcode_exactly():
    """Derived gpt2/distilgpt2 dims equal the legacy ``_GPT2_FAMILY_DIMS`` /
    ``_KNOWN_MODELS`` hardcode field-for-field (no regression)."""
    for model_id, n_layer in (("gpt2", 12), ("distilgpt2", 6)):
        info = li._derive_model_info(_gpt2_style_config(n_layer), model_id)
        assert info.num_layers == n_layer == li._KNOWN_MODELS[model_id]
        assert info.head_size == li._GPT2_FAMILY_DIMS["head_size"]
        assert info.hidden_dim == li._GPT2_FAMILY_DIMS["hidden_dim"]
        assert info.intermediate_dim == li._GPT2_FAMILY_DIMS["intermediate_dim"]
        assert info.num_attention_heads == li._GPT2_FAMILY_DIMS["num_attention_heads"]
        assert info.num_kv_heads == li._GPT2_FAMILY_DIMS["num_kv_heads"]
        assert info.vocab_size == li._GPT2_FAMILY_DIMS["vocab_size"]


def test_distilgpt2_layer_count():
    """distilgpt2-style (n_layer=6) -> num_layers=6, same family dims."""
    info = li._derive_model_info(_gpt2_style_config(6), "distilgpt2")
    assert info.num_layers == 6
    assert info.hidden_dim == 768
    assert info.head_size == 64
    assert info.num_attention_heads == 12
    assert info.num_kv_heads == 12


# ---------------------------------------------------------------------------
# (b) MODERN dense (Llama-ish) — GQA preserved, head_size from hidden/heads
# ---------------------------------------------------------------------------
def test_modern_dense_llama_gqa():
    """Llama-3-8B-ish modern config -> correct dims, GQA kv-heads preserved."""
    cfg = SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        intermediate_size=14336,
        vocab_size=128256,
    )
    info = li._derive_model_info(cfg, "meta-llama/Llama-3-8B")
    assert info.num_layers == 32
    assert info.hidden_dim == 4096
    assert info.head_size == 128  # 4096 // 32
    assert info.num_attention_heads == 32
    assert info.num_kv_heads == 8  # GQA preserved (NOT collapsed to 32)
    assert info.intermediate_dim == 14336
    assert info.vocab_size == 128256
    assert info.model_name == "meta-llama/Llama-3-8B"
    assert info.mlx_model_name == "meta-llama/Llama-3-8B-mlx"


def test_modern_explicit_head_dim_respected():
    """When the modern config carries an explicit ``head_dim``, it wins over
    hidden_size // num_attention_heads (some configs set a non-divisible dim)."""
    cfg = SimpleNamespace(
        num_hidden_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        intermediate_size=18944,
        vocab_size=152064,
        head_dim=256,  # explicit, != 3584 // 28 (=128)
    )
    info = li._derive_model_info(cfg, "Qwen/Qwen2-7B")
    assert info.head_size == 256
    assert info.num_kv_heads == 4


# ---------------------------------------------------------------------------
# (c) GQA absent -> num_kv_heads defaults to num_attention_heads (MHA)
# ---------------------------------------------------------------------------
def test_modern_without_kv_heads_defaults_to_mha():
    """Modern config WITHOUT num_key_value_heads -> num_kv_heads == num_attention_heads."""
    cfg = SimpleNamespace(
        num_hidden_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        intermediate_size=5632,
        vocab_size=32000,
        # no num_key_value_heads
    )
    info = li._derive_model_info(cfg, "some/dense-mha")
    assert info.num_attention_heads == 16
    assert info.num_kv_heads == 16  # MHA fallback
    assert info.head_size == 128  # 2048 // 16


# ---------------------------------------------------------------------------
# (d) intermediate absent -> 4 * hidden_dim
# ---------------------------------------------------------------------------
def test_gpt2_style_n_inner_none_defaults_to_4x_hidden():
    """gpt2-style with n_inner=None -> intermediate_dim == 4 * hidden_dim."""
    cfg = SimpleNamespace(
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_inner=None,  # HF gpt2 commonly leaves this None
        vocab_size=50257,
    )
    info = li._derive_model_info(cfg, "gpt2")
    assert info.intermediate_dim == 4 * 768 == 3072


def test_modern_without_intermediate_defaults_to_4x_hidden():
    """Modern config WITHOUT intermediate_size -> intermediate_dim == 4 * hidden_dim."""
    cfg = SimpleNamespace(
        num_hidden_layers=8,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=16,
        vocab_size=32000,
        # no intermediate_size
    )
    info = li._derive_model_info(cfg, "some/no-intermediate")
    assert info.intermediate_dim == 4 * 1024 == 4096


# ---------------------------------------------------------------------------
# (e) FALLBACK — _model_info_for survives AutoConfig load failure
# ---------------------------------------------------------------------------
def test_model_info_for_falls_back_on_autoconfig_failure(monkeypatch):
    """When AutoConfig.from_pretrained raises (offline/uncached/no transformers),
    ``_model_info_for`` returns the gpt2-hardcode ModelInfo without crashing."""
    def _boom(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise OSError("offline / not cached")

    # Patch the AutoConfig the loader uses, regardless of import strategy.
    import transformers
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(_boom))

    info = li._model_info_for("gpt2")
    assert isinstance(info, ModelInfo)
    assert info.model_name == "gpt2"
    assert info.num_layers == 12
    assert info.head_size == li._GPT2_FAMILY_DIMS["head_size"]
    assert info.hidden_dim == li._GPT2_FAMILY_DIMS["hidden_dim"]
    assert info.intermediate_dim == li._GPT2_FAMILY_DIMS["intermediate_dim"]
    assert info.num_attention_heads == li._GPT2_FAMILY_DIMS["num_attention_heads"]
    assert info.num_kv_heads == li._GPT2_FAMILY_DIMS["num_kv_heads"]
    assert info.vocab_size == li._GPT2_FAMILY_DIMS["vocab_size"]


def test_model_info_for_unknown_model_fallback_uses_12_layers(monkeypatch):
    """Unknown model id with AutoConfig failure -> the legacy default of 12 layers."""
    import transformers
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))),
    )
    info = li._model_info_for("some/uncached-model")
    assert info.num_layers == 12
    assert info.hidden_dim == 768


# ---------------------------------------------------------------------------
# Optional real-config integration check (skips if distilgpt2 not cached)
# ---------------------------------------------------------------------------
def test_real_distilgpt2_config_if_cached():
    """If distilgpt2 is locally cached, the real AutoConfig drives the SAME
    proven dims (6 layers, 768 hidden, 12 heads, 64 head_size, 50257 vocab)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained("distilgpt2", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("distilgpt2 config not locally cached")

    info = li._derive_model_info(cfg, "distilgpt2")
    assert info.num_layers == 6
    assert info.hidden_dim == 768
    assert info.num_attention_heads == 12
    assert info.num_kv_heads == 12
    assert info.head_size == 64
    assert info.vocab_size == 50257
    assert info.intermediate_dim == 3072
