"""Sprint 1134 — catalog-miss daemon staging falls back to the model's own HF config.

sp1034 auto-stages PRSM_PARALLAX_HF_MODEL_ID into the registry at daemon start, but
resolves num_layers ONLY from the parallax catalog file — so a production model the
operator configured but that is ABSENT from the bundled catalog is SKIPPED
("skipped:no-num-layers"), forcing a manual catalog edit (the "last manual step").

sp1134 closes that: on a catalog miss, derive num_layers AUTHORITATIVELY from the
model's own locally-cached HF config (AutoConfig) — the same config sp1133 uses. It
still NEVER guesses: if neither the catalog nor the config yields a usable layer
count, staging is skipped exactly as before (no model registered with a wrong
layer_range).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

_CATALOG = (
    Path(__file__).resolve().parents[2]
    / "config" / "parallax" / "model_catalog.json"
)
# A safe model id (matches _SAFE_FS_MODEL_ID) that is NOT in the bundled catalog.
_ABSENT_MODEL = "test-llama-8b"


def _ident(name="n1"):
    from prsm.node.identity import generate_node_identity
    return generate_node_identity(display_name=name)


def _env(root, **over):
    e = {
        "PRSM_INFERENCE_EXECUTOR": "parallax",
        "PRSM_PARALLAX_HF_MODEL_ID": _ABSENT_MODEL,
        "PRSM_MODEL_REGISTRY_ROOT": str(root),
        "PRSM_PARALLAX_MODEL_CATALOG_FILE": str(_CATALOG),
    }
    e.update(over)
    return e


def _patch_autoconfig(monkeypatch, *, num_hidden_layers=None, raise_exc=None):
    """Make transformers.AutoConfig.from_pretrained return a fake config (or raise),
    so the test never touches the network / a real model cache."""
    import transformers

    def _fake(model_id, *a, **k):
        if raise_exc is not None:
            raise raise_exc
        return SimpleNamespace(num_hidden_layers=num_hidden_layers)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _fake)


# ── the helper itself ──────────────────────────────────────────────────────────

def test_helper_reads_num_hidden_layers(monkeypatch):
    from prsm.node.inference_wiring import _resolve_num_layers_from_hf_config
    _patch_autoconfig(monkeypatch, num_hidden_layers=32)
    assert _resolve_num_layers_from_hf_config(_ABSENT_MODEL) == 32


def test_helper_reads_gpt2_style_n_layer(monkeypatch):
    import transformers
    from prsm.node.inference_wiring import _resolve_num_layers_from_hf_config
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained",
        lambda *a, **k: SimpleNamespace(n_layer=6),  # gpt2-style, no num_hidden_layers
    )
    assert _resolve_num_layers_from_hf_config("distil-thing") == 6


def test_helper_returns_none_on_load_failure_never_guesses(monkeypatch):
    from prsm.node.inference_wiring import _resolve_num_layers_from_hf_config
    _patch_autoconfig(monkeypatch, raise_exc=OSError("not cached"))
    assert _resolve_num_layers_from_hf_config(_ABSENT_MODEL) is None


def test_helper_returns_none_on_missing_or_bogus_layer_count(monkeypatch):
    import transformers
    from prsm.node.inference_wiring import _resolve_num_layers_from_hf_config
    # config present but declares no usable layer count (None / bool / <1) -> None
    for cfg in (SimpleNamespace(), SimpleNamespace(num_hidden_layers=None),
                SimpleNamespace(num_hidden_layers=True), SimpleNamespace(num_hidden_layers=0)):
        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, _c=cfg, **k: _c)
        assert _resolve_num_layers_from_hf_config(_ABSENT_MODEL) is None


# ── end-to-end through ensure_hf_model_staged ────────────────────────────────────

def test_catalog_miss_stages_from_hf_config(monkeypatch, tmp_path):
    """The model is absent from the catalog but its HF config declares 32 layers ->
    it now REGISTERS with layer_range (0, 32) instead of being skipped."""
    from prsm.node.inference_wiring import ensure_hf_model_staged
    from prsm.compute.model_registry.registry import FilesystemModelRegistry
    _patch_autoconfig(monkeypatch, num_hidden_layers=32)
    ident = _ident()
    assert ensure_hf_model_staged(ident, env=_env(tmp_path)) == "registered"
    loaded = FilesystemModelRegistry(root=tmp_path).get(_ABSENT_MODEL)
    assert tuple(loaded.shards[0].layer_range) == (0, 32)


def test_catalog_miss_and_no_config_still_skips(monkeypatch, tmp_path):
    """Neither the catalog nor a loadable config yields a layer count -> still skipped
    (the never-guess invariant is preserved; no model registered)."""
    from prsm.node.inference_wiring import ensure_hf_model_staged
    _patch_autoconfig(monkeypatch, raise_exc=OSError("not cached"))
    ident = _ident()
    assert ensure_hf_model_staged(ident, env=_env(tmp_path)) == f"skipped:no-num-layers:{_ABSENT_MODEL}"
    assert not (tmp_path / _ABSENT_MODEL).exists()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
