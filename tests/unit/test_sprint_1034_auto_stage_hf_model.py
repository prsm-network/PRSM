"""Sprint 1034 — daemon auto-stages the configured HF model into the registry.

The last manual operator step the Tier-1 cross-host bench exposed: an unstaged
model makes the layer_stage chain server return MODEL_NOT_FOUND ("model 'gpt2'
not in local registry") at inference time, forcing operators to run
scripts/stage_hf_model.py by hand on every node. The daemon now auto-stages
PRSM_PARALLAX_HF_MODEL_ID into PRSM_MODEL_REGISTRY_ROOT at startup —
idempotent, publisher-aware (never clobbers another publisher's manifest), and
fail-open (returns a status string, never raises, so it can't crash start()).

num_layers comes from the parallax catalog (never defaulted); the sentinel
manifest matches scripts/stage_hf_model.py (the HF runner loads real weights
from the HF cache at inference time — the registry just needs the model known
with the right layer_range).
"""
from __future__ import annotations

from pathlib import Path

_CATALOG = (
    Path(__file__).resolve().parents[2]
    / "config" / "parallax" / "model_catalog.json"
)


def _ident(name="n1"):
    from prsm.node.identity import generate_node_identity
    return generate_node_identity(display_name=name)


def _env(root, **over):
    e = {
        "PRSM_INFERENCE_EXECUTOR": "parallax",
        "PRSM_PARALLAX_HF_MODEL_ID": "gpt2",
        "PRSM_MODEL_REGISTRY_ROOT": str(root),
        "PRSM_PARALLAX_MODEL_CATALOG_FILE": str(_CATALOG),
    }
    e.update(over)
    return e


def test_stages_configured_model(tmp_path):
    from prsm.node.inference_wiring import ensure_hf_model_staged
    from prsm.compute.model_registry.registry import FilesystemModelRegistry
    ident = _ident()
    assert ensure_hf_model_staged(ident, env=_env(tmp_path)) == "registered"
    reg = FilesystemModelRegistry(root=tmp_path)
    man = reg.get_manifest("gpt2")
    assert man.publisher_node_id == ident.node_id
    loaded = reg.get("gpt2")                       # full verify roundtrip
    assert tuple(loaded.shards[0].layer_range) == (0, 12)   # gpt2 = 12 layers


def test_idempotent_same_publisher(tmp_path):
    from prsm.node.inference_wiring import ensure_hf_model_staged
    ident = _ident()
    assert ensure_hf_model_staged(ident, env=_env(tmp_path)) == "registered"
    assert ensure_hf_model_staged(ident, env=_env(tmp_path)) == "already"


def test_refuses_to_clobber_different_publisher(tmp_path):
    from prsm.node.inference_wiring import ensure_hf_model_staged
    from prsm.compute.model_registry.registry import FilesystemModelRegistry
    a, b = _ident("a"), _ident("b")
    assert ensure_hf_model_staged(a, env=_env(tmp_path)) == "registered"
    assert (
        ensure_hf_model_staged(b, env=_env(tmp_path))
        == "refused:different-publisher"
    )
    # A's manifest must be intact — not clobbered by B.
    man = FilesystemModelRegistry(root=tmp_path).get_manifest("gpt2")
    assert man.publisher_node_id == a.node_id


def test_distilgpt2_layer_count_from_catalog(tmp_path):
    from prsm.node.inference_wiring import ensure_hf_model_staged
    from prsm.compute.model_registry.registry import FilesystemModelRegistry
    ident = _ident()
    s = ensure_hf_model_staged(
        ident, env=_env(tmp_path, PRSM_PARALLAX_HF_MODEL_ID="distilgpt2"),
    )
    assert s == "registered"
    loaded = FilesystemModelRegistry(root=tmp_path).get("distilgpt2")
    assert tuple(loaded.shards[0].layer_range) == (0, 6)   # distilgpt2 = 6


def test_skipped_not_parallax(tmp_path):
    from prsm.node.inference_wiring import ensure_hf_model_staged
    s = ensure_hf_model_staged(
        _ident(), env=_env(tmp_path, PRSM_INFERENCE_EXECUTOR="local"),
    )
    assert s == "skipped:not-parallax"


def test_skipped_no_model_id(tmp_path):
    from prsm.node.inference_wiring import ensure_hf_model_staged
    s = ensure_hf_model_staged(
        _ident(), env=_env(tmp_path, PRSM_PARALLAX_HF_MODEL_ID=""),
    )
    assert s == "skipped:no-model-id"


def test_skipped_no_registry_root(tmp_path):
    from prsm.node.inference_wiring import ensure_hf_model_staged
    s = ensure_hf_model_staged(
        _ident(), env=_env(tmp_path, PRSM_MODEL_REGISTRY_ROOT=""),
    )
    assert s == "skipped:no-registry-root"


def test_skipped_no_identity(tmp_path):
    from prsm.node.inference_wiring import ensure_hf_model_staged
    assert ensure_hf_model_staged(None, env=_env(tmp_path)) == "skipped:no-identity"


def test_skipped_model_not_in_catalog(tmp_path):
    """Never default num_layers — a model absent from the catalog is skipped
    with a clear status, not registered with a wrong layer count."""
    from prsm.node.inference_wiring import ensure_hf_model_staged
    s = ensure_hf_model_staged(
        _ident(), env=_env(tmp_path, PRSM_PARALLAX_HF_MODEL_ID="not-a-real-model"),
    )
    assert s.startswith("skipped:no-num-layers")


def test_never_raises_on_bad_registry_root(tmp_path):
    """Registry root is a FILE, not a dir → the register path errors, but the
    function returns a string ('error:...') rather than raising — so a staging
    failure can never crash daemon initialize()."""
    from prsm.node.inference_wiring import ensure_hf_model_staged
    bad = tmp_path / "iam_a_file"
    bad.write_text("x")
    s = ensure_hf_model_staged(
        _ident(), env=_env(tmp_path, PRSM_MODEL_REGISTRY_ROOT=str(bad)),
    )
    assert isinstance(s, str)
    assert s.startswith("error:") or s.startswith("skipped:")


def _write_catalog(path, models):
    import json
    path.write_text(json.dumps({"schema_version": "v1", "models": models}))


def test_bool_num_layers_rejected(tmp_path):
    """isinstance(True, int) is True in Python — a JSON boolean must NOT pass as
    a layer count (it would silently stage layer_range=(0,1))."""
    from prsm.node.inference_wiring import ensure_hf_model_staged
    cat = tmp_path / "boolcat.json"
    _write_catalog(cat, {"gpt2": {"num_layers": True}})
    s = ensure_hf_model_staged(
        _ident(), env=_env(tmp_path, PRSM_PARALLAX_MODEL_CATALOG_FILE=str(cat)),
    )
    assert s.startswith("skipped:no-num-layers")


def test_invalid_model_id_skipped_before_path_join(tmp_path):
    """A traversal/odd model_id is rejected BEFORE any filesystem Path-join, so
    the manifest pre-check can't stat/read outside the registry root."""
    from prsm.node.inference_wiring import ensure_hf_model_staged
    for bad in ("../evil", "a/b", "..", ".", "has space"):
        s = ensure_hf_model_staged(
            _ident(), env=_env(tmp_path, PRSM_PARALLAX_HF_MODEL_ID=bad),
        )
        assert s == "skipped:invalid-model-id", bad


def test_stale_layer_range_surfaced(tmp_path):
    """If the catalog's layer count changes after a model was staged, the stale
    on-disk manifest is detected + surfaced (not silently trusted)."""
    from prsm.node.inference_wiring import ensure_hf_model_staged
    ident = _ident()
    assert ensure_hf_model_staged(ident, env=_env(tmp_path)) == "registered"  # gpt2=12
    cat = tmp_path / "cat10.json"
    _write_catalog(cat, {"gpt2": {"num_layers": 10}})   # corrected to 10
    s = ensure_hf_model_staged(
        ident, env=_env(tmp_path, PRSM_PARALLAX_MODEL_CATALOG_FILE=str(cat)),
    )
    assert s.startswith("already:stale-layer-range")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
