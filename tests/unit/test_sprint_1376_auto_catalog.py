"""Sprint 1376 — auto-catalog the operator's configured HF model.

Serving a real HF model beyond the bundled catalog used to require hand-running
gen_parallax_catalog.py + editing the catalog file (the manual step the 2026-07-04 mainnet GPU
canary hit). augment_catalog_with_configured_hf_model derives the ModelInfo from the model's HF
config on demand. Injectable config_loader/entry_deriver keep the test network-free.
"""
from prsm.node.inference_wiring import augment_catalog_with_configured_hf_model


class _FakeModelInfo:
    def __init__(self, **kw):
        self.kw = kw


def _deriver(_cfg, mid):
    return {"num_layers": 48, "model_name": mid, "hidden_dim": 5120}


def test_adds_missing_configured_model():
    cat = {}
    augment_catalog_with_configured_hf_model(
        cat, _FakeModelInfo, environ={"PRSM_PARALLAX_HF_MODEL_ID": "org/Big-Model"},
        config_loader=lambda mid: object(), entry_deriver=_deriver)
    assert "org/Big-Model" in cat                          # auto-added
    assert cat["org/Big-Model"].kw["num_layers"] == 48     # via the derived entry


def test_noop_when_already_present():
    sentinel = object()
    cat = {"org/Big-Model": sentinel}
    called = []
    augment_catalog_with_configured_hf_model(
        cat, _FakeModelInfo, environ={"PRSM_PARALLAX_HF_MODEL_ID": "org/Big-Model"},
        config_loader=lambda mid: called.append(mid), entry_deriver=_deriver)
    assert cat["org/Big-Model"] is sentinel                # untouched
    assert not called                                      # deriver never invoked (no re-fetch)


def test_noop_when_env_unset():
    cat = {}
    augment_catalog_with_configured_hf_model(
        cat, _FakeModelInfo, environ={}, config_loader=lambda m: 1 / 0, entry_deriver=_deriver)
    assert cat == {}                                       # nothing configured → nothing fetched


def test_fail_soft_on_derive_error():
    def boom(_cfg, _mid):
        raise RuntimeError("config.json unreachable")
    cat = {"existing": 1}
    augment_catalog_with_configured_hf_model(
        cat, _FakeModelInfo, environ={"PRSM_PARALLAX_HF_MODEL_ID": "org/Bad"},
        config_loader=lambda mid: object(), entry_deriver=boom)
    assert cat == {"existing": 1}                          # unchanged, no raise (fail-soft)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
