"""Sprint 1033 — actionable 'inference executor not initialized' diagnostic.

The Tier-1 cross-host bench lost real time to a silent operator footgun: a GPU
operator who installed only `.[ml]` (torch/transformers) but not `.[blockchain]`
got `web3`/`eth_account` missing → the production trust stack's anchor client
couldn't build → build_parallax_executor_or_none returned None → /compute/inference
returned a GENERIC 503 "Inference executor not initialized" with no cause. The
real reason was only in the daemon log (had to grep). Same class: an unset
PRSM_PARALLAX_MODEL_CATALOG_FILE, or PRSM_INFERENCE_EXECUTOR not set at all.

diagnose_inference_executor_unavailable() re-derives the most likely actionable
cause (sp1027 pattern: surface the real reason in the 503 detail) from the env +
import availability + catalog file, so the next operator gets a fix, not a grep.
"""
from __future__ import annotations


def _diag(**kw):
    from prsm.node.inference_wiring import (
        diagnose_inference_executor_unavailable,
    )
    return diagnose_inference_executor_unavailable(**kw)


_ALL_PRESENT = lambda m: True          # noqa: E731
_NONE_PRESENT = lambda m: False        # noqa: E731
_EXISTS = lambda p: True               # noqa: E731
_MISSING = lambda p: False             # noqa: E731


def test_executor_unset_points_at_env_var():
    msg = _diag(env={}, module_check=_ALL_PRESENT, path_check=_EXISTS)
    assert "PRSM_INFERENCE_EXECUTOR" in msg
    assert "local" in msg and "parallax" in msg


def test_parallax_missing_web3_points_at_blockchain_extra():
    """The painful live case: .[ml] installed, .[blockchain] not."""
    msg = _diag(
        env={"PRSM_INFERENCE_EXECUTOR": "parallax"},
        module_check=lambda m: m not in ("web3", "eth_account"),
        path_check=_EXISTS,
    )
    assert "web3" in msg
    assert "[blockchain]" in msg


def test_parallax_catalog_unset_points_at_catalog():
    msg = _diag(
        env={"PRSM_INFERENCE_EXECUTOR": "parallax"},
        module_check=_ALL_PRESENT,
        path_check=_EXISTS,
    )
    assert "PRSM_PARALLAX_MODEL_CATALOG_FILE" in msg


def test_parallax_catalog_missing_file_points_at_path():
    msg = _diag(
        env={
            "PRSM_INFERENCE_EXECUTOR": "parallax",
            "PRSM_PARALLAX_MODEL_CATALOG_FILE": "/nope/catalog.json",
        },
        module_check=_ALL_PRESENT,
        path_check=_MISSING,
    )
    assert "does not exist" in msg
    assert "/nope/catalog.json" in msg


def test_parallax_required_kind_unset_named():
    msg = _diag(
        env={
            "PRSM_INFERENCE_EXECUTOR": "parallax",
            "PRSM_PARALLAX_MODEL_CATALOG_FILE": "/ok/catalog.json",
            "PRSM_PARALLAX_GPU_POOL_KIND": "dht-backed",
            # PRSM_PARALLAX_TRUST_STACK_KIND intentionally unset
        },
        module_check=_ALL_PRESENT,
        path_check=_EXISTS,
    )
    assert "PRSM_PARALLAX_TRUST_STACK_KIND" in msg


def test_parallax_all_set_falls_back_to_log_hint():
    msg = _diag(
        env={
            "PRSM_INFERENCE_EXECUTOR": "parallax",
            "PRSM_PARALLAX_MODEL_CATALOG_FILE": "/ok/catalog.json",
            "PRSM_PARALLAX_GPU_POOL_KIND": "dht-backed",
            "PRSM_PARALLAX_TRUST_STACK_KIND": "production",
        },
        module_check=_ALL_PRESENT,
        path_check=_EXISTS,
    )
    assert "log" in msg.lower()


def test_local_missing_torch_points_at_ml_extra():
    msg = _diag(
        env={"PRSM_INFERENCE_EXECUTOR": "local"},
        module_check=lambda m: m not in ("torch", "transformers"),
        path_check=_EXISTS,
    )
    assert "[ml]" in msg


def test_unknown_executor_value_flagged():
    msg = _diag(
        env={"PRSM_INFERENCE_EXECUTOR": "banana"},
        module_check=_ALL_PRESENT,
        path_check=_EXISTS,
    )
    assert "banana" in msg
    assert "recogni" in msg.lower()


def test_default_checks_are_callable_without_injection():
    """With no injected checkers it must still return a non-empty string
    (uses the real importlib/os checks) — never crash the 503 path."""
    msg = _diag(env={"PRSM_INFERENCE_EXECUTOR": "parallax"})
    assert isinstance(msg, str) and msg


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
