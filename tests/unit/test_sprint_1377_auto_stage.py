"""Sprint 1377 — auto-stage the configured HF model into the local registry at node-start.

build_layer_stage_server_executor needs a signed manifest in the FilesystemModelRegistry; without
it the worker's stage executor is a stub and the multi-stage chain fails (the 2026-07-04 mainnet GPU
canary hit this — the model had to be staged by hand). ensure_hf_model_registered folds that
registration into node-start. Idempotent + fail-soft.
"""
from prsm.node.chain_executor_adapters import ensure_hf_model_registered
from prsm.node.identity import generate_node_identity


def test_registers_then_idempotent(tmp_path):
    from prsm.compute.model_registry.registry import FilesystemModelRegistry
    ident = generate_node_identity(display_name="op")
    ok = ensure_hf_model_registered(
        registry_root=str(tmp_path), model_id="org/Big-14B", num_layers=48, identity=ident)
    assert ok is True
    m = FilesystemModelRegistry(root=tmp_path).get("org/Big-14B")
    assert m is not None
    assert m.shards[0].layer_range == (0, 48)              # the runner learns the layer count
    # second call is a no-op success (manifest already present) — never raises
    assert ensure_hf_model_registered(
        registry_root=str(tmp_path), model_id="org/Big-14B", num_layers=48, identity=ident) is True


def test_fail_soft_on_bad_identity(tmp_path):
    # a None identity can't sign the manifest → helper must fail-soft (False, no raise)
    ok = ensure_hf_model_registered(
        registry_root=str(tmp_path), model_id="org/X", num_layers=48, identity=None)
    assert ok is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
