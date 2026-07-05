"""Sprint 1378 — build the static-file pool + publisher-key anchor for a controlled GPU cluster."""
import pytest

from prsm.node.cluster_config import build_cluster_pool_and_anchor

_A = {"node_id": "a" * 32, "pubkey": "PUBA="}
_B = {"node_id": "b" * 32, "pubkey": "PUBB="}


def test_builds_pool_and_anchor():
    pool, anchor = build_cluster_pool_and_anchor([_A, _B], memory_gb=40.0, region="canary")
    assert [g["node_id"] for g in pool["gpus"]] == ["a" * 32, "b" * 32]
    assert all(g["region"] == "canary" and g["memory_gb"] == 40.0 for g in pool["gpus"])
    assert all(g["device"] == "cuda" and g["stake_amount"] > 0 for g in pool["gpus"])
    assert anchor == {"a" * 32: "PUBA=", "b" * 32: "PUBB="}   # node_id → pubkey map


def test_per_node_overrides_win():
    pool, _ = build_cluster_pool_and_anchor(
        [{**_A, "memory_gb": 24.0, "region": "us"}, _B], memory_gb=40.0, region="canary")
    a = next(g for g in pool["gpus"] if g["node_id"] == "a" * 32)
    assert a["memory_gb"] == 24.0 and a["region"] == "us"     # explicit override
    b = next(g for g in pool["gpus"] if g["node_id"] == "b" * 32)
    assert b["memory_gb"] == 40.0 and b["region"] == "canary"  # falls back to cluster default


def test_needs_two_nodes():
    with pytest.raises(ValueError):
        build_cluster_pool_and_anchor([_A], memory_gb=40.0)


@pytest.mark.parametrize("bad", [
    [{"node_id": "a" * 32}, _B],              # missing pubkey
    [{"pubkey": "P="}, _B],                    # missing node_id
    [_A, {"node_id": "a" * 32, "pubkey": "X="}],  # duplicate node_id
])
def test_rejects_malformed(bad):
    with pytest.raises(ValueError):
        build_cluster_pool_and_anchor(bad, memory_gb=40.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
