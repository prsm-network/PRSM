"""Sprint 1378 — build the static-file pool + publisher-key anchor for a controlled GPU cluster.

Running a big model across a pinned set of nodes (the productionized multi-stage path) needs two
config files per node, both keyed by the participating nodes' identities:

  - ``parallax_pool.json`` — the sp1371 static-file GPU pool (PRSM_PARALLAX_GPU_POOL_KIND=static-file):
    every node listed, same region, memory_gb tuned so the model can't fit one card → forced split.
  - ``anchor.json`` — the static publisher-key anchor (PRSM_PUBLISHER_KEY_ANCHOR_KIND=static): the
    node_id → base64-pubkey map the layer-stage server checks for upstream-token verification.

On the 2026-07-04 mainnet canary these were hand-written per node. This module derives BOTH from a
single list of node descriptors (each node's ``node_id`` + ``pubkey``, collected from ``GET /info``
+ the node's identity), so a deploy script writes one source of truth and drops the two files on
every node. Pure — no I/O, no network — so it's trivially testable; the thin CLI wrapper lives in
``scripts/gen_cluster_config.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

_DEFAULT_STAKE_AMOUNT = 10 ** 18   # 1 FTNS — advisory eligibility passes; real stake overrides
_DEFAULT_LAYER_CAPACITY = 48


def build_cluster_pool_and_anchor(
    nodes: List[Dict[str, Any]],
    *,
    memory_gb: float,
    region: str = "cluster",
    device: str = "cuda",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """From a list of node descriptors, build ``(pool_dict, anchor_dict)``.

    Each descriptor is ``{"node_id": <hex>, "pubkey": <base64>, ...}`` with optional per-node
    overrides ``region`` / ``memory_gb`` / ``layer_capacity`` / ``stake_amount`` / ``device``.
    ``memory_gb`` (the split lever) is required as the cluster default. Raises ``ValueError`` on a
    descriptor missing node_id/pubkey or on <2 distinct nodes (a static multi-stage pool needs ≥2).
    """
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError(
            f"a multi-stage cluster needs >=2 node descriptors, got {len(nodes) if isinstance(nodes, list) else type(nodes).__name__}")
    pool_gpus: List[Dict[str, Any]] = []
    anchor: Dict[str, str] = {}
    seen = set()
    for i, n in enumerate(nodes):
        if not isinstance(n, dict):
            raise ValueError(f"node[{i}] must be an object, got {type(n).__name__}")
        nid = str(n.get("node_id", "")).strip()
        pub = str(n.get("pubkey", "")).strip()
        if not nid or not pub:
            raise ValueError(f"node[{i}] needs both node_id and pubkey")
        if nid in seen:
            raise ValueError(f"duplicate node_id {nid!r}")
        seen.add(nid)
        pool_gpus.append({
            "node_id": nid,
            "region": str(n.get("region", region)),
            "layer_capacity": int(n.get("layer_capacity", _DEFAULT_LAYER_CAPACITY)),
            "memory_gb": float(n.get("memory_gb", memory_gb)),
            "stake_amount": int(n.get("stake_amount", _DEFAULT_STAKE_AMOUNT)),
            "device": str(n.get("device", device)),
        })
        anchor[nid] = pub
    return {"gpus": pool_gpus}, anchor
