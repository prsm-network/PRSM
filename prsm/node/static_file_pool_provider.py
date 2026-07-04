"""Sprint 1371 — static-file GPU pool provider.

The dht-backed pool (sprint 682) builds the parallax GPU pool from self + peer-advertised
``hardware_profile``s propagated over the discovery layer. That propagation is a **libp2p** feature
(``libp2p_discovery.py``); over the default WebSocket transport peer profiles do NOT propagate, so a
head only ever sees ITSELF in the pool (``gpu_count == 1``) and can never plan a multi-stage split —
even when peers are transport-connected via ``/peers/connect``. (Diagnosed live 2026-07-03 during the
mainnet multi-stage settlement canary.)

This provider lets an operator PIN the pool explicitly from a JSON file: each entry becomes a
``ParallaxGPU`` with the operator-chosen ``node_id`` / ``region`` / ``layer_capacity``. A controlled
multi-node deployment (e.g. a settlement canary, or a private cluster) can thus force a real
cross-node split without libp2p — list both nodes in the SAME region with a ``layer_capacity`` below
the model's layer count, and the planner shards across them. Opt-in via
``PRSM_PARALLAX_GPU_POOL_KIND=static-file`` + ``PRSM_PARALLAX_GPU_POOL_FILE=<path>``.

Re-reads the file on every call (like the dht-backed provider) so edits take effect without a
restart. FAIL-SOFT: a missing/unparseable file yields ``[]`` (the request then fails Phase-1
allocation with a clear error, never a crash); a malformed entry is skipped while valid entries are
kept.

File format — a bare list, or ``{"gpus": [...]}``; each entry:
    {"node_id": "<32-hex>", "region": "canary", "layer_capacity": 6, "memory_gb": 16.0,
     // optional: "stake_amount", "tflops_fp16", "memory_bandwidth_gbps", "device",
     //           "gpu_name", "num_gpus", "tier_attestation"}
Note: with ``PRSM_PARALLAX_STAKE_ELIGIBILITY=enforced`` an entry needs a positive ``stake_amount``
(or relax eligibility) to be schedulable — this provider does not fake stake.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, List

from prsm.compute.parallax_scheduling.prsm_types import (
    TIER_ATTESTATION_NONE,
    ParallaxGPU,
)

logger = logging.getLogger(__name__)

_POOL_FILE_ENV = "PRSM_PARALLAX_GPU_POOL_FILE"

# Defaults for optional fields — chosen so a minimal entry
# (node_id / region / layer_capacity / memory_gb) builds a VALID ParallaxGPU. tflops + bandwidth are
# scheduling-roofline inputs the operator can override per entry.
_DEFAULT_TFLOPS = 100.0
_DEFAULT_BANDWIDTH_GBPS = 100.0


def _gpu_from_entry(entry: dict) -> ParallaxGPU:
    """Build a ParallaxGPU from one file entry. Raises (KeyError/ValueError/TypeError) on a malformed
    entry — the caller skips it fail-soft. Required keys: node_id, region, layer_capacity, memory_gb."""
    return ParallaxGPU(
        node_id=str(entry["node_id"]),
        region=str(entry["region"]),
        layer_capacity=int(entry["layer_capacity"]),
        stake_amount=int(entry.get("stake_amount", 0)),
        tier_attestation=str(entry.get("tier_attestation") or TIER_ATTESTATION_NONE),
        tflops_fp16=float(entry.get("tflops_fp16", _DEFAULT_TFLOPS)),
        memory_gb=float(entry["memory_gb"]),
        memory_bandwidth_gbps=float(
            entry.get("memory_bandwidth_gbps", _DEFAULT_BANDWIDTH_GBPS)),
        gpu_name=str(entry.get("gpu_name", "") or ""),
        device=str(entry.get("device", "cuda") or "cuda"),
        num_gpus=int(entry.get("num_gpus", 1)),
        # sp1372 — optional per-entry inter-node RTT map (node_id → ms). Missing links get
        # the full-mesh default below so the DP router can route (absent RTT → inf → no chain).
        rtt_to_nodes=(
            {str(k): float(v) for k, v in entry["rtt_to_nodes"].items()}
            if isinstance(entry.get("rtt_to_nodes"), dict) else None),
    )


def _patch_full_mesh_rtt(gpus: List[ParallaxGPU]) -> List[ParallaxGPU]:
    """Sprint 1372 — mirror the dht provider (dht_backed_pool_provider): the Phase-2 DP router
    computes inter-stage cost from ``node.rtt_to(other)``, which returns ``inf`` when
    ``rtt_to_nodes`` is absent — so a chain across statically-pinned nodes is rejected as
    infeasible (``path=[] latency=inf``) and only single-node requests route. Patch a full-mesh
    default RTT (``PRSM_PARALLAX_DEFAULT_RTT_MS``, else 100 ms) for any missing link so a
    multi-node static pool is ROUTABLE, not just allocatable. Per-entry ``rtt_to_nodes`` wins."""
    if len(gpus) <= 1:
        return gpus
    default_rtt = 100.0
    raw = (os.environ.get("PRSM_PARALLAX_DEFAULT_RTT_MS", "") or "").strip()
    if raw:
        try:
            default_rtt = float(raw)
        except ValueError:
            pass
    from dataclasses import replace as _dc_replace
    patched: List[ParallaxGPU] = []
    for g in gpus:
        rtt_map = dict(g.rtt_to_nodes) if g.rtt_to_nodes else {}
        for other in gpus:
            if other.node_id != g.node_id and other.node_id not in rtt_map:
                rtt_map[other.node_id] = default_rtt
        patched.append(_dc_replace(g, rtt_to_nodes=rtt_map))
    return patched


def _load_entries(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        logger.warning(
            "static-file pool: %s=%s does not exist — empty pool", _POOL_FILE_ENV, path)
        return []
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("static-file pool: %s failed to load (%s) — empty pool", path, exc)
        return []
    if isinstance(raw, dict):
        raw = raw.get("gpus", [])
    if not isinstance(raw, list):
        logger.warning(
            "static-file pool: %s must be a list or {\"gpus\": [...]}; got %s — empty pool",
            path, type(raw).__name__)
        return []
    return [e for e in raw if isinstance(e, dict)]


def build_static_file_pool_provider(node: Any = None) -> Callable[[], List[ParallaxGPU]]:
    """Return a pool provider that reads ``PRSM_PARALLAX_GPU_POOL_FILE`` fresh on each call and
    yields one ``ParallaxGPU`` per valid entry. ``node`` is accepted for dispatch-signature parity
    with ``build_dht_backed_pool_provider`` and is unused."""
    def _provider() -> List[ParallaxGPU]:
        path = (os.environ.get(_POOL_FILE_ENV, "") or "").strip()
        if not path:
            logger.warning("static-file pool: %s unset — empty pool", _POOL_FILE_ENV)
            return []
        gpus: List[ParallaxGPU] = []
        for entry in _load_entries(path):
            try:
                gpus.append(_gpu_from_entry(entry))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "static-file pool: skipping malformed entry %s: %s",
                    entry.get("node_id", "<no node_id>"), exc)
        return _patch_full_mesh_rtt(gpus)
    return _provider
