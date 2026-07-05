#!/usr/bin/env python3
"""Sprint 1378 — generate parallax_pool.json + anchor.json for a controlled multi-stage GPU cluster.

Usage:
    python scripts/gen_cluster_config.py --nodes nodes.json --memory-gb 40 --out-dir .

nodes.json = a JSON list of {"node_id": "<hex>", "pubkey": "<base64>", ...optional overrides:
region / memory_gb / layer_capacity / stake_amount / device}. Collect each node's node_id from
GET /info and its base64 pubkey from ~/.prsm/identity.json.

Writes parallax_pool.json (PRSM_PARALLAX_GPU_POOL_FILE) + anchor.json (PRSM_PUBLISHER_KEY_ANCHOR_FILE)
into --out-dir; drop both on every node. Exit 0 ok, 1 bad descriptors, 2 bad --nodes file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True, help="JSON file: list of {node_id, pubkey, ...}")
    ap.add_argument("--memory-gb", type=float, required=True,
                    help="per-node memory_gb — the split lever: set BELOW the model's single-node footprint")
    ap.add_argument("--region", default="cluster")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args(argv)
    try:
        nodes = json.loads(Path(args.nodes).read_text(encoding="utf-8"))
        if isinstance(nodes, dict) and "nodes" in nodes:
            nodes = nodes["nodes"]
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not read --nodes {args.nodes}: {exc}", file=sys.stderr)
        return 2
    from prsm.node.cluster_config import build_cluster_pool_and_anchor
    try:
        pool, anchor = build_cluster_pool_and_anchor(
            nodes, memory_gb=args.memory_gb, region=args.region)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parallax_pool.json").write_text(json.dumps(pool))
    (out / "anchor.json").write_text(json.dumps(anchor))
    print(f"wrote {out/'parallax_pool.json'} ({len(pool['gpus'])} nodes) + {out/'anchor.json'}")
    print("On EVERY node, set: PRSM_PARALLAX_GPU_POOL_KIND=static-file "
          f"PRSM_PARALLAX_GPU_POOL_FILE={out/'parallax_pool.json'} "
          "PRSM_PUBLISHER_KEY_ANCHOR_KIND=static "
          f"PRSM_PUBLISHER_KEY_ANCHOR_FILE={out/'anchor.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
