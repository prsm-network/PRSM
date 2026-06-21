#!/usr/bin/env python3
"""Generate a parallax model catalog (sprint 1205) from HuggingFace model configs.

The multi-host parallax scheduler (PRSM_INFERENCE_EXECUTOR=parallax) reads a v1 catalog
JSON (PRSM_PARALLAX_MODEL_CATALOG_FILE) mapping model_id → ModelInfo (num_layers,
hidden_dim, num_kv_heads, head_size, ...). Hand-authoring that for a production instruct
model is error-prone, so this fetches each model's HF config and derives the entry via
the same _derive_model_info the local path uses — so the layer slicing is correct.

Only the config.json is downloaded (a few KB), NOT the weights.

Usage:
    python scripts/gen_parallax_catalog.py Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3
    python scripts/gen_parallax_catalog.py Qwen/Qwen2.5-7B-Instruct --out catalog.json
    python scripts/gen_parallax_catalog.py NewModel --merge-into catalog.json   # add to existing

Then point the daemon at it:
    PRSM_INFERENCE_EXECUTOR=parallax PRSM_PARALLAX_MODEL_CATALOG_FILE=catalog.json \
      PRSM_PARALLAX_TRUST_STACK_KIND=... PRSM_PARALLAX_GPU_POOL_KIND=... prsm node start
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a v1 parallax model catalog.")
    ap.add_argument("model_ids", nargs="+", help="HF model ids (e.g. Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--out", default=None, help="write the catalog here (default: stdout)")
    ap.add_argument("--merge-into", default=None,
                    help="merge the new models into an existing v1 catalog file (in place)")
    args = ap.parse_args()

    try:
        from transformers import AutoConfig
        from prsm.compute.inference.local_inference import catalog_entry_for_config
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: needs transformers + the repo ({exc}). pip install -e '.[ml]'",
              file=sys.stderr)
        return 2

    models: dict = {}
    if args.merge_into:
        try:
            existing = json.loads(open(args.merge_into, encoding="utf-8").read())
            if isinstance(existing.get("models"), dict):
                models.update(existing["models"])
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: could not read --merge-into {args.merge_into}: {exc}",
                  file=sys.stderr)
            return 2

    for mid in args.model_ids:
        try:
            cfg = AutoConfig.from_pretrained(mid)  # fetches config.json only
            entry = catalog_entry_for_config(cfg, mid)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: could not derive ModelInfo for {mid!r}: {exc}", file=sys.stderr)
            return 1
        models[mid] = entry
        print(f"  derived {mid}: num_layers={entry['num_layers']} "
              f"hidden_dim={entry['hidden_dim']} num_kv_heads={entry['num_kv_heads']}",
              file=sys.stderr)

    catalog = {"schema_version": "v1", "models": models}
    text = json.dumps(catalog, indent=2)
    out = args.out or args.merge_into
    if out:
        open(out, "w", encoding="utf-8").write(text + "\n")
        print(f"wrote {len(models)} model(s) → {out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
