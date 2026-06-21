"""Sprint 845 — stage an HF model into the local PRSM FilesystemModelRegistry.

Generalized from the one-off staging steps in scripts/sprint_625*.py +
scripts/sprint_626*.py. Reusable across operators + models.

For HF-runner-based operators
(``PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=huggingface``), the registry
manifest is a sentinel — actual model weights load from the HF cache
via ``transformers.AutoModelForCausalLM.from_pretrained(model_id)`` at
inference time. The registry just needs the model to be "known" with
correct ``layer_range`` so the runner knows how many transformer
blocks to forward through.

Usage:
    PRSM_DEPLOYER_PRIVATE_KEY unused  (signing uses node identity, not chain)
    python scripts/stage_hf_model.py <model_id>
    python scripts/stage_hf_model.py <model_id> --layers 6
    python scripts/stage_hf_model.py <model_id> --registry-root /custom/path
    python scripts/stage_hf_model.py <model_id> --identity-file ~/.prsm/identity.json

Resolution order for num_layers:
  1. --layers CLI flag (operator override)
  2. config/parallax/model_catalog.json (canonical catalog)
  3. Error — operator must pass --layers

The model manifest is signed by the local operator's NodeIdentity
(read from PRSM_NODE_IDENTITY_PATH or ~/.prsm/identity.json).

Idempotent: skips re-registration if the manifest already exists at
``<registry_root>/<model_id>/manifest.json`` AND its publisher node_id
matches the current operator (prevents one operator overwriting
another operator's signed manifest).

Exit codes:
  0 — registered (or already present + signed by us)
  1 — bad args / missing identity / signing failure
  2 — already registered by a different publisher (refuse to overwrite)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


DEFAULT_REGISTRY_ROOT = Path("/var/lib/prsm-registry")
DEFAULT_IDENTITY_PATH = Path.home() / ".prsm" / "identity.json"
DEFAULT_CATALOG_PATH = (
    Path(__file__).parent.parent
    / "config" / "parallax" / "model_catalog.json"
)


def _resolve_num_layers(
    model_id: str,
    cli_layers: Optional[int],
    catalog_path: Path,
) -> int:
    """Resolve num_layers from CLI flag → catalog → error."""
    if cli_layers is not None:
        if cli_layers < 1:
            raise ValueError(
                f"--layers must be >= 1, got {cli_layers}",
            )
        return cli_layers
    if not catalog_path.exists():
        raise ValueError(
            f"--layers not passed and catalog at {catalog_path} "
            "does not exist. Pass --layers explicitly."
        )
    try:
        catalog = json.loads(catalog_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"failed to read catalog {catalog_path}: {exc}",
        )
    models = catalog.get("models") or {}
    entry = models.get(model_id)
    if entry is None:
        raise ValueError(
            f"model_id {model_id!r} not in catalog {catalog_path}; "
            f"pass --layers explicitly or add a catalog entry first. "
            f"Available: {sorted(models.keys())}"
        )
    n = entry.get("num_layers")
    if not isinstance(n, int) or n < 1:
        raise ValueError(
            f"catalog entry for {model_id!r} has invalid num_layers "
            f"{n!r}; pass --layers explicitly"
        )
    return n


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage an HF model in the local PRSM model registry."
        ),
    )
    parser.add_argument("model_id", help="HF model id (e.g. distilgpt2)")
    parser.add_argument(
        "--layers", type=int, default=None,
        help=(
            "Override num_layers. Default: read from canonical "
            "model_catalog.json."
        ),
    )
    parser.add_argument(
        "--registry-root", default=str(DEFAULT_REGISTRY_ROOT),
        help=f"Registry root dir (default: {DEFAULT_REGISTRY_ROOT})",
    )
    parser.add_argument(
        "--identity-file", default=str(DEFAULT_IDENTITY_PATH),
        help=(
            "Path to operator's identity.json "
            f"(default: {DEFAULT_IDENTITY_PATH})"
        ),
    )
    parser.add_argument(
        "--catalog", default=str(DEFAULT_CATALOG_PATH),
        help=f"Catalog path (default: {DEFAULT_CATALOG_PATH})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Re-register even if manifest already present + signed "
            "by us. Default skips (idempotent)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        num_layers = _resolve_num_layers(
            args.model_id, args.layers, Path(args.catalog),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    registry_root = Path(args.registry_root)
    identity_path = Path(args.identity_file)

    if not identity_path.exists():
        print(
            f"ERROR: identity file {identity_path} does not exist. "
            "Either run `prsm node start` once to generate it, or "
            "pass --identity-file pointing at the operator's identity.",
            file=sys.stderr,
        )
        return 1

    # Imports deferred so --help works without PRSM env.
    from prsm.node.identity import load_node_identity
    from prsm.compute.model_registry.registry import (
        FilesystemModelRegistry,
    )
    from prsm.compute.model_sharding.models import (
        ShardedModel, ModelShard,
    )

    identity = load_node_identity(identity_path)
    print(f"Operator (publisher) node_id: {identity.node_id}")
    print(f"Registry root: {registry_root}")
    print(f"Model: {args.model_id} (num_layers={num_layers})")

    # Idempotency check — refuse to clobber another publisher's manifest.
    manifest_path = registry_root / args.model_id / "manifest.json"
    if manifest_path.exists() and not args.force:
        try:
            existing = json.loads(manifest_path.read_text())
            existing_publisher = existing.get("publisher_node_id", "")
            if existing_publisher == identity.node_id:
                print(
                    f"Already registered by this operator (skipping). "
                    f"Use --force to re-register."
                )
                return 0
            print(
                f"REFUSING TO CLOBBER: manifest at {manifest_path} "
                f"is signed by a different publisher "
                f"({existing_publisher[:16]}...). Use --force only "
                f"if you intend to overwrite.",
                file=sys.stderr,
            )
            return 2
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"WARNING: existing manifest at {manifest_path} "
                f"unreadable ({exc}); proceeding to overwrite.",
                file=sys.stderr,
            )

    registry_root.mkdir(parents=True, exist_ok=True)
    registry = FilesystemModelRegistry(root=registry_root)

    # Single-shard sentinel covering all layers. HF runner ignores
    # tensor_data + uses transformers.from_pretrained() at runtime.
    # sp1215 — the shard_id is a filesystem-mapped identifier (a filename in shards/),
    # so it must stay slash-free even though sp1214 lets the model_id be an HF org/model
    # id. Sanitize the model_id portion ('/'->'_'); the id is just a stored, read-back
    # label so this is self-consistent.
    shard = ModelShard(
        shard_id=f"{args.model_id.replace('/', '_')}-shard-0",
        model_id=args.model_id,
        shard_index=0,
        total_shards=1,
        tensor_data=b"\x00" * 32,
        tensor_shape=(32,),
        layer_range=(0, num_layers),
        size_bytes=32,
    )
    model = ShardedModel(
        model_id=args.model_id,
        model_name=f"HF runner sentinel for {args.model_id}",
        total_shards=1,
        shards=[shard],
    )

    try:
        manifest = registry.register(model, identity=identity)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: registry.register failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nRegistered:")
    print(f"  model_id:           {manifest.model_id}")
    print(f"  publisher_node_id:  {manifest.publisher_node_id}")
    print(f"  layer_range:        (0, {num_layers})")
    print(f"  shards:             {len(manifest.shards)}")
    print(f"  signature:          {manifest.publisher_signature[:32]}...")

    # Roundtrip sanity check.
    loaded = registry.get(args.model_id)
    if loaded is None:
        print(
            "WARNING: registry.get returned None after register; "
            "registry may be inconsistent",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nRoundtrip OK: registry.get({args.model_id!r}) → "
        f"ShardedModel(shards={len(loaded.shards)}, "
        f"layer_range={loaded.shards[0].layer_range})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
