"""Sprint 627 — ManifestShardEntry.layer_range persistence.

Sprint 626 live-test surfaced: ShardedModel.shards[].layer_range is
LOST when the model is persisted to a manifest, then defaults to the
(0,0) sentinel when reloaded via registry.get(). This causes
_is_final_stage(model, layer_range) in chain_rpc/server.py to return
False for what should be a final-stage request → LayerStageServer
skips the LM-head application → wire returns raw hidden_states
instead of logits.

Sprint 627 fixes the persistence:
  - Add layer_range field to ManifestShardEntry (default (0,0) for
    back-compat)
  - Update to_dict/from_dict to round-trip the field
  - Include layer_range in ModelManifest.signing_payload with
    omit-when-default semantics (old manifests with (0,0) still
    verify byte-identical to pre-627)
  - manifest_from_model populates layer_range from the source shard
  - FilesystemModelRegistry.get() reconstructs shards with the
    manifest's layer_range, not hardcoded (0,0)

After sprint 627, droplet's _is_final_stage() correctly returns
True for the gpt2 shim's layer_range=(0,12) request, lm_head + ln_f
get applied server-side, sprint-626 Mac-side workaround is no longer
needed.
"""
from __future__ import annotations

from pathlib import Path


def test_manifest_shard_entry_has_layer_range_field():
    from prsm.compute.model_registry.models import ManifestShardEntry
    e = ManifestShardEntry(
        shard_id="s", shard_index=0, tensor_shape=(1,),
        sha256="0" * 64, size_bytes=1,
    )
    assert hasattr(e, "layer_range")
    assert e.layer_range == (0, 0)  # default = back-compat sentinel


def test_manifest_shard_entry_accepts_explicit_layer_range():
    from prsm.compute.model_registry.models import ManifestShardEntry
    e = ManifestShardEntry(
        shard_id="s", shard_index=0, tensor_shape=(1,),
        sha256="0" * 64, size_bytes=1, layer_range=(3, 12),
    )
    assert e.layer_range == (3, 12)


def test_to_dict_includes_layer_range():
    from prsm.compute.model_registry.models import ManifestShardEntry
    e = ManifestShardEntry(
        shard_id="s", shard_index=0, tensor_shape=(1,),
        sha256="0" * 64, size_bytes=1, layer_range=(0, 12),
    )
    d = e.to_dict()
    assert d.get("layer_range") == [0, 12]


def test_from_dict_round_trips_layer_range():
    from prsm.compute.model_registry.models import ManifestShardEntry
    d = {
        "shard_id": "s", "shard_index": 0, "tensor_shape": [1],
        "sha256": "0" * 64, "size_bytes": 1, "layer_range": [3, 12],
    }
    e = ManifestShardEntry.from_dict(d)
    assert e.layer_range == (3, 12)


def test_from_dict_missing_layer_range_defaults_to_sentinel():
    """Existing pre-627 manifests have no layer_range field → must
    parse cleanly with default (0,0)."""
    from prsm.compute.model_registry.models import ManifestShardEntry
    d = {
        "shard_id": "s", "shard_index": 0, "tensor_shape": [1],
        "sha256": "0" * 64, "size_bytes": 1,
    }
    e = ManifestShardEntry.from_dict(d)
    assert e.layer_range == (0, 0)


def test_signing_payload_omits_layer_range_when_default():
    """Back-compat invariant: a manifest where ALL shards have
    layer_range=(0,0) must produce a byte-identical signing payload
    to the pre-627 schema (no layer_range in the per-shard line).
    Existing signed manifests must still verify post-upgrade.
    """
    from prsm.compute.model_registry.models import (
        ManifestShardEntry, ModelManifest,
    )
    shard = ManifestShardEntry(
        shard_id="s0", shard_index=0, tensor_shape=(1,),
        sha256="a" * 64, size_bytes=1,
        # layer_range defaults to (0, 0)
    )
    manifest = ModelManifest(
        model_id="m", model_name="M",
        publisher_node_id="n",
        total_shards=1, shards=(shard,),
        published_at=12345.0,
    )
    payload = manifest.signing_payload().decode()
    # The shard line must NOT contain "0,1" or any layer-range syntax
    # since the value is the default sentinel
    shard_line = payload.split("\n")[-1]
    assert "0:s0:" in shard_line
    # Pre-627 format: <idx>:<id>:<sha>:<size>:<shape>
    # Post-627 format with omit-when-default = same as pre-627
    # If layer_range is appended, the line ends with the shape, not "/0,0"
    assert not shard_line.endswith(":0,0")
    assert "layer_range" not in shard_line


def test_signing_payload_includes_layer_range_when_non_default():
    """When layer_range != (0,0), it MUST be in the signing payload —
    otherwise signature would not commit to the layer mapping (the
    bug that sprint 626 surfaced)."""
    from prsm.compute.model_registry.models import (
        ManifestShardEntry, ModelManifest,
    )
    shard = ManifestShardEntry(
        shard_id="s0", shard_index=0, tensor_shape=(1,),
        sha256="a" * 64, size_bytes=1, layer_range=(0, 12),
    )
    manifest = ModelManifest(
        model_id="m", model_name="M", publisher_node_id="n",
        total_shards=1, shards=(shard,), published_at=12345.0,
    )
    payload = manifest.signing_payload().decode()
    # Either "0:12" or "(0,12)" or "0,12" — any encoding that
    # commits to the layer_range
    assert "0" in payload and "12" in payload
    shard_line = payload.split("\n")[-1]
    # layer_range encoding must be present somewhere in the line
    # past the shape field
    assert "0,12" in shard_line or "0-12" in shard_line or "[0,12]" in shard_line


def test_manifest_from_model_propagates_layer_range():
    """manifest_from_model in registry.py must read each shard's
    layer_range and put it on the ManifestShardEntry — without
    this, the (0,0) loss persists."""
    from prsm.compute.model_registry.registry import manifest_from_model
    from prsm.compute.model_sharding.models import ShardedModel, ModelShard

    shard = ModelShard(
        shard_id="x", model_id="m", shard_index=0, total_shards=1,
        tensor_data=b"\x00" * 4, tensor_shape=(4,),
        layer_range=(3, 7), size_bytes=4,
    )
    model = ShardedModel(
        model_id="m", model_name="M", total_shards=1, shards=[shard],
    )
    manifest = manifest_from_model(model, publisher_node_id="n")
    assert manifest.shards[0].layer_range == (3, 7)


def test_filesystem_registry_get_round_trips_layer_range(
    tmp_path, prsm_home_with_identity,
):
    """End-to-end: register a model with layer_range=(0,12), get it
    back, verify the loaded shard's layer_range matches.

    sp1418 — the identity comes from the prsm_home_with_identity fixture.
    This used to do `load_node_identity(NodeConfig.load().identity_path)`,
    i.e. read the DEVELOPER'S real ~/.prsm: green on any box that had run
    `prsm setup`, and on a clean CI runner it got identity=None and the
    round-trip blew up. The registry's signing path is what's under test
    here, not identity discovery — so hand it a real, isolated identity.
    """
    from prsm.compute.model_registry.registry import FilesystemModelRegistry
    from prsm.compute.model_sharding.models import ShardedModel, ModelShard

    identity = prsm_home_with_identity
    root = tmp_path / "sprint627"
    root.mkdir(parents=True, exist_ok=True)
    registry = FilesystemModelRegistry(root=root)
    shard = ModelShard(
        shard_id="full", model_id="sprint627-roundtrip-test",
        shard_index=0, total_shards=1,
        tensor_data=b"\x01" * 16, tensor_shape=(16,),
        layer_range=(0, 12), size_bytes=16,
    )
    model = ShardedModel(
        model_id="sprint627-roundtrip-test", model_name="rt",
        total_shards=1, shards=[shard],
    )
    registry.register(model, identity=identity)
    loaded = registry.get("sprint627-roundtrip-test")
    assert loaded.shards[0].layer_range == (0, 12), (
        f"layer_range lost on round-trip: got {loaded.shards[0].layer_range}"
    )
