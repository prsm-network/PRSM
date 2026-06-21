"""
Unit tests — Phase 3.x.2 Task 3 — ModelRegistry ABC + InMemoryModelRegistry.

Acceptance per design plan §4 Task 3: drop-in replacement for the
Phase 3.x.1 dict-based pattern; the existing test_supported_models_lists_registry
shape passes against InMemoryModelRegistry.

Real ShardedModel + ModelShard with deterministic numpy bytes; real
NodeIdentity Ed25519 signing — no crypto/numerics mocks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from prsm.compute.model_registry import (
    FilesystemModelRegistry,
    InMemoryModelRegistry,
    ManifestVerificationError,
    ModelAlreadyRegisteredError,
    ModelNotFoundError,
    ModelRegistryError,
    manifest_from_model,
    sign_manifest,
)
from prsm.compute.model_registry.registry import _hash_shard
from prsm.compute.model_sharding.models import ModelShard, ShardedModel
from prsm.node.identity import NodeIdentity, generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fixtures — real ShardedModel + real NodeIdentity
# ──────────────────────────────────────────────────────────────────────────


def _make_shard(model_id: str, index: int, total: int, rows: int = 4, cols: int = 8) -> ModelShard:
    rng = np.random.default_rng(seed=2000 + index)
    tensor = rng.standard_normal(size=(rows, cols))
    data = tensor.tobytes()
    return ModelShard(
        shard_id=f"{model_id}-shard-{index}",
        model_id=model_id,
        shard_index=index,
        total_shards=total,
        tensor_data=data,
        tensor_shape=(rows, cols),
        layer_range=(0, 0),
        size_bytes=len(data),
        checksum=hashlib.sha256(data).hexdigest(),
    )


def _make_model(model_id: str = "test-llama", num_shards: int = 3) -> ShardedModel:
    shards = [_make_shard(model_id, i, num_shards) for i in range(num_shards)]
    return ShardedModel(
        model_id=model_id,
        model_name=f"{model_id} display",
        total_shards=num_shards,
        shards=shards,
    )


@pytest.fixture
def identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.2-task3-publisher")


@pytest.fixture
def other_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.2-task3-impostor")


@pytest.fixture
def registry() -> InMemoryModelRegistry:
    return InMemoryModelRegistry()


@pytest.fixture
def model() -> ShardedModel:
    return _make_model()


# ──────────────────────────────────────────────────────────────────────────
# Helper — manifest_from_model
# ──────────────────────────────────────────────────────────────────────────


class TestManifestFromModel:
    def test_builds_entry_per_shard(self, model):
        m = manifest_from_model(model, publisher_node_id="pub")
        assert len(m.shards) == model.total_shards
        assert m.total_shards == model.total_shards

    def test_sha256_commits_to_actual_bytes(self, model):
        m = manifest_from_model(model, publisher_node_id="pub")
        for shard, entry in zip(
            sorted(model.shards, key=lambda s: s.shard_index), m.shards
        ):
            assert entry.sha256 == hashlib.sha256(shard.tensor_data).hexdigest()
            assert entry.size_bytes == len(shard.tensor_data)

    def test_unsigned_by_default(self, model):
        m = manifest_from_model(model, publisher_node_id="pub")
        assert m.publisher_signature == b""

    def test_publisher_node_id_stamped(self, model):
        m = manifest_from_model(model, publisher_node_id="alice-node")
        assert m.publisher_node_id == "alice-node"

    def test_explicit_published_at_used(self, model):
        m = manifest_from_model(model, publisher_node_id="pub", published_at=42.0)
        assert m.published_at == 42.0


# ──────────────────────────────────────────────────────────────────────────
# register — write path
# ──────────────────────────────────────────────────────────────────────────


class TestRegister:
    def test_basic_register_returns_signed_manifest(self, registry, model, identity):
        manifest = registry.register(model, identity=identity)
        assert manifest.model_id == model.model_id
        assert manifest.publisher_node_id == identity.node_id
        assert len(manifest.publisher_signature) == 64

    def test_register_stores_model(self, registry, model, identity):
        registry.register(model, identity=identity)
        assert model.model_id in registry.list_models()

    def test_duplicate_register_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        with pytest.raises(ModelAlreadyRegisteredError, match=model.model_id):
            registry.register(model, identity=identity)

    def test_duplicate_register_blocks_different_publisher(
        self, registry, model, identity, other_identity
    ):
        # First-write-wins per node — even a different publisher can't
        # overwrite the existing registration in this registry.
        registry.register(model, identity=identity)
        with pytest.raises(ModelAlreadyRegisteredError):
            registry.register(model, identity=other_identity)

    def test_multiple_models_can_register(self, registry, identity):
        m1 = _make_model("m1")
        m2 = _make_model("m2")
        registry.register(m1, identity=identity)
        registry.register(m2, identity=identity)
        assert sorted(registry.list_models()) == ["m1", "m2"]

    def test_register_signs_under_supplied_identity(
        self, registry, model, identity, other_identity
    ):
        # Two different identities → distinct signatures
        m1 = _make_model("m1")
        m2 = _make_model("m2")
        s1 = registry.register(m1, identity=identity)
        s2 = registry.register(m2, identity=other_identity)
        assert s1.publisher_node_id != s2.publisher_node_id


# ──────────────────────────────────────────────────────────────────────────
# get — read path with verification
# ──────────────────────────────────────────────────────────────────────────


class TestGet:
    def test_get_returns_same_model(self, registry, model, identity):
        registry.register(model, identity=identity)
        retrieved = registry.get(model.model_id)
        # Same shard contents (object equality not required; byte-equality is).
        assert retrieved.model_id == model.model_id
        assert len(retrieved.shards) == len(model.shards)
        for s_in, s_out in zip(
            sorted(model.shards, key=lambda s: s.shard_index),
            sorted(retrieved.shards, key=lambda s: s.shard_index),
        ):
            assert s_in.tensor_data == s_out.tensor_data

    def test_get_unknown_raises_not_found(self, registry):
        with pytest.raises(ModelNotFoundError, match="not-registered"):
            registry.get("not-registered")

    def test_get_after_signature_tamper_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        # Reach into private state to flip a byte in the stored signature
        signed = registry._manifests[model.model_id]
        bad_sig = bytes(
            [signed.publisher_signature[0] ^ 0xFF]
        ) + signed.publisher_signature[1:]
        registry._manifests[model.model_id] = dataclasses.replace(
            signed, publisher_signature=bad_sig
        )
        with pytest.raises(ManifestVerificationError, match="signature"):
            registry.get(model.model_id)

    def test_get_after_shard_byte_tamper_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        # Flip a byte in the stored shard's tensor_data
        stored = registry._models[model.model_id]
        s0 = stored.shards[0]
        s0_tampered = ModelShard(
            shard_id=s0.shard_id, model_id=s0.model_id,
            shard_index=s0.shard_index, total_shards=s0.total_shards,
            tensor_data=bytes([s0.tensor_data[0] ^ 0xFF]) + s0.tensor_data[1:],
            tensor_shape=s0.tensor_shape, layer_range=s0.layer_range,
            size_bytes=s0.size_bytes, checksum=s0.checksum,
        )
        stored.shards[0] = s0_tampered
        with pytest.raises(ManifestVerificationError, match="sha256 mismatch"):
            registry.get(model.model_id)

    def test_get_after_shard_size_tamper_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        stored = registry._models[model.model_id]
        s0 = stored.shards[0]
        # Truncate tensor_data — sha256 also changes, so the sha256
        # check fires first. Construct a case where size is wrong but
        # sha256 happens to match by truncating zero-bytes — easier to
        # just check that truncation raises (whichever path).
        s0_short = ModelShard(
            shard_id=s0.shard_id, model_id=s0.model_id,
            shard_index=s0.shard_index, total_shards=s0.total_shards,
            tensor_data=s0.tensor_data[:-1],
            tensor_shape=s0.tensor_shape, layer_range=s0.layer_range,
            size_bytes=s0.size_bytes, checksum=s0.checksum,
        )
        stored.shards[0] = s0_short
        with pytest.raises(ManifestVerificationError):
            registry.get(model.model_id)

    def test_get_after_shard_removal_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        stored = registry._models[model.model_id]
        # Drop a shard from the model (but not the manifest)
        del stored.shards[-1]
        with pytest.raises(ManifestVerificationError, match="shards"):
            registry.get(model.model_id)

    def test_get_after_shard_index_swap_raises(self, registry, model, identity):
        # Two shards with swapped indices but original bytes → sha256
        # check on each slot fails because slot 0 now holds shard-1
        # bytes whose sha256 doesn't match slot 0's manifest entry.
        registry.register(model, identity=identity)
        stored = registry._models[model.model_id]
        s0, s1 = stored.shards[0], stored.shards[1]
        stored.shards[0] = ModelShard(
            shard_id=s0.shard_id, model_id=s0.model_id,
            shard_index=0, total_shards=s0.total_shards,
            tensor_data=s1.tensor_data,
            tensor_shape=s1.tensor_shape, layer_range=s1.layer_range,
            size_bytes=s1.size_bytes, checksum=s1.checksum,
        )
        with pytest.raises(ManifestVerificationError):
            registry.get(model.model_id)


# ──────────────────────────────────────────────────────────────────────────
# list_models / get_manifest
# ──────────────────────────────────────────────────────────────────────────


class TestListAndGetManifest:
    def test_list_empty(self, registry):
        assert registry.list_models() == []

    def test_list_sorted(self, registry, identity):
        registry.register(_make_model("zz"), identity=identity)
        registry.register(_make_model("aa"), identity=identity)
        registry.register(_make_model("mm"), identity=identity)
        assert registry.list_models() == ["aa", "mm", "zz"]

    def test_get_manifest_returns_signed_manifest(self, registry, model, identity):
        registry.register(model, identity=identity)
        m = registry.get_manifest(model.model_id)
        assert m.model_id == model.model_id
        assert len(m.publisher_signature) == 64

    def test_get_manifest_unknown_raises(self, registry):
        with pytest.raises(ModelNotFoundError):
            registry.get_manifest("missing")


# ──────────────────────────────────────────────────────────────────────────
# verify — audit-only
# ──────────────────────────────────────────────────────────────────────────


class TestVerify:
    def test_verify_passes_for_clean_registration(self, registry, model, identity):
        registry.register(model, identity=identity)
        assert registry.verify(model.model_id) is True

    def test_verify_returns_false_for_unknown(self, registry):
        assert registry.verify("nope") is False

    def test_verify_returns_false_after_tamper(self, registry, model, identity):
        registry.register(model, identity=identity)
        registry._models[model.model_id].shards[0] = ModelShard(
            shard_id="x", model_id=model.model_id,
            shard_index=0, total_shards=model.total_shards,
            tensor_data=b"different bytes",
            tensor_shape=(1,), layer_range=(0, 0),
            size_bytes=15, checksum="",
        )
        assert registry.verify(model.model_id) is False

    def test_verify_does_not_swallow_unknown_errors(self, registry, model, identity):
        # If the registry raises something OTHER than the documented
        # exception types, verify must NOT silently turn it into False
        # — that would mask bugs.
        registry.register(model, identity=identity)

        original_get = registry.get
        def boom(model_id):
            raise RuntimeError("registry corrupted")
        registry.get = boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="corrupted"):
            registry.verify(model.model_id)
        registry.get = original_get  # type: ignore[method-assign]


# ──────────────────────────────────────────────────────────────────────────
# Drop-in for the Phase 3.x.1 dict pattern (acceptance per §4 Task 3)
# ──────────────────────────────────────────────────────────────────────────


class TestPhase3x1DropIn:
    def test_supported_models_lists_registry_shape(self, registry, identity):
        # The Phase 3.x.1 executor's TestConstruction::test_supported_models_lists_registry
        # asserts ex.supported_models() == sorted(model_ids). The registry's
        # list_models() must produce the same shape.
        registry.register(_make_model("test-llama"), identity=identity)
        assert registry.list_models() == ["test-llama"]

    def test_register_and_get_round_trip(self, registry, model, identity):
        # Mirrors the Phase 3.x.1 dict pattern: registered models
        # round-trip through get() unchanged.
        registry.register(model, identity=identity)
        out = registry.get(model.model_id)
        assert out.model_id == model.model_id
        assert out.total_shards == model.total_shards


# ──────────────────────────────────────────────────────────────────────────
# Exception hierarchy contract
# ──────────────────────────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        assert issubclass(ModelNotFoundError, ModelRegistryError)
        assert issubclass(ModelAlreadyRegisteredError, ModelRegistryError)
        assert issubclass(ManifestVerificationError, ModelRegistryError)

    def test_all_distinguishable(self):
        # Distinct types so callers can `except ModelNotFoundError`
        # without catching tampering errors.
        assert ModelNotFoundError is not ManifestVerificationError
        assert ModelAlreadyRegisteredError is not ManifestVerificationError


# ──────────────────────────────────────────────────────────────────────────
# Internal _hash_shard helper — pinned so future "optimization" can't
# silently desync from the manifest writer
# ──────────────────────────────────────────────────────────────────────────


class TestHashShard:
    def test_matches_sha256_of_tensor_data(self):
        shard = _make_shard("m", 0, 1)
        assert _hash_shard(shard) == hashlib.sha256(shard.tensor_data).hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# FilesystemModelRegistry
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fs_registry(tmp_path):
    return FilesystemModelRegistry(tmp_path)


class TestFilesystemRegistryConstruction:
    def test_missing_root_raises(self, tmp_path):
        bogus = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            FilesystemModelRegistry(bogus)

    def test_root_is_file_raises(self, tmp_path):
        f = tmp_path / "im-a-file"
        f.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            FilesystemModelRegistry(f)

    def test_accepts_str_path(self, tmp_path):
        # Construction accepts both Path and str
        reg = FilesystemModelRegistry(str(tmp_path))
        assert reg.list_models() == []

    def test_empty_root_lists_no_models(self, fs_registry):
        assert fs_registry.list_models() == []


class TestFilesystemRegistryRegister:
    def test_register_writes_manifest_pubkey_shards(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        model_dir = tmp_path / model.model_id
        assert (model_dir / "manifest.json").exists()
        assert (model_dir / "publisher.pubkey").exists()
        shards_dir = model_dir / "shards"
        assert shards_dir.exists()
        for shard in model.shards:
            assert (shards_dir / f"{shard.shard_id}.bin").exists()

    def test_register_returns_signed_manifest(self, fs_registry, model, identity):
        manifest = fs_registry.register(model, identity=identity)
        assert manifest.publisher_node_id == identity.node_id
        assert len(manifest.publisher_signature) == 64

    def test_register_writes_canonical_json(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        manifest_path = tmp_path / model.model_id / "manifest.json"
        text = manifest_path.read_text()
        # sort_keys=True means re-loading then re-dumping with the same
        # flag must produce identical bytes.
        loaded = json.loads(text)
        assert json.dumps(loaded, sort_keys=True, indent=2) == text

    def test_pubkey_sidecar_holds_b64(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        text = (tmp_path / model.model_id / "publisher.pubkey").read_text().strip()
        assert text == identity.public_key_b64

    def test_shard_files_hold_raw_bytes(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        for shard in model.shards:
            data = (tmp_path / model.model_id / "shards" / f"{shard.shard_id}.bin").read_bytes()
            assert data == shard.tensor_data

    def test_duplicate_register_raises(self, fs_registry, model, identity):
        fs_registry.register(model, identity=identity)
        with pytest.raises(ModelAlreadyRegisteredError):
            fs_registry.register(model, identity=identity)

    def test_unsafe_model_id_rejected(self, fs_registry, identity):
        bad = ShardedModel(
            model_id="../escape",
            model_name="bad",
            total_shards=1,
            shards=[_make_shard("../escape", 0, 1)],
        )
        with pytest.raises(ValueError, match="model_id"):
            fs_registry.register(bad, identity=identity)

    def test_unsafe_shard_id_rejected(self, fs_registry, identity):
        good_model_id = "ok-model"
        bad_shard = _make_shard(good_model_id, 0, 1)
        bad_shard.shard_id = "/etc/passwd"
        m = ShardedModel(
            model_id=good_model_id, model_name="ok", total_shards=1, shards=[bad_shard]
        )
        with pytest.raises(ValueError, match="shard_id"):
            fs_registry.register(m, identity=identity)

    def test_dotdot_model_id_rejected(self, fs_registry, identity, tmp_path):
        # Round 2 review HIGH finding: the regex [A-Za-z0-9._-]+
        # accepts the bare string ".." (each char is in the allowlist).
        # Rejected explicitly via _RESERVED_FS_NAMES + defense-in-depth
        # is_relative_to(root) check in _model_dir.
        bad_model = ShardedModel(
            model_id="..",
            model_name="escape attempt",
            total_shards=1,
            shards=[_make_shard("..", 0, 1)],
        )
        with pytest.raises(ValueError, match="reserved|escapes|unsafe"):
            fs_registry.register(bad_model, identity=identity)
        # Verify nothing was written outside the registry root
        parent = tmp_path.parent
        for entry in parent.iterdir():
            assert entry == tmp_path or "manifest.json" not in str(entry)

    def test_dot_model_id_rejected(self, fs_registry, identity):
        # Bare "." would map to <root>/. (current dir). Same defense.
        bad_model = ShardedModel(
            model_id=".",
            model_name="escape attempt",
            total_shards=1,
            shards=[_make_shard(".", 0, 1)],
        )
        with pytest.raises(ValueError, match="reserved|escapes|unsafe"):
            fs_registry.register(bad_model, identity=identity)

    def test_dotdot_shard_id_rejected(self, fs_registry, identity):
        # Same finding applies to shard_id — a malicious shard_id of
        # ".." would escape the model's shards/ subdirectory.
        bad_shard = _make_shard("ok-model", 0, 1)
        bad_shard.shard_id = ".."
        m = ShardedModel(
            model_id="ok-model", model_name="ok", total_shards=1, shards=[bad_shard]
        )
        with pytest.raises(ValueError, match="reserved|shard_id"):
            fs_registry.register(m, identity=identity)

    def test_get_with_dotdot_rejected(self, fs_registry):
        # The read path also runs _validate_fs_id, so bare ".." raises
        # before any disk access.
        with pytest.raises(ValueError, match="reserved|unsafe"):
            fs_registry.get("..")


class TestFilesystemRegistryGet:
    def test_get_returns_byte_identical_model(
        self, fs_registry, model, identity
    ):
        fs_registry.register(model, identity=identity)
        out = fs_registry.get(model.model_id)
        assert out.model_id == model.model_id
        assert out.total_shards == model.total_shards
        for s_in, s_out in zip(
            sorted(model.shards, key=lambda s: s.shard_index),
            sorted(out.shards, key=lambda s: s.shard_index),
        ):
            assert s_in.tensor_data == s_out.tensor_data
            assert s_in.tensor_shape == s_out.tensor_shape

    def test_get_unknown_raises(self, fs_registry):
        with pytest.raises(ModelNotFoundError):
            fs_registry.get("does-not-exist")

    def test_unsafe_model_id_rejected(self, fs_registry):
        with pytest.raises(ValueError, match="model_id"):
            fs_registry.get("../escape")

    def test_shard_bit_flip_detected(
        self, fs_registry, model, identity, tmp_path
    ):
        # Register, then mutate a single byte in a shard's .bin file.
        fs_registry.register(model, identity=identity)
        shard_path = (
            tmp_path / model.model_id / "shards" /
            f"{model.shards[0].shard_id}.bin"
        )
        data = bytearray(shard_path.read_bytes())
        data[0] ^= 0xFF
        shard_path.write_bytes(bytes(data))
        with pytest.raises(ManifestVerificationError, match="sha256 mismatch"):
            fs_registry.get(model.model_id)

    def test_missing_shard_file_detected(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        shard_path = (
            tmp_path / model.model_id / "shards" /
            f"{model.shards[0].shard_id}.bin"
        )
        shard_path.unlink()
        with pytest.raises(ManifestVerificationError, match="missing"):
            fs_registry.get(model.model_id)

    def test_shard_truncation_detected(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        shard_path = (
            tmp_path / model.model_id / "shards" /
            f"{model.shards[0].shard_id}.bin"
        )
        truncated = shard_path.read_bytes()[:-1]
        shard_path.write_bytes(truncated)
        with pytest.raises(ManifestVerificationError):
            fs_registry.get(model.model_id)

    def test_signature_tamper_in_manifest_detected(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        manifest_path = tmp_path / model.model_id / "manifest.json"
        data = json.loads(manifest_path.read_text())
        # Flip a bit in the hex signature
        sig = data["publisher_signature"]
        flipped_first_byte = (int(sig[:2], 16) ^ 0xFF)
        data["publisher_signature"] = f"{flipped_first_byte:02x}" + sig[2:]
        manifest_path.write_text(json.dumps(data, sort_keys=True, indent=2))
        with pytest.raises(ManifestVerificationError, match="signature"):
            fs_registry.get(model.model_id)

    def test_missing_pubkey_sidecar_detected(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        (tmp_path / model.model_id / "publisher.pubkey").unlink()
        with pytest.raises(ManifestVerificationError, match="publisher.pubkey"):
            fs_registry.get(model.model_id)

    def test_corrupt_manifest_json_detected(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        (tmp_path / model.model_id / "manifest.json").write_text("not valid json {{")
        with pytest.raises(ManifestVerificationError, match="unreadable|corrupt"):
            fs_registry.get(model.model_id)

    def test_manifest_schema_corruption_detected(
        self, fs_registry, model, identity, tmp_path
    ):
        # Valid JSON but missing required field
        fs_registry.register(model, identity=identity)
        manifest_path = tmp_path / model.model_id / "manifest.json"
        manifest_path.write_text(json.dumps({"model_id": "x"}))
        with pytest.raises(ManifestVerificationError, match="schema"):
            fs_registry.get(model.model_id)


class TestFilesystemRegistryListAndManifest:
    def test_list_returns_only_dirs_with_manifest(
        self, fs_registry, model, identity, tmp_path
    ):
        fs_registry.register(model, identity=identity)
        # Stray dir without manifest.json must be ignored.
        (tmp_path / "stray-dir").mkdir()
        # Stray file at root must be ignored.
        (tmp_path / "stray.txt").write_text("x")
        # Directory whose name violates safe-id rules must be ignored
        # (spaces aren't in the [A-Za-z0-9._-] allowlist).
        (tmp_path / "has space").mkdir()
        (tmp_path / "has space" / "manifest.json").write_text("{}")
        assert fs_registry.list_models() == [model.model_id]

    def test_list_sorted(self, fs_registry, identity):
        for mid in ["zz", "aa", "mm"]:
            fs_registry.register(_make_model(mid), identity=identity)
        assert fs_registry.list_models() == ["aa", "mm", "zz"]

    def test_get_manifest_works_independently_of_get(
        self, fs_registry, model, identity, tmp_path
    ):
        # Register, delete a shard file (so get() would fail), but
        # get_manifest() should still work — it doesn't touch shards.
        fs_registry.register(model, identity=identity)
        (tmp_path / model.model_id / "shards" /
         f"{model.shards[0].shard_id}.bin").unlink()
        # get_manifest succeeds (metadata only)
        m = fs_registry.get_manifest(model.model_id)
        assert m.model_id == model.model_id
        # get fails (shard byte verification)
        with pytest.raises(ManifestVerificationError):
            fs_registry.get(model.model_id)


class TestFilesystemRegistryRestartSimulation:
    """Two registry instances pointing at the same root must see each
    other's writes after they hit the filesystem — this is the whole
    point of persistence."""

    def test_register_then_get_via_fresh_instance(
        self, tmp_path, model, identity
    ):
        # Instance A writes
        a = FilesystemModelRegistry(tmp_path)
        a.register(model, identity=identity)
        # Instance B (simulating restart) reads
        b = FilesystemModelRegistry(tmp_path)
        out = b.get(model.model_id)
        assert out.model_id == model.model_id
        for s_in, s_out in zip(
            sorted(model.shards, key=lambda s: s.shard_index),
            sorted(out.shards, key=lambda s: s.shard_index),
        ):
            assert s_in.tensor_data == s_out.tensor_data

    def test_list_models_seen_across_instances(self, tmp_path, identity):
        a = FilesystemModelRegistry(tmp_path)
        a.register(_make_model("first"), identity=identity)
        a.register(_make_model("second"), identity=identity)
        b = FilesystemModelRegistry(tmp_path)
        assert b.list_models() == ["first", "second"]

    def test_verify_across_instances(self, tmp_path, model, identity):
        a = FilesystemModelRegistry(tmp_path)
        a.register(model, identity=identity)
        b = FilesystemModelRegistry(tmp_path)
        assert b.verify(model.model_id) is True

    def test_duplicate_register_blocked_across_instances(
        self, tmp_path, model, identity
    ):
        a = FilesystemModelRegistry(tmp_path)
        a.register(model, identity=identity)
        b = FilesystemModelRegistry(tmp_path)
        with pytest.raises(ModelAlreadyRegisteredError):
            b.register(model, identity=identity)


class TestFilesystemRegistryAnchorIntegration:
    """Phase 3.x.3 Task 5 — anchor= kwarg routes signature-verify
    through the on-chain anchor instead of the local sidecar.
    Closes the cross-node trust-boundary caveat from Phase 3.x.2.
    """

    @pytest.fixture
    def fake_anchor(self, identity):
        """Mock anchor returning identity.public_key_b64 for the
        registered publisher's node_id; None for unregistered."""
        from unittest.mock import MagicMock
        anchor = MagicMock()

        def _lookup(node_id):
            if node_id == identity.node_id:
                return identity.public_key_b64
            return None

        anchor.lookup = MagicMock(side_effect=_lookup)
        return anchor

    def test_anchor_path_verifies_after_sidecar_tamper(
        self, tmp_path, identity, fake_anchor, model
    ):
        # Register without anchor; tamper the sidecar; reopen WITH
        # anchor — verification should still succeed because anchor
        # takes precedence over the sidecar.
        FilesystemModelRegistry(tmp_path).register(model, identity=identity)
        # Tamper sidecar with a bogus key — would fail sidecar-only verify
        (tmp_path / model.model_id / "publisher.pubkey").write_text(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        )

        anchor_reg = FilesystemModelRegistry(tmp_path, anchor=fake_anchor)
        out = anchor_reg.get(model.model_id)
        assert out.model_id == model.model_id
        # Anchor was consulted with the manifest's publisher_node_id
        fake_anchor.lookup.assert_called_with(identity.node_id)

    def test_no_anchor_uses_sidecar_unchanged(
        self, tmp_path, identity, model
    ):
        # Phase 3.x.2 back-compat: anchor=None is the default.
        # Sidecar-tampered registry MUST fail without anchor.
        FilesystemModelRegistry(tmp_path).register(model, identity=identity)
        (tmp_path / model.model_id / "publisher.pubkey").write_text(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        )
        reg = FilesystemModelRegistry(tmp_path)
        with pytest.raises(ManifestVerificationError):
            reg.get(model.model_id)

    def test_unregistered_publisher_raises(
        self, tmp_path, identity, model
    ):
        from unittest.mock import MagicMock
        # Anchor that always returns None (publisher not registered)
        empty_anchor = MagicMock()
        empty_anchor.lookup = MagicMock(return_value=None)

        FilesystemModelRegistry(tmp_path).register(model, identity=identity)
        reg = FilesystemModelRegistry(tmp_path, anchor=empty_anchor)
        with pytest.raises(ManifestVerificationError, match="not registered"):
            reg.get(model.model_id)

    def test_anchor_returning_wrong_key_fails(
        self, tmp_path, identity, other_identity, model
    ):
        # Anchor returns the WRONG key for the publisher.
        # verify_manifest must fail under that key → ManifestVerificationError.
        from unittest.mock import MagicMock
        wrong_anchor = MagicMock()
        wrong_anchor.lookup = MagicMock(return_value=other_identity.public_key_b64)

        FilesystemModelRegistry(tmp_path).register(model, identity=identity)
        reg = FilesystemModelRegistry(tmp_path, anchor=wrong_anchor)
        with pytest.raises(ManifestVerificationError, match="signature"):
            reg.get(model.model_id)

    def test_anchor_does_not_affect_register(
        self, tmp_path, identity, fake_anchor, model
    ):
        # Anchor only affects read paths. register() still writes the
        # sidecar (offline-verifier metadata).
        reg = FilesystemModelRegistry(tmp_path, anchor=fake_anchor)
        reg.register(model, identity=identity)
        sidecar = (tmp_path / model.model_id / "publisher.pubkey").read_text().strip()
        assert sidecar == identity.public_key_b64


class TestFilesystemRegistryParity:
    """Same interface tests must pass against FilesystemModelRegistry as
    against InMemoryModelRegistry. Acceptance per design plan §4 Task 4."""

    @pytest.fixture
    def fs_reg(self, tmp_path):
        return FilesystemModelRegistry(tmp_path)

    def test_supported_models_list_shape(self, fs_reg, identity):
        fs_reg.register(_make_model("test-llama"), identity=identity)
        assert fs_reg.list_models() == ["test-llama"]

    def test_round_trip_preserves_model(self, fs_reg, model, identity):
        fs_reg.register(model, identity=identity)
        out = fs_reg.get(model.model_id)
        assert out.total_shards == model.total_shards

    def test_verify_false_for_unknown(self, fs_reg):
        assert fs_reg.verify("nope") is False

    def test_verify_true_for_clean(self, fs_reg, model, identity):
        fs_reg.register(model, identity=identity)
        assert fs_reg.verify(model.model_id) is True

    def test_verify_false_after_tamper(
        self, fs_reg, model, identity, tmp_path
    ):
        fs_reg.register(model, identity=identity)
        shard_path = (
            tmp_path / model.model_id / "shards" /
            f"{model.shards[0].shard_id}.bin"
        )
        data = bytearray(shard_path.read_bytes())
        data[0] ^= 0xFF
        shard_path.write_bytes(bytes(data))
        assert fs_reg.verify(model.model_id) is False


class TestFilesystemRegistryDHTIntegration:
    """Phase 3.x.5 Task 5 — dht= kwarg.

    Hooks the registry into a manifest DHT so:
      - register() also announces on the DHT (best-effort; failures
        are logged but don't fail the registration).
      - get_manifest() / get() fall back to dht.get_manifest() when
        the local manifest is missing. The fetched manifest is
        anchor-verified by the DHT layer; cached locally so subsequent
        reads are fast.
    """

    @pytest.fixture
    def fake_anchor(self, identity):
        from unittest.mock import MagicMock
        anchor = MagicMock()
        anchor.lookup = MagicMock(
            side_effect=lambda nid: identity.public_key_b64
            if nid == identity.node_id else None
        )
        return anchor

    @pytest.fixture
    def fake_dht(self):
        """Mock DHT client exposing announce + get_manifest."""
        from unittest.mock import MagicMock
        return MagicMock()

    def test_dht_requires_anchor(self, tmp_path, fake_dht):
        with pytest.raises(RuntimeError, match="dht= requires anchor="):
            FilesystemModelRegistry(tmp_path, dht=fake_dht)

    def test_register_announces_on_dht(
        self, tmp_path, identity, fake_anchor, fake_dht, model
    ):
        reg = FilesystemModelRegistry(
            tmp_path, anchor=fake_anchor, dht=fake_dht
        )
        reg.register(model, identity=identity)

        # announce called with (model_id, manifest_path)
        assert fake_dht.announce.call_count == 1
        call_args = fake_dht.announce.call_args
        assert call_args.args[0] == model.model_id
        assert call_args.args[1] == (
            tmp_path / model.model_id / "manifest.json"
        )

    def test_register_succeeds_when_dht_announce_raises(
        self, tmp_path, identity, fake_anchor, fake_dht, model
    ):
        # DHT announce blowing up MUST NOT fail the registration —
        # the local registry is the write authority.
        fake_dht.announce = MagicMock(
            side_effect=RuntimeError("network down")
        )

        reg = FilesystemModelRegistry(
            tmp_path, anchor=fake_anchor, dht=fake_dht
        )
        signed = reg.register(model, identity=identity)
        assert signed.model_id == model.model_id
        # Local manifest is durable
        assert (tmp_path / model.model_id / "manifest.json").exists()

    def test_get_manifest_falls_back_to_dht(
        self, tmp_path, identity, fake_anchor, fake_dht, model
    ):
        # Build a real signed manifest "on another node" — since the
        # DHT layer guarantees anchor verification, our fake_dht just
        # returns the pre-built manifest as if the network had served it.
        unsigned = manifest_from_model(
            model, publisher_node_id=identity.node_id
        )
        signed = sign_manifest(unsigned, identity)
        fake_dht.get_manifest = MagicMock(return_value=signed)

        # Empty registry — no local manifest for this model_id.
        reg = FilesystemModelRegistry(
            tmp_path, anchor=fake_anchor, dht=fake_dht
        )

        out = reg.get_manifest(model.model_id)
        assert out.model_id == model.model_id
        fake_dht.get_manifest.assert_called_once_with(model.model_id)

        # Manifest is cached locally — second call doesn't re-fetch.
        cached_path = tmp_path / model.model_id / "manifest.json"
        assert cached_path.exists()

        fake_dht.get_manifest.reset_mock()
        out2 = reg.get_manifest(model.model_id)
        assert out2.model_id == model.model_id
        fake_dht.get_manifest.assert_not_called()

    def test_dht_lookup_failure_yields_model_not_found(
        self, tmp_path, fake_anchor, fake_dht
    ):
        from prsm.network.manifest_dht import ManifestNotFoundError
        fake_dht.get_manifest = MagicMock(
            side_effect=ManifestNotFoundError("no providers")
        )

        reg = FilesystemModelRegistry(
            tmp_path, anchor=fake_anchor, dht=fake_dht
        )
        with pytest.raises(ModelNotFoundError, match="DHT lookup"):
            reg.get_manifest("absent-model")

    def test_no_dht_no_fallback(self, tmp_path, fake_anchor):
        # Without dht=, behavior is unchanged from Phase 3.x.3 —
        # missing manifest → ModelNotFoundError, no network call.
        reg = FilesystemModelRegistry(tmp_path, anchor=fake_anchor)
        with pytest.raises(ModelNotFoundError):
            reg.get_manifest("absent-model")

    def test_get_falls_back_then_fails_at_shards(
        self, tmp_path, identity, fake_anchor, fake_dht, model
    ):
        # get() goes through _load_manifest_or_raise → DHT fallback
        # populates the manifest, but the shards aren't on disk
        # (DHT distributes manifests only, not shards). The downstream
        # shard-existence check raises ManifestVerificationError —
        # operator's signal that the model is announced but bytes
        # need to be obtained out-of-band.
        unsigned = manifest_from_model(
            model, publisher_node_id=identity.node_id
        )
        signed = sign_manifest(unsigned, identity)
        fake_dht.get_manifest = MagicMock(return_value=signed)

        reg = FilesystemModelRegistry(
            tmp_path, anchor=fake_anchor, dht=fake_dht
        )
        with pytest.raises(ManifestVerificationError, match="shard file missing"):
            reg.get(model.model_id)

    def test_dht_cached_manifest_reverifies_via_anchor(
        self, tmp_path, identity, fake_anchor, fake_dht, model
    ):
        # After caching a DHT-fetched manifest, a subsequent
        # get_manifest() reads the cached bytes and (when get() runs)
        # re-verifies them via the anchor. Since the registry's
        # anchor matches the DHT's anchor, this should succeed —
        # no sidecar is needed.
        unsigned = manifest_from_model(
            model, publisher_node_id=identity.node_id
        )
        signed = sign_manifest(unsigned, identity)
        fake_dht.get_manifest = MagicMock(return_value=signed)

        reg = FilesystemModelRegistry(
            tmp_path, anchor=fake_anchor, dht=fake_dht
        )
        # Trigger fallback + cache.
        reg.get_manifest(model.model_id)
        # Sidecar was NOT written — DHT-fetched manifest doesn't have one.
        sidecar = tmp_path / model.model_id / "publisher.pubkey"
        assert not sidecar.exists()
        # But anchor-based reads still work.
        out = reg.get_manifest(model.model_id)
        assert out.model_id == model.model_id

    def test_register_without_dht_unchanged(
        self, tmp_path, identity, fake_anchor, model
    ):
        # Sanity: anchor-only construction (no dht=) doesn't try to
        # announce. Phase 3.x.3 behavior is preserved.
        reg = FilesystemModelRegistry(tmp_path, anchor=fake_anchor)
        reg.register(model, identity=identity)
        # No announce attempted — there's no dht to announce on.
        assert (tmp_path / model.model_id / "manifest.json").exists()
