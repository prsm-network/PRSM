"""
Model registry — abstract interface + in-memory implementation.

Phase 3.x.2 Task 3.

The ABC pins the contract every backend (in-memory, filesystem,
future DHT, future on-chain anchor) must satisfy:

- ``register(model, identity)`` builds a signed ``ModelManifest`` from
  the model's actual bytes and stores it.
- ``get(model_id)`` returns the model only after BOTH cryptographic
  checks pass: (a) the manifest's publisher signature verifies, AND
  (b) every shard's actual ``tensor_data`` sha256 matches the manifest
  entry. Any mismatch raises ``ManifestVerificationError`` — fail closed.
- ``verify(model_id)`` is the audit-only variant; default impl wraps
  ``get`` and catches.

The in-memory implementation here is the drop-in replacement for the
``Dict[str, ShardedModel]`` that ``TensorParallelInferenceExecutor``
accepted in Phase 3.x.1; preserves test ergonomics and the dict-arg
back-compat path lands in Task 5.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prsm.compute.model_registry.models import (
    ManifestShardEntry,
    ModelManifest,
)
from prsm.compute.model_registry.signing import sign_manifest, verify_manifest
from prsm.compute.model_sharding.models import ModelShard, ShardedModel
from prsm.node.identity import NodeIdentity


logger = logging.getLogger(__name__)


# Strict allowlist for filesystem-mapped identifiers. Rejects path
# traversal (.. /), absolute paths, null bytes, and any character that
# would surprise a typical filesystem. Publishers picking model/shard
# ids that don't match this need to use the InMemoryRegistry or hash
# the id themselves before registration.
#
# IMPORTANT: the regex alone permits the bare strings "." and ".." (each
# is one or more characters from the allowlist). Both are illegal as
# directory names because they resolve to the parent or current dir.
# _validate_fs_id() rejects them explicitly; do not loosen this without
# the corresponding `path.is_relative_to(root)` defense-in-depth check.
_SAFE_FS_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_RESERVED_FS_NAMES = frozenset({".", ".."})


def _validate_fs_id(kind: str, value: str) -> None:
    """Reject identifiers unsafe for direct filesystem mapping."""
    if not value or not _SAFE_FS_ID.fullmatch(value):
        raise ValueError(
            f"{kind}={value!r} unsafe for filesystem registry: must match "
            f"{_SAFE_FS_ID.pattern} (got non-conforming characters)"
        )
    if value in _RESERVED_FS_NAMES:
        raise ValueError(
            f"{kind}={value!r} is a reserved filesystem name "
            f"(would resolve to current/parent dir and escape registry root)"
        )


def _validate_model_id(value: str) -> None:
    """Sprint 1214 — a model_id may be a HuggingFace ``org/model`` id (slash-separated,
    fs-safe segments), unlike shard/publisher ids which stay strict (no slash). Each
    ``/``-separated segment must match ``_SAFE_FS_ID`` and not be a reserved name; that
    plus ``_model_dir``'s ``is_relative_to(root)`` guard keeps path traversal impossible
    while letting the registry hold standard HF ids like 'Qwen/Qwen2.5-3B-Instruct'
    (pre-1214 the strict slash-rejecting check failed every org/model id — only slashless
    ids like 'gpt2' could be staged, which blocked production models on the parallax path).
    """
    if not value:
        raise ValueError("model_id must be a non-empty string")
    for seg in value.split("/"):
        if not _SAFE_FS_ID.fullmatch(seg) or seg in _RESERVED_FS_NAMES:
            raise ValueError(
                f"model_id={value!r} unsafe for filesystem registry: each "
                f"'/'-separated segment must match {_SAFE_FS_ID.pattern} and not be "
                f"'.'/'..' (offending segment: {seg!r})"
            )


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ModelRegistryError(Exception):
    """Base error for any registry-layer failure."""


class ModelNotFoundError(ModelRegistryError):
    """No manifest registered for the given model_id."""


class ModelAlreadyRegisteredError(ModelRegistryError):
    """A manifest already exists for the model_id; first-write-wins per node."""


class ManifestVerificationError(ModelRegistryError):
    """Manifest signature failed verification, or a shard's actual sha256
    didn't match its manifest entry. Either way, do not trust the model.
    """


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _hash_shard(shard: ModelShard) -> str:
    """sha256 of a shard's tensor_data bytes — the load-bearing
    commitment that tampering breaks."""
    return hashlib.sha256(shard.tensor_data).hexdigest()


def manifest_from_model(
    model: ShardedModel,
    *,
    publisher_node_id: str,
    published_at: Optional[float] = None,
) -> ModelManifest:
    """Build an unsigned manifest from a ``ShardedModel``.

    Hashes every shard's ``tensor_data`` to produce the canonical
    sha256 commitment in each ``ManifestShardEntry``. The manifest is
    returned UNSIGNED (publisher_signature=b""); call
    :func:`prsm.compute.model_registry.signing.sign_manifest` to seal it.

    Exposed as a module-level helper so callers building manifests
    outside the registry (e.g., for offline-signing workflows or
    cross-registry verification) reuse the same hashing rules the
    registry uses internally.
    """
    entries = tuple(
        ManifestShardEntry(
            shard_id=s.shard_id,
            shard_index=s.shard_index,
            tensor_shape=tuple(s.tensor_shape),
            sha256=_hash_shard(s),
            size_bytes=len(s.tensor_data),
            # Sprint 627 — preserve the source shard's layer_range on
            # the manifest so registry.get() can reconstruct it. Default
            # (0, 0) keeps signing-payload back-compat for old shards
            # that don't set the field.
            layer_range=tuple(getattr(s, "layer_range", (0, 0))),
        )
        for s in sorted(model.shards, key=lambda s: s.shard_index)
    )
    return ModelManifest(
        model_id=model.model_id,
        model_name=model.model_name,
        publisher_node_id=publisher_node_id,
        total_shards=model.total_shards,
        shards=entries,
        published_at=published_at if published_at is not None else time.time(),
    )


# --------------------------------------------------------------------------
# ABC
# --------------------------------------------------------------------------


class ModelRegistry(abc.ABC):
    """Abstract registry of signed model manifests.

    All implementations MUST honor:

    1. ``register`` builds the manifest from the model's actual bytes,
       signs it under ``identity``, and stores enough state to verify
       later. Raise :class:`ModelAlreadyRegisteredError` on duplicate
       ``model_id`` (no first-write-wins assumption changes per backend).
    2. ``get`` returns the model only if BOTH the manifest signature
       AND every shard's sha256 verify. Anything else → raise
       :class:`ManifestVerificationError`.
    3. Missing ``model_id`` → :class:`ModelNotFoundError`.

    The ABC's ``verify`` default impl reuses ``get``; subclasses MAY
    override for audit-faster paths that skip returning the model bytes.
    """

    @abc.abstractmethod
    def register(
        self, model: ShardedModel, *, identity: NodeIdentity
    ) -> ModelManifest:
        """Register and sign. Returns the signed manifest."""

    @abc.abstractmethod
    def get(self, model_id: str) -> ShardedModel:
        """Load + verify. Raises ``ModelNotFoundError`` or
        ``ManifestVerificationError``."""

    @abc.abstractmethod
    def list_models(self) -> List[str]:
        """Sorted list of registered ``model_id`` values."""

    @abc.abstractmethod
    def get_manifest(self, model_id: str) -> ModelManifest:
        """Return the stored manifest (metadata only; no shard bytes)."""

    def verify(self, model_id: str) -> bool:
        """Audit-only: True iff ``get(model_id)`` would succeed.

        Catches both ``ModelNotFoundError`` and
        ``ManifestVerificationError``. Anything else propagates so a
        caller can distinguish "model doesn't verify" from "registry
        is broken."
        """
        try:
            self.get(model_id)
            return True
        except (ModelNotFoundError, ManifestVerificationError):
            return False


# --------------------------------------------------------------------------
# In-memory implementation
# --------------------------------------------------------------------------


class InMemoryModelRegistry(ModelRegistry):
    """Process-local model registry — backwards-compatible with the
    Phase 3.x.1 ``Dict[str, ShardedModel]`` consumer pattern.

    Suitable for tests, single-process scaffolds, and the dict-arg
    back-compat path in ``TensorParallelInferenceExecutor`` (Task 5).
    NOT suitable for production deployment alone — restart drops all
    registrations. Use ``FilesystemModelRegistry`` (Task 4) for
    persistent state.

    Single-writer assumption: no internal locking. Concurrent
    registration of the same ``model_id`` from multiple threads has
    undefined ordering; one will win, the loser raises
    :class:`ModelAlreadyRegisteredError`.

    Verification stash: the registry stores the publisher's
    ``public_key_b64`` at register time so ``get()`` can verify the
    signature without needing the publisher's full ``NodeIdentity``
    on the read path. This matches the offline-verifier model: a
    client holding only the publisher's public key can verify any
    manifest, no live trust path needed.
    """

    def __init__(self) -> None:
        # Three parallel dicts keyed by model_id. Kept separate so
        # get_manifest() can return metadata without touching the
        # heavyweight model bytes.
        self._manifests: Dict[str, ModelManifest] = {}
        self._models: Dict[str, ShardedModel] = {}
        self._publisher_keys: Dict[str, str] = {}  # b64-encoded pubkey

    # -- write path --

    def register(
        self, model: ShardedModel, *, identity: NodeIdentity
    ) -> ModelManifest:
        if model.model_id in self._manifests:
            raise ModelAlreadyRegisteredError(
                f"model_id {model.model_id!r} already registered "
                f"(publisher: {self._manifests[model.model_id].publisher_node_id})"
            )

        unsigned = manifest_from_model(
            model, publisher_node_id=identity.node_id
        )
        signed = sign_manifest(unsigned, identity)

        # Sanity check: signature must verify under the same identity
        # we just used to sign. A failure here means something is very
        # wrong with the signing flow; better to fail fast than ship
        # an unverifiable manifest.
        if not verify_manifest(signed, identity=identity):
            raise ManifestVerificationError(
                "post-sign self-verification failed; signing flow is broken"
            )

        self._manifests[model.model_id] = signed
        self._models[model.model_id] = model
        self._publisher_keys[model.model_id] = identity.public_key_b64
        return signed

    # -- read path --

    def get(self, model_id: str) -> ShardedModel:
        manifest = self._get_manifest_or_raise(model_id)
        public_key_b64 = self._publisher_keys[model_id]

        # Step 1: signature check
        if not verify_manifest(manifest, public_key_b64=public_key_b64):
            raise ManifestVerificationError(
                f"manifest signature for {model_id!r} failed verification"
            )

        # Step 2: shard-byte check — every shard's actual tensor_data
        # must hash to the manifest's recorded sha256. Catches in-place
        # shard tampering between register() and get().
        model = self._models[model_id]
        manifest_by_index = {e.shard_index: e for e in manifest.shards}
        for shard in model.shards:
            entry = manifest_by_index.get(shard.shard_index)
            if entry is None:
                raise ManifestVerificationError(
                    f"shard {shard.shard_index} present in model but missing "
                    f"from manifest of {model_id!r}"
                )
            actual_sha = _hash_shard(shard)
            if actual_sha != entry.sha256:
                raise ManifestVerificationError(
                    f"shard {shard.shard_index} of {model_id!r}: "
                    f"sha256 mismatch (manifest={entry.sha256}, "
                    f"actual={actual_sha})"
                )
            if len(shard.tensor_data) != entry.size_bytes:
                raise ManifestVerificationError(
                    f"shard {shard.shard_index} of {model_id!r}: "
                    f"size mismatch (manifest={entry.size_bytes}, "
                    f"actual={len(shard.tensor_data)})"
                )
        # Confirm the model's shard count matches the manifest's
        # — guards against silently stripping a shard post-register.
        if len(model.shards) != len(manifest.shards):
            raise ManifestVerificationError(
                f"{model_id!r}: model has {len(model.shards)} shards, "
                f"manifest expects {len(manifest.shards)}"
            )
        return model

    def list_models(self) -> List[str]:
        return sorted(self._manifests.keys())

    def get_manifest(self, model_id: str) -> ModelManifest:
        return self._get_manifest_or_raise(model_id)

    # -- internals --

    def _get_manifest_or_raise(self, model_id: str) -> ModelManifest:
        manifest = self._manifests.get(model_id)
        if manifest is None:
            raise ModelNotFoundError(f"no model registered for {model_id!r}")
        return manifest


# --------------------------------------------------------------------------
# Filesystem implementation
# --------------------------------------------------------------------------


# Filename for the publisher's b64-encoded public key, stored alongside
# the manifest so a process restart can verify the manifest signature
# without needing the publisher's live NodeIdentity. NOT included in the
# manifest itself — the manifest is the wire format and stays minimal;
# the sidecar is local-verification metadata.
_PUBLISHER_KEY_FILENAME = "publisher.pubkey"
_MANIFEST_FILENAME = "manifest.json"
_SHARDS_DIRNAME = "shards"


class FilesystemModelRegistry(ModelRegistry):
    """Persistent model registry — manifest.json + shards/*.bin on disk.

    Layout per design plan §3.2::

        <root>/
        ├── <model_id>/
        │   ├── manifest.json        — ModelManifest as canonical JSON
        │   ├── publisher.pubkey     — b64 publisher pubkey (sidecar)
        │   └── shards/
        │       ├── <shard_id>.bin   — raw tensor_data bytes
        │       └── ...

    Survives node restarts. Two registry instances pointing at the same
    ``root`` see each other's writes after they hit the filesystem
    (no shared state in memory; reads always go to disk).

    Identifier safety: ``model_id`` and every ``shard_id`` MUST match
    ``[A-Za-z0-9._-]+``. Path traversal (``..``, ``/``, ``\\``, null
    bytes) is rejected at register time with ``ValueError``. Publishers
    needing arbitrary identifiers should hash them before registration
    or use ``InMemoryModelRegistry``.

    Atomicity: manifest + sidecar writes are atomic
    (``.tmp`` + ``os.replace``). Shard writes are not atomic
    individually but the manifest is written LAST, so a crashed
    registration leaves either a complete model or no manifest (which
    the next ``list_models()`` call won't surface — no half-published
    state visible to readers).

    Single-writer assumption per node; no cross-process locking.
    Concurrent writers on the same root may interleave shards from
    different registrations; the last-writer-wins on the manifest
    means the orphaned shards waste disk but don't corrupt reads.

    SECURITY — TRUST BOUNDARY:
        The registry root is assumed to be a local trust boundary.
        An attacker with write access to ``<root>/<model_id>/`` can
        replace ``publisher.pubkey`` AND re-sign ``manifest.json``
        under their own key — the registry will happily verify the
        substitute. This is acceptable for a node-local registry
        protected by filesystem permissions; it is NOT acceptable
        as a cross-node chain-of-custody anchor. Cross-node
        verification requires the on-chain manifest anchor planned
        for Phase 3.x.3, where the publisher's public key resolves
        from publisher_node_id via an authoritative source rather
        than a sidecar file.

    WRITE-ORDER INVARIANT (do not change without updating tests):
        register() writes shards/*.bin → publisher.pubkey →
        manifest.json. The manifest is the publication marker —
        list_models() and get() both gate on its presence. A crashed
        registration leaves either an incomplete-but-invisible
        directory (no manifest) or a complete model. Reordering
        could create a manifest-present-but-pubkey-missing window;
        get() would then raise ManifestVerificationError, which is
        safe but visible to readers as a pseudo-corruption.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        anchor=None,
        dht: Any = None,
    ) -> None:
        """
        Args:
            root: Filesystem root for the registry.
            anchor: Optional ``PublisherKeyAnchorClient`` (or any object
                with ``lookup(node_id) → Optional[str]``). When supplied,
                ``get()`` resolves the publisher's pubkey via
                ``anchor.lookup`` instead of trusting the on-disk
                ``publisher.pubkey`` sidecar — closes the cross-node
                trust-boundary caveat from Phase 3.x.2 per Phase 3.x.3
                design plan §1.4. The sidecar is still written at
                register time as offline-verifier metadata.

                When ``anchor`` is None, behavior is unchanged from
                Phase 3.x.2 (sidecar-only trust).
            dht: Optional ``ManifestDHTClient`` (or any object exposing
                ``announce(model_id, manifest_path)`` +
                ``get_manifest(model_id) → ModelManifest``). When
                supplied, ``register()`` ALSO announces the model on
                the DHT after the local manifest is durable, and
                ``get()`` / ``get_manifest()`` fall back to a DHT fetch
                when the local manifest is missing. DHT-fetched
                manifests are anchor-verified by the DHT client itself
                (Phase 3.x.5 invariant — no trust-the-network mode), so
                ``dht=`` requires ``anchor=`` for read-side consistency
                with how cached manifests will be re-verified on
                subsequent reads.

                When ``dht`` is None, behavior is unchanged from
                Phase 3.x.3.
        """
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(
                f"FilesystemModelRegistry root {self._root} does not exist; "
                f"create the directory before constructing the registry"
            )
        if not self._root.is_dir():
            raise NotADirectoryError(
                f"FilesystemModelRegistry root {self._root} is not a directory"
            )
        if dht is not None and anchor is None:
            # The DHT client itself anchor-verifies on fetch. But once we
            # cache the fetched manifest to disk, subsequent reads route
            # through the registry's own verify path — which without an
            # anchor would fall back to the publisher.pubkey sidecar,
            # which DHT-fetched manifests don't have. Require both so
            # the cache is verifiable on re-read.
            raise RuntimeError(
                "FilesystemModelRegistry: dht= requires anchor=. "
                "DHT-fetched manifests are anchor-verified at fetch time; "
                "the registry needs the same anchor to re-verify the "
                "cached copy on subsequent reads."
            )
        self._anchor = anchor
        self._dht = dht

    # -- write path --

    def register(
        self, model: ShardedModel, *, identity: NodeIdentity
    ) -> ModelManifest:
        _validate_model_id(model.model_id)
        for shard in model.shards:
            _validate_fs_id("shard_id", shard.shard_id)

        model_dir = self._model_dir(model.model_id)
        if (model_dir / _MANIFEST_FILENAME).exists():
            raise ModelAlreadyRegisteredError(
                f"model_id {model.model_id!r} already registered at {model_dir}"
            )

        unsigned = manifest_from_model(
            model, publisher_node_id=identity.node_id
        )
        signed = sign_manifest(unsigned, identity)

        # Defense-in-depth: signing flow self-check before we write
        # anything to disk.
        if not verify_manifest(signed, identity=identity):
            raise ManifestVerificationError(
                "post-sign self-verification failed; signing flow is broken"
            )

        # Create the model directory + shards subdirectory.
        shards_dir = model_dir / _SHARDS_DIRNAME
        shards_dir.mkdir(parents=True, exist_ok=True)

        # Write shards FIRST. The manifest is the publication marker
        # (its presence is what list_models() looks for). Writing
        # shards before the manifest means a crashed registration
        # leaves an incomplete model directory but no visible
        # registration — readers will see "model not found," not
        # "manifest signed but bytes missing."
        for shard in model.shards:
            shard_path = shards_dir / f"{shard.shard_id}.bin"
            self._atomic_write_bytes(shard_path, shard.tensor_data)

        # Sidecar pubkey before manifest, same reasoning.
        self._atomic_write_text(
            model_dir / _PUBLISHER_KEY_FILENAME,
            identity.public_key_b64,
        )

        # Manifest LAST — its presence means "everything else is on disk."
        manifest_json = json.dumps(signed.to_dict(), sort_keys=True, indent=2)
        manifest_path = model_dir / _MANIFEST_FILENAME
        self._atomic_write_text(manifest_path, manifest_json)

        # DHT announce — best-effort. The local registry is the write
        # authority; if the network is partitioned or the DHT layer
        # raises, we keep the local registration and surface the
        # availability degradation in logs. Catching `Exception` here
        # is intentional (the DHT layer may raise transport, protocol,
        # or local-index errors — none of which should regress the
        # local register() contract).
        if self._dht is not None:
            try:
                self._dht.announce(model.model_id, manifest_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FilesystemModelRegistry.register: DHT announce for "
                    "%r failed: %s — registration succeeded locally, "
                    "network availability degraded",
                    model.model_id, exc,
                )

        return signed

    # -- read path --

    def get(self, model_id: str) -> ShardedModel:
        _validate_model_id(model_id)
        manifest = self._load_manifest_or_raise(model_id)

        # Resolve publisher pubkey: anchor takes precedence over
        # sidecar when configured (Phase 3.x.3 trust upgrade). With no
        # anchor, falls back to the sidecar at the journal root —
        # Phase 3.x.2 behavior preserved unchanged.
        if self._anchor is not None:
            public_key_b64 = self._anchor.lookup(manifest.publisher_node_id)
            if public_key_b64 is None:
                raise ManifestVerificationError(
                    f"publisher {manifest.publisher_node_id!r} not registered "
                    f"on the configured anchor; cannot verify {model_id!r}. "
                    f"Either register the publisher on-chain or remove the "
                    f"anchor= kwarg to fall back to sidecar trust."
                )
        else:
            public_key_b64 = self._load_publisher_key_or_raise(model_id)

        # Step 1: signature
        if not verify_manifest(manifest, public_key_b64=public_key_b64):
            raise ManifestVerificationError(
                f"manifest signature for {model_id!r} failed verification"
            )

        # Step 2: reconstruct each ModelShard from on-disk bytes,
        # verifying sha256 + size as we go.
        model_dir = self._model_dir(model_id)
        shards_dir = model_dir / _SHARDS_DIRNAME

        reconstructed: List[ModelShard] = []
        for entry in manifest.shards:
            shard_path = shards_dir / f"{entry.shard_id}.bin"
            if not shard_path.exists():
                raise ManifestVerificationError(
                    f"shard file missing for {model_id!r} shard "
                    f"{entry.shard_index} ({entry.shard_id!r}): {shard_path}"
                )
            data = shard_path.read_bytes()
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != entry.sha256:
                raise ManifestVerificationError(
                    f"shard {entry.shard_index} of {model_id!r}: "
                    f"sha256 mismatch (manifest={entry.sha256}, "
                    f"actual={actual_sha})"
                )
            if len(data) != entry.size_bytes:
                raise ManifestVerificationError(
                    f"shard {entry.shard_index} of {model_id!r}: "
                    f"size mismatch (manifest={entry.size_bytes}, "
                    f"actual={len(data)})"
                )
            reconstructed.append(
                ModelShard(
                    shard_id=entry.shard_id,
                    model_id=manifest.model_id,
                    shard_index=entry.shard_index,
                    total_shards=manifest.total_shards,
                    tensor_data=data,
                    tensor_shape=tuple(entry.tensor_shape),
                    # Sprint 627 — use the manifest's persisted
                    # layer_range. Pre-627 manifests with no field
                    # come back as (0, 0) (back-compat sentinel).
                    layer_range=tuple(entry.layer_range),
                    size_bytes=entry.size_bytes,
                    checksum=entry.sha256,
                )
            )

        return ShardedModel(
            model_id=manifest.model_id,
            model_name=manifest.model_name,
            total_shards=manifest.total_shards,
            shards=reconstructed,
        )

    def list_models(self) -> List[str]:
        models: List[str] = []
        if not self._root.exists():
            return models
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            if not _SAFE_FS_ID.fullmatch(child.name):
                # Skip directories that don't match our naming policy
                # — they're foreign to this registry.
                continue
            if (child / _MANIFEST_FILENAME).exists():
                models.append(child.name)
        return sorted(models)

    def get_manifest(self, model_id: str) -> ModelManifest:
        _validate_model_id(model_id)
        return self._load_manifest_or_raise(model_id)

    # -- internals --

    def _model_dir(self, model_id: str) -> Path:
        # Pre: model_id was validated by _validate_fs_id() in the
        # public method. Defense in depth: even if a regression in the
        # validator someday lets a traversal slip through, the
        # is_relative_to() check below stops the request before any
        # filesystem state changes. Always raise ValueError on escape
        # — same exception type as _validate_fs_id() for caller parity.
        candidate = (self._root / model_id).resolve()
        root_resolved = self._root.resolve()
        if not candidate.is_relative_to(root_resolved):
            raise ValueError(
                f"model_id={model_id!r} resolves to {candidate} which "
                f"escapes registry root {root_resolved}"
            )
        return candidate

    def _load_manifest_or_raise(self, model_id: str) -> ModelManifest:
        manifest_path = self._model_dir(model_id) / _MANIFEST_FILENAME
        if not manifest_path.exists():
            # DHT fallback: when configured, ask the network. The DHT
            # client is responsible for anchor-verifying every fetched
            # manifest — we don't re-verify here. We DO cache the bytes
            # to disk so subsequent reads are local; the cached manifest
            # gets re-verified through the registry's normal anchor
            # path on next get().
            if self._dht is not None:
                return self._fetch_manifest_via_dht(model_id)
            raise ModelNotFoundError(f"no model registered for {model_id!r}")
        try:
            data = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestVerificationError(
                f"manifest.json for {model_id!r} unreadable or corrupt: {exc}"
            ) from exc
        try:
            return ModelManifest.from_dict(data)
        except (TypeError, ValueError, KeyError) as exc:
            raise ManifestVerificationError(
                f"manifest.json for {model_id!r} schema error: {exc}"
            ) from exc

    def _fetch_manifest_via_dht(self, model_id: str) -> ModelManifest:
        """Pull a manifest from the DHT and cache it locally. Called
        only when the local manifest.json is missing AND a DHT is
        configured. The DHT client anchor-verifies the fetched bytes
        before returning; bytes that fail verification are dropped at
        the DHT layer (caller-side raises ManifestNotFoundError when
        no provider serves verifying bytes), which we surface here as
        ``ModelNotFoundError`` since the local registry has no path
        to serve the model.

        Caches successful fetches as ``manifest.json`` under the
        model directory so subsequent reads short-circuit. Does NOT
        cache a publisher.pubkey sidecar — DHT-fetched manifests are
        only re-verifiable via the anchor (which is required at
        construction when ``dht=`` is set).
        """
        # Late import — avoid pulling the network module into import
        # graphs that don't use DHT.
        from prsm.network.manifest_dht import ManifestNotFoundError

        try:
            manifest = self._dht.get_manifest(model_id)
        except ManifestNotFoundError as exc:
            raise ModelNotFoundError(
                f"no local manifest for {model_id!r}; DHT lookup also "
                f"failed: {exc}"
            ) from exc

        # Cache to disk so subsequent reads bypass the network. The
        # model directory may not exist yet (first time we've heard of
        # this model_id on this node).
        model_dir = self._model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        manifest_json = json.dumps(
            manifest.to_dict(), sort_keys=True, indent=2
        )
        manifest_path = model_dir / _MANIFEST_FILENAME
        self._atomic_write_text(manifest_path, manifest_json)

        # Announce the cached manifest so this node can now serve it
        # to peers. Without this step the DHT server's local_index
        # doesn't know about the cached manifest, and a peer asking
        # "who has model X?" won't see this node — even though we have
        # the bytes on disk. Best-effort, matching register() semantics:
        # cache success + announce failure is still a win for the local
        # caller (they got the manifest), and downgrades to
        # publisher-only DHT visibility for peers.
        try:
            self._dht.announce(model_id, manifest_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FilesystemModelRegistry._fetch_manifest_via_dht: "
                "post-cache DHT announce for %r failed: %s — local cache "
                "is intact, but this node won't serve the model to peers",
                model_id, exc,
            )

        return manifest

    def _load_publisher_key_or_raise(self, model_id: str) -> str:
        key_path = self._model_dir(model_id) / _PUBLISHER_KEY_FILENAME
        if not key_path.exists():
            raise ManifestVerificationError(
                f"publisher.pubkey sidecar missing for {model_id!r}: "
                f"can't verify signature without the publisher's public key"
            )
        return key_path.read_text().strip()

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        """Write bytes atomically: <path>.tmp, fsync, os.replace."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        FilesystemModelRegistry._atomic_write_bytes(
            path, text.encode("utf-8")
        )
