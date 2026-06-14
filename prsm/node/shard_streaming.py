"""Streaming chunker + assembler for shards that exceed the libp2p ≤10MB path.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §3.5, §5.2, §6 Task 6.

Plan §3.5 specifies that shards >10 MB use gRPC-over-TLS direct between
operator and dispatcher rather than the libp2p path. Receipt signing is
unchanged — both paths produce the same `ShardExecutionReceipt`.

This module owns the payload-chunking + integrity logic that sits inside
the gRPC stream. The actual gRPC servicer is a thin wrapper around this
class at Foundation ops time; the chunking + verification logic lives
here so it is testable without `grpcio` or a real socket.

Integrity guarantees:

  * Per-chunk SHA-256 catches single-chunk corruption.
  * Overall SHA-256 over the full payload catches reordering and
    cross-chunk integrity violations.
  * Sequence numbers catch drops and reorders explicitly; the assembler
    rejects a stream whose chunks arrive out of order rather than
    silently buffering (a lenient buffer would mask transport bugs).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List


def _max_reassembly_bytes() -> int:
    """sp1097 (Domain-02 review F4) — receiver-side ceiling on a reassembled shard
    payload. A hostile sender could declare a huge ``payload_bytes`` / ``total_chunks``
    and stream valid-checksum chunks, OOMing the receiver before the finalize-time length
    check fires. Default 1 GiB (matches the streamed-activation cap); PRSM_MAX_SHARD_
    REASSEMBLY_BYTES overrides. Falls back to the default on a missing/invalid value."""
    raw = (os.environ.get("PRSM_MAX_SHARD_REASSEMBLY_BYTES", "") or "").strip()
    if not raw:
        return 1 << 30
    try:
        v = int(raw)
    except (ValueError, TypeError):
        return 1 << 30
    return v if v > 0 else 1 << 30


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "ChunkChecksumMismatch",
    "ChunkOutOfOrder",
    "PayloadChecksumMismatch",
    "ShardAssembler",
    "ShardChunk",
    "ShardChunker",
    "ShardManifest",
    "StreamingError",
]


# 1 MiB default — sized to fit comfortably in a single gRPC message below
# the typical 4 MiB max-message cap without tuning. Operators can pick
# larger values if they've bumped the gRPC cap.
DEFAULT_CHUNK_BYTES = 1 * 1024 * 1024


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class StreamingError(Exception):
    """Base for shard-streaming failures."""


class ChunkChecksumMismatch(StreamingError):
    """A chunk's per-chunk sha256 did not match the stated digest."""


class ChunkOutOfOrder(StreamingError):
    """A chunk arrived with a sequence number that breaks the 0..N-1 order."""


class PayloadChecksumMismatch(StreamingError):
    """The reassembled payload did not match the manifest's overall sha256."""


# -----------------------------------------------------------------------------
# Wire data
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardManifest:
    """Metadata a chunk stream is validated against on the receiver.

    Sent as the first frame of a stream before any `ShardChunk`. The
    `payload_sha256` + `total_chunks` are the integrity envelope.
    """

    shard_id: str
    payload_sha256: str  # hex digest
    payload_bytes: int
    total_chunks: int
    chunk_bytes: int


@dataclass(frozen=True)
class ShardChunk:
    shard_id: str
    sequence: int  # 0-indexed
    data: bytes
    chunk_sha256: str  # hex digest of `data`


# -----------------------------------------------------------------------------
# Chunker
# -----------------------------------------------------------------------------


class ShardChunker:
    """Produces a `(manifest, chunk-iterator)` pair from a bytes payload.

    Pure / deterministic — same input yields byte-identical output across
    runs. No streaming I/O; the caller is expected to hand the output to
    the gRPC servicer.
    """

    def __init__(self, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")
        self._chunk_bytes = chunk_bytes

    def chunk(
        self, shard_id: str, payload: bytes
    ) -> tuple[ShardManifest, List[ShardChunk]]:
        total_chunks = self._compute_total_chunks(len(payload))
        chunks: List[ShardChunk] = []
        for seq in range(total_chunks):
            start = seq * self._chunk_bytes
            end = start + self._chunk_bytes
            data = payload[start:end]
            chunks.append(
                ShardChunk(
                    shard_id=shard_id,
                    sequence=seq,
                    data=data,
                    chunk_sha256=hashlib.sha256(data).hexdigest(),
                )
            )

        manifest = ShardManifest(
            shard_id=shard_id,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            payload_bytes=len(payload),
            total_chunks=total_chunks,
            chunk_bytes=self._chunk_bytes,
        )
        return manifest, chunks

    def _compute_total_chunks(self, payload_bytes: int) -> int:
        if payload_bytes == 0:
            return 0
        return (payload_bytes + self._chunk_bytes - 1) // self._chunk_bytes


# -----------------------------------------------------------------------------
# Assembler
# -----------------------------------------------------------------------------


class ShardAssembler:
    """Receives a manifest then a chunk stream, yields the reassembled
    payload after verifying per-chunk + overall checksums.

    Strict-order: chunks MUST arrive in sequence 0..N-1. Out-of-order
    arrival raises ChunkOutOfOrder. A lenient reorder-buffer would mask
    transport bugs; we reject loud rather than mask.
    """

    def __init__(self, manifest: ShardManifest) -> None:
        # sp1097 (F4) — refuse a manifest declaring more than the reassembly ceiling
        # BEFORE buffering anything, so an oversized declaration can't OOM the receiver.
        _cap = _max_reassembly_bytes()
        if int(getattr(manifest, "payload_bytes", 0)) > _cap:
            raise StreamingError(
                f"manifest payload_bytes {manifest.payload_bytes} exceeds the reassembly "
                f"cap {_cap} (PRSM_MAX_SHARD_REASSEMBLY_BYTES)")
        self._manifest = manifest
        self._max_bytes = _cap
        self._received_bytes = 0
        self._buffer: List[bytes] = []
        self._next_sequence = 0
        self._hasher = hashlib.sha256()
        self._finalised = False

    @property
    def manifest(self) -> ShardManifest:
        return self._manifest

    def feed(self, chunk: ShardChunk) -> None:
        if self._finalised:
            raise StreamingError("assembler already finalised")

        if chunk.shard_id != self._manifest.shard_id:
            raise StreamingError(
                f"shard_id mismatch: chunk={chunk.shard_id!r}, "
                f"manifest={self._manifest.shard_id!r}"
            )
        if chunk.sequence != self._next_sequence:
            raise ChunkOutOfOrder(
                f"expected sequence {self._next_sequence}, got {chunk.sequence}"
            )

        actual_digest = hashlib.sha256(chunk.data).hexdigest()
        if actual_digest != chunk.chunk_sha256:
            raise ChunkChecksumMismatch(
                f"chunk {chunk.sequence}: digest mismatch"
            )

        # sp1097 (F4) — bound the running reassembly size against BOTH the declared
        # payload and the absolute cap, so a sender that streams more than it declared
        # (or a huge declaration that slipped __init__) is rejected mid-stream, not after
        # buffering gigabytes.
        self._received_bytes += len(chunk.data)
        _limit = min(int(self._manifest.payload_bytes), self._max_bytes)
        if self._received_bytes > _limit:
            raise StreamingError(
                f"reassembly exceeded {_limit} bytes at chunk {chunk.sequence} "
                f"(declared payload_bytes={self._manifest.payload_bytes})")

        self._buffer.append(chunk.data)
        self._hasher.update(chunk.data)
        self._next_sequence += 1

    def finalize(self) -> bytes:
        if self._finalised:
            raise StreamingError("assembler already finalised")
        if self._next_sequence != self._manifest.total_chunks:
            raise StreamingError(
                f"incomplete stream: got {self._next_sequence} chunks, "
                f"expected {self._manifest.total_chunks}"
            )

        payload = b"".join(self._buffer)
        if len(payload) != self._manifest.payload_bytes:
            raise StreamingError(
                f"payload length mismatch: got {len(payload)}, "
                f"expected {self._manifest.payload_bytes}"
            )
        overall = self._hasher.hexdigest()
        if overall != self._manifest.payload_sha256:
            raise PayloadChecksumMismatch(
                f"overall sha256 mismatch: got {overall}, "
                f"expected {self._manifest.payload_sha256}"
            )

        self._finalised = True
        return payload

    @classmethod
    def reassemble(
        cls, manifest: ShardManifest, chunks: Iterable[ShardChunk]
    ) -> bytes:
        """One-shot convenience for tests + simple callers."""
        asm = cls(manifest)
        for chunk in chunks:
            asm.feed(chunk)
        return asm.finalize()
