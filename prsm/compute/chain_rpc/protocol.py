"""Phase 3.x.7 Task 1 — Cross-host ChainExecutor wire protocol.

The wire layer for activation handoff between consecutive stages of
a `GPUChain` (Phase 3.x.6). Mirrors the Phase 3.x.5 manifest DHT and
Phase 3.x.6 profile DHT patterns:
  - JSON-canonical-bytes wire format with version
  - Bounded message size (handshake messages only — large activation
    blobs ride the Phase 6 streaming chunker out-of-band)
  - All wire dataclasses are immutable (`@dataclass(frozen=True)`)
  - Anchor-verify-on-read for `HandoffToken`: pubkey resolved from the
    Phase 3.x.3 anchor at verify time, never trusted from any embedded
    field

Three wire types:

  RunLayerSliceRequest    Executor (or upstream stage) → stage. Carries
                          model_id + layer_range + activation blob +
                          signed `HandoffToken`. Stage validates the
                          token against the anchor before executing.

  RunLayerSliceResponse   Stage → executor on success. Carries downstream
                          activation blob + per-stage TEE attestation +
                          stage's signature over the response.

  StageError              Stage → executor on any failure. Structured
                          enum-coded error so the client can route to
                          a structured InferenceResult.failure rather
                          than parsing prose.

Plus the load-bearing trust artifact:

  HandoffToken            Settler-signed credential authorizing one
                          specific stage to execute one specific
                          (request, layer-range) at the settler's
                          expense. Bound to chain_stage_index +
                          request_id; not replayable across requests
                          or stages.

Activation tensor encoding sits in `activation_codec.py` (Task 3); the
wire types here just carry the encoded bytes + shape + dtype string.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity, verify_signature
from prsm.node.shard_streaming import ShardManifest


def request_input_commitment(
    activation_blob: bytes,
    activation_manifest: Optional["ShardManifest"] = None,
) -> str:
    """sp927 — per-iteration commitment to the activation INPUT a stage
    processed.

    For inline inputs it is ``sha256(activation_blob)``; for streamed inputs the
    manifest's ``payload_sha256`` already commits the to-be-assembled bytes.
    Binding this into the stage's RESPONSE signature ties each response to the
    specific input it answered. The KV-cache is keyed on ``request_id`` so
    prefill + every incremental token in a decode session share one request_id;
    without this commitment a malicious non-tail stage could REPLAY a prior
    iteration's signed response (e.g. its prefill output) on a later step and the
    signature would still verify (silent, undetectable output corruption). The
    executor supplies the EXPECTED commitment from the request IT sent — which
    differs every iteration — so a replayed response (bound to a different input)
    fails verification. Both signer and verifier compute it identically here."""
    if activation_manifest is not None:
        return str(activation_manifest.payload_sha256)
    return hashlib.sha256(bytes(activation_blob)).hexdigest()


def response_input_commitment_for_request(request: Any) -> Optional[str]:
    """sp927 — the input commitment a stage binds into its RESPONSE for THIS
    request (and the executor expects at verify), or ``None`` when no binding
    applies.

    Bound ONLY for non-PREFILL decode (INCREMENTAL / VERIFY), where prefill +
    every incremental token in a session share one ``request_id`` (KV-cache key)
    so a prior response could be replayed. PREFILL / non-decode return None →
    the field is omitted from the signing payload → byte-equivalent with pre-fix
    signed bytes. BOTH the signer (stage server) and the verifier (executor)
    call THIS one function on the same request, so the gating decision + the
    committed value are identical by construction — no lockstep drift."""
    decode_mode = getattr(request, "decode_mode", DecodeMode.PREFILL)
    if decode_mode == DecodeMode.PREFILL:
        return None
    return request_input_commitment(
        getattr(request, "activation_blob", b"") or b"",
        getattr(request, "activation_manifest", None),
    )


# ──────────────────────────────────────────────────────────────────────────
# Protocol constants
# ──────────────────────────────────────────────────────────────────────────


CHAIN_RPC_PROTOCOL_VERSION = 2
"""Local wire-format version. Bumped to 2 in Phase 3.x.7.1 to add the
optional ``activation_manifest`` field on ``RunLayerSliceRequest`` /
``Response`` for the chunked-streaming path. v1 inline messages
remain byte-identical for the signing payload (the new field is
omitted from the payload when ``activation_manifest is None``), so
inline interop with v1 peers is preserved."""

SUPPORTED_PROTOCOL_VERSIONS: frozenset = frozenset({1, 2})
"""Protocol versions the parser accepts. v2 nodes accept both v1 and
v2 messages from peers (servers are backward-compat). v1 nodes
accept only v1; a v2 client sending a v2-formatted message to a v1
server gets rejected at the v1 parser. Production rolling-deploy
flows handle this by upgrading servers before clients."""

MAX_HANDSHAKE_BYTES = 64 * 1024 * 1024

MAX_VERIFY_BATCH_TOKENS = 65
"""Phase 3.x.11.y — cap on the number of verified tokens returned
in a VERIFY response (parent + K drafts). 64 is well above any
reasonable speculation depth (literature reports 4-8 as the
practical sweet spot); 65 = 64 drafts + 1 parent verification
position. Defends against a hostile peer claiming a huge
``verified_token_ids`` list that explodes server-side memory at
parse time."""
"""Hard cap on a JSON-encoded chain-RPC message.

v1 inlines activation blobs hex-encoded inside the JSON envelope.
Hex doubles size + JSON adds modest framing overhead; a 64 MiB cap
swallows ~25 MiB of raw activation bytes — enough for typical LLM
activations (e.g., 2048-token × 4096-dim float16 ≈ 16 MiB raw,
≈ 33 MiB hex+JSON).

Future work (Phase 3.x.7.x): wire the Phase 6 streaming chunker
(``activation_codec.chunk_activation``) for activations exceeding
``CHUNK_THRESHOLD_BYTES`` so the wire layer doesn't have to allocate
the full hex string in memory. The codec is shipped (Task 3); only
the transport-side wiring remains. Until then, the inline path
handles the production-sized activation regime by raising the cap.

Stages enforcing a different ceiling can wrap parse_message in their
own size check; the wire protocol's contract is "messages larger
than this are rejected at parse time," not "messages must fit this
cap."
"""


# ──────────────────────────────────────────────────────────────────────────
# Message types + error codes
# ──────────────────────────────────────────────────────────────────────────


class ChainRpcMessageType(str, Enum):
    RUN_LAYER_SLICE_REQUEST = "run_layer_slice_request"
    RUN_LAYER_SLICE_RESPONSE = "run_layer_slice_response"
    STAGE_ERROR = "stage_error"
    ACTIVATION_CHUNK = "activation_chunk"  # v2: streamed-path chunk frame
    # Phase 3.x.8 — streaming-token output (additive on protocol v2):
    TOKEN_FRAME = "token_frame"  # incremental token output from tail stage
    STREAM_FINAL_FRAME = "stream_final_frame"  # terminal frame; embeds signed response
    # Phase 3.x.11 — sharded-decode KV-cache eviction signal:
    EVICT_CACHE_REQUEST = "evict_cache_request"  # executor → stage broadcast on terminal/cancel
    EVICT_CACHE_RESPONSE = "evict_cache_response"  # stage → executor ack
    # Phase 3.x.11.y — speculative-decoding cache rollback signal:
    ROLLBACK_CACHE_REQUEST = "rollback_cache_request"  # executor → stage on rejected speculative suffix
    ROLLBACK_CACHE_RESPONSE = "rollback_cache_response"  # stage → executor ack
    # QueryOrchestrator B3.1b — aggregator handoff RPC for the
    # data-query path. Per-shard partials → selected aggregator →
    # combined plaintext + commit (A9). See
    # docs/2026-05-08-aggregate-rpc-design.md.
    AGGREGATE_REQUEST = "aggregate_request"
    AGGREGATE_RESPONSE = "aggregate_response"


class DecodeMode(str, Enum):
    """Phase 3.x.11 — sharded autoregressive decode mode.

    A sharded autoregressive request runs the chain in distinct
    modes:

    - ``PREFILL`` — first dispatch in a request; full prompt
      forward through the stage's layers; allocates fresh
      KV-cache keyed on ``request_id``.
    - ``INCREMENTAL`` — per-token dispatch; single-position
      forward with cached KV; mutates existing cache.
    - ``VERIFY`` *(Phase 3.x.11.y — speculative decoding)* —
      batched K+1-position forward with cached KV from the
      parent token. Stage 1 input is
      ``[parent_token_id, draft_1, draft_2, ..., draft_K]``;
      Stage > 1 input is K+1 positions of hidden state. Tail
      stage samples K+1 logits + computes ``accepted_count``
      (longest matching prefix between draft and verified
      argmaxes). Executor accepts the prefix + rolls back the
      KV-cache for any rejected suffix via
      ``RollbackCacheRequest``.

    Default ``PREFILL`` preserves byte-equivalence with
    pre-3.x.11 messages (omit-when-default canonical
    encoding, mirroring Phase 3.x.10.x's max_tokens /
    temperature optionality pattern).
    """

    PREFILL = "prefill"
    INCREMENTAL = "incremental"
    VERIFY = "verify"


class StageErrorCode(str, Enum):
    """Structured error codes for cross-host stage failures.

    Values are stable strings (kept lowercase + dash-free) for forward
    compatibility — adding new codes is non-breaking, but renaming an
    existing code requires a protocol-version bump.
    """

    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    INVALID_TOKEN = "INVALID_TOKEN"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    SHARD_MISSING = "SHARD_MISSING"
    TIER_GATE = "TIER_GATE"
    LAYER_RANGE_INVALID = "LAYER_RANGE_INVALID"
    ACTIVATION_INVALID = "ACTIVATION_INVALID"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # QueryOrchestrator B3.1b — aggregate-RPC failure modes per
    # docs/2026-05-08-aggregate-rpc-design.md. EXPIRED_DEADLINE
    # is intentionally omitted; reuse DEADLINE_EXCEEDED above.
    DP_NOISE_MARKER_MISSING = "DP_NOISE_MARKER_MISSING"  # A5 enforcement at server
    PRIVACY_BUDGET_EXHAUSTED = "PRIVACY_BUDGET_EXHAUSTED"  # epsilon ceiling tripped
    INVALID_PARTIAL_SIGNATURE = "INVALID_PARTIAL_SIGNATURE"  # source agent's sig fails


# ──────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────


class ChainRpcProtocolError(Exception):
    """Base error for chain-RPC wire-format failures."""


class ChainRpcMalformedError(ChainRpcProtocolError):
    """Wire bytes did not parse as a valid chain-RPC message."""


class ChainRpcUnknownTypeError(ChainRpcProtocolError):
    """Message ``type`` field doesn't match a known ``ChainRpcMessageType``."""


class ChainRpcVersionMismatchError(ChainRpcProtocolError):
    """Peer's ``protocol_version`` doesn't match local."""


# ──────────────────────────────────────────────────────────────────────────
# HandoffToken — the load-bearing trust artifact
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HandoffToken:
    """Settler-signed credential authorizing one stage to execute one
    (request, layer-range) at the settler's expense.

    The ``settler_node_id`` is the node that's paying for the chain —
    typically the same node that signs the final ``InferenceReceipt``.
    Stages verify the token's signature against the settler's pubkey
    via the Phase 3.x.3 anchor before doing any work.

    Bindings:
      - ``request_id``: a stage cannot reuse a token across requests
      - ``chain_stage_index``: a stage cannot reuse a token at a
        different position in the chain
      - ``deadline_unix``: stale tokens are rejected; bounds the
        replay window even if a token leaks

    Security model: forging a token requires forging an Ed25519
    signature against an anchor-registered pubkey. A stage that
    accepts a forged token has not been compromised — its pre-work
    validation simply failed. We treat forged tokens as protocol
    bugs rather than attestation failures (Phase 7.1 challenges fire
    on output mismatch, not on token-validation failure).
    """

    request_id: str
    settler_node_id: str
    chain_stage_index: int
    chain_total_stages: int
    deadline_unix: float
    signature_b64: str
    # Phase 3.x.11.q.y' — per-request ECDH ephemeral public key.
    # When set, carries an X25519 public key (32 raw bytes) that
    # the receiving stage uses with its long-term X25519 private
    # key to derive a per-request AES-256 key (HKDF-SHA256 over
    # the shared secret + ``(request_id, stage_index)``). When
    # None, the operator-distributed PSK path applies (Phase
    # 3.x.11.q.y baseline). Covered by ``signing_payload`` so a
    # relay adversary cannot substitute a different ephemeral
    # pubkey without invalidating the settler's signature.
    ephemeral_pubkey: Optional[bytes] = None

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("settler_node_id", self.settler_node_id)
        _validate_str_field("signature_b64", self.signature_b64)
        # Phase 3.x.11.q.y' — ephemeral_pubkey shape validation.
        # Exactly 32 bytes when set (X25519 public key length per
        # RFC 7748). Bytes type only; defends against length
        # forgery + type confusion.
        if self.ephemeral_pubkey is not None:
            if not isinstance(
                self.ephemeral_pubkey, (bytes, bytearray),
            ):
                raise ChainRpcMalformedError(
                    f"ephemeral_pubkey must be bytes, got "
                    f"{type(self.ephemeral_pubkey).__name__}"
                )
            if len(self.ephemeral_pubkey) != 32:
                raise ChainRpcMalformedError(
                    f"ephemeral_pubkey must be exactly 32 bytes "
                    f"(X25519 public key), got "
                    f"{len(self.ephemeral_pubkey)}"
                )
        # Reject bool: isinstance(True, int) is True in Python; without
        # this guard a peer sending {"chain_stage_index": false} would
        # produce a token with bool-typed numeric fields.
        if (
            not isinstance(self.chain_stage_index, int)
            or isinstance(self.chain_stage_index, bool)
            or self.chain_stage_index < 0
        ):
            raise ChainRpcMalformedError(
                f"chain_stage_index must be non-negative int, got "
                f"{self.chain_stage_index!r}"
            )
        if (
            not isinstance(self.chain_total_stages, int)
            or isinstance(self.chain_total_stages, bool)
            or self.chain_total_stages <= 0
        ):
            raise ChainRpcMalformedError(
                f"chain_total_stages must be positive int, got "
                f"{self.chain_total_stages!r}"
            )
        if self.chain_stage_index >= self.chain_total_stages:
            raise ChainRpcMalformedError(
                f"chain_stage_index ({self.chain_stage_index}) must be < "
                f"chain_total_stages ({self.chain_total_stages})"
            )
        if not isinstance(self.deadline_unix, (int, float)):
            raise ChainRpcMalformedError(
                f"deadline_unix must be numeric, got "
                f"{type(self.deadline_unix).__name__}"
            )

    @staticmethod
    def signing_payload(
        request_id: str,
        settler_node_id: str,
        chain_stage_index: int,
        chain_total_stages: int,
        deadline_unix: float,
        ephemeral_pubkey: Optional[bytes] = None,
    ) -> bytes:
        """Canonical bytes signed by the settler. Both signer and
        verifier MUST construct identical bytes from the same logical
        token, so all numeric fields are coerced to their JSON-stable
        types and `sort_keys=True` enforces deterministic key ordering.

        Phase 3.x.11.q.y' — ``ephemeral_pubkey`` (when set) is
        committed in the payload via hex encoding. Omit-when-None
        preserves byte-equivalence for pre-q.y' tokens that
        didn't have the field (key absent from the canonical
        payload, not present-as-null)."""
        payload = {
            "request_id": request_id,
            "settler_node_id": settler_node_id,
            "chain_stage_index": int(chain_stage_index),
            "chain_total_stages": int(chain_total_stages),
            "deadline_unix": float(deadline_unix),
        }
        if ephemeral_pubkey is not None:
            payload["ephemeral_pubkey_hex"] = bytes(
                ephemeral_pubkey,
            ).hex()
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def sign(
        cls,
        identity: NodeIdentity,
        request_id: str,
        chain_stage_index: int,
        chain_total_stages: int,
        deadline_unix: float,
        ephemeral_pubkey: Optional[bytes] = None,
    ) -> "HandoffToken":
        """Mint + sign a fresh token under the settler ``identity``."""
        payload = cls.signing_payload(
            request_id,
            identity.node_id,
            chain_stage_index,
            chain_total_stages,
            deadline_unix,
            ephemeral_pubkey,
        )
        sig = identity.sign(payload)
        return cls(
            request_id=request_id,
            settler_node_id=identity.node_id,
            chain_stage_index=int(chain_stage_index),
            chain_total_stages=int(chain_total_stages),
            deadline_unix=float(deadline_unix),
            signature_b64=sig,
            ephemeral_pubkey=(
                bytes(ephemeral_pubkey)
                if ephemeral_pubkey is not None else None
            ),
        )

    def verify_with_anchor(self, anchor: Any) -> bool:
        """Verify this token's signature using the settler's pubkey
        resolved from the on-chain anchor.

        Returns False (not an exception) for any failure path so the
        caller can simply route to ``StageError(INVALID_TOKEN)`` —
        matches the Phase 3.x.5 / 3.x.6 anchor-verify pattern. Distinct
        from anchor-RPC errors which raise (those are transient
        infrastructure, not "this token is bad").

        Anchor-verify-on-read invariant: the pubkey is resolved from
        the anchor via ``settler_node_id``. The token does NOT carry
        an embedded pubkey field; a malicious settler cannot include
        their own pubkey + signature that "verifies" trivially.
        """
        if anchor is None or not hasattr(anchor, "lookup"):
            return False
        settler_pubkey_b64 = anchor.lookup(self.settler_node_id)
        if not settler_pubkey_b64:
            return False
        payload = self.signing_payload(
            self.request_id,
            self.settler_node_id,
            self.chain_stage_index,
            self.chain_total_stages,
            self.deadline_unix,
            self.ephemeral_pubkey,
        )
        return verify_signature(
            settler_pubkey_b64, payload, self.signature_b64
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "request_id": self.request_id,
            "settler_node_id": self.settler_node_id,
            "chain_stage_index": self.chain_stage_index,
            "chain_total_stages": self.chain_total_stages,
            "deadline_unix": self.deadline_unix,
            "signature_b64": self.signature_b64,
        }
        # Phase 3.x.11.q.y' — omit-when-None for backwards-compat:
        # pre-q.y' tokens stay byte-identical on the wire.
        if self.ephemeral_pubkey is not None:
            out["ephemeral_pubkey_hex"] = bytes(
                self.ephemeral_pubkey,
            ).hex()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffToken":
        eph_hex = data.get("ephemeral_pubkey_hex")
        eph_bytes: Optional[bytes] = None
        if eph_hex is not None:
            if not isinstance(eph_hex, str):
                raise ChainRpcMalformedError(
                    f"ephemeral_pubkey_hex must be str, got "
                    f"{type(eph_hex).__name__}"
                )
            try:
                eph_bytes = bytes.fromhex(eph_hex)
            except ValueError as exc:
                raise ChainRpcMalformedError(
                    f"ephemeral_pubkey_hex is not valid hex: {exc}"
                ) from exc
        return cls(
            request_id=_required_str(data, "request_id"),
            settler_node_id=_required_str(data, "settler_node_id"),
            chain_stage_index=_required_int(data, "chain_stage_index"),
            chain_total_stages=_required_int(data, "chain_total_stages"),
            deadline_unix=_required_number(data, "deadline_unix"),
            signature_b64=_required_str(data, "signature_b64"),
            ephemeral_pubkey=eph_bytes,
        )


# ──────────────────────────────────────────────────────────────────────────
# ShardManifest serialization helpers
# (Phase 6 ShardManifest is owned by node/shard_streaming; we serialize
# it inline here to keep the wire format self-contained.)
# ──────────────────────────────────────────────────────────────────────────


def _shard_manifest_to_dict(m: ShardManifest) -> Dict[str, Any]:
    return {
        "shard_id": m.shard_id,
        "payload_sha256": m.payload_sha256,
        "payload_bytes": m.payload_bytes,
        "total_chunks": m.total_chunks,
        "chunk_bytes": m.chunk_bytes,
    }


def _shard_manifest_from_dict(data: Dict[str, Any]) -> ShardManifest:
    if not isinstance(data, dict):
        raise ChainRpcMalformedError(
            f"shard_manifest must be dict, got {type(data).__name__}"
        )
    return ShardManifest(
        shard_id=_required_str(data, "shard_id"),
        payload_sha256=_required_str(data, "payload_sha256"),
        payload_bytes=_required_int(data, "payload_bytes"),
        total_chunks=_required_int(data, "total_chunks"),
        chunk_bytes=_required_int(data, "chunk_bytes"),
    )


def _validate_shard_manifest(m: ShardManifest, *, field: str) -> None:
    """Structural sanity checks on a manifest before signing /
    verification. Stricter than ShardManifest's own validation;
    catches obvious malformations a peer might inject."""
    if not isinstance(m, ShardManifest):
        raise ChainRpcMalformedError(
            f"{field} must be ShardManifest, got {type(m).__name__}"
        )
    if m.payload_bytes < 0:
        raise ChainRpcMalformedError(
            f"{field}.payload_bytes must be non-negative, got {m.payload_bytes}"
        )
    if m.total_chunks < 0:
        raise ChainRpcMalformedError(
            f"{field}.total_chunks must be non-negative, got {m.total_chunks}"
        )
    if m.chunk_bytes <= 0:
        raise ChainRpcMalformedError(
            f"{field}.chunk_bytes must be positive, got {m.chunk_bytes}"
        )
    if not m.shard_id:
        raise ChainRpcMalformedError(
            f"{field}.shard_id must be non-empty"
        )
    # payload_sha256 must look like a sha256 hex string.
    if (
        not isinstance(m.payload_sha256, str)
        or len(m.payload_sha256) != 64
        or any(c not in "0123456789abcdef" for c in m.payload_sha256)
    ):
        raise ChainRpcMalformedError(
            f"{field}.payload_sha256 must be 64-char lowercase hex, "
            f"got {m.payload_sha256!r}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Wire messages
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActivationChunk:
    """Per-chunk frame for the v2 streaming path.

    Wraps a single Phase 6 ``ShardChunk`` for cross-host transport.
    Bound to its parent ``RunLayerSliceRequest`` / ``Response`` via
    ``request_id`` so a relay can't splice chunks from one stream
    into another.

    Chunks are small (≤1 MiB by default — well under
    ``MAX_HANDSHAKE_BYTES``). The streaming transport sends them as
    individual frames over a bidi gRPC stream; the wire format here
    just describes the on-the-wire encoding of one frame.
    """

    request_id: str
    sequence: int  # 0-indexed; matches ShardChunk.sequence
    data: bytes
    chunk_sha256: str  # hex digest of `data`
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.ACTIVATION_CHUNK.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("chunk_sha256", self.chunk_sha256)
        _validate_version(self.protocol_version)
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise ChainRpcMalformedError(
                f"sequence must be non-negative int, got {self.sequence!r}"
            )
        if not isinstance(self.data, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"data must be bytes, got {type(self.data).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "data_hex": bytes(self.data).hex(),
            "chunk_sha256": self.chunk_sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivationChunk":
        _expect_type(data, ChainRpcMessageType.ACTIVATION_CHUNK)
        try:
            data_bytes = bytes.fromhex(_required_str(data, "data_hex"))
        except ValueError as exc:
            raise ChainRpcMalformedError(
                f"data_hex is not valid hex: {exc}"
            ) from exc
        return cls(
            request_id=_required_str(data, "request_id"),
            sequence=_required_int(data, "sequence"),
            data=data_bytes,
            chunk_sha256=_required_str(data, "chunk_sha256"),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class RunLayerSliceRequest:
    """Executor (or upstream stage) → stage: 'run layers
    [layer_range[0], layer_range[1]) on this activation'."""

    request_id: str
    model_id: str
    layer_range: Tuple[int, int]  # half-open [start, end)
    privacy_tier: PrivacyLevel
    content_tier: ContentTier
    activation_blob: bytes
    activation_shape: Tuple[int, ...]
    activation_dtype: str
    upstream_token: HandoffToken
    deadline_unix: float
    # v2 streaming: when present, ``activation_blob`` MUST be empty
    # (b"") and the bytes ride out-of-band as ``ActivationChunk``
    # frames over the streaming transport. ``activation_shape`` +
    # ``activation_dtype`` still describe the post-assembly tensor
    # so the server can decode without an extra round-trip. The
    # manifest's ``payload_sha256`` is the cryptographic commitment
    # to the to-be-assembled bytes.
    activation_manifest: Optional[ShardManifest] = None
    # Phase 3.x.8 — streaming-token output. When True AND the
    # dispatched stage is the chain tail (``is_final_stage`` at the
    # server side), the server returns a TokenFrame stream followed
    # by a StreamFinalFrame instead of a unary RunLayerSliceResponse.
    # When True on a non-tail stage the server rejects with
    # MALFORMED_REQUEST. When False (default), the wire format is
    # byte-identical to a pre-3.x.8 message — the field is OMITTED
    # from the canonical JSON entirely (mirroring the
    # activation_manifest conditional encoding pattern).
    streaming: bool = False
    # Phase 3.x.10.x — streaming-tail sampling overrides. When set,
    # the server passes them through to the streaming runner via
    # the ``request=`` parameter so user-specified
    # ``InferenceRequest.max_tokens`` / ``.temperature`` reach the
    # runner instead of dead-lettering at the wire boundary. Both
    # default to None and are OMITTED from the canonical signing
    # payload when None — pre-3.x.10.x messages with both unset
    # produce byte-identical signed bytes (mirroring the
    # ``activation_manifest`` + ``streaming`` conditional-encoding
    # pattern). Both fields are streaming-only metadata in v1; the
    # unary path ignores them at the server side.
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # Phase 3.x.11 — sharded autoregressive decode mode.
    # ``PREFILL`` (default) is the first dispatch in a request,
    # full prompt forward, fresh KV-cache allocation.
    # ``INCREMENTAL`` is a per-token dispatch with cached KV,
    # single-position forward, mutates existing cache. Pre-
    # 3.x.11 messages omit the field; the canonical signing
    # payload OMITS the key when ``decode_mode == PREFILL``,
    # preserving byte-equivalence with pre-3.x.11 signed bytes
    # (same omit-when-default pattern Phase 3.x.10.x used for
    # ``max_tokens`` / ``temperature``).
    decode_mode: DecodeMode = DecodeMode.PREFILL
    # Phase 3.x.11.y — speculative-decoding draft proposals. Set
    # only on VERIFY dispatches (the executor's speculation loop
    # carries the K draft tokens from the draft model so the tail
    # stage can compute ``accepted_count`` against the verifier's
    # K+1 argmaxes). On non-VERIFY dispatches MUST be None.
    # ``len`` is capped at ``MAX_VERIFY_BATCH_TOKENS - 1`` (since K
    # drafts produce K+1 verified positions, mirrors the response-
    # side cap on ``verified_token_ids``). Default None preserves
    # byte-equivalence with pre-3.x.11.y signed bytes (omit-when-
    # default canonical encoding pattern, same as ``decode_mode`` /
    # ``streaming`` / sampling fields).
    proposed_token_ids: Optional[Tuple[int, ...]] = None
    # Phase 3.x.11.y.x — sampling-correct speculation under
    # temperature > 0. Tuple of per-draft probabilities (the draft
    # model's ``q(d_i)`` per proposed token). Co-set with
    # ``proposed_token_ids``: both set together OR both None.
    # When set, the tail routes to
    # ``apply_lm_head_and_sample_batch_with_rejection`` (Leviathan-
    # 2023 rejection sampling); when None on a VERIFY request, the
    # tail falls back to greedy ``apply_lm_head_and_sample_batch``
    # (v1 path; argmax + longest-prefix-match). Each prob in
    # ``[0, 1]``; length must equal ``len(proposed_token_ids)``.
    # Default None preserves byte-equivalence with pre-3.x.11.y.x
    # signed bytes (omit-when-default canonical encoding mirrors
    # the ``proposed_token_ids`` pattern).
    proposed_token_probs: Optional[Tuple[float, ...]] = None
    # Phase 3.x.11.q.y — encrypted-wire variant of
    # ``proposed_token_probs``. AES-GCM ciphertext bytes (key
    # derived per-request via ECDH ephemeral piggybacked on the
    # HandoffToken). Co-set EXCLUSION invariant with
    # ``proposed_token_probs``: at most one of the two can be set
    # on a single request. Operators choose plaintext or
    # encrypted at config time (Tier C operators set
    # ``encrypted_probs_mode=True`` on RpcChainExecutor; Tier A/B
    # operators leave it default False and use plaintext).
    #
    # When set, the tail decrypts the ciphertext using the
    # request's DH-derived symmetric key, then routes to the
    # standard rejection-sampling helper exactly as it would for
    # plaintext probs. The wire surface becomes content-
    # uncorrelated ciphertext — defends against passive network
    # observers but NOT against stage operators with valid
    # anchor-verified identity (R3 S1 attacker model unchanged).
    #
    # Bytes max length cap: ``MAX_VERIFY_BATCH_TOKENS - 1``
    # plaintext probs at 8 bytes/float = ~512 bytes; AES-GCM adds
    # 12-byte nonce + 16-byte tag = ~540 bytes max. Cap at 1024
    # bytes for headroom + future-proofing against alternative
    # ciphersuites. Default None preserves byte-equivalence with
    # pre-3.x.11.q.y signed bytes (omit-when-default canonical
    # encoding mirrors the ``proposed_token_probs`` pattern).
    #
    # Note on end-to-end authentication: like all
    # RunLayerSliceRequest body fields (proposed_token_ids,
    # proposed_token_probs, etc.), the ciphertext is NOT
    # end-to-end-signed at the request level — only the
    # HandoffToken's boundary fields (request_id, settler,
    # stage_index, total_stages, deadline) are signed. A relay
    # adversary can substitute the ciphertext bytes in transit;
    # mitigated by the AES-GCM AAD bind (request_id || stage_index)
    # which causes decryption to fail at the tail on substitution.
    # Carry-forward of the Phase 3.x.11.y.x M2 honest-scope point.
    encrypted_proposed_token_probs: Optional[bytes] = None
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("model_id", self.model_id)
        _validate_str_field("activation_dtype", self.activation_dtype)
        _validate_version(self.protocol_version)
        if not isinstance(self.layer_range, tuple) or len(self.layer_range) != 2:
            raise ChainRpcMalformedError(
                f"layer_range must be (start, end) tuple, got "
                f"{self.layer_range!r}"
            )
        start, end = self.layer_range
        if not isinstance(start, int) or not isinstance(end, int):
            raise ChainRpcMalformedError(
                f"layer_range entries must be int, got ({type(start).__name__}, "
                f"{type(end).__name__})"
            )
        if start < 0 or end <= start:
            raise ChainRpcMalformedError(
                f"layer_range must satisfy 0 <= start < end, got "
                f"({start}, {end})"
            )
        if not isinstance(self.activation_blob, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"activation_blob must be bytes, got "
                f"{type(self.activation_blob).__name__}"
            )
        if not isinstance(self.activation_shape, tuple):
            raise ChainRpcMalformedError(
                f"activation_shape must be tuple, got "
                f"{type(self.activation_shape).__name__}"
            )
        for dim in self.activation_shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ChainRpcMalformedError(
                    f"activation_shape entries must be positive int, got "
                    f"{dim!r}"
                )
        if not isinstance(self.privacy_tier, PrivacyLevel):
            raise ChainRpcMalformedError(
                f"privacy_tier must be PrivacyLevel, got "
                f"{type(self.privacy_tier).__name__}"
            )
        if not isinstance(self.content_tier, ContentTier):
            raise ChainRpcMalformedError(
                f"content_tier must be ContentTier, got "
                f"{type(self.content_tier).__name__}"
            )
        if not isinstance(self.upstream_token, HandoffToken):
            raise ChainRpcMalformedError(
                f"upstream_token must be HandoffToken, got "
                f"{type(self.upstream_token).__name__}"
            )
        # Cross-field consistency: token's request_id must match the
        # request's request_id. Forging this would require forging the
        # token signature, but we still want a parse-time fast-fail.
        if self.upstream_token.request_id != self.request_id:
            raise ChainRpcMalformedError(
                f"upstream_token.request_id ({self.upstream_token.request_id!r}) "
                f"must match request.request_id ({self.request_id!r})"
            )
        if not isinstance(self.deadline_unix, (int, float)):
            raise ChainRpcMalformedError(
                f"deadline_unix must be numeric, got "
                f"{type(self.deadline_unix).__name__}"
            )
        # v2 inline-XOR-streamed integrity check. Exactly one of the
        # two payload paths must be active: inline blob (non-empty) OR
        # streamed manifest. Both-or-neither is malformed.
        #   - manifest present + blob non-empty → ambiguous (M3 round-1)
        #   - manifest absent  + blob empty     → no payload (M3 round-1)
        # ``activation_shape`` entries are positive ints (>0), so any
        # well-formed inline activation has at least dtype.itemsize
        # bytes and an empty blob is structurally meaningless.
        if self.activation_manifest is not None:
            _validate_shard_manifest(
                self.activation_manifest, field="activation_manifest"
            )
            if self.activation_blob:
                raise ChainRpcMalformedError(
                    "streamed mode: activation_blob must be empty when "
                    "activation_manifest is present (chunks ride out-"
                    "of-band)"
                )
        elif not self.activation_blob:
            raise ChainRpcMalformedError(
                "request has neither inline activation_blob nor "
                "activation_manifest — exactly one payload path is "
                "required"
            )
        # Phase 3.x.8 — streaming flag must be a real bool (not int).
        # Catches the same M1-style coercion hole the protocol_version
        # check defends against.
        if not isinstance(self.streaming, bool):
            raise ChainRpcMalformedError(
                f"streaming must be bool, got {type(self.streaming).__name__}"
            )
        # Phase 3.x.10.x — sampling-override fields. None passes
        # through (server falls back to runner's SamplingDefaults);
        # set values are validated tightly so a malformed peer can't
        # smuggle e.g. negative caps or out-of-range temps.
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool) or not isinstance(
                self.max_tokens, int
            ):
                raise ChainRpcMalformedError(
                    f"max_tokens must be int, got "
                    f"{type(self.max_tokens).__name__}"
                )
            if self.max_tokens <= 0:
                raise ChainRpcMalformedError(
                    f"max_tokens must be positive, got {self.max_tokens}"
                )
        if self.temperature is not None:
            if isinstance(self.temperature, bool) or not isinstance(
                self.temperature, (int, float)
            ):
                raise ChainRpcMalformedError(
                    f"temperature must be number, got "
                    f"{type(self.temperature).__name__}"
                )
            if not (0.0 <= float(self.temperature) <= 2.0):
                raise ChainRpcMalformedError(
                    f"temperature must satisfy 0.0 <= t <= 2.0, "
                    f"got {self.temperature}"
                )
        # Phase 3.x.11 — decode_mode must be a real DecodeMode
        # enum member. Bool / string / arbitrary-int passes
        # would slip through the JSON-string layer; explicit
        # type-check defends against malformed peer input
        # (mirrors ``streaming`` bool-rejection at line 631).
        if not isinstance(self.decode_mode, DecodeMode):
            raise ChainRpcMalformedError(
                f"decode_mode must be DecodeMode, got "
                f"{type(self.decode_mode).__name__}"
            )
        # Phase 3.x.11.y — proposed_token_ids validation. Set
        # iff decode_mode == VERIFY; non-empty tuple of non-
        # negative ints; length capped at
        # MAX_VERIFY_BATCH_TOKENS - 1 (K drafts → K+1 verified;
        # K cap matches the response-side verified-len cap minus
        # the +1 verifier output).
        # Phase 3.x.11.y Task 9 round-1 MEDIUM-1 remediation:
        # symmetric co-set requirement. VERIFY mode REQUIRES
        # proposed_token_ids to be set (closes the asymmetry
        # where a malformed peer could send VERIFY without
        # drafts; the runner caught it on tail-only via
        # _sample_tail_verify but non-tail stages would have
        # processed the K+1 batch silently). Mirrors the
        # response-side verified_token_ids ⇔ accepted_count
        # co-set invariant.
        if (
            self.decode_mode == DecodeMode.VERIFY
            and self.proposed_token_ids is None
        ):
            raise ChainRpcMalformedError(
                "decode_mode=VERIFY requires proposed_token_ids "
                "to be set (the executor's speculation loop "
                "carries the K draft tokens for accepted_count "
                "comparison)"
            )
        if self.proposed_token_ids is not None:
            if self.decode_mode != DecodeMode.VERIFY:
                raise ChainRpcMalformedError(
                    f"proposed_token_ids set but decode_mode is "
                    f"{self.decode_mode.value!r}; proposed_token_ids "
                    f"is meaningful only on VERIFY dispatches"
                )
            if not isinstance(self.proposed_token_ids, tuple):
                raise ChainRpcMalformedError(
                    f"proposed_token_ids must be tuple, got "
                    f"{type(self.proposed_token_ids).__name__}"
                )
            if not self.proposed_token_ids:
                raise ChainRpcMalformedError(
                    "proposed_token_ids must be non-empty when set "
                    "(at least one draft token; speculation_depth >= 1)"
                )
            if len(self.proposed_token_ids) >= MAX_VERIFY_BATCH_TOKENS:
                raise ChainRpcMalformedError(
                    f"proposed_token_ids length "
                    f"{len(self.proposed_token_ids)} exceeds K cap "
                    f"{MAX_VERIFY_BATCH_TOKENS - 1} (K drafts "
                    f"produce K+1 verified positions; K must "
                    f"satisfy K+1 <= MAX_VERIFY_BATCH_TOKENS)"
                )
            for tok in self.proposed_token_ids:
                if isinstance(tok, bool) or not isinstance(tok, int):
                    raise ChainRpcMalformedError(
                        f"proposed_token_ids entries must be int, "
                        f"got {type(tok).__name__}"
                    )
                if tok < 0:
                    raise ChainRpcMalformedError(
                        f"proposed_token_ids entries must be non-"
                        f"negative, got {tok}"
                    )
        # Phase 3.x.11.y.x — proposed_token_probs validation.
        # Co-set with proposed_token_ids: both set together OR
        # both None. Each prob in [0, 1]. Length must match
        # proposed_token_ids. ``None`` here is the v1 greedy path
        # (tail uses argmax-based prefix-match); set here is the
        # v2 stochastic path (tail uses Leviathan-2023 rejection
        # sampling).
        if self.proposed_token_probs is not None:
            if self.proposed_token_ids is None:
                raise ChainRpcMalformedError(
                    "proposed_token_probs set but proposed_token_ids "
                    "is None — both must be co-set (v2 stochastic "
                    "speculation requires both per-draft probabilities "
                    "AND the proposed token ids)"
                )
            if not isinstance(self.proposed_token_probs, tuple):
                raise ChainRpcMalformedError(
                    f"proposed_token_probs must be tuple, got "
                    f"{type(self.proposed_token_probs).__name__}"
                )
            if len(self.proposed_token_probs) != len(self.proposed_token_ids):
                raise ChainRpcMalformedError(
                    f"proposed_token_probs length "
                    f"{len(self.proposed_token_probs)} must equal "
                    f"proposed_token_ids length "
                    f"{len(self.proposed_token_ids)}"
                )
            for p in self.proposed_token_probs:
                if isinstance(p, bool) or not isinstance(p, (int, float)):
                    raise ChainRpcMalformedError(
                        f"proposed_token_probs entries must be number, "
                        f"got {type(p).__name__}"
                    )
                if not (0.0 <= float(p) <= 1.0):
                    raise ChainRpcMalformedError(
                        f"proposed_token_probs entries must be in "
                        f"[0, 1], got {p}"
                    )
        # Phase 3.x.11.q.y — encrypted_proposed_token_probs
        # validation. Exclusion invariant with proposed_token_probs:
        # at most one of the two can be set on a single request
        # (operators choose plaintext OR encrypted at config; mixing
        # would imply ambiguous routing on the tail). Bytes cap
        # at 1024 bytes (see field docstring for derivation).
        if self.encrypted_proposed_token_probs is not None:
            if self.proposed_token_probs is not None:
                raise ChainRpcMalformedError(
                    "encrypted_proposed_token_probs and "
                    "proposed_token_probs are mutually exclusive — "
                    "at most one can be set on a single VERIFY "
                    "request (operators choose plaintext OR "
                    "encrypted at RpcChainExecutor construction)"
                )
            if self.proposed_token_ids is None:
                raise ChainRpcMalformedError(
                    "encrypted_proposed_token_probs set but "
                    "proposed_token_ids is None — both must be "
                    "co-set (the encrypted ciphertext encrypts "
                    "the per-draft probabilities for those ids)"
                )
            if not isinstance(
                self.encrypted_proposed_token_probs, (bytes, bytearray),
            ):
                raise ChainRpcMalformedError(
                    f"encrypted_proposed_token_probs must be bytes, "
                    f"got "
                    f"{type(self.encrypted_proposed_token_probs).__name__}"
                )
            if len(self.encrypted_proposed_token_probs) == 0:
                raise ChainRpcMalformedError(
                    "encrypted_proposed_token_probs must be non-empty "
                    "(empty bytes carry no AES-GCM nonce + tag)"
                )
            if len(self.encrypted_proposed_token_probs) > 1024:
                raise ChainRpcMalformedError(
                    f"encrypted_proposed_token_probs bytes "
                    f"{len(self.encrypted_proposed_token_probs)} "
                    f"exceeds 1024-byte cap (defends against memory "
                    f"exhaustion from a hostile peer claiming huge "
                    f"ciphertext)"
                )

    def to_dict(self) -> Dict[str, Any]:
        # activation_blob → hex for JSON-safety. For streamed (v2)
        # requests, activation_blob is empty bytes and the manifest
        # describes the chunked payload.
        out: Dict[str, Any] = {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "layer_range": list(self.layer_range),
            "privacy_tier": self.privacy_tier.value,
            "content_tier": self.content_tier.value,
            "activation_blob_hex": bytes(self.activation_blob).hex(),
            "activation_shape": list(self.activation_shape),
            "activation_dtype": self.activation_dtype,
            "upstream_token": self.upstream_token.to_dict(),
            "deadline_unix": self.deadline_unix,
        }
        if self.activation_manifest is not None:
            out["activation_manifest"] = _shard_manifest_to_dict(
                self.activation_manifest
            )
        # Conditional encoding: streaming=False omits the field
        # entirely so a v2-but-pre-3.x.8 serializer's bytes are
        # byte-identical to a 3.x.8 serializer's. Only streaming=True
        # adds the key.
        if self.streaming:
            out["streaming"] = True
        # Phase 3.x.10.x — sampling overrides. Omit-when-None
        # preserves byte-equivalence with pre-3.x.10.x signed bytes.
        if self.max_tokens is not None:
            out["max_tokens"] = int(self.max_tokens)
        if self.temperature is not None:
            out["temperature"] = float(self.temperature)
        # Phase 3.x.11 — decode_mode. Omit-when-PREFILL
        # (default) preserves byte-equivalence with pre-3.x.11
        # signed bytes.
        if self.decode_mode != DecodeMode.PREFILL:
            out["decode_mode"] = self.decode_mode.value
        # Phase 3.x.11.y — proposed_token_ids. Omit-when-None
        # preserves byte-equivalence with pre-3.x.11.y signed
        # bytes (proposed_token_ids is set only on VERIFY
        # dispatches, which by definition didn't exist pre-3.x.11.y).
        if self.proposed_token_ids is not None:
            out["proposed_token_ids"] = list(self.proposed_token_ids)
        # Phase 3.x.11.y.x — proposed_token_probs. Omit-when-None
        # preserves byte-equivalence with pre-3.x.11.y.x signed
        # bytes (v1 greedy callers leave probs unset; only v2
        # stochastic callers populate the field). Encoded as a
        # list of floats; co-set invariant with proposed_token_ids
        # is enforced at __post_init__.
        if self.proposed_token_probs is not None:
            out["proposed_token_probs"] = [
                float(p) for p in self.proposed_token_probs
            ]
        # Phase 3.x.11.q.y — encrypted_proposed_token_probs.
        # Hex-encoded bytes for JSON-safety; omit-when-None
        # preserves byte-equivalence with pre-3.x.11.q.y signed
        # bytes. Mutually exclusive with proposed_token_probs at
        # __post_init__ — at most one of the two will be in the
        # serialized output.
        if self.encrypted_proposed_token_probs is not None:
            out["encrypted_proposed_token_probs_hex"] = bytes(
                self.encrypted_proposed_token_probs,
            ).hex()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunLayerSliceRequest":
        _expect_type(data, ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST)
        token_raw = data.get("upstream_token")
        if not isinstance(token_raw, dict):
            raise ChainRpcMalformedError(
                f"upstream_token must be dict, got {type(token_raw).__name__}"
            )
        layer_range_raw = data.get("layer_range")
        if not isinstance(layer_range_raw, list) or len(layer_range_raw) != 2:
            raise ChainRpcMalformedError(
                f"layer_range must be [start, end] list, got {layer_range_raw!r}"
            )
        shape_raw = data.get("activation_shape")
        if not isinstance(shape_raw, list):
            raise ChainRpcMalformedError(
                f"activation_shape must be list, got {type(shape_raw).__name__}"
            )
        # activation_blob_hex is REQUIRED but MAY be empty (streamed
        # mode carries the bytes out-of-band as ActivationChunk frames).
        blob_hex = data.get("activation_blob_hex")
        if not isinstance(blob_hex, str):
            raise ChainRpcMalformedError(
                "activation_blob_hex must be string"
            )
        try:
            blob_bytes = bytes.fromhex(blob_hex)
        except ValueError as exc:
            raise ChainRpcMalformedError(
                f"activation_blob_hex is not valid hex: {exc}"
            ) from exc
        try:
            privacy = PrivacyLevel(_required_str(data, "privacy_tier"))
        except ValueError as exc:
            raise ChainRpcMalformedError(
                f"privacy_tier invalid: {exc}"
            ) from exc
        try:
            content = ContentTier(_required_str(data, "content_tier"))
        except ValueError as exc:
            raise ChainRpcMalformedError(
                f"content_tier invalid: {exc}"
            ) from exc
        manifest_raw = data.get("activation_manifest")
        manifest: Optional[ShardManifest] = None
        if manifest_raw is not None:
            manifest = _shard_manifest_from_dict(manifest_raw)
        # streaming flag (Phase 3.x.8): default False when absent
        # (preserves byte-equivalence with v2-pre-3.x.8 messages).
        # When present, MUST be a real bool — int 0/1 rejected to
        # preserve M1-style coercion-safety invariant.
        streaming_raw = data.get("streaming", False)
        if not isinstance(streaming_raw, bool):
            raise ChainRpcMalformedError(
                f"streaming must be bool, got "
                f"{type(streaming_raw).__name__}"
            )
        # Phase 3.x.10.x — sampling overrides. Default None when
        # absent (preserves byte-equivalence with pre-3.x.10.x
        # messages). Type-checked tightly here so a hostile peer
        # can't smuggle bool-as-int or string-as-number through.
        # Range/positivity validation runs in __post_init__.
        max_tokens_raw = data.get("max_tokens")
        if max_tokens_raw is not None:
            if isinstance(max_tokens_raw, bool) or not isinstance(
                max_tokens_raw, int
            ):
                raise ChainRpcMalformedError(
                    f"max_tokens must be int, got "
                    f"{type(max_tokens_raw).__name__}"
                )
        temperature_raw = data.get("temperature")
        if temperature_raw is not None:
            if isinstance(temperature_raw, bool) or not isinstance(
                temperature_raw, (int, float)
            ):
                raise ChainRpcMalformedError(
                    f"temperature must be number, got "
                    f"{type(temperature_raw).__name__}"
                )
            temperature_raw = float(temperature_raw)
        # Phase 3.x.11 — decode_mode. Default PREFILL when absent
        # (preserves byte-equivalence with pre-3.x.11 messages).
        # When present, must be a string matching a DecodeMode
        # member; bool/int/unknown-string rejected.
        decode_mode_raw = data.get("decode_mode")
        if decode_mode_raw is None:
            decode_mode = DecodeMode.PREFILL
        else:
            if not isinstance(decode_mode_raw, str):
                raise ChainRpcMalformedError(
                    f"decode_mode must be string, got "
                    f"{type(decode_mode_raw).__name__}"
                )
            try:
                decode_mode = DecodeMode(decode_mode_raw)
            except ValueError as exc:
                raise ChainRpcMalformedError(
                    f"decode_mode invalid: {exc}"
                ) from exc
        # Phase 3.x.11.y — proposed_token_ids. Default None when
        # absent (preserves byte-equivalence with pre-3.x.11.y
        # messages). When present, MUST be a list of non-negative
        # ints (bool rejected via the inner type check).
        proposed_raw = data.get("proposed_token_ids")
        proposed_tuple: Optional[Tuple[int, ...]] = None
        if proposed_raw is not None:
            if not isinstance(proposed_raw, list):
                raise ChainRpcMalformedError(
                    f"proposed_token_ids must be list, got "
                    f"{type(proposed_raw).__name__}"
                )
            for tok in proposed_raw:
                if isinstance(tok, bool) or not isinstance(tok, int):
                    raise ChainRpcMalformedError(
                        f"proposed_token_ids entries must be int, "
                        f"got {type(tok).__name__}"
                    )
            proposed_tuple = tuple(int(t) for t in proposed_raw)
        # Phase 3.x.11.y.x — proposed_token_probs. Default None
        # when absent (preserves byte-equivalence with pre-
        # 3.x.11.y.x messages). When present, MUST be a list of
        # numbers (bool rejected via the inner type check; range
        # check + co-set invariant land in __post_init__).
        probs_raw = data.get("proposed_token_probs")
        probs_tuple: Optional[Tuple[float, ...]] = None
        if probs_raw is not None:
            if not isinstance(probs_raw, list):
                raise ChainRpcMalformedError(
                    f"proposed_token_probs must be list, got "
                    f"{type(probs_raw).__name__}"
                )
            for p in probs_raw:
                if isinstance(p, bool) or not isinstance(p, (int, float)):
                    raise ChainRpcMalformedError(
                        f"proposed_token_probs entries must be number, "
                        f"got {type(p).__name__}"
                    )
            probs_tuple = tuple(float(p) for p in probs_raw)
        # Phase 3.x.11.q.y — encrypted_proposed_token_probs.
        # Hex-decoded; default None when absent (preserves byte-
        # equivalence). Mutual exclusion with proposed_token_probs
        # validated at __post_init__.
        encrypted_probs_raw = data.get("encrypted_proposed_token_probs_hex")
        encrypted_probs_bytes: Optional[bytes] = None
        if encrypted_probs_raw is not None:
            if not isinstance(encrypted_probs_raw, str):
                raise ChainRpcMalformedError(
                    f"encrypted_proposed_token_probs_hex must be str, "
                    f"got {type(encrypted_probs_raw).__name__}"
                )
            try:
                encrypted_probs_bytes = bytes.fromhex(encrypted_probs_raw)
            except ValueError as exc:
                raise ChainRpcMalformedError(
                    f"encrypted_proposed_token_probs_hex must be valid "
                    f"hex string: {exc}"
                ) from exc
        return cls(
            request_id=_required_str(data, "request_id"),
            model_id=_required_str(data, "model_id"),
            layer_range=(int(layer_range_raw[0]), int(layer_range_raw[1])),
            privacy_tier=privacy,
            content_tier=content,
            activation_blob=blob_bytes,
            activation_shape=tuple(int(d) for d in shape_raw),
            activation_dtype=_required_str(data, "activation_dtype"),
            upstream_token=HandoffToken.from_dict(token_raw),
            deadline_unix=_required_number(data, "deadline_unix"),
            activation_manifest=manifest,
            streaming=streaming_raw,
            max_tokens=max_tokens_raw,
            temperature=temperature_raw,
            decode_mode=decode_mode,
            proposed_token_ids=proposed_tuple,
            proposed_token_probs=probs_tuple,
            encrypted_proposed_token_probs=encrypted_probs_bytes,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class RunLayerSliceResponse:
    """Stage → executor on success."""

    request_id: str
    activation_blob: bytes
    activation_shape: Tuple[int, ...]
    activation_dtype: str
    duration_seconds: float
    tee_attestation: bytes
    tee_type: TEEType
    epsilon_spent: float
    stage_signature_b64: str
    stage_node_id: str
    # v2 streaming: when present, ``activation_blob`` MUST be empty
    # and the bytes ride out-of-band as ActivationChunk frames.
    activation_manifest: Optional[ShardManifest] = None
    # Phase 3.x.11 — sharded autoregressive tail-stage signal.
    # ``next_token_id`` is the token sampled by the chain tail
    # for this iteration; the executor feeds it back as Stage 1's
    # input on the next ``INCREMENTAL`` chain pass. Non-tail
    # stages leave it None. ``is_terminal`` is True iff the
    # tail's sampling hit EOS or the request's max_tokens cap;
    # signals the executor to stop the per-token chain loop.
    # Both default to None/False so pre-3.x.11 responses
    # produce byte-identical signed bytes (omit-when-default
    # canonical encoding pattern).
    next_token_id: Optional[int] = None
    is_terminal: bool = False
    # Phase 3.x.11.y — speculative-decoding tail-stage signals.
    # ``verified_token_ids`` (when set) carries K+1 argmax tokens
    # — one per VERIFY input position. ``accepted_count`` is the
    # length of the longest matching prefix between the request's
    # draft proposals and the verified argmaxes (``0..K``;
    # `0` means no draft tokens accepted, `K` means all accepted).
    # Both are populated only on tail-stage VERIFY dispatches;
    # non-VERIFY responses (PREFILL / INCREMENTAL) leave them
    # None. Default None preserves byte-equivalence with
    # pre-3.x.11.y signed bytes (omit-when-default canonical
    # encoding pattern).
    verified_token_ids: Optional[Tuple[int, ...]] = None
    accepted_count: Optional[int] = None
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.RUN_LAYER_SLICE_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("activation_dtype", self.activation_dtype)
        _validate_str_field("stage_signature_b64", self.stage_signature_b64)
        _validate_str_field("stage_node_id", self.stage_node_id)
        _validate_version(self.protocol_version)
        if not isinstance(self.activation_blob, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"activation_blob must be bytes, got "
                f"{type(self.activation_blob).__name__}"
            )
        if not isinstance(self.activation_shape, tuple):
            raise ChainRpcMalformedError(
                f"activation_shape must be tuple, got "
                f"{type(self.activation_shape).__name__}"
            )
        for dim in self.activation_shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ChainRpcMalformedError(
                    f"activation_shape entries must be positive int, got "
                    f"{dim!r}"
                )
        if not isinstance(self.duration_seconds, (int, float)) or self.duration_seconds < 0:
            raise ChainRpcMalformedError(
                f"duration_seconds must be non-negative numeric, got "
                f"{self.duration_seconds!r}"
            )
        if not isinstance(self.tee_attestation, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"tee_attestation must be bytes, got "
                f"{type(self.tee_attestation).__name__}"
            )
        if not isinstance(self.tee_type, TEEType):
            raise ChainRpcMalformedError(
                f"tee_type must be TEEType, got "
                f"{type(self.tee_type).__name__}"
            )
        if not isinstance(self.epsilon_spent, (int, float)):
            raise ChainRpcMalformedError(
                f"epsilon_spent must be numeric, got "
                f"{type(self.epsilon_spent).__name__}"
            )
        # v2 inline-XOR-streamed integrity check (M3 round-1 mirror).
        # Exactly one payload path active. ``activation_shape`` entries
        # are positive ints, so any well-formed inline response carries
        # at least dtype.itemsize bytes — empty + no manifest = no
        # payload.
        if self.activation_manifest is not None:
            _validate_shard_manifest(
                self.activation_manifest, field="activation_manifest"
            )
            if self.activation_blob:
                raise ChainRpcMalformedError(
                    "streamed mode: activation_blob must be empty when "
                    "activation_manifest is present"
                )
        elif not self.activation_blob:
            raise ChainRpcMalformedError(
                "response has neither inline activation_blob nor "
                "activation_manifest — exactly one payload path is "
                "required"
            )
        # Phase 3.x.11 — sharded autoregressive tail signals.
        # ``next_token_id`` (when set) must be non-negative int;
        # ``is_terminal`` must be a real bool (mirrors
        # ``streaming`` / ``decode_mode`` bool-rejection on the
        # request side).
        if self.next_token_id is not None:
            if isinstance(self.next_token_id, bool) or not isinstance(
                self.next_token_id, int
            ):
                raise ChainRpcMalformedError(
                    f"next_token_id must be int, got "
                    f"{type(self.next_token_id).__name__}"
                )
            if self.next_token_id < 0:
                raise ChainRpcMalformedError(
                    f"next_token_id must be non-negative, got "
                    f"{self.next_token_id}"
                )
        if not isinstance(self.is_terminal, bool):
            raise ChainRpcMalformedError(
                f"is_terminal must be bool, got "
                f"{type(self.is_terminal).__name__}"
            )
        # Phase 3.x.11.y — speculative-decoding tail signals.
        # verified_token_ids must be a tuple of non-negative
        # ints with length capped at MAX_VERIFY_BATCH_TOKENS
        # (defends against a hostile peer claiming a huge
        # speculation depth that explodes server-side memory).
        # accepted_count must be a non-negative int <= len(
        # verified_token_ids) - 1 (since one position is the
        # parent's verifier-output, accepted_count is the
        # number of DRAFT tokens accepted, max == K when all
        # K drafts match).
        # Both fields must be co-set: setting one without the
        # other is a malformed VERIFY response.
        verified_set = self.verified_token_ids is not None
        accepted_set = self.accepted_count is not None
        if verified_set != accepted_set:
            raise ChainRpcMalformedError(
                "verified_token_ids and accepted_count must be "
                "co-set on VERIFY responses (got "
                f"verified={self.verified_token_ids!r}, "
                f"accepted={self.accepted_count!r})"
            )
        if verified_set:
            if not isinstance(self.verified_token_ids, tuple):
                raise ChainRpcMalformedError(
                    f"verified_token_ids must be tuple, got "
                    f"{type(self.verified_token_ids).__name__}"
                )
            if not self.verified_token_ids:
                raise ChainRpcMalformedError(
                    "verified_token_ids must be non-empty when "
                    "set (must contain at least the parent's "
                    "verifier-output)"
                )
            if len(self.verified_token_ids) > MAX_VERIFY_BATCH_TOKENS:
                raise ChainRpcMalformedError(
                    f"verified_token_ids length "
                    f"{len(self.verified_token_ids)} exceeds cap "
                    f"{MAX_VERIFY_BATCH_TOKENS} — defends against "
                    f"hostile peer claiming huge speculation depth"
                )
            for tok in self.verified_token_ids:
                if isinstance(tok, bool) or not isinstance(tok, int):
                    raise ChainRpcMalformedError(
                        f"verified_token_ids entries must be int, "
                        f"got {type(tok).__name__}"
                    )
                if tok < 0:
                    raise ChainRpcMalformedError(
                        f"verified_token_ids entries must be "
                        f"non-negative, got {tok}"
                    )
            if (
                isinstance(self.accepted_count, bool)
                or not isinstance(self.accepted_count, int)
            ):
                raise ChainRpcMalformedError(
                    f"accepted_count must be int, got "
                    f"{type(self.accepted_count).__name__}"
                )
            if self.accepted_count < 0:
                raise ChainRpcMalformedError(
                    f"accepted_count must be non-negative, got "
                    f"{self.accepted_count}"
                )
            # accepted_count <= K, where K = len(verified) - 1
            # (the first verified position is the parent's
            # output; tokens 1..K are the K drafts that may or
            # may not match).
            max_accepted = len(self.verified_token_ids) - 1
            if self.accepted_count > max_accepted:
                raise ChainRpcMalformedError(
                    f"accepted_count {self.accepted_count} exceeds "
                    f"max {max_accepted} (K = len(verified)-1)"
                )

    @staticmethod
    def signing_payload(
        request_id: str,
        activation_blob: bytes,
        activation_shape: Tuple[int, ...],
        activation_dtype: str,
        duration_seconds: float,
        tee_attestation: bytes,
        tee_type: TEEType,
        epsilon_spent: float,
        stage_node_id: str,
        activation_manifest: Optional[ShardManifest] = None,
        next_token_id: Optional[int] = None,
        is_terminal: bool = False,
        verified_token_ids: Optional[Tuple[int, ...]] = None,
        accepted_count: Optional[int] = None,
        input_commitment: Optional[str] = None,
    ) -> bytes:
        """Canonical bytes the stage signs over the response.

        The signature commits the stage to its activation output bytes
        + attestation + claimed timing. A malicious downstream that
        tampered with the response (e.g. swapped attestation) would
        invalidate the signature; the executor catches this at
        anchor-verify time and surfaces it as Phase 7.1 challenge
        material.

        v1↔v2 byte-equivalence for inline messages: when
        ``activation_manifest`` is None, the field is OMITTED from the
        canonical JSON entirely — so a v2 caller signing inline gets
        the same payload bytes a v1 caller would. Only streamed
        responses carry the additional ``activation_manifest_envelope``
        key, which commits the stage to ALL FIVE manifest fields
        (shard_id + payload_sha256 + payload_bytes + total_chunks +
        chunk_bytes). Without committing the envelope's shape fields,
        a network-level relay could inflate ``payload_bytes`` /
        ``total_chunks`` while leaving ``payload_sha256`` intact and
        trigger a client-side memory DoS during reassembly — the
        H2 round-1 finding from Phase 3.x.7.1 Task 6 review.
        """
        payload = {
            "request_id": request_id,
            "activation_blob_hex": bytes(activation_blob).hex(),
            "activation_shape": list(activation_shape),
            "activation_dtype": activation_dtype,
            "duration_seconds": float(duration_seconds),
            "tee_attestation_hex": bytes(tee_attestation).hex(),
            "tee_type": tee_type.value,
            "epsilon_spent": float(epsilon_spent),
            "stage_node_id": stage_node_id,
        }
        # Conditional manifest envelope encoding: omitted entirely when
        # the response is inline (so v1↔v2 inline messages produce
        # byte-identical canonical JSON). Streamed responses commit
        # to ALL five manifest fields — tampering ANY of them
        # invalidates the signature.
        if activation_manifest is not None:
            payload["activation_manifest_envelope"] = {
                "shard_id": activation_manifest.shard_id,
                "payload_sha256": activation_manifest.payload_sha256,
                "payload_bytes": activation_manifest.payload_bytes,
                "total_chunks": activation_manifest.total_chunks,
                "chunk_bytes": activation_manifest.chunk_bytes,
            }
        # Phase 3.x.11 Task 5 — commit the tail-sample fields when
        # set. Omit-when-default preserves pre-3.x.11 byte-equivalence:
        # responses with next_token_id=None AND is_terminal=False
        # produce the exact same signing payload as pre-3.x.11
        # signed bytes. Sharded-mode tail responses gain commitment
        # to the sampled token + terminal flag — without this, a
        # malicious downstream relay could swap next_token_id
        # without invalidating the signature.
        if next_token_id is not None:
            payload["next_token_id"] = int(next_token_id)
        if is_terminal:
            payload["is_terminal"] = True
        # Phase 3.x.11.y — commit speculative-decoding tail
        # signals when set. Omit-when-default preserves
        # pre-3.x.11.y byte-equivalence: PREFILL/INCREMENTAL
        # responses (no VERIFY signals) produce the same
        # signing payload as pre-3.x.11.y signed bytes.
        # Co-set invariant enforced at __post_init__; here
        # both keys are added together.
        if verified_token_ids is not None:
            # Round-1 MEDIUM-3 remediation: defend against a
            # caller passing verified_token_ids without
            # accepted_count via the staticmethod entry. The
            # __post_init__ co-set invariant catches this on
            # dataclass construction; the staticmethod is also
            # callable directly by ``verify_with_anchor``, so
            # we explicitly raise here instead of silently
            # coercing ``None or 0 → 0``.
            if accepted_count is None:
                raise ChainRpcMalformedError(
                    "RunLayerSliceResponse.signing_payload: "
                    "verified_token_ids is set but accepted_count "
                    "is None — both must be co-set"
                )
            payload["verified_token_ids"] = [
                int(t) for t in verified_token_ids
            ]
            payload["accepted_count"] = int(accepted_count)
        # sp927 — per-iteration INPUT commitment. Omit-when-None preserves
        # byte-equivalence: prefill / non-decode / pre-fix callers pass None and
        # produce identical signed bytes. When set (incremental/verify decode),
        # the response is bound to the specific input it answered, so a malicious
        # stage cannot replay a prior iteration's signed response.
        if input_commitment is not None:
            payload["input_commitment"] = str(input_commitment)
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def sign(
        cls,
        identity: NodeIdentity,
        request_id: str,
        activation_blob: bytes,
        activation_shape: Tuple[int, ...],
        activation_dtype: str,
        duration_seconds: float,
        tee_attestation: bytes,
        tee_type: TEEType,
        epsilon_spent: float,
        activation_manifest: Optional[ShardManifest] = None,
        next_token_id: Optional[int] = None,
        is_terminal: bool = False,
        verified_token_ids: Optional[Tuple[int, ...]] = None,
        accepted_count: Optional[int] = None,
        input_commitment: Optional[str] = None,
    ) -> "RunLayerSliceResponse":
        """Construct + sign a fresh response under the stage ``identity``.

        For streamed responses, pass ``activation_manifest=manifest`` and
        ``activation_blob=b""``. The manifest's ``payload_sha256``
        commits the stage to the to-be-assembled bytes via the
        signing payload.

        For Phase 3.x.11 sharded-decode tail responses, pass
        ``next_token_id=<sampled>`` and ``is_terminal=<EOS or
        max_tokens hit>``. Both fields are committed in the
        signing payload (omit-when-default) — sharded-mode tails
        prevent a malicious downstream relay from swapping the
        sampled token without invalidating the signature.
        """
        payload = cls.signing_payload(
            request_id,
            activation_blob,
            activation_shape,
            activation_dtype,
            duration_seconds,
            tee_attestation,
            tee_type,
            epsilon_spent,
            identity.node_id,
            activation_manifest=activation_manifest,
            next_token_id=next_token_id,
            is_terminal=is_terminal,
            verified_token_ids=verified_token_ids,
            accepted_count=accepted_count,
            input_commitment=input_commitment,
        )
        sig = identity.sign(payload)
        return cls(
            request_id=request_id,
            activation_blob=bytes(activation_blob),
            activation_shape=tuple(activation_shape),
            activation_dtype=activation_dtype,
            duration_seconds=float(duration_seconds),
            tee_attestation=bytes(tee_attestation),
            tee_type=tee_type,
            epsilon_spent=float(epsilon_spent),
            stage_signature_b64=sig,
            stage_node_id=identity.node_id,
            activation_manifest=activation_manifest,
            next_token_id=next_token_id,
            is_terminal=is_terminal,
            verified_token_ids=(
                tuple(verified_token_ids)
                if verified_token_ids is not None else None
            ),
            accepted_count=accepted_count,
        )

    def verify_with_anchor(
        self, anchor: Any, *, expected_stage_node_id: str,
        expected_input_commitment: Optional[str] = None,
    ) -> bool:
        """Verify this response's stage signature against the EXPECTED
        stage's pubkey resolved from the on-chain anchor.

        ``expected_stage_node_id`` MUST be the node_id the caller
        dispatched the request to — typically the GPUChain stage at
        the current chain index. The check rejects when:

          - the response's claimed ``stage_node_id`` doesn't match
            ``expected_stage_node_id`` (defends against an attacker
            with ANY anchor-registered identity replacing the legit
            stage's response with their own genuine signature; without
            this check, the lookup-by-self-claim path would accept
            Mallory's response under Mallory's pubkey because Mallory
            IS anchor-registered)

          - the resolved pubkey doesn't verify the stage's signature

        The Phase 3.x.5 SignedManifestEntry / Phase 3.x.6
        SignedProfileEntry pattern is functionally equivalent: the
        identity to look up is supplied EXTERNALLY by the caller, not
        pulled from the response. RunLayerSliceResponse takes the
        externally-supplied identity as a kwarg to make the contract
        impossible to misuse.

        Returns False on any failure path; raises only on transient
        anchor-RPC infrastructure errors.
        """
        if anchor is None or not hasattr(anchor, "lookup"):
            return False
        if not isinstance(expected_stage_node_id, str) or not expected_stage_node_id:
            return False
        # Anchor-verify against the EXPECTED identity, not the
        # response's self-declared one. If a malicious peer signed
        # with their own (also anchor-registered) key, the lookup
        # below uses the expected node's pubkey, so the bad signature
        # fails verification.
        if self.stage_node_id != expected_stage_node_id:
            return False
        stage_pubkey_b64 = anchor.lookup(expected_stage_node_id)
        if not stage_pubkey_b64:
            return False
        payload = self.signing_payload(
            self.request_id,
            self.activation_blob,
            self.activation_shape,
            self.activation_dtype,
            self.duration_seconds,
            self.tee_attestation,
            self.tee_type,
            self.epsilon_spent,
            self.stage_node_id,
            activation_manifest=self.activation_manifest,
            next_token_id=self.next_token_id,
            is_terminal=self.is_terminal,
            verified_token_ids=self.verified_token_ids,
            accepted_count=self.accepted_count,
            # sp927 — the EXPECTED input commitment is supplied EXTERNALLY by
            # the executor (from the request it sent this iteration), NOT taken
            # from the response — mirroring expected_stage_node_id. A replayed
            # response (signed over a different input) reconstructs to different
            # bytes here and fails verification.
            input_commitment=expected_input_commitment,
        )
        return verify_signature(
            stage_pubkey_b64, payload, self.stage_signature_b64
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "activation_blob_hex": bytes(self.activation_blob).hex(),
            "activation_shape": list(self.activation_shape),
            "activation_dtype": self.activation_dtype,
            "duration_seconds": self.duration_seconds,
            "tee_attestation_hex": bytes(self.tee_attestation).hex(),
            "tee_type": self.tee_type.value,
            "epsilon_spent": self.epsilon_spent,
            "stage_signature_b64": self.stage_signature_b64,
            "stage_node_id": self.stage_node_id,
        }
        if self.activation_manifest is not None:
            out["activation_manifest"] = _shard_manifest_to_dict(
                self.activation_manifest
            )
        # Phase 3.x.11 — sharded autoregressive tail signals.
        # Omit-when-default canonical encoding: pre-3.x.11
        # responses (next_token_id=None, is_terminal=False)
        # produce byte-identical signed bytes.
        if self.next_token_id is not None:
            out["next_token_id"] = int(self.next_token_id)
        if self.is_terminal:
            out["is_terminal"] = True
        # Phase 3.x.11.y — speculative-decoding tail signals.
        # Omit-when-default canonical encoding: VERIFY responses
        # carry both keys; non-VERIFY responses produce byte-
        # identical wire bytes with pre-3.x.11.y. Co-set
        # invariant enforced by __post_init__.
        if self.verified_token_ids is not None:
            # Round-1 MEDIUM-3 remediation: __post_init__ co-set
            # invariant guarantees ``accepted_count is not None``
            # whenever verified_token_ids is set. Use direct
            # int() to make the invariant explicit (the prior
            # ``self.accepted_count or 0`` silently masked an
            # invalid state).
            out["verified_token_ids"] = [
                int(t) for t in self.verified_token_ids
            ]
            out["accepted_count"] = int(self.accepted_count)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunLayerSliceResponse":
        _expect_type(data, ChainRpcMessageType.RUN_LAYER_SLICE_RESPONSE)
        shape_raw = data.get("activation_shape")
        if not isinstance(shape_raw, list):
            raise ChainRpcMalformedError(
                f"activation_shape must be list, got {type(shape_raw).__name__}"
            )
        # activation_blob_hex is REQUIRED but MAY be empty (streamed
        # mode carries the bytes out-of-band).
        blob_hex_raw = data.get("activation_blob_hex")
        if not isinstance(blob_hex_raw, str):
            raise ChainRpcMalformedError(
                "activation_blob_hex must be string"
            )
        try:
            blob_bytes = bytes.fromhex(blob_hex_raw)
            attest_bytes = bytes.fromhex(
                _required_str(data, "tee_attestation_hex")
            )
        except ValueError as exc:
            raise ChainRpcMalformedError(f"hex decode failed: {exc}") from exc
        try:
            tee_type = TEEType(_required_str(data, "tee_type"))
        except ValueError as exc:
            raise ChainRpcMalformedError(f"tee_type invalid: {exc}") from exc
        manifest_raw = data.get("activation_manifest")
        manifest: Optional[ShardManifest] = None
        if manifest_raw is not None:
            manifest = _shard_manifest_from_dict(manifest_raw)
        # Phase 3.x.11 — sharded autoregressive tail signals.
        # Default None / False when absent (preserves byte-
        # equivalence with pre-3.x.11 responses). Type-checked
        # tightly here so a hostile peer can't smuggle bool-
        # as-int or unknown types through.
        next_token_id_raw = data.get("next_token_id")
        if next_token_id_raw is not None:
            if isinstance(next_token_id_raw, bool) or not isinstance(
                next_token_id_raw, int
            ):
                raise ChainRpcMalformedError(
                    f"next_token_id must be int, got "
                    f"{type(next_token_id_raw).__name__}"
                )
        is_terminal_raw = data.get("is_terminal", False)
        if not isinstance(is_terminal_raw, bool):
            raise ChainRpcMalformedError(
                f"is_terminal must be bool, got "
                f"{type(is_terminal_raw).__name__}"
            )
        # Phase 3.x.11.y — speculative-decoding tail signals.
        # Default None when absent. Type-checked tightly: list
        # of non-negative ints capped at MAX_VERIFY_BATCH_TOKENS;
        # accepted_count non-negative int. Co-set invariant
        # enforced at __post_init__ via the constructed
        # dataclass — same boundary catches malformed wire.
        verified_raw = data.get("verified_token_ids")
        verified_token_ids: Optional[Tuple[int, ...]] = None
        if verified_raw is not None:
            if not isinstance(verified_raw, list):
                raise ChainRpcMalformedError(
                    f"verified_token_ids must be list, got "
                    f"{type(verified_raw).__name__}"
                )
            if len(verified_raw) > MAX_VERIFY_BATCH_TOKENS:
                raise ChainRpcMalformedError(
                    f"verified_token_ids length {len(verified_raw)} "
                    f"exceeds cap {MAX_VERIFY_BATCH_TOKENS}"
                )
            for t in verified_raw:
                if isinstance(t, bool) or not isinstance(t, int):
                    raise ChainRpcMalformedError(
                        f"verified_token_ids entries must be int, "
                        f"got {type(t).__name__}"
                    )
            verified_token_ids = tuple(int(t) for t in verified_raw)
        accepted_raw = data.get("accepted_count")
        accepted_count: Optional[int] = None
        if accepted_raw is not None:
            if isinstance(accepted_raw, bool) or not isinstance(
                accepted_raw, int
            ):
                raise ChainRpcMalformedError(
                    f"accepted_count must be int, got "
                    f"{type(accepted_raw).__name__}"
                )
            accepted_count = int(accepted_raw)
        return cls(
            request_id=_required_str(data, "request_id"),
            activation_blob=blob_bytes,
            activation_shape=tuple(int(d) for d in shape_raw),
            activation_dtype=_required_str(data, "activation_dtype"),
            duration_seconds=_required_number(data, "duration_seconds"),
            tee_attestation=attest_bytes,
            tee_type=tee_type,
            epsilon_spent=_required_number(data, "epsilon_spent"),
            stage_signature_b64=_required_str(data, "stage_signature_b64"),
            stage_node_id=_required_str(data, "stage_node_id"),
            activation_manifest=manifest,
            next_token_id=next_token_id_raw,
            is_terminal=is_terminal_raw,
            verified_token_ids=verified_token_ids,
            accepted_count=accepted_count,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class StageError:
    """Stage → executor on any failure. Structured enum-coded reason."""

    request_id: str
    code: str          # StageErrorCode value (str-Enum, stored as str
                       # so unknown codes from a future protocol round-
                       # trip cleanly without rejection)
    message: str
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.STAGE_ERROR.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("code", self.code)
        if not isinstance(self.message, str):
            raise ChainRpcMalformedError(
                f"message must be str, got {type(self.message).__name__}"
            )
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "code": self.code,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageError":
        _expect_type(data, ChainRpcMessageType.STAGE_ERROR)
        return cls(
            request_id=_required_str(data, "request_id"),
            code=_required_str(data, "code"),
            message=data.get("message", "") if isinstance(data.get("message"), str) else "",
            protocol_version=_required_int(data, "protocol_version"),
        )


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.8 — streaming-token output wire types
# ──────────────────────────────────────────────────────────────────────────


# Permitted finish_reason values on TokenFrame / StreamFinalFrame's
# embedded response. ``None`` means "more frames coming"; a non-None
# value is set on the FINAL frame the runner emits.
_FINISH_REASONS = frozenset({"stop", "max_tokens", "cancelled", "error"})


@dataclass(frozen=True)
class TokenFrame:
    """Incremental token-output frame emitted by the tail stage during
    autoregressive decode (Phase 3.x.8).

    Multiple ``TokenFrame`` frames per stream, ordered strictly by
    ``sequence_index`` (0-indexed, no gaps). The terminal frame on a
    stream is always a ``StreamFinalFrame`` carrying the signed
    ``RunLayerSliceResponse`` over the joined output bytes.

    Bound to its parent request via ``request_id`` so a network relay
    can't splice frames from one stream into another (mirrors the
    relay-defense binding on ``ActivationChunk`` from 3.x.7.1).

    The joined ``text_delta`` across all frames in a stream MUST equal
    the UTF-8 decoding of the embedded response's ``activation_blob``
    in the terminal ``StreamFinalFrame`` — this is how the stage's
    Ed25519 signature commits to the streamed output (existing
    ``RunLayerSliceResponse.signing_payload`` hex-encodes
    ``activation_blob``; no new signing-payload field needed).
    """

    request_id: str
    sequence_index: int  # 0-indexed; strictly increasing across stream
    text_delta: str      # text emitted by this frame (may be empty)
    token_id: Optional[int] = None        # vocab id if available
    finish_reason: Optional[str] = None   # set on the LAST TokenFrame
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.TOKEN_FRAME.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)
        if (
            not isinstance(self.sequence_index, int)
            or isinstance(self.sequence_index, bool)
            or self.sequence_index < 0
        ):
            raise ChainRpcMalformedError(
                f"sequence_index must be non-negative int, got "
                f"{self.sequence_index!r}"
            )
        if not isinstance(self.text_delta, str):
            raise ChainRpcMalformedError(
                f"text_delta must be str, got "
                f"{type(self.text_delta).__name__}"
            )
        if self.token_id is not None and (
            not isinstance(self.token_id, int)
            or isinstance(self.token_id, bool)
        ):
            raise ChainRpcMalformedError(
                f"token_id must be int or None, got "
                f"{type(self.token_id).__name__}"
            )
        if self.finish_reason is not None:
            if not isinstance(self.finish_reason, str):
                raise ChainRpcMalformedError(
                    f"finish_reason must be str or None, got "
                    f"{type(self.finish_reason).__name__}"
                )
            if self.finish_reason not in _FINISH_REASONS:
                raise ChainRpcMalformedError(
                    f"finish_reason must be one of "
                    f"{sorted(_FINISH_REASONS)}, got "
                    f"{self.finish_reason!r}"
                )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "sequence_index": self.sequence_index,
            "text_delta": self.text_delta,
        }
        # Optional fields use conditional encoding so frames without
        # token_id / finish_reason don't gain spurious null keys.
        if self.token_id is not None:
            out["token_id"] = self.token_id
        if self.finish_reason is not None:
            out["finish_reason"] = self.finish_reason
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenFrame":
        _expect_type(data, ChainRpcMessageType.TOKEN_FRAME)
        token_id_raw = data.get("token_id")
        if token_id_raw is not None and (
            not isinstance(token_id_raw, int)
            or isinstance(token_id_raw, bool)
        ):
            raise ChainRpcMalformedError(
                f"token_id must be int or null, got "
                f"{type(token_id_raw).__name__}"
            )
        finish_raw = data.get("finish_reason")
        if finish_raw is not None and not isinstance(finish_raw, str):
            raise ChainRpcMalformedError(
                f"finish_reason must be str or null, got "
                f"{type(finish_raw).__name__}"
            )
        text_raw = data.get("text_delta")
        if not isinstance(text_raw, str):
            raise ChainRpcMalformedError(
                "text_delta must be string"
            )
        return cls(
            request_id=_required_str(data, "request_id"),
            sequence_index=_required_int(data, "sequence_index"),
            text_delta=text_raw,
            token_id=token_id_raw,
            finish_reason=finish_raw,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class StreamFinalFrame:
    """Terminal frame on a streaming-token output stream
    (Phase 3.x.8).

    Carries the FULL signed ``RunLayerSliceResponse`` whose
    ``activation_blob`` is the UTF-8-encoded joined output text
    across all preceding ``TokenFrame.text_delta`` values. The
    embedded response's stage signature commits to this joined text
    via the existing ``signing_payload`` (hex-encodes
    ``activation_blob``) — so a relay that tampers any TokenFrame's
    ``text_delta`` causes the joined-bytes hash to diverge from
    what the stage signed, invalidating the stream as a whole.

    The ``activation_shape`` on the embedded response describes the
    joined-text bytes (always 1-D byte tensor of length
    ``len(joined_utf8)``); ``activation_dtype`` is ``"uint8"``.
    """

    response: RunLayerSliceResponse
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.STREAM_FINAL_FRAME.value

    def __post_init__(self) -> None:
        _validate_version(self.protocol_version)
        if not isinstance(self.response, RunLayerSliceResponse):
            raise ChainRpcMalformedError(
                f"response must be RunLayerSliceResponse, got "
                f"{type(self.response).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "response": self.response.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamFinalFrame":
        _expect_type(data, ChainRpcMessageType.STREAM_FINAL_FRAME)
        response_raw = data.get("response")
        if not isinstance(response_raw, dict):
            raise ChainRpcMalformedError(
                f"response must be dict, got "
                f"{type(response_raw).__name__}"
            )
        return cls(
            response=RunLayerSliceResponse.from_dict(response_raw),
            protocol_version=_required_int(data, "protocol_version"),
        )


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11 — KV-cache eviction signal
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvictCacheRequest:
    """Phase 3.x.11 — executor → stage broadcast.

    Sent on every terminal exit path of
    ``RpcChainExecutor._execute_chain_streaming_sharded``
    (natural completion, EOS, max_tokens cap, GeneratorExit
    cancellation, exception). Best-effort delivery — server-side
    TTL sweeper bounds the leak window if a broadcast misses
    (e.g., stage temporarily unreachable, executor crash).

    Wire: a single ``request_id`` payload. No signature — the
    eviction signal is non-load-bearing for correctness (the
    server-side TTL sweeper closes the same hole eventually).
    Operators wanting cryptographic eviction proof wire a
    pre-shared HMAC at the transport layer (out of v1 scope).

    The handler at the server side calls
    ``KVCacheManager.evict(request_id)`` which is idempotent —
    duplicate broadcasts (e.g., from retried executor close) are
    safe.
    """

    request_id: str
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.EVICT_CACHE_REQUEST.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvictCacheRequest":
        _expect_type(data, ChainRpcMessageType.EVICT_CACHE_REQUEST)
        return cls(
            request_id=_required_str(data, "request_id"),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class EvictCacheResponse:
    """Phase 3.x.11 — stage → executor ack for ``EvictCacheRequest``.

    ``evicted=True`` indicates the manager actually had a handle
    for ``request_id`` and dropped it; ``evicted=False`` means
    the no-op idempotent path (e.g., handle was already evicted
    by TTL sweeper before the executor's broadcast arrived).

    Operators may surface ``evicted=False`` rates as a metric to
    detect misbehaving cache lifecycles; the executor itself
    treats both responses as success.
    """

    request_id: str
    evicted: bool
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.EVICT_CACHE_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        if not isinstance(self.evicted, bool):
            raise ChainRpcMalformedError(
                f"evicted must be bool, got "
                f"{type(self.evicted).__name__}"
            )
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "evicted": self.evicted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvictCacheResponse":
        _expect_type(data, ChainRpcMessageType.EVICT_CACHE_RESPONSE)
        evicted_raw = data.get("evicted")
        if not isinstance(evicted_raw, bool):
            raise ChainRpcMalformedError(
                f"evicted must be bool, got {type(evicted_raw).__name__}"
            )
        return cls(
            request_id=_required_str(data, "request_id"),
            evicted=evicted_raw,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class RollbackCacheRequest:
    """Phase 3.x.11.y — executor → stage on rejected speculative
    suffix.

    Sent by the executor's speculation loop when a VERIFY round
    accepted fewer than K draft tokens. The server-side handler
    truncates the LAST ``n_positions_to_drop`` positions from
    the cache. Idempotent: rollback past position 0 (or larger
    than current seq_len) is a no-op + returns
    ``rolled_back=False``.

    Wire: ``request_id`` + ``n_positions_to_drop``. No signature
    — non-load-bearing for correctness because the next VERIFY
    dispatch's signed response commits to the (correct) cache
    state, and a malicious rollback that drops too much state
    surfaces as MALFORMED_REQUEST or wrong-output downstream.

    Cap on ``n_positions_to_drop`` is ``MAX_VERIFY_BATCH_TOKENS``
    — defends against a hostile peer claiming a huge rollback
    that doesn't correspond to any speculation round.
    """

    request_id: str
    n_positions_to_drop: int
    # Phase 3.x.11.q.y' — replay-accepted-prefix protocol. When the
    # executor runs in always_rollback_k mode (constant-K wire-side
    # rollback, see threat-model addendum §3.8), the rollback is
    # accompanied by the accepted prefix bytes so the stage can
    # repopulate its cache to the correct state after the constant-K
    # truncation. Mutual-exclusion: at most one of
    # ``replay_accepted_prefix`` (plaintext) and
    # ``encrypted_replay_accepted_prefix`` (AES-GCM ciphertext under
    # AAD = request_id || stage_index || b"rollback") may be set.
    # When neither is set, this is a v1-style truncation-only
    # rollback (backwards-compat path; pre-q.y' deployments).
    replay_accepted_prefix: Optional[Tuple[int, ...]] = None
    encrypted_replay_accepted_prefix: Optional[bytes] = None
    # Phase 3.x.11.q.y' — stage_index for AAD binding when
    # encrypted_replay_accepted_prefix is set. The executor sets
    # this to the destination stage's chain_stage_index so the
    # server can derive the correct AAD without extra context.
    # Required co-set with encrypted_replay_accepted_prefix (else
    # the server can't decrypt). Range [0, 255] mirrors the
    # ProbsCipher AAD validator.
    target_stage_index: Optional[int] = None
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.ROLLBACK_CACHE_REQUEST.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        if (
            isinstance(self.n_positions_to_drop, bool)
            or not isinstance(self.n_positions_to_drop, int)
        ):
            raise ChainRpcMalformedError(
                f"n_positions_to_drop must be int, got "
                f"{type(self.n_positions_to_drop).__name__}"
            )
        if self.n_positions_to_drop < 0:
            raise ChainRpcMalformedError(
                f"n_positions_to_drop must be non-negative, got "
                f"{self.n_positions_to_drop}"
            )
        if self.n_positions_to_drop > MAX_VERIFY_BATCH_TOKENS:
            raise ChainRpcMalformedError(
                f"n_positions_to_drop {self.n_positions_to_drop} "
                f"exceeds cap {MAX_VERIFY_BATCH_TOKENS}"
            )
        # Phase 3.x.11.q.y' — replay-prefix mutual-exclusion +
        # type/cap validators.
        if (
            self.replay_accepted_prefix is not None
            and self.encrypted_replay_accepted_prefix is not None
        ):
            raise ChainRpcMalformedError(
                "RollbackCacheRequest cannot carry both "
                "replay_accepted_prefix (plaintext) and "
                "encrypted_replay_accepted_prefix (ciphertext); "
                "set exactly one or neither (v1-truncation-only)"
            )
        if self.replay_accepted_prefix is not None:
            if not isinstance(self.replay_accepted_prefix, tuple):
                raise ChainRpcMalformedError(
                    f"replay_accepted_prefix must be tuple, got "
                    f"{type(self.replay_accepted_prefix).__name__}"
                )
            if len(self.replay_accepted_prefix) > MAX_VERIFY_BATCH_TOKENS:
                raise ChainRpcMalformedError(
                    f"replay_accepted_prefix length "
                    f"{len(self.replay_accepted_prefix)} exceeds "
                    f"cap {MAX_VERIFY_BATCH_TOKENS}"
                )
            for i, t in enumerate(self.replay_accepted_prefix):
                if isinstance(t, bool) or not isinstance(t, int):
                    raise ChainRpcMalformedError(
                        f"replay_accepted_prefix[{i}] must be int, "
                        f"got {type(t).__name__}"
                    )
                if t < 0:
                    raise ChainRpcMalformedError(
                        f"replay_accepted_prefix[{i}] must be "
                        f"non-negative, got {t}"
                    )
        if self.encrypted_replay_accepted_prefix is not None:
            if not isinstance(
                self.encrypted_replay_accepted_prefix,
                (bytes, bytearray),
            ):
                raise ChainRpcMalformedError(
                    f"encrypted_replay_accepted_prefix must be "
                    f"bytes, got "
                    f"{type(self.encrypted_replay_accepted_prefix).__name__}"
                )
            if len(self.encrypted_replay_accepted_prefix) == 0:
                raise ChainRpcMalformedError(
                    "encrypted_replay_accepted_prefix must be "
                    "non-empty when set"
                )
            # Plaintext prefix is at most MAX_VERIFY_BATCH_TOKENS
            # int64s = MAX_VERIFY_BATCH_TOKENS * 8 bytes. AES-GCM
            # adds 12-byte nonce + 16-byte tag = 28 bytes. Cap at
            # 4096 bytes accommodates K up to ~509 with overhead.
            if len(self.encrypted_replay_accepted_prefix) > 4096:
                raise ChainRpcMalformedError(
                    f"encrypted_replay_accepted_prefix bytes "
                    f"{len(self.encrypted_replay_accepted_prefix)} "
                    f"exceeds 4096 cap"
                )
            # Phase 3.x.11.q.y' — co-set invariant: target_stage_index
            # MUST be set when encrypted_replay_accepted_prefix is
            # set (the AAD binding requires it). Defends against a
            # caller-side bug where the cipher uses one stage_index
            # but the server tries to decrypt with another.
            if self.target_stage_index is None:
                raise ChainRpcMalformedError(
                    "encrypted_replay_accepted_prefix requires "
                    "target_stage_index to be set (AAD binding)"
                )
        if self.target_stage_index is not None:
            if (
                isinstance(self.target_stage_index, bool)
                or not isinstance(self.target_stage_index, int)
            ):
                raise ChainRpcMalformedError(
                    f"target_stage_index must be int, got "
                    f"{type(self.target_stage_index).__name__}"
                )
            if not (0 <= self.target_stage_index <= 255):
                raise ChainRpcMalformedError(
                    f"target_stage_index must be in [0, 255], "
                    f"got {self.target_stage_index}"
                )
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "n_positions_to_drop": self.n_positions_to_drop,
        }
        # Phase 3.x.11.q.y' — omit-when-None canonical encoding for
        # backwards-compat (pre-q.y' rollbacks remain byte-identical).
        if self.replay_accepted_prefix is not None:
            out["replay_accepted_prefix"] = list(
                self.replay_accepted_prefix,
            )
        if self.encrypted_replay_accepted_prefix is not None:
            out["encrypted_replay_accepted_prefix_hex"] = bytes(
                self.encrypted_replay_accepted_prefix,
            ).hex()
        if self.target_stage_index is not None:
            out["target_stage_index"] = int(self.target_stage_index)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RollbackCacheRequest":
        _expect_type(data, ChainRpcMessageType.ROLLBACK_CACHE_REQUEST)
        n_raw = data.get("n_positions_to_drop")
        if isinstance(n_raw, bool) or not isinstance(n_raw, int):
            raise ChainRpcMalformedError(
                f"n_positions_to_drop must be int, got "
                f"{type(n_raw).__name__}"
            )
        replay_raw = data.get("replay_accepted_prefix")
        replay_tuple: Optional[Tuple[int, ...]] = None
        if replay_raw is not None:
            if not isinstance(replay_raw, list):
                raise ChainRpcMalformedError(
                    f"replay_accepted_prefix must be list, got "
                    f"{type(replay_raw).__name__}"
                )
            replay_tuple = tuple(int(t) for t in replay_raw)
        enc_hex = data.get("encrypted_replay_accepted_prefix_hex")
        enc_bytes: Optional[bytes] = None
        if enc_hex is not None:
            if not isinstance(enc_hex, str):
                raise ChainRpcMalformedError(
                    f"encrypted_replay_accepted_prefix_hex must be "
                    f"str, got {type(enc_hex).__name__}"
                )
            try:
                enc_bytes = bytes.fromhex(enc_hex)
            except ValueError as exc:
                raise ChainRpcMalformedError(
                    f"encrypted_replay_accepted_prefix_hex is not "
                    f"valid hex: {exc}"
                ) from exc
        target_raw = data.get("target_stage_index")
        target_idx: Optional[int] = None
        if target_raw is not None:
            if isinstance(target_raw, bool) or not isinstance(
                target_raw, int,
            ):
                raise ChainRpcMalformedError(
                    f"target_stage_index must be int, got "
                    f"{type(target_raw).__name__}"
                )
            target_idx = int(target_raw)
        return cls(
            request_id=_required_str(data, "request_id"),
            n_positions_to_drop=int(n_raw),
            replay_accepted_prefix=replay_tuple,
            encrypted_replay_accepted_prefix=enc_bytes,
            target_stage_index=target_idx,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class RollbackCacheResponse:
    """Phase 3.x.11.y — stage → executor ack for
    ``RollbackCacheRequest``.

    ``rolled_back=True`` means the manager truncated at least
    one position; ``rolled_back=False`` is the no-op idempotent
    path (e.g., the request asked to drop more positions than
    the cache held — TTL eviction may have raced; or the cache
    was already empty).

    ``actual_dropped`` returns the actual number of positions
    truncated — useful for metrics + executor-side rollback
    accounting (the executor knows EXACTLY how much it tried
    to roll back, vs. how much actually came off).
    """

    request_id: str
    rolled_back: bool
    actual_dropped: int
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.ROLLBACK_CACHE_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        if not isinstance(self.rolled_back, bool):
            raise ChainRpcMalformedError(
                f"rolled_back must be bool, got "
                f"{type(self.rolled_back).__name__}"
            )
        if (
            isinstance(self.actual_dropped, bool)
            or not isinstance(self.actual_dropped, int)
        ):
            raise ChainRpcMalformedError(
                f"actual_dropped must be int, got "
                f"{type(self.actual_dropped).__name__}"
            )
        if self.actual_dropped < 0:
            raise ChainRpcMalformedError(
                f"actual_dropped must be non-negative, got "
                f"{self.actual_dropped}"
            )
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "rolled_back": self.rolled_back,
            "actual_dropped": self.actual_dropped,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RollbackCacheResponse":
        _expect_type(data, ChainRpcMessageType.ROLLBACK_CACHE_RESPONSE)
        rolled_raw = data.get("rolled_back")
        if not isinstance(rolled_raw, bool):
            raise ChainRpcMalformedError(
                f"rolled_back must be bool, got "
                f"{type(rolled_raw).__name__}"
            )
        actual_raw = data.get("actual_dropped")
        if isinstance(actual_raw, bool) or not isinstance(actual_raw, int):
            raise ChainRpcMalformedError(
                f"actual_dropped must be int, got "
                f"{type(actual_raw).__name__}"
            )
        return cls(
            request_id=_required_str(data, "request_id"),
            rolled_back=rolled_raw,
            actual_dropped=int(actual_raw),
            protocol_version=_required_int(data, "protocol_version"),
        )


# ──────────────────────────────────────────────────────────────────────────
# Codec
# ──────────────────────────────────────────────────────────────────────────


_MESSAGE_TYPE_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST.value: RunLayerSliceRequest.from_dict,
    ChainRpcMessageType.RUN_LAYER_SLICE_RESPONSE.value: RunLayerSliceResponse.from_dict,
    ChainRpcMessageType.STAGE_ERROR.value: StageError.from_dict,
    ChainRpcMessageType.ACTIVATION_CHUNK.value: ActivationChunk.from_dict,
    ChainRpcMessageType.TOKEN_FRAME.value: TokenFrame.from_dict,
    ChainRpcMessageType.STREAM_FINAL_FRAME.value: StreamFinalFrame.from_dict,
    ChainRpcMessageType.EVICT_CACHE_REQUEST.value: EvictCacheRequest.from_dict,
    ChainRpcMessageType.EVICT_CACHE_RESPONSE.value: EvictCacheResponse.from_dict,
    ChainRpcMessageType.ROLLBACK_CACHE_REQUEST.value: RollbackCacheRequest.from_dict,
    ChainRpcMessageType.ROLLBACK_CACHE_RESPONSE.value: RollbackCacheResponse.from_dict,
}


def parse_message(payload: bytes) -> Any:
    """Decode wire-format bytes into the matching dataclass.

    Mirrors the manifest-DHT and profile-DHT parsers. Size cap fires
    BEFORE ``json.loads`` allocates; version mismatch raises a
    dedicated exception so callers can distinguish from malformed
    input.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise ChainRpcMalformedError(
            f"payload must be bytes, got {type(payload).__name__}"
        )
    if len(payload) > MAX_HANDSHAKE_BYTES:
        raise ChainRpcMalformedError(
            f"payload exceeds MAX_HANDSHAKE_BYTES "
            f"({len(payload)} > {MAX_HANDSHAKE_BYTES})"
        )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ChainRpcMalformedError(f"JSON parse failed: {exc}") from exc
    if not isinstance(data, dict):
        raise ChainRpcMalformedError(
            f"top-level message must be dict, got {type(data).__name__}"
        )
    msg_type = data.get("type")
    if not isinstance(msg_type, str):
        raise ChainRpcMalformedError(
            f"missing or non-string 'type' field; got {msg_type!r}"
        )
    constructor = _MESSAGE_TYPE_REGISTRY.get(msg_type)
    if constructor is None:
        raise ChainRpcUnknownTypeError(
            f"unknown message type {msg_type!r}; "
            f"known: {sorted(_MESSAGE_TYPE_REGISTRY)}"
        )
    # v2 nodes accept v1 + v2 (forward-compat for rolling deploys);
    # v1 nodes still reject anything other than 1 because they only
    # know about CHAIN_RPC_PROTOCOL_VERSION = 1. bool rejection
    # preserved (M1 round-1 fix from Phase 3.x.7 Task 8).
    #
    # L1 round-1 (3.x.7.1): missing or non-int protocol_version is a
    # version-negotiation failure, not a malformed-message failure —
    # surface it as ``ChainRpcVersionMismatchError`` so the executor
    # maps it to ``UNSUPPORTED_VERSION`` instead of
    # ``MALFORMED_RESPONSE``.
    if "protocol_version" not in data:
        raise ChainRpcVersionMismatchError(
            "peer message missing 'protocol_version'; "
            f"local SUPPORTED_PROTOCOL_VERSIONS="
            f"{sorted(SUPPORTED_PROTOCOL_VERSIONS)}"
        )
    version = data["protocol_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ChainRpcVersionMismatchError(
            f"peer protocol_version must be int, got "
            f"{type(version).__name__} ({version!r}); "
            f"local SUPPORTED_PROTOCOL_VERSIONS="
            f"{sorted(SUPPORTED_PROTOCOL_VERSIONS)}"
        )
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ChainRpcVersionMismatchError(
            f"peer protocol_version={version}; "
            f"local SUPPORTED_PROTOCOL_VERSIONS="
            f"{sorted(SUPPORTED_PROTOCOL_VERSIONS)}"
        )
    return constructor(data)


def encode_message(message: Any) -> bytes:
    """Encode a wire-format dataclass to canonical JSON bytes."""
    if not hasattr(message, "to_dict"):
        raise ChainRpcMalformedError(
            f"message must have to_dict(); got {type(message).__name__}"
        )
    return json.dumps(message.to_dict(), sort_keys=True).encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────
# Validation helpers (private)
# ──────────────────────────────────────────────────────────────────────────


def _validate_str_field(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ChainRpcMalformedError(
            f"{name} must be str, got {type(value).__name__}"
        )
    if not value:
        raise ChainRpcMalformedError(f"{name} must be non-empty")


def _validate_version(version: Any) -> None:
    # bool is a subclass of int in Python; reject explicitly so a
    # peer sending {"protocol_version": true} doesn't slip through
    # via True == 1.
    if not isinstance(version, int) or isinstance(version, bool):
        raise ChainRpcMalformedError(
            f"protocol_version must be int, got {type(version).__name__}"
        )


def _expect_type(
    data: Dict[str, Any], expected: ChainRpcMessageType
) -> None:
    actual = data.get("type")
    if actual != expected.value:
        raise ChainRpcMalformedError(
            f"expected type={expected.value!r}, got {actual!r}"
        )


def _required_str(data: Dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ChainRpcMalformedError(
            f"{field_name} must be non-empty str"
        )
    return value


def _required_int(data: Dict[str, Any], field_name: str) -> int:
    value = data.get(field_name)
    # Reject bool explicitly; isinstance(True, int) is True in Python
    # so {"chain_stage_index": true} would otherwise produce a token
    # whose int field is True, polluting downstream telemetry +
    # equality semantics.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ChainRpcMalformedError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    return value


def _required_number(data: Dict[str, Any], field_name: str) -> float:
    value = data.get(field_name)
    if not isinstance(value, (int, float)):
        raise ChainRpcMalformedError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    return float(value)
