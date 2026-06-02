"""Phase 3.x.7 Task 2 — Cross-host LayerStageServer.

Per-node server that receives a ``RunLayerSliceRequest`` from an
upstream stage (or the executor itself for the chain head), validates
it through the 8-step gate, executes the requested contiguous layer
slice via a local layer runner, and returns a stage-signed
``RunLayerSliceResponse``.

8-step validation order (matches design plan §3.4):

  1. Parse the request                  → MALFORMED_REQUEST
  2. verify_with_anchor on token        → INVALID_TOKEN
  3. token.deadline_unix > clock()      → DEADLINE_EXCEEDED
                                          (also checks request deadline)
  4. registry.get(model_id)             → MODEL_NOT_FOUND
  5. layer_range covered by local shards → SHARD_MISSING
  6. privacy_tier vs tee_runtime.tee_type → TIER_GATE
  7. Decode activation → run layer slice → encode output → handle TIMEOUT
  8. Sign the response under self.identity

"Never raises" invariant: every failure is mapped to a structured
``StageError`` and returned as encoded bytes. Caller transport never
sees an exception. Mirrors Phase 3.x.5 ``ManifestDHTServer.handle()``
and Phase 3.x.6 ``ProfileDHTServer.handle()``.

This module is intentionally narrow: the public surface is the
``LayerStageServer.handle(bytes) → bytes`` callable plus the
``LayerSliceRunner`` Protocol that callers inject for production
wiring (Task 6 wraps Phase 2 Ring 8's ``TensorParallelExecutor``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Protocol, Tuple

import numpy as np

from prsm.compute.chain_rpc.activation_codec import (
    DEFAULT_CHUNK_BYTES_ACTIVATION,
    ActivationCodecError,
    chunk_activation,
    decode_activation,
    encode_activation,
    reassemble_chunked,
)
from prsm.compute.chain_rpc.kv_cache import KVCacheManager
from prsm.compute.chain_rpc.protocol import (
    ActivationChunk,
    ChainRpcMalformedError,
    ChainRpcProtocolError,
    ChainRpcUnknownTypeError,
    ChainRpcVersionMismatchError,
    DecodeMode,
    EvictCacheRequest,
    EvictCacheResponse,
    RollbackCacheRequest,
    RollbackCacheResponse,
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    response_input_commitment_for_request,
    StageError,
    StageErrorCode,
    StreamFinalFrame,
    TokenFrame,
    encode_message,
    parse_message,
)
from prsm.node.shard_streaming import ShardChunk
from prsm.compute.tee.models import HARDWARE_TEE_TYPES, PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity


logger = logging.getLogger(__name__)


_UNKNOWN_REQUEST_ID = "<unknown>"

# Phase 3.x.7.1 H1 round-1 remediation: hard cap on the assembled
# activation payload size for the streamed path. Chosen at 1 GiB —
# generous enough for batched LLM activations (e.g. batch=16 of 64 MiB
# each = 1 GiB) but short of the kind of inflated values a hostile
# peer could ship to exhaust server memory. Operators tuning this
# down for stricter memory budgets pass it via the constructor.
DEFAULT_MAX_STREAMED_PAYLOAD_BYTES = 1 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class _GateResult:
    """Outcome of ``_run_validation_gates``. Either ``model`` is
    populated (all gates pass) or ``error`` is populated (first gate
    that failed)."""

    model: Optional[Any] = None
    error: Optional[Tuple[StageErrorCode, str]] = None


# ──────────────────────────────────────────────────────────────────────────
# LayerSliceRunner Protocol — what the server calls to actually run layers
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerSliceResult:
    """Per-stage output bundled for the response.

    Mirrors the shape that ``RunLayerSliceResponse.sign(...)`` expects.
    Production wiring (Task 6) wraps the existing
    ``TensorParallelExecutor`` to satisfy this Protocol; tests inject
    a fake.
    """

    output: np.ndarray
    duration_seconds: float
    tee_attestation: bytes
    tee_type: TEEType
    epsilon_spent: float


class LayerSliceRunner(Protocol):
    """Adapter that runs a contiguous layer range on a local model.

    The runner is responsible for:
      - Loading the requested layer slice's weights into the local
        accelerator (or reusing if already loaded).
      - Running the forward pass for those layers on the input
        ``activation`` ndarray.
      - Applying DP noise + producing the TEE attestation IF this is
        the chain's tail stage (the server passes ``apply_dp=True``
        only when it knows from the request layout that this is the
        final stage; v1 always sets it based on the layer_range
        covering the model's last layer).
      - Returning a ``LayerSliceResult`` with the post-layer-slice
        activation, timing, attestation, and any DP epsilon spent.

    Errors propagate. The server wraps the call in try/except and
    maps to ``StageError(INTERNAL_ERROR)``.
    """

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
    ) -> LayerSliceResult:
        ...


# ──────────────────────────────────────────────────────────────────────────
# Layer-coverage helpers
# ──────────────────────────────────────────────────────────────────────────


def _shards_cover_range(model: Any, layer_range: Tuple[int, int]) -> bool:
    """Return True iff every layer in [start, end) is covered by at
    least one shard's ``layer_range`` on the local model.

    Tolerates models where shards have ``layer_range == (0, 0)`` —
    that's the registry's "layer assignment unset" sentinel from
    Phase 3.x.2 Task 5; production allocators set real ranges before
    publishing. When EVERY shard has the sentinel, we treat the
    model as covering all layers (single-stage fallback) to preserve
    the existing single-host code path.
    """
    if not hasattr(model, "shards"):
        return False
    shards = list(model.shards)
    if not shards:
        return False
    sentinel = all(
        getattr(s, "layer_range", (0, 0)) == (0, 0) for s in shards
    )
    if sentinel:
        return True
    start, end = layer_range
    if start >= end:
        return False
    covered = [False] * (end - start)
    for shard in shards:
        s_start, s_end = getattr(shard, "layer_range", (0, 0))
        if s_start == 0 and s_end == 0:
            continue
        for layer in range(max(s_start, start), min(s_end, end)):
            covered[layer - start] = True
    return all(covered)


def _is_final_stage(model: Any, layer_range: Tuple[int, int]) -> bool:
    """Heuristic: this stage is the tail iff the requested
    layer_range's end is at or beyond the model's last layer.

    For shards using the (0, 0) sentinel from Phase 3.x.2, we fall
    back to assuming non-final (DP noise applied at the executor
    level instead). v1 conservative — final-stage detection is best-
    effort signal for the LayerSliceRunner.
    """
    if not hasattr(model, "shards") or not model.shards:
        return False
    max_end = 0
    for shard in model.shards:
        s_start, s_end = getattr(shard, "layer_range", (0, 0))
        if s_start == 0 and s_end == 0:
            continue
        max_end = max(max_end, s_end)
    if max_end == 0:
        return False
    return layer_range[1] >= max_end


# ──────────────────────────────────────────────────────────────────────────
# LayerStageServer
# ──────────────────────────────────────────────────────────────────────────


class LayerStageServer:
    """Per-node handler for ``RunLayerSliceRequest``.

    Constructor args:
      identity      The stage's NodeIdentity (used to sign responses).
                    The same identity is what callers register on the
                    Phase 3.x.3 anchor.
      registry      ``ModelRegistry`` for ``model_id → ShardedModel``
                    lookup. Production: ``FilesystemModelRegistry``.
      runner        ``LayerSliceRunner`` that actually executes the
                    layer slice.
      tee_runtime   Phase 2 Ring 8 ``TEERuntime``. The server reads
                    ``tee_runtime.tee_type`` to enforce the privacy-
                    tier gate.
      anchor        Phase 3.x.3 anchor for verifying upstream tokens.
      clock         Injected for tests; defaults to ``time.time``.
    """

    def __init__(
        self,
        *,
        identity: NodeIdentity,
        registry: Any,
        runner: LayerSliceRunner,
        tee_runtime: Any,
        anchor: Any,
        clock: Callable[[], float] = time.time,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES_ACTIVATION,
        max_streamed_payload_bytes: int = DEFAULT_MAX_STREAMED_PAYLOAD_BYTES,
        streaming_runner: Optional[Any] = None,
        tier_c_streaming_decorator: Optional[
            Callable[[Any], Any]
        ] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        sharded_runner: Optional[Any] = None,
        encrypted_probs_cipher: Optional[Any] = None,
    ) -> None:
        if identity is None or not hasattr(identity, "node_id"):
            raise RuntimeError(
                "LayerStageServer requires a NodeIdentity for response signing"
            )
        if registry is None or not hasattr(registry, "get"):
            raise RuntimeError(
                "LayerStageServer requires a ModelRegistry with .get(model_id)"
            )
        if runner is None or not hasattr(runner, "run_layer_range"):
            raise RuntimeError(
                "LayerStageServer requires a LayerSliceRunner with "
                ".run_layer_range(...)"
            )
        if tee_runtime is None or not hasattr(tee_runtime, "tee_type"):
            raise RuntimeError(
                "LayerStageServer requires a tee_runtime with .tee_type"
            )
        if anchor is None or not hasattr(anchor, "lookup"):
            raise RuntimeError(
                "LayerStageServer requires an anchor with .lookup(node_id)"
            )
        if chunk_bytes <= 0:
            raise ValueError(
                f"chunk_bytes must be positive, got {chunk_bytes}"
            )
        if max_streamed_payload_bytes <= 0:
            raise ValueError(
                f"max_streamed_payload_bytes must be positive, got "
                f"{max_streamed_payload_bytes}"
            )
        # Phase 3.x.8: streaming_runner is optional. When None, the
        # token-stream handler returns a structured StageError on any
        # streaming request (server doesn't support streaming). When
        # set, must be a callable-bearing object with
        # ``run_layer_slice_streaming``.
        if streaming_runner is not None and not hasattr(
            streaming_runner, "run_layer_slice_streaming"
        ):
            raise RuntimeError(
                "LayerStageServer: streaming_runner must expose "
                ".run_layer_slice_streaming(...) if provided"
            )
        # Phase 3.x.10.y Task 4: tier_c_streaming_decorator is
        # operator-supplied callable that wraps the streaming
        # runner per-request when the request is Tier C content.
        # When set, it MUST be callable; we can't fully validate
        # what it returns at construction (decorator may close
        # over operator state and only fail when invoked), but
        # the per-dispatch wrap result IS validated structurally
        # (must expose ``run_layer_slice_streaming``) before
        # invocation.
        # When unset + Tier C streaming dispatch arrives, the
        # token-stream handler rejects with INTERNAL_ERROR
        # ("Tier C streaming requires constant-time padding
        # decorator") — operators must opt in to Tier C
        # streaming explicitly per the timing-sidechannel memo
        # §6 default-deny invariant.
        if (
            tier_c_streaming_decorator is not None
            and not callable(tier_c_streaming_decorator)
        ):
            raise RuntimeError(
                "LayerStageServer: tier_c_streaming_decorator must "
                "be callable if provided"
            )
        # Phase 3.x.11 — kv_cache_manager is opt-in. When wired,
        # the server handles EvictCacheRequest by routing to
        # ``manager.evict``; when None, EvictCacheRequests are
        # rejected with INTERNAL_ERROR (operator must wire the
        # manager + the sharded-decode tail-runner together at
        # node startup).
        if (
            kv_cache_manager is not None
            and not isinstance(kv_cache_manager, KVCacheManager)
        ):
            raise RuntimeError(
                "LayerStageServer: kv_cache_manager must be a "
                "KVCacheManager instance if provided"
            )
        # Phase 3.x.11 — sharded_runner is opt-in. When wired,
        # the server routes ALL RunLayerSliceRequests to it via
        # ``run_layer_slice_unary`` (sharded autoregressive
        # decode); the regular ``runner`` becomes the back-compat
        # path for non-sharded operators. A node that wants to
        # serve sharded decode wires ``sharded_runner`` +
        # ``kv_cache_manager`` together.
        if sharded_runner is not None and not callable(
            getattr(sharded_runner, "run_layer_slice_unary", None),
        ):
            raise RuntimeError(
                "LayerStageServer: sharded_runner must expose "
                ".run_layer_slice_unary(...) if provided "
                "(see ShardedAutoregressiveRunner)"
            )

        self._identity = identity
        self._registry = registry
        self._runner = runner
        self._tee_runtime = tee_runtime
        self._anchor = anchor
        self._clock = clock
        self._chunk_bytes = int(chunk_bytes)
        self._max_streamed_payload_bytes = int(max_streamed_payload_bytes)
        self._streaming_runner = streaming_runner
        self._tier_c_streaming_decorator = tier_c_streaming_decorator
        self._kv_cache_manager = kv_cache_manager
        self._sharded_runner = sharded_runner
        # Phase 3.x.11.q.y — encrypted_probs_cipher is opt-in. When
        # wired, the server decrypts encrypted_proposed_token_probs
        # at the boundary using the same cipher (operator distributes
        # the PSK out-of-band). When None + the wire field is set,
        # the server returns MALFORMED_REQUEST (the executor
        # mistakenly dispatched encrypted probs to a server that
        # can't decrypt — operator misconfig).
        if encrypted_probs_cipher is not None:
            for method in ("encrypt", "decrypt"):
                if not callable(
                    getattr(encrypted_probs_cipher, method, None),
                ):
                    raise RuntimeError(
                        f"LayerStageServer: encrypted_probs_cipher "
                        f"must implement {method}(...) per "
                        f"prsm.compute.chain_rpc.probs_cipher.ProbsCipher"
                    )
        self._encrypted_probs_cipher = encrypted_probs_cipher

    # ── public API ────────────────────────────────────────────────────

    def handle(self, request_bytes: bytes) -> bytes:
        """Parse → validate → run → encode response. NEVER raises.

        Wraps the entire pipeline in defensive try/except so any
        failure path emits a structured ``StageError`` rather than
        propagating an exception through the transport.
        """
        # Step 1: parse.
        try:
            request = parse_message(request_bytes)
        except ChainRpcVersionMismatchError as exc:
            return self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.UNSUPPORTED_VERSION,
                str(exc),
            )
        except (ChainRpcMalformedError, ChainRpcUnknownTypeError) as exc:
            return self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except ChainRpcProtocolError as exc:
            return self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LayerStageServer.handle: unexpected parse failure")
            return self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during parse: {exc.__class__.__name__}",
            )

        # Phase 3.x.11 Task 6: EvictCacheRequest is a separate
        # cache-management op that doesn't run through the
        # normal layer-slice dispatch.
        if isinstance(request, EvictCacheRequest):
            try:
                return self._handle_evict_cache(request)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "LayerStageServer.handle: unexpected eviction "
                    "failure for request_id=%r",
                    request.request_id,
                )
                return self._error(
                    request.request_id,
                    StageErrorCode.INTERNAL_ERROR,
                    f"internal error during evict_cache: "
                    f"{exc.__class__.__name__}",
                )

        # Phase 3.x.11.y Task 6: RollbackCacheRequest — speculative-
        # decoding rollback after rejected suffix. Routes to
        # ``KVCacheManager.rollback`` via the runner's
        # ``rollback_cache`` wrapper (which provides the model's
        # ``truncate_cache`` as the manager's truncate_fn).
        if isinstance(request, RollbackCacheRequest):
            try:
                return self._handle_rollback_cache(request)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "LayerStageServer.handle: unexpected rollback "
                    "failure for request_id=%r",
                    request.request_id,
                )
                return self._error(
                    request.request_id,
                    StageErrorCode.INTERNAL_ERROR,
                    f"internal error during rollback_cache: "
                    f"{exc.__class__.__name__}",
                )

        if not isinstance(request, RunLayerSliceRequest):
            return self._error(
                getattr(request, "request_id", _UNKNOWN_REQUEST_ID),
                StageErrorCode.MALFORMED_REQUEST,
                f"server only handles RunLayerSliceRequest; got "
                f"{type(request).__name__}",
            )

        # Step 2-8 in a try/except so any unexpected raise gets mapped
        # to INTERNAL_ERROR rather than propagating.
        try:
            return self._dispatch(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer.handle: unexpected dispatch failure for "
                "request_id=%r",
                request.request_id,
            )
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during dispatch: {exc.__class__.__name__}",
            )

    # ── streamed-path public entry ────────────────────────────────────

    def handle_streamed(
        self,
        manifest_bytes: bytes,
        chunk_iter: Iterable[bytes],
    ) -> Tuple[bytes, Iterable[bytes]]:
        """Parse manifest → validate → assemble chunks → run → chunk
        output → sign + return.

        Mirrors ``handle()`` for the v2 streamed path. Validation
        gates (token verify, deadline, registry, shard coverage, tier
        gate) fire BEFORE chunk consumption so a peer that exhausts
        the stream without a valid token wastes only the manifest
        parse cost.

        NEVER raises. Returns ``(error_bytes, empty_iter)`` on any
        failure path so the streaming transport never sees an
        exception.
        """
        # Step 1: parse the manifest.
        try:
            request = parse_message(manifest_bytes)
        except ChainRpcVersionMismatchError as exc:
            return self._streamed_error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.UNSUPPORTED_VERSION,
                str(exc),
            )
        except (ChainRpcMalformedError, ChainRpcUnknownTypeError) as exc:
            return self._streamed_error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except ChainRpcProtocolError as exc:
            return self._streamed_error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer.handle_streamed: parse raised"
            )
            return self._streamed_error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during parse: {exc.__class__.__name__}",
            )

        if not isinstance(request, RunLayerSliceRequest):
            return self._streamed_error(
                getattr(request, "request_id", _UNKNOWN_REQUEST_ID),
                StageErrorCode.MALFORMED_REQUEST,
                f"streamed handler only accepts RunLayerSliceRequest; got "
                f"{type(request).__name__}",
            )

        # Streamed-handler contract: the request manifest MUST carry
        # an activation_manifest. A peer using the streamed RPC method
        # for an inline-formatted request is a protocol violation;
        # reject explicitly.
        if request.activation_manifest is None:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.MALFORMED_REQUEST,
                "streamed dispatch: request lacks activation_manifest "
                "(use handle() for inline requests)",
            )

        try:
            return self._dispatch_streamed(request, chunk_iter)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer.handle_streamed: dispatch raised "
                "for request_id=%r",
                request.request_id,
            )
            return self._streamed_error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during dispatch: {exc.__class__.__name__}",
            )

    # ── token-stream public entry (Phase 3.x.8) ──────────────────────

    def handle_token_stream(
        self,
        request_bytes: bytes,
    ) -> Iterable[bytes]:
        """Streaming-decode path. ONLY invoked for tail stages with
        ``streaming=True``. Mirrors ``handle()``'s validation gates;
        yields ``TokenFrame`` wire frames as the runner produces
        chunks; finalizes with exactly one ``StreamFinalFrame``
        carrying the signed ``RunLayerSliceResponse`` over the joined
        output bytes.

        NEVER raises through the wire boundary. Any failure path
        emits a SOLE encoded ``StageError`` frame and the iterator
        terminates. Consumers of this iterator MUST handle either
        terminal type:
          - ``StreamFinalFrame`` → stream completed cleanly.
          - ``StageError`` → stream errored (no signed receipt
            material available).

        v1 contract: streaming + chunked-input composition is NOT
        yet wired (a request with ``activation_manifest`` set
        rejects with ``MALFORMED_REQUEST``). Phase 3.x.8 Task 7
        revisits — for v1, the executor reassembles upstream chunked
        activations into inline form before dispatching to the
        streaming tail.
        """
        # Step 1: parse.
        try:
            request = parse_message(request_bytes)
        except ChainRpcVersionMismatchError as exc:
            yield self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.UNSUPPORTED_VERSION,
                str(exc),
            )
            return
        except (ChainRpcMalformedError, ChainRpcUnknownTypeError) as exc:
            yield self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
            return
        except ChainRpcProtocolError as exc:
            yield self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer.handle_token_stream: parse raised"
            )
            yield self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during parse: {exc.__class__.__name__}",
            )
            return

        if not isinstance(request, RunLayerSliceRequest):
            yield self._error(
                getattr(request, "request_id", _UNKNOWN_REQUEST_ID),
                StageErrorCode.MALFORMED_REQUEST,
                f"token-stream handler only accepts "
                f"RunLayerSliceRequest; got {type(request).__name__}",
            )
            return

        # Routing invariant: token-stream handler requires streaming=True.
        # A peer that calls this method on a non-streaming request is
        # using the wrong RPC method; reject.
        if not request.streaming:
            yield self._error(
                request.request_id,
                StageErrorCode.MALFORMED_REQUEST,
                "token-stream handler requires streaming=True "
                "(use handle() for unary requests)",
            )
            return

        # v1 scope: inline activations only on the streaming-output
        # path. Streaming-input + streaming-output composition is a
        # Task 7 follow-up; reject explicitly for now.
        if request.activation_manifest is not None:
            yield self._error(
                request.request_id,
                StageErrorCode.MALFORMED_REQUEST,
                "token-stream handler v1: chunked-input + streaming-"
                "output composition not yet supported (executor must "
                "reassemble upstream chunks before dispatching to the "
                "streaming tail)",
            )
            return

        # Operator config: token-stream requests rejected when no
        # streaming_runner is wired. INTERNAL_ERROR (not MALFORMED)
        # because the request itself is well-formed; the server is
        # simply not configured for streaming.
        if self._streaming_runner is None:
            yield self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                "server has no streaming_runner configured",
            )
            return

        try:
            yield from self._dispatch_token_stream(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer.handle_token_stream: dispatch raised "
                "for request_id=%r",
                request.request_id,
            )
            yield self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during dispatch: {exc.__class__.__name__}",
            )

    # ── pipeline ──────────────────────────────────────────────────────

    def _dispatch(self, request: RunLayerSliceRequest) -> bytes:
        # Streamed requests routed to handle_streamed; reject explicitly
        # if a streamed request reaches the inline path.
        if request.activation_manifest is not None:
            return self._error(
                request.request_id,
                StageErrorCode.MALFORMED_REQUEST,
                "inline dispatch: request carries activation_manifest "
                "(use handle_streamed() for streamed requests)",
            )

        # Phase 3.x.11 — sharded-decode dispatch. When the operator
        # wired a sharded_runner, route here regardless of decode_mode
        # (sharded operators don't co-host the regular non-sharded
        # path on the same node). When NO sharded_runner is wired
        # but the request has decode_mode != PREFILL, reject —
        # back-compat regular runners can't honor sharded semantics.
        if self._sharded_runner is not None:
            return self._dispatch_sharded(request)
        # Sprint 657 — KV-cache arc piece 4: INCREMENTAL dispatch on
        # the inline path. The kv_cache_manager must be wired AND the
        # runner must support run_layer_range_incremental (sprint 656).
        # If either is missing, fall through to the sprint-654 error
        # breadcrumb with the actionable hint.
        if request.decode_mode != DecodeMode.PREFILL:
            if self._kv_cache_manager is None:
                return self._error(
                    request.request_id,
                    StageErrorCode.MALFORMED_REQUEST,
                    f"server has no sharded_runner AND no "
                    f"kv_cache_manager wired but request carries "
                    f"decode_mode={request.decode_mode.value!r}; "
                    f"set PRSM_PARALLAX_KV_CACHE_ENABLED=1 to engage "
                    f"the sprint-654-657 KV-cache path on the inline "
                    f"dispatcher",
                )
            if not hasattr(
                self._runner, "run_layer_range_incremental",
            ):
                return self._error(
                    request.request_id,
                    StageErrorCode.MALFORMED_REQUEST,
                    f"server has kv_cache_manager wired but runner "
                    f"{type(self._runner).__name__} doesn't implement "
                    f"run_layer_range_incremental (sprint 656 adds "
                    f"it for HuggingFaceLayerSliceRunner)",
                )
            # Both gates passed → INCREMENTAL inline dispatch.
            return self._dispatch_inline_incremental(request)

        # Steps 2-6: shared validation gates.
        gate_result = self._run_validation_gates(request)
        if gate_result.error is not None:
            return self._error(
                request.request_id, gate_result.error[0], gate_result.error[1]
            )
        model = gate_result.model

        # Step 7a: decode inline activation.
        try:
            activation = decode_activation(
                request.activation_blob,
                request.activation_shape,
                request.activation_dtype,
            )
        except ActivationCodecError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                str(exc),
            )

        # Step 7b: run + encode + sign (inline).
        result_or_error = self._run_layer_slice(request, model, activation)
        if isinstance(result_or_error, tuple):
            return self._error(
                request.request_id, result_or_error[0], result_or_error[1]
            )
        result = result_or_error

        try:
            output_blob, output_shape, output_dtype = encode_activation(
                result.output
            )
        except ActivationCodecError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"output encode failure: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: output encode failed for request_id=%r",
                request.request_id,
            )
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"output encode failure: {exc.__class__.__name__}",
            )

        # Step 8: sign and return.
        response = RunLayerSliceResponse.sign(
            identity=self._identity,
            request_id=request.request_id,
            activation_blob=output_blob,
            activation_shape=output_shape,
            activation_dtype=output_dtype,
            duration_seconds=result.duration_seconds,
            tee_attestation=result.tee_attestation,
            tee_type=result.tee_type,
            epsilon_spent=result.epsilon_spent,
            input_commitment=response_input_commitment_for_request(request),
        )
        return encode_message(response)

    # ── Phase 3.x.11 sharded dispatch ─────────────────────────────────

    def _dispatch_sharded(
        self, request: RunLayerSliceRequest,
    ) -> bytes:
        """Sharded-decode unary dispatch. Routes to
        ``self._sharded_runner.run_layer_slice_unary``.

        Validation gates (model lookup, layer-range, content tier,
        deadline) reuse the regular ``_run_validation_gates`` —
        sharded decode honors all the same operator-side
        constraints. Tier C is currently NOT supported on the
        sharded path (per design plan §2 honest scope); the
        gate here rejects Tier C dispatches with INTERNAL_ERROR
        until Phase 3.x.11.q lands.

        ``is_final_stage`` is inferred from the chain context via
        ``upstream_token.chain_total_stages`` —
        the tail is always the highest stage_index.
        """
        # Reuse the standard validation gates.
        gate_result = self._run_validation_gates(request)
        if gate_result.error is not None:
            return self._error(
                request.request_id,
                gate_result.error[0], gate_result.error[1],
            )

        # Honest-scope Tier C deny on sharded path. Mirrors
        # Phase 3.x.10.y Task 4's tier-c-streaming default-deny;
        # sharded per-token wire dispatch has its own timing
        # surface that the existing constant-time decorators
        # don't cover.
        from prsm.compute.inference.content_tier_gate import ContentTier
        if request.content_tier == ContentTier.C:
            return self._error(
                request.request_id,
                StageErrorCode.TIER_GATE,
                "sharded decode does not yet support Tier C "
                "(per-token wire dispatch creates a new timing "
                "surface; see Phase 3.x.11.q honest scope)",
            )

        # Decode the inline activation_blob.
        try:
            activation = decode_activation(
                request.activation_blob,
                request.activation_shape,
                request.activation_dtype,
            )
        except ActivationCodecError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                str(exc),
            )

        # Determine tail role from the handoff token. The chain
        # tail's stage_index == chain_total_stages - 1.
        token = request.upstream_token
        is_final_stage = (
            token.chain_stage_index
            == token.chain_total_stages - 1
        )

        # Run the sharded forward. Phase 3.x.11.y.x backwards-compat:
        # only pass proposed_token_probs when the wire field is
        # actually set. Pre-3.x.11.y.x runners predate the kwarg
        # and would TypeError on it; v1 (greedy) traffic from a new
        # executor leaves the field None, so the legacy call shape
        # carries forward unchanged. When probs ARE set, we require
        # a v2-capable runner — TypeError surfaces as
        # MALFORMED_REQUEST so the executor learns the tail can't
        # route stochastic traffic.
        runner_kwargs = dict(
            activation_or_input_ids=(
                activation.tolist()
                if activation.dtype.kind in ("i", "u")
                else activation
            ),
            request_id=request.request_id,
            decode_mode=request.decode_mode,
            is_final_stage=is_final_stage,
            request=request,
            # Phase 3.x.11.y — pass proposed_token_ids through
            # to the runner. None on non-VERIFY dispatches; on
            # VERIFY, carries the K draft tokens the executor's
            # speculation loop produced. Tail uses them to
            # compute accepted_count.
            proposed_token_ids=(
                list(request.proposed_token_ids)
                if request.proposed_token_ids is not None
                else None
            ),
        )
        if request.proposed_token_probs is not None:
            # Phase 3.x.11.y.x — v2 stochastic dispatch. Runner
            # routes to apply_lm_head_and_sample_batch_with_rejection
            # (Leviathan-2023 §2.2) when probs co-set with ids AND
            # temperature > 0. Server forwards faithfully; routing
            # decision lives on the runner side (Task 4).
            runner_kwargs["proposed_token_probs"] = list(
                request.proposed_token_probs,
            )
        elif request.encrypted_proposed_token_probs is not None:
            # Phase 3.x.11.q.y — encrypted probs path. Decrypt at
            # the boundary using the operator-wired cipher; pass
            # plaintext to the runner just like the legacy path.
            # The cipher's AAD bind (request_id || stage_index)
            # causes decryption to fail loudly on cross-stage
            # replay; we surface that as MALFORMED_REQUEST.
            if self._encrypted_probs_cipher is None:
                return self._error(
                    request.request_id,
                    StageErrorCode.MALFORMED_REQUEST,
                    "encrypted_proposed_token_probs set on the wire "
                    "but server has no encrypted_probs_cipher wired "
                    "— operator misconfig (the executor encrypted "
                    "but the tail can't decrypt)",
                )
            stage_index = (
                request.upstream_token.chain_stage_index
                if request.upstream_token is not None else 0
            )
            expected_k = (
                len(request.proposed_token_ids)
                if request.proposed_token_ids is not None else 0
            )
            try:
                plaintext_probs = self._encrypted_probs_cipher.decrypt(
                    ciphertext=bytes(
                        request.encrypted_proposed_token_probs,
                    ),
                    request_id=request.request_id,
                    stage_index=stage_index,
                    expected_k=expected_k,
                )
            except Exception as exc:  # noqa: BLE001
                return self._error(
                    request.request_id,
                    StageErrorCode.MALFORMED_REQUEST,
                    f"encrypted_proposed_token_probs decrypt failed: "
                    f"{exc.__class__.__name__}: {exc}",
                )
            runner_kwargs["proposed_token_probs"] = plaintext_probs

        start_ts = self._clock()
        try:
            result = self._sharded_runner.run_layer_slice_unary(
                **runner_kwargs,
            )
        except TypeError as exc:
            # Most-likely cause: stale runner that doesn't accept
            # proposed_token_probs= (pre-3.x.11.y.x). Surface as
            # MALFORMED_REQUEST so the executor knows the tail
            # can't honor v2 stochastic dispatch.
            if (
                "proposed_token_probs" in str(exc)
                or "unexpected keyword argument" in str(exc)
            ):
                logger.warning(
                    "LayerStageServer._dispatch_sharded: runner "
                    "rejected proposed_token_probs= for "
                    "request_id=%r — stale runner predates v2 "
                    "speculation; returning MALFORMED_REQUEST so "
                    "the executor falls back to v1 or fails loud",
                    request.request_id,
                )
                return self._error(
                    request.request_id,
                    StageErrorCode.MALFORMED_REQUEST,
                    f"sharded runner does not support v2 stochastic "
                    f"speculation (Phase 3.x.11.y.x); upgrade the "
                    f"runner or set request.temperature=0.0. "
                    f"Underlying: {exc}",
                )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer._dispatch_sharded: runner raised "
                "for request_id=%r",
                request.request_id,
            )
            # Map MalformedCacheStateError + MissingVerifyCapabilityError
            # to MALFORMED_REQUEST so the executor can distinguish
            # caller bug from internal crash.
            if exc.__class__.__name__ in (
                "MalformedCacheStateError",
                "MissingVerifyCapabilityError",
            ):
                return self._error(
                    request.request_id,
                    StageErrorCode.MALFORMED_REQUEST,
                    str(exc),
                )
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"sharded runner raised: {exc.__class__.__name__}: "
                f"{exc}",
            )

        # Encode the boundary hidden state for the wire.
        try:
            output_blob, output_shape, output_dtype = encode_activation(
                result.hidden_state,
            )
        except ActivationCodecError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"output encode failure: {exc}",
            )

        duration = self._clock() - start_ts
        # Sharded servers commit to the runner's TEE state via
        # the stage's identity. The activation isn't routed
        # through the TEE runtime in v1 (HF model loaded in plain
        # Python process); the response's tee_attestation is the
        # stage's local attestation bytes.
        tee_attestation = self._tee_runtime.get_attestation_bytes()
        # Phase 3.x.11.y — propagate VERIFY tail signals through the
        # signed response. The runner's LayerSliceResult carries
        # ``verified_token_ids`` + ``accepted_count`` only on tail
        # VERIFY dispatches; non-tail / non-VERIFY paths leave them
        # None and the response signing payload omits-when-default
        # (preserves byte-equivalence with pre-3.x.11.y signed bytes).
        response = RunLayerSliceResponse.sign(
            identity=self._identity,
            request_id=request.request_id,
            activation_blob=output_blob,
            activation_shape=output_shape,
            activation_dtype=output_dtype,
            duration_seconds=float(duration),
            tee_attestation=tee_attestation,
            tee_type=self._tee_runtime.tee_type,
            epsilon_spent=0.0,
            next_token_id=result.next_token_id,
            is_terminal=result.is_terminal,
            verified_token_ids=getattr(result, "verified_token_ids", None),
            accepted_count=getattr(result, "accepted_count", None),
            input_commitment=response_input_commitment_for_request(request),
        )
        return encode_message(response)

    # ── streamed dispatch ─────────────────────────────────────────────

    def _dispatch_streamed(
        self,
        request: RunLayerSliceRequest,
        chunk_iter: Iterable[bytes],
    ) -> Tuple[bytes, Iterable[bytes]]:
        """Streamed-path body. Validation BEFORE chunk consumption so
        rejected requests don't pay assembly cost."""
        # Phase 3.x.11.x Task 3: sharded decode supports streamed
        # PREFILL composition (the executor side allows chunked
        # transport for PREFILL when enable_sharded_decode + the
        # activation crosses the inline threshold). INCREMENTAL
        # stays unary-only (single-position activations are tiny
        # by construction; chunked INCREMENTAL is structurally
        # pointless). Reject INCREMENTAL streamed requests when
        # sharded_runner is wired; route PREFILL to the sharded
        # streamed handler.
        if self._sharded_runner is not None:
            if request.decode_mode == DecodeMode.PREFILL:
                return self._dispatch_streamed_sharded(
                    request, chunk_iter,
                )
            return self._streamed_error(
                request.request_id,
                StageErrorCode.MALFORMED_REQUEST,
                "sharded INCREMENTAL is unary-only (design plan "
                "§3.4); chunked transport is supported on PREFILL "
                "only (Phase 3.x.11.x). Single-position INCREMENTAL "
                "activations don't benefit from chunking.",
            )
        # Steps 2-6: shared validation gates (BEFORE consuming chunks).
        gate_result = self._run_validation_gates(request)
        if gate_result.error is not None:
            return self._streamed_error(
                request.request_id, gate_result.error[0], gate_result.error[1]
            )
        model = gate_result.model

        # H1 + M1 round-1 remediation: validate the manifest envelope
        # AGAINST the request's claimed shape/dtype + the operator's
        # max_streamed_payload_bytes ceiling, BEFORE consuming any
        # chunks. A hostile peer that ships an inflated payload_bytes
        # would otherwise cause ShardAssembler to buffer arbitrary
        # bytes during reassembly.
        envelope_err = self._validate_streamed_envelope(request)
        if envelope_err is not None:
            return self._streamed_error(
                request.request_id, envelope_err[0], envelope_err[1]
            )

        # Step 7a-streamed: assemble chunks → decode activation. Both
        # ShardAssembler errors and decode errors map to ACTIVATION_INVALID.
        # Bounded by manifest.total_chunks — _reassemble_inbound_chunks
        # raises if the iterator yields excess frames.
        try:
            shard_chunks = self._reassemble_inbound_chunks(
                chunk_iter,
                expected_request_id=request.request_id,
                manifest_shard_id=request.activation_manifest.shard_id,
                expected_total_chunks=request.activation_manifest.total_chunks,
            )
        except ActivationCodecError as exc:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed input chunk parse failed: {exc}",
            )

        from prsm.compute.chain_rpc.activation_codec import (
            ChunkedActivation as _CA,
        )
        chunked_in = _CA(
            manifest=request.activation_manifest,
            chunks=shard_chunks,
            shape=request.activation_shape,
            dtype_str=request.activation_dtype,
        )
        try:
            activation = reassemble_chunked(chunked_in, chunks=shard_chunks)
        except ActivationCodecError as exc:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed input reassembly failed: {exc}",
            )

        # Step 7b: run.
        result_or_error = self._run_layer_slice(request, model, activation)
        if isinstance(result_or_error, tuple):
            return self._streamed_error(
                request.request_id, result_or_error[0], result_or_error[1]
            )
        result = result_or_error

        # Step 7c-streamed: chunk the output for streaming back. v1
        # contract: streamed request → streamed response (always).
        try:
            chunked_out = chunk_activation(
                result.output,
                activation_id=f"{request.request_id}::resp",
                chunk_bytes=self._chunk_bytes,
            )
        except ActivationCodecError as exc:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed output encode failed: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: streamed output encode failed for "
                "request_id=%r",
                request.request_id,
            )
            return self._streamed_error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"streamed output encode failure: {exc.__class__.__name__}",
            )

        # Step 8: sign + emit. The signing payload commits to
        # manifest.payload_sha256 (v2 conditional encoding); response
        # activation_blob is empty.
        response = RunLayerSliceResponse.sign(
            identity=self._identity,
            request_id=request.request_id,
            activation_blob=b"",
            activation_shape=chunked_out.shape,
            activation_dtype=chunked_out.dtype_str,
            duration_seconds=result.duration_seconds,
            tee_attestation=result.tee_attestation,
            tee_type=result.tee_type,
            epsilon_spent=result.epsilon_spent,
            activation_manifest=chunked_out.manifest,
            input_commitment=response_input_commitment_for_request(request),
        )
        response_manifest_bytes = encode_message(response)

        # Encode each chunk as an ActivationChunk wire frame. Bind to
        # request.request_id (relay-defense — chunks can't be spliced
        # across streams).
        response_chunk_frames: List[bytes] = []
        for shard_chunk in chunked_out.chunks:
            frame = ActivationChunk(
                request_id=request.request_id,
                sequence=shard_chunk.sequence,
                data=shard_chunk.data,
                chunk_sha256=shard_chunk.chunk_sha256,
            )
            response_chunk_frames.append(encode_message(frame))

        return response_manifest_bytes, iter(response_chunk_frames)

    # ── Phase 3.x.11.x sharded streamed dispatch ─────────────────────

    def _dispatch_streamed_sharded(
        self,
        request: RunLayerSliceRequest,
        chunk_iter: Iterable[bytes],
    ) -> Tuple[bytes, Iterable[bytes]]:
        """Phase 3.x.11.x — chunked + sharded PREFILL composition.

        Mirrors ``_dispatch_streamed`` but routes to the sharded
        runner. Caller (``_dispatch_streamed``) has already
        verified ``self._sharded_runner is not None`` and
        ``request.decode_mode == PREFILL`` — INCREMENTAL is
        rejected upstream (single-position activations are tiny).

        Reuses existing machinery:
          - ``_run_validation_gates`` for token / deadline /
            registry / shard / tier
          - ``_validate_streamed_envelope`` for the
            payload_bytes ceiling defence (Phase 3.x.7.1 H1+M1)
          - ``_reassemble_inbound_chunks`` for bounded chunk
            iteration
          - ``reassemble_chunked`` for tensor reassembly
          - ``chunk_activation`` for the response side
          - ``RunLayerSliceResponse.sign`` with the Phase 3.x.11
            ``next_token_id`` + ``is_terminal`` extension

        Tier C structurally denied (mirrors ``_dispatch_sharded``;
        sharded decode introduces a new timing surface that the
        Phase 3.x.10.y constant-time padding decorators don't
        cover; Phase 3.x.11.q deferred).
        """
        # Phase 3.x.11.x Task 6 round-1 LOW-2 remediation:
        # defensive guard against future-refactor seam-bugs.
        # Caller (``_dispatch_streamed`` line 905-913) routes
        # only PREFILL here; INCREMENTAL is rejected upstream.
        # If a future caller wires INCREMENTAL into this path,
        # the runner would attempt PREFILL semantics on a
        # single-position INCREMENTAL input and produce wrong
        # output. Enforce the contract at method entry.
        if request.decode_mode != DecodeMode.PREFILL:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"_dispatch_streamed_sharded invariant: caller "
                f"must verify decode_mode=PREFILL upstream; got "
                f"{request.decode_mode.value!r}",
            )

        # Validation gates.
        gate_result = self._run_validation_gates(request)
        if gate_result.error is not None:
            return self._streamed_error(
                request.request_id,
                gate_result.error[0], gate_result.error[1],
            )

        # Tier C structural deny (mirrors _dispatch_sharded line
        # 783-789).
        from prsm.compute.inference.content_tier_gate import ContentTier
        if request.content_tier == ContentTier.C:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.TIER_GATE,
                "sharded decode does not yet support Tier C "
                "(per-token wire dispatch creates a new timing "
                "surface; see Phase 3.x.11.q honest scope)",
            )

        # Envelope validation BEFORE chunk consumption (Phase
        # 3.x.7.1 H1+M1 round-1 remediation — defends against
        # hostile peers shipping inflated payload_bytes).
        envelope_err = self._validate_streamed_envelope(request)
        if envelope_err is not None:
            return self._streamed_error(
                request.request_id, envelope_err[0], envelope_err[1],
            )

        # Reassemble chunks → activation tensor.
        try:
            shard_chunks = self._reassemble_inbound_chunks(
                chunk_iter,
                expected_request_id=request.request_id,
                manifest_shard_id=request.activation_manifest.shard_id,
                expected_total_chunks=(
                    request.activation_manifest.total_chunks
                ),
            )
        except ActivationCodecError as exc:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"sharded streamed input chunk parse failed: {exc}",
            )

        from prsm.compute.chain_rpc.activation_codec import (
            ChunkedActivation as _CA,
        )
        chunked_in = _CA(
            manifest=request.activation_manifest,
            chunks=shard_chunks,
            shape=request.activation_shape,
            dtype_str=request.activation_dtype,
        )
        try:
            activation = reassemble_chunked(
                chunked_in, chunks=shard_chunks,
            )
        except ActivationCodecError as exc:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"sharded streamed input reassembly failed: {exc}",
            )

        # Tail role inferred from the handoff token (settler-signed
        # so non-forgeable). Mirrors _dispatch_sharded line 808-811.
        token = request.upstream_token
        is_final_stage = (
            token.chain_stage_index == token.chain_total_stages - 1
        )

        # Run the sharded forward.
        start_ts = self._clock()
        try:
            result = self._sharded_runner.run_layer_slice_unary(
                activation_or_input_ids=(
                    activation.tolist()
                    if activation.dtype.kind in ("i", "u")
                    else activation
                ),
                request_id=request.request_id,
                decode_mode=request.decode_mode,
                is_final_stage=is_final_stage,
                request=request,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer._dispatch_streamed_sharded: runner "
                "raised for request_id=%r",
                request.request_id,
            )
            if exc.__class__.__name__ == "MalformedCacheStateError":
                return self._streamed_error(
                    request.request_id,
                    StageErrorCode.MALFORMED_REQUEST,
                    str(exc),
                )
            return self._streamed_error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"sharded streamed runner raised: "
                f"{exc.__class__.__name__}: {exc}",
            )

        # Chunk the boundary hidden state for streaming back.
        # Streamed-request → streamed-response invariant carries
        # forward (matches _dispatch_streamed line 980-981).
        try:
            chunked_out = chunk_activation(
                result.hidden_state,
                activation_id=f"{request.request_id}::resp",
                chunk_bytes=self._chunk_bytes,
            )
        except ActivationCodecError as exc:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"sharded streamed output encode failed: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer._dispatch_streamed_sharded: output "
                "encode failed for request_id=%r",
                request.request_id,
            )
            return self._streamed_error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"sharded streamed output encode failure: "
                f"{exc.__class__.__name__}",
            )

        # Sign + emit. Signing payload commits to manifest.payload_sha256
        # (v2 conditional encoding) AND to next_token_id + is_terminal
        # (Phase 3.x.11 Task 5 critical fix).
        duration = self._clock() - start_ts
        tee_attestation = self._tee_runtime.get_attestation_bytes()
        response = RunLayerSliceResponse.sign(
            identity=self._identity,
            request_id=request.request_id,
            activation_blob=b"",
            activation_shape=chunked_out.shape,
            activation_dtype=chunked_out.dtype_str,
            duration_seconds=float(duration),
            tee_attestation=tee_attestation,
            tee_type=self._tee_runtime.tee_type,
            epsilon_spent=0.0,
            activation_manifest=chunked_out.manifest,
            next_token_id=result.next_token_id,
            is_terminal=result.is_terminal,
            input_commitment=response_input_commitment_for_request(request),
        )
        response_manifest_bytes = encode_message(response)

        # Encode each output chunk as an ActivationChunk wire frame.
        response_chunk_frames: List[bytes] = []
        for shard_chunk in chunked_out.chunks:
            frame = ActivationChunk(
                request_id=request.request_id,
                sequence=shard_chunk.sequence,
                data=shard_chunk.data,
                chunk_sha256=shard_chunk.chunk_sha256,
            )
            response_chunk_frames.append(encode_message(frame))

        return response_manifest_bytes, iter(response_chunk_frames)

    # ── token-stream dispatch (Phase 3.x.8) ──────────────────────────

    def _dispatch_token_stream(
        self,
        request: RunLayerSliceRequest,
    ) -> Iterable[bytes]:
        """Token-stream pipeline body. Validation BEFORE invoking the
        runner; runner emits ``StreamingChunk``s that get encoded
        into ``TokenFrame`` wire frames; terminal chunk's aggregate
        fields populate the signed ``RunLayerSliceResponse`` carried
        by the closing ``StreamFinalFrame``."""
        # Steps 2-6: shared validation gates (token, deadline,
        # registry, shard, tier).
        gate_result = self._run_validation_gates(request)
        if gate_result.error is not None:
            yield self._error(
                request.request_id, gate_result.error[0], gate_result.error[1]
            )
            return
        model = gate_result.model

        # Tail-stage invariant: streaming requests are only valid on
        # the chain's tail. The executor's job is to dispatch
        # streaming=True only to the tail; this gate catches an
        # executor bug or an adversary trying to coerce a non-tail
        # stage into emitting tokens.
        if not _is_final_stage(model, request.layer_range):
            yield self._error(
                request.request_id,
                StageErrorCode.MALFORMED_REQUEST,
                "token-stream handler: streaming=True is only valid "
                "on the chain's tail stage; this stage's layer_range "
                f"{request.layer_range} is not the model tail",
            )
            return

        # Step 7a: decode the inline activation.
        try:
            activation = decode_activation(
                request.activation_blob,
                request.activation_shape,
                request.activation_dtype,
            )
        except ActivationCodecError as exc:
            yield self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                str(exc),
            )
            return

        # Step 7b: invoke the streaming runner. Each ``StreamingChunk``
        # becomes a TokenFrame; the terminal chunk also drives the
        # signed StreamFinalFrame.
        # Phase 3.x.10.x: forward sampling overrides via the
        # ``StreamingSamplingShim``. The wire request's
        # ``max_tokens`` / ``temperature`` (None when the caller
        # didn't override) reach the runner so user-specified
        # ``InferenceRequest.max_tokens`` / ``.temperature`` no
        # longer dead-letter. Empty shim (both None) means runner
        # falls back to ``SamplingDefaults`` — same behavior as
        # pre-3.x.10.x dispatches.
        # Lazy import to keep the chain_rpc → inference dependency
        # one-directional (matches the existing StreamingChunk
        # lazy-import pattern at server.py:867).
        from prsm.compute.inference.streaming_runner import (
            StreamingSamplingShim as _StreamingSamplingShim,
        )
        sampling_shim = _StreamingSamplingShim(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Phase 3.x.10.y Task 4 — Tier C dispatch-layer gate.
        # Tier C content (encrypted/private inference) requires
        # constant-time padding to mask the per-token
        # inter-token-latency side-channel characterized in the
        # timing-sidechannel memo. Default-deny: a Tier C
        # streaming request without ``tier_c_streaming_decorator``
        # configured rejects with INTERNAL_ERROR rather than
        # silently leaking timing. Operators opt in by passing a
        # decorator (typically ``BatchedTrailingStreamingRunner``
        # or ``FixedRateStreamingRunner``) at server construction.
        from prsm.compute.inference.models import ContentTier
        active_runner = self._streaming_runner
        if request.content_tier == ContentTier.C:
            if self._tier_c_streaming_decorator is None:
                yield self._error(
                    request.request_id,
                    StageErrorCode.INTERNAL_ERROR,
                    "Tier C streaming requires constant-time "
                    "padding decorator (tier_c_streaming_decorator "
                    "not configured at server construction)",
                )
                return
            try:
                active_runner = self._tier_c_streaming_decorator(
                    self._streaming_runner,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "LayerStageServer: tier_c_streaming_decorator "
                    "raised during Tier C wrap for request_id=%r",
                    request.request_id,
                )
                yield self._error(
                    request.request_id,
                    StageErrorCode.INTERNAL_ERROR,
                    f"tier_c_streaming_decorator failure: "
                    f"{exc.__class__.__name__}",
                )
                return
            if not hasattr(active_runner, "run_layer_slice_streaming"):
                yield self._error(
                    request.request_id,
                    StageErrorCode.INTERNAL_ERROR,
                    "tier_c_streaming_decorator returned an object "
                    "without .run_layer_slice_streaming",
                )
                return

        chunk_iter = None
        try:
            chunk_iter = active_runner.run_layer_slice_streaming(
                model=model,
                layer_range=request.layer_range,
                activation=activation,
                privacy_tier=request.privacy_tier,
                is_final_stage=True,
                request=sampling_shim,
            )
        except TimeoutError as exc:
            yield self._error(
                request.request_id, StageErrorCode.TIMEOUT, str(exc)
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: streaming_runner setup raised for "
                "request_id=%r",
                request.request_id,
            )
            yield self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"streaming-runner failure: {exc.__class__.__name__}",
            )
            return

        # Iterate the runner. We don't know the terminal chunk until
        # ``finish_reason`` is non-None on a yielded ``StreamingChunk``.
        # Track joined-text invariant: the StreamFinalFrame's signed
        # response.activation_blob MUST equal "".join(text_deltas)
        # encoded UTF-8 — that's how the stage signature commits to
        # the streamed output.
        # Lazy import to avoid circular dependency: streaming_runner.py
        # imports LayerSliceRunner from this module.
        from prsm.compute.inference.streaming_runner import (
            StreamingChunk as _StreamingChunk,
        )
        joined_parts: List[str] = []
        terminal_seen = False
        expected_seq = 0

        # Phase 3.x.8 Task 6 — cancellation cleanup. The outer
        # try/finally ensures the runner's generator is .close()'d
        # on EVERY exit path (caller GeneratorExit, normal return,
        # or unhandled exception). Without this, a caller closing
        # the executor's generator could leave the runner generator
        # alive until the next gc cycle — undesirable for runners
        # holding accelerator state (KV cache pinned memory etc.).
        #
        # v1 honest scope: cancellation = clean cleanup, NOT
        # delivery of a partial-output StreamFinalFrame. Python's
        # GeneratorExit semantics forbid yielding additional values
        # after .close(); a partial-receipt-on-cancel pathway
        # requires either an in-band cancel-sentinel via .send()
        # or a side-channel inspection API — both deferred to a
        # Phase 3.x.8.x follow-up.
        try:
            for chunk in chunk_iter:
                if not isinstance(chunk, _StreamingChunk):
                    yield self._error(
                        request.request_id,
                        StageErrorCode.INTERNAL_ERROR,
                        f"streaming_runner yielded {type(chunk).__name__}; "
                        f"expected StreamingChunk",
                    )
                    return
                # Sequence-index invariant: 0-indexed, strictly
                # increasing, no gaps. A runner that violates this
                # is a bug — surface as INTERNAL_ERROR.
                if chunk.sequence_index != expected_seq:
                    yield self._error(
                        request.request_id,
                        StageErrorCode.INTERNAL_ERROR,
                        f"streaming_runner emitted sequence_index="
                        f"{chunk.sequence_index} (expected "
                        f"{expected_seq})",
                    )
                    return
                expected_seq += 1
                joined_parts.append(chunk.text_delta)

                # M1 round-1 remediation: terminal-chunk integrity
                # checks run BEFORE the terminal TokenFrame is
                # emitted on the wire. The "sole error frame on
                # failure" invariant in the handler's docstring
                # holds: a runner producing an inconsistent
                # terminal chunk yields a StageError WITHOUT any
                # preceding terminal TokenFrame. Non-terminal
                # chunks (finish_reason=None) emit normally — the
                # joined-text invariant only resolves on the
                # terminal chunk anyway.
                if chunk.finish_reason is not None:
                    # The terminal chunk MUST carry the final-
                    # aggregate fields the StreamFinalFrame needs.
                    if (
                        chunk.full_output_text is None
                        or chunk.duration_seconds is None
                        or chunk.tee_attestation is None
                        or chunk.tee_type is None
                        or chunk.epsilon_spent is None
                    ):
                        yield self._error(
                            request.request_id,
                            StageErrorCode.INTERNAL_ERROR,
                            "streaming_runner terminal chunk missing "
                            "final-aggregate fields (full_output_text / "
                            "duration_seconds / tee_attestation / "
                            "tee_type / epsilon_spent)",
                        )
                        return

                    # Joined-text invariant check: the stage signs
                    # over full_output_text bytes; the consumer
                    # joins TokenFrame deltas and asserts they
                    # match. We enforce here BEFORE signing so a
                    # runner that produces inconsistent text fails
                    # without emitting either the terminal
                    # TokenFrame OR the StreamFinalFrame.
                    joined = "".join(joined_parts)
                    if joined != chunk.full_output_text:
                        yield self._error(
                            request.request_id,
                            StageErrorCode.INTERNAL_ERROR,
                            "streaming_runner: joined text_deltas do "
                            "not match terminal chunk's "
                            "full_output_text (runner produced "
                            "inconsistent stream)",
                        )
                        return

                # Encode + yield the TokenFrame wire bytes. For the
                # terminal chunk this only runs after the integrity
                # checks above have passed.
                frame = TokenFrame(
                    request_id=request.request_id,
                    sequence_index=chunk.sequence_index,
                    text_delta=chunk.text_delta,
                    token_id=chunk.token_id,
                    finish_reason=chunk.finish_reason,
                )
                yield encode_message(frame)

                if chunk.finish_reason is not None:
                    terminal_seen = True
                    # Build + yield the signed StreamFinalFrame.
                    yield self._build_stream_final_frame(
                        request_id=request.request_id,
                        chunk=chunk,
                    )
                    return
        except TimeoutError as exc:
            yield self._error(
                request.request_id, StageErrorCode.TIMEOUT, str(exc)
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: streaming_runner iteration raised for "
                "request_id=%r",
                request.request_id,
            )
            yield self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"streaming-runner iteration failure: "
                f"{exc.__class__.__name__}",
            )
            return
        finally:
            # Explicit close on every exit path. ``close()`` on a
            # generator triggers GeneratorExit at its current yield
            # point; close() on a non-generator iterator that
            # implements the .close() protocol is also forwarded.
            # Iterators without .close() (plain lists, etc.) are a
            # no-op — getattr keeps us safe.
            close = getattr(chunk_iter, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    # Cleanup-time exceptions are swallowed —
                    # never propagate through GeneratorExit. The
                    # runner has already produced its output (or
                    # the caller cancelled); a close-time exception
                    # is non-actionable at the wire boundary.
                    logger.debug(
                        "LayerStageServer: streaming_runner.close() "
                        "raised during cleanup (request_id=%r); "
                        "swallowed",
                        request.request_id,
                    )

        # Iterator exhausted without ever emitting a terminal chunk
        # (a runner that returns mid-stream silently). Surface as
        # INTERNAL_ERROR so the consumer doesn't hang waiting for a
        # StreamFinalFrame that's never coming.
        if not terminal_seen:
            yield self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                "streaming_runner exhausted without a terminal chunk "
                "(no finish_reason set on any yielded chunk)",
            )

    def _build_stream_final_frame(
        self,
        *,
        request_id: str,
        chunk: Any,  # StreamingChunk — lazy-typed to avoid circular import
    ) -> bytes:
        """Encode the signed terminal frame for a token stream.

        The embedded ``RunLayerSliceResponse``'s ``activation_blob``
        is the UTF-8 encoding of the joined output text. The stage's
        existing ``signing_payload`` hex-encodes ``activation_blob``
        — so the signature commits to the streamed output without
        any new signing-payload field. A relay that tampers any
        ``TokenFrame.text_delta`` causes the joined-bytes hash to
        diverge from what the stage signed, invalidating the stream
        as a whole.
        """
        joined_bytes = chunk.full_output_text.encode("utf-8")  # type: ignore[union-attr]
        response = RunLayerSliceResponse.sign(
            identity=self._identity,
            request_id=request_id,
            activation_blob=joined_bytes,
            activation_shape=(len(joined_bytes),)
            if joined_bytes
            else (0,),
            activation_dtype="uint8",
            duration_seconds=chunk.duration_seconds,  # type: ignore[arg-type]
            tee_attestation=chunk.tee_attestation,  # type: ignore[arg-type]
            tee_type=chunk.tee_type,  # type: ignore[arg-type]
            epsilon_spent=chunk.epsilon_spent,  # type: ignore[arg-type]
        )
        return encode_message(StreamFinalFrame(response=response))

    # ── shared validation + run ───────────────────────────────────────

    def _run_validation_gates(
        self, request: RunLayerSliceRequest
    ) -> "_GateResult":
        """Steps 2-6: token verify → deadline → registry → shard
        coverage → tier gate. Shared by inline + streamed paths.
        Returns a ``_GateResult`` with either ``model`` populated (all
        gates pass) or ``error`` populated (first failure)."""
        # Step 2: anchor-verify the upstream token.
        if not request.upstream_token.verify_with_anchor(self._anchor):
            return _GateResult(error=(
                StageErrorCode.INVALID_TOKEN,
                f"upstream_token failed anchor verification (settler="
                f"{request.upstream_token.settler_node_id!r})",
            ))

        # Step 3: deadline check.
        now = self._clock()
        if request.upstream_token.deadline_unix <= now:
            return _GateResult(error=(
                StageErrorCode.DEADLINE_EXCEEDED,
                f"token deadline {request.upstream_token.deadline_unix} "
                f"already past clock {now}",
            ))
        if request.deadline_unix <= now:
            return _GateResult(error=(
                StageErrorCode.DEADLINE_EXCEEDED,
                f"request deadline {request.deadline_unix} already past "
                f"clock {now}",
            ))

        # Step 4: registry lookup.
        try:
            model = self._registry.get(request.model_id)
        except Exception as exc:  # noqa: BLE001
            class_name = exc.__class__.__name__
            if "NotFound" in class_name:
                return _GateResult(error=(
                    StageErrorCode.MODEL_NOT_FOUND,
                    f"model {request.model_id!r} not in local registry",
                ))
            if "Verification" in class_name:
                return _GateResult(error=(
                    StageErrorCode.MODEL_NOT_FOUND,
                    f"model {request.model_id!r} failed registry "
                    f"verification: {exc}",
                ))
            logger.exception(
                "LayerStageServer: registry.get raised unexpectedly"
            )
            return _GateResult(error=(
                StageErrorCode.INTERNAL_ERROR,
                f"registry error: {class_name}",
            ))

        # Step 5: layer-range coverage.
        if not _shards_cover_range(model, request.layer_range):
            return _GateResult(error=(
                StageErrorCode.SHARD_MISSING,
                f"local shards do not cover layer_range "
                f"{request.layer_range} for model {request.model_id!r}",
            ))

        # Step 6: privacy-tier gate against the local TEE runtime.
        if request.privacy_tier != PrivacyLevel.NONE:
            tee_type = self._tee_runtime.tee_type
            if not self._is_hardware_tee(tee_type):
                return _GateResult(error=(
                    StageErrorCode.TIER_GATE,
                    f"privacy_tier={request.privacy_tier.value} requires "
                    f"hardware TEE; local runtime is {tee_type.value}",
                ))

        return _GateResult(model=model)

    def _dispatch_inline_incremental(
        self, request: RunLayerSliceRequest,
    ) -> bytes:
        """Sprint 657 — INCREMENTAL inline dispatch with KVCacheManager.

        Steps:
          1. Validate the request (shared validation gates)
          2. Decode the activation
          3. Get prev_kv_state from kv_cache_manager (None on first
             INCREMENTAL of a new request_id, or after TTL evict)
          4. Call _run_layer_slice_incremental
          5. Put new_kv_state to cache
          6. Encode + sign + return

        Cache key is the request_id — sprint 658 will add TTL +
        eviction. For now the cache lives for the daemon's lifetime
        (KVCacheManager's bounded LRU keeps it sane).
        """
        gate_result = self._run_validation_gates(request)
        if gate_result.error is not None:
            return self._error(
                request.request_id,
                gate_result.error[0], gate_result.error[1],
            )
        model = gate_result.model

        try:
            activation = decode_activation(
                request.activation_blob,
                request.activation_shape,
                request.activation_dtype,
            )
        except ActivationCodecError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                str(exc),
            )

        # Sprint 658 — KVCacheManager API: allocate(request_id, n_layers)
        # creates a fresh handle; get(request_id) returns the existing
        # handle or None. handle.payload is the DynamicCache (mutable).
        # The manager doesn't have a `put` method — sprint 657 used
        # that misnamed call, corrected here.
        handle = None
        try:
            handle = self._kv_cache_manager.get(request.request_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LayerStageServer: kv_cache_manager.get raised for "
                "request_id=%r — proceeding with cold cache: %s",
                request.request_id, exc,
            )
        if handle is None:
            # Cold path: allocate a new handle. n_layers = the layer
            # range's width; the manager only uses it for the
            # cached_positions accounting.
            n_layers = int(request.layer_range[1] - request.layer_range[0])
            try:
                handle = self._kv_cache_manager.allocate(
                    request.request_id,
                    max(1, n_layers),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LayerStageServer: kv_cache_manager.allocate "
                    "raised for request_id=%r — INCREMENTAL forward "
                    "will run uncached: %s",
                    request.request_id, exc,
                )

        prev_kv_state = handle.payload if handle is not None else None

        # Run incremental forward.
        result_or_error = self._run_layer_slice_incremental(
            request, model, activation, prev_kv_state,
        )
        if (
            isinstance(result_or_error, tuple)
            and len(result_or_error) == 2
            and isinstance(result_or_error[0], StageErrorCode)
        ):
            return self._error(
                request.request_id,
                result_or_error[0], result_or_error[1],
            )
        # Success → unpack (result, new_kv_state).
        # DynamicCache is mutated in-place by the runner, but we
        # store the reference on the handle explicitly so an
        # observability snapshot of handle.payload reflects the
        # latest state regardless of internal aliasing.
        result, new_kv_state = result_or_error
        if handle is not None:
            handle.payload = new_kv_state
            # cached_positions accounting — at least one position
            # was forwarded.
            try:
                handle.cached_positions = int(
                    new_kv_state.get_seq_length()
                )
            except Exception:  # noqa: BLE001
                pass

        # Encode + sign + return (mirrors the PREFILL path).
        try:
            output_blob, output_shape, output_dtype = encode_activation(
                result.output
            )
        except ActivationCodecError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"output encode failure: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: output encode failed for "
                "request_id=%r (INCREMENTAL)",
                request.request_id,
            )
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"output encode failure: {exc.__class__.__name__}",
            )
        response = RunLayerSliceResponse.sign(
            identity=self._identity,
            request_id=request.request_id,
            activation_blob=output_blob,
            activation_shape=output_shape,
            activation_dtype=output_dtype,
            duration_seconds=result.duration_seconds,
            tee_attestation=result.tee_attestation,
            tee_type=result.tee_type,
            epsilon_spent=result.epsilon_spent,
            input_commitment=response_input_commitment_for_request(request),
        )
        return encode_message(response)

    def _run_layer_slice(
        self,
        request: RunLayerSliceRequest,
        model: Any,
        activation: np.ndarray,
    ):
        """Wraps runner.run_layer_range with TIMEOUT / INTERNAL_ERROR
        mapping. Returns a ``LayerSliceResult`` on success or a
        ``(StageErrorCode, message)`` tuple on failure."""
        try:
            return self._runner.run_layer_range(
                model=model,
                layer_range=request.layer_range,
                activation=activation,
                privacy_tier=request.privacy_tier,
                is_final_stage=_is_final_stage(model, request.layer_range),
            )
        except TimeoutError as exc:
            return (StageErrorCode.TIMEOUT, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: runner.run_layer_range raised for "
                "request_id=%r",
                request.request_id,
            )
            return (
                StageErrorCode.INTERNAL_ERROR,
                f"layer-runner failure: {exc.__class__.__name__}",
            )

    def _run_layer_slice_incremental(
        self,
        request: RunLayerSliceRequest,
        model: Any,
        activation: np.ndarray,
        prev_kv_state: Any,
    ):
        """Sprint 655 — INCREMENTAL inline-path delegator.

        Wraps runner.run_layer_range_incremental(...) with the same
        TIMEOUT / INTERNAL_ERROR mapping as the PREFILL delegator.
        Returns either a (LayerSliceResult, new_kv_state) tuple on
        success OR a (StageErrorCode, message) tuple on failure.

        Sprint 656 will add `run_layer_range_incremental` to
        HuggingFaceLayerSliceRunner. Sprint 657 will wire
        KVCacheManager.get/put around the call site. Until those
        sprints land, this method correctly surfaces NOT_IMPLEMENTED
        when called against a runner that doesn't yet implement
        INCREMENTAL — the path is structurally in place so sprints
        656-660 are pure additive plumbing rather than scaffolding.

        ``prev_kv_state``: opaque value from the previous INCREMENTAL
        or PREFILL call's `new_kv_state` output, OR None for a fresh
        cache (server-side TTL eviction OR client-side first INCREMENTAL).
        """
        if not hasattr(self._runner, "run_layer_range_incremental"):
            return (
                StageErrorCode.MALFORMED_REQUEST,
                f"runner {type(self._runner).__name__} does not "
                f"implement run_layer_range_incremental; INCREMENTAL "
                f"decode requires runner-side support (sprint 656 "
                f"adds it for HuggingFaceLayerSliceRunner)",
            )
        try:
            return self._runner.run_layer_range_incremental(
                model=model,
                layer_range=request.layer_range,
                activation=activation,
                privacy_tier=request.privacy_tier,
                is_final_stage=_is_final_stage(
                    model, request.layer_range,
                ),
                prev_kv_state=prev_kv_state,
            )
        except TimeoutError as exc:
            return (StageErrorCode.TIMEOUT, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: runner.run_layer_range_incremental "
                "raised for request_id=%r",
                request.request_id,
            )
            return (
                StageErrorCode.INTERNAL_ERROR,
                f"layer-runner failure: {exc.__class__.__name__}",
            )

    def _validate_streamed_envelope(
        self, request: RunLayerSliceRequest
    ) -> Optional[Tuple[StageErrorCode, str]]:
        """H1 + M1 round-1 remediation: pre-consumption envelope sanity
        check. Returns ``None`` if the manifest's claimed envelope is
        consistent with the request's shape/dtype AND below the
        operator's ``max_streamed_payload_bytes`` ceiling. Otherwise
        returns a (code, message) error tuple.

        This fires BEFORE any chunk consumption — defense against
        hostile peers that inflate ``payload_bytes`` / ``total_chunks``
        to exhaust server memory during reassembly. The H2 fix on the
        signing side ensures that authentic stages can't lie about
        these values without invalidating the upstream signature; this
        gate handles the request-side path where chunks haven't yet
        been signed by anyone.
        """
        m = request.activation_manifest
        # Shape × dtype must equal payload_bytes. Anything else is a
        # malformed manifest (or a relay attempting DoS).
        try:
            dtype = np.dtype(request.activation_dtype)
        except TypeError:
            return (
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed: unrecognized activation_dtype "
                f"{request.activation_dtype!r}",
            )
        expected_payload_bytes = (
            int(np.prod(request.activation_shape)) * dtype.itemsize
        )
        if m.payload_bytes != expected_payload_bytes:
            return (
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed: manifest.payload_bytes ({m.payload_bytes}) "
                f"does not match shape {request.activation_shape} × "
                f"dtype {request.activation_dtype} "
                f"({expected_payload_bytes})",
            )
        # Hard cap against memory DoS.
        if m.payload_bytes > self._max_streamed_payload_bytes:
            return (
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed: manifest.payload_bytes ({m.payload_bytes}) "
                f"exceeds max_streamed_payload_bytes "
                f"({self._max_streamed_payload_bytes})",
            )
        # total_chunks consistency with chunk_bytes + payload_bytes.
        # ShardAssembler will catch over-count via finalize(), but
        # rejecting now skips the assembly cost entirely.
        if m.total_chunks > 0:
            expected_total_chunks = (
                m.payload_bytes + m.chunk_bytes - 1
            ) // m.chunk_bytes
            if m.total_chunks != expected_total_chunks:
                return (
                    StageErrorCode.ACTIVATION_INVALID,
                    f"streamed: manifest.total_chunks ({m.total_chunks}) "
                    f"inconsistent with payload_bytes ({m.payload_bytes}) "
                    f"/ chunk_bytes ({m.chunk_bytes}); expected "
                    f"{expected_total_chunks}",
                )
        elif m.payload_bytes > 0:
            return (
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed: manifest.total_chunks == 0 but "
                f"payload_bytes == {m.payload_bytes}",
            )
        return None

    @staticmethod
    def _reassemble_inbound_chunks(
        frame_iter: Iterable[bytes],
        *,
        expected_request_id: str,
        manifest_shard_id: str,
        expected_total_chunks: int,
    ) -> List[ShardChunk]:
        """Decode incoming ActivationChunk frames back to Phase 6
        ``ShardChunk`` objects. Validates the relay-defense binding
        (chunk.request_id must match the parent request); rewraps the
        shard_id to match the inbound manifest so the assembler's
        per-chunk validation succeeds.

        H1 round-1 remediation: bounded by ``expected_total_chunks``.
        Stops reading after ``expected_total_chunks`` frames have been
        accepted — a peer that keeps shipping bytes past that count
        gets ``ActivationCodecError("excess chunks")`` rather than
        unbounded memory growth.
        """
        out: List[ShardChunk] = []
        for raw in frame_iter:
            if len(out) >= expected_total_chunks:
                raise ActivationCodecError(
                    f"streamed: peer shipped excess chunks "
                    f"(expected_total_chunks={expected_total_chunks})"
                )
            try:
                msg = parse_message(raw)
            except (
                ChainRpcMalformedError,
                ChainRpcUnknownTypeError,
                ChainRpcProtocolError,
                ChainRpcVersionMismatchError,
            ) as exc:
                raise ActivationCodecError(
                    f"streamed chunk frame parse failed: {exc}"
                ) from exc
            if not isinstance(msg, ActivationChunk):
                raise ActivationCodecError(
                    f"streamed chunk frame: expected ActivationChunk, "
                    f"got {type(msg).__name__}"
                )
            if msg.request_id != expected_request_id:
                raise ActivationCodecError(
                    f"streamed chunk request_id mismatch: got "
                    f"{msg.request_id!r}, expected {expected_request_id!r}"
                )
            out.append(ShardChunk(
                shard_id=manifest_shard_id,
                sequence=msg.sequence,
                data=msg.data,
                chunk_sha256=msg.chunk_sha256,
            ))
        return out

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_hardware_tee(tee_type: TEEType) -> bool:
        """Same hardware-TEE policy used by Phase 3.x.6 Task 5
        ``TierGateAdapter``: anything in ``HARDWARE_TEE_TYPES`` counts;
        ``SOFTWARE`` does not.

        Sprint 702 — `PRSM_PARALLAX_TIER_GATE=advisory` extends to the
        server-side runtime check (in addition to the trust-stack-side
        Adapter B in `_wrap_tier_gate_for_advisory`). When advisory,
        software-TEE runtimes accept tier≥standard requests so the
        activation-DP injection code path can be live-attested without
        real TEE hardware. Production posture (default + typo
        fallthrough) is unchanged.
        """
        import os as _os
        if (
            _os.environ.get("PRSM_PARALLAX_TIER_GATE", "")
            .strip().lower() == "advisory"
        ):
            return True
        return tee_type.value in HARDWARE_TEE_TYPES

    def _error(
        self,
        request_id: str,
        code: StageErrorCode,
        message: str,
    ) -> bytes:
        """Encode a ``StageError`` for return; never raises (helpers
        within encode_message are pure JSON serialization)."""
        return encode_message(
            StageError(
                request_id=request_id or _UNKNOWN_REQUEST_ID,
                code=code.value,
                message=message,
            )
        )

    # ── Phase 3.x.11 — KV-cache eviction handler ─────────────────────

    def _handle_evict_cache(self, request: EvictCacheRequest) -> bytes:
        """Route an ``EvictCacheRequest`` to the bound
        ``KVCacheManager``. Returns an ``EvictCacheResponse``
        with ``evicted=True/False``.

        When ``kv_cache_manager`` was not wired at construction,
        rejects with ``INTERNAL_ERROR`` — the server can't honor
        sharded-decode eviction without a cache lifecycle owner.
        """
        if self._kv_cache_manager is None:
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                "evict_cache: server has no kv_cache_manager wired "
                "(operator must pass kv_cache_manager= when sharded "
                "decode is enabled)",
            )
        evicted = self._kv_cache_manager.evict(request.request_id)
        return encode_message(
            EvictCacheResponse(
                request_id=request.request_id,
                evicted=bool(evicted),
            )
        )

    # ── Phase 3.x.11.y — speculative-decoding rollback handler ───────

    def _handle_rollback_cache(
        self, request: RollbackCacheRequest,
    ) -> bytes:
        """Route a ``RollbackCacheRequest`` to the bound
        ``ShardedAutoregressiveRunner.rollback_cache``. Returns a
        ``RollbackCacheResponse`` with ``rolled_back=True/False``
        and ``actual_dropped`` (the count actually removed; may be
        less than ``n_positions_to_drop`` per the manager's
        idempotent over-drop semantics).

        Sharded-decode + speculation requires both
        ``kv_cache_manager`` (for cache lifecycle) AND
        ``sharded_runner`` (for the model's ``truncate_cache``
        delegate). When either is missing, returns INTERNAL_ERROR
        — the server can't honor speculative rollback without
        both wired.

        ``MissingVerifyCapabilityError`` (raised by the runner
        when the model omits ``truncate_cache``) maps to
        MALFORMED_REQUEST so the executor can distinguish caller
        bug from internal crash.
        """
        if self._kv_cache_manager is None:
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                "rollback_cache: server has no kv_cache_manager "
                "wired (operator must pass kv_cache_manager= when "
                "speculative decode is enabled)",
            )
        if self._sharded_runner is None:
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                "rollback_cache: server has no sharded_runner wired "
                "(operator must pass sharded_runner= when "
                "speculative decode is enabled — the runner provides "
                "the model's truncate_cache delegate)",
            )
        if not callable(getattr(self._sharded_runner, "rollback_cache", None)):
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                "rollback_cache: sharded_runner does not expose "
                ".rollback_cache(...) — operator wired a non-"
                "Phase-3.x.11.y-capable runner",
            )
        try:
            rolled_back, actual_dropped = (
                self._sharded_runner.rollback_cache(
                    request.request_id,
                    int(request.n_positions_to_drop),
                )
            )
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "MissingVerifyCapabilityError":
                return self._error(
                    request.request_id,
                    StageErrorCode.MALFORMED_REQUEST,
                    str(exc),
                )
            raise
        # Phase 3.x.11.q.y' — replay accepted prefix to repopulate
        # the cache after a constant-K rollback. Decrypt at the
        # boundary if the prefix arrived encrypted; surface
        # MALFORMED_REQUEST on cipher failure (mirrors the
        # encrypted_proposed_token_probs decrypt path from
        # Phase 3.x.11.q.y). Replay is best-effort at the runner
        # level — non-stage-0 stages cannot replay without
        # upstream hidden state and return False without raising.
        # Round-1 review L1 remediation: replay-prefix decrypt is
        # best-effort. The truncation above already succeeded; if we
        # surface MALFORMED_REQUEST here, the cache is left in a
        # truncated-but-not-replayed state which is worse than just
        # logging the decrypt failure and continuing. The wire-side
        # leak closure is independent of replay correctness; only
        # the LOCAL cache state suffers when replay can't run, and
        # the TTL sweeper bounds that. Mirrors the runner-side
        # replay_accepted_prefix best-effort semantics.
        prefix_tokens: Optional[List[int]] = None
        if request.replay_accepted_prefix is not None:
            prefix_tokens = list(request.replay_accepted_prefix)
        elif request.encrypted_replay_accepted_prefix is not None:
            if self._encrypted_probs_cipher is None:
                logger.warning(
                    "_handle_rollback_cache: "
                    "encrypted_replay_accepted_prefix set on wire "
                    "but server has no encrypted_probs_cipher wired "
                    "— skipping replay (truncation succeeded; cache "
                    "state may be inconsistent on next VERIFY round, "
                    "TTL sweeper bounds the impact). Operator should "
                    "wire encrypted_probs_cipher= for full q.y' "
                    "closure.",
                )
            elif not callable(getattr(
                self._encrypted_probs_cipher, "decrypt_prefix", None,
            )):
                logger.warning(
                    "_handle_rollback_cache: cipher does not "
                    "implement decrypt_prefix(...) — operator wired "
                    "a pre-q.y' cipher. Skipping replay; cache "
                    "state may be inconsistent on next VERIFY "
                    "round (TTL sweeper bounds the impact)."
                )
            else:
                # Phase 3.x.11.q.y' — stage_index for AAD binding
                # comes from the wire field (validator already
                # enforced co-set with
                # encrypted_replay_accepted_prefix in the protocol
                # layer; we know it's set here).
                stage_index = int(request.target_stage_index)
                # We don't know K a priori from the rollback envelope
                # (it's not co-set with proposed_token_ids the way
                # the probs path is). Try sequential K values
                # 1..n_positions_to_drop and accept the one that
                # decrypts cleanly. Bounded by n_positions_to_drop
                # so the overhead is constant per round.
                decrypted: Optional[List[int]] = None
                for try_k in range(
                    1, int(request.n_positions_to_drop) + 1,
                ):
                    try:
                        decrypted = (
                            self._encrypted_probs_cipher.decrypt_prefix(
                                ciphertext=bytes(
                                    request.encrypted_replay_accepted_prefix,
                                ),
                                request_id=request.request_id,
                                stage_index=stage_index,
                                expected_k=try_k,
                            )
                        )
                        break
                    except Exception:  # noqa: BLE001
                        continue
                if decrypted is None:
                    logger.warning(
                        "_handle_rollback_cache: "
                        "encrypted_replay_accepted_prefix decrypt "
                        "failed for all expected_k in "
                        "[1, n_positions_to_drop=%d] — ciphertext "
                        "was tampered, wrong key, or AAD-replayed "
                        "across (request_id, stage_index). "
                        "Skipping replay; cache state may be "
                        "inconsistent on next VERIFY round (TTL "
                        "sweeper bounds the impact).",
                        int(request.n_positions_to_drop),
                    )
                else:
                    prefix_tokens = decrypted
        if prefix_tokens:
            replay_fn = getattr(
                self._sharded_runner, "replay_accepted_prefix", None,
            )
            if callable(replay_fn):
                try:
                    replay_fn(
                        request_id=request.request_id,
                        prefix_token_ids=prefix_tokens,
                    )
                except Exception as exc:  # noqa: BLE001
                    if exc.__class__.__name__ == (
                        "MissingVerifyCapabilityError"
                    ):
                        return self._error(
                            request.request_id,
                            StageErrorCode.MALFORMED_REQUEST,
                            str(exc),
                        )
                    # Best-effort: log + continue. The rollback
                    # itself succeeded (wire leak is closed); the
                    # cache state may be inconsistent on the next
                    # VERIFY round but the executor's TTL-sweeper
                    # bounds the impact.
                    pass
        return encode_message(
            RollbackCacheResponse(
                request_id=request.request_id,
                rolled_back=bool(rolled_back),
                actual_dropped=int(actual_dropped),
            )
        )

    def _streamed_error(
        self,
        request_id: str,
        code: StageErrorCode,
        message: str,
    ) -> Tuple[bytes, Iterable[bytes]]:
        """Streamed-path failure: encode StageError as the manifest
        return slot + empty chunk iterator. Caller never sees a raise."""
        return self._error(request_id, code, message), iter(())
