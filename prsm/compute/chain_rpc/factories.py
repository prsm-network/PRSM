"""Phase 3.x.7 Task 6 — production-wiring factories for chain-RPC.

One-call construction helpers so production callers (the orchestrator,
the operator's node bootstrapping path) don't need full-submodule
imports + deep parameter knowledge to wire an ``RpcChainExecutor``
into a ``ParallaxScheduledExecutor``.

Two factories:

  ``make_rpc_chain_executor(...)`` — RpcChainExecutor with sensible
  defaults. Production callers MUST override the prompt_encoder and
  output_decoder for real tokenization; the UTF-8 byte-passthrough
  defaults are sufficient for tests + integration but won't work
  with real LLMs.

  ``make_layer_stage_server(...)`` — LayerStageServer wired up with
  the canonical Phase 2 Ring 8 + 3.x.2 + 3.x.3 dependencies. Reduces
  the boilerplate at node-startup to a single call.

Default tokenizer adapter (``utf8_prompt_encoder`` /
``utf8_output_decoder``):
  - Encodes the prompt's UTF-8 bytes as an int32 numpy array.
  - Decodes by reversing the operation, stripping zero padding.
  - Round-trip safe for ASCII; UTF-8 boundaries preserved as long
    as the chain doesn't reorder bytes mid-tensor.
  - Production callers replace this with the model's real tokenizer
    + de-tokenizer (typically wraps the HuggingFace ``AutoTokenizer``
    bound to the model's vocab).

The factory makes the SDK-style "construct with one call" surface the
design plan §4 Task 6 acceptance criterion calls for.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from prsm.compute.chain_rpc.activation_codec import (
    CHUNK_THRESHOLD_BYTES,
    DEFAULT_CHUNK_BYTES_ACTIVATION,
)
from prsm.compute.chain_rpc.client import (
    AddressResolver,
    OutputDecoder,
    PromptEncoder,
    RpcChainExecutor,
    SendMessage,
    StreamedSendMessage,
)
from prsm.compute.chain_rpc.server import (
    LayerSliceRunner,
    LayerStageServer,
)
from prsm.node.identity import NodeIdentity


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Default tokenizer adapter (UTF-8 byte-passthrough)
# ──────────────────────────────────────────────────────────────────────────


def utf8_prompt_encoder(prompt: str) -> np.ndarray:
    """Encode a prompt's UTF-8 bytes as a 1-D int32 numpy array.

    Pads to a 4-byte boundary with zeros so the byte buffer is a
    clean int32 view. Round-trips with ``utf8_output_decoder`` for
    ASCII + UTF-8 inputs.

    PRODUCTION USE WARNING: this is a transparent byte-passthrough
    suitable for tests and integration scenarios where the chain's
    layer runners just shuffle activations through. Real LLM
    inference requires tokenizing the prompt into the model's vocab
    space + an embedding lookup; production callers replace this
    with their model's tokenizer adapter.
    """
    raw = prompt.encode("utf-8")
    pad = (4 - len(raw) % 4) % 4
    raw = raw + b"\x00" * pad
    return np.frombuffer(raw, dtype=np.int32).copy()


def utf8_output_decoder(arr: np.ndarray) -> str:
    """Inverse of ``utf8_prompt_encoder``. Strips trailing zero bytes
    and decodes UTF-8 with replacement on invalid sequences (so a
    corrupted tail doesn't crash the executor)."""
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32, copy=False)
    return arr.tobytes().rstrip(b"\x00").decode("utf-8", errors="replace")


# ──────────────────────────────────────────────────────────────────────────
# RpcChainExecutor factory
# ──────────────────────────────────────────────────────────────────────────


def make_rpc_chain_executor(
    *,
    settler_identity: NodeIdentity,
    send_message: SendMessage,
    anchor: Any,
    prompt_encoder: Optional[PromptEncoder] = None,
    output_decoder: Optional[OutputDecoder] = None,
    address_resolver: Optional[AddressResolver] = None,
    streamed_send_message: Optional[StreamedSendMessage] = None,
    token_stream_send_message: Optional[Any] = None,
    chunk_threshold_bytes: int = CHUNK_THRESHOLD_BYTES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES_ACTIVATION,
    default_deadline_seconds: float = 30.0,
    wrap_topology_aware: bool = True,
    wrap_activation_dp_aware: bool = True,
    dp_clip_norm: float = 1.0,
    dp_delta: float = 1e-5,
) -> Any:
    """Build an ``RpcChainExecutor`` with production-friendly defaults.

    Required:
      settler_identity   The settling node's NodeIdentity. Mints
                         per-stage HandoffTokens. Same identity that
                         signs the final InferenceReceipt at the
                         ParallaxScheduledExecutor layer.
      send_message       Phase 6 transport callable: (address, bytes)
                         → bytes. Used for inline-sized activations
                         (≤ ``chunk_threshold_bytes``).
      anchor             Phase 3.x.3 anchor for verifying each stage's
                         response signature.

    Optional (defaults are test-friendly, not production-ready):
      prompt_encoder     Default: ``utf8_prompt_encoder``. Replace
                         with the model's tokenizer adapter for real
                         LLM inference.
      output_decoder     Default: ``utf8_output_decoder``. Replace
                         with the model's de-tokenizer for real LLM
                         inference.
      address_resolver   Default: identity (node_id == address).
                         Suitable when the Phase 6 peer registry uses
                         node_ids directly.
      streamed_send_message  Phase 3.x.7.1 streamed transport: ``(address,
                         manifest_bytes, chunk_bytes_iter) → (response_
                         manifest_bytes, response_chunk_bytes_iter)``.
                         Production wires this to Phase 6 gRPC bidi-
                         streaming. When None, activations exceeding
                         ``chunk_threshold_bytes`` raise
                         ``ChainExecutionError(ACTIVATION_TOO_LARGE)``.
      chunk_threshold_bytes  Inline-vs-streamed cutoff (default 10 MiB
                         from ``CHUNK_THRESHOLD_BYTES``). Activations
                         below this ride inline; above, streamed.
      chunk_bytes        Per-chunk size on the streamed path (default
                         1 MiB from Phase 6 ``ShardChunker``).
      default_deadline_seconds  Default 30s. Overridable via the env
                         ``PRSM_PARALLAX_CHAIN_DEADLINE_S`` — the per-stage
                         HandoffToken deadline budget. 30s is fine for WARM
                         inference, but a node COLD-loading a big model's layer
                         slice (sp1220-1225 slice-load: read ~GBs of
                         safetensors + build the skeleton + move to GPU) can
                         take minutes on the first request, which would expire a
                         30s token mid-load (DEADLINE_EXCEEDED). Operators
                         serving large models set this higher (e.g. 600).
    """
    import os as _os
    _dl_env = (_os.environ.get("PRSM_PARALLAX_CHAIN_DEADLINE_S", "") or "").strip()
    if _dl_env:
        try:
            _dl_val = float(_dl_env)
            if _dl_val > 0:
                default_deadline_seconds = _dl_val
        except ValueError:
            logger.warning(
                "PRSM_PARALLAX_CHAIN_DEADLINE_S=%r not a positive float; "
                "keeping default_deadline_seconds=%s",
                _dl_env, default_deadline_seconds,
            )
    if prompt_encoder is None or output_decoder is None:
        # Loud one-time warning when defaults activate. The docstring
        # already calls this out, but an SDK user constructing the
        # factory with minimum args might not read it; the runtime
        # warning ensures the test-friendly default never silently
        # ships to prod with a real LLM model.
        logger.warning(
            "make_rpc_chain_executor: using utf8_prompt_encoder / "
            "utf8_output_decoder defaults — these are TEST-FRIENDLY "
            "byte-passthrough adapters, NOT real tokenizers. Production "
            "callers MUST pass prompt_encoder= + output_decoder= "
            "wrapping the model's actual tokenizer + de-tokenizer. "
            "See factories.py docstring for details."
        )
    base = RpcChainExecutor(
        settler_identity=settler_identity,
        send_message=send_message,
        streamed_send_message=streamed_send_message,
        token_stream_send_message=token_stream_send_message,
        anchor=anchor,
        prompt_encoder=prompt_encoder or utf8_prompt_encoder,
        output_decoder=output_decoder or utf8_output_decoder,
        address_resolver=address_resolver,
        chunk_threshold_bytes=chunk_threshold_bytes,
        chunk_bytes=chunk_bytes,
        default_deadline_seconds=default_deadline_seconds,
    )
    if not wrap_topology_aware and not wrap_activation_dp_aware:
        return base
    # Sprint 417 — wrap with sprint-414's
    # TopologyAwareChainExecutor by default so live
    # InferenceReceipts immediately carry a verifiable
    # topology_assignment hash. Operators wanting the raw
    # executor (e.g., for unit-test fixtures that assert
    # on the bare receipt shape) pass
    # wrap_topology_aware=False.
    inner_after_topology: Any = base
    if wrap_topology_aware:
        from prsm.compute.inference.topology_aware_executor import (
            TopologyAwareChainExecutor,
        )
        inner_after_topology = TopologyAwareChainExecutor(inner=base)
    if not wrap_activation_dp_aware:
        return inner_after_topology
    # Sprint 546 — wrap with sprint-419's
    # ActivationDPAwareChainExecutor (OUTERMOST per its
    # docstring composition order ``base → topology → dp``)
    # so live InferenceReceipts also carry a verifiable
    # activation_noise_trace. Tier NONE requests pass
    # through unchanged (the decorator skips the hook +
    # leaves trace=None). Operators wanting topology-only
    # receipts pass wrap_activation_dp_aware=False.
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    return ActivationDPAwareChainExecutor(
        inner=inner_after_topology,
        clip_norm=dp_clip_norm,
        delta=dp_delta,
    )


# ──────────────────────────────────────────────────────────────────────────
# LayerStageServer factory
# ──────────────────────────────────────────────────────────────────────────


def make_layer_stage_server(
    *,
    identity: NodeIdentity,
    registry: Any,
    runner: LayerSliceRunner,
    tee_runtime: Any,
    anchor: Any,
    clock: Optional[Callable[[], float]] = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES_ACTIVATION,
    streaming_runner: Optional[Any] = None,
    tier_c_streaming_decorator: Optional[
        Callable[[Any], Any]
    ] = None,
) -> LayerStageServer:
    """Build a ``LayerStageServer`` for a node hosting one or more
    chain stages.

    Required:
      identity      This node's NodeIdentity. Signs RunLayerSliceResponse
                    payloads; the same identity callers register on the
                    Phase 3.x.3 anchor.
      registry      Phase 3.x.2 ModelRegistry (typically
                    FilesystemModelRegistry).
      runner        LayerSliceRunner — the actual layer-execution
                    adapter. Production wraps Phase 2 Ring 8's
                    TensorParallelExecutor; a tiny wrapper module
                    will land in a future task once the
                    TensorParallelExecutor.run_layer_range signature
                    is finalized.
      tee_runtime   Phase 2 Ring 8 TEERuntime (SoftwareTEERuntime
                    for dev; SgxTEERuntime / etc. for production).
      anchor        Phase 3.x.3 anchor for verifying upstream tokens.

    Optional:
      clock              Defaults to ``time.time``. Tests override.
      chunk_bytes        Per-chunk size when chunking response activations
                         on the streamed path (Phase 3.x.7.1). Default
                         1 MiB from Phase 6 ``ShardChunker``.
      streaming_runner   Phase 3.x.10.x — optional
                         ``StreamingLayerRunner`` (Phase 3.x.8) wired
                         in for the token-stream path. When None
                         (default), the server rejects
                         ``handle_token_stream`` requests with
                         ``INTERNAL_ERROR`` ("not configured for
                         streaming") — back-compat with operators
                         not opting into streaming. When set
                         (typically built via
                         ``make_autoregressive_streaming_runner``),
                         the chat-style streaming UX through
                         ``prsm_inference`` MCP tool is live.
    """
    kwargs = dict(
        identity=identity,
        registry=registry,
        runner=runner,
        tee_runtime=tee_runtime,
        anchor=anchor,
        chunk_bytes=chunk_bytes,
    )
    if clock is not None:
        kwargs["clock"] = clock
    if streaming_runner is not None:
        kwargs["streaming_runner"] = streaming_runner
    if tier_c_streaming_decorator is not None:
        kwargs["tier_c_streaming_decorator"] = tier_c_streaming_decorator
    return LayerStageServer(**kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.q — Tier C chain-level constant-time decorator factory
# ──────────────────────────────────────────────────────────────────────────


def make_tier_c_sharded_executor(
    inner: Any,
    *,
    mode: str,
    cadence_seconds: Optional[float] = None,
    pad_to_bytes: Optional[int] = None,
):
    """Wrap an inner chain executor in one of the Phase 3.x.11.q
    constant-time decorators. Mirrors Phase 3.x.10.y's pattern of
    selecting M1 vs M2 by mode string.

    Args:
      inner:           Object exposing
                       ``execute_chain_streaming(request=, chain=)`` —
                       typically an ``RpcChainExecutor``.
      mode:            ``"m2"`` (BatchedTrailing — single trailing
                       frame) or ``"m1"`` (FixedRate — cadence-driven
                       per-token yield).
      cadence_seconds: required for ``mode="m1"``; rejected for
                       ``mode="m2"`` (M2 has no cadence concept).
      pad_to_bytes:    Phase 3.x.11.q.x — applies to ``mode="m2"``
                       only; pads or truncates the joined trailing
                       frame to exactly ``pad_to_bytes`` UTF-8
                       bytes (closes §7.13 honest-scope item #2,
                       M2 response-size leak). Rejected for
                       ``mode="m1"``.

    Returns:
      A ``BatchedTrailingShardedExecutor`` or
      ``FixedRateShardedExecutor`` instance.

    Raises:
      ValueError: unknown mode; cadence misconfig; pad_to_bytes
                  misapplied.
    """
    from prsm.compute.chain_rpc.tier_c_sharded_executors import (
        BatchedTrailingShardedExecutor,
        FixedRateShardedExecutor,
    )

    if mode == "m2":
        if cadence_seconds is not None:
            raise ValueError(
                "make_tier_c_sharded_executor: cadence_seconds is "
                "not applicable for mode='m2' (BatchedTrailing has "
                "no per-token cadence concept)"
            )
        return BatchedTrailingShardedExecutor(
            inner=inner, pad_to_bytes=pad_to_bytes,
        )
    if mode == "m1":
        if cadence_seconds is None:
            raise ValueError(
                "make_tier_c_sharded_executor: cadence_seconds is "
                "required for mode='m1' (FixedRate cadence-driven "
                "yield)"
            )
        if pad_to_bytes is not None:
            raise ValueError(
                "make_tier_c_sharded_executor: pad_to_bytes is "
                "applicable only for mode='m2' (M1 emits per-token "
                "frames; response-size masking belongs at the M2 "
                "trailing-frame boundary)"
            )
        return FixedRateShardedExecutor(
            inner=inner, cadence_seconds=cadence_seconds,
        )
    raise ValueError(
        f"make_tier_c_sharded_executor: unknown mode {mode!r}; "
        f"expected 'm2' or 'm1'"
    )
