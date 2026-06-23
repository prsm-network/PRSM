"""Sprint 592 (Phase 2A) — chain-executor send-message adapter scaffolding.

Sprint 578 plumbed PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=rpc as a hook
for the real ``make_rpc_chain_executor`` factory. Phase 2 wiring
requires bridging the gap between:

- ``SendMessage = Callable[[str, bytes], bytes]`` (factory contract;
  SYNC; address → response bytes); see
  ``prsm/compute/chain_rpc/client.py``
- ``transport.send_to_peer(peer_id, P2PMessage) -> bool`` (async;
  awaits + does not return response bytes directly)

This module scaffolds the adapter contract. Phase 2A (sprint 592)
introduces the type signature + Protocol + a NotImplementedError
placeholder ``build_send_message_adapter()``. Phase 2B (sprint 593)
adds the address-resolver helper. Phase 2C (sprint 594) implements
the async-to-sync bridge using ``asyncio.run_coroutine_threadsafe``.
Phase 2D (sprint 595) wires it into ``_build_chain_executor``.

This staged approach keeps each sprint scoped + reviewable. The
scaffolding lets test code reference the eventual contract today.
"""
from __future__ import annotations

import asyncio
from typing import (
    Any, Callable, Coroutine, Dict, Optional, Protocol, runtime_checkable,
)


# Re-export the canonical SendMessage signature from the factory's
# source of truth — adapters wired to make_rpc_chain_executor MUST
# conform to this Callable shape.
SendMessage = Callable[[str, bytes], bytes]


@runtime_checkable
class SendMessageAdapter(Protocol):
    """Phase 2 adapter contract.

    Implementations bridge between the daemon's async transport
    layer and the sync SendMessage contract required by
    ``make_rpc_chain_executor``. The adapter is responsible for:

    1. Resolving ``stage_address`` to a transport peer_id
       (delegates to a Phase-2B address resolver).
    2. Constructing a ``P2PMessage`` wrapping ``request_bytes``.
    3. Scheduling the async ``transport.send_to_peer`` on the
       daemon's running event loop (e.g., via
       ``asyncio.run_coroutine_threadsafe``).
    4. Blocking the calling thread until a response arrives or
       a deadline elapses; raising on transport failure.

    Phase 2A — this module. Protocol only; no implementation.
    """

    def __call__(self, stage_address: str, request_bytes: bytes) -> bytes:
        ...


class _Phase2AdapterNotReady(NotImplementedError):
    """Sprint 592 — raised by the scaffolding placeholder when
    callers try to USE the adapter before Phase 2C lands the real
    async-to-sync bridge.

    Distinct exception class so `_build_chain_executor`'s `rpc`
    branch can detect Phase 2 non-readiness cleanly + log the
    structured warning sprint 578 already established.
    """


def run_async_on_loop(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, Any],
    timeout: float,
) -> Any:
    """Sprint 594 (Phase 2C) — async-to-sync bridge primitive.

    Schedules ``coro`` on a running event loop from a DIFFERENT
    thread + returns the result synchronously. Thin wrapper around
    ``asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)``.

    Thread-safety contract:
      - ``loop`` MUST be running on a different thread than the
        caller. Calling from the loop's own thread deadlocks
        because the loop cannot make progress while blocked in
        ``.result()``.

    Used by Phase 2D wiring (sprint 595+) to bridge the sync
    SendMessage contract over async transport calls. Exposed as a
    standalone helper so the threading primitive is unit-testable
    in isolation from the chain-executor protocol layer.

    Raises whatever the coroutine raises (passed through);
    ``concurrent.futures.TimeoutError`` on timeout expiry.
    """
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


class PeerNotFound(RuntimeError):
    """Sprint 593 (Phase 2B) — raised by the address resolver when
    a chain stage's node_id isn't currently in ``transport.peers``.

    Operators triaging chain-executor failures see the missing
    node_id in the exception message. Sprint-595 wiring will catch
    this at executor-startup time + report it via the trust-stack
    observability surfaces (sprint 579 CLI + 582 /health/detailed)
    so the failure is visible BEFORE a real inference request hits
    the dispatch.
    """


def build_address_resolver(node: Any) -> Callable[[str], str]:
    """Sprint 593 (Phase 2B) — map chain stage node_id → transport
    peer.address by looking up ``node.transport.peers[node_id]``.

    Raises ``PeerNotFound`` when the node isn't currently in
    transport.peers. The chain executor's dispatch loop catches
    this to surface "stage N unreachable" cleanly rather than
    propagating a KeyError up.

    Returns the canonical ``AddressResolver = Callable[[str], str]``
    shape from ``prsm/compute/chain_rpc/client.py``.
    """
    transport = node.transport

    def _resolve(node_id: str) -> str:
        # Sprint 687 F34 — self-dispatch sentinel. When the chain
        # routes a stage to the local node, transport.peers cannot
        # contain self (only remote connections live there). Return
        # the node_id unchanged so the downstream send_message
        # adapter detects self via identity comparison + short-
        # circuits to the local StageExecutor. Without this, the
        # resolver raises PeerNotFound and the entire chain aborts
        # — even though local execution doesn't need any peer
        # connection.
        self_node_id = getattr(
            getattr(node, "identity", None), "node_id", None,
        )
        if self_node_id is not None and node_id == self_node_id:
            return node_id
        peer = transport.peers.get(node_id)
        if peer is None:
            raise PeerNotFound(
                f"chain stage node_id {node_id!r} not currently "
                f"in transport.peers; cannot dispatch. Likely "
                f"causes: peer dropped connection, auto-dial sweep "
                f"hasn't reached this peer yet (sprint 573), or "
                f"peer never registered against the same bootstrap."
            )
        # Sprint 695 F45 fix — return node_id (peer_id) not peer.address.
        # Sprint 596's convention: "stage_address IS the target peer_id".
        # The send_message adapter calls
        # `transport.send_to_peer(stage_address, msg)` which looks up
        # via `self.peers.get(stage_address)` — a peer_id, not a
        # network address. Pre-fix the resolver returned peer.address
        # ("1.2.3.4:9001") which caused send_to_peer to fail (no
        # such peer_id), only visible now that sprint-695's
        # rtt_to_nodes fix lets the routing actually reach
        # cross-host dispatch.
        return node_id

    return _resolve


# Sprint 596 — chain-executor wire-protocol identifiers.
# Chain-executor request messages set payload[CHAIN_REQ_KEY] = request_id;
# response messages set payload[CHAIN_RESP_KEY] = request_id + payload
# bytes (base64-encoded in the JSON envelope). Sprint 597 wires the
# response handler that reads CHAIN_RESP_KEY and resolves the pending
# Future identified by request_id.
CHAIN_MSG_TYPE = "chain_executor_rpc"
CHAIN_REQ_KEY = "chain_req_id"
CHAIN_RESP_KEY = "chain_resp_id"
CHAIN_PAYLOAD_KEY = "chain_payload_b64"
# Sprint 601 (Phase 2E-1) — error indicator in server-side response.
# Lets the requester distinguish "stage handler returned an error"
# from "no stage handler exists yet" (Phase 2E-1 ships scaffolding;
# Phase 2E-2+ replaces this with real stage execution).
CHAIN_ERROR_KEY = "chain_error"

# Sprint 711 (F40 remote streaming) — token-stream wire protocol.
# Distinct subtype from unary CHAIN_MSG_TYPE so the receive-side
# router can dispatch streaming and unary independently. Wire format:
#
#   Requester → Server: ONE STREAM_REQ message
#     payload[CHAIN_STREAM_REQ_KEY] = stream_id
#     payload[CHAIN_PAYLOAD_KEY]    = base64(request_bytes)
#
#   Server → Requester: MULTIPLE STREAM_FRAME messages, one per
#   frame yielded by LayerStageServer.handle_token_stream:
#     payload[CHAIN_STREAM_FRAME_KEY] = stream_id
#     payload[CHAIN_STREAM_SEQ_KEY]   = frame index (0-based)
#     payload[CHAIN_PAYLOAD_KEY]      = base64(frame_bytes)
#
#   Server → Requester: ONE terminal STREAM_END message:
#     payload[CHAIN_STREAM_END_KEY]   = stream_id
#     payload[CHAIN_STREAM_SEQ_KEY]   = total frame count
#     (optional) payload[CHAIN_ERROR_KEY] = error message if mid-
#                                            stream failure
#
# The terminal STREAM_END is required even on error so the requester
# closes its iterator cleanly. Frame ordering is by SEQ_KEY (NOT by
# arrival time) so out-of-order delivery on the underlying transport
# doesn't corrupt the stream.
CHAIN_STREAM_MSG_TYPE = "chain_executor_stream_rpc"
CHAIN_STREAM_REQ_KEY = "chain_stream_req_id"
CHAIN_STREAM_FRAME_KEY = "chain_stream_frame_id"
CHAIN_STREAM_END_KEY = "chain_stream_end_id"
CHAIN_STREAM_SEQ_KEY = "chain_stream_seq"


class StageExecutionError(RuntimeError):
    """Sprint 602 (Phase 2E-2) — raised by StageExecutor.execute()
    when forward execution of a chain stage fails.

    Subclasses RuntimeError so generic operator error-handling
    paths catch it via isinstance(exc, RuntimeError) naturally.
    """


@runtime_checkable
class StageExecutor(Protocol):
    """Sprint 602 (Phase 2E-2) — server-side stage executor contract.

    Receives a chain stage's serialized request bytes (typically
    activation bytes from the previous stage) and returns the
    response bytes (output activation passed to the next stage).

    Phase 2E-4 (sprint 604) wires this Protocol into
    handle_chain_executor_request — request bytes flow in, response
    bytes flow back out via the existing wire format.

    Implementations may raise ``StageExecutionError`` on failure;
    the request handler converts to a CHAIN_ERROR_KEY response.
    """

    async def execute(self, request_bytes: bytes) -> bytes:
        ...


def build_stub_stage_executor() -> StageExecutor:
    """Sprint 602 (Phase 2E-2) — placeholder StageExecutor that
    raises StageExecutionError with an actionable message.

    Phase 2E-3 (sprint 603) ships a real implementation that
    decodes activation bytes + forwards through a model layer.
    Phase 2E-4 (sprint 604) wires this into the request handler
    so it's actually called.
    """

    class _StubStageExecutor:
        async def execute(self, request_bytes: bytes) -> bytes:
            raise StageExecutionError(
                "Sprint 602 Phase 2E-2 stub: StageExecutor.execute() "
                "is not yet implemented. Phase 2E-3 (sprint 603) "
                "will ship the real chain-stage forward execution. "
                "Until then, this node ACKs incoming requests via "
                "the sprint-601 request handler but cannot actually "
                "produce next-stage activations."
            )

    return _StubStageExecutor()


class LayerStageServerStageExecutor:
    """Sprint 606 (Phase 2F-1) — wraps a chain_rpc LayerStageServer
    as a StageExecutor.

    The existing
    ``prsm.compute.chain_rpc.server.LayerStageServer.handle(bytes) -> bytes``
    already has the exact signature our StageExecutor Protocol
    expects (sync, bytes-in/bytes-out, never raises by design).
    This adapter:
      - Provides the async ``execute()`` Protocol method
      - Dispatches the (sync, potentially CPU-heavy) handle()
        call via ``loop.run_in_executor`` so the event loop
        thread isn't blocked
      - Wraps any unexpected exception in ``StageExecutionError``
        (defense-in-depth — handle() shouldn't raise but if it
        does, the request handler surfaces it cleanly)

    Phase 2F-2 (sprint 607+) ships the factory that constructs
    the underlying LayerStageServer with the operator's identity +
    registry + runner + tee_runtime + anchor.
    """

    def __init__(self, *, server: Any) -> None:
        if server is None or not hasattr(server, "handle"):
            raise ValueError(
                "LayerStageServerStageExecutor requires a server "
                "with a .handle(bytes) -> bytes method"
            )
        self._server = server

    async def execute(self, request_bytes: bytes) -> bytes:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._server.handle, request_bytes,
            )
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"LayerStageServer.handle raised {type(exc).__name__}: "
                f"{exc}"
            ) from exc


# Sprint 1220 (Brick 1) — process-shared HF model cache. The layer-slice
# runner + the prompt encoder + the streaming runner each from_pretrained'd the
# FULL model separately (2-4 full copies on the head) and reloaded per request.
# Keyed by (model_id, device, dtype) so one process loads each model ONCE; the
# instance-level `self._hf_model` still short-circuits, but a fresh
# runner/encoder now reuses the cached weights instead of re-reading them.
_HF_MODEL_CACHE: Dict[tuple, Any] = {}


def _torch_dtype_from_str(dtype_str: str) -> Any:
    """Map a dtype string → torch dtype (fp32 fallback)."""
    import torch as _torch
    return {
        "float16": _torch.float16,
        "bfloat16": _torch.bfloat16,
        "float32": _torch.float32,
    }.get(dtype_str, _torch.float32)


def _resolve_model_dtype(device: str) -> str:
    """Sprint 1220 — load dtype string for the HF runner/encoder: fp16 on CUDA
    (≈2x less VRAM → a 7-8B model fits a 24GB A10), fp32 elsewhere. Reuses
    local_inference.resolve_dtype so PRSM_LOCAL_INFERENCE_DTYPE overrides apply
    uniformly. Greedy decode stays deterministic at any dtype (§7-safe)."""
    from prsm.compute.inference.local_inference import resolve_dtype
    return resolve_dtype(None, device)


def _model_compute_dtype(hf_model: Any) -> Any:
    """Sprint 1220 (adversarial-verify C1) — the dtype of the model's params,
    for the activation-boundary cast. Feeding an fp32 activation into an fp16
    nn.Linear raises 'mat1 and mat2 must have the same dtype', so the cast MUST
    match the loaded model — not the old hardcoded float32. fp32 fallback when
    the model exposes no params (defensive)."""
    import torch as _torch
    try:
        return next(hf_model.parameters()).dtype
    except (StopIteration, Exception):  # noqa: BLE001
        return _torch.float32


def _load_hf_model_cached(model_id: str, device: str, dtype_str: str) -> Any:
    """Sprint 1220 — load (or fetch from the process cache) an HF causal-LM in
    the resolved dtype on the target device. Collapses the prior multi-copy +
    per-request reloads to ONE load per (model_id, device, dtype). transformers
    v5 renamed torch_dtype→dtype (and ignores torch_dtype), so the dtype kwarg
    name is version-resolved — getting it wrong silently loads fp32 and OOMs."""
    key = (model_id, device, dtype_str)
    cached = _HF_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    import transformers  # noqa: F401
    from transformers import AutoModelForCausalLM
    from prsm.compute.inference.local_inference import (
        dtype_kwarg_for_transformers,
    )
    # getattr default keeps mocked-`transformers` test doubles (no __version__)
    # working; "" → dtype_kwarg_for_transformers falls back to torch_dtype.
    dtype_kwarg = dtype_kwarg_for_transformers(
        getattr(transformers, "__version__", "") or "",
    )
    kwargs = {
        dtype_kwarg: _torch_dtype_from_str(dtype_str),
        # sp702 F48 — materialize weights during load (not meta) so
        # `.to(device)` doesn't raise "Cannot copy out of meta tensor".
        "low_cpu_mem_usage": False,
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = model.to(device).eval()
    _HF_MODEL_CACHE[key] = model
    return model


# ── Sprint 1221 (Brick 2) — pure slice-key resolver + sharded-safetensors reader ──
# A stage loads ONLY the checkpoint tensors it owns: its decoder layers
# [start,end) plus, on the final stage, the final norm + lm_head (untied) or
# embed_tokens (tied — tie_weights wires lm_head to it). No model construction
# here; these are pure functions over a snapshot's weight_map + safetensors.

def _resolve_weight_map(snapshot_dir: str) -> Dict[str, str]:
    """Map checkpoint-key → shard-filename. Handles BOTH the sharded case
    (model.safetensors.index.json `weight_map`) AND the single-file case
    (a lone model.safetensors with NO index — small models; adversarial-verify
    CO1)."""
    import json
    import os
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fh:
            idx = json.load(fh)
        wm = idx.get("weight_map") or {}
        if not isinstance(wm, dict) or not wm:
            raise StageExecutionError(
                f"slice-load: {index_path} has no usable weight_map",
            )
        return dict(wm)
    single = os.path.join(snapshot_dir, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt", device="cpu") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise StageExecutionError(
        f"slice-load: no safetensors checkpoint found in {snapshot_dir} "
        f"(expected model.safetensors[.index.json])",
    )


def _layer_index_of(key: str, layers_prefix: str) -> Optional[int]:
    """Parse the decoder-layer index N from a key like
    f'{layers_prefix}.{N}.…' (e.g. 'model.layers.7.self_attn.q_proj.weight' →
    7). Returns None for non-layer keys (embed/norm/lm_head)."""
    pre = layers_prefix + "."
    if not key.startswith(pre):
        return None
    head = key[len(pre):].split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _remap_layer_key(key: str, layers_prefix: str, start: int) -> str:
    """Relative-remap a layer key's index by -start (model.layers.{start+i} →
    model.layers.{i}); non-layer keys pass through unchanged. This is what lets
    a [start,end) slice load into a reduced-config skeleton whose layer list is
    indexed 0..(end-start-1)."""
    idx = _layer_index_of(key, layers_prefix)
    if idx is None:
        return key
    pre = layers_prefix + "."
    rest = key[len(pre):]
    suffix = rest.split(".", 1)[1] if "." in rest else ""
    new_idx = idx - start
    return f"{pre}{new_idx}.{suffix}" if suffix else f"{pre}{new_idx}"


def _resolve_slice_owned_keys(
    all_keys: Any, *, layers_prefix: str, norm_key: str, lm_head_key: str,
    embed_key: str, start: int, end: int, is_final_stage: bool,
    tie_word_embeddings: bool,
) -> set:
    """The exact checkpoint keys this stage OWNS. Decoder layers in [start,end)
    (ALL of each layer's tensors — incl. Qwen2's q/k/v .bias). The final stage
    also owns the final norm + the head: lm_head.weight when untied, else
    embed_tokens.weight (the tied source — tie_weights() points lm_head at it).
    Stage 0 owns NO input embedding: run_layer_range never embeds (the head's
    prompt encoder applies embeddings upstream)."""
    owned = set()
    for k in all_keys:
        idx = _layer_index_of(k, layers_prefix)
        if idx is not None and start <= idx < end:
            owned.add(k)
    if is_final_stage:
        if norm_key in all_keys:
            owned.add(norm_key)
        target = embed_key if tie_word_embeddings else lm_head_key
        if target in all_keys:
            owned.add(target)
    return owned


def _read_owned_state_dict(
    snapshot_dir: str, weight_map: Dict[str, str], owned_keys: Any, *,
    layers_prefix: str, start: int, dtype: str,
) -> Dict[str, Any]:
    """Read ONLY the owned tensors from the sharded safetensors, casting to
    `dtype` and relative-remapping layer keys. Groups owned keys BY SHARD (a
    single layer's tensors can straddle two shards) so each shard file is opened
    exactly once."""
    import os
    from safetensors import safe_open
    by_shard: Dict[str, list] = {}
    for k in owned_keys:
        shard = weight_map.get(k)
        if shard is None:
            raise StageExecutionError(
                f"slice-load: owned key {k!r} absent from weight_map",
            )
        by_shard.setdefault(shard, []).append(k)
    td = _torch_dtype_from_str(dtype)
    sd: Dict[str, Any] = {}
    for shard, keys in by_shard.items():
        path = os.path.join(snapshot_dir, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in keys:
                sd[_remap_layer_key(k, layers_prefix, start)] = (
                    f.get_tensor(k).to(td)
                )
    return sd


# ── Sprint 1222 (Brick 3) — reduced-config slice loader ──
# Standard LLaMA/Qwen/Mistral (rotary `.model.layers`) state-dict keys. Slice
# loading is gated to this architecture family (Brick 4); gpt2 (.transformer.h,
# no model-level rotary) keeps the full-load path.
_SLICE_LAYERS_PREFIX = "model.layers"
_SLICE_NORM_KEY = "model.norm.weight"
_SLICE_LM_HEAD_KEY = "lm_head.weight"
_SLICE_EMBED_KEY = "model.embed_tokens.weight"


def _resolve_snapshot_dir(model_id: str) -> str:
    """Sprint 1222 (adversarial-verify CO3) — resolve a model id → its local HF
    snapshot dir (holding config.json + model.safetensors[.index.json]). A
    local path is used directly; otherwise the HF cache is resolved offline
    (the model must be fully downloaded — the daemon stages it at startup)."""
    import os
    if os.path.isdir(model_id) and os.path.exists(
        os.path.join(model_id, "config.json")
    ):
        return model_id
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_id, local_files_only=True,
            allow_patterns=["*.json", "*.safetensors"],
        )
    except Exception as exc:  # noqa: BLE001
        raise StageExecutionError(
            f"slice-load: cannot resolve a local snapshot for {model_id!r} "
            f"({type(exc).__name__}: {exc}). The model must be fully "
            f"downloaded (daemon startup staging / snapshot_download)."
        ) from exc


def _shard_fetch_enabled() -> bool:
    """Sprint 1228 — OPT-IN per-node shard-only download (PRSM_PARALLAX_SHARD_FETCH).
    When on, a node fetches ONLY the safetensors shards holding its assigned
    layers (+ config + index) from the HF hub, instead of the whole model — so
    it DOWNLOADS + STORES ~its slice (GBs), not the full model (hundreds of GB
    for a frontier model). The loader already READS only its owned shards
    (sp1221); this closes the FETCH side, realizing the storage half of the
    distributed-model end-state. Default OFF preserves the full-snapshot path
    (back-compat + the local-dir unit tests)."""
    import os
    return str(os.environ.get("PRSM_PARALLAX_SHARD_FETCH", "")).strip().lower() in (
        "1", "true", "yes", "on",
    )


def _fetch_only_files(model_id: str, shards: Any = None) -> str:
    """Sprint 1228 — download ONLY config.json + the safetensors index (both
    tiny) + the given `shards` from the HF hub. Idempotent (hf_hub_download is
    cached). Returns the snapshot dir (all fetched files land there). A node
    never pulls the full model — only the shards for the layers it owns."""
    import os
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
    snapshot_dir = os.path.dirname(cfg_path)
    try:
        hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
    except Exception:  # noqa: BLE001 — single-file model has no index
        pass
    for s in (shards or []):
        hf_hub_download(repo_id=model_id, filename=s)
    return snapshot_dir


def build_reduced_config_slice_model(
    model_id: str, *, start: int, end: int, is_final_stage: bool,
    device: str, dtype: str,
) -> tuple:
    """Sprint 1222 (Brick 3) — build a reduced-config skeleton holding ONLY the
    [start,end) decoder layers (+ on the final stage the norm + lm_head/embed)
    and materialize ONLY this stage's owned weights from the sharded
    safetensors. Resident weights ≈ the slice (≈6.5GB/side for a 7B 14+14 fp16)
    instead of the full model — the unblock for >24GB models multi-host.
    Returns ``(model, full_num_layers)``.

    Adversarial-review fixes folded in:
      C2 — shrink num_hidden_layers AND every config list of length
           full_num_layers (Qwen2/3 carry ``layer_types`` validated against it;
           setting only num_hidden_layers raises ValueError in transformers v5).
      C3 — run the fail-loud key assertion AFTER tie_weights() (a tied
           lm_head.weight reads as 'missing' until tie_weights wires it, which
           would FALSE-FAIL tied models and silently defeat sharding).
      M1 — delete the from_config full-vocab modules this stage never reads
           BEFORE loading, to cap the transient memory spike.
    The model-level rotary_emb (sp1216) is auto-built by from_config (inv_freq
    is a non-persistent buffer, not a checkpoint tensor) — free + correct."""
    import torch as _torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if not (0 <= start < end):
        raise StageExecutionError(
            f"slice-load: invalid range start={start} end={end}",
        )
    # sp1228 — shard-fetch: pull ONLY config+index now (tiny); the owned shards
    # are fetched below, once we know which layers (and thus shards) we own. A
    # local dir (tests / pre-staged) skips fetch entirely. Off → full snapshot.
    import os as _os_sf
    _shard_fetch = _shard_fetch_enabled() and not _os_sf.path.isdir(model_id)
    if _shard_fetch:
        snapshot_dir = _fetch_only_files(model_id)
    else:
        snapshot_dir = _resolve_snapshot_dir(model_id)
    cfg = AutoConfig.from_pretrained(snapshot_dir)
    full_num_layers = int(getattr(cfg, "num_hidden_layers"))
    if end > full_num_layers:
        raise StageExecutionError(
            f"slice-load: end={end} exceeds model depth {full_num_layers} "
            f"for {model_id!r}",
        )
    tie = bool(getattr(cfg, "tie_word_embeddings", False))

    # Resolve the weight_map + arch-check EARLY (cheap index.json read) so an
    # unsupported architecture (gpt2's .transformer.h, no model.layers) fails
    # FAST here → the runner's cheap fallback to the proven full-load path,
    # without first building a from_config skeleton.
    weight_map = _resolve_weight_map(snapshot_dir)
    if not any(
        _layer_index_of(k, _SLICE_LAYERS_PREFIX) is not None for k in weight_map
    ):
        raise StageExecutionError(
            f"slice-load: {model_id!r} has no {_SLICE_LAYERS_PREFIX}.* keys — "
            f"unsupported architecture for slicing (e.g. gpt2). Use full-load.",
        )

    # C2 — slice every config list coupled to the layer count (layer_types et
    # al.), THEN set num_hidden_layers, so from_config's length validation
    # passes.
    n_slice = end - start
    for attr, val in list(vars(cfg).items()):
        if isinstance(val, list) and len(val) == full_num_layers:
            setattr(cfg, attr, val[start:end])
    cfg.num_hidden_layers = n_slice

    # sp1227 — build the skeleton on the META device: instant, and crucially
    # NO wasteful random-init of the ~billions of params we immediately
    # overwrite via load_state_dict. That CPU-bound init took MINUTES for a
    # 14B's slices and starved the node's P2P heartbeat → the worker dropped →
    # pool churn → slice rebuild (cascade that blocked the live 14B chain).
    # transformers 5.2 removed no_init_weights; meta is the accelerate-free way
    # to skip init. Owned weights are materialized off-meta by
    # load_state_dict(assign=True); the model-level rotary buffer is rebuilt and
    # unused modules are deleted below, so no meta tensor survives to .to().
    with _torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    # M1 — free the full-vocab modules this stage never reads, BEFORE load.
    # Non-final (incl. stage 0): the head embeds upstream + only the final
    # stage decodes → neither embed nor lm_head is used. Final-untied: drop
    # embed (lm_head is loaded). Final-tied: keep both (embed loaded; lm_head
    # tied to it by tie_weights()).
    if not is_final_stage:
        model.model.embed_tokens = None
        model.lm_head = None
        # sp1227 — the final norm is unused on a non-final stage; delete it so
        # it doesn't survive as an (unloaded) meta tensor that .to() can't move.
        model.model.norm = None
    elif not tie:
        model.model.embed_tokens = None

    owned = _resolve_slice_owned_keys(
        set(weight_map), layers_prefix=_SLICE_LAYERS_PREFIX,
        norm_key=_SLICE_NORM_KEY, lm_head_key=_SLICE_LM_HEAD_KEY,
        embed_key=_SLICE_EMBED_KEY, start=start, end=end,
        is_final_stage=is_final_stage, tie_word_embeddings=tie,
    )
    # sp1228 — now that we know our owned keys, fetch ONLY the shards holding
    # them (config+index were fetched above). This node downloads ~its slice,
    # never the full model.
    if _shard_fetch:
        _fetch_only_files(model_id, {weight_map[k] for k in owned})
    slice_sd = _read_owned_state_dict(
        snapshot_dir, weight_map, owned,
        layers_prefix=_SLICE_LAYERS_PREFIX, start=start, dtype=dtype,
    )
    result = model.load_state_dict(slice_sd, strict=False, assign=True)
    if is_final_stage and tie:
        model.tie_weights()

    # sp1227 — the meta-built skeleton's model-level rotary buffer (inv_freq) is
    # a meta tensor (it's computed in __init__, not a checkpoint weight, so
    # assign didn't materialize it). Rebuild the rotary module on the real
    # device — recomputes a real, finite inv_freq. Version-agnostic via
    # type(rotary)(config=cfg). Needed by EVERY stage (all decoder layers use
    # the model-level cos/sin — sp1216).
    _inner = getattr(model, "model", None)
    if _inner is not None and getattr(_inner, "rotary_emb", None) is not None:
        try:
            _inner.rotary_emb = type(_inner.rotary_emb)(config=cfg).to(device)
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"slice-load: failed to rebuild rotary_emb after meta-build "
                f"for {model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

    # C3 — fail-loud AFTER tie_weights: no OWNED key may be missing; nothing
    # unexpected (an off-by-one relative remap surfaces here as an unexpected
    # key). This is the silent-corruption guard for strict=False.
    missing = set(getattr(result, "missing_keys", []) or [])
    unexpected = set(getattr(result, "unexpected_keys", []) or [])
    owned_remapped = {
        _remap_layer_key(k, _SLICE_LAYERS_PREFIX, start) for k in owned
    }
    still_missing = owned_remapped & missing
    if is_final_stage and tie:
        still_missing.discard(_SLICE_LM_HEAD_KEY)  # satisfied by tie_weights
    if unexpected:
        raise StageExecutionError(
            f"slice-load: {model_id!r} [{start},{end}) produced UNEXPECTED "
            f"keys (relative-remap bug?): {sorted(unexpected)[:5]}",
        )
    if still_missing:
        raise StageExecutionError(
            f"slice-load: {model_id!r} [{start},{end}) failed to load OWNED "
            f"keys (silent-corruption guard): {sorted(still_missing)[:5]}",
        )

    # sp1227 — guard: NO meta tensor may survive (the .to(device) below cannot
    # copy out of meta). Every owned param is off-meta via assign, the rotary
    # was rebuilt, and unused modules were deleted — a leftover meta tensor is a
    # materialization bug, surfaced loudly rather than crashing the .to().
    _meta = [n for n, t in model.named_parameters() if t.is_meta]
    _meta += [n for n, t in model.named_buffers() if t.is_meta]
    if _meta:
        raise StageExecutionError(
            f"slice-load: {model_id!r} [{start},{end}) left meta tensors after "
            f"materialization (incomplete load): {sorted(_meta)[:5]}",
        )

    model = model.to(device=device, dtype=_torch_dtype_from_str(dtype)).eval()
    return model, full_num_layers


def _slice_load_enabled() -> bool:
    """Sprint 1223 (Brick 4) — per-stage layer-slice weight loading is OPT-IN
    via PRSM_PARALLAX_SLICE_LOAD. Default OFF keeps the proven full-load path
    (gpt2/1.5B) byte-identical; on, the runner loads only its assigned layers
    so a >24GB model shards across nodes."""
    import os
    return str(os.environ.get("PRSM_PARALLAX_SLICE_LOAD", "")).strip().lower() in (
        "1", "true", "yes", "on",
    )


def _build_embedding_only(model_id: str, device: str, dtype: str) -> Any:
    """Sprint 1224 (Brick 5) — load ONLY the input embedding
    (model.embed_tokens) as a standalone nn.Embedding, so a slice-load HEAD
    (which runs the prompt encoder + its stage-0 slice) does NOT carry a full
    model just to embed the prompt. Required for genuinely-too-big-for-one-GPU
    models: the head can't load the full model to encode. Rotary arch only —
    raises (→ caller falls back to full-load) if there's no model.embed_tokens,
    which keeps gpt2's learned-position (wpe) path on the full model."""
    import torch as _torch
    import os as _os_sf
    # sp1228 — shard-fetch: pull config+index, then ONLY the embed shard (the
    # head needs just the embedding to encode, not the whole model).
    _shard_fetch = _shard_fetch_enabled() and not _os_sf.path.isdir(model_id)
    if _shard_fetch:
        snapshot_dir = _fetch_only_files(model_id)
    else:
        snapshot_dir = _resolve_snapshot_dir(model_id)
    weight_map = _resolve_weight_map(snapshot_dir)
    if _SLICE_EMBED_KEY not in weight_map:
        raise StageExecutionError(
            f"embed-only: {model_id!r} has no {_SLICE_EMBED_KEY} "
            f"(non-rotary arch, e.g. gpt2) — use full-load.",
        )
    if _shard_fetch:
        _fetch_only_files(model_id, {weight_map[_SLICE_EMBED_KEY]})
    sd = _read_owned_state_dict(
        snapshot_dir, weight_map, {_SLICE_EMBED_KEY},
        layers_prefix=_SLICE_LAYERS_PREFIX, start=0, dtype=dtype,
    )
    w = sd[_SLICE_EMBED_KEY]  # [vocab, hidden]
    # from_pretrained builds an Embedding whose weight IS w — no full-vocab
    # random alloc; freeze (inference). Bit-identical lookup to the full
    # model's get_input_embeddings() (same weight, and rotary adds no wpe).
    emb = _torch.nn.Embedding.from_pretrained(w, freeze=True)
    return emb.to(device).eval()


def _resolve_hf_layers(hf_model: Any) -> Any:
    """Sprint 612 (Phase 2F-5c) — polymorphic HF layer extraction.

    Try known causal-LM layer-list paths in order:
      model.model.layers      LLaMA / Mistral / Qwen
      model.transformer.h     GPT-2 / Falcon / GPT-J
      model.gpt_neox.layers   GPT-NeoX / Pythia

    Returns the first path that resolves. Raises StageExecutionError
    listing all attempted paths if none match — operator triages.

    The "first that resolves" rule means LLaMA-style models with a
    populated ``model.model.layers`` always take that path even if
    other attributes are coincidentally present.
    """
    attempted = []
    # LLaMA / Mistral / Qwen style
    attempted.append(".model.layers")
    sub = getattr(hf_model, "model", None)
    if sub is not None:
        layers = getattr(sub, "layers", None)
        if layers is not None:
            return layers
    # GPT-2 / Falcon / GPT-J style
    attempted.append(".transformer.h")
    sub = getattr(hf_model, "transformer", None)
    if sub is not None:
        layers = getattr(sub, "h", None)
        if layers is not None:
            return layers
    # GPT-NeoX / Pythia style
    attempted.append(".gpt_neox.layers")
    sub = getattr(hf_model, "gpt_neox", None)
    if sub is not None:
        layers = getattr(sub, "layers", None)
        if layers is not None:
            return layers
    raise StageExecutionError(
        f"HuggingFaceLayerSliceRunner: model does not expose any "
        f"known layer-list path. Attempted: {', '.join(attempted)}. "
        f"Add support for this architecture in _resolve_hf_layers."
    )


def _parallax_chat_template_enabled(env: Any = None) -> bool:
    """Sprint 1230 — apply the model's chat template on the parallax encode
    path (default ON). Delegates to the single source of truth in
    local_inference (shared with the sp1231 streaming path) so the
    ``PRSM_PARALLAX_CHAT_TEMPLATE`` toggle has one definition."""
    from prsm.compute.inference.local_inference import (
        parallax_chat_template_enabled,
    )
    return parallax_chat_template_enabled(env)


def _encode_prompt_tokens(tokenizer: Any, prompt: str, *, env: Any = None) -> Any:
    """Sprint 1230 — tokenize a prompt for the parallax encode path, applying
    the model's chat template for instruct models (mirrors the single-node
    sp1208 path by REUSING its tested helpers). Returns the bare ``input_ids``
    tensor the embedding lookup expects. Base models (no ``chat_template``) →
    raw encode, byte-identical to the pre-sp1230 ``tokenizer.encode`` path.
    """
    from prsm.compute.inference.local_inference import (
        should_use_chat_template,
        encode_for_generation,
    )
    use_ct = should_use_chat_template(
        tokenizer, enabled=_parallax_chat_template_enabled(env),
    )
    enc = encode_for_generation(tokenizer, prompt, use_chat_template=use_ct)
    # encode_for_generation returns a BatchEncoding/dict (chat-template path and
    # the raw tokenizer(...) path both do); extract input_ids. A bare tensor
    # (older transformers fallback) passes through unchanged.
    if isinstance(enc, dict) or hasattr(enc, "keys"):
        return enc["input_ids"]
    return enc


def build_hf_prompt_encoder(
    *,
    model_id: str,
    device: str = "cpu",
) -> Callable[[str], Any]:
    """Sprint 614 (Phase 2F-5e) — client-side HF PromptEncoder.

    Returns a closure that maps a prompt string to the first stage's
    activation tensor (token embeddings). Wires into the
    ``RpcChainExecutor`` factory's optional ``prompt_encoder`` arg
    (sprint 615 ships the env-wired hookup).

    Flow on first encoder() call:
      1. Lazy-load HF AutoTokenizer + AutoModelForCausalLM
         (cached for subsequent calls)
      2. tokenizer.encode(prompt, return_tensors="pt") → token_ids
      3. model.get_input_embeddings()(token_ids) → embeddings tensor
      4. Convert to numpy ndarray, return

    Failure modes (transformers missing, model 404, etc.) wrap
    as StageExecutionError so callers see a structured error
    instead of a bare HuggingFace exception.

    Memory note: loads the FULL model just to access its embedding
    layer. For very large models this is wasteful. Phase 2F-5e+ may
    add a thinner path that loads only the embedding matrix +
    vocab.
    """
    if not model_id or not isinstance(model_id, str):
        raise ValueError(
            "build_hf_prompt_encoder requires model_id "
            "(HuggingFace identifier)"
        )
    # Sprint 617 — auto-detect / safe-fallback device selection.
    device = _resolve_hf_device(device)

    # Closed-over cache so we only load once.
    _state: Dict[str, Any] = {
        "tokenizer": None,
        "model": None,
        "embed_layer": None,
    }

    def _encode(prompt: str) -> Any:
        if _state["tokenizer"] is None:
            try:
                import transformers  # noqa: F401
                import torch  # noqa: F401
            except ImportError as exc:
                raise StageExecutionError(
                    f"build_hf_prompt_encoder requires `transformers` "
                    f"+ `torch` packages installed: {exc}."
                ) from exc
            try:
                from transformers import AutoTokenizer
                import torch as _torch  # noqa: F401
                tok = AutoTokenizer.from_pretrained(model_id)
                _dtype = _resolve_model_dtype(device)
                # sp1224 (Brick 5) — under slice-load, materialize ONLY the
                # input embedding (rotary arch) so a sliced head doesn't carry
                # the full model just to embed. Falls back to the full-model
                # load (sp1220 shared cache) for gpt2/wpe or any embed-only
                # failure. _state["model"]=None on the embed-only path → the
                # wpe branch below is skipped (rotary needs none).
                _embed_only = None
                if _slice_load_enabled():
                    try:
                        _embed_only = _build_embedding_only(
                            model_id, device, _dtype,
                        )
                    except Exception as _eo_exc:  # noqa: BLE001 — full fallback
                        import logging as _lg
                        _lg.getLogger(__name__).warning(
                            "embed-only encoder load failed for %s — falling "
                            "back to full-model embed: %s", model_id, _eo_exc,
                        )
                if _embed_only is not None:
                    model = None
                    embed_layer = _embed_only
                else:
                    # sp1220 — share the process-level model cache with the
                    # layer-slice runner + streaming runner (one load per
                    # (model_id, device, dtype); fp16 on CUDA, fp32 on CPU).
                    model = _load_hf_model_cached(model_id, device, _dtype)
                    embed_layer = model.get_input_embeddings()
            except Exception as exc:  # noqa: BLE001
                raise StageExecutionError(
                    f"build_hf_prompt_encoder failed to load "
                    f"{model_id!r}: {type(exc).__name__}: {exc}"
                ) from exc
            _state["tokenizer"] = tok
            _state["model"] = model
            _state["embed_layer"] = embed_layer
        try:
            import torch as _torch
            # Sprint 1230 — apply the model's chat template for instruct models
            # (was raw tokenizer.encode → instruct models tokenized
            # completion-style on the distributed path). No-op for base models.
            token_ids = _encode_prompt_tokens(_state["tokenizer"], prompt)
            # Sprint 698 F47 fix — move token_ids to the model's
            # device before embedding lookup. tokenizer.encode
            # returns CPU tensor; embed_layer is on the model's
            # device. Without this, GPU deployments raise
            # "Expected all tensors to be on the same device".
            token_ids = token_ids.to(device)
            # Move tokens to device + run embedding lookup under no_grad
            with _torch.no_grad():
                embeddings = _state["embed_layer"](token_ids)
                # Sprint 688 F38 — add position embeddings for gpt2-
                # style models (those with a separate `wpe` matrix).
                # LLaMA / Qwen / Mistral use rotary embeddings inside
                # attention so the wte-only path works. GPT-2 /
                # Falcon / GPT-NeoX add learned position embeddings
                # explicitly at the embedding stage — without this
                # the layers see token vectors without position
                # info → semantically garbage output (live-attest:
                # "The capital of France is" → " all" instead of
                # " the").
                _wpe = None
                _model = _state["model"]
                if hasattr(_model, "transformer") and hasattr(
                    _model.transformer, "wpe",
                ):
                    _wpe = _model.transformer.wpe
                elif hasattr(_model, "gpt_neox") and hasattr(
                    _model.gpt_neox, "embed_in",
                ) and hasattr(_model.gpt_neox, "wpe"):
                    # GPT-NeoX uses both embed_in (token) + wpe
                    _wpe = _model.gpt_neox.wpe
                if _wpe is not None:
                    seq_len = token_ids.shape[-1]
                    position_ids = _torch.arange(
                        seq_len, device=token_ids.device,
                    )
                    pos_embeds = _wpe(position_ids)
                    embeddings = embeddings + pos_embeds
            return embeddings.cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"build_hf_prompt_encoder encode failed for "
                f"{model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

    return _encode


def _resolve_hf_device(requested: str) -> str:
    """Sprint 617 (Phase 2F-5h) — centralized HF device selection.

    Behavior:
      "cpu"             → "cpu" (always honored — no override)
      "cuda" + CUDA OK  → "cuda"
      "cuda" + no CUDA  → "cpu" + WARNING (operator typo /
                          missing CUDA install / cpu-only host)
      "auto" / ""       → "cuda" if available, else "cpu"
      other             → treat as auto

    torch import failure → "cpu" safe-fallback (no torch means no
    GPU possible; constructing the runner will still surface the
    underlying ImportError at first use).

    Used by HuggingFaceLayerSliceRunner / build_hf_prompt_encoder /
    build_hf_output_decoder so operators flipping
    PRSM_PARALLAX_HF_DEVICE=auto get sane defaults pipeline-wide.
    """
    req = (requested or "auto").lower().strip()
    if req == "cpu":
        return "cpu"
    try:
        import torch as _torch
        cuda_ok = bool(_torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False
    if req == "cuda":
        if cuda_ok:
            return "cuda"
        import logging as _l
        _l.getLogger(__name__).warning(
            "Sprint 617 _resolve_hf_device: device=cuda requested "
            "but torch.cuda.is_available() returned False (no GPU "
            "or torch missing CUDA build). Falling back to cpu."
        )
        return "cpu"
    # auto / empty / unknown → pick best available
    return "cuda" if cuda_ok else "cpu"


def build_hf_output_decoder(
    *,
    model_id: str,
    device: str = "cpu",
) -> Callable[[Any], str]:
    """Sprint 616 (Phase 2F-5g) — client-side HF OutputDecoder.

    Symmetric inverse of sprint-614's prompt_encoder. Returns a
    closure that maps the chain tail's final activation (logits,
    after sprint-613 LM head) to a decoded string.

    Flow on first decoder() call:
      1. Lazy-load HF AutoTokenizer (cached for subsequent calls;
         lighter than sprint-614 which also loads the full model
         for embedding lookup — decoder only needs vocab)
      2. argmax over the vocab dimension at the last sequence
         position → token_id
      3. tokenizer.decode([token_id]) → str

    Logits shape handling:
      [B, S, V] → argmax(logits[..., -1, :]) at last position
      [S, V]    → argmax(logits[-1, :])
      [V]       → argmax(logits)

    Wraps load/decode failures in StageExecutionError.
    """
    if not model_id or not isinstance(model_id, str):
        raise ValueError(
            "build_hf_output_decoder requires model_id"
        )
    del device  # tokenizer is CPU-only; arg kept for API symmetry

    _state: Dict[str, Any] = {"tokenizer": None}

    def _decode(logits: Any) -> str:
        if _state["tokenizer"] is None:
            try:
                import transformers  # noqa: F401
            except ImportError as exc:
                raise StageExecutionError(
                    f"build_hf_output_decoder requires `transformers` "
                    f"installed: {exc}."
                ) from exc
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model_id)
            except Exception as exc:  # noqa: BLE001
                raise StageExecutionError(
                    f"build_hf_output_decoder failed to load "
                    f"tokenizer for {model_id!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            _state["tokenizer"] = tok
        try:
            # Normalize to last-position logits over vocab
            arr = logits
            while arr.ndim > 1:
                arr = arr[-1]
            # arr is now 1D over vocab
            token_id = int(arr.argmax())
            return _state["tokenizer"].decode([token_id])
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"build_hf_output_decoder decode failed for "
                f"{model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

    return _decode


def _resolve_hf_lm_head(hf_model: Any) -> Any:
    """Sprint 613 (Phase 2F-5d) — polymorphic HF LM-head resolution.

    Try known causal-LM head attribute paths:
      .lm_head      LLaMA / Mistral / GPT-2 / Falcon / Qwen
      .embed_out    GPT-NeoX / Pythia

    Returns the first that's truthy. Raises StageExecutionError
    listing attempted paths if neither resolves — operator triages
    by name.
    """
    head = getattr(hf_model, "lm_head", None)
    if head is not None:
        return head
    head = getattr(hf_model, "embed_out", None)
    if head is not None:
        return head
    raise StageExecutionError(
        "HuggingFaceLayerSliceRunner: model does not expose an "
        "LM head. Attempted: .lm_head, .embed_out. Add support "
        "for this architecture in _resolve_hf_lm_head."
    )


def _resolve_hf_rotary_position_embeddings(
    hf_model: Any, hidden: Any, position_ids: Any,
) -> Any:
    """Sprint 1216 — compute model-level rotary ``(cos, sin)`` once.

    transformers v4.4x+/v5 moved rotary-embedding computation OUT of each
    attention sub-layer UP to the model level: ``Qwen2Model.forward`` (and
    LLaMA / Mistral) computes
    ``position_embeddings = self.rotary_emb(hidden, position_ids)`` ONCE
    and threads the tuple to every decoder layer, which then does
    ``cos, sin = position_embeddings`` internally.

    The layer-slice runner drives the decoder layers DIRECTLY, so it must
    supply that tuple; otherwise ``position_embeddings`` defaults to
    ``None`` and the layer's internal unpack raises
    "cannot unpack non-iterable NoneType object" (the 2-A10 Qwen-1.5B
    bench failure).

    Returns the ``(cos, sin)`` tuple for rotary architectures
    (LLaMA / Qwen / Mistral, which expose ``.model.rotary_emb``), else
    ``None`` for GPT-2 / GPT-NeoX (learned absolute / per-layer positions
    → the existing ``position_ids``-only call is correct and unchanged).

    ``cos``/``sin`` depend only on ``position_ids`` (and ``hidden``'s
    dtype/device), not on the hidden VALUES, so computing once before the
    layer loop is correct.
    """
    inner = getattr(hf_model, "model", None)
    if inner is None:
        return None
    rotary_emb = getattr(inner, "rotary_emb", None)
    if rotary_emb is None:
        return None
    return rotary_emb(hidden, position_ids)


class HuggingFaceLayerSliceRunner:
    """Sprint 610 (Phase 2F-5a) — first real-model LayerSliceRunner
    skeleton.

    Constructor accepts a HuggingFace ``model_id`` (e.g.,
    "meta-llama/Llama-3.2-1B") + target device (default "cpu" —
    safe on GPU-less machines).

    Phase 2F-5a is interface only. Phase 2F-5b ships:
      - Lazy ``transformers`` import + model loading from HF Hub
        or local checkpoint
      - Layer extraction + forward pass on the activation tensor
      - Layer-range validation against the model's actual depth
      - TEE attestation hook (Phase 2F-5c)

    Conforms to ``LayerSliceRunner`` Protocol. The ``model`` arg
    in ``run_layer_range`` (which the LayerStageServer passes from
    ``registry.get(model_id)``) is IGNORED by the HF runner — the
    HF runner owns its own model loading via the ``transformers``
    library. ShardedModel-based registry is for PRSM's own
    sharding scheme; HF runner is a parallel path.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
    ) -> None:
        if not model_id or not isinstance(model_id, str):
            raise ValueError(
                "HuggingFaceLayerSliceRunner requires model_id "
                "(HuggingFace model identifier, e.g., "
                "'meta-llama/Llama-3.2-1B')"
            )
        self.model_id = model_id
        # Sprint 617 — auto-detect / safe-fallback device selection.
        self.device = _resolve_hf_device(device)
        # Lazy-cached on first run_layer_range call (full-load path).
        self._hf_model: Any = None
        # sp1223 (Brick 4, adversarial-verify C4) — per-(start,end,is_final)
        # slice-model cache. The runner is shared across calls and a node can
        # host >1 range in a chain, so slice models are keyed PER RANGE (not a
        # single swap). Opt-in via PRSM_PARALLAX_SLICE_LOAD; any loader error or
        # unsupported (gpt2-style) arch falls back to the full-load path.
        self._slice_models: Dict[tuple, Any] = {}

    def _resolve_model_for_range(
        self, start: int, end: int, is_final_stage: bool,
    ) -> tuple:
        """sp1223 (Brick 4) — return ``(model, layers, slice_loaded,
        full_depth)`` for THIS range. Slice-loads (cached per
        (start,end,is_final)) when PRSM_PARALLAX_SLICE_LOAD is on AND the arch
        supports it; otherwise (or on ANY slice-load error) full-loads the
        proven path. Per-call → safe for a shared runner hosting multiple
        ranges."""
        if _slice_load_enabled():
            key = (int(start), int(end), bool(is_final_stage))
            cached = self._slice_models.get(key)
            if cached is not None:
                model, full_depth = cached
                return model, _resolve_hf_layers(model), True, full_depth
            try:
                dtype_str = _resolve_model_dtype(self.device)
                model, full_depth = build_reduced_config_slice_model(
                    self.model_id, start=int(start), end=int(end),
                    is_final_stage=bool(is_final_stage),
                    device=self.device, dtype=dtype_str,
                )
                self._slice_models[key] = (model, full_depth)
                return model, _resolve_hf_layers(model), True, full_depth
            except Exception as exc:  # noqa: BLE001 — fall back to full-load
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "slice-load failed for %s [%s,%s) is_final=%s — falling "
                    "back to full-load: %s",
                    self.model_id, start, end, is_final_stage, exc,
                )
        model = self._ensure_model_loaded()
        layers = _resolve_hf_layers(model)
        return model, layers, False, len(layers)

    def _ensure_model_loaded(self) -> Any:
        """Sprint 611 (Phase 2F-5b) — lazy model load + cache."""
        if self._hf_model is not None:
            return self._hf_model
        # Lazy imports keep transformers off the import path for
        # daemons not using the HF runner.
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner requires `transformers` "
                f"+ `torch` packages installed: {exc}. Install via "
                f"`pip install transformers torch`."
            ) from exc
        try:
            # sp1220 — resolve dtype (fp16 on CUDA fits 7B on 24GB; fp32 on
            # CPU keeps the proven path) + fetch from the process-shared cache.
            dtype_str = _resolve_model_dtype(self.device)
            model = _load_hf_model_cached(self.model_id, self.device, dtype_str)
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner failed to load "
                f"{self.model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc
        self._hf_model = model
        return model

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Any,
        activation: Any,
        privacy_tier: Any,
        is_final_stage: bool,
    ) -> Any:
        """Sprint 611 (Phase 2F-5b) — real HF model forward pass.

        Workflow:
          1. Lazy-load HF model (cached after first call)
          2. Extract layer list (LLaMA-style ``.model.layers``)
          3. Convert numpy activation → torch [B, S, H]
          4. Build position_ids = arange(S)
          5. Forward through layers[start:end]
          6. Convert back → numpy + return LayerSliceResult

        The ``model`` arg (passed from registry.get) is ignored;
        HF runner owns its own model loading.

        Assumes LLaMA-style architecture (.model.layers). Future
        sprints handle GPT-style (.transformer.h) + other variants.
        """
        import time as _time
        import torch as _torch
        from prsm.compute.chain_rpc.server import LayerSliceResult
        from prsm.compute.tee.models import TEEType

        start_t = _time.monotonic()
        start, end = int(layer_range[0]), int(layer_range[1])
        # sp1223 (Brick 4) — resolve the model for THIS range: a slice-loaded
        # skeleton (relative-indexed layers 0..n-1) when PRSM_PARALLAX_SLICE_LOAD
        # is on + the arch supports it, else the full model (absolute indices,
        # the proven path). Per-call → safe for a shared runner / multi-range
        # node.
        hf_model, layers, slice_loaded, full_depth = (
            self._resolve_model_for_range(start, end, is_final_stage)
        )
        if start < 0 or end > full_depth or start >= end:
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner: invalid layer_range "
                f"({start}, {end}) for model with {full_depth} layers"
            )
        original_ndim = activation.ndim
        # sp1220 (adversarial-verify C1) — match the LOADED model's compute
        # dtype, NOT hardcoded fp32. numpy activations default to float64;
        # feeding an fp32/fp64 activation into an fp16 nn.Linear (the whole
        # point of Brick 1's fp16 load) raises "mat1 and mat2 must have the
        # same dtype". Cast at the boundary to the model's dtype (fp32 on the
        # CPU/gpt2 path → unchanged behavior there).
        hidden = _torch.from_numpy(activation).to(
            self.device, dtype=_model_compute_dtype(hf_model),
        )
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        # position_ids = arange(S) broadcast over batch — sufficient
        # for non-cached forward; KV-cache + rolling positions land
        # in a later phase.
        seq_len = hidden.shape[-2] if hidden.dim() >= 2 else 1
        position_ids = _torch.arange(
            seq_len, device=self.device,
        ).unsqueeze(0)
        # Sprint 1216 — rotary models (LLaMA/Qwen/Mistral) need the
        # model-level (cos, sin) tuple threaded into each layer; gpt2
        # returns None here and keeps the position_ids-only call.
        position_embeddings = _resolve_hf_rotary_position_embeddings(
            hf_model, hidden, position_ids,
        )
        layer_kwargs: Dict[str, Any] = {"position_ids": position_ids}
        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings
        try:
            with _torch.no_grad():
                # sp1223 (Brick 4) — a slice-loaded model's `layers` already
                # holds EXACTLY the [start,end) slice at relative indices
                # 0..n-1, so iterate relatively; the full-load path keeps the
                # proven absolute indices. (indexing_decision: relative iff
                # slice_loaded — the only bit-exact-loop edit, guarded.)
                _indices = range(end - start) if slice_loaded else range(start, end)
                for i in _indices:
                    out = layers[i](hidden, **layer_kwargs)
                    # LLaMA layer returns tuple (hidden_state, ...)
                    hidden = out[0] if isinstance(out, tuple) else out
                # Sprint 613 (Phase 2F-5d) — chain tail: apply LM
                # head to produce logits. The OutputDecoder on the
                # client side does argmax/sampling + detokenization.
                if is_final_stage:
                    # Sprint 626 fix — apply final LayerNorm before
                    # lm_head when present. GPT-2 has model.transformer.ln_f;
                    # LLaMA has model.model.norm; both must run AFTER
                    # all transformer blocks but BEFORE lm_head. Without
                    # this, logits are off and argmax returns garbage
                    # (surfaced live-testing gpt2 with "The capital of
                    # France is" → got "age" instead of " Paris").
                    final_ln = None
                    if hasattr(hf_model, "transformer") and hasattr(
                        hf_model.transformer, "ln_f",
                    ):
                        final_ln = hf_model.transformer.ln_f
                    elif hasattr(hf_model, "model") and hasattr(
                        hf_model.model, "norm",
                    ):
                        final_ln = hf_model.model.norm
                    elif hasattr(hf_model, "gpt_neox") and hasattr(
                        hf_model.gpt_neox, "final_layer_norm",
                    ):
                        final_ln = hf_model.gpt_neox.final_layer_norm
                    if final_ln is not None:
                        hidden = final_ln(hidden)
                    lm_head = _resolve_hf_lm_head(hf_model)
                    hidden = lm_head(hidden)
        except StageExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner forward pass failed "
                f"in {self.model_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # Convert back to numpy, restore original ndim.
        if original_ndim == 2:
            hidden = hidden.squeeze(0)
        output_np = hidden.cpu().numpy()
        duration = _time.monotonic() - start_t
        return LayerSliceResult(
            output=output_np,
            duration_seconds=duration,
            tee_attestation=b"hf-runner-software-attestation",
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )

    def run_layer_range_incremental(
        self,
        *,
        model: Any,
        layer_range: Any,
        activation: Any,
        privacy_tier: Any,
        is_final_stage: bool,
        prev_kv_state: Any,
    ) -> Any:
        """Sprint 656 (KV-cache arc piece 3) — incremental forward
        using HF DynamicCache.

        Returns ``(LayerSliceResult, new_kv_state)``:
          - LayerSliceResult: same shape as ``run_layer_range``
          - new_kv_state: the ``DynamicCache`` object after this
            forward — sprint 657's KVCacheManager wiring stores
            it for the next INCREMENTAL call

        When ``prev_kv_state is None``: cold cache — the activation
        carries the FULL prefix's embeddings, we forward all S
        positions, return a freshly-initialized DynamicCache. This
        is equivalent to PREFILL with cache-building.

        When ``prev_kv_state`` is given (a DynamicCache): hot cache —
        the activation should carry ONLY the new token's embedding
        (S=1). cache_position starts at past_seq_len so positional
        encoding aligns with the cached prefix. The Cache object
        is mutated in-place (HF API) AND returned for clarity.
        """
        import time as _time
        import torch as _torch
        from transformers.cache_utils import DynamicCache
        from prsm.compute.chain_rpc.server import LayerSliceResult
        from prsm.compute.tee.models import TEEType

        start_t = _time.monotonic()
        start, end = int(layer_range[0]), int(layer_range[1])
        # sp1223 (Brick 4) — slice-aware model resolution (relative indices when
        # sliced, absolute on the proven full-load path); per-call.
        hf_model, layers, slice_loaded, full_depth = (
            self._resolve_model_for_range(start, end, is_final_stage)
        )
        if start < 0 or end > full_depth or start >= end:
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner: invalid layer_range "
                f"({start}, {end}) for model with {full_depth} layers"
            )
        original_ndim = activation.ndim
        # sp1220 (adversarial-verify C1) — match the LOADED model's compute
        # dtype, NOT hardcoded fp32. numpy activations default to float64;
        # feeding an fp32/fp64 activation into an fp16 nn.Linear (the whole
        # point of Brick 1's fp16 load) raises "mat1 and mat2 must have the
        # same dtype". Cast at the boundary to the model's dtype (fp32 on the
        # CPU/gpt2 path → unchanged behavior there).
        hidden = _torch.from_numpy(activation).to(
            self.device, dtype=_model_compute_dtype(hf_model),
        )
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        # past_seq_len from the cache OR 0 for cold start.
        if prev_kv_state is not None:
            past_seq_len = int(prev_kv_state.get_seq_length())
            cache = prev_kv_state
        else:
            past_seq_len = 0
            cache = DynamicCache()
        seq_len = hidden.shape[-2] if hidden.dim() >= 2 else 1
        # cache_position: which positions THESE new hidden states
        # occupy in the FULL [past + new] sequence.
        cache_position = _torch.arange(
            past_seq_len, past_seq_len + seq_len, device=self.device,
        )
        # Sprint 1216 — rotary models need (cos, sin) computed from the
        # ABSOLUTE positions of the new tokens (= cache_position), threaded
        # into each layer. gpt2 → None (keeps cache_position-only call).
        position_embeddings = _resolve_hf_rotary_position_embeddings(
            hf_model, hidden, cache_position.unsqueeze(0),
        )
        layer_kwargs: Dict[str, Any] = {
            "past_key_values": cache,
            "cache_position": cache_position,
            "use_cache": True,
        }
        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings

        try:
            with _torch.no_grad():
                # sp1223 (Brick 4) — relative indices on a slice-loaded model,
                # absolute on the proven full-load path.
                _indices = range(end - start) if slice_loaded else range(start, end)
                for i in _indices:
                    out = layers[i](hidden, **layer_kwargs)
                    hidden = out[0] if isinstance(out, tuple) else out
                # Final-stage: same ln_f + lm_head as run_layer_range
                if is_final_stage:
                    final_ln = None
                    if hasattr(hf_model, "transformer") and hasattr(
                        hf_model.transformer, "ln_f",
                    ):
                        final_ln = hf_model.transformer.ln_f
                    elif hasattr(hf_model, "model") and hasattr(
                        hf_model.model, "norm",
                    ):
                        final_ln = hf_model.model.norm
                    elif hasattr(hf_model, "gpt_neox") and hasattr(
                        hf_model.gpt_neox, "final_layer_norm",
                    ):
                        final_ln = hf_model.gpt_neox.final_layer_norm
                    if final_ln is not None:
                        hidden = final_ln(hidden)
                    lm_head = _resolve_hf_lm_head(hf_model)
                    hidden = lm_head(hidden)
        except StageExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner incremental forward "
                f"failed in {self.model_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if original_ndim == 2:
            hidden = hidden.squeeze(0)
        output_np = hidden.cpu().numpy()
        duration = _time.monotonic() - start_t
        result = LayerSliceResult(
            output=output_np,
            duration_seconds=duration,
            tee_attestation=b"hf-runner-software-attestation-incremental",
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )
        return (result, cache)


class IdentityLayerSliceRunner:
    """Sprint 608 (Phase 2F-3) — smallest possible LayerSliceRunner.

    Returns the input activation UNCHANGED. Lets the FULL real-model
    server wire (chain_rpc.server.LayerStageServer parses the
    RunLayerSliceRequest, verifies upstream HandoffToken, signs the
    RunLayerSliceResponse) be exercised end-to-end without needing
    actual model weights loaded.

    Analogue to sprint-603 EchoStageExecutor but at the LayerSliceRunner
    layer (deeper in the stack — LayerStageServer.handle still runs
    full parse + sign + validate paths).

    Phase 2F-4+ ships real model-framework runners (HuggingFace
    transformers first probable target).
    """

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Any,
        activation: Any,
        privacy_tier: Any,
        is_final_stage: bool,
    ) -> Any:
        # Lazy imports — keep numpy off the import chain for daemons
        # that never enable the rpc/layer_stage kind.
        from prsm.compute.chain_rpc.server import LayerSliceResult
        from prsm.compute.tee.models import TEEType
        return LayerSliceResult(
            output=activation.copy(),  # defensive copy
            duration_seconds=0.0,
            tee_attestation=b"identity-runner-no-attestation",
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )


def build_layer_stage_server_executor(
    *,
    node: Any,
    runner: Any,
    model_registry_root_env: str = "PRSM_MODEL_REGISTRY_ROOT",
) -> "LayerStageServerStageExecutor":
    """Sprint 607 (Phase 2F-2) — factory for the real-model
    StageExecutor backed by chain_rpc.LayerStageServer.

    Required operator config:
      - ``$PRSM_MODEL_REGISTRY_ROOT`` env var (path to local
        model-registry directory)
      - ``$PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS`` env var
        (sprint 580 _build_anchor_or_none reads this)
      - ``runner`` argument — operator-supplied
        ``LayerSliceRunner`` (Phase 2F-3+ ships specific impls)
      - ``node.identity`` — present after PRSMNode.start

    Raises ``StageExecutionError`` with a structured message naming
    the missing piece when any prerequisite is absent, so operators
    flipping rpc kind on get actionable feedback.

    Lazy imports for the heavy ML deps (FilesystemModelRegistry,
    LayerStageServer, SoftwareTEERuntime) — default operators on
    PRSM_PARALLAX_STAGE_EXECUTOR_KIND=stub never trigger these.
    """
    import os as _os
    root = (_os.environ.get(model_registry_root_env, "") or "").strip()
    if not root:
        raise StageExecutionError(
            f"PRSM_MODEL_REGISTRY_ROOT unset; "
            f"build_layer_stage_server_executor needs the path to "
            f"the local FilesystemModelRegistry root directory."
        )
    from prsm.node.inference_wiring import _build_anchor_or_none
    anchor = _build_anchor_or_none()
    if anchor is None:
        raise StageExecutionError(
            "anchor unavailable (set PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS); "
            "LayerStageServer requires it for upstream-token verification."
        )
    if runner is None or not hasattr(runner, "run_layer_range"):
        raise StageExecutionError(
            "runner is required (LayerSliceRunner with "
            ".run_layer_range method). Phase 2F-3+ ships specific "
            "runner implementations; pass one here."
        )
    if getattr(node, "identity", None) is None:
        raise StageExecutionError(
            "node.identity is None; daemon must be started "
            "(PRSMNode.start) before constructing the LayerStageServer "
            "executor."
        )

    # Lazy imports — only paid when this kind is actually selected.
    from prsm.compute.model_registry.registry import (
        FilesystemModelRegistry,
    )
    from prsm.compute.chain_rpc.server import LayerStageServer
    from prsm.compute.tee.runtime import SoftwareTEERuntime

    registry = FilesystemModelRegistry(root=root, anchor=anchor)
    tee_runtime = SoftwareTEERuntime()

    # Sprint 618 (Phase 2F-5i) — env-tunable KVCacheManager for
    # INCREMENTAL decode_mode. Opt-in via
    # PRSM_PARALLAX_KV_CACHE_ENABLED=1 OR by setting any of the
    # tuning env vars (implicit opt-in, less surprising for ops).
    import os as _os
    _kv_enabled_raw = (_os.environ.get(
        "PRSM_PARALLAX_KV_CACHE_ENABLED", "",
    ) or "").strip()
    _kv_max_raw = (_os.environ.get(
        "PRSM_PARALLAX_KV_CACHE_MAX_REQUESTS", "",
    ) or "").strip()
    _kv_ttl_raw = (_os.environ.get(
        "PRSM_PARALLAX_KV_CACHE_TTL_SECONDS", "",
    ) or "").strip()
    _kv_enabled = (
        _kv_enabled_raw.lower() in ("1", "true", "yes", "on")
        or bool(_kv_max_raw)
        or bool(_kv_ttl_raw)
    )
    kv_cache_manager = None
    if _kv_enabled:
        from prsm.compute.chain_rpc.kv_cache import KVCacheManager
        try:
            max_req = int(_kv_max_raw) if _kv_max_raw else 64
        except ValueError:
            max_req = 64
        try:
            ttl = float(_kv_ttl_raw) if _kv_ttl_raw else 300.0
        except ValueError:
            ttl = 300.0
        kv_cache_manager = KVCacheManager(
            max_cached_requests=max_req, ttl_seconds=ttl,
        )

    # Sprint 692 F41 fix — construct a streaming_runner so
    # LayerStageServer.handle_token_stream can actually produce
    # frames. SyntheticStreamingRunner wraps the existing unary
    # runner: runs full forward, decodes activation → text,
    # chunks into streaming frames. NOT real autoregressive
    # decode (max_tokens is honored by the chunker, not by an
    # actual token-by-token generation loop) — that's
    # AutoregressiveStreamingRunner / sprint 693+ work.
    # Output_decoder reuses the HF-tokenizer wiring from sprint
    # 616. Skipped silently if HF model id missing — server
    # falls back to "no streaming_runner configured" error
    # which sprint 691's adapter surfaces clearly.
    streaming_runner = None
    _hf_id_raw = (_os.environ.get(
        "PRSM_PARALLAX_HF_MODEL_ID", "",
    ) or "").strip()
    _stream_kind = (_os.environ.get(
        "PRSM_PARALLAX_STREAMING_RUNNER_KIND", "",
    ) or "").strip().lower() or "embedder_backed"
    # sp1223 (Brick 4, adversarial-verify streaming guard) — the
    # embedder_backed/synthetic streaming runners call model.generate on the
    # FULL model (runner._ensure_model_loaded). Under slice-load a node holds
    # only its layer slice (and a 7B that NEEDS sharding can't even load the
    # full model), so streaming is unsupported: leave streaming_runner=None so
    # the streaming path surfaces "no streaming_runner configured" rather than
    # OOM'ing or handing generate() a layer slice. Unary inference is unaffected.
    if _hf_id_raw and _slice_load_enabled():
        import logging as _lg
        _lg.getLogger(__name__).warning(
            "Sprint 1223 — streaming runner DISABLED under "
            "PRSM_PARALLAX_SLICE_LOAD (a sliced node cannot run model.generate; "
            "use unary /compute/inference). Model: %s", _hf_id_raw,
        )
        _hf_id_raw = ""  # skip the streaming-runner construction below
    if _hf_id_raw:
        try:
            _hf_device_raw = (_os.environ.get(
                "PRSM_PARALLAX_HF_DEVICE", "",
            ) or "").strip() or "cpu"
            if _stream_kind == "synthetic":
                # Sprint 692 — text-chunk streaming via unary
                # forward + tokenizer decode + whitespace split.
                # Functional but emits ONE forward pass's output
                # chunked into N frames, not true token-by-token.
                from prsm.compute.inference.streaming_runner import (
                    SyntheticStreamingRunner,
                )
                output_decoder = build_hf_output_decoder(
                    model_id=_hf_id_raw, device=_hf_device_raw,
                )
                streaming_runner = SyntheticStreamingRunner(
                    runner=runner, output_decoder=output_decoder,
                )
            else:
                # Sprint 693 — real autoregressive streaming via
                # HF model.generate(inputs_embeds=activation, ...).
                # Bypasses the F42 prompt-registry problem entirely
                # by working from the activation tensor instead of
                # re-tokenizing the original prompt text.
                from prsm.compute.inference.autoregressive_runner import (
                    EmbedderBackedStreamingRunner, SamplingDefaults,
                )
                from prsm.compute.tee.models import TEEType
                # Lazy-load model + tokenizer via the same path
                # the HF runner uses, ensuring single shared cache.
                _model = runner._ensure_model_loaded() if hasattr(
                    runner, "_ensure_model_loaded",
                ) else None
                if _model is None:
                    raise RuntimeError(
                        "embedder_backed streaming kind requires "
                        "the HF layer-slice runner (model cached "
                        "via _ensure_model_loaded)"
                    )
                from transformers import AutoTokenizer
                _tok = AutoTokenizer.from_pretrained(_hf_id_raw)
                # Sprint 694 F43 fix — default to greedy (temperature
                # 0.0) so streaming output is deterministic and
                # matches HF reference byte-for-byte. SamplingDefaults's
                # class-level default is temperature=1.0 (sampling)
                # which produced non-deterministic divergence at the
                # 4th-token boundary in sprint 693's live-attest.
                # Callers can override via request.temperature.
                streaming_runner = EmbedderBackedStreamingRunner(
                    model=_model,
                    tokenizer=_tok,
                    tee_attestation=b"identity-runner-no-attestation",
                    tee_type=TEEType.SOFTWARE,
                    sampling_defaults=SamplingDefaults(temperature=0.0),
                )
        except Exception as exc:  # noqa: BLE001
            import logging as _l
            _l.getLogger(__name__).warning(
                "Sprint 692/693 streaming_runner construction "
                "failed: %s. LayerStageServer.handle_token_stream "
                "will surface 'no streaming_runner configured' on "
                "streaming requests.", exc,
            )
            streaming_runner = None

    server = LayerStageServer(
        identity=node.identity,
        registry=registry,
        runner=runner,
        tee_runtime=tee_runtime,
        anchor=anchor,
        kv_cache_manager=kv_cache_manager,
        streaming_runner=streaming_runner,
    )
    return LayerStageServerStageExecutor(server=server)


def build_echo_stage_executor() -> StageExecutor:
    """Sprint 603 (Phase 2E-3) — diagnostic StageExecutor that
    echoes its input bytes back unchanged.

    NOT a real chain-stage forward pass (no model computation).
    Purpose: end-to-end wire testing of the Phase 2 client +
    server round-trip without requiring a model integration.

    Phase 2E-4 (sprint 604) wires this in behind an env var
    (PRSM_PARALLAX_STAGE_EXECUTOR_KIND=echo) for operator opt-in
    fleet diagnostics. Default stays at sprint-602 stub (raises)
    so production behavior isn't silently masked.

    Phase 2F+ ships real-model StageExecutor variants.
    """

    class _EchoStageExecutor:
        async def execute(self, request_bytes: bytes) -> bytes:
            return request_bytes

    return _EchoStageExecutor()


def _build_layer_slice_runner_from_env() -> Any:
    """Sprint 609 (Phase 2F-4) — env-driven LayerSliceRunner selection.

    Read PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND:
      - unset / "identity" → IdentityLayerSliceRunner (sprint 608,
        passthrough for testing). Default keeps testing-mode
        accessible without extra config.
      - anything else      → identity + WARNING (future: huggingface,
        torch, etc., shipped as Phase 2F-5+ runners).
    """
    import os as _os
    kind = (_os.environ.get(
        "PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND", "",
    ) or "").strip().lower() or "identity"
    if kind == "identity":
        return IdentityLayerSliceRunner()
    if kind == "huggingface":
        # Sprint 611 Phase 2F-5b — real HF transformer runner.
        # Operator MUST set PRSM_PARALLAX_HF_MODEL_ID to a valid HF
        # model identifier (e.g., "meta-llama/Llama-3.2-1B").
        import os as _os
        model_id = (_os.environ.get(
            "PRSM_PARALLAX_HF_MODEL_ID", "",
        ) or "").strip()
        if not model_id:
            import logging as _l
            _l.getLogger(__name__).warning(
                "Sprint 611 layer-slice-runner: huggingface kind "
                "requires PRSM_PARALLAX_HF_MODEL_ID env. Falling "
                "back to identity."
            )
            return IdentityLayerSliceRunner()
        device = (_os.environ.get(
            "PRSM_PARALLAX_HF_DEVICE", "",
        ) or "").strip() or "cpu"
        return HuggingFaceLayerSliceRunner(
            model_id=model_id, device=device,
        )
    import logging as _l
    _l.getLogger(__name__).warning(
        "Sprint 609/611 layer-slice-runner selector: "
        "PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=%r unknown; falling "
        "back to identity. Valid: identity, huggingface.",
        kind,
    )
    return IdentityLayerSliceRunner()


def _build_stage_executor_from_env(node: Any = None) -> StageExecutor:
    """Sprint 604 (Phase 2E-4) — env-driven StageExecutor selection.
    Sprint 609 (Phase 2F-4) — extended with ``layer_stage`` kind.

    Read PRSM_PARALLAX_STAGE_EXECUTOR_KIND:
      - unset / "stub"   → build_stub_stage_executor (raises on
        execute(); production default — operators opt in explicitly).
      - "echo"           → build_echo_stage_executor (returns bytes
        unchanged; for end-to-end wire testing only).
      - "layer_stage"    → build_layer_stage_server_executor with
        runner from PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND. Real-model
        server-side path. Falls back to stub if factory raises
        StageExecutionError (missing env, missing anchor, etc.) with
        structured warning so operator sees the gap.
      - anything else    → stub fallback + structured warning.

    The optional ``node`` arg threads through for ``layer_stage``
    kind (needed for identity + downstream wiring). When None,
    layer_stage falls back to stub (sprint 604 callers pre-2F-4
    may not pass node).
    """
    import os as _os
    kind_raw = _os.environ.get(
        "PRSM_PARALLAX_STAGE_EXECUTOR_KIND", "",
    ).strip().lower()
    kind = kind_raw or "stub"
    if kind == "echo":
        return build_echo_stage_executor()
    if kind == "stub":
        return build_stub_stage_executor()
    if kind == "layer_stage":
        if node is None:
            import logging as _l
            _l.getLogger(__name__).warning(
                "Sprint 609 stage-executor selector: "
                "PRSM_PARALLAX_STAGE_EXECUTOR_KIND=layer_stage but "
                "the env-driven path is being called without a node "
                "arg (legacy caller pre-Phase 2F-4). Falling back "
                "to stub. Production callers in sprint-604 thread "
                "node through."
            )
            return build_stub_stage_executor()
        runner = _build_layer_slice_runner_from_env()
        try:
            return build_layer_stage_server_executor(
                node=node, runner=runner,
            )
        except StageExecutionError as exc:
            import logging as _l
            _l.getLogger(__name__).warning(
                "Sprint 609 stage-executor selector: "
                "build_layer_stage_server_executor raised %s. "
                "Falling back to stub.",
                exc,
            )
            return build_stub_stage_executor()
    # Unknown — warn + fall back to stub.
    import logging as _l
    _l.getLogger(__name__).warning(
        "Sprint 604/609 stage-executor selector: "
        "PRSM_PARALLAX_STAGE_EXECUTOR_KIND=%r unknown; falling "
        "back to stub. Valid: stub, echo, layer_stage.",
        kind_raw,
    )
    return build_stub_stage_executor()


async def handle_chain_executor_request(node: Any, msg: Any) -> bool:
    """Sprint 601 (Phase 2E-1) — server-side request handler.
    Sprint 604 (Phase 2E-4) — delegates execution to a StageExecutor.

    Receives a chain_executor_rpc REQUEST message (CHAIN_REQ_KEY set,
    CHAIN_RESP_KEY absent), decodes the base64 payload, hands the
    bytes to a ``StageExecutor`` (selected via
    ``PRSM_PARALLAX_STAGE_EXECUTOR_KIND`` env var), then ships the
    result back as a response message.

    Returns:
      True  — handled (response sent or attempted)
      False — not for us (wrong subtype, response message)

    Stage-executor selection:
      - ``node._chain_stage_executor`` attr if present (test
        injection / future production wiring)
      - else ``_build_stage_executor_from_env()`` (env-driven)

    Defensive semantics:
      - Wrong subtype → False
      - Has CHAIN_RESP_KEY → False (sprint 597 handles those)
      - Malformed base64 payload → CHAIN_ERROR_KEY response
      - StageExecutionError raised → CHAIN_ERROR_KEY response w/ msg
      - send_to_peer raises → log + return True (we tried)
    """
    import base64
    payload = getattr(msg, "payload", None) or {}
    if payload.get("subtype") != CHAIN_MSG_TYPE:
        return False
    request_id = payload.get(CHAIN_REQ_KEY)
    if not request_id:
        return False
    if payload.get(CHAIN_RESP_KEY):
        return False

    sender_id = getattr(msg, "sender_id", None)
    if not sender_id:
        return False

    # Sprint 726 F59 fix: per-peer concurrent unary-request cap
    # (mirror of sprint-723 F56 on the streaming path). Pre-726,
    # one peer could open unlimited concurrent CHAIN_REQs against
    # this server → each held an in-flight executor.execute()
    # coroutine + (for real autoregressive runner) KV cache →
    # trivial memory-DoS. Bookkeeping lives on
    # `node._chain_executor_serving_unary_by_peer` (peer_id →
    # active count). Incremented on accepted REQ; decremented in
    # finally so every code path (decode-fail / exec-fail / send-
    # fail / clean) releases the slot.
    _unary_per_peer_cap = _resolve_per_peer_unary_concurrency()
    _serving_unary = getattr(
        node, "_chain_executor_serving_unary_by_peer", None,
    )
    # MagicMock test fixtures auto-create attributes that aren't
    # dicts — narrow the type check to isinstance(dict) so we
    # always end up with a real dict for arithmetic.
    if not isinstance(_serving_unary, dict):
        _serving_unary = {}
        try:
            node._chain_executor_serving_unary_by_peer = _serving_unary
        except Exception:  # noqa: BLE001 — defensive
            pass
    if _unary_per_peer_cap > 0:
        active = _serving_unary.get(sender_id, 0)
        if active >= _unary_per_peer_cap:
            from prsm.node.transport import P2PMessage, MSG_DIRECT
            resp = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=getattr(
                    getattr(node, "identity", None), "node_id", "",
                ),
                payload={
                    "subtype": CHAIN_MSG_TYPE,
                    CHAIN_REQ_KEY: request_id,
                    CHAIN_RESP_KEY: request_id,
                    CHAIN_ERROR_KEY: (
                        f"peer concurrent unary-request limit "
                        f"exceeded ({active} >= "
                        f"{_unary_per_peer_cap}); tune via "
                        f"PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY"
                    ),
                    CHAIN_PAYLOAD_KEY: "",
                },
            )
            try:
                await node.transport.send_to_peer(sender_id, resp)
            except Exception:  # noqa: BLE001
                pass
            return True
    _serving_unary[sender_id] = _serving_unary.get(sender_id, 0) + 1
    try:
        return await _handle_chain_executor_request_body(
            node, msg, request_id, sender_id, payload,
        )
    finally:
        new_count = _serving_unary.get(sender_id, 0) - 1
        if new_count <= 0:
            _serving_unary.pop(sender_id, None)
        else:
            _serving_unary[sender_id] = new_count


async def _handle_chain_executor_request_body(
    node: Any, msg: Any, request_id: Any, sender_id: Any, payload: Any,
) -> bool:
    """Sprint 726 — body of `handle_chain_executor_request` extracted
    so the per-peer concurrency counter can wrap it in a single
    try/finally without losing readability. Every original return
    path returns here; the wrapper decrements the counter on exit.
    Same refactor pattern as sprint-723 on the streaming side.
    """
    import base64

    # Decode incoming payload bytes.
    # Sprint 725 F58 fix: pre-decode size gate (mirror of sprint-721
    # F55 on the streaming path). A malicious peer sending a 100MB
    # base64 string would otherwise force ~75MB allocation on the
    # b64decode step → easy memory-DoS on 2GB-RAM operator droplets.
    decode_error = None
    payload_bytes: bytes = b""
    _unary_max_bytes = _resolve_unary_request_max_bytes()
    payload_b64 = payload.get(CHAIN_PAYLOAD_KEY, "")
    if _unary_max_bytes > 0:
        # base64 expands by ~4/3; check len * 3 / 4 against limit.
        estimated_decoded = (len(payload_b64) * 3) // 4
        if estimated_decoded > _unary_max_bytes:
            decode_error = (
                f"chain-executor request payload exceeds max bytes "
                f"({estimated_decoded} > {_unary_max_bytes}); tune "
                f"via PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES"
            )
    if decode_error is None:
        try:
            payload_bytes = base64.b64decode(payload_b64)
        except Exception as exc:  # noqa: BLE001
            decode_error = (
                f"chain-executor request payload base64-decode "
                f"failed: {type(exc).__name__}: {exc}"
            )
        else:
            # Sprint 725 defense-in-depth: post-decode size check.
            if (
                _unary_max_bytes > 0
                and len(payload_bytes) > _unary_max_bytes
            ):
                decoded_len = len(payload_bytes)
                payload_bytes = b""  # free immediately
                decode_error = (
                    f"chain-executor decoded payload exceeds max "
                    f"bytes ({decoded_len} > {_unary_max_bytes}); "
                    f"tune via PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES"
                )

    # Stage execution (skip if we already have a decode error).
    # Sprint 728 F61 fix: bound execution wall-clock so a hung
    # executor doesn't hold the sprint-726 per-peer cap slot
    # indefinitely. Defaults to 60s; operators tune via
    # PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S. Timeout converts to a
    # CHAIN_ERROR_KEY response naming the env var so operators
    # know what to tune.
    response_bytes: bytes = b""
    exec_error: str | None = decode_error
    if exec_error is None:
        executor: StageExecutor = getattr(
            node, "_chain_stage_executor", None,
        ) or _build_stage_executor_from_env(node=node)
        _exec_timeout = _resolve_unary_execution_timeout()
        try:
            if _exec_timeout > 0:
                import asyncio as _asyncio
                response_bytes = await _asyncio.wait_for(
                    executor.execute(payload_bytes),
                    timeout=_exec_timeout,
                )
            else:
                response_bytes = await executor.execute(payload_bytes)
        except StageExecutionError as exc:
            exec_error = (
                f"stage-executor raised StageExecutionError: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            # asyncio.TimeoutError is a subclass of Exception. Detect
            # by name so we ship an actionable error that names the
            # env var (so operators reading logs know to tune).
            exc_name = type(exc).__name__
            if exc_name in ("TimeoutError", "CancelledError"):
                exec_error = (
                    f"stage-executor exceeded execution timeout of "
                    f"{_exec_timeout}s; tune via "
                    f"PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S"
                )
            else:
                exec_error = (
                    f"stage-executor raised {exc_name}: {exc}"
                )

    # Build + ship response.
    try:
        from prsm.node.transport import P2PMessage, MSG_DIRECT
        if exec_error is not None:
            resp_payload = {
                "subtype": CHAIN_MSG_TYPE,
                CHAIN_REQ_KEY: request_id,
                CHAIN_RESP_KEY: request_id,
                CHAIN_ERROR_KEY: exec_error,
                CHAIN_PAYLOAD_KEY: "",
            }
        else:
            resp_payload = {
                "subtype": CHAIN_MSG_TYPE,
                CHAIN_REQ_KEY: request_id,
                CHAIN_RESP_KEY: request_id,
                CHAIN_PAYLOAD_KEY: base64.b64encode(
                    response_bytes,
                ).decode("ascii"),
            }
        response = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=getattr(
                getattr(node, "identity", None), "node_id", "",
            ),
            payload=resp_payload,
        )
        await node.transport.send_to_peer(sender_id, response)
    except Exception as exc:  # noqa: BLE001
        import logging as _l
        _l.getLogger(__name__).warning(
            "Sprint 601/604 chain-executor request handler: "
            "failed to send response to %s: %s",
            sender_id, exc,
        )
    return True


def handle_chain_executor_response(node: Any, msg: Any) -> bool:
    """Sprint 597 (Phase 2D step 3) — resolve a pending chain-executor
    Future on receipt of a response message.

    Reads the response wire-protocol fields from ``msg.payload``:
      ``CHAIN_RESP_KEY``     → request_id (hex sha256 string)
      ``CHAIN_PAYLOAD_KEY``  → base64-encoded response bytes

    If the request_id is in ``node._chain_executor_pending``, sets
    the Future result. If not (request already timed out + popped,
    or unsolicited message), returns False without raising.

    Must be called from the loop thread (via the transport's normal
    inbound-message dispatch); the Future is on that loop. Use
    ``loop.call_soon_threadsafe`` if invoking from another thread.

    Returns ``True`` if the response was matched + Future resolved;
    ``False`` if no matching pending request (silent drop).
    """
    import base64
    payload = getattr(msg, "payload", None) or {}
    # Only handle our wire-protocol envelope; ignore other subtypes
    if payload.get("subtype") != CHAIN_MSG_TYPE:
        return False
    request_id = payload.get(CHAIN_RESP_KEY)
    if not request_id:
        return False
    pending = getattr(node, "_chain_executor_pending", None)
    if pending is None:
        return False
    entry = pending.get(request_id)
    if entry is None:
        # No pending request (timed out + cleaned up). Silent drop.
        return False

    # Sprint 727 F60 fix: bind incoming response's sender to the
    # expected dispatch peer. Pre-727, pending[request_id] stored
    # just the future — any peer who learned the request_id could
    # forge a CHAIN_RESP and resolve the victim's future with
    # attacker bytes. Same fix-class as sprint-719 F53 on
    # streaming. Post-727, pending stores (future, expected_sender);
    # responses from any other sender are silently dropped (return
    # False so the genuine response can still resolve the future
    # when it arrives).
    if isinstance(entry, tuple) and len(entry) == 2:
        future, expected_sender = entry
        sender_id = getattr(msg, "sender_id", None)
        if expected_sender and sender_id and sender_id != expected_sender:
            return False  # forged response from wrong peer — drop
    else:  # legacy bare-future shape (defensive — should never hit)
        future = entry

    if future.done():
        # Already resolved (duplicate response). Silent drop.
        return False
    # Sprint 630 — CHAIN_ERROR_KEY propagation. The server-side
    # request handler sets this field (with an empty CHAIN_PAYLOAD_KEY)
    # when the stage executor raised. Without this branch the handler
    # below would base64-decode the empty payload to b"" and resolve
    # the Future with empty bytes — sprint 624 hit this and saw
    # silent "size_bytes=0" instead of a useful error. Treat any
    # non-empty error string as the canonical signal, regardless of
    # whether CHAIN_PAYLOAD_KEY is also present.
    error_msg = payload.get(CHAIN_ERROR_KEY, "")
    if error_msg:
        future.set_exception(StageExecutionError(error_msg))
        return True
    payload_b64 = payload.get(CHAIN_PAYLOAD_KEY, "")
    try:
        response_bytes = base64.b64decode(payload_b64)
    except Exception as exc:  # noqa: BLE001
        # Malformed payload → reject this response. Future stays
        # pending so the original SendMessage call still times out
        # cleanly rather than getting bogus bytes.
        future.set_exception(
            RuntimeError(
                f"chain-executor response payload base64-decode "
                f"failed for request_id={request_id}: {exc}"
            )
        )
        return True
    future.set_result(response_bytes)
    return True


def _resolve_peer_address(node: Any, peer_id: str) -> Optional[str]:
    """Sprint 679 — look up a peer's network address by peer_id.
    Used by the chain-exec-ping adapter when send_to_peer fails
    silently and we want to auto-redial before giving up.

    Returns ``host:port`` string or None if unknown.

    Sources, in priority order:
      1. node.transport.peers[peer_id] — currently-known connection
         info (may have a host:port already even if the WS is dead)
      2. node.discovery.known_peers[peer_id] — bootstrap-mediated
         peer registry (PeerInfo.address)
    """
    try:
        transport_peers = getattr(node, "transport", None)
        if transport_peers is not None:
            peer = getattr(transport_peers, "peers", {}).get(peer_id)
            if peer is not None:
                addr = getattr(peer, "address", None)
                if addr:
                    return addr
    except Exception:  # noqa: BLE001
        pass
    try:
        discovery = getattr(node, "discovery", None)
        if discovery is not None:
            info = getattr(discovery, "known_peers", {}).get(peer_id)
            if info is not None:
                addr = getattr(info, "address", None)
                if addr:
                    return addr
    except Exception:  # noqa: BLE001
        pass
    return None


def build_send_message_adapter(
    node: Any,
    timeout: float = 30.0,
) -> SendMessageAdapter:
    """Sprint 596 (Phase 2D step 2) — real SendMessage adapter.

    Bridges the sync ``SendMessage = Callable[[str, bytes], bytes]``
    contract over the daemon's async transport. Workflow:

      1. Hash request_bytes → request_id (sha256 hex).
      2. Resolve stage_address → peer_id via build_address_resolver-
         compatible lookup against ``node.transport.peers``.
      3. Create asyncio.Future, store in
         ``node._chain_executor_pending[request_id]``.
      4. Schedule coroutine on ``node._loop`` that sends a P2PMessage
         + awaits the future (resolved by sprint-597 response handler).
      5. Return response bytes synchronously.

    Phase 2D step 3 (sprint 597) wires the response handler that
    resolves the Future when a CHAIN_RESP_KEY message arrives.
    Until that lands, calls will time out — but the wire is sound.

    Raises:
      _Phase2AdapterNotReady — when node._loop is None (daemon
        not started or not running on asyncio context).
      PeerNotFound — when stage_address isn't in transport.peers.
      TimeoutError — when no response arrives within ``timeout``.
    """
    import base64
    import hashlib

    def _adapter(stage_address: str, request_bytes: bytes) -> bytes:
        loop = getattr(node, "_loop", None)
        if loop is None:
            raise _Phase2AdapterNotReady(
                "Sprint 596 SendMessage adapter: node._loop is None. "
                "Daemon must be started (sprint-595 captures the loop "
                "at PRSMNode.start). Until then, set "
                "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=stub for a working "
                "daemon."
            )

        # Sprint 687 F34 fix — self-dispatch shortcut. When Phase-1
        # allocation routes a stage to the local node (which IS the
        # expected case for any 2-node deployment serving its own
        # requests), the dispatcher must NOT try to look up
        # transport.peers[self] — self is never in that dict.
        # Execute the stage locally via the same StageExecutor the
        # server-side handle_chain_executor_request uses.
        self_node_id = getattr(
            getattr(node, "identity", None), "node_id", None,
        )
        if self_node_id is not None and stage_address == self_node_id:
            executor: StageExecutor = getattr(
                node, "_chain_stage_executor", None,
            ) or _build_stage_executor_from_env(node=node)

            async def _local_execute() -> bytes:
                return await executor.execute(request_bytes)

            future = asyncio.run_coroutine_threadsafe(
                _local_execute(), loop,
            )
            return future.result(timeout=timeout)

        # Sprint 724 F57 fix: same fix-class as sprint-718 F52 on
        # the streaming side. Pre-724 derivation was purely
        # deterministic — two concurrent identical unary requests
        # (idempotent retry from coordinator, multi-stage chain
        # re-dispatch on transient failure) computed the same
        # request_id; second `pending[request_id] = future`
        # overwrote the first's future, then the first dispatch
        # waited 30s for a response that resolved into the wrong
        # future and timed out. Mix 8 bytes os.urandom to make
        # collision negligible while preserving wire-id length.
        import os as _os
        request_id = hashlib.sha256(
            request_bytes + b":unary:" + _os.urandom(8),
        ).hexdigest()
        pending = node._chain_executor_pending

        async def _send_and_wait() -> bytes:
            # Import here to avoid circular: transport imports node
            # indirectly via __init__
            from prsm.node.transport import P2PMessage, MSG_DIRECT
            future = loop.create_future()
            # Sprint 727 F60 fix: bind expected_sender (stage_address
            # = the peer we dispatched to) so handle_chain_executor_
            # response can verify msg.sender_id matches before
            # resolving the future. Pre-727, response handler
            # routed by request_id alone — any peer that learned
            # the id (network observation, pre-724 deterministic
            # derivation, etc.) could forge CHAIN_RESP and resolve
            # the victim's future with attacker bytes BEFORE the
            # legitimate response arrived. Same fix-class as
            # sprint 719's F53 on streaming.
            pending[request_id] = (future, stage_address)
            try:
                msg = P2PMessage(
                    msg_type=MSG_DIRECT,
                    sender_id=node.identity.node_id,
                    payload={
                        "subtype": CHAIN_MSG_TYPE,
                        CHAIN_REQ_KEY: request_id,
                        CHAIN_PAYLOAD_KEY: base64.b64encode(
                            request_bytes,
                        ).decode("ascii"),
                    },
                )
                # Resolve address → peer_id by lookup. The chain
                # executor passes stage_address that downstream code
                # treats as a peer_id (per sprint 593 resolver
                # contract). For Phase 2D the convention is:
                # stage_address IS the target peer_id.
                sent = await node.transport.send_to_peer(
                    stage_address, msg,
                )
                if not sent:
                    # Sprint 679 — auto-redial on silent WS drop.
                    # Long inference waits (gpt2 cold-load on a fresh
                    # peer can take 25-30s) can outlast the peer's
                    # WS idle timeout, causing send_to_peer to fail
                    # silently on the next request. Look up the
                    # peer's address from known_peers + reconnect.
                    address = _resolve_peer_address(node, stage_address)
                    if address is not None:
                        # Sprint 679 — evict stale peer first so
                        # connect_to_peer does a FRESH WS dial
                        # rather than returning the dead-but-cached
                        # PeerConnection from transport.peers.
                        try:
                            transport_peers = getattr(
                                node.transport, "peers", None,
                            )
                            if transport_peers is not None and (
                                stage_address in transport_peers
                            ):
                                stale = transport_peers.pop(
                                    stage_address, None,
                                )
                                # Try to close the WS cleanly so
                                # the OS releases the FD.
                                if stale is not None:
                                    close = getattr(stale, "close", None)
                                    if callable(close):
                                        try:
                                            result = close()
                                            if hasattr(result, "__await__"):
                                                await result
                                        except Exception:  # noqa: BLE001
                                            pass
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            reconnected = await node.transport.connect_to_peer(
                                address,
                            )
                        except Exception as exc:  # noqa: BLE001
                            reconnected = None
                            import logging as _l
                            _l.getLogger(__name__).debug(
                                "Sprint 679 redial of %s @ %s raised: %s",
                                stage_address, address, exc,
                            )
                        if reconnected is not None:
                            import logging as _l
                            _l.getLogger(__name__).info(
                                "Sprint 679 redial succeeded for "
                                "peer_id=%s @ %s; retrying send",
                                stage_address, address,
                            )
                            sent = await node.transport.send_to_peer(
                                stage_address, msg,
                            )
                    if not sent:
                        raise RuntimeError(
                            f"transport.send_to_peer returned False for "
                            f"peer_id={stage_address!r}; chain stage "
                            f"unreachable (sprint 679 redial attempt "
                            f"also failed or peer address unknown)"
                        )
                import asyncio as _asyncio
                response_payload = await _asyncio.wait_for(
                    future, timeout=timeout,
                )
                return response_payload
            finally:
                pending.pop(request_id, None)

        return run_async_on_loop(loop, _send_and_wait(), timeout=timeout + 1.0)

    return _adapter


def build_token_stream_send_message_adapter(node: Any) -> Callable[
    [str, bytes], Any,
]:
    """Sprint 691 + 711 — token-stream transport adapter.

    Wires the chain executor's ``token_stream_send_message`` slot
    so ``execute_chain_streaming`` can dispatch streaming requests
    both locally (sprint 691 self-dispatch) AND remotely (sprint
    711 cross-host streaming via the CHAIN_STREAM_MSG_TYPE wire
    protocol).

    For self-dispatch:
      - Resolves the local StageExecutor via _build_stage_executor_
        from_env (or node._chain_stage_executor injection).
      - If the executor's underlying ._server has handle_token_stream
        (LayerStageServer with streaming_runner wired), call it
        and yield each frame.
      - Else raises StageExecutionError naming the gap so operators
        see WHY streaming doesn't work on their config.

    For remote (stage_address != self_node_id):
      - Delegates to ``_remote_token_stream_dispatch`` which sends
        ONE CHAIN_STREAM_REQ message, iterates incoming
        CHAIN_STREAM_FRAME messages keyed by stream_id, and
        terminates on CHAIN_STREAM_END (with optional error
        propagation). Closes the audit-doc §7.3 known limit.
    """
    def _stream(stage_address: str, request_bytes: bytes) -> Any:
        self_node_id = getattr(
            getattr(node, "identity", None), "node_id", None,
        )
        # Sprint 711 F40 — remote dispatch via P2P stream RPC.
        # The CHAIN_STREAM_MSG_TYPE wire protocol carries one
        # STREAM_REQ message + multiple STREAM_FRAME messages +
        # one terminal STREAM_END. The frame ordering is by
        # CHAIN_STREAM_SEQ_KEY (not arrival time) so out-of-order
        # delivery on the underlying transport doesn't corrupt
        # the stream. See module-top comment for full schema.
        if self_node_id is None or stage_address != self_node_id:
            yield from _remote_token_stream_dispatch(
                node, stage_address, request_bytes,
            )
            return

        # Self-dispatch — call the local executor's streaming side
        executor: StageExecutor = getattr(
            node, "_chain_stage_executor", None,
        ) or _build_stage_executor_from_env(node=node)
        server = getattr(executor, "_server", None)
        if server is None or not hasattr(server, "handle_token_stream"):
            raise StageExecutionError(
                "token_stream_send_message: local stage executor "
                "lacks .handle_token_stream — set "
                "PRSM_PARALLAX_STAGE_EXECUTOR_KIND=layer_stage + "
                "ensure the runner exposes "
                "run_layer_slice_streaming (sprint 691 surface)."
            )
        # handle_token_stream returns Iterable[bytes] — yield each frame
        for frame in server.handle_token_stream(request_bytes):
            yield frame

    return _stream


def _remote_token_stream_dispatch(
    node: Any, stage_address: str, request_bytes: bytes,
) -> Any:
    """Sprint 711 F40 — remote-side token-stream client.

    Sends ONE CHAIN_STREAM_REQ to the stage's peer over MSG_DIRECT,
    then iterates incoming CHAIN_STREAM_FRAME messages keyed by
    stream_id, ordered by sequence number. Yields each frame as
    bytes. Terminal CHAIN_STREAM_END message closes the iterator;
    if it carries CHAIN_ERROR_KEY, raises StageExecutionError so
    the caller surfaces the failure cleanly.

    Threading: the chain executor's send_message contract is sync;
    this function is called from a worker thread (sprint 687 F35
    fix runs `chain_executor.execute_chain_streaming` via
    `run_in_executor`). We use `asyncio.run_coroutine_threadsafe`
    to schedule the send + use `node._chain_executor_pending_streams`
    (a per-request asyncio.Queue) to receive frames.
    """
    import asyncio as _asyncio
    import base64
    import hashlib

    loop = getattr(node, "_loop", None)
    if loop is None:
        raise _Phase2AdapterNotReady(
            "Sprint 711 remote streaming: node._loop is None. "
            "Daemon must be started before remote token-stream "
            "dispatch."
        )

    # Sprint 718 F52 fix: stream_id MUST include per-invocation
    # entropy. The pre-718 derivation `sha256(request_bytes + ":stream")`
    # was purely deterministic — concurrent identical requests (e.g.,
    # an idempotent retry from a coordinator) computed the same
    # stream_id, the second invocation's `pending[stream_id] = queue`
    # silently OVERWROTE the first's queue, and the first dispatcher
    # then got the second's frames (or nothing at all on EOF). 8
    # bytes of os.urandom is enough to make collision negligible
    # (~birthday bound at 2^32 in-flight streams) while keeping the
    # id length identical to the pre-718 sha256 hex output.
    import os as _os
    stream_id = hashlib.sha256(
        request_bytes + b":stream:" + _os.urandom(8),
    ).hexdigest()
    # Per-stream queue for incoming frames + end-signal.
    pending = getattr(node, "_chain_executor_pending_streams", None)
    if pending is None:
        pending = {}
        try:
            node._chain_executor_pending_streams = pending
        except Exception:  # noqa: BLE001 — defensive; some test mocks
            pass

    maxsize = _resolve_stream_queue_maxsize()
    queue: _asyncio.Queue = _asyncio.run_coroutine_threadsafe(
        _make_async_queue(loop, maxsize), loop,
    ).result(timeout=5.0)
    # Sprint 719 F53 fix: bind the registered queue to the
    # expected sender. handle_chain_stream_response now verifies
    # msg.sender_id == expected_sender BEFORE routing frames, so a
    # malicious or buggy third peer that learns the stream_id
    # cannot forge frames into the victim's queue. Pre-719,
    # pending[stream_id] was just the queue → any peer with the
    # stream_id could inject frames.
    pending[stream_id] = (queue, stage_address)

    async def _send_request() -> None:
        from prsm.node.transport import P2PMessage, MSG_DIRECT
        msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=node.identity.node_id,
            payload={
                "subtype": CHAIN_STREAM_MSG_TYPE,
                CHAIN_STREAM_REQ_KEY: stream_id,
                CHAIN_PAYLOAD_KEY: base64.b64encode(
                    request_bytes,
                ).decode("ascii"),
            },
        )
        await node.transport.send_to_peer(stage_address, msg)

    # Fire-and-forget send (response frames arrive via gossip handler)
    _asyncio.run_coroutine_threadsafe(_send_request(), loop)

    # Frame loop. Each iteration awaits the next queued frame. The
    # terminal entry is a sentinel dict carrying CHAIN_STREAM_END_KEY
    # + optional CHAIN_ERROR_KEY.
    try:
        while True:
            entry = _asyncio.run_coroutine_threadsafe(
                queue.get(), loop,
            ).result(timeout=300.0)
            if isinstance(entry, dict) and entry.get("end"):
                err = entry.get("error")
                if err:
                    raise StageExecutionError(
                        f"remote token-stream error from "
                        f"{stage_address[:16]}: {err}"
                    )
                return
            # Frame bytes
            yield entry
    finally:
        pending.pop(stream_id, None)


def _resolve_unary_execution_timeout() -> float:
    """Sprint 728 F61 — max wall-clock seconds for executor.execute()
    on a single unary CHAIN_REQ. Defends against an executor that
    hangs (model load deadlock, infinite loop, blocked network
    call) holding the per-peer cap slot indefinitely + tying up
    the requester's pending future.

    Default 60s (covers gpt2 inference on CPU with 100-token
    prompt; tune up for slower models / larger inputs). Operators
    tune via PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S.
    Values <=0 mean no timeout (pre-728 behavior).
    Non-float safely defaults to 60.0.
    """
    import os
    raw = os.environ.get("PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S")
    if raw is None or raw == "":
        return 60.0
    try:
        val = float(raw)
    except ValueError:
        return 60.0
    if val <= 0:
        return 0.0  # disabled
    return val


def _resolve_unary_request_max_bytes() -> int:
    """Sprint 725 F58 — max request_bytes accepted by
    handle_chain_executor_request. Mirrors sprint-721 F55 on the
    unary path. Defends against a malicious peer sending an
    enormous CHAIN_REQ payload to exhaust server memory on the
    b64-decode step.

    Default 16 MiB (matches the streaming limit; sized for a real
    gpt2 activation blob with 100-token prompt + metadata).
    Operators tune via PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES.
    Values <=0 mean unbounded; non-int safely defaults to 16 MiB.
    """
    import os
    raw = os.environ.get("PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES")
    default_bytes = 16 * 1024 * 1024  # 16 MiB
    if raw is None or raw == "":
        return default_bytes
    try:
        val = int(raw)
    except ValueError:
        return default_bytes
    if val <= 0:
        return 0  # unbounded
    return val


def _resolve_per_peer_unary_concurrency() -> int:
    """Sprint 726 F59 — max concurrent unary CHAIN_REQs a single
    peer can have in-flight against this server. Mirrors sprint-
    723 F56 on the unary path. Defends against a peer opening
    thousands of CHAIN_REQs to exhaust server memory + execution
    slots.

    Default 8 (same as streaming cap; covers realistic multi-
    request coordinator workloads). Operators tune via
    PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY. Values <=0 mean
    unbounded; non-int safely defaults to 8.
    """
    import os
    raw = os.environ.get("PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY")
    if raw is None or raw == "":
        return 8
    try:
        val = int(raw)
    except ValueError:
        return 8
    if val <= 0:
        return 0
    return val


def _resolve_per_peer_stream_concurrency() -> int:
    """Sprint 723 F56 — max concurrent streams a single peer can
    have in-flight against this server. Defends against a malicious
    or buggy peer opening thousands of CHAIN_STREAM_REQs to exhaust
    server memory + GPU.

    Default 8 (covers realistic multi-stream workloads — e.g., a
    coordinator running 4 parallel inferences with 2-stage chain
    each = 8 concurrent server-side streams per peer). Operators
    tune via PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY.
    Values <=0 mean unbounded (pre-723 behavior — disabled gate).
    Non-int safely defaults to 8.
    """
    import os
    raw = os.environ.get("PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY")
    if raw is None or raw == "":
        return 8
    try:
        val = int(raw)
    except ValueError:
        return 8
    if val <= 0:
        return 0  # unbounded
    return val


def _resolve_stream_request_max_bytes() -> int:
    """Sprint 721 F55 — max request_bytes accepted by
    handle_chain_stream_request. Defends against a malicious peer
    sending an enormous CHAIN_STREAM_REQ payload to exhaust server
    memory on the b64-decode step.

    Default 16 MiB (an honest gpt2 activation blob with 100-token
    prompt is ~4 MB; 16 MB covers larger contexts and metadata
    overhead). Operators tune via PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES.
    Values <=0 mean unbounded (pre-721 behavior — disabled gate).
    Non-int values default to 16 MiB (safe rather than failing
    streaming setup mysteriously).
    """
    import os
    raw = os.environ.get("PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES")
    default_bytes = 16 * 1024 * 1024  # 16 MiB
    if raw is None or raw == "":
        return default_bytes
    try:
        val = int(raw)
    except ValueError:
        return default_bytes
    if val <= 0:
        return 0  # unbounded
    return val


def _resolve_stream_execution_timeout() -> float:
    """Sprint 729 F62 — max wall-clock seconds per server-side
    streaming response. Defends against a slow or malicious
    StageExecutor that yields frames at a glacial rate (or pauses
    indefinitely BETWEEN yields), holding the sprint-723 per-peer
    cap slot longer than any honest stream should need.

    Default 300s (5 minutes — covers gpt2 streaming a 200-token
    response at CPU rate ~1.5s/token). Operators tune via
    PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S. Values <=0 mean no
    timeout. Non-float safely defaults to 300s.

    Note: this bounds CUMULATIVE wall-time across the stream. It
    does NOT protect against a single `__next__()` call that hangs
    indefinitely (that would block the event loop and require a
    thread-offload refactor — different attack surface).
    """
    import os
    raw = os.environ.get("PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S")
    if raw is None or raw == "":
        return 300.0
    try:
        val = float(raw)
    except ValueError:
        return 300.0
    if val <= 0:
        return 0.0
    return val


def _resolve_stream_queue_maxsize() -> int:
    """Sprint 713 — bounded receive-queue size for streaming
    back-pressure. Default 64 frames (chosen as 2× typical inference
    chunk-count; high enough not to stall fast streams under normal
    load, low enough to bound memory if the consumer hangs).

    Operators can override via PRSM_CHAIN_STREAM_QUEUE_MAXSIZE.
    Values <= 0 mean unbounded (pre-713 behavior — disabled gate).
    Typos / non-int values default to 64 (safe rather than failing
    streaming setup mysteriously).
    """
    import os
    raw = os.environ.get("PRSM_CHAIN_STREAM_QUEUE_MAXSIZE")
    if raw is None or raw == "":
        return 64
    try:
        val = int(raw)
    except ValueError:
        return 64
    if val <= 0:
        return 0  # unbounded
    return val


async def _make_async_queue(loop, maxsize: int = 0):
    """Construct an asyncio.Queue on the given loop. Wrapped because
    Queue() must be instantiated FROM the loop's thread.

    Sprint 713: `maxsize` param adds bounded back-pressure. 0 =
    unbounded (pre-713 behavior); >0 = back-pressure kicks in when
    queue fills, blocking new put_nowait calls with QueueFull.
    """
    import asyncio as _asyncio
    return _asyncio.Queue(maxsize=maxsize)


async def handle_chain_stream_request(node: Any, msg: Any) -> bool:
    """Sprint 711 F40 — server-side token-stream request handler.

    Mirrors `handle_chain_executor_request` but yields multiple
    CHAIN_STREAM_FRAME response messages instead of one CHAIN_RESP.

    Returns:
      True  — handled (frames sent + terminal STREAM_END dispatched)
      False — not for us (wrong subtype, frame/end message)
    """
    import base64
    payload = getattr(msg, "payload", None) or {}
    if payload.get("subtype") != CHAIN_STREAM_MSG_TYPE:
        return False
    stream_id = payload.get(CHAIN_STREAM_REQ_KEY)
    if not stream_id:
        # Not a request — could be a frame/end message destined for
        # the response handler. Defer to that.
        return False
    sender_id = getattr(msg, "sender_id", None)
    if not sender_id:
        return False

    from prsm.node.transport import P2PMessage, MSG_DIRECT

    # Sprint 723 F56 fix: per-peer concurrent-stream cap. Pre-723,
    # one peer could open unlimited CHAIN_STREAM_REQs against this
    # server, accumulating asyncio.Queues + in-flight generator
    # state + (for real autoregressive runner) KV cache per stream
    # → trivial memory-exhaustion DoS. Bookkeeping lives on
    # `node._chain_executor_serving_streams_by_peer` (peer_id →
    # active count). Incremented on accepted REQ; decremented in
    # the finally block below so every code path (clean / error /
    # peer-disconnect) releases the slot.
    _per_peer_cap = _resolve_per_peer_stream_concurrency()
    _serving = getattr(
        node, "_chain_executor_serving_streams_by_peer", None,
    )
    # isinstance(dict) check — see sprint 726 lesson: MagicMock
    # auto-children pass `is None` checks but aren't dicts.
    if not isinstance(_serving, dict):
        _serving = {}
        try:
            node._chain_executor_serving_streams_by_peer = _serving
        except Exception:  # noqa: BLE001 — defensive
            pass
    if _per_peer_cap > 0:
        active = _serving.get(sender_id, 0)
        if active >= _per_peer_cap:
            end_msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=node.identity.node_id,
                payload={
                    "subtype": CHAIN_STREAM_MSG_TYPE,
                    CHAIN_STREAM_END_KEY: stream_id,
                    CHAIN_STREAM_SEQ_KEY: 0,
                    CHAIN_ERROR_KEY: (
                        f"peer concurrent stream limit exceeded "
                        f"({active} >= {_per_peer_cap}); tune via "
                        f"PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY"
                    ),
                },
            )
            try:
                await node.transport.send_to_peer(sender_id, end_msg)
            except Exception:  # noqa: BLE001
                pass
            return True
    # Accepted — increment the peer's active count. Decremented in
    # the outer try/finally below (wraps EVERY return-True path so
    # the counter never leaks even on size-gate / decode-fail /
    # disconnect / exception).
    _serving[sender_id] = _serving.get(sender_id, 0) + 1
    try:
        return await _handle_stream_request_body(
            node, msg, stream_id, sender_id, payload,
        )
    finally:
        # Decrement; clean up the key if it goes to zero to avoid
        # unbounded dict growth from many distinct one-off peers.
        new_count = _serving.get(sender_id, 0) - 1
        if new_count <= 0:
            _serving.pop(sender_id, None)
        else:
            _serving[sender_id] = new_count


async def _handle_stream_request_body(
    node: Any, msg: Any, stream_id: Any, sender_id: Any, payload: Any,
) -> bool:
    """Sprint 723 — body of `handle_chain_stream_request` extracted
    so the per-peer concurrency counter can wrap it in a single
    try/finally without losing readability of the original control
    flow. Every original return-True path continues to return True
    here; the wrapper decrements the counter on exit."""
    import base64
    from prsm.node.transport import P2PMessage, MSG_DIRECT

    # Sprint 721 F55 fix: bound request_bytes BEFORE the b64-decode
    # step. A malicious peer sending a 100MB base64 string would
    # otherwise allocate ~75MB on decode + hand it to
    # handle_token_stream — easy memory-exhaustion DoS on 2GB
    # droplets. Check encoded length first (cheap), then decoded
    # length (defense-in-depth). Both checks fire terminal STREAM_END
    # with a clear error so the requester closes cleanly.
    _max_bytes = _resolve_stream_request_max_bytes()
    payload_b64 = payload.get(CHAIN_PAYLOAD_KEY, "")
    if _max_bytes > 0:
        # base64 expands by ~4/3; check len * 3 / 4 against limit.
        estimated_decoded = (len(payload_b64) * 3) // 4
        if estimated_decoded > _max_bytes:
            end_msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=node.identity.node_id,
                payload={
                    "subtype": CHAIN_STREAM_MSG_TYPE,
                    CHAIN_STREAM_END_KEY: stream_id,
                    CHAIN_STREAM_SEQ_KEY: 0,
                    CHAIN_ERROR_KEY: (
                        f"request payload exceeds max bytes "
                        f"({estimated_decoded} > {_max_bytes}); "
                        f"tune via PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"
                    ),
                },
            )
            try:
                await node.transport.send_to_peer(sender_id, end_msg)
            except Exception:  # noqa: BLE001
                pass
            return True

    try:
        request_bytes = base64.b64decode(payload_b64)
    except Exception as exc:  # noqa: BLE001
        # Malformed request — send terminal STREAM_END with error.
        end_msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=node.identity.node_id,
            payload={
                "subtype": CHAIN_STREAM_MSG_TYPE,
                CHAIN_STREAM_END_KEY: stream_id,
                CHAIN_STREAM_SEQ_KEY: 0,
                CHAIN_ERROR_KEY: (
                    f"payload decode failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            },
        )
        try:
            await node.transport.send_to_peer(sender_id, end_msg)
        except Exception:  # noqa: BLE001
            pass
        return True
    # Sprint 721 defense-in-depth: post-decode size check. The
    # pre-decode estimate is over-cautious (base64 expands ~4/3)
    # but the decoded length is the real memory cost. Pin both.
    if _max_bytes > 0 and len(request_bytes) > _max_bytes:
        decoded_len = len(request_bytes)
        # Free the bytes immediately — we know we won't process
        # them. Helps the GC reclaim memory before send_to_peer's
        # await suspension point.
        del request_bytes
        end_msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=node.identity.node_id,
            payload={
                "subtype": CHAIN_STREAM_MSG_TYPE,
                CHAIN_STREAM_END_KEY: stream_id,
                CHAIN_STREAM_SEQ_KEY: 0,
                CHAIN_ERROR_KEY: (
                    f"decoded payload exceeds max bytes "
                    f"({decoded_len} > {_max_bytes}); "
                    f"tune via PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"
                ),
            },
        )
        try:
            await node.transport.send_to_peer(sender_id, end_msg)
        except Exception:  # noqa: BLE001
            pass
        return True

    # Resolve the local executor + handle_token_stream surface.
    executor = getattr(
        node, "_chain_stage_executor", None,
    ) or _build_stage_executor_from_env(node=node)
    server = getattr(executor, "_server", None)
    if server is None or not hasattr(server, "handle_token_stream"):
        end_msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=node.identity.node_id,
            payload={
                "subtype": CHAIN_STREAM_MSG_TYPE,
                CHAIN_STREAM_END_KEY: stream_id,
                CHAIN_STREAM_SEQ_KEY: 0,
                CHAIN_ERROR_KEY: (
                    "no streaming-capable stage executor on remote — "
                    "remote must run PRSM_PARALLAX_STAGE_EXECUTOR_KIND="
                    "layer_stage with a streaming_runner wired"
                ),
            },
        )
        await node.transport.send_to_peer(sender_id, end_msg)
        return True

    # Iterate the server's stream + ship one CHAIN_STREAM_FRAME per
    # yielded chunk. Terminal STREAM_END with seq=frame_count closes.
    #
    # Sprint 720 F54 fix: check send_to_peer return value + break on
    # False (requester disconnected). Pre-720, the loop kept calling
    # `server.handle_token_stream(...)` after the peer was gone — for
    # a real autoregressive runner that's burning GPU on tokens that
    # are discarded. Closing the iterator with `.close()` (if it's a
    # generator) releases server-side resources (KV cache, model
    # context). For non-generator iterables, the early break still
    # bounds wasted work.
    #
    # Sprint 729 F62 fix: total-wall-time budget. A slow or
    # malicious StageExecutor that yields slowly would otherwise
    # hold the sprint-723 per-peer cap slot beyond any honest use
    # case. Bound via wall-clock check on each iteration.
    import time as _time
    seq = 0
    stream_iter = server.handle_token_stream(request_bytes)
    _stream_timeout = _resolve_stream_execution_timeout()
    _stream_start = _time.monotonic()
    try:
        for frame in stream_iter:
            if (
                _stream_timeout > 0
                and (_time.monotonic() - _stream_start) > _stream_timeout
            ):
                # Time budget exceeded — close iter + ship terminal
                # STREAM_END with actionable error referencing the
                # env var.
                _close_fn = getattr(stream_iter, "close", None)
                if callable(_close_fn):
                    try:
                        _close_fn()
                    except Exception:  # noqa: BLE001
                        pass
                end_msg = P2PMessage(
                    msg_type=MSG_DIRECT,
                    sender_id=node.identity.node_id,
                    payload={
                        "subtype": CHAIN_STREAM_MSG_TYPE,
                        CHAIN_STREAM_END_KEY: stream_id,
                        CHAIN_STREAM_SEQ_KEY: seq,
                        CHAIN_ERROR_KEY: (
                            f"stream execution exceeded total "
                            f"wall-time budget of {_stream_timeout}s; "
                            f"tune via "
                            f"PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S"
                        ),
                    },
                )
                try:
                    await node.transport.send_to_peer(
                        sender_id, end_msg,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return True
            frame_msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=node.identity.node_id,
                payload={
                    "subtype": CHAIN_STREAM_MSG_TYPE,
                    CHAIN_STREAM_FRAME_KEY: stream_id,
                    CHAIN_STREAM_SEQ_KEY: seq,
                    CHAIN_PAYLOAD_KEY: base64.b64encode(
                        frame,
                    ).decode("ascii"),
                },
            )
            send_ok = await node.transport.send_to_peer(
                sender_id, frame_msg,
            )
            if not send_ok:
                # Requester disconnected mid-stream — stop burning
                # compute on output nobody will receive. Close the
                # underlying generator if possible so server-side
                # state (KV cache, etc.) is released promptly.
                _close_fn = getattr(stream_iter, "close", None)
                if callable(_close_fn):
                    try:
                        _close_fn()
                    except Exception:  # noqa: BLE001 — defensive
                        pass
                return True
            seq += 1
        # Clean terminal
        end_msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=node.identity.node_id,
            payload={
                "subtype": CHAIN_STREAM_MSG_TYPE,
                CHAIN_STREAM_END_KEY: stream_id,
                CHAIN_STREAM_SEQ_KEY: seq,
            },
        )
        await node.transport.send_to_peer(sender_id, end_msg)
    except Exception as exc:  # noqa: BLE001
        # Mid-stream failure — ship terminal with error so the
        # requester closes cleanly.
        end_msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=node.identity.node_id,
            payload={
                "subtype": CHAIN_STREAM_MSG_TYPE,
                CHAIN_STREAM_END_KEY: stream_id,
                CHAIN_STREAM_SEQ_KEY: seq,
                CHAIN_ERROR_KEY: (
                    f"{type(exc).__name__}: {exc}"
                ),
            },
        )
        try:
            await node.transport.send_to_peer(sender_id, end_msg)
        except Exception:  # noqa: BLE001
            pass
    return True


def handle_chain_stream_response(node: Any, msg: Any) -> bool:
    """Sprint 711 F40 — client-side token-stream frame/end handler.

    Routes incoming CHAIN_STREAM_FRAME + CHAIN_STREAM_END messages
    to the per-stream asyncio.Queue created by
    `_remote_token_stream_dispatch`. Returns True if the message
    was a stream-response (handled), False otherwise.

    Sprint 713: when the receive queue is bounded (default 64;
    `PRSM_CHAIN_STREAM_QUEUE_MAXSIZE`), this scheduler hands off to
    an async coroutine that AWAITS `queue.put(...)`. That awaits if
    the queue is full → producer side blocks → real back-pressure.
    """
    import base64
    payload = getattr(msg, "payload", None) or {}
    if payload.get("subtype") != CHAIN_STREAM_MSG_TYPE:
        return False
    # Either a frame for an in-flight stream OR a terminal end.
    stream_id = (
        payload.get(CHAIN_STREAM_FRAME_KEY)
        or payload.get(CHAIN_STREAM_END_KEY)
    )
    if not stream_id:
        return False  # request-side message; not for us
    pending = getattr(node, "_chain_executor_pending_streams", {}) or {}
    entry = pending.get(stream_id)
    if entry is None:
        return False  # unknown stream (race; requester gave up)

    # Sprint 719 F53 fix: bind incoming frame's sender to the
    # expected dispatch peer. Pre-719, pending[stream_id] stored
    # just the queue — any peer that learned the stream_id could
    # forge frames into the victim's pending queue (P2P networks
    # are open by default, so a peered third party could connect
    # + inject). Post-719, pending stores (queue, expected_sender);
    # frames from any other sender are silently dropped (return
    # False so other dispatchers can claim them if needed; never
    # let them into the queue).
    if isinstance(entry, tuple) and len(entry) == 2:
        queue, expected_sender = entry
        sender_id = getattr(msg, "sender_id", None)
        if expected_sender and sender_id and sender_id != expected_sender:
            return False  # forged frame from wrong peer — drop
    else:  # legacy shape (defensive — should never hit post-719)
        queue = entry

    loop = getattr(node, "_loop", None)
    if loop is None:
        return False

    if payload.get(CHAIN_STREAM_END_KEY):
        # Terminal entry — pass through end + optional error.
        # Sprint 715 F50 fix: END must use the SAME scheduling
        # primitive as frames (queue.put coroutine via
        # run_coroutine_threadsafe) so wire order is preserved.
        # The original sprint-713 put_nowait-for-END created a race:
        # call_soon callbacks run before deferred tasks, so a quick
        # END could land in the queue ahead of the frame puts that
        # preceded it on the wire → dispatcher saw END first, yielded
        # zero frames. END no longer "bypasses" back-pressure — it
        # waits with the frames. Cannot deadlock because if the queue
        # is full the producer side is also blocked at its puts (so
        # neither side advances), and the consumer eventually drains.
        end_entry = {
            "end": True,
            "error": payload.get(CHAIN_ERROR_KEY),
        }
        import asyncio as _asyncio
        async def _put_end():
            await queue.put(end_entry)
        try:
            _asyncio.run_coroutine_threadsafe(_put_end(), loop)
        except Exception:  # noqa: BLE001 — defensive fallback
            try:
                loop.call_soon_threadsafe(queue.put_nowait, end_entry)
            except Exception:  # noqa: BLE001
                pass
        return True

    # Mid-stream frame — decode + enqueue (with back-pressure).
    # The wire spec says frames arrive in-order over the same
    # MSG_DIRECT connection. If transport later allows reordering,
    # this is where we'd buffer + sort by CHAIN_STREAM_SEQ_KEY.
    try:
        frame_bytes = base64.b64decode(
            payload.get(CHAIN_PAYLOAD_KEY, ""),
        )
    except Exception:  # noqa: BLE001
        # Malformed frame — treat as terminal error so caller closes.
        # Sprint 716 F51 fix: same race as sprint-715 F50. The
        # put_nowait callback would run before any preceding
        # queue.put coroutines, so a malformed-frame error could
        # land in the queue ahead of valid frames that were
        # awaiting the consumer drain. Use the same scheduling
        # primitive as good frames so wire order is preserved.
        err_end_entry = {
            "end": True,
            "error": "malformed frame payload (base64 decode failed)",
        }
        import asyncio as _asyncio
        async def _put_malformed_end():
            await queue.put(err_end_entry)
        try:
            _asyncio.run_coroutine_threadsafe(_put_malformed_end(), loop)
        except Exception:  # noqa: BLE001 — defensive fallback
            try:
                loop.call_soon_threadsafe(
                    queue.put_nowait, err_end_entry,
                )
            except Exception:  # noqa: BLE001
                pass
        return True

    # Sprint 713: bounded-queue back-pressure. Schedule an AWAIT-put
    # via run_coroutine_threadsafe so we block at the producer side
    # rather than dropping frames on QueueFull.
    import asyncio as _asyncio
    async def _put_with_backpressure():
        await queue.put(frame_bytes)
    try:
        _asyncio.run_coroutine_threadsafe(_put_with_backpressure(), loop)
    except Exception:  # noqa: BLE001 — defensive fallback
        try:
            loop.call_soon_threadsafe(queue.put_nowait, frame_bytes)
        except Exception:  # noqa: BLE001
            pass
    return True
