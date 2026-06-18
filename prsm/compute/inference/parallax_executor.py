"""
ParallaxScheduledExecutor — Phase 3.x.6 Task 6.

Implements the ``InferenceExecutor`` Protocol on top of the vendored
Parallax scheduler + the four PRSM trust adapters (Phase 3.x.6 Task 5).

Composition (per design plan §3.2):

  1. Gather current GPU pool via ``gpu_pool_provider()``.
  2. ``TrustStack.filter_pool`` — anchor-verify (Adapter A).
  3. Phase-1 ``allocate_across_regions`` (cached per model+pool).
  4. Per-request: ``TrustStack.filter_for_request`` — tier gate (Adapter B).
  5. Build ``RequestRouter`` over the (allocation, profile_source, pool)
     where the profile_source is already wrapped by Adapter C.
  6. ``router.route(...)`` → ``GPUChain``.
  7. Hand the chain to an injected ``ChainExecutor`` for actual layer
     dispatch; receive ``(output, duration, tee_attestation, tee_type,
     epsilon_spent)``.
  8. ``ConsensusMismatchHook`` post-route: if sampled, route + execute
     an alternate chain and compare outputs (Adapter D).
  9. Sign ``InferenceReceipt`` under the executor's ``NodeIdentity``.

Phase-1 cache invalidation (paper §3.4):
  - Cache hit: cached allocation reused IFF every stage in the cached
    pipelines is still present in the current trust-filtered pool. This
    is the "localized GPU-leave absorbed" case.
  - Cache miss / coverage gap: rebuild the allocation, retry the route
    once. A second NoCoverageError is a routing failure, not a cache
    issue, and surfaces as ``InferenceResult.failure``.

Drop-in replacement: callers wiring the orchestrator to either
``TensorParallelInferenceExecutor`` or ``ParallaxScheduledExecutor`` get
the same ``execute / estimate_cost / supported_models`` surface.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import (
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from prsm.compute.inference.executor import (
    InferenceExecutor,
    UnsupportedModelError,
)
from prsm.compute.inference.models import (
    ContentTier,
    InferenceReceipt,
    InferenceRequest,
    InferenceResult,
)
from prsm.compute.inference.receipt import sign_receipt
from prsm.compute.parallax_scheduling.model_info import ModelInfo
from prsm.compute.parallax_scheduling.prsm_request_router import (
    BudgetExceededError,
    EmptyAllocationError,
    GPUChain,
    NoCoverageError,
    RegionNotFoundError,
    RequestRouter,
    RouteRequest,
)
from prsm.compute.parallax_scheduling.prsm_types import (
    AllocationError,
    AllocationResult,
    EmptyPoolError,
    InsufficientCapacityError,
    ParallaxGPU,
    allocate_across_regions,
)
from prsm.compute.parallax_scheduling.trust_adapter import (
    ChallengeRecord,
    TierGateRejected,
    TrustStack,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# ChainExecutor Protocol — abstracts actual layer dispatch
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChainExecutionResult:
    """What a ``ChainExecutor`` returns to the scheduler.

    Fields:
      output                  Generated text.
      duration_seconds        Wall-clock time for the chain dispatch.
      tee_attestation         Attestation bytes from the chain head.
      tee_type                Effective TEE type (worst case across stages).
      epsilon_spent           DP epsilon recorded by the privacy budget.
      activation_noise_trace  (sprint 413) Optional sprint-295 DP trace
                              when the chain executor invoked per-stage
                              noise injection. Threaded into the
                              InferenceReceipt by the scheduler.
      topology_assignment     (sprint 413) Optional sprint-296 chain-
                              rotation assignment hash. Same wiring
                              path as activation_noise_trace.

    The two privacy fields default to None so pre-sprint-413
    ChainExecutors that don't populate them continue to work
    unchanged — pure-additive contract.
    """

    output: str
    duration_seconds: float
    tee_attestation: bytes
    tee_type: TEEType
    epsilon_spent: float
    activation_noise_trace: Optional[Any] = None
    topology_assignment: Optional[Any] = None
    # Sprint 778 — partial-completion marker (Vision §4.5). Set by
    # preemption-aware chain executors when the inference runner
    # stops mid-stream (sprint-772 detector flag flipped, timeout,
    # or runtime error). None by default; pre-778 executors that
    # don't populate it produce byte-identical signed receipts.
    partial_completion: Optional[Any] = None
    # Sprint 1110 (Domain-03 F1/F2, brick 4) — the per-stage SIGNED activation chain
    # (StageActivationChain) the chain executor assembled from each worker's proof, when
    # every stage supplied one. Threaded into the receipt by _build_signed_receipt.
    stage_activation_chain: Optional[Any] = None
    # Sprint 1156 (on-chain per-stage payee arc, brick 2) — the per-node
    # NodeSignatureMaterial dict (keyed by node_id) the chain executor assembled from
    # each worker's emitted challenge-defensible settlement leaf signature, when EVERY
    # stage supplied one (else None → single-payee fallback). This is the INPUT brick 1's
    # per_stage_settlement_split.split_receipt_to_per_node_batched_receipts consumes; it
    # is carried OUT-OF-BAND (each per-node sig is self-securing, so it need NOT be in the
    # signed receipt payload). Brick 4 (per-node commit) is the runtime/multi-node
    # follow-on that calls the splitter with this material; it is NOT wired here.
    per_stage_settlement_signatures: Optional[Any] = None


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.8.1 — streaming events
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InferenceTokenEvent:
    """Per-token event yielded by ``execute_streaming``. Mirrors the
    chain-RPC layer's ``StreamToken`` user-visible fields without
    exposing internal wire-protocol types — the SSE endpoint encodes
    these directly into ``event: token`` frames.

    The terminal token of a stream has ``finish_reason`` non-None;
    the executor's streaming generator follows it with exactly one
    ``InferenceResult`` (success or failure)."""

    sequence_index: int
    text_delta: str
    token_id: Optional[int] = None
    finish_reason: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────
# Pre-execute gate result — shared between execute() and execute_streaming
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _GateOutcome:
    """Result of ``_pre_execute_gates``. Either ``failure`` is
    populated (a gate rejected the request) or ``chain``/``cost``/
    ``allocation``/``tier_filtered``/``model_info`` are populated
    (request is ready to dispatch).

    Phase 3.x.8.1 Task 1 — extracted from ``execute()`` so both the
    unary and streaming paths share identical pre-execute semantics
    without code duplication. Behavior of ``execute()`` is unchanged.
    """

    failure: Optional[InferenceResult] = None
    # On success: the chain to dispatch + the gate-locked cost +
    # the allocation/tier-filtered context for the consensus check.
    chain: Optional[GPUChain] = None
    cost: Optional[Decimal] = None
    allocation: Optional[AllocationResult] = None
    tier_filtered: Optional[Sequence[ParallaxGPU]] = None
    model_info: Optional[ModelInfo] = None


class ChainExecutor(Protocol):
    """Adapter: given a chosen ``GPUChain``, run inference end-to-end.

    Production: wraps ``TensorParallelExecutor.execute_parallel(...)``
    over the chain's stages with cross-host RPC. Tests: a fake that
    returns deterministic output without spinning real shards.
    """

    def execute_chain(
        self, *, request: InferenceRequest, chain: GPUChain
    ) -> ChainExecutionResult:
        ...


# ──────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────


GpuPoolProvider = Callable[[], Sequence[ParallaxGPU]]
"""Callable that returns the current candidate GPU pool. Called once
per ``execute()`` invocation. Production wires this to the live
``ProfileDHT`` membership snapshot."""


# ──────────────────────────────────────────────────────────────────────────
# ParallaxScheduledExecutor
# ──────────────────────────────────────────────────────────────────────────


class ParallaxScheduledExecutor(InferenceExecutor):
    """``InferenceExecutor`` Protocol implementation backed by the
    vendored Parallax scheduler + PRSM trust stack.

    Constructor args:
      gpu_pool_provider      Callable returning the current pool.
      trust_stack            Composed ``TrustStack`` (Phase 3.x.6 Task 5).
      model_catalog          Mapping ``model_id → ModelInfo`` for any
                             model this executor advertises.
      chain_executor         Concrete chain-dispatch adapter. Production
                             callers use ``make_rpc_chain_executor(...)``
                             from ``prsm.compute.chain_rpc`` (Phase
                             3.x.7) which wires the cross-host RPC
                             path. Tests + dev paths inject a fake
                             that satisfies the ``ChainExecutor``
                             Protocol.
      node_identity          ``NodeIdentity`` used to sign receipts
                             (this node = the settling/serving node).
      cost_per_layer         Base FTNS price per model layer for the
                             roofline cost estimate. Default 0.01.
      privacy_overhead       Per-PrivacyLevel cost multipliers. Default
                             matches Phase 3.x.1 schedule (1.00 / 1.10
                             / 1.25 / 1.50 for NONE/STANDARD/HIGH/MAX).
      allow_partial_regions  Forwarded to ``allocate_across_regions``.
                             False (default) raises if any region has
                             insufficient capacity. True skips them.
    """

    DEFAULT_PRIVACY_OVERHEAD: Mapping[PrivacyLevel, Decimal] = {
        PrivacyLevel.NONE: Decimal("1.00"),
        PrivacyLevel.STANDARD: Decimal("1.10"),
        PrivacyLevel.HIGH: Decimal("1.25"),
        PrivacyLevel.MAXIMUM: Decimal("1.50"),
    }

    def __init__(
        self,
        *,
        gpu_pool_provider: GpuPoolProvider,
        trust_stack: TrustStack,
        model_catalog: Mapping[str, ModelInfo],
        chain_executor: ChainExecutor,
        node_identity: NodeIdentity,
        cost_per_layer: Decimal = Decimal("0.01"),
        privacy_overhead: Optional[Mapping[PrivacyLevel, Decimal]] = None,
        allow_partial_regions: bool = False,
        tier_c_chain_executor: Optional[Any] = None,
        tier_c_speculation_enabled: bool = False,
    ) -> None:
        if gpu_pool_provider is None or not callable(gpu_pool_provider):
            raise RuntimeError(
                "ParallaxScheduledExecutor requires a callable gpu_pool_provider"
            )
        if trust_stack is None:
            raise RuntimeError(
                "ParallaxScheduledExecutor requires a trust_stack"
            )
        if chain_executor is None or not hasattr(chain_executor, "execute_chain"):
            raise RuntimeError(
                "ParallaxScheduledExecutor requires a chain_executor with "
                ".execute_chain(...) → ChainExecutionResult"
            )
        if node_identity is None or not hasattr(node_identity, "node_id"):
            raise RuntimeError(
                "ParallaxScheduledExecutor requires a NodeIdentity for "
                "receipt signing"
            )
        # Phase 3.x.11.q — Tier C chain executor is the constant-time
        # routing target for Tier C streaming requests. When None,
        # Tier C streaming surfaces a structured failure (operator
        # opt-in: only Tier A/B streaming is enabled by default).
        # The decorator MUST expose execute_chain_streaming; we
        # enforce that at construction so a misconfig is caught
        # before any request lands.
        if (
            tier_c_chain_executor is not None
            and not hasattr(
                tier_c_chain_executor, "execute_chain_streaming",
            )
        ):
            raise RuntimeError(
                "ParallaxScheduledExecutor: tier_c_chain_executor "
                "must expose execute_chain_streaming(request=, chain=) "
                "— wrap RpcChainExecutor in a Phase 3.x.11.q decorator "
                "(BatchedTrailingShardedExecutor / "
                "FixedRateShardedExecutor) or use "
                "make_tier_c_sharded_executor(...)"
            )

        self._pool_provider = gpu_pool_provider
        self._trust = trust_stack
        self._catalog = dict(model_catalog)
        self._chain_executor = chain_executor
        self._tier_c_chain_executor = tier_c_chain_executor
        # Phase 3.x.11.q.y — operator opt-in for speculation +
        # Tier C composition. When False (default), Tier C requests
        # with temperature > 0 still surface a structured failure
        # (speculation under Tier C is denied unless this is True
        # AND the wired tier_c_chain_executor is speculation-capable
        # with encrypted_probs_cipher + flat_k_mode + the tail's
        # constant_k_commitment). When True, the routing layer
        # forwards Tier C + temp>0 to tier_c_chain_executor without
        # blocking.
        self._tier_c_speculation_enabled = bool(tier_c_speculation_enabled)
        self._identity = node_identity
        self._cost_per_layer = Decimal(cost_per_layer)
        self._privacy_overhead = dict(
            privacy_overhead or self.DEFAULT_PRIVACY_OVERHEAD
        )
        self._allow_partial_regions = allow_partial_regions

        # Sprint 811 — output cache for deterministic
        # (privacy_tier=none) repeat inferences. Default None;
        # operators opt in by setting `executor._output_cache = OutputCache(...)`
        # at construction time. See sprint 810's resolve_output_cache_from_env.
        self._output_cache: Optional[Any] = None

        # Phase-1 allocation cache:
        #   model_id → (AllocationResult, frozenset_of_stage_node_ids)
        # Localized GPU-leave is absorbed when the cached stage-set is
        # still a subset of the trust-filtered pool's node-ids; otherwise
        # the cache is invalidated and Phase-1 recomputes.
        self._alloc_cache: dict[str, Tuple[AllocationResult, frozenset]] = {}

        # Public-facing recompute counter — operators monitor this for
        # DHT churn / pool-instability signals. Tests assert on it to
        # verify §3.4 behavior.
        self._phase1_recompute_count: int = 0

    # ── InferenceExecutor surface ─────────────────────────────────────

    def supported_models(self) -> list[str]:
        return list(self._catalog.keys())

    async def estimate_cost(self, request: InferenceRequest) -> Decimal:
        """Roofline cost — scales linearly with layer count and the
        per-tier overhead. Independent of pool composition; the actual
        chain may execute faster or slower but the price is fixed at
        request time so callers can pre-validate budgets."""
        if request.model_id not in self._catalog:
            raise UnsupportedModelError(f"Unknown model_id: {request.model_id}")
        model_info = self._catalog[request.model_id]
        base = self._cost_per_layer * Decimal(model_info.num_layers)
        overhead = self._privacy_overhead.get(
            request.privacy_tier, Decimal("1.0")
        )
        return base * overhead

    def _try_output_cache(
        self, *, request: "InferenceRequest", cost: Decimal,
        streamed: bool = False,
    ) -> "Optional[InferenceResult]":
        """Sprint 811/813 — return a fresh signed InferenceResult
        from the output cache, or None on miss / non-cacheable
        tier / cache absent.

        Cache hit derives a FRESH receipt (new job_id + new
        signature). Privacy invariant: only privacy_tier="none"
        ever reaches the lookup (make_cache_key returns "" for
        other tiers; OutputCache.get("") returns None).

        Sprint 813: `streamed` kwarg threads through
        `_build_signed_receipt` so the streaming path's cache
        hits get receipts with `streamed_output=True` baked
        into the signing payload (Phase 3.x.8 downgrade-
        resistance).
        """
        if self._output_cache is None:
            return None
        from prsm.compute.inference.output_cache import (
            make_cache_key,
        )
        key = make_cache_key(
            request.prompt,
            request.model_id,
            int(request.max_tokens),
            request.privacy_tier.value,
        )
        if not key:
            return None
        hit = self._output_cache.get(key)
        if hit is None:
            return None
        cached_value, _age = hit
        # Build a synthetic outcome that the receipt builder
        # consumes. duration_seconds=0 + epsilon_spent=0 reflect
        # the no-compute reality. tee_attestation comes from the
        # cached entry — replaying the original is honest since
        # the output_hash is bound to it via signing_payload.
        from prsm.compute.tee.models import TEEType
        synthetic_outcome = ChainExecutionResult(
            output=cached_value.get("output", ""),
            duration_seconds=0.0,
            tee_attestation=cached_value.get("tee_attestation", b""),
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )
        receipt = self._build_signed_receipt(
            request=request,
            cost=cost,
            outcome=synthetic_outcome,
            streamed=streamed,
        )
        return InferenceResult(
            request_id=request.request_id,
            success=True,
            output=synthetic_outcome.output,
            receipt=receipt,
        )

    def _put_output_cache(
        self, *, request: "InferenceRequest",
        outcome: "ChainExecutionResult",
    ) -> None:
        """Sprint 811 — store output in cache for future hits.

        No-op when cache absent or privacy_tier != "none" (the
        empty cache key from make_cache_key short-circuits the
        cache.put — defense-in-depth on the privacy invariant)."""
        if self._output_cache is None:
            return
        from prsm.compute.inference.output_cache import (
            make_cache_key,
        )
        key = make_cache_key(
            request.prompt,
            request.model_id,
            int(request.max_tokens),
            request.privacy_tier.value,
        )
        if not key:
            return
        self._output_cache.put(key, {
            "output": outcome.output,
            "tee_attestation": outcome.tee_attestation,
        })

    async def execute(self, request: InferenceRequest) -> InferenceResult:
        """End-to-end: filter → allocate → route → execute → sign."""
        # Phase 3.x.8.1 Task 1 refactor: pre-execute gates 1-6
        # extracted into ``_pre_execute_gates`` so the streaming path
        # (``execute_streaming``) reuses them verbatim. Behavior of
        # this method is unchanged.
        gate_outcome = await self._pre_execute_gates(request)
        if gate_outcome.failure is not None:
            return gate_outcome.failure
        chain = gate_outcome.chain
        cost = gate_outcome.cost
        allocation = gate_outcome.allocation
        tier_filtered = gate_outcome.tier_filtered
        model_info = gate_outcome.model_info

        # Sprint 811 — output cache lookup. On hit, return a
        # fresh signed receipt + cached output WITHOUT running
        # the chain executor. Only fires for privacy_tier="none"
        # (cache key is "" sentinel for other tiers).
        cached = self._try_output_cache(
            request=request, cost=cost,
        )
        if cached is not None:
            return cached

        # 7. Execute primary chain (unary).
        # Sprint 687 F35 — `execute_chain` is SYNC and internally
        # blocks on `send_message_adapter.future.result()`. When the
        # send_message adapter routes a stage to self (sprint 687
        # self-dispatch) and schedules the local executor on the
        # SAME loop, calling execute_chain directly from this async
        # handler deadlocks the loop. Move it to a worker thread via
        # run_in_executor so the loop stays free to process scheduled
        # coroutines (including the self-dispatch one).
        loop = asyncio.get_event_loop()
        try:
            primary_outcome = await loop.run_in_executor(
                None,
                lambda: self._chain_executor.execute_chain(
                    request=request, chain=chain,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ParallaxScheduledExecutor: primary chain executor raised"
            )
            return InferenceResult.failure(
                request.request_id, f"chain execution failure: {exc}"
            )

        # 8. Adapter D — optional redundant execution + mismatch check.
        self._maybe_run_consensus_check(
            request=request,
            allocation=allocation,
            tier_filtered=tier_filtered,
            model_info=model_info,
            primary_chain=chain,
            primary_outcome=primary_outcome,
        )

        # Sprint 811 — store the result so identical future
        # requests hit cache. Pure side-effect; failures are
        # silently absorbed inside the helper.
        self._put_output_cache(
            request=request, outcome=primary_outcome,
        )

        # 9. Build + sign receipt; return InferenceResult.
        receipt = self._build_signed_receipt(
            request=request,
            cost=cost,
            outcome=primary_outcome,
            streamed=False,
        )
        return InferenceResult(
            request_id=request.request_id,
            success=True,
            output=primary_outcome.output,
            receipt=receipt,
        )

    async def execute_streaming(
        self, request: InferenceRequest,
    ) -> AsyncIterator[Union[InferenceTokenEvent, InferenceResult]]:
        """Streaming counterpart of ``execute``. Yields per-token
        ``InferenceTokenEvent``s as the chain's tail stage emits them,
        then exactly one terminal ``InferenceResult`` (success or
        failure).

        Pre-execute gate semantics match ``execute()`` exactly — both
        paths share ``_pre_execute_gates``. On gate failure: yields a
        single ``InferenceResult.failure(...)`` and terminates with
        NO token events emitted (caller observed: empty stream + sole
        terminal failure event).

        On success: yields N token events followed by exactly one
        ``InferenceResult.success(receipt)`` whose receipt has
        ``streamed_output=True`` (Phase 3.x.8 Task 4 downgrade-
        resistant signing-payload commit).

        Mid-stream chain-executor exceptions surface as a terminal
        ``InferenceResult.failure(...)`` — non-fatal at the wire
        layer so the SSE endpoint can encode the error event +
        refund the escrow.
        """
        gate_outcome = await self._pre_execute_gates(request)
        if gate_outcome.failure is not None:
            yield gate_outcome.failure
            return
        chain = gate_outcome.chain
        cost = gate_outcome.cost

        # Sprint 813 — output cache lookup on the streaming path.
        # On hit: emit one token event carrying the full cached
        # output (finish_reason="cache_hit"), then one terminal
        # InferenceResult with a fresh streamed-flagged receipt.
        # No chain executor runs.
        cached = self._try_output_cache(
            request=request, cost=cost, streamed=True,
        )
        if cached is not None:
            yield InferenceTokenEvent(
                sequence_index=0,
                text_delta=cached.output,
                token_id=None,
                finish_reason="cache_hit",
            )
            yield cached
            return

        # Phase 3.x.11.q — Tier C streaming routes through the
        # constant-time chain decorator when wired. Tier A/B
        # continues to use the default chain_executor. When Tier C
        # is requested but no decorator is wired, surface a
        # structured failure so operators learn the deploy needs
        # the decorator (rather than silently falling back to the
        # leaky path).
        chain_executor = self._chain_executor
        if request.content_tier == ContentTier.C:
            if self._tier_c_chain_executor is None:
                yield InferenceResult.failure(
                    request.request_id,
                    "Tier C streaming requires Phase 3.x.11.q "
                    "constant-time decorator — wire "
                    "tier_c_chain_executor= via "
                    "make_tier_c_sharded_executor(...) at "
                    "ParallaxScheduledExecutor construction",
                )
                return
            chain_executor = self._tier_c_chain_executor

            # Phase 3.x.11.q.y — Tier C + temperature > 0
            # (i.e. speculation) is denied unless the operator
            # explicitly opts in via ``tier_c_speculation_enabled``
            # AND wires a speculation-capable
            # tier_c_chain_executor (encrypted_probs_cipher +
            # flat_k_mode + a tail with constant_k_commitment).
            # Otherwise vanilla speculation under Tier C would
            # leak the per-iteration acceptance count on the
            # wire and burn the chain-level constant-time
            # invariant. See docs §7.14 (audit-prep).
            request_temp = getattr(request, "temperature", None)
            if (
                request_temp is not None
                and float(request_temp) > 0.0
                and not self._tier_c_speculation_enabled
            ):
                yield InferenceResult.failure(
                    request.request_id,
                    "Tier C streaming with temperature > 0 "
                    "(speculation) requires "
                    "tier_c_speculation_enabled=True at "
                    "ParallaxScheduledExecutor construction "
                    "AND a speculation-capable "
                    "tier_c_chain_executor wired with "
                    "encrypted_probs_cipher + flat_k_mode + a "
                    "tail with constant_k_commitment=True "
                    "(Phase 3.x.11.q.y bundle). Until then, "
                    "Tier C streaming is restricted to "
                    "temperature == 0 (greedy).",
                )
                return

        # Drive the chain executor's streaming generator. Each
        # ``StreamToken`` becomes an ``InferenceTokenEvent``; the
        # terminal ``ChainExecutionResult`` drives the signed receipt.
        # Lazy import to avoid a hard dep on chain_rpc at module
        # import time — keeps the executor module loadable for tests
        # that don't exercise the streaming surface.
        from prsm.compute.chain_rpc.client import StreamToken

        outcome: Optional[ChainExecutionResult] = None
        try:
            for item in chain_executor.execute_chain_streaming(
                request=request, chain=chain,
            ):
                if isinstance(item, StreamToken):
                    yield InferenceTokenEvent(
                        sequence_index=item.sequence_index,
                        text_delta=item.text_delta,
                        token_id=item.token_id,
                        finish_reason=item.finish_reason,
                    )
                elif isinstance(item, ChainExecutionResult):
                    outcome = item
                    break
                else:
                    yield InferenceResult.failure(
                        request.request_id,
                        f"chain executor yielded unexpected type "
                        f"{type(item).__name__}; expected StreamToken "
                        f"or ChainExecutionResult",
                    )
                    return
        except AttributeError as exc:
            # Wrapped chain_executor doesn't implement
            # execute_chain_streaming — operator misconfig. Surface
            # as a structured failure rather than crashing.
            logger.exception(
                "ParallaxScheduledExecutor.execute_streaming: "
                "chain_executor has no execute_chain_streaming"
            )
            yield InferenceResult.failure(
                request.request_id,
                f"chain executor does not support streaming: {exc}",
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ParallaxScheduledExecutor.execute_streaming: "
                "chain_executor.execute_chain_streaming raised"
            )
            yield InferenceResult.failure(
                request.request_id, f"chain execution failure: {exc}"
            )
            return

        if outcome is None:
            yield InferenceResult.failure(
                request.request_id,
                "chain executor exhausted without yielding a terminal "
                "ChainExecutionResult",
            )
            return

        # Sprint 813 — record the result so identical future
        # streaming requests hit cache and skip the chain
        # executor entirely. Same cache key as unary (sprint 811);
        # streamed flag tracked at receipt-build time, not in key.
        self._put_output_cache(
            request=request, outcome=outcome,
        )

        # Build + sign receipt with streamed_output=True (Phase 3.x.8
        # Task 4 downgrade-resistant flag). Skips consensus check on
        # the streaming path: redundant execution would require
        # running TWO streaming chains in parallel, doubling
        # bandwidth/latency for a sampled request — Phase 3.x.10
        # revisit.
        receipt = self._build_signed_receipt(
            request=request,
            cost=cost,
            outcome=outcome,
            streamed=True,
        )
        yield InferenceResult(
            request_id=request.request_id,
            success=True,
            output=outcome.output,
            receipt=receipt,
        )

    async def _pre_execute_gates(
        self, request: InferenceRequest,
    ) -> _GateOutcome:
        """Steps 1-6 of execute() / execute_streaming(): catalog
        lookup, budget gate, pool gathering, anchor filter, tier
        filter, allocation, routing. Returns a ``_GateOutcome``
        whose ``failure`` is set on ANY gate rejection (caller yields
        it as the sole terminal event); on success carries the
        ``chain`` + ``cost`` + ``allocation`` + ``tier_filtered`` +
        ``model_info`` for downstream dispatch + consensus check.

        Phase 3.x.8.1 Task 1 — extracted from ``execute()`` so the
        streaming path reuses identical pre-execute semantics. No
        behavior change to ``execute()``.
        """
        # 1. Catalog lookup.
        if request.model_id not in self._catalog:
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id,
                f"Unknown model_id: {request.model_id}",
            ))
        model_info = self._catalog[request.model_id]

        # 2. Budget gate.
        cost = await self.estimate_cost(request)
        if request.budget_ftns < cost:
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id,
                f"Insufficient budget: {request.budget_ftns} FTNS < "
                f"required {cost} FTNS",
            ))

        # 3. Pool gathering + Adapter A (anchor verify).
        try:
            raw_pool = list(self._pool_provider())
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ParallaxScheduledExecutor: gpu_pool_provider raised"
            )
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id,
                f"GPU pool provider failure: {exc}",
            ))
        if not raw_pool:
            # Sprint 631 — actionable error. Pre-sprint the message
            # was just "GPU pool is empty"; operators with all other
            # PRSM_PARALLAX_* env vars set correctly hit this and had
            # no breadcrumb pointing at the static-empty default. The
            # canonical fix (DHT-backed provider per sprint 561) is
            # the right answer; this message at least documents it.
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id,
                "GPU pool is empty (gpu_pool_provider returned []). "
                "Default PRSM_PARALLAX_GPU_POOL_KIND=static-empty "
                "provides no GPUs by design; sprint 561+ wires a "
                "DHT-backed provider that discovers peer GPUs. "
                "Until then, /compute/inference and /compute/inference/"
                "stream cannot route through ParallaxScheduledExecutor."
            ))
        anchor_filtered = self._trust.filter_pool(raw_pool)
        if not anchor_filtered:
            # Sprint 686 F32 fix — `filter_pool` runs BOTH the anchor
            # verify adapter AND the stake-eligibility check. The
            # pre-686 message blamed anchor verification alone; in
            # practice stake eligibility is the more common rejection
            # path (F31 — pubkey/address mismatch) and operators
            # correlate the wrong logs. The honest error names both.
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id,
                "no GPU passed pool gating (Adapter A anchor "
                "verification AND/OR stake-eligibility check). "
                "Check journalctl for AnchorVerifyAdapter rejections; "
                "if those are empty, the cause is stake eligibility — "
                "set PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory for "
                "pre-stake-ceremony live-attest, or post real stake "
                "via StakeBond contract.",
            ))

        # 4. Adapter B — tier gate.
        try:
            tier_filtered = self._trust.filter_for_request(
                anchor_filtered, request.privacy_tier
            )
        except TierGateRejected as exc:
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id, f"tier gate refusal: {exc}"
            ))

        if not tier_filtered:
            # NONE-tier with empty pool — treated the same as no GPUs.
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id, "no eligible GPU after tier gate"
            ))

        # 5. Phase-1 allocation (cache or recompute).
        try:
            allocation = self._get_or_build_allocation(
                request.model_id, tier_filtered, model_info
            )
        except EmptyPoolError as exc:
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id, f"empty pool: {exc}"
            ))
        except InsufficientCapacityError as exc:
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id, f"insufficient capacity: {exc}"
            ))
        except AllocationError as exc:
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id, f"allocation failure: {exc}"
            ))

        # 6. Build router; route. Retry once on NoCoverageError after
        # forced Phase-1 recompute (paper §3.4).
        try:
            chain = self._route_with_retry(
                allocation=allocation,
                model_info=model_info,
                tier_filtered=tier_filtered,
                request=request,
                model_id=request.model_id,
            )
        except (NoCoverageError, EmptyAllocationError) as exc:
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id, f"routing coverage failure: {exc}"
            ))
        except RegionNotFoundError as exc:
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id, f"region not found: {exc}"
            ))
        except BudgetExceededError as exc:
            return _GateOutcome(failure=InferenceResult.failure(
                request.request_id, f"latency budget exceeded: {exc}"
            ))

        # All gates pass — return the dispatch context.
        return _GateOutcome(
            chain=chain,
            cost=cost,
            allocation=allocation,
            tier_filtered=tier_filtered,
            model_info=model_info,
        )

    # ── Allocation cache ──────────────────────────────────────────────

    @property
    def phase1_recompute_count(self) -> int:
        """Counter incremented every time Phase-1 alloc is rebuilt.
        Tests use this to verify §3.4 (recompute-only-on-coverage-gap)."""
        return self._phase1_recompute_count

    def _get_or_build_allocation(
        self,
        model_id: str,
        pool: Sequence[ParallaxGPU],
        model_info: ModelInfo,
    ) -> AllocationResult:
        """Cache hit when the cached stage-set is non-empty AND a
        subset of the current trust-filtered pool. Otherwise rebuild.

        The non-empty guard guards against cache poisoning under
        ``allow_partial_regions=True``: an allocation with zero
        pipelines (every region failed) has stage_set == ∅, which
        is trivially a subset of any pool, so a naive issubset check
        would re-use the empty allocation forever and route() would
        fail every subsequent request even after the pool recovers."""
        pool_ids = frozenset(g.node_id for g in pool)
        cached = self._alloc_cache.get(model_id)
        if cached is not None:
            cached_alloc, cached_stage_set = cached
            if cached_stage_set and cached_stage_set.issubset(pool_ids):
                return cached_alloc
        # Rebuild.
        return self._rebuild_allocation(
            model_id, pool, model_info, pool_ids
        )

    def _rebuild_allocation(
        self,
        model_id: str,
        pool: Sequence[ParallaxGPU],
        model_info: ModelInfo,
        pool_ids: frozenset,
    ) -> AllocationResult:
        allocation = allocate_across_regions(
            list(pool),
            model_info,
            allow_partial=self._allow_partial_regions,
        )
        stage_set = frozenset(
            node_id
            for pipelines in allocation.region_to_pipelines.values()
            for pipeline in pipelines
            for node_id in pipeline.stages
        )
        self._phase1_recompute_count += 1
        if not stage_set:
            # Don't poison the cache with an empty allocation (only
            # reachable under allow_partial_regions=True when every
            # region failed). Force the next request to retry against
            # a possibly-recovered pool.
            return allocation
        self._alloc_cache[model_id] = (allocation, stage_set)
        return allocation

    def _route_with_retry(
        self,
        *,
        allocation: AllocationResult,
        model_info: ModelInfo,
        tier_filtered: Sequence[ParallaxGPU],
        request: InferenceRequest,
        model_id: str,
    ) -> GPUChain:
        """Try the cached/built allocation; on coverage gap, force one
        recompute and retry. Second NoCoverageError propagates."""
        router = RequestRouter(
            allocation=allocation,
            profile_source=self._trust.profile_source,
            model_info=model_info,
            original_gpus=list(tier_filtered),
        )
        route_req = RouteRequest(request_id=request.request_id)
        try:
            return router.route(route_req)
        except NoCoverageError:
            # Force rebuild and retry — even if the cached allocation
            # had every node, the membership-changed shape may need
            # different stage assignments. Caller above wraps a second
            # NoCoverageError into InferenceResult.failure.
            pool_ids = frozenset(g.node_id for g in tier_filtered)
            fresh_alloc = self._rebuild_allocation(
                model_id, tier_filtered, model_info, pool_ids
            )
            fresh_router = RequestRouter(
                allocation=fresh_alloc,
                profile_source=self._trust.profile_source,
                model_info=model_info,
                original_gpus=list(tier_filtered),
            )
            return fresh_router.route(route_req)

    # ── Consensus-mismatch hook ───────────────────────────────────────

    def _maybe_run_consensus_check(
        self,
        *,
        request: InferenceRequest,
        allocation: AllocationResult,
        tier_filtered: Sequence[ParallaxGPU],
        model_info: ModelInfo,
        primary_chain: GPUChain,
        primary_outcome: ChainExecutionResult,
    ) -> Optional[ChallengeRecord]:
        """When sampled, attempt to route + execute an alternate chain
        and ask the consensus hook to compare. Failures inside this
        path are logged but do NOT propagate — primary already succeeded
        and the user's request is satisfied."""
        hook = self._trust.consensus_hook
        if not hook.should_sample_redundant(request.request_id):
            return None

        secondary_chain = self._try_route_alternate_chain(
            allocation=allocation,
            tier_filtered=tier_filtered,
            model_info=model_info,
            request=request,
            primary_chain=primary_chain,
        )
        if secondary_chain is None:
            logger.info(
                "ParallaxScheduledExecutor: sampled request %s for "
                "redundant execution but no alternate chain available; "
                "skipping consensus check",
                request.request_id,
            )
            return None

        try:
            secondary_outcome = self._chain_executor.execute_chain(
                request=request, chain=secondary_chain
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "ParallaxScheduledExecutor: secondary chain executor raised "
                "for request %s; skipping consensus check",
                request.request_id,
            )
            return None

        return hook.compare_and_challenge(
            request_id=request.request_id,
            primary_output=primary_outcome.output.encode("utf-8"),
            secondary_output=secondary_outcome.output.encode("utf-8"),
            primary_chain=primary_chain,
            secondary_chain=secondary_chain,
        )

    def _try_route_alternate_chain(
        self,
        *,
        allocation: AllocationResult,
        tier_filtered: Sequence[ParallaxGPU],
        model_info: ModelInfo,
        request: InferenceRequest,
        primary_chain: GPUChain,
    ) -> Optional[GPUChain]:
        """Build an alternate chain by excluding primary's stages from
        the pool and re-running Phase-1 on the remainder. Returns None
        if the remainder can't cover the model — that's a normal "not
        enough redundancy" signal, not an error."""
        primary_node_ids = set(primary_chain.stages)
        alternate_pool = [
            gpu for gpu in tier_filtered
            if gpu.node_id not in primary_node_ids
        ]
        if not alternate_pool:
            return None
        try:
            alt_allocation = allocate_across_regions(
                alternate_pool,
                model_info,
                allow_partial=True,  # alternates are best-effort
            )
        except (AllocationError, InsufficientCapacityError, EmptyPoolError):
            return None

        if alt_allocation.total_pipeline_count() == 0:
            return None

        try:
            alt_router = RequestRouter(
                allocation=alt_allocation,
                profile_source=self._trust.profile_source,
                model_info=model_info,
                original_gpus=alternate_pool,
            )
            return alt_router.route(
                RouteRequest(request_id=f"{request.request_id}-alt")
            )
        except (NoCoverageError, EmptyAllocationError, RegionNotFoundError):
            return None

    # ── Receipt assembly ──────────────────────────────────────────────

    def _build_signed_receipt(
        self,
        *,
        request: InferenceRequest,
        cost: Decimal,
        outcome: ChainExecutionResult,
        streamed: bool = False,
    ) -> InferenceReceipt:
        """Assemble + Ed25519-sign the receipt under this node's identity.

        Phase 3.x.8.1 Task 1 — ``streamed`` parameter closes the
        Phase 3.x.8 round-1 L3 follow-up: unary + streaming receipt-
        build paths share this single helper. The flag is part of
        the signed payload (Phase 3.x.8 Task 4 conditional encoding),
        so a relay can't downgrade a streamed receipt to unary or
        vice versa.

        ``job_id`` prefix differs slightly so audit logs can filter
        streamed-vs-unary jobs without parsing the full receipt:
        ``parallax-job-<hex>`` for unary, ``parallax-stream-job-<hex>``
        for streamed. Both prefixes are non-load-bearing — the flag
        in the signed payload is authoritative.
        """
        output_hash = hashlib.sha256(outcome.output.encode("utf-8")).digest()
        # Sprint 1099 (Domain-03 review F1, first brick) — bind the INPUT prompt
        # into the signed receipt. Without it the receipt commits only to the
        # output, so a head node could return a (cheaper / canned) answer
        # computed for a DIFFERENT prompt and the receipt still verified. A
        # caller recomputes sha256(their prompt) and confirms it matches.
        prompt_hash = hashlib.sha256(
            (getattr(request, "prompt", "") or "").encode("utf-8")
        ).digest()
        prefix = "parallax-stream-job" if streamed else "parallax-job"
        # Sprint 413 — thread sprint-297 privacy fields from
        # the chain executor's outcome into the InferenceReceipt.
        # When the chain executor doesn't populate them
        # (pre-sprint-413 callers OR chains running without DP),
        # the Optional fields stay None and the receipt
        # signing-payload is byte-identical to pre-sprint-413
        # receipts (sprint 297 conditional encoding handles this).
        unsigned = InferenceReceipt(
            job_id=f"{prefix}-{uuid.uuid4().hex[:12]}",
            request_id=request.request_id,
            model_id=request.model_id,
            content_tier=request.content_tier,
            privacy_tier=request.privacy_tier,
            epsilon_spent=outcome.epsilon_spent,
            tee_type=outcome.tee_type,
            tee_attestation=outcome.tee_attestation,
            output_hash=output_hash,
            duration_seconds=outcome.duration_seconds,
            cost_ftns=cost,
            streamed_output=streamed,
            activation_noise_trace=getattr(
                outcome, "activation_noise_trace", None,
            ),
            topology_assignment=getattr(
                outcome, "topology_assignment", None,
            ),
            # Sprint 778 — thread partial-completion marker into
            # the signed receipt. Same defensive-getattr pattern
            # as sprint 413 for activation_noise_trace +
            # topology_assignment: legacy ChainExecutor implementers
            # without the field get None → byte-identical pre-778
            # signing payload.
            partial_completion=getattr(
                outcome, "partial_completion", None,
            ),
            prompt_hash=prompt_hash,
            # Sprint 1110 (Domain-03 F1/F2, brick 4) — carry the per-stage signed
            # activation chain into the receipt (None for single-node/streaming paths
            # that don't assemble one → byte-identical pre-1110 receipt).
            stage_activation_chain=getattr(
                outcome, "stage_activation_chain", None,
            ),
        )
        return sign_receipt(unsigned, self._identity)
