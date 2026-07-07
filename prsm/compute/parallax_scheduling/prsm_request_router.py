"""
PRSM-side adapter for the Phase-2 per-request routing.

Phase 3.x.6 Task 3 — wraps the vendored ``DynamicProgrammingRouting``
(from upstream's ``request_routing.py``) with a clean PRSM-side API:

  - ``ProfileSource`` Protocol — abstract source for live (τ_g,ℓ, ρ_g,g')
    measurements. Concrete implementation lands in Task 4 (profile DHT
    on Phase 6 transport). Tests use a synthetic in-memory source.

  - ``RequestRouter`` — given a Phase-1 ``AllocationResult`` + a
    ``ProfileSource``, picks the minimum-latency chain for an
    inference request.

  - ``GPUChain`` — typed result: ordered list of (node_id, layer_range)
    tuples, the chosen region, and the chain's estimated total
    latency.

What this module owns:

  - Profile-snapshot application: for each Node in the allocation,
    write the latest (τ, ρ) values onto the upstream Node so the
    vendored DP sees them. Pure adapter logic — algorithm itself
    is unchanged.

  - Stale-profile fallback: if a node has no live snapshot, the
    upstream's ``layer_latency_ms`` already falls through to
    ``roofline_layer_latency_ms()`` (hardware-derived static
    estimate). That's the "last-known-good" behavior the design
    plan §4 Task 3 calls for; we surface it explicitly in the
    profile-staleness count returned alongside the chain.

  - Coverage-gap detection: if the requested region has no allocation,
    or the allocation's union of layer ranges doesn't cover [0, L),
    we raise ``NoCoverageError`` immediately rather than letting
    the upstream return ``([], inf)`` and forcing callers to interpret
    that.

  - Deterministic tie-breaks: when multiple chains have identical
    latency (e.g., a synthetic test pool with all-equal profiles),
    we sort candidates by (latency_ms, sorted_stage_ids) so test
    runs are reproducible. The upstream DP is already deterministic
    given a deterministic input; PRSM-side determinism is ensured
    by our region-iteration order + the sorted profile-application
    loop.

What this module does NOT own:

  - The DP algorithm itself — vendored in ``request_routing.py``.
  - The trust adapters (anchor verify, tier gate, stake-weighted,
    consensus mismatch) — Task 5.
  - The actual layer execution — that's Task 6's
    ``ParallaxScheduledExecutor`` calling into PRSM's TensorParallelExecutor.
  - Live profile publishing — Task 4 ships the DHT producer side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from prsm.compute.parallax_scheduling.model_info import ModelInfo
from prsm.compute.parallax_scheduling.node import Node
from prsm.compute.parallax_scheduling.node_management import NodeManager, NodeState
from prsm.compute.parallax_scheduling.prsm_types import (
    AllocationResult,
    ParallaxGPU,
    RegionPipeline,
    to_parallax_node,
)


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Profile data types
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProfileSnapshot:
    """Latest live measurements published by a single GPU.

    Per the paper §3.3, every GPU publishes (τ_g,ℓ, ρ_g,g') every 1-2s
    to a DHT. PRSM mirrors this with an opt-in profile DHT on Phase 6
    transport (Task 4) — but the router is source-agnostic and
    consumes any ``ProfileSource`` Protocol implementation.

    Fields:
      node_id            Identity of the publishing GPU.
      layer_latency_ms   τ_g,ℓ — average per-layer compute latency
                         this GPU is observing right now. Per-layer,
                         not per-stage, so we can compose chains
                         where different stages have different layer
                         counts.
      rtt_to_peers       ρ_g,g' — half-RTT (one-way) from this GPU
                         to each named peer. Symmetric in
                         expectation; we don't enforce that.
      timestamp_unix     When this snapshot was taken. Stale snapshots
                         (older than the source's TTL) are excluded
                         by the source itself.
    """

    node_id: str
    layer_latency_ms: float
    rtt_to_peers: Dict[str, float]
    timestamp_unix: float


class ProfileSource(Protocol):
    """Read-only source of live profile snapshots.

    Concrete implementations:
      - Task 4: ``ProfileDHT`` reads from the live PRSM-native DHT.
      - Tests: ``InMemoryProfileSource`` (below) returns hard-coded
        snapshots for deterministic test pools.
      - Operator override: a static-config source that reads from
        the operator's wiring file (useful for staging deployments
        without DHT churn).

    The Protocol is shaped so the router never needs to know which
    backend is in play.

    Reserved name: ``is_eligible(node_id) -> bool``
      Implementations MAY define an ``is_eligible`` method, in which
      case ``TrustStack.filter_pool`` will call it as a pre-Phase-1
      pool-admission gate. Currently used by
      ``StakeWeightedTrustAdapter`` to enforce a minimum-stake
      participation threshold; the upstream router's roofline
      fallback would otherwise route to a zero-stake liar based on
      advertised hardware specs alone.

      If you implement ``is_eligible`` on a custom ``ProfileSource``,
      use the same semantics: True means "this node is admissible to
      the allocation pool"; False means "exclude before Phase-1".
      Any other semantic will silently shift filter_pool's behavior.
      Implementations that don't define ``is_eligible`` pass through
      unchanged (anchor-only filtering).
    """

    def get_snapshot(self, node_id: str) -> Optional[ProfileSnapshot]:
        """Return the latest non-stale snapshot for ``node_id``, or
        None if no current snapshot exists. None triggers the
        upstream's roofline fallback path (hardware-derived static
        estimate)."""
        ...


@dataclass
class InMemoryProfileSource:
    """Test-only profile source that returns pre-loaded snapshots.

    Tests construct one of these with the snapshots they want to
    exercise. The router treats it identically to a real DHT — the
    Protocol shape is the only contract.
    """

    snapshots: Dict[str, ProfileSnapshot] = field(default_factory=dict)

    def get_snapshot(self, node_id: str) -> Optional[ProfileSnapshot]:
        return self.snapshots.get(node_id)

    def set_snapshot(self, snapshot: ProfileSnapshot) -> None:
        self.snapshots[snapshot.node_id] = snapshot

    def clear(self, node_id: str) -> None:
        """Simulate a stale / departed node — subsequent get returns None."""
        self.snapshots.pop(node_id, None)


# ──────────────────────────────────────────────────────────────────────────
# Request + result types
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteRequest:
    """Input to the router.

    Fields:
      request_id         Unique correlator. Logged + carried into
                         the upstream's RequestSignal.
      preferred_region   If set, route within this region. If None,
                         the router picks the lowest-latency region.
      max_latency_ms     Optional latency budget. If the chosen
                         chain exceeds it, raise ``BudgetExceededError``.
                         Default None = no budget gate.
    """

    request_id: str
    preferred_region: Optional[str] = None
    max_latency_ms: Optional[float] = None


@dataclass(frozen=True)
class GPUChain:
    """Output of the router — the chosen pipeline chain.

    Fields:
      request_id         Echoed from the input.
      region             Region the chain runs in.
      stages             Ordered node_ids; first node hosts layer 0.
      layer_ranges       Half-open (start, end) per stage.
      total_latency_ms   Estimated end-to-end latency from the DP.
      stale_profile_count How many stages had no live snapshot at
                         routing time (fell back to roofline).
                         Operators monitor this to detect DHT churn.
    """

    request_id: str
    region: str
    stages: Tuple[str, ...]
    layer_ranges: Tuple[Tuple[int, int], ...]
    total_latency_ms: float
    stale_profile_count: int


# ──────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────


class RoutingError(Exception):
    """Base error for routing failures."""


class EmptyAllocationError(RoutingError):
    """The supplied AllocationResult has no pipelines.

    Caller is responsible for distinguishing 'allocation never ran'
    from 'allocation ran but had no GPUs'."""


def _empty_allocation_detail(model_info: "ModelInfo", gpus: "List[ParallaxGPU]") -> str:
    """Sprint 1396 — build an ACTIONABLE message for an empty allocation: the memory-derived decoder
    capacity the pool can host vs the model's layer count, per node, plus concrete fixes. Fail-soft:
    on any error, fall back to the original terse message so diagnostics never mask the failure."""
    base = ("AllocationResult has no pipelines — Phase-1 allocator produced an empty result, or you "
            "passed allocation that was never run")
    try:
        if not gpus:
            return base + " (the pool is empty — no nodes advertised compute capacity)."
        from prsm.compute.parallax_scheduling.prsm_types import to_parallax_node
        caps = []
        total_cap = 0
        for g in gpus:
            c = int(to_parallax_node(g, model_info).get_decoder_layer_capacity())
            caps.append(f"{g.node_id[:8]}={c}L@{g.memory_gb}GB")
            total_cap += c
        need = int(model_info.num_layers)
        if total_cap >= need:
            return base  # not a capacity shortfall — keep the terse message
        return (
            f"AllocationResult has no pipelines: the {len(gpus)}-node pool can host {total_cap} "
            f"decoder layer(s) but the model needs {need}. Per-node capacity (memory-derived): "
            + ", ".join(caps)
            + ". Fixes: raise PRSM_PARALLAX_MEMORY_GB_OVERRIDE (allocator budgets ~14 layers/GB), "
            "add nodes to the pool, use PRSM_INFERENCE_EXECUTOR=local for robust single-node "
            "serving, or serve a smaller model.")
    except Exception:  # noqa: BLE001 — diagnostics must never crash the (already-failing) path
        return base


class NoCoverageError(RoutingError):
    """No chain in the (preferred or any) region covers all model
    layers. Caller should rerun Phase-1 allocation — this is the
    paper §3.4 'global rebalance' trigger condition."""


class RegionNotFoundError(RoutingError):
    """The request's ``preferred_region`` has no allocation in this
    AllocationResult. Distinct from NoCoverageError because the fix
    is operator-side (advertise GPUs in that region) rather than
    Phase-1 rerun."""


class BudgetExceededError(RoutingError):
    """The minimum-latency chain exceeds the request's max_latency_ms.
    Surfaces with the chain attached so callers can decide whether
    to fall back to a different model size, retry later, or fail
    the request."""

    def __init__(self, message: str, chain: GPUChain):
        super().__init__(message)
        self.chain = chain


# ──────────────────────────────────────────────────────────────────────────
# Request router — composes vendored DP with PRSM profile source
# ──────────────────────────────────────────────────────────────────────────


class RequestRouter:
    """Per-request DAG-DP routing on top of a Phase-1 allocation.

    Constructed once per (allocation, profile_source) pair. Reused
    across requests — each ``route()`` call applies the latest
    profile snapshots to a fresh NodeManager copy and runs the
    upstream DP.

    Why fresh NodeManager per request: the upstream's DP mutates
    Node state (start_layer/end_layer, layer_latency_ms) in place.
    To avoid bleed-over between concurrent requests, we rebuild
    the NodeManager per call. This is cheap — sub-ms — relative to
    the DP cost itself (single-digit ms at hundreds of GPUs per
    paper §4.3).
    """

    def __init__(
        self,
        allocation: AllocationResult,
        profile_source: ProfileSource,
        model_info: ModelInfo,
        original_gpus: List[ParallaxGPU],
    ) -> None:
        if allocation.total_pipeline_count() == 0:
            # sp1396 — a bare "no pipelines" dead-ends the operator (I hit it live on a single-node
            # seed: memory_gb=0.8 derived capacity 11 < gpt2's 12 layers → empty, with no hint why).
            # Explain the exact capacity shortfall + how to fix it.
            raise EmptyAllocationError(_empty_allocation_detail(model_info, original_gpus))
        self._allocation = allocation
        self._profile_source = profile_source
        self._model_info = model_info
        # Index original ParallaxGPUs by node_id for hardware-spec lookup
        # at routing time. Algorithm-unchanged invariant: we don't ALTER
        # hardware specs at routing time — we just need them to
        # reconstruct fresh upstream Nodes per request.
        self._gpu_by_id: Dict[str, ParallaxGPU] = {
            gpu.node_id: gpu for gpu in original_gpus
        }

    def route(self, request: RouteRequest) -> GPUChain:
        """Pick the minimum-latency chain for ``request``.

        Selection logic:
          1. If ``request.preferred_region`` is set, route within
             that region (raises ``RegionNotFoundError`` if absent).
          2. Otherwise, route within each region independently and
             return the global minimum.
          3. Apply ``max_latency_ms`` budget if set.

        Determinism note: ties between equal-latency chains are
        broken by (region_name_sorted, sorted_stages) — reproducible
        across runs given the same inputs.
        """
        candidate_regions: List[str]
        if request.preferred_region is not None:
            if request.preferred_region not in self._allocation.region_to_pipelines:
                raise RegionNotFoundError(
                    f"preferred_region {request.preferred_region!r} has no "
                    f"allocation; available regions: "
                    f"{sorted(self._allocation.region_to_pipelines.keys())}"
                )
            candidate_regions = [request.preferred_region]
        else:
            # Sort for deterministic iteration order — matters when
            # two regions produce identical-latency chains.
            candidate_regions = sorted(self._allocation.region_to_pipelines.keys())

        best: Optional[GPUChain] = None
        for region in candidate_regions:
            chain = self._route_within_region(region, request)
            if chain is None:
                continue
            if best is None or _chain_lt(chain, best):
                best = chain

        if best is None:
            raise NoCoverageError(
                f"no chain in regions {candidate_regions} covers all "
                f"{self._model_info.num_layers} layers — Phase-1 rerun needed"
            )

        if (
            request.max_latency_ms is not None
            and best.total_latency_ms > request.max_latency_ms
        ):
            raise BudgetExceededError(
                f"chain latency {best.total_latency_ms:.2f}ms exceeds "
                f"budget {request.max_latency_ms:.2f}ms",
                chain=best,
            )

        return best

    # ── Internals ─────────────────────────────────────────────────────

    def _route_within_region(
        self, region: str, request: RouteRequest
    ) -> Optional[GPUChain]:
        """Apply current snapshots + run upstream DP for one region.

        Returns None if no chain in this region covers all layers
        (so the outer caller can try the next region or raise
        NoCoverageError if all regions fail).
        """
        pipelines = self._allocation.region_to_pipelines.get(region, [])
        if not pipelines:
            return None

        # Build a fresh NodeManager from the region's pipeline node_ids.
        # Apply current profile snapshots; fall through to roofline
        # for any node missing live data.
        nm, stale_count = self._build_node_manager_for_region(pipelines)

        # The vendored DP wants nodes in the ACTIVE state to consider
        # them for routing. STANDBY nodes are skipped.
        nm.activate(list(self._iter_node_ids(pipelines)))

        # Late import — keeps module-load cost low and isolates the
        # vendored algorithm dep from this PRSM-side adapter.
        from prsm.compute.parallax_scheduling.request_routing import (
            DynamicProgrammingRouting,
        )

        router = DynamicProgrammingRouting(
            node_manager=nm,
            total_layers=self._model_info.num_layers,
        )
        node_path, total_latency_ms = router.find_optimal_path()

        if not node_path or total_latency_ms == float("inf"):
            return None

        # Reconstruct layer_ranges from the chosen Node path.
        layer_ranges: List[Tuple[int, int]] = []
        for node_id in node_path:
            node = nm.get(node_id)
            if node is None or node.start_layer is None or node.end_layer is None:
                # Should not happen — DP only returns nodes with
                # valid layer assignments — but guard for the
                # never-raises invariant.
                logger.warning(
                    "RequestRouter: node %s in DP path has no layer assignment",
                    node_id,
                )
                return None
            layer_ranges.append((node.start_layer, node.end_layer))

        # Verify the reconstruction covers [0, num_layers) — defense
        # in depth against a bug in the upstream DP returning a path
        # that doesn't actually tile the model.
        if not layer_ranges or layer_ranges[0][0] != 0:
            return None
        if layer_ranges[-1][1] != self._model_info.num_layers:
            return None
        for prev, nxt in zip(layer_ranges, layer_ranges[1:]):
            if prev[1] != nxt[0]:
                return None

        return GPUChain(
            request_id=request.request_id,
            region=region,
            stages=tuple(node_path),
            layer_ranges=tuple(layer_ranges),
            total_latency_ms=total_latency_ms,
            stale_profile_count=stale_count,
        )

    def _build_node_manager_for_region(
        self, pipelines: List[RegionPipeline]
    ) -> Tuple[NodeManager, int]:
        """Reconstruct upstream Nodes from ParallaxGPUs + apply
        current profile snapshots. Returns (node_manager, stale_count).
        """
        # Sort node_ids deterministically for reproducible Manager
        # iteration order. Matters for tiebreaks.
        unique_node_ids = sorted({nid for p in pipelines for nid in p.stages})
        nodes: List[Node] = []
        stale_count = 0

        for node_id in unique_node_ids:
            gpu = self._gpu_by_id.get(node_id)
            if gpu is None:
                # An allocation referenced a node that's not in the
                # original GPU pool. Allocation/router decoupling is
                # the caller's responsibility; we surface this as a
                # silent skip + log warning so a route can still
                # succeed if the rest of the chain covers the layers.
                logger.warning(
                    "RequestRouter: node %s in allocation has no "
                    "ParallaxGPU spec; excluded from routing",
                    node_id,
                )
                continue
            node = to_parallax_node(gpu, self._model_info)

            # Apply this node's layer assignment from the allocation
            # (so the upstream DP sees a valid (start_layer, end_layer)
            # pair and can compute roofline latency for the right
            # number of layers).
            for pipeline in pipelines:
                if node_id in pipeline.stages:
                    idx = pipeline.stages.index(node_id)
                    start, end = pipeline.layer_ranges[idx]
                    node.set_layer_allocation(start, end)
                    break

            # Apply latest live snapshot — falls through to roofline
            # if absent.
            snap = self._profile_source.get_snapshot(node_id)
            if snap is None:
                stale_count += 1
            else:
                node.set_layer_latency_ms(snap.layer_latency_ms)
                # Apply RTTs (only those targeting other nodes in this
                # routing set — others are irrelevant).
                for peer_id, rtt_ms in snap.rtt_to_peers.items():
                    if peer_id in unique_node_ids:
                        node.update_rtt(peer_id, rtt_ms)

            nodes.append(node)

        nm = NodeManager(initial_nodes=nodes)
        return nm, stale_count

    @staticmethod
    def _iter_node_ids(pipelines: List[RegionPipeline]) -> List[str]:
        seen: List[str] = []
        seen_set: set = set()
        for pipeline in pipelines:
            for node_id in pipeline.stages:
                if node_id not in seen_set:
                    seen.append(node_id)
                    seen_set.add(node_id)
        return seen


# ──────────────────────────────────────────────────────────────────────────
# Tiebreak comparator
# ──────────────────────────────────────────────────────────────────────────


def _chain_lt(a: GPUChain, b: GPUChain) -> bool:
    """Strict less-than comparison: first by latency, then by sorted
    stages tuple (deterministic). Used for cross-region best-pick.
    """
    if a.total_latency_ms != b.total_latency_ms:
        return a.total_latency_ms < b.total_latency_ms
    return tuple(sorted(a.stages)) < tuple(sorted(b.stages))
