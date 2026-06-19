"""QueryOrchestrator — SwarmDispatcher adapter.

Production wiring for the `SwarmDispatcher` Protocol declared in
`swarm_runner.py`. Wraps the existing requester-side
`prsm/compute/agents/dispatcher.py::AgentDispatcher` so the
QueryOrchestrator can fan WASM agents out across shard candidates
through the bid-based dispatch lifecycle that already exists.

Per-shard flow inside `fan_out(manifest, shards)`:

    1. Build a MobileAgent that carries the InstructionManifest +
       per-shard FTNS budget + WASM executor binary.
    2. await agent_dispatcher.dispatch(agent) — gossip-publishes the
       manifest, opens escrow, kicks off bidding.
    3. await agent_dispatcher.wait_for_result(agent.agent_id, timeout)
       — blocks until the chosen provider returns the agent's signed
       output, or until timeout.
    4. Convert the result dict to a `PartialResult`. Skip the shard
       if the wait timed out — the orchestrator's retry-loop owns
       retry policy.

What this adapter does NOT do:

  - Apply DP noise. The source agent invokes `dp_noise.py` primitives
    on its WASM output BEFORE signing, then sets `dp_noise_applied`
    in its reply. This adapter just threads the marker forward to
    the PartialResult; `swarm_runner._enforce_a5_dp_noise` does the
    actual A5 gating downstream.

  - Sign partials. The source agent signs its own output with its
    Ed25519 identity key.

  - Verify signatures. Aggregator-side verification covers per-partial
    signatures (per `docs/2026-05-08-aggregate-rpc-design.md` §"A5
    enforcement at server side"). The runner's A5 marker check is a
    fail-fast hint, not the security boundary.

  - Bid selection. `AgentDispatcher` already owns scoring + selection
    via `_select_best_bid`. The adapter is structurally orthogonal to
    bid logic.

The adapter is intentionally thin — keep it that way. Any logic
heavier than "construct agent, await dispatch, await result, convert"
belongs upstream (orchestrator) or downstream (dispatcher / source
agent / aggregator).

Threat-model coverage anchored in this layer: NONE. A1/A2/A6/A7/A8/A10
land at aggregator_selector. A3/A4 land at the retry-loop shell. A5
+ A9 land at swarm_runner. The adapter is purely transport.

Refs:
  - prsm/compute/query_orchestrator/swarm_runner.py — Protocol shape
  - prsm/compute/agents/dispatcher.py — wrapped class
  - prsm/compute/agents/models.py — MobileAgent / DispatchRecord shapes
  - docs/2026-05-08-aggregate-rpc-design.md §B4
"""
from __future__ import annotations

import logging
import uuid
from typing import Sequence

from prsm.compute.agents.dispatcher import AgentDispatcher
from prsm.compute.agents.instruction_set import InstructionManifest
from prsm.compute.agents.models import AgentManifest, MobileAgent
from prsm.compute.query_orchestrator.shard_finder import ShardCandidate
from prsm.compute.query_orchestrator.swarm_runner import PartialResult

logger = logging.getLogger(__name__)


# Default per-shard agent TTL. The bidding window inside
# AgentDispatcher is bounded separately; this is the upper-bound on
# the total agent lifetime once it lands on a provider.
DEFAULT_TTL_SECONDS = 60

# Default per-shard wait timeout for `wait_for_result`. Distinct from
# AgentDispatcher.result_timeout — the adapter takes a per-call
# override so the orchestrator can budget wall-time across the swarm.
DEFAULT_RESULT_TIMEOUT_SECONDS = 30.0


class SwarmDispatcherAdapter:
    """Satisfies SwarmDispatcher Protocol by orchestrating per-shard
    MobileAgent dispatch via the existing AgentDispatcher.

    Parameters
    ----------
    agent_dispatcher:
        Instance of `prsm/compute/agents/dispatcher.py::AgentDispatcher`
        responsible for the bid-based requester-side lifecycle.
    wasm_executor_binary:
        Pre-built WASM executor that interprets the InstructionManifest
        on the target node. Same binary for every shard; the per-shard
        variation lives in the manifest passed to `fan_out`.
    per_shard_budget_ftns:
        FTNS budget allocated to a single shard's agent. The
        orchestrator typically computes this as
        `total_query_budget // len(shards)` upstream.
    ttl_seconds:
        Per-agent TTL (seconds). Defaults to 60s.
    result_timeout_seconds:
        Per-shard `wait_for_result` timeout in seconds. Defaults to
        30s. The orchestrator should pass a tighter value if it has
        less wall-time budget across the swarm.
    """

    def __init__(
        self,
        agent_dispatcher: AgentDispatcher,
        *,
        wasm_executor_binary: bytes,
        per_shard_budget_ftns: int,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        result_timeout_seconds: float = DEFAULT_RESULT_TIMEOUT_SECONDS,
    ) -> None:
        if per_shard_budget_ftns <= 0:
            raise ValueError(
                f"per_shard_budget_ftns must be positive, "
                f"got {per_shard_budget_ftns}"
            )
        if ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be positive, got {ttl_seconds}"
            )
        if result_timeout_seconds <= 0:
            raise ValueError(
                f"result_timeout_seconds must be positive, "
                f"got {result_timeout_seconds}"
            )
        if not wasm_executor_binary:
            raise ValueError("wasm_executor_binary must be non-empty")

        self._dispatcher = agent_dispatcher
        self._wasm_binary = wasm_executor_binary
        self._budget = per_shard_budget_ftns
        self._ttl = ttl_seconds
        self._result_timeout = result_timeout_seconds

    # ── Protocol method ─────────────────────────────────────────────

    async def fan_out(
        self,
        manifest: InstructionManifest,
        shards: Sequence[ShardCandidate],
    ) -> list[PartialResult]:
        """Dispatch one MobileAgent per shard, await results, return
        successful PartialResults in shard-order.

        Shards whose `wait_for_result` returns None (timeout) are
        skipped — the orchestrator's retry-loop shell decides whether
        to re-poll the selector or fail the query.
        """
        if not shards:
            return []

        partials: list[PartialResult] = []
        for shard in shards:
            agent = self._build_agent(manifest, shard)

            # Open the dispatch (gossip-publish manifest, escrow hold,
            # bidding starts).
            await self._dispatcher.dispatch(agent)

            # Block for the chosen provider's signed result, or skip
            # this shard on timeout.
            result = await self._dispatcher.wait_for_result(
                agent.agent_id,
                timeout=self._result_timeout,
            )
            if result is None:
                logger.info(
                    "shard %s timed out (no result within %.1fs) — "
                    "skipping; orchestrator retry-loop owns retries",
                    shard.cid,
                    self._result_timeout,
                )
                continue

            partials.append(self._to_partial(shard, result))

        return partials

    # ── Internals ───────────────────────────────────────────────────

    def _build_agent(
        self,
        instruction_manifest: InstructionManifest,
        shard: ShardCandidate,
    ) -> MobileAgent:
        """Construct a per-shard MobileAgent carrying:
          - resource declaration (AgentManifest)
          - configured FTNS budget + executor binary
          - per-shard agent_id

        The InstructionManifest (DSL payload the WASM executor will
        consume via `to_wasm_input()`) is NOT carried on
        `MobileAgent.manifest` — that field is an `AgentManifest`
        declaring resource needs, not a DSL payload. The InstructionManifest
        is delivered to the WASM executor through a separate channel
        the dispatcher's transfer flow owns (see follow-on note below).

        Type unification: this commit closes the B4 teammate's
        flagged divergence — `agent.manifest.to_dict()` (called by
        the dispatcher's gossip payload at
        `prsm/compute/agents/dispatcher.py:135`) now correctly
        produces an `AgentManifest` dict, not an `InstructionManifest`
        dict.

        Follow-on (separate sprint, NOT in this commit): wire the
        InstructionManifest to the target WASM executor at agent
        transfer time. The orchestrator's swarm_runner already passes
        the InstructionManifest into `fan_out`; the dispatcher's
        transfer pathway needs a sidecar field to carry the
        `to_wasm_input()` bytes alongside the WASM binary itself.
        Until that lands, the WASM executor on the target node
        receives no DSL — production deployment requires the
        orchestrator-wiring task to bridge that gap.
        """
        agent_manifest = AgentManifest(
            required_content_ids=[shard.cid],
            # Tier inherited from shard's source listing if available
            # (per Vision §6 the shard holder's tier was verified at
            # marketplace ingestion). Default to "t1" for v1 — Tier C
            # constraints land at the aggregator-selector via
            # `requires_tee` rather than at agent-dispatch time.
            min_hardware_tier="t1",
            max_memory_bytes=256 * 1024 * 1024,
            max_execution_seconds=60,
            max_output_bytes=10 * 1024 * 1024,
            required_capabilities=[],
        )
        return MobileAgent(
            agent_id=str(uuid.uuid4()),
            wasm_binary=self._wasm_binary,
            manifest=agent_manifest,
            origin_node="",  # filled in by dispatcher's create_agent path
            signature="",
            ftns_budget=self._budget,
            ttl=self._ttl,
        )

    @staticmethod
    def _to_partial(
        shard: ShardCandidate,
        result: dict,
    ) -> PartialResult:
        """Convert an AgentDispatcher result dict to a PartialResult.

        Field threading:
          - shard_cid + creator_id: from the shard candidate
          - payload + agent_signature: from the result (source agent's
            WASM output + identity-key sig over it)
          - dp_noise_applied: from the result, defaulting to False if
            the source agent didn't set the marker. Defaulting to
            False is the fail-safe: the runner's A5 gate will reject
            the swarm and the orchestrator's retry-loop will refund.
          - source_agent_pubkey: 32-byte Ed25519 public key the source
            agent signed payload+manifest with. Defaults to 32 zero
            bytes if the result dict omits it (back-compat with older
            dispatchers that don't surface the agent's identity key —
            aggregator-side signature verification will fail loudly,
            which is the correct fail-loud behavior).
          - privacy_budget_consumed: epsilon spent applying DP noise.
            Defaults to 0.0 — the aggregator's sum_privacy_budgets
            ceiling check is permissive when no budget is reported.
          - pcu_consumed: the agent's measured PCU (Provider Compute
            Units), sourced from the result dict's ``"pcu"`` key
            (emitted by AgentExecutor via WASMExecutionResult.pcu()).
            Sprint 1178 — pre-fix this was NOT extracted, so it
            defaulted to 0.0 all the way to ParticipantAttribution,
            and compute_split_amounts (which PCU-weights only when
            ALL participants report pcu > 0) ALWAYS fell back to a
            uniform split — the PCU-weighted settlement attribution
            was inert. Now threaded so a swarm where every agent
            reports real PCU settles proportional to compute spent.
            Defaults to 0.0 (→ uniform fallback) when unreported.
        """
        # Sprint 176 — derive source_agent_pubkey with priority:
        #   1. result["source_agent_pubkey"]: explicit 32-byte raw
        #      bytes from a dispatcher that surfaces the agent's
        #      signing key directly.
        #   2. result["provider_public_key"]: base64-encoded
        #      Ed25519 pubkey from `AgentExecutor` (the canonical
        #      v1 path — same as source_agent_pubkey since the
        #      provider IS the source agent in Phase 3).
        #   3. 32 zero bytes — fail-loud default; aggregator-side
        #      signature verification will reject.
        # Pre-sprint-176 only path (1) was honored; path (2) was
        # the v1 placeholder gap from the wiring-complete MEMORY
        # entry's deferred-followups list.
        _pubkey_raw = result.get("source_agent_pubkey")
        if _pubkey_raw is None:
            _b64 = result.get("provider_public_key")
            if _b64:
                try:
                    import base64 as _b64mod
                    _pubkey_raw = _b64mod.b64decode(_b64)
                except (ValueError, TypeError):
                    _pubkey_raw = b"\x00" * 32
            else:
                _pubkey_raw = b"\x00" * 32
        # Defensive: enforce 32-byte width.
        _pubkey_raw = bytes(_pubkey_raw)
        if len(_pubkey_raw) != 32:
            _pubkey_raw = b"\x00" * 32

        return PartialResult(
            shard_cid=shard.cid,
            payload=result["payload"],
            agent_signature=result["agent_signature"],
            creator_id=shard.creator_id,
            dp_noise_applied=bool(result.get("dp_noise_applied", False)),
            source_agent_pubkey=_pubkey_raw,
            privacy_budget_consumed=float(
                result.get("privacy_budget_consumed", 0.0)
            ),
            # Sp1178 — thread the agent's measured PCU (result key
            # "pcu", per AgentExecutor) so settlement can PCU-weight
            # the compute split. Was omitted → always 0.0 → uniform.
            pcu_consumed=float(result.get("pcu", 0.0) or 0.0),
        )
