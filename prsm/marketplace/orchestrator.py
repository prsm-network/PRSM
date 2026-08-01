"""Phase 3 Task 7: MarketplaceOrchestrator.

The integration layer. One entrypoint `orchestrate_sharded_inference`
composes:

  directory (Task 2) → filter (Task 4) → randomizer (Phase 2.1 Line B)
  → price-handshake (Task 5 server + Task 7 client)
  → RemoteShardDispatcher (Phase 2)
  → ReceiptOnlyVerification + optional TEE attestation check (Phase 2.1 Line C)
  → ReputationTracker (Task 6)

Per-shard dispatch loop: try providers from the eligible pool one at a
time, moving on after quote rejection / preemption. A provider that
raises ShardDispatchError (malicious / unrecoverable failure) is
recorded as a failure and the error propagates — that's the caller's
signal to back off entirely. Preempted and quote-rejected providers
are retried on different nodes.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prsm.compute.model_sharding.models import (
    ModelShard,
    PipelineStakeTier,
)
from prsm.compute.remote_dispatcher import (
    MissingAttestationError,
    ShardDispatchError,
    ShardPreemptedError,
)
from prsm.marketplace.directory import MarketplaceDirectory
from prsm.marketplace.errors import (
    JobBudgetExceededError,
    NoEligibleProvidersError,
)
from prsm.marketplace.filter import EligibilityFilter
from prsm.marketplace.listing import ProviderListing
from prsm.marketplace.policy import DispatchPolicy
from prsm.marketplace.price_handshake import PriceNegotiator, PriceQuote
from prsm.marketplace.reputation import ReputationTracker

# Phase 7 Task 5 — on-chain tier gating. Ordering mirrors
# EligibilityFilter._TIER_ORDER — kept in lock-step because we reuse it
# to compare the on-chain effectiveTier string against policy.min_stake_tier.
_TIER_ORDER = {"open": 0, "standard": 1, "premium": 2, "critical": 3}

logger = logging.getLogger(__name__)


def _stake_tier_for(label: str) -> PipelineStakeTier:
    """Map a listing's stake_tier string to the PipelineStakeTier enum
    RemoteShardDispatcher expects."""
    for t in PipelineStakeTier:
        if t.label == label:
            return t
    return PipelineStakeTier.STANDARD


class MarketplaceOrchestrator:
    """Ties the full Phase 3 marketplace flow together.

    Constructor wires all dependencies; the single public entrypoint
    `orchestrate_sharded_inference` runs one inference end-to-end.
    """

    def __init__(
        self,
        identity,
        directory: MarketplaceDirectory,
        eligibility_filter: EligibilityFilter,
        reputation: ReputationTracker,
        price_negotiator: PriceNegotiator,
        remote_dispatcher,
        # Phase 3.1 (Task 7) — optional batched-settlement integration.
        # When both are wired, after each successful remote dispatch the
        # orchestrator forwards a BatchedReceipt to the settlement client.
        # When either is None, the orchestrator behaves as Phase 3 did
        # (local-ledger-only settlement via the dispatcher's escrow flow).
        batch_settlement_client=None,
        provider_address_resolver=None,
        # Phase 7 Task 5 — on-chain tier gating. When wired, the
        # orchestrator verifies StakeBond.effectiveTier(provider) meets
        # policy.min_stake_tier before dispatching. Closes the gap where
        # a provider claims a tier in their listing but hasn't actually
        # bonded stake to back it. When None, gating is skipped and the
        # orchestrator behaves as Phase 3.1 did (listing-claim-only tier).
        stake_manager_client=None,
        # Phase 7.1 Task 5 — redundant-execution consensus. When wired AND
        # policy.consensus_mode is set, _dispatch_one_shard routes to the
        # MultiShardDispatcher for k-way parallel dispatch + consensus
        # voting. When None or when policy.consensus_mode is None, the
        # orchestrator follows the single-provider path (Phase 7 behavior
        # preserved byte-for-byte).
        multi_dispatcher=None,
    ):
        self.identity = identity
        self.directory = directory
        self.eligibility_filter = eligibility_filter
        self.reputation = reputation
        self.price_negotiator = price_negotiator
        self.remote_dispatcher = remote_dispatcher
        self.batch_settlement_client = batch_settlement_client
        self.provider_address_resolver = provider_address_resolver
        self.stake_manager_client = stake_manager_client
        self.multi_dispatcher = multi_dispatcher
        # Phase 7.1: minority receipts from consensus dispatches accumulate
        # here for downstream challenge submission. The orchestrator does
        # NOT submit challenges on-chain itself — challenges require a
        # batch-committed merkle tree + proof, which the settlement-
        # accumulator owns. A higher-level service drains this queue after
        # batches are committed to the registry (Phase 7.1 Task 7 E2E
        # demonstrates the full path; a production challenge-submitter
        # service is Phase 7.1x scope).
        self.consensus_minority_queue: List[Any] = []

    async def orchestrate_sharded_inference(
        self,
        shards: List[ModelShard],
        input_tensor: np.ndarray,
        job_id: str,
        policy: DispatchPolicy,
    ) -> np.ndarray:
        """Run one sharded inference end-to-end across the marketplace.

        Returns the concatenated per-shard outputs (axis 0 — matches
        the Phase 2 integration test's row-parallel assembly).
        """
        listings = self.directory.list_active_providers()
        eligible = self.eligibility_filter.filter(listings, policy)
        if not eligible:
            raise NoEligibleProvidersError(
                f"no eligible providers for job {job_id!r}: "
                f"directory has {len(listings)} listings, filter excluded all"
            )

        # sp1498 — CHEAPEST FIRST. EligibilityFilter deliberately preserves input
        # order, and that input is gossip-ARRIVAL order — so dispatch used to take
        # whichever provider happened to gossip first and pay its asking price.
        # That is not a market: it rewards being early, not being cheap, and it is
        # the selection half of the unbounded-spend problem. The filter's ordering
        # contract is unchanged; the ordering decision belongs here, to the caller.
        # Ties break on provider_id so two requesters with the same directory make
        # the same choice.
        eligible = sorted(
            eligible, key=lambda l: (float(l.price_per_shard_ftns), str(l.provider_id)))

        # sp1498 — PER-JOB ceiling. A per-shard cap alone does not bound a job:
        # N shards at the cap is N times the cap, and N comes from the model, not
        # from the requester. Budget is enforced against the QUOTED prices as they
        # are agreed, in _dispatch_one_shard.
        self._job_budget_ftns = float(
            getattr(policy, "max_total_job_ftns", 0.0) or 0.0)
        self._job_spent_ftns = 0.0

        outputs: List[np.ndarray] = []
        for shard in shards:
            if self._consensus_requested(policy):
                output = await self._dispatch_one_shard_with_consensus(
                    shard=shard,
                    input_tensor=input_tensor,
                    job_id=job_id,
                    eligible=eligible,
                    policy=policy,
                )
            else:
                output = await self._dispatch_one_shard(
                    shard=shard,
                    input_tensor=input_tensor,
                    job_id=job_id,
                    eligible=eligible,
                    policy=policy,
                )
            outputs.append(output)

        return np.concatenate(outputs, axis=0)

    @staticmethod
    def _consensus_requested(policy: DispatchPolicy) -> bool:
        """Phase 7.1 routing guard. Consensus is active only when the
        policy explicitly requests it (not None, not the degenerate
        "single" value that means "skip multi-dispatcher")."""
        return (
            policy.consensus_mode is not None
            and policy.consensus_mode != "single"
        )

    async def _dispatch_one_shard(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        job_id: str,
        eligible: List[ProviderListing],
        policy: DispatchPolicy,
    ) -> np.ndarray:
        """Try providers from the eligible pool one at a time until one
        succeeds. Raises NoEligibleProvidersError if the pool is
        exhausted; raises ShardDispatchError as-is for malicious
        failures (caller decides whether to abort the job)."""
        last_reason: Optional[str] = None
        for listing in eligible:

            # Phase 7 Task 5: on-chain tier gate. A listing-claimed tier
            # can be a lie; StakeBond.effectiveTier is the ground truth.
            # Check BEFORE the price handshake so a cheating provider
            # doesn't get to occupy a quote slot.
            if not self._onchain_tier_meets(listing, policy):
                self.reputation.record_failure(listing.provider_id)
                last_reason = "tier_below_required_onchain"
                continue

            # Step 1: price handshake.
            quote = await self.price_negotiator.request_quote(
                listing=listing,
                shard_index=shard.shard_index,
                shard_size_bytes=len(shard.tensor_data),
                max_acceptable_price_ftns=policy.max_price_per_shard_ftns,
                # sp1499 — the floor was enforced on LISTINGS but never on the
                # QUOTE, which is the number actually escrowed.
                min_acceptable_price_ftns=policy.min_price_per_shard_ftns,
            )
            if quote is None:
                last_reason = "quote_timeout"
                continue
            if not isinstance(quote, PriceQuote):
                # PriceQuoteRejected — try next provider.
                last_reason = f"quote_rejected:{quote.reason}"
                continue

            # sp1498 — enforce the per-shard ceiling against the QUOTE, not just
            # the advertised listing. request_quote passes the ceiling to the
            # provider, but the provider is the one answering: an honest ceiling
            # check on the far side is not a check. Verify locally before escrow.
            quoted = float(quote.quoted_price_ftns)
            if quoted > float(policy.max_price_per_shard_ftns):
                last_reason = (
                    f"quote_over_ceiling:{quoted}>{policy.max_price_per_shard_ftns}")
                logger.warning(
                    "job %s shard %s: provider %s quoted %.6f FTNS, above the "
                    "policy ceiling %.6f — skipping. A provider cannot be trusted "
                    "to enforce the requester's own limit.",
                    job_id[:8], shard.shard_index, listing.provider_id[:12],
                    quoted, policy.max_price_per_shard_ftns)
                continue
            if quoted < 0 or quoted != quoted:
                last_reason = f"quote_not_finite:{quoted}"
                continue

            # sp1498 — and against the per-JOB budget, so N shards at the per-shard
            # cap cannot silently multiply into N times the cap.
            budget = getattr(self, "_job_budget_ftns", 0.0)
            if budget > 0:
                spent = getattr(self, "_job_spent_ftns", 0.0)
                if spent + quoted > budget:
                    raise JobBudgetExceededError(
                        f"job {job_id!r} would spend {spent + quoted:.6f} FTNS "
                        f"(shard {shard.shard_index} quoted {quoted:.6f}) against a "
                        f"budget of {budget:.6f}. Aborting before escrow rather "
                        "than part-paying a job that cannot complete."
                    )

            # Step 2: dispatch the shard under the quoted price.
            # Use dispatch_with_receipt when Phase 3.1 batched settlement
            # is wired so we can forward the verified receipt to the
            # settlement accumulator; otherwise use the Phase 2 dispatch
            # API that returns just the output.
            started = time.time()
            use_batched_settlement = (
                self.batch_settlement_client is not None
                and self.provider_address_resolver is not None
                and getattr(self.identity, "ethereum_address", None) is not None
            )
            try:
                if use_batched_settlement:
                    result = await self.remote_dispatcher.dispatch_with_receipt(
                        shard=shard,
                        input_tensor=input_tensor,
                        node_id=listing.provider_id,
                        job_id=job_id,
                        stake_tier=_stake_tier_for(listing.stake_tier),
                        escrow_amount_ftns=quote.quoted_price_ftns,
                        require_tee_attestation=policy.require_tee,
                    )
                    output = result.output
                    receipt = result.receipt
                else:
                    output = await self.remote_dispatcher.dispatch(
                        shard=shard,
                        input_tensor=input_tensor,
                        node_id=listing.provider_id,
                        job_id=job_id,
                        stake_tier=_stake_tier_for(listing.stake_tier),
                        escrow_amount_ftns=quote.quoted_price_ftns,
                        require_tee_attestation=policy.require_tee,
                    )
                    receipt = None
            except ShardPreemptedError:
                # Honest-work failure — no reputation penalty.
                self.reputation.record_preemption(listing.provider_id)
                last_reason = "preempted"
                continue
            except MissingAttestationError:
                # Provider lied about TEE support in the listing.
                # Treat as a failure (reputation penalty) so the
                # liar gets filtered out on subsequent inferences.
                self.reputation.record_failure(listing.provider_id)
                last_reason = "missing_attestation"
                continue
            except ShardDispatchError as exc:
                self.reputation.record_failure(listing.provider_id)
                raise

            # Success.
            latency_ms = (time.time() - started) * 1000
            self.reputation.record_success(listing.provider_id, latency_ms)
            # sp1498 — count the spend only once the shard actually succeeded, so a
            # failed attempt that was never escrow-released does not consume budget.
            self._job_spent_ftns = getattr(self, "_job_spent_ftns", 0.0) + quoted

            # Phase 3.1 (Task 7): accumulate the verified receipt for
            # batched on-chain settlement, when wired. Silently skip if
            # receipt is empty (fallback path) or addresses aren't
            # resolvable — we never want to block Phase 3 success on an
            # optional settlement failure.
            if use_batched_settlement and receipt:
                await self._accumulate_settlement_receipt(
                    listing=listing,
                    job_id=job_id,
                    shard_index=shard.shard_index,
                    receipt=receipt,
                    quoted_price_ftns=quote.quoted_price_ftns,
                )

            return output

        # Exhausted the eligible pool without a success.
        raise NoEligibleProvidersError(
            f"shard {shard.shard_index} of job {job_id!r} exhausted the "
            f"eligible pool (size={len(eligible)}) without success; "
            f"last failure reason: {last_reason or 'unknown'}"
        )

    async def _dispatch_one_shard_with_consensus(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        job_id: str,
        eligible: List[ProviderListing],
        policy: DispatchPolicy,
    ) -> np.ndarray:
        """Phase 7.1 consensus path. Runs one shard on k providers
        concurrently and returns the majority-agreed output.

        Flow:
          1. Apply the Phase 7 on-chain tier gate to the eligible pool.
          2. Rank the survivors by (reputation * 1/price) and pick top-k.
          3. Hand the k node_ids to MultiShardDispatcher.
          4. Record reputation outcomes: successes for majority,
             failures for minority.
          5. Queue minority receipts for challenge (no on-chain submission
             here — that's a post-batch-commit concern).
          6. Return the agreed ndarray.

        Preconditions: self.multi_dispatcher is not None (checked at
        the routing decision in orchestrate_sharded_inference via
        _consensus_requested → MultiDispatcher presence).
        """
        if self.multi_dispatcher is None:
            raise RuntimeError(
                "consensus_mode requested but multi_dispatcher is not wired"
            )

        # Step 1: tier gate. Reuse the single-path helper — any listing
        # whose on-chain tier doesn't meet the policy's floor is excluded
        # BEFORE we pay the asyncio.gather cost on k dispatches.
        tier_eligible = [
            l for l in eligible if self._onchain_tier_meets(l, policy)
        ]
        for excluded in eligible:
            if excluded not in tier_eligible:
                self.reputation.record_failure(excluded.provider_id)

        # Step 2: select top-k.
        k = policy.consensus_k
        if len(tier_eligible) < k:
            raise NoEligibleProvidersError(
                f"consensus_mode={policy.consensus_mode!r} k={k} needs at "
                f"least {k} eligible providers after tier gating; got "
                f"{len(tier_eligible)}"
            )
        selected = self._select_top_k(
            tier_eligible, k, policy.requester_priority_boost,
        )

        # Step 3: fair escrow split across the k. MVP uses the policy's
        # max_price_per_shard_ftns as the budget (no per-provider price
        # handshake in consensus path — price discovery in k-of-n is
        # deferred to Phase 7.1x per design §3.3). If the policy has no
        # ceiling, fall back to the maximum listing price among selected.
        total_budget = (
            policy.max_price_per_shard_ftns
            if policy.max_price_per_shard_ftns != float("inf")
            else max(l.price_per_shard_ftns for l in selected) * k
        )
        per_provider_escrow = total_budget / k

        # Step 4: dispatch via multi-dispatcher.
        from prsm.compute.multi_dispatcher import (  # lazy import
            ConsensusFailedError,
            InsufficientResponsesError,
        )
        started = time.time()
        try:
            receipt = await self.multi_dispatcher.dispatch_with_consensus(
                shard=shard,
                input_tensor=input_tensor,
                node_ids=[l.provider_id for l in selected],
                job_id=job_id,
                stake_tier=_stake_tier_for(selected[0].stake_tier),
                escrow_amount_per_provider_ftns=per_provider_escrow,
                require_tee_attestation=policy.require_tee,
            )
        except (InsufficientResponsesError, ConsensusFailedError) as exc:
            # Both failures are recorded as failures against the whole
            # selected set. A production challenge-submitter may treat
            # ConsensusFailed more aggressively (challenge every
            # disagreer), but that needs the actual receipts we don't
            # have in the exception. MVP: blanket-fail and surface.
            for listing in selected:
                self.reputation.record_failure(listing.provider_id)
            raise NoEligibleProvidersError(
                f"consensus dispatch failed for shard {shard.shard_index} "
                f"of job {job_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

        # Step 5: reputation updates.
        latency_ms = (time.time() - started) * 1000
        majority_ids = {r.provider_id for r in receipt.majority_receipts}
        minority_ids = {r.provider_id for r in receipt.minority_receipts}
        # Providers who failed to respond at all (neither majority nor
        # minority) pass the partial-response logic in MultiDispatcher;
        # they're NOT in `receipt`. We don't record a failure for them
        # here — preemption/TEE-missing are honest-work or advertising
        # concerns handled inside the single dispatcher already.
        for provider_id in majority_ids:
            self.reputation.record_success(provider_id, latency_ms)
        for provider_id in minority_ids:
            self.reputation.record_failure(provider_id)

        # Step 6: queue minority for downstream challenge.
        if receipt.minority_receipts:
            # Convert per-provider FTNS price (float) to wei (uint128) so
            # the submitter can build ReceiptLeafFields without having
            # to know the policy. Queue entry carries the numeric value
            # the challenge tx eventually asserts in the leaf.
            per_provider_wei = int(round(per_provider_escrow * 10**18))
            self._queue_consensus_challenges(receipt, per_provider_wei)

        return receipt.agreed_output

    @staticmethod
    def _select_top_k(
        listings: List[ProviderListing], k: int,
        priority_boost: float = 0.0,
    ) -> List[ProviderListing]:
        """Phase 7.1 Task 5 top-k selection.

        Rank by `reputation_proxy * (1 / price)`. Since the orchestrator's
        ReputationTracker holds the actual score and selection needs to be
        pure / testable, we use the listing's advertised stake_tier as
        reputation proxy for MVP:
          - open=1, standard=2, premium=3, critical=4
        Multiply by (1 / price) and take the top k. Ties broken by
        lexicographic provider_id (deterministic — the requester can
        audit why a given set was picked).

        sp906 — staking priority access. When ``priority_boost`` > 0 (a
        requester with an active staking lock, see DispatchPolicy
        .requester_priority_boost), the score is scaled by
        ``(1 + priority_boost * capacity_norm)`` where capacity_norm is
        the provider's capacity relative to the fastest in the set. This
        biases selection toward faster providers — "priority access" =
        access to faster service — without overriding tier/price for
        unboosted (boost=0 → multiplier 1.0 → identical to pre-sp906)
        requests. The influence is bounded (normalized, scaled by boost).
        """
        tier_score = {"open": 1, "standard": 2, "premium": 3, "critical": 4}
        # sp926 — defensively clamp capacity to >= 0 in the boost math so a
        # negative (or otherwise unvalidated) capacity that slipped past
        # verify_listing can't produce a negative/scrambled cap_norm. The
        # upper-bound check lives in verify_listing (ingestion).
        def _cap(l: "ProviderListing") -> float:
            return max(float(l.capacity_shards_per_sec), 0.0)

        max_cap = max((_cap(l) for l in listings), default=0.0)

        def score(listing: ProviderListing) -> Tuple[float, str]:
            tier = tier_score.get(listing.stake_tier, 0)
            # Guard against division by zero even though the filter's
            # min_price_per_shard_ftns rejects <=0 listings already.
            price = max(listing.price_per_shard_ftns, 1e-9)
            base = tier / price
            if priority_boost > 0 and max_cap > 0:
                cap_norm = _cap(listing) / max_cap
                base = base * (1.0 + priority_boost * cap_norm)
            # Negate because we'll sort ascending by (score, then
            # ascending by provider_id — stable tiebreak).
            return (-base, listing.provider_id)

        sorted_listings = sorted(listings, key=score)
        return sorted_listings[:k]

    def drain_consensus_minority_queue(self) -> List[Dict[str, Any]]:
        """Phase 7.1x — pop all pending minority-receipt entries off the
        in-process queue. Returns them in FIFO order for a downstream
        `ConsensusChallengeSubmitter` to turn into on-chain challenges.

        This is the orchestrator's side of the §8.6 seam the Task 8
        review flagged. The submitter service calls this method after
        the settlement accumulator commits the minority batches on-
        chain, pairs each entry with the committed batch_ids, and fires
        the `CONSENSUS_MISMATCH` challenges.

        Idempotent + crash-atomic from the orchestrator's perspective:
        once drained, entries are the caller's responsibility. If the
        submitter crashes mid-drain, the orchestrator's queue is
        already empty and those minority receipts are lost. Persistence
        is future work (a Phase 7.1x.next follow-up); MVP assumes the
        submitter runs in the same process as the orchestrator or
        handles its own persistence after drain.
        """
        drained = list(self.consensus_minority_queue)
        self.consensus_minority_queue.clear()
        return drained

    def _queue_consensus_challenges(
        self, receipt, value_ftns_per_provider_wei: int,
    ) -> None:
        """Phase 7.1 Task 5 — accumulate minority receipts for the
        downstream challenge-submitter service to consume.

        No on-chain submission here. The orchestrator runs inside the
        dispatch path; challenges are a post-batch-commit action that
        needs the Merkle tree + proof, which only the settlement
        accumulator has once the batch is committed. A production
        challenge-submitter drains this queue after batch commit.

        Entry shape: (job_id, shard_index, agreed_output_hash,
        majority_receipts, minority_receipts, value_ftns_per_provider_wei).
        The per-provider wei value is carried here so the submitter
        (or ConsensusChallengeQueue) can build ReceiptLeafFields
        without knowing the dispatch policy. The submitter service
        supplies batch_ids when draining.
        """
        self.consensus_minority_queue.append({
            "job_id": receipt.job_id,
            "shard_index": receipt.shard_index,
            "agreed_output_hash": receipt.agreed_output_hash,
            "majority_receipts": receipt.majority_receipts,
            "minority_receipts": receipt.minority_receipts,
            "value_ftns_per_provider_wei": value_ftns_per_provider_wei,
        })
        logger.info(
            f"queued {len(receipt.minority_receipts)} consensus-mismatch "
            f"challenge(s) for job {receipt.job_id!r} shard "
            f"{receipt.shard_index}"
        )

    def _onchain_tier_meets(
        self,
        listing: ProviderListing,
        policy: DispatchPolicy,
    ) -> bool:
        """Phase 7 Task 5 — Return True iff the listing should be allowed
        to serve the job under the on-chain tier policy.

        Three short-circuits preserve the additive integration pattern:
          - If the policy's min_stake_tier is "open" (the default), any
            on-chain tier is acceptable; skip the RPC.
          - If stake_manager_client OR provider_address_resolver is None,
            the orchestrator is not wired for Phase 7 — fall back to the
            listing-claim tier (already enforced by EligibilityFilter).
          - If the resolver can't map the provider to an Ethereum address,
            skip the check (providers without chain identity predate
            Phase 7; the listing filter has already approved them).

        If the RPC itself raises, we fail OPEN with a loud warning.
        Closed-fail here would let an attacker DoS the StakeBond RPC to
        block all dispatch — and the listing filter plus slashing-on-
        misbehavior still protect against the cheat case. Operators
        watching the warning log can harden to fail-closed later.
        """
        if policy.min_stake_tier == "open":
            return True
        if self.stake_manager_client is None or self.provider_address_resolver is None:
            return True

        provider_eth = self.provider_address_resolver(listing.provider_id)
        if provider_eth is None:
            return True

        try:
            onchain_tier = self.stake_manager_client.effective_tier(
                provider_eth
            )
        except Exception as exc:
            logger.warning(
                f"on-chain tier check failed for provider "
                f"{listing.provider_id[:12]}…: "
                f"{type(exc).__name__}: {exc}; failing open"
            )
            return True

        required_ord = _TIER_ORDER.get(policy.min_stake_tier, -1)
        actual_ord = _TIER_ORDER.get(onchain_tier, -1)
        if actual_ord < required_ord:
            logger.info(
                f"provider {listing.provider_id[:12]}… rejected: on-chain "
                f"tier {onchain_tier!r} below required {policy.min_stake_tier!r} "
                f"(listing claimed {listing.stake_tier!r})"
            )
            return False
        return True

    async def _accumulate_settlement_receipt(
        self,
        listing: ProviderListing,
        job_id: str,
        shard_index: int,
        receipt: Dict[str, Any],
        quoted_price_ftns: float,
    ) -> None:
        """Phase 3.1 (Task 7) — build a BatchedReceipt and forward to
        the settlement accumulator. Lazy imports so callers without
        Phase 3.1 wiring don't pay the import cost.

        Never raises: if the provider address resolver returns None, or
        the settlement client's accumulate fails, we log and swallow.
        Phase 3.1 batched settlement is strictly additive; its failures
        must not block Phase 3 dispatch success."""
        try:
            from prsm.compute.shard_receipt import ShardExecutionReceipt
            from prsm.settlement.accumulator import BatchedReceipt

            provider_eth = self.provider_address_resolver(listing.provider_id)
            if provider_eth is None:
                logger.debug(
                    f"batched settlement skipped for provider "
                    f"{listing.provider_id[:12]}…: no Ethereum address resolved"
                )
                return

            # Reconstruct a ShardExecutionReceipt dataclass from the
            # verified receipt dict so the settlement layer's canonical-
            # form encoding (Task 5) can reference typed fields.
            shard_receipt = ShardExecutionReceipt(
                job_id=receipt.get("job_id", ""),
                shard_index=receipt.get("shard_index", shard_index),
                provider_id=receipt.get("provider_id", ""),
                provider_pubkey_b64=receipt.get("provider_pubkey_b64", ""),
                output_hash=receipt.get("output_hash", ""),
                executed_at_unix=receipt.get("executed_at_unix", 0),
                signature=receipt.get("signature", ""),
                tee_attestation=receipt.get("tee_attestation"),
            )

            # Value field is in FTNS base units (wei, 18 decimals). The
            # orchestrator receives the quoted price as a float FTNS
            # value per Phase 3's price-handshake semantics.
            value_ftns_wei = int(round(quoted_price_ftns * 10**18))

            br = BatchedReceipt(
                receipt=shard_receipt,
                requester_address=self.identity.ethereum_address,
                provider_address=provider_eth,
                value_ftns=value_ftns_wei,
                local_escrow_id=f"{job_id}:shard:{shard_index}",
            )
            await self.batch_settlement_client.accumulate(br)
        except Exception as exc:
            logger.warning(
                f"batched settlement accumulate failed for shard "
                f"{shard_index} of job {job_id!r}: "
                f"{type(exc).__name__}: {exc}"
            )
