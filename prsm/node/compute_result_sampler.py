"""
Compute Result Sampler — optimistic verification for the single-provider pay path
=================================================================================

The single-provider compute pay path (``ComputeRequester._on_job_result``) pays
a provider on a VALID SIGNATURE ALONE. The signature proves *who* produced the
result, not that the work was actually done — a provider can sign a
well-formed-but-fabricated result and collect payment (compute/inference review
finding #2). The marketplace k-of-n path already runs consensus + challenge +
slash, but the single-provider path has had zero verification.

This module closes the gap with **optimistic verification**, the standard
decentralized-compute defense:

  * Pay optimistically (the hot path is untouched — the local ledger move stays
    immediate, so there is no clawback to engineer and no new HELD state to
    break the sp924 idempotency guard).
  * With probability ``sample_rate``, re-execute the same job on an INDEPENDENT
    provider and compare outputs.
  * On a mismatch, escalate to a tiebreaker so the **outlier** (the liar) is
    identified by majority — slashing the wrong provider on a bare 1-vs-1
    disagreement would itself be a critical bug, so a mismatch with no majority
    is INCONCLUSIVE and penalizes nobody.
  * Penalty is reputation-always; a *bonded* provider additionally gets a
    ``CONSENSUS_MISMATCH`` challenge routed into the existing on-chain slash
    rails. (An unbonded provider cannot be slashed on-chain — ``StakeBond.slash``
    reverts silently — so its only deterrent is reputation / eligibility.)

Economics: steady-state overhead is ``sample_rate`` x the job's compute. A
provider that fabricates is caught with probability ~``sample_rate`` per job and
slashed; for any bonded stake worth more than ``1 / sample_rate`` jobs, fraud is
-EV. A provider that tries to return a *correct* output to dodge detection has
to do the real work anyway, which is the point.

Every external effect is injected (re-executor, reputation sink, challenge sink,
RNG, comparator) so the orchestrator is fully unit-testable without a multi-node
bench. Production wires thin adapters over the live re-dispatch, reputation and
challenge-queue machinery.

NOTE on comparison: the default comparator is exact output-hash equality, which
is correct for deterministic (greedy / temperature-0) decoding. Non-greedy or
heterogeneous-hardware execution can drift; supply a tolerance comparator
(logprob / similarity) for those policies. The verdict logic deliberately never
slashes without a strict majority, so honest drift degrades to INCONCLUSIVE
rather than a false-positive slash.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Conservative default: 1-in-20 jobs verified. A no-op on single-provider
# setups (the re-executor finds nobody else → SKIPPED). Operators tune via
# the PRSM_COMPUTE_VERIFY_SAMPLE_RATE env var at the wiring layer.
DEFAULT_SAMPLE_RATE = 0.05


@dataclass(frozen=True)
class ReExecResult:
    """One independent re-execution of a sampled job.

    sp957: ``bonded``/``stake_wei``/``operator_address`` carry the participant's
    on-chain stake posture so that, when this participant is the outlier, the
    routed CONSENSUS_MISMATCH evidence is self-describing — a downstream
    authority can size/route an on-chain challenge without re-querying chain
    state. They are *descriptive only*: nothing in this module moves stake."""
    provider_id: str
    output_hash: str
    bonded: bool = False
    stake_wei: int = 0
    operator_address: Optional[str] = None


class VerificationVerdict(str, Enum):
    SKIPPED = "skipped"            # no independent provider available — cannot verify
    VERIFIED = "verified"         # the original agreed with the established majority
    MISMATCH = "mismatch"         # the original was the outlier — penalized
    INCONCLUSIVE = "inconclusive"  # no majority among samples — flagged, nobody penalized
    ERROR = "error"               # re-execution raised — logged, nobody penalized


@dataclass
class VerificationOutcome:
    job_id: str
    verdict: VerificationVerdict
    original_provider_id: str
    original_output_hash: str
    samples: List[ReExecResult] = field(default_factory=list)
    penalized_provider_ids: List[str] = field(default_factory=list)
    majority_output_hash: Optional[str] = None
    detail: str = ""


# Injected effect signatures (documentation):
#   re_executor:    async (job_type, payload, exclude: Set[str]) -> Optional[ReExecResult]
#   reputation_sink: (provider_id: str, outcome: VerificationOutcome) -> None
#   challenge_sink:  async (evidence: dict) -> None
#   comparator:     (hash_a: str, hash_b: str) -> bool
#   roll:           () -> float in [0, 1)
ReExecutor = Callable[[Any, Dict[str, Any], Set[str]], Awaitable[Optional[ReExecResult]]]


class ComputeResultSampler:
    def __init__(
        self,
        *,
        re_executor: ReExecutor,
        reputation_sink: Optional[Callable[[str, VerificationOutcome], None]] = None,
        challenge_sink: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        comparator: Optional[Callable[[str, str], bool]] = None,
        roll: Optional[Callable[[], float]] = None,
        max_tiebreakers: int = 1,
    ):
        self._re_executor = re_executor
        self._reputation_sink = reputation_sink
        self._challenge_sink = challenge_sink
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self._comparator = comparator or (lambda a, b: a == b)
        # Default RNG is a CSPRNG so a provider cannot predict whether a given
        # job will be verified (a job-id-derived decision would let a provider
        # behave honestly only on jobs it computes will be sampled).
        self._roll = roll or (lambda: secrets.randbelow(1_000_000) / 1_000_000)
        self.max_tiebreakers = max(0, int(max_tiebreakers))
        self._stats = {
            "sampled": 0, "verified": 0, "mismatch": 0,
            "inconclusive": 0, "skipped": 0, "error": 0, "penalized": 0,
        }

    # ── sampling decision ────────────────────────────────────────────────

    def should_sample(self) -> bool:
        """Unpredictable, rate-respecting decision. rate<=0 never, rate>=1 always."""
        if self.sample_rate <= 0.0:
            return False
        if self.sample_rate >= 1.0:
            return True
        return self._roll() < self.sample_rate

    # ── verification ──────────────────────────────────────────────────────

    async def verify(
        self,
        *,
        job_id: str,
        job_type: Any,
        payload: Dict[str, Any],
        original_provider_id: str,
        original_output_hash: str,
        original_bonded: bool = False,
        original_stake_wei: int = 0,
        original_operator_address: Optional[str] = None,
    ) -> VerificationOutcome:
        """Re-execute on independent providers and adjudicate the original."""
        self._stats["sampled"] += 1
        outcome = VerificationOutcome(
            job_id=job_id,
            verdict=VerificationVerdict.SKIPPED,
            original_provider_id=original_provider_id,
            original_output_hash=original_output_hash,
        )

        exclude: Set[str] = {original_provider_id}
        try:
            first = await self._re_executor(job_type, payload, set(exclude))
        except Exception as e:  # re-dispatch must never crash the node
            logger.warning("sp928: re-execution failed for job %s: %s", job_id[:8], e)
            outcome.verdict = VerificationVerdict.ERROR
            outcome.detail = str(e)
            self._stats["error"] += 1
            return outcome

        if first is None:
            # No independent provider available (single-node / single-provider).
            outcome.detail = "no independent provider available"
            self._stats["skipped"] += 1
            return outcome

        outcome.samples.append(first)
        exclude.add(first.provider_id)

        if self._comparator(original_output_hash, first.output_hash):
            outcome.verdict = VerificationVerdict.VERIFIED
            outcome.majority_output_hash = original_output_hash
            self._stats["verified"] += 1
            return outcome

        # Disagreement: gather tiebreakers to find the majority / outlier.
        for _ in range(self.max_tiebreakers):
            try:
                extra = await self._re_executor(job_type, payload, set(exclude))
            except Exception as e:
                logger.warning("sp928: tiebreaker re-execution failed for job %s: %s", job_id[:8], e)
                break
            if extra is None:
                break
            outcome.samples.append(extra)
            exclude.add(extra.provider_id)

        original = ReExecResult(
            provider_id=outcome.original_provider_id,
            output_hash=outcome.original_output_hash,
            bonded=original_bonded,
            stake_wei=original_stake_wei,
            operator_address=original_operator_address,
        )
        return await self._adjudicate(outcome, original)

    async def _adjudicate(
        self, outcome: VerificationOutcome, original: ReExecResult
    ) -> VerificationOutcome:
        """Group all participants by output and penalize outliers vs the majority."""
        # Participants: the original + every re-execution sample. Each carries
        # (provider_id, output_hash, bonded, stake_wei, operator_address).
        participants: List[ReExecResult] = [original] + list(outcome.samples)
        total = len(participants)

        # Group by output hash (first-match grouping so a tolerance comparator
        # still clusters near-equal outputs).
        groups: List[Dict[str, Any]] = []  # {hash, members: List[ReExecResult]}
        for p in participants:
            placed = False
            for g in groups:
                if self._comparator(g["hash"], p.output_hash):
                    g["members"].append(p)
                    placed = True
                    break
            if not placed:
                groups.append({"hash": p.output_hash, "members": [p]})

        largest = max(groups, key=lambda g: len(g["members"]))
        # Strict majority required to establish "truth" — never slash on a tie
        # or a bare 1-vs-1 disagreement.
        if len(largest["members"]) * 2 <= total:
            outcome.verdict = VerificationVerdict.INCONCLUSIVE
            outcome.detail = "no strict majority among re-executions"
            self._stats["inconclusive"] += 1
            return outcome

        truth_hash = largest["hash"]
        outcome.majority_output_hash = truth_hash
        truth_ids = {m.provider_id for m in largest["members"]}

        # Penalize every participant outside the majority (could be the original
        # or a lying re-execution provider).
        outliers = [p for p in participants if p.provider_id not in truth_ids]
        for p in outliers:
            outcome.penalized_provider_ids.append(p.provider_id)
            self._stats["penalized"] += 1
            self._penalize_reputation(p.provider_id, outcome)
            if p.bonded:
                await self._route_challenge(outcome, accused=p, truth_hash=truth_hash)

        # The verdict describes the ORIGINAL: was it the honest majority or the
        # outlier? (Lying re-execution providers are still penalized above.)
        original_is_outlier = outcome.original_provider_id not in truth_ids
        if original_is_outlier:
            outcome.verdict = VerificationVerdict.MISMATCH
            self._stats["mismatch"] += 1
        else:
            outcome.verdict = VerificationVerdict.VERIFIED
            self._stats["verified"] += 1
        return outcome

    # ── penalty routing ──────────────────────────────────────────────────

    def _penalize_reputation(self, provider_id: str, outcome: VerificationOutcome) -> None:
        if self._reputation_sink is None:
            return
        try:
            self._reputation_sink(provider_id, outcome)
        except Exception as e:  # observability sink failure must not abort
            logger.warning("sp928: reputation sink failed for %s: %s", provider_id[:8], e)

    async def _route_challenge(
        self, outcome: VerificationOutcome, *, accused: ReExecResult, truth_hash: str
    ) -> None:
        if self._challenge_sink is None:
            return
        witnesses = [
            s.provider_id for s in outcome.samples
            if self._comparator(s.output_hash, truth_hash)
        ]
        if self._comparator(outcome.original_output_hash, truth_hash):
            witnesses.append(outcome.original_provider_id)
        evidence = {
            "reason": "CONSENSUS_MISMATCH",
            "job_id": outcome.job_id,
            "accused_provider_id": accused.provider_id,
            "accused_output_hash": accused.output_hash,
            "majority_output_hash": truth_hash,
            "witness_provider_ids": witnesses,
            # sp957: self-describing stake posture so a downstream authority can
            # size/route an on-chain challenge without re-querying chain state.
            "accused_bonded": accused.bonded,
            "accused_stake_wei": accused.stake_wei,
            "accused_operator_address": accused.operator_address,
        }
        try:
            await self._challenge_sink(evidence)
        except Exception as e:
            logger.warning(
                "sp928: challenge sink failed for %s: %s", accused.provider_id[:8], e
            )

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
