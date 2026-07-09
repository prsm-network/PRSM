"""
Compute Requester
=================

Submits compute jobs to the PRSM network and collects results.
Broadcasts job offers via gossip, waits for provider acceptance,
verifies results, and records payments.

Supports capability-based smart routing to target capable peers first.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from prsm.node.compute_provider import JobStatus, JobType
from prsm.node.compute_result_sampler import (
    DEFAULT_SAMPLE_RATE,
    ComputeResultSampler,
    ReExecResult,
    VerificationVerdict,
)
from prsm.node.gossip import (
    GOSSIP_JOB_ACCEPT,
    GOSSIP_JOB_CONFIRM,
    GOSSIP_JOB_OFFER,
    GOSSIP_JOB_RESULT,
    GOSSIP_PAYMENT_CONFIRM,
    GossipProtocol,
)
from prsm.node.identity import NodeIdentity, verify_signature
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.transport import WebSocketTransport
from prsm.node.payment_escrow import PaymentEscrow
from prsm.node.operator_delegation import verify_operator_delegation_blob

if TYPE_CHECKING:
    from prsm.node.discovery import PeerDiscovery

logger = logging.getLogger(__name__)


# Mapping of job types to required capabilities
JOB_TYPE_CAPABILITIES = {
    JobType.INFERENCE: "inference",
    JobType.EMBEDDING: "embedding",
    JobType.BENCHMARK: "benchmark",
}

# Mapping of job types to preferred backends
JOB_TYPE_PREFERRED_BACKENDS = {
    JobType.INFERENCE: ["anthropic", "openai", "local"],
    JobType.EMBEDDING: ["openai", "local"],
    JobType.BENCHMARK: ["local", "anthropic", "openai"],
}


@dataclass
class SubmittedJob:
    """Tracks a job submitted to the network."""
    job_id: str
    job_type: JobType
    payload: Dict[str, Any]
    ftns_budget: float
    status: JobStatus = JobStatus.PENDING
    provider_id: Optional[str] = None
    provider_public_key: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    result_verified: bool = False
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    # sp928 — a verification re-run of another job; suppresses nested sampling
    # and is itself never re-verified (avoids verification regress).
    verification_run: bool = False
    # sp928 — when set, ONLY these providers may be confirmed for this job
    # (ENFORCED in _on_job_accept). For verification re-runs this is the
    # independent capable set with the original provider excluded, so the
    # provider under test cannot grab its own re-run and self-confirm a forged
    # result (target_peers in the offer is only a hint a malicious peer ignores).
    allowed_providers: Optional[Set[str]] = None
    _result_event: asyncio.Event = field(default_factory=asyncio.Event)


class ComputeRequester:
    """Submits compute jobs to the PRSM network.

    Flow:
    1. submit_job() broadcasts a job_offer via gossip
    2. Providers accept (job_accept) — first accept wins
    3. Provider executes and sends job_result
    4. We verify the result signature and record payment
    """

    def __init__(
        self,
        identity: NodeIdentity,
        transport: WebSocketTransport,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        discovery: Optional["PeerDiscovery"] = None,
        accept_timeout: float = 30.0,
        result_timeout: float = 300.0,
        smart_routing: bool = True,
    ):
        self.identity = identity
        self.transport = transport
        self.gossip = gossip
        self.ledger = ledger
        self.discovery = discovery
        self.accept_timeout = accept_timeout
        self.result_timeout = result_timeout
        self.smart_routing = smart_routing

        self.submitted_jobs: Dict[str, SubmittedJob] = {}
        self._running = False
        self.ledger_sync = None  # Set by node.py after construction
        self.escrow: Optional["PaymentEscrow"] = None  # Set by node.py after construction
        # sp928 — optimistic verification of the single-provider pay path.
        # Built lazily in start() (or injected by node.py / tests before start).
        self.sampler: Optional[ComputeResultSampler] = None
        self._verify_tasks: Set[asyncio.Task] = set()
        # sp957 — CONSENSUS_MISMATCH challenge routing. Both injected by node.py
        # after construction (kept optional so tests / minimal wirings still run):
        #   stake_reader  → resolves a provider's on-chain bond posture
        #   mismatch_log  → persistent, operator-reviewable evidence store; its
        #                   async record() becomes the sampler's challenge_sink.
        self.stake_reader: Optional[Any] = None
        self.mismatch_log: Optional[Any] = None

    async def start(self) -> None:
        """Register gossip handlers."""
        self._running = True
        if self.sampler is None:
            self.sampler = self._build_default_sampler()
        self.gossip.subscribe(GOSSIP_JOB_ACCEPT, self._on_job_accept)
        self.gossip.subscribe(GOSSIP_JOB_RESULT, self._on_job_result)
        logger.info("Compute requester started")

    def _build_default_sampler(self) -> ComputeResultSampler:
        """Wire the sampler over this requester's live re-dispatch + reputation.

        Sample rate is env-tunable (``PRSM_COMPUTE_VERIFY_SAMPLE_RATE``).
        Verification re-runs are always free (see ``_reexec_for_verification``).

        sp957: a caught bonded liar's CONSENSUS_MISMATCH evidence is now routed
        to ``self.mismatch_log.record`` (a persistent, operator-reviewable store)
        when a log is injected. This does NOT slash on-chain — an autonomous
        slash from a single node's re-execution is unsound (StakeBond.slash is
        slasher-only; the open challengeReceipt rail re-verifies a Merkle proof
        the off-chain single-provider pay path never produces). The evidence is
        the corpus a future authority-gated bridge consumes. The always-on
        deterrent remains reputation / eligibility removal.
        """
        rate = DEFAULT_SAMPLE_RATE
        try:
            rate = float(os.getenv("PRSM_COMPUTE_VERIFY_SAMPLE_RATE", str(DEFAULT_SAMPLE_RATE)))
        except (TypeError, ValueError):
            logger.warning("sp928: bad PRSM_COMPUTE_VERIFY_SAMPLE_RATE — using default")
        challenge_sink = (
            self.mismatch_log.record if self.mismatch_log is not None else None
        )
        return ComputeResultSampler(
            re_executor=self._reexec_for_verification,
            reputation_sink=self._record_verification_failure,
            challenge_sink=challenge_sink,
            sample_rate=rate,
        )

    def _resolve_stake_posture(self, provider_id: str):
        """Resolve a provider's on-chain bond posture for evidence routing.

        Returns ``(bonded, stake_wei, operator_address)``. A provider's CLAIMED
        ``operator_address`` (from its hardware_profile) is trusted ONLY after a
        valid operator-delegation proof (sprint 788) — a spoofed claim, a missing
        peer record, or an unavailable stake reader all degrade to the unbonded
        ``(False, 0, None)`` safe default. This is descriptive metadata for the
        CONSENSUS_MISMATCH evidence log; it grants no authority and moves no stake.
        """
        try:
            disc = self.discovery
            known = getattr(disc, "known_peers", None) if disc is not None else None
            peer = known.get(provider_id) if isinstance(known, dict) else None
            if peer is None:
                return (False, 0, None)
            hw = getattr(peer, "hardware_profile", None) or {}
            operator_address = (hw.get("operator_address") or "").strip()
            if not operator_address:
                return (False, 0, None)
            # sprint-788 gate: an unverified claim cannot ride another bond.
            if not verify_operator_delegation_blob(
                node_id=provider_id,
                operator_address=operator_address,
                delegation=hw.get("operator_delegation"),
            ):
                return (False, 0, None)
            if self.stake_reader is None:
                # Operator is proven, but we can't size the bond → treat as
                # unbonded (safe degrade) while still surfacing the address.
                return (False, 0, operator_address)
            stake_wei = int(self.stake_reader.stake_amount_for(operator_address) or 0)
            return (stake_wei > 0, stake_wei, operator_address)
        except Exception as e:  # never let posture resolution break verification
            logger.debug("sp957: stake-posture resolution failed for %s: %s",
                         str(provider_id)[:8], e)
            return (False, 0, None)

    async def stop(self) -> None:
        self._running = False
        # sp928 — cancel any in-flight verification tasks so they don't outlive
        # the requester / dangle on the event loop at shutdown.
        for task in list(self._verify_tasks):
            task.cancel()
        self._verify_tasks.clear()

    def set_discovery(self, discovery: "PeerDiscovery") -> None:
        """Set the peer discovery instance for smart routing."""
        self.discovery = discovery

    def _get_capable_peers(self, job_type: JobType) -> List[str]:
        """Find peers capable of handling a specific job type.

        Args:
            job_type: The type of job to find capable peers for.

        Returns:
            List of peer IDs that have the required capabilities.
        """
        if not self.discovery:
            return []

        # Get required capability for job type
        required_capability = JOB_TYPE_CAPABILITIES.get(job_type)
        if not required_capability:
            logger.debug(f"No capability mapping for job type {job_type.value}")
            return []

        # Find peers with the required capability
        capable_peers = self.discovery.find_peers_with_capability(required_capability)

        # Optionally filter by preferred backends
        preferred_backends = JOB_TYPE_PREFERRED_BACKENDS.get(job_type, [])
        if preferred_backends:
            backend_peers = []
            for backend in preferred_backends:
                backend_peers.extend(self.discovery.find_peers_with_backend(backend))
            # Prefer peers that have both capability and preferred backend
            backend_peer_ids = {p.node_id for p in backend_peers}
            capable_peers = [p for p in capable_peers if p.node_id in backend_peer_ids] or capable_peers

        # sp958 — EXCLUDE confirmed repeat-liars from dispatch (not merely sort
        # them last). A provider with >= PRSM_DISPATCH_MAX_CONSENSUS_MISMATCHES
        # confirmed consensus-mismatch events (the sp957 evidence log) caught
        # fabricating results stops receiving paid jobs from this node. This is
        # the sampling-immune signal (a reliability ratio is useless: at a 5%
        # sample rate a 100%-fabricating provider still scores ~0.95) and a
        # sound, LOCAL, reversible enforcement — it moves no stake.
        before = len(capable_peers)
        capable_peers = [p for p in capable_peers if not self._dispatch_excluded(p.node_id)]
        if len(capable_peers) < before:
            logger.warning(
                "sp958: excluded %d confirmed repeat-liar(s) from %s dispatch",
                before - len(capable_peers), job_type.value,
            )

        # Sort by reliability — unreliable peers fall to the bottom
        capable_peers.sort(key=lambda p: p.reliability_score, reverse=True)

        return [p.node_id for p in capable_peers]

    def _dispatch_excluded(self, node_id: str) -> bool:
        """True if ``node_id`` has accumulated enough confirmed consensus
        mismatches (sp957 evidence log) to be barred from compute dispatch.

        No log wired (minimal/legacy) → never excluded (current behavior).
        ``PRSM_DISPATCH_MAX_CONSENSUS_MISMATCHES`` (default 2) is the threshold;
        ``PRSM_DISPATCH_MISMATCH_WINDOW_SEC`` (default 0 = all-time) optionally
        decays old evidence so a reformed provider re-enters. Fail-open: any
        error in the check leaves the provider eligible (never silently starve
        dispatch on a bookkeeping glitch)."""
        log = getattr(self, "mismatch_log", None)
        if log is None:
            return False
        try:
            threshold, since = self._dispatch_exclusion_policy()
            if threshold <= 0:
                return False
            return log.count_for(node_id, since_timestamp=since) >= threshold
        except Exception as e:  # never starve dispatch on a check error
            logger.debug("sp958: dispatch-eligibility check failed for %s: %s",
                         str(node_id)[:8], e)
            return False

    def _dispatch_exclusion_policy(self):
        """Resolve (threshold, since_timestamp) from env — the single source of
        truth shared by the gate and the operator-facing summary (no drift)."""
        threshold = int(os.getenv("PRSM_DISPATCH_MAX_CONSENSUS_MISMATCHES", "2"))
        window = float(os.getenv("PRSM_DISPATCH_MISMATCH_WINDOW_SEC", "0") or 0)
        since = (time.time() - window) if window > 0 else None
        return threshold, since

    def dispatch_exclusion_summary(self) -> Dict[str, Any]:
        """sp959 — operator visibility into the sp958 enforcement policy: which
        providers this node is currently refusing to pay, and the count of
        confirmed mismatches behind each. Returns the active threshold/window
        plus a per-provider breakdown (count + excluded bool). Empty when no
        evidence log is wired."""
        log = getattr(self, "mismatch_log", None)
        threshold = int(os.getenv("PRSM_DISPATCH_MAX_CONSENSUS_MISMATCHES", "2"))
        window = float(os.getenv("PRSM_DISPATCH_MISMATCH_WINDOW_SEC", "0") or 0)
        out: Dict[str, Any] = {
            "threshold": threshold,
            "window_sec": window,
            "providers": [],
        }
        if log is None:
            return out
        try:
            _, since = self._dispatch_exclusion_policy()
            rows = []
            for pid in log.providers():
                count = log.count_for(pid, since_timestamp=since)
                rows.append({
                    "provider_id": pid,
                    "mismatch_count": count,
                    "excluded": threshold > 0 and count >= threshold,
                })
            rows.sort(key=lambda r: r["mismatch_count"], reverse=True)
            out["providers"] = rows
        except Exception as e:
            logger.debug("sp959: dispatch-exclusion summary failed: %s", e)
        return out

    async def submit_job(
        self,
        job_type: JobType,
        payload: Dict[str, Any],
        ftns_budget: float = 0.0,
        target_peers: Optional[List[str]] = None,
        use_escrow: bool = True,
        job_id: Optional[str] = None,
        verification_run: bool = False,
        allowed_providers: Optional[Set[str]] = None,
    ) -> SubmittedJob:
        """Submit a compute job to the network, optionally with escrow.

        Args:
            job_id: Optional pre-assigned job ID (e.g. from API escrow).
                    If None, a new UUID is generated.
            verification_run: sp928 — mark this as a re-execution issued to
                    verify another job's result. Such jobs are never themselves
                    sampled (no verification regress).
        """
        if ftns_budget > 0:
            balance = await self.ledger.get_balance(self.identity.node_id)
            if balance < ftns_budget:
                raise ValueError(f"Insufficient FTNS balance: {balance:.2f} < {ftns_budget:.2f}")

        job = SubmittedJob(
            job_id=job_id or uuid.uuid4().hex,
            job_type=job_type,
            payload=payload,
            ftns_budget=ftns_budget,
            verification_run=verification_run,
            allowed_providers=allowed_providers,
        )
        self.submitted_jobs[job.job_id] = job

        # Create escrow for the job (Phase 1: on-chain FTNS economy)
        if use_escrow and self.escrow is not None and ftns_budget > 0:
            escrow_entry = await self.escrow.create_escrow(
                job_id=job.job_id,
                amount=ftns_budget,
                requester_id=self.identity.node_id,
            )
            if escrow_entry:
                job.escrow_id = escrow_entry.escrow_id
                logger.info(
                    f"Escrow created for {job.job_id[:8]}: "
                    f"{ftns_budget:.6f} FTNS locked"
                )
            else:
                logger.warning(
                    f"Escrow creation failed for {job.job_id[:8]} — "
                    f"proceeding without escrow"
                )
        elif ftns_budget > 0 and self.escrow is None:
            logger.debug(
                f"No escrow available for {job.job_id[:8]} — "
                f"running without on-chain escrow"
            )

        # Determine target peers for smart routing
        routing_targets = target_peers
        routing_mode = "broadcast"

        if self.smart_routing and self.discovery and target_peers is None:
            capable_peers = self._get_capable_peers(job_type)
            if capable_peers:
                routing_targets = capable_peers
                routing_mode = "targeted"
                logger.info(
                    f"Smart routing: targeting {len(capable_peers)} "
                    f"capable peers for {job_type.value}"
                )
            else:
                logger.info(
                    f"No capable peers found for {job_type.value}, falling back to broadcast"
                )

        # Broadcast job offer
        job_offer = {
            "job_id": job.job_id,
            "job_type": job_type.value,
            "requester_id": self.identity.node_id,
            "payload": payload,
            "ftns_budget": ftns_budget,
            # sp1405 — advertise the payer eth-address so the PROVIDER can name it as `requester`
            # when it commits its earning on-chain (omitted when off-chain settlement isn't configured).
            "requester_operator_address": (getattr(self, "operator_address", "") or "") or None,
        }

        # Add target peers if smart routing
        if routing_targets:
            job_offer["target_peers"] = routing_targets

        await self.gossip.publish(GOSSIP_JOB_OFFER, job_offer)

        logger.info(
            f"Submitted job {job.job_id[:8]} ({job_type.value}), "
            f"budget: {ftns_budget} FTNS, routing: {routing_mode}"
        )
        return job

    async def get_result(self, job_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for a job result. Returns the result dict or None on timeout."""
        job = self.submitted_jobs.get(job_id)
        if not job:
            return None

        timeout = timeout or self.result_timeout
        try:
            await asyncio.wait_for(job._result_event.wait(), timeout=timeout)
            return job.result
        except asyncio.TimeoutError:
            job.status = JobStatus.FAILED
            job.error = "Timed out waiting for result"
            return None

    # ── Gossip handlers ──────────────────────────────────────────

    async def _on_job_accept(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle a provider accepting our job.

        First-accept-wins: the first provider to accept is confirmed via
        GOSSIP_JOB_CONFIRM.  Other providers that accepted should see the
        confirm message (with a different provider_id) and drop the job.
        """
        job_id = data.get("job_id", "")
        job = self.submitted_jobs.get(job_id)
        if not job:
            return

        # First accept wins — ignore subsequent accepts
        if job.status != JobStatus.PENDING:
            return

        provider_id = data.get("provider_id", origin)

        # sp928 — ENFORCE the allowed-provider restriction. target_peers in the
        # job offer is only a routing hint a malicious provider can ignore; the
        # binding decision is made HERE. For a verification re-run this rejects
        # an accept from the provider under test (excluded), so it cannot
        # self-confirm its own re-run and forge agreement with its forged result.
        if job.allowed_providers is not None and provider_id not in job.allowed_providers:
            logger.warning(
                "Job %s: rejecting accept from disallowed provider %s "
                "(verification re-run / targeted job)",
                job_id[:8], str(provider_id)[:8],
            )
            return

        job.status = JobStatus.ACCEPTED
        job.provider_id = provider_id
        job.provider_public_key = data.get("public_key", "")

        # Confirm this provider so it starts execution;
        # other providers that accepted will see the confirm and drop the job.
        await self.gossip.publish(GOSSIP_JOB_CONFIRM, {
            "job_id": job_id,
            "provider_id": provider_id,
            "requester_id": self.identity.node_id,
        })

        logger.info(f"Job {job_id[:8]} confirmed provider {provider_id[:8]}")

    async def _on_job_result(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle a job result from a provider."""
        job_id = data.get("job_id", "")
        job = self.submitted_jobs.get(job_id)
        if not job:
            return

        # sp924 — IDEMPOTENCY: once a job is COMPLETED (and the provider paid),
        # ignore any further result. The transport nonce-dedup only covers a
        # ~300s window; a JOB_RESULT re-delivered after expiry (peer reconnect /
        # re-broadcast) would otherwise re-run this handler and re-pay (the
        # sp898/899 gossip-money double-credit class). A cross-restart replay is
        # already safe — the job is gone from submitted_jobs above.
        if job.status == JobStatus.COMPLETED:
            return

        provider_id = data.get("provider_id", origin)
        status = data.get("status", "")

        # sp924 — WRONG-PAYEE guard: provider_id (and public_key, below) come
        # from the UNTRUSTED gossip message. Without binding them to the
        # provider this requester ACCEPTED (job.provider_id / provider_public_key,
        # set at job-accept time), a third party can publish a JOB_RESULT for
        # someone else's job signed with their OWN key + their OWN payout id —
        # the signature verifies (against the attacker's key) and payment is
        # redirected to the attacker. Reject a result that isn't from the
        # accepted provider. (Enforced only once a provider was accepted;
        # self-compute sets provider_id == own node id and passes.)
        if job.provider_id and provider_id != job.provider_id:
            logger.warning(
                "Job %s: result from unexpected provider %s (accepted %s) — "
                "ignoring (possible payment redirect)",
                job_id[:8], str(provider_id)[:8], str(job.provider_id)[:8],
            )
            return

        if status == "failed":
            job.status = JobStatus.FAILED
            job.error = data.get("error", "Unknown error")
            job.completed_at = time.time()
            job._result_event.set()
            logger.warning(f"Job {job_id[:8]} failed: {job.error}")
            # Record failure in discovery for reliability tracking
            if self.discovery:
                self.discovery.record_job_failure(provider_id)
            return

        # Verify result signature — required for payment
        result = data.get("result", {})
        signature = data.get("signature", "")
        pub_key = data.get("public_key", job.provider_public_key or "")

        # sp924 — bind the verifying key to the accepted provider's key too
        # (defense-in-depth alongside the provider_id check): a valid signature
        # under an ATTACKER's key must not authorize payment. Enforced only when
        # the accepted provider's key is known.
        if job.provider_public_key and pub_key != job.provider_public_key:
            logger.warning(
                "Job %s: result public key does not match the accepted "
                "provider's key — ignoring (possible payment redirect)",
                job_id[:8],
            )
            return

        verified = False
        if pub_key and signature:
            result_bytes = json.dumps(result, sort_keys=True).encode()
            verified = verify_signature(pub_key, result_bytes, signature)

        if not verified and provider_id != self.identity.node_id:
            # Reject unsigned/unverified results from remote providers.
            # Self-compute results (same node) are trusted without signature.
            logger.warning(
                f"Job {job_id[:8]}: rejecting unverified result from "
                f"{provider_id[:8]} (missing or invalid signature)"
            )
            return

        # sp1408 — MOCK GUARD: a signature proves ORIGIN, not that any compute happened.
        # A provider with no inference backend (bare `pip install prsm`, no `.[ml]` extra)
        # does not fail the job — `compute_provider._run_inference` fabricates a placeholder
        # and tags it `source: "mock"`, then signs it with the provider's real key. That
        # signature verifies, so the job completed and the locked escrow was RELEASED: the
        # requester paid its full budget for "[PRSM node abcd1234 processed inference]".
        # Never pay for a result the provider's own software labels a mock. Self-compute is
        # exempt (payer == payee, so no value moves).
        if (
            provider_id != self.identity.node_id
            and isinstance(result, dict)
            and result.get("source") == "mock"
        ):
            job.status = JobStatus.FAILED
            job.error = (
                f"provider {str(provider_id)[:8]} returned a MOCK result "
                f"(no inference backend configured on that node) — not paying for it"
            )
            job.completed_at = time.time()
            logger.warning("Job %s: %s", job_id[:8], job.error)
            if self.discovery:
                try:
                    self.discovery.record_job_failure(provider_id)
                except Exception as _track_exc:  # noqa: BLE001 — tracking must never raise here
                    logger.debug("record_job_failure failed (non-fatal): %s", _track_exc)
            job._result_event.set()
            return

        job.status = JobStatus.COMPLETED
        job.result = result
        job.result_verified = verified
        job.completed_at = time.time()

        # Record success in discovery for reliability tracking. sp1402 — best-effort: a tracking
        # error (e.g. a discovery impl missing this method) must NEVER block the provider payment
        # below. Before this guard, an AttributeError here silently skipped every cross-node payout.
        if self.discovery:
            try:
                self.discovery.record_job_success(provider_id)
            except Exception as _track_exc:  # noqa: BLE001
                logger.debug("record_job_success failed (non-fatal): %s", _track_exc)

        # Record payment — only for remote providers.
        # Self-compute payment is handled by the API escrow release.
        try:
            if provider_id != self.identity.node_id and job.ftns_budget > 0:
                # sp1401 — if this job locked ESCROW at submit-time, RELEASE that escrow to the
                # provider. A second direct transfer here would DOUBLE-PAY: the requester's funds are
                # already locked in the escrow wallet. release_escrow moves the locked funds to the
                # provider and is idempotent (returns None on an already-released escrow). Only fall
                # back to a direct transfer for a non-escrow job.
                if getattr(job, "escrow_id", None) and self.escrow is not None:
                    tx = await self.escrow.release_escrow(
                        job_id=job_id,
                        provider_id=provider_id,
                        consensus_reached=True,
                    )
                else:
                    tx = await self.ledger.transfer(
                        from_wallet=self.identity.node_id,
                        to_wallet=provider_id,
                        amount=job.ftns_budget,
                        tx_type=TransactionType.COMPUTE_PAYMENT,
                        description=f"Payment for job {job_id[:8]}",
                    )

                # Broadcast payment via ledger sync so the provider's OWN ledger credits itself
                # (its _on_ftns_transaction applies the tx whose to_wallet is the provider).
                if tx is not None and self.ledger_sync:
                    try:
                        await self.ledger_sync.broadcast_transaction(tx)
                    except Exception:
                        pass

                # Confirm payment on network
                await self.gossip.publish(GOSSIP_PAYMENT_CONFIRM, {
                    "job_id": job_id,
                    "requester_id": self.identity.node_id,
                    "provider_id": provider_id,
                    "amount": job.ftns_budget,
                })
        except ValueError as e:
            logger.error(f"Payment failed for job {job_id[:8]}: {e}")

        # sp1405 — on-chain settlement is PROVIDER-side (the earner commits: commitBatch →
        # provider=msg.sender). The requester does NOT accumulate/commit — it funds the on-chain escrow
        # and is named as `requester` in the batch the provider commits. (Superseded the sp1403
        # requester-side accumulate, which would have set the payer as the on-chain earner.)
        job._result_event.set()

    def deliver_local_result(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Sprint 1390 — deliver a SELF-COMPUTED result directly into the submitting job's tracking,
        bypassing gossip. ``transport.gossip`` only sends to network peers (no loopback), so a
        single-node self-compute result never reaches the requester via the normal GOSSIP_JOB_RESULT
        path — this closes that gap so `compute submit` completes. Trusted (same node, no signature
        check); self-compute payment is the API escrow release's job. Returns True iff a still-pending
        job was completed."""
        job = self.submitted_jobs.get(job_id)
        if job is None or job._result_event.is_set():
            return False
        job.status = JobStatus.COMPLETED
        job.result = result
        job.result_verified = True
        job.completed_at = time.time()
        job._result_event.set()
        return True

        logger.info(
            f"Job {job_id[:8]} completed by {provider_id[:8]}, "
            f"verified={verified}, paid {job.ftns_budget} FTNS"
        )

        # sp928 — OPTIMISTIC VERIFICATION (mis-pay #2). A valid signature proves
        # WHO produced the result, not that the work was done. With probability
        # sample_rate, re-execute this job on an independent provider and
        # penalize the outlier on a mismatch. Fire-and-forget: a verification
        # error must never block or unwind the pay path above.
        if (
            self.sampler is not None
            and not job.verification_run
            and provider_id != self.identity.node_id
            and self.sampler.should_sample()
        ):
            original_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True).encode()
            ).hexdigest()
            task = asyncio.create_task(
                self._run_verification(job, provider_id, original_hash)
            )
            # Keep a strong ref so the task is not GC'd mid-flight.
            self._verify_tasks.add(task)
            task.add_done_callback(self._verify_tasks.discard)

    async def _run_verification(
        self, job: SubmittedJob, provider_id: str, original_hash: str
    ) -> None:
        """Drive sampler.verify() for a completed job (fire-and-forget safe)."""
        try:
            bonded, stake_wei, operator_address = self._resolve_stake_posture(provider_id)
            outcome = await self.sampler.verify(
                job_id=job.job_id,
                job_type=job.job_type,
                payload=job.payload,
                original_provider_id=provider_id,
                original_output_hash=original_hash,
                original_bonded=bonded,
                original_stake_wei=stake_wei,
                original_operator_address=operator_address,
            )
            if outcome.verdict == VerificationVerdict.MISMATCH:
                logger.warning(
                    "sp928: job %s FAILED verification — provider %s is the "
                    "outlier vs the re-execution majority (penalized)",
                    job.job_id[:8], provider_id[:8],
                )
            elif outcome.verdict == VerificationVerdict.VERIFIED:
                logger.info("sp928: job %s passed re-execution verification", job.job_id[:8])
            else:
                logger.debug("sp928: job %s verification %s", job.job_id[:8], outcome.verdict.value)
        except Exception as e:
            logger.warning("sp928: verification task errored for job %s: %s", job.job_id[:8], e)

    async def _reexec_for_verification(
        self, job_type: JobType, payload: Dict[str, Any], exclude: Set[str]
    ) -> Optional[ReExecResult]:
        """Re-dispatch a job to an INDEPENDENT provider and hash its result.

        Returns None (→ verification SKIPPED, never a false slash) when no other
        capable provider is available or the re-run produces no result.
        """
        capable = [p for p in self._get_capable_peers(job_type) if p not in exclude]
        if not capable:
            return None
        allowed = set(capable)

        # Re-runs are ALWAYS free (ftns_budget=0). Paying a verification provider
        # on its signature alone would reintroduce mis-pay #2 on the re-run (the
        # re-run is itself never sampled), so a paid-verification policy must wait
        # for conditional, comparison-gated payment (follow-on). A provider that
        # declines the free re-run simply yields no result → verification SKIPPED
        # (safe degrade), never a false slash.
        try:
            rerun = await self.submit_job(
                job_type, payload,
                ftns_budget=0.0,
                target_peers=list(allowed),
                use_escrow=False,
                verification_run=True,
                allowed_providers=allowed,
            )
        except Exception as e:
            logger.warning("sp928: verification re-dispatch failed: %s", e)
            return None

        res = await self.get_result(rerun.job_id, timeout=self.result_timeout)
        if res is None:
            return None
        rerun_job = self.submitted_jobs.get(rerun.job_id)
        rerun_provider = (rerun_job.provider_id if rerun_job else None) or capable[0]
        out_hash = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
        # sp957: carry the re-run provider's own bond posture so a lying
        # re-exec provider's stake is recorded in the routed evidence.
        bonded, stake_wei, operator_address = self._resolve_stake_posture(rerun_provider)
        return ReExecResult(
            provider_id=rerun_provider,
            output_hash=out_hash,
            bonded=bonded,
            stake_wei=stake_wei,
            operator_address=operator_address,
        )

    def _record_verification_failure(self, provider_id: str, outcome) -> None:
        """Reputation deterrent — drop a caught liar's reliability score."""
        if self.discovery:
            try:
                self.discovery.record_job_failure(provider_id)
            except Exception as e:
                logger.warning("sp928: failed to record reputation hit for %s: %s", provider_id[:8], e)

    def get_stats(self) -> Dict[str, Any]:
        """Return requester statistics."""
        statuses = {}
        for job in self.submitted_jobs.values():
            statuses[job.status.value] = statuses.get(job.status.value, 0) + 1

        total_spent = sum(
            j.ftns_budget for j in self.submitted_jobs.values()
            if j.status == JobStatus.COMPLETED
        )

        return {
            "total_submitted": len(self.submitted_jobs),
            "by_status": statuses,
            "total_ftns_spent": total_spent,
        }
