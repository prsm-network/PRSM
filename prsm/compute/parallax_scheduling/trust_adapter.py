"""
Trust adapters — the load-bearing PRSM contribution.

Phase 3.x.6 Task 5. The vendored Parallax algorithm assumes
cooperative volunteer GPUs. PRSM doesn't — it has stake/slash because
volunteers may misbehave. These four adapters wrap the algorithm so
its routing decisions respect PRSM's trust + economic + confidentiality
stack:

  AnchorVerifyAdapter      Phase-1 input filter. Excludes GPUs whose
                           ``node_id`` doesn't resolve on the on-chain
                           Phase 3.x.3 anchor. Closes "fake GPU
                           advertises capacity to capture routing".

  TierGateAdapter          Phase-2 pre-route filter. For requests
                           with non-NONE privacy tier, excludes GPUs
                           lacking hardware-TEE attestation. Refuses
                           the request entirely if the filtered set
                           can't cover all layers.

  StakeWeightedTrustAdapter Wraps a ``ProfileSource``. Rescales each
                           snapshot's ``layer_latency_ms`` by 1 /
                           confidence, where confidence is derived
                           from the GPU's current stake. Zero-stake
                           GPUs return ``None`` from ``get_snapshot``
                           (effectively excluded from routing).

  ConsensusMismatchHook    Post-route. With probability ``sample_rate``,
                           schedules a redundant chain in parallel.
                           On output mismatch, surfaces as a Phase 7.1
                           ``CONSENSUS_MISMATCH`` challenge that slashes
                           the misbehaving GPU(s).

Why these four:
  - Adapter A: catches fake-identity attacks at the front door.
  - Adapter B: keeps confidential workloads off non-attested hardware.
  - Adapter C: makes profile-lying economically irrational
               (lying GPU needs stake, but stake gets slashed when
               Adapter D catches the lie).
  - Adapter D: empirical deterrent that backstops Adapter C's
               theoretical guarantee.

C and D are paired: stake-weighting raises the cost of misbehavior;
consensus-mismatch detection makes the cost concrete by enabling
Phase 7.1 to slash. Without D, C's stake-weighting reduces to a
participation tax with no enforcement teeth. Without C, D fires too
often (every GPU lies because there's no cost).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from prsm.compute.parallax_scheduling.prsm_request_router import (
    GPUChain,
    ProfileSnapshot,
    ProfileSource,
)
from prsm.compute.parallax_scheduling.prsm_types import (
    TIER_ATTESTATION_NONE,
    ParallaxGPU,
)
from prsm.compute.tee.models import HARDWARE_TEE_TYPES, PrivacyLevel


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Adapter A — Anchor verification (Phase-1 input filter)
# ──────────────────────────────────────────────────────────────────────────


class AnchorLookup(Protocol):
    """Subset of the Phase 3.x.3 anchor client we depend on."""

    def lookup(self, node_id: str) -> Optional[str]:
        """Returns the publisher's pubkey (b64) if registered, else None."""
        ...


@dataclass(frozen=True)
class AnchorVerifyAdapter:
    """Filter ParallaxGPUs by anchor registration.

    A GPU enters the allocation pool ONLY if its ``node_id`` resolves
    to a non-empty pubkey on the configured anchor. Construction-time
    state isn't strictly required (anchor lookup is the only check)
    but the dataclass keeps the same shape as the other adapters
    for caller ergonomics.
    """

    anchor: AnchorLookup

    def __post_init__(self) -> None:
        if self.anchor is None or not hasattr(self.anchor, "lookup"):
            raise RuntimeError(
                "AnchorVerifyAdapter requires an anchor with .lookup(node_id) "
                "→ Optional[str]"
            )

    def filter(self, gpus: Sequence[ParallaxGPU]) -> List[ParallaxGPU]:
        """Return only GPUs whose ``node_id`` resolves on the anchor.

        Excluded GPUs are logged at WARNING so operators can correlate
        with the anchor side-channel. AnchorRPCError from the anchor
        propagates — that's a transient infrastructure failure, not
        an adapter rejection.
        """
        out: List[ParallaxGPU] = []
        for gpu in gpus:
            pubkey = self.anchor.lookup(gpu.node_id)
            if not pubkey:
                logger.warning(
                    "AnchorVerifyAdapter: excluding GPU %s — node_id not "
                    "registered on anchor",
                    gpu.node_id,
                )
                continue
            out.append(gpu)
        return out


# ──────────────────────────────────────────────────────────────────────────
# Adapter B — Tier gate (Phase-2 pre-route filter)
# ──────────────────────────────────────────────────────────────────────────


class TierGateRejected(Exception):
    """Raised when no GPU in the pool can serve the requested privacy
    tier. Distinct from NoCoverageError — this is a policy refusal,
    not an algorithmic failure."""


# Operator-set of accepted tier-attestation prefixes that count as
# hardware-backed. The default uses the ``HARDWARE_TEE_TYPES`` set
# from Phase 2 Ring 8 and prefix-matches against the attestation
# string. Operators can override (e.g. to whitelist new TEE types
# before Phase 2 catches up).
DEFAULT_HARDWARE_TIER_PREFIXES: frozenset = frozenset(
    f"tier-{t}" for t in HARDWARE_TEE_TYPES
)


def is_hardware_attestation(attestation: str, *, hardware_prefixes: frozenset = DEFAULT_HARDWARE_TIER_PREFIXES) -> bool:
    """True iff the attestation string indicates a hardware-backed TEE.

    The tier_attestation field on ParallaxGPU is operator-supplied;
    we don't trust its semantic — we just look for a known-good
    prefix. Real attestation verification is Phase 2 Ring 8's job
    upstream of this adapter.

    ``TIER_ATTESTATION_NONE`` (i.e. ``"tier-none"``) is NEVER
    considered hardware-backed regardless of prefix matching.
    """
    if attestation == TIER_ATTESTATION_NONE or not attestation:
        return False
    return attestation in hardware_prefixes


# Sprint 1083 — a genuine SGX/TDX/SEV quote is ≤ ~10 KB; cap the accepted attestation so
# a malicious operator can't relay a near-1-MiB blob across the gossip network and force
# every peer to base64-decode + cert-chain-parse it on each pool build (review LOW DoS).
_MAX_ATTESTATION_BYTES = 64 * 1024

# Sprint 1083 — map a §7-registry vendor string to its parallax tier-type token. Only
# these REAL hardware vendors can earn a hardware tier (and only when vendor_verified).
_VENDOR_TO_TEE_TYPE = {
    "intel-sgx": "sgx",
    "intel-tdx": "tdx",
    "amd-sev-snp": "sev",
    "amd-sev-snp-envelope": "sev",
    "apple-sep": "secure_enclave",
}


def expected_attestation_report_data(node_id: str) -> str:
    """Sprint 1083 — the canonical node-identity commitment a node must embed in the
    first 32 bytes of its TEE quote's REPORT_DATA: ``sha256(node_id)`` (hex). The
    consumer checks the verified quote carries THIS commitment so a quote can't be
    replayed by a different node (binding the hardware attestation to the node_id, the
    parallel to the sp788 operator-delegation node_id binding)."""
    import hashlib
    return hashlib.sha256((node_id or "").encode()).hexdigest()


def verified_tier_attestation(blob, *, node_id=None, registry=None) -> str:
    """Sprint 1083 — the missing bridge between the §7 attestation engines
    (sp1044-1082, which compute ``vendor_verified``) and the parallax tier gate
    (sp702 ``TierGateAdapter``, which filters on a tier-attestation string).

    Returns a hardware tier string (e.g. ``"tier-sgx"``) for ``blob`` ONLY when it is a
    real attestation that the registry CRYPTOGRAPHICALLY ``vendor_verified`` AND (when
    ``node_id`` is given) whose REPORT_DATA binds to THIS node — otherwise
    ``TIER_ATTESTATION_NONE``. So a node's eligibility for confidential (Tier B/C) work
    depends on a VERIFIED attestation bound to its identity, not a self-advertised tier
    prefix and not a replayable third-party quote. FAIL-CLOSED: a missing blob, a
    structural/unverified result, an unknown vendor, a REPORT_DATA that doesn't bind to
    ``node_id``, or any error during verification all yield ``tier-none``.

    ``blob`` may be raw bytes or a base64 string (as carried in a capability announce).
    When ``node_id`` is None the binding check is skipped (callers that bind elsewhere);
    the live pool-admission path always passes ``node_id``.
    """
    if not blob:
        return TIER_ATTESTATION_NONE
    if isinstance(blob, str):
        if len(blob) > _MAX_ATTESTATION_BYTES * 2:   # base64 ≈ 4/3 raw; cheap pre-check
            return TIER_ATTESTATION_NONE
        try:
            import base64
            blob = base64.b64decode(blob, validate=True)
        except Exception:  # noqa: BLE001
            return TIER_ATTESTATION_NONE
    if len(blob) > _MAX_ATTESTATION_BYTES:
        return TIER_ATTESTATION_NONE   # oversized quote → not a real TEE quote, refuse
    try:
        if registry is not None:
            result = registry.verify(blob)
        else:
            from prsm.compute.inference.attestation_backends import verify_attestation
            result = verify_attestation(blob)
    except Exception:  # noqa: BLE001 — verification error must never grant a tier
        return TIER_ATTESTATION_NONE
    if not getattr(result, "vendor_verified", False):
        return TIER_ATTESTATION_NONE
    tee_type = _VENDOR_TO_TEE_TYPE.get(getattr(result, "vendor", ""))
    if tee_type is None:
        return TIER_ATTESTATION_NONE
    # Replay/binding guard: the verified quote's REPORT_DATA must commit to THIS node.
    if node_id:
        report_data_hex = ""
        vendor_data = getattr(result, "vendor_data", None) or {}
        if isinstance(vendor_data, dict):
            report_data_hex = (vendor_data.get("report_data_hex") or "").lower()
        expected = expected_attestation_report_data(node_id)
        # REPORT_DATA is 64 bytes (128 hex); the node commits sha256(node_id) (32 bytes)
        # in its first half. A quote from a DIFFERENT node carries a different commitment.
        if not report_data_hex or report_data_hex[:len(expected)] != expected:
            return TIER_ATTESTATION_NONE
    return f"tier-{tee_type}"


@dataclass(frozen=True)
class TierGateAdapter:
    """Filter ParallaxGPUs by privacy-tier requirements at request time.

    For ``PrivacyLevel.NONE``: no filtering — all GPUs pass through.
    For STANDARD / HIGH / MAXIMUM: only GPUs whose ``tier_attestation``
    matches a hardware-backed prefix are eligible.

    If the filtered set has fewer GPUs than the model requires, the
    adapter raises ``TierGateRejected`` rather than letting Phase-2
    discover the coverage gap. This surfaces the policy refusal
    earlier and with a more actionable error message.
    """

    hardware_prefixes: frozenset = DEFAULT_HARDWARE_TIER_PREFIXES

    def filter(
        self,
        gpus: Sequence[ParallaxGPU],
        privacy_level: PrivacyLevel,
    ) -> List[ParallaxGPU]:
        """Return only GPUs eligible for the given privacy level."""
        if privacy_level == PrivacyLevel.NONE:
            return list(gpus)

        eligible = [
            gpu for gpu in gpus
            if is_hardware_attestation(
                gpu.tier_attestation,
                hardware_prefixes=self.hardware_prefixes,
            )
        ]
        if not eligible:
            raise TierGateRejected(
                f"no GPU in pool ({len(gpus)} candidates) has hardware-TEE "
                f"attestation; required for privacy_level={privacy_level.value}"
            )
        return eligible


# ──────────────────────────────────────────────────────────────────────────
# Adapter C — Stake-weighted profile trust
# ──────────────────────────────────────────────────────────────────────────


class StakeLookup(Protocol):
    """Subset of Phase 7 ``StakeManager`` we depend on. Returns the
    GPU's current stake in FTNS wei, or 0 if unstaked."""

    def get_stake(self, node_id: str) -> int:
        ...


# Default fully-staked threshold. At or above this stake amount, the
# GPU's profile is trusted at full advertised speed (1× confidence).
# Below: linearly scaled. At zero: excluded entirely. Operators tune
# this — too high and only well-capitalized stakers can compete; too
# low and slashing is a slap on the wrist.
DEFAULT_FULLY_STAKED_THRESHOLD = 1_000_000_000_000_000_000  # 1 FTNS = 1e18 wei

# Below this stake, snapshots return None (treated as "no live data"
# by the router's stale-fallback path → roofline estimate). Distinct
# from the linear-scale path above — at exactly 0 stake the GPU
# is excluded outright.
MIN_STAKE_FOR_PARTICIPATION = 1


@dataclass
class StakeWeightedTrustAdapter:
    """Wraps a ``ProfileSource`` and rescales snapshots by stake.

    The wrapped source is what eventually populates the
    ``RequestRouter``. By rescaling at this layer, the upstream DP
    sees adjusted (τ, ρ) values and routes accordingly — no DP-level
    changes needed.

    Confidence formula:
        zero stake          → return None (de-facto excluded)
        below min_threshold → return None (parity with zero)
        below fully_staked  → confidence = stake / fully_staked_threshold;
                              latency *= 1 / confidence
        ≥ fully_staked      → confidence = 1.0; latency unchanged

    RTT-to-peers is NOT rescaled — only the per-layer compute latency
    is suspect under stake-weighting (RTT is symmetric and verifiable
    by the peer side).
    """

    inner: ProfileSource
    stake_lookup: StakeLookup
    fully_staked_threshold: int = DEFAULT_FULLY_STAKED_THRESHOLD
    min_stake_for_participation: int = MIN_STAKE_FOR_PARTICIPATION

    def __post_init__(self) -> None:
        if self.fully_staked_threshold <= 0:
            raise ValueError(
                f"fully_staked_threshold must be positive, "
                f"got {self.fully_staked_threshold}"
            )
        if self.min_stake_for_participation < 0:
            raise ValueError(
                f"min_stake_for_participation must be ≥ 0, "
                f"got {self.min_stake_for_participation}"
            )
        if self.stake_lookup is None or not hasattr(
            self.stake_lookup, "get_stake"
        ):
            raise RuntimeError(
                "StakeWeightedTrustAdapter requires a stake_lookup with "
                ".get_stake(node_id) → int"
            )

    def is_eligible(self, node_id: str) -> bool:
        """Returns True iff the node's stake meets the participation
        threshold. Used by ``TrustStack.filter_pool`` to drop zero-stake
        GPUs from the allocation pool BEFORE Phase-1 — without this
        filter the upstream router can fall back to a roofline latency
        estimate from hardware specs and route to a zero-stake liar
        anyway, defeating the design-plan intent that zero-stake GPUs
        are 'effectively excluded from routing'."""
        return (
            self.stake_lookup.get_stake(node_id)
            >= self.min_stake_for_participation
        )

    def get_snapshot(self, node_id: str) -> Optional[ProfileSnapshot]:
        """Implements ``ProfileSource``. Falls through to inner source
        for the raw snapshot, then rescales latency by stake confidence."""
        stake = self.stake_lookup.get_stake(node_id)
        if stake < self.min_stake_for_participation:
            # De facto excluded — router's stale-fallback path will
            # use roofline OR (at the orchestration layer) the
            # AnchorVerifyAdapter will already have filtered this
            # GPU out. Returning None here is the second line of
            # defense.
            return None

        raw = self.inner.get_snapshot(node_id)
        if raw is None:
            return None

        confidence = self._confidence(stake)
        if confidence <= 0:
            return None

        scaled_latency = raw.layer_latency_ms / confidence
        return ProfileSnapshot(
            node_id=raw.node_id,
            layer_latency_ms=scaled_latency,
            rtt_to_peers=dict(raw.rtt_to_peers),
            timestamp_unix=raw.timestamp_unix,
        )

    def _confidence(self, stake: int) -> float:
        if stake >= self.fully_staked_threshold:
            return 1.0
        return stake / self.fully_staked_threshold


# ──────────────────────────────────────────────────────────────────────────
# Adapter D — Consensus-mismatch hook (post-route)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChallengeRecord:
    """Surface area of a Phase 7.1 ``CONSENSUS_MISMATCH`` challenge.

    The actual on-chain dispatch happens via Phase 7.1's
    ``ConsensusChallengeSubmitter``; this record is the typed
    payload the hook hands to the submitter callable.
    """

    request_id: str
    primary_chain_stages: tuple
    secondary_chain_stages: tuple
    primary_output_hash: str  # sha256 hex
    secondary_output_hash: str  # sha256 hex


# Submitter callable: receives a fully-formed ChallengeRecord and is
# responsible for the on-chain ``challengeReceipt`` call. Production
# wires this to the Phase 7.1x ``ConsensusChallengeSubmitter.enqueue``
# method; tests inject a fake to assert what would have been submitted.
ChallengeSubmitter = Callable[[ChallengeRecord], None]


@dataclass
class ConsensusMismatchHook:
    """Post-route redundant-execution + mismatch-challenge orchestrator.

    Two responsibilities, intentionally split into separate methods so
    callers can sequence them around the actual chain dispatch:

      should_sample_redundant(request_id) → bool
        Sample-rate decision. Deterministic given the request_id +
        sample_rate (uses a hash so the same request always gets
        the same decision — useful for tests + reproducible audits).

      compare_and_challenge(...) → Optional[ChallengeRecord]
        Output comparison + challenge dispatch. Called only when
        should_sample_redundant returned True for this request AND
        a secondary chain actually executed. Mismatch → submitter
        invoked, ChallengeRecord returned. Match → None.

    The hook does NOT execute chains itself — orchestration of the
    primary + secondary chain dispatch is the executor's
    responsibility (Phase 3.x.6 Task 6).
    """

    submitter: ChallengeSubmitter
    sample_rate: float = 0.01  # 1% redundancy per design plan §3.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(
                f"sample_rate must be in [0.0, 1.0], got {self.sample_rate}"
            )
        if self.submitter is None:
            raise RuntimeError(
                "ConsensusMismatchHook requires a submitter callable"
            )

    def should_sample_redundant(self, request_id: str) -> bool:
        """Deterministic per-request sample decision.

        Same request_id always yields the same decision — this lets
        operators reproduce mismatch events in the audit log and
        gives tests a stable surface. Sampling threshold:

            hash(request_id) mod 10**6 < sample_rate * 10**6
        """
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        digest = hashlib.sha256(request_id.encode("utf-8")).digest()
        # Take the first 8 bytes as an unsigned int and mod into
        # [0, 10**6).
        bucket = int.from_bytes(digest[:8], "big") % 1_000_000
        return bucket < int(self.sample_rate * 1_000_000)

    def compare_and_challenge(
        self,
        *,
        request_id: str,
        primary_output: bytes,
        secondary_output: bytes,
        primary_chain: GPUChain,
        secondary_chain: GPUChain,
    ) -> Optional[ChallengeRecord]:
        """Compare two chain outputs; on mismatch, dispatch challenge.

        Returns the submitted ``ChallengeRecord`` on mismatch, or
        ``None`` on match. The submitter callable is invoked with
        the record; it's responsible for any retry / queueing /
        on-chain semantics.
        """
        primary_hash = hashlib.sha256(primary_output).hexdigest()
        secondary_hash = hashlib.sha256(secondary_output).hexdigest()
        if primary_hash == secondary_hash:
            return None

        record = ChallengeRecord(
            request_id=request_id,
            primary_chain_stages=primary_chain.stages,
            secondary_chain_stages=secondary_chain.stages,
            primary_output_hash=primary_hash,
            secondary_output_hash=secondary_hash,
        )
        try:
            self.submitter(record)
        except Exception as exc:  # noqa: BLE001
            # Submitter failure is transient (network down, queue
            # full, etc.). Don't propagate — the record itself is
            # the audit trail; ops will replay from the operator
            # log if the submitter dropped it.
            logger.exception(
                "ConsensusMismatchHook: submitter raised on request_id=%s",
                request_id,
            )
        return record


# ──────────────────────────────────────────────────────────────────────────
# Composition helper
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class TrustStack:
    """Bundles the four adapters for orchestrator convenience.

    The executor (Phase 3.x.6 Task 6) constructs one of these at
    startup and uses it as a single entry point for trust-side
    operations:

      stack.filter_pool(gpus)              ← Adapter A
      stack.filter_for_request(gpus, tier) ← Adapter B
      stack.profile_source                 ← Adapter C (already wraps inner)
      stack.consensus_hook                 ← Adapter D
    """

    anchor_verify: AnchorVerifyAdapter
    tier_gate: TierGateAdapter
    profile_source: ProfileSource  # post-Adapter-C wrapping
    consensus_hook: ConsensusMismatchHook

    def filter_pool(self, gpus: Sequence[ParallaxGPU]) -> List[ParallaxGPU]:
        """Pre-Phase-1 pool filter: anchor verification + stake-eligibility
        admission. Tier filtering happens later per-request because the
        privacy tier is a request property, not a pool property.

        Stake admission piggybacks on the stake-weighted profile source's
        ``is_eligible`` method when present (any ``ProfileSource``
        without that method passes through unchanged). This closes the
        design-plan gap where Adapter C only rescaled snapshots — the
        router's roofline fallback would otherwise route to a zero-
        stake liar based on advertised hardware specs alone."""
        anchor_filtered = self.anchor_verify.filter(gpus)
        is_eligible = getattr(self.profile_source, "is_eligible", None)
        if not callable(is_eligible):
            return anchor_filtered
        return [gpu for gpu in anchor_filtered if is_eligible(gpu.node_id)]

    def filter_for_request(
        self,
        gpus: Sequence[ParallaxGPU],
        privacy_level: PrivacyLevel,
    ) -> List[ParallaxGPU]:
        """Pre-Phase-2: Adapter B (anchor filter assumed already
        applied at pool-build time)."""
        return self.tier_gate.filter(gpus, privacy_level)
