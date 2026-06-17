"""Sprint 558 — production wiring for ParallaxScheduledExecutor.

First sprint in the arc taking /compute/inference from "mock only"
(sprint 438) to "real ParallaxScheduledExecutor". Sprint 546 wired
sprint-419's ActivationDPAware decorator into the chain-RPC
factory, but no production caller invokes that factory — /compute/
inference still falls through to MockInferenceExecutor or 503.

Sprint 558 closes the loop at the OPT-IN PATH:

  PRSM_INFERENCE_EXECUTOR=parallax  → node.py calls
      build_parallax_executor_or_none(node)

The builder reads three component-kind env vars:

  PRSM_PARALLAX_MODEL_CATALOG_FILE — JSON file, top-level dict
      mapping model_id → ModelInfo kwargs.
  PRSM_PARALLAX_TRUST_STACK_KIND   — currently only "mock"; sprint
      560 adds "production" backed by real anchor + stake-lookup
      adapters.
  PRSM_PARALLAX_GPU_POOL_KIND      — currently only "static-empty"
      (no peers); sprint 561 adds "dht" backed by bootstrap-derived
      peer list.

Every missing/invalid piece logs a structured warning naming the
env var + returns None. The daemon stays bootable; /compute/
inference returns the existing 503 from sprint 438 with the
operator-readable detail. No new failure modes for existing
operators — this is a purely additive opt-in surface.

Future sprints fill in the "production" / "dht" kinds + the
chain_executor wiring (sprint 546's factory). Sprint 558 just
locks the contract so the component knobs are operationally
discoverable.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


_KNOWN_TRUST_STACK_KINDS = ("mock", "production")
_KNOWN_GPU_POOL_KINDS = ("static-empty", "dht-backed")

# Sprint 559 — catalog schema versioning. v1 shape:
#   {
#     "schema_version": "v1",
#     "models": { "<model_id>": { ...ModelInfo kwargs... }, ... }
#   }
_CATALOG_SCHEMA_VERSIONS = ("v1",)

# Required ModelInfo fields per entry. ModelInfo's __init__ silently
# accepts ANY kwargs (no positional validation), so operator typos
# like "model_namee" used to construct a degenerate ModelInfo that
# failed at scheduling time with cryptic errors. Sprint 559 catches
# the typo at boot.
_REQUIRED_MODEL_INFO_FIELDS = (
    "model_name",
    "num_layers",
    "hidden_dim",
    "num_attention_heads",
    "num_kv_heads",
    "vocab_size",
    "head_size",
    "intermediate_dim",
)


def _build_mock_trust_stack():
    """Permissive trust stack — passes every GPU through unchanged.
    Clearly labeled in env-var ("mock"); production operators MUST
    pass PRSM_PARALLAX_TRUST_STACK_KIND=production (sprint 560)."""
    from prsm.compute.parallax_scheduling.trust_adapter import (
        AnchorVerifyAdapter,
        ConsensusMismatchHook,
        StakeWeightedTrustAdapter,
        TierGateAdapter,
        TrustStack,
    )
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )

    class _AcceptAllAnchor:
        """Test/dev anchor — claims every node_id is registered."""
        def lookup(self, node_id: str):
            return f"{node_id}.local"

    class _ZeroStakeLookup:
        def get_stake(self, node_id: str) -> int:
            return 0

    # Sprint 562: mock kind uses the same _LoggingChallengeSubmitter
    # as production — fixes the structurally-broken _RecordingSubmitter
    # (had async def submit() but the hook invokes the submitter as
    # a sync Callable). Sample-rate=0.0 in both kinds today so the
    # hook never actually fires; the fix is defensive.
    return TrustStack(
        anchor_verify=AnchorVerifyAdapter(anchor=_AcceptAllAnchor()),
        tier_gate=TierGateAdapter(),
        profile_source=StakeWeightedTrustAdapter(
            inner=InMemoryProfileSource(snapshots={}),
            stake_lookup=_ZeroStakeLookup(),
        ),
        consensus_hook=ConsensusMismatchHook(
            submitter=_build_consensus_submitter(),
            sample_rate=0.0,
        ),
    )


class _LoggingChallengeSubmitter:
    """Sprint 562 — interim consensus_hook submitter.

    The trust-stack's ``ConsensusMismatchHook`` calls this when a
    sampled redundant execution produces a different output than the
    primary chain. The legacy ``_NoOpSubmitter`` silently dropped
    these — operators had zero visibility into compute-side fraud
    attempts. This submitter emits a structured WARNING log naming
    the request_id, both chain stage lists, and both output hashes
    so the audit trail is complete even before the real
    ``ConsensusChallengeSubmitter`` (Phase 7.1x on-chain dispatch)
    is wired.

    The full on-chain submitter requires a translation layer from
    ``ChallengeRecord`` to the ``challengeReceipt(batchId, leaf,
    merkleProof, reason, auxData)`` ABI shape — its own multi-piece
    concern. Logging closes the silent-drop bug TODAY without
    blocking on that translation layer.

    Honors the ``ChallengeSubmitter`` Callable contract: synchronous,
    returns None, must not raise. Malformed records (defensive
    against future schema changes) still log + return None instead
    of raising.
    """

    def __call__(self, record) -> None:
        try:
            request_id = getattr(record, "request_id", "<missing>")
            primary = getattr(
                record, "primary_chain_stages", "<missing>",
            )
            secondary = getattr(
                record, "secondary_chain_stages", "<missing>",
            )
            primary_hash = getattr(
                record, "primary_output_hash", "<missing>",
            )
            secondary_hash = getattr(
                record, "secondary_output_hash", "<missing>",
            )
            logger.warning(
                "Consensus mismatch detected (sprint 562 logging "
                "submitter): request_id=%s primary_chain=%s "
                "secondary_chain=%s primary_output_hash=%s "
                "secondary_output_hash=%s. On-chain challenge "
                "dispatch deferred — Phase 7.1x translation layer "
                "not yet wired.",
                request_id, primary, secondary,
                primary_hash, secondary_hash,
            )
        except Exception as exc:  # noqa: BLE001
            # Defensive — submitter must not raise per the
            # ChallengeSubmitter contract. Log + swallow.
            logger.warning(
                "Consensus mismatch submitter raised on log "
                "formatting (non-fatal): %s", exc,
            )


def _wrap_tier_gate_for_advisory(tier_gate: Any) -> Any:
    """Sprint 702 — PRSM_PARALLAX_TIER_GATE=advisory bypass.

    Live-attest of `privacy_tier=standard` against software-only
    operators was correctly rejected by Adapter B: "no GPU in pool
    has hardware-TEE attestation". This is the right production
    behavior.

    For dev/test environments that need to exercise the activation-DP
    injection code path (sprints 295/413/414) without real TEE
    hardware, this advisory mode wraps the TierGateAdapter with a
    permissive filter that passes ALL GPUs through regardless of
    tier_attestation. Production posture is UNCHANGED — default is
    enforced, typos fall through to enforced (no accidental
    disabling), loud WARNING fires on construction.

    Mirrors sprint 686's `_wrap_stake_lookup_for_eligibility`
    pattern.
    """
    mode = (
        os.environ.get("PRSM_PARALLAX_TIER_GATE", "")
        .strip().lower()
    )
    if mode != "advisory":
        return tier_gate

    logger.warning(
        "Sprint 702: PRSM_PARALLAX_TIER_GATE=advisory — Adapter B "
        "hardware-TEE attestation requirement DISABLED. Every GPU "
        "passes through the tier filter regardless of tier_attestation. "
        "Activation-DP injection still fires for tier ≥ standard "
        "requests, but the TEE attestation chain is software-only. "
        "This mode is intended for dev/test exercise of the DP code "
        "path only. Production deploy MUST flip back to "
        "PRSM_PARALLAX_TIER_GATE=enforced (or unset) once operators "
        "have real TEE hardware (SGX/TDX/SEV-SNP/TrustZone/Secure "
        "Enclave) attestation backends wired."
    )

    class _PermitAllTierGate:
        """Sprint 702 advisory wrapper. Returns all gpus regardless
        of privacy_level / tier_attestation."""
        def filter(self, gpus, privacy_level):  # noqa: D401
            return list(gpus)

    return _PermitAllTierGate()


def _wrap_stake_lookup_for_eligibility(stake_lookup: Any) -> Any:
    """Sprint 686 — PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory bypass.

    Returns the input stake_lookup unchanged unless the operator
    has explicitly opted into ``advisory`` mode, in which case the
    lookup is wrapped to return MIN_STAKE_FOR_PARTICIPATION for
    every node_id — effectively disabling stake-based eligibility
    filtering.

    Why this exists: live-attest of sprint 685's DHT-backed pool
    surfaced F31 (AnchorMediatedStakeLookup passes the anchor's
    base64 pubkey to StakeManagerClient.stake_of which expects an
    ETH address — the contract has no node_id → operator-address
    mapping). Until F31 is fixed properly via a new on-chain
    mapping OR via hardware_profile.operator_address propagation
    in the trust stack itself, advisory mode is the live-attest
    path.

    Defaults to enforced (the safe production posture). Unknown
    values also default to enforced — no typo'd env var can
    accidentally disable enforcement. Advisory mode logs a loud
    WARNING at construction so operators don't silently run
    without stake enforcement.
    """
    mode = (
        os.environ.get("PRSM_PARALLAX_STAKE_ELIGIBILITY", "")
        .strip().lower()
    )
    if mode != "advisory":
        return stake_lookup

    logger.warning(
        "Sprint 686: PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory — "
        "stake-based pool-eligibility filtering DISABLED. Every "
        "anchor-registered peer is treated as stake-eligible "
        "regardless of on-chain bond. This mode is intended for "
        "pre-stake-ceremony live-attest only. Production deploy "
        "MUST flip to PRSM_PARALLAX_STAKE_ELIGIBILITY=enforced "
        "(or unset) once operators have posted real stake."
    )

    class _PermitAllStakeLookup:
        def get_stake(self, node_id: str) -> int:
            # Sprint 686 — return MIN_STAKE_FOR_PARTICIPATION so
            # is_eligible() returns True for everyone, but don't
            # claim more than that (StakeWeightedTrustAdapter's
            # latency-rescaling still treats them as low-confidence).
            from prsm.compute.parallax_scheduling.trust_adapter import (
                MIN_STAKE_FOR_PARTICIPATION,
            )
            return MIN_STAKE_FOR_PARTICIPATION

    return _PermitAllStakeLookup()


class PoolBackedStakeLookup:
    """Sprint 690 F31 proper fix — read stake from the latest pool
    snapshot rather than via the broken anchor-mediated path.

    The DHT-backed GpuPoolProvider (sprint 682) already populates
    ``ParallaxGPU.stake_amount`` correctly when a peer advertises
    ``operator_address`` in its hardware_profile (sprint 690 piece
    1 plumbs this from PRSM_OPERATOR_ADDRESS env). Sprint 683's
    OnChainStakeReader reads the real on-chain stake via
    StakeManagerClient.stake_of(operator_address).amount_wei
    inside the pool provider, with 60s TTL caching.

    This lookup just reads the ParallaxGPU.stake_amount field on
    the GPU matching the queried node_id. No anchor.lookup call,
    no pubkey-vs-ETH-address misuse. Closes the F31 advisory-
    mode bypass shipped in sprint 686.

    Behavior:
      - provider raises → 0 (defensive — pre-route filter must
        never crash)
      - node_id absent from pool → 0 (matches
        AnchorMediatedStakeLookup's "unknown node = unstaked"
        semantics)
      - matched gpu → gpu.stake_amount (already populated by
        sprint 683's OnChainStakeReader path)
    """

    def __init__(self, *, pool_provider: Any) -> None:
        if pool_provider is None or not callable(pool_provider):
            raise RuntimeError(
                "PoolBackedStakeLookup requires a callable "
                "pool_provider"
            )
        self._pool_provider = pool_provider

    def get_stake(self, node_id: str) -> int:
        try:
            pool = list(self._pool_provider())
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "PoolBackedStakeLookup: pool_provider raised %s; "
                "returning 0 stake for %s",
                type(exc).__name__, node_id,
            )
            return 0
        for gpu in pool:
            if getattr(gpu, "node_id", None) == node_id:
                return int(getattr(gpu, "stake_amount", 0) or 0)
        return 0


class AnchorMediatedStakeLookup:
    """Sprint 561 — production stake_lookup. Maps trust-stack node_id
    (32-char hex publisher ID) → ETH address via the anchor's
    ``lookup`` method, then queries on-chain stake via the
    StakeManagerClient.

    Conservative fail-soft semantics:
      - anchor.lookup returns None  → 0 stake (unknown node)
      - stake_of raises (RPC error) → 0 stake (logged at DEBUG to
                                      avoid log spam on transient
                                      network errors)

    Returning 0 under uncertainty matches the
    ``StakeWeightedTrustAdapter`` design: zero stake = "effectively
    excluded from routing" — the conservative posture under any
    resolution failure.
    """

    def __init__(self, *, anchor, stake_client):
        if anchor is None or not hasattr(anchor, "lookup"):
            raise RuntimeError(
                "AnchorMediatedStakeLookup requires an anchor with "
                ".lookup(node_id) → Optional[str]"
            )
        if stake_client is None or not hasattr(
            stake_client, "stake_of",
        ):
            raise RuntimeError(
                "AnchorMediatedStakeLookup requires a stake_client "
                "with .stake_of(provider) → StakeRecord"
            )
        self._anchor = anchor
        self._stake = stake_client

    def get_stake(self, node_id: str) -> int:
        try:
            eth = self._anchor.lookup(node_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "AnchorMediatedStakeLookup: anchor.lookup raised "
                "for node_id=%s: %s", node_id, exc,
            )
            return 0
        if not eth:
            return 0
        try:
            record = self._stake.stake_of(eth)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "AnchorMediatedStakeLookup: stake_of raised for "
                "node_id=%s eth=%s: %s", node_id, eth, exc,
            )
            return 0
        return int(getattr(record, "amount_wei", 0))


def _build_production_trust_stack_or_none(pool_provider=None):
    """Sprint 560 + 561 — `production` trust-stack kind.

    Wires REAL components from operator-supplied env vars:

      - anchor          : PublisherKeyAnchorClient @
                          PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS
                          (required — no fallback).
      - stake_lookup    : (sprint 561) AnchorMediatedStakeLookup
                          via StakeManagerClient @
                          PRSM_STAKE_BOND_ADDRESS. Falls back to
                          ZeroStakeLookup PLACEHOLDER + warning
                          when the env var is unset OR construction
                          fails (operator on partial-production).

    Still PLACEHOLDER (pending sprint 562):
      - profile_source  → empty InMemoryProfileSource.
      - consensus_hook  → no-op submitter.

    Returns None when the anchor address isn't set — there's no
    sensible fallback for the anchor under `production` kind.

    Caller logs the per-component REAL/PLACEHOLDER enumeration at
    INFO level so operators see what's actually verified.
    """
    from prsm.compute.parallax_scheduling.trust_adapter import (
        AnchorVerifyAdapter,
        ConsensusMismatchHook,
        StakeWeightedTrustAdapter,
        TierGateAdapter,
        TrustStack,
    )
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )

    # Sprint 580 — anchor construction extracted to module helper
    # so chain_executor Phase 2 can share. None handling stays here
    # since "production trust-stack requires anchor" is a trust-stack
    # invariant, not an anchor-helper concern.
    anchor_addr = os.environ.get(
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "",
    ).strip()
    if not anchor_addr:
        logger.warning(
            "Sprint 560 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_TRUST_STACK_KIND=production requires "
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS (the Phase-3.x.3 "
            "publisher-key anchor contract address). Set it to "
            "the deployed anchor address on your chain, or use "
            "PRSM_PARALLAX_TRUST_STACK_KIND=mock for dev."
        )
        return None
    anchor = _build_anchor_or_none()
    if anchor is None:
        # _build_anchor_or_none already logged the underlying
        # construction failure; the trust-stack caller surfaces
        # the kind-specific guidance.
        return None

    # Sprint 561 — production stake_lookup. PRSM_STAKE_BOND_ADDRESS
    # set → build real StakeManagerClient + AnchorMediatedStakeLookup.
    # Unset OR construction fails → fall back to ZeroStakeLookup
    # placeholder with a structured warning (operator on partial-
    # production keeps a working daemon).
    class _ZeroStakeLookup:
        def get_stake(self, node_id: str) -> int:
            return 0

    # Sprint 580 — rpc_url resolved here (was earlier in pre-580
    # inline construction; sprint-561 StakeManagerClient still needs it).
    rpc_url = os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    stake_addr = os.environ.get(
        "PRSM_STAKE_BOND_ADDRESS", "",
    ).strip()
    if stake_addr:
        # Sprint 690 F31 proper fix — prefer PoolBackedStakeLookup
        # when a pool_provider is available. The DHT pool provider
        # (sprint 682+683) already populates ParallaxGPU.stake_amount
        # via OnChainStakeReader → operator_address → stake_of. The
        # pool-backed lookup just reads that already-correct value
        # instead of doing a doomed-from-the-start
        # anchor.lookup(node_id) → stake_of(pubkey) dance (F31).
        if pool_provider is not None:
            stake_lookup = PoolBackedStakeLookup(pool_provider=pool_provider)
            stake_lookup_status = "REAL (pool-backed, sprint 690)"
        else:
            try:
                from prsm.economy.web3.stake_manager import (
                    StakeManagerClient,
                )
                stake_client = StakeManagerClient(
                    contract_address=stake_addr,
                    rpc_url=rpc_url,
                )
                stake_lookup = AnchorMediatedStakeLookup(
                    anchor=anchor, stake_client=stake_client,
                )
                stake_lookup_status = (
                    "LEGACY (AnchorMediatedStakeLookup — F31 bug, "
                    "always returns 0; supply pool_provider for "
                    "sprint-690 PoolBackedStakeLookup)"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Sprint 561 ParallaxScheduledExecutor wiring: "
                    "StakeManagerClient construction failed against %s "
                    "— %s. Falling back to ZeroStakeLookup placeholder.",
                    stake_addr, exc,
                )
                stake_lookup = _ZeroStakeLookup()
                stake_lookup_status = (
                    "PLACEHOLDER (StakeManagerClient construction failed)"
                )
    else:
        logger.warning(
            "Sprint 561 ParallaxScheduledExecutor wiring: "
            "PRSM_STAKE_BOND_ADDRESS not set; stake_lookup falls "
            "back to ZeroStakeLookup placeholder. Set it to the "
            "deployed StakeBond contract address (Base mainnet "
            "default: see networks.py:stake_bond) to wire real "
            "on-chain stake verification."
        )
        stake_lookup = _ZeroStakeLookup()
        stake_lookup_status = "PLACEHOLDER (zero stake)"

    # Sprint 686 — optional advisory-mode bypass for live-attest
    # before stake ceremony. Default enforced.
    stake_lookup = _wrap_stake_lookup_for_eligibility(stake_lookup)

    # Sprint 562 — consensus_hook submitter upgraded from
    # structurally-broken _NoOpSubmitter (had .submit() method but
    # hook invokes the submitter as a Callable) to a
    # _LoggingChallengeSubmitter that emits WARNING per mismatch.
    # On-chain Phase 7.1x ConsensusChallengeSubmitter wiring needs
    # a translation layer (ChallengeRecord → on-chain ABI shape) —
    # deferred until that translation lands.
    logger.info(
        "Sprint 560+561+562 ParallaxScheduledExecutor wiring: "
        "trust_stack=production. anchor=REAL "
        "(PublisherKeyAnchorClient @ %s); stake_lookup=%s; "
        "profile_source=PLACEHOLDER (empty InMemoryProfileSource "
        "— ProfileDHT requires multi-host send_message + peers, "
        "out of scope for single-node); consensus_hook=LOGGING "
        "(WARNING per ChallengeRecord — on-chain dispatch via "
        "Phase 7.1x ConsensusChallengeSubmitter deferred pending "
        "translation layer). Operators must not treat partial-"
        "production as fully production-verified until all 4 "
        "components are REAL.",
        anchor_addr, stake_lookup_status,
    )
    # Sprint 576 — inner ProfileSource selection is env-driven via
    # PRSM_PARALLAX_PROFILE_SOURCE_KIND (default in_memory preserves
    # behavior; "dht" is the Phase 2 hook).
    return TrustStack(
        anchor_verify=AnchorVerifyAdapter(anchor=anchor),
        tier_gate=_wrap_tier_gate_for_advisory(TierGateAdapter()),
        profile_source=StakeWeightedTrustAdapter(
            inner=_build_inner_profile_source(),
            stake_lookup=stake_lookup,
        ),
        consensus_hook=ConsensusMismatchHook(
            submitter=_build_consensus_submitter(),
            sample_rate=0.0,
        ),
    )


def _build_anchor_or_none():
    """Sprint 580 — env-driven PublisherKeyAnchorClient construction.

    Single source of truth for anchor resolution. Used by:
      - ``_build_production_trust_stack_or_none`` (existing caller)
      - ``_build_chain_executor`` (sprint-581+ Phase 2 — needs the
        anchor for ``make_rpc_chain_executor``)

    Semantics:
      - no address resolvable (env unset AND networks.py default
        unset for current network) → return None
      - construction succeeds → return PublisherKeyAnchorClient
      - construction fails → log WARNING + return None

    Sprint 629 fix: routes through ``prsm.config.networks.resolve_endpoints()``
    so the baked-in networks.py default (e.g. Base mainnet Phase 3.x.3
    anchor 0xd811ad99... per sprint 621) is honored when
    PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS is unset. Pre-629 the helper
    only honored explicit env; fresh-install operators with
    PRSM_PARALLAX_TRUST_STACK_KIND=production silently got the stub
    anchor when they should have gotten the deployed contract.

    RPC URL also comes from resolve_endpoints() so PRSM_NETWORK +
    BASE_RPC_URL overrides apply uniformly.
    """
    try:
        from prsm.config.networks import resolve_endpoints
        endpoints = resolve_endpoints()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 629 _build_anchor_or_none: "
            "resolve_endpoints() failed — %s. Returning None.",
            exc,
        )
        return None

    anchor_addr = (endpoints.publisher_key_anchor or "").strip()
    if not anchor_addr:
        return None
    rpc_url = endpoints.rpc_url
    try:
        from prsm.security.publisher_key_anchor.client import (
            PublisherKeyAnchorClient,
        )
        return PublisherKeyAnchorClient(
            contract_address=anchor_addr,
            rpc_url=rpc_url,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 580 _build_anchor_or_none: "
            "PublisherKeyAnchorClient construction failed against "
            "%s — %s. Anchor unavailable; returning None.",
            anchor_addr, exc,
        )
        return None


def _build_chain_executor(node: Any) -> Any:
    """Sprint 578 — env-driven chain_executor construction.

    Read PRSM_PARALLAX_CHAIN_EXECUTOR_KIND:
      - unset / "stub" → _StubChainExecutor (current default;
        raises on execute_chain so callers must hit a non-empty
        pool first — sprint 558 invariant).
      - "rpc" → Phase 2 wires real via make_rpc_chain_executor
        with node.identity/transport/anchor. Phase 1 logs WARNING
        + falls back to stub so daemon stays up.
      - anything else → WARNING + stub fallback.

    Mirrors sprints-576/577 plumbing pattern. Phase 2 (real RPC
    chain executor) becomes a non-churn additive change here.
    """
    kind_raw = os.environ.get(
        "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND", "",
    ).strip().lower()
    kind = kind_raw or "stub"
    if kind == "stub":
        return _StubChainExecutor()
    if kind == "rpc":
        # Sprint 598 (Phase 2D step 4) — real RPC chain executor.
        # Construction needs: settler_identity, send_message
        # (sprint-596 adapter), anchor (sprint-580 helper),
        # address_resolver (sprint-593 helper). Falls back to stub
        # when any dependency is missing, with structured logging
        # so operators see WHICH piece blocked production wiring.
        missing = []
        if getattr(node, "identity", None) is None:
            missing.append("node.identity")
        if getattr(node, "transport", None) is None:
            missing.append("node.transport")
        if getattr(node, "_loop", None) is None:
            missing.append("node._loop (set in sprint 595 at PRSMNode.start)")
        anchor = _build_anchor_or_none()
        if anchor is None:
            missing.append(
                "anchor (set PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS)"
            )
        if missing:
            logger.warning(
                "Sprint 598 _build_chain_executor: rpc kind cannot "
                "wire because dependencies are missing: %s. Falling "
                "back to _StubChainExecutor. Use "
                "`prsm node section7-readiness` for full operator "
                "preflight (sprint 585).",
                ", ".join(missing),
            )
            return _StubChainExecutor()
        try:
            from prsm.compute.chain_rpc.factories import (
                make_rpc_chain_executor,
            )
            from prsm.node.chain_executor_adapters import (
                build_send_message_adapter,
                build_address_resolver,
                build_hf_prompt_encoder,
                build_hf_output_decoder,
            )
            # Sprint 615 (Phase 2F-5f) — env-wired prompt_encoder.
            # PRSM_PARALLAX_PROMPT_ENCODER_KIND=huggingface +
            # PRSM_PARALLAX_HF_MODEL_ID=<id> → real HF tokenizer +
            # embedding lookup at chain head. Default (unset) keeps
            # the factory's utf8 default — fine for wire tests.
            prompt_encoder_kwarg = {}
            _pe_kind = (os.environ.get(
                "PRSM_PARALLAX_PROMPT_ENCODER_KIND", "",
            ) or "").strip().lower()
            if _pe_kind == "huggingface":
                _hf_id = (os.environ.get(
                    "PRSM_PARALLAX_HF_MODEL_ID", "",
                ) or "").strip()
                if _hf_id:
                    _hf_device = (os.environ.get(
                        "PRSM_PARALLAX_HF_DEVICE", "",
                    ) or "").strip() or "cpu"
                    prompt_encoder_kwarg["prompt_encoder"] = (
                        build_hf_prompt_encoder(
                            model_id=_hf_id, device=_hf_device,
                        )
                    )
                else:
                    logger.warning(
                        "Sprint 615 _build_chain_executor: "
                        "PRSM_PARALLAX_PROMPT_ENCODER_KIND=huggingface "
                        "but PRSM_PARALLAX_HF_MODEL_ID unset; "
                        "falling back to utf8 default encoder."
                    )
            # Sprint 616 (Phase 2F-5g) — env-wired output_decoder.
            # PRSM_PARALLAX_OUTPUT_DECODER_KIND=huggingface +
            # PRSM_PARALLAX_HF_MODEL_ID=<id> → real HF tokenizer.decode
            # at chain tail.
            output_decoder_kwarg = {}
            _od_kind = (os.environ.get(
                "PRSM_PARALLAX_OUTPUT_DECODER_KIND", "",
            ) or "").strip().lower()
            if _od_kind == "huggingface":
                _hf_id = (os.environ.get(
                    "PRSM_PARALLAX_HF_MODEL_ID", "",
                ) or "").strip()
                if _hf_id:
                    _hf_device = (os.environ.get(
                        "PRSM_PARALLAX_HF_DEVICE", "",
                    ) or "").strip() or "cpu"
                    output_decoder_kwarg["output_decoder"] = (
                        build_hf_output_decoder(
                            model_id=_hf_id, device=_hf_device,
                        )
                    )
                else:
                    logger.warning(
                        "Sprint 616 _build_chain_executor: "
                        "PRSM_PARALLAX_OUTPUT_DECODER_KIND=huggingface "
                        "but PRSM_PARALLAX_HF_MODEL_ID unset; "
                        "falling back to default decoder."
                    )
            # Sprint 687 — operator-tunable timeout. Default 30s
            # covers a single layer execution + network roundtrip
            # for remote stages, but local self-dispatch can take
            # 30+s on cold gpt2 load on small droplets. Env override
            # lets operators raise it without code change.
            _timeout_raw = os.environ.get(
                "PRSM_PARALLAX_SEND_MESSAGE_TIMEOUT_S", "",
            ).strip()
            _send_timeout = 30.0
            if _timeout_raw:
                try:
                    _send_timeout = max(1.0, float(_timeout_raw))
                except ValueError:
                    logger.debug(
                        "PRSM_PARALLAX_SEND_MESSAGE_TIMEOUT_S=%r "
                        "not numeric; using 30.0s default",
                        _timeout_raw,
                    )
            # Sprint 691 F40 fix — wire token_stream_send_message
            # so /compute/inference/stream can dispatch streaming
            # requests for self-dispatch (single-host streaming).
            # Remote streaming raises a structured error at dispatch
            # time per the adapter's contract.
            from prsm.node.chain_executor_adapters import (
                build_token_stream_send_message_adapter,
            )
            executor = make_rpc_chain_executor(
                settler_identity=node.identity,
                send_message=build_send_message_adapter(
                    node, timeout=_send_timeout,
                ),
                token_stream_send_message=(
                    build_token_stream_send_message_adapter(node)
                ),
                anchor=anchor,
                address_resolver=build_address_resolver(node),
                **prompt_encoder_kwarg,
                **output_decoder_kwarg,
            )
            logger.info(
                "Sprint 598 _build_chain_executor: real "
                "RpcChainExecutor constructed (anchor=REAL, "
                "send_message=PHASE-2D-bridged over transport, "
                "address_resolver=transport.peers lookup). "
                "Note: response routing (sprint 599 follow-on) "
                "still pending — chain dispatches will time out "
                "until handle_chain_executor_response is wired "
                "into transport's message dispatch."
            )
            return executor
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Sprint 598 _build_chain_executor: "
                "make_rpc_chain_executor construction raised %s: %s. "
                "Falling back to _StubChainExecutor.",
                type(exc).__name__, exc,
            )
            return _StubChainExecutor()
    logger.warning(
        "Sprint 578 ParallaxScheduledExecutor wiring: "
        "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=%r is unknown; falling "
        "back to _StubChainExecutor. Valid values: stub, rpc "
        "(Phase 2 pending).",
        kind_raw,
    )
    return _StubChainExecutor()


def _build_consensus_submitter():
    """Sprint 577 — env-driven consensus_hook submitter construction.

    Read PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND:
      - unset / "logging" → _LoggingChallengeSubmitter (current
        default; WARNING per ChallengeRecord; no on-chain dispatch).
      - "onchain" → Phase 2 returns real OnChainChallengeSubmitter
        that maps ChallengeRecord → Phase 7.1x
        ConsensusChallengeSubmitter ABI + dispatches. Phase 1 logs
        WARNING + falls back to logging so daemon stays up.
      - anything else → WARNING + logging fallback.

    Mirrors sprint-576's profile_source helper pattern. Phase 2
    (real on-chain) becomes a non-churn additive change here.
    """
    kind_raw = os.environ.get(
        "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND", "",
    ).strip().lower()
    kind = kind_raw or "logging"
    if kind == "logging":
        return _LoggingChallengeSubmitter()
    if kind == "onchain":
        logger.warning(
            "Sprint 577 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND=onchain set but "
            "Phase 2 (real OnChainChallengeSubmitter w/ "
            "ChallengeRecord → Phase 7.1x ABI translation layer) "
            "has not landed yet — falling back to "
            "_LoggingChallengeSubmitter. Track sprint 578+ for the "
            "actual on-chain dispatch wiring."
        )
        return _LoggingChallengeSubmitter()
    logger.warning(
        "Sprint 577 ParallaxScheduledExecutor wiring: "
        "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND=%r is unknown; "
        "falling back to _LoggingChallengeSubmitter. Valid values: "
        "logging, onchain (Phase 2 pending).",
        kind_raw,
    )
    return _LoggingChallengeSubmitter()


def _build_inner_profile_source():
    """Sprint 576 — env-driven inner ProfileSource construction.

    Read PRSM_PARALLAX_PROFILE_SOURCE_KIND:
      - unset / "in_memory" → InMemoryProfileSource(snapshots={})
        (current default behavior; existing operators unaffected)
      - "dht"               → Phase 2 will return ProfileDHT(...);
        Phase 1 logs a structured warning + falls back to
        in_memory so the daemon stays up.
      - anything else       → warn + in_memory fallback.

    The trust-stack constructor calls this helper instead of
    hardcoding the InMemoryProfileSource — Phase 2 (real DHT
    wiring) becomes a non-churn additive change here.
    """
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )
    kind_raw = os.environ.get(
        "PRSM_PARALLAX_PROFILE_SOURCE_KIND", "",
    ).strip().lower()
    kind = kind_raw or "in_memory"
    if kind == "in_memory":
        return InMemoryProfileSource(snapshots={})
    if kind == "dht":
        logger.warning(
            "Sprint 576 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_PROFILE_SOURCE_KIND=dht set but Phase 2 "
            "(real ProfileDHT integration) has not landed yet — "
            "falling back to InMemoryProfileSource(snapshots={}). "
            "Track sprint 577+ for the actual DHT-backed source."
        )
        return InMemoryProfileSource(snapshots={})
    logger.warning(
        "Sprint 576 ParallaxScheduledExecutor wiring: "
        "PRSM_PARALLAX_PROFILE_SOURCE_KIND=%r is unknown; falling "
        "back to InMemoryProfileSource. Valid values: in_memory, "
        "dht (Phase 2 pending).",
        kind_raw,
    )
    return InMemoryProfileSource(snapshots={})


def _build_static_empty_pool_provider():
    """Pool provider that returns an empty list. /compute/inference
    requests will fail Phase-1 allocation with a clear error; this
    is the explicit "no peers wired" path. Sprint 561 replaces with
    a DHT-backed provider."""
    def _provider():
        return []
    return _provider


class _StubChainExecutor:
    """Sprint 558 placeholder. The real chain executor comes from
    sprint 546's make_rpc_chain_executor factory in a future sprint.
    Today, no /compute/inference request gets this far because the
    static-empty pool fails Phase-1 first."""

    def execute_chain(self, *, request, chain, **kwargs):
        raise RuntimeError(
            "Sprint 558 stub chain executor — not callable until "
            "sprint 562+ wires make_rpc_chain_executor."
        )


def _module_available(name: str) -> bool:
    """True if a module can be imported, without importing it."""
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def diagnose_inference_executor_unavailable(
    env=None,
    module_check=None,
    path_check=None,
) -> str:
    """Sprint 1033 — the actionable reason the inference executor is unavailable.

    Returned in the /compute/inference 503 detail so an operator gets a FIX
    instead of grepping the daemon log. The executor build
    (build_parallax_executor_or_none) already logs the real reason at startup;
    this reconstructs the actionable hint at request time from the env + import
    availability + catalog file. Live-motivated by the Tier-1 bench, where a
    GPU operator who installed only `.[ml]` (not `.[blockchain]`) hit a generic
    503 that hid the real cause (web3/eth_account missing → anchor build failed).

    ``env`` / ``module_check`` / ``path_check`` are injectable for testing;
    defaults read os.environ / importlib / os.path. Always returns a non-empty
    string and never raises — it sits on the 503 path.
    """
    env = env if env is not None else os.environ
    module_check = module_check or _module_available
    path_check = path_check or os.path.exists

    executor = (env.get("PRSM_INFERENCE_EXECUTOR", "") or "").strip().lower()
    if not executor:
        return (
            "PRSM_INFERENCE_EXECUTOR is unset — set it to 'local' (single-node "
            "real HuggingFace inference) or 'parallax' (the verifiable GPU "
            "pool). A bare default node serves no inference by design."
        )
    if executor == "parallax":
        if not (module_check("web3") and module_check("eth_account")):
            return (
                "PRSM_INFERENCE_EXECUTOR=parallax needs the on-chain anchor "
                "client, but web3/eth_account are not installed — run "
                "`pip install -e '.[blockchain]'` (they are NOT in the [ml] "
                "extra)."
            )
        catalog = (
            env.get("PRSM_PARALLAX_MODEL_CATALOG_FILE", "") or ""
        ).strip()
        if not catalog:
            return (
                "PRSM_PARALLAX_MODEL_CATALOG_FILE is unset — point it at "
                "config/parallax/model_catalog.json."
            )
        if not path_check(catalog):
            return (
                f"PRSM_PARALLAX_MODEL_CATALOG_FILE={catalog!r} does not exist "
                "— point it at a valid catalog "
                "(config/parallax/model_catalog.json)."
            )
        for var in (
            "PRSM_PARALLAX_TRUST_STACK_KIND",
            "PRSM_PARALLAX_GPU_POOL_KIND",
        ):
            if not (env.get(var, "") or "").strip():
                return (
                    f"{var} is unset — required for the parallax executor "
                    "(see docs/operations/parallax-inference-deploy.md)."
                )
        return (
            "Parallax env looks set but the executor failed to build — check "
            "the daemon log for the build_parallax_executor_or_none WARNING "
            "naming the missing piece (e.g. the anchor RPC was unreachable)."
        )
    if executor == "local":
        if not (module_check("torch") and module_check("transformers")):
            return (
                "PRSM_INFERENCE_EXECUTOR=local needs torch + transformers — "
                "run `pip install -e '.[ml]'`."
            )
        return (
            "PRSM_INFERENCE_EXECUTOR=local is set but the executor failed to "
            "build — check the daemon log for the specific reason."
        )
    if executor in ("mock", "mock-streaming"):
        return (
            f"PRSM_INFERENCE_EXECUTOR={executor} is set but the mock executor "
            "failed to build — check the daemon log."
        )
    return (
        f"PRSM_INFERENCE_EXECUTOR={executor!r} is not a recognized value "
        "(expected one of: local, parallax, mock, mock-streaming)."
    )


# Sentinel-safe model_id allowlist — same pattern FilesystemModelRegistry
# enforces (^[A-Za-z0-9._-]+$). Validate BEFORE any filesystem Path-join so a
# traversal model_id can't even stat/read outside the registry root.
_SAFE_FS_MODEL_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _manifest_layer_count(manifest_dict):
    """The staged layer count (layer_range hi) from a manifest dict, or None if
    it can't be determined (so we never false-positive a stale warning)."""
    try:
        shards = manifest_dict.get("shards") or []
        layer_range = shards[0].get("layer_range")
        if layer_range and len(layer_range) >= 2:
            return int(layer_range[1])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    return None


def _resolve_num_layers_from_catalog(model_id: str, catalog_path: str):
    """num_layers for model_id from the parallax catalog, or None if absent /
    unreadable / invalid. Never raises (mirrors stage_hf_model._resolve_num_layers
    but returns None instead of raising — the daemon caller skips on None rather
    than guessing a layer count)."""
    try:
        catalog = json.loads(Path(catalog_path).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    entry = (catalog.get("models") or {}).get(model_id)
    if not isinstance(entry, dict):
        return None
    n = entry.get("num_layers")
    # bool is an int subclass in Python — exclude True/False explicitly so a
    # JSON boolean can't masquerade as a layer count (would yield (0,1)).
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return None
    return n


def _resolve_num_layers_from_hf_config(model_id: str):
    """sp1134 — authoritative num_layers from the model's own locally-cached HF
    config (AutoConfig), the fallback when the parallax catalog lacks the model so
    an operator can serve ANY cached production model without hand-editing the
    catalog (the sp1034 "last manual step", generalized). The HF runner loads these
    same weights at inference, so the config is local. Returns None on any failure
    or if the config declares no usable layer count — NEVER guesses (preserves the
    never-defaulted staging invariant; a None here means staging is skipped, not
    registered with a wrong layer_range). Mirrors sp1133's config-derivation but is
    stricter: no 12-layer default."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 — offline / uncached / missing dep → no guess
        return None
    for attr in ("num_hidden_layers", "n_layer"):
        v = getattr(cfg, attr, None)
        # bool is an int subclass — exclude True/False so a stray flag can't pose
        # as a layer count.
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            continue
        return v
    return None


def ensure_hf_model_staged(identity, env=None) -> str:
    """Sprint 1034 — auto-stage the configured HF model into the local
    FilesystemModelRegistry at daemon startup so an operator no longer has to run
    scripts/stage_hf_model.py by hand on every node (the last manual step the
    Tier-1 cross-host bench exposed: an unstaged model → the layer_stage chain
    server returns MODEL_NOT_FOUND at inference time).

    No-op unless PRSM_INFERENCE_EXECUTOR=parallax AND PRSM_PARALLAX_HF_MODEL_ID
    + PRSM_MODEL_REGISTRY_ROOT are set AND ``identity`` is present. num_layers is
    resolved from the parallax catalog (PRSM_PARALLAX_MODEL_CATALOG_FILE, else
    the bundled config/parallax/model_catalog.json) and NEVER defaulted — a model
    absent from the catalog is skipped, not registered with a wrong layer count.

    Idempotent + publisher-aware: skips if THIS node already staged the model;
    REFUSES to clobber a manifest signed by a different publisher (mirrors
    stage_hf_model.py). The sentinel manifest matches the script byte-for-byte
    (single shard, 32 zero bytes, layer_range=(0,num_layers)); the HF runner
    loads real weights from the HF cache at inference, so the registry only needs
    the model "known" with the right layer_range.

    NEVER raises — it runs on the daemon-start path (the parallax branch in
    node.initialize() is not otherwise try/except-wrapped). Returns a short
    status for the caller to log:
      'registered' | 'already' | 'refused:different-publisher' |
      'skipped:<reason>' | 'error:<detail>'.
    """
    try:
        environ = env if env is not None else os.environ
        executor = (
            environ.get("PRSM_INFERENCE_EXECUTOR", "") or ""
        ).strip().lower()
        if executor != "parallax":
            return "skipped:not-parallax"
        if identity is None or not getattr(identity, "node_id", ""):
            return "skipped:no-identity"
        model_id = (environ.get("PRSM_PARALLAX_HF_MODEL_ID", "") or "").strip()
        if not model_id:
            return "skipped:no-model-id"
        # Reject traversal / unsafe ids BEFORE any Path-join (defense-in-depth;
        # FilesystemModelRegistry.register would also reject at write time).
        if model_id in (".", "..") or not _SAFE_FS_MODEL_ID.match(model_id):
            return "skipped:invalid-model-id"
        root_str = (environ.get("PRSM_MODEL_REGISTRY_ROOT", "") or "").strip()
        if not root_str:
            return "skipped:no-registry-root"
        catalog_path = (
            environ.get("PRSM_PARALLAX_MODEL_CATALOG_FILE", "") or ""
        ).strip()
        if not catalog_path:
            # Fall back to the catalog bundled in the repo (config/parallax/).
            catalog_path = str(
                Path(__file__).resolve().parents[2]
                / "config" / "parallax" / "model_catalog.json"
            )
        num_layers = _resolve_num_layers_from_catalog(model_id, catalog_path)
        if num_layers is None:
            # sp1134 — catalog miss: derive authoritatively from the model's own
            # locally-cached HF config so any configured model auto-stages without
            # a manual catalog edit. Still never guesses — a derivation failure
            # falls through to the skip below (no wrong layer_range registered).
            num_layers = _resolve_num_layers_from_hf_config(model_id)
        if num_layers is None:
            return f"skipped:no-num-layers:{model_id}"

        registry_root = Path(root_str)
        # Publisher-aware idempotency: FilesystemModelRegistry.register() raises
        # on ANY pre-existing manifest with no publisher comparison, so check the
        # manifest ourselves first (mirrors stage_hf_model.py:169-194).
        manifest_path = registry_root / model_id / "manifest.json"
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text())
                if existing.get("publisher_node_id", "") == identity.node_id:
                    staged = _manifest_layer_count(existing)
                    if staged is not None and staged != num_layers:
                        # The catalog's layer count changed after this model was
                        # staged. Surface the drift (operator should re-stage)
                        # rather than silently trusting a stale manifest; we
                        # don't auto-clobber our own manifest at startup.
                        return (
                            f"already:stale-layer-range:{staged}!={num_layers}"
                        )
                    return "already"
                return "refused:different-publisher"
            except (OSError, json.JSONDecodeError, ValueError):
                pass  # unreadable manifest → fall through and (re)register

        # Lazy imports — keep the heavy registry/sharding modules off the
        # module-import path (default operators never trigger them).
        from prsm.compute.model_registry.registry import (
            FilesystemModelRegistry,
            ModelAlreadyRegisteredError,
        )
        from prsm.compute.model_sharding.models import ShardedModel, ModelShard

        registry_root.mkdir(parents=True, exist_ok=True)
        registry = FilesystemModelRegistry(root=registry_root)
        shard = ModelShard(
            shard_id=f"{model_id}-shard-0",
            model_id=model_id,
            shard_index=0,
            total_shards=1,
            tensor_data=b"\x00" * 32,
            tensor_shape=(32,),
            layer_range=(0, num_layers),
            size_bytes=32,
        )
        model = ShardedModel(
            model_id=model_id,
            model_name=f"HF runner sentinel for {model_id}",
            total_shards=1,
            shards=[shard],
        )
        try:
            registry.register(model, identity=identity)
        except ModelAlreadyRegisteredError:
            return "already"  # race between the pre-check and register()
        return "registered"
    except Exception as exc:  # noqa: BLE001 — must never crash daemon start
        return f"error:{type(exc).__name__}: {exc}"


def build_parallax_executor_or_none(node: Any) -> Optional[Any]:
    """Construct a ParallaxScheduledExecutor if every required
    operator-supplied component is present + valid.

    Returns ``None`` and logs a structured warning naming the
    missing piece otherwise. Daemon caller treats None the same
    as "no inference executor wired" (the existing default).

    Component contract (env vars):
      PRSM_PARALLAX_MODEL_CATALOG_FILE — JSON path
      PRSM_PARALLAX_TRUST_STACK_KIND   — "mock"
      PRSM_PARALLAX_GPU_POOL_KIND      — "static-empty"

    Each kind is a tagged-union opt-in. Sprint 558 ships the dev/
    mock kinds only; sprints 560/561 add production kinds.
    """
    catalog_path = os.environ.get(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE", "",
    ).strip()
    trust_kind = os.environ.get(
        "PRSM_PARALLAX_TRUST_STACK_KIND", "",
    ).strip()
    pool_kind = os.environ.get(
        "PRSM_PARALLAX_GPU_POOL_KIND", "",
    ).strip()

    # No-opt-in case: silent return. The operator simply didn't set
    # any of the parallax env vars.
    if not (catalog_path or trust_kind or pool_kind):
        return None

    # ── model catalog ────────────────────────────────────
    if not catalog_path:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_MODEL_CATALOG_FILE not set. Provide a "
            "JSON file mapping model_id → ModelInfo kwargs."
        )
        return None
    cat_p = Path(catalog_path)
    if not cat_p.exists():
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_MODEL_CATALOG_FILE=%s does not exist "
            "on disk.", catalog_path,
        )
        return None
    try:
        raw = json.loads(cat_p.read_text())
    except json.JSONDecodeError as exc:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: model "
            "catalog %s failed JSON parse: %s",
            catalog_path, exc,
        )
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: model "
            "catalog %s top-level must be a dict; got %s",
            catalog_path, type(raw).__name__,
        )
        return None

    # Sprint 559 — schema version + required-field validation. The
    # canonical v1 shape is {schema_version: "v1", models: {...}}.
    # Sprint 558's transitional shape (top-level dict of models)
    # is rejected with a migration hint.
    schema_version = raw.get("schema_version")
    if schema_version is None:
        if "models" in raw:
            logger.warning(
                "Sprint 559 ParallaxScheduledExecutor wiring: "
                "model catalog %s missing required top-level "
                "`schema_version` field. Add `\"schema_version\": "
                "\"v1\"` to the top level.",
                catalog_path,
            )
        else:
            # Legacy sprint-558 shape (top-level dict of models).
            logger.warning(
                "Sprint 559 ParallaxScheduledExecutor wiring: "
                "model catalog %s uses the deprecated sprint-558 "
                "top-level dict format. Migrate to the v1 schema: "
                "{\"schema_version\": \"v1\", \"models\": {...}}.",
                catalog_path,
            )
        return None
    if schema_version not in _CATALOG_SCHEMA_VERSIONS:
        logger.warning(
            "Sprint 559 ParallaxScheduledExecutor wiring: model "
            "catalog %s has schema_version=%r — this daemon only "
            "supports %s. Either upgrade the daemon or downgrade "
            "the catalog format.",
            catalog_path, schema_version, _CATALOG_SCHEMA_VERSIONS,
        )
        return None

    models_section = raw.get("models")
    if not isinstance(models_section, dict):
        logger.warning(
            "Sprint 559 ParallaxScheduledExecutor wiring: model "
            "catalog %s missing/invalid `models` section "
            "(must be a dict mapping model_id → ModelInfo kwargs; "
            "got %s).",
            catalog_path, type(models_section).__name__,
        )
        return None

    # Sprint 559 — per-entry required-field validation. ModelInfo's
    # __init__(**kwargs) silently accepts any input; operator typos
    # in field names would otherwise fail at scheduling time with
    # cryptic AttributeErrors.
    for model_id, kw in models_section.items():
        if not isinstance(kw, dict):
            logger.warning(
                "Sprint 559 ParallaxScheduledExecutor wiring: "
                "model catalog %s entry %r must be a dict of "
                "ModelInfo kwargs; got %s.",
                catalog_path, model_id, type(kw).__name__,
            )
            return None
        missing = [
            f for f in _REQUIRED_MODEL_INFO_FIELDS if f not in kw
        ]
        if missing:
            logger.warning(
                "Sprint 559 ParallaxScheduledExecutor wiring: "
                "model catalog %s entry %r is missing required "
                "ModelInfo field(s): %s",
                catalog_path, model_id, missing,
            )
            return None

    try:
        from prsm.compute.parallax_scheduling.model_info import (
            ModelInfo,
        )
        catalog = {
            mid: ModelInfo(**kw)
            for mid, kw in models_section.items()
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 559 ParallaxScheduledExecutor wiring: model "
            "catalog %s contains an entry that failed ModelInfo "
            "construction: %s", catalog_path, exc,
        )
        return None

    if not catalog:
        logger.warning(
            "Sprint 559 ParallaxScheduledExecutor wiring: model "
            "catalog %s has an empty `models` section — daemon "
            "will advertise no models. Add at least one entry to "
            "serve inference.",
            catalog_path,
        )

    # ── trust stack ──────────────────────────────────────
    if trust_kind not in _KNOWN_TRUST_STACK_KINDS:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_TRUST_STACK_KIND=%r unrecognized. "
            "Known: %s", trust_kind, _KNOWN_TRUST_STACK_KINDS,
        )
        return None
    # ── gpu pool ─────────────────────────────────────────
    # Sprint 690 — built BEFORE trust_stack so the production trust
    # stack can wire PoolBackedStakeLookup against it (F31 proper
    # fix replaces broken AnchorMediatedStakeLookup).
    if pool_kind not in _KNOWN_GPU_POOL_KINDS:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_GPU_POOL_KIND=%r unrecognized. "
            "Known: %s", pool_kind, _KNOWN_GPU_POOL_KINDS,
        )
        return None
    if pool_kind == "static-empty":
        pool_provider = _build_static_empty_pool_provider()
    elif pool_kind == "dht-backed":
        # Sprint 682 — DHT-backed pool reads self + peer-advertised
        # hardware_profile from the discovery layer at each call.
        from prsm.node.dht_backed_pool_provider import (
            build_dht_backed_pool_provider,
        )
        pool_provider = build_dht_backed_pool_provider(node)
    else:  # defensive
        return None

    if trust_kind == "mock":
        trust_stack = _build_mock_trust_stack()
    elif trust_kind == "production":
        trust_stack = _build_production_trust_stack_or_none(
            pool_provider=pool_provider,
        )
        if trust_stack is None:
            return None
    else:  # defensive — already gated above
        return None

    # ── identity ─────────────────────────────────────────
    identity = getattr(node, "identity", None)
    if identity is None:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: node has "
            "no .identity — daemon must initialize identity before "
            "this builder runs."
        )
        return None

    # ── construct ────────────────────────────────────────
    try:
        from prsm.compute.inference.parallax_executor import (
            ParallaxScheduledExecutor,
        )
        executor = ParallaxScheduledExecutor(
            gpu_pool_provider=pool_provider,
            trust_stack=trust_stack,
            model_catalog=catalog,
            chain_executor=_build_chain_executor(node),
            node_identity=identity,
        )
        # Sprint 812 — opt-in output cache via env. Fail-soft:
        # any error here keeps the executor working without
        # cache (pre-811 behavior).
        try:
            from prsm.compute.inference.output_cache import (
                resolve_output_cache_from_env,
            )
            cache = resolve_output_cache_from_env()
            if cache is not None:
                executor._output_cache = cache
                logger.info(
                    "Sprint 812 — output cache enabled "
                    "(max_entries=%d, ttl_seconds=%s)",
                    cache.max_entries, cache.ttl_seconds,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Sprint 812 — output cache wire-up failed; "
                "executor will run without cache: %s", exc,
            )
        return executor
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "ParallaxScheduledExecutor(...) raised at construction: "
            "%s", exc,
        )
        return None
