"""Sprint 682 — DHT-backed GpuPoolProvider.

Translates the local hardware_profile (sprint 681) + DHT-propagated
peer hardware_profiles (sprint 680) into the
``Sequence[ParallaxGPU]`` that ``ParallaxScheduledExecutor`` needs.

Replaces sprint 558's static-empty provider when the operator opts
in via ``PRSM_PARALLAX_GPU_POOL_KIND=dht-backed``.

Sprint 683 layers on-chain stake reads on top of the
``stake_amount=0`` placeholder used here. Until then, Adapter C
(stake-weighted trust) effectively treats every peer as unstaked —
which is correct behavior pre-commissioning.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from prsm.compute.parallax_scheduling.trust_adapter import (
    verified_tier_attestation,
)
from prsm.compute.parallax_scheduling.prsm_types import (
    ParallaxGPU, TIER_ATTESTATION_NONE,
)

logger = logging.getLogger(__name__)


# Conservative default for memory bandwidth when the profiler
# doesn't surface a real number. 50 GB/s ≈ typical DDR4/DDR5 dual-
# channel desktop. Better than crashing on the dataclass's positive-
# only validator.
_DEFAULT_MEMORY_BANDWIDTH_GBPS = 50.0

# Layer capacity heuristic: ~2 GB per transformer layer for a 7B
# model in fp16. Operators can override later.
_BYTES_PER_LAYER_GB = 2.0


def _hw_dict_to_parallax_gpu(
    node_id: str, hw: Any, region: str,
    stake_reader: Optional[Any] = None,
    *,
    node_id_authenticated: bool = True,
) -> Optional[ParallaxGPU]:
    """Map a HardwareProfile.to_dict() shape → ParallaxGPU.

    Returns None when essential fields can't be derived (no tflops,
    no memory). Defensive against malicious/buggy peers — never
    raises."""
    if not isinstance(hw, dict):
        return None

    tflops_fp16 = float(hw.get("tflops_fp16", 0.0) or 0.0)
    if tflops_fp16 <= 0.0:
        # Estimate from fp32 (consumer GPUs ≈ 2× fp32 for fp16)
        tflops_fp32 = float(hw.get("tflops_fp32", 0.0) or 0.0)
        if tflops_fp32 > 0.0:
            tflops_fp16 = tflops_fp32 * 2.0
    if tflops_fp16 <= 0.0:
        return None

    gpu_vram_gb = float(hw.get("gpu_vram_gb", 0.0) or 0.0)
    ram_total_gb = float(hw.get("ram_total_gb", 0.0) or 0.0)
    memory_gb = gpu_vram_gb if gpu_vram_gb > 0 else ram_total_gb
    if memory_gb <= 0:
        return None

    # Sprint 843 — per-peer override field takes precedence over
    # consumer-side env. Producer (sp681 hardware_profile_loader)
    # writes operator's PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE into
    # the advertised hw_profile as `tflops_fp16_override`; relay
    # carries it to other consumers via sp838. Coerce defensively
    # — wire data is untrusted.
    _peer_tflops_override = hw.get("tflops_fp16_override")
    if isinstance(_peer_tflops_override, (int, float)) and \
            _peer_tflops_override > 0:
        tflops_fp16 = float(_peer_tflops_override)
    else:
        # Sprint 695 — operator override for the advertised tflops_fp16.
        # The Phase-2 router's optimization uses tflops_fp16 to compute
        # per-layer compute latency; for CPU peers with realistic
        # benchmark numbers (~0.07 tflops on a 1vCPU droplet), the
        # router rejects the resulting chain as too slow. Operators
        # testing multi-stage allocation on slow hardware can override
        # this to a higher value to clear the threshold. Pure-additive.
        # Sprint 843: consumer-side env now acts as fallback for peers
        # that haven't advertised an explicit override field.
        _tflops_override_raw = os.environ.get(
            "PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE", "",
        ).strip()
        if _tflops_override_raw:
            try:
                override = float(_tflops_override_raw)
                if override > 0:
                    tflops_fp16 = override
            except ValueError:
                logger.debug(
                    "PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE=%r not float; "
                    "keeping detected tflops_fp16=%s",
                    _tflops_override_raw, tflops_fp16,
                )

    # Sprint 843 — per-peer memory override takes precedence
    # over consumer-side env.
    _peer_mem_override = hw.get("memory_gb_override")
    if isinstance(_peer_mem_override, (int, float)) and \
            _peer_mem_override > 0:
        memory_gb = float(_peer_mem_override)
    else:
        # Sprint 695 — operator override for the advertised memory_gb.
        # The upstream Parallax allocator uses memory_gb as input to
        # its water-filling step that decides per-stage layer counts;
        # large memory → all layers fit on one GPU → 1-stage allocation
        # (live-attested in sprint 688 for gpt2 on 1.92GB droplets).
        # Operators wanting to force a multi-stage split for testing
        # (or to reserve memory for runtime overhead) can pin this to
        # a smaller value. Pure-additive: env unset → existing
        # heuristic preserved.
        # Sprint 843: consumer-side env now acts as fallback.
        _mem_override_raw = os.environ.get(
            "PRSM_PARALLAX_MEMORY_GB_OVERRIDE", "",
        ).strip()
        if _mem_override_raw:
            try:
                memory_gb = float(_mem_override_raw)
                if memory_gb <= 0:
                    return None
            except ValueError:
                logger.debug(
                    "PRSM_PARALLAX_MEMORY_GB_OVERRIDE=%r not float; "
                    "falling back to advertised memory_gb=%s",
                    _mem_override_raw, memory_gb,
                )

    memory_bandwidth_gbps = float(
        hw.get("memory_bandwidth_gbps", _DEFAULT_MEMORY_BANDWIDTH_GBPS)
        or _DEFAULT_MEMORY_BANDWIDTH_GBPS
    )
    if memory_bandwidth_gbps <= 0:
        memory_bandwidth_gbps = _DEFAULT_MEMORY_BANDWIDTH_GBPS

    # Sprint 843 — per-peer layer_capacity override takes
    # precedence. When producer (operator) sets
    # PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE, sp681's
    # _merge_hardware_overrides writes it as
    # `layer_capacity_override` into the advertised hw_profile;
    # sp838 relays it; we honor it here per-peer. Fallback
    # order: peer field → consumer env → memory heuristic.
    _peer_cap_override = hw.get("layer_capacity_override")
    if isinstance(_peer_cap_override, int) and \
            _peer_cap_override > 0:
        layer_capacity = _peer_cap_override
    else:
        # Sprint 686 — operator override for the layer_capacity
        # heuristic. The default 2GB-per-layer formula targets 7B-class
        # fp16 models; operators serving smaller models (gpt2, phi-2,
        # etc.) need a higher cap or Phase-1 allocation can't place
        # all layers. Set PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE=<int>
        # to pin per-GPU capacity explicitly. Live-attest of sprint
        # 685's gpt2 (12 layers / ~27MB each) surfaced this.
        # Sprint 843: consumer-side env now acts as fallback.
        _override_raw = os.environ.get(
            "PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE", "",
        ).strip()
        if _override_raw:
            try:
                layer_capacity = max(1, int(_override_raw))
            except ValueError:
                logger.debug(
                    "PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE=%r not "
                    "integer; falling back to memory heuristic",
                    _override_raw,
                )
                layer_capacity = max(1, int(memory_gb / _BYTES_PER_LAYER_GB))
        else:
            layer_capacity = max(1, int(memory_gb / _BYTES_PER_LAYER_GB))

    gpu_api = hw.get("gpu_api", "") or ""
    device_map = {
        "cuda": "cuda", "rocm": "cuda", "metal": "mps", "": "cpu",
    }
    device = device_map.get(gpu_api, "cpu")

    # Sprint 683 — optionally resolve stake via on-chain reader
    # Sprint 788 — REQUIRE a valid operator_delegation before
    # trusting the claimed operator_address. Pre-788 trusted the
    # bare claim, allowing peer A to ride peer B's stake by
    # announcing operator_address=B. Missing or invalid
    # delegation → operator_address treated as unset → stake=0.
    stake_amount = 0
    operator_address = hw.get("operator_address", "") or ""
    if operator_address:
        from prsm.node.operator_delegation import (
            verify_operator_delegation_blob,
        )
        delegation = hw.get("operator_delegation")
        if not verify_operator_delegation_blob(
            node_id=node_id,
            operator_address=operator_address,
            delegation=delegation,
        ):
            logger.debug(
                "operator_address=%s on node %s rejected — "
                "missing or invalid operator_delegation (sprint "
                "788). Treating as unstaked.",
                operator_address[:10], node_id[:8],
            )
            operator_address = ""
    if stake_reader is not None and operator_address:
        try:
            stake_amount = int(
                stake_reader.stake_amount_for(operator_address) or 0
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "stake_reader.stake_amount_for(%s) raised: %s — "
                "treating as unstaked", operator_address[:10], exc,
            )
            stake_amount = 0

    # Sprint 1083 — derive the tier-attestation from a CRYPTOGRAPHICALLY VERIFIED
    # attestation blob, never from a self-advertised tier string. Mirrors the sp788
    # operator-delegation pattern: a node self-asserts a capability (here an attestation
    # quote) and the consumer verifies it before trusting it. No blob / unverified /
    # software → TIER_ATTESTATION_NONE (the safe default — ineligible for confidential
    # Tier B/C work via the sp702 TierGateAdapter). Never raises.
    tier_attestation = TIER_ATTESTATION_NONE
    _att_blob = hw.get("attestation")
    if _att_blob and not node_id_authenticated:
        # Sprint 1085 — the discovery transport did NOT authenticate this peer's node_id
        # (the libp2p gossip path trusts a payload-supplied node_id; sp1010 Residual A).
        # The sp1083 report_data↔node_id binding is then meaningless — an attacker picks
        # the node_id — so an attestation must NOT grant a hardware tier here. Fail-closed
        # to tier-none (ineligible for confidential work) until libp2p origin-auth lands.
        logger.debug("ignoring attestation from %s — node_id NOT authenticated by the "
                     "discovery transport (libp2p origin-auth gap); tier-none",
                     node_id[:8])
    elif _att_blob:
        try:
            # node_id binding closes the replay gap: the quote's REPORT_DATA must commit
            # to THIS node, so a genuine quote from another TEE node can't be re-advertised.
            tier_attestation = verified_tier_attestation(_att_blob, node_id=node_id)
        except Exception as exc:  # noqa: BLE001 — fail closed, never raise
            logger.debug("attestation verification raised for node %s: %s — tier-none",
                         node_id[:8], exc)
            tier_attestation = TIER_ATTESTATION_NONE

    try:
        return ParallaxGPU(
            node_id=node_id,
            region=region,
            layer_capacity=layer_capacity,
            stake_amount=stake_amount,
            tier_attestation=tier_attestation,
            tflops_fp16=tflops_fp16,
            memory_gb=memory_gb,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            gpu_name=hw.get("gpu_name", "") or "",
            device=device,
            num_gpus=1,
        )
    except ValueError as exc:
        logger.debug(
            "Skipping peer %s — ParallaxGPU rejected derived "
            "fields: %s", node_id[:16], exc,
        )
        return None


def build_dht_backed_pool_provider(
    node: Any,
    stake_reader: Optional[Any] = None,
) -> Callable[[], Sequence[ParallaxGPU]]:
    """Returns a callable suitable for the
    ``ParallaxScheduledExecutor.gpu_pool_provider`` slot.

    Each invocation reads the *current* state of the node's
    discovery layer + own hardware profile — so peers joining/
    leaving the DHT show up in the next allocation pass without
    restarting the daemon."""
    region = os.environ.get("PRSM_PARALLAX_REGION", "default") or "default"

    # Sprint 683 — lazy-construct an OnChainStakeReader when the
    # caller didn't supply one + PRSM_STAKE_BOND_ADDRESS is set.
    # Tests pass an explicit reader via the kwarg.
    if stake_reader is None and os.environ.get(
        "PRSM_STAKE_BOND_ADDRESS", "",
    ).strip():
        try:
            from prsm.node.onchain_stake_reader import OnChainStakeReader
            stake_reader = OnChainStakeReader()
        except Exception as exc:  # noqa: BLE001
            logger.debug("OnChainStakeReader construction failed: %s", exc)
            stake_reader = None

    def _provider() -> List[ParallaxGPU]:
        gpus: List[ParallaxGPU] = []
        own_id: Optional[str] = None
        try:
            identity = getattr(node, "identity", None)
            if identity is not None:
                own_id = getattr(identity, "node_id", None)
        except Exception:  # noqa: BLE001
            pass

        discovery = getattr(node, "discovery", None)
        if discovery is None:
            return gpus

        own_hw = getattr(discovery, "_local_hardware_profile", None)
        if own_id and own_hw is not None:
            gpu = _hw_dict_to_parallax_gpu(
                own_id, own_hw, region, stake_reader=stake_reader,
            )
            if gpu is not None:
                gpus.append(gpu)

        # Sprint 836 — F31 fix: when admit-unknown-hardware is
        # enabled, fall back to a synthetic conservative
        # hardware profile for peers that haven't advertised
        # one via DISCOVERY_ANNOUNCE. Closes the cold-start
        # gossip gap surfaced during multi-host re-attest:
        # bootstrap-server doesn't propagate hardware_profile,
        # so a fresh-joining operator sees known peers (peer_id +
        # address + capabilities) but no hw → DHT pool silently
        # excludes them → 0 GPUs in pool → can't dispatch. Env-
        # gated opt-in keeps production behavior strict (real
        # hardware required for staking + tier policy); dogfood/
        # multi-host tests set the flag to admit cold-start
        # peers under conservative defaults.
        _admit_unknown_raw = os.environ.get(
            "PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE", "",
        ).strip().lower()
        _admit_unknown = _admit_unknown_raw in ("1", "true", "yes")
        # Sprint 836 — F31 fix part 2: PRSM has two discovery
        # implementations (Libp2pDiscovery for libp2p transport,
        # PeerDiscovery for WebSocket transport). Only the
        # latter exposes a `known_peers` dict attribute; the
        # former only exposes `get_known_peers()` method.
        # Pre-836 the pool provider read `discovery.known_peers`
        # which silently returned {} on Libp2p — so the
        # dht-backed pool always saw 0 peers under libp2p
        # transport (the default). Prefer the API-stable
        # `get_known_peers()` method (both implementations
        # provide it). Fall back to `.known_peers.values()` for
        # callers (legacy test fixtures, mock objects) where
        # the method returns something unusable.
        # Sprint 1085/1246 — the COARSE transport-level authentication default, used
        # ONLY as the fallback when a peer's per-entry marker is None (see below). The
        # WebSocket PeerDiscovery verifies sha256(pubkey)==node_id + a signature on every
        # announce; the libp2p path now ALSO authenticates — discovery announces
        # self-verify (sp1086/1087) and gossip authenticates the origin (sp1246) — but its
        # class default stays False because the only entries reaching here with per-entry
        # None are the unauthenticated connectivity stubs (peer_join / DHT find_providers,
        # explicitly marked False). When a node_id isn't authenticated, its attestation
        # must not grant a hardware tier (the sp1083 binding is void). Default True for
        # back-compat (WS + test fixtures).
        _node_id_authd = bool(getattr(discovery, "node_id_authenticated", True))
        peer_list: List[Any] = []
        getter = getattr(discovery, "get_known_peers", None)
        if callable(getter):
            try:
                got = getter()
                if got is not None:
                    peer_list = list(got)
            except Exception:
                peer_list = []
        if not peer_list:
            attr_peers = getattr(discovery, "known_peers", None)
            if isinstance(attr_peers, dict):
                peer_list = list(attr_peers.values())

        for info in peer_list:
            peer_id = getattr(info, "node_id", None)
            if not peer_id or peer_id == own_id:
                continue
            peer_hw = getattr(info, "hardware_profile", None)
            if peer_hw is None:
                if not _admit_unknown:
                    continue
                # Conservative synthetic profile — matches a
                # generic 1vCPU / 1GB-RAM CPU operator (the
                # weakest realistic peer). Memory math gates
                # downstream allocation; tflops_fp16=0.1 lets
                # Phase-2 routing see SOME path but the DP
                # optimizer will favor better peers when
                # they're present.
                peer_hw = {
                    "tflops_fp16": 0.1,
                    "ram_total_gb": 1.0,
                    "gpu_vram_gb": 0.0,
                    "memory_bandwidth_gbps": 25.0,
                }
            # Sprint 1088 — a per-peer marker (set by an authenticated gossip announce
            # or a verified bootstrap-relay credential) overrides the coarse
            # discovery-level default; None → use the transport default (_node_id_authd).
            _peer_authd = getattr(info, "node_id_authenticated", None)
            _authd = _node_id_authd if _peer_authd is None else bool(_peer_authd)
            gpu = _hw_dict_to_parallax_gpu(
                peer_id, peer_hw, region, stake_reader=stake_reader,
                node_id_authenticated=_authd,
            )
            if gpu is not None:
                gpus.append(gpu)

        # Sprint 695 F44 fix — populate rtt_to_nodes for routing DP.
        # Phase-2 routing's DynamicProgrammingRouting.find_optimal_path
        # uses `node.get_rtt_to(other)` to compute inter-stage transition
        # cost; absent rtt_to_nodes → inf → chain rejected as infeasible.
        # Sprint 562's InMemoryProfileSource is empty in production; until
        # a real RTT-measurement source ships, use a default RTT (env-
        # configurable). Real numbers can be added later by populating
        # hardware_profile.rtt_to_nodes per-peer.
        _default_rtt_raw = os.environ.get(
            "PRSM_PARALLAX_DEFAULT_RTT_MS", "",
        ).strip()
        _default_rtt = 100.0  # conservative inter-region default
        if _default_rtt_raw:
            try:
                _default_rtt = float(_default_rtt_raw)
            except ValueError:
                pass
        if len(gpus) > 1:
            patched: List[ParallaxGPU] = []
            for g in gpus:
                rtt_map = dict(g.rtt_to_nodes) if g.rtt_to_nodes else {}
                for other in gpus:
                    if other.node_id != g.node_id and (
                        other.node_id not in rtt_map
                    ):
                        rtt_map[other.node_id] = _default_rtt
                # ParallaxGPU is frozen — replace via dataclasses.replace
                from dataclasses import replace as _dc_replace
                patched.append(_dc_replace(g, rtt_to_nodes=rtt_map))
            gpus = patched

        return gpus

    return _provider
