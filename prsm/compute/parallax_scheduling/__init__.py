# Derived from GradientHQ/parallax (Apache-2.0)
# https://github.com/GradientHQ/parallax — commit c8c8ebdaaf2924b6d25e2d1caff61e27374cce0b
# Original copyright (c) 2025 Gradient and contributors
# Modifications copyright (c) 2026 Prismatica, Inc.
# See licenses/PARALLAX-APACHE-2.0.txt for the upstream license terms.
# See licenses/PARALLAX-NOTICE.txt + git history for the full diff vs. upstream.

"""Phase 3.x.6 Parallax-derived decentralized inference scheduler.

Public exports curate the PRSM-original surface (data layer, router,
profile DHT, four trust adapters, TrustStack composition helper). The
vendored upstream symbols (DynamicProgrammingLayerAllocator, NodeManager,
the upstream Node dataclass, etc.) are intentionally NOT re-exported —
callers should consume them only through the PRSM-original
``allocate_across_regions`` / ``RequestRouter`` / ``ProfileDHT`` shells
so the trust boundary between vendored and PRSM code stays visible.
"""

from prsm.compute.parallax_scheduling.profile_dht import (
    DEFAULT_PUBLISH_INTERVAL_SECONDS,
    DEFAULT_TTL_SECONDS,
    MAX_MESSAGE_BYTES,
    PROFILE_DHT_PROTOCOL_VERSION,
    ProfileDHT,
    ProfileDHTServer,
    ProfileErrorCode,
    ProfileMalformedError,
    ProfileProtocolError,
    ProfileUnknownTypeError,
    ProfileVersionMismatchError,
    SignedProfileEntry,
)
from prsm.compute.parallax_scheduling.prsm_request_router import (
    BudgetExceededError,
    EmptyAllocationError,
    GPUChain,
    InMemoryProfileSource,
    NoCoverageError,
    ProfileSnapshot,
    ProfileSource,
    RegionNotFoundError,
    RequestRouter,
    RouteRequest,
    RoutingError,
)
from prsm.compute.parallax_scheduling.prsm_types import (
    TIER_ATTESTATION_NONE,
    AllocationError,
    AllocationResult,
    EmptyPoolError,
    InsufficientCapacityError,
    ParallaxGPU,
    RegionPipeline,
    allocate_across_regions,
    partition_by_region,
    to_parallax_node,
)
from prsm.compute.parallax_scheduling.trust_adapter import (
    DEFAULT_FULLY_STAKED_THRESHOLD,
    DEFAULT_HARDWARE_TIER_PREFIXES,
    MIN_STAKE_FOR_PARTICIPATION,
    AnchorLookup,
    AnchorVerifyAdapter,
    ChallengeRecord,
    ChallengeSubmitter,
    ConsensusMismatchHook,
    StakeLookup,
    StakeWeightedTrustAdapter,
    TierGateAdapter,
    TierGateRejected,
    TrustStack,
    is_hardware_attestation,
    verified_tier_attestation,
)

__all__ = [
    # Data layer
    "ParallaxGPU",
    "AllocationResult",
    "RegionPipeline",
    "TIER_ATTESTATION_NONE",
    "allocate_across_regions",
    "partition_by_region",
    "to_parallax_node",
    "AllocationError",
    "EmptyPoolError",
    "InsufficientCapacityError",
    # Phase-2 router
    "RequestRouter",
    "RouteRequest",
    "GPUChain",
    "ProfileSource",
    "ProfileSnapshot",
    "InMemoryProfileSource",
    "RoutingError",
    "EmptyAllocationError",
    "NoCoverageError",
    "RegionNotFoundError",
    "BudgetExceededError",
    # Profile DHT
    "ProfileDHT",
    "ProfileDHTServer",
    "SignedProfileEntry",
    "ProfileErrorCode",
    "ProfileProtocolError",
    "ProfileMalformedError",
    "ProfileUnknownTypeError",
    "ProfileVersionMismatchError",
    "PROFILE_DHT_PROTOCOL_VERSION",
    "DEFAULT_TTL_SECONDS",
    "DEFAULT_PUBLISH_INTERVAL_SECONDS",
    "MAX_MESSAGE_BYTES",
    # Trust adapters
    "AnchorVerifyAdapter",
    "AnchorLookup",
    "TierGateAdapter",
    "TierGateRejected",
    "StakeWeightedTrustAdapter",
    "StakeLookup",
    "ConsensusMismatchHook",
    "ChallengeRecord",
    "ChallengeSubmitter",
    "TrustStack",
    "is_hardware_attestation",
    "verified_tier_attestation",
    "DEFAULT_HARDWARE_TIER_PREFIXES",
    "DEFAULT_FULLY_STAKED_THRESHOLD",
    "MIN_STAKE_FOR_PARTICIPATION",
]
