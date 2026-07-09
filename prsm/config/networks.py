"""Per-network contract addresses + RPC endpoints.

Single source of truth for which contracts a node should talk to on each
chain. Loaded by `prsm.node.node.PRSMNode`, the bootstrap protocol, and
the CLI's `prsm join-testnet` command.

To add a new network: append an entry to NETWORK_CONFIGS keyed by the
network name. Network names are stable user-facing strings (e.g.,
"mainnet", "testnet"); chain IDs are the EIP-155 numeric identifier.

Mainnet contract addresses are pinned to Base mainnet deployments from
Phase 1.3 Task 8 (2026-05-04). Testnet contract addresses are filled in
post-T1 deploy (see docs/2026-05-05-public-testnet-deploy-plan.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NetworkConfig:
    """Immutable per-network configuration."""

    name: str
    chain_id: int
    rpc_url_default: str
    explorer_url: str

    # Phase 1.3 contracts (deployed)
    ftns_token: Optional[str] = None
    provenance_registry: Optional[str] = None
    # provenance_registry_v2 — added 2026-05-09 in lockstep with the
    # A-08 RoyaltyDistributor v2 mainnet ceremony. The v2 RoyaltyDistributor
    # uses _registry == provenance_registry_v2; operators wiring the
    # provenance pipeline should pin to V2 going forward. V1 retained
    # for legacy Item 7 client compatibility.
    provenance_registry_v2: Optional[str] = None
    royalty_distributor: Optional[str] = None
    foundation_safe: Optional[str] = None

    # Audit-bundle (deployed at L4 audit clear on mainnet; deployed at T1 on testnet)
    escrow_pool: Optional[str] = None
    stake_bond: Optional[str] = None
    settlement_registry: Optional[str] = None
    signature_verifier: Optional[str] = None

    # Phase 8 emission stack
    emission_controller: Optional[str] = None
    compensation_distributor: Optional[str] = None

    # Phase 7-storage
    storage_slashing: Optional[str] = None
    key_distribution: Optional[str] = None
    # sp1364 — Tier B/C paid-decrypt ContentAccessVerifier (the IRoyaltyPaymentVerifier gating
    # the payment-gated key serve). None until deployed per network.
    content_access_verifier: Optional[str] = None

    # Phase 3.x.3 publisher key anchor
    publisher_key_anchor: Optional[str] = None

    # §14 anti-spam creator stake (sp976 CreatorStakeRegistry). None until the
    # commissioning ceremony deploys it + records the address here (the canonical
    # home, like every other contract). The Python CreatorStakeClient reads this.
    creator_stake_registry: Optional[str] = None

    # Network-specific operational notes for users
    notes: tuple[str, ...] = field(default_factory=tuple)

    def is_complete(self) -> bool:
        """True if all expected contracts are deployed (no None placeholders)."""
        required = [
            self.ftns_token,
            self.provenance_registry,
            self.royalty_distributor,
            self.foundation_safe,
            self.escrow_pool,
            self.stake_bond,
            self.settlement_registry,
            self.signature_verifier,
            self.emission_controller,
            self.compensation_distributor,
            self.storage_slashing,
            self.key_distribution,
        ]
        return all(addr is not None for addr in required)


# ────────────────────────────────────────────────────────────────────────
# Mainnet (Base, chainId 8453)
# ────────────────────────────────────────────────────────────────────────
MAINNET = NetworkConfig(
    name="mainnet",
    chain_id=8453,
    rpc_url_default="https://mainnet.base.org",
    explorer_url="https://basescan.org",
    # Phase 1.3 Task 8 deploys (2026-05-04):
    ftns_token="0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
    provenance_registry="0xdF470BFa9eF310B196801D5105468515d0069915",  # V1 (retained for legacy Item 7 ProvenanceRegistry callers)
    provenance_registry_v2="0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf",  # V2 deployed 2026-05-06 per PRSM-CR-2026-05-06-2; wired as _registry on the A-08 v2 RoyaltyDistributor
    royalty_distributor="0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e",  # v2 (A-08 ceremony 2026-05-09); was v1 0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2 (retained for legacy claimable balances per ceremony plan §5.3)
    foundation_safe="0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791",
    # Audit-bundle + Phase 8 + Phase 7-storage deployed 2026-05-07
    # under PRSM-POL-2 §4.1 substituted-audit framework
    # (agent-teams self-audit + Slither static pass + OZ Pausable +
    # TVL caps off-chain). Deployer hot key: 0xF7d8...11c2.
    # Total deploy spend: ~$0.25.
    # sp1300 — F-activation cutover (2026-06-29): the canonical audit-bundle is now the
    # TEE Tier-3 roadmap-F bundle (sp1240 commitBatchWithAttestation), live on Base mainnet
    # + Foundation-owned (2-of-3 acceptOwnership) + Basescan-verified. The PRE-F bundle is
    # RETIRED: drain check passed (0 PENDING batches; 1 historical finalized canary), so no
    # value was migrated. Its EscrowPool holds 1.0 FTNS recoverable residual — depositor
    # recovers via withdraw(); the OLD EscrowPool stays unpaused forever for that.
    #   RETIRED  escrow_pool         0x526D40C08524670846ab811C95691845374122aF
    #   RETIRED  stake_bond          0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed (0 stake)
    #   RETIRED  settlement_registry 0x48fFab641b9D638F312FFA776818756a326F995B
    #   RETIRED  signature_verifier  0xac6a73b270A49Fb62985AbA6bFD6a949577032E5
    escrow_pool="0x4e93a04b3A0C5063FE584980e6c2B1429495EEa1",
    stake_bond="0x21B5de0f65B9273A715C6a02b7085a8ABE8adA72",
    settlement_registry="0x12a01F6C487d765af389bC7D95D90b3136a391F2",
    signature_verifier="0x9d369312bf3b502Bc07c5859a18f7158c22A31e1",  # production Ed25519Verifier (F bundle)
    emission_controller="0x13A0D76895c196B795b94fe843F76B6e145AeaAE",
    compensation_distributor="0xa9551F5a3AeAB39cc8315AcD8caC2886Bd04f244",
    storage_slashing="0x0e9cAfadCCCe0987C773B5FdFF295c2Aa6F03337",
    key_distribution="0x51AF73Aa098E3b12Da78167c25c3d1D98059c8Ff",
    # sp1364 — Tier B/C paid-decrypt verifier, Base mainnet (deployed + Basescan-verified 2026-07-02;
    # wired to FTNS 0x5276...16e5 + ProvenanceRegistry V2 0xe0ce...0dbf).
    content_access_verifier="0xF32A9049EeB1Fa3eE9f76A085a6b8662d5c9aE59",
    publisher_key_anchor="0xd811ad9986f44f404b0fd992168a7cc76206df03",  # Phase 3.x.3 deployed 2026-05-20 via Foundation Safe + EIP-2470 CREATE2 factory (sprints 619-621); on-chain tx 0x55087635ae9543ae2a844e112d76c04f93f19c72d0e9f7c736eafed7fb3510b0 block 46248175; admin() == Foundation Safe verified
    creator_stake_registry=None,  # sp976 CreatorStakeRegistry — NOT yet deployed. RECORD the address HERE after the gated Base mainnet ceremony (transferOwnership→Foundation Safe→2-of-3 acceptOwnership); the CreatorStakeClient then reads it. See docs/2026-06-03-creator-stake-registry-sepolia-runbook.md.
    notes=(
        "Mainnet uses the real 2-of-3 Foundation Safe (Ledger + Trezor + OneKey) at 0x91b0...5791.",
        "Audit-bundle ownership-transfer ceremony complete 2026-05-07: all 7 Ownable2Step "
        "contracts (EscrowPool, BSR, StakeBond, EmissionController, CompensationDistributor, "
        "StorageSlashing, KeyDistribution) now owned by Foundation Safe via 2-of-3 hardware "
        "multisig acceptOwnership batch. Deployer hot key 0xF7d8…11c2 retains zero authority "
        "on these contracts.",
        "Foundation reserve wallet on StakeBond + all 3 CompensationDistributor pool sinks "
        "+ StorageSlashing.authorizedVerifier all point at the Foundation Safe — passes MED-4 "
        "code.length check natively (Safes are smart contracts).",
        "Provenance + RoyaltyDistributor (Phase 1.3) live since 2026-05-04; FTNS token live since "
        "Phase 1.3 Task 8 with 100M minted to Foundation Safe via PRSM-CR-2026-05-06-3.",
        "TVL caps for v1 are off-chain operational discipline + Forta alerts on Deposit/Bond "
        "event sums (POL-2 §4.3 default $10K each on EscrowPool + StakeBond). On-chain caps "
        "are a v2 enhancement.",
    ),
)

# ────────────────────────────────────────────────────────────────────────
# Testnet (Base Sepolia, chainId 84532)
# Addresses filled in post-T1 deploy.
# ────────────────────────────────────────────────────────────────────────
TESTNET = NetworkConfig(
    name="testnet",
    chain_id=84532,
    rpc_url_default="https://sepolia.base.org",
    explorer_url="https://sepolia.basescan.org",
    # T1 deploy 2026-05-07 (post-HIGH-1/2 + MED-cluster + A-06 fixes; deployer
    # `PRSM Testnet Deployer` = 0xCCAc7b21695De068979b1ca47B0cfBD328654220):
    ftns_token="0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a",
    # Provenance + Royalty are Phase 1.3 surfaces deployed independently;
    # not in scope of this T1 rehearsal. Testnet versions can be deployed
    # later if/when content-registration flow needs validating on testnet.
    provenance_registry=None,
    # sp1413 — V2 IS deployed on Sepolia and the Tier B/C ContentAccessVerifier
    # (0x99264Bca..) reads getCreatorAndRate from it. Leaving this None made the
    # node skip the V2 client entirely on testnet (node.py: `v2_addr = ep.
    # provenance_registry_v2 or ""`), so a paid-content publisher registered
    # nowhere the verifier looks and payForAccess reverted ContentNotRegistered.
    # Pinned to the CAV's registry so testnet mirrors mainnet's single-registry
    # shape. (Superseded: 0xe75F0c24.., 3 placeholder registrations, dormant
    # since 2026-05-06 — see contract_addresses.json base-sepolia.)
    provenance_registry_v2="0xCBe377Ae09fdD5F63875Aa5313C65A3C8C073731",
    royalty_distributor=None,
    foundation_safe="0xCCAc7b21695De068979b1ca47B0cfBD328654220",  # deployer EOA per §9 ratified decision
    escrow_pool="0xaa28b5818242608e04C1773c3e34bF7bFfb96248",
    stake_bond="0xF93aCa6551F408fFfe24292288d5488864D5264c",
    settlement_registry="0xF8BEEb4362222b50109b6034767322B31aA92449",
    signature_verifier="0x208dc98545Fe062d0B13Ac07b073633E6a62b5A9",  # production Ed25519Verifier
    # T10 (2026-05-07) re-deploy: 1-hour epoch (vs mainnet 4 years) so
    # halving curve is observable at testnet timescales. Old pair
    # (EmissionController 0x30b6810F…, CompensationDistributor 0x18c87574…)
    # is orphaned but retained on-chain — superseded by these.
    emission_controller="0x1478F8f5F13a5BDeBc2a0b7C185D19BEE15f312e",
    compensation_distributor="0xFd730f8E513eD184F255cb1a62791e711B2e81b9",
    storage_slashing="0x2ba1B361d2AD49f15F1131762fA3512d7824EB06",
    key_distribution="0xdB41A471AAC86285cD855bEdC27D7FC810dc3318",
    # sp1364 — Tier B/C paid-decrypt verifier (deployed 2026-07-02, verified on Basescan). Wired to
    # a fresh ProvenanceRegistryV2 0xCBe377Ae09fdD5F63875Aa5313C65A3C8C073731 (NOT the pre-existing
    # 0xe75F...), so a paid-content publisher registers in 0xCBe377... for the fee-payee lookup.
    content_access_verifier="0x99264Bca75d63DB9b8B5C7C1e2ECBf78d133905a",
    publisher_key_anchor=None,  # not yet deployed on Base Sepolia; Phase 3.x.3 Sepolia deploy was on Ethereum Sepolia
    creator_stake_registry=None,  # sp976 CreatorStakeRegistry — record the Base Sepolia rehearsal address here if a testnet gate is wanted (the rehearsal script is self-contained and does not require this).
    notes=(
        "TESTNET — testnet-FTNS has zero monetary value.",
        "Foundation 'Safe' on testnet is a single deployer EOA, NOT the real 2-of-3 mainnet multisig.",
        "Foundation reserve wallet on StakeBond is set to the FTNS token address itself "
        "(passes MED-4 code.length>0 gate; foundation-share slashes accumulate passively at "
        "the FTNS contract — no economic-recovery path on testnet, but slashing flow IS exercisable).",
        "Halving curve uses 1-hour epoch (vs mainnet 4 years) — T10 redeploy 2026-05-07. "
        "EmissionController.EPOCH_DURATION_SECONDS is now constructor-set (immutable); "
        "mainnet (chainId 8453) constructor enforces exactly 4 years.",
        "Provenance + Royalty contracts are NOT deployed on testnet — content-registration "
        "flow remains mainnet-only for now.",
        "Faucet: ask in #testnet-faucet on Discord; founder airdrops within 24h.",
    ),
)

# ────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────
NETWORK_CONFIGS: dict[str, NetworkConfig] = {
    "mainnet": MAINNET,
    "testnet": TESTNET,
}

DEFAULT_NETWORK = "mainnet"


def get_network_config(name: Optional[str] = None) -> NetworkConfig:
    """Look up a network by name. Falls back to DEFAULT_NETWORK if name is None.

    Raises KeyError if the name doesn't match any known network.
    """
    key = (name or DEFAULT_NETWORK).lower()
    if key not in NETWORK_CONFIGS:
        known = ", ".join(sorted(NETWORK_CONFIGS.keys()))
        raise KeyError(
            f"unknown network {name!r}; known networks: {known}"
        )
    return NETWORK_CONFIGS[key]


# ────────────────────────────────────────────────────────────────────────
# Env-var resolution for runtime consumers
# ────────────────────────────────────────────────────────────────────────
#
# Several runtime modules — `prsm.economy.ftns_onchain.OnChainFTNSLedger`,
# `prsm.node.content_economy._build_provenance_client_or_none`,
# `prsm.node.node._build_provenance_client_or_none` — historically read
# their RPC URL + contract addresses from independent env vars and
# defaulted to mainnet. That meant `prsm join-testnet`'s env-var bundle
# (which uses `BASE_SEPOLIA_RPC_URL` + `FTNS_TOKEN_ADDRESS` +
# `PRSM_NETWORK=testnet`) didn't actually flow through; on-chain calls
# would silently target Base mainnet despite the testnet onboarding.
#
# `resolve_endpoints()` centralises the resolution so all three consumers
# pick up the same network selection. Resolution order per field:
#   1. Explicit `network` kwarg (if passed by the caller).
#   2. `PRSM_NETWORK` env var (testnet | mainnet).
#   3. Default network (mainnet).
# Per-field overrides (`FTNS_TOKEN_ADDRESS`, `PRSM_BASE_RPC_URL`,
# `BASE_RPC_URL`, `BASE_SEPOLIA_RPC_URL`,
# `PRSM_PROVENANCE_REGISTRY_ADDRESS`, …) take precedence over the
# resolved network defaults — operators can still pin a specific address
# when they want to.
import os as _os  # local alias to avoid leaking `os` into the public symbol surface


@dataclass(frozen=True)
class ResolvedEndpoints:
    """Materialised view of the network config + env-var overrides.

    Caller-visible field surface stays stable: each field is the value
    that on-chain clients should actually use. None means "no
    deployment available for this field on the resolved network", which
    callers are expected to handle (typically by skipping on-chain
    routing for that surface).
    """

    network_name: str
    chain_id: int
    rpc_url: str
    explorer_url: str
    ftns_token: Optional[str]
    provenance_registry: Optional[str]
    provenance_registry_v2: Optional[str]
    royalty_distributor: Optional[str]
    foundation_safe: Optional[str]
    publisher_key_anchor: Optional[str]
    creator_stake_registry: Optional[str]
    settlement_registry: Optional[str]
    escrow_pool: Optional[str]
    stake_bond: Optional[str]
    emission_controller: Optional[str]
    compensation_distributor: Optional[str]
    storage_slashing: Optional[str]
    key_distribution: Optional[str]
    content_access_verifier: Optional[str]


def _resolve_network_name(network: Optional[str] = None) -> str:
    """Pick the network name from explicit arg → PRSM_NETWORK → default.

    Empty string from `os.getenv` is treated as unset.
    """
    if network:
        return network.lower()
    env = (_os.getenv("PRSM_NETWORK") or "").strip().lower()
    if env:
        return env
    return DEFAULT_NETWORK


def _resolve_rpc_url(cfg: NetworkConfig) -> str:
    """Pick the RPC URL respecting per-network env-var aliases.

    Resolution order:
      mainnet: BASE_RPC_URL → PRSM_BASE_RPC_URL → cfg.rpc_url_default
      testnet: BASE_SEPOLIA_RPC_URL → PRSM_BASE_RPC_URL → cfg.rpc_url_default
    `PRSM_BASE_RPC_URL` is a network-agnostic operator override that wins
    over the per-network env var so a single env file can target either.
    """
    explicit = (_os.getenv("PRSM_BASE_RPC_URL") or "").strip()
    if explicit:
        return explicit
    if cfg.chain_id == 84532:  # Base Sepolia
        sepolia = (_os.getenv("BASE_SEPOLIA_RPC_URL") or "").strip()
        if sepolia:
            return sepolia
    else:  # Base mainnet (and any future networks default to BASE_RPC_URL)
        mainnet = (_os.getenv("BASE_RPC_URL") or "").strip()
        if mainnet:
            return mainnet
    return cfg.rpc_url_default


def resolve_endpoints(network: Optional[str] = None) -> ResolvedEndpoints:
    """Resolve the runtime on-chain endpoint set.

    Reads PRSM_NETWORK + per-field env-var overrides. Returns a
    materialised dataclass that on-chain clients can plumb directly.
    """
    name = _resolve_network_name(network)
    cfg = get_network_config(name)

    def _override(env_name: str, fallback: Optional[str]) -> Optional[str]:
        v = (_os.getenv(env_name) or "").strip()
        return v if v else fallback

    return ResolvedEndpoints(
        network_name=name,
        chain_id=cfg.chain_id,
        rpc_url=_resolve_rpc_url(cfg),
        explorer_url=cfg.explorer_url,
        # Phase 1.3 contracts — operators can pin individual addresses.
        # `FTNS_CONTRACT_ADDRESS` is the legacy alias kept for backwards
        # compatibility with existing `OnChainFTNSLedger` operators.
        ftns_token=_override(
            "FTNS_TOKEN_ADDRESS",
            _override("FTNS_CONTRACT_ADDRESS", cfg.ftns_token),
        ),
        provenance_registry=_override(
            "PRSM_PROVENANCE_REGISTRY_ADDRESS", cfg.provenance_registry
        ),
        # Sprint 525 — V2 canonical (post-A-08 ceremony). Operators
        # should pin to V2 going forward; V1 retained as legacy fallback.
        provenance_registry_v2=_override(
            "PRSM_PROVENANCE_REGISTRY_V2_ADDRESS",
            cfg.provenance_registry_v2,
        ),
        royalty_distributor=_override(
            "PRSM_ROYALTY_DISTRIBUTOR_ADDRESS", cfg.royalty_distributor
        ),
        foundation_safe=_override("PRSM_FOUNDATION_SAFE", cfg.foundation_safe),
        publisher_key_anchor=_override(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", cfg.publisher_key_anchor
        ),
        # sp981 — §14 creator-stake registry. Primary override env is the
        # non-prefixed CREATOR_STAKE_REGISTRY_ADDRESS (the name from_env + the
        # deploy script already use); the PRSM_-prefixed alias is accepted for
        # consistency with the other contracts.
        creator_stake_registry=_override(
            "CREATOR_STAKE_REGISTRY_ADDRESS",
            _override(
                "PRSM_CREATOR_STAKE_REGISTRY_ADDRESS", cfg.creator_stake_registry
            ),
        ),
        # Audit-bundle + Phase 8 + Phase 7-storage — pinning individual
        # addresses is rarer in practice but we still expose the override
        # surface for completeness.
        settlement_registry=_override(
            "PRSM_SETTLEMENT_REGISTRY_ADDRESS", cfg.settlement_registry
        ),
        escrow_pool=_override("PRSM_ESCROW_POOL_ADDRESS", cfg.escrow_pool),
        stake_bond=_override("PRSM_STAKE_BOND_ADDRESS", cfg.stake_bond),
        emission_controller=_override(
            "PRSM_EMISSION_CONTROLLER_ADDRESS", cfg.emission_controller
        ),
        compensation_distributor=_override(
            "PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS", cfg.compensation_distributor
        ),
        storage_slashing=_override(
            "PRSM_STORAGE_SLASHING_ADDRESS", cfg.storage_slashing
        ),
        key_distribution=_override(
            "PRSM_KEY_DISTRIBUTION_ADDRESS", cfg.key_distribution
        ),
        content_access_verifier=_override(
            "PRSM_CONTENT_ACCESS_VERIFIER", cfg.content_access_verifier
        ),
    )
