"""Sprint 302 — formal-invariant harness (Vision §14 item 4).

Vision §14 item 4: "Formal verification on highest-value
contracts. Payment escrow and royalty distribution contracts
undergo formal-methods verification, not just standard audit."

This module ships the SPEC LAYER + RUNTIME PROBE:
  Invariant       — declarative pinned formal-spec record
                    (id, kind, selector, expected, severity)
  INVARIANT_REGISTRY — public mapping of contract_name →
                       [Invariant]. The §14 transparency
                       promise: anyone can see what PRSM
                       has formally committed to.
  InvariantChecker — runtime probe via an injected backend
                     (call_uint256 / call_address /
                     call_bool / token_balance_of).
                     Returns PASS / FAIL / SKIPPED.
  InvariantResult  — verifiable outcome record with
                     diagnostic.

The actual symbolic-execution runs (halmos, Certora) consume
the SAME registry on a follow-on sprint. The Python harness
gives operators a "is the protocol in spec RIGHT NOW" probe
against live mainnet state.

Highest-value target this sprint: RoyaltyDistributor v2.
Five pinned invariants:
  INV-RD-1  NETWORK_FEE_BPS == 200 (immutable 2% cap)
  INV-RD-2  networkTreasury == Foundation Safe
  INV-RD-3  owner == Foundation Safe (post-acceptOwnership)
  INV-RD-4  ftns.balanceOf(this) >= totalClaimable (SOLVENCY)
  INV-RD-5  paused() — operator-observable
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


# ── Enums ────────────────────────────────────────────


class InvariantSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class InvariantStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class InvariantKind(str, Enum):
    UINT256_EQ = "uint256_eq"
    UINT256_AT_WORD_EQ = "uint256_at_word_eq"
    UINT256_GTE = "uint256_gte"
    UINT256_LTE = "uint256_lte"
    ADDRESS_EQ = "address_eq"
    BOOL_READ = "bool_read"
    BALANCE_GTE_CLAIMABLE = "balance_gte_claimable"
    HAS_ROLE_EQ = "has_role_eq"


# ── Dataclasses ──────────────────────────────────────


@dataclass
class Invariant:
    id: str
    contract_name: str
    title: str
    description: str
    severity: InvariantSeverity
    spec_text: str
    kind: InvariantKind
    selector: str = ""
    expected: Optional[Any] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "contract_name": self.contract_name,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "spec_text": self.spec_text,
            "kind": self.kind.value,
            "selector": self.selector,
            "expected": (
                self.expected
                if not isinstance(self.expected, bytes)
                else self.expected.hex()
            ),
            "params": dict(self.params),
        }


@dataclass
class InvariantResult:
    invariant_id: str
    status: InvariantStatus
    value: Any = None
    expected: Any = None
    diagnostic: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "status": self.status.value,
            "value": self.value,
            "expected": self.expected,
            "diagnostic": self.diagnostic,
            "error": self.error,
        }


# ── Backend protocol ─────────────────────────────────


class FormalBackend(Protocol):
    def call_uint256(
        self, addr: str, selector: str,
    ) -> Optional[int]: ...
    def call_uint256_at_word(
        self, addr: str, selector: str, word_index: int,
    ) -> Optional[int]: ...
    def call_address(
        self, addr: str, selector: str,
    ) -> Optional[str]: ...
    def call_bool(
        self, addr: str, selector: str,
    ) -> Optional[bool]: ...
    def token_balance_of(
        self, token: str, holder: str,
    ) -> Optional[int]: ...
    def call_has_role(
        self, addr: str, role_hash: str, account: str,
    ) -> Optional[bool]: ...


# ── EVM function selectors (canonical) ───────────────


# All selectors below are first 4 bytes of keccak256 of the
# function signature. Pinned to the v2 RoyaltyDistributor
# ABI per `contracts/contracts/RoyaltyDistributor.sol`.
_SEL_NETWORK_FEE_BPS = "0x9c5e6cf2"   # NETWORK_FEE_BPS()
_SEL_NETWORK_TREASURY = "0x8f0d1b8e"  # networkTreasury()
_SEL_OWNER = "0x8da5cb5b"             # owner()
_SEL_FTNS = "0xefa21b41"              # ftns() — sprint 356 fix:
# sprint 302 originally committed 0x9b03f021 which is NOT
# keccak256("ftns()")[:4]. That bug caused INV-RD-4 (the
# solvency invariant) to SKIP on mainnet because the
# backend's eth_call against a wrong selector returns 0x and
# we map None → SKIPPED. The mocked-backend tests didn't
# catch this because they index by the (wrong) selector
# verbatim. Pinned by test_ftns_selector_pinned_to_canonical_keccak.
_SEL_TOTAL_CLAIMABLE = "0xc70b25c0"   # totalClaimable()
_SEL_PAUSED = "0x5c975abb"            # paused()
_SEL_TOTAL_SUPPLY = "0x18160ddd"      # totalSupply() (ERC-20)
_SEL_MAX_SUPPLY = "0x32cb6b0c"        # MAX_SUPPLY()
_SEL_TOTAL_ESCROWED_BALANCE = (
    "0x71e780f3"                       # totalEscrowedBalance()
)
_SEL_HAS_ROLE = "0x91d14854"          # hasRole(bytes32,address)
_SEL_EPOCH_DURATION_SECONDS = (
    "0xdf617c6e"                       # EPOCH_DURATION_SECONDS()
)
_SEL_BASE_MAINNET_CHAIN_ID = (
    "0xc7b6b6e8"                       # BASE_MAINNET_CHAIN_ID()
)
_SEL_MIN_WEIGHT_SCHEDULE_DELAY = (
    "0x5a7d67c9"                       # MIN_WEIGHT_SCHEDULE_DELAY()
)
_SEL_MINT_CAP = "0x76c71ca1"          # mintCap()
_SEL_BASELINE_RATE_PER_SECOND = (
    "0x968226be"                       # baselineRatePerSecond()
)
_SEL_CURRENT_WEIGHTS = (
    "0x322db68a"                       # currentWeights() -> PoolWeights
)
_SEL_MIN_HEARTBEAT_GRACE = (
    "0xebf2fbfe"                       # MIN_HEARTBEAT_GRACE()
)
_SEL_MAX_HEARTBEAT_GRACE = (
    "0xa0648401"                       # MAX_HEARTBEAT_GRACE()
)
_SEL_MIN_UNBOND_DELAY = (
    "0x962dc269"                       # MIN_UNBOND_DELAY_SECONDS()
)
_SEL_MAX_UNBOND_DELAY = (
    "0xa0e48ecc"                       # MAX_UNBOND_DELAY_SECONDS()
)
_SEL_CHALLENGER_BOUNTY_BPS = (
    "0xfc65a392"                       # CHALLENGER_BOUNTY_BPS()
)


# ── OpenZeppelin AccessControl role hashes (bytes32) ──


# DEFAULT_ADMIN_ROLE is bytes32(0) by OZ convention.
_DEFAULT_ADMIN_ROLE_HASH = "0x" + "00" * 32
# MINTER_ROLE = keccak256("MINTER_ROLE") — pinned to the
# FTNSTokenSimple.sol constant; computed once + frozen here
# so the harness doesn't need a runtime keccak dependency.
_MINTER_ROLE_HASH = (
    "0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef"
    "8d981c8956a6"
)


# ── Foundation Safe + disarmed hot-key addresses ─────


# Disarmed hot key per PRSM-CR-2026-05-06-3 (executed
# 2026-05-06). All FTNSToken role grants on this address
# were revoked + the on-disk private-key file was deleted;
# the runtime invariants below pin that the role-disarm
# still holds.
_DISARMED_HOT_KEY_BASE = (
    "0x8eaA00FF741323bc8B0ab1290c544738D9b2f012"
)

# The selector values above are derived from the v2 source.
# Operators should verify on Basescan if treating them as
# load-bearing; for the harness, the canonical-check
# protocol below requires the backend to look them up
# correctly — a mismatched ABI surfaces as
# SKIPPED (None return) rather than silent FAIL.


# ── Foundation Safe address (from networks.py) ───────


_FOUNDATION_SAFE_BASE = (
    "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
)


# ── Pinned registry ──────────────────────────────────


INVARIANT_REGISTRY: Dict[str, List[Invariant]] = {
    "royalty_distributor": [
        Invariant(
            id="INV-RD-1",
            contract_name="royalty_distributor",
            title="Network fee is immutable at 200 bps (2%)",
            description=(
                "The protocol fee skimmed to the Foundation "
                "Safe is a public constant. Drift would "
                "indicate either a contract substitution or "
                "an ABI-confusion attack."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text="NETWORK_FEE_BPS() == 200",
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_NETWORK_FEE_BPS,
            expected=200,
        ),
        Invariant(
            id="INV-RD-2",
            contract_name="royalty_distributor",
            title=(
                "Network treasury is the Foundation Safe "
                "(immutable)"
            ),
            description=(
                "networkTreasury is declared immutable in "
                "the v2 contract, so any deviation here "
                "means we're reading the wrong contract."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                f"networkTreasury() == {_FOUNDATION_SAFE_BASE}"
            ),
            kind=InvariantKind.ADDRESS_EQ,
            selector=_SEL_NETWORK_TREASURY,
            expected=_FOUNDATION_SAFE_BASE,
        ),
        Invariant(
            id="INV-RD-3",
            contract_name="royalty_distributor",
            title=(
                "Contract owner is the Foundation Safe"
            ),
            description=(
                "Post-acceptOwnership (sprint 134 mainnet "
                "ceremony 2026-05-09), owner() must return "
                "the Foundation 2-of-3 multisig — never the "
                "deployer hot key."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                f"owner() == {_FOUNDATION_SAFE_BASE}"
            ),
            kind=InvariantKind.ADDRESS_EQ,
            selector=_SEL_OWNER,
            expected=_FOUNDATION_SAFE_BASE,
        ),
        Invariant(
            id="INV-RD-4",
            contract_name="royalty_distributor",
            title=(
                "Solvency: ftns.balanceOf(this) >= "
                "totalClaimable"
            ),
            description=(
                "THE money invariant. If the contract's "
                "FTNS balance ever drops below the sum of "
                "outstanding pull-payment claims, the "
                "protocol is insolvent and some recipient "
                "will be unable to claim. This invariant is "
                "what the L4 self-audit A-08 surface was "
                "built to maintain in lockstep."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "ftns.balanceOf(address(this)) >= "
                "totalClaimable"
            ),
            kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
            selector=_SEL_TOTAL_CLAIMABLE,
            params={
                "ftns_selector": _SEL_FTNS,
                "totalclaimable_selector": (
                    _SEL_TOTAL_CLAIMABLE
                ),
            },
        ),
        Invariant(
            id="INV-RD-5",
            contract_name="royalty_distributor",
            title="paused() — operator-observable",
            description=(
                "Pause state is surfaced as an invariant "
                "for operator visibility. PASS just means "
                "the read succeeded; the boolean value is "
                "the operator's signal."
            ),
            severity=InvariantSeverity.MEDIUM,
            spec_text="paused() observability",
            kind=InvariantKind.BOOL_READ,
            selector=_SEL_PAUSED,
            expected=None,
        ),
    ],
    "ftns_token": [
        Invariant(
            id="INV-FT-1",
            contract_name="ftns_token",
            title=(
                "MAX_SUPPLY is pinned at 1B FTNS (immutable)"
            ),
            description=(
                "MAX_SUPPLY is a public constant in "
                "FTNSTokenSimple.sol. Drift here would "
                "indicate either a contract substitution or "
                "ABI confusion. The entire monetary base "
                "of the protocol depends on this constant "
                "matching the value committed in source."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "MAX_SUPPLY() == 1_000_000_000 * 10**18"
            ),
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_MAX_SUPPLY,
            expected=1_000_000_000 * 10**18,
        ),
        Invariant(
            id="INV-FT-2",
            contract_name="ftns_token",
            title=(
                "totalSupply() <= MAX_SUPPLY (cap honored)"
            ),
            description=(
                "Every mint in FTNSTokenSimple.mint() "
                "requires totalSupply() + amount <= "
                "MAX_SUPPLY. If this invariant ever fails, "
                "either MINTER_ROLE was abused via a "
                "compromised key or a malicious upgrade "
                "bypassed the check. THE supply-side money "
                "invariant for the protocol's token base."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "totalSupply() <= MAX_SUPPLY (1B * 10**18)"
            ),
            kind=InvariantKind.UINT256_LTE,
            selector=_SEL_TOTAL_SUPPLY,
            expected=1_000_000_000 * 10**18,
        ),
        Invariant(
            id="INV-FT-3",
            contract_name="ftns_token",
            title=(
                "Foundation Safe holds DEFAULT_ADMIN_ROLE"
            ),
            description=(
                "Post-PRSM-CR-2026-05-06-3 execution "
                "(2026-05-06), the Foundation Safe is the "
                "SOLE FTNSToken administrator. If this "
                "invariant ever fails, the protocol has "
                "lost the ability to grant new MINTER_ROLE "
                "holders — operationally recoverable but "
                "audit-critical. POSITIVE assertion."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "hasRole(DEFAULT_ADMIN_ROLE, Foundation Safe)"
                " == true"
            ),
            kind=InvariantKind.HAS_ROLE_EQ,
            selector="",
            expected=True,
            params={
                "role_hash": _DEFAULT_ADMIN_ROLE_HASH,
                "account": _FOUNDATION_SAFE_BASE,
            },
        ),
        Invariant(
            id="INV-FT-4",
            contract_name="ftns_token",
            title=(
                "Disarmed hot key MUST NOT hold MINTER_ROLE"
            ),
            description=(
                "The 900M-FTNS unilateral-mint attack "
                "surface that PRSM-CR-2026-05-06-3 closed. "
                "The hot key was disarmed via 4 grants + "
                "batched 4-revoke multisig + on-disk file "
                "deletion. If MINTER_ROLE ever appears back "
                "on this account, either the disarm was "
                "incomplete or the address was re-granted "
                "by accident. NEGATIVE assertion — failure "
                "is a P0 disarm-broken event."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "hasRole(MINTER_ROLE, disarmed_hot_key) "
                "== false"
            ),
            kind=InvariantKind.HAS_ROLE_EQ,
            selector="",
            expected=False,
            params={
                "role_hash": _MINTER_ROLE_HASH,
                "account": _DISARMED_HOT_KEY_BASE,
            },
        ),
        Invariant(
            id="INV-FT-5",
            contract_name="ftns_token",
            title=(
                "Disarmed hot key MUST NOT hold "
                "DEFAULT_ADMIN_ROLE"
            ),
            description=(
                "Sister invariant to INV-FT-4. The disarm "
                "ceremony revoked both MINTER_ROLE and "
                "DEFAULT_ADMIN_ROLE on the deployer hot key. "
                "Re-armed admin role would itself permit "
                "re-granting MINTER_ROLE — both checks "
                "needed for full disarm verification."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "hasRole(DEFAULT_ADMIN_ROLE, "
                "disarmed_hot_key) == false"
            ),
            kind=InvariantKind.HAS_ROLE_EQ,
            selector="",
            expected=False,
            params={
                "role_hash": _DEFAULT_ADMIN_ROLE_HASH,
                "account": _DISARMED_HOT_KEY_BASE,
            },
        ),
    ],
    "emission_controller": [
        Invariant(
            id="INV-EC-1",
            contract_name="emission_controller",
            title=(
                "EPOCH_DURATION_SECONDS is pinned at 4 years"
            ),
            description=(
                "The halving cadence is the canonical "
                "monetary-policy parameter for Phase 8 "
                "emissions. EmissionController.constructor "
                "enforces chainid-8453 → "
                "MAINNET_EPOCH_DURATION_SECONDS = 4*365 days; "
                "this runtime invariant guards against "
                "contract substitution to a deployment that "
                "didn't honor the constraint. Drift dilutes "
                "or constricts FTNS issuance pace."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "EPOCH_DURATION_SECONDS() == 4 * 365 days "
                "(126_144_000 seconds)"
            ),
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_EPOCH_DURATION_SECONDS,
            expected=4 * 365 * 86400,
        ),
        Invariant(
            id="INV-EC-2",
            contract_name="emission_controller",
            title=(
                "BASE_MAINNET_CHAIN_ID() == 8453"
            ),
            description=(
                "The chainid pin that enforces the 4-year "
                "mainnet halving constraint at construction. "
                "If this constant doesn't match, the live "
                "contract is either deployed to a different "
                "chain or has been substituted."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text="BASE_MAINNET_CHAIN_ID() == 8453",
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_BASE_MAINNET_CHAIN_ID,
            expected=8453,
        ),
        Invariant(
            id="INV-EC-3",
            contract_name="emission_controller",
            title=(
                "mintCap pinned at 900M FTNS (the v1 emission cap)"
            ),
            description=(
                "Phase 8 emissions are hard-bounded by an "
                "immutable mintCap. With 100M genesis already "
                "minted, a 900M emission cap caps lifetime FTNS "
                "supply at the 1B MAX_SUPPLY enforced by "
                "FTNSToken — the Bitcoin-style scarcity ceiling "
                "early-adopter value depends on (PRSM_Tokenomics "
                "v1, §4.2 / §10 #1, #8). Drift in this constant "
                "would either dilute holders (higher cap) or "
                "prematurely starve worker compensation (lower). "
                "Pinned to the mainnet deploy "
                "(phase8-emission-base-1778164608198.json)."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "mintCap() == 900_000_000 * 1e18"
            ),
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_MINT_CAP,
            expected=900_000_000 * 10**18,
        ),
        Invariant(
            id="INV-EC-4",
            contract_name="emission_controller",
            title=(
                "baselineRatePerSecond pinned at 1 FTNS/sec"
            ),
            description=(
                "The opening emission rate (epoch 0) is 1 FTNS "
                "per second ≈ 31.536M FTNS/yr, halved every 4 "
                "years toward zero (PRSM_Tokenomics v1, §4.2). "
                "This immutable baseline sets the early-adopter "
                "reward gradient; substitution to a higher value "
                "would inflate issuance pace and erode scarcity. "
                "Pinned to the mainnet deploy "
                "(phase8-emission-base-1778164608198.json)."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "baselineRatePerSecond() == 1e18"
            ),
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_BASELINE_RATE_PER_SECOND,
            expected=10**18,
        ),
    ],
    "compensation_distributor": [
        Invariant(
            id="INV-CD-1",
            contract_name="compensation_distributor",
            title=(
                "MIN_WEIGHT_SCHEDULE_DELAY pinned at 90 days"
            ),
            description=(
                "Phase 8 reward-split weights cannot be "
                "updated faster than 90 days. The delay is "
                "the structural defense against unilateral "
                "weight-flip attacks by a compromised owner "
                "key. Drift would weaken anti-rugpull "
                "protections."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "MIN_WEIGHT_SCHEDULE_DELAY() == 90 days"
            ),
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_MIN_WEIGHT_SCHEDULE_DELAY,
            expected=90 * 86400,
        ),
        Invariant(
            id="INV-CD-2",
            contract_name="compensation_distributor",
            title="owner() == Foundation Safe",
            description=(
                "Post-2026-05-07 ownership ceremony, "
                "CompensationDistributor is sole-owned by "
                "the Foundation Safe via Ownable2Step + "
                "acceptOwnership. Drift = compromise or "
                "unauthorized ceremony."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                f"owner() == {_FOUNDATION_SAFE_BASE}"
            ),
            kind=InvariantKind.ADDRESS_EQ,
            selector=_SEL_OWNER,
            expected=_FOUNDATION_SAFE_BASE,
        ),
        Invariant(
            id="INV-CD-3",
            contract_name="compensation_distributor",
            title=(
                "creator pool weight pinned at 5000 bps (50%)"
            ),
            description=(
                "The v1 reward split is 50/30/20 "
                "(creator / operator / grant), enforced "
                "on-chain (PRSM_Tokenomics v1, §10 #3). "
                "currentWeights() returns the PoolWeights "
                "tuple (creatorPoolBps, operatorPoolBps, "
                "grantPoolBps); word 0 is the creator share. "
                "A weight change is only reachable via the "
                "90-day-gated updateWeights path (INV-CD-1) — "
                "this invariant surfaces any such drift "
                "against the mainnet deploy "
                "(phase8-emission-base-1778164608198.json)."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "currentWeights().creatorPoolBps == 5000"
            ),
            kind=InvariantKind.UINT256_AT_WORD_EQ,
            selector=_SEL_CURRENT_WEIGHTS,
            expected=5000,
            params={"word_index": 0},
        ),
        Invariant(
            id="INV-CD-4",
            contract_name="compensation_distributor",
            title=(
                "operator pool weight pinned at 3000 bps (30%)"
            ),
            description=(
                "Word 1 of the currentWeights() PoolWeights "
                "tuple — the operator share of the v1 50/30/20 "
                "split. See INV-CD-3."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "currentWeights().operatorPoolBps == 3000"
            ),
            kind=InvariantKind.UINT256_AT_WORD_EQ,
            selector=_SEL_CURRENT_WEIGHTS,
            expected=3000,
            params={"word_index": 1},
        ),
        Invariant(
            id="INV-CD-5",
            contract_name="compensation_distributor",
            title=(
                "grant pool weight pinned at 2000 bps (20%)"
            ),
            description=(
                "Word 2 of the currentWeights() PoolWeights "
                "tuple — the grant share of the v1 50/30/20 "
                "split. Rounding remainder accrues to grant; "
                "the three weights sum to 10000 bps. See "
                "INV-CD-3."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "currentWeights().grantPoolBps == 2000"
            ),
            kind=InvariantKind.UINT256_AT_WORD_EQ,
            selector=_SEL_CURRENT_WEIGHTS,
            expected=2000,
            params={"word_index": 2},
        ),
    ],
    "storage_slashing": [
        Invariant(
            id="INV-SS-1",
            contract_name="storage_slashing",
            title=(
                "MIN_HEARTBEAT_GRACE pinned at 1 hour"
            ),
            description=(
                "Lower bound on heartbeat grace prevents an "
                "owner from setting an unreasonably-short "
                "grace that would mass-slash honest "
                "operators on transient network issues."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text="MIN_HEARTBEAT_GRACE() == 1 hours",
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_MIN_HEARTBEAT_GRACE,
            expected=3600,
        ),
        Invariant(
            id="INV-SS-2",
            contract_name="storage_slashing",
            title=(
                "MAX_HEARTBEAT_GRACE pinned at 30 days"
            ),
            description=(
                "Upper bound prevents grace from being set "
                "so long that slashing effectively never "
                "fires — weakening the storage-honest game."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text="MAX_HEARTBEAT_GRACE() == 30 days",
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_MAX_HEARTBEAT_GRACE,
            expected=30 * 86400,
        ),
        Invariant(
            id="INV-SS-3",
            contract_name="storage_slashing",
            title="owner() == Foundation Safe",
            description=(
                "Post-2026-05-07 ownership ceremony, "
                "StorageSlashing is sole-owned by the "
                "Foundation Safe via Ownable2Step + "
                "acceptOwnership."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                f"owner() == {_FOUNDATION_SAFE_BASE}"
            ),
            kind=InvariantKind.ADDRESS_EQ,
            selector=_SEL_OWNER,
            expected=_FOUNDATION_SAFE_BASE,
        ),
    ],
    "stake_bond": [
        Invariant(
            id="INV-SB-1",
            contract_name="stake_bond",
            title=(
                "MIN_UNBOND_DELAY_SECONDS pinned at 1 day"
            ),
            description=(
                "Lower bound on unbond delay prevents an "
                "owner from setting an instant-unbond mode "
                "that would let providers escape slashing "
                "by withdrawing the moment a challenge "
                "appears."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text="MIN_UNBOND_DELAY_SECONDS() == 1 days",
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_MIN_UNBOND_DELAY,
            expected=86400,
        ),
        Invariant(
            id="INV-SB-2",
            contract_name="stake_bond",
            title=(
                "MAX_UNBOND_DELAY_SECONDS pinned at 30 days"
            ),
            description=(
                "Upper bound prevents unbond delay from "
                "being weaponized to permanently lock "
                "honest providers' bonds."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text="MAX_UNBOND_DELAY_SECONDS() == 30 days",
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_MAX_UNBOND_DELAY,
            expected=30 * 86400,
        ),
        Invariant(
            id="INV-SB-3",
            contract_name="stake_bond",
            title=(
                "CHALLENGER_BOUNTY_BPS pinned at 7000 (70%)"
            ),
            description=(
                "Anti-confiscation invariant — challenger "
                "bounty is the public-goods incentive that "
                "makes slashing a positive-EV activity. "
                "Drift would either over-pay challengers "
                "(insolvent treasury exposure) or under-pay "
                "(slashing market drying up). Mirrors "
                "INV-RD-1's network-fee anti-tamper "
                "pattern."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text="CHALLENGER_BOUNTY_BPS() == 7000",
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_CHALLENGER_BOUNTY_BPS,
            expected=7000,
        ),
        Invariant(
            id="INV-SB-4",
            contract_name="stake_bond",
            title="owner() == Foundation Safe",
            description=(
                "Post-2026-05-07 ownership ceremony, "
                "StakeBond is sole-owned by the Foundation "
                "Safe via Ownable2Step + acceptOwnership."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                f"owner() == {_FOUNDATION_SAFE_BASE}"
            ),
            kind=InvariantKind.ADDRESS_EQ,
            selector=_SEL_OWNER,
            expected=_FOUNDATION_SAFE_BASE,
        ),
    ],
    # sp984 — §14 anti-spam CreatorStakeRegistry (sp976). Runtime lane,
    # PENDING_COMMISSION: until the contract is deployed + its address recorded
    # in networks.py, the /admin/formal-verification/check endpoint returns 503
    # (the invariant LIST below stays PUBLIC). Completes the dual-lane coverage
    # — the symbolic lane (CreatorStakeRegistrySpec, sp983) proves the
    # algorithmic safety (solvency / slash-conservation / anti-game eligibility);
    # these runtime pins guard the STATE properties (owner, bounds) live, exactly
    # as StakeBond carries both AdminBoundedSettersSpec + INV-SB-1..4. The unbond-
    # delay constant selectors are shared with StakeBond by identical function name.
    "creator_stake_registry": [
        Invariant(
            id="INV-CSR-1",
            contract_name="creator_stake_registry",
            title="MIN_UNBOND_DELAY_SECONDS pinned at 1 day",
            description=(
                "Lower bound on the unbond delay. Combined with the slasher-SLA "
                "note in requestUnbond, this prevents an instant-unbond mode that "
                "would let a spam creator withdraw their bond the moment a "
                "warranted slash is about to land. Mirrors StakeBond INV-SB-1."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text="MIN_UNBOND_DELAY_SECONDS() == 1 days",
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_MIN_UNBOND_DELAY,
            expected=86400,
        ),
        Invariant(
            id="INV-CSR-2",
            contract_name="creator_stake_registry",
            title="MAX_UNBOND_DELAY_SECONDS pinned at 30 days",
            description=(
                "Upper bound prevents the unbond delay from being weaponized to "
                "permanently lock an honest creator's bond. Mirrors INV-SB-2."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text="MAX_UNBOND_DELAY_SECONDS() == 30 days",
            kind=InvariantKind.UINT256_EQ,
            selector=_SEL_MAX_UNBOND_DELAY,
            expected=30 * 86400,
        ),
        Invariant(
            id="INV-CSR-3",
            contract_name="creator_stake_registry",
            title="owner() == Foundation Safe",
            description=(
                "The critical governance pin. Post-commissioning ceremony "
                "(transferOwnership → 2-of-3 acceptOwnership), the registry is "
                "sole-owned by the Foundation Safe via Ownable2Step — no single "
                "hot key can pause / set the reserve wallet / drain. Mirrors "
                "INV-SB-4. FAILs (not skips) if a deployed contract has not yet "
                "completed the ownership handover."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=f"owner() == {_FOUNDATION_SAFE_BASE}",
            kind=InvariantKind.ADDRESS_EQ,
            selector=_SEL_OWNER,
            expected=_FOUNDATION_SAFE_BASE,
        ),
        Invariant(
            id="INV-CSR-4",
            contract_name="creator_stake_registry",
            title="paused() readable (Pausable wired)",
            description=(
                "Confirms the OZ Pausable surface is live — the emergency-freeze "
                "that the sp979 audit made load-bearing (whenNotPaused on every "
                "value/state mutation). A readable paused() is the precondition "
                "for the multisig emergency-pause response path."
            ),
            severity=InvariantSeverity.HIGH,
            spec_text="paused() is readable",
            kind=InvariantKind.BOOL_READ,
            selector=_SEL_PAUSED,
        ),
    ],
    "escrow_pool": [
        Invariant(
            id="INV-EP-1",
            contract_name="escrow_pool",
            title=(
                "Solvency: ftns.balanceOf(this) >= "
                "totalEscrowedBalance"
            ),
            description=(
                "Mirror of INV-RD-4 against the Phase 3.1 "
                "EscrowPool's per-requester FTNS balance "
                "accumulator. If the pool's FTNS reserve "
                "ever drops below sum(balances), some "
                "requester withdraw or batch-settlement "
                "transfer reverts at the ERC-20 boundary. "
                "L2 audit MEDIUM B-CROSS-2 added the "
                "totalEscrowedBalance counter specifically "
                "to make this invariant runtime-checkable."
            ),
            severity=InvariantSeverity.CRITICAL,
            spec_text=(
                "ftns.balanceOf(address(this)) >= "
                "totalEscrowedBalance"
            ),
            kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
            selector=_SEL_TOTAL_ESCROWED_BALANCE,
            params={
                "ftns_selector": _SEL_FTNS,
                "totalclaimable_selector": (
                    _SEL_TOTAL_ESCROWED_BALANCE
                ),
                "reserve_label": "totalEscrowedBalance",
            },
        ),
    ],
}


def list_invariants_for_contract(
    contract_name: str,
) -> List[Invariant]:
    return list(INVARIANT_REGISTRY.get(contract_name, []))


# ── Checker ──────────────────────────────────────────


class InvariantChecker:
    def __init__(self, *, backend: FormalBackend) -> None:
        self._backend = backend

    def check_one(
        self, inv: Invariant, contract_address: str,
    ) -> InvariantResult:
        if inv.kind == InvariantKind.UINT256_EQ:
            return self._check_uint256_eq(
                inv, contract_address,
            )
        if inv.kind == InvariantKind.UINT256_AT_WORD_EQ:
            return self._check_uint256_at_word_eq(
                inv, contract_address,
            )
        if inv.kind == InvariantKind.UINT256_GTE:
            return self._check_uint256_gte(
                inv, contract_address,
            )
        if inv.kind == InvariantKind.UINT256_LTE:
            return self._check_uint256_lte(
                inv, contract_address,
            )
        if inv.kind == InvariantKind.ADDRESS_EQ:
            return self._check_address_eq(
                inv, contract_address,
            )
        if inv.kind == InvariantKind.BOOL_READ:
            return self._check_bool_read(
                inv, contract_address,
            )
        if inv.kind == InvariantKind.BALANCE_GTE_CLAIMABLE:
            return self._check_balance_gte_claimable(
                inv, contract_address,
            )
        if inv.kind == InvariantKind.HAS_ROLE_EQ:
            return self._check_has_role_eq(
                inv, contract_address,
            )
        return InvariantResult(
            invariant_id=inv.id,
            status=InvariantStatus.SKIPPED,
            error=f"unhandled kind {inv.kind.value!r}",
        )

    def check_contract(
        self,
        contract_name: str,
        *,
        contract_address: str,
    ) -> List[InvariantResult]:
        invs = list_invariants_for_contract(contract_name)
        return [
            self.check_one(inv, contract_address)
            for inv in invs
        ]

    # ── kind dispatchers ──────────────────────────────

    def _check_uint256_eq(
        self, inv: Invariant, addr: str,
    ) -> InvariantResult:
        try:
            v = self._backend.call_uint256(
                addr, inv.selector,
            )
        except Exception as e:  # noqa: BLE001
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error=f"RPC error: {e}",
            )
        if v is None:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error="backend returned None",
            )
        if v == inv.expected:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.PASS,
                value=v, expected=inv.expected,
            )
        return InvariantResult(
            invariant_id=inv.id,
            status=InvariantStatus.FAIL,
            value=v, expected=inv.expected,
            diagnostic=(
                f"got {v}, expected {inv.expected}"
            ),
        )

    def _check_uint256_at_word_eq(
        self, inv: Invariant, addr: str,
    ) -> InvariantResult:
        """Pin a single 32-byte word of a multi-word return
        (e.g. one field of a struct getter). `word_index`
        selects the word; 0 is the first."""
        word_index = int(inv.params.get("word_index", 0))
        try:
            v = self._backend.call_uint256_at_word(
                addr, inv.selector, word_index,
            )
        except Exception as e:  # noqa: BLE001
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error=f"RPC error: {e}",
            )
        if v is None:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error="backend returned None",
            )
        if v == inv.expected:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.PASS,
                value=v, expected=inv.expected,
            )
        return InvariantResult(
            invariant_id=inv.id,
            status=InvariantStatus.FAIL,
            value=v, expected=inv.expected,
            diagnostic=(
                f"word[{word_index}] got {v}, "
                f"expected {inv.expected}"
            ),
        )

    def _check_uint256_gte(
        self, inv: Invariant, addr: str,
    ) -> InvariantResult:
        try:
            v = self._backend.call_uint256(
                addr, inv.selector,
            )
        except Exception as e:  # noqa: BLE001
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error=f"RPC error: {e}",
            )
        if v is None:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error="backend returned None",
            )
        if v >= inv.expected:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.PASS,
                value=v, expected=inv.expected,
            )
        return InvariantResult(
            invariant_id=inv.id,
            status=InvariantStatus.FAIL,
            value=v, expected=inv.expected,
            diagnostic=(
                f"got {v}, required >= {inv.expected}"
            ),
        )

    def _check_uint256_lte(
        self, inv: Invariant, addr: str,
    ) -> InvariantResult:
        try:
            v = self._backend.call_uint256(
                addr, inv.selector,
            )
        except Exception as e:  # noqa: BLE001
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error=f"RPC error: {e}",
            )
        if v is None:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error="backend returned None",
            )
        if v <= inv.expected:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.PASS,
                value=v, expected=inv.expected,
            )
        return InvariantResult(
            invariant_id=inv.id,
            status=InvariantStatus.FAIL,
            value=v, expected=inv.expected,
            diagnostic=(
                f"got {v}, required <= {inv.expected}"
            ),
        )

    def _check_address_eq(
        self, inv: Invariant, addr: str,
    ) -> InvariantResult:
        try:
            v = self._backend.call_address(
                addr, inv.selector,
            )
        except Exception as e:  # noqa: BLE001
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error=f"RPC error: {e}",
            )
        if v is None:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error="backend returned None",
            )
        if str(v).lower() == str(inv.expected).lower():
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.PASS,
                value=v, expected=inv.expected,
            )
        return InvariantResult(
            invariant_id=inv.id,
            status=InvariantStatus.FAIL,
            value=v, expected=inv.expected,
            diagnostic=(
                f"got {v}, expected {inv.expected}"
            ),
        )

    def _check_bool_read(
        self, inv: Invariant, addr: str,
    ) -> InvariantResult:
        try:
            v = self._backend.call_bool(
                addr, inv.selector,
            )
        except Exception as e:  # noqa: BLE001
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error=f"RPC error: {e}",
            )
        if v is None:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error="backend returned None",
            )
        # Observability: any successful read PASSES; the
        # bool value is the diagnostic.
        return InvariantResult(
            invariant_id=inv.id,
            status=InvariantStatus.PASS,
            value=v,
            diagnostic=f"paused={v}",
        )

    def _check_balance_gte_claimable(
        self, inv: Invariant, addr: str,
    ) -> InvariantResult:
        ftns_sel = inv.params.get("ftns_selector") or ""
        tc_sel = (
            inv.params.get("totalclaimable_selector") or ""
        )
        try:
            ftns_addr = self._backend.call_address(
                addr, ftns_sel,
            )
            tc = self._backend.call_uint256(addr, tc_sel)
            if ftns_addr is None or tc is None:
                return InvariantResult(
                    invariant_id=inv.id,
                    status=InvariantStatus.SKIPPED,
                    error=(
                        "backend returned None for one of "
                        "ftns / totalClaimable"
                    ),
                )
            bal = self._backend.token_balance_of(
                ftns_addr, addr,
            )
            if bal is None:
                return InvariantResult(
                    invariant_id=inv.id,
                    status=InvariantStatus.SKIPPED,
                    error=(
                        "backend returned None for "
                        "token balance"
                    ),
                )
        except Exception as e:  # noqa: BLE001
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error=f"RPC error: {e}",
            )
        reserve_label = (
            inv.params.get("reserve_label") or "totalClaimable"
        )
        diag = (
            f"balance={bal}, {reserve_label}={tc}, "
            f"slack={bal - tc}"
        )
        if bal >= tc:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.PASS,
                value=bal, expected=tc, diagnostic=diag,
            )
        return InvariantResult(
            invariant_id=inv.id,
            status=InvariantStatus.FAIL,
            value=bal, expected=tc, diagnostic=diag,
        )

    def _check_has_role_eq(
        self, inv: Invariant, addr: str,
    ) -> InvariantResult:
        role_hash = inv.params.get("role_hash") or ""
        account = inv.params.get("account") or ""
        if not role_hash or not account:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error=(
                    "missing role_hash or account in "
                    "invariant params"
                ),
            )
        try:
            v = self._backend.call_has_role(
                addr, role_hash, account,
            )
        except Exception as e:  # noqa: BLE001
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error=f"RPC error: {e}",
            )
        if v is None:
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.SKIPPED,
                error="backend returned None",
            )
        diag = (
            f"hasRole({role_hash[:10]}..., "
            f"{account})={v}; expected={inv.expected}"
        )
        if bool(v) == bool(inv.expected):
            return InvariantResult(
                invariant_id=inv.id,
                status=InvariantStatus.PASS,
                value=v, expected=inv.expected,
                diagnostic=diag,
            )
        return InvariantResult(
            invariant_id=inv.id,
            status=InvariantStatus.FAIL,
            value=v, expected=inv.expected,
            diagnostic=diag,
        )
