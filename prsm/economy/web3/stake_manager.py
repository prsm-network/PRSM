"""StakeBond Web3 Client — Phase 7 Task 4.

Thin Python wrapper around the on-chain StakeBond contract. Mirrors the
patterns established by provenance_registry.py — Web3.py 7.x,
synchronous calls, explicit private-key signing, per-client tx lock to
prevent nonce races.

StakeBond is the provider-stake registry that backs Tier C slashing.
Providers bond FTNS to claim a tier; successful challenges against them
invalidate the receipt AND burn/redirect their stake (70% to challenger,
30% to Foundation reserve; 100% to Foundation on self-slash).

Writes exposed: bond, request_unbond, withdraw, claim_bounty.
Reads exposed: stake_of, effective_tier, slashed_bounty_payable,
foundation_reserve_balance, unbond_delay_seconds.

Governance entries (setSlasher/setUnbondDelay/setFoundationReserveWallet)
are intentionally NOT wrapped — they run from the owner multi-sig via
the hardhat CLI, not from production Python code.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)

logger = logging.getLogger(__name__)

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# Tier thresholds in FTNS wei — must mirror StakeBond.sol:effectiveTier.
# If the contract thresholds change, these MUST be updated in lock-step.
TIER_STANDARD_MIN_WEI = 5_000 * 10**18
TIER_PREMIUM_MIN_WEI = 25_000 * 10**18
TIER_CRITICAL_MIN_WEI = 50_000 * 10**18

# Tier → recommended slash rate (bps). Callers pass one of these to bond()
# unless they are intentionally bonding at a non-standard rate.
SLASH_RATE_STANDARD_BPS = 5_000   # 50%
SLASH_RATE_PREMIUM_BPS = 10_000   # 100%
SLASH_RATE_CRITICAL_BPS = 10_000  # 100%


class ReasonCode(IntEnum):
    """Mirrors BatchSettlementRegistry.sol:ReasonCode.

    Ordering is stable — new codes append. Python callers who pass a
    reason into challengeReceipt must use the integer value from this
    enum; the contract's function signature accepts `uint8`.
    """
    DOUBLE_SPEND = 0
    INVALID_SIGNATURE = 1
    NO_ESCROW = 2
    EXPIRED = 3
    MALFORMED = 4          # reserved; reverts on-chain if used
    CONSENSUS_MISMATCH = 5  # Phase 7.1 — k-of-n redundant-execution disagreement


# Reasons that trigger on-chain slashing via the registry's challenge
# hook. Kept as a frozenset so callers can cheaply test membership when
# deciding whether to expect a Slashed event on a challenge receipt.
SLASHABLE_REASONS = frozenset({
    ReasonCode.DOUBLE_SPEND,
    ReasonCode.INVALID_SIGNATURE,
    ReasonCode.CONSENSUS_MISMATCH,
})


def _reason_keccak(reason: "ReasonCode | str") -> bytes:
    """Return keccak256(reason_name_utf8). Useful when callers need a
    bytes32 identifier for off-chain correlation (e.g., logging a slash
    event alongside the reason that triggered it).

    Accepts the enum OR the raw string so callers can build identifiers
    for reason codes the current Python build doesn't know about —
    future-proof against new ReasonCode values landing on-chain first.
    """
    if not HAS_WEB3:
        raise RuntimeError("web3 package required for keccak hashing")
    name = reason.name if isinstance(reason, ReasonCode) else str(reason)
    return bytes(Web3.keccak(name.encode("utf-8")))


# Precomputed keccak identifiers for the slashable reasons. Useful for
# event-log correlation and audit trails that want a canonical 32-byte
# tag for each reason independent of the (small, potentially re-ordered)
# enum integer. Note: the StakeBond.Slashed event's `reasonId` is the
# batchId the slash was tied to, NOT this value — these are separate
# identifiers that serve different observability needs.
REASON_DOUBLE_SPEND_KECCAK = None  # set below once web3 is imported
REASON_INVALID_SIGNATURE_KECCAK = None
REASON_CONSENSUS_MISMATCH_KECCAK = None
if HAS_WEB3:
    REASON_DOUBLE_SPEND_KECCAK = _reason_keccak(ReasonCode.DOUBLE_SPEND)
    REASON_INVALID_SIGNATURE_KECCAK = _reason_keccak(ReasonCode.INVALID_SIGNATURE)
    REASON_CONSENSUS_MISMATCH_KECCAK = _reason_keccak(ReasonCode.CONSENSUS_MISMATCH)


class StakeStatus(IntEnum):
    """Mirrors StakeBond.sol:StakeStatus."""
    NONE = 0
    BONDED = 1
    UNBONDING = 2
    WITHDRAWN = 3


# Hardcoded ABI subset — only the surface this wrapper exposes. Matches
# contracts/artifacts/contracts/StakeBond.sol/StakeBond.json.
STAKE_BOND_ABI = [
    {
        "inputs": [
            {"internalType": "uint128", "name": "amount", "type": "uint128"},
            {"internalType": "uint16", "name": "tierSlashRateBps", "type": "uint16"},
        ],
        "name": "bond",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "requestUnbond",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "withdraw",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "claimBounty",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "provider", "type": "address"}],
        "name": "stakeOf",
        "outputs": [
            {
                "components": [
                    {"internalType": "uint128", "name": "amount", "type": "uint128"},
                    {"internalType": "uint64", "name": "bonded_at_unix", "type": "uint64"},
                    {"internalType": "uint64", "name": "unbond_eligible_at", "type": "uint64"},
                    {"internalType": "uint8", "name": "status", "type": "uint8"},
                    {"internalType": "uint16", "name": "tier_slash_rate_bps", "type": "uint16"},
                ],
                "internalType": "struct StakeBond.Stake",
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "provider", "type": "address"}],
        "name": "effectiveTier",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "challenger", "type": "address"}],
        "name": "slashedBountyPayable",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "foundationReserveBalance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "unbondDelaySeconds",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "provider", "type": "address"},
            {"indexed": False, "internalType": "uint128", "name": "amount", "type": "uint128"},
            {"indexed": False, "internalType": "uint16", "name": "tierSlashRateBps", "type": "uint16"},
            {"indexed": False, "internalType": "uint64", "name": "bondedAt", "type": "uint64"},
        ],
        "name": "Bonded",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "provider", "type": "address"},
            {"indexed": False, "internalType": "uint64", "name": "eligibleAt", "type": "uint64"},
        ],
        "name": "UnbondRequested",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "provider", "type": "address"},
            {"indexed": False, "internalType": "uint128", "name": "amount", "type": "uint128"},
        ],
        "name": "Withdrawn",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "provider", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "challenger", "type": "address"},
            {"indexed": True, "internalType": "bytes32", "name": "reasonId", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "slashAmount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "challengerBounty", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "foundationShare", "type": "uint256"},
        ],
        "name": "Slashed",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "challenger", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "BountyClaimed",
        "type": "event",
    },
]


@dataclass
class StakeRecord:
    """View of a provider's on-chain stake."""
    amount_wei: int
    bonded_at_unix: int
    unbond_eligible_at: int
    status: StakeStatus
    tier_slash_rate_bps: int

    @property
    def is_bonded(self) -> bool:
        return self.status == StakeStatus.BONDED

    @property
    def is_unbonding(self) -> bool:
        return self.status == StakeStatus.UNBONDING


class StakeManagerClient:
    """Sync Web3 client for StakeBond.

    One client per provider keypair. Governance addresses (owner /
    slasher) are NOT routed through this wrapper — those operate
    from the owner multi-sig via hardhat.
    """

    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        private_key: Optional[str] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError(
                "web3 package is required (pip install web3 eth-account)"
            )

        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.web3.eth.contract(
            address=self.contract_address,
            abi=STAKE_BOND_ABI,
        )

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

        # Per-account lock from the process-wide registry. Cross-client
        # nonce safety: any other web3 client (e.g., ProvenanceRegistry,
        # BatchSettlement) signing with the same account address acquires
        # the same lock instance, serializing the
        # build_transaction → sign → broadcast sequence across clients.
        # Resolves Phase 7 §8.8. Falls back to a private Lock if there's
        # no account (read-only client, unreachable from writes).
        from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY

        if self._account is not None:
            self._tx_lock = TX_LOCK_REGISTRY.get_lock(self._account.address)
        else:
            self._tx_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Writes ─────────────────────────────────────────────────

    def bond(
        self, amount_wei: int, tier_slash_rate_bps: int
    ) -> Tuple[str, TransferStatus]:
        """Post FTNS stake and snapshot the slash rate.

        Caller must have already approved the StakeBond address for
        `amount_wei` on the FTNS token contract — this wrapper does NOT
        issue the approval (keeping token and stake concerns separate).

        Returns (tx_hash_hex, TransferStatus). May raise
        BroadcastFailedError, OnChainPendingError, or OnChainRevertedError.
        """
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if amount_wei <= 0:
            raise ValueError(f"amount_wei must be positive (got {amount_wei})")
        if amount_wei >= 2**128:
            raise ValueError(
                f"amount_wei must fit in uint128 (got {amount_wei})"
            )
        if not (0 <= tier_slash_rate_bps <= 10_000):
            raise ValueError(
                f"tier_slash_rate_bps must be in [0, 10000] "
                f"(got {tier_slash_rate_bps})"
            )

        with self._tx_lock:
            tx = self.contract.functions.bond(
                amount_wei, tier_slash_rate_bps
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    def approve_and_bond(
        self,
        *,
        ftns_token_address: str,
        amount_wei: int,
        tier_slash_rate_bps: int,
    ) -> Tuple[str, TransferStatus]:
        """Sprint 1379 — one-call operator stake: approve FTNS to the StakeBond contract (only when
        the current allowance is short), then bond. ``bond()`` deliberately keeps token + stake
        concerns separate; this is the convenience the ``prsm node stake-bond`` CLI uses so an
        operator posts stake in ONE command. The approve is awaited (``_sign_and_send`` blocks on the
        receipt) before the bond, so the bond's ``transferFrom`` sees the allowance even against a
        lagging RPC replica, and the sequential ``pending`` nonces stay correct. Returns the BOND
        ``(tx_hash_hex, TransferStatus)``."""
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if amount_wei <= 0:
            raise ValueError(f"amount_wei must be positive (got {amount_wei})")
        if amount_wei >= 2 ** 128:
            raise ValueError(f"amount_wei must fit in uint128 (got {amount_wei})")
        if not (0 <= tier_slash_rate_bps <= 10_000):
            raise ValueError(
                f"tier_slash_rate_bps must be in [0, 10000] (got {tier_slash_rate_bps})")
        _erc20_abi = [
            {"name": "approve", "type": "function", "stateMutability": "nonpayable",
             "inputs": [{"name": "spender", "type": "address"},
                        {"name": "amount", "type": "uint256"}],
             "outputs": [{"name": "", "type": "bool"}]},
            {"name": "allowance", "type": "function", "stateMutability": "view",
             "inputs": [{"name": "owner", "type": "address"},
                        {"name": "spender", "type": "address"}],
             "outputs": [{"name": "", "type": "uint256"}]},
        ]
        ftns = self.web3.eth.contract(
            address=Web3.to_checksum_address(ftns_token_address), abi=_erc20_abi)
        owner = self._account.address
        with self._tx_lock:
            allowance = int(
                ftns.functions.allowance(owner, self.contract_address).call())
            if allowance < amount_wei:
                approve_tx = ftns.functions.approve(
                    self.contract_address, int(amount_wei)
                ).build_transaction(self._tx_overrides())
                self._sign_and_send(approve_tx)   # awaited — bond's transferFrom needs the allowance
            bond_tx = self.contract.functions.bond(
                int(amount_wei), int(tier_slash_rate_bps)
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(bond_tx)

    def request_unbond(self) -> Tuple[str, TransferStatus]:
        """Transition BONDED → UNBONDING. Slashing remains active during
        the unbond delay — a provider caught mid-unbond still forfeits."""
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        with self._tx_lock:
            tx = self.contract.functions.requestUnbond().build_transaction(
                self._tx_overrides()
            )
            return self._sign_and_send(tx)

    def withdraw(self) -> Tuple[str, TransferStatus]:
        """Finalize unbond: UNBONDING → WITHDRAWN and return FTNS.
        Reverts on-chain if the unbond delay hasn't elapsed."""
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        with self._tx_lock:
            tx = self.contract.functions.withdraw().build_transaction(
                self._tx_overrides()
            )
            return self._sign_and_send(tx)

    def claim_bounty(self) -> Tuple[str, TransferStatus]:
        """Claim any slashed-bounty FTNS owed to caller as challenger.
        Reverts on-chain if the caller has zero bounty payable."""
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        with self._tx_lock:
            tx = self.contract.functions.claimBounty().build_transaction(
                self._tx_overrides()
            )
            return self._sign_and_send(tx)

    # ── Reads ──────────────────────────────────────────────────

    def stake_of(self, provider: str) -> StakeRecord:
        """Return the full stake record for `provider`. If the provider
        has never bonded, the record's status is NONE and amount is 0."""
        provider_cs = Web3.to_checksum_address(provider)
        raw = self.contract.functions.stakeOf(provider_cs).call()
        # raw is a tuple (amount, bonded_at, unbond_eligible_at, status, rate_bps)
        amount, bonded_at, unbond_eligible_at, status, rate_bps = raw
        return StakeRecord(
            amount_wei=int(amount),
            bonded_at_unix=int(bonded_at),
            unbond_eligible_at=int(unbond_eligible_at),
            status=StakeStatus(int(status)),
            tier_slash_rate_bps=int(rate_bps),
        )

    def effective_tier(self, provider: str) -> str:
        """Return the current effective tier ("open"/"standard"/
        "premium"/"critical"). Drops to "open" during UNBONDING."""
        provider_cs = Web3.to_checksum_address(provider)
        return self.contract.functions.effectiveTier(provider_cs).call()

    def slashed_bounty_payable(self, challenger: str) -> int:
        """Return FTNS wei claimable by `challenger` from past slashings."""
        challenger_cs = Web3.to_checksum_address(challenger)
        return int(
            self.contract.functions.slashedBountyPayable(challenger_cs).call()
        )

    def foundation_reserve_balance(self) -> int:
        return int(self.contract.functions.foundationReserveBalance().call())

    def unbond_delay_seconds(self) -> int:
        return int(self.contract.functions.unbondDelaySeconds().call())

    # ── Internals ──────────────────────────────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending"
            ),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_and_send(self, tx: dict) -> Tuple[str, TransferStatus]:
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )

        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise BroadcastFailedError(f"broadcast failed: {exc}") from exc

        tx_hash_hex = "0x" + tx_hash_bytes.hex()

        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash_bytes, timeout=120
            )
        except Exception as exc:
            raise OnChainPendingError(
                f"broadcast OK but receipt unknown: {exc}", tx_hash=tx_hash_hex
            ) from exc

        if receipt.status != 1:
            raise OnChainRevertedError(f"tx reverted: {tx_hash_hex}")

        return tx_hash_hex, TransferStatus.CONFIRMED


# Re-export the shared exception types so callers importing from this
# module don't need to reach into provenance_registry.
__all__ = [
    "StakeManagerClient",
    "StakeRecord",
    "StakeStatus",
    "ReasonCode",
    "SLASHABLE_REASONS",
    "TransferStatus",
    "BroadcastFailedError",
    "OnChainPendingError",
    "OnChainRevertedError",
    "TIER_STANDARD_MIN_WEI",
    "TIER_PREMIUM_MIN_WEI",
    "TIER_CRITICAL_MIN_WEI",
    "SLASH_RATE_STANDARD_BPS",
    "SLASH_RATE_PREMIUM_BPS",
    "SLASH_RATE_CRITICAL_BPS",
    "REASON_DOUBLE_SPEND_KECCAK",
    "REASON_INVALID_SIGNATURE_KECCAK",
    "REASON_CONSENSUS_MISMATCH_KECCAK",
]
