"""Sprint 978 — web3 read-backend for CreatorStakeRegistry (creator-stake
commissioning, Python leg).

Decision A (runbook §5): the §14 creator-stake gate keys on the creator's ETH
ADDRESS (the canonical creator identity in §14's fingerprint dedup + content
royalty, threaded through uploads as creator_eth_address). So this backend's
`balance_of(creator)` treats `creator` as that ETH address and reads the on-chain
`CreatorStakeRegistry.creatorStakeOf(address)` — the bonded collateral that gates
HIGH creator-tier eligibility.

This is a READ backend. `stake()` is a CREATOR-WALLET action (the creator bonds
their own FTNS via the contract; the node can't do it on their behalf) and
`slash()` is SLASHER-ONLY (the governance/Foundation authority) — both raise here
rather than silently mutate, so the in-memory scaffold's "credit anyone" semantics
never carry to a commissioned node. Reads degrade to 0 on any RPC error
(fail-CLOSED for an eligibility gate: a creator simply isn't counted as HIGH-tier
if the chain is unreachable), mirroring OnChainStakeReader.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Minimal ABI — only the read surface this backend uses. Matches
# CreatorStakeRegistry.sol creatorStakeOf(address) -> uint256.
CREATOR_STAKE_REGISTRY_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "creator", "type": "address"}],
        "name": "creatorStakeOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


class CreatorStakeServerActionError(ValueError):
    """Raised when stake()/slash() is attempted server-side. Staking is a
    creator-wallet action; slashing is slasher-only — neither is executable by
    the node. (Subclasses ValueError so API handlers surface it as a 422.)"""


class CreatorStakeRegistryBackend:
    """Read-only on-chain backend for CreatorStakeClient (decision A:
    address-keyed). Construct with the registry address + an RPC URL."""

    def __init__(
        self,
        registry_address: str,
        rpc_url: str,
        *,
        web3_factory: Optional[Any] = None,
    ) -> None:
        self._registry_address = registry_address
        self._rpc_url = rpc_url
        self._web3_factory = web3_factory
        self._contract: Any = None
        self._construction_failed = False

    def _get_contract(self) -> Optional[Any]:
        if self._contract is not None:
            return self._contract
        if self._construction_failed:
            return None
        try:
            if self._web3_factory is not None:
                self._contract = self._web3_factory()
                return self._contract
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(self._rpc_url))
            self._contract = w3.eth.contract(
                address=Web3.to_checksum_address(self._registry_address),
                abi=CREATOR_STAKE_REGISTRY_ABI,
            )
            return self._contract
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "CreatorStakeRegistryBackend: contract construction failed: %s",
                exc,
            )
            self._construction_failed = True
            return None

    def balance_of(self, creator_id: str) -> int:
        """On-chain bonded stake for `creator_id` (the creator's ETH address per
        decision A). Returns 0 on any error (fail-closed eligibility read)."""
        if not creator_id:
            return 0
        contract = self._get_contract()
        if contract is None:
            return 0
        try:
            from web3 import Web3
            addr = Web3.to_checksum_address(creator_id)
        except Exception:
            # creator_id isn't a valid ETH address → no on-chain stake possible.
            logger.debug(
                "CreatorStakeRegistryBackend: %r is not a valid ETH address "
                "(decision A keys stake by creator_eth_address) — returning 0",
                creator_id,
            )
            return 0
        try:
            return int(contract.functions.creatorStakeOf(addr).call() or 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "CreatorStakeRegistryBackend.creatorStakeOf(%s) raised: %s — "
                "returning 0 (fail-closed)", creator_id[:10], exc,
            )
            return 0

    def stake(self, creator_id: str, amount_wei: int) -> None:
        raise CreatorStakeServerActionError(
            "Staking is a creator-wallet action: the creator must bond FTNS "
            "directly via CreatorStakeRegistry.stake() from their own wallet. "
            "The node cannot stake on a creator's behalf."
        )

    def slash(self, creator_id: str, amount_wei: int, reason: str) -> None:
        raise CreatorStakeServerActionError(
            "Slashing is slasher-only: only the governance/Foundation slash "
            "authority may call CreatorStakeRegistry.slash(). It is not a "
            "node-server action."
        )
