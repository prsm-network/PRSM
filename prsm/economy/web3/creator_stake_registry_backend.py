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

sp995 — blip tolerance + caching. The §14 search-tier path reads stake per result
row, so a naive backend issued one blocking on-chain call per row AND fail-closed to
0 on any transient RPC hiccup — silently demoting/excluding a genuinely-staked HIGH
creator on RPC weather. `balance_of` now keeps a short-TTL per-address cache: a fresh
entry is served without a chain call (dedups within + across requests), reads retry a
couple times, and on an exhausted read it serves the last-known-good value while it
is within a longer stale-grace window — only fail-closing to 0 once that grace also
expires (preserving fail-closed for a SUSTAINED outage, not a momentary blip).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v >= 0 else default
    except (TypeError, ValueError):
        logger.warning("%s=%r not a number; using %s", name, raw, default)
        return default


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry_address = registry_address
        self._rpc_url = rpc_url
        self._web3_factory = web3_factory
        self._contract: Any = None
        self._construction_failed = False
        # sp995 — blip-tolerant read cache. addr → (balance_wei, read_ts).
        self._clock = clock
        self._cache: Dict[str, Tuple[int, float]] = {}
        # Within TTL: serve cached, no chain call. On read failure: serve cached
        # if within the (longer) stale-grace, else fail-closed to 0.
        self._cache_ttl_s = _env_float("PRSM_CREATOR_STAKE_CACHE_TTL_S", 60.0)
        self._stale_grace_s = _env_float(
            "PRSM_CREATOR_STAKE_STALE_GRACE_S", 600.0,
        )
        self._read_attempts = max(
            1, int(_env_float("PRSM_CREATOR_STAKE_READ_ATTEMPTS", 3.0)),
        )
        self._cache_max = 4096  # bound vs arbitrary address fan-out

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

    def _cache_put(self, addr: str, value: int, now: float) -> None:
        if len(self._cache) >= self._cache_max and addr not in self._cache:
            # bound: evict the oldest entry
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            self._cache.pop(oldest, None)
        self._cache[addr] = (value, now)

    def balance_of(self, creator_id: str) -> int:
        """On-chain bonded stake for `creator_id` (the creator's ETH address per
        decision A). Blip-tolerant: serves a fresh cached value without a chain
        call, retries a failing read, and on an exhausted read serves the
        last-known-good value while within the stale-grace window — only
        fail-closing to 0 once the grace also expires (sustained outage)."""
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

        now = self._clock()
        cached = self._cache.get(addr)
        # Fast path: a fresh cached value is served without a chain call. This
        # dedups the per-result-row reads on /content/search (one read per
        # distinct creator per TTL) and avoids the event-loop-blocking fan-out.
        if cached is not None and (now - cached[1]) < self._cache_ttl_s:
            return cached[0]

        last_exc: Optional[Exception] = None
        for _ in range(self._read_attempts):
            try:
                val = int(contract.functions.creatorStakeOf(addr).call() or 0)
                self._cache_put(addr, val, now)
                return val
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

        # Read failed after retries. Prefer last-known-good over a false 0 so a
        # momentary RPC blip doesn't silently demote a genuinely-staked creator;
        # only fail-closed to 0 once the cached value is also beyond stale-grace
        # (a SUSTAINED outage — the documented fail-closed contract).
        if cached is not None and (now - cached[1]) < self._stale_grace_s:
            logger.warning(
                "CreatorStakeRegistryBackend.creatorStakeOf(%s) read failed "
                "(%s) — serving last-known-good %s wei (age %.0fs)",
                creator_id[:10], last_exc, cached[0], now - cached[1],
            )
            return cached[0]
        logger.warning(
            "CreatorStakeRegistryBackend.creatorStakeOf(%s) read failed (%s) "
            "and no fresh cache — fail-closed to 0", creator_id[:10], last_exc,
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
