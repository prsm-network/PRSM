"""Sprint 290 — Creator stake client (Vision §14 item 2).

Implements the operator-side scaffold for "Staking
requirements for high-tier creator status — uploading
requires collateral that can be slashed." The economic
intuition: a creator who spams loses bonded FTNS, making
spam-uploading economically unattractive.

This v1 follows the Phase 5 pattern (Coinbase WaaS,
paymaster): Python adapter with dependency-injected backend.
PENDING_COMMISSION pre-deploy — `stake`/`slash`/`balance`
operate against an in-memory mirror; calls record state
locally without on-chain settlement. Post-deploy + env-
configured (CREATOR_STAKE_REGISTRY_ADDRESS + BASE_RPC_URL),
the backend delegates to the real `CreatorStakeRegistry.sol`
contract.

`apply_stake_gate` is a pure function that layers stake-
eligibility on top of the sprint-288 score-based tier — the
CreatorReputationTracker stays unchanged (score-only); api.py
composes the score-based tier with the stake gate. This
matches the architecture pattern where the tracker is the
data substrate and gates are layered.

Threshold: MIN_HIGH_TIER_STAKE_WEI defaults to 1000 FTNS
(1000 × 10**18 wei). Tunable via env var at import time;
operators can pick higher thresholds for stricter networks.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Protocol

from prsm.marketplace.creator_reputation import (
    TIER_HIGH, TIER_MEDIUM,
)

logger = logging.getLogger(__name__)


def _read_min_stake_from_env() -> int:
    raw = (
        os.environ.get("PRSM_MIN_HIGH_TIER_STAKE_WEI") or ""
    ).strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except (ValueError, TypeError):
            logger.warning(
                "PRSM_MIN_HIGH_TIER_STAKE_WEI=%r is not a "
                "positive integer; using default.",
                raw,
            )
    # 1000 FTNS = 1000 × 10**18 wei (18-decimal FTNS token)
    return 1000 * (10 ** 18)


MIN_HIGH_TIER_STAKE_WEI = _read_min_stake_from_env()


class _StakeBackend(Protocol):
    """Dependency-injected backend. Production wraps the real
    CreatorStakeRegistry.sol contract; tests use a fake."""

    def stake(self, creator_id: str, amount_wei: int) -> None: ...
    def slash(
        self, creator_id: str, amount_wei: int, reason: str,
    ) -> None: ...
    def balance_of(self, creator_id: str) -> int: ...


class CreatorStakeClient:
    """Operator-side stake adapter with in-memory fallback."""

    def __init__(
        self,
        registry_address: Optional[str] = None,
        rpc_url: Optional[str] = None,
        *,
        backend: Optional[_StakeBackend] = None,
    ) -> None:
        self._registry_address = registry_address
        self._rpc_url = rpc_url
        self._backend = backend
        # In-memory mirror (used when uncommissioned OR when
        # backend reads fail).
        self._balances: Dict[str, int] = {}

    @classmethod
    def from_env(
        cls, *, backend: Optional[_StakeBackend] = None,
    ) -> "CreatorStakeClient":
        # sp981 — resolve the address + RPC through the canonical per-network
        # registry (prsm.config.networks), the same path every other deployed
        # contract uses. The operator records the post-ceremony address ONCE in
        # networks.py (or via the CREATOR_STAKE_REGISTRY_ADDRESS env override) and
        # the gate goes live against the network's default Base RPC — no separate
        # RPC env required. Fail-soft to the legacy direct-env read if resolution
        # raises (e.g. a misconfigured PRSM_NETWORK), so from_env never crashes
        # the gate at node startup.
        addr: Optional[str]
        rpc: Optional[str]
        try:
            from prsm.config.networks import resolve_endpoints
            ep = resolve_endpoints()
            addr = ep.creator_stake_registry or None
            # An explicit RPC env still wins; otherwise the resolved network RPC
            # (which is never None — falls back to the Base default).
            rpc = (
                os.environ.get("BASE_RPC_URL")
                or ep.rpc_url
                or None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CreatorStakeClient.from_env: network resolution failed (%s) — "
                "falling back to direct env vars.", exc,
            )
            addr = os.environ.get("CREATOR_STAKE_REGISTRY_ADDRESS") or None
            rpc = os.environ.get("BASE_RPC_URL") or None
        # sp978 — when commissioned (registry address + RPC set) and no explicit
        # backend was injected, construct the real on-chain read backend so the
        # stake gate has actual teeth (pre-sp978 from_env never wired a backend →
        # the gate ran on the in-memory mirror forever). Fail-soft: if the
        # backend can't be built, fall back to the in-memory dev scaffold.
        if backend is None and addr and rpc:
            try:
                from prsm.economy.web3.creator_stake_registry_backend import (
                    CreatorStakeRegistryBackend,
                )
                backend = CreatorStakeRegistryBackend(addr, rpc)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CreatorStakeClient.from_env: could not build the on-chain "
                    "backend (%s) — using the in-memory dev scaffold.", exc,
                )
                backend = None
        return cls(
            registry_address=addr, rpc_url=rpc, backend=backend,
        )

    def is_commissioned(self) -> bool:
        """True iff both registry address and RPC URL are set.
        Backend presence is independent — operators may have
        env wired but no backend implementation yet; that
        still counts as commissioned for the
        is_commissioned() predicate (operator can verify env
        config before plugging the SDK)."""
        return bool(self._registry_address and self._rpc_url)

    # ── Mutations ────────────────────────────────────────

    def stake(self, creator_id: str, amount_wei: int) -> None:
        if not creator_id:
            raise ValueError("creator_id must be non-empty")
        if not isinstance(amount_wei, int) or amount_wei <= 0:
            raise ValueError(
                f"amount_wei must be a positive integer, "
                f"got {amount_wei!r}"
            )
        # sp978 — when a real backend is wired, it is authoritative: do NOT fall
        # back to the in-memory mirror on error (that fallback was the toothless
        # free-stake bypass — a backend error or a non-server-action would
        # silently credit the gate). The in-memory mirror is ONLY the
        # uncommissioned (backend=None) dev scaffold.
        if self._backend is not None:
            self._backend.stake(creator_id, amount_wei)
            return
        self._balances[creator_id] = (
            self._balances.get(creator_id, 0) + amount_wei
        )

    def slash(
        self,
        creator_id: str,
        amount_wei: int,
        reason: str,
    ) -> None:
        if not creator_id:
            raise ValueError("creator_id must be non-empty")
        if not isinstance(amount_wei, int) or amount_wei <= 0:
            raise ValueError(
                f"amount_wei must be a positive integer, "
                f"got {amount_wei!r}"
            )
        if not reason:
            raise ValueError("reason must be non-empty")
        # sp978 — backend authoritative when wired (no in-memory fallback; see
        # stake()). In-memory only for the uncommissioned dev scaffold.
        if self._backend is not None:
            self._backend.slash(creator_id, amount_wei, reason)
            return
        current = self._balances.get(creator_id, 0)
        self._balances[creator_id] = max(0, current - amount_wei)

    # ── Reads ────────────────────────────────────────────

    def stake_balance(self, creator_id: str) -> int:
        if self._backend is not None:
            try:
                return int(self._backend.balance_of(creator_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CreatorStakeClient: backend.balance_of "
                    "raised: %s — returning 0 (fail-soft)",
                    exc,
                )
                return 0
        return self._balances.get(creator_id, 0)

    def is_high_tier_eligible(self, creator_id: str) -> bool:
        return (
            self.stake_balance(creator_id)
            >= MIN_HIGH_TIER_STAKE_WEI
        )


# ── Pure tier-gate function ──────────────────────────────


def apply_stake_gate(
    tier: str,
    creator_eth_address: Optional[str],
    stake_client: Optional[CreatorStakeClient],
) -> str:
    """Layer stake-eligibility on top of a score-based tier.

    Demotes HIGH → MEDIUM when the creator hasn't bonded the
    minimum stake. Other tiers pass through unchanged
    (stake doesn't buy you score, and the gate only acts on
    HIGH).

    sp978 (decision A): the stake identity is the creator's ETH
    ADDRESS (the §14 canonical creator key — fingerprint dedup +
    content royalty already use it), NOT the node_id. A falsy
    address (a record with no creator_eth_address) can't have
    bonded stake → demote to MEDIUM (safe default).

    stake_client=None is a no-op passthrough — preserves
    backwards-compat for sprint 287/288/289 callers that
    don't yet thread the stake client.
    """
    if stake_client is None:
        return tier
    if tier != TIER_HIGH:
        return tier
    if not creator_eth_address:
        return TIER_MEDIUM  # no eth address → no bonded stake → demote
    try:
        if stake_client.is_high_tier_eligible(creator_eth_address):
            return TIER_HIGH
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "apply_stake_gate: eligibility check raised "
            "for %s: %s — demoting to MEDIUM (defensive)",
            creator_eth_address, exc,
        )
    return TIER_MEDIUM
