"""Phase 3 Task 3: MarketplaceAdvertiser.

Provider-side counterpart to MarketplaceDirectory. Runs a background
loop that re-broadcasts the provider's ProviderListing at a jittered
interval so the directory has fresh data before any given ttl expires.

Auto-downgrade: when the bound ComputeProvider's in-flight shard count
reaches max_concurrent_jobs, the advertiser re-signs a listing with
capacity_shards_per_sec=0.0 instead of the configured base capacity.
Requesters' EligibilityFilter (Task 4) will skip zero-capacity listings
so an overloaded provider stops attracting new dispatch requests until
load eases.

Design contract (docs/2026-04-20-phase3-marketplace-design.md §3.1):
  - Each re-broadcast carries a fresh advertised_at_unix. Stale
    listings aren't replayed.
  - rebroadcast_interval_sec defaults to 90 (per §3.1: "every 60-120s,
    jittered"). Jitter is ±25% to avoid a gossip thundering herd when
    many nodes start simultaneously.
  - ttl_seconds should be > 2 * rebroadcast_interval_sec so one missed
    broadcast does not expire the listing in the directory.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import List, Optional

from prsm.marketplace.listing import ProviderListing, sign_listing
from prsm.node.gossip import GOSSIP_MARKETPLACE_LISTING

logger = logging.getLogger(__name__)


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _resolve_stake_binding(provider_id, env):
    """sp1459 — resolve the sp1457 on-chain-stake binding for the advertised listing.

    Prefers a PRE-PRODUCED (PRSM_STAKE_ETH_ADDRESS + PRSM_STAKE_BINDING_SIG) pair (the operator
    signs it out of band with the stake-holding eth key, so that key need never touch the node).
    Falls back to deriving it at startup from PRSM_STAKE_ETH_KEY when supplied. Returns (None, None)
    when no binding is configured — the listing then advertises no binding (weighted at the tier
    label unless a selector runs in PRSM_REQUIRE_STAKE_BINDING mode)."""
    addr = str(env.get("PRSM_STAKE_ETH_ADDRESS", "") or "").strip()
    sig = str(env.get("PRSM_STAKE_BINDING_SIG", "") or "").strip()
    if addr and sig:
        return addr, sig
    key = str(env.get("PRSM_STAKE_ETH_KEY", "") or "").strip()
    if key:
        try:
            from prsm.marketplace.listing import sign_stake_binding
            return sign_stake_binding(provider_id, key)
        except Exception as exc:  # noqa: BLE001 — a bad key must not break node startup
            logger.warning(
                "marketplace: could not derive the stake binding from PRSM_STAKE_ETH_KEY "
                "(advertising WITHOUT a verified binding): %s", exc)
    return None, None


def build_marketplace_advertiser_from_env(
    *, identity, gossip, compute_provider, env=None,
) -> Optional["MarketplaceAdvertiser"]:
    """sp1459 — construct a MarketplaceAdvertiser from operator env config, or None.

    OPT-IN + FAIL-SAFE: returns None unless PRSM_MARKETPLACE_ADVERTISE is truthy AND a
    ComputeProvider exists to advertise. So an existing node's behavior is UNCHANGED by default —
    the marketplace supply side (directory → candidate pool → stake-weighted aggregator selection,
    plus the price-quote handler) is only ACTIVATED when an operator explicitly opts in. This closes
    the long-standing gap where MarketplaceAdvertiser had a full lifecycle but was never constructed
    in prod (compute_provider._marketplace_advertiser stayed None), so no node ever advertised and
    the whole selection machinery fired on an empty directory.

    Listing params come from PRSM_MARKETPLACE_* env (sensible defaults). The on-chain stake binding
    (sp1457) is attached per _resolve_stake_binding."""
    import os
    e = env if env is not None else os.environ
    if not _truthy(e.get("PRSM_MARKETPLACE_ADVERTISE", "")):
        return None
    if compute_provider is None or gossip is None or identity is None:
        return None

    def _f(key, default):
        try:
            return float(str(e.get(key, "") or "").strip() or default)
        except (ValueError, TypeError):
            return default

    def _i(key, default):
        try:
            return int(str(e.get(key, "") or "").strip() or default)
        except (ValueError, TypeError):
            return default

    dtypes_raw = str(e.get("PRSM_MARKETPLACE_DTYPES", "") or "").strip()
    supported_dtypes = [d.strip() for d in dtypes_raw.split(",") if d.strip()] or [
        "float16", "float32"]
    stake_tier = str(e.get("PRSM_MARKETPLACE_STAKE_TIER", "") or "open").strip() or "open"
    stake_eth_address, stake_binding_sig = _resolve_stake_binding(identity.node_id, e)

    return MarketplaceAdvertiser(
        identity=identity,
        gossip=gossip,
        compute_provider=compute_provider,
        capacity_shards_per_sec=_f("PRSM_MARKETPLACE_CAPACITY_SHARDS_PER_SEC", 1.0),
        max_shard_bytes=_i("PRSM_MARKETPLACE_MAX_SHARD_BYTES", 8 * 1024 * 1024),
        supported_dtypes=supported_dtypes,
        price_per_shard_ftns=_f("PRSM_MARKETPLACE_PRICE_PER_SHARD_FTNS", 1.0),
        tee_capable=_truthy(e.get("PRSM_MARKETPLACE_TEE_CAPABLE", "")),
        stake_tier=stake_tier,
        rebroadcast_interval_sec=_f("PRSM_MARKETPLACE_REBROADCAST_INTERVAL_SEC", 90.0),
        ttl_seconds=_i("PRSM_MARKETPLACE_TTL_SECONDS", 300),
        stake_eth_address=stake_eth_address,
        stake_binding_sig=stake_binding_sig,
    )


class MarketplaceAdvertiser:
    """Periodic broadcaster for a provider's marketplace listing.

    Owned by the node's ComputeProvider (wired in node.initialize).
    Exposes start()/stop() lifecycle and a current_price_ftns()
    getter so the Task 5 price-quote handler can answer with the
    live advertised price.
    """

    _JITTER_FRACTION = 0.25

    def __init__(
        self,
        identity,
        gossip,
        compute_provider,
        capacity_shards_per_sec: float,
        max_shard_bytes: int,
        supported_dtypes: List[str],
        price_per_shard_ftns: float,
        tee_capable: bool = False,
        stake_tier: str = "open",
        rebroadcast_interval_sec: float = 90.0,
        ttl_seconds: int = 300,
        stake_eth_address: Optional[str] = None,
        stake_binding_sig: Optional[str] = None,
    ):
        self.identity = identity
        self.gossip = gossip
        self.compute_provider = compute_provider
        self.base_capacity = capacity_shards_per_sec
        self.max_shard_bytes = max_shard_bytes
        self.supported_dtypes = list(supported_dtypes)
        self.price_per_shard_ftns = price_per_shard_ftns
        self.tee_capable = tee_capable
        self.stake_tier = stake_tier
        self.rebroadcast_interval_sec = rebroadcast_interval_sec
        self.ttl_seconds = ttl_seconds
        # sp1457 — the operator's on-chain-stake binding (produced once via sign_stake_binding with
        # the stake eth key), attached to every re-broadcast so the selector can weight by real stake.
        self.stake_eth_address = stake_eth_address
        self.stake_binding_sig = stake_binding_sig

        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    def current_price_ftns(self) -> float:
        """Expose the live advertised price to the Task 5 price-quote
        handler. Kept as a method (not a property) so future Phase 6
        per-request pricing can swap in a more dynamic computation
        without changing the callsite."""
        return self.price_per_shard_ftns

    def _current_listing(self) -> ProviderListing:
        """Build + sign a listing reflecting current capacity state.

        Returns a listing with capacity_shards_per_sec=0 when the bound
        ComputeProvider is at or above max_concurrent_jobs — signals to
        requesters that we're full without dropping off the directory
        entirely (better UX than a disappearing provider)."""
        effective_capacity = (
            0.0
            if self.compute_provider._current_jobs >= self.compute_provider.max_concurrent_jobs
            else self.base_capacity
        )
        return sign_listing(
            identity=self.identity,
            capacity_shards_per_sec=effective_capacity,
            max_shard_bytes=self.max_shard_bytes,
            supported_dtypes=self.supported_dtypes,
            price_per_shard_ftns=self.price_per_shard_ftns,
            tee_capable=self.tee_capable,
            stake_tier=self.stake_tier,
            ttl_seconds=self.ttl_seconds,
            stake_eth_address=self.stake_eth_address,
            stake_binding_sig=self.stake_binding_sig,
        )

    async def _broadcast_once(self) -> ProviderListing:
        """Build a current listing and publish it. Returns the listing
        so tests can inspect what was sent."""
        listing = self._current_listing()
        await self.gossip.publish(
            GOSSIP_MARKETPLACE_LISTING, listing.to_dict()
        )
        return listing

    async def _broadcast_loop(self) -> None:
        """Background loop: emit, sleep (jittered), repeat until stopped."""
        while not self._stopping.is_set():
            try:
                await self._broadcast_once()
            except Exception as exc:
                logger.warning(
                    f"marketplace advertiser broadcast failed: {exc}"
                )

            jitter = 1.0 + random.uniform(-self._JITTER_FRACTION, self._JITTER_FRACTION)
            wait = self.rebroadcast_interval_sec * jitter
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=wait)
            except asyncio.TimeoutError:
                # Normal path — time to broadcast again.
                continue

    async def start(self) -> None:
        """Emit the first broadcast immediately, then schedule the
        background re-broadcast loop. Safe to call multiple times
        (idempotent — second call is a no-op if already running)."""
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        await self._broadcast_once()
        self._task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        """Cancel the background loop and wait for it to exit."""
        self._stopping.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
