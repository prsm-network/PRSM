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
