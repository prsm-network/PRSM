"""Phase 3 Task 2: MarketplaceDirectory.

Requester-side aggregator. Subscribes to GOSSIP_MARKETPLACE_LISTING
and maintains an in-memory dict of verified, unexpired provider
listings. Exposes a simple query API for the EligibilityFilter
(Task 4) and MarketplaceOrchestrator (Task 7).

Design contract (docs/2026-04-20-phase3-marketplace-design.md §3.2):
  - Verify signature on every ingress. Drop invalid listings silently.
  - Replace an existing listing for a provider_id only if the new
    advertised_at_unix is strictly greater — protects against replay
    of old listings with better terms.
  - Evict expired listings lazily (on list/get/size) rather than on
    a timer. Avoids a background task; simpler and matches Phase 2's
    event-driven style.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import os

from prsm.marketplace.listing import ProviderListing, verify_listing
from prsm.node.gossip import GOSSIP_MARKETPLACE_LISTING

logger = logging.getLogger(__name__)

# sp1460 — hard cap on distinct provider listings held in memory. verify_listing requires
# provider_id == sha256(pubkey) + a valid sig, but keypairs are cheap, so a flood of DISTINCT valid
# listings would otherwise grow this dict without bound → OOM on every listening node. Generous
# default; env-tunable via PRSM_MARKETPLACE_DIRECTORY_MAX.
_DEFAULT_DIRECTORY_MAX = 10_000


def _max_directory_size() -> int:
    try:
        return max(1, int(os.environ.get("PRSM_MARKETPLACE_DIRECTORY_MAX", _DEFAULT_DIRECTORY_MAX)))
    except (TypeError, ValueError):
        return _DEFAULT_DIRECTORY_MAX


class MarketplaceDirectory:
    """In-memory directory of active provider listings.

    One instance per requester node. Aggregates gossip-advertised
    listings and lets Phase 3 components (filter, orchestrator,
    reputation display) query who's currently offering what.
    """

    def __init__(self, gossip):
        self._listings: Dict[str, ProviderListing] = {}
        self._gossip = gossip
        gossip.subscribe(GOSSIP_MARKETPLACE_LISTING, self._on_listing)

    async def _on_listing(
        self, subtype: str, data: Dict[str, Any], origin: str,
    ) -> None:
        """Handle an incoming GOSSIP_MARKETPLACE_LISTING event.

        Verifies signature + binding; replaces any existing entry for
        the same provider_id only if strictly newer.
        """
        try:
            listing = ProviderListing.from_dict(data)
        except (KeyError, TypeError) as exc:
            logger.warning(
                f"marketplace directory dropped malformed listing: {exc}"
            )
            return

        if not verify_listing(listing):
            # verify_listing already logs the specific reason.
            return

        existing = self._listings.get(listing.provider_id)
        if existing is not None:
            if existing.advertised_at_unix >= listing.advertised_at_unix:
                # Old or equal — ignore. Prevents replay of stale listings
                # with attacker-favorable terms (e.g., lower price).
                return
            # A newer listing for an ALREADY-PRESENT provider: replace in place. The size cap gates
            # only NEW provider_ids, so an honest provider's periodic refresh is never starved by a
            # full directory.
            self._listings[listing.provider_id] = listing
            return

        # sp1460 — a NEW provider_id: enforce the size cap so a flood of distinct valid listings
        # cannot exhaust memory. Reclaim expired entries first; if still full, drop the new listing
        # (availability over completeness — the node stays up with a bounded, if flood-contended,
        # directory rather than OOMing). Logged, never silent.
        cap = _max_directory_size()
        if len(self._listings) >= cap:
            self._evict_expired(int(time.time()))
            if len(self._listings) >= cap:
                logger.warning(
                    "marketplace directory at capacity (%d) — dropping new listing from "
                    "provider %s… (possible listing flood)",
                    cap, str(listing.provider_id)[:12],
                )
                return

        self._listings[listing.provider_id] = listing

    def _evict_expired(self, at_unix: int) -> None:
        """Drop listings whose TTL has elapsed. Called lazily by query
        methods to keep the dict small without a background task."""
        stale = [
            pid for pid, l in self._listings.items()
            if l.is_expired(at_unix)
        ]
        for pid in stale:
            del self._listings[pid]

    def list_active_providers(
        self, at_unix: Optional[int] = None,
    ) -> List[ProviderListing]:
        """Return all currently-active listings.

        Evicts expired entries first so the returned list is always
        fresh (no need for callers to filter again)."""
        now = at_unix if at_unix is not None else int(time.time())
        self._evict_expired(now)
        return list(self._listings.values())

    def get_listing(
        self, provider_id: str, at_unix: Optional[int] = None,
    ) -> Optional[ProviderListing]:
        """Return the listing for a provider_id if it exists and isn't
        expired. Evicts on access to stay consistent with
        list_active_providers."""
        now = at_unix if at_unix is not None else int(time.time())
        listing = self._listings.get(provider_id)
        if listing is None:
            return None
        if listing.is_expired(now):
            del self._listings[provider_id]
            return None
        return listing

    def size(self, at_unix: Optional[int] = None) -> int:
        """Number of active (unexpired) listings."""
        now = at_unix if at_unix is not None else int(time.time())
        self._evict_expired(now)
        return len(self._listings)
