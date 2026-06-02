"""Phase 3 Task 1: ProviderListing wire format + sign/verify.

A provider broadcasts its available capacity + price as a signed
ProviderListing over GOSSIP_MARKETPLACE_LISTING. Requesters aggregate
these in MarketplaceDirectory (Task 2), filter by policy (Task 4),
and dispatch via the Phase 2 RemoteShardDispatcher.

Security model:
  - Ed25519 signature over a keccak256 canonical payload. Same
    pattern as the Phase 2 ShardExecutionReceipt.
  - provider_id must match hex(sha256(public_key_bytes))[:32] —
    the NodeIdentity derivation. Closes the "adversary publishes
    listings under another node's identity" attack at the listing
    layer, mirroring Phase 2's receipt-verification guard.
  - Stale listings (advertised_at + ttl < now) are dropped by the
    directory. Providers re-broadcast periodically to stay live.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from eth_utils import keccak

logger = logging.getLogger(__name__)

# sp926 — implausibility ceiling on a self-reported capacity claim. A single
# node cannot realistically serve more than this many shards/sec; a claim above
# it is rejected at ingestion so it can't game the priority-boost ranking.
# Generous default (env-tunable) — the point is to reject the absurd, not to
# tightly bound legitimate high-end operators.
_DEFAULT_MAX_LISTING_CAPACITY = 1_000_000.0


def _max_listing_capacity() -> float:
    """sp926 — upper bound on capacity_shards_per_sec. PRSM_MAX_LISTING_CAPACITY_SHARDS_PER_SEC."""
    try:
        return max(
            1.0,
            float(os.environ.get(
                "PRSM_MAX_LISTING_CAPACITY_SHARDS_PER_SEC",
                _DEFAULT_MAX_LISTING_CAPACITY,
            )),
        )
    except (TypeError, ValueError):
        return _DEFAULT_MAX_LISTING_CAPACITY


def build_listing_signing_payload(
    listing_id: str,
    provider_id: str,
    capacity_shards_per_sec: float,
    max_shard_bytes: int,
    price_per_shard_ftns: float,
    tee_capable: bool,
    stake_tier: str,
    advertised_at_unix: int,
    ttl_seconds: int,
) -> bytes:
    """Canonical bytes the provider signs over for a listing.

    Format (design §3.1): keccak256 of a pipe-delimited UTF-8 string
    composed of all price/capacity-relevant fields. Matches the on-
    chain-compatible hashing scheme used elsewhere in PRSM (Phase 7
    slashing will verify signatures over the same scheme).
    """
    raw = (
        f"{listing_id}||{provider_id}||{capacity_shards_per_sec}||"
        f"{max_shard_bytes}||{price_per_shard_ftns}||{tee_capable}||"
        f"{stake_tier}||{advertised_at_unix}||{ttl_seconds}"
    ).encode("utf-8")
    return keccak(raw)


def _derive_node_id_from_pubkey_b64(pubkey_b64: str) -> Optional[str]:
    """Recompute NodeIdentity.node_id from an advertised pubkey.

    Must match generate_node_identity() exactly — see prsm/node/identity.py.
    Returns None on decode failure so callers can treat it as verification
    failure rather than raise.
    """
    try:
        pub_bytes = base64.b64decode(pubkey_b64)
    except Exception:
        return None
    return hashlib.sha256(pub_bytes).hexdigest()[:32]


@dataclass(frozen=True)
class ProviderListing:
    """Signed advertisement of a provider's capacity + price.

    Carried as the payload of a GOSSIP_MARKETPLACE_LISTING message.
    """
    listing_id: str
    provider_id: str
    provider_pubkey_b64: str
    capacity_shards_per_sec: float
    max_shard_bytes: int
    supported_dtypes: List[str]
    price_per_shard_ftns: float
    tee_capable: bool
    stake_tier: str
    advertised_at_unix: int
    ttl_seconds: int
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderListing":
        return cls(
            listing_id=data["listing_id"],
            provider_id=data["provider_id"],
            provider_pubkey_b64=data["provider_pubkey_b64"],
            capacity_shards_per_sec=data["capacity_shards_per_sec"],
            max_shard_bytes=data["max_shard_bytes"],
            supported_dtypes=list(data["supported_dtypes"]),
            price_per_shard_ftns=data["price_per_shard_ftns"],
            tee_capable=data["tee_capable"],
            stake_tier=data["stake_tier"],
            advertised_at_unix=data["advertised_at_unix"],
            ttl_seconds=data["ttl_seconds"],
            signature=data["signature"],
        )

    def is_expired(self, at_unix: Optional[int] = None) -> bool:
        """Return True if advertised_at + ttl_seconds is in the past
        relative to at_unix (defaults to current wall time)."""
        now = at_unix if at_unix is not None else int(time.time())
        return (self.advertised_at_unix + self.ttl_seconds) < now


def sign_listing(
    identity,
    capacity_shards_per_sec: float,
    max_shard_bytes: int,
    supported_dtypes: List[str],
    price_per_shard_ftns: float,
    tee_capable: bool,
    stake_tier: str,
    ttl_seconds: int = 300,
    listing_id: Optional[str] = None,
    advertised_at_unix: Optional[int] = None,
) -> ProviderListing:
    """Construct + sign a ProviderListing using a NodeIdentity.

    The listing's provider_id and provider_pubkey_b64 are populated
    from the identity directly — callers cannot inject a mismatched
    claim here. (They CAN construct a ProviderListing manually via
    __init__; verify_listing then rejects the mismatch.)
    """
    lid = listing_id or f"listing-{uuid.uuid4().hex[:16]}"
    when = advertised_at_unix if advertised_at_unix is not None else int(time.time())
    payload = build_listing_signing_payload(
        listing_id=lid,
        provider_id=identity.node_id,
        capacity_shards_per_sec=capacity_shards_per_sec,
        max_shard_bytes=max_shard_bytes,
        price_per_shard_ftns=price_per_shard_ftns,
        tee_capable=tee_capable,
        stake_tier=stake_tier,
        advertised_at_unix=when,
        ttl_seconds=ttl_seconds,
    )
    sig = identity.sign(payload)
    return ProviderListing(
        listing_id=lid,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        capacity_shards_per_sec=capacity_shards_per_sec,
        max_shard_bytes=max_shard_bytes,
        supported_dtypes=list(supported_dtypes),
        price_per_shard_ftns=price_per_shard_ftns,
        tee_capable=tee_capable,
        stake_tier=stake_tier,
        advertised_at_unix=when,
        ttl_seconds=ttl_seconds,
        signature=sig,
    )


def verify_listing(listing: ProviderListing) -> bool:
    """Four checks (all must pass):
      1. ttl_seconds >= 0 and price_per_shard_ftns >= 0 (sanity).
      2. provider_id == hex(sha256(provider_pubkey_b64))[:32]
         (identity binding — closes the forged-provider_id attack).
      3. Ed25519 signature valid against provider_pubkey_b64 over
         build_listing_signing_payload(...) (authenticity).
      4. supported_dtypes non-empty (a listing with no dtypes can
         never be selected by any policy — drop at ingestion).

    Never raises; returns False and logs warning on any failure.
    """
    if listing.ttl_seconds < 0 or listing.price_per_shard_ftns < 0:
        logger.warning(
            f"listing {listing.listing_id!r} rejected: negative ttl or price"
        )
        return False

    # sp926 — capacity sanity. A negative capacity is nonsensical (and would
    # make the selection cap_norm negative); an implausibly large one is a
    # ranking-gaming claim. Bound it at ingestion.
    cap_max = _max_listing_capacity()
    if listing.capacity_shards_per_sec < 0 or listing.capacity_shards_per_sec > cap_max:
        logger.warning(
            f"listing {listing.listing_id!r} rejected: capacity "
            f"{listing.capacity_shards_per_sec} out of bounds [0, {cap_max}]"
        )
        return False

    if not listing.supported_dtypes:
        logger.warning(
            f"listing {listing.listing_id!r} rejected: empty supported_dtypes"
        )
        return False

    derived = _derive_node_id_from_pubkey_b64(listing.provider_pubkey_b64)
    if derived is None or derived != listing.provider_id:
        logger.warning(
            f"listing {listing.listing_id!r} rejected: provider_id "
            f"{str(listing.provider_id)[:12]}… does not match pubkey-derived "
            f"node_id {str(derived)[:12]}…"
        )
        return False

    payload = build_listing_signing_payload(
        listing_id=listing.listing_id,
        provider_id=listing.provider_id,
        capacity_shards_per_sec=listing.capacity_shards_per_sec,
        max_shard_bytes=listing.max_shard_bytes,
        price_per_shard_ftns=listing.price_per_shard_ftns,
        tee_capable=listing.tee_capable,
        stake_tier=listing.stake_tier,
        advertised_at_unix=listing.advertised_at_unix,
        ttl_seconds=listing.ttl_seconds,
    )

    try:
        from prsm.node.identity import verify_signature
        if not verify_signature(listing.provider_pubkey_b64, payload, listing.signature):
            logger.warning(
                f"listing {listing.listing_id!r} rejected: signature invalid "
                f"for provider {listing.provider_id[:12]}…"
            )
            return False
    except ImportError:
        logger.warning(
            f"listing {listing.listing_id!r} rejected: verify_signature unavailable"
        )
        return False
    except Exception as exc:
        logger.warning(
            f"listing {listing.listing_id!r} verification raised: {exc}"
        )
        return False

    return True
