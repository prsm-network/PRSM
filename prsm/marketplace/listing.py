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


# sp1460 — upper bound on ttl_seconds. Generous (1 day) — the point is to reject the ABSURD, not to
# bound a legitimate operator. Without it a listing with a decades-long ttl never expires, defeating
# the directory's lazy expiry-eviction so a flood of long-ttl valid listings grows a listening node's
# directory permanently (memory DoS). Env-tunable: PRSM_MAX_LISTING_TTL_SECONDS.
_DEFAULT_MAX_LISTING_TTL_SECONDS = 86_400


def _max_listing_ttl() -> int:
    try:
        return max(
            1,
            int(os.environ.get(
                "PRSM_MAX_LISTING_TTL_SECONDS",
                _DEFAULT_MAX_LISTING_TTL_SECONDS,
            )),
        )
    except (TypeError, ValueError):
        return _DEFAULT_MAX_LISTING_TTL_SECONDS


# sp1462 — allowance for benign clock skew between nodes on advertised_at_unix. A listing timestamped
# further than this into the future is rejected: is_expired() is (advertised_at + ttl) < now, so an
# UNBOUNDED future advertised_at makes a listing never expire, defeating the sp1460 ttl cap (the
# expiry anchor would be attacker-controlled) → immortal listings that permanently occupy the
# size-capped directory. Env-tunable: PRSM_MAX_LISTING_FUTURE_SKEW_SEC.
_DEFAULT_MAX_LISTING_FUTURE_SKEW_SEC = 300


def _max_listing_future_skew() -> int:
    try:
        return max(
            0,
            int(os.environ.get(
                "PRSM_MAX_LISTING_FUTURE_SKEW_SEC",
                _DEFAULT_MAX_LISTING_FUTURE_SKEW_SEC,
            )),
        )
    except (TypeError, ValueError):
        return _DEFAULT_MAX_LISTING_FUTURE_SKEW_SEC


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
    supported_dtypes: Optional[List[str]] = None,
    stake_eth_address: Optional[str] = None,
    stake_binding_sig: Optional[str] = None,
) -> bytes:
    """Canonical bytes the provider signs over for a listing.

    Format (design §3.1): keccak256 of a pipe-delimited UTF-8 string
    composed of ALL selection-relevant fields. Matches the on-chain-
    compatible hashing scheme used elsewhere in PRSM (Phase 7 slashing
    will verify signatures over the same scheme).

    sp1461 — the payload also binds ``supported_dtypes`` (an eligibility
    input) and the sp1457 stake binding (``stake_eth_address`` +
    ``stake_binding_sig``). Without them a relay on the gossip path could
    tamper with dtypes or STRIP/swap the binding while keeping the
    Ed25519 signature valid — altering who is eligible / how a provider is
    stake-weighted. Order-preserving join (the provider signs its declared
    dtype order; verify reproduces from the same list)."""
    dtypes_canon = ",".join(supported_dtypes or [])
    addr_canon = str(stake_eth_address or "")
    bind_canon = str(stake_binding_sig or "")
    raw = (
        f"{listing_id}||{provider_id}||{capacity_shards_per_sec}||"
        f"{max_shard_bytes}||{price_per_shard_ftns}||{tee_capable}||"
        f"{stake_tier}||{advertised_at_unix}||{ttl_seconds}||"
        f"{dtypes_canon}||{addr_canon}||{bind_canon}"
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


def build_stake_binding_message(provider_id: str, stake_eth_address: str) -> str:
    """sp1457 — the EIP-191 message the provider's STAKE-HOLDING eth key signs to prove it
    authorized ``provider_id`` to advertise ``stake_eth_address``'s on-chain stake.

    Closes the self-declared-stake-tier selection-weight exploit: aggregator selection weights a
    provider by stake, but the stake_tier was self-asserted (a zero-stake node could claim T4 and
    win ~25x more paid work, unslashable). Requiring this proof means the selector can read the
    REAL on-chain ``stake_of(stake_eth_address)`` and trust the address belongs to this provider.
    A relay cannot forge the eth signature, and the message binds a SPECIFIC provider_id so the
    proof can never be reused for a different node identity."""
    return f"PRSM-stake-binding:v1:{str(provider_id)}:{str(stake_eth_address).lower()}"


def sign_stake_binding(provider_id: str, eth_private_key: str) -> "tuple[str, str]":
    """sp1457 — PROVIDER side: sign the provider_id↔stake_address binding with the STAKE-HOLDING
    eth key, so the operator can advertise a listing whose stake is on-chain-verifiable. Returns
    ``(stake_eth_address, stake_binding_sig)`` to pass to ``sign_listing`` / MarketplaceAdvertiser.
    The eth key must be the one controlling the operator's bonded StakeBond stake."""
    from eth_account import Account
    from eth_account.messages import encode_defunct
    acct = Account.from_key(eth_private_key)
    msg = encode_defunct(text=build_stake_binding_message(provider_id, acct.address))
    sig = acct.sign_message(msg).signature.hex()
    if not sig.startswith("0x"):
        sig = "0x" + sig
    return acct.address, sig


def verify_stake_binding(
    provider_id: str, stake_eth_address: str, stake_binding_sig: str,
) -> bool:
    """True iff ``stake_binding_sig`` is a valid EIP-191 signature over
    ``build_stake_binding_message(provider_id, stake_eth_address)`` by the key controlling
    ``stake_eth_address`` — i.e. the on-chain stake holder authorized THIS provider_id. Never
    raises; returns False on any missing/malformed input, recovery failure, or address mismatch."""
    if not (provider_id and stake_eth_address and stake_binding_sig):
        return False
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        from eth_utils import to_checksum_address
        msg = encode_defunct(
            text=build_stake_binding_message(provider_id, stake_eth_address))
        recovered = Account.recover_message(msg, signature=stake_binding_sig)
        return to_checksum_address(recovered) == to_checksum_address(stake_eth_address)
    except Exception:  # noqa: BLE001 — malformed sig/address → not verified
        return False


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
    # sp1457 — OPTIONAL authenticated on-chain-stake binding. When present, stake_binding_sig is
    # an EIP-191 signature by the key controlling stake_eth_address over
    # build_stake_binding_message(provider_id, stake_eth_address), proving the on-chain stake
    # holder authorized this provider_id. The selector then weights by the REAL stake_of(address)
    # instead of the self-asserted stake_tier. Absent (None) → unverified stake (legacy behavior).
    stake_eth_address: Optional[str] = None
    stake_binding_sig: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def has_verified_stake_binding(self) -> bool:
        """True iff this listing carries a stake binding that cryptographically verifies (the
        stake_eth_address key authorized this provider_id). Selection uses the REAL on-chain
        stake only when this holds."""
        return bool(self.stake_eth_address) and verify_stake_binding(
            self.provider_id, self.stake_eth_address or "", self.stake_binding_sig or "")

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
            stake_eth_address=data.get("stake_eth_address"),
            stake_binding_sig=data.get("stake_binding_sig"),
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
    stake_eth_address: Optional[str] = None,
    stake_binding_sig: Optional[str] = None,
) -> ProviderListing:
    """Construct + sign a ProviderListing using a NodeIdentity.

    The listing's provider_id and provider_pubkey_b64 are populated
    from the identity directly — callers cannot inject a mismatched
    claim here. (They CAN construct a ProviderListing manually via
    __init__; verify_listing then rejects the mismatch.)

    sp1457 — pass ``stake_eth_address`` + ``stake_binding_sig`` (an EIP-191 signature by the
    stake-holding eth key over ``build_stake_binding_message(provider_id, stake_eth_address)``)
    to authenticate the provider's on-chain stake for weight-verified selection. The eth-key
    signature is produced by the operator OUT OF BAND (this node holds only the Ed25519 identity)."""
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
        supported_dtypes=list(supported_dtypes),
        stake_eth_address=stake_eth_address,
        stake_binding_sig=stake_binding_sig,
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
        stake_eth_address=stake_eth_address,
        stake_binding_sig=stake_binding_sig,
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

    # sp1460 — reject an absurdly-long ttl so every accepted listing expires within a bounded window
    # (else lazy expiry-eviction can never reclaim a flood of "immortal" listings → memory DoS).
    ttl_max = _max_listing_ttl()
    if listing.ttl_seconds > ttl_max:
        logger.warning(
            f"listing {listing.listing_id!r} rejected: ttl {listing.ttl_seconds} "
            f"exceeds max {ttl_max}"
        )
        return False

    # sp1462 — reject a listing dated too far in the FUTURE. Without this an attacker sets
    # advertised_at_unix decades ahead so (advertised_at + ttl) never elapses → the listing never
    # expires and the sp1460 ttl cap is meaningless (the expiry anchor is attacker-controlled),
    # letting a one-shot flood permanently occupy the size-capped directory and lock out honest
    # providers. A small skew allowance tolerates benign NTP drift between nodes.
    skew = _max_listing_future_skew()
    if listing.advertised_at_unix > int(time.time()) + skew:
        logger.warning(
            f"listing {listing.listing_id!r} rejected: advertised_at_unix "
            f"{listing.advertised_at_unix} is more than {skew}s in the future"
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
        supported_dtypes=listing.supported_dtypes,
        stake_eth_address=listing.stake_eth_address,
        stake_binding_sig=listing.stake_binding_sig,
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
