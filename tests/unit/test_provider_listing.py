"""Unit tests for ProviderListing + sign/verify.

Phase 3 Task 1. Exercises:
  - sign/verify roundtrip (happy path)
  - tamper detection (field mutation breaks signature)
  - forged provider_id (claim A's identity with B's pubkey)
  - dict-based serialization roundtrip
  - ttl expiry
  - sanity-check rejection (negative ttl, empty dtypes)
"""
from __future__ import annotations

import time

from prsm.marketplace.listing import (
    ProviderListing,
    build_listing_signing_payload,
    sign_listing,
    verify_listing,
)
from prsm.node.identity import generate_node_identity


def _fresh():
    return generate_node_identity(display_name="test-provider")


def _make_valid_listing(identity, **overrides):
    kwargs = dict(
        capacity_shards_per_sec=10.0,
        max_shard_bytes=10 * 1024 * 1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.05,
        tee_capable=False,
        stake_tier="standard",
        ttl_seconds=300,
    )
    kwargs.update(overrides)
    return sign_listing(identity=identity, **kwargs)


def test_listing_signing_roundtrip():
    identity = _fresh()
    listing = _make_valid_listing(identity)
    assert verify_listing(listing) is True
    assert listing.provider_id == identity.node_id
    assert listing.provider_pubkey_b64 == identity.public_key_b64


def test_listing_tamper_detection_price():
    """Flipping price after signing breaks the signature."""
    identity = _fresh()
    listing = _make_valid_listing(identity)
    tampered = ProviderListing(
        **{**listing.to_dict(), "price_per_shard_ftns": 0.01}
    )
    assert verify_listing(tampered) is False


def test_listing_tamper_detection_capacity():
    identity = _fresh()
    listing = _make_valid_listing(identity)
    tampered = ProviderListing(
        **{**listing.to_dict(), "capacity_shards_per_sec": 99999.0}
    )
    assert verify_listing(tampered) is False


def test_listing_provider_id_must_match_pubkey():
    """Closes the 'claim provider A's node_id while carrying B's pubkey'
    attack at the listing layer — same guard Phase 2 receipts use."""
    victim = _fresh()
    attacker = _fresh()
    assert victim.node_id != attacker.node_id

    listing = _make_valid_listing(attacker)
    forged = ProviderListing(
        **{**listing.to_dict(), "provider_id": victim.node_id}
    )
    assert verify_listing(forged) is False


def test_listing_roundtrip_serialization():
    identity = _fresh()
    listing = _make_valid_listing(identity, stake_tier="premium", tee_capable=True)
    as_dict = listing.to_dict()
    restored = ProviderListing.from_dict(as_dict)
    assert restored == listing
    assert verify_listing(restored) is True


def test_listing_is_expired():
    identity = _fresh()
    now = int(time.time())
    listing = sign_listing(
        identity=identity,
        capacity_shards_per_sec=1.0,
        max_shard_bytes=1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.01,
        tee_capable=False,
        stake_tier="open",
        ttl_seconds=1,
        advertised_at_unix=now,
    )
    assert listing.is_expired(at_unix=now + 2) is True
    assert listing.is_expired(at_unix=now) is False


def test_listing_rejects_empty_dtypes():
    """A listing with no supported_dtypes can never be selected —
    verify drops it at ingestion so it never clutters the directory."""
    identity = _fresh()
    # sign_listing happily signs anything, so we construct by hand.
    payload = build_listing_signing_payload(
        listing_id="test", provider_id=identity.node_id,
        capacity_shards_per_sec=1.0, max_shard_bytes=1024,
        price_per_shard_ftns=0.01, tee_capable=False,
        stake_tier="open", advertised_at_unix=int(time.time()),
        ttl_seconds=60,
    )
    sig = identity.sign(payload)
    listing = ProviderListing(
        listing_id="test",
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        capacity_shards_per_sec=1.0,
        max_shard_bytes=1024,
        supported_dtypes=[],  # empty
        price_per_shard_ftns=0.01,
        tee_capable=False,
        stake_tier="open",
        advertised_at_unix=int(time.time()),
        ttl_seconds=60,
        signature=sig,
    )
    assert verify_listing(listing) is False


def test_listing_rejects_negative_ttl():
    identity = _fresh()
    payload = build_listing_signing_payload(
        listing_id="neg", provider_id=identity.node_id,
        capacity_shards_per_sec=1.0, max_shard_bytes=1024,
        price_per_shard_ftns=0.01, tee_capable=False,
        stake_tier="open", advertised_at_unix=int(time.time()),
        ttl_seconds=-1,
    )
    sig = identity.sign(payload)
    listing = ProviderListing(
        listing_id="neg", provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        capacity_shards_per_sec=1.0, max_shard_bytes=1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.01, tee_capable=False,
        stake_tier="open", advertised_at_unix=int(time.time()),
        ttl_seconds=-1, signature=sig,
    )
    assert verify_listing(listing) is False


# ── sp1457: authenticated on-chain-stake binding ──────────────────────────────
# Closes the self-declared-stake-tier selection-weight exploit: the provider's STAKE-HOLDING eth
# key must sign a provider_id↔address binding, so the selector can trust (and read) the real
# on-chain stake instead of a self-asserted tier.

from eth_account import Account as _Account
from eth_account.messages import encode_defunct as _encode_defunct

from prsm.marketplace.listing import (
    build_stake_binding_message,
    verify_stake_binding,
)

_ETH_KEY = "0x" + "a1" * 32
_ETH = _Account.from_key(_ETH_KEY)


def _bind(provider_id, address=None, key=None):
    acct = _Account.from_key(key) if key else _ETH
    addr = address or acct.address
    msg = build_stake_binding_message(provider_id, addr)
    return acct.sign_message(_encode_defunct(text=msg)).signature.hex()


def test_stake_binding_verifies_for_the_controlling_key():
    pid = "f" * 32
    sig = _bind(pid)
    assert verify_stake_binding(pid, _ETH.address, sig) is True


def test_stake_binding_rejects_a_different_provider_id():
    # A binding signed for provider A cannot be replayed onto provider B's listing.
    sig = _bind("a" * 32)
    assert verify_stake_binding("b" * 32, _ETH.address, sig) is False


def test_stake_binding_rejects_claiming_an_address_you_dont_control():
    # A liar advertises a RICH staker's address but can't produce its signature.
    pid = "c" * 32
    rich_addr = _Account.from_key("0x" + "bb" * 32).address
    forged = _bind(pid)  # signed by the liar's OWN key, not rich_addr's
    assert verify_stake_binding(pid, rich_addr, forged) is False


def test_stake_binding_rejects_garbage_and_missing():
    pid = "d" * 32
    assert verify_stake_binding(pid, _ETH.address, "0xdead") is False
    assert verify_stake_binding(pid, _ETH.address, "") is False
    assert verify_stake_binding(pid, "", _bind(pid)) is False
    assert verify_stake_binding("", _ETH.address, _bind(pid)) is False


def test_listing_has_verified_stake_binding_true_when_bound():
    identity = _fresh()
    sig = _bind(identity.node_id)
    listing = sign_listing(
        identity=identity, capacity_shards_per_sec=10.0, max_shard_bytes=1024,
        supported_dtypes=["float64"], price_per_shard_ftns=0.05, tee_capable=False,
        stake_tier="T4", ttl_seconds=300,
        stake_eth_address=_ETH.address, stake_binding_sig=sig)
    assert verify_listing(listing) is True                     # base listing still valid
    assert listing.has_verified_stake_binding() is True        # + stake binding verifies


def test_listing_binding_false_when_absent_or_forged():
    identity = _fresh()
    # legacy listing (no binding fields) → not verified, but still a valid listing (backward-compat)
    legacy = _make_valid_listing(identity)
    assert verify_listing(legacy) is True
    assert legacy.has_verified_stake_binding() is False
    # a forged binding for an address the provider doesn't control → not verified
    forged = sign_listing(
        identity=identity, capacity_shards_per_sec=10.0, max_shard_bytes=1024,
        supported_dtypes=["float64"], price_per_shard_ftns=0.05, tee_capable=False,
        stake_tier="T4", ttl_seconds=300,
        stake_eth_address=_Account.from_key("0x" + "cc" * 32).address,
        stake_binding_sig=_bind(identity.node_id))  # sig is by _ETH, not the cc address
    assert forged.has_verified_stake_binding() is False


def test_listing_dict_roundtrip_preserves_binding():
    identity = _fresh()
    sig = _bind(identity.node_id)
    listing = sign_listing(
        identity=identity, capacity_shards_per_sec=10.0, max_shard_bytes=1024,
        supported_dtypes=["float64"], price_per_shard_ftns=0.05, tee_capable=False,
        stake_tier="T4", ttl_seconds=300,
        stake_eth_address=_ETH.address, stake_binding_sig=sig)
    back = ProviderListing.from_dict(listing.to_dict())
    assert back.stake_eth_address == _ETH.address
    assert back.has_verified_stake_binding() is True


def test_sign_stake_binding_roundtrips_with_verify():
    # PROVIDER side: sign with the stake eth key → the verifier accepts it.
    from prsm.marketplace.listing import sign_stake_binding
    pid = "e" * 32
    eth_key = "0x" + "7f" * 32
    address, sig = sign_stake_binding(pid, eth_key)
    assert sig.startswith("0x")
    assert verify_stake_binding(pid, address, sig) is True
    # and it's bound to THIS provider_id only
    assert verify_stake_binding("0" * 32, address, sig) is False
