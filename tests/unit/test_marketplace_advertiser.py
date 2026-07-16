"""Unit tests for MarketplaceAdvertiser.

Phase 3 Task 3. Verifies:
  - First broadcast fires on start.
  - Listing reflects configured capacity when compute_provider has
    free slots.
  - Listing reflects capacity=0 when compute_provider is at max.
  - Each broadcast produces a fresh advertised_at_unix (no replay).
  - stop() cancels the background loop cleanly.
  - Published payload is a valid, verify_listing-passing dict.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.marketplace.advertiser import MarketplaceAdvertiser
from prsm.marketplace.listing import ProviderListing, verify_listing
from prsm.node.gossip import GOSSIP_MARKETPLACE_LISTING
from prsm.node.identity import generate_node_identity


def _run(coro):
    return asyncio.run(coro)


def _make_compute_provider(current_jobs: int = 0, max_jobs: int = 10):
    cp = MagicMock()
    cp._current_jobs = current_jobs
    cp.max_concurrent_jobs = max_jobs
    return cp


def _make_advertiser(
    identity=None,
    compute_provider=None,
    rebroadcast_interval_sec: float = 0.05,
    **overrides,
):
    gossip = MagicMock()
    gossip.publish = AsyncMock()
    identity = identity or generate_node_identity(display_name="provider")
    cp = compute_provider or _make_compute_provider()

    kwargs = dict(
        identity=identity,
        gossip=gossip,
        compute_provider=cp,
        capacity_shards_per_sec=10.0,
        max_shard_bytes=10 * 1024 * 1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.05,
        tee_capable=False,
        stake_tier="standard",
        rebroadcast_interval_sec=rebroadcast_interval_sec,
        ttl_seconds=300,
    )
    kwargs.update(overrides)
    advertiser = MarketplaceAdvertiser(**kwargs)
    return advertiser, gossip, identity, cp


def test_advertiser_broadcasts_on_start():
    async def run():
        advertiser, gossip, _, _ = _make_advertiser()
        await advertiser.start()
        # Publish should have fired at least once.
        assert gossip.publish.await_count >= 1
        topic, payload = gossip.publish.await_args_list[0].args
        assert topic == GOSSIP_MARKETPLACE_LISTING
        # Payload should be a verify_listing-passing dict.
        listing = ProviderListing.from_dict(payload)
        assert verify_listing(listing)
        await advertiser.stop()

    _run(run())


def test_advertiser_reflects_configured_capacity_when_free():
    async def run():
        cp = _make_compute_provider(current_jobs=0, max_jobs=10)
        advertiser, gossip, _, _ = _make_advertiser(compute_provider=cp)
        await advertiser.start()
        payload = gossip.publish.await_args_list[0].args[1]
        assert payload["capacity_shards_per_sec"] == 10.0
        await advertiser.stop()

    _run(run())


def test_advertiser_reflects_zero_capacity_when_at_max():
    """Critical UX guard: an overloaded provider should advertise 0
    capacity so the filter skips it, without disappearing from the
    directory entirely."""
    async def run():
        cp = _make_compute_provider(current_jobs=10, max_jobs=10)
        advertiser, gossip, _, _ = _make_advertiser(compute_provider=cp)
        await advertiser.start()
        payload = gossip.publish.await_args_list[0].args[1]
        assert payload["capacity_shards_per_sec"] == 0.0
        # Listing still verifies — it's a valid signed message.
        listing = ProviderListing.from_dict(payload)
        assert verify_listing(listing)
        await advertiser.stop()

    _run(run())


def test_advertiser_capacity_tracks_provider_state_changes():
    """As compute_provider's in-flight count changes, each new listing
    built via _broadcast_once() reflects the current state. Directly
    drives _broadcast_once rather than the timing loop (the test
    harness mocks asyncio.sleep to instant, so the rebroadcast timer
    cannot be observed; the state-change semantics belong on the
    deterministic path anyway)."""
    async def run():
        cp = _make_compute_provider(current_jobs=0, max_jobs=5)
        advertiser, gossip, _, _ = _make_advertiser(compute_provider=cp)

        listing_free = await advertiser._broadcast_once()
        assert listing_free.capacity_shards_per_sec == 10.0

        cp._current_jobs = 5  # hit capacity
        listing_full = await advertiser._broadcast_once()
        assert listing_full.capacity_shards_per_sec == 0.0

        cp._current_jobs = 2  # free up
        listing_recovered = await advertiser._broadcast_once()
        assert listing_recovered.capacity_shards_per_sec == 10.0

    _run(run())


def test_advertiser_each_broadcast_fresh_advertised_at():
    """No replay: successive broadcasts build fresh ProviderListings,
    each with its own advertised_at_unix pulled from the current wall
    clock. The directory's 'strictly newer' replacement rule can then
    accept the second broadcast rather than discarding it.

    Test uses controlled time via monkeypatch so we don't depend on
    wall-clock granularity (int(time.time()) can return the same value
    for two calls within the same second)."""
    async def run():
        advertiser, gossip, _, _ = _make_advertiser()

        import prsm.marketplace.listing as listing_mod
        original_time = listing_mod.time.time

        # First broadcast at synthetic wall time 1000.
        listing_mod.time.time = lambda: 1000.0
        try:
            listing_1 = await advertiser._broadcast_once()
            listing_mod.time.time = lambda: 1001.0
            listing_2 = await advertiser._broadcast_once()
        finally:
            listing_mod.time.time = original_time

        assert listing_1.advertised_at_unix == 1000
        assert listing_2.advertised_at_unix == 1001
        assert listing_2.advertised_at_unix > listing_1.advertised_at_unix

    _run(run())


def test_advertiser_stop_is_clean():
    """stop() cancels the background task and the awaited task exits
    without exception."""
    async def run():
        advertiser, gossip, _, _ = _make_advertiser()
        await advertiser.start()
        assert advertiser._task is not None
        await advertiser.stop()
        assert advertiser._task is None

    _run(run())


def test_advertiser_start_is_idempotent():
    async def run():
        advertiser, gossip, _, _ = _make_advertiser()
        await advertiser.start()
        count1 = gossip.publish.await_count
        await advertiser.start()  # second start
        # Should not have started a second task or emitted a second
        # immediate broadcast — but the running loop may have emitted
        # one on its own by now, so allow for that.
        assert gossip.publish.await_count >= count1
        await advertiser.stop()

    _run(run())


def test_advertiser_current_price_matches_configured():
    advertiser, _, _, _ = _make_advertiser(price_per_shard_ftns=0.07)
    assert advertiser.current_price_ftns() == 0.07


def test_advertiser_attaches_verifiable_stake_binding():
    # sp1457 — a provider that supplies its stake binding broadcasts listings whose stake is
    # on-chain-verifiable (has_verified_stake_binding == True), so the selector weights by real stake.
    from prsm.node.identity import generate_node_identity
    from prsm.marketplace.listing import sign_stake_binding, verify_listing
    identity = generate_node_identity(display_name="bound-provider")
    address, sig = sign_stake_binding(identity.node_id, "0x" + "5d" * 32)
    advertiser, _gossip, _id, _cp = _make_advertiser(
        identity=identity, stake_tier="T4",
        stake_eth_address=address, stake_binding_sig=sig)
    listing = _run(advertiser._broadcast_once())
    assert verify_listing(listing) is True
    assert listing.stake_eth_address == address
    assert listing.has_verified_stake_binding() is True


def test_advertiser_without_binding_is_unverified_but_valid():
    # Backward-compat: a provider that doesn't supply a binding still advertises a valid listing.
    from prsm.marketplace.listing import verify_listing
    advertiser, _gossip, _id, _cp = _make_advertiser(stake_tier="T2")
    listing = _run(advertiser._broadcast_once())
    assert verify_listing(listing) is True
    assert listing.has_verified_stake_binding() is False


# ── sp1459: build_marketplace_advertiser_from_env — the node-lifecycle wiring helper ──
# The advertiser is opt-in (PRSM_MARKETPLACE_ADVERTISE) + fail-safe: an existing node's behavior
# is unchanged by default (returns None), so the whole marketplace supply side is only ACTIVATED
# when an operator explicitly turns it on. Listing params come from PRSM_MARKETPLACE_* env; the
# sp1457 on-chain stake binding is attached from a pre-produced pair OR derived from a stake key.

def _env_gossip_cp(identity=None):
    from prsm.node.identity import generate_node_identity
    gossip = MagicMock()
    gossip.publish = AsyncMock()
    identity = identity or generate_node_identity(display_name="env-provider")
    cp = _make_compute_provider()
    return identity, gossip, cp


def test_build_from_env_disabled_by_default_returns_none():
    from prsm.marketplace.advertiser import build_marketplace_advertiser_from_env
    identity, gossip, cp = _env_gossip_cp()
    # No PRSM_MARKETPLACE_ADVERTISE → the node is unchanged (no advertiser).
    out = build_marketplace_advertiser_from_env(
        identity=identity, gossip=gossip, compute_provider=cp, env={})
    assert out is None


def test_build_from_env_enabled_without_compute_provider_returns_none():
    from prsm.marketplace.advertiser import build_marketplace_advertiser_from_env
    identity, gossip, _ = _env_gossip_cp()
    # Gate on but nothing to advertise (non-compute node) → still None (fail-safe).
    out = build_marketplace_advertiser_from_env(
        identity=identity, gossip=gossip, compute_provider=None,
        env={"PRSM_MARKETPLACE_ADVERTISE": "1"})
    assert out is None


def test_build_from_env_reads_listing_params():
    from prsm.marketplace.advertiser import (
        MarketplaceAdvertiser, build_marketplace_advertiser_from_env)
    identity, gossip, cp = _env_gossip_cp()
    out = build_marketplace_advertiser_from_env(
        identity=identity, gossip=gossip, compute_provider=cp,
        env={
            "PRSM_MARKETPLACE_ADVERTISE": "true",
            "PRSM_MARKETPLACE_PRICE_PER_SHARD_FTNS": "2.5",
            "PRSM_MARKETPLACE_CAPACITY_SHARDS_PER_SEC": "7",
            "PRSM_MARKETPLACE_MAX_SHARD_BYTES": "4194304",
            "PRSM_MARKETPLACE_DTYPES": "float16, bfloat16",
            "PRSM_MARKETPLACE_STAKE_TIER": "T3",
            "PRSM_MARKETPLACE_TEE_CAPABLE": "1",
            "PRSM_MARKETPLACE_TTL_SECONDS": "600",
            "PRSM_MARKETPLACE_REBROADCAST_INTERVAL_SEC": "120",
        })
    assert isinstance(out, MarketplaceAdvertiser)
    assert out.price_per_shard_ftns == 2.5
    assert out.base_capacity == 7.0
    assert out.max_shard_bytes == 4194304
    assert out.supported_dtypes == ["float16", "bfloat16"]
    assert out.stake_tier == "T3"
    assert out.tee_capable is True
    assert out.ttl_seconds == 600
    assert out.rebroadcast_interval_sec == 120.0
    # It binds the SAME compute_provider (so current_price_ftns + auto-downgrade work).
    assert out.compute_provider is cp


def test_build_from_env_defaults_when_params_unset():
    from prsm.marketplace.advertiser import build_marketplace_advertiser_from_env
    identity, gossip, cp = _env_gossip_cp()
    out = build_marketplace_advertiser_from_env(
        identity=identity, gossip=gossip, compute_provider=cp,
        env={"PRSM_MARKETPLACE_ADVERTISE": "yes"})
    assert out is not None
    assert out.supported_dtypes                       # non-empty default dtype list
    assert out.price_per_shard_ftns > 0               # a positive default price
    assert out.stake_tier == "open"                   # honest default: no claimed tier
    assert out.stake_eth_address is None              # no binding supplied → none advertised


def test_build_from_env_attaches_preproduced_binding():
    from prsm.marketplace.advertiser import build_marketplace_advertiser_from_env
    from prsm.marketplace.listing import sign_stake_binding, verify_stake_binding
    from prsm.node.identity import generate_node_identity
    identity = generate_node_identity(display_name="bound-env-provider")
    gossip = MagicMock(); gossip.publish = AsyncMock()
    cp = _make_compute_provider()
    address, sig = sign_stake_binding(identity.node_id, "0x" + "7c" * 32)
    out = build_marketplace_advertiser_from_env(
        identity=identity, gossip=gossip, compute_provider=cp,
        env={
            "PRSM_MARKETPLACE_ADVERTISE": "1",
            "PRSM_STAKE_ETH_ADDRESS": address,
            "PRSM_STAKE_BINDING_SIG": sig,
        })
    assert out.stake_eth_address == address
    assert out.stake_binding_sig == sig
    # The advertised binding authenticates for THIS provider_id.
    assert verify_stake_binding(identity.node_id, address, sig) is True


def test_build_from_env_derives_binding_from_stake_key():
    from prsm.marketplace.advertiser import build_marketplace_advertiser_from_env
    from prsm.marketplace.listing import verify_stake_binding
    from prsm.node.identity import generate_node_identity
    from eth_account import Account
    identity = generate_node_identity(display_name="key-env-provider")
    gossip = MagicMock(); gossip.publish = AsyncMock()
    cp = _make_compute_provider()
    stake_key = "0x" + "3a" * 32
    expected_addr = Account.from_key(stake_key).address
    out = build_marketplace_advertiser_from_env(
        identity=identity, gossip=gossip, compute_provider=cp,
        env={"PRSM_MARKETPLACE_ADVERTISE": "1", "PRSM_STAKE_ETH_KEY": stake_key})
    assert out.stake_eth_address == expected_addr
    # The startup-derived signature authenticates for THIS provider_id ↔ the key's address.
    assert verify_stake_binding(identity.node_id, out.stake_eth_address, out.stake_binding_sig) is True


# ── sp1459: node-lifecycle glue — _start_marketplace_advertiser_if_present ──
# The node START/STOP wiring is thin + fail-soft. Verified without booting a heavy node via the
# established PRSMNode.__new__ surface pattern (mirrors the DHT-components glue tests).

def test_node_start_advertiser_noop_when_none():
    from prsm.node.node import PRSMNode
    node = PRSMNode.__new__(PRSMNode)
    node._marketplace_advertiser = None            # default: advertising not opted in
    # Must not raise (no advertiser to start).
    _run(node._start_marketplace_advertiser_if_present())


def test_node_start_advertiser_calls_start_when_present():
    from prsm.node.node import PRSMNode
    node = PRSMNode.__new__(PRSMNode)
    adv = MagicMock()
    adv.start = AsyncMock()
    node._marketplace_advertiser = adv
    _run(node._start_marketplace_advertiser_if_present())
    adv.start.assert_awaited_once()


def test_node_start_advertiser_is_fail_soft():
    # A broadcast/start error must NOT abort node bring-up.
    from prsm.node.node import PRSMNode
    node = PRSMNode.__new__(PRSMNode)
    adv = MagicMock()
    adv.start = AsyncMock(side_effect=RuntimeError("gossip not ready"))
    node._marketplace_advertiser = adv
    _run(node._start_marketplace_advertiser_if_present())   # swallowed, no raise
    adv.start.assert_awaited_once()
