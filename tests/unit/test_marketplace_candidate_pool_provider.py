"""MarketplaceCandidatePoolProvider — bridges existing primitives
(MarketplaceDirectory + ReputationTracker) to the QueryOrchestrator's
`candidate_pool_provider` callable contract.

Per the QueryOrchestrator wiring-readiness assessment, the
QueryOrchestrator class needs a `Callable[[], tuple[StakedNode, ...]]`
that returns the current T2+ stake pool snapshot. This module wires
that callable from the existing marketplace gossip directory.

v1 limitations (documented inline in the impl):
  - stake_amount_ftns is derived from the tier label, not from a
    per-listing on-chain stake_of read (avoiding N RPC calls per
    selection)
  - has_tee defaults to False — listings don't carry that marker yet;
    Tier C queries will find no candidates until ProviderListing is
    extended (separate follow-on)
"""
from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass

import pytest

from prsm.compute.query_orchestrator import StakedNode
from prsm.compute.query_orchestrator.marketplace_candidate_pool_provider import (
    DEFAULT_STAKE_PER_TIER,
    MarketplaceCandidatePoolProvider,
)
from prsm.marketplace.listing import ProviderListing
from prsm.marketplace.reputation import ReputationTracker


# ──────────────────────────────────────────────────────────────────────
# Stubs
# ──────────────────────────────────────────────────────────────────────


def _make_listing(
    *,
    provider_id: str,
    pubkey_b64: str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    stake_tier: str = "T2",
    tee_capable: bool = False,
) -> ProviderListing:
    """Build a ProviderListing for tests."""
    return ProviderListing(
        listing_id="listing-x",
        provider_id=provider_id,
        provider_pubkey_b64=pubkey_b64,
        capacity_shards_per_sec=1.0,
        max_shard_bytes=1024 * 1024,
        supported_dtypes=["int64"],
        price_per_shard_ftns=0.001,
        tee_capable=tee_capable,
        stake_tier=stake_tier,
        advertised_at_unix=int(time.time()),
        ttl_seconds=300,
        signature="AAAA",
    )


@dataclass
class _StubDirectory:
    canned_listings: list

    def list_active_providers(self, at_unix=None):
        return list(self.canned_listings)


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_returns_tuple_of_staked_nodes(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="provider-1", stake_tier="T2"),
            _make_listing(provider_id="provider-2", stake_tier="T3"),
        ])
        reputation = ReputationTracker()
        provider = MarketplaceCandidatePoolProvider(
            directory=directory,
            reputation=reputation,
        )
        pool = provider()
        assert isinstance(pool, tuple)
        assert len(pool) == 2
        for node in pool:
            assert isinstance(node, StakedNode)

    def test_t1_filtered_out(self):
        # T1 is below the aggregator-eligibility threshold (Vision §6).
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="t1-only", stake_tier="T1"),
            _make_listing(provider_id="t2-eligible", stake_tier="T2"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory,
            reputation=ReputationTracker(),
        )
        pool = provider()
        ids = {n.node_id for n in pool}
        assert "t2-eligible" in ids
        assert "t1-only" not in ids

    def test_unknown_tier_filtered_out(self):
        # Defensive: a malformed listing shouldn't sneak past the gate.
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="bad", stake_tier="T99"),
            _make_listing(provider_id="good", stake_tier="T4"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory,
            reputation=ReputationTracker(),
        )
        pool = provider()
        assert {n.node_id for n in pool} == {"good"}


# ──────────────────────────────────────────────────────────────────────
# Field threading
# ──────────────────────────────────────────────────────────────────────


class TestFieldThreading:
    def test_stake_amount_derived_from_tier(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id=f"p-{tier}", stake_tier=tier)
            for tier in ["T2", "T3", "T4"]
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=ReputationTracker(),
        )
        pool = provider()
        per_id = {n.node_id: n for n in pool}
        # T4 should outweigh T3 should outweigh T2 — derived from tier.
        assert (
            per_id["p-T2"].stake_amount_ftns
            < per_id["p-T3"].stake_amount_ftns
            < per_id["p-T4"].stake_amount_ftns
        )

    def test_pubkey_hash_is_sha256_of_decoded_pubkey(self):
        # 32 zero bytes encoded as base64 — easy to verify.
        zero_pub_b64 = base64.b64encode(b"\x00" * 32).decode("ascii")
        directory = _StubDirectory(canned_listings=[
            _make_listing(
                provider_id="p-1", pubkey_b64=zero_pub_b64, stake_tier="T2",
            ),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=ReputationTracker(),
        )
        [node] = provider()
        assert node.pubkey_hash == hashlib.sha256(b"\x00" * 32).digest()
        assert len(node.pubkey_hash) == 32

    def test_reputation_score_threaded(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="rep-test", stake_tier="T2"),
        ])
        reputation = ReputationTracker()
        # Push a slash so score drops.
        reputation.record_slash(
            provider_id="rep-test",
            batch_id="batch-1",
            slash_amount_wei=1,
            reason="DOUBLE_SPEND",
        )
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=reputation,
        )
        [node] = provider()
        # Slashed → score below NEUTRAL_SCORE.
        assert node.reputation_score < ReputationTracker.NEUTRAL_SCORE

    def test_unknown_provider_gets_neutral_reputation(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="never-tracked", stake_tier="T2"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=ReputationTracker(),
        )
        [node] = provider()
        assert node.reputation_score == ReputationTracker.NEUTRAL_SCORE


# ──────────────────────────────────────────────────────────────────────
# Stake-per-tier override
# ──────────────────────────────────────────────────────────────────────


class TestStakePerTierOverride:
    def test_default_table_has_t2_t3_t4(self):
        # Pin the default table — Foundation council can ratify
        # different values via constructor override.
        assert "T2" in DEFAULT_STAKE_PER_TIER
        assert "T3" in DEFAULT_STAKE_PER_TIER
        assert "T4" in DEFAULT_STAKE_PER_TIER
        assert (
            DEFAULT_STAKE_PER_TIER["T2"]
            < DEFAULT_STAKE_PER_TIER["T3"]
            < DEFAULT_STAKE_PER_TIER["T4"]
        )

    def test_override_threads_through(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="p", stake_tier="T2"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory,
            reputation=ReputationTracker(),
            stake_amount_per_tier={"T2": 999, "T3": 1000, "T4": 1001},
        )
        [node] = provider()
        assert node.stake_amount_ftns == 999


# ──────────────────────────────────────────────────────────────────────
# Empty pool / degenerate
# ──────────────────────────────────────────────────────────────────────


class TestDegenerate:
    def test_empty_directory_returns_empty_tuple(self):
        provider = MarketplaceCandidatePoolProvider(
            directory=_StubDirectory(canned_listings=[]),
            reputation=ReputationTracker(),
        )
        pool = provider()
        assert pool == ()

    def test_malformed_pubkey_skipped_silently(self):
        # Invalid base64 should not blow the whole call — just drop
        # the offending listing.
        directory = _StubDirectory(canned_listings=[
            _make_listing(
                provider_id="malformed",
                pubkey_b64="!!!not-base64!!!",
                stake_tier="T2",
            ),
            _make_listing(provider_id="good", stake_tier="T2"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=ReputationTracker(),
        )
        pool = provider()
        assert {n.node_id for n in pool} == {"good"}


# ──────────────────────────────────────────────────────────────────────
# Callable contract pin
# ──────────────────────────────────────────────────────────────────────


class TestCallableContract:
    """The QueryOrchestrator class accepts
    `candidate_pool_provider: Callable[[], tuple[StakedNode, ...]]`.
    Pin that the adapter satisfies it."""

    def test_provider_is_callable(self):
        provider = MarketplaceCandidatePoolProvider(
            directory=_StubDirectory(canned_listings=[]),
            reputation=ReputationTracker(),
        )
        assert callable(provider)

    def test_provider_called_with_no_args_returns_tuple(self):
        provider = MarketplaceCandidatePoolProvider(
            directory=_StubDirectory(canned_listings=[]),
            reputation=ReputationTracker(),
        )
        result = provider()
        assert isinstance(result, tuple)


# ── sp1457: verified on-chain-stake binding weighting (closes the self-declared-tier exploit) ──

from eth_account import Account as _Account
from eth_account.messages import encode_defunct as _encode_defunct

from prsm.marketplace.listing import build_stake_binding_message as _bind_msg

_ETHK = "0x" + "3c" * 32


def _bound_listing(provider_id, *, stake_tier="T4", eth_key=_ETHK):
    acct = _Account.from_key(eth_key)
    sig = acct.sign_message(
        _encode_defunct(text=_bind_msg(provider_id, acct.address))).signature.hex()
    base = _make_listing(provider_id=provider_id, stake_tier=stake_tier)
    listing = ProviderListing(**{**base.to_dict(),
                                 "stake_eth_address": acct.address,
                                 "stake_binding_sig": sig})
    return listing, acct.address


def test_verified_binding_uses_real_onchain_stake_not_tier_label():
    # ★ THE FIX: a listing claims T4 (tier label 25000) but its verified on-chain stake is only 100.
    pid = "a" * 32
    listing, addr = _bound_listing(pid, stake_tier="T4")
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([listing]), reputation=ReputationTracker(),
        stake_reader=lambda a: 100 if a == addr else None)
    (node,) = pool()
    assert node.stake_amount_ftns == 100                 # REAL stake, not 25000 → exploit closed


def test_legacy_listing_without_reader_keeps_tier_label():
    pid = "b" * 32
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([_make_listing(provider_id=pid, stake_tier="T4")]),
        reputation=ReputationTracker())
    (node,) = pool()
    assert node.stake_amount_ftns == DEFAULT_STAKE_PER_TIER["T4"]   # backward-compat


def test_unbound_listing_with_reader_falls_back_to_tier_in_default_mode():
    pid = "c" * 32
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([_make_listing(provider_id=pid, stake_tier="T3")]),
        reputation=ReputationTracker(), stake_reader=lambda a: 999)
    (node,) = pool()
    assert node.stake_amount_ftns == DEFAULT_STAKE_PER_TIER["T3"]   # no binding → tier label


def test_require_binding_excludes_unbound_listings():
    pid = "d" * 32
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([_make_listing(provider_id=pid, stake_tier="T4")]),
        reputation=ReputationTracker(), stake_reader=lambda a: 100,
        require_stake_binding=True)
    assert pool() == ()                                   # unverifiable stake → excluded


def test_require_binding_includes_bound_listing_with_real_stake():
    pid = "e" * 32
    listing, addr = _bound_listing(pid, stake_tier="T4")
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([listing]), reputation=ReputationTracker(),
        stake_reader=lambda a: 50000 if a == addr else None, require_stake_binding=True)
    (node,) = pool()
    assert node.stake_amount_ftns == 50000


def test_forged_binding_is_not_honored():
    # Listing claims a binding to an address it cannot sign for → not verified.
    pid = "f" * 32
    base = _make_listing(provider_id=pid, stake_tier="T4")
    wrong_addr = _Account.from_key("0x" + "99" * 32).address
    real_sig = _Account.from_key(_ETHK).sign_message(
        _encode_defunct(text=_bind_msg(pid, _Account.from_key(_ETHK).address))).signature.hex()
    forged = ProviderListing(**{**base.to_dict(),
                                "stake_eth_address": wrong_addr,   # not the signer of real_sig
                                "stake_binding_sig": real_sig})
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([forged]), reputation=ReputationTracker(),
        stake_reader=lambda a: 25000)
    (node,) = pool()
    assert node.stake_amount_ftns == DEFAULT_STAKE_PER_TIER["T4"]   # forged binding ignored
    strict = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([forged]), reputation=ReputationTracker(),
        stake_reader=lambda a: 25000, require_stake_binding=True)
    assert strict() == ()                                 # and excluded under enforcement


# ── sp1457: make_ftns_stake_reader (wei → whole FTNS adapter over OnChainStakeReader) ──

from prsm.compute.query_orchestrator.marketplace_candidate_pool_provider import (
    make_ftns_stake_reader,
)


class _FakeOnChainReader:
    def __init__(self, wei_by_addr):
        self._m = wei_by_addr

    def stake_amount_for(self, address):
        if address == "__boom__":
            raise RuntimeError("rpc down")
        return self._m.get(address, 0)


def test_make_ftns_stake_reader_converts_wei_to_whole_ftns():
    reader = make_ftns_stake_reader(_FakeOnChainReader({"0xabc": 5000 * 10**18}))
    assert reader("0xabc") == 5000        # wei → whole FTNS (matches tier scale)
    assert reader("0xunknown") == 0       # unstaked → 0 (verified-but-no-stake → 0 weight)


def test_make_ftns_stake_reader_returns_none_on_error():
    reader = make_ftns_stake_reader(_FakeOnChainReader({}))
    assert reader("__boom__") is None     # RPC error → None → fall back / exclude, never over-weight


def test_make_ftns_stake_reader_wires_into_pool_end_to_end():
    # A verified-bound listing claiming T4 (tier 25000) but only 300 FTNS on-chain → weight 300.
    pid = "a" * 32
    listing, addr = _bound_listing(pid, stake_tier="T4")
    reader = make_ftns_stake_reader(_FakeOnChainReader({addr: 300 * 10**18}))
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([listing]), reputation=ReputationTracker(),
        stake_reader=reader)
    (node,) = pool()
    assert node.stake_amount_ftns == 300


# ── sp1463 (capstone audit F2): shared-stake Sybil amplification ──
# The binding proves an eth address authorized a provider_id, but nothing enforced one-provider-per-
# address. A single bonded address R could back N distinct identities, each weighted at the FULL
# stake_of(R) → N·S selection mass for S stake (decisive in require_stake_binding mode). Fix: SPLIT
# R's real stake across the identities bound to it, so collective weight ∝ stake, counted once.

def test_shared_stake_address_splits_weight_across_clones():
    l1, addr = _bound_listing("clone-1", stake_tier="T4")
    l2, addr2 = _bound_listing("clone-2", stake_tier="T4")
    l3, addr3 = _bound_listing("clone-3", stake_tier="T4")   # same default eth key → same address
    assert addr == addr2 == addr3
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([l1, l2, l3]), reputation=ReputationTracker(),
        stake_reader=lambda a: 900 if a == addr else None)
    nodes = pool()
    assert len(nodes) == 3
    for n in nodes:
        assert n.stake_amount_ftns == 300                    # 900 // 3, NOT 900 → amplification closed
    assert sum(n.stake_amount_ftns for n in nodes) == 900    # collective == the real stake, once


def test_single_bound_provider_keeps_full_stake():
    listing, addr = _bound_listing("solo", stake_tier="T4")
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([listing]), reputation=ReputationTracker(),
        stake_reader=lambda a: 900 if a == addr else None)
    (node,) = pool()
    assert node.stake_amount_ftns == 900                     # N=1 → unaffected by the split


def test_distinct_stake_addresses_are_not_split_together():
    l1, a1 = _bound_listing("p1", stake_tier="T4", eth_key="0x" + "11" * 32)
    l2, a2 = _bound_listing("p2", stake_tier="T4", eth_key="0x" + "22" * 32)
    assert a1 != a2
    reader = lambda a: {a1: 900, a2: 500}.get(a)
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([l1, l2]), reputation=ReputationTracker(), stake_reader=reader)
    per = {n.node_id: n.stake_amount_ftns for n in pool()}
    assert per["p1"] == 900 and per["p2"] == 500             # different addresses → each full


def test_shared_stake_split_applies_under_require_binding():
    # Strict mode is where the amplification was decisive — confirm the split applies there too.
    l1, addr = _bound_listing("c1", stake_tier="T4")
    l2, _a2 = _bound_listing("c2", stake_tier="T4")
    pool = MarketplaceCandidatePoolProvider(
        directory=_StubDirectory([l1, l2]), reputation=ReputationTracker(),
        stake_reader=lambda a: 1000 if a == addr else None, require_stake_binding=True)
    nodes = pool()
    assert len(nodes) == 2
    assert all(n.stake_amount_ftns == 500 for n in nodes)    # 1000 // 2
