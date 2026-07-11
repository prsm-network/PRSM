"""Sprint 1424 — on-chain slashes must actually penalize provider reputation.

Before this, ReputationTracker was read on every marketplace dispatch and written NEVER on the
live path: record_success/record_failure only fire from the never-instantiated
MarketplaceOrchestrator, and record_slash had zero production callers. So score_for() returned a
constant 0.5 for everyone forever, and a provider slashed on-chain for double-spending kept full
selection weight while /marketplace/reputation reported it clean.

These tests use a real ReputationTracker and injected slash events + an injected node_id resolver,
so they run fully offline (no chain).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from prsm.marketplace.reputation import ReputationTracker
from prsm.marketplace.slash_reputation_bridge import (
    apply_onchain_slashes_to_reputation,
    slash_dedup_key,
)

# node_id keyspace (sha256(pubkey)[:32]) vs operator eth-address keyspace — the whole point.
NODE_A = "a" * 32
OP_A = "0x000000000000000000000000000000000000000A"
NODE_B = "b" * 32
OP_B = "0x000000000000000000000000000000000000000B"


@dataclass
class _Slash:
    """Mirrors stake_manager.SlashEvent (only the fields the bridge reads)."""
    provider: str          # operator eth address
    reason_id: str         # == batchId on-chain
    slash_amount_wei: int
    tx_hash: str


def _resolver(mapping):
    return lambda op: mapping.get(str(op))


class TestSlashesReachTheScore:
    def test_a_slashed_provider_loses_reputation_and_is_flagged(self):
        tracker = ReputationTracker()
        assert tracker.score_for(NODE_A) == 0.5, "precondition: neutral before any slash"
        assert tracker.has_been_slashed(NODE_A) is False

        seen = set()
        recorded, unmapped = apply_onchain_slashes_to_reputation(
            events=[_Slash(OP_A, "0x" + "11" * 32, 10**18, "0x" + "aa" * 32)],
            tracker=tracker,
            resolve_node_id=_resolver({OP_A: NODE_A}),
            already_recorded=seen,
        )

        assert (recorded, unmapped) == (1, 0)
        assert tracker.has_been_slashed(NODE_A) is True, (
            "a provider slashed on-chain is still reported clean — it keeps full selection weight"
        )
        assert tracker.score_for(NODE_A) < 0.5, (
            "the slash did not lower the reputation score; selection weighting is unaffected"
        )

    def test_the_slash_is_recorded_under_the_node_id_not_the_eth_address(self):
        """The keyspace bridge is the crux: score_for is called with node_id, so the slash must
        land under node_id, not the operator address the event carries."""
        tracker = ReputationTracker()
        apply_onchain_slashes_to_reputation(
            events=[_Slash(OP_A, "0x" + "11" * 32, 10**18, "0x" + "aa" * 32)],
            tracker=tracker,
            resolve_node_id=_resolver({OP_A: NODE_A}),
            already_recorded=set(),
        )
        assert tracker.has_been_slashed(NODE_A) is True
        # The raw eth address must NOT be a key — that would never be consulted by score_for.
        assert tracker.has_been_slashed(OP_A) is False


class TestIdempotency:
    def test_rescanning_the_same_slash_does_not_double_count(self):
        tracker = ReputationTracker()
        ev = _Slash(OP_A, "0x" + "11" * 32, 10**18, "0x" + "aa" * 32)
        seen = set()
        r1, _ = apply_onchain_slashes_to_reputation(
            events=[ev], tracker=tracker, resolve_node_id=_resolver({OP_A: NODE_A}),
            already_recorded=seen,
        )
        r2, _ = apply_onchain_slashes_to_reputation(
            events=[ev], tracker=tracker, resolve_node_id=_resolver({OP_A: NODE_A}),
            already_recorded=seen,  # SAME set — simulates the next scan of overlapping windows
        )
        assert (r1, r2) == (1, 0), "the second scan re-recorded a slash it had already applied"
        assert tracker.slashed_count(NODE_A) == 1, "double-counted the same on-chain slash"

    def test_dedup_key_is_tx_and_batch(self):
        ev = _Slash(OP_A, "0x11", 5, "0xABCDEF")
        assert slash_dedup_key(ev) == ("0xabcdef", "0x11")


class TestUnmappableSlashes:
    def test_an_unmappable_slash_is_counted_and_loudly_logged_not_dropped(self, caplog):
        tracker = ReputationTracker()
        with caplog.at_level(logging.WARNING):
            recorded, unmapped = apply_onchain_slashes_to_reputation(
                events=[_Slash(OP_B, "0x" + "22" * 32, 10**18, "0x" + "bb" * 32)],
                tracker=tracker,
                resolve_node_id=_resolver({}),  # OP_B not known to discovery
                already_recorded=set(),
            )
        assert (recorded, unmapped) == (0, 1)
        assert any("could NOT be mapped" in r.message for r in caplog.records), (
            "an unmappable slash must warn loudly — a slashed provider is out there unaccounted for"
        )

    def test_mixed_batch_records_the_mappable_and_counts_the_rest(self):
        tracker = ReputationTracker()
        recorded, unmapped = apply_onchain_slashes_to_reputation(
            events=[
                _Slash(OP_A, "0x" + "11" * 32, 10**18, "0x" + "aa" * 32),
                _Slash(OP_B, "0x" + "22" * 32, 10**18, "0x" + "bb" * 32),
            ],
            tracker=tracker,
            resolve_node_id=_resolver({OP_A: NODE_A}),  # only A is known
            already_recorded=set(),
        )
        assert (recorded, unmapped) == (1, 1)
        assert tracker.has_been_slashed(NODE_A) is True
        assert tracker.has_been_slashed(NODE_B) is False


class TestFailSafe:
    def test_one_bad_event_does_not_abort_the_batch(self):
        tracker = ReputationTracker()

        class _Bad:
            @property
            def provider(self):
                raise RuntimeError("corrupt log")

        good = _Slash(OP_A, "0x" + "11" * 32, 10**18, "0x" + "aa" * 32)
        recorded, _ = apply_onchain_slashes_to_reputation(
            events=[_Bad(), good],
            tracker=tracker,
            resolve_node_id=_resolver({OP_A: NODE_A}),
            already_recorded=set(),
        )
        assert recorded == 1, "a single corrupt event swallowed a real, recordable slash"
        assert tracker.has_been_slashed(NODE_A) is True

    def test_a_throwing_resolver_is_treated_as_unmapped_not_a_crash(self):
        tracker = ReputationTracker()

        def _boom(_op):
            raise ConnectionError("discovery down")

        recorded, unmapped = apply_onchain_slashes_to_reputation(
            events=[_Slash(OP_A, "0x" + "11" * 32, 10**18, "0x" + "aa" * 32)],
            tracker=tracker,
            resolve_node_id=_boom,
            already_recorded=set(),
        )
        assert (recorded, unmapped) == (0, 1)


class TestTheWatchIsWiredIntoTheNode:
    """A bridge nothing calls is exactly the inert-defense bug this closes. Pin the wiring."""

    def test_node_launches_the_slash_watch_and_drains_it(self):
        import inspect

        from prsm.node import node as node_mod

        src = inspect.getsource(node_mod)
        assert "self._reputation_slash_watch_task = asyncio.create_task(" in src, (
            "the slash-watch is never launched — the bridge would be dead code, exactly how the "
            "reputation defense was inert in the first place"
        )
        assert "async def _reputation_slash_watch_loop" in src
        assert '"_reputation_slash_watch_task",  # sp1424' in src, (
            "the watch task is not in the shutdown drain list — it would leak on stop"
        )
        assert hasattr(node_mod.PRSMNode, "_reputation_slash_watch_loop")

    def test_the_loop_reads_chain_and_records_but_never_signs(self):
        import inspect

        from prsm.node.node import PRSMNode

        body = inspect.getsource(PRSMNode._reputation_slash_watch_loop)
        assert "apply_onchain_slashes_to_reputation" in body
        assert "get_all_slash_events" in body
        # It must be READ-ONLY on chain — a reputation watch never sends a transaction.
        for forbidden in ("_sign_send_wait", ".slash(", "commitBatch", "send_transaction", "broadcast("):
            assert forbidden not in body, (
                f"the slash-watch must never {forbidden} — it is read-only chain observability"
            )
