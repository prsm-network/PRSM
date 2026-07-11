"""Sprint 1422 — actually LOOK for batches drawn against my escrow.

sp1146 built the matcher. sp1147 built the challenge assembly. sp1172 built the issued-auth
store. sp1421 finally populated that store. But NOTHING ever went and looked on chain for
batches committed against this requester — `match_unauthorized_batches` takes a list of
committed batches, and no production code ever produced one. So the whole defense, however
well-built, could never fire.

It cannot come from the node's own PublishedBatchStore (that holds batches THIS node published);
an attacker's forged batch exists only on chain. And `event BatchCommitted(bytes32 indexed
batchId, address indexed provider, ...)` does not carry the requester AT ALL — so a victim cannot
filter for "batches against me" and must enumerate every batch and read each requester from
storage. That on-chain gap is why this scan is O(batches).

`scan_for_unauthorized_batches` closes the loop. These tests use injected readers, so they run
fully offline — no RPC, no chain.
"""
from __future__ import annotations

import logging
import time

import pytest
from eth_account import Account

from prsm.settlement.issued_authorization_store import (
    IssuedAuthorizationStore,
    build_and_record_payment_authorization,
)
from prsm.settlement.no_escrow_assembler import scan_for_unauthorized_batches

HONEST = "0x000000000000000000000000000000000000BEEF"
ATTACKER = "0x000000000000000000000000000000000000DEAD"

PENDING, FINALIZED = 1, 2


class _Observed:
    """Mirrors settlement_audit_engine.ObservedBatch (what the real enumerator returns)."""

    def __init__(self, batch_id: bytes, requester_address: str):
        self.batch_id = batch_id
        self.requester_address = requester_address


@pytest.fixture
def victim():
    return Account.create()


@pytest.fixture
def store(tmp_path):
    return IssuedAuthorizationStore(tmp_path / "issued.json")


def _authorize(store, victim, *, provider=HONEST, max_spend=5.0):
    """The victim genuinely pays `provider` — recorded, as sp1421 now does on every payment."""
    return build_and_record_payment_authorization(
        store=store,
        requester_key=victim.key.hex(),
        provider_address=provider,
        model_id="gpt2",
        prompt="hello",
        max_tokens=0,
        privacy_tier="none",
        content_tier="none",
        max_spend_ftns=max_spend,
        expiry_unix=int(time.time()) + 300,
        chain_id=8453,
    )


def _readers(batches):
    """batches: {batch_id: dict(requester, provider, total_value_ftns, commit_timestamp, status)}"""
    def enumerate_batches(_from, _to):
        return [_Observed(bid, b["requester"]) for bid, b in batches.items()]

    def read_batch(bid):
        return batches[bytes(bid)]

    return enumerate_batches, read_batch


def _batch(requester, provider, *, value=int(2e18), status=PENDING):
    return {
        "requester": requester, "provider": provider,
        "total_value_ftns": value, "commit_timestamp": int(time.time()),
        "status": status,
    }


class TestTheDrainIsDetected:
    def test_a_forged_batch_against_my_escrow_is_flagged(self, store, victim):
        """The attack: an address I never paid commits a batch naming ME as requester."""
        forged = b"\xaa" * 32
        enum_b, read_b = _readers({forged: _batch(victim.address, ATTACKER)})

        alerts = scan_for_unauthorized_batches(
            enumerate_batches=enum_b, read_batch=read_b, store=store,
            my_address=victim.address, from_block=0, to_block=99,
        )

        assert len(alerts) == 1, "the escrow-drain attempt went undetected"
        a = alerts[0]
        assert a.classification.provider.lower() == ATTACKER.lower()
        assert a.challengeable is True, "PENDING batch — the funds are still savable"
        assert a.already_drained is False

    def test_my_own_authorized_batch_is_not_flagged(self, store, victim):
        """A false NO_ESCROW would deny an HONEST provider payment. Must never happen."""
        _authorize(store, victim, provider=HONEST, max_spend=5.0)
        legit = b"\xbb" * 32
        enum_b, read_b = _readers({legit: _batch(victim.address, HONEST)})

        assert scan_for_unauthorized_batches(
            enumerate_batches=enum_b, read_batch=read_b, store=store,
            my_address=victim.address, from_block=0, to_block=99,
        ) == [], "flagged a batch the requester genuinely authorized — this would grief an honest provider"

    def test_forged_and_legit_together_only_the_forged_is_flagged(self, store, victim):
        _authorize(store, victim, provider=HONEST, max_spend=5.0)
        legit, forged = b"\xbb" * 32, b"\xaa" * 32
        enum_b, read_b = _readers({
            legit: _batch(victim.address, HONEST),
            forged: _batch(victim.address, ATTACKER),
        })

        alerts = scan_for_unauthorized_batches(
            enumerate_batches=enum_b, read_batch=read_b, store=store,
            my_address=victim.address, from_block=0, to_block=99,
        )
        assert [bytes(a.classification.batch_id) for a in alerts] == [forged]

    def test_batches_against_other_requesters_are_ignored(self, store, victim):
        """Only the requester may raise NO_ESCROW (_handleNoEscrow: msg.sender == b.requester)."""
        other = Account.create().address
        enum_b, read_b = _readers({b"\xcc" * 32: _batch(other, ATTACKER)})

        assert scan_for_unauthorized_batches(
            enumerate_batches=enum_b, read_batch=read_b, store=store,
            my_address=victim.address, from_block=0, to_block=99,
        ) == []

    def test_an_already_finalized_drain_is_reported_as_a_post_mortem(self, store, victim):
        """Too late to challenge — but the victim must still be TOLD they were robbed."""
        drained = b"\xdd" * 32
        enum_b, read_b = _readers({
            drained: _batch(victim.address, ATTACKER, status=FINALIZED),
        })

        [a] = scan_for_unauthorized_batches(
            enumerate_batches=enum_b, read_batch=read_b, store=store,
            my_address=victim.address, from_block=0, to_block=99,
        )
        assert a.already_drained is True
        assert a.challengeable is False

    def test_savable_batches_are_reported_before_lost_ones(self, store, victim):
        lost, savable = b"\xdd" * 32, b"\xaa" * 32
        enum_b, read_b = _readers({
            lost: _batch(victim.address, ATTACKER, status=FINALIZED),
            savable: _batch(victim.address, ATTACKER, status=PENDING),
        })
        alerts = scan_for_unauthorized_batches(
            enumerate_batches=enum_b, read_batch=read_b, store=store,
            my_address=victim.address, from_block=0, to_block=99,
        )
        assert [bytes(a.classification.batch_id) for a in alerts] == [savable, lost], (
            "the still-challengeable drain must be surfaced first — it is the one with a deadline"
        )

    def test_the_alert_is_logged_at_critical(self, store, victim, caplog):
        """An operator must not have to read a return value to learn they are being robbed."""
        enum_b, read_b = _readers({b"\xaa" * 32: _batch(victim.address, ATTACKER)})
        with caplog.at_level(logging.CRITICAL):
            scan_for_unauthorized_batches(
                enumerate_batches=enum_b, read_batch=read_b, store=store,
                my_address=victim.address, from_block=0, to_block=99,
            )
        assert any(r.levelno >= logging.CRITICAL for r in caplog.records)


class TestTheScanFailsSafely:
    def test_a_dead_rpc_does_not_crash_the_audit_loop(self, store, victim, caplog):
        def boom(_f, _t):
            raise ConnectionError("rpc down")

        with caplog.at_level(logging.ERROR):
            assert scan_for_unauthorized_batches(
                enumerate_batches=boom, read_batch=lambda b: {}, store=store,
                my_address=victim.address, from_block=0, to_block=99,
            ) == []
        assert any("BLIND" in r.message or "blind" in r.message.lower() for r in caplog.records), (
            "a failed scan must say loudly that the requester is now undefended, not fail silently"
        )

    def test_one_unreadable_batch_does_not_blind_the_whole_scan(self, store, victim):
        good, bad = b"\xaa" * 32, b"\xee" * 32
        batches = {good: _batch(victim.address, ATTACKER), bad: _batch(victim.address, ATTACKER)}

        def enum_b(_f, _t):
            return [_Observed(bid, b["requester"]) for bid, b in batches.items()]

        def read_b(bid):
            if bytes(bid) == bad:
                raise TimeoutError("rpc hiccup")
            return batches[bytes(bid)]

        alerts = scan_for_unauthorized_batches(
            enumerate_batches=enum_b, read_batch=read_b, store=store,
            my_address=victim.address, from_block=0, to_block=99,
        )
        assert [bytes(a.classification.batch_id) for a in alerts] == [good], (
            "one flaky read swallowed the other, real, detectable drain"
        )


class TestTheWatchIsActuallyWiredIntoTheNode:
    """A scanner nothing calls is exactly the bug this sprint exists to fix.

    sp1146/1147/1172 all shipped correct, well-tested code that no production path ever
    invoked. So pin the WIRING, not just the logic.
    """

    def test_the_node_launches_the_watch_and_drains_it_on_shutdown(self):
        import inspect

        from prsm.node import node as node_mod

        src = inspect.getsource(node_mod)

        assert "self._escrow_drain_watch_task = asyncio.create_task(" in src, (
            "the escrow-drain watch is never launched — the scanner would be dead code, which "
            "is precisely how the NO_ESCROW defense died in the first place"
        )
        assert "async def _escrow_drain_watch_loop" in src
        assert '"_escrow_drain_watch_task",  # sp1422' in src, (
            "the watch task is not in the shutdown drain list — it would leak on stop"
        )
        assert hasattr(node_mod.PRSMNode, "_escrow_drain_watch_loop")

    def test_the_loop_calls_the_scanner(self):
        import inspect

        from prsm.node.node import PRSMNode

        body = inspect.getsource(PRSMNode._escrow_drain_watch_loop)
        assert "scan_for_unauthorized_batches" in body
        # It must NEVER auto-submit: raising NO_ESCROW is a signed, user-gated action.
        for forbidden in ("submit_challenge(", "challengeReceipt(", "broadcast("):
            assert forbidden not in body, (
                f"the watch must never {forbidden} — challenge submission is user-gated"
            )
