"""§4 step 6 — compute-participant settlement post-QO-aggregation.

Two surfaces tested:
  1. ``PaymentEscrow.release_escrow_split`` — multi-recipient atomic
     distribution. Closes the gap where the prompter's compute
     budget was previously paid entirely to the prompter's own node.
  2. ``/compute/forge`` integration — when the QueryOrchestrator
     returns ``participants`` on the AggregatedResult, the endpoint
     dispatches an escrow split (aggregator share + uniform per-
     participant compute share) instead of the legacy single-
     provider release.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import (
    EscrowAlreadyFinalizedError,
    EscrowEntry,
    EscrowStatus,
    PaymentEscrow,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _Tx:
    tx_id: str


class _StubLedger:
    """Tracks transfers for assertion. Always succeeds unless seeded
    with a per-recipient failure."""

    def __init__(self, escrow_balance: float = 100.0, fail_for=None):
        self._balance = escrow_balance
        self.transfers = []
        self._counter = 0
        self._fail_for = fail_for or set()

    async def get_balance(self, wallet: str) -> float:
        return self._balance

    async def transfer(
        self, from_wallet: str, to_wallet: str, amount: float,
        description: str,
    ):
        if to_wallet in self._fail_for:
            raise ValueError(f"simulated failure for {to_wallet}")
        self._counter += 1
        tx = _Tx(tx_id=f"tx-{self._counter}")
        self.transfers.append({
            "from": from_wallet,
            "to": to_wallet,
            "amount": amount,
            "tx_id": tx.tx_id,
            "description": description,
        })
        return tx


def _seed_escrow(
    escrow: PaymentEscrow, *, job_id="job-x", amount=100.0,
    requester_id="prompter-1",
) -> EscrowEntry:
    entry = EscrowEntry(
        escrow_id=f"esc-{job_id}",
        job_id=job_id,
        requester_id=requester_id,
        amount=amount,
        status=EscrowStatus.PENDING,
    )
    escrow._escrows[entry.escrow_id] = entry
    return entry


def _make_escrow(escrow_balance: float = 100.0, fail_for=None):
    ledger = _StubLedger(escrow_balance=escrow_balance, fail_for=fail_for)
    return PaymentEscrow(ledger=ledger, node_id="test-node"), ledger


# ──────────────────────────────────────────────────────────────────────
# PaymentEscrow.release_escrow_split
# ──────────────────────────────────────────────────────────────────────


class TestSplitHappyPath:
    def test_distributes_to_each_recipient(self):
        escrow, ledger = _make_escrow()
        _seed_escrow(escrow)
        splits = [("agg-node", 5.0), ("worker-a", 47.5), ("worker-b", 47.5)]
        txs = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=splits,
        ))
        assert txs is not None and len(txs) == 3
        # Each recipient got one transfer with the right amount.
        per_recipient = {t["to"]: t["amount"] for t in ledger.transfers if t["to"] != "prompter-1"}
        assert per_recipient == {"agg-node": 5.0, "worker-a": 47.5, "worker-b": 47.5}

    def test_remainder_refunded_to_requester(self):
        escrow, ledger = _make_escrow(escrow_balance=100.0)
        _seed_escrow(escrow, amount=100.0)
        splits = [("agg-node", 5.0), ("worker-a", 45.0)]  # only 50 of 100
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=splits,
        ))
        # Remainder transfer landed on the requester.
        refund = next(t for t in ledger.transfers if t["to"] == "prompter-1")
        assert refund["amount"] == pytest.approx(50.0)

    def test_no_remainder_when_total_equals_amount(self):
        escrow, ledger = _make_escrow()
        _seed_escrow(escrow, amount=100.0)
        splits = [("a", 60.0), ("b", 40.0)]
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=splits,
        ))
        # No transfer went to the requester.
        assert not any(t["to"] == "prompter-1" for t in ledger.transfers)

    def test_escrow_status_marked_released(self):
        escrow, _ = _make_escrow()
        entry = _seed_escrow(escrow)
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[("a", 10.0)],
        ))
        assert entry.status == EscrowStatus.RELEASED
        assert entry.completed_at is not None
        assert entry.tx_release is not None

    def test_metadata_records_split_breakdown(self):
        # An audit trail must reconstruct who got paid what without
        # log-replay. Pin the metadata shape.
        escrow, _ = _make_escrow()
        entry = _seed_escrow(escrow)
        splits = [("a", 30.0), ("b", 20.0)]
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=splits,
        ))
        recorded = entry.metadata.get("splits", [])
        assert len(recorded) == 2
        recipients = {entry["recipient"] for entry in recorded}
        assert recipients == {"a", "b"}
        # Each entry has tx_id linkage.
        for r in recorded:
            assert r["tx_id"] is not None

    def test_provider_winner_marked_split(self):
        # No single winner — provider_winner reflects this so audit
        # tooling doesn't misinterpret a split as single-recipient.
        escrow, _ = _make_escrow()
        entry = _seed_escrow(escrow)
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[("a", 10.0), ("b", 10.0)],
        ))
        assert entry.provider_winner == "split:2"


class TestSplitValidation:
    def test_sum_exceeds_escrow_amount_raises(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow, amount=10.0)
        with pytest.raises(ValueError, match="exceeds escrow amount"):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x",
                splits=[("a", 6.0), ("b", 5.0)],  # sum=11 > 10
            ))

    def test_empty_splits_returns_none(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        result = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[],
        ))
        assert result is None

    def test_negative_amount_rejected(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        with pytest.raises(ValueError, match="positive number"):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x", splits=[("a", -1.0)],
            ))

    def test_zero_amount_rejected(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        with pytest.raises(ValueError, match="positive number"):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x", splits=[("a", 0.0)],
            ))

    def test_empty_recipient_id_rejected(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x", splits=[("", 5.0)],
            ))


class TestSplitFloatTolerance:
    """sp997 — compute_split_amounts normalizes splits to sum to the budget, but
    float rounding can make sum land a few ULPs OVER the escrow balance (~8% of
    multi-recipient forge splits). The old guard `escrow_balance < total_split`
    had no tolerance (unlike the +1e-9 raise guard) so that float noise returned
    None → zero transfers, escrow stranded PENDING, providers paid 0 — while the
    forge handler still reported the job COMPLETED + 200."""

    def test_float_overshoot_within_tolerance_settles_not_strands(self):
        escrow, ledger = _make_escrow(escrow_balance=0.3)
        entry = _seed_escrow(escrow, amount=0.3)
        # 0.1+0.1+0.1 == 0.30000000000000004 in float — a real overshoot pattern.
        splits = [("a", 0.1), ("b", 0.1), ("c", 0.1)]
        assert sum(a for _, a in splits) > 0.3  # confirm the float overshoot
        txs = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=splits,
        ))
        assert txs is not None  # SETTLES (pre-sp997 this returned None → strand)
        assert entry.status == EscrowStatus.RELEASED
        paid = {
            t["to"]: t["amount"]
            for t in ledger.transfers if t["to"] != "prompter-1"
        }
        assert set(paid) == {"a", "b", "c"}  # every worker paid
        # The clamp shaved the float overshoot off the largest leg; the legs sum
        # to ~the balance (no over-debit, no spurious strand).
        assert sum(paid.values()) == pytest.approx(0.3, abs=1e-9)

    def test_real_shortfall_beyond_tolerance_still_strands(self):
        # Nominal escrow amount 10.0 (so splits summing to 10 pass the +1e-9 raise
        # guard), but the wallet is genuinely underfunded (balance 5.0 << 10).
        # A real shortfall (>> 1e-9) must still strand — never release more than held.
        escrow, ledger = _make_escrow(escrow_balance=5.0)
        entry = _seed_escrow(escrow, amount=10.0)
        result = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[("a", 6.0), ("b", 4.0)],  # sum 10.0 > balance 5.0
        ))
        assert result is None  # real shortfall → strands (correct)
        assert entry.status == EscrowStatus.PENDING
        assert not ledger.transfers  # released nothing


class TestSplitStateMachine:
    def test_double_release_is_no_op(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        first = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[("a", 10.0)],
        ))
        assert first is not None
        # Second call returns None (idempotent).
        second = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[("a", 10.0)],
        ))
        assert second is None

    def test_release_after_refund_raises(self):
        escrow, _ = _make_escrow()
        entry = _seed_escrow(escrow)
        # Manually set REFUNDED to simulate prior refund without
        # going through the full refund path.
        entry.status = EscrowStatus.REFUNDED
        with pytest.raises(EscrowAlreadyFinalizedError):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x", splits=[("a", 10.0)],
            ))

    def test_unknown_job_returns_none(self):
        escrow, _ = _make_escrow()
        # No seeded escrow.
        result = asyncio.run(escrow.release_escrow_split(
            job_id="missing", splits=[("a", 10.0)],
        ))
        assert result is None


class TestSplitFailureMode:
    def test_per_recipient_failure_keeps_escrow_pending(self):
        # If one recipient transfer fails partway, we want the
        # caller to see None + the escrow stay PENDING (so cleanup
        # can refund or operator can retry).
        escrow, ledger = _make_escrow(fail_for={"bad-worker"})
        entry = _seed_escrow(escrow)
        result = asyncio.run(escrow.release_escrow_split(
            job_id="job-x",
            splits=[("good-a", 10.0), ("bad-worker", 10.0)],
        ))
        assert result is None
        assert entry.status == EscrowStatus.PENDING


class TestSplitAtomicRollback:
    """v2 partial-failure recovery (shipped 2026-05-09): when one
    recipient transfer fails after others have succeeded, compensating
    reverse-transfers restore the escrow wallet to its pre-call state.
    Operator no longer has to manually reconcile leaked legs.
    """

    def test_successful_first_leg_compensated_on_second_leg_failure(self):
        """First transfer succeeds, second fails. Atomic-rollback
        issues a compensating transfer (recipient → escrow_wallet)
        to undo the first leg. Net effect on escrow_wallet balance
        is zero."""
        escrow, ledger = _make_escrow(fail_for={"bad-worker"})
        entry = _seed_escrow(escrow, amount=100.0)
        result = asyncio.run(escrow.release_escrow_split(
            job_id="job-x",
            splits=[("good-a", 30.0), ("bad-worker", 20.0)],
        ))
        assert result is None
        assert entry.status == EscrowStatus.PENDING
        # Compensating transfer: good-a → escrow_wallet for 30.0.
        compensations = [
            t for t in ledger.transfers
            if t["from"] == "good-a"
        ]
        assert len(compensations) == 1
        assert compensations[0]["to"].startswith("escrow-")
        assert compensations[0]["amount"] == pytest.approx(30.0)

    def test_compensation_records_in_metadata_for_audit(self):
        """The metadata must record what was rolled back so audit
        trails can reconstruct what happened, even though the
        escrow is reverted to PENDING."""
        escrow, ledger = _make_escrow(fail_for={"bad-worker"})
        entry = _seed_escrow(escrow, amount=100.0)
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x",
            splits=[("good-a", 30.0), ("bad-worker", 20.0)],
        ))
        # New metadata key: list of (recipient, amount) tuples that
        # were issued + then compensated.
        assert "rollback_history" in entry.metadata
        history = entry.metadata["rollback_history"]
        assert len(history) >= 1
        # Most recent rollback entry covers the failed split.
        last = history[-1]
        assert last["failed_recipient"] == "bad-worker"
        assert last["compensated"] == [
            {"recipient": "good-a", "amount": 30.0},
        ]

    def test_three_successful_then_one_fail_compensates_all_three(self):
        """When N-1 transfers succeed and the Nth fails, all N-1
        prior legs get compensating reverses."""
        escrow, ledger = _make_escrow(fail_for={"bad-worker"})
        entry = _seed_escrow(escrow, amount=100.0)
        result = asyncio.run(escrow.release_escrow_split(
            job_id="job-x",
            splits=[
                ("a", 10.0), ("b", 15.0), ("c", 20.0), ("bad-worker", 5.0),
            ],
        ))
        assert result is None
        assert entry.status == EscrowStatus.PENDING
        # Three compensating reverses, one per successful leg.
        compensations = {
            t["from"]: t["amount"] for t in ledger.transfers
            if t["to"].startswith("escrow-")
        }
        assert compensations == {"a": 10.0, "b": 15.0, "c": 20.0}

    def test_first_leg_failure_no_compensations_needed(self):
        """If the very first transfer fails, no compensating
        reverses are needed (no leaked legs). The escrow stays
        PENDING and rollback_history records empty compensated."""
        escrow, ledger = _make_escrow(fail_for={"bad-worker"})
        entry = _seed_escrow(escrow, amount=100.0)
        result = asyncio.run(escrow.release_escrow_split(
            job_id="job-x",
            splits=[("bad-worker", 10.0), ("good-a", 30.0)],
        ))
        assert result is None
        assert entry.status == EscrowStatus.PENDING
        # No compensating reverses; only the failed-first attempt.
        assert not any(
            t["to"].startswith("escrow-")
            for t in ledger.transfers
        )
        # rollback_history records the failed-first case explicitly.
        history = entry.metadata.get("rollback_history", [])
        assert len(history) == 1
        assert history[-1]["failed_recipient"] == "bad-worker"
        assert history[-1]["compensated"] == []

    def test_compensation_failure_records_unrecoverable_flag(self):
        """If compensating transfer ALSO fails (e.g., recipient
        wallet unreachable), escrow stays PENDING but metadata
        records `partial_release_unrecoverable: True` + the
        leaked-recipient list. Operator can manually reconcile
        only the leaked legs from the audit trail."""
        # good-a accepts the forward transfer but rejects the
        # compensating reverse (simulates wallet locked / RPC
        # error on the reverse direction).
        class _OneWayLedger(_StubLedger):
            def __init__(self, **kw):
                super().__init__(**kw)

            async def transfer(
                self, from_wallet, to_wallet, amount, description,
            ):
                # Reverse direction (recipient → escrow_wallet) for
                # good-a fails; everything else uses normal logic.
                if from_wallet == "good-a" and to_wallet.startswith(
                    "escrow-"
                ):
                    raise ValueError(
                        "compensating reverse failed for good-a"
                    )
                return await super().transfer(
                    from_wallet, to_wallet, amount, description,
                )

        ledger = _OneWayLedger(
            escrow_balance=100.0, fail_for={"bad-worker"},
        )
        escrow = PaymentEscrow(ledger=ledger, node_id="test-node")
        entry = _seed_escrow(escrow, amount=100.0)

        result = asyncio.run(escrow.release_escrow_split(
            job_id="job-x",
            splits=[("good-a", 30.0), ("bad-worker", 20.0)],
        ))
        assert result is None
        assert entry.status == EscrowStatus.PENDING
        # Unrecoverable flag set; leaked recipients listed.
        assert entry.metadata.get("partial_release_unrecoverable") is True
        leaked = entry.metadata.get("leaked_recipients", [])
        assert {"recipient": "good-a", "amount": 30.0} in leaked


# ──────────────────────────────────────────────────────────────────────
# /compute/forge integration — §4 step 6 settlement routing
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _FakeAggregatedResult:
    query_id: bytes
    payload: bytes
    aggregator_node_id: str
    contributing_shards: tuple
    participants: tuple


@dataclass
class _FakeParticipant:
    shard_cid: str
    source_agent_pubkey: bytes
    creator_id: str
    pcu_consumed: float = 0.0  # Sprint 239 — threaded through api.py


class _FakeOrchestrator:
    def __init__(self, participants):
        self._participants = participants

    async def dispatch_query(
        self, *, query, prompter_node_id, query_id,
        requires_tee=False, governance_denylist=frozenset(),
    ):
        return _FakeAggregatedResult(
            query_id=query_id,
            payload=b'{"count": 3}',
            aggregator_node_id="agg-node-7",
            contributing_shards=tuple(p.shard_cid for p in self._participants),
            participants=tuple(self._participants),
        )


def _node_with_orchestrator(participants, escrow):
    node = MagicMock()
    node.identity.node_id = "prompter-1"
    node.privacy_budget = None
    node.agent_forge = _FakeOrchestrator(participants)
    node._payment_escrow = escrow
    return node


def _client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


class TestForgeSplitRouting:
    """End-to-end through /compute/forge: confirm the QO path
    triggers a multi-recipient escrow split rather than the legacy
    single-provider release."""

    def _setup(
        self, *, monkeypatch, agg_share_bps="500", n_participants=3,
        budget=10.0,
    ):
        # Real PaymentEscrow with stubbed ledger.
        ledger = _StubLedger(escrow_balance=budget)
        escrow = PaymentEscrow(ledger=ledger, node_id="test-node")
        # Pre-create the escrow that /compute/forge expects.
        # /compute/forge calls create_escrow internally; the stub
        # ledger doesn't validate balances on create, so we let
        # the endpoint do its thing.
        async def _create_escrow_stub(**kwargs):
            entry = EscrowEntry(
                escrow_id=f"esc-{kwargs['job_id']}",
                job_id=kwargs["job_id"],
                requester_id=kwargs["requester_id"],
                amount=kwargs["amount"],
                status=EscrowStatus.PENDING,
            )
            escrow._escrows[entry.escrow_id] = entry
            return entry
        escrow.create_escrow = _create_escrow_stub  # monkey-patch

        if agg_share_bps is not None:
            monkeypatch.setenv("PRSM_AGGREGATOR_SHARE_BPS", agg_share_bps)
        else:
            monkeypatch.delenv("PRSM_AGGREGATOR_SHARE_BPS", raising=False)

        participants = [
            _FakeParticipant(
                shard_cid=f"cid-{i}",
                source_agent_pubkey=bytes([i] * 32),
                creator_id=f"creator-{i}",
            )
            for i in range(n_participants)
        ]
        node = _node_with_orchestrator(participants, escrow)
        return _client(node), ledger, escrow, participants

    def test_split_dispatched_when_participants_present(
        self, monkeypatch,
    ):
        client, ledger, escrow, participants = self._setup(
            monkeypatch=monkeypatch, n_participants=3, budget=10.0,
        )
        resp = client.post("/compute/forge", json={
            "query": "Count records",
            "budget_ftns": 10.0,
        })
        assert resp.status_code == 200
        # Aggregator + 3 participants = 4 recipient transfers.
        non_refund = [t for t in ledger.transfers if t["to"] != "prompter-1"]
        assert len(non_refund) == 4
        recipients = {t["to"] for t in non_refund}
        assert "agg-node-7" in recipients
        # source_agent_pubkey is the recipient — hex-encoded in the
        # /compute/forge marshal step.
        for p in participants:
            assert p.source_agent_pubkey.hex() in recipients

    def test_default_aggregator_share_is_5pct(self, monkeypatch):
        # Default PRSM_AGGREGATOR_SHARE_BPS = 500 → 5% of budget.
        client, ledger, escrow, _ = self._setup(
            monkeypatch=monkeypatch, agg_share_bps=None, budget=100.0,
            n_participants=2,
        )
        client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 100.0,
        })
        agg_tx = next(t for t in ledger.transfers if t["to"] == "agg-node-7")
        assert agg_tx["amount"] == pytest.approx(5.0)

    def test_aggregator_share_env_override_honored(self, monkeypatch):
        # Set 10% (1000 bps) and verify aggregator gets 10 of 100.
        client, ledger, escrow, _ = self._setup(
            monkeypatch=monkeypatch, agg_share_bps="1000", budget=100.0,
            n_participants=4,
        )
        client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 100.0,
        })
        agg_tx = next(t for t in ledger.transfers if t["to"] == "agg-node-7")
        assert agg_tx["amount"] == pytest.approx(10.0)
        # Compute total = 90; per-participant = 22.5 across 4.
        worker_txs = [
            t for t in ledger.transfers
            if t["to"] not in ("agg-node-7", "prompter-1")
        ]
        assert all(
            t["amount"] == pytest.approx(22.5) for t in worker_txs
        )

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        # Garbage env value → use 500 default.
        client, ledger, _, _ = self._setup(
            monkeypatch=monkeypatch, agg_share_bps="not-a-number",
            budget=100.0, n_participants=1,
        )
        client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 100.0,
        })
        agg_tx = next(t for t in ledger.transfers if t["to"] == "agg-node-7")
        assert agg_tx["amount"] == pytest.approx(5.0)  # 5% default

    def test_response_surfaces_participants(self, monkeypatch):
        # MCP clients consuming /compute/forge can inspect the
        # participants list in the response (audit-side surface).
        client, _, _, participants = self._setup(
            monkeypatch=monkeypatch, n_participants=2, budget=10.0,
        )
        body = client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 10.0,
        }).json()
        assert "participants" in body["result"]
        assert len(body["result"]["participants"]) == 2
        for entry in body["result"]["participants"]:
            assert "shard_cid" in entry
            assert "source_agent_pubkey_hex" in entry
            assert "creator_id" in entry
