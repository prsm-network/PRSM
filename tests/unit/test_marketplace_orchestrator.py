"""Unit tests for MarketplaceOrchestrator.

Phase 3 Task 7. Unit-scoped — uses mocks for directory, dispatcher,
price negotiator, reputation. The full wiring is covered by the
integration test in tests/integration/test_phase3_marketplace_e2e.py.
"""
from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from prsm.compute.model_sharding.models import ModelShard, PipelineStakeTier
from prsm.compute.remote_dispatcher import (
    MissingAttestationError,
    ShardDispatchError,
    ShardPreemptedError,
)
from prsm.marketplace.errors import NoEligibleProvidersError
from prsm.marketplace.filter import EligibilityFilter
from prsm.marketplace.listing import sign_listing
from prsm.marketplace.orchestrator import MarketplaceOrchestrator
from prsm.marketplace.policy import DispatchPolicy
from prsm.marketplace.price_handshake import PriceQuote, PriceQuoteRejected
from prsm.marketplace.reputation import ReputationTracker
from prsm.node.identity import generate_node_identity


def _make_shard(index: int = 0) -> ModelShard:
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tensor_bytes = tensor.tobytes()
    return ModelShard(
        shard_id=f"shard-{index}",
        model_id="m-test",
        shard_index=index,
        total_shards=1,
        tensor_data=tensor_bytes,
        tensor_shape=tensor.shape,
        size_bytes=len(tensor_bytes),
        checksum=hashlib.sha256(tensor_bytes).hexdigest(),
    )


def _make_listing(**overrides):
    identity = overrides.pop("identity", None) or generate_node_identity(
        display_name="p"
    )
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


def _make_orchestrator(listings, *, verify_tee=False):
    identity = generate_node_identity(display_name="requester")
    directory = MagicMock()
    directory.list_active_providers = MagicMock(return_value=listings)
    directory.get_listing = MagicMock(
        side_effect=lambda pid, **kw: next(
            (l for l in listings if l.provider_id == pid), None
        )
    )

    reputation = ReputationTracker()
    eligibility = EligibilityFilter(reputation_tracker=reputation)

    price_negotiator = MagicMock()
    price_negotiator.request_quote = AsyncMock()

    remote_dispatcher = MagicMock()
    remote_dispatcher.dispatch = AsyncMock()

    orchestrator = MarketplaceOrchestrator(
        identity=identity,
        directory=directory,
        eligibility_filter=eligibility,
        reputation=reputation,
        price_negotiator=price_negotiator,
        remote_dispatcher=remote_dispatcher,
    )
    return (
        orchestrator, price_negotiator, remote_dispatcher, reputation,
    )


def _make_quote(listing, price: float = 0.05) -> PriceQuote:
    return PriceQuote(
        request_id="q-1",
        listing_id=listing.listing_id,
        shard_index=0,
        quoted_price_ftns=price,
        quote_expires_unix=9999999999,
        provider_id=listing.provider_id,
        provider_pubkey_b64=listing.provider_pubkey_b64,
        signature="sig",
    )


def _run(coro):
    return asyncio.run(coro)


def test_orchestrator_happy_path_single_shard():
    listing = _make_listing()
    orch, quoter, dispatcher, reputation = _make_orchestrator([listing])

    expected_output = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.return_value = expected_output

    shard = _make_shard(0)
    input_tensor = np.array([1.0, 1.0], dtype=np.float64)
    policy = DispatchPolicy()

    result = _run(orch.orchestrate_sharded_inference(
        shards=[shard], input_tensor=input_tensor,
        job_id="job-1", policy=policy,
    ))

    np.testing.assert_array_equal(result, expected_output)
    quoter.request_quote.assert_awaited_once()
    dispatcher.dispatch.assert_awaited_once()
    # Reputation recorded success with some latency.
    rep = reputation.get_reputation(listing.provider_id)
    assert rep is not None
    assert len(rep.successful_dispatches) == 1
    assert len(rep.latency_samples_ms) == 1


def test_orchestrator_no_eligible_providers_raises():
    """When filter excludes every listing, raise NoEligibleProvidersError
    without any network traffic."""
    # Listing too expensive for the policy.
    listing = _make_listing(price_per_shard_ftns=10.0)
    orch, quoter, dispatcher, _ = _make_orchestrator([listing])

    shard = _make_shard(0)
    input_tensor = np.array([1.0, 1.0], dtype=np.float64)
    policy = DispatchPolicy(max_price_per_shard_ftns=0.10)

    with pytest.raises(NoEligibleProvidersError):
        _run(orch.orchestrate_sharded_inference(
            shards=[shard], input_tensor=input_tensor,
            job_id="job-1", policy=policy,
        ))

    quoter.request_quote.assert_not_called()
    dispatcher.dispatch.assert_not_called()


def test_orchestrator_empty_directory_raises():
    orch, _, _, _ = _make_orchestrator([])

    shard = _make_shard(0)
    input_tensor = np.array([1.0, 1.0], dtype=np.float64)

    with pytest.raises(NoEligibleProvidersError):
        _run(orch.orchestrate_sharded_inference(
            shards=[shard], input_tensor=input_tensor,
            job_id="job-1", policy=DispatchPolicy(),
        ))


def test_orchestrator_quote_rejection_tries_next_provider():
    """First provider rejects the quote; orchestrator moves to the second."""
    # sp1498 — selection is now CHEAPEST-first with a provider_id tie-break, so
    # the order these are tried in must be stated by price, not by construction
    # order. The behaviour under test (reject -> try next) is unchanged.
    listing_bad = _make_listing(price_per_shard_ftns=0.04)
    listing_good = _make_listing(price_per_shard_ftns=0.05)
    orch, quoter, dispatcher, reputation = _make_orchestrator(
        [listing_bad, listing_good]
    )

    expected = np.array([42.0], dtype=np.float64)

    # First call rejects, second accepts.
    quoter.request_quote.side_effect = [
        PriceQuoteRejected(
            request_id="q-1", listing_id=listing_bad.listing_id,
            reason="overloaded",
        ),
        _make_quote(listing_good),
    ]
    dispatcher.dispatch.return_value = expected

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    np.testing.assert_array_equal(result, expected)
    assert quoter.request_quote.await_count == 2
    assert dispatcher.dispatch.await_count == 1
    # Good provider got the success.
    _, kwargs = dispatcher.dispatch.await_args_list[0].args, dispatcher.dispatch.await_args_list[0].kwargs
    assert kwargs["node_id"] == listing_good.provider_id


def test_orchestrator_preemption_reroutes_to_fresh_pool():
    """Preempted provider → record_preemption (no reputation penalty),
    try next. Honest-work failure stays off the score denominator."""
    listing_preempted = _make_listing(price_per_shard_ftns=0.04)
    listing_ok = _make_listing(price_per_shard_ftns=0.05)
    orch, quoter, dispatcher, reputation = _make_orchestrator(
        [listing_preempted, listing_ok]
    )

    quoter.request_quote.side_effect = [
        _make_quote(listing_preempted),
        _make_quote(listing_ok),
    ]
    dispatcher.dispatch.side_effect = [
        ShardPreemptedError(
            shard_index=0, node_id=listing_preempted.provider_id,
            reason="spot_eviction",
        ),
        np.array([1.0], dtype=np.float64),
    ]

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    np.testing.assert_array_equal(result, np.array([1.0]))
    # Preempted provider: preemption counted, NOT a failure.
    rep_preempted = reputation.get_reputation(listing_preempted.provider_id)
    assert len(rep_preempted.preempted_dispatches) == 1
    assert len(rep_preempted.failed_dispatches) == 0
    # OK provider: success recorded.
    rep_ok = reputation.get_reputation(listing_ok.provider_id)
    assert len(rep_ok.successful_dispatches) == 1


def test_orchestrator_missing_attestation_penalizes_and_retries():
    """When policy.require_tee=True and a provider's receipt omits
    tee_attestation, the dispatcher raises MissingAttestationError.
    Orchestrator records failure (liar penalty) and moves on."""
    identity_bad = generate_node_identity(display_name="liar")
    identity_ok = generate_node_identity(display_name="honest")
    listing_liar = _make_listing(identity=identity_bad, tee_capable=True,
                                 price_per_shard_ftns=0.04)
    listing_honest = _make_listing(identity=identity_ok, tee_capable=True,
                                   price_per_shard_ftns=0.05)
    orch, quoter, dispatcher, reputation = _make_orchestrator(
        [listing_liar, listing_honest]
    )

    quoter.request_quote.side_effect = [
        _make_quote(listing_liar),
        _make_quote(listing_honest),
    ]
    dispatcher.dispatch.side_effect = [
        MissingAttestationError("no tee quote"),
        np.array([1.0], dtype=np.float64),
    ]

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1",
        policy=DispatchPolicy(require_tee=True),
    ))

    np.testing.assert_array_equal(result, np.array([1.0]))
    # Liar: failure recorded.
    rep_liar = reputation.get_reputation(listing_liar.provider_id)
    assert len(rep_liar.failed_dispatches) == 1
    assert len(rep_liar.successful_dispatches) == 0


def test_orchestrator_shard_dispatch_error_propagates():
    """Malicious/unrecoverable ShardDispatchError from the dispatcher
    escalates immediately — we don't silently retry on a different
    provider because the error isn't provider-specific (could be
    verification failure on a signed receipt, which is a protocol-
    level problem)."""
    listing = _make_listing()
    orch, quoter, dispatcher, reputation = _make_orchestrator([listing])

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.side_effect = ShardDispatchError("bad receipt")

    with pytest.raises(ShardDispatchError):
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job-1", policy=DispatchPolicy(),
        ))

    rep = reputation.get_reputation(listing.provider_id)
    assert len(rep.failed_dispatches) == 1


def test_orchestrator_exhausted_pool_raises_no_eligible():
    """All providers in the eligible pool rejected or preempted →
    NoEligibleProvidersError with a summary of the last failure."""
    listing_a = _make_listing()
    listing_b = _make_listing()
    orch, quoter, dispatcher, _ = _make_orchestrator([listing_a, listing_b])

    quoter.request_quote.return_value = PriceQuoteRejected(
        request_id="q", listing_id="x", reason="overloaded",
    )

    with pytest.raises(NoEligibleProvidersError) as excinfo:
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job-1", policy=DispatchPolicy(),
        ))

    assert "exhausted" in str(excinfo.value).lower()
    dispatcher.dispatch.assert_not_called()


def test_orchestrator_multiple_shards_concatenates_in_order():
    """Two shards → two dispatches → concatenate axis=0 in shard index
    order."""
    listing = _make_listing()
    orch, quoter, dispatcher, _ = _make_orchestrator([listing])

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.side_effect = [
        np.array([1.0, 2.0], dtype=np.float64),  # shard 0
        np.array([3.0, 4.0], dtype=np.float64),  # shard 1
    ]

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0), _make_shard(1)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    np.testing.assert_array_equal(
        result, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )


# ── Phase 3.1 Task 7: batched-settlement accumulation hook ────────


def _make_orchestrator_with_settlement(listings):
    """Variant of _make_orchestrator that wires an AsyncMock
    BatchSettlementClient + a provider_address_resolver so the
    Phase 3.1 hook path is active."""
    identity = generate_node_identity(display_name="requester")
    identity.ethereum_address = "0x" + "e" * 40  # set to activate the hook

    directory = MagicMock()
    directory.list_active_providers = MagicMock(return_value=listings)
    directory.get_listing = MagicMock(
        side_effect=lambda pid, **kw: next(
            (l for l in listings if l.provider_id == pid), None
        )
    )
    reputation = ReputationTracker()
    eligibility = EligibilityFilter(reputation_tracker=reputation)

    price_negotiator = MagicMock()
    price_negotiator.request_quote = AsyncMock()

    remote_dispatcher = MagicMock()
    remote_dispatcher.dispatch = AsyncMock()
    remote_dispatcher.dispatch_with_receipt = AsyncMock()

    batch_client = MagicMock()
    batch_client.accumulate = AsyncMock()

    # Simple resolver: maps provider_id → a synthetic Ethereum address.
    def resolver(provider_id):
        return "0x" + "f" * 40  # always return a stub address

    orchestrator = MarketplaceOrchestrator(
        identity=identity,
        directory=directory,
        eligibility_filter=eligibility,
        reputation=reputation,
        price_negotiator=price_negotiator,
        remote_dispatcher=remote_dispatcher,
        batch_settlement_client=batch_client,
        provider_address_resolver=resolver,
    )
    return (orchestrator, price_negotiator, remote_dispatcher, batch_client)


def _make_dispatch_result(output: np.ndarray, node_id: str, amount: float = 0.05):
    """Build the DispatchResult shape dispatch_with_receipt returns."""
    from prsm.compute.remote_dispatcher import DispatchResult
    return DispatchResult(
        output=output,
        receipt={
            "job_id": "job-1",
            "shard_index": 0,
            "provider_id": "provider-id-hex",
            "provider_pubkey_b64": "AAAA",
            "output_hash": "a" * 64,
            "executed_at_unix": 1700000000,
            "signature": "BBBB",
        },
        provider_node_id=node_id,
        escrow_amount_ftns=amount,
    )


def test_orchestrator_accumulates_when_settlement_client_wired():
    """Happy path: a successful dispatch forwards a BatchedReceipt to
    the settlement client."""
    listing = _make_listing()
    orch, quoter, dispatcher, batch = _make_orchestrator_with_settlement([listing])

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch_with_receipt.return_value = _make_dispatch_result(
        np.array([1.0, 2.0], dtype=np.float64), listing.provider_id,
    )

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    # Settlement client was invoked with a BatchedReceipt.
    batch.accumulate.assert_awaited_once()
    br = batch.accumulate.await_args.args[0]
    assert br.requester_address == orch.identity.ethereum_address
    assert br.provider_address == "0x" + "f" * 40  # from resolver
    assert br.receipt.job_id == "job-1"


def test_orchestrator_does_not_use_dispatch_with_receipt_when_not_wired():
    """Phase 3 preservation: without a settlement client, the orchestrator
    uses the Phase 2 dispatch API that returns only np.ndarray."""
    listing = _make_listing()
    orch, quoter, dispatcher, _ = _make_orchestrator([listing])  # no batch client

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.return_value = np.array([1.0], dtype=np.float64)

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    # Old API was used; new API was NOT.
    dispatcher.dispatch.assert_awaited_once()
    dispatcher.dispatch_with_receipt.assert_not_called()


def test_orchestrator_skips_accumulation_when_ethereum_address_missing():
    """If identity.ethereum_address is None (Phase 2/3 node without
    Phase 3.1 setup), fall back to Phase 2 dispatch API — no batched
    settlement attempted."""
    listing = _make_listing()
    orch, quoter, dispatcher, batch = _make_orchestrator_with_settlement([listing])
    orch.identity.ethereum_address = None  # clear it

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.return_value = np.array([1.0], dtype=np.float64)

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    # Fell back to Phase 2 API; no settlement accumulation.
    dispatcher.dispatch.assert_awaited_once()
    dispatcher.dispatch_with_receipt.assert_not_called()
    batch.accumulate.assert_not_called()


def test_orchestrator_skips_accumulation_when_resolver_returns_none():
    """Resolver returning None for a provider (unknown Ethereum address)
    skips accumulation for that shard gracefully — no raise."""
    listing = _make_listing()
    orch, quoter, dispatcher, batch = _make_orchestrator_with_settlement([listing])
    # Override resolver to return None.
    orch.provider_address_resolver = lambda pid: None

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch_with_receipt.return_value = _make_dispatch_result(
        np.array([1.0], dtype=np.float64), listing.provider_id,
    )

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    # Dispatch succeeded; accumulate skipped without raising.
    dispatcher.dispatch_with_receipt.assert_awaited_once()
    batch.accumulate.assert_not_called()


def test_orchestrator_accumulation_failure_does_not_break_dispatch():
    """If the settlement client's accumulate raises, the orchestrator
    still returns the dispatch output — batched settlement is strictly
    additive; its failures must never block Phase 3 success."""
    listing = _make_listing()
    orch, quoter, dispatcher, batch = _make_orchestrator_with_settlement([listing])

    batch.accumulate.side_effect = RuntimeError("settlement RPC down")

    quoter.request_quote.return_value = _make_quote(listing)
    expected = np.array([42.0], dtype=np.float64)
    dispatcher.dispatch_with_receipt.return_value = _make_dispatch_result(
        expected, listing.provider_id,
    )

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    np.testing.assert_array_equal(result, expected)


def test_orchestrator_value_ftns_converted_to_wei():
    """Phase 3's quoted_price is a float FTNS; Phase 3.1 uses uint128
    wei. Orchestrator must multiply by 10^18."""
    listing = _make_listing()
    orch, quoter, dispatcher, batch = _make_orchestrator_with_settlement([listing])

    # Quote 0.05 FTNS = 5 * 10^16 wei
    quote = PriceQuote(
        request_id="q-1",
        listing_id=listing.listing_id,
        shard_index=0,
        quoted_price_ftns=0.05,
        quote_expires_unix=9999999999,
        provider_id=listing.provider_id,
        provider_pubkey_b64=listing.provider_pubkey_b64,
        signature="sig",
    )
    quoter.request_quote.return_value = quote
    dispatcher.dispatch_with_receipt.return_value = _make_dispatch_result(
        np.array([1.0], dtype=np.float64), listing.provider_id, amount=0.05,
    )

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    br = batch.accumulate.await_args.args[0]
    assert br.value_ftns == 5 * 10**16


def test_orchestrator_empty_receipt_on_fallback_path_skips_accumulation():
    """If dispatch_with_receipt returns an empty-receipt DispatchResult
    (e.g., local_fallback path), no accumulation happens."""
    listing = _make_listing()
    orch, quoter, dispatcher, batch = _make_orchestrator_with_settlement([listing])

    from prsm.compute.remote_dispatcher import DispatchResult
    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch_with_receipt.return_value = DispatchResult(
        output=np.array([1.0], dtype=np.float64),
        receipt={},  # empty — fallback path
        provider_node_id=listing.provider_id,
        escrow_amount_ftns=0.0,
    )

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    batch.accumulate.assert_not_called()


# ── Phase 7 Task 5: on-chain tier gating ─────────────────────────────────


def _make_orchestrator_with_stake_gate(listings, *, resolver=None):
    """Orchestrator wired with a stake_manager_client + provider address
    resolver so Phase 7 tier gating is active."""
    identity = generate_node_identity(display_name="requester")

    directory = MagicMock()
    directory.list_active_providers = MagicMock(return_value=listings)
    directory.get_listing = MagicMock(
        side_effect=lambda pid, **kw: next(
            (l for l in listings if l.provider_id == pid), None
        )
    )
    reputation = ReputationTracker()
    eligibility = EligibilityFilter(reputation_tracker=reputation)

    price_negotiator = MagicMock()
    price_negotiator.request_quote = AsyncMock()

    remote_dispatcher = MagicMock()
    remote_dispatcher.dispatch = AsyncMock()

    stake_manager = MagicMock()
    # effective_tier is a sync read; use MagicMock, not AsyncMock.
    stake_manager.effective_tier = MagicMock(return_value="critical")

    if resolver is None:
        def resolver(provider_id):
            return "0x" + "a" * 40

    orchestrator = MarketplaceOrchestrator(
        identity=identity,
        directory=directory,
        eligibility_filter=eligibility,
        reputation=reputation,
        price_negotiator=price_negotiator,
        remote_dispatcher=remote_dispatcher,
        provider_address_resolver=resolver,
        stake_manager_client=stake_manager,
    )
    return (
        orchestrator, price_negotiator, remote_dispatcher,
        reputation, stake_manager,
    )


def test_onchain_tier_gate_skipped_when_policy_is_open():
    """If policy.min_stake_tier=='open' (the default), the on-chain
    gate short-circuits — no RPC to StakeBond. Preserves Phase 3
    zero-cost-latency on low-value jobs."""
    listing = _make_listing()
    orch, quoter, dispatcher, _, stake_manager = (
        _make_orchestrator_with_stake_gate([listing])
    )

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.return_value = np.array([1.0], dtype=np.float64)

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(min_stake_tier="open"),
    ))

    stake_manager.effective_tier.assert_not_called()
    dispatcher.dispatch.assert_awaited_once()


def test_onchain_tier_gate_skipped_when_stake_manager_unwired():
    """Phase 3.1 preservation: if stake_manager_client is None, Phase 7
    gating is invisible even under a premium policy."""
    # Build a non-phase-7 orchestrator (no stake_manager_client).
    listing = _make_listing(stake_tier="premium")
    orch, quoter, dispatcher, _ = _make_orchestrator([listing])

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.return_value = np.array([1.0], dtype=np.float64)

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1",
        policy=DispatchPolicy(min_stake_tier="premium"),
    ))

    dispatcher.dispatch.assert_awaited_once()


def test_onchain_tier_gate_skipped_when_resolver_returns_none():
    """Providers without a resolvable Ethereum address (pre-Phase 7
    listings) pass the gate — the listing-claim filter has already
    approved them."""
    listing = _make_listing(stake_tier="premium")
    orch, quoter, dispatcher, _, stake_manager = (
        _make_orchestrator_with_stake_gate(
            [listing], resolver=lambda pid: None,
        )
    )

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.return_value = np.array([1.0], dtype=np.float64)

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1",
        policy=DispatchPolicy(min_stake_tier="premium"),
    ))

    # Resolver returned None → no RPC issued, dispatch proceeds.
    stake_manager.effective_tier.assert_not_called()
    dispatcher.dispatch.assert_awaited_once()


def test_onchain_tier_gate_passes_when_stake_meets_required():
    """Provider with on-chain tier ≥ required → dispatch proceeds and
    the RPC was consulted once."""
    listing = _make_listing(stake_tier="premium")
    orch, quoter, dispatcher, _, stake_manager = (
        _make_orchestrator_with_stake_gate([listing])
    )
    stake_manager.effective_tier.return_value = "premium"

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.return_value = np.array([1.0], dtype=np.float64)

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1",
        policy=DispatchPolicy(min_stake_tier="premium"),
    ))

    stake_manager.effective_tier.assert_called_once()
    dispatcher.dispatch.assert_awaited_once()


def test_onchain_tier_gate_rejects_cheater_and_retries_next():
    """Provider listing claims 'premium' but StakeBond says 'standard'
    (or 'open') → gate rejects, reputation.record_failure, move to next
    provider. The quote RPC is never issued for the cheater."""
    listing_cheater = _make_listing(stake_tier="premium", price_per_shard_ftns=0.04)
    listing_honest = _make_listing(stake_tier="premium", price_per_shard_ftns=0.05)
    orch, quoter, dispatcher, reputation, stake_manager = (
        _make_orchestrator_with_stake_gate([listing_cheater, listing_honest])
    )

    # Cheater returns "standard" on chain; honest returns "premium".
    def by_addr(addr):
        # Both providers resolve to stub addresses; differentiate by call
        # order via side_effect list below.
        return addr

    tier_sequence = ["standard", "premium"]
    stake_manager.effective_tier.side_effect = lambda addr: tier_sequence.pop(0)

    quoter.request_quote.return_value = _make_quote(listing_honest)
    dispatcher.dispatch.return_value = np.array([42.0], dtype=np.float64)

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1",
        policy=DispatchPolicy(min_stake_tier="premium"),
    ))

    np.testing.assert_array_equal(result, np.array([42.0]))
    # Cheater: failure recorded, no quote issued.
    rep_cheater = reputation.get_reputation(listing_cheater.provider_id)
    assert len(rep_cheater.failed_dispatches) == 1
    # Gate was checked twice (cheater + honest); quote only for honest.
    assert stake_manager.effective_tier.call_count == 2
    assert quoter.request_quote.await_count == 1


def test_onchain_tier_gate_exhausts_pool_when_all_cheat():
    """Pool entirely of listing-cheaters → every gate check rejects,
    ends in NoEligibleProvidersError."""
    listings = [_make_listing(stake_tier="critical") for _ in range(3)]
    orch, quoter, dispatcher, reputation, stake_manager = (
        _make_orchestrator_with_stake_gate(listings)
    )
    stake_manager.effective_tier.return_value = "open"  # every provider cheats

    with pytest.raises(NoEligibleProvidersError) as excinfo:
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job-1",
            policy=DispatchPolicy(min_stake_tier="critical"),
        ))

    assert "tier_below_required_onchain" in str(excinfo.value)
    # No quote RPCs, no dispatches — cheaters get stopped at the gate.
    quoter.request_quote.assert_not_called()
    dispatcher.dispatch.assert_not_called()
    # Each cheater logged a failure.
    for listing in listings:
        rep = reputation.get_reputation(listing.provider_id)
        assert len(rep.failed_dispatches) == 1


def test_onchain_tier_gate_fails_open_on_rpc_error():
    """If StakeBond RPC raises, the gate fails OPEN (logs warning,
    allows dispatch). Closed-fail would let an attacker DoS the RPC
    to block all dispatch; the listing filter + slashing-on-misbehavior
    still protect the cheat case."""
    listing = _make_listing(stake_tier="premium")
    orch, quoter, dispatcher, _, stake_manager = (
        _make_orchestrator_with_stake_gate([listing])
    )
    stake_manager.effective_tier.side_effect = RuntimeError("RPC timeout")

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.return_value = np.array([1.0], dtype=np.float64)

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1",
        policy=DispatchPolicy(min_stake_tier="premium"),
    ))

    # RPC raised but dispatch proceeded.
    stake_manager.effective_tier.assert_called_once()
    dispatcher.dispatch.assert_awaited_once()


def test_onchain_tier_gate_ordering_standard_below_critical():
    """Ordinal check: a provider with on-chain 'standard' cannot
    serve a 'critical' job."""
    listing = _make_listing(stake_tier="critical")
    orch, quoter, dispatcher, reputation, stake_manager = (
        _make_orchestrator_with_stake_gate([listing])
    )
    stake_manager.effective_tier.return_value = "standard"

    with pytest.raises(NoEligibleProvidersError):
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job-1",
            policy=DispatchPolicy(min_stake_tier="critical"),
        ))

    rep = reputation.get_reputation(listing.provider_id)
    assert len(rep.failed_dispatches) == 1
    quoter.request_quote.assert_not_called()


def test_onchain_tier_gate_runs_before_price_handshake():
    """The gate must run BEFORE the price handshake so a cheater
    doesn't get to occupy a quote slot (they'd waste resources on
    the quoter side and could cause DoS)."""
    listing = _make_listing(stake_tier="premium")
    orch, quoter, dispatcher, _, stake_manager = (
        _make_orchestrator_with_stake_gate([listing])
    )
    stake_manager.effective_tier.return_value = "open"  # cheater

    with pytest.raises(NoEligibleProvidersError):
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job-1",
            policy=DispatchPolicy(min_stake_tier="premium"),
        ))

    # Quote was never solicited — we didn't waste a quote slot on cheater.
    quoter.request_quote.assert_not_called()
