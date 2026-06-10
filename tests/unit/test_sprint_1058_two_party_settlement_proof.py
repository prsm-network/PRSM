"""Sprint 1058 — two-party settlement proof orchestration (requester-payment brick 5).

The capstone that joins the requester-payment auth (bricks 1-4) to the on-chain
settlement rail (sp1035-1053): requester A signs a PaymentAuthorization; provider B
verifies it, builds a batched receipt naming A as requester + B as provider, funds
A's escrow, and commits. finalizeBatch then draws from A's escrow → B's wallet —
REAL cross-party value, unlike the sp1042 self-pay proof. The live Base-Sepolia run
is user-gated (funded two keys); these tests drive the composition with mocks.
"""
from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock

import pytest
from eth_account import Account

from prsm.node.identity import generate_node_identity
from prsm.settlement.accumulator import TriggerReason
from prsm.settlement.client import CommittedBatch
from prsm.settlement.payment_authorization_verifier import (
    AuthorizationRejected, InMemoryNonceStore, PaymentAuthorizationVerifier,
)
from prsm.settlement.payment_client import build_payment_authorization
from prsm.settlement.e2e_proof import run_two_party_deposit_and_commit

ONE_FTNS = 10**18
REQUESTER_KEY = "0x" + "11" * 32
REQUESTER = Account.from_key(REQUESTER_KEY).address
PROVIDER = Account.from_key("0x" + "22" * 32).address
_REQ = dict(model_id="gpt2", prompt="two-party hello", max_tokens=16,
            privacy_tier="standard", content_tier="A")


def _run(coro):
    return asyncio.run(coro)


def _committed():
    return CommittedBatch(
        batch_id=b"\xbb" * 32, tx_hash="0xcommit", provider_address=PROVIDER,
        requester_address=REQUESTER, merkle_root=b"\xcd" * 32, receipt_count=1,
        total_value_ftns=ONE_FTNS, commit_timestamp=1700000000,
        leaf_hashes=(b"\x01" * 32,), trigger_reason=TriggerReason.VALUE)


def _escrow_client(balance=ONE_FTNS):
    m = AsyncMock()
    m.balance_of.return_value = balance
    m.deposit.return_value = "0xdeposit"
    m.ftns_balance_of.return_value = 0
    return m


def _settlement_client():
    m = AsyncMock()
    m.accumulate.return_value = None
    m.commit_ready_batches.return_value = [_committed()]
    return m


def _verifier():
    return PaymentAuthorizationVerifier(
        provider_address=PROVIDER, nonce_store=InMemoryNonceStore(),
        now=lambda: 1_000_000_000.0)


def _auth():
    return build_payment_authorization(
        requester_key=REQUESTER_KEY, provider_address=PROVIDER,
        max_spend_ftns=1.0, expiry_unix=2_000_000_000, **_REQ)


def _call(**over):
    kw = dict(
        escrow_client=_escrow_client(), settlement_client=_settlement_client(),
        verifier=_verifier(), payment_authorization=_auth(), request_fields=_REQ,
        provider_address=PROVIDER, identity=generate_node_identity(display_name="prov-B"),
        job_id="job-2p", output_hash=hashlib.sha256(b"out").hexdigest(),
        executed_at_unix=1700000000, value_wei=ONE_FTNS,
        local_escrow_id="esc-2p", deposit_amount_wei=ONE_FTNS)
    kw.update(over)
    return run_two_party_deposit_and_commit(**kw)


def test_two_party_commit_names_distinct_requester_and_provider():
    report = _run(_call())
    assert report["requester_address"].lower() == REQUESTER.lower()
    assert report["provider_address"].lower() == PROVIDER.lower()
    assert report["requester_address"].lower() != report["provider_address"].lower()
    assert report["batch_id"] == (b"\xbb" * 32).hex()


def test_deposit_targets_requester_escrow_and_commit_happens():
    settlement = _settlement_client()
    escrow = _escrow_client()
    _run(_call(settlement_client=settlement, escrow_client=escrow))
    # the receipt accumulated names the authenticated requester
    accumulated = settlement.accumulate.call_args.args[0]
    assert accumulated.requester_address.lower() == REQUESTER.lower()
    assert accumulated.provider_address.lower() == PROVIDER.lower()
    settlement.commit_ready_batches.assert_awaited_once()


def test_rejected_authorization_aborts_before_any_commit():
    """A bad auth (wrong provider pin) must NOT deposit or commit."""
    settlement = _settlement_client()
    escrow = _escrow_client()
    bad = build_payment_authorization(
        requester_key=REQUESTER_KEY, provider_address=Account.from_key("0x" + "33" * 32).address,
        max_spend_ftns=1.0, expiry_unix=2_000_000_000, **_REQ)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(_call(payment_authorization=bad, settlement_client=settlement, escrow_client=escrow))
    assert ei.value.reason == "provider-mismatch"
    settlement.commit_ready_batches.assert_not_awaited()
    escrow.deposit.assert_not_awaited()


def test_price_over_max_spend_aborts():
    settlement = _settlement_client()
    with pytest.raises(AuthorizationRejected) as ei:
        _run(_call(value_wei=ONE_FTNS + 1, settlement_client=settlement))  # auth max_spend == ONE_FTNS
    assert ei.value.reason == "exceeds-max-spend"
    settlement.commit_ready_batches.assert_not_awaited()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
