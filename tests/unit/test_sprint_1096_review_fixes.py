"""Sprint 1096 — folds the cross-cutting review (Domain 01) findings:
- A2: a multi-stage relayer job skips the on-chain accumulate, so the capture must release
  the FULL reservation (nothing settled on-chain), not reserved-minus-offchain-release.
- A1 (eviction): an evicted relayer carrier entry never settled → release its held budget.
- #5: PaymentAuthorizationVerifier now serializes verify() with a lock (parity with the
  relayer verifier), closing the seen()->remember() TOCTOU.
"""
from __future__ import annotations

import types

import pytest
from eth_account import Account

from prsm.node.api import (
    _PAID_REQUESTER_CARRIER_CAP,
    _capture_relayer_budget,
    _record_paid_requester,
)
from prsm.settlement.delegation_budget import InMemoryDelegationBudgetStore
from prsm.settlement.payment_authorization import (
    canonical_request_hash, inference_request_fields, sign_payment_authorization,
)
from prsm.settlement.payment_authorization_verifier import (
    AuthorizationRejected, PaymentAuthorizationVerifier, RelayerAuthorizationVerifier,
)

_CAP = 10**18
_PROVIDER = Account.create()


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _node_with_budget():
    store = InMemoryDelegationBudgetStore()
    verifier = RelayerAuthorizationVerifier(
        provider_address=_PROVIDER.address, budget_store=store, chain_id=8453)
    node = types.SimpleNamespace(_relayer_verifier=verifier, _paid_requester_by_job={})
    return node, store


# ── A2: skipped on-chain accumulate → release the FULL reservation ───────────────

def test_capture_releases_full_reservation_when_accumulate_skipped():
    node, store = _node_with_budget()
    nonce = "0x" + "cd" * 32
    store.reserve(nonce, 10**17, _CAP)              # reserved the ceiling
    paid = {"delegation_nonce": nonce, "reserved_wei": 10**17}
    # multi-stage job: nothing settled on-chain → FULL release despite an off-chain release
    _run(_capture_relayer_budget(node, paid, settled_ftns=0.05,
                                 accumulate_result="skipped:multi-stage-deferred"))
    assert store.consumed(nonce) == 0   # full reservation returned, not reserved-settled


def test_capture_releases_only_remainder_on_real_settle():
    node, store = _node_with_budget()
    nonce = "0x" + "ce" * 32
    store.reserve(nonce, 10**17, _CAP)
    paid = {"delegation_nonce": nonce, "reserved_wei": 10**17}
    # real on-chain settle of 0.05 FTNS (5e16 wei) → release reserved-settled = 5e16
    _run(_capture_relayer_budget(node, paid, settled_ftns=0.05, accumulate_result="batch-1"))
    assert store.consumed(nonce) == 5 * 10**16


# ── A1: eviction of a relayer entry releases its held reservation ────────────────

def test_eviction_releases_relayer_reservation(monkeypatch):
    import prsm.node.api as api
    monkeypatch.setattr(api, "_PAID_REQUESTER_CARRIER_CAP", 2)
    node, store = _node_with_budget()
    # three relayer jobs, each reserving 0.1 FTNS under its own delegation
    for i in range(3):
        n = "0x" + f"{i:02d}" * 32
        store.reserve(n, 10**17, _CAP)
        _record_paid_requester(node, f"job{i}", {"delegation_nonce": n, "reserved_wei": 10**17})
    # cap=2, so job0 was evicted on the 3rd record → its reservation released
    assert store.consumed("0x" + "00" * 32) == 0          # evicted → released
    assert store.consumed("0x" + "02" * 32) == 10**17       # still in carrier → held


def test_eviction_noop_for_self_pay_entries(monkeypatch):
    import prsm.node.api as api
    monkeypatch.setattr(api, "_PAID_REQUESTER_CARRIER_CAP", 1)
    node, store = _node_with_budget()
    # self-signed / self-pay entries have no delegation_nonce → eviction is a no-op (no crash)
    _record_paid_requester(node, "j0", {"requester": "0xabc", "max_spend_wei": 1})
    _record_paid_requester(node, "j1", {"requester": "0xdef", "max_spend_wei": 1})  # evicts j0
    assert "j1" in node._paid_requester_by_job


# ── #5: the self-signed verifier now serializes same-nonce verify() ──────────────

def test_self_signed_concurrent_same_nonce_only_one_succeeds():
    requester = Account.create()
    rh = canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="hi", max_tokens=8, privacy_tier="none", content_tier="A"))
    payload = {"requester": requester.address, "provider": _PROVIDER.address,
               "max_spend_wei": 10**17, "job_nonce": "0x" + "ab" * 32,
               "expiry_unix": 9999999999, "request_hash": rh}
    sig = sign_payment_authorization(payload, requester.key, chain_id=8453)

    import asyncio

    async def slow_balance(_addr):
        await asyncio.sleep(0.01)   # yield between seen() and remember()
        return 10**18
    v = PaymentAuthorizationVerifier(
        provider_address=_PROVIDER.address, balance_reader=slow_balance,
        chain_id=8453, now=lambda: 100.0)

    async def one():
        try:
            return await v.verify(payload, sig, request_hash=rh, quoted_price_wei=10**16)
        except AuthorizationRejected as exc:
            return exc.reason

    async def both():
        return await asyncio.gather(one(), one())
    results = _run(both())
    ok = [r for r in results if isinstance(r, str) and r.lower() == requester.address.lower()]
    rejects = [r for r in results if r == "nonce-replayed"]
    assert len(ok) == 1 and len(rejects) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
