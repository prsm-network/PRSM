"""sp1452 (money-path audit wuh7kwl7i) — a relayer cannot ALIAS one funder-signed delegation into
N budget buckets to drain the funder's escrow past the signed cumulative cap.

The delegation's cumulative cap (max_total_spend_wei) is enforced OFF-CHAIN by DelegationBudgetStore
keyed by delegation_nonce (the on-chain EscrowPool checks balance, not the per-delegation cap). The
EIP-712 delegation signature canonicalizes the nonce via _to_bytes32 (strips a leading '0x',
hex-case-insensitive), so '0x<hex>' and bare '<hex>' (and case variants) share ONE digest → ONE funder
signature verifies all of them. But the budget key was the RAW request string, only .lower()'d — so a
malicious relayer (holding ONLY its own key + ONE funder-signed delegation) spends the cap C, then
edits ONLY the delegation_nonce STRING (strip '0x') WITHOUT re-signing, lands a DISTINCT budget bucket,
and drains another C → 2×C, a 100% overspend beyond what the funder authorized. Real FTNS on Base.

Fix: canonicalize the budget key the SAME way the signature does (delegation_budget._key), so every
signature-equivalent spelling collapses to ONE bucket. Money assertions — never weaken.
"""
from __future__ import annotations

import asyncio

import pytest
from eth_account import Account

from prsm.settlement.delegation_budget import (
    DurableDelegationBudgetStore,
    InMemoryDelegationBudgetStore,
)
from prsm.settlement.payment_authorization import (
    canonical_request_hash,
    inference_request_fields,
    sign_payment_authorization,
)
from prsm.settlement.payment_authorization_verifier import (
    AuthorizationRejected,
    RelayerAuthorizationVerifier,
)
from prsm.settlement.payment_delegation import sign_payment_delegation

_CHAIN = 8453
_FUNDER = Account.create()
_RELAYER = Account.create()
_P1 = Account.create()


def _run(coro):
    return asyncio.run(coro)


# ── unit: the budget store collapses signature-equivalent nonce spellings ──


def test_inmemory_budget_collapses_0x_and_bare_nonce():
    store = InMemoryDelegationBudgetStore()
    n0x = "0x" + "ab" * 32
    nbare = "ab" * 32  # SAME funder-signed nonce (identical EIP-712 digest), just no '0x'
    assert store.reserve(n0x, 100, 100) is True  # spend the whole cap once
    assert store.reserve(nbare, 1, 100) is False, (
        "a signature-equivalent nonce spelling got a FRESH cap bucket — relayer drains 2×C")
    assert store.consumed(nbare) == 100  # same bucket as n0x


def test_durable_budget_collapses_case_and_release_hits_same_bucket(tmp_path):
    store = DurableDelegationBudgetStore(tmp_path / "budget.json")
    n0x = "0x" + "cd" * 32
    nUP = "0x" + "CD" * 32  # case variant — same canonical bytes32
    assert store.reserve(n0x, 100, 100) is True
    assert store.reserve(nUP, 1, 100) is False, "case-variant nonce got a fresh cap bucket"
    # a release via the BARE spelling must reclaim from the SAME bucket
    store.release("cd" * 32, 100)
    assert store.consumed(n0x) == 0
    assert store.reserve(nUP, 100, 100) is True  # bucket freed


# ── end-to-end money assertion at the live verifier (mirrors the audit repro) ──


def _rhash():
    return canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="hi", max_tokens=8, privacy_tier="none", content_tier="A"))


def _deleg(*, nonce, max_total):
    return {
        "requester": _FUNDER.address, "relayer": _RELAYER.address, "provider": _P1.address,
        "max_total_spend_wei": max_total, "delegation_nonce": nonce, "expiry_unix": 9999999999,
    }


def _auth(*, job_nonce, max_spend=10 ** 17):
    return {
        "requester": _FUNDER.address, "provider": _P1.address, "max_spend_wei": max_spend,
        "job_nonce": job_nonce, "expiry_unix": 9999999999, "request_hash": _rhash(),
    }


def test_aliased_delegation_nonce_cannot_double_the_cap_end_to_end():
    cap = 10 ** 17  # == the per-request ceiling, so the SECOND request must exceed the cumulative cap
    deleg = _deleg(nonce="0x" + "cd" * 32, max_total=cap)
    dsig = sign_payment_delegation(deleg, _FUNDER.key, chain_id=_CHAIN)
    v = RelayerAuthorizationVerifier(provider_address=_P1.address, chain_id=_CHAIN, now=lambda: 100.0)

    a1 = _auth(job_nonce="0x" + "aa" * 32)
    _run(v.verify(a1, sign_payment_authorization(a1, _RELAYER.key, chain_id=_CHAIN),
                  deleg, dsig, request_hash=a1["request_hash"], quoted_price_wei=10 ** 16))

    # ATTACK: reuse the SAME funder signature; strip ONLY '0x' from the delegation nonce STRING. The
    # chain still verifies (dsig re-canonicalizes to the same bytes32 → recovers the funder), but the
    # budget MUST land in the same bucket (cumulative 2×cap) → rejected, not granted a fresh cap.
    deleg_aliased = _deleg(nonce="cd" * 32, max_total=cap)  # same nonce, no '0x'
    a2 = _auth(job_nonce="0x" + "bb" * 32)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(a2, sign_payment_authorization(a2, _RELAYER.key, chain_id=_CHAIN),
                      deleg_aliased, dsig, request_hash=a2["request_hash"], quoted_price_wei=10 ** 16))
    assert ei.value.reason == "delegation-budget-exceeded", (
        f"aliased-nonce second request was NOT budget-rejected (reason={ei.value.reason!r}) — "
        "the relayer drained a second full cap C past the funder's signed cumulative cap")
