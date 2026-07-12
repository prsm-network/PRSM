"""sp1437 (money-path audit #3) — bind a PaymentDelegation to a single provider.

The gap this closes
-------------------
A ``PaymentDelegation`` caps cumulative spend via ``max_total_spend_wei``, but that cap is
enforced by a PER-NODE budget store (``DelegationBudgetStore``) keyed only by
``delegation_nonce``. The per-request ``PaymentAuthorization`` is pinned to a provider
(``auth.provider == this_node``), but the DELEGATION was provider-AGNOSTIC. So a relayer
holding one funder-signed delegation with cap C could mint a fresh per-request auth for each
of N different providers and present ``(auth_for_Pi, delegation)`` to each Pi: every node
independently verifies the chain and reserves up to C against its OWN budget store (which
starts at consumed=0), draining the funder's escrow up to C×N — N times the cap the funder
actually signed.

The fix binds the delegation to a specific ``provider`` inside the signed EIP-712 struct and
makes ``verify_delegated_authorization`` reject any auth whose provider differs from the
delegation's. Combined with the verifier's existing ``auth.provider == this_node`` pin, a
delegation naming P1 is spendable ONLY at P1 (whose per-node store then correctly caps it at
C). To fund N providers a funder signs N delegations, each with its own cap — so total
exposure is exactly the sum of the caps the funder signed, never a multiple of one.

These are money assertions (CLAUDE.md: never weaken to pass). A red here means the relayer
path can be drained past the signed cap.
"""
from __future__ import annotations

import pytest
from eth_account import Account

from prsm.settlement.payment_authorization import (
    canonical_request_hash,
    inference_request_fields,
    sign_payment_authorization,
)
from prsm.settlement.payment_delegation import (
    DelegatedAuthError,
    sign_payment_delegation,
    verify_delegated_authorization,
)
from prsm.settlement.payment_authorization_verifier import (
    AuthorizationRejected,
    RelayerAuthorizationVerifier,
)

_CHAIN = 8453
_FUNDER = Account.create()
_RELAYER = Account.create()
_P1 = Account.create()
_P2 = Account.create()


def _rhash():
    return canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="hi", max_tokens=8, privacy_tier="none", content_tier="A"))


def _deleg(provider, *, max_total=10**18, nonce="0x" + "cd" * 32, expiry=9999999999):
    return {
        "requester": _FUNDER.address,
        "relayer": _RELAYER.address,
        "provider": provider,
        "max_total_spend_wei": max_total,
        "delegation_nonce": nonce,
        "expiry_unix": expiry,
    }


def _auth(provider, *, max_spend=10**17, job_nonce="0x" + "ab" * 32, expiry=9999999999, rhash=None):
    return {
        "requester": _FUNDER.address,
        "provider": provider,
        "max_spend_wei": max_spend,
        "job_nonce": job_nonce,
        "expiry_unix": expiry,
        "request_hash": rhash or _rhash(),
    }


def _sign_auth(auth):
    return sign_payment_authorization(auth, _RELAYER.key, chain_id=_CHAIN)


def _sign_deleg(deleg):
    return sign_payment_delegation(deleg, _FUNDER.key, chain_id=_CHAIN)


def _verifier(provider):
    return RelayerAuthorizationVerifier(
        provider_address=provider, chain_id=_CHAIN, now=lambda: 100.0)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# ── 1. The money shot: one delegation can't be spent at two providers ────────


def test_delegation_bound_to_one_provider_cannot_be_spent_at_another():
    """The C×N drain, blocked. A delegation naming P1 verifies at P1 but a different
    provider P2 (its own verifier + own budget store) rejects the SAME delegation."""
    deleg = _deleg(_P1.address)
    dsig = _sign_deleg(deleg)

    # At P1 with an auth naming P1 → accepted.
    a1 = _auth(_P1.address, job_nonce="0x" + "11" * 32)
    funder = _run(_verifier(_P1.address).verify(
        a1, _sign_auth(a1), deleg, dsig,
        request_hash=a1["request_hash"], quoted_price_wei=10**16))
    assert funder.lower() == _FUNDER.address.lower()

    # At P2 the relayer mints an auth naming P2 and presents the SAME P1 delegation.
    # P2's node rejects it — the delegation was never authorized for P2.
    a2 = _auth(_P2.address, job_nonce="0x" + "22" * 32)
    with pytest.raises(AuthorizationRejected):
        _run(_verifier(_P2.address).verify(
            a2, _sign_auth(a2), deleg, dsig,
            request_hash=a2["request_hash"], quoted_price_wei=10**16))


# ── 2. The primitive: provider binding is enforced + signed ──────────────────


def test_primitive_rejects_auth_provider_not_matching_delegation():
    deleg = _deleg(_P1.address)
    dsig = _sign_deleg(deleg)
    a2 = _auth(_P2.address)  # auth for a DIFFERENT provider than the delegation binds
    with pytest.raises(DelegatedAuthError):
        verify_delegated_authorization(
            auth_payload=a2, auth_signature=_sign_auth(a2),
            delegation_payload=deleg, delegation_signature=dsig,
            chain_id=_CHAIN, now=lambda: 100.0)


def test_primitive_accepts_matching_provider():
    deleg = _deleg(_P1.address)
    a1 = _auth(_P1.address)
    funder = verify_delegated_authorization(
        auth_payload=a1, auth_signature=_sign_auth(a1),
        delegation_payload=deleg, delegation_signature=_sign_deleg(deleg),
        chain_id=_CHAIN, now=lambda: 100.0)
    assert funder.lower() == _FUNDER.address.lower()


def test_delegation_without_provider_is_rejected_fail_closed():
    """A provider-less delegation must not be honored — that IS the pre-fix wildcard hole.
    Missing the field is a hard reject, never a silent 'any provider'."""
    deleg = _deleg(_P1.address)
    deleg.pop("provider")
    a1 = _auth(_P1.address)
    with pytest.raises((DelegatedAuthError, ValueError, KeyError)):
        verify_delegated_authorization(
            auth_payload=a1, auth_signature=_sign_auth(a1),
            delegation_payload=deleg, delegation_signature=_sign_deleg(_deleg(_P1.address)),
            chain_id=_CHAIN, now=lambda: 100.0)


def test_provider_is_part_of_the_signed_struct():
    """Signing a delegation for P1 then swapping provider→P2 in the payload must break the
    signature recovery (the funder is no longer the recovered signer) — proving provider is
    inside the EIP-712 digest, not an unsigned tag a relayer could flip."""
    signed_for_p1 = _deleg(_P1.address)
    dsig = _sign_deleg(signed_for_p1)
    tampered = dict(signed_for_p1, provider=_P2.address)
    a2 = _auth(_P2.address)
    with pytest.raises(DelegatedAuthError):
        verify_delegated_authorization(
            auth_payload=a2, auth_signature=_sign_auth(a2),
            delegation_payload=tampered, delegation_signature=dsig,
            chain_id=_CHAIN, now=lambda: 100.0)


# ── 3. Regression: within one provider the cumulative cap still holds ─────────


def test_cumulative_cap_still_enforced_within_one_provider():
    """The fix must not disturb the sp1092 per-provider cumulative cap. cap=1.5×per-request:
    the first request reserves the ceiling, the second exceeds the delegation budget."""
    deleg = _deleg(_P1.address, max_total=15 * 10**16)  # 0.15 FTNS; per-request ceiling 0.1
    dsig = _sign_deleg(deleg)
    v = _verifier(_P1.address)

    a1 = _auth(_P1.address, job_nonce="0x" + "aa" * 32)
    _run(v.verify(a1, _sign_auth(a1), deleg, dsig,
                  request_hash=a1["request_hash"], quoted_price_wei=10**16))

    a2 = _auth(_P1.address, job_nonce="0x" + "bb" * 32)
    with pytest.raises(AuthorizationRejected) as ei:
        _run(v.verify(a2, _sign_auth(a2), deleg, dsig,
                      request_hash=a2["request_hash"], quoted_price_wei=10**16))
    assert ei.value.reason == "delegation-budget-exceeded"
