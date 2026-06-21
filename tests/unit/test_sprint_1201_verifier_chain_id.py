"""Sprint 1201 — the requester-payment verifier must use the NODE'S chainId.

LIVE BUG (caught by the sp1200 front-door harness on Base Sepolia): the CLI signs a
PaymentAuthorization over an EIP-712 domain that includes chainId (84532 on testnet),
but build_payment_verifier_or_none constructed the verifier with no chain_id, so
verify_payment_authorization_signature fell back to DEFAULT_CHAIN_ID=8453 (Base
mainnet). The verifier then recomputed the digest with the WRONG chainId and recovered
a different address -> every testnet-signed auth was rejected with "signer-mismatch".

Fix: build_payment_verifier_or_none / build_relayer_verifier_or_none resolve the active
network's chainId (resolve_endpoints) and pass it to the verifier, so signer + verifier
agree. The sp1054-1058 tests never caught this because they round-tripped sign+verify
with the SAME chain_id.
"""
from __future__ import annotations

import pytest

from prsm.settlement.client_wiring import (
    build_payment_verifier_or_none,
    build_relayer_verifier_or_none,
)
from prsm.settlement.payment_client import build_payment_authorization
from prsm.settlement.payment_authorization import (
    canonical_request_hash,
    inference_request_fields,
)
from prsm.settlement.payment_authorization_verifier import AuthorizationRejected

_REQ_KEY = "0x" + "11" * 32  # test requester key
_FAR_FUTURE = 9_999_999_999
_TESTNET_CHAIN = 84532
_MAINNET_CHAIN = 8453


def _provider_address():
    from eth_account import Account
    return Account.from_key(_REQ_KEY).address


def _verifier(network):
    env = {
        "PRSM_REQUESTER_PAYMENT": "1",
        "PRSM_NETWORK": network,
        "PRSM_PAYMENT_NONCE_FILE": ":memory:",
    }
    return build_payment_verifier_or_none(env, provider_address=_provider_address())


# ── the fix: verifier chain_id == the node's network chain ───────────────────────────

def test_testnet_verifier_uses_84532():
    v = _verifier("testnet")
    assert v is not None
    assert v._chain_id == _TESTNET_CHAIN


def test_mainnet_verifier_uses_8453():
    v = _verifier("mainnet")
    assert v is not None
    assert v._chain_id == _MAINNET_CHAIN


def test_relayer_verifier_also_pins_chain_id():
    env = {
        "PRSM_REQUESTER_PAYMENT": "1", "PRSM_NETWORK": "testnet",
        "PRSM_RELAYER_NONCE_FILE": ":memory:", "PRSM_DELEGATION_BUDGET_FILE": ":memory:",
    }
    v = build_relayer_verifier_or_none(env, provider_address=_provider_address())
    assert v is not None and v._chain_id == _TESTNET_CHAIN


# ── the live-bug reproduction: testnet-signed auth verifies on a testnet node ─────────

def _build_auth(chain_id):
    provider = _provider_address()  # self-pay: provider == requester
    return build_payment_authorization(
        requester_key=_REQ_KEY, provider_address=provider,
        model_id="gpt2", prompt="hi", max_tokens=8,
        privacy_tier="none", content_tier="A",
        max_spend_ftns=1.0, expiry_unix=_FAR_FUTURE, chain_id=chain_id)


def _request_hash():
    return canonical_request_hash(inference_request_fields(
        model_id="gpt2", prompt="hi", max_tokens=8,
        privacy_tier="none", content_tier="A"))


async def test_testnet_signed_auth_accepted_by_testnet_verifier():
    # This is the exact flow that FAILED live before the fix.
    auth = _build_auth(_TESTNET_CHAIN)
    verifier = _verifier("testnet")
    requester = await verifier.verify(
        auth["payload"], auth["signature"],
        request_hash=_request_hash(), quoted_price_wei=int(0.5 * 10 ** 18))
    assert requester.lower() == _provider_address().lower()


async def test_testnet_signed_auth_rejected_by_mainnet_verifier():
    # The chain binding works both ways: an 84532-signed auth must NOT verify on a
    # node that recovers with 8453 (proves the digest really is chain-bound).
    auth = _build_auth(_TESTNET_CHAIN)
    verifier = _verifier("mainnet")
    with pytest.raises(AuthorizationRejected) as ei:
        await verifier.verify(
            auth["payload"], auth["signature"],
            request_hash=_request_hash(), quoted_price_wei=int(0.5 * 10 ** 18))
    assert ei.value.reason == "signer-mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
