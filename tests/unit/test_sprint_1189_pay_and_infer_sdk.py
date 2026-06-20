"""Sprint 1189 — user-facing pay-for-inference SDK (day-one-live blocker #4).

The settlement rail + requester-payment verifier are live, but a user had no SDK/CLI to
deposit FTNS to escrow or sign the EIP-712 PaymentAuthorization — they'd hand-craft the
typed data + request_hash. This adds PRSMClient.pay_and_infer (discover operator payee →
build+sign auth bound to the exact request → POST with payment_authorization) and
deposit_escrow (fund the escrow once), wrapping the existing primitives.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from eth_account import Account

from prsm.sdk.client import PRSMClient


def _run(coro):
    return asyncio.run(coro)


# A deterministic test requester key (testnet-only; never a real funded key).
_REQ_KEY = "0x" + "11" * 32
_REQ_ADDR = Account.from_key(_REQ_KEY).address
_PROVIDER = "0x" + "22" * 20


class _CapturingClient(PRSMClient):
    """Stubs the HTTP layer to capture the posted body + serve /info."""
    def __init__(self, *, info=None, post_result=None):
        super().__init__()
        self._info = info if info is not None else {"operator_address": _PROVIDER}
        self._post_result = post_result or {"success": True, "output": " ok"}
        self.posted_path = None
        self.posted_body = None
        self.got_path = None

    async def _get(self, path):
        self.got_path = path
        return self._info

    async def _post(self, path, data):
        self.posted_path = path
        self.posted_body = data
        return self._post_result


# ── pay_and_infer ────────────────────────────────────────────────────────────────────

def test_pay_and_infer_signs_auth_bound_to_request_and_posts():
    c = _CapturingClient()
    res = _run(c.pay_and_infer(
        "hello", requester_key=_REQ_KEY, provider_address=_PROVIDER,
        model_id="gpt2", max_tokens=8, budget_ftns=1.0, chain_id=84532,
        expiry_unix=9999999999))
    assert res["success"] is True
    assert c.posted_path == "/compute/inference"
    body = c.posted_body
    # the request fields are present + the signed authorization is attached
    assert body["prompt"] == "hello" and body["model_id"] == "gpt2"
    auth = body["payment_authorization"]
    assert auth["payload"]["requester"] == _REQ_ADDR
    assert auth["payload"]["provider"] == _PROVIDER
    assert auth["signature"].startswith("0x")
    # max_spend defaults to budget_ftns → 1 FTNS in wei
    assert auth["payload"]["max_spend_wei"] == 10**18


def test_pay_and_infer_discovers_operator_address_from_info():
    c = _CapturingClient(info={"operator_address": _PROVIDER})
    _run(c.pay_and_infer("hi", requester_key=_REQ_KEY, expiry_unix=9999999999,
                         chain_id=84532))
    assert c.got_path == "/info"  # discovered
    assert c.posted_body["payment_authorization"]["payload"]["provider"] == _PROVIDER


def test_pay_and_infer_errors_when_no_operator_address():
    c = _CapturingClient(info={})  # operator published no payee
    with pytest.raises(ValueError, match="operator"):
        _run(c.pay_and_infer("hi", requester_key=_REQ_KEY))


def test_pay_and_infer_request_hash_matches_provider_recompute():
    """The auth's request_hash must equal what the server recomputes from the SAME body
    fields — otherwise the provider rejects (request-hash mismatch)."""
    from prsm.settlement.payment_authorization import (
        canonical_request_hash, inference_request_fields,
    )
    c = _CapturingClient()
    _run(c.pay_and_infer("bind me", requester_key=_REQ_KEY, provider_address=_PROVIDER,
                         model_id="gpt2", max_tokens=5, privacy_tier="none",
                         content_tier="A", chain_id=84532, expiry_unix=9999999999))
    body = c.posted_body
    expected = canonical_request_hash(inference_request_fields(
        model_id=body["model_id"], prompt=body["prompt"],
        max_tokens=int(body["max_tokens"] or 0),
        privacy_tier=body["privacy_tier"], content_tier=body["content_tier"]))
    assert body["payment_authorization"]["payload"]["request_hash"] == expected


# ── deposit_escrow ─────────────────────────────────────────────────────────────────────

class _FakeEscrow:
    def __init__(self):
        self.deposited_wei = None
    async def deposit(self, amount_wei):
        self.deposited_wei = amount_wei
        return "0xdeposittx"


def test_deposit_escrow_converts_ftns_to_wei_and_deposits():
    c = PRSMClient()
    fake = _FakeEscrow()
    tx = _run(c.deposit_escrow(requester_key=_REQ_KEY, amount_ftns="2.5", _client=fake))
    assert tx == "0xdeposittx"
    assert fake.deposited_wei == int(Decimal("2.5") * (10**18))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
