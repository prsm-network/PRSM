"""Sprint 1311 — relayer / sponsored payment for wallet-less users.

A FUNDER (gateway/sponsor) issues ONE PaymentDelegation authorizing a RELAYER to sign
per-request auths on its behalf; the relayer then runs inference for END-USERS who hold
no wallet. The provider verifies the two-signature chain and settles from the funder's
escrow. The server already wires resolve_relayer_requester (api.py:780); these tests pin
the missing CLIENT pieces: build_payment_delegation (funder) + PRSMClient.relayed_infer
(relayer).
"""
from __future__ import annotations

import pytest
from eth_account import Account

from prsm.settlement.payment_client import build_payment_delegation
from prsm.settlement.payment_delegation import verify_payment_delegation_signature

_RELAYER = "0x" + "22" * 20
_PROVIDER = "0x" + "33" * 20


# ── build_payment_delegation (funder side) ───────────────────────────────────

def test_delegation_shape_and_fields():
    funder = Account.create()
    d = build_payment_delegation(
        funder_key=funder.key.hex(), relayer_address=_RELAYER,
        max_total_spend_ftns=10, expiry_unix=9999999999, chain_id=84532)
    assert set(d) == {"payload", "signature"}
    p = d["payload"]
    assert p["requester"].lower() == funder.address.lower()   # funder pays
    assert p["relayer"].lower() == _RELAYER.lower()
    assert p["max_total_spend_wei"] == 10 * 10 ** 18
    assert p["expiry_unix"] == 9999999999
    assert p["delegation_nonce"].startswith("0x") and len(p["delegation_nonce"]) == 66
    assert d["signature"].startswith("0x")


def test_delegation_signature_recovers_funder():
    funder = Account.create()
    d = build_payment_delegation(
        funder_key=funder.key.hex(), relayer_address=_RELAYER,
        max_total_spend_ftns=5, expiry_unix=9999999999, chain_id=8453)
    rec = verify_payment_delegation_signature(d["payload"], d["signature"], chain_id=8453)
    assert rec.lower() == funder.address.lower()


def test_delegation_chain_id_bound():
    """Signed for chain 8453 must NOT recover the funder under 84532 (domain binds chainId)."""
    funder = Account.create()
    d = build_payment_delegation(
        funder_key=funder.key.hex(), relayer_address=_RELAYER,
        max_total_spend_ftns=1, expiry_unix=9999999999, chain_id=8453)
    rec = verify_payment_delegation_signature(d["payload"], d["signature"], chain_id=84532)
    assert rec.lower() != funder.address.lower()


def test_delegation_nonce_random_per_call_but_explicit_honored():
    funder = Account.create()
    d1 = build_payment_delegation(funder_key=funder.key.hex(), relayer_address=_RELAYER,
                                  max_total_spend_ftns=1, expiry_unix=9999999999)
    d2 = build_payment_delegation(funder_key=funder.key.hex(), relayer_address=_RELAYER,
                                  max_total_spend_ftns=1, expiry_unix=9999999999)
    assert d1["payload"]["delegation_nonce"] != d2["payload"]["delegation_nonce"]
    fixed = "0x" + "ab" * 32
    d3 = build_payment_delegation(funder_key=funder.key.hex(), relayer_address=_RELAYER,
                                  max_total_spend_ftns=1, expiry_unix=9999999999,
                                  delegation_nonce=fixed)
    assert d3["payload"]["delegation_nonce"] == fixed


# ── PRSMClient.relayed_infer (relayer side) ──────────────────────────────────

@pytest.mark.asyncio
async def test_relayed_infer_sends_auth_and_delegation():
    from prsm.sdk.client import PRSMClient
    funder, relayer = Account.create(), Account.create()
    delegation = build_payment_delegation(
        funder_key=funder.key.hex(), relayer_address=relayer.address,
        max_total_spend_ftns=10, expiry_unix=9999999999, chain_id=84532)

    captured = {}

    async def _fake_post(path, body):
        captured["path"] = path
        captured["body"] = body
        return {"success": True, "output": "hi", "ftns_charged": "0.1", "receipt": {}}

    c = PRSMClient("http://node:8000")
    c._post = _fake_post  # type: ignore

    result = await c.relayed_infer(
        prompt="Hi", relayer_key=relayer.key.hex(), payment_delegation=delegation,
        provider_address=_PROVIDER, chain_id=84532)
    assert result["output"] == "hi"
    assert captured["path"] == "/compute/inference"
    body = captured["body"]
    # BOTH blobs present
    assert "payment_authorization" in body and "payment_delegation" in body
    assert body["payment_delegation"] is delegation
    # the per-request auth is signed by the RELAYER (auth.requester == relayer)
    assert body["payment_authorization"]["payload"]["requester"].lower() \
        == relayer.address.lower()
    # the delegation names funder->relayer
    assert body["payment_delegation"]["payload"]["relayer"].lower() \
        == relayer.address.lower()


@pytest.mark.asyncio
async def test_relayed_infer_verify_pubkey_sets_receipt_verified():
    from prsm.sdk.client import PRSMClient
    funder, relayer = Account.create(), Account.create()
    delegation = build_payment_delegation(
        funder_key=funder.key.hex(), relayer_address=relayer.address,
        max_total_spend_ftns=10, expiry_unix=9999999999)

    async def _fake_post(path, body):
        return {"success": True, "output": "x", "ftns_charged": "0.1", "receipt": {}}

    c = PRSMClient("http://node:8000")
    c._post = _fake_post  # type: ignore
    result = await c.relayed_infer(
        prompt="Hi", relayer_key=relayer.key.hex(), payment_delegation=delegation,
        provider_address=_PROVIDER, verify_pubkey_b64="QUJD")
    assert "receipt_verified" in result and isinstance(result["receipt_verified"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
