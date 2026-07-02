"""Sprint 1360 — F1 redesign R4: paid-key store + end-to-end through the real HTTP endpoint.

Wires the retained-key store to GET /content/paid-key and proves the whole binding-gate over real
HTTP: an unpaid fetcher is refused (402), a paid + authenticated fetcher gets the wrapped key, and
publish → retain → serve → fetch+verify → decrypt composes end to end.
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.paid_key_serve import (
    PaidKeyStore,
    build_paid_key_verify_payment,
    paid_key_challenge,
)

_CH = bytes.fromhex("cd" * 32)
_FEE = 10 ** 18
_PRIV = "0x" + "11" * 32
_PAYER = Account.from_key(_PRIV).address


def _sign(content_hash=_CH, nonce="n1", priv=_PRIV):
    return Account.from_key(priv).sign_message(
        encode_defunct(text=paid_key_challenge(content_hash, nonce))).signature.hex()


def _app(store, verify_payment):
    node = MagicMock()
    node._paid_key_store = store
    node._paid_key_verify_payment = verify_payment
    return TestClient(create_api_app(node, enable_security=False), raise_server_exceptions=False)


# ── store ─────────────────────────────────────────────────────────────────────

def test_store_put_get():
    s = PaidKeyStore()
    s.put(_CH, b"WK", _FEE)
    assert s.get(_CH) == {"wrapped_key": b"WK", "fee_wei": _FEE}
    assert s.get(b"z" * 32) is None
    assert len(s) == 1


def test_build_verify_payment_adapts_client():
    vc = MagicMock()
    vc.verify_payment.return_value = True
    vp = build_paid_key_verify_payment(vc)
    assert vp(_PAYER, _CH, _FEE) is True
    vc.verify_payment.assert_called_once_with(_PAYER, _CH, _FEE)


# ── the real endpoint ─────────────────────────────────────────────────────────

def test_endpoint_disabled_without_flag(monkeypatch):
    monkeypatch.delenv("PRSM_PAID_KEY_SERVE", raising=False)
    store = PaidKeyStore()
    store.put(_CH, b"WK", _FEE)
    client = _app(store, lambda *a: True)
    r = client.get(f"/content/paid-key/0x{_CH.hex()}",
                   params={"nonce": "n1", "signature": _sign()})
    assert r.status_code == 503


def test_endpoint_gates_unpaid_then_serves_paid(monkeypatch):
    monkeypatch.setenv("PRSM_PAID_KEY_SERVE", "1")
    store = PaidKeyStore()
    store.put(_CH, b"THE-WRAPPED-KEY", _FEE)
    paid = {"ok": False}
    client = _app(store, lambda payer, ch, fee: (
        paid["ok"] and payer == _PAYER and bytes(ch) == _CH and fee == _FEE))
    url = f"/content/paid-key/0x{_CH.hex()}"
    params = {"nonce": "n1", "signature": _sign()}

    r = client.get(url, params=params)                 # unpaid → 402
    assert r.status_code == 402

    paid["ok"] = True
    r = client.get(url, params=params)                 # paid → served
    assert r.status_code == 200
    assert base64.b64decode(r.json()["wrapped_key_b64"]) == b"THE-WRAPPED-KEY"


def test_endpoint_unknown_content_404(monkeypatch):
    monkeypatch.setenv("PRSM_PAID_KEY_SERVE", "1")
    client = _app(PaidKeyStore(), lambda *a: True)     # empty store
    r = client.get(f"/content/paid-key/0x{_CH.hex()}",
                   params={"nonce": "n1", "signature": _sign()})
    assert r.status_code == 404


# ── ★ full: publish → retain → serve → fetch+verify → decrypt ─────────────────

def test_publish_to_endpoint_to_decrypt(monkeypatch):
    monkeypatch.setenv("PRSM_PAID_KEY_SERVE", "1")
    from prsm.economy.paid_content import publish_paid_content
    from prsm.economy.web3.key_acquisition import fetch_and_verify_wrapped_key
    from prsm.enterprise.recipient_encryption import (
        EnterpriseRecipient, generate_recipient_keypair,
    )
    from prsm.storage.paid_unlock import (
        deserialize_encrypted_content, reconstruct_paid_content,
    )

    store = PaidKeyStore()
    kd = MagicMock()
    kd.deposit_key.return_value = ("0xdep", None)
    buyer_priv, buyer_pub = generate_recipient_keypair()

    res = publish_paid_content(
        plaintext=b"proprietary rows served over HTTP",
        recipients=[EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)],
        royalty_verifier_address="0x" + "ab" * 20, release_fee_ftns_wei=_FEE, key_client=kd,
        publish_ciphertext=lambda ch, ct: None,
        retain_wrapped_key=lambda ch, wk, fee: store.put(ch, wk, fee))
    ch = res["content_hash"]

    # a PAID buyer fetches the wrapped key through the real endpoint, verifies vs commitment,
    # and decrypts the (separately-held) ciphertext.
    client = _app(store, lambda payer, c, fee: payer == Account.from_key(_PRIV).address)
    resp = client.get(f"/content/paid-key/0x{ch.hex()}",
                      params={"nonce": ch.hex()[:16], "signature": _sign(ch, ch.hex()[:16])})
    assert resp.status_code == 200
    wrapped = base64.b64decode(resp.json()["wrapped_key_b64"])

    verified = fetch_and_verify_wrapped_key(lambda _c: wrapped, ch, res["commitment"])
    content = deserialize_encrypted_content(res["ciphertext"])
    assert reconstruct_paid_content(verified, buyer_priv, content) == b"proprietary rows served over HTTP"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
