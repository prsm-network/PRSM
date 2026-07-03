"""Sprint 1367 — Tier B/C paid-decrypt PUBLISHER surface, brick 2: POST /content/paid/publish.

The node route makes publish reachable: it uploads the ciphertext (content layer serves it +
registers creator == publisher), then deposits the commitment naming the CAV + retains the wrapped
key. Tested through the real FastAPI route via TestClient, then the retained key + served ciphertext
flow to a consumer decrypt (end to end over the endpoint).
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct
from fastapi.testclient import TestClient

from prsm.enterprise.recipient_encryption import generate_recipient_keypair
from prsm.node.api import create_api_app
from prsm.node.paid_key_serve import PaidKeyStore, paid_key_challenge, serve_paid_key
from prsm.storage.paid_unlock import deserialize_encrypted_content, reconstruct_paid_content

_FEE = 10 ** 18
_CAV = "0x" + "ab" * 20
_PLAIN = b"proprietary dataset published via the node endpoint"


def _app(monkeypatch, store, served):
    monkeypatch.setenv("PRSM_PAID_KEY_SERVE", "1")
    monkeypatch.setenv("PRSM_CONTENT_ACCESS_VERIFIER", _CAV)   # resolves via networks.py override

    import hashlib as _h

    async def _upload(ciphertext, filename, meta=None, creator_eth_address=None):
        ch_hex = _h.sha256(ciphertext).hexdigest()            # same hash publish uses (aligned)
        served[bytes.fromhex(ch_hex)] = ciphertext
        return MagicMock(content_hash=ch_hex)

    uploader = MagicMock()
    uploader.upload = AsyncMock(side_effect=_upload)

    kd = MagicMock()
    kd.deposit_key.return_value = ("0xdep", None)
    kd.get_deposit.return_value = None                        # fresh hash → anti-squat passes
    kd.address = "0x" + "11" * 20

    node = MagicMock()
    node._paid_key_store = store
    node.content_uploader = uploader
    node._paid_publish_key_client = kd
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False), kd


def test_publish_disabled_without_flag(monkeypatch):
    monkeypatch.delenv("PRSM_PAID_KEY_SERVE", raising=False)
    node = MagicMock()
    client = TestClient(create_api_app(node, enable_security=False), raise_server_exceptions=False)
    r = client.post("/content/paid/publish", json={
        "plaintext_b64": base64.b64encode(_PLAIN).decode(), "buyer_x25519_pubkeys": ["x"],
        "fee_wei": _FEE})
    assert r.status_code == 503


def test_publish_deposits_commitment_retains_and_serves(monkeypatch):
    store, served = PaidKeyStore(), {}
    client, kd = _app(monkeypatch, store, served)
    _, buyer_pub = generate_recipient_keypair()

    r = client.post("/content/paid/publish", json={
        "plaintext_b64": base64.b64encode(_PLAIN).decode(),
        "buyer_x25519_pubkeys": [buyer_pub], "fee_wei": _FEE})
    assert r.status_code == 200, r.text
    out = r.json()
    ch = bytes.fromhex(out["content_hash"][2:])

    # deposit named the CAV + carried the 32-byte commitment (not the key)
    _ch, deposited, verifier, fee = kd.deposit_key.call_args[0]
    assert bytes(_ch) == ch and len(deposited) == 32 and verifier == _CAV and fee == _FEE
    assert out["commitment"][2:] == deposited.hex()
    # ciphertext served under the same hash + key retained
    assert ch in served and store.get(ch) is not None
    assert _PLAIN not in served[ch]                            # served bytes are ciphertext


def test_publish_then_consumer_unlocks_end_to_end(monkeypatch):
    store, served = PaidKeyStore(), {}
    client, _kd = _app(monkeypatch, store, served)
    buyer_priv, buyer_pub = generate_recipient_keypair()

    out = client.post("/content/paid/publish", json={
        "plaintext_b64": base64.b64encode(_PLAIN).decode(),
        "buyer_x25519_pubkeys": [buyer_pub], "fee_wei": _FEE}).json()
    ch = bytes.fromhex(out["content_hash"][2:])
    commitment = bytes.fromhex(out["commitment"][2:])

    # a paid consumer fetches the retained key through the real serve gate + decrypts
    from prsm.economy.web3.key_acquisition import fetch_and_verify_wrapped_key
    payer_priv = "0x" + "33" * 32
    payer = Account.from_key(payer_priv).address
    sig = Account.from_key(payer_priv).sign_message(
        encode_defunct(text=paid_key_challenge(ch, "n1"))).signature.hex()
    wrapped = serve_paid_key(ch, "n1", sig, key_store=store,
                             verify_payment=lambda p, c, f: p == payer)
    verified = fetch_and_verify_wrapped_key(lambda _c: wrapped, ch, commitment)
    content = deserialize_encrypted_content(served[ch])
    assert reconstruct_paid_content(verified, buyer_priv, content) == _PLAIN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
