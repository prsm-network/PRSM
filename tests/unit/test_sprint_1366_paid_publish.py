"""Sprint 1366 — Tier B/C paid-decrypt PUBLISHER surface, brick 1: node-side paid publish.

run_paid_publish makes the deployed ContentAccessVerifier usable by publishers: it deposits the
sha256 commitment (naming the CAV) + retains the wrapped key for the serve endpoint, on top of the
reused upload path. The capstone ties publisher → serve → consumer: publish on the node, then the
served ciphertext + retained key flow through the real serve_paid_key gate to a consumer decrypt.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from prsm.economy.web3.key_acquisition import fetch_and_verify_wrapped_key
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.node.paid_key_serve import PaidKeyStore, paid_key_challenge, serve_paid_key
from prsm.node.paid_publish import run_paid_publish
from prsm.storage.paid_unlock import (
    deserialize_encrypted_content,
    key_commitment,
    reconstruct_paid_content,
)

_FEE = 10 ** 18
_CAV = "0x" + "ab" * 20
_PLAIN = b"proprietary dataset published from the operator node"


def _kd():
    kd = MagicMock()
    kd.deposit_key.return_value = ("0xdep", None)
    kd.get_deposit.return_value = None          # fresh hash — not squatted (sp1365 guard passes)
    kd.address = "0x" + "11" * 20
    return kd


def _publish(store, served, buyer_pub, plaintext=_PLAIN):
    return run_paid_publish(
        plaintext=plaintext,
        recipients=[EnterpriseRecipient(identifier="buyer", x25519_pubkey_b64=buyer_pub)],
        fee_wei=_FEE, verifier_address=_CAV, key_client=_kd_shared[0],
        serve_ciphertext=lambda ch, ct: served.__setitem__(bytes(ch), ct),
        paid_key_store=store)


_kd_shared = [None]


def test_deposits_commitment_names_cav_and_retains_key():
    kd = _kd()
    _kd_shared[0] = kd
    store, served = PaidKeyStore(), {}
    _, buyer_pub = generate_recipient_keypair()
    res = _publish(store, served, buyer_pub)

    ch, deposited, verifier, fee = kd.deposit_key.call_args[0]
    assert deposited == res["commitment"] == key_commitment(res["wrapped_key"])  # commitment, not key
    assert len(deposited) == 32 and verifier == _CAV and fee == _FEE
    assert bytes(ch) in served                                   # ciphertext served
    assert store.get(bytes(ch))["wrapped_key"] == res["wrapped_key"]  # key retained for serving
    assert store.get(bytes(ch))["fee_wei"] == _FEE


def test_publish_to_unlock_end_to_end():
    kd = _kd()
    _kd_shared[0] = kd
    store, served = PaidKeyStore(), {}
    buyer_priv, buyer_pub = generate_recipient_keypair()
    res = _publish(store, served, buyer_pub)
    ch, commitment = res["content_hash"], res["commitment"]

    # a party holding only the served ciphertext learns nothing
    assert _PLAIN not in served[bytes(ch)]

    # consumer pays (simulated) → the gated serve hands over the retained key → verify → decrypt
    payer_priv = "0x" + "22" * 32
    payer = Account.from_key(payer_priv).address
    sig = Account.from_key(payer_priv).sign_message(
        encode_defunct(text=paid_key_challenge(ch, "n1"))).signature.hex()
    wrapped = serve_paid_key(ch, "n1", sig, key_store=store,
                             verify_payment=lambda p, c, f: p == payer and bytes(c) == bytes(ch))
    verified = fetch_and_verify_wrapped_key(lambda _c: wrapped, ch, commitment)
    content = deserialize_encrypted_content(served[bytes(ch)])
    assert reconstruct_paid_content(verified, buyer_priv, content) == _PLAIN


def test_non_buyer_cannot_decrypt_published_content():
    kd = _kd()
    _kd_shared[0] = kd
    store, served = PaidKeyStore(), {}
    _buyer_priv, buyer_pub = generate_recipient_keypair()
    other_priv, _ = generate_recipient_keypair()
    res = _publish(store, served, buyer_pub)
    ch = res["content_hash"]

    entry = store.get(bytes(ch))
    content = deserialize_encrypted_content(served[bytes(ch)])
    from prsm.storage.paid_unlock import PaidUnlockError
    with pytest.raises(PaidUnlockError, match="unwrap"):
        reconstruct_paid_content(entry["wrapped_key"], other_priv, content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_large_file_streaming_publish_to_unlock_end_to_end(tmp_path):
    # ★ sp1458 — the full LARGE-FILE paid flow: stream-publish a big dataset from the node → serve the
    # ciphertext FILE → payment-gated key release → stream-decrypt back to byte-identical plaintext.
    from pathlib import Path
    from prsm.node.paid_publish import run_paid_publish_from_path
    from prsm.storage.paid_unlock import reconstruct_paid_content_from_file

    kd = _kd()
    store = PaidKeyStore()
    served = {}   # content_hash -> ciphertext file path
    buyer_priv, buyer_pub = generate_recipient_keypair()
    plaintext = (b"large proprietary dataset row; " * 100_000) + b"tail"
    src = tmp_path / "plain.bin"
    src.write_bytes(plaintext)
    ct = tmp_path / "cipher.bin"

    res = run_paid_publish_from_path(
        plaintext_path=src, ciphertext_path=ct,
        recipients=[EnterpriseRecipient(identifier="buyer", x25519_pubkey_b64=buyer_pub)],
        fee_wei=_FEE, verifier_address=_CAV, key_client=kd,
        serve_ciphertext_from_path=lambda ch, p: served.__setitem__(bytes(ch), str(p)),
        paid_key_store=store)
    ch, commitment = res["content_hash"], res["commitment"]

    # commitment (not key) deposited; key retained for the gated serve; ciphertext FILE served.
    _dch, deposited, verifier, fee = kd.deposit_key.call_args[0]
    assert deposited == commitment == key_commitment(res["wrapped_key"])
    assert verifier == _CAV and fee == _FEE
    assert store.get(bytes(ch))["wrapped_key"] == res["wrapped_key"]
    assert bytes(ch) in served
    assert plaintext[:32] not in Path(served[bytes(ch)]).read_bytes()   # served bytes are encrypted

    # consumer pays → gated serve hands over the retained key → verify vs commitment → stream-decrypt.
    payer_priv = "0x" + "22" * 32
    payer = Account.from_key(payer_priv).address
    sig = Account.from_key(payer_priv).sign_message(
        encode_defunct(text=paid_key_challenge(ch, "n1"))).signature.hex()
    wrapped = serve_paid_key(ch, "n1", sig, key_store=store,
                             verify_payment=lambda p, c, f: p == payer and bytes(c) == bytes(ch))
    verified = fetch_and_verify_wrapped_key(lambda _c: wrapped, ch, commitment)
    out = tmp_path / "recovered.bin"
    reconstruct_paid_content_from_file(verified, buyer_priv, served[bytes(ch)], out)
    assert out.read_bytes() == plaintext
