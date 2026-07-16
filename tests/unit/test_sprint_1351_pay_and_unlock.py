"""Sprint 1351 (+1359 F1 redesign) — brick 3: pay -> unlock orchestration.

Composes fee-settlement (injectable) + brick 2 (fetch the wrapped key off-chain + verify vs the
on-chain commitment) + retrieval + brick 1 (reconstruct). Tested end-to-end with REAL crypto; the
wrapped key comes from an injected fetch (simulating the payment-gated serve endpoint, sp1358).
"""
from __future__ import annotations

import pytest

from prsm.economy.paid_content import pay_and_unlock
from prsm.economy.web3.key_acquisition import (
    KeyCommitmentMismatchError,
    KeyNotReleasedError,
)
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import (
    PaidUnlockError,
    key_commitment,
    wrap_content_key_for_deposit,
)

_CH = bytes.fromhex("cd" * 32)


def _fixture(plaintext=b"paywalled dataset rows"):
    content_key = generate_key()
    content = encrypt(plaintext, content_key)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="buyer", x25519_pubkey_b64=buyer_pub)])
    return content, wrapped, key_commitment(wrapped), buyer_priv, plaintext


def _call(content, wrapped, commitment, buyer_priv, *, settle_fee=None,
          fetch=None, retrieve=None):
    return pay_and_unlock(
        content_hash=_CH, recipient_privkey_b64=buyer_priv, commitment=commitment,
        fetch_wrapped_key=fetch if fetch is not None else (lambda ch: wrapped),
        retrieve_content=retrieve if retrieve is not None else (lambda ch: content),
        settle_fee=settle_fee)


# ── happy path ────────────────────────────────────────────────────────────────

def test_full_flow_pays_then_fetches():
    content, wrapped, commitment, buyer_priv, plaintext = _fixture()
    order = []
    out = _call(content, wrapped, commitment, buyer_priv,
                settle_fee=lambda: order.append("pay"),
                fetch=lambda ch: (order.append("fetch"), wrapped)[1])
    assert out == plaintext
    assert order == ["pay", "fetch"]                   # fee settled BEFORE fetching the key


def test_already_paid_no_settle():
    content, wrapped, commitment, buyer_priv, plaintext = _fixture()
    out = _call(content, wrapped, commitment, buyer_priv, settle_fee=None)
    assert out == plaintext


def test_authoritative_commitment_read_after_payment():
    # sp1363 (R5 MEDIUM): when no commitment is supplied, it's read via fetch_commitment AFTER
    # settle (the on-chain release is gated on payment) — never trusted from the serve path.
    content, wrapped, commitment, buyer_priv, plaintext = _fixture()
    order = []
    out = pay_and_unlock(
        content_hash=_CH, recipient_privkey_b64=buyer_priv,
        fetch_wrapped_key=lambda ch: wrapped, retrieve_content=lambda ch: content,
        fetch_commitment=lambda: (order.append("read-commitment"), commitment)[1],
        settle_fee=lambda: order.append("pay"))
    assert out == plaintext
    assert order == ["pay", "read-commitment"]         # commitment read AFTER payment


def test_missing_both_commitment_and_fetch_raises():
    content, wrapped, _commitment, buyer_priv, _ = _fixture()
    with pytest.raises(PaidUnlockError, match="commitment or fetch_commitment"):
        pay_and_unlock(content_hash=_CH, recipient_privkey_b64=buyer_priv,
                       fetch_wrapped_key=lambda ch: wrapped, retrieve_content=lambda ch: content)


# ── fail-loud at each stage ───────────────────────────────────────────────────

def test_unpaid_fetch_returns_nothing_surfaces_key_not_released():
    content, wrapped, commitment, buyer_priv, _ = _fixture()
    # the payment-gated endpoint serves nothing to an unpaid fetcher
    with pytest.raises(KeyNotReleasedError, match="no key"):
        _call(content, wrapped, commitment, buyer_priv, fetch=lambda ch: None)


def test_wrong_served_key_fails_commitment_check():
    content, wrapped, commitment, buyer_priv, _ = _fixture()
    with pytest.raises(KeyCommitmentMismatchError, match="WRONG key"):
        _call(content, wrapped, commitment, buyer_priv, fetch=lambda ch: b"a substituted key")


def test_ciphertext_not_retrievable_raises():
    content, wrapped, commitment, buyer_priv, _ = _fixture()
    with pytest.raises(PaidUnlockError, match="not retrievable"):
        _call(content, wrapped, commitment, buyer_priv, retrieve=lambda ch: None)


def test_wrong_buyer_key_raises():
    content, wrapped, commitment, _buyer_priv, _ = _fixture()
    other_priv, _ = generate_recipient_keypair()
    with pytest.raises(PaidUnlockError, match="unwrap"):
        _call(content, wrapped, commitment, other_priv)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── sp1458: pay_and_unlock_to_file (streaming large-file consumer) ─────────────

def test_pay_and_unlock_to_file_streams_end_to_end(tmp_path):
    from prsm.economy.paid_content import (
        build_paid_content_from_path, pay_and_unlock_to_file)
    plaintext = (b"paywalled large dataset row; " * 100_000) + b"tail"
    src = tmp_path / "plain.bin"
    src.write_bytes(plaintext)
    ct = tmp_path / "cipher.bin"
    buyer_priv, buyer_pub = generate_recipient_keypair()
    built = build_paid_content_from_path(
        src, ct, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])

    order = []
    out = pay_and_unlock_to_file(
        content_hash=built["content_hash"], recipient_privkey_b64=buyer_priv,
        commitment=built["commitment"],
        fetch_wrapped_key=lambda ch: (order.append("fetch"), built["wrapped_key"])[1],
        retrieve_content_to_file=lambda ch: (order.append("retrieve"), str(ct))[1],
        dest_path=tmp_path / "out.bin",
        settle_fee=lambda: order.append("pay"))
    # money flow preserved: pay BEFORE fetching the key BEFORE retrieving the ciphertext
    assert order == ["pay", "fetch", "retrieve"]
    assert out == tmp_path / "out.bin"
    assert (tmp_path / "out.bin").read_bytes() == plaintext   # streamed decrypt, byte-identical


def test_pay_and_unlock_to_file_wrong_buyer_key_fails_loud(tmp_path):
    from prsm.economy.paid_content import (
        build_paid_content_from_path, pay_and_unlock_to_file)
    src = tmp_path / "plain.bin"
    src.write_bytes(b"secret dataset " * 50_000)
    _buyer_priv, buyer_pub = generate_recipient_keypair()
    other_priv, _other_pub = generate_recipient_keypair()
    ct = tmp_path / "cipher.bin"
    built = build_paid_content_from_path(
        src, ct, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])
    with pytest.raises(PaidUnlockError):
        pay_and_unlock_to_file(
            content_hash=built["content_hash"], recipient_privkey_b64=other_priv,
            commitment=built["commitment"],
            fetch_wrapped_key=lambda ch: built["wrapped_key"],
            retrieve_content_to_file=lambda ch: str(ct),
            dest_path=tmp_path / "out.bin", settle_fee=lambda: None)
    assert not (tmp_path / "out.bin").exists()


def test_pay_and_unlock_to_file_unretrievable_ciphertext_fails_loud(tmp_path):
    from prsm.economy.paid_content import (
        build_paid_content_from_path, pay_and_unlock_to_file)
    src = tmp_path / "plain.bin"
    src.write_bytes(b"data " * 20_000)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    ct = tmp_path / "cipher.bin"
    built = build_paid_content_from_path(
        src, ct, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])
    with pytest.raises(PaidUnlockError, match="not retrievable"):
        pay_and_unlock_to_file(
            content_hash=built["content_hash"], recipient_privkey_b64=buyer_priv,
            commitment=built["commitment"],
            fetch_wrapped_key=lambda ch: built["wrapped_key"],
            retrieve_content_to_file=lambda ch: None,   # no provider has it
            dest_path=tmp_path / "out.bin", settle_fee=lambda: None)
