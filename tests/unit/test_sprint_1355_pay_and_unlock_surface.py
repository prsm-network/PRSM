"""Sprint 1355 — Tier B/C paid-decrypt arc, brick 4 SURFACE: SDK + CLI buy-and-decrypt.

PRSMClient.pay_and_unlock_content (pay the release fee → key release → fetch ciphertext → decrypt)
and the `prsm content unlock` CLI over it. SDK tested end-to-end with injected web3 clients +
ciphertext; CLI tested for the keys-from-env guard (keys must never come from argv).
"""
from __future__ import annotations

import asyncio

import pytest

from prsm.economy.web3.key_distribution import KeyNotFoundError
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.sdk.client import PRSMClient
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import wrap_content_key_for_deposit

_CH = bytes.fromhex("cd" * 32)
_FEE = 10 ** 18
_BUYER_ETH = "0x" + "33" * 20


class _Ev:
    def __init__(self, ch, r, k):
        self.content_hash, self.recipient, self.encrypted_key = ch, r, k


class _KD:
    """KeyDistribution whose release is gated on payment (mirrors the real verifyPayment gate)."""

    def __init__(self, wrapped, order):
        self._wrapped, self._order = wrapped, order
        self._paid = False
        self._released = []

    def latest_block(self):
        return 100

    def get_key_released_events(self, fb, tb, *, argument_filters=None):
        return list(self._released)

    def release(self, ch, recipient):
        self._order.append("release")
        if not self._paid:
            raise KeyNotFoundError("PaymentNotVerified")
        self._released.append(_Ev(bytes(ch), recipient, self._wrapped))
        return ("0xr", None)


class _Verifier:
    def __init__(self, kd, order):
        self.address = _BUYER_ETH
        self._kd, self._order = kd, order

    def pay_for_access(self, ch, fee):
        self._order.append("pay")
        self._kd._paid = True


def _fixture(plaintext=b"paywalled dataset via the SDK"):
    order = []
    content_key = generate_key()
    content = encrypt(plaintext, content_key)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])
    kd = _KD(wrapped, order)
    vc = _Verifier(kd, order)
    return order, content, buyer_priv, kd, vc, plaintext


async def _unlock(content_hash, content, buyer_priv, kd, vc):
    client = PRSMClient()
    try:
        return await client.pay_and_unlock_content(
            content_hash, requester_key="0x" + "01" * 32, x25519_privkey_b64=buyer_priv,
            fee_wei=_FEE, verifier_address="0x" + "ab" * 20,
            _verifier_client=vc, _key_client=kd, _content=content)
    finally:
        await client.close()


# ── SDK ───────────────────────────────────────────────────────────────────────

def test_sdk_pay_and_unlock_full_flow():
    order, content, buyer_priv, kd, vc, plaintext = _fixture()
    out = asyncio.run(_unlock(_CH, content, buyer_priv, kd, vc))
    assert out == plaintext
    assert order == ["pay", "release"]                 # paid, THEN the gated release succeeded


def test_sdk_accepts_hex_content_hash():
    order, content, buyer_priv, kd, vc, plaintext = _fixture()
    out = asyncio.run(_unlock("0x" + _CH.hex(), content, buyer_priv, kd, vc))
    assert out == plaintext


def test_sdk_validates_content_hash_and_fee():
    _o, content, buyer_priv, kd, vc, _ = _fixture()
    with pytest.raises(ValueError, match="32 bytes"):
        asyncio.run(_unlock("dead", content, buyer_priv, kd, vc))

    async def _zero_fee():
        client = PRSMClient()
        try:
            return await client.pay_and_unlock_content(
                _CH, requester_key="0x" + "01" * 32, x25519_privkey_b64=buyer_priv,
                fee_wei=0, verifier_address="0x" + "ab" * 20,
                _verifier_client=vc, _key_client=kd, _content=content)
        finally:
            await client.close()
    with pytest.raises(ValueError, match="positive"):
        asyncio.run(_zero_fee())


def test_sdk_wrong_x25519_key_raises():
    from prsm.storage.paid_unlock import PaidUnlockError
    order, content, _buyer_priv, kd, vc, _ = _fixture()
    other_priv, _ = generate_recipient_keypair()
    with pytest.raises(PaidUnlockError, match="unwrap"):
        asyncio.run(_unlock(_CH, content, other_priv, kd, vc))


# ── CLI: keys must come from the environment, never argv ──────────────────────

def test_cli_unlock_requires_env_keys(monkeypatch):
    from click.testing import CliRunner
    from prsm.cli import main
    monkeypatch.delenv("PRSM_REQUESTER_KEY", raising=False)
    monkeypatch.delenv("PRSM_X25519_PRIVKEY", raising=False)
    r = CliRunner().invoke(main, [
        "content", "unlock", "0x" + _CH.hex(), "--fee", "1",
        "--verifier-address", "0x" + "ab" * 20])
    assert r.exit_code == 1
    assert "Missing keys" in r.output


def test_cli_unlock_requires_verifier(monkeypatch):
    from click.testing import CliRunner
    from prsm.cli import main
    monkeypatch.setenv("PRSM_REQUESTER_KEY", "0x" + "01" * 32)
    monkeypatch.setenv("PRSM_X25519_PRIVKEY", "somekey")
    monkeypatch.delenv("PRSM_CONTENT_ACCESS_VERIFIER", raising=False)
    r = CliRunner().invoke(main, ["content", "unlock", "0x" + _CH.hex(), "--fee", "1"])
    assert r.exit_code == 1
    assert "verifier-address" in r.output.lower() or "verifier" in r.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
