"""Sprint 1355 (+1359 F1 redesign) — brick 4 SURFACE: SDK + CLI buy-and-decrypt.

PRSMClient.pay_and_unlock_content (pay → FETCH the wrapped key from the payment-gated endpoint →
verify vs the on-chain commitment → decrypt) and the `prsm content unlock` CLI over it. SDK tested
end-to-end with an injected fetch (gated on payment); CLI tested for the keys-from-env guard.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)
from prsm.sdk.client import PRSMClient
from prsm.storage.encryption import encrypt, generate_key
from prsm.storage.paid_unlock import key_commitment, wrap_content_key_for_deposit

_CH = bytes.fromhex("cd" * 32)
_FEE = 10 ** 18
_BUYER_ETH = "0x" + "33" * 20


def _fixture(plaintext=b"paywalled dataset via the SDK"):
    order = []
    content_key = generate_key()
    content = encrypt(plaintext, content_key)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])
    commitment = key_commitment(wrapped)

    paid = {"ok": False}
    vc = MagicMock()
    vc.address = _BUYER_ETH
    vc.verify_payment.return_value = False                          # not paid → settle pays
    vc.pay_for_access.side_effect = lambda *a: (order.append("pay"), paid.__setitem__("ok", True))

    def fetch(ch):                                                 # the gated serve endpoint
        order.append("fetch")
        return wrapped if paid["ok"] else None

    return order, content, buyer_priv, commitment, vc, fetch, plaintext


async def _unlock(content_hash, content, buyer_priv, commitment, vc, fetch, *, fee_wei=_FEE):
    from prsm.economy.web3.key_distribution import KeyDeposit
    kc = MagicMock()
    kc.get_deposit.return_value = KeyDeposit(                       # fee matches → check passes
        publisher="0xpub", royalty="0xroy", release_fee_ftns_wei=_FEE, active=True)
    client = PRSMClient()
    try:
        return await client.pay_and_unlock_content(
            content_hash, requester_key="0x" + "01" * 32, x25519_privkey_b64=buyer_priv,
            fee_wei=fee_wei, verifier_address="0x" + "ab" * 20, commitment=commitment,
            _verifier_client=vc, _content=content, _fetch_wrapped_key=fetch, _key_client=kc)
    finally:
        await client.close()


# ── SDK ───────────────────────────────────────────────────────────────────────

def test_sdk_pay_then_fetch_then_decrypt():
    order, content, buyer_priv, commitment, vc, fetch, plaintext = _fixture()
    out = asyncio.run(_unlock(_CH, content, buyer_priv, commitment, vc, fetch))
    assert out == plaintext
    assert order == ["pay", "fetch"]                    # paid, THEN the gated fetch succeeded


def test_sdk_accepts_hex_content_hash():
    order, content, buyer_priv, commitment, vc, fetch, plaintext = _fixture()
    out = asyncio.run(_unlock("0x" + _CH.hex(), content, buyer_priv, commitment, vc, fetch))
    assert out == plaintext


def test_sdk_validates_content_hash_and_fee():
    _o, content, buyer_priv, commitment, vc, fetch, _ = _fixture()
    with pytest.raises(ValueError, match="32 bytes"):
        asyncio.run(_unlock("dead", content, buyer_priv, commitment, vc, fetch))
    with pytest.raises(ValueError, match="positive"):
        asyncio.run(_unlock(_CH, content, buyer_priv, commitment, vc, fetch, fee_wei=0))


def test_sdk_wrong_served_key_fails_commitment():
    from prsm.economy.web3.key_acquisition import KeyCommitmentMismatchError
    order, content, buyer_priv, commitment, vc, _fetch, _ = _fixture()
    paid = {"ok": False}
    vc.pay_for_access.side_effect = lambda *a: paid.__setitem__("ok", True)
    bad_fetch = lambda ch: (b"a substituted key" if paid["ok"] else None)
    with pytest.raises(KeyCommitmentMismatchError, match="WRONG key"):
        asyncio.run(_unlock(_CH, content, buyer_priv, commitment, vc, bad_fetch))


def test_sdk_reads_authoritative_commitment_when_not_supplied():
    # sp1363 (R5 MEDIUM): commitment=None → the SDK reads it from the gated KeyReleased event
    # (acquire_released_key) AFTER paying, not from the untrusted serve path.
    from eth_account import Account
    from prsm.economy.web3.key_distribution import KeyDeposit, KeyNotFoundError
    from prsm.storage.encryption import encrypt, generate_key
    from prsm.storage.paid_unlock import key_commitment, wrap_content_key_for_deposit

    content_key = generate_key()
    plaintext = b"authoritative-commitment path"
    content = encrypt(plaintext, content_key)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])
    commitment = key_commitment(wrapped)
    payer = Account.from_key("0x" + "01" * 32).address

    state = {"paid": False}
    vc = MagicMock()
    vc.address = payer
    vc.verify_payment.return_value = False
    vc.pay_for_access.side_effect = lambda *a: state.__setitem__("paid", True)

    class _Ev:
        content_hash, recipient, encrypted_key = _CH, payer, commitment

    class _KD:                                          # release emits the COMMITMENT in the event
        def __init__(self):
            self._rel = []

        def get_deposit(self, ch):
            return KeyDeposit(publisher="0xp", royalty="0xr", release_fee_ftns_wei=_FEE, active=True)

        def latest_block(self):
            return 100

        def get_key_released_events(self, fb, tb, *, argument_filters=None):
            return list(self._rel)

        def release(self, ch, recipient):
            if not state["paid"]:
                raise KeyNotFoundError("unpaid")
            self._rel.append(_Ev())
            return ("0xr", None)

    async def _go():
        client = PRSMClient()
        try:
            return await client.pay_and_unlock_content(
                _CH, requester_key="0x" + "01" * 32, x25519_privkey_b64=buyer_priv,
                fee_wei=_FEE, verifier_address="0x" + "ab" * 20, commitment=None,
                _verifier_client=vc, _content=content, _key_client=_KD(),
                _fetch_wrapped_key=lambda ch: (wrapped if state["paid"] else None))
        finally:
            await client.close()

    assert asyncio.run(_go()) == plaintext


# ── CLI: keys must come from the environment, never argv ──────────────────────

_COMMIT = "0x" + "ee" * 32


def test_cli_unlock_requires_env_keys(monkeypatch):
    from click.testing import CliRunner
    from prsm.cli import main
    monkeypatch.delenv("PRSM_REQUESTER_KEY", raising=False)
    monkeypatch.delenv("PRSM_X25519_PRIVKEY", raising=False)
    r = CliRunner().invoke(main, [
        "content", "unlock", "0x" + _CH.hex(), "--fee", "1", "--commitment", _COMMIT,
        "--verifier-address", "0x" + "ab" * 20])
    assert r.exit_code == 1
    assert "Missing keys" in r.output


def test_cli_unlock_requires_verifier(monkeypatch):
    from click.testing import CliRunner
    from prsm.cli import main
    monkeypatch.setenv("PRSM_REQUESTER_KEY", "0x" + "01" * 32)
    monkeypatch.setenv("PRSM_X25519_PRIVKEY", "somekey")
    monkeypatch.delenv("PRSM_CONTENT_ACCESS_VERIFIER", raising=False)
    r = CliRunner().invoke(main, [
        "content", "unlock", "0x" + _CH.hex(), "--fee", "1", "--commitment", _COMMIT])
    assert r.exit_code == 1
    assert "verifier" in r.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── sp1458: pay_and_unlock_content(dest_path=...) — streaming large-file unlock ──

def test_sdk_pay_and_unlock_content_to_file_streams(tmp_path):
    from prsm.economy.paid_content import build_paid_content_from_path
    from prsm.economy.web3.key_distribution import KeyDeposit

    plaintext = (b"paywalled large dataset via the SDK " * 100_000) + b"tail"
    src = tmp_path / "plain.bin"
    src.write_bytes(plaintext)
    ct = tmp_path / "cipher.bin"
    buyer_priv, buyer_pub = generate_recipient_keypair()
    built = build_paid_content_from_path(
        src, ct, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])

    order = []
    paid = {"ok": False}
    vc = MagicMock()
    vc.address = _BUYER_ETH
    vc.verify_payment.return_value = False
    vc.pay_for_access.side_effect = lambda *a: (order.append("pay"), paid.__setitem__("ok", True))

    def fetch(ch):
        order.append("fetch")
        return built["wrapped_key"] if paid["ok"] else None

    kc = MagicMock()
    kc.get_deposit.return_value = KeyDeposit(
        publisher="0xpub", royalty="0xroy", release_fee_ftns_wei=_FEE, active=True)

    async def _go():
        client = PRSMClient()
        try:
            return await client.pay_and_unlock_content(
                built["content_hash"], requester_key="0x" + "01" * 32,
                x25519_privkey_b64=buyer_priv, fee_wei=_FEE,
                verifier_address="0x" + "ab" * 20, commitment=built["commitment"],
                dest_path=str(tmp_path / "out.bin"),
                _verifier_client=vc, _content=str(ct), _fetch_wrapped_key=fetch, _key_client=kc)
        finally:
            await client.close()

    out = asyncio.run(_go())
    assert out == str(tmp_path / "out.bin")
    assert order == ["pay", "fetch"]                        # paid, THEN the gated key fetch
    assert (tmp_path / "out.bin").read_bytes() == plaintext  # streamed decrypt, byte-identical


def test_cli_unlock_stream_requires_output(monkeypatch):
    from click.testing import CliRunner
    from prsm.cli import main
    monkeypatch.setenv("PRSM_REQUESTER_KEY", "0x" + "01" * 32)
    monkeypatch.setenv("PRSM_X25519_PRIVKEY", "k")
    r = CliRunner().invoke(main, [
        "content", "unlock", "0x" + _CH.hex(), "--fee", "1", "--commitment", _COMMIT,
        "--verifier-address", "0x" + "ab" * 20, "--stream"])
    assert r.exit_code == 1
    assert "--stream requires --output" in r.output


def test_cli_unlock_stream_passes_dest_path(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock, patch
    from pathlib import Path
    from click.testing import CliRunner
    from prsm.cli import main
    monkeypatch.setenv("PRSM_REQUESTER_KEY", "0x" + "01" * 32)
    monkeypatch.setenv("PRSM_X25519_PRIVKEY", "k")
    out = tmp_path / "big.bin"
    captured = {}

    async def _fake_unlock(content_hash, **kw):
        captured.update(kw)
        Path(kw["dest_path"]).write_bytes(b"streamed plaintext")
        return kw["dest_path"]

    with patch("prsm.sdk.client.PRSMClient.pay_and_unlock_content",
               new=AsyncMock(side_effect=_fake_unlock)), \
         patch("prsm.sdk.client.PRSMClient.close", new=AsyncMock()):
        r = CliRunner().invoke(main, [
            "content", "unlock", "0x" + _CH.hex(), "--fee", "1", "--commitment", _COMMIT,
            "--verifier-address", "0x" + "ab" * 20, "--stream", "--output", str(out)])
    assert r.exit_code == 0, r.output
    assert "Unlocked (streamed)" in r.output
    assert captured["dest_path"] == str(out)              # --stream threads dest_path
    assert out.read_bytes() == b"streamed plaintext"
