"""Sprint 1361 — F1 redesign hardening: fold in the B5 review's remaining code-level fixes.

F3/F4/F12 (fee-validation): read the AUTHORITATIVE on-chain deposit fee and refuse to pay a
mismatched one (which would be pulled with no way to unlock). F5 (publish order): deposit LAST so a
live gate always implies the content is deliverable.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.paid_content import (
    FeeMismatchError,
    assert_fee_matches_deposit,
    publish_paid_content,
)
from prsm.economy.web3.key_distribution import KeyDeposit

_CH = bytes.fromhex("ab" * 32)
_FEE = 10 ** 18


# ── assert_fee_matches_deposit (F3/F4/F12) ────────────────────────────────────

def _client(deposit):
    c = MagicMock()
    c.get_deposit.return_value = deposit
    return c


def test_matching_fee_passes():
    c = _client(KeyDeposit(publisher="0xpub", royalty="0xroy",
                           release_fee_ftns_wei=_FEE, active=True))
    assert_fee_matches_deposit(c, _CH, _FEE)            # no raise
    c.get_deposit.assert_called_once_with(_CH)


def test_mismatched_fee_raises_before_paying():
    c = _client(KeyDeposit(publisher="0xpub", royalty="0xroy",
                           release_fee_ftns_wei=_FEE, active=True))
    with pytest.raises(FeeMismatchError, match="won't unlock"):
        assert_fee_matches_deposit(c, _CH, _FEE + 1)


def test_fail_closed_when_client_present_but_unconfirmable():
    # sp1362 (R5 low): only a MISSING client skips; a real client that can't confirm FAILS CLOSED
    assert_fee_matches_deposit(None, _CH, _FEE)                     # no client → skip (opt-out)
    with pytest.raises(FeeMismatchError, match="nothing to unlock"):
        assert_fee_matches_deposit(_client(None), _CH, _FEE)       # deposit None → fail-closed
    raising = MagicMock()
    raising.get_deposit.side_effect = RuntimeError("rpc down")
    with pytest.raises(FeeMismatchError, match="could not confirm"):
        assert_fee_matches_deposit(raising, _CH, _FEE)             # RPC error → fail-closed


# ── publish order: deposit LAST (F5) ──────────────────────────────────────────

def test_publish_deposits_last():
    from prsm.enterprise.recipient_encryption import generate_recipient_keypair, EnterpriseRecipient
    order = []
    kd = MagicMock()
    kd.deposit_key.side_effect = lambda *a: (order.append("deposit"), ("0xdep", None))[1]
    kd.get_deposit.return_value = None            # fresh content_hash — not squatted (sp1365 guard)
    _, pub = generate_recipient_keypair()
    retained, served = {}, {}
    publish_paid_content(
        plaintext=b"rows", recipients=[EnterpriseRecipient(identifier="b", x25519_pubkey_b64=pub)],
        royalty_verifier_address="0x" + "ab" * 20, release_fee_ftns_wei=_FEE, key_client=kd,
        publish_ciphertext=lambda ch, ct: (order.append("publish"), served.__setitem__(bytes(ch), ct)),
        retain_wrapped_key=lambda ch, wk, fee: (order.append("retain"),
                                                retained.__setitem__(bytes(ch), wk)))
    # retain + serve the content BEFORE the on-chain gate goes live
    assert order == ["retain", "publish", "deposit"]


# ── KeyDistributionClient.get_deposit reads records (omitting encryptedKey) ────

def test_get_deposit_parses_records(monkeypatch):
    from prsm.economy.web3 import key_distribution as kd_mod

    class _Fn:
        def __init__(self, ret):
            self._ret = ret

        def call(self):
            return self._ret

    class _Functions:
        def records(self, ch):
            return _Fn(["0xPUB", "0xROY", _FEE, True])

    client = kd_mod.KeyDistributionClient.__new__(kd_mod.KeyDistributionClient)
    client.contract = MagicMock()
    client.contract.functions = _Functions()
    dep = client.get_deposit(_CH)
    assert dep.publisher == "0xPUB" and dep.release_fee_ftns_wei == _FEE and dep.active is True


def test_get_deposit_none_for_zero_publisher():
    from prsm.economy.web3 import key_distribution as kd_mod

    class _Fn:
        def call(self):
            return [kd_mod.ZERO_ADDRESS, kd_mod.ZERO_ADDRESS, 0, False]

    class _Functions:
        def records(self, ch):
            return _Fn()

    client = kd_mod.KeyDistributionClient.__new__(kd_mod.KeyDistributionClient)
    client.contract = MagicMock()
    client.contract.functions = _Functions()
    assert client.get_deposit(_CH) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
