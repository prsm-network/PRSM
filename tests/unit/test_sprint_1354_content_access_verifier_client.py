"""Sprint 1354 — Tier B/C paid-decrypt arc, brick 4 CONSUMER side: the pay path.

ContentAccessVerifierClient (approve FTNS + payForAccess + verifyPayment + claim) and
build_content_access_settle_fee, which turns the client into the real ``settle_fee`` that
pay_and_unlock expects — so the live consumer flow is: settle_fee pays the release fee here →
acquire_released_key's release passes its verifyPayment gate → decrypt. Client tested against a
web3 fake; the builder tested wired into pay_and_unlock.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.paid_content import build_content_access_settle_fee, pay_and_unlock
from prsm.economy.web3.content_access_verifier import ContentAccessVerifierClient

_CH = bytes.fromhex("ab" * 32)
_FEE = 10 ** 18
_PAYER = "0x" + "11" * 20


# ── web3 fake harness (mirrors test_key_distribution_client) ───────────────────

class _State:
    def __init__(self, *, allowance=0, verify=False, claimable=0, receipt_status=1):
        self.allowance = allowance
        self.verify = verify
        self.claimable = claimable
        self.receipt_status = receipt_status
        self.calls = []


class _Fn:
    def __init__(self, state, name, args):
        self._state, self._name, self._args = state, name, args

    def build_transaction(self, overrides):
        self._state.calls.append((self._name, self._args))
        return {"to": "0xc", "data": "0x", **overrides}

    def call(self):
        self._state.calls.append((self._name, self._args))
        return {
            "allowance": self._state.allowance,
            "verifyPayment": self._state.verify,
            "claimable": self._state.claimable,
        }.get(self._name, 0)


class _Functions:
    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        state = self._state
        return lambda *args: _Fn(state, name, args)


class _Contract:
    def __init__(self, state):
        self.functions = _Functions(state)
        self.events = MagicMock()


class _FakeAccount:
    address = _PAYER
    key = b"\x01" * 32


class _FakeEth:
    def __init__(self, state):
        self._state = state
        self.chain_id = 84532
        self.gas_price = 10 ** 9
        self.account = MagicMock()
        signed = MagicMock()
        signed.raw_transaction = b"\xff" * 32
        self.account.sign_transaction.return_value = signed

    def get_transaction_count(self, addr, *_):
        return 3

    def contract(self, address, abi):
        return _Contract(self._state)

    def send_raw_transaction(self, raw):
        return b"\xab" * 32

    def wait_for_transaction_receipt(self, tx_hash, timeout=120):
        return MagicMock(status=self._state.receipt_status)


_ACTIVE = {"state": None}


class _FakeWeb3:
    def __init__(self, *a, **k):
        self.eth = _FakeEth(_ACTIVE["state"])

    @staticmethod
    def to_checksum_address(a):
        return a

    @staticmethod
    def HTTPProvider(u):
        return object()


def _client(state, *, with_key=True, expected_chain_id=None):
    _ACTIVE["state"] = state
    with patch("prsm.economy.web3.content_access_verifier.Web3", _FakeWeb3), \
         patch("prsm.economy.web3.content_access_verifier.Account") as acct:
        acct.from_key.return_value = _FakeAccount()
        return ContentAccessVerifierClient(
            rpc_url="http://t", verifier_address="0x" + "ab" * 20,
            ftns_token_address="0x" + "cd" * 20,
            private_key=("0x" + "01" * 32) if with_key else None,
            expected_chain_id=expected_chain_id)


def test_client_chain_id_mismatch_raises():
    # sp1356 (review F7): the fake RPC reports chainId 84532; pinning a different one fails loud.
    with pytest.raises(RuntimeError, match="chainId"):
        _client(_State(), expected_chain_id=999)


def test_client_chain_id_match_ok():
    assert _client(_State(), expected_chain_id=84532) is not None


def _names(state):
    return [c[0] for c in state.calls]


# ── ContentAccessVerifierClient ───────────────────────────────────────────────

def test_verify_payment_reads_the_gate():
    st = _State(verify=True)
    assert _client(st).verify_payment(_PAYER, _CH, _FEE) is True
    assert ("verifyPayment", (_PAYER, _CH, _FEE)) in st.calls


def test_pay_for_access_approves_when_allowance_short():
    st = _State(allowance=0)                       # under the fee → must approve first
    _client(st).pay_for_access(_CH, _FEE)
    assert _names(st) == ["allowance", "approve", "payForAccess"]


def test_pay_for_access_skips_approve_when_already_allowed():
    st = _State(allowance=_FEE)                     # already approved → no approve tx
    _client(st).pay_for_access(_CH, _FEE)
    assert _names(st) == ["allowance", "payForAccess"]


def test_pay_for_access_validates_inputs():
    st = _State(allowance=_FEE)
    with pytest.raises(ValueError, match="32 bytes"):
        _client(st).pay_for_access(b"short", _FEE)
    with pytest.raises(ValueError, match="positive"):
        _client(st).pay_for_access(_CH, 0)


def test_write_without_key_raises():
    with pytest.raises(RuntimeError, match="private_key"):
        _client(_State(), with_key=False).pay_for_access(_CH, _FEE)


def test_claim_calls_claim():
    st = _State()
    _client(st).claim()
    assert "claim" in _names(st)


# ── build_content_access_settle_fee + pay_and_unlock integration ──────────────

def test_settle_fee_builder_calls_pay_for_access():
    fake = MagicMock()
    fake.verify_payment.return_value = False               # not yet paid → must pay
    settle = build_content_access_settle_fee(fake, _CH, _FEE)
    settle()
    fake.pay_for_access.assert_called_once_with(_CH, _FEE)


def test_settle_fee_skips_when_already_paid(monkeypatch):
    # sp1356 (review F8/F11): verify-before-pay — a retry after payment is a no-op (no double-charge)
    fake = MagicMock()
    fake.address = _PAYER
    fake.verify_payment.return_value = True                # already settled
    settle = build_content_access_settle_fee(fake, _CH, _FEE)
    assert settle() is None
    fake.pay_for_access.assert_not_called()
    fake.verify_payment.assert_called_once_with(_PAYER, _CH, _FEE)


def test_settle_fee_wired_into_pay_and_unlock_end_to_end():
    from prsm.economy.web3.key_distribution import KeyNotFoundError
    from prsm.enterprise.recipient_encryption import (
        EnterpriseRecipient, generate_recipient_keypair,
    )
    from prsm.storage.encryption import encrypt, generate_key
    from prsm.storage.paid_unlock import wrap_content_key_for_deposit

    # publisher-side artifacts
    content_key = generate_key()
    content = encrypt(b"paywalled rows", content_key)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(
        content_key, [EnterpriseRecipient(identifier="b", x25519_pubkey_b64=buyer_pub)])

    # a KeyDistribution that only releases AFTER the fee is paid (mirrors the real gate)
    order = []

    class _Ev:
        def __init__(s, ch, r, k):
            s.content_hash, s.recipient, s.encrypted_key = ch, r, k

    class _KD:
        def __init__(s):
            s._released, s._paid = [], False

        def latest_block(s):
            return 100

        def get_key_released_events(s, fb, tb, *, argument_filters=None):
            return list(s._released)

        def release(s, ch, recipient):
            order.append("release")
            if not s._paid:
                raise KeyNotFoundError("PaymentNotVerified")   # release gated on payment
            s._released.append(_Ev(bytes(ch), recipient, wrapped))
            return ("0xr", None)

    kd = _KD()

    verifier_client = MagicMock()
    verifier_client.verify_payment.return_value = False    # not yet paid → settle pays
    verifier_client.pay_for_access.side_effect = lambda *a: (order.append("pay"),
                                                             setattr(kd, "_paid", True))
    settle = build_content_access_settle_fee(verifier_client, _CH, _FEE)

    out = pay_and_unlock(
        content_hash=_CH, recipient=_PAYER, recipient_privkey_b64=buyer_priv, key_client=kd,
        retrieve_content=lambda c: content, settle_fee=settle)

    assert out == b"paywalled rows"
    assert order == ["pay", "release"]                  # paid, THEN the gated release succeeded
    verifier_client.pay_for_access.assert_called_once_with(_CH, _FEE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
