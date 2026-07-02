"""Sprint 1358 — F1 redesign R2: the payment-gated wrapped-key serve core.

serve_paid_key authenticates the fetcher's ETH signature, gates on an on-chain verifyPayment, and
only then returns the retained wrapped key — so the designated buyer can no longer obtain the key
for free (the F1 hole). Tested with real EIP-191 signatures + injected verify_payment/key_store.
"""
from __future__ import annotations

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from prsm.node.paid_key_serve import (
    PaidKeyServeError,
    paid_key_challenge,
    serve_paid_key,
)

_CH = bytes.fromhex("cd" * 32)
_FEE = 10 ** 18
_PRIV = "0x" + "11" * 32
_PAYER = Account.from_key(_PRIV).address


def _sign(content_hash=_CH, nonce="n1", priv=_PRIV):
    msg = encode_defunct(text=paid_key_challenge(content_hash, nonce))
    return Account.from_key(priv).sign_message(msg).signature.hex()


class _Store:
    def __init__(self, entry):
        self._entry = entry

    def get(self, ch):
        return self._entry


_ENTRY = {"wrapped_key": b"THE-WRAPPED-KEY", "fee_wei": _FEE}


def test_serves_key_to_a_paid_signer():
    def vp(payer, ch, fee):
        return payer == _PAYER and ch == _CH and fee == _FEE
    out = serve_paid_key(_CH, "n1", _sign(), key_store=_Store(_ENTRY), verify_payment=vp)
    assert out == b"THE-WRAPPED-KEY"


def test_unpaid_signer_is_402():
    with pytest.raises(PaidKeyServeError) as ei:
        serve_paid_key(_CH, "n1", _sign(), key_store=_Store(_ENTRY),
                       verify_payment=lambda *a: False)
    assert ei.value.status == 402


def test_no_retained_key_is_404():
    with pytest.raises(PaidKeyServeError) as ei:
        serve_paid_key(_CH, "n1", _sign(), key_store=_Store(None),
                       verify_payment=lambda *a: True)
    assert ei.value.status == 404


def test_bad_signature_is_401():
    with pytest.raises(PaidKeyServeError) as ei:
        serve_paid_key(_CH, "n1", "0xdeadbeef", key_store=_Store(_ENTRY),
                       verify_payment=lambda *a: True)
    assert ei.value.status == 401


def test_rpc_failure_is_503_not_a_leak():
    def vp(*a):
        raise RuntimeError("rpc down")
    with pytest.raises(PaidKeyServeError) as ei:
        serve_paid_key(_CH, "n1", _sign(), key_store=_Store(_ENTRY), verify_payment=vp)
    assert ei.value.status == 503


def test_gate_binds_to_the_actual_payer():
    # a DIFFERENT key signs; only _PAYER paid → the other signer is unpaid → refused
    sig = _sign(priv="0x" + "22" * 32)
    with pytest.raises(PaidKeyServeError) as ei:
        serve_paid_key(_CH, "n1", sig, key_store=_Store(_ENTRY),
                       verify_payment=lambda payer, ch, fee: payer == _PAYER)
    assert ei.value.status == 402


def test_signature_is_bound_to_the_content():
    # a signature over content _CH must not authenticate a fetch for a different content
    other_ch = bytes.fromhex("ef" * 32)
    sig_for_ch = _sign(content_hash=_CH)
    with pytest.raises(PaidKeyServeError):
        serve_paid_key(other_ch, "n1", sig_for_ch, key_store=_Store(_ENTRY),
                       verify_payment=lambda payer, ch, fee: payer == _PAYER)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
