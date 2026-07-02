"""Sprint 1365 — F9/F2 squatting hardening (client-layer, deployed contracts unchanged).

contentHash is a public squattable id, and neither deployed KeyDistribution.depositKey nor
ProvenanceRegistry.registerContent binds authority to the publisher. The invariant that closes both:
the fee payee (registry creator the CAV credits) must equal the key depositor (KeyDistribution
publisher who controls the content). Consumer refuses to pay on mismatch; publisher refuses to
publish into a hash another party already deposited.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.paid_content import (
    SquatMismatchError,
    assert_content_hash_unclaimed,
    assert_publisher_controls_payee,
    publish_paid_content,
)
from prsm.economy.web3.key_distribution import KeyDeposit
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    generate_recipient_keypair,
)

_CH = bytes.fromhex("ab" * 32)
_PUB = "0x" + "11" * 20
_OTHER = "0x" + "22" * 20
_ZERO = "0x" + "0" * 40


def _kc(publisher):
    c = MagicMock()
    c.get_deposit.return_value = (
        None if publisher is None
        else KeyDeposit(publisher=publisher, royalty="0xr", release_fee_ftns_wei=1, active=True))
    return c


# ── consumer guard: fee payee (creator) must == key depositor (publisher) ─────

def test_payee_matches_publisher_ok():
    assert_publisher_controls_payee(_kc(_PUB), lambda c: _PUB, _CH)   # no raise


def test_payee_differs_from_depositor_raises():
    with pytest.raises(SquatMismatchError, match="squatting"):
        assert_publisher_controls_payee(_kc(_PUB), lambda c: _OTHER, _CH)


def test_zero_creator_raises():
    with pytest.raises(SquatMismatchError, match="no registered creator"):
        assert_publisher_controls_payee(_kc(_PUB), lambda c: _ZERO, _CH)


def test_no_reader_or_client_skips():
    assert_publisher_controls_payee(_kc(_PUB), None, _CH)             # no reader → skip
    assert_publisher_controls_payee(None, lambda c: _OTHER, _CH)      # no client → skip


def test_no_deposit_skips_here():
    # a missing deposit is fail-closed by assert_fee_matches_deposit, not this guard
    assert_publisher_controls_payee(_kc(None), lambda c: _OTHER, _CH)  # no raise


# ── publisher guard: don't publish into another party's deposited hash ────────

def test_unclaimed_hash_ok():
    assert_content_hash_unclaimed(_kc(None), _CH, _PUB)              # no prior deposit → ok


def test_own_prior_deposit_ok():
    assert_content_hash_unclaimed(_kc(_PUB), _CH, _PUB)             # same publisher → ok


def test_other_party_deposit_raises():
    with pytest.raises(SquatMismatchError, match="already deposited"):
        assert_content_hash_unclaimed(_kc(_OTHER), _CH, _PUB)


def test_publish_rejects_squatted_content_hash():
    _, pub = generate_recipient_keypair()
    kd = MagicMock()
    kd.address = _PUB
    kd.get_deposit.return_value = KeyDeposit(     # the hash is already deposited by someone else
        publisher=_OTHER, royalty="0xr", release_fee_ftns_wei=1, active=True)
    with pytest.raises(SquatMismatchError, match="already deposited"):
        publish_paid_content(
            plaintext=b"x",
            recipients=[EnterpriseRecipient(identifier="b", x25519_pubkey_b64=pub)],
            royalty_verifier_address="0x" + "ab" * 20, release_fee_ftns_wei=10 ** 18,
            key_client=kd, publish_ciphertext=lambda c, ct: None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
