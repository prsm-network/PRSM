"""Sprint 1490 — a deposit must never credit MORE FTNS than arrived on chain.

`amount_wei / 1e18` sends the int through float64, which represents integers
exactly only up to 2**53. One FTNS is 10**18 wei, so every realistic deposit
exceeds that and the quotient is rounded to NEAREST — which can round UP.

The magnitude is tiny (sub-wei), but the direction is the one the whole money-in
audit arc (sp1472/1473/1478) existed to close: an over-credit is an unbacked mint,
while an under-credit is recoverable. So the conversion floors.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from prsm.economy.ftns_onchain import wei_to_ftns_floor

WEI = 10**18


def _exact(wei):
    return Decimal(wei) / Decimal(WEI)


def test_the_float_path_really_does_round_up():
    """★ The bug this exists to prevent — proof it is real, not theoretical.
    One wei SHORT of 1 FTNS becomes exactly 1.0 through the float path."""
    one_wei_short = WEI - 1
    assert one_wei_short / 1e18 == 1.0            # over-credit
    assert wei_to_ftns_floor(one_wei_short) < 1.0  # floored


def test_never_credits_more_than_deposited():
    """★ The invariant, across values chosen to straddle 2**53 and force rounding."""
    cases = [
        1, 999, WEI - 1, WEI, WEI + 1,
        1234567890123456789,
        999999999999999999,
        10**24 + 1,
        2**53 - 1, 2**53, 2**53 + 1,
        123456789 * WEI + 987654321,
    ]
    for wei in cases:
        credited = Decimal(wei_to_ftns_floor(wei))
        assert credited <= _exact(wei), (
            f"OVER-CREDIT for {wei} wei: credited {credited} > actual {_exact(wei)}")


def test_exact_values_are_preserved():
    """Flooring must not shave whole amounts — a 1 FTNS deposit credits 1.0."""
    for n in (1, 5, 100, 12345):
        assert wei_to_ftns_floor(n * WEI) == float(n)


def test_zero_and_dust():
    assert wei_to_ftns_floor(0) == 0.0
    assert wei_to_ftns_floor(1) > 0.0          # dust still credits something
    assert wei_to_ftns_floor(1) <= 1e-18


def test_loss_is_bounded_and_negligible():
    """Flooring must cost at most one float ULP, not a meaningful amount."""
    for wei in (1234567890123456789, 10**24 + 1, 987654321 * WEI + 13):
        lost = _exact(wei) - Decimal(wei_to_ftns_floor(wei))
        assert lost >= 0
        assert lost < Decimal(_exact(wei)) * Decimal("1e-15")


def test_honours_non_18_decimals():
    assert wei_to_ftns_floor(5 * 10**6, decimals=6) == 5.0


# ── the scanner and credit path actually use it ─────────────────────

def test_scanner_emits_the_exact_wei():
    """★ Binding test: the helper is useless if the scan dict still carries only
    the lossy float."""
    from unittest.mock import MagicMock

    from prsm.economy.ftns_onchain import scan_inbound_transfers

    log = MagicMock()
    log.blockNumber, log.logIndex = 100, 0
    log.transactionHash = b"\xaa" * 32
    log.args = {"from": "0x" + "a1" * 20, "to": "0x" + "b2" * 20,
                "value": WEI - 1}
    contract = MagicMock()
    contract.events.Transfer.get_logs.return_value = [log]

    out = scan_inbound_transfers(contract, "0x" + "b2" * 20, 0, 200)
    assert len(out) == 1
    assert out[0]["amount_wei"] == WEI - 1, "exact wei must be carried through"


def test_credit_path_prefers_amount_wei_over_the_float():
    """★ Binding test: assert the credit site reads amount_wei, not amount_ftns."""
    import inspect

    from prsm.economy import ftns_onchain

    src = inspect.getsource(ftns_onchain.InboundMonitor._credit_deposit)
    assert "wei_to_ftns_floor" in src
    assert 'transfer.get("amount_wei")' in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
