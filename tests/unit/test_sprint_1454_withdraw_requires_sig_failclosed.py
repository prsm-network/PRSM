"""sp1454 (off-ramp audit w11uxmoxt) — the /wallet/withdraw signature-requirement gate must FAIL CLOSED.

The off-ramp/withdrawal audit came back CLEAN on exploitable money-loss (Path A fiat→Stripe is unwired
scaffold; the live Path B on-chain withdraw is value-conserved — the off-chain debit runs before each
broadcast under a serializing lock with a balance re-check). But it flagged one fail-OPEN worth
hardening: the handler read the wallet's requires_user_signature flag as
`try: requires_sig = await get_requires_user_signature(...) except Exception: requires_sig = False`.
On a read error it defaulted to NOT requiring a signature → the entire EIP-712 signature block is
skipped → an UNSIGNED withdraw proceeds against a signature-required wallet. A security gate must fail
CLOSED: if we cannot PROVE a wallet is signature-optional, REQUIRE the signature.

The trigger (a SELECT raising on the shared single-connection WAL ledger) is unlikely, so this is
defense-in-depth, not a live exploit — but fail-closed is strictly safer at zero money-risk cost (the
worst case is a legitimate no-sig wallet is briefly asked for a signature during a DB error).
"""
from __future__ import annotations

import asyncio

from prsm.economy.withdraw_signature import resolve_requires_signature_failclosed


def _run(coro):
    return asyncio.run(coro)


def test_returns_flag_value_on_success():
    async def _true():
        return True

    async def _false():
        return False

    assert _run(resolve_requires_signature_failclosed(_true)) is True
    assert _run(resolve_requires_signature_failclosed(_false)) is False


def test_fails_closed_on_read_error():
    async def _raises():
        raise RuntimeError("database is locked")

    # The money assertion: a read error must REQUIRE a signature (True), NOT skip it (False).
    assert _run(resolve_requires_signature_failclosed(_raises)) is True, (
        "signature-requirement gate failed OPEN on a read error — an unsigned withdraw could proceed "
        "against a signature-required wallet")


def test_non_bool_truthy_is_coerced():
    async def _one():
        return 1  # a truthy non-bool must still gate as required

    assert _run(resolve_requires_signature_failclosed(_one)) is True
