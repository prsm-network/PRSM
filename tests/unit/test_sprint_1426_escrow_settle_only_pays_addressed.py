"""Sprint 1426 — MultiPartyEscrow batch settlement booked UNPAID creators as settled.

`_execute_onchain_settlement` drops any creator without an on-chain address from the actual
transfer (with require_onchain_address=False, the default, it just silently omits them from
recipients/amounts). But the ATOMIC settlement branch of `settle_batch`
(transfer_batch / multicall / simulation — none set `partial`) then deleted EVERY creator in the
batch from `_pending` and added the FULL batch total to `_total_settled` — including the
address-less creators who were never paid.

So in a MIXED batch (some creators have addresses, some don't), each address-less creator's
accumulated royalty is:
  - dropped from _pending (the record that would let it be retried once an address resolves), and
  - counted in _total_settled (over-reporting), while
  - never included in any on-chain transfer.

Its on-chain claim is marked satisfied without payment — a fund loss the moment this path carries
value. The `partial` (individual-transfer fallback) branch already does the right thing (clears
only creators whose transfer succeeded); the atomic branch did not. This aligns them.

LATENT TODAY (documented, not fixed here): (a) the accumulate() call in content_economy passes
source_content_id= instead of the required source_cid=, so accumulation TypeErrors into a swallowed
except — the escrow never fills; and (b) node.py builds MultiPartyEscrow with no creator_registry
and passes no creator_address, so every creator is address-less and a real batch returns "No valid
recipients". Those keep the fund-loss dormant; this sprint disarms the landmine so it stays safe
when the on-chain path is eventually wired.
"""
from __future__ import annotations

import pytest

from prsm.node.multi_party_escrow import EscrowConfig, MultiPartyEscrow


class _BatchLedger:
    """A minimal ftns_ledger exposing transfer_batch — the ATOMIC on-chain settlement path."""

    def __init__(self):
        self.paid: list[tuple[str, float]] = []

    async def transfer_batch(self, recipients, amounts):
        self.paid = list(zip(recipients, amounts))
        return {"success": True, "tx_hash": "0x" + "ab" * 32, "gas_used": 21000}


@pytest.fixture
def escrow():
    led = _BatchLedger()
    esc = MultiPartyEscrow(
        ftns_ledger=led,
        config=EscrowConfig(min_batch_size=1, min_batch_value=0.0),
    )
    return esc, led


async def _accumulate(esc, creator_id, amount, address):
    await esc.accumulate(
        creator_id=creator_id, amount=amount, source_cid="cid-1",
        accessor_id="accessor-1", creator_address=address,
    )


class TestOnlyPaidCreatorsAreBooked:
    async def test_addressless_creator_stays_pending_and_is_not_booked(self, escrow):
        esc, led = escrow
        await _accumulate(esc, "creatorA", 5.0, "0x000000000000000000000000000000000000aAaA")
        await _accumulate(esc, "creatorB", 3.0, None)  # no on-chain address

        batch = await esc.settle_batch(force=True)
        assert batch is not None

        # Only creatorA (with an address) was actually paid on-chain.
        assert led.paid == [("0x000000000000000000000000000000000000aAaA", 5.0)], (
            "the on-chain transfer paid an unexpected set — the address-less creator must NOT be paid"
        )

        # creatorA: paid -> cleared from pending + counted in total_settled.
        assert "creatorA" not in esc._pending
        # creatorB: NOT paid -> MUST stay pending (retry once an address resolves) and NOT counted.
        assert "creatorB" in esc._pending, (
            "the address-less creator was deleted from pending despite never being paid — its "
            "royalty is now unrecoverable"
        )
        assert esc._pending["creatorB"].total_amount == 3.0

        # _total_settled must equal ONLY what was actually transferred on-chain.
        assert esc._total_settled == 5.0, (
            f"_total_settled={esc._total_settled} over-reports: it counted the 3.0 FTNS that was "
            f"never paid to the address-less creator"
        )

    async def test_all_addressed_still_settles_everyone(self, escrow):
        """Guard against over-correction: when everyone has an address, all are paid + cleared."""
        esc, led = escrow
        await _accumulate(esc, "creatorA", 5.0, "0x000000000000000000000000000000000000aAaA")
        await _accumulate(esc, "creatorB", 3.0, "0x000000000000000000000000000000000000bBbB")

        await esc.settle_batch(force=True)

        assert esc._pending == {}
        assert esc._total_settled == 8.0
        assert dict(led.paid) == {
            "0x000000000000000000000000000000000000aAaA": 5.0,
            "0x000000000000000000000000000000000000bBbB": 3.0,
        }

    async def test_all_addressless_settles_nobody_and_loses_nothing(self, escrow):
        """The current production shape (no registry): nothing resolvable -> nothing booked."""
        esc, led = escrow
        await _accumulate(esc, "creatorA", 5.0, None)
        await _accumulate(esc, "creatorB", 3.0, None)

        await esc.settle_batch(force=True)

        # No valid recipients -> on-chain settlement fails -> nothing paid, nothing booked, nothing lost.
        assert led.paid == []
        assert esc._pending.keys() == {"creatorA", "creatorB"}
        assert esc._total_settled == 0.0
