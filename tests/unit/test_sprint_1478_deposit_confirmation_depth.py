"""Sprint 1478 — bridge-deposit reorg/finality confirmation depth (audit wf_1160e260 LOW).

The bridge-deposit MINT-path audit (sp1472) flagged a now-live-relevant LOW: the
InboundMonitor scanned + credited deposits up to the raw chain TIP
(scan_to = current_block). A Transfer in a block that later REORGS OUT would then
leave an UNBACKED off-chain credit (the mint/counterfeit direction) with no
reversal path, and the checkpoint would have advanced past it so it is never
re-examined.

sp1478 gates crediting on a confirmation depth: only Transfers >= _confirmations
blocks deep are credited (scan_to = current_block - _confirmations), and the
checkpoint advances only to that confirmed tip — so blocks in the confirmation
window are re-scanned on a later tick once deep enough (idempotent under the
sp1472/1473 dedup). A shallow sequencer reorg can no longer produce an unbacked
credit. confirmations=0 preserves the legacy tip-credit behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.ftns_onchain import InboundMonitor, OnChainFTNSLedger

_ADDR = "0x" + "a" * 40


def _mon(confirmations, block_number=110):
    led = OnChainFTNSLedger(node_id="t", wallet_private_key=None)
    led._connected_address = _ADDR
    led.w3 = MagicMock()
    led.w3.eth.block_number = block_number
    led._token = MagicMock()
    mon = InboundMonitor(led, interval_seconds=60, confirmations=confirmations)
    mon._last_scanned_block = 100   # skip the baseline tick
    return led, mon


def _range_aware_scan(dep_block, credited_ranges=None):
    """A scan that returns a deposit at dep_block ONLY when the queried range
    covers it — so it models the real 'a Transfer is only seen once its block is
    scanned' behavior the confirmation gate depends on."""
    def _scan(token, *, recipient, from_block, to_block, **kw):
        if credited_ranges is not None:
            credited_ranges.append((from_block, to_block))
        if from_block <= dep_block <= to_block:
            return [{
                "block_number": dep_block, "tx_hash": "0x" + "11" * 32,
                "from_address": "0xFEED", "to_address": recipient,
                "amount_ftns": 2.0,
            }]
        return []
    return _scan


pytestmark = pytest.mark.asyncio


async def test_deposit_in_confirmation_window_not_credited(monkeypatch):
    """★ A deposit at block 108 with tip 110 + confirmations 5 is within the
    unconfirmed window (108 > 110-5=105) → NOT scanned/credited yet, and the
    checkpoint stops at the CONFIRMED tip (105), not the raw tip (110)."""
    led, mon = _mon(confirmations=5, block_number=110)
    credited = []

    async def _spy(t):
        credited.append(t)
    monkeypatch.setattr(mon, "_credit_deposit", _spy)
    monkeypatch.setattr("prsm.economy.ftns_onchain.scan_inbound_transfers_chunked",
                        _range_aware_scan(108))

    await mon._tick_async()
    assert credited == []                      # 108 not yet confirmation-deep
    assert mon._last_scanned_block == 105       # ★ confirmed tip, NOT 110


async def test_deposit_credited_once_confirmation_deep(monkeypatch):
    """The same deposit IS credited on a later tick once it is >= 5 blocks deep,
    exactly once, and the checkpoint then reaches the new confirmed tip."""
    led, mon = _mon(confirmations=5, block_number=110)
    credited = []

    async def _spy(t):
        credited.append(t)
    monkeypatch.setattr(mon, "_credit_deposit", _spy)
    monkeypatch.setattr("prsm.economy.ftns_onchain.scan_inbound_transfers_chunked",
                        _range_aware_scan(108))

    await mon._tick_async()                      # tip 110 → confirmed 105, no credit
    assert credited == []
    led.w3.eth.block_number = 115                # chain advances; 108 now 7 deep
    await mon._tick_async()                      # confirmed 110, scan [106,110] covers 108
    assert len(credited) == 1
    assert credited[0]["block_number"] == 108
    assert mon._last_scanned_block == 110


async def test_confirmations_zero_is_legacy_tip_credit(monkeypatch):
    """confirmations=0 preserves the pre-sp1478 behavior: credit at the tip."""
    led, mon = _mon(confirmations=0, block_number=110)
    credited = []

    async def _spy(t):
        credited.append(t)
    monkeypatch.setattr(mon, "_credit_deposit", _spy)
    monkeypatch.setattr("prsm.economy.ftns_onchain.scan_inbound_transfers_chunked",
                        _range_aware_scan(108))

    await mon._tick_async()
    assert len(credited) == 1                    # scan_to=110 covers 108
    assert mon._last_scanned_block == 110


async def test_confirmations_default_from_env(monkeypatch):
    monkeypatch.setenv("PRSM_DEPOSIT_CONFIRMATIONS", "3")
    _, mon = _mon(confirmations=None, block_number=110)
    assert mon._confirmations == 3


async def test_confirmations_default_when_unset(monkeypatch):
    monkeypatch.delenv("PRSM_DEPOSIT_CONFIRMATIONS", raising=False)
    _, mon = _mon(confirmations=None, block_number=110)
    assert mon._confirmations == 5              # safe default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
