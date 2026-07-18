"""Sprint 514 — background inbound poller + webhook.

Sprint 512 ships /wallet/transactions/onchain/inbound. Sprint 513
wraps it in CLI. Both are pull-based: operator has to actively
query. Sprint 514 adds the push complement:

  - InboundMonitor: periodic background task that scans for new
    Transfer events targeting operator wallet (every
    PRSM_INBOUND_MONITOR_INTERVAL_SECONDS, default 60s)
  - Tracks last_scanned_block to avoid re-scanning the same
    range
  - For each new inbound event: log info + fire webhook
    (gas.transition pattern from sprint 507)
  - Webhook event: "ftns.inbound" with full transfer payload

Boundary: tests inject mocked w3 + scan helper, verify the
poller advances last_scanned_block + fires the webhook on new
events.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.economy.ftns_onchain import (
    InboundMonitor,
    OnChainFTNSLedger,
)


@pytest.fixture(autouse=True)
def _legacy_confirmations(monkeypatch):
    # sp1478 — these tests assert exact checkpoint mechanics against the chain
    # TIP; run them in legacy tip-credit mode (confirmation gate disabled). The
    # reorg-confirmation-depth behavior is covered by test_sprint_1478.
    monkeypatch.setenv("PRSM_DEPOSIT_CONFIRMATIONS", "0")


def _build_ledger():
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
    )
    led._connected_address = (
        "0x" + "a" * 40
    )
    led.w3 = MagicMock()
    led.w3.eth.block_number = 100
    led._token = MagicMock()
    led.w3.eth.contract.return_value = led._token
    return led


@pytest.mark.asyncio
async def test_first_tick_sets_baseline_without_firing():
    """First tick should just record the current block as
    baseline — no scanning, no fire (operator startup log
    + immediate poll covers that)."""
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock()
    led = _build_ledger()
    mon = InboundMonitor(
        led, interval_seconds=60,
        webhook_deliverer=deliverer,
        webhook_url="http://hook",
    )
    await mon._tick_async()
    deliverer.deliver.assert_not_called()
    assert mon._last_scanned_block == 100


@pytest.mark.asyncio
async def test_tick_with_new_inbound_fires_webhook(
    monkeypatch,
):
    """When new Transfer events appear in the new block
    window, fire one webhook per inbound event."""
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock()
    led = _build_ledger()
    mon = InboundMonitor(
        led, interval_seconds=60,
        webhook_deliverer=deliverer,
        webhook_url="http://hook",
    )
    # Baseline
    await mon._tick_async()
    deliverer.deliver.reset_mock()

    # Advance chain
    led.w3.eth.block_number = 110

    # Mock scan to return 2 new inbound events
    fake_transfers = [
        {
            "block_number": 105,
            "tx_hash": "0x" + "11" * 32,
            "from_address": "0xFFFF",
            "to_address": led._connected_address,
            "amount_ftns": 1.5,
        },
        {
            "block_number": 109,
            "tx_hash": "0x" + "22" * 32,
            "from_address": "0xEEEE",
            "to_address": led._connected_address,
            "amount_ftns": 0.25,
        },
    ]
    monkeypatch.setattr(
        "prsm.economy.ftns_onchain.scan_inbound_transfers",
        lambda *a, **kw: fake_transfers,
    )

    await mon._tick_async()

    assert deliverer.deliver.call_count == 2
    payloads = [
        c.kwargs["payload"] for c in deliverer.deliver.call_args_list
    ]
    assert payloads[0]["event"] == "ftns.inbound"
    assert payloads[0]["amount_ftns"] == 1.5
    assert payloads[0]["from_address"] == "0xFFFF"
    assert payloads[1]["amount_ftns"] == 0.25
    assert mon._last_scanned_block == 110


@pytest.mark.asyncio
async def test_no_w3_is_silent_noop():
    """If w3 is not initialized, the monitor must not
    crash on tick."""
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock()
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
    )
    led.w3 = None
    mon = InboundMonitor(
        led, interval_seconds=60,
        webhook_deliverer=deliverer,
        webhook_url="http://hook",
    )
    await mon._tick_async()
    deliverer.deliver.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_failure_does_not_crash(monkeypatch):
    """Webhook delivery exception must NOT crash the
    monitor loop — last_scanned_block still advances so
    we don't re-fire the same event on next tick."""
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock(
        side_effect=RuntimeError("hook down"),
    )
    led = _build_ledger()
    mon = InboundMonitor(
        led, interval_seconds=60,
        webhook_deliverer=deliverer,
        webhook_url="http://hook",
    )
    await mon._tick_async()  # baseline
    led.w3.eth.block_number = 110
    monkeypatch.setattr(
        "prsm.economy.ftns_onchain.scan_inbound_transfers",
        lambda *a, **kw: [
            {
                "block_number": 105,
                "tx_hash": "0x" + "11" * 32,
                "from_address": "0xFFFF",
                "to_address": led._connected_address,
                "amount_ftns": 1.0,
            },
        ],
    )
    # Must not raise
    await mon._tick_async()
    assert mon._last_scanned_block == 110


@pytest.mark.asyncio
async def test_no_op_when_no_new_blocks(monkeypatch):
    """If block_number hasn't advanced, no scan call, no
    fire."""
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock()
    led = _build_ledger()
    mon = InboundMonitor(
        led, interval_seconds=60,
        webhook_deliverer=deliverer,
        webhook_url="http://hook",
    )
    await mon._tick_async()  # baseline at 100

    called = {"n": 0}

    def fake_scan(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(
        "prsm.economy.ftns_onchain.scan_inbound_transfers",
        fake_scan,
    )
    # block_number still 100 — no scan should fire
    await mon._tick_async()
    assert called["n"] == 0
    deliverer.deliver.assert_not_called()
