"""Sprint 545 — don't advance the InboundMonitor checkpoint past a
failed scan window.

Pre-sprint bug introduced by sprint 543's persistence wiring: when
``scan_inbound_transfers_chunked`` raises (RPC outage mid-scan, the
public RPC returning a transient error on one sub-window, etc.) the
monitor's except branch does:

    self._last_scanned_block = current_block
    self._checkpoint_store.set_last_scanned_block(addr, current_block)
    return

That advances the persistent checkpoint to ``current_block`` — past
the block the scan failed on, AND past all the blocks we DIDN'T scan
because the chunker re-raised. Next tick will start from
``current_block + 1``, so the failed window's events are permanently
unrecoverable via the background path. (Sprint 512's pull endpoint
can still scan them manually, but a daemon that only ever fires
auto-credit will silently drop them.)

Fix: on scan failure, leave the checkpoint at its pre-tick value so
the next tick retries the same window. Idempotent — repeated retries
on a persistent RPC outage will keep failing the same way until the
outage clears, then succeed normally. Sprint 544's persistent dedup
prevents double-credit if the failure happened AFTER some events were
already credited (impossible today because the chunker re-raises
without exposing partial results, but the dedup is the belt that the
suspenders depend on).
"""

import pytest


@pytest.fixture(autouse=True)
def _legacy_confirmations(monkeypatch):
    # sp1478 — assert checkpoint failure semantics against the chain TIP (legacy
    # tip-credit mode; the reorg-confirmation gate is covered by test_sprint_1478).
    monkeypatch.setenv("PRSM_DEPOSIT_CONFIRMATIONS", "0")


class _StubW3:
    def __init__(self, block_number):
        self.eth = _StubEth(block_number)


class _StubEth:
    def __init__(self, block_number):
        self.block_number = block_number


class _RaisingToken:
    """A token whose events.Transfer.get_logs raises on every call —
    simulates a sustained RPC outage."""

    class _EventsHolder:
        def __init__(self, parent):
            self.Transfer = parent

    @property
    def events(self):
        return _RaisingToken._EventsHolder(self)

    def get_logs(self, from_block, to_block, argument_filters=None):
        raise RuntimeError("simulated RPC outage")


class _StubLedger:
    def __init__(self, address, block_number, token):
        self.node_id = "stub"
        self._connected_address = address
        self.w3 = _StubW3(block_number)
        self._token = token


@pytest.mark.asyncio
async def test_checkpoint_NOT_advanced_on_scan_failure(tmp_path):
    """The core invariant: if the scan raises, the persisted checkpoint
    must stay at its pre-tick value. Next tick retries the SAME
    window, eventually succeeding when the RPC outage clears."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    db = str(tmp_path / "ck.db")

    # Seed checkpoint at block 100. Daemon was healthy through 100.
    store_seed = InboundCheckpointStore(db)
    store_seed.set_last_scanned_block(addr, 100)

    store = InboundCheckpointStore(db)
    ledger = _StubLedger(
        address=addr,
        block_number=200,  # 100 blocks of catch-up needed
        token=_RaisingToken(),
    )
    monitor = InboundMonitor(
        ledger=ledger, checkpoint_store=store,
    )

    # Tick raises internally — monitor must NOT swallow the failure
    # by advancing the checkpoint.
    await monitor._tick_async()

    # Critical: checkpoint must still read 100 (pre-tick value), not
    # 200 (current_block).
    fresh = InboundCheckpointStore(db)
    assert fresh.get_last_scanned_block(addr) == 100, (
        f"Checkpoint advanced past failed scan! Got "
        f"{fresh.get_last_scanned_block(addr)}, expected 100. "
        "Next tick will skip blocks 101-200 forever."
    )


@pytest.mark.asyncio
async def test_in_memory_state_NOT_advanced_on_scan_failure(tmp_path):
    """Same invariant for the in-memory ``_last_scanned_block``: if
    we advanced it but not the persisted value, next tick within the
    same process would still skip the failed window."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    store.set_last_scanned_block(addr, 100)

    ledger = _StubLedger(
        address=addr, block_number=200,
        token=_RaisingToken(),
    )
    monitor = InboundMonitor(
        ledger=ledger, checkpoint_store=store,
    )

    await monitor._tick_async()

    assert monitor._last_scanned_block == 100, (
        f"In-memory _last_scanned_block advanced past failed scan! "
        f"Got {monitor._last_scanned_block}, expected 100."
    )


@pytest.mark.asyncio
async def test_recovery_after_transient_failure(tmp_path):
    """End-to-end retry semantics: tick 1 raises (RPC outage), tick 2
    succeeds (outage cleared), tick 2 should scan blocks 101-200
    cleanly — proving the bug fix doesn't introduce a permanent
    skipped-window symptom of its own."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    store.set_last_scanned_block(addr, 100)

    class _ToggleToken:
        def __init__(self):
            self.fail_next = True
            self.calls = []

        class _EH:
            def __init__(self, p):
                self.Transfer = p

        @property
        def events(self):
            return _ToggleToken._EH(self)

        def get_logs(self, from_block, to_block,
                     argument_filters=None):
            self.calls.append((from_block, to_block, self.fail_next))
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("transient")
            return []

    token = _ToggleToken()
    ledger = _StubLedger(
        address=addr, block_number=200, token=token,
    )
    monitor = InboundMonitor(
        ledger=ledger, checkpoint_store=store,
    )

    # Tick 1: fails. Checkpoint stays at 100.
    await monitor._tick_async()
    assert store.get_last_scanned_block(addr) == 100

    # Tick 2: succeeds. Scans 101-200 (the SAME window tick 1 tried).
    await monitor._tick_async()
    assert store.get_last_scanned_block(addr) == 200
    # Tick 2's underlying call was (101, 200).
    assert any(
        c[0] == 101 and c[1] == 200 and c[2] is False
        for c in token.calls
    ), f"Expected tick 2 to retry (101, 200); calls={token.calls!r}"


@pytest.mark.asyncio
async def test_first_tick_no_checkpoint_failure_still_baselines(
    tmp_path,
):
    """If the very first tick (no prior checkpoint) hits an RPC
    failure on the BLOCK_NUMBER lookup (not the scan), the monitor
    must not write a corrupt checkpoint."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    class _BadW3:
        class _Eth:
            @property
            def block_number(self):
                raise RuntimeError("RPC down")
        eth = _Eth()

    class _Ledger:
        node_id = "stub"
        _connected_address = "0xabc"
        w3 = _BadW3()
        _token = _RaisingToken()

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    monitor = InboundMonitor(
        ledger=_Ledger(), checkpoint_store=store,
    )

    await monitor._tick_async()

    # block_number raised → tick returns early, no checkpoint write.
    assert store.get_last_scanned_block("0xabc") is None, (
        "Failed block_number lookup must not pollute checkpoint."
    )
