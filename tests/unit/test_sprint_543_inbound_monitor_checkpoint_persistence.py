"""Sprint 543 — persist ``InboundMonitor._last_scanned_block`` across
restart so deposits arriving during daemon downtime are still credited
when the daemon comes back up.

Pre-sprint behavior: monitor's first tick records ``current_block`` as
the baseline without scanning anything before it. On restart this resets,
silently dropping coverage of any blocks scanned in the previous run —
including any FTNS transfers that arrived to linked operator addresses
during downtime. Pattern A's bridge correctness depends on this gap
being closed: a deposit detected only by chain-side broadcast but never
credited to an off-chain wallet is exactly the failure mode Pattern A
was supposed to avoid (Pattern B contract bridges escape this by
construction — funds locked on-chain stay locked even if the
coordinator daemon is down).

New: ``InboundCheckpointStore`` — tiny SQLite-backed K/V keyed by
``recipient_address``. InboundMonitor takes an optional store; on each
tick it persists the block it scanned through. On startup if a value
exists, ``_last_scanned_block`` is preloaded (catch-up happens on first
tick). ``MAX_CATCHUP_BLOCKS`` (default 100_000 ≈ 56hrs on Base 2s
blocks) clamps unbounded catch-up; chunked scanning (sprint 542)
already handles wide ranges server-side.

Honest scope: the store also dedup-protects against double-credit on
restart via sprint 540's ``_credited_tx_hashes`` set + sprint 501's
onchain_tx.db status check — that's a separate concern. Sprint 543
only owns the block checkpoint.
"""

import sqlite3
from pathlib import Path

import pytest


# ── pin tests ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _legacy_confirmations(monkeypatch):
    # sp1478 — assert checkpoint mechanics against the chain TIP (legacy
    # tip-credit mode; the reorg-confirmation gate is covered by test_sprint_1478).
    monkeypatch.setenv("PRSM_DEPOSIT_CONFIRMATIONS", "0")


def test_checkpoint_store_empty_returns_none(tmp_path):
    """Brand-new store: get returns None for any address."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    assert store.get_last_scanned_block(
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    ) is None


def test_checkpoint_store_set_and_get_roundtrip(tmp_path):
    """set then get returns the same block."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    store.set_last_scanned_block(addr, 46197586)

    assert store.get_last_scanned_block(addr) == 46197586


def test_checkpoint_store_persists_across_instances(tmp_path):
    """Two stores pointed at the same path see each other's writes —
    the production scenario: daemon writes on tick, restart constructs
    a new store, reads it back."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    db = str(tmp_path / "ck.db")
    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"

    s1 = InboundCheckpointStore(db)
    s1.set_last_scanned_block(addr, 46100000)

    s2 = InboundCheckpointStore(db)
    assert s2.get_last_scanned_block(addr) == 46100000


def test_checkpoint_store_set_overwrites(tmp_path):
    """Later set replaces earlier value (the tick-by-tick advance)."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"

    store.set_last_scanned_block(addr, 46_100_000)
    store.set_last_scanned_block(addr, 46_200_000)

    assert store.get_last_scanned_block(addr) == 46_200_000


def test_checkpoint_store_per_address_isolation(tmp_path):
    """Two distinct addresses have independent checkpoints."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    a = "0x4acdE458766C704B2511583572303e77109cFFE8"
    b = "0xCba5b409a504480d4C969a47FC74cd8c109F8B15"

    store.set_last_scanned_block(a, 46_100_000)
    store.set_last_scanned_block(b, 46_200_000)

    assert store.get_last_scanned_block(a) == 46_100_000
    assert store.get_last_scanned_block(b) == 46_200_000


def test_checkpoint_store_creates_directory(tmp_path):
    """If parent dir doesn't exist, store creates it (matches the
    sprint-501 ``onchain_tx.db`` parent-dir handling)."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    nested = tmp_path / "a" / "b" / "ck.db"
    assert not nested.parent.exists()

    store = InboundCheckpointStore(str(nested))
    store.set_last_scanned_block("0xabc", 123)

    assert nested.exists()
    assert store.get_last_scanned_block("0xabc") == 123


def test_checkpoint_store_in_memory_mode(tmp_path):
    """``:memory:`` is an explicit opt-out (matches PRSM_ONCHAIN_TX_DB
    contract). No file is created; values are intra-process only."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    store = InboundCheckpointStore(":memory:")
    store.set_last_scanned_block("0xabc", 42)
    # Same-instance round-trip still works.
    assert store.get_last_scanned_block("0xabc") == 42


# ── InboundMonitor wiring ──────────────────────────────────────────


class _StubLedger:
    """Minimal ledger surface so InboundMonitor can run a tick."""
    def __init__(self, address, block_number, transfers=()):
        # InboundMonitor reads these attrs.
        self.node_id = "stub"
        self._connected_address = address
        self.w3 = _StubW3(block_number)
        self._token = _StubToken(transfers)


class _StubW3:
    def __init__(self, block_number):
        self.eth = _StubEth(block_number)


class _StubEth:
    def __init__(self, block_number):
        self.block_number = block_number


class _StubToken:
    """Carries a fixed transfer list; ignores from_block/to_block."""
    def __init__(self, transfers):
        self._transfers = transfers
        self.calls = []

    class _EventsHolder:
        def __init__(self, parent):
            self._parent = parent
            self.Transfer = parent

        def __getattr__(self, name):
            raise AttributeError(name)

    @property
    def events(self):
        # InboundMonitor → scan_inbound_transfers calls
        # contract.events.Transfer.get_logs(...).
        # Bind self as the Transfer "event" so get_logs lives on us.
        return _StubToken._EventsHolder(self)

    def get_logs(self, from_block, to_block, argument_filters=None):
        self.calls.append((from_block, to_block))
        # Filter the seeded transfers to the requested window.
        out = []
        for t in self._transfers:
            if from_block <= t["blockNumber"] <= to_block:
                out.append(_StubLog(t))
        return out


class _StubLog:
    def __init__(self, t):
        self.blockNumber = t["blockNumber"]
        self.transactionHash = bytes.fromhex(
            t["tx_hash"].removeprefix("0x")
        )

        class _A:
            def __getitem__(self, k):
                return {
                    "from": t["from"], "to": t["to"],
                    "value": t["value"],
                }[k]
        self.args = _A()


@pytest.mark.asyncio
async def test_inbound_monitor_loads_checkpoint_on_first_tick(tmp_path):
    """If a checkpoint exists, first tick uses it as the scan window
    floor instead of re-baselining to current_block."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    db = str(tmp_path / "ck.db")

    # Persisted: last scan ended at block 46_100_000.
    store_seed = InboundCheckpointStore(db)
    store_seed.set_last_scanned_block(addr, 46_100_000)

    # Restart: new monitor, new store instance, same db path.
    store = InboundCheckpointStore(db)
    ledger = _StubLedger(
        address=addr,
        block_number=46_100_100,
        transfers=[],  # window empty for simplicity
    )
    monitor = InboundMonitor(
        ledger=ledger, checkpoint_store=store,
    )

    await monitor._tick_async()

    # Critical: monitor should have scanned the *delta* from the
    # persisted checkpoint (not no-op'd as a fresh first tick).
    assert ledger._token.calls == [(46_100_001, 46_100_100)]


@pytest.mark.asyncio
async def test_inbound_monitor_persists_after_each_tick(tmp_path):
    """Each tick that scans must write the new end-block to the store."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    db = str(tmp_path / "ck.db")
    store = InboundCheckpointStore(db)
    # Seed with a baseline so the first tick actually scans.
    store.set_last_scanned_block(addr, 46_000_000)

    ledger = _StubLedger(
        address=addr, block_number=46_000_500, transfers=[],
    )
    monitor = InboundMonitor(
        ledger=ledger, checkpoint_store=store,
    )
    await monitor._tick_async()

    # Post-tick: store reflects the latest scanned block.
    fresh_store = InboundCheckpointStore(db)
    assert fresh_store.get_last_scanned_block(addr) == 46_000_500


@pytest.mark.asyncio
async def test_inbound_monitor_no_checkpoint_falls_back_to_baseline(tmp_path):
    """Brand-new node, no persisted checkpoint: first tick records
    current_block as baseline (the pre-sprint behavior — sprint-512's
    pull endpoint covers historical scan)."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    db = str(tmp_path / "ck.db")
    store = InboundCheckpointStore(db)  # empty

    ledger = _StubLedger(
        address=addr, block_number=46_100_000, transfers=[],
    )
    monitor = InboundMonitor(
        ledger=ledger, checkpoint_store=store,
    )
    await monitor._tick_async()

    # First tick: no scan, just baseline.
    assert ledger._token.calls == []
    # But the baseline IS persisted now, so future restarts catch up.
    assert (
        InboundCheckpointStore(db).get_last_scanned_block(addr)
        == 46_100_000
    )


@pytest.mark.asyncio
async def test_inbound_monitor_clamps_to_max_catchup(tmp_path):
    """If the persisted checkpoint is so old that catch-up would scan
    > MAX_CATCHUP_BLOCKS, clamp the window floor to
    current_block - MAX_CATCHUP_BLOCKS. The chunker (sprint 542)
    handles the still-large window — this is about bounding restart
    cost so a multi-week downtime doesn't trigger an unbounded scan."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    db = str(tmp_path / "ck.db")

    # Persisted block is 200k behind — half a week of Base 2s blocks.
    store_seed = InboundCheckpointStore(db)
    store_seed.set_last_scanned_block(addr, 46_000_000)

    store = InboundCheckpointStore(db)
    ledger = _StubLedger(
        address=addr, block_number=46_200_000, transfers=[],
    )
    monitor = InboundMonitor(
        ledger=ledger,
        checkpoint_store=store,
        max_catchup_blocks=100_000,
    )
    await monitor._tick_async()

    # Without clamping: (46_000_001, 46_200_000) = 200k blocks.
    # With clamping at 100k: window starts at 46_100_001
    # (current_block - max_catchup_blocks + 1). Sprint 542's chunker
    # then splits the 100k range into ≤9k sub-windows. We assert the
    # CLAMP invariant: first sub-window starts at the clamped floor;
    # last sub-window ends at current_block; nothing scans below.
    calls = ledger._token.calls
    assert calls, "expected at least one underlying RPC call"
    assert calls[0][0] == 46_100_001, (
        f"first sub-window must start at clamped floor; got {calls[0]}"
    )
    assert calls[-1][1] == 46_200_000, (
        f"last sub-window must end at current_block; got {calls[-1]}"
    )
    # No sub-window may dip below the clamped floor.
    assert all(c[0] >= 46_100_001 for c in calls)


@pytest.mark.asyncio
async def test_inbound_monitor_no_store_backcompat(tmp_path):
    """When checkpoint_store=None (sprint 540 default), monitor must
    keep working exactly as before — in-memory only, no persistence,
    first-tick records baseline."""
    from prsm.economy.ftns_onchain import InboundMonitor

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    ledger = _StubLedger(
        address=addr, block_number=46_100_000, transfers=[],
    )
    monitor = InboundMonitor(
        ledger=ledger,  # no checkpoint_store kwarg
    )
    await monitor._tick_async()
    # First tick: baseline + no scan.
    assert ledger._token.calls == []
    assert monitor._last_scanned_block == 46_100_000


def test_node_wiring_passes_checkpoint_store():
    """The production wiring (node.py) should construct the
    checkpoint store with the same db_path policy as the on-chain TX
    DB (sprint 501) and hand it to InboundMonitor."""
    import inspect
    import prsm.node.node as node_mod

    src = inspect.getsource(node_mod)
    # The wiring must reference both the new store class and the
    # checkpoint_store kwarg on InboundMonitor.
    assert "InboundCheckpointStore" in src, (
        "node.py must construct InboundCheckpointStore so the "
        "checkpoint survives restart."
    )
    assert "checkpoint_store=" in src, (
        "node.py must pass checkpoint_store= to InboundMonitor."
    )
