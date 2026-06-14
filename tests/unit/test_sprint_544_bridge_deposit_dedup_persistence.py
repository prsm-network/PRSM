"""Sprint 544 — persist InboundMonitor's bridge-deposit dedup set
across restart so the sprint-543 catch-up scan path can't
double-credit a tx that was already credited in a previous run.

Pre-sprint regression: ``_credited_tx_hashes`` was in-memory only.
Pre-sprint-543 this was safe because restart re-baselined the
checkpoint to ``current_block`` — old blocks were never re-scanned.
Sprint 543's checkpoint persistence flipped that: on restart the
monitor now resumes from the persisted block, re-scans any blocks
since, and any Transfer event in the catch-up window would be
re-credited because the in-memory dedup set is empty.

The sprint-540 design comment "Persistence across restart is provided
by sprint-501's onchain_tx.db (we check the status before crediting)"
described OUTBOUND tx audit (which sprint 501 does cover), not the
INBOUND credit-side dedup. Sprint 544 closes the gap properly.

Fix shape (mirrors sprint 543's checkpoint store):
``InboundCheckpointStore`` gains ``has_credited_tx(address, tx_hash)``
+ ``mark_credited(address, tx_hash)`` on a new ``credited_deposits``
table. Monitor checks store on every credit attempt; marks on success.
In-memory ``_credited_tx_hashes`` set stays as a fast-path cache for
the common within-process retry case but is no longer the source of
truth.
"""

import pytest


# ── Persistent dedup store ─────────────────────────────────────────


def test_credited_dedup_store_unknown_tx_returns_false(tmp_path):
    """Brand-new store: has_credited_tx returns False."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    assert store.has_credited_tx(
        "0x4acdE458766C704B2511583572303e77109cFFE8",
        "0xa1" * 32,
    ) is False


def test_credited_dedup_store_mark_then_query(tmp_path):
    """mark_credited → has_credited_tx returns True for that tuple."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    tx = "0x0a0d63e6f63515a3bf94af4b1e506be58c4ced357e6ca453466534984a57bc84"

    assert store.has_credited_tx(addr, tx) is False
    store.mark_credited(addr, tx)
    assert store.has_credited_tx(addr, tx) is True


def test_credited_dedup_store_persists_across_instances(tmp_path):
    """Two stores at the same path see each other's writes (the
    restart scenario)."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    db = str(tmp_path / "ck.db")
    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    tx = "0x0a" * 32

    InboundCheckpointStore(db).mark_credited(addr, tx)

    fresh = InboundCheckpointStore(db)
    assert fresh.has_credited_tx(addr, tx) is True


def test_credited_dedup_store_per_address_isolation(tmp_path):
    """tx_hash dedup is scoped to recipient address. Same tx_hash
    appearing for two different recipients (highly unlikely in
    practice but no harm in being precise) is dedup'd independently.
    """
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    a = "0x4acdE458766C704B2511583572303e77109cFFE8"
    b = "0xCba5b409a504480d4C969a47FC74cd8c109F8B15"
    tx = "0x0a" * 32

    store.mark_credited(a, tx)
    assert store.has_credited_tx(a, tx) is True
    assert store.has_credited_tx(b, tx) is False


def test_credited_dedup_store_idempotent_mark(tmp_path):
    """Marking the same tuple twice is a no-op (no UNIQUE violation)."""
    from prsm.economy.ftns_onchain import InboundCheckpointStore

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    addr = "0xabc"
    tx = "0xdeadbeef"
    store.mark_credited(addr, tx)
    store.mark_credited(addr, tx)  # must not raise
    assert store.has_credited_tx(addr, tx) is True


# ── InboundMonitor integration ─────────────────────────────────────


class _StubLedger:
    def __init__(self, address, block_number, transfers=()):
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
    def __init__(self, transfers):
        self._transfers = transfers
        self.calls = []

    class _EventsHolder:
        def __init__(self, parent):
            self.Transfer = parent

    @property
    def events(self):
        return _StubToken._EventsHolder(self)

    def get_logs(self, from_block, to_block, argument_filters=None):
        self.calls.append((from_block, to_block))
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
                return {"from": t["from"], "to": t["to"],
                        "value": t["value"]}[k]
        self.args = _A()


class _RecordingLocalLedger:
    """Captures every credit() call so tests can assert single-credit
    invariants across restart simulation."""

    def __init__(self, linked_eth_address, wallet_id):
        self._linked = linked_eth_address.lower()
        self._wallet = wallet_id
        self.credits: list = []

    async def wallet_for_eth_address(self, eth_address: str):
        if eth_address.lower() == self._linked:
            return self._wallet
        return None

    async def credit(self, *, wallet_id, amount, tx_type, description,
                     idempotency_key=None):
        # sp1101 — mirror the real LocalLedger.credit signature (the inbound
        # monitor now passes a deterministic idempotency_key for exactly-once).
        self.credits.append({
            "wallet_id": wallet_id, "amount": amount,
            "tx_type": tx_type, "description": description,
            "idempotency_key": idempotency_key,
        })


@pytest.mark.asyncio
async def test_monitor_does_not_double_credit_across_restart(tmp_path):
    """The restart double-credit scenario (sprint 543 regression):

    1. Daemon v1 boots. Persists checkpoint = block 100. Tick scans
       blocks 101-110, sees a Transfer at block 105, credits 1 FTNS.
       Marks tx as credited in the persistent dedup store. Crashes.
    2. Daemon v2 boots. Reads checkpoint = block 100 (sprint 543).
       Tick scans 101-current. Sees the SAME Transfer at block 105.
       Pre-sprint-544: in-memory _credited_tx_hashes is empty →
       credit fires AGAIN → 2 FTNS credited from a 1 FTNS deposit.
       Post-sprint-544: persistent dedup store has the tx → skip.
    """
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr_lower = "0x4acde458766c704b2511583572303e77109cffe8"
    addr_checksum = "0x4acdE458766C704B2511583572303e77109cFFE8"
    sender = "0xCba5b409a504480d4C969a47FC74cd8c109F8B15"
    tx_hash = (
        "0x0a0d63e6f63515a3bf94af4b1e506be58c4ced357e6ca453466534984a57bc84"
    )

    db = str(tmp_path / "ck.db")

    # ── v1: seed checkpoint @ 100, run a tick that credits.
    store_v1 = InboundCheckpointStore(db)
    store_v1.set_last_scanned_block(addr_checksum, 100)

    ledger_v1 = _StubLedger(
        address=addr_checksum,
        block_number=110,
        transfers=[{
            "blockNumber": 105,
            "tx_hash": tx_hash,
            "from": sender,
            "to": addr_checksum,
            "value": 1_000_000_000_000_000_000,  # 1 FTNS in wei
        }],
    )
    local_v1 = _RecordingLocalLedger(
        linked_eth_address=sender,  # sender's eth → wallet "alice"
        wallet_id="alice",
    )
    monitor_v1 = InboundMonitor(
        ledger=ledger_v1,
        local_ledger=local_v1,
        checkpoint_store=store_v1,
    )
    await monitor_v1._tick_async()

    # v1 credited once.
    assert len(local_v1.credits) == 1
    assert local_v1.credits[0]["amount"] == 1.0

    # ── v2: fresh process, same DB, ledger CURRENTLY at 110 still
    # (in reality could be higher; what matters is checkpoint sat at
    # 100 due to a crash mid-scan that DIDN'T advance it, so v2 re-
    # scans 101-110).
    # Simulate by manually re-seeding the checkpoint to 100 (the
    # crash-before-advance case).
    InboundCheckpointStore(db).set_last_scanned_block(addr_checksum, 100)

    store_v2 = InboundCheckpointStore(db)
    ledger_v2 = _StubLedger(
        address=addr_checksum,
        block_number=110,
        transfers=[{
            "blockNumber": 105,
            "tx_hash": tx_hash,
            "from": sender,
            "to": addr_checksum,
            "value": 1_000_000_000_000_000_000,
        }],
    )
    local_v2 = _RecordingLocalLedger(
        linked_eth_address=sender, wallet_id="alice",
    )
    monitor_v2 = InboundMonitor(
        ledger=ledger_v2,
        local_ledger=local_v2,
        checkpoint_store=store_v2,
    )
    await monitor_v2._tick_async()

    # CRITICAL INVARIANT: v2 must NOT re-credit the tx that v1
    # already processed.
    assert len(local_v2.credits) == 0, (
        f"v2 double-credited! credits={local_v2.credits!r}"
    )


@pytest.mark.asyncio
async def test_monitor_credits_first_time_only(tmp_path):
    """Within a single process, a new tx is credited; a second tick
    seeing the same tx (e.g. RPC re-emission) is dedup'd. Same logic
    as before sprint 544 — the in-memory fast-path stays, just backed
    by persistent state too."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    sender = "0xCba5b409a504480d4C969a47FC74cd8c109F8B15"
    tx = "0x" + "be" * 32

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    store.set_last_scanned_block(addr, 100)

    ledger = _StubLedger(
        address=addr,
        block_number=110,
        transfers=[{
            "blockNumber": 105, "tx_hash": tx,
            "from": sender, "to": addr,
            "value": 5_000_000_000_000_000_000,  # 5 FTNS
        }],
    )
    local = _RecordingLocalLedger(
        linked_eth_address=sender, wallet_id="bob",
    )
    monitor = InboundMonitor(
        ledger=ledger, local_ledger=local, checkpoint_store=store,
    )
    # First tick: credit fires.
    await monitor._tick_async()
    assert len(local.credits) == 1

    # Second tick from block 110 → 120 sees no new transfers (the
    # block-105 tx is now BELOW last_scanned_block); credit total
    # stays at 1.
    ledger.w3.eth.block_number = 120
    await monitor._tick_async()
    assert len(local.credits) == 1


@pytest.mark.asyncio
async def test_monitor_unlinked_address_does_not_pollute_dedup_store(tmp_path):
    """If the sender's eth address is not linked, the credit is
    skipped — but we should NOT mark the tx as credited (so if the
    user links the address later and we re-scan, the credit still
    fires)."""
    from prsm.economy.ftns_onchain import (
        InboundMonitor, InboundCheckpointStore,
    )

    addr = "0x4acdE458766C704B2511583572303e77109cFFE8"
    unlinked_sender = "0xDeadBeefDeadBeefDeadBeefDeadBeefDeadBeef"
    tx = "0x" + "fe" * 32

    store = InboundCheckpointStore(str(tmp_path / "ck.db"))
    store.set_last_scanned_block(addr, 100)

    ledger = _StubLedger(
        address=addr,
        block_number=110,
        transfers=[{
            "blockNumber": 105, "tx_hash": tx,
            "from": unlinked_sender, "to": addr,
            "value": 1_000_000_000_000_000_000,
        }],
    )
    local = _RecordingLocalLedger(
        linked_eth_address="0xSomethingElseEntirely",
        wallet_id="charlie",
    )
    monitor = InboundMonitor(
        ledger=ledger, local_ledger=local, checkpoint_store=store,
    )
    await monitor._tick_async()
    assert len(local.credits) == 0
    assert store.has_credited_tx(addr, tx) is False, (
        "Unlinked-address skip must not mark tx as credited — the "
        "user might link the address later and replay should credit."
    )
