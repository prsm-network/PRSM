"""Watcher persistence integration: KeyDistributionWatcher /
StorageSlashingWatcher / CompensationDistributorWatcher accept an
optional `state_store` kwarg; when provided, restart-resilient
last_processed_block tracking replaces today's chain-tip-baseline-
only behavior.

Closes the afternoon-arc deferred item from
`project_phase78_afternoon_arc_2026_05_08.md` honest-scope.
"""
from __future__ import annotations

import asyncio
from typing import List

import pytest

from prsm.economy.web3.compensation_distributor import DistributedEvent
from prsm.economy.web3.compensation_distributor_watcher import (
    CompensationDistributorWatcher,
)
from prsm.economy.web3.key_distribution import KeyReleasedEvent
from prsm.economy.web3.key_distribution_watcher import (
    KeyDistributionWatcher,
)
from prsm.economy.web3.last_processed_block_store import (
    InMemoryLastProcessedBlockStore,
)
from prsm.economy.web3.storage_slashing import (
    HeartbeatMissingSlashedEvent,
)
from prsm.economy.web3.storage_slashing_watcher import (
    StorageSlashingWatcher,
)


# ──────────────────────────────────────────────────────────────────────
# Stub clients (mirror the existing per-watcher test fixtures)
# ──────────────────────────────────────────────────────────────────────


# sp1457: Base mainnet rejects any eth_getLogs range wider than 10k blocks (-32614). A watcher that
# restarts with a persisted baseline >10k blocks behind must NOT issue one oversized poll and wedge;
# `range_cap` on these fakes reproduces that RPC limit so the tests are discriminating.
_RPC_RANGE_CAP = 10_000


def _enforce_range_cap(cap, from_block, to_block):
    if cap is not None and (int(to_block) - int(from_block) + 1) > cap:
        raise ValueError(
            f"eth_getLogs is limited to a {cap} range (-32614): got "
            f"{int(to_block) - int(from_block) + 1} blocks [{from_block}, {to_block}]")


class _FakeKeyDistributionClient:
    def __init__(self, *, latest_block: int = 100, range_cap=None):
        self._latest_block = latest_block
        self._released: List = []
        self.released_calls = []
        self._range_cap = range_cap

    def latest_block(self):
        return self._latest_block

    def advance_to(self, block):
        self._latest_block = block

    def queue_released(self, ev):
        self._released.append(ev)

    def get_key_released_events(self, from_block, to_block):
        _enforce_range_cap(self._range_cap, from_block, to_block)
        self.released_calls.append((from_block, to_block))
        events = self._released
        self._released = []
        return events

    def get_key_deposited_events(self, from_block, to_block):
        return []

    def get_key_deauthorized_events(self, from_block, to_block):
        return []


class _FakeStorageSlashingClient:
    def __init__(self, *, latest_block: int = 100, range_cap=None):
        self._latest_block = latest_block
        self._missing: List = []
        self.missing_calls = []
        self._range_cap = range_cap

    def latest_block(self):
        return self._latest_block

    def advance_to(self, block):
        self._latest_block = block

    def queue_missing(self, ev):
        self._missing.append(ev)

    def get_heartbeat_missing_slashed_events(self, from_block, to_block):
        _enforce_range_cap(self._range_cap, from_block, to_block)
        self.missing_calls.append((from_block, to_block))
        events = self._missing
        self._missing = []
        return events

    def get_heartbeat_recorded_events(self, from_block, to_block):
        return []

    def get_proof_failure_slashed_events(self, from_block, to_block):
        return []


class _FakeCompensationDistributorClient:
    def __init__(self, *, latest_block: int = 100, range_cap=None):
        self._latest_block = latest_block
        self._distributed: List = []
        self.distributed_calls = []
        self._range_cap = range_cap

    def latest_block(self):
        return self._latest_block

    def advance_to(self, block):
        self._latest_block = block

    def queue_distributed(self, ev):
        self._distributed.append(ev)

    def get_distributed_events(self, from_block, to_block):
        _enforce_range_cap(self._range_cap, from_block, to_block)
        self.distributed_calls.append((from_block, to_block))
        events = self._distributed
        self._distributed = []
        return events


# ──────────────────────────────────────────────────────────────────────
# KeyDistributionWatcher persistence
# ──────────────────────────────────────────────────────────────────────


class TestKeyDistributionWatcherPersistence:
    @pytest.mark.asyncio
    async def test_first_tick_loads_from_store(self):
        """When state_store has a persisted block, first tick should
        use it (instead of the chain-tip baseline). Subsequent ticks
        poll forward from the persisted point."""
        store = InMemoryLastProcessedBlockStore()
        store.save("key_distribution", 50)  # restart point

        client = _FakeKeyDistributionClient(latest_block=200)
        # Queue an event that landed during the "downtime window"
        # (block range 51-200 — events the watcher would have missed
        # without persistence).
        client.queue_released(KeyReleasedEvent(
            content_hash=b"\xaa" * 32,
            recipient="0x" + "11" * 20,
            encrypted_key=b"ct1",
        ))
        events = []

        async def cb(ev):
            events.append(ev)

        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
            state_store=store,
        )
        await watcher.tick()

        # Polled the downtime range [51, 200] — NOT first-tick
        # baselined at 200.
        assert client.released_calls == [(51, 200)]
        assert len(events) == 1
        assert watcher.last_processed_block == 200
        # Persisted the new advance.
        assert store.load("key_distribution") == 200

    @pytest.mark.asyncio
    async def test_first_tick_with_empty_store_baselines_at_tip(self):
        """When state_store is empty, first tick falls back to
        chain-tip baseline (current behavior preserved). The new
        baseline is then persisted."""
        store = InMemoryLastProcessedBlockStore()
        client = _FakeKeyDistributionClient(latest_block=500)
        async def cb(ev):
            pass

        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
            state_store=store,
        )
        await watcher.tick()

        assert client.released_calls == []  # no history replay
        assert watcher.last_processed_block == 500
        # Baseline persisted.
        assert store.load("key_distribution") == 500

    @pytest.mark.asyncio
    async def test_subsequent_tick_persists_advance(self):
        """Each baseline advance triggers a save call."""
        store = InMemoryLastProcessedBlockStore()
        client = _FakeKeyDistributionClient(latest_block=100)
        async def cb(ev):
            pass

        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
            state_store=store,
        )
        await watcher.tick()  # baseline 100
        assert store.load("key_distribution") == 100

        client.advance_to(150)
        await watcher.tick()
        assert store.load("key_distribution") == 150

        client.advance_to(200)
        await watcher.tick()
        assert store.load("key_distribution") == 200

    @pytest.mark.asyncio
    async def test_no_state_store_preserves_legacy_behavior(self):
        """When state_store kwarg is omitted, the watcher behaves
        EXACTLY as it did pre-persistence (chain-tip baseline; no
        history replay; no persistence)."""
        client = _FakeKeyDistributionClient(latest_block=500)
        async def cb(ev):
            pass

        # NO state_store kwarg — legacy path.
        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
        )
        await watcher.tick()
        assert watcher.last_processed_block == 500
        # No replay attempted (no events polled on first tick).
        assert client.released_calls == []

    @pytest.mark.asyncio
    async def test_store_save_failure_does_not_crash_watcher(self):
        """If state_store.save raises, the watcher must not crash —
        next successful tick re-persists."""
        client = _FakeKeyDistributionClient(latest_block=100)

        class _BrokenStore:
            def load(self, key):
                return None
            def save(self, key, block):
                raise OSError("disk full")
            def delete(self, key):
                pass

        async def cb(ev):
            pass

        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
            state_store=_BrokenStore(),
        )
        # Must not raise.
        await watcher.tick()
        # Baseline still advanced internally.
        assert watcher.last_processed_block == 100

    @pytest.mark.asyncio
    async def test_store_load_returns_none_on_corrupt_falls_back_to_tip(self):
        """When store.load returns None (e.g., corrupt file), watcher
        falls back to chain-tip baseline rather than refusing to
        start."""
        class _AlwaysNoneStore:
            def load(self, key):
                return None
            def save(self, key, block):
                pass
            def delete(self, key):
                pass

        client = _FakeKeyDistributionClient(latest_block=999)
        async def cb(ev):
            pass

        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
            state_store=_AlwaysNoneStore(),
        )
        await watcher.tick()
        assert watcher.last_processed_block == 999


# ──────────────────────────────────────────────────────────────────────
# StorageSlashingWatcher persistence
# ──────────────────────────────────────────────────────────────────────


class TestStorageSlashingWatcherPersistence:
    @pytest.mark.asyncio
    async def test_first_tick_loads_from_store(self):
        store = InMemoryLastProcessedBlockStore()
        store.save("storage_slashing", 50)

        client = _FakeStorageSlashingClient(latest_block=200)
        client.queue_missing(HeartbeatMissingSlashedEvent(
            provider="0x" + "11" * 20,
            challenger="0x" + "22" * 20,
            last_heartbeat_at=1700000000,
            slash_id=b"\x55" * 32,
        ))
        events = []

        async def cb(ev):
            events.append(ev)

        watcher = StorageSlashingWatcher(
            client=client,
            on_heartbeat_missing_slashed=cb,
            state_store=store,
        )
        await watcher.tick()

        assert client.missing_calls == [(51, 200)]
        assert len(events) == 1
        assert store.load("storage_slashing") == 200

    @pytest.mark.asyncio
    async def test_no_store_preserves_legacy_behavior(self):
        client = _FakeStorageSlashingClient(latest_block=500)
        async def cb(ev):
            pass
        watcher = StorageSlashingWatcher(
            client=client,
            on_heartbeat_missing_slashed=cb,
        )
        await watcher.tick()
        assert watcher.last_processed_block == 500
        assert client.missing_calls == []


# ──────────────────────────────────────────────────────────────────────
# CompensationDistributorWatcher persistence
# ──────────────────────────────────────────────────────────────────────


class TestCompensationDistributorWatcherPersistence:
    @pytest.mark.asyncio
    async def test_first_tick_loads_from_store(self):
        store = InMemoryLastProcessedBlockStore()
        store.save("compensation_distributor", 50)

        client = _FakeCompensationDistributorClient(latest_block=200)
        client.queue_distributed(DistributedEvent(
            to_creator=10**17, to_operator=10**17, to_grant=10**17,
        ))
        events = []

        async def cb(ev):
            events.append(ev)

        watcher = CompensationDistributorWatcher(
            client=client,
            on_distributed=cb,
            state_store=store,
        )
        await watcher.tick()

        assert client.distributed_calls == [(51, 200)]
        assert len(events) == 1
        assert store.load("compensation_distributor") == 200

    @pytest.mark.asyncio
    async def test_no_store_preserves_legacy_behavior(self):
        client = _FakeCompensationDistributorClient(latest_block=500)
        async def cb(ev):
            pass
        watcher = CompensationDistributorWatcher(
            client=client,
            on_distributed=cb,
        )
        await watcher.tick()
        assert watcher.last_processed_block == 500
        assert client.distributed_calls == []


# ──────────────────────────────────────────────────────────────────────
# Cross-watcher: distinct keys don't collide
# ──────────────────────────────────────────────────────────────────────


class TestCrossWatcherKeyIsolation:
    @pytest.mark.asyncio
    async def test_three_watchers_share_store_without_collision(self):
        """All three watchers can share the same state_store instance —
        each uses its own watcher_key namespace."""
        store = InMemoryLastProcessedBlockStore()

        kd_client = _FakeKeyDistributionClient(latest_block=100)
        ss_client = _FakeStorageSlashingClient(latest_block=200)
        cd_client = _FakeCompensationDistributorClient(latest_block=300)

        async def cb(ev):
            pass

        kd_watcher = KeyDistributionWatcher(
            client=kd_client, on_key_released=cb, state_store=store,
        )
        ss_watcher = StorageSlashingWatcher(
            client=ss_client, on_heartbeat_missing_slashed=cb,
            state_store=store,
        )
        cd_watcher = CompensationDistributorWatcher(
            client=cd_client, on_distributed=cb, state_store=store,
        )

        await kd_watcher.tick()
        await ss_watcher.tick()
        await cd_watcher.tick()

        # Each watcher's key persisted independently.
        assert store.load("key_distribution") == 100
        assert store.load("storage_slashing") == 200
        assert store.load("compensation_distributor") == 300


# ──────────────────────────────────────────────────────────────────────
# sp1457: restart-after-long-downtime must not wedge on the RPC getLogs cap
# ──────────────────────────────────────────────────────────────────────
#
# A node down > ~10k blocks (~5.5h on Base) restarts with a persisted baseline that far behind.
# The old tick polled [persisted+1, latest] in ONE call; on Base that exceeds the 10k eth_getLogs
# range cap (-32614), the watcher swallows the error and never advances → it wedges SILENTLY forever
# (every tick retries the same oversized range). The fix caps each poll to a <=9k-block window so a
# far-behind watcher advances one bounded window per tick and catches up across successive ticks.


class TestWatcherDowntimeBackfillStaysUnderRpcCap:
    @pytest.mark.asyncio
    async def test_key_distribution_far_behind_does_not_wedge(self):
        store = InMemoryLastProcessedBlockStore()
        store.save("key_distribution", 1_000)                 # restart baseline, far behind
        client = _FakeKeyDistributionClient(latest_block=40_000, range_cap=_RPC_RANGE_CAP)

        async def cb(ev):
            pass

        watcher = KeyDistributionWatcher(
            client=client, on_key_released=cb, state_store=store)

        await watcher.tick()
        # Progress made (not wedged at 1000) and the oversized [1001, 40000] call was never issued.
        assert watcher.last_processed_block > 1_000
        assert all(hi - lo + 1 <= _RPC_RANGE_CAP for (lo, hi) in client.released_calls)

        # Successive ticks catch all the way up, every call staying under the cap.
        for _ in range(10):
            await watcher.tick()
        assert watcher.last_processed_block == 40_000
        assert all(hi - lo + 1 <= _RPC_RANGE_CAP for (lo, hi) in client.released_calls)

    @pytest.mark.asyncio
    async def test_storage_slashing_far_behind_does_not_wedge(self):
        store = InMemoryLastProcessedBlockStore()
        store.save("storage_slashing", 500)
        client = _FakeStorageSlashingClient(latest_block=30_000, range_cap=_RPC_RANGE_CAP)

        async def cb(ev):
            pass

        watcher = StorageSlashingWatcher(
            client=client, on_heartbeat_missing_slashed=cb, state_store=store)

        await watcher.tick()
        assert watcher.last_processed_block > 500
        assert all(hi - lo + 1 <= _RPC_RANGE_CAP for (lo, hi) in client.missing_calls)

        for _ in range(10):
            await watcher.tick()
        assert watcher.last_processed_block == 30_000
        assert all(hi - lo + 1 <= _RPC_RANGE_CAP for (lo, hi) in client.missing_calls)

    @pytest.mark.asyncio
    async def test_compensation_distributor_far_behind_does_not_wedge(self):
        store = InMemoryLastProcessedBlockStore()
        store.save("compensation_distributor", 2_000)
        client = _FakeCompensationDistributorClient(latest_block=25_000, range_cap=_RPC_RANGE_CAP)

        async def cb(ev):
            pass

        watcher = CompensationDistributorWatcher(
            client=client, on_distributed=cb, state_store=store)

        await watcher.tick()
        assert watcher.last_processed_block > 2_000
        assert all(hi - lo + 1 <= _RPC_RANGE_CAP for (lo, hi) in client.distributed_calls)

        for _ in range(10):
            await watcher.tick()
        assert watcher.last_processed_block == 25_000
        assert all(hi - lo + 1 <= _RPC_RANGE_CAP for (lo, hi) in client.distributed_calls)
