"""Sprint 1140 — settlement-receipt data plane BRICK E.2: the live-daemon WIRING.

Brick E.1 (sp1139) produced the PURE, offline-verified observe pass
(``SettlementAuditLoop.run_once``). Brick E.2 is the OPT-IN daemon glue that turns it
into a running node capability — and the logic that node.py defers to lives in the
fully-unit-tested ``prsm.settlement.settlement_audit_wiring`` module so node.py keeps
only thin env-gated glue.

These tests assert the load-bearing properties (NO network — fake gossip / fetcher /
chain-head / a tmp_path cursor; the REAL engine/cache/selectors/loop where cheap):

  (a) ``audit_enabled`` parses the env flag correctly (on/off/garbage).
  (b) ``build_settlement_audit_components`` returns None when disabled; a full bundle
      when enabled.
  (c) SCHEDULER block-cursor: a tick scans [cursor+1, head], advances the cursor to
      head on success; a SECOND tick with the SAME head does nothing (to_block <
      from_block) — no gap, no re-scan/overlap.
  (d) RETRY-ON-FAILURE: a tick whose run_once raises does NOT advance the cursor and
      does NOT crash (the next tick retries the same window).
  (e) ``announce_committed_batches`` announces each batch; an announcer error never
      raises.
  (f) NO-BROADCAST: a tick driven end-to-end with an injected dry_run_client whose
      submit/broadcast is a Mock asserts those are NEVER called.
  (g) the composite selector unions OwnStake + ConsensusGroup correctly.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional
from unittest.mock import Mock

import pytest

from prsm.settlement.settlement_audit_engine import (
    ConsensusGroupSelector,
    ObservedBatch,
    OwnStakeSelector,
    SettlementAuditEngine,
    VerifiedBatchCache,
)
from prsm.settlement.settlement_audit_loop import AuditRunResult, SettlementAuditLoop
from prsm.settlement.settlement_audit_wiring import (
    BlockCursor,
    SettlementAuditScheduler,
    announce_committed_batches,
    audit_enabled,
    build_settlement_audit_components,
)


# ── fakes ─────────────────────────────────────────────────────────────


class _FakeGossip:
    """Records subscriptions + publishes; satisfies SettlementBatchAdIndex
    (subscribe on construction) + SettlementBatchAnnouncer (publish)."""

    def __init__(self):
        self.subscriptions = []
        self.published = []

    def subscribe(self, subtype, handler):
        self.subscriptions.append((subtype, handler))

    async def publish(self, subtype, data):
        self.published.append((subtype, data))


class _FakeEnumerator:
    """Records the [from_block, to_block] windows it was asked to scan; returns []."""

    def __init__(self):
        self.windows: List[tuple] = []

    async def enumerate_committed_batches(self, from_block, to_block, *, provider=None):
        self.windows.append((from_block, to_block))
        return []


class _RaisingEnumerator:
    def __init__(self):
        self.calls = 0

    async def enumerate_committed_batches(self, from_block, to_block, *, provider=None):
        self.calls += 1
        raise RuntimeError("chain RPC down")


class _FakeFetcher:
    async def fetch(self, cid):
        raise AssertionError("fetch must not be called when there is nothing selected")


class _FakeContractClient:
    """Enumerator + has no broadcast surface — the loop only ever reads from it."""

    def __init__(self):
        self.windows: List[tuple] = []

    async def enumerate_committed_batches(self, from_block, to_block, *, provider=None):
        self.windows.append((from_block, to_block))
        return []


# ── (a) audit_enabled ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("Yes", True),
        ("on", True),
        ("  on  ", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("garbage", False),
    ],
)
def test_audit_enabled_parses_flag(value, expected):
    assert audit_enabled({"PRSM_SETTLEMENT_AUDIT": value}) is expected


def test_audit_enabled_default_off_when_unset():
    assert audit_enabled({}) is False


# ── (b) build_settlement_audit_components gating ─────────────────────────


def _build(enabled, **over):
    gossip = over.pop("gossip", _FakeGossip())
    contract = over.pop("contract_client", _FakeContractClient())
    cursor = over.pop("cursor", BlockCursor(path=":memory:"))
    kwargs = dict(
        enabled=enabled,
        gossip=gossip,
        fetcher=_FakeFetcher(),
        contract_client=contract,
        my_address="0xabc0000000000000000000000000000000000001",
        group_ids=set(),
        dry_run_client=None,
        cursor=cursor,
    )
    kwargs.update(over)
    return build_settlement_audit_components(**kwargs)


def test_build_returns_none_when_disabled():
    assert _build(enabled=False) is None


def test_build_returns_full_bundle_when_enabled():
    gossip = _FakeGossip()
    bundle = _build(enabled=True, gossip=gossip)
    assert bundle is not None
    # The ad index subscribed to the availability subtype on construction.
    assert len(gossip.subscriptions) == 1
    # Every load-bearing component is present.
    assert isinstance(bundle.ad_index, object)
    assert isinstance(bundle.cache, VerifiedBatchCache)
    assert isinstance(bundle.engine, SettlementAuditEngine)
    assert isinstance(bundle.audit_loop, SettlementAuditLoop)
    assert isinstance(bundle.scheduler, SettlementAuditScheduler)
    assert bundle.selector is not None


# ── (c) scheduler block-cursor advancement, no gaps/overlap ──────────────


@pytest.mark.asyncio
async def test_scheduler_advances_cursor_to_head_then_idles_on_same_head(tmp_path):
    enumerator = _FakeEnumerator()
    cursor = BlockCursor(path=str(tmp_path / "cursor.json"))
    cursor.save(100)  # last fully-scanned block

    head = {"v": 150}

    async def chain_head_fn():
        return head["v"]

    loop = Mock()

    async def _run_once(from_block, to_block, *, provider=None):
        enumerator.windows.append((from_block, to_block))
        return AuditRunResult(enumerated=0)

    loop.run_once = _run_once

    sched = SettlementAuditScheduler(
        loop, chain_head_fn, cursor=cursor, interval_s=0.0,
    )

    # First tick: scans (101, 150), advances cursor to 150.
    await sched.run_once_tick()
    assert enumerator.windows == [(101, 150)]
    assert cursor.load() == 150

    # Second tick, SAME head: from_block (151) > to_block (150) => no scan, no
    # advance. No gap, no overlap.
    await sched.run_once_tick()
    assert enumerator.windows == [(101, 150)]
    assert cursor.load() == 150

    # Head moves forward: scans exactly (151, 175) — contiguous, no gap.
    head["v"] = 175
    await sched.run_once_tick()
    assert enumerator.windows == [(101, 150), (151, 175)]
    assert cursor.load() == 175


@pytest.mark.asyncio
async def test_scheduler_cursor_persists_across_instances(tmp_path):
    cursor_path = str(tmp_path / "cursor.json")
    c1 = BlockCursor(path=cursor_path)
    c1.save(4242)
    c2 = BlockCursor(path=cursor_path)
    assert c2.load() == 4242


# ── (d) retry-on-failure: a failed tick does not advance the cursor ──────


@pytest.mark.asyncio
async def test_scheduler_failed_tick_does_not_advance_cursor_or_crash(tmp_path):
    cursor = BlockCursor(path=str(tmp_path / "cursor.json"))
    cursor.save(10)

    async def chain_head_fn():
        return 20

    calls = {"n": 0}

    async def _run_once(from_block, to_block, *, provider=None):
        calls["n"] += 1
        raise RuntimeError("transient audit failure")

    loop = Mock()
    loop.run_once = _run_once

    sched = SettlementAuditScheduler(
        loop, chain_head_fn, cursor=cursor, interval_s=0.0,
    )

    # Tick raises internally but the scheduler swallows it (never crashes the daemon).
    await sched.run_once_tick()  # must not raise
    assert calls["n"] == 1
    # Cursor NOT advanced — the same window retries next tick.
    assert cursor.load() == 10

    # Next tick retries the SAME window [11, 20].
    await sched.run_once_tick()
    assert calls["n"] == 2
    assert cursor.load() == 10


@pytest.mark.asyncio
async def test_scheduler_chain_head_error_does_not_advance_or_crash(tmp_path):
    cursor = BlockCursor(path=str(tmp_path / "cursor.json"))
    cursor.save(10)

    async def chain_head_fn():
        raise RuntimeError("eth_blockNumber failed")

    loop = Mock()
    loop.run_once = Mock(side_effect=AssertionError("run_once must not run on head failure"))

    sched = SettlementAuditScheduler(
        loop, chain_head_fn, cursor=cursor, interval_s=0.0,
    )
    await sched.run_once_tick()  # must not raise
    assert cursor.load() == 10


@pytest.mark.asyncio
async def test_scheduler_run_forever_loops_and_stops_on_cancel(tmp_path):
    cursor = BlockCursor(path=str(tmp_path / "cursor.json"))
    cursor.save(0)
    ticks = {"n": 0}

    async def chain_head_fn():
        ticks["n"] += 1
        return ticks["n"]

    async def _run_once(from_block, to_block, *, provider=None):
        return AuditRunResult()

    loop = Mock()
    loop.run_once = _run_once

    slept = {"n": 0}

    async def _sleep(_s):
        slept["n"] += 1
        if slept["n"] >= 3:
            raise asyncio.CancelledError()

    sched = SettlementAuditScheduler(
        loop, chain_head_fn, cursor=cursor, interval_s=0.01, sleep=_sleep,
    )
    with pytest.raises(asyncio.CancelledError):
        await sched.run_forever()
    # It ran multiple ticks and advanced the cursor as the head climbed.
    assert ticks["n"] >= 3
    assert cursor.load() >= 3


# ── (e) announce_committed_batches ───────────────────────────────────────


class _RecordingAnnouncer:
    def __init__(self):
        self.announced: List[bytes] = []

    async def announce(self, batch_id):
        self.announced.append(bytes(batch_id))
        return True


class _RaisingAnnouncer:
    async def announce(self, batch_id):
        raise RuntimeError("gossip down")


class _Batch:
    def __init__(self, batch_id):
        self.batch_id = batch_id


@pytest.mark.asyncio
async def test_announce_committed_batches_announces_each():
    announcer = _RecordingAnnouncer()
    batches = [_Batch(b"\x01" * 32), _Batch(b"\x02" * 32)]
    n = await announce_committed_batches(announcer, batches)
    assert n == 2
    assert announcer.announced == [b"\x01" * 32, b"\x02" * 32]


@pytest.mark.asyncio
async def test_announce_accepts_raw_batch_id_bytes():
    announcer = _RecordingAnnouncer()
    n = await announce_committed_batches(announcer, [b"\x09" * 32])
    assert n == 1
    assert announcer.announced == [b"\x09" * 32]


@pytest.mark.asyncio
async def test_announce_committed_batches_never_raises_on_error():
    announcer = _RaisingAnnouncer()
    # A raising announcer must be swallowed (best-effort); count of successes is 0.
    n = await announce_committed_batches(announcer, [_Batch(b"\x03" * 32)])
    assert n == 0


@pytest.mark.asyncio
async def test_announce_no_announcer_is_noop():
    n = await announce_committed_batches(None, [_Batch(b"\x04" * 32)])
    assert n == 0


@pytest.mark.asyncio
async def test_announce_empty_list_is_noop():
    announcer = _RecordingAnnouncer()
    n = await announce_committed_batches(announcer, [])
    assert n == 0
    assert announcer.announced == []


# ── (f) NO-BROADCAST: a full tick never submits/broadcasts ───────────────


@pytest.mark.asyncio
async def test_full_tick_never_broadcasts(tmp_path):
    """Drive a real bundle's scheduler tick end-to-end with an injected dry_run_client
    whose submit/broadcast/finalize are Mocks; assert NONE are ever called. The audit
    path is read-only (it only ever calls dry_run on the engine, which we exercise via
    the empty enumerate path here — the property under test is that the WIRING never
    reaches a broadcast surface)."""
    dry_run_client = Mock()
    dry_run_client.submit = Mock()
    dry_run_client.broadcast = Mock()
    dry_run_client.finalize = Mock()
    dry_run_client.commit_batch = Mock()
    # dry_run itself returns a benign verdict if ever called.
    dry_run_client.dry_run = Mock(
        return_value=Mock(would_succeed=False, revert_reason="n/a")
    )

    gossip = _FakeGossip()
    contract = _FakeContractClient()
    cursor = BlockCursor(path=str(tmp_path / "cursor.json"))
    cursor.save(0)

    bundle = build_settlement_audit_components(
        enabled=True,
        gossip=gossip,
        fetcher=_FakeFetcher(),
        contract_client=contract,
        my_address="0xabc0000000000000000000000000000000000001",
        group_ids=set(),
        dry_run_client=dry_run_client,
        cursor=cursor,
    )
    assert bundle is not None

    head = {"v": 50}

    async def chain_head_fn():
        return head["v"]

    bundle.scheduler._chain_head_fn = chain_head_fn  # inject the fake head
    await bundle.scheduler.run_once_tick()

    # The window was scanned (read-only) and the cursor advanced...
    assert contract.windows == [(1, 50)]
    assert cursor.load() == 50
    # ...and NOTHING was ever broadcast/submitted/finalized.
    dry_run_client.submit.assert_not_called()
    dry_run_client.broadcast.assert_not_called()
    dry_run_client.finalize.assert_not_called()
    dry_run_client.commit_batch.assert_not_called()
    # No gossip publish happened either (no batches => nothing to announce here).
    assert gossip.published == []


# ── (g) composite selector unions OwnStake + ConsensusGroup ──────────────


def _ob(batch_id, requester, group_id=None):
    return ObservedBatch(
        batch_id=batch_id,
        provider_address="0xprov",
        requester_address=requester,
        merkle_root=b"\x00" * 32,
        consensus_group_id=group_id if group_id is not None else b"\x00" * 32,
    )


def test_composite_selector_unions_ownstake_and_group(tmp_path):
    my_addr = "0xMyAddrAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    my_group = b"\x07" * 32
    bundle = build_settlement_audit_components(
        enabled=True,
        gossip=_FakeGossip(),
        fetcher=_FakeFetcher(),
        contract_client=_FakeContractClient(),
        my_address=my_addr,
        group_ids={my_group},
        dry_run_client=None,
        cursor=BlockCursor(path=":memory:"),
    )
    assert bundle is not None

    mine_by_stake = _ob(b"\x01" * 32, my_addr)
    mine_by_group = _ob(b"\x02" * 32, "0xother", group_id=my_group)
    both = _ob(b"\x03" * 32, my_addr, group_id=my_group)  # in BOTH sets — no dupe
    neither = _ob(b"\x04" * 32, "0xother", group_id=b"\x08" * 32)

    selected = bundle.selector.select([mine_by_stake, mine_by_group, both, neither])
    ids = {bytes(b.batch_id) for b in selected}
    assert ids == {b"\x01" * 32, b"\x02" * 32, b"\x03" * 32}
    # No duplicate for the batch that matched both selectors.
    assert len(selected) == 3


def test_composite_selector_ownstake_only_when_no_groups(tmp_path):
    my_addr = "0xMyAddrAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    bundle = build_settlement_audit_components(
        enabled=True,
        gossip=_FakeGossip(),
        fetcher=_FakeFetcher(),
        contract_client=_FakeContractClient(),
        my_address=my_addr,
        group_ids=set(),  # no groups => OwnStake only
        dry_run_client=None,
        cursor=BlockCursor(path=":memory:"),
    )
    assert bundle is not None
    mine = _ob(b"\x01" * 32, my_addr)
    grouped = _ob(b"\x02" * 32, "0xother", group_id=b"\x07" * 32)
    selected = bundle.selector.select([mine, grouped])
    ids = {bytes(b.batch_id) for b in selected}
    assert ids == {b"\x01" * 32}
