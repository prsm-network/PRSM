"""Sprint 1039 — durable settlement batch state (brick 2.5).

The on-chain settlement client (BatchSettlementClient) holds its post-commit
lifecycle state in three in-memory dicts:
  - _tracked          committed batches awaiting finalization
  - _pending_commits  broadcast-but-unconfirmed commits (the sp1022 quarantine)
  - _finalized_ids    batches seen finalized

On-chain money is ALREADY escrowed for every entry in these dicts, so losing
them on a daemon restart STRANDS that money: a committed batch is never
finalized (escrow never releases) and a quarantined commit is never reconciled.
Brick 2 (the poll loop) is therefore unsafe to ever run with a funded key until
this state is durable. That is what brick 2.5 adds.

This suite drives:
  1. SettlementStateStore — an atomic JSON dict store (save/load), loud (not
     silent-empty) on a corrupt file, no .tmp left behind.
  2. BatchSettlementClient rehydration — every one of the three state dicts
     survives a "restart" (a fresh client over the same store), finalize +
     reconcile work on the rehydrated state, and persistence is PER-MUTATION
     (a failure mid commit-loop never loses an already-committed batch).
  3. Opt-out — store=None preserves the exact pre-1039 in-memory behavior and
     writes no file.
"""
from __future__ import annotations

import hashlib
from base64 import b64encode
from unittest.mock import AsyncMock

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import (
    AccumulatorConfig,
    BatchedReceipt,
    ReceiptAccumulator,
)
from prsm.settlement.client import BatchSettlementClient


ONE_FTNS = 10**18
PROVIDER_ADDR = "0x" + "b" * 40
REQUESTER_A = "0x" + "a" * 40


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _make_batched(shard_index: int = 0, value_ftns: int = ONE_FTNS) -> BatchedReceipt:
    receipt = ShardExecutionReceipt(
        job_id=f"job-{shard_index}",
        shard_index=shard_index,
        provider_id="provider-id-string",
        provider_pubkey_b64=b64encode(b"pubkey-raw").decode(),
        output_hash=hashlib.sha256(f"out-{shard_index}".encode()).hexdigest(),
        executed_at_unix=1700000000 + shard_index,
        signature=b64encode(b"sig-raw").decode(),
    )
    return BatchedReceipt(
        receipt=receipt,
        requester_address=REQUESTER_A,
        provider_address=PROVIDER_ADDR,
        value_ftns=value_ftns,
        local_escrow_id=f"escrow-{shard_index}",
    )


def _commit_one_acc() -> ReceiptAccumulator:
    """Accumulator that fires a ready batch on a single receipt."""
    return ReceiptAccumulator(AccumulatorConfig(count_threshold=1))


def _mock_contract(commit_return=None, finalizable=False, status=1):
    mock = AsyncMock()
    if commit_return is None:
        n = {"i": 0}

        async def commit_default(**kw):
            n["i"] += 1
            return (hashlib.sha256(f"batch-{n['i']}".encode()).digest(),
                    1700000000 + n["i"])
        mock.commit_batch.side_effect = commit_default
    else:
        mock.commit_batch.return_value = commit_return
    mock.is_finalizable.return_value = finalizable
    mock.finalize_batch.return_value = None
    mock.get_batch_status.return_value = status
    return mock


def _store(tmp_path):
    from prsm.settlement.state_store import SettlementStateStore
    return SettlementStateStore(tmp_path / "settlement_state.json")


# ── 1. SettlementStateStore ───────────────────────────────────────────


def test_state_store_save_load_roundtrip(tmp_path):
    s = _store(tmp_path)
    payload = {"tracked": {"x": 1}, "finalized_ids": ["a", "b"], "n": 2**90}
    s.save(payload)
    assert s.load() == payload


def test_state_store_load_missing_returns_none(tmp_path):
    assert _store(tmp_path).load() is None


def test_state_store_corrupt_file_raises_not_silent(tmp_path):
    from prsm.settlement.state_store import (
        SettlementStateStore,
        SettlementStateCorruptError,
    )
    p = tmp_path / "settlement_state.json"
    p.write_text("{ this is not valid json ")
    # Loud, NOT silent-empty: an empty rehydrate would strand committed escrow.
    with pytest.raises(SettlementStateCorruptError):
        SettlementStateStore(p).load()


def test_state_store_non_object_json_raises_not_silent(tmp_path):
    """A file truncated/replaced to valid-JSON-but-not-an-object (null, 0, [])
    must fail loud, not rehydrate to silent-empty."""
    from prsm.settlement.state_store import (
        SettlementStateStore,
        SettlementStateCorruptError,
    )
    for content in ("null", "0", "[]", '"hi"'):
        p = tmp_path / f"state_{content.strip('[]\"')}.json"
        p.write_text(content)
        with pytest.raises(SettlementStateCorruptError):
            SettlementStateStore(p).load()


def test_state_store_save_leaves_no_tmp_behind(tmp_path):
    s = _store(tmp_path)
    s.save({"a": 1})
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_state_store_creates_parent_dir(tmp_path):
    from prsm.settlement.state_store import SettlementStateStore
    nested = tmp_path / "deep" / "nested" / "state.json"
    s = SettlementStateStore(nested)
    s.save({"a": 1})
    assert nested.exists() and s.load() == {"a": 1}


# ── 2. Client rehydration ─────────────────────────────────────────────


def test_committed_batch_survives_restart_byte_exact(tmp_path):
    store = _store(tmp_path)
    c1 = BatchSettlementClient(
        _commit_one_acc(), _mock_contract(), PROVIDER_ADDR, state_store=store)
    _run(c1.accumulate(_make_batched()))
    committed = _run(c1.commit_ready_batches())
    assert len(committed) == 1
    orig = committed[0]

    # "Restart": a brand-new client over the same store.
    c2 = BatchSettlementClient(
        _commit_one_acc(), _mock_contract(), PROVIDER_ADDR, state_store=store)
    reh = c2.get_tracked(orig.batch_id)
    assert reh is not None
    # byte/enum/tuple fidelity — the fields finalize + challenge-proofs depend on
    assert reh.batch_id == orig.batch_id
    assert reh.merkle_root == orig.merkle_root
    assert reh.leaf_hashes == orig.leaf_hashes
    assert reh.trigger_reason == orig.trigger_reason
    assert reh.total_value_ftns == orig.total_value_ftns
    assert reh.commit_timestamp == orig.commit_timestamp


def test_finalize_works_on_rehydrated_batch(tmp_path):
    store = _store(tmp_path)
    c1 = BatchSettlementClient(
        _commit_one_acc(), _mock_contract(), PROVIDER_ADDR, state_store=store)
    _run(c1.accumulate(_make_batched()))
    bid = _run(c1.commit_ready_batches())[0].batch_id

    # Restart with a contract that now reports the batch finalizable.
    contract = _mock_contract(finalizable=True)
    c2 = BatchSettlementClient(
        _commit_one_acc(), contract, PROVIDER_ADDR, state_store=store)
    out = _run(c2.finalize_ready_batches())
    assert len(out) == 1 and out[0].batch_id == bid and out[0].tx_submitted
    contract.finalize_batch.assert_awaited_once()  # the rehydrated batch settled


def test_pending_commit_survives_restart_and_reconciles(tmp_path):
    from prsm.economy.web3.provenance_registry import OnChainPendingError
    store = _store(tmp_path)
    # commit broadcasts but its receipt is lost -> quarantine.
    pend = _mock_contract()
    pend.commit_batch.side_effect = OnChainPendingError("unknown", tx_hash="0xdead")
    c1 = BatchSettlementClient(
        _commit_one_acc(), pend, PROVIDER_ADDR, state_store=store)
    _run(c1.accumulate(_make_batched()))
    assert _run(c1.commit_ready_batches()) == []
    assert len(c1.pending_commits()) == 1

    # Restart: the quarantine must survive (else the tx is never reconciled and
    # any landed commit strands). Reconcile adopts it now that the tx is found.
    landed_bid = hashlib.sha256(b"landed").digest()
    contract = _mock_contract()
    contract.get_committed_batch_for_tx.return_value = (landed_bid, 1700001234)
    c2 = BatchSettlementClient(
        _commit_one_acc(), contract, PROVIDER_ADDR, state_store=store)
    assert len(c2.pending_commits()) == 1
    adopted = _run(c2.reconcile_pending_commits())
    assert adopted == 1
    assert c2.get_tracked(landed_bid) is not None
    assert c2.pending_commits() == []


def test_finalized_ids_survive_restart(tmp_path):
    store = _store(tmp_path)
    c1 = BatchSettlementClient(
        _commit_one_acc(), _mock_contract(finalizable=True), PROVIDER_ADDR,
        state_store=store)
    _run(c1.accumulate(_make_batched()))
    bid = _run(c1.commit_ready_batches())[0].batch_id
    _run(c1.finalize_ready_batches())
    assert c1.is_finalized_locally(bid)

    c2 = BatchSettlementClient(
        _commit_one_acc(), _mock_contract(finalizable=True), PROVIDER_ADDR,
        state_store=store)
    assert c2.is_finalized_locally(bid)  # won't re-finalize across restart


def test_per_mutation_persist_keeps_committed_batch_when_loop_later_fails(tmp_path):
    """A failure mid commit-loop must not lose an ALREADY-committed batch:
    persistence is per-mutation, not end-of-method."""
    store = _store(tmp_path)
    acc = ReceiptAccumulator(AccumulatorConfig(count_threshold=1))
    # two distinct ready batches (distinct requesters keep separate keys)
    _run_acc = acc
    br0 = _make_batched(shard_index=0)
    br1 = BatchedReceipt(
        receipt=_make_batched(shard_index=1).receipt,
        requester_address="0x" + "e" * 40,  # distinct key -> second batch
        provider_address=PROVIDER_ADDR,
        value_ftns=ONE_FTNS,
        local_escrow_id="escrow-1",
    )

    contract = AsyncMock()
    calls = {"i": 0}

    async def commit(**kw):
        calls["i"] += 1
        if calls["i"] == 1:
            return (hashlib.sha256(b"first").digest(), 1700000001)
        raise RuntimeError("simulated crash committing the second batch")
    contract.commit_batch.side_effect = commit
    contract.is_finalizable.return_value = False

    c1 = BatchSettlementClient(acc, contract, PROVIDER_ADDR, state_store=store)
    _run(c1.accumulate(br0))
    _run(c1.accumulate(br1))
    _run(c1.commit_ready_batches())  # first persists, second raises (non-tx)

    # The first batch's on-chain commit must be durable despite the later error.
    c2 = BatchSettlementClient(
        ReceiptAccumulator(AccumulatorConfig(count_threshold=1)),
        _mock_contract(), PROVIDER_ADDR, state_store=store)
    assert c2.get_tracked(hashlib.sha256(b"first").digest()) is not None


# ── 3. Opt-out (no store) preserves pre-1039 behavior ─────────────────


def test_no_store_writes_no_file_and_still_commits(tmp_path):
    contract = _mock_contract()
    c = BatchSettlementClient(_commit_one_acc(), contract, PROVIDER_ADDR)
    _run(c.accumulate(_make_batched()))
    assert len(_run(c.commit_ready_batches())) == 1
    # store=None must not touch the filesystem
    assert list(tmp_path.iterdir()) == []


# ── 3b. Schema-corruption is as loud as syntax-corruption (review #1/#4) ─


def test_version_mismatch_raises_corrupt_on_rehydrate(tmp_path):
    import json
    from prsm.settlement.state_store import SettlementStateCorruptError
    p = tmp_path / "settlement_state.json"
    p.write_text(json.dumps(
        {"version": 2, "tracked": {}, "finalized_ids": [], "pending_commits": {}}))
    store = _store(tmp_path)
    with pytest.raises(SettlementStateCorruptError):
        BatchSettlementClient(
            _commit_one_acc(), _mock_contract(), PROVIDER_ADDR, state_store=store)


def test_empty_object_no_version_raises_corrupt(tmp_path):
    """A file truncated to '{}' has no version -> must fail loud, not silent-empty."""
    from prsm.settlement.state_store import SettlementStateCorruptError
    (tmp_path / "settlement_state.json").write_text("{}")
    with pytest.raises(SettlementStateCorruptError):
        BatchSettlementClient(
            _commit_one_acc(), _mock_contract(), PROVIDER_ADDR,
            state_store=_store(tmp_path))


def test_missing_required_key_raises_corrupt(tmp_path):
    import json
    from prsm.settlement.state_store import SettlementStateCorruptError
    # version is right, but a tracked record omits merkle_root
    bad = {
        "version": 1,
        "tracked": {"aa" * 32: {"batch_id": "aa" * 32, "tx_hash": "",
                                "provider_address": PROVIDER_ADDR,
                                "requester_address": REQUESTER_A,
                                # merkle_root MISSING
                                "receipt_count": 1, "total_value_ftns": 1,
                                "commit_timestamp": 1, "leaf_hashes": [],
                                "trigger_reason": "count"}},
        "finalized_ids": [], "pending_commits": {},
    }
    (tmp_path / "settlement_state.json").write_text(json.dumps(bad))
    with pytest.raises(SettlementStateCorruptError):
        BatchSettlementClient(
            _commit_one_acc(), _mock_contract(), PROVIDER_ADDR,
            state_store=_store(tmp_path))


def test_persist_failure_is_loud_and_reraised(tmp_path):
    """A persist IO failure must re-raise (not silently diverge in-memory)."""
    store = _store(tmp_path)
    c = BatchSettlementClient(
        _commit_one_acc(), _mock_contract(), PROVIDER_ADDR, state_store=store)
    _run(c.accumulate(_make_batched()))

    def boom(_state):
        raise OSError("disk full")
    store.save = boom  # type: ignore[assignment]
    with pytest.raises(OSError):
        _run(c.commit_ready_batches())


# ── 4. Wiring: the daemon-built client gets a durable store by default ─


def _build(env):
    from prsm.settlement.client_wiring import (
        build_onchain_settlement_client_or_none,
    )
    return build_onchain_settlement_client_or_none(
        provider_address="0x" + "11" * 20, env=env,
    )


def test_build_attaches_durable_store_by_default(tmp_path):
    client = _build({
        "PRSM_ONCHAIN_SETTLEMENT": "1",
        "PRSM_SETTLEMENT_STATE_FILE": str(tmp_path / "s.json"),
    })
    assert client is not None
    assert client._store is not None
    assert str(client._store.path).endswith("s.json")


def test_state_file_memory_disables_durable_store(tmp_path):
    client = _build({
        "PRSM_ONCHAIN_SETTLEMENT": "1",
        "PRSM_SETTLEMENT_STATE_FILE": ":memory:",
    })
    assert client is not None
    assert client._store is None   # explicit opt-out -> in-memory only


def test_corrupt_state_file_turns_settlement_off(tmp_path):
    """A corrupt money-state file must NOT silently start empty (which would
    strand committed escrow). The build refuses -> settlement OFF."""
    p = tmp_path / "s.json"
    p.write_text("{ corrupt ")
    client = _build({
        "PRSM_ONCHAIN_SETTLEMENT": "1",
        "PRSM_SETTLEMENT_STATE_FILE": str(p),
    })
    assert client is None


def test_version_skew_state_file_turns_settlement_off(tmp_path):
    """Schema corruption (a future version) is as loud as syntax corruption:
    the build refuses -> settlement OFF, not a silent empty start."""
    import json
    p = tmp_path / "s.json"
    p.write_text(json.dumps(
        {"version": 999, "tracked": {}, "finalized_ids": [], "pending_commits": {}}))
    client = _build({
        "PRSM_ONCHAIN_SETTLEMENT": "1",
        "PRSM_SETTLEMENT_STATE_FILE": str(p),
    })
    assert client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
