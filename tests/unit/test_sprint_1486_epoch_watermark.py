"""Sprint 1486 — durable epoch watermark: never re-pay a batch, prefer late over twice.

sp1481's exactly-once attribution held only if the caller remembered which batches
prior epochs paid. Nothing persisted that, so the guarantee was theoretical — the
first real epoch job would have kept it in memory and double-paid every provider
after a restart.

The asymmetry these tests encode: losing the watermark pays providers TWICE from a
shared pot (unrecoverable, silently underpaying everyone else), while skipping a
batch merely pays someone LATE (the next epoch picks it up). Every ambiguous case
must fail toward "late".
"""
from __future__ import annotations

import json

import pytest

from prsm.settlement.emission_epoch import FinalizedBatch, build_emission_epoch
from prsm.settlement.epoch_watermark import (
    EpochWatermarkStore,
    WatermarkIntegrityError,
)

A = "0x" + "a1" * 20
B = "0x" + "b2" * 20


def _store(tmp_path):
    return EpochWatermarkStore(tmp_path / "wm.json").load()


def test_cold_start_is_empty_and_epoch_ids_begin_at_one(tmp_path):
    s = _store(tmp_path)
    assert s.consumed == set()
    assert s.last_epoch_id is None
    assert s.next_epoch_id() == 1


def test_commit_persists_across_restart(tmp_path):
    """★ The whole point: a restart must NOT forget what was already paid."""
    s = _store(tmp_path)
    s.commit_epoch(1, ["0xaa", "0xbb"])
    reopened = EpochWatermarkStore(tmp_path / "wm.json").load()
    assert reopened.consumed == {"0xaa", "0xbb"}
    assert reopened.last_epoch_id == 1
    assert reopened.next_epoch_id() == 2


def test_batch_ids_are_case_insensitive(tmp_path):
    s = _store(tmp_path)
    s.commit_epoch(1, ["0xAABB"])
    assert s.has_consumed("0xaabb") and s.has_consumed("0xAABB")


def test_corrupt_watermark_refuses_rather_than_starting_empty(tmp_path):
    """★ An empty watermark is indistinguishable from 'nothing was ever paid', so
    treating a corrupt file as empty would re-pay EVERY historical batch. Fail loud."""
    p = tmp_path / "wm.json"
    p.write_text("{ this is not json")
    with pytest.raises(WatermarkIntegrityError, match="unreadable"):
        EpochWatermarkStore(p).load()


def test_malformed_shape_refuses(tmp_path):
    p = tmp_path / "wm.json"
    p.write_text(json.dumps({"something_else": 1}))
    with pytest.raises(WatermarkIntegrityError, match="malformed"):
        EpochWatermarkStore(p).load()


def test_use_before_load_refuses(tmp_path):
    """Using an unloaded store would silently present an empty consumed set."""
    s = EpochWatermarkStore(tmp_path / "wm.json")
    with pytest.raises(WatermarkIntegrityError, match="before load"):
        _ = s.consumed


def test_missing_file_is_a_legitimate_cold_start(tmp_path):
    """Distinct from corrupt: never-existed is a real first run and must be allowed."""
    s = EpochWatermarkStore(tmp_path / "nope.json").load()
    assert s.consumed == set()


def test_commit_is_idempotent_for_the_same_epoch(tmp_path):
    """A retry after an ambiguous publish must not rewind or duplicate."""
    s = _store(tmp_path)
    s.commit_epoch(1, ["0xaa"])
    added = s.commit_epoch(1, ["0xaa"])
    assert added == 0
    assert s.last_epoch_id == 1 and s.consumed == {"0xaa"}


def test_older_epoch_cannot_rewind_the_watermark(tmp_path):
    s = _store(tmp_path)
    s.commit_epoch(5, ["0xaa"])
    s.commit_epoch(2, ["0xbb"])          # late/replayed older epoch
    assert s.last_epoch_id == 5          # not rewound
    assert s.consumed == {"0xaa", "0xbb"}  # but its batches are still absorbed


def test_atomic_write_leaves_prior_state_on_failure(tmp_path, monkeypatch):
    """★ A crash mid-write must not truncate the set — a partial file's missing
    tail would be re-paid."""
    s = _store(tmp_path)
    s.commit_epoch(1, ["0xaa"])
    import os as _os
    monkeypatch.setattr(_os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        s.commit_epoch(2, ["0xbb"])
    reopened = EpochWatermarkStore(tmp_path / "wm.json").load()
    assert reopened.consumed == {"0xaa"}      # previous complete state intact
    assert reopened.last_epoch_id == 1
    # ...and no temp litter left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["wm.json"]


# ─────────────── the crash window: publish-then-persist ───────────────

def test_reconcile_adopts_an_epoch_the_chain_already_published(tmp_path):
    """★ Crashed between broadcast and persist. The CHAIN is authoritative — adopt
    its epoch so the next one does not re-attribute those batches."""
    s = _store(tmp_path)
    recovered = s.reconcile_from_chain(1, ["0xaa", "0xbb"], published_on_chain=True)
    assert recovered is True
    assert s.consumed == {"0xaa", "0xbb"} and s.last_epoch_id == 1


def test_reconcile_is_a_noop_when_the_chain_has_not_published(tmp_path):
    """Must NOT mark batches consumed for an epoch that never landed — that is the
    silent non-payment direction."""
    s = _store(tmp_path)
    assert s.reconcile_from_chain(1, ["0xaa"], published_on_chain=False) is False
    assert s.consumed == set()


def test_reconcile_does_not_redo_an_already_committed_epoch(tmp_path):
    s = _store(tmp_path)
    s.commit_epoch(1, ["0xaa"])
    assert s.reconcile_from_chain(1, ["0xaa"], published_on_chain=True) is False


# ─────────────── integration with the epoch builder ───────────────

def test_two_sequential_epochs_never_double_pay_across_a_restart(tmp_path):
    """★ END TO END: build epoch 1, commit, RESTART the store from disk, then feed
    the SAME batches to epoch 2 — nothing must be payable again."""
    batches = [
        FinalizedBatch("0xaa", A, 100, 1000),
        FinalizedBatch("0xbb", B, 100, 1001),
    ]
    s = _store(tmp_path)
    plan1 = build_emission_epoch(
        epoch_id=s.next_epoch_id(), batches=batches, pot_wei=10**18,
        consumed_batch_ids=s.consumed)
    s.commit_epoch(plan1.epoch_id, plan1.consumed_batch_ids)

    reopened = EpochWatermarkStore(tmp_path / "wm.json").load()   # simulate restart
    with pytest.raises(ValueError, match="no eligible finalized batches"):
        build_emission_epoch(
            epoch_id=reopened.next_epoch_id(), batches=batches, pot_wei=10**18,
            consumed_batch_ids=reopened.consumed)


def test_new_batches_after_a_restart_are_still_payable(tmp_path):
    """The watermark must block only what was already paid — not new work."""
    s = _store(tmp_path)
    plan1 = build_emission_epoch(
        epoch_id=s.next_epoch_id(),
        batches=[FinalizedBatch("0xaa", A, 100, 1000)], pot_wei=10**18,
        consumed_batch_ids=s.consumed)
    s.commit_epoch(plan1.epoch_id, plan1.consumed_batch_ids)

    reopened = EpochWatermarkStore(tmp_path / "wm.json").load()
    plan2 = build_emission_epoch(
        epoch_id=reopened.next_epoch_id(),
        batches=[FinalizedBatch("0xaa", A, 100, 1000),
                 FinalizedBatch("0xcc", B, 50, 1002)],
        pot_wei=10**18, consumed_batch_ids=reopened.consumed)
    assert plan2.epoch_id == 2
    assert plan2.consumed_batch_ids == ["0xcc"]        # only the NEW batch
    assert [e.account for e in plan2.entries] == [__import__("eth_utils").to_checksum_address(B)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
