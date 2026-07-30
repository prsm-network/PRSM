"""Sprint 1487 — the emission epoch job. Refusals matter more than the happy path.

Publishing is irreversible and an epoch id can never be rewritten, so most of the
value in this job is in what it declines to do. Every abort below leaves batches
UNCONSUMED, which means the next successful epoch pays them — lateness, not loss.
"""
from __future__ import annotations

import json

import pytest

from prsm.settlement.emission_epoch import FinalizedBatch
from prsm.settlement.epoch_runner import (
    EpochRunAborted,
    WatermarkLostError,
    run_epoch,
)
from prsm.settlement.epoch_watermark import EpochWatermarkStore

A = "0x" + "a1" * 20
B = "0x" + "b2" * 20
POT = 10**18


class FakeChain:
    """Stands in for OperatorRewardPool. Records what would be published."""

    def __init__(self, existing=(), unreserved=POT, fail_publish=False):
        self.existing = set(existing)
        self.unreserved = unreserved
        self.fail_publish = fail_publish
        self.published = []

    def epoch_exists(self, epoch_id):
        return epoch_id in self.existing

    def unreserved_balance_wei(self):
        return self.unreserved

    def publish_epoch(self, epoch_id, merkle_root, total_amount_wei):
        if self.fail_publish:
            raise RuntimeError("broadcast failed")
        self.published.append((epoch_id, merkle_root, total_amount_wei))
        self.existing.add(epoch_id)
        return f"0xtx{epoch_id}"


def _wm(tmp_path):
    return EpochWatermarkStore(tmp_path / "wm.json").load()


def _batches():
    return [
        FinalizedBatch("0xaa", A, 600, 1000),
        FinalizedBatch("0xbb", B, 400, 1001),
    ]


# ─────────────────────────── the happy path ───────────────────────────

def test_dry_run_publishes_nothing_and_leaves_the_watermark_untouched(tmp_path):
    """★ Default is dry run: an irreversible step must be opted into."""
    chain = FakeChain()
    wm = _wm(tmp_path)
    r = run_epoch(chain=chain, watermark=wm, batches=_batches(),
                  manifest_dir=tmp_path)
    assert r.dry_run and not r.published
    assert chain.published == []
    assert wm.last_epoch_id is None          # nothing consumed
    assert r.recipients == 2
    assert r.merkle_root and r.merkle_root.startswith("0x")


def test_publish_advances_watermark_and_writes_a_manifest(tmp_path):
    chain = FakeChain()
    wm = _wm(tmp_path)
    r = run_epoch(chain=chain, watermark=wm, batches=_batches(),
                  manifest_dir=tmp_path, dry_run=False)
    assert r.published and r.tx_hash == "0xtx1"
    assert len(chain.published) == 1
    assert chain.published[0][2] == POT               # spends the pot exactly
    assert wm.last_epoch_id == 1
    assert wm.consumed == {"0xaa", "0xbb"}

    man = json.loads((tmp_path / "epoch-1.json").read_text())
    assert man["merkle_root"] == r.merkle_root
    assert sum(int(e["amount_wei"]) for e in man["entries"]) == POT
    assert all(e["proof"] is not None for e in man["entries"])
    assert man["consumed_batch_ids"] == ["0xaa", "0xbb"]


def test_second_run_after_a_restart_finds_nothing_to_pay(tmp_path):
    """★ End to end: same batches, fresh process — must not pay again."""
    chain = FakeChain()
    run_epoch(chain=chain, watermark=_wm(tmp_path), batches=_batches(),
              manifest_dir=tmp_path, dry_run=False)
    reopened = EpochWatermarkStore(tmp_path / "wm.json").load()
    with pytest.raises(ValueError, match="no eligible finalized batches"):
        run_epoch(chain=chain, watermark=reopened, batches=_batches(),
                  manifest_dir=tmp_path, dry_run=False)
    assert len(chain.published) == 1


# ─────────────────── the refusal that prevents double-pay ───────────────────

def test_LOST_watermark_is_refused_not_treated_as_a_cold_start(tmp_path):
    """★★ THE test. Watermark says 'never published'; the chain says epoch 1 exists.
    Running would re-attribute all history into a new epoch and double-pay."""
    chain = FakeChain(existing=[1])
    with pytest.raises(WatermarkLostError, match="lost, not cold"):
        run_epoch(chain=chain, watermark=_wm(tmp_path), batches=_batches(),
                  manifest_dir=tmp_path, dry_run=False)
    assert chain.published == []


def test_lost_watermark_is_refused_even_in_dry_run(tmp_path):
    """The plan itself is the dangerous artifact — refuse before building it."""
    chain = FakeChain(existing=[1])
    with pytest.raises(WatermarkLostError):
        run_epoch(chain=chain, watermark=_wm(tmp_path), batches=_batches())


def test_genuine_cold_start_still_runs(tmp_path):
    """The refusal must not fire when the chain agrees nothing was published."""
    chain = FakeChain()
    r = run_epoch(chain=chain, watermark=_wm(tmp_path), batches=_batches(),
                  manifest_dir=tmp_path, dry_run=False)
    assert r.published and r.epoch_id == 1


# ───────────────── crash between publish and persist ─────────────────

def test_recovers_an_epoch_published_but_not_persisted(tmp_path):
    """★ Crash after broadcast. The chain has epoch 2; our watermark stops at 1.
    Adopt epoch 2 from its manifest, then plan epoch 3."""
    chain = FakeChain()
    wm = _wm(tmp_path)
    run_epoch(chain=chain, watermark=wm, batches=_batches(),
              manifest_dir=tmp_path, dry_run=False)          # epoch 1 committed

    # Simulate: epoch 2 published + manifest written, watermark never advanced.
    (tmp_path / "epoch-2.json").write_text(json.dumps(
        {"epoch_id": 2, "consumed_batch_ids": ["0xcc"]}))
    chain.existing.add(2)

    r = run_epoch(chain=chain, watermark=wm,
                  batches=[FinalizedBatch("0xcc", A, 100, 1002),
                           FinalizedBatch("0xdd", B, 100, 1003)],
                  manifest_dir=tmp_path, dry_run=False)
    assert r.recovered_from_chain
    assert r.epoch_id == 3
    assert wm.has_consumed("0xcc")                # adopted, not re-paid
    assert r.consumed_batches == 1                # only 0xdd is new


def test_refuses_when_the_published_epoch_has_no_manifest_to_recover_from(tmp_path):
    """Without the consumed list we cannot know what that epoch paid. Guessing
    would double-pay, so abort and say what to recover."""
    chain = FakeChain()
    wm = _wm(tmp_path)
    run_epoch(chain=chain, watermark=wm, batches=_batches(),
              manifest_dir=tmp_path, dry_run=False)
    chain.existing.add(2)                          # published, no manifest
    with pytest.raises(EpochRunAborted, match="no manifest"):
        run_epoch(chain=chain, watermark=wm,
                  batches=[FinalizedBatch("0xcc", A, 100, 1002)],
                  manifest_dir=tmp_path, dry_run=False)


def test_a_failed_broadcast_leaves_the_watermark_unadvanced(tmp_path):
    """★ Publish threw. Those batches must stay unconsumed so a retry pays them —
    persisting first would have silently voided real work."""
    chain = FakeChain(fail_publish=True)
    wm = _wm(tmp_path)
    with pytest.raises(RuntimeError, match="broadcast failed"):
        run_epoch(chain=chain, watermark=wm, batches=_batches(),
                  manifest_dir=tmp_path, dry_run=False)
    assert wm.last_epoch_id is None
    assert EpochWatermarkStore(tmp_path / "wm.json").load().consumed == set()

    chain.fail_publish = False                     # retry succeeds
    r = run_epoch(chain=chain, watermark=wm, batches=_batches(),
                  manifest_dir=tmp_path, dry_run=False)
    assert r.published and r.consumed_batches == 2


# ─────────────────────────── pot safety ───────────────────────────

def test_refuses_a_pot_larger_than_the_unreserved_balance(tmp_path):
    """Reserved funds back earlier epochs' unclaimed leaves — spending them would
    strand those claims."""
    chain = FakeChain(unreserved=500)
    with pytest.raises(EpochRunAborted, match="exceeds the pool's UNRESERVED"):
        run_epoch(chain=chain, watermark=_wm(tmp_path), batches=_batches(),
                  pot_wei=1000, manifest_dir=tmp_path, dry_run=False)
    assert chain.published == []


def test_refuses_when_there_is_nothing_unreserved_to_distribute(tmp_path):
    chain = FakeChain(unreserved=0)
    with pytest.raises(EpochRunAborted, match="no unreserved balance"):
        run_epoch(chain=chain, watermark=_wm(tmp_path), batches=_batches(),
                  manifest_dir=tmp_path, dry_run=False)
    assert chain.published == []


def test_defaults_the_pot_to_the_unreserved_balance(tmp_path):
    chain = FakeChain(unreserved=777)
    r = run_epoch(chain=chain, watermark=_wm(tmp_path), batches=_batches(),
                  manifest_dir=tmp_path, dry_run=False)
    assert r.total_amount_wei == 777
    assert chain.published[0][2] == 777


def test_allocation_spends_the_pot_exactly_across_recipients(tmp_path):
    """No wei may be created or lost — the contract reserves exactly this total."""
    chain = FakeChain(unreserved=1_000_000_007)      # prime-ish, forces remainders
    r = run_epoch(chain=chain, watermark=_wm(tmp_path), batches=_batches(),
                  manifest_dir=tmp_path, dry_run=False)
    man = json.loads((tmp_path / "epoch-1.json").read_text())
    assert sum(int(e["amount_wei"]) for e in man["entries"]) == 1_000_000_007
    assert r.total_amount_wei == 1_000_000_007


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
