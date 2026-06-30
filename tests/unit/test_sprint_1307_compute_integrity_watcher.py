"""Sprint 1307 — activate the ChallengeWatcher for compute-integrity self-audit.

The ChallengeWatcher (sp1129) was built but never scheduled — nothing ran the §7
verifier against committed batches. sp1307 wires it: ``build_compute_integrity_watcher``
correlates the node's two opt-in audit stores (PublishedBatchStore committed batches ×
the §7 InferenceReceiptStore keyed by leaf hash) into the watcher's pluggable source, and
the node launches a read-only/never-broadcast scan loop alongside the settlement audit.

These tests pin the BUILDER's correlation + None-handling (the loop is thin node glue).
The merkle leaf functions are monkeypatched so the source can be exercised without real
BatchedReceipts/crypto; the verifier/assembler/dry-run are the watcher's own (tested
elsewhere).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import prsm.settlement.merkle as merkle
from prsm.settlement.settlement_audit_wiring import build_compute_integrity_watcher


def _rec(tag):
    return SimpleNamespace(
        inference_receipt=SimpleNamespace(tag=tag),
        settler_public_key_b64=f"PK-{tag}",
        stage_public_keys={f"node-{tag}": "k"},
    )


def _leaf_for(br):
    """Deterministic 32-byte leaf from a sentinel receipt (bytes)."""
    return (bytes(br) + b"\x00" * 32)[:32]


@pytest.fixture
def _patch_merkle(monkeypatch):
    # identity batched_receipt_to_leaf + a deterministic hash_leaf so the source can
    # correlate sentinel receipts without real BatchedReceipts.
    monkeypatch.setattr(merkle, "batched_receipt_to_leaf", lambda br: br)
    monkeypatch.setattr(merkle, "hash_leaf", _leaf_for)


class _PBS:
    def __init__(self, batches):
        self._b = batches

    def all_batches(self):
        return self._b


class _IRS:
    """Returns a §7 record only for the leaves we say are retained."""
    def __init__(self, retained):  # retained: {leaf_bytes: rec}
        self._r = retained

    def get(self, leaf):
        return self._r.get(bytes(leaf))


# ── None-handling ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pbs,irs,dry", [
    (None, object(), object()),
    (object(), None, object()),
    (object(), object(), None),
])
def test_returns_none_when_any_dep_absent(pbs, irs, dry):
    assert build_compute_integrity_watcher(
        published_batch_store=pbs, inference_receipt_store=irs,
        dry_run_client=dry) is None


def test_returns_watcher_when_all_present(_patch_merkle):
    w = build_compute_integrity_watcher(
        published_batch_store=_PBS([]), inference_receipt_store=_IRS({}),
        dry_run_client=object())
    from prsm.settlement.challenge_watcher import ChallengeWatcher
    assert isinstance(w, ChallengeWatcher)


# ── source correlation ───────────────────────────────────────────────────────

async def _units(watcher):
    return [u async for u in watcher._source()]


@pytest.mark.asyncio
async def test_source_yields_only_retained_leaves(_patch_merkle):
    batch = SimpleNamespace(batch_id=b"BID", receipts=[b"r0", b"r1", b"r2"])
    # only r0 + r2 have a retained §7 receipt
    irs = _IRS({_leaf_for(b"r0"): _rec("0"), _leaf_for(b"r2"): _rec("2")})
    w = build_compute_integrity_watcher(
        published_batch_store=_PBS([batch]), inference_receipt_store=irs,
        dry_run_client=object())
    units = await _units(w)
    assert len(units) == 2
    by_idx = {u.target_index: u for u in units}
    assert set(by_idx) == {0, 2}
    u0 = by_idx[0]
    assert u0.batch_id == b"BID"
    assert u0.inference_receipt.tag == "0"
    assert u0.settler_public_key_b64 == "PK-0"
    assert u0.stage_public_keys == {"node-0": "k"}
    # the assembler needs the FULL ordered receipt set
    assert u0.batch_receipts == [b"r0", b"r1", b"r2"]


@pytest.mark.asyncio
async def test_source_skips_leaf_when_no_receipt_retained(_patch_merkle):
    batch = SimpleNamespace(batch_id=b"BID", receipts=[b"r0", b"r1"])
    w = build_compute_integrity_watcher(
        published_batch_store=_PBS([batch]), inference_receipt_store=_IRS({}),
        dry_run_client=object())
    assert await _units(w) == []


@pytest.mark.asyncio
async def test_source_skips_batch_without_batch_id(_patch_merkle):
    batch = SimpleNamespace(batch_id=None, receipts=[b"r0"])
    irs = _IRS({_leaf_for(b"r0"): _rec("0")})
    w = build_compute_integrity_watcher(
        published_batch_store=_PBS([batch]), inference_receipt_store=irs,
        dry_run_client=object())
    assert await _units(w) == []


@pytest.mark.asyncio
async def test_source_fail_soft_when_leaf_compute_raises(monkeypatch):
    monkeypatch.setattr(merkle, "batched_receipt_to_leaf", lambda br: br)

    def _boom(_leaf):
        raise ValueError("bad receipt")
    monkeypatch.setattr(merkle, "hash_leaf", _boom)
    batch = SimpleNamespace(batch_id=b"BID", receipts=[b"r0"])
    w = build_compute_integrity_watcher(
        published_batch_store=_PBS([batch]),
        inference_receipt_store=_IRS({_leaf_for(b"r0"): _rec("0")}),
        dry_run_client=object())
    assert await _units(w) == []   # the raising leaf is skipped, not propagated


@pytest.mark.asyncio
async def test_source_spans_multiple_batches(_patch_merkle):
    b1 = SimpleNamespace(batch_id=b"B1", receipts=[b"a"])
    b2 = SimpleNamespace(batch_id=b"B2", receipts=[b"b"])
    irs = _IRS({_leaf_for(b"a"): _rec("a"), _leaf_for(b"b"): _rec("b")})
    w = build_compute_integrity_watcher(
        published_batch_store=_PBS([b1, b2]), inference_receipt_store=irs,
        dry_run_client=object())
    units = await _units(w)
    assert {u.batch_id for u in units} == {b"B1", b"B2"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
