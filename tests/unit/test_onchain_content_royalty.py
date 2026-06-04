"""Sprint 247 — on-chain content-access royalty dispatcher.

Wires the sprint-246 resolver to the existing
RoyaltyDistributorClient.distribute_royalty() so settlement can
ship the access fee to the on-chain contract instead of (or in
addition to) the local-ledger flow.

Contract:
  dispatch_content_access_royalties(
    shards, content_index, royalty_client,
    serving_node_address, gross_per_shard_wei,
  ) -> List[DispatchResult]

Each shard becomes one distribute_royalty tx. Fail-soft per
shard so a single bad tx doesn't crash the batch. Skips shards
with missing/malformed content_hash. Returns a per-shard
result list with status (sent/skipped_no_hash/failed/...).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.onchain_content_royalty import (
    DispatchResult,
    dispatch_content_access_royalties,
)


def _fake_record(cid, content_hash_hex, provenance_hash_hex=None):
    r = MagicMock()
    r.cid = cid
    r.content_hash = content_hash_hex
    # sp996 — the on-chain dispatch keys on provenance_hash (the ProvenanceRegistry
    # key = keccak256(creator || sha3_256(bytes))), NOT the raw SHA-256
    # content_hash. These tests pass the registered hash as content_hash_hex, so
    # default provenance_hash to it; the dedicated regression test below sets them
    # to DIFFERENT values to prove the dispatch uses provenance_hash.
    r.provenance_hash = (
        provenance_hash_hex if provenance_hash_hex is not None
        else content_hash_hex
    )
    return r


def _fake_index(records_by_cid):
    idx = MagicMock()
    idx.lookup = lambda cid: records_by_cid.get(cid)
    return idx


def _fake_client(returns):
    """returns = list of (tx_hash, status) tuples (one per call)
    OR an Exception instance (raised on every call)."""
    client = MagicMock()
    if isinstance(returns, Exception):
        client.distribute_royalty = MagicMock(side_effect=returns)
    else:
        client.distribute_royalty = MagicMock(side_effect=returns)
    return client


def test_dispatches_one_tx_per_shard():
    records = {
        "c1": _fake_record("c1", "ab" * 32),
        "c2": _fake_record("c2", "cd" * 32),
    }
    client = _fake_client([
        ("0xtx1", "CONFIRMED"),
        ("0xtx2", "CONFIRMED"),
    ])
    results = dispatch_content_access_royalties(
        shards=["c1", "c2"],
        content_index=_fake_index(records),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_per_shard_wei=1_000_000_000_000_000_000,  # 1 FTNS
    )
    assert len(results) == 2
    assert results[0].cid == "c1"
    assert results[0].tx_hash == "0xtx1"
    assert results[0].status == "sent"
    assert results[1].cid == "c2"
    assert results[1].tx_hash == "0xtx2"
    # Verify client invocation
    assert client.distribute_royalty.call_count == 2
    args, _ = client.distribute_royalty.call_args_list[0]
    # Args: content_hash bytes, serving_node, gross
    assert args[0] == bytes.fromhex("ab" * 32)
    assert args[1] == "0x" + "1" * 40
    assert args[2] == 1_000_000_000_000_000_000


def test_skips_shard_with_missing_record():
    records = {"c1": _fake_record("c1", "ab" * 32)}
    client = _fake_client([("0xtx1", "CONFIRMED")])
    results = dispatch_content_access_royalties(
        shards=["c1", "missing"],
        content_index=_fake_index(records),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_per_shard_wei=1,
    )
    assert len(results) == 2
    assert results[0].status == "sent"
    assert results[1].status == "skipped_no_record"
    assert client.distribute_royalty.call_count == 1


def test_skips_shard_with_bad_hash_length():
    records = {
        "short": _fake_record("short", "abcd"),  # not 64 chars
        "good": _fake_record("good", "ab" * 32),
    }
    client = _fake_client([("0xtx1", "CONFIRMED")])
    results = dispatch_content_access_royalties(
        shards=["short", "good"],
        content_index=_fake_index(records),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_per_shard_wei=1,
    )
    assert results[0].status == "skipped_bad_hash"
    assert results[1].status == "sent"
    assert client.distribute_royalty.call_count == 1


def test_skips_shard_with_non_hex_hash():
    records = {
        "bad": _fake_record("bad", "z" * 64),
    }
    client = _fake_client([])
    results = dispatch_content_access_royalties(
        shards=["bad"],
        content_index=_fake_index(records),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_per_shard_wei=1,
    )
    assert results[0].status == "skipped_bad_hash"


def test_failed_tx_doesnt_break_batch():
    """If one tx raises, the rest still go through."""
    records = {
        "c1": _fake_record("c1", "ab" * 32),
        "c2": _fake_record("c2", "cd" * 32),
        "c3": _fake_record("c3", "ef" * 32),
    }
    client = _fake_client([
        ("0xtx1", "CONFIRMED"),
        RuntimeError("rpc down"),
        ("0xtx3", "CONFIRMED"),
    ])
    results = dispatch_content_access_royalties(
        shards=["c1", "c2", "c3"],
        content_index=_fake_index(records),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_per_shard_wei=1,
    )
    # sp932 — sp918 status vocabulary: an UNTYPED exception (here a bare
    # RuntimeError, not a BroadcastFailedError) yields "error", NOT "failed".
    # "error" is the fail-safe — the chain state is unknown so the idempotency
    # claim is kept (excluded from keys_to_release) to avoid a double-pay; only
    # a typed BroadcastFailedError ("nothing reached the chain") yields "failed"
    # and is safe to release. The batch still continues past the failure.
    assert [r.status for r in results] == [
        "sent", "error", "sent",
    ]
    assert "rpc down" in results[1].error


def test_zero_gross_rejected():
    with pytest.raises(ValueError):
        dispatch_content_access_royalties(
            shards=["c1"],
            content_index=_fake_index({}),
            royalty_client=_fake_client([]),
            serving_node_address="0x" + "1" * 40,
            gross_per_shard_wei=0,
        )


def test_negative_gross_rejected():
    with pytest.raises(ValueError):
        dispatch_content_access_royalties(
            shards=["c1"],
            content_index=_fake_index({}),
            royalty_client=_fake_client([]),
            serving_node_address="0x" + "1" * 40,
            gross_per_shard_wei=-1,
        )


def test_no_shards_returns_empty():
    results = dispatch_content_access_royalties(
        shards=[],
        content_index=_fake_index({}),
        royalty_client=_fake_client([]),
        serving_node_address="0x" + "1" * 40,
        gross_per_shard_wei=1,
    )
    assert results == []


def test_dispatch_uses_provenance_hash_not_content_hash():
    """sp996 regression — the on-chain ProvenanceRegistry/RoyaltyDistributor key
    on the provenance_hash (keccak256(creator || sha3_256(bytes))), NOT the raw
    SHA-256 content_hash. Dispatching with content_hash made getCreatorAndRate
    return address(0) → distributeRoyalty reverted 'Not registered' every time, so
    the creator's on-chain royalty leg never paid. Assert the dispatcher passes the
    PROVENANCE hash bytes (and explicitly NOT the SHA-256 content_hash)."""
    sha256_hex = "ab" * 32   # the record's content_hash (raw SHA-256)
    prov_hex = "cd" * 32     # the registered provenance_hash (keccak) — distinct
    rec = _fake_record("c1", sha256_hex, provenance_hash_hex=prov_hex)
    client = _fake_client([("0xtx", "CONFIRMED")])
    results = dispatch_content_access_royalties(
        shards=["c1"],
        content_index=_fake_index({"c1": rec}),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_per_shard_wei=10**18,
    )
    assert results[0].status == "sent"
    args, _ = client.distribute_royalty.call_args_list[0]
    assert args[0] == bytes.fromhex(prov_hex)        # the registered key
    assert args[0] != bytes.fromhex(sha256_hex)      # NOT the SHA-256


def test_skips_record_with_no_provenance_hash():
    """A record never registered on-chain (no provenance_hash) is skipped as
    'skipped_unregistered' — it can't be paid the on-chain leg (would revert
    'Not registered'). NOT dispatched (avoids a guaranteed revert per shard)."""
    rec = MagicMock()
    rec.cid = "c1"
    rec.content_hash = "ab" * 32  # has a SHA-256 but was never registered on-chain
    rec.provenance_hash = None
    client = _fake_client([("0xtx", "CONFIRMED")])
    results = dispatch_content_access_royalties(
        shards=["c1"],
        content_index=_fake_index({"c1": rec}),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_per_shard_wei=10**18,
    )
    assert results[0].status == "skipped_unregistered"
    client.distribute_royalty.assert_not_called()
