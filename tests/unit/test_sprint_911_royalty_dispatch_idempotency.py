"""Sprint 911 — on-chain royalty-dispatch idempotency + pending status.

The money-path review confirmed two HIGH issues in
dispatch_content_access_royalties:

1. NO idempotency key — a retry of the same settlement re-dispatches every
   shard and DOUBLE-PAYS on chain. Fix: a deterministic
   royalty_dispatch_key(settlement_key, cid) atomically claimed via a
   caller-supplied claim_fn (the sp898 record_nonce primitive) BEFORE the
   tx; an already-claimed shard short-circuits as
   skipped_already_dispatched.

2. OnChainPendingError flattened to status='failed' — a broadcast-but-
   unconfirmed tx looked like a plain failure, inviting the operator to
   re-dispatch a tx that actually settled. Fix: distinct 'pending' status
   carrying tx_hash; BroadcastFailedError → 'failed' (safe retry);
   OnChainRevertedError → 'reverted' (safe fall-back).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.onchain_content_royalty import (
    dispatch_content_access_royalties,
    royalty_dispatch_key,
)
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
)


def _record(cid, h="ab" * 32):
    r = MagicMock()
    r.content_hash = h
    # sp996 — on-chain dispatch keys on provenance_hash (the registry key), not
    # content_hash. Mirror the registered hash (else MagicMock auto-attr →
    # skipped_unregistered).
    r.provenance_hash = h
    r.royalty_rate = 0.05
    return r


def _index(recs):
    idx = MagicMock()
    idx.lookup = lambda cid: recs.get(cid)
    return idx


def _client(returns=None, side_effect=None):
    c = MagicMock()
    if side_effect is not None:
        c.distribute_royalty = MagicMock(side_effect=side_effect)
    else:
        c.distribute_royalty = MagicMock(return_value=returns or ("0xtx", "confirmed"))
    return c


# ── Idempotency ──────────────────────────────────────────


def test_retry_does_not_redispatch_already_claimed_shards():
    recs = {"c1": _record("c1"), "c2": _record("c2")}
    client = _client(("0xtx", "confirmed"))
    # Durable claim store: first claim of a key wins, repeats lose.
    claimed = set()
    def claim(key):
        if key in claimed:
            return False
        claimed.add(key)
        return True

    common = dict(
        shards=["c1", "c2"], content_index=_index(recs),
        royalty_client=client, serving_node_address="0xop",
        gross_per_shard_wei=1000, settlement_key="settle-A", claim_fn=claim,
    )
    first = dispatch_content_access_royalties(**common)
    second = dispatch_content_access_royalties(**common)  # retry

    assert [r.status for r in first] == ["sent", "sent"]
    assert [r.status for r in second] == [
        "skipped_already_dispatched", "skipped_already_dispatched",
    ]
    # The on-chain tx fired exactly twice total (once per shard), NOT four.
    assert client.distribute_royalty.call_count == 2


def test_claim_error_fails_closed_no_dispatch():
    recs = {"c1": _record("c1")}
    client = _client(("0xtx", "confirmed"))
    def boom(key):
        raise RuntimeError("claim store down")
    out = dispatch_content_access_royalties(
        shards=["c1"], content_index=_index(recs), royalty_client=client,
        serving_node_address="0xop", gross_per_shard_wei=1000,
        settlement_key="s", claim_fn=boom,
    )
    assert out[0].status == "failed"
    assert client.distribute_royalty.call_count == 0  # fail-CLOSED: no pay


def test_no_claim_fn_keeps_legacy_behavior():
    recs = {"c1": _record("c1")}
    client = _client(("0xtx", "confirmed"))
    out = dispatch_content_access_royalties(
        shards=["c1"], content_index=_index(recs), royalty_client=client,
        serving_node_address="0xop", gross_per_shard_wei=1000,
    )
    assert out[0].status == "sent"


def test_dispatch_key_is_deterministic_and_per_shard():
    assert royalty_dispatch_key("s", "c1") == royalty_dispatch_key("s", "c1")
    assert royalty_dispatch_key("s", "c1") != royalty_dispatch_key("s", "c2")
    assert royalty_dispatch_key("s1", "c1") != royalty_dispatch_key("s2", "c1")


# ── Distinct on-chain error statuses ─────────────────────


def test_pending_error_yields_pending_status_with_tx_hash():
    recs = {"c1": _record("c1")}
    client = _client(side_effect=OnChainPendingError("unconfirmed", "0xpend"))
    out = dispatch_content_access_royalties(
        shards=["c1"], content_index=_index(recs), royalty_client=client,
        serving_node_address="0xop", gross_per_shard_wei=1000,
    )
    assert out[0].status == "pending"   # NOT "failed"
    assert out[0].tx_hash == "0xpend"   # so operator reconciles, not re-sends


def test_reverted_error_yields_reverted_status():
    recs = {"c1": _record("c1")}
    client = _client(side_effect=OnChainRevertedError("reverted"))
    out = dispatch_content_access_royalties(
        shards=["c1"], content_index=_index(recs), royalty_client=client,
        serving_node_address="0xop", gross_per_shard_wei=1000,
    )
    assert out[0].status == "reverted"


def test_broadcast_failed_yields_failed_status():
    recs = {"c1": _record("c1")}
    client = _client(side_effect=BroadcastFailedError("never broadcast"))
    out = dispatch_content_access_royalties(
        shards=["c1"], content_index=_index(recs), royalty_client=client,
        serving_node_address="0xop", gross_per_shard_wei=1000,
    )
    assert out[0].status == "failed"
