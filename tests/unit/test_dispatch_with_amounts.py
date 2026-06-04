"""Sprint 257 — dispatcher accepts gross_amounts_wei dict.

Extends the sprint-248 dispatcher with a per-shard amount dict
input (output of sprint-256 allocate_royalty_amounts). Adds
"skipped_zero_amount" status for shards allocated zero from a
rate-weighted policy that didn't redistribute.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.onchain_content_royalty import (
    dispatch_content_access_royalties,
)


def _rec(cid, content_hash="ab" * 32):
    r = MagicMock()
    r.cid = cid
    r.content_hash = content_hash
    # sp996 — on-chain dispatch keys on provenance_hash (the registry key), not
    # content_hash. Mirror the registered hash onto provenance_hash (else the
    # MagicMock auto-attr routes the record to skipped_unregistered).
    r.provenance_hash = content_hash
    return r


def _index(records):
    idx = MagicMock()
    idx.lookup = lambda cid: records.get(cid)
    return idx


def _client_seq(returns):
    client = MagicMock()
    client.distribute_royalty = MagicMock(side_effect=returns)
    return client


def test_mutex_kwarg_validation():
    """Caller must supply exactly one of the amount kwargs."""
    with pytest.raises(ValueError):
        dispatch_content_access_royalties(
            shards=["a"],
            content_index=_index({}),
            royalty_client=_client_seq([]),
            serving_node_address="0x" + "1" * 40,
            # neither
        )
    with pytest.raises(ValueError):
        dispatch_content_access_royalties(
            shards=["a"],
            content_index=_index({}),
            royalty_client=_client_seq([]),
            serving_node_address="0x" + "1" * 40,
            gross_per_shard_wei=10,
            gross_amounts_wei={"a": 10},
        )


def test_negative_amount_in_dict_rejected():
    with pytest.raises(ValueError):
        dispatch_content_access_royalties(
            shards=["a"],
            content_index=_index({}),
            royalty_client=_client_seq([]),
            serving_node_address="0x" + "1" * 40,
            gross_amounts_wei={"a": -1},
        )


def test_per_shard_amounts_routed_individually():
    """Each shard's tx gets its own wei from the dict."""
    records = {
        "a": _rec("a", "aa" * 32),
        "b": _rec("b", "bb" * 32),
    }
    client = _client_seq([
        ("0xtxa", "CONFIRMED"),
        ("0xtxb", "CONFIRMED"),
    ])
    dispatch_content_access_royalties(
        shards=["a", "b"],
        content_index=_index(records),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_amounts_wei={"a": 100, "b": 200},
    )
    assert client.distribute_royalty.call_count == 2
    args_a, _ = client.distribute_royalty.call_args_list[0]
    args_b, _ = client.distribute_royalty.call_args_list[1]
    assert args_a[2] == 100
    assert args_b[2] == 200


def test_zero_amount_short_circuits_to_skipped_status():
    """A shard with zero allocation (e.g. missing record in
    rate_weighted mode) should NOT fire a tx — no point paying
    zero wei. Return distinct status for audit-ring telemetry."""
    records = {
        "a": _rec("a", "aa" * 32),
        "b": _rec("b", "bb" * 32),
    }
    client = _client_seq([
        ("0xtxa", "CONFIRMED"),
        # b never reached
    ])
    results = dispatch_content_access_royalties(
        shards=["a", "b"],
        content_index=_index(records),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_amounts_wei={"a": 100, "b": 0},
    )
    statuses = [r.status for r in results]
    assert statuses == ["sent", "skipped_zero_amount"]
    # Only one tx fired.
    assert client.distribute_royalty.call_count == 1


def test_missing_cid_in_amounts_dict_defaults_to_zero():
    """Cid in shards but missing from amounts dict → zero."""
    records = {"a": _rec("a")}
    client = _client_seq([])
    results = dispatch_content_access_royalties(
        shards=["a"],
        content_index=_index(records),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_amounts_wei={},  # empty
    )
    assert results[0].status == "skipped_zero_amount"
    assert client.distribute_royalty.call_count == 0


def test_legacy_uniform_signature_still_works():
    """Backwards-compat: sprint-248 callers using
    gross_per_shard_wei=N keep working unchanged."""
    records = {"a": _rec("a"), "b": _rec("b")}
    client = _client_seq([
        ("0xtxa", "CONFIRMED"),
        ("0xtxb", "CONFIRMED"),
    ])
    results = dispatch_content_access_royalties(
        shards=["a", "b"],
        content_index=_index(records),
        royalty_client=client,
        serving_node_address="0x" + "1" * 40,
        gross_per_shard_wei=42,
    )
    assert [r.status for r in results] == ["sent", "sent"]
    for call in client.distribute_royalty.call_args_list:
        args, _ = call
        assert args[2] == 42
