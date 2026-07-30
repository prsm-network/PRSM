"""Sprint 1487 — scanning BatchFinalized: the epoch job's input must be reorg-safe.

An emission epoch is a published Merkle root that can NEVER be rewritten. So paying
against a BatchFinalized log at the chain head means that if a reorg un-finalizes
that batch, the pot has already been spent on work the chain no longer agrees
happened — unrecoverable. Waiting for confirmations merely pays late.

This is the same gate sp1478 put on bridge deposit credits, for the same reason.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.web3.batch_settlement_contract_client import (
    BATCH_SETTLEMENT_REGISTRY_ABI,
    Web3SettlementContractClient,
)

PROV_A = "0x" + "a1" * 20
PROV_B = "0x" + "b2" * 20


def _entry(batch_id_byte, provider, value, ts):
    return {"args": {
        "batchId": bytes([batch_id_byte]) * 32,
        "provider": provider,
        "finalValueFTNS": value,
        "invalidatedValueFTNS": 0,
        "finalizeTimestamp": ts,
    }}


def _client(head=1000, entries=None):
    c = object.__new__(Web3SettlementContractClient)
    c.web3 = MagicMock()
    c.web3.eth.block_number = head
    c.contract = MagicMock()
    flt = MagicMock()
    flt.get_all_entries.return_value = list(entries or [])
    c.contract.events.BatchFinalized.return_value.create_filter.return_value = flt
    return c


def test_abi_exposes_batch_finalized():
    ev = [e for e in BATCH_SETTLEMENT_REGISTRY_ABI
          if e.get("type") == "event" and e.get("name") == "BatchFinalized"]
    assert len(ev) == 1
    names = [i["name"] for i in ev[0]["inputs"]]
    assert names == ["batchId", "provider", "finalValueFTNS",
                     "invalidatedValueFTNS", "finalizeTimestamp"]
    # batchId and provider indexed -> the RPC can filter server-side by provider.
    idx = {i["name"]: i["indexed"] for i in ev[0]["inputs"]}
    assert idx["batchId"] and idx["provider"]


def test_scan_returns_finalized_batches():
    c = _client(entries=[_entry(0xaa, PROV_A, 500, 1000),
                         _entry(0xbb, PROV_B, 300, 1001)])
    out = c.scan_finalized_batches(from_block=0)
    assert len(out) == 2
    assert out[0].provider == PROV_A and out[0].final_value_wei == 500
    assert out[0].batch_id == "0x" + "aa" * 32


def test_scan_STOPS_at_the_confirmation_depth():
    """★ THE guard. Head is 1000; with 12 confirmations nothing past 988 is safe to
    pay against, because a reorg there un-finalizes work whose payment is already an
    unrewritable published root."""
    c = _client(head=1000, entries=[])
    c.scan_finalized_batches(from_block=0, confirmations=12)
    kwargs = c.contract.events.BatchFinalized.return_value.create_filter.call_args.kwargs
    assert kwargs["to_block"] == 988


def test_an_explicit_to_block_is_CLAMPED_not_trusted():
    """A caller asking for the head must not be able to opt out of the depth."""
    c = _client(head=1000, entries=[])
    c.scan_finalized_batches(from_block=0, to_block=1000, confirmations=12)
    kwargs = c.contract.events.BatchFinalized.return_value.create_filter.call_args.kwargs
    assert kwargs["to_block"] == 988


def test_an_explicit_to_block_below_the_safe_head_is_respected():
    c = _client(head=1000, entries=[])
    c.scan_finalized_batches(from_block=0, to_block=500, confirmations=12)
    kwargs = c.contract.events.BatchFinalized.return_value.create_filter.call_args.kwargs
    assert kwargs["to_block"] == 500


def test_a_chain_shallower_than_the_depth_yields_nothing():
    """Not a crash, and not 'scan from 0' — a fresh chain simply has nothing settled
    deeply enough to pay for yet."""
    c = _client(head=5, entries=[_entry(0xaa, PROV_A, 500, 1000)])
    assert c.scan_finalized_batches(from_block=0, confirmations=12) == []


def test_start_past_the_safe_head_scans_nothing():
    c = _client(head=1000, entries=[_entry(0xaa, PROV_A, 5, 1)])
    assert c.scan_finalized_batches(from_block=999, confirmations=12) == []


def test_duplicate_logs_are_deduped():
    """Overlapping windows must not double-weight a provider."""
    c = _client(entries=[_entry(0xaa, PROV_A, 500, 1000),
                         _entry(0xaa, PROV_A, 500, 1000)])
    out = c.scan_finalized_batches(from_block=0)
    assert len(out) == 1


def test_zero_value_batches_are_returned_so_they_can_be_consumed():
    """A fully-challenged batch pays nothing, but the planner must still mark it
    consumed — otherwise every epoch re-examines it forever."""
    c = _client(entries=[_entry(0xaa, PROV_A, 0, 1000)])
    out = c.scan_finalized_batches(from_block=0)
    assert len(out) == 1 and out[0].final_value_wei == 0


def test_output_order_is_deterministic():
    """Two parties scanning the same range must build the same epoch."""
    c1 = _client(entries=[_entry(0xbb, PROV_B, 1, 20), _entry(0xaa, PROV_A, 1, 10)])
    c2 = _client(entries=[_entry(0xaa, PROV_A, 1, 10), _entry(0xbb, PROV_B, 1, 20)])
    assert ([b.batch_id for b in c1.scan_finalized_batches(from_block=0)]
            == [b.batch_id for b in c2.scan_finalized_batches(from_block=0)])


def test_provider_filter_is_pushed_to_the_rpc():
    c = _client(entries=[])
    c.scan_finalized_batches(from_block=0, provider=PROV_A)
    kwargs = c.contract.events.BatchFinalized.return_value.create_filter.call_args.kwargs
    assert kwargs["argument_filters"]["provider"].lower() == PROV_A.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
