"""Sprint 975 — LocalLedger + DAGLedger share ONE node-services implementation.

sp966 mirrored the gossip-log + provenance methods verbatim into DAGLedger to fix
the dormancy where they were absent from the default backend. But verbatim copies
DRIFT — a fix landing in one ledger but not the other (the sp966 parity pin only
guards method EXISTENCE, not that they're the same code). sp975 collapses both
ledgers onto a single LedgerNodeServicesMixin, so the two backends can never
diverge by construction.

This pin asserts both ledgers RESOLVE these methods to the mixin (same function
object via MRO) — if anyone re-adds a per-ledger override, it fails here.
"""
from __future__ import annotations

import json

import pytest

from prsm.node.dag_ledger import DAGLedger
from prsm.node.ledger_node_services import LedgerNodeServicesMixin
from prsm.node.local_ledger import LocalLedger


_NODE_SERVICES = (
    "log_gossip", "get_recent_gossip", "prune_gossip_log",
    "upsert_provenance", "get_provenance", "get_signed_provenance",
)


def test_both_ledgers_inherit_the_mixin():
    assert issubclass(LocalLedger, LedgerNodeServicesMixin)
    assert issubclass(DAGLedger, LedgerNodeServicesMixin)


def test_node_services_methods_resolve_to_the_shared_mixin():
    """No per-ledger override may shadow the mixin — that's exactly the
    duplication/drift sp975 removed."""
    for name in _NODE_SERVICES:
        mixin_fn = getattr(LedgerNodeServicesMixin, name)
        assert getattr(LocalLedger, name) is mixin_fn, (
            f"LocalLedger.{name} overrides the shared mixin — re-duplication"
        )
        assert getattr(DAGLedger, name) is mixin_fn, (
            f"DAGLedger.{name} overrides the shared mixin — re-duplication"
        )


# ── behaviour-preservation: both ledgers still work identically ────────────


async def _fresh(ledger_cls):
    if ledger_cls is DAGLedger:
        led = DAGLedger(":memory:", verify_signatures=False)
    else:
        led = LocalLedger(":memory:")
    await led.initialize()
    return led


@pytest.mark.parametrize("ledger_cls", [LocalLedger, DAGLedger])
@pytest.mark.asyncio
async def test_gossip_and_provenance_roundtrip_identical(ledger_cls):
    led = await _fresh(ledger_cls)
    # gossip-log roundtrip (with sp961 attestation)
    att = {"origin_time": 1.0, "origin_pubkey": "P", "origin_sig": "S"}
    await led.log_gossip(nonce="n1", subtype="job_offer", origin="node_a",
                         payload={"job_id": "j1"}, ttl=1, attestation=att)
    rows = await led.get_recent_gossip(since=0)
    assert len(rows) == 1
    assert rows[0]["payload"] == {"job_id": "j1"}
    assert rows[0]["attestation"] == att
    # provenance roundtrip (with sp965 verbatim signed_record)
    rec = {"cid": "C1", "content_hash": "ab", "creator_id": "alice",
           "creator_public_key": "pk", "filename": "f", "size_bytes": 1,
           "royalty_rate": 0.01, "parent_cids": [], "metadata": {},
           "signature": "sig"}
    await led.upsert_provenance(rec)
    got = await led.get_provenance("C1")
    assert got["creator_id"] == "alice"
    assert await led.get_signed_provenance("C1") == rec
