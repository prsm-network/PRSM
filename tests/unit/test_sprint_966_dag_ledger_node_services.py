"""Sprint 966 — revive the node-services ledger surface on the DEFAULT (dag) backend.

config.ledger_type defaults to "dag" -> node wires the raw DAGLedger as self.ledger
(node.py:1770) and as gossip.ledger / content_index.ledger. But the raw DAGLedger
defined NONE of the node-services methods (log_gossip / get_recent_gossip /
prune_gossip_log / upsert_provenance / get_provenance / get_signed_provenance) and
had no provenance_chains table — so gossip-log persistence + digest catch-up and
the durable provenance cache/lookup were SILENTLY DEAD on the default backend
(every call AttributeError'd and was swallowed by callers' try/except). The
DAGLedgerAdapter built to provide this surface was never wired. This made
sp961/964/965 dormant on default nodes.

Fix: additively mirror the node-services surface (the same SQL LocalLedger uses)
onto the raw DAGLedger, plus a parity pin so the two ledgers can never drift again.
"""
from __future__ import annotations

import json

import pytest

from prsm.node.content_index import ContentIndex
from prsm.node.dag_ledger import DAGLedger
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from unittest.mock import AsyncMock, MagicMock


_NODE_SERVICES_METHODS = (
    "log_gossip", "get_recent_gossip", "prune_gossip_log",
    "upsert_provenance", "get_provenance", "get_signed_provenance",
)


async def _dag():
    led = DAGLedger(":memory:", verify_signatures=False)
    await led.initialize()
    return led


def _signed_record(identity, *, cid="C1", **overrides):
    prov = {
        "cid": cid, "content_hash": "ab" * 32,
        "creator_id": identity.node_id,
        "creator_public_key": identity.public_key_b64,
        "filename": "f.txt", "size_bytes": 10, "created_at": 123.0,
        "metadata": {}, "royalty_rate": 0.01, "parent_cids": [],
    }
    prov.update(overrides)
    sig = identity.sign(json.dumps(prov, sort_keys=True).encode())
    return {**prov, "signature": sig}


# ── parity pin: the two ledgers must never drift again ─────────────────────


def test_dag_and_local_ledger_expose_same_node_services_surface():
    for m in _NODE_SERVICES_METHODS:
        assert hasattr(DAGLedger, m), f"DAGLedger missing node-services method {m!r}"
        assert hasattr(LocalLedger, m), f"LocalLedger missing node-services method {m!r}"


# ── gossip-log persistence (was dead on dag) ───────────────────────────────


@pytest.mark.asyncio
async def test_dag_gossip_log_roundtrip_with_attestation():
    led = await _dag()
    att = {"origin_time": 1.0, "origin_pubkey": "P", "origin_sig": "S"}
    await led.log_gossip(nonce="n1", subtype="job_offer", origin="node_a",
                         payload={"job_id": "j1"}, ttl=1, attestation=att)
    rows = await led.get_recent_gossip(since=0)
    assert len(rows) == 1
    assert rows[0]["payload"] == {"job_id": "j1"}
    assert rows[0]["attestation"] == att


@pytest.mark.asyncio
async def test_dag_prune_gossip_log():
    led = await _dag()
    await led.log_gossip(nonce="n1", subtype="job_offer", origin="a",
                         payload={}, ttl=1)
    pruned = await led.prune_gossip_log(max_age=-1)  # cutoff in the future → prune all
    assert pruned == 1
    assert await led.get_recent_gossip(since=0) == []


# ── provenance (was dead on dag) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_dag_provenance_roundtrip():
    led = await _dag()
    alice = generate_node_identity()
    rec = _signed_record(alice, cid="C1")
    await led.upsert_provenance(rec)
    got = await led.get_provenance("C1")
    assert got is not None
    assert got["creator_id"] == alice.node_id
    # verbatim signed record preserved (sp965 response-path verifiability)
    assert await led.get_signed_provenance("C1") == rec


@pytest.mark.asyncio
async def test_dag_get_signed_provenance_absent_is_none():
    led = await _dag()
    assert await led.get_signed_provenance("nope") is None


# ── end-to-end: the gossip-authorship arc now works on the DEFAULT backend ─


@pytest.mark.asyncio
async def test_provenance_register_and_query_work_on_dag_backend():
    led = await _dag()
    alice = generate_node_identity()
    idx = ContentIndex(gossip=MagicMock(), ledger=led)
    idx.gossip.publish = AsyncMock()
    # REGISTER (sp964 verification) now actually persists on a dag-backed node.
    await idx._on_provenance_register(
        "provenance_register", _signed_record(alice, cid="C1"), origin=alice.node_id)
    stored = await led.get_provenance("C1")
    assert stored is not None and stored["creator_id"] == alice.node_id
    # QUERY responder (sp965) serves the verbatim signed form.
    await idx._on_provenance_query(
        "provenance_query", {"cid": "C1", "requester_id": "B"}, origin="B")
    idx.gossip.publish.assert_awaited()
    _sub, payload = idx.gossip.publish.await_args[0]
    assert payload["provenance"].get("signature")
    assert payload["provenance"].get("created_at") == 123.0  # verbatim, not registered_at


@pytest.mark.asyncio
async def test_provenance_first_writer_wins_on_dag_backend():
    led = await _dag()
    alice = generate_node_identity()
    mallory = generate_node_identity()
    idx = ContentIndex(gossip=MagicMock(), ledger=led)
    await idx._on_provenance_register(
        "provenance_register", _signed_record(alice, cid="C1"), origin=alice.node_id)
    await idx._on_provenance_register(
        "provenance_register", _signed_record(mallory, cid="C1", royalty_rate=0.5),
        origin=mallory.node_id)
    stored = await led.get_provenance("C1")
    assert stored["creator_id"] == alice.node_id  # first-writer-wins holds on dag too
