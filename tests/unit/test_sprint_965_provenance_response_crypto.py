"""Sprint 965 — make the provenance RESPONSE path cryptographically verifiable.

sp964 hardened the broadcast REGISTER path (full signature + creator-binding) but
left the cross-node RESPONSE path on first-writer-wins ONLY, because the responder
re-served a LOSSY typed row from get_provenance (registered_at != the signed
created_at; is_sharded/provenance_hash dropped), whose signature could not be
reconstructed. So a peer answering a GOSSIP_PROVENANCE_QUERY for a brand-new cid
could still feed us a forged creator (informational, MEDIUM, but unsound).

Fix: persist the VERBATIM signed record (a nullable signed_record JSON column,
idempotent migration) and have the query responder re-serve THAT verbatim form, so
the receiver runs the same require_signature=True check as the REGISTER path. With
this, both gossip provenance ingestion paths are fully authenticated; a forged
response cannot be persisted, and an honest one survives the storage + wire
round-trips and verifies.
"""
from __future__ import annotations

import json

import pytest

from prsm.node.content_index import ContentIndex
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from unittest.mock import AsyncMock, MagicMock


async def _ledger():
    led = LocalLedger(":memory:")
    await led.initialize()
    return led


def _signed_record(identity, *, cid="C1", **overrides):
    prov = {
        "cid": cid,
        "content_hash": "ab" * 32,
        "creator_id": identity.node_id,
        "creator_public_key": identity.public_key_b64,
        "filename": "f.txt",
        "size_bytes": 10,
        "created_at": 1780000000.5,
        "metadata": {"k": "v"},
        "royalty_rate": 0.01,
        "parent_cids": [],
        "is_sharded": False,
        "provenance_hash": "ff" * 16,
    }
    prov.update(overrides)
    sig = identity.sign(json.dumps(prov, sort_keys=True).encode())
    return {**prov, "signature": sig}


# ── ledger: verbatim signed-record persistence ─────────────────────────────


@pytest.mark.asyncio
async def test_ledger_persists_and_returns_verbatim_signed_record():
    led = await _ledger()
    alice = generate_node_identity()
    rec = _signed_record(alice, cid="C1")
    await led.upsert_provenance(rec)
    got = await led.get_signed_provenance("C1")
    assert got is not None
    # Verbatim: the signed_record round-trips byte-stable for re-verification.
    assert got == rec


@pytest.mark.asyncio
async def test_ledger_signed_record_absent_is_none():
    led = await _ledger()
    # A row written without a verbatim record (legacy) returns None.
    assert await led.get_signed_provenance("nope") is None


# ── responder re-serves the verbatim signed record ─────────────────────────


@pytest.mark.asyncio
async def test_query_responder_serves_verbatim_signed_record():
    led = await _ledger()
    alice = generate_node_identity()
    idx = ContentIndex(gossip=MagicMock(), ledger=led)
    idx.gossip.publish = AsyncMock()
    await idx._on_provenance_register(
        "provenance_register", _signed_record(alice, cid="C1"), origin=alice.node_id)
    await idx._on_provenance_query(
        "provenance_query", {"cid": "C1", "requester_id": "peerB"}, origin="peerB")
    idx.gossip.publish.assert_awaited()
    _subtype, payload = idx.gossip.publish.await_args[0]
    served = payload["provenance"]
    # The served record must carry the signature + signed fields so a peer can verify.
    assert served.get("signature")
    assert served.get("creator_public_key") == alice.public_key_b64
    assert served.get("created_at") == 1780000000.5  # signed field, not registered_at


# ── full cross-node round-trip: register -> query -> response -> verify ────


@pytest.mark.asyncio
async def test_response_path_accepts_verified_record_end_to_end():
    alice = generate_node_identity()
    # Node A stores a valid record and answers a query with the verbatim form.
    led_a = await _ledger()
    idx_a = ContentIndex(gossip=MagicMock(), ledger=led_a)
    idx_a.gossip.publish = AsyncMock()
    await idx_a._on_provenance_register(
        "provenance_register", _signed_record(alice, cid="C1"), origin=alice.node_id)
    await idx_a._on_provenance_query(
        "provenance_query", {"cid": "C1", "requester_id": "B"}, origin="B")
    _subtype, response_payload = idx_a.gossip.publish.await_args[0]
    # Simulate the wire: the response JSON-serializes across the network.
    wire = json.loads(json.dumps(response_payload))

    # Node B receives the response and must VERIFY it before persisting.
    led_b = await _ledger()
    idx_b = ContentIndex(gossip=MagicMock(), ledger=led_b)
    await idx_b._on_provenance_response("provenance_response", wire, origin="A")
    stored = await led_b.get_provenance("C1")
    assert stored is not None, "an honest verbatim-signed response must verify + persist"
    assert stored["creator_id"] == alice.node_id


@pytest.mark.asyncio
async def test_response_path_rejects_forged_record_for_new_cid():
    """A malicious responder answers our query for an unseen cid with a record
    whose signature does not verify -> must NOT be persisted (sp964 left this
    open on the response path; sp965 closes it)."""
    led_b = await _ledger()
    idx_b = ContentIndex(gossip=MagicMock(), ledger=led_b)
    attacker = generate_node_identity()
    forged = _signed_record(attacker, cid="Cnew")
    forged["creator_id"] = "victim_creator_id_0000000000aabb"  # claim a victim, sig now invalid for binding
    await idx_b._on_provenance_response(
        "provenance_response", {"cid": "Cnew", "provenance": forged}, origin="attacker")
    assert await led_b.get_provenance("Cnew") is None


@pytest.mark.asyncio
async def test_response_forged_does_not_resolve_future_or_persist():
    """A forged response must NOT resolve a pending get_provenance() future (the
    caller must never receive forged provenance — it waits for an honest answer
    or times out to None) and must NOT be persisted."""
    import asyncio
    led_b = await _ledger()
    idx_b = ContentIndex(gossip=MagicMock(), ledger=led_b)
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    idx_b._pending_provenance["Cnew"] = fut
    attacker = generate_node_identity()
    forged = _signed_record(attacker, cid="Cnew")
    forged["creator_id"] = "victim_creator_id_0000000000aabb"
    await idx_b._on_provenance_response(
        "provenance_response", {"cid": "Cnew", "provenance": forged}, origin="attacker")
    assert not fut.done()  # caller NOT handed forged data
    assert await led_b.get_provenance("Cnew") is None  # forged record not cached


@pytest.mark.asyncio
async def test_response_verified_resolves_future():
    """An authentic response resolves the pending future (caller unblocked) and
    persists the record."""
    import asyncio
    alice = generate_node_identity()
    rec = _signed_record(alice, cid="Cok")
    led_b = await _ledger()
    idx_b = ContentIndex(gossip=MagicMock(), ledger=led_b)
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    idx_b._pending_provenance["Cok"] = fut
    wire = json.loads(json.dumps({"cid": "Cok", "provenance": rec}))
    await idx_b._on_provenance_response("provenance_response", wire, origin="A")
    assert fut.done()
    assert fut.result()["creator_id"] == alice.node_id
    assert (await led_b.get_provenance("Cok"))["creator_id"] == alice.node_id
