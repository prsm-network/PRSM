"""Sprint 964 — authenticate GOSSIP_PROVENANCE_REGISTER before persisting.

`ContentIndex._on_provenance_register` ignored the authenticated `origin` and
passed the publisher-controlled `data` straight to `ledger.upsert_provenance`,
which blind-overwrites `creator_id`/`creator_pubkey`/`royalty_rate` on
`ON CONFLICT(cid)`. The `signature` field (a real ed25519 sig the uploader makes
over `json.dumps(provenance_data, sort_keys=True)`, with creator_id = node_id =
sha256(pubkey)[:32]) was stored but NEVER verified. So any gossip-connected peer
could forge or hijack the §14 first-creator-wins provenance record on every node
and re-serve it over the cross-node provenance API.

Severity is MEDIUM (integrity/authority spoofing, NOT fund loss — no payment path
reads this ledger table; on-chain royalty reads the immutable on-chain
ProvenanceRegistry and local FTNS royalty credits the node's own identity with a
self-clamped rate). But it is one wiring change from a payment bug and forging the
provenance authority is wrong on its own.

Fix (this sprint): on the broadcast REGISTER path, verify the signature + bind the
pubkey to the claimed creator_id (the proven _authenticate_origin scheme), and
enforce first-writer-wins so a gossip record can never change an existing cid's
creator. The RESPONSE path (a reply to our own query, carrying a lossy re-
serialized stored row whose signature can't be reconstructed) gets first-writer-
wins only; full response-path crypto is a documented follow-on.
"""
from __future__ import annotations

import json

import pytest

from prsm.node.content_index import ContentIndex
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from unittest.mock import MagicMock


async def _ledger():
    led = LocalLedger(":memory:")
    await led.initialize()
    return led


def _signed_record(identity, *, cid="C1", royalty_rate=0.01, **overrides):
    """Build a provenance record the way ContentUploader publishes it:
    creator_id = node_id, creator_public_key = pubkey, signature over the
    sort_keys json of the dict WITHOUT the signature field."""
    prov = {
        "cid": cid,
        "content_hash": "ab" * 32,
        "creator_id": identity.node_id,
        "creator_public_key": identity.public_key_b64,
        "filename": "f.txt",
        "size_bytes": 10,
        "created_at": 123.0,
        "metadata": {},
        "royalty_rate": royalty_rate,
        "parent_cids": [],
    }
    prov.update(overrides)
    sig = identity.sign(json.dumps(prov, sort_keys=True).encode())
    return {**prov, "signature": sig}


def _idx(led):
    return ContentIndex(gossip=MagicMock(), ledger=led)


# ── REGISTER path: full crypto verification ────────────────────────────────


@pytest.mark.asyncio
async def test_valid_signed_register_for_new_cid_accepted():
    led = await _ledger()
    alice = generate_node_identity()
    rec = _signed_record(alice, cid="C1")
    await _idx(led)._on_provenance_register("provenance_register", rec, origin=alice.node_id)
    stored = await led.get_provenance("C1")
    assert stored is not None
    assert stored["creator_id"] == alice.node_id


@pytest.mark.asyncio
async def test_unsigned_register_rejected():
    led = await _ledger()
    alice = generate_node_identity()
    rec = _signed_record(alice, cid="C1")
    rec.pop("signature")
    await _idx(led)._on_provenance_register("provenance_register", rec, origin=alice.node_id)
    assert await led.get_provenance("C1") is None


@pytest.mark.asyncio
async def test_tampered_register_rejected():
    led = await _ledger()
    alice = generate_node_identity()
    rec = _signed_record(alice, cid="C1", royalty_rate=0.01)
    rec["royalty_rate"] = 0.99   # tamper AFTER signing → signature no longer matches
    await _idx(led)._on_provenance_register("provenance_register", rec, origin=alice.node_id)
    assert await led.get_provenance("C1") is None


@pytest.mark.asyncio
async def test_pubkey_creator_binding_enforced():
    """A record whose signature verifies against its pubkey but whose pubkey
    does NOT hash to the claimed creator_id (impersonating a victim creator_id)
    must be rejected."""
    led = await _ledger()
    attacker = generate_node_identity()
    rec = _signed_record(attacker, cid="C1")
    # Claim a victim's creator_id while keeping the attacker's real pubkey+sig.
    rec_forged = dict(rec)
    rec_forged["creator_id"] = "victim_creator_id_000000000000aa"
    # Re-sign so the signature is valid for the forged dict, but the pubkey
    # still hashes to the attacker's node_id, not the claimed creator_id.
    rec_forged.pop("signature")
    rec_forged["signature"] = attacker.sign(
        json.dumps(rec_forged, sort_keys=True).encode())
    await _idx(led)._on_provenance_register("provenance_register", rec_forged, origin=attacker.node_id)
    assert await led.get_provenance("C1") is None


# ── first-writer-wins (the documented hijack) ──────────────────────────────


@pytest.mark.asyncio
async def test_hijack_of_existing_cid_rejected_even_if_validly_signed():
    led = await _ledger()
    alice = generate_node_identity()
    mallory = generate_node_identity()
    # Alice legitimately registers C1.
    await _idx(led)._on_provenance_register(
        "provenance_register", _signed_record(alice, cid="C1", royalty_rate=0.01),
        origin=alice.node_id)
    # Mallory broadcasts her OWN validly-signed record for the SAME cid C1
    # (binding + signature PASS — it's a genuine record from mallory).
    mallory_rec = _signed_record(mallory, cid="C1", royalty_rate=0.5)
    await _idx(led)._on_provenance_register(
        "provenance_register", mallory_rec, origin=mallory.node_id)
    stored = await led.get_provenance("C1")
    assert stored["creator_id"] == alice.node_id, "first-writer-wins: alice keeps C1"
    assert stored["creator_public_key"] == alice.public_key_b64
    assert stored["royalty_rate"] == 0.01


@pytest.mark.asyncio
async def test_same_creator_metadata_update_allowed():
    led = await _ledger()
    alice = generate_node_identity()
    idx = _idx(led)
    await idx._on_provenance_register(
        "provenance_register", _signed_record(alice, cid="C1", filename="old.txt"),
        origin=alice.node_id)
    # Alice re-registers the same cid with an updated filename (valid signed).
    await idx._on_provenance_register(
        "provenance_register", _signed_record(alice, cid="C1", filename="new.txt"),
        origin=alice.node_id)
    stored = await led.get_provenance("C1")
    assert stored["creator_id"] == alice.node_id
    assert stored["filename"] == "new.txt"   # mutable field updated by same creator


# ── RESPONSE path: first-writer-wins (lossy record, no sig verify) ─────────


@pytest.mark.asyncio
async def test_honest_record_verifies_after_gossip_wire_roundtrip():
    """The highest-risk regression: the gossip transport JSON-serializes the
    record on the wire. An honest record must STILL verify after a full
    json.dumps -> json.loads round-trip (floats, ints, nested dicts), or the fix
    would silently reject every legitimate provenance registration in production.
    Mirrors the real uploader field set incl. is_sharded/embedding_id/parent_cids/
    near_duplicate_of/provenance_hash + a float created_at + nested metadata."""
    led = await _ledger()
    alice = generate_node_identity()
    prov = {
        "cid": "Cwire",
        "content_hash": "ab" * 32,
        "creator_id": alice.node_id,
        "creator_public_key": alice.public_key_b64,
        "filename": "f.txt",
        "size_bytes": 4096,
        "created_at": 1780000000.123456,   # float — the classic round-trip risk
        "metadata": {"tags": ["a", "b"], "nested": {"x": 1}},
        "royalty_rate": 0.025,
        "parent_cids": ["p1", "p2"],
        "is_sharded": False,
        "embedding_id": "emb-1",
        "near_duplicate_of": None,
        "provenance_hash": "ff" * 16,
    }
    sig = alice.sign(json.dumps(prov, sort_keys=True).encode())
    record = {**prov, "signature": sig}
    # Simulate the gossip transport: serialize + deserialize exactly as the wire does.
    wire = json.loads(json.dumps(record))
    await _idx(led)._on_provenance_register("provenance_register", wire, origin=alice.node_id)
    stored = await led.get_provenance("Cwire")
    assert stored is not None, "honest record must survive the wire round-trip + verify"
    assert stored["creator_id"] == alice.node_id


@pytest.mark.asyncio
async def test_response_cannot_hijack_existing_cid():
    led = await _ledger()
    alice = generate_node_identity()
    mallory = generate_node_identity()
    idx = _idx(led)
    await idx._on_provenance_register(
        "provenance_register", _signed_record(alice, cid="C1"), origin=alice.node_id)
    # A malicious GOSSIP_PROVENANCE_RESPONSE claiming mallory created C1.
    forged = {
        "cid": "C1", "creator_id": mallory.node_id,
        "creator_public_key": mallory.public_key_b64, "content_hash": "cd" * 32,
        "filename": "x", "size_bytes": 1, "royalty_rate": 0.9, "parent_cids": [],
        "metadata": {}, "signature": "whatever",
    }
    await idx._on_provenance_response(
        "provenance_response", {"cid": "C1", "provenance": forged}, origin=mallory.node_id)
    stored = await led.get_provenance("C1")
    assert stored["creator_id"] == alice.node_id  # first-writer-wins holds on response path too
