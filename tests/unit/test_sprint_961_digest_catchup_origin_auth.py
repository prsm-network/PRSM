"""Sprint 961 — authenticate the gossip ORIGIN on the digest catch-up path.

sp934 authenticated the origin on the LIVE gossip path (`_handle_gossip` calls
`_authenticate_origin`: trust `payload['origin']` only if its pubkey hashes to
the claimed node_id AND the signature verifies, else fall back to the
transport-authenticated `sender_id`). But the catch-up path
(`_handle_digest_response`) bypassed it entirely — it took each replayed
message's `origin` straight from the relaying peer's payload and delivered/stored
it unverified. So a malicious peer answering a digest request could forge the
AUTHORSHIP of any caught-up content/job/provenance/agent advertisement (the same
full-impersonation hole sp934 closed on the live path) and poison the local
gossip log with forged-origin entries.

Root cause: the gossip log only persisted the (already-authenticated) origin, not
the attestation (pubkey/sig/origin_time) needed to RE-verify it relayer-
independently. Fix: persist the attestation (a nullable `attestation` column,
idempotent migration — `payload` still stores just `data`, so existing consumers
are untouched) and re-run `_authenticate_origin` on every caught-up message.
Backward-compatible: pre-upgrade rows / messages without an attestation fall back
to the relaying `sender_id` — never the spoofed claim.

Scope: the WebSocket `gossip.py` path + `local_ledger`. The `libp2p_gossip.py`
path already uses `sender_id` (never a claimed origin), so it is not vulnerable
to this spoof; `dag_ledger` gains the same optional attestation param for
signature parity. Both flagged as the immediate follow-on for full parity.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.gossip import (
    GossipProtocol,
    GOSSIP_DIGEST_RESPONSE,
    build_gossip_origin_fields,
)
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger
from prsm.node.transport import MSG_GOSSIP, P2PMessage


# ── ledger: attestation persistence roundtrip ──────────────────────────────


@pytest.mark.asyncio
async def test_ledger_persists_and_returns_attestation():
    led = LocalLedger(":memory:")
    await led.initialize()
    attestation = {
        "origin_time": 123.0,
        "origin_pubkey": "PUBKEY_B64",
        "origin_sig": "SIG_B64",
    }
    await led.log_gossip(
        nonce="n1", subtype="job_offer", origin="node_a",
        payload={"job_id": "j1"}, ttl=1, attestation=attestation,
    )
    rows = await led.get_recent_gossip(since=0)
    assert len(rows) == 1
    assert rows[0]["payload"] == {"job_id": "j1"}  # payload still == data
    assert rows[0]["attestation"] == attestation


@pytest.mark.asyncio
async def test_ledger_attestation_absent_is_none():
    led = LocalLedger(":memory:")
    await led.initialize()
    await led.log_gossip(
        nonce="n1", subtype="job_offer", origin="node_a",
        payload={"job_id": "j1"}, ttl=1,
    )
    rows = await led.get_recent_gossip(since=0)
    assert rows[0]["attestation"] is None


# ── gossip catch-up: re-authenticate the origin ────────────────────────────


def _gossip(receiver):
    transport = MagicMock()
    transport.identity = receiver
    transport.on_message = MagicMock()
    transport.gossip = AsyncMock(return_value=0)
    transport.send_to_peer = AsyncMock(return_value=True)
    gp = GossipProtocol(transport)
    gp.ledger = MagicMock()
    gp.ledger.log_gossip = AsyncMock()
    gp.ledger.get_recent_gossip = AsyncMock(return_value=[])
    return gp


async def _deliver_catchup(gp, message_data, sender_id="relayer_node"):
    seen = []

    async def cb(subtype, data, origin):
        seen.append(origin)

    gp._subscribers[message_data["subtype"]] = [cb]
    gp._is_duplicate = AsyncMock(return_value=False)
    msg = P2PMessage(
        msg_type=MSG_GOSSIP, sender_id=sender_id,
        payload={"subtype": GOSSIP_DIGEST_RESPONSE,
                 "data": {"messages": [message_data], "total_count": 1}},
        ttl=1,
    )
    await gp._handle_digest_response(msg, MagicMock())
    return seen


def _attested_entry(author, *, subtype="job_offer", data=None, nonce="cn1"):
    data = data if data is not None else {"job_id": "j1"}
    fields = build_gossip_origin_fields(author, subtype, data, 123.0, nonce)
    return {
        "nonce": nonce,
        "subtype": subtype,
        "origin": fields["origin"],
        "payload": data,
        "received_at": 1.0,
        "attestation": {
            "origin_time": fields["origin_time"],
            "origin_pubkey": fields["origin_pubkey"],
            "origin_sig": fields["origin_sig"],
        },
    }


@pytest.mark.asyncio
async def test_catchup_valid_attestation_trusts_real_author():
    receiver = generate_node_identity()
    author = generate_node_identity()
    gp = _gossip(receiver)
    entry = _attested_entry(author)
    seen = await _deliver_catchup(gp, entry)
    assert seen == [author.node_id], (
        "a caught-up message with a valid attestation must be attributed to its "
        "real signed author"
    )


@pytest.mark.asyncio
async def test_catchup_forged_origin_falls_back_to_relayer():
    """Attacker relays an entry claiming a VICTIM origin but signs the
    attestation with its OWN key — pubkey won't hash to the claimed victim id,
    so the claim must be rejected and origin must NOT be the victim."""
    receiver = generate_node_identity()
    attacker = generate_node_identity()
    gp = _gossip(receiver)
    entry = _attested_entry(attacker)        # validly signed by attacker
    entry["origin"] = "VICTIM_NODE_ID"        # ...but claims to be the victim
    seen = await _deliver_catchup(gp, entry, sender_id="attacker_relayer")
    assert seen == ["attacker_relayer"], (
        "a forged origin claim must fall back to the transport-authenticated "
        "relayer, never the spoofed victim id"
    )
    assert "VICTIM_NODE_ID" not in seen


@pytest.mark.asyncio
async def test_catchup_no_attestation_falls_back_to_relayer():
    receiver = generate_node_identity()
    gp = _gossip(receiver)
    entry = {
        "nonce": "cn2", "subtype": "job_offer", "origin": "claimed_author",
        "payload": {"job_id": "j2"}, "received_at": 1.0,
        # no "attestation" — pre-upgrade / unsigned
    }
    seen = await _deliver_catchup(gp, entry, sender_id="relayer_x")
    assert seen == ["relayer_x"], (
        "an unattested caught-up message must fall back to the relayer, not the "
        "bare claimed origin"
    )


@pytest.mark.asyncio
async def test_catchup_stores_authenticated_origin_not_claim():
    receiver = generate_node_identity()
    attacker = generate_node_identity()
    gp = _gossip(receiver)
    entry = _attested_entry(attacker)
    entry["origin"] = "VICTIM_NODE_ID"
    await _deliver_catchup(gp, entry, sender_id="attacker_relayer")
    gp.ledger.log_gossip.assert_called_once()
    stored_origin = gp.ledger.log_gossip.call_args[1]["origin"]
    assert stored_origin == "attacker_relayer"
    assert stored_origin != "VICTIM_NODE_ID"  # forged origin not persisted


# ── live path persists the attestation (so a later digest carries it) ──────


@pytest.mark.asyncio
async def test_live_handler_persists_attestation_for_later_catchup():
    receiver = generate_node_identity()
    author = generate_node_identity()
    gp = _gossip(receiver)
    nonce = "live1"
    fields = build_gossip_origin_fields(author, "job_offer", {"job_id": "j1"}, 99.0, nonce)
    msg = P2PMessage(
        msg_type=MSG_GOSSIP, sender_id="some_relayer",
        payload={"subtype": "job_offer", "data": {"job_id": "j1"}, **fields},
        ttl=1, nonce=nonce,
    )
    await gp._handle_gossip(msg, MagicMock())
    gp.ledger.log_gossip.assert_called_once()
    kw = gp.ledger.log_gossip.call_args[1]
    assert kw["origin"] == author.node_id           # authenticated on the way in
    assert kw["attestation"]["origin_pubkey"] == fields["origin_pubkey"]
    assert kw["attestation"]["origin_sig"] == fields["origin_sig"]
    assert kw["attestation"]["origin_time"] == 99.0
