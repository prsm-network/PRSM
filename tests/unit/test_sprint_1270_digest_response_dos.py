"""Sprint 1270 — bound the catch-up messages from a digest RESPONSE (audit round 5, HIGH).

_handle_digest_response looped over a peer-controlled `messages` list with NO cap, calling
_is_duplicate (a ledger scan) per entry — so a malicious digest_response with a huge list was
a DoS amplifier (one ledger scan per attacker-supplied message). The request path already
caps at 100 (sp1182 lineage); this mirrors that on the response path
(_MAX_DIGEST_RESPONSE_MESSAGES = 200, generous slack over the legit 100).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.gossip import GossipProtocol, _MAX_DIGEST_RESPONSE_MESSAGES


def _gp():
    gp = GossipProtocol(transport=MagicMock())
    gp.ledger = None
    gp._subscribers = {}
    return gp


def _msg(n):
    messages = [
        {"subtype": "content", "payload": {"x": i}, "nonce": f"n{i}",
         "origin": "origin", "attestation": None}
        for i in range(n)
    ]
    m = MagicMock()
    m.payload = {"data": {"messages": messages}}
    m.sender_id = "peerabcd1234"
    return m


@pytest.mark.asyncio
async def test_oversized_response_truncated_to_cap(monkeypatch):
    gp = _gp()
    calls = {"n": 0}

    async def _dup(nonce):
        calls["n"] += 1
        return True  # treat all as duplicate → fast skip after the dedup call

    monkeypatch.setattr(gp, "_is_duplicate", _dup)
    await gp._handle_digest_response(_msg(5000), MagicMock())
    # the per-message ledger scan ran AT MOST the cap many times, not 5000
    assert calls["n"] == _MAX_DIGEST_RESPONSE_MESSAGES


@pytest.mark.asyncio
async def test_under_cap_all_processed(monkeypatch):
    gp = _gp()
    calls = {"n": 0}

    async def _dup(nonce):
        calls["n"] += 1
        return True

    monkeypatch.setattr(gp, "_is_duplicate", _dup)
    await gp._handle_digest_response(_msg(10), MagicMock())
    assert calls["n"] == 10  # honest small responses are fully processed


@pytest.mark.asyncio
async def test_non_list_messages_ignored(monkeypatch):
    gp = _gp()
    calls = {"n": 0}

    async def _dup(nonce):
        calls["n"] += 1
        return True

    monkeypatch.setattr(gp, "_is_duplicate", _dup)
    m = MagicMock()
    m.payload = {"data": {"messages": {"not": "a list"}}}
    m.sender_id = "peerabcd1234"
    await gp._handle_digest_response(m, MagicMock())
    assert calls["n"] == 0  # malformed → ignored, no scans


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
