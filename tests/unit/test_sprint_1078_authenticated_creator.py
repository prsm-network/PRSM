"""Sprint 1078 — authenticate the off-chain creator (content-data-plane Gap B, creator leg).

The reputation auto-record (retrieve → CreatorReputationTracker.record_access) keys on
the record's creator_eth_address, which the unauthenticated GOSSIP_CONTENT_ADVERTISE
lane lets a malicious first-advertiser set — poisoning the §14 reputation of a creator
they don't control. The registry's `creator` IS an eth address, so it can authenticate
this directly (same pattern as the sp1077 rate leg, no node-id↔eth binding needed for
the reputation leg). For content with a REGISTERED provenance_hash, the api.py retrieve
path overrides creator_eth_address with the on-chain-authoritative creator before
record_access.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.web3.provenance_registry import ContentRecord
from prsm.node.content_economy import ContentEconomy

REGISTERED_CREATOR = "0x" + "ab" * 20
_PH = "0x" + "cd" * 32


def _econ(client=None):
    e = ContentEconomy.__new__(ContentEconomy)
    e._provenance_client = client
    e._get_provenance_client = lambda: client
    return e


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _client(creator):
    c = MagicMock()
    c.get_content.return_value = ContentRecord(
        creator=creator, royalty_rate_bps=200, registered_at=1, metadata_uri="")
    return c


def test_registered_creator_returned():
    e = _econ(_client(REGISTERED_CREATOR))
    assert _run(e._authenticated_creator({"provenance_hash": _PH})) == REGISTERED_CREATOR


def test_unregistered_returns_none():
    client = MagicMock()
    client.get_content.return_value = None
    assert _run(_econ(client)._authenticated_creator({"provenance_hash": _PH})) is None


def test_no_provenance_hash_returns_none():
    assert _run(_econ(_client(REGISTERED_CREATOR))._authenticated_creator({})) is None


def test_no_client_returns_none():
    assert _run(_econ(None)._authenticated_creator({"provenance_hash": _PH})) is None


def test_malformed_creator_returns_none():
    assert _run(_econ(_client("not-an-address"))._authenticated_creator(
        {"provenance_hash": _PH})) is None
    assert _run(_econ(_client("0x1234"))._authenticated_creator(
        {"provenance_hash": _PH})) is None   # wrong length


def test_rate_and_creator_share_one_rpc():
    """sp1078 — rate + creator derive from the SAME cached registry record (one RPC)."""
    client = _client(REGISTERED_CREATOR)
    e = _econ(client)
    rate = _run(e._authenticated_royalty_rate({"provenance_hash": _PH}))
    creator = _run(e._authenticated_creator({"provenance_hash": _PH}))
    assert rate == pytest.approx(0.02) and creator == REGISTERED_CREATOR
    assert client.get_content.call_count == 1   # shared cache → single lookup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
