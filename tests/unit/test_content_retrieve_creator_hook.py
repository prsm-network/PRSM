"""Sprint 288 — /content/retrieve auto-records creator access.

When a content piece is successfully retrieved, the API
handler records the access against the creator's reputation
tracker. Best-effort: tracker exceptions must not break the
primary retrieve surface.

Signals to record:
  - creator_id:    from ContentRecord.creator_eth_address
                    (already propagated via sprint-243 arc)
  - purchaser_id:  the operator's connected address
                    (operator's local node view; per-node
                    no-gossip contract)
  - content_id:    the CID itself
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from prsm.marketplace.creator_reputation import (
    CreatorReputationTracker,
)
from prsm.node.api import create_api_app


class _FakeContentProvider:
    """Returns a known small payload for any CID. Tracks
    `request_content` calls."""

    def __init__(self):
        self.requests = []

    def get_stats(self):
        return {"providers_attempted": 0}

    async def request_content(
        self, cid, timeout=30.0, verify_hash=True,
    ):
        self.requests.append(cid)
        return b"hello world"


class _FakeContentIndex:
    """Returns a ContentRecord-shaped object with the
    creator_eth_address surface."""

    def __init__(self, creator_addr=None):
        self._creator = creator_addr

    def lookup(self, cid):
        rec = MagicMock()
        rec.content_hash = "sha256-deadbeef"
        rec.filename = "file.bin"
        rec.creator_eth_address = self._creator
        return rec


def _client(
    tracker=None, creator_addr="0xcreator",
    operator_addr="0xoperator",
):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = MagicMock()
    node.ftns_ledger._connected_address = operator_addr
    node._creator_reputation_tracker = tracker
    node._content_filter_store = None
    node.content_provider = _FakeContentProvider()
    node.content_index = _FakeContentIndex(creator_addr)
    # Sprint 494 added a fallback in the retrieve hook: when the
    # content_index yields no creator_eth_address, it reads the LOCAL
    # uploaded_content record. With a bare MagicMock node,
    # `uploaded_content.get(cid)` returns a truthy mock whose
    # `.creator_eth_address` is also truthy — which silently defeats
    # the "creator unknown" cases (and any other test relying on the
    # index being the only creator source). Use a real empty dict so
    # the fallback finds nothing and creator resolution reflects only
    # what the test wired into content_index.
    node.content_uploader.uploaded_content = {}
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_retrieve_records_access_on_tracker():
    t = CreatorReputationTracker()
    resp = _client(tracker=t).get(
        "/content/retrieve/bafy-abc",
    )
    assert resp.status_code == 200
    assert t.access_count("0xcreator") == 1


def test_retrieve_records_purchaser_as_operator_address():
    t = CreatorReputationTracker()
    _client(
        tracker=t,
        operator_addr="0xoperator",
    ).get("/content/retrieve/bafy-abc")
    e = t.get_entry("0xcreator")
    assert "0xoperator" in e.purchaser_counts


def test_retrieve_records_content_id_as_cid():
    """The content_id signal lets the tracker tell apart
    "same purchaser, same piece N times" (single repeat) from
    "same purchaser, N distinct pieces" (true repeat
    purchaser). Verify the CID flows through."""
    t = CreatorReputationTracker()
    cli = _client(tracker=t)
    # Same operator retrieves 3 distinct pieces from same
    # creator
    cli.get("/content/retrieve/bafy-piece1")
    cli.get("/content/retrieve/bafy-piece2")
    cli.get("/content/retrieve/bafy-piece3")
    assert t.access_count("0xcreator") == 3
    # All from one purchaser
    assert t.distinct_purchasers("0xcreator") == 1


def test_retrieve_skips_recording_when_creator_unknown():
    """No creator_eth_address on the ContentRecord — nothing
    to record. Retrieve still succeeds."""
    t = CreatorReputationTracker()
    resp = _client(
        tracker=t, creator_addr=None,
    ).get("/content/retrieve/bafy-abc")
    assert resp.status_code == 200
    assert t.access_count("") == 0
    assert t.known_creators() == []


def test_retrieve_skips_recording_when_tracker_unwired():
    """No tracker → handler still works (per-node tracker is
    optional)."""
    resp = _client(tracker=None).get(
        "/content/retrieve/bafy-abc",
    )
    assert resp.status_code == 200


def test_retrieve_tracker_exception_does_not_break_primary():
    """Tracker raises → retrieve still returns the content.
    Defense-in-depth: telemetry failures never deny primary
    service."""
    class BoomTracker:
        def record_access(self, **kwargs):
            raise RuntimeError("tracker explosion")
    resp = _client(tracker=BoomTracker()).get(
        "/content/retrieve/bafy-abc",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"


def test_retrieve_skips_recording_when_no_operator_address():
    """No operator address available (e.g., ftns_ledger
    unwired). Skip recording rather than recording an empty
    purchaser_id."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None  # no operator address
    t = CreatorReputationTracker()
    node._creator_reputation_tracker = t
    node._content_filter_store = None
    node.content_provider = _FakeContentProvider()
    node.content_index = _FakeContentIndex("0xcreator")
    cli = TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )
    resp = cli.get("/content/retrieve/bafy-abc")
    assert resp.status_code == 200
    # No record without an operator identity
    assert t.access_count("0xcreator") == 0


def test_retrieve_overrides_forged_creator_with_registered():
    """sp1078 (content-data-plane Gap B, creator leg) — when the content has a
    REGISTERED provenance_hash, the forgeable advertise-lane creator_eth_address is
    overridden by the on-chain-authoritative creator before reputation record_access.
    So a malicious advertiser can't poison the §14 reputation of a creator it doesn't
    control."""
    from unittest.mock import AsyncMock
    t = CreatorReputationTracker()
    forged = "0x" + "f" * 40       # attacker's advertised creator
    authentic = "0x" + "a" * 40    # the on-chain-registered creator

    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = MagicMock()
    node.ftns_ledger._connected_address = "0xoperator"
    node._creator_reputation_tracker = t
    node._content_filter_store = None
    node.content_provider = _FakeContentProvider()

    rec = MagicMock()
    rec.content_hash = "h"
    rec.filename = "f"
    rec.creator_eth_address = forged
    rec.provenance_hash = "0x" + "cd" * 32   # registered on-chain
    idx = MagicMock()
    idx.lookup = lambda cid: rec
    node.content_index = idx
    node.content_uploader.uploaded_content = {}

    econ = MagicMock()
    econ._authenticated_creator = AsyncMock(return_value=authentic)
    node.content_economy = econ

    cli = TestClient(create_api_app(node, enable_security=False),
                     raise_server_exceptions=False)
    resp = cli.get("/content/retrieve/bafy-abc")
    assert resp.status_code == 200
    assert t.access_count(authentic) == 1   # credited to the AUTHENTIC creator
    assert t.access_count(forged) == 0      # NOT the forged advertise value


def test_retrieve_not_recorded_on_not_found():
    """Content not found → no access to record."""
    class _NoneProvider:
        def get_stats(self):
            return {}
        async def request_content(self, cid, **kwargs):
            return None
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = MagicMock()
    node.ftns_ledger._connected_address = "0xoperator"
    t = CreatorReputationTracker()
    node._creator_reputation_tracker = t
    node._content_filter_store = None
    node.content_provider = _NoneProvider()
    node.content_index = _FakeContentIndex("0xcreator")
    cli = TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )
    cli.get("/content/retrieve/bafy-abc")
    assert t.access_count("0xcreator") == 0
