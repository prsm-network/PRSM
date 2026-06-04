"""Sprint 244 — verify creator_eth_address propagates through:
   ContentUploadRequest → upload_text() → UploadedContent
   ContentIndex.advertise → ContentRecord → /content/{cid}.
"""
from __future__ import annotations

import pytest

from prsm.node.content_uploader import UploadedContent


def test_uploaded_content_has_field():
    uc = UploadedContent(
        content_id="c1",
        filename="x.txt",
        size_bytes=10,
        content_hash="00" * 32,
        creator_id="creator-a",
        creator_eth_address="0x" + "a" * 40,
    )
    assert uc.creator_eth_address == "0x" + "a" * 40


def test_uploaded_content_field_optional():
    uc = UploadedContent(
        content_id="c1",
        filename="x.txt",
        size_bytes=10,
        content_hash="00" * 32,
        creator_id="creator-a",
    )
    assert uc.creator_eth_address is None


def test_content_record_has_field():
    from prsm.node.content_index import ContentRecord
    r = ContentRecord(
        cid="c1",
        filename="x.txt",
        size_bytes=10,
        content_hash="00" * 32,
        creator_id="creator-a",
        creator_eth_address="0x" + "b" * 40,
    )
    assert r.creator_eth_address == "0x" + "b" * 40


@pytest.mark.asyncio
async def test_content_index_ingests_field_from_advertise():
    """ContentIndex._on_content_advertise wires the gossip
    payload through to ContentRecord. New field carried through."""
    from unittest.mock import MagicMock
    from prsm.node.content_index import ContentIndex

    idx = ContentIndex(gossip=MagicMock())
    await idx._on_content_advertise(
        subtype="content.advertise",
        data={
            "cid": "c1",
            "provider_id": "peer-a",
            "filename": "x.txt",
            "size_bytes": 10,
            "content_hash": "00" * 32,
            "creator_id": "creator-a",
            "creator_eth_address": "0x" + "c" * 40,
        },
        origin="peer-a",
    )
    record = idx.lookup("c1")
    assert record is not None
    assert record.creator_eth_address == "0x" + "c" * 40


@pytest.mark.asyncio
async def test_content_index_handles_missing_field():
    """Pre-sprint-244 peers won't include the field. Ingest gracefully."""
    from unittest.mock import MagicMock
    from prsm.node.content_index import ContentIndex

    idx = ContentIndex(gossip=MagicMock())
    await idx._on_content_advertise(
        subtype="content.advertise",
        data={
            "cid": "c2",
            "provider_id": "peer-a",
            "filename": "x.txt",
            "size_bytes": 10,
            "content_hash": "00" * 32,
            "creator_id": "creator-a",
            # creator_eth_address omitted
        },
        origin="peer-a",
    )
    record = idx.lookup("c2")
    assert record is not None
    assert record.creator_eth_address is None


@pytest.mark.asyncio
async def test_creator_eth_address_backfilled_on_later_full_advertise():
    """sp995 (fix B) — a record first created from a MINIMAL replica/announce
    advertise (which omits creator_eth_address) is REPAIRED when the uploader's
    full advertise (carrying the field) arrives later. Without the backfill branch
    the field stayed None forever, so the §14 stake gate keyed on None and wrongly
    demoted a stake-eligible HIGH creator at activation."""
    from unittest.mock import MagicMock
    from prsm.node.content_index import ContentIndex

    idx = ContentIndex(gossip=MagicMock())
    # 1) minimal advertise (e.g. a replica/announce) — no creator_eth_address.
    await idx._on_content_advertise(
        subtype="content.advertise",
        data={
            "cid": "c3", "provider_id": "replica-peer",
            "filename": "x.txt", "size_bytes": 10,
            "content_hash": "00" * 32, "creator_id": "creator-a",
        },
        origin="replica-peer",
    )
    assert idx.lookup("c3").creator_eth_address is None
    # 2) full advertise from the uploader — carries creator_eth_address.
    eth = "0x" + "d" * 40
    await idx._on_content_advertise(
        subtype="content.advertise",
        data={
            "cid": "c3", "provider_id": "uploader-peer",
            "filename": "x.txt", "size_bytes": 10,
            "content_hash": "00" * 32, "creator_id": "creator-a",
            "creator_eth_address": eth,
        },
        origin="uploader-peer",
    )
    assert idx.lookup("c3").creator_eth_address == eth  # backfilled


@pytest.mark.asyncio
async def test_creator_eth_address_backfill_first_non_none_wins():
    """A later minimal advertise (no field) must NOT clobber an address already
    recorded — first-non-None-wins, like the other backfilled optional fields."""
    from unittest.mock import MagicMock
    from prsm.node.content_index import ContentIndex

    idx = ContentIndex(gossip=MagicMock())
    eth = "0x" + "e" * 40
    await idx._on_content_advertise(
        subtype="content.advertise",
        data={
            "cid": "c4", "provider_id": "p", "filename": "x.txt",
            "size_bytes": 1, "content_hash": "00" * 32,
            "creator_id": "a", "creator_eth_address": eth,
        },
        origin="p",
    )
    await idx._on_content_advertise(
        subtype="content.advertise",
        data={
            "cid": "c4", "provider_id": "p2", "filename": "x.txt",
            "size_bytes": 1, "content_hash": "00" * 32, "creator_id": "a",
        },
        origin="p2",
    )
    assert idx.lookup("c4").creator_eth_address == eth
