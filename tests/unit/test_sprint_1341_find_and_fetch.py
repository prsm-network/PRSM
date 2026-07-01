"""Sprint 1341 — PRSMClient.find_and_fetch: the one-call find→fetch→verify capstone.

Turns the three proven data-consumer primitives (network-wide topic search sp1339/1340,
provenance-bearing retrieve sp1338) into a single flagship call: topic-search → fetch the top
hit → INDEPENDENTLY re-verify the bytes client-side (sha256 == content_hash, not trusting the
server) → return the content + verifiable creator/provenance.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib

import pytest

from prsm.sdk.client import PRSMClient

_CREATOR = "0x" + "a" * 40
_PROV = "0x" + "cd" * 32


def _client(search_payload, fetch_payload):
    c = PRSMClient(base_url="http://x")

    async def _search(query, **kw):
        return search_payload

    async def _fetch(cid, **kw):
        return fetch_payload

    c.search_content = _search
    c.fetch_content = _fetch
    return c


def _b64(raw):
    return base64.b64encode(raw).decode()


def test_find_and_fetch_verifies_integrity_and_surfaces_provenance():
    raw = b"the nada dataset bytes"
    ch = hashlib.sha256(raw).hexdigest()
    search = {"results": [{
        "cid": "bafy-nada", "filename": "nada.csv", "creator_tier": "high",
        "creator_eth_address": _CREATOR, "provenance_hash": _PROV}], "count": 1}
    fetch = {"status": "success", "data": _b64(raw), "content_hash": ch,
             "creator_eth_address": _CREATOR, "provenance_hash": _PROV,
             "filename": "nada.csv", "size_bytes": len(raw)}
    res = asyncio.run(_client(search, fetch).find_and_fetch("nutrition"))
    assert res["cid"] == "bafy-nada"
    assert res["integrity_verified"] is True          # client re-hash matched
    assert res["creator_eth_address"] == _CREATOR
    assert res["provenance_hash"] == _PROV
    assert res["matched"]["creator_tier"] == "high"


def test_integrity_fails_on_tampered_bytes():
    """The verify is REAL: bytes that don't hash to content_hash → integrity_verified False."""
    ch = hashlib.sha256(b"honest bytes").hexdigest()
    fetch = {"status": "success", "data": _b64(b"EVIL swapped bytes"), "content_hash": ch}
    res = asyncio.run(_client({"results": [{"cid": "c", "filename": "f"}]}, fetch)
                      .find_and_fetch("q"))
    assert res["integrity_verified"] is False


def test_no_search_results_raises():
    with pytest.raises(FileNotFoundError, match="no content found"):
        asyncio.run(_client({"results": []}, {}).find_and_fetch("nothing here"))


def test_top_hit_not_retrievable_raises():
    with pytest.raises(FileNotFoundError, match="not retrievable"):
        asyncio.run(_client({"results": [{"cid": "c", "filename": "f"}]},
                            {"status": "not_found"}).find_and_fetch("q"))


def test_provenance_falls_back_to_search_row_when_fetch_lacks_it():
    raw = b"x"
    ch = hashlib.sha256(raw).hexdigest()
    search = {"results": [{"cid": "c", "filename": "f",
                           "creator_eth_address": "0x" + "b" * 40,
                           "provenance_hash": "0x" + "ef" * 32}]}
    fetch = {"status": "success", "data": _b64(raw), "content_hash": ch}  # no attribution
    res = asyncio.run(_client(search, fetch).find_and_fetch("q"))
    assert res["creator_eth_address"] == "0x" + "b" * 40
    assert res["provenance_hash"] == "0x" + "ef" * 32


def test_missing_content_hash_is_unverified_not_crash():
    fetch = {"status": "success", "data": _b64(b"bytes"), "content_hash": None}
    res = asyncio.run(_client({"results": [{"cid": "c"}]}, fetch).find_and_fetch("q"))
    assert res["integrity_verified"] is False


def test_cli_get_command_registered():
    from prsm.cli import content
    assert "get" in content.commands


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
