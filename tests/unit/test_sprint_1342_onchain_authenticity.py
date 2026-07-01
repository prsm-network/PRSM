"""Sprint 1342 — full ON-CHAIN authenticity in find_and_fetch.

sp1341 gave the capstone an INTEGRITY check (bytes match content_hash). This adds the trustless
AUTHENTICITY check: resolve the content's provenance_hash against the on-chain ProvenanceRegistry
and confirm the claimed creator IS the on-chain-registered creator — so an untrusted serving node
can't lie about who made the data. The client reads the chain itself (never asks the node).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib

import pytest

from prsm.sdk.client import PRSMClient

_CLAIMED = "0x" + "a" * 40
_PROV = "0x" + "cd" * 32


def _client(*, claimed=_CLAIMED, prov=_PROV):
    raw = b"the dataset bytes"
    ch = hashlib.sha256(raw).hexdigest()
    c = PRSMClient(base_url="http://x")

    async def _search(query, **kw):
        return {"results": [{"cid": "bafy-c", "filename": "nada.csv", "creator_tier": "high",
                             "creator_eth_address": claimed, "provenance_hash": prov}]}

    async def _fetch(cid, **kw):
        return {"status": "success", "data": base64.b64encode(raw).decode(),
                "content_hash": ch, "creator_eth_address": claimed,
                "provenance_hash": prov, "filename": "nada.csv", "size_bytes": len(raw)}

    c.search_content = _search
    c.fetch_content = _fetch
    return c


def _run(c, **kw):
    return asyncio.run(c.find_and_fetch("nutrition", verify_provenance=True, **kw))


def test_authentic_when_onchain_creator_matches_claim():
    res = _run(_client(), provenance_verifier=lambda ph: _CLAIMED)
    assert res["authenticity_verified"] is True
    assert res["registered_creator"] == _CLAIMED
    assert res["authenticity_detail"] == "verified"


def test_fails_when_provider_lies_about_creator():
    """THE trustless property: content claims creator A, on-chain it's B → authenticity FAILS."""
    res = _run(_client(claimed="0x" + "a" * 40),
               provenance_verifier=lambda ph: "0x" + "b" * 40)
    assert res["authenticity_verified"] is False
    assert res["registered_creator"] == "0x" + "b" * 40
    assert "does NOT match" in res["authenticity_detail"]


def test_fails_when_provenance_unregistered():
    res = _run(_client(), provenance_verifier=lambda ph: None)
    assert res["authenticity_verified"] is False
    assert "not registered" in res["authenticity_detail"]


def test_fails_when_no_provenance_hash():
    res = _run(_client(prov=None), provenance_verifier=lambda ph: _CLAIMED)
    assert res["authenticity_verified"] is False
    assert "no provenance_hash" in res["authenticity_detail"]


def test_no_verifier_configured_is_honest_false(monkeypatch):
    c = _client()
    monkeypatch.setattr(c, "_default_provenance_verifier", lambda: None)
    res = asyncio.run(c.find_and_fetch("q", verify_provenance=True))
    assert res["authenticity_verified"] is False
    assert "no on-chain verifier" in res["authenticity_detail"]


def test_lookup_exception_is_not_authentic():
    def _boom(ph):
        raise RuntimeError("rpc down")
    res = _run(_client(), provenance_verifier=_boom)
    assert res["authenticity_verified"] is False
    assert "lookup failed" in res["authenticity_detail"]


def test_async_verifier_supported():
    async def _averify(ph):
        return _CLAIMED
    res = _run(_client(), provenance_verifier=_averify)
    assert res["authenticity_verified"] is True


def test_case_insensitive_address_match():
    res = _run(_client(claimed="0x" + "A" * 40),
               provenance_verifier=lambda ph: "0x" + "a" * 40)
    assert res["authenticity_verified"] is True


def test_default_off_adds_no_authenticity_fields():
    """Opt-in: without verify_provenance the lightweight path is unchanged (no chain read)."""
    res = asyncio.run(_client().find_and_fetch("q"))
    assert "authenticity_verified" not in res
    assert res["integrity_verified"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
