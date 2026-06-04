"""Sprint 1007 — bound the per-CID provider set (memory-DoS + availability-eclipse).

The P2P-substrate integrity hunt (workflow wbu7u2ftm, finding 6, HIGH) confirmed
that ContentRecord.providers grows without bound: each GOSSIP_CONTENT_ADVERTISE
adds `data.get("provider_id", origin)` to the set, and provider_id is
attacker-chosen (only the gossip ORIGIN is sp934-authenticated, not the claimed
provider_id). A peer can therefore advertise one CID under thousands of distinct
provider_ids — growing the set unbounded (memory DoS) and flooding the provider
list with attacker entries (availability-eclipse: the retriever's provider
selection is dominated by attacker-controlled ids).

Fix: cap record.providers at PRSM_MAX_PROVIDERS_PER_CID (default 64). An
already-listed provider always refreshes; a new provider_id beyond the cap is
dropped. A realistic CID has a handful of replicas, so the cap never bites
legitimate content.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.content_index import (
    MAX_PROVIDERS_PER_CID_DEFAULT,
    ContentIndex,
    _max_providers_per_cid,
)


def _idx():
    return ContentIndex(gossip=MagicMock())


async def _advertise(idx, cid, provider_id):
    await idx._on_content_advertise(
        "content.advertise", {"cid": cid, "provider_id": provider_id}, provider_id
    )


@pytest.mark.asyncio
async def test_provider_set_capped(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_PROVIDERS_PER_CID", "5")
    assert _max_providers_per_cid() == 5
    idx = _idx()
    for i in range(50):
        await _advertise(idx, "c1", f"prov{i}")
    assert len(idx.lookup("c1").providers) <= 5


@pytest.mark.asyncio
async def test_existing_provider_still_refreshes_at_cap(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_PROVIDERS_PER_CID", "3")
    idx = _idx()
    for pid in ("a", "b", "c"):
        await _advertise(idx, "c1", pid)
    assert idx.lookup("c1").providers == {"a", "b", "c"}
    # A 4th distinct provider is dropped (cap), but re-advertising an existing
    # one is a no-op that does not error or evict.
    await _advertise(idx, "c1", "d")
    await _advertise(idx, "c1", "a")
    provs = idx.lookup("c1").providers
    assert len(provs) <= 3
    assert "a" in provs  # existing provider preserved


@pytest.mark.asyncio
async def test_legitimate_replica_count_not_capped(monkeypatch):
    monkeypatch.delenv("PRSM_MAX_PROVIDERS_PER_CID", raising=False)
    assert _max_providers_per_cid() == MAX_PROVIDERS_PER_CID_DEFAULT
    idx = _idx()
    for i in range(8):  # a handful of real replicas
        await _advertise(idx, "c1", f"replica{i}")
    assert len(idx.lookup("c1").providers) == 8


def test_max_providers_default_sane():
    assert MAX_PROVIDERS_PER_CID_DEFAULT >= 16
    assert _max_providers_per_cid() > 0


@pytest.mark.asyncio
async def test_bad_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_PROVIDERS_PER_CID", "not-an-int")
    assert _max_providers_per_cid() == MAX_PROVIDERS_PER_CID_DEFAULT
