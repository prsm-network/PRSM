"""Sprint 1077 — authenticate the off-chain royalty rate (content-data-plane Gap B).

The GOSSIP_CONTENT_ADVERTISE lane is unauthenticated + first-writer-wins, so a
malicious first advertiser sets a record's royalty_rate to an attacker-chosen value
(sp1004 bounds it to a sane range but a within-bounds value still skews off-chain
credit — pool-split weighting + the per-access payment amount). The authoritative
rate lives in the on-chain ProvenanceRegistry (the same source the on-chain royalty
leg dispatches on, sp996). This binds the off-chain rate to it: for content with a
REGISTERED provenance_hash, process_content_access uses the registered rate, not the
forgeable advertise copy. Unregistered / no-client → the bounded advertise value
stands (no authenticated source exists for it).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.web3.provenance_registry import ContentRecord
from prsm.node.content_economy import ContentEconomy


def _econ(client=None):
    e = ContentEconomy.__new__(ContentEconomy)   # bypass heavy __init__
    e._provenance_client = client
    e._get_provenance_client = lambda: client
    return e


def _run(coro):
    import asyncio
    return asyncio.run(coro)


_PH = "0x" + "ab" * 32   # 32-byte provenance hash


def test_registered_rate_overrides_advertise_rate():
    client = MagicMock()
    client.get_content.return_value = ContentRecord(
        creator="0x" + "11" * 20, royalty_rate_bps=500,  # 5% = registered
        registered_at=1, metadata_uri="")
    e = _econ(client)
    # advertise lane claims 50%; registry says 5%
    rate = _run(e._authenticated_royalty_rate(
        {"provenance_hash": _PH, "royalty_rate": 0.50}))
    assert rate == pytest.approx(0.05)


def test_unregistered_returns_none_keeps_advertise():
    client = MagicMock()
    client.get_content.return_value = None   # not registered on-chain
    e = _econ(client)
    assert _run(e._authenticated_royalty_rate(
        {"provenance_hash": _PH, "royalty_rate": 0.50})) is None


def test_no_provenance_hash_returns_none():
    e = _econ(MagicMock())
    assert _run(e._authenticated_royalty_rate({"royalty_rate": 0.50})) is None


def test_no_client_returns_none():
    e = _econ(None)
    assert _run(e._authenticated_royalty_rate(
        {"provenance_hash": _PH, "royalty_rate": 0.50})) is None


def test_malformed_provenance_hash_returns_none():
    e = _econ(MagicMock())
    assert _run(e._authenticated_royalty_rate(
        {"provenance_hash": "not-hex"})) is None
    assert _run(e._authenticated_royalty_rate(
        {"provenance_hash": "0x" + "ab" * 10})) is None   # wrong length (10 bytes)


def test_lookup_error_returns_none():
    client = MagicMock()
    client.get_content.side_effect = RuntimeError("rpc down")
    e = _econ(client)
    assert _run(e._authenticated_royalty_rate({"provenance_hash": _PH})) is None


def test_result_is_cached_no_second_rpc():
    client = MagicMock()
    client.get_content.return_value = ContentRecord(
        creator="0x" + "11" * 20, royalty_rate_bps=300, registered_at=1, metadata_uri="")
    e = _econ(client)
    a = _run(e._authenticated_royalty_rate({"provenance_hash": _PH}))
    b = _run(e._authenticated_royalty_rate({"provenance_hash": _PH}))
    assert a == b == pytest.approx(0.03)
    assert client.get_content.call_count == 1   # cached on the second call


def test_out_of_range_bps_rejected():
    client = MagicMock()
    client.get_content.return_value = ContentRecord(
        creator="0x" + "11" * 20, royalty_rate_bps=99999, registered_at=1, metadata_uri="")
    e = _econ(client)
    assert _run(e._authenticated_royalty_rate({"provenance_hash": _PH})) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
