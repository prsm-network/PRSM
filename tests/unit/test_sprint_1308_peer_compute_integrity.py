"""Sprint 1308 — cross-provider compute-integrity audit.

sp1307 self-audits the node's own committed batches; sp1308 audits OTHER providers:
the peer source iterates FOREIGN committed batches (verified_batch_store), resolves
each producer's receipt-serve endpoint (injected resolver), fetches the foreign §7
receipt (sp1305/1306 fetch_retained_receipt), and yields a foreign WatchUnit so the
existing ChallengeWatcher verifies + dry-runs it uniformly.

Tests pin (a) the env endpoint resolver and (b) the source's correlation + fail-soft,
isolated from real HTTP/crypto via injected http_get + monkeypatched merkle/receipt.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import prsm.settlement.merkle as merkle
import prsm.settlement.receipt_challenge_client as rcc
from prsm.settlement.settlement_audit_wiring import (
    build_env_endpoint_resolver,
    build_peer_compute_integrity_watcher,
)


# ── env endpoint resolver ────────────────────────────────────────────────────

def test_env_resolver_none_when_unset():
    assert build_env_endpoint_resolver({}) is None
    assert build_env_endpoint_resolver({"PRSM_PEER_RECEIPT_ENDPOINTS": "  "}) is None


def test_env_resolver_invalid_json_none():
    assert build_env_endpoint_resolver({"PRSM_PEER_RECEIPT_ENDPOINTS": "{bad"}) is None
    assert build_env_endpoint_resolver({"PRSM_PEER_RECEIPT_ENDPOINTS": "[1,2]"}) is None
    assert build_env_endpoint_resolver({"PRSM_PEER_RECEIPT_ENDPOINTS": "{}"}) is None


def test_env_resolver_maps_case_insensitive():
    r = build_env_endpoint_resolver(
        {"PRSM_PEER_RECEIPT_ENDPOINTS": '{"0xAbC123":"http://prod:8000"}'})
    assert r is not None
    assert r("0xabc123") == "http://prod:8000"   # case-insensitive
    assert r("0xABC123") == "http://prod:8000"
    assert r("0xunknown") is None


# ── peer-audit watcher: None-handling ────────────────────────────────────────

@pytest.mark.parametrize("vbs,res,dry", [
    (None, (lambda p: "u"), object()),
    (object(), None, object()),
    (object(), (lambda p: "u"), None),
])
def test_returns_none_when_any_dep_absent(vbs, res, dry):
    assert build_peer_compute_integrity_watcher(
        verified_batch_store=vbs, endpoint_resolver=res, dry_run_client=dry) is None


# ── peer-audit source correlation ────────────────────────────────────────────

class _Resp:
    def __init__(self, status, body):
        self.status_code, self._b = status, body
    def json(self):
        return self._b


@pytest.fixture
def _patched(monkeypatch):
    monkeypatch.setattr(merkle, "batched_receipt_to_leaf", lambda br: br)
    monkeypatch.setattr(merkle, "hash_leaf", lambda br: (br.tag + b"\x00" * 32)[:32])

    class _FakeIR:
        @staticmethod
        def from_dict(d):
            return SimpleNamespace(settler_node_id="nid", _d=d)
    monkeypatch.setattr(rcc, "InferenceReceipt", _FakeIR)


def _vbs(batches):
    return SimpleNamespace(all_batches=lambda: batches)


async def _units(watcher):
    return [u async for u in watcher._source()]


@pytest.mark.asyncio
async def test_source_fetches_and_builds_foreign_watchunits(_patched):
    br0 = SimpleNamespace(provider_address="0xPROV", tag=b"r0")
    br1 = SimpleNamespace(provider_address="0xPROV", tag=b"r1")
    batch = SimpleNamespace(batch_id=b"BID", receipts=[br0, br1])

    def _get(url, **k):
        leaf = url.rsplit("/", 1)[-1]
        return _Resp(200, {"leaf_hash": leaf, "inference_receipt": {"x": 1},
                           "settler_public_key_b64": "PK", "stage_public_keys": {"n": "k"}})

    w = build_peer_compute_integrity_watcher(
        verified_batch_store=_vbs([batch]),
        endpoint_resolver=lambda p: "http://prod:8000" if p == "0xPROV" else None,
        dry_run_client=object(), http_get=_get)
    units = await _units(w)
    assert len(units) == 2
    u0 = units[0]
    assert u0.batch_id == b"BID"
    assert u0.target_index == 0
    assert u0.batch_receipts == [br0, br1]          # FULL foreign ordered set (for assembler)
    assert u0.settler_public_key_b64 == "PK"
    assert u0.stage_public_keys == {"n": "k"}
    assert u0.inference_receipt.settler_node_id == "nid"


@pytest.mark.asyncio
async def test_source_skips_when_endpoint_unresolved(_patched):
    br = SimpleNamespace(provider_address="0xUNKNOWN", tag=b"r0")
    batch = SimpleNamespace(batch_id=b"BID", receipts=[br])
    w = build_peer_compute_integrity_watcher(
        verified_batch_store=_vbs([batch]),
        endpoint_resolver=lambda p: None,          # no endpoint => skip (not fraud)
        dry_run_client=object(), http_get=lambda *a, **k: _Resp(200, {}))
    assert await _units(w) == []


@pytest.mark.asyncio
async def test_source_skips_on_fetch_error(_patched):
    br = SimpleNamespace(provider_address="0xPROV", tag=b"r0")
    batch = SimpleNamespace(batch_id=b"BID", receipts=[br])

    def _boom(url, **k):
        raise OSError("connection refused")
    w = build_peer_compute_integrity_watcher(
        verified_batch_store=_vbs([batch]),
        endpoint_resolver=lambda p: "http://prod:8000",
        dry_run_client=object(), http_get=_boom)
    assert await _units(w) == []                   # fail-soft: skip, never break the scan


@pytest.mark.asyncio
async def test_source_skips_receipt_without_provider(_patched):
    br = SimpleNamespace(provider_address=None, tag=b"r0")
    batch = SimpleNamespace(batch_id=b"BID", receipts=[br])
    w = build_peer_compute_integrity_watcher(
        verified_batch_store=_vbs([batch]),
        endpoint_resolver=lambda p: "http://prod:8000",
        dry_run_client=object(), http_get=lambda *a, **k: _Resp(200, {}))
    assert await _units(w) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
