"""sp1451 — per-stage committed batches are RETAINED + ANNOUNCED so the observer challenge data
plane (double-spend / invalid-signature / expired scanners) can SEE them.

The per-stage challenge-coverage audit (workflow wqpo3rg5u) found the observer-side mirror of the
sp1450 requester-side gap: the settlement audit engine's fraud scanners read
VerifiedBatchCache.verified_batches(), which is populated by INGESTING receipts fetched via a gossiped
announce CID. The announce step (node.py) advertises every batch in the node's
_settlement_published_batch_store — but the DEDICATED per-stage client was built with
published_batch_store=None (resolve_per_stage_settlement_client), so per-stage committed batches were
never retained, never announced, and the observer's double-spend/invalid-sig/expired scanners were
STRUCTURALLY BLIND to per-stage fraud while single-stage batches were covered — an asymmetric gap.

Fix: give the per-stage client the node's OWN _settlement_published_batch_store (the same store the
single-stage client uses + the announce step reads). Retention (client.py commit tail) + the existing
announce then cover per-stage batches too — observer-challenge parity. Bounded like single-stage: the
store only exists when the audit data plane (PRSM_SETTLEMENT_AUDIT) is enabled, else None (no change).
"""
from __future__ import annotations

from types import SimpleNamespace

import prsm.settlement.client_wiring as cw

_OP = "0x" + "11" * 20


def _capturing_build(captured):
    def fake_build(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(_published_batch_store=kwargs.get("published_batch_store"))
    return fake_build


def test_per_stage_client_receives_the_nodes_published_batch_store(monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    sentinel_store = object()  # stands in for the node's PublishedBatchStore (audit plane ON)
    node = SimpleNamespace(
        _operator_address=_OP,
        _settlement_published_batch_store=sentinel_store,
    )
    captured: dict = {}
    monkeypatch.setattr(cw, "build_onchain_settlement_client_or_none", _capturing_build(captured))

    client = cw.resolve_per_stage_settlement_client(node)
    assert client is not None
    assert captured.get("published_batch_store") is sentinel_store, (
        "per-stage client built WITHOUT the node's published_batch_store — its committed batches are "
        "never retained/announced, so the observer double-spend/invalid-sig/expired scanners are blind "
        "to per-stage fraud")


def test_per_stage_client_store_is_none_when_audit_plane_off(monkeypatch):
    """Parity with single-stage: when the audit data plane is OFF (no _settlement_published_batch_store)
    the per-stage client also gets published_batch_store=None — no behavior change in the default config."""
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    node = SimpleNamespace(_operator_address=_OP)  # no _settlement_published_batch_store attribute
    captured: dict = {}
    monkeypatch.setattr(cw, "build_onchain_settlement_client_or_none", _capturing_build(captured))

    cw.resolve_per_stage_settlement_client(node)
    assert captured.get("published_batch_store") is None
