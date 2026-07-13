"""sp1447 — the per-stage FINALIZE cycle (run_per_stage_finalize_cycle).

The big-model paid multi-stage settlement had a commit driver (sp1322 run_per_stage_commit_cycle)
but NO finalize driver for the per-stage client's committed share-batches — so a self-committed
share would stay PENDING (escrow locked) forever and the payee would never be paid on-chain after
the challenge window. This adds the finalize half, wired into the node settlement poll loop right
after the commit cycle, with the same gate + fail-soft discipline.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from prsm.settlement.client_wiring import run_per_stage_finalize_cycle


def _run(coro):
    return asyncio.run(coro)


class _FakeClient:
    """Records finalize + reconcile-finalized calls on the per-stage client."""

    def __init__(self, *, finalize_raises=False):
        self.finalized = 0
        self.reconciled = 0
        self._finalize_raises = finalize_raises

    async def finalize_ready_batches(self):
        if self._finalize_raises:
            raise RuntimeError("private_key required")  # the default VIEW-ONLY client
        self.finalized += 1
        return ["fb1"]

    async def reconcile_finalized(self):
        self.reconciled += 1
        return None


def test_finalize_skipped_when_disabled(monkeypatch):
    monkeypatch.delenv("PRSM_MULTISTAGE_SETTLEMENT", raising=False)
    client = _FakeClient()
    node = SimpleNamespace(_onchain_per_stage_settlement_client=client)
    r = _run(run_per_stage_finalize_cycle(node))
    assert r["per_stage_finalize"] == "skipped:disabled"
    assert client.finalized == 0, "finalize ran while the multi-stage path is gated OFF"


def test_finalize_skipped_when_no_client(monkeypatch):
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    node = SimpleNamespace(_onchain_per_stage_settlement_client=None)  # no client resolvable
    r = _run(run_per_stage_finalize_cycle(node))
    assert r["per_stage_finalize"] == "skipped:no-client"


def test_finalize_drives_finalize_and_reconcile(monkeypatch):
    """The money shot: enabled + a per-stage client → finalize_ready_batches (releases the escrow to
    the payee once the challenge window elapsed) + reconcile_finalized both run."""
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    client = _FakeClient()
    node = SimpleNamespace(_onchain_per_stage_settlement_client=client)
    r = _run(run_per_stage_finalize_cycle(node))
    assert client.finalized == 1, "the committed share-batch was never finalized (escrow stranded)"
    assert client.reconciled == 1
    assert r["per_stage_finalize"] == "['fb1']"
    assert r["per_stage_reconcile_finalized"] == "ok"


def test_finalize_view_only_client_is_inert_and_never_raises(monkeypatch):
    """The default VIEW-ONLY client (no funded per-stage settler key) raises private_key required on
    finalize — the cycle records it as an error, isolates the phase, and NEVER raises."""
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    client = _FakeClient(finalize_raises=True)
    node = SimpleNamespace(_onchain_per_stage_settlement_client=client)
    r = _run(run_per_stage_finalize_cycle(node))  # must not raise
    assert r["per_stage_finalize"].startswith("error:")
    assert client.reconciled == 1, "phase isolation broke — reconcile skipped after finalize error"
