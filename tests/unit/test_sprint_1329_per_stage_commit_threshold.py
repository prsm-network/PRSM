"""Sprint 1329 — per-stage commit uses a dedicated count_threshold=1 client (small shares commit).

The 2026-06-30 testnet GO surfaced this: a 0.14 FTNS per-stage share never crossed the shared
single-stage client's value threshold, so it silently never settled (we committed via a manual
low-threshold client). Per-stage share-batches are committed INDIVIDUALLY (one per node per job),
so the per-stage client uses count_threshold=1 — a single staged share is immediately ready.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from prsm.settlement.accumulator import AccumulatorConfig, ReceiptAccumulator
from prsm.settlement import client_wiring


# ── the core fix: count_threshold=1 readies a single small share ──────────────

def test_default_config_count_threshold_is_high():
    """Why the bug existed: the default accumulator needs 1000 receipts (or the value/time
    threshold) before a batch is ready — a lone small share never triggers it."""
    assert AccumulatorConfig().count_threshold == 1000


def test_per_stage_config_is_count_threshold_1():
    cfg = AccumulatorConfig(count_threshold=1)
    assert cfg.count_threshold == 1
    # constructs a real accumulator without error (validation passes)
    assert ReceiptAccumulator(cfg) is not None


# ── resolver: gated, count_threshold=1, distinct state file, cached ───────────

def test_resolver_none_when_settlement_off(monkeypatch):
    monkeypatch.delenv("PRSM_ONCHAIN_SETTLEMENT", raising=False)
    node = SimpleNamespace(_operator_address="0x" + "ab" * 20,
                           _onchain_per_stage_settlement_client=None)
    assert client_wiring.resolve_per_stage_settlement_client(node) is None


def test_resolver_returns_cached(monkeypatch):
    sentinel = MagicMock()
    node = SimpleNamespace(_onchain_per_stage_settlement_client=sentinel)
    assert client_wiring.resolve_per_stage_settlement_client(node) is sentinel


def test_resolver_builds_with_count_threshold_1_and_distinct_state(monkeypatch, tmp_path):
    captured = {}

    def _fake_build(*, provider_address, env=None, accumulator_config=None, state_store=None,
                    published_batch_store=None):
        captured["provider"] = provider_address
        captured["acc_cfg"] = accumulator_config
        captured["state_store"] = state_store
        return MagicMock(name="per_stage_client")

    monkeypatch.setattr(client_wiring, "build_onchain_settlement_client_or_none", _fake_build)
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT_STATE_FILE", str(tmp_path / "ps_state.json"))
    node = SimpleNamespace(_operator_address="0x" + "cd" * 20,
                           _onchain_per_stage_settlement_client=None)

    client = client_wiring.resolve_per_stage_settlement_client(node)
    assert client is not None
    # count_threshold=1 → a single small share commits immediately
    assert captured["acc_cfg"] is not None and captured["acc_cfg"].count_threshold == 1
    # a DISTINCT state store (its own file, not the single-stage default)
    assert captured["state_store"] is not None
    assert "ps_state.json" in str(captured["state_store"].path)
    assert captured["provider"] == "0x" + "cd" * 20
    # cached on the node
    assert node._onchain_per_stage_settlement_client is client
    assert client_wiring.resolve_per_stage_settlement_client(node) is client


def test_builder_accepts_accumulator_config_and_state_store_kwargs():
    """The builder must accept the new kwargs (default None → unchanged) — pin the signature so a
    future refactor can't drop them."""
    import inspect
    sig = inspect.signature(client_wiring.build_onchain_settlement_client_or_none)
    assert "accumulator_config" in sig.parameters
    assert "state_store" in sig.parameters
    assert sig.parameters["accumulator_config"].default is None
    assert sig.parameters["state_store"].default is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
