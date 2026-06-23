"""Sprint 262 — /compute/inference/quote returns
epsilon_estimated + privacy_budget_remaining.

Pre-fix quote surfaced cost_ftns + tiers only. Callers had to
re-derive the projected ε spend from PrivacyLevel docs and
guess whether they had enough remaining privacy budget. The
/compute/inference pre-flight gate uses the same ε value, so
exposing it via quote closes the "ε exhausted before FTNS"
foot-gun.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node(remaining=10.0, privacy_wired=True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.inference_executor = MagicMock()
    node.inference_executor.supported_models = MagicMock(
        return_value=["mock-llama-3-8b"],
    )
    node.inference_executor.estimate_cost = AsyncMock(
        return_value=Decimal("0.10"),
    )
    if privacy_wired:
        pb = MagicMock()
        pb.remaining = MagicMock(return_value=remaining)
        node.privacy_budget = pb
    else:
        node.privacy_budget = None
    return node


def _client(node):
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_quote_includes_epsilon_for_standard():
    """privacy_tier=standard → ε=8.0 (per PrivacyLevel config)."""
    resp = _client(_node()).post(
        "/compute/inference/quote",
        json={
            "prompt": "hi",
            "model_id": "mock-llama-3-8b",
            "privacy_tier": "standard",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["epsilon_estimated"] == 8.0


def test_quote_epsilon_zero_for_none_tier():
    resp = _client(_node()).post(
        "/compute/inference/quote",
        json={
            "prompt": "hi",
            "model_id": "mock-llama-3-8b",
            "privacy_tier": "none",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["epsilon_estimated"] == 0.0


def test_quote_surfaces_remaining_budget():
    resp = _client(_node(remaining=12.5)).post(
        "/compute/inference/quote",
        json={
            "prompt": "hi",
            "model_id": "mock-llama-3-8b",
            "privacy_tier": "standard",
        },
    )
    body = resp.json()
    assert body["privacy_budget_remaining"] == 12.5


def test_quote_remaining_null_when_unwired():
    resp = _client(_node(privacy_wired=False)).post(
        "/compute/inference/quote",
        json={
            "prompt": "hi",
            "model_id": "mock-llama-3-8b",
            "privacy_tier": "standard",
        },
    )
    body = resp.json()
    assert body["privacy_budget_remaining"] is None


def test_quote_pre_sprint_262_fields_still_present():
    """Sprint 237 contract preserved: cost_ftns, privacy_tier,
    content_tier, model_id all still there."""
    resp = _client(_node()).post(
        "/compute/inference/quote",
        json={
            "prompt": "hi",
            "model_id": "mock-llama-3-8b",
        },
    )
    body = resp.json()
    assert body["model_id"] == "mock-llama-3-8b"
    assert body["cost_ftns"] == "0.10"
    assert body["privacy_tier"] == "none"  # sp1234 — quote default is now none (DP is experimental)
    assert body["content_tier"] == "A"
