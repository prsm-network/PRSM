"""Sprint 963 — surface the sp960 public-bind auth posture in /health/detailed.

sp960 assesses the network-exposure auth posture (non-loopback bind + no
PRSM_NODE_API_KEY → protected money endpoints unauthenticated) ONCE at startup
and emits a log warning. An operator who missed that log line, or who wants to
re-check after a config change without restarting, had no way to query it.

This adds a `security_posture` block to /health/detailed (the operator's existing
subsystem-status surface) reporting the same classification, so the posture is
queryable any time.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node(*, listen_host="0.0.0.0"):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node.config = MagicMock()
    node.config.listen_host = listen_host
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_health_detailed_flags_insecure_public_bind(monkeypatch):
    monkeypatch.delenv("PRSM_NODE_API_KEY", raising=False)
    body = _client(_node(listen_host="0.0.0.0")).get("/health/detailed").json()
    posture = body["security_posture"]
    assert posture["level"] == "insecure"
    assert posture["listen_host"] == "0.0.0.0"
    assert posture["api_key_set"] is False
    assert "PRSM_NODE_API_KEY" in posture["message"]


def test_health_detailed_ok_with_api_key(monkeypatch):
    monkeypatch.setenv("PRSM_NODE_API_KEY", "secret")
    body = _client(_node(listen_host="0.0.0.0")).get("/health/detailed").json()
    posture = body["security_posture"]
    assert posture["level"] == "ok"
    assert posture["api_key_set"] is True


def test_health_detailed_ok_on_loopback(monkeypatch):
    monkeypatch.delenv("PRSM_NODE_API_KEY", raising=False)
    body = _client(_node(listen_host="127.0.0.1")).get("/health/detailed").json()
    assert body["security_posture"]["level"] == "ok"
