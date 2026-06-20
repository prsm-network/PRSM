"""Sprint 1196 — /info advertises whether the operator ACCEPTS requester payment.

operator_address alone is the node's general on-chain identity and can be set
without PRSM_REQUESTER_PAYMENT enabled — in which case a paid /compute/inference
is rejected (402 verifier-not-wired). /info now exposes requester_payment_accepted
(a payment or relayer verifier is wired) so a requester — and the `pay-infer
--dry-run` preflight — can tell BEFORE signing whether payment will be honored.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _info(node):
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    ).get("/info").json()


def test_info_accepted_true_when_payment_verifier_wired():
    node = MagicMock()
    node._payment_verifier = object()  # wired
    node._relayer_verifier = None
    assert _info(node)["requester_payment_accepted"] is True


def test_info_accepted_true_when_only_relayer_verifier_wired():
    node = MagicMock()
    node._payment_verifier = None
    node._relayer_verifier = object()  # relayer path counts as accepting payment
    assert _info(node)["requester_payment_accepted"] is True


def test_info_accepted_false_when_no_verifier():
    node = MagicMock()
    node._payment_verifier = None
    node._relayer_verifier = None
    assert _info(node)["requester_payment_accepted"] is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
