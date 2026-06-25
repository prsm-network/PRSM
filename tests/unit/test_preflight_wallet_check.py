"""Preflight diagnostic for FTNS_WALLET_PRIVATE_KEY (sprint 126).

New check surfaces wallet config status pre-startup so operators
can verify the right key is wired before the node begins on-chain
operations.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.cli import _node_preflight_diagnostics


# Test private key (well-known test value, never used on mainnet)
_TEST_PK = (
    "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
)


def _config():
    cfg = MagicMock()
    cfg.config_path = MagicMock()
    cfg.config_path.exists = MagicMock(return_value=False)
    cfg.api_port = 0  # arbitrary; bind probe uses ephemeral
    cfg.bootstrap_nodes = []
    cfg.p2p_port = 9001
    return cfg


def _wallet_check(checks):
    return next(
        (c for c in checks if c.name == "Wallet config (optional)"),
        None,
    )


class TestWalletPreflight:
    def test_no_source_warns(self):
        # sp1259: with NEITHER FTNS_WALLET_PRIVATE_KEY nor PRSM_OPERATOR_ADDRESS, the
        # check WARNs and its remediation must point at BOTH ways to configure one.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FTNS_WALLET_PRIVATE_KEY", None)
            os.environ.pop("PRSM_OPERATOR_ADDRESS", None)
            checks = _node_preflight_diagnostics(_config())
        wc = _wallet_check(checks)
        assert wc is not None
        assert wc.status == "WARN"
        assert "not configured" in wc.details
        assert "PRSM_OPERATOR_ADDRESS" in wc.remediation
        assert "FTNS_WALLET_PRIVATE_KEY" in wc.remediation

    def test_explicit_operator_address_passes_without_wallet_key(self):
        # sp1259 regression: the requester-payment / read-only operator model sets only
        # PRSM_OPERATOR_ADDRESS — it must PASS, not WARN "FTNS_WALLET_PRIVATE_KEY not set".
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FTNS_WALLET_PRIVATE_KEY", None)
            os.environ["PRSM_OPERATOR_ADDRESS"] = (
                "0xC09bAbCdEf0123456789AbCdEf0123456789aBcD"
            )
            checks = _node_preflight_diagnostics(_config())
        wc = _wallet_check(checks)
        assert wc is not None
        assert wc.status == "PASS"
        assert "Operator address:" in wc.details
        assert "PRSM_OPERATOR_ADDRESS" in wc.details

    def test_valid_pk_shows_address(self):
        with patch.dict(
            os.environ, {"FTNS_WALLET_PRIVATE_KEY": _TEST_PK},
            clear=False,
        ):
            os.environ.pop("PRSM_OPERATOR_ADDRESS", None)  # exercise the derived path
            checks = _node_preflight_diagnostics(_config())
        wc = _wallet_check(checks)
        assert wc is not None
        assert wc.status == "PASS"
        # Truncated address format
        assert "Operator address:" in wc.details
        assert "..." in wc.details

    def test_malformed_pk_warns(self):
        with patch.dict(
            os.environ,
            {"FTNS_WALLET_PRIVATE_KEY": "definitely-not-a-key"},
            clear=False,
        ):
            os.environ.pop("PRSM_OPERATOR_ADDRESS", None)  # no explicit fallback
            checks = _node_preflight_diagnostics(_config())
        wc = _wallet_check(checks)
        assert wc is not None
        assert wc.status == "WARN"
