"""Sprint 1259 — preflight Wallet-config check honors the explicit-operator-address path.

Found during the live front-door validation: the `prsm node start` preflight "Wallet config"
diagnostic keyed SOLELY on FTNS_WALLET_PRIVATE_KEY, so an operator running the
requester-payment / explicit-address model (PRSM_OPERATOR_ADDRESS set, no wallet key) saw a
misleading "FTNS_WALLET_PRIVATE_KEY not set" WARN — even though resolve_operator_address()
honors PRSM_OPERATOR_ADDRESS FIRST and the node operates fine on it.

The preflight now mirrors resolve_operator_address()'s real precedence: it reports the resolved
operator address (naming the source) regardless of which env supplied it, and only WARNs when
NEITHER source yields an address.
"""
from __future__ import annotations

import pytest

from prsm.cli import (
    PREFLIGHT_PASS,
    PREFLIGHT_WARN,
    _operator_wallet_preflight,
)

# eth_account test vector: private key 0x..01 → this address.
_TEST_PK = "0x0000000000000000000000000000000000000000000000000000000000000001"
_TEST_ADDR = "0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf"

_VARS = ("PRSM_OPERATOR_ADDRESS", "FTNS_WALLET_PRIVATE_KEY")


@pytest.fixture
def clean_env(monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def test_explicit_operator_address_no_wallet_key_passes(clean_env):
    # The regression: explicit address, no wallet key → must PASS, not WARN "not set".
    clean_env.setenv("PRSM_OPERATOR_ADDRESS", "0xC09bAbCdEf0123456789AbCdEf0123456789aBcD")
    r = _operator_wallet_preflight()
    assert r.status == PREFLIGHT_PASS
    assert "Operator address:" in r.details
    assert "PRSM_OPERATOR_ADDRESS" in r.details          # source named
    assert "not set" not in r.details


def test_explicit_wins_over_wallet_key(clean_env):
    clean_env.setenv("PRSM_OPERATOR_ADDRESS", "0xC09bAbCdEf0123456789AbCdEf0123456789aBcD")
    clean_env.setenv("FTNS_WALLET_PRIVATE_KEY", _TEST_PK)
    r = _operator_wallet_preflight()
    assert r.status == PREFLIGHT_PASS
    assert "PRSM_OPERATOR_ADDRESS" in r.details           # source = explicit, not derived
    assert "0xC09bAb" in r.details                          # the EXPLICIT address is shown
    assert _TEST_ADDR[:8] not in r.details                  # NOT the wallet-derived address


def test_wallet_key_derivation_passes(clean_env):
    clean_env.setenv("FTNS_WALLET_PRIVATE_KEY", _TEST_PK)
    r = _operator_wallet_preflight()
    assert r.status == PREFLIGHT_PASS
    assert _TEST_ADDR[:8] in r.details                     # derived address shown
    assert "FTNS_WALLET_PRIVATE_KEY" in r.details          # source named


def test_malformed_wallet_key_warns(clean_env):
    clean_env.setenv("FTNS_WALLET_PRIVATE_KEY", "not-a-valid-key")
    r = _operator_wallet_preflight()
    assert r.status == PREFLIGHT_WARN
    assert "malformed" in r.details.lower()


def test_nothing_configured_warns_mentioning_both_sources(clean_env):
    r = _operator_wallet_preflight()
    assert r.status == PREFLIGHT_WARN
    # remediation must point at BOTH ways to configure an operator address.
    blob = (r.details + " " + r.remediation)
    assert "PRSM_OPERATOR_ADDRESS" in blob
    assert "FTNS_WALLET_PRIVATE_KEY" in blob


def test_helper_is_wired_into_full_preflight(clean_env):
    # The wallet-config check must still be emitted by the full diagnostics pass.
    clean_env.setenv("PRSM_OPERATOR_ADDRESS", "0xC09bAbCdEf0123456789AbCdEf0123456789aBcD")
    from prsm.cli import _node_preflight_diagnostics
    from prsm.node.node import NodeConfig

    checks = _node_preflight_diagnostics(NodeConfig())
    wallet = [c for c in checks if c.name.startswith("Wallet config")]
    assert len(wallet) == 1
    assert wallet[0].status == PREFLIGHT_PASS
    assert "PRSM_OPERATOR_ADDRESS" in wallet[0].details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
