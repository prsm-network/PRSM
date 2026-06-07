"""Sprint 1036 — build_onchain_settlement_client_or_none (on-chain settlement
client instantiation, brick 1).

Assembles a prsm.settlement.client.BatchSettlementClient (ReceiptAccumulator +
Web3SettlementContractClient + this node's provider_address) for on-chain
cross-node settlement. OFF by default; opt-in via PRSM_ONCHAIN_SETTLEMENT.

The settler key (FTNS_WALLET_PRIVATE_KEY) is OPTIONAL: without it the client is
VIEW-ONLY — local accumulation works now; commit/finalize (which need msg.sender)
raise until a funded key is supplied (brick 2 + the key ceremony). When a key IS
present, its eth address MUST equal provider_address, else the eventual
commitBatch would settle funds to the wrong address → refuse (return None).

Fail-open: never raises (it sits on the daemon-start path).
"""
from __future__ import annotations

import types


def _build(**over):
    from prsm.settlement.client_wiring import (
        build_onchain_settlement_client_or_none,
    )
    provider_address = over.pop("provider_address", "0x" + "11" * 20)
    return build_onchain_settlement_client_or_none(
        provider_address=provider_address, env=over.pop("env"),
    )


def _eth_keypair():
    from eth_account import Account
    acct = Account.create()
    return acct.key.hex(), acct.address


def test_off_by_default():
    assert _build(env={}) is None
    assert _build(env={"PRSM_ONCHAIN_SETTLEMENT": "0"}) is None


def test_on_requires_provider_address():
    assert _build(env={"PRSM_ONCHAIN_SETTLEMENT": "1"}, provider_address="") is None


def test_on_view_only_no_key_returns_client():
    """Opted-in with a provider address but NO key → a view-only client
    (accumulation works; commit deferred). Not None."""
    from prsm.settlement.client import BatchSettlementClient
    client = _build(env={"PRSM_ONCHAIN_SETTLEMENT": "true"})
    assert isinstance(client, BatchSettlementClient)
    assert client._provider == "0x" + "11" * 20   # bound identity for accumulate gate


def test_on_with_matching_key_returns_client():
    from prsm.settlement.client import BatchSettlementClient
    key, addr = _eth_keypair()
    client = _build(
        env={"PRSM_ONCHAIN_SETTLEMENT": "yes", "FTNS_WALLET_PRIVATE_KEY": key},
        provider_address=addr,
    )
    assert isinstance(client, BatchSettlementClient)
    assert client._provider == addr


def test_refuses_key_not_controlling_provider_address():
    """Funds-safety: a key whose eth address != provider_address must be refused
    (commitBatch settles to msg.sender == key's address)."""
    key, _addr = _eth_keypair()
    client = _build(
        env={"PRSM_ONCHAIN_SETTLEMENT": "1", "FTNS_WALLET_PRIVATE_KEY": key},
        provider_address="0x" + "22" * 20,   # NOT the key's address
    )
    assert client is None


def test_malformed_key_fails_open_to_none():
    client = _build(
        env={"PRSM_ONCHAIN_SETTLEMENT": "1", "FTNS_WALLET_PRIVATE_KEY": "not-a-key"},
        provider_address="0x" + "11" * 20,
    )
    assert client is None


def test_unresolvable_registry_returns_none(monkeypatch):
    monkeypatch.setattr(
        "prsm.config.networks.resolve_endpoints",
        lambda *a, **k: types.SimpleNamespace(
            rpc_url="https://x", settlement_registry=None,
        ),
    )
    assert _build(env={"PRSM_ONCHAIN_SETTLEMENT": "1"}) is None


def test_key_0x_prefix_normalized():
    """A key without the 0x prefix still matches its address."""
    from prsm.settlement.client import BatchSettlementClient
    key, addr = _eth_keypair()
    bare = key[2:] if key.startswith("0x") else key
    client = _build(
        env={"PRSM_ONCHAIN_SETTLEMENT": "1", "FTNS_WALLET_PRIVATE_KEY": bare},
        provider_address=addr,
    )
    assert isinstance(client, BatchSettlementClient)


def test_bare_provider_address_with_matching_key_normalizes():
    """A 0x-less PRSM_OPERATOR_ADDRESS with the correct key must still match
    (common copy/paste form) — not silently disable settlement (fail-safe but a
    confusing foot-gun the review flagged)."""
    from prsm.settlement.client import BatchSettlementClient
    key, addr = _eth_keypair()
    bare = addr[2:] if addr.startswith("0x") else addr
    client = _build(
        env={"PRSM_ONCHAIN_SETTLEMENT": "1", "FTNS_WALLET_PRIVATE_KEY": key},
        provider_address=bare,
    )
    assert isinstance(client, BatchSettlementClient)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
