"""T6 (post-2026-05-07) — `prsm.config.networks.resolve_endpoints` unit tests.

Covers the env-var resolution contract that on-chain clients
(`OnChainFTNSLedger`, `ProvenanceRegistryClient`,
`PublisherKeyAnchorClient`, …) depend on. The bug this regression
suite locks in: pre-T6, those clients defaulted to mainnet RPC + the
mainnet FTNS address even when `PRSM_NETWORK=testnet` was set, because
the env-var alias surface (`FTNS_TOKEN_ADDRESS` vs `FTNS_CONTRACT_ADDRESS`,
`BASE_SEPOLIA_RPC_URL` vs `BASE_RPC_URL`) was inconsistent across
modules. `resolve_endpoints()` is the single point of truth.
"""
from __future__ import annotations

import os
import pytest

from prsm.config.networks import (
    DEFAULT_NETWORK,
    MAINNET,
    TESTNET,
    ResolvedEndpoints,
    resolve_endpoints,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every PRSM-/network-related env var so each test gets a
    deterministic starting state."""
    for k in [
        "PRSM_NETWORK",
        "FTNS_TOKEN_ADDRESS",
        "FTNS_CONTRACT_ADDRESS",
        "BASE_RPC_URL",
        "BASE_SEPOLIA_RPC_URL",
        "PRSM_BASE_RPC_URL",
        "PRSM_PROVENANCE_REGISTRY_ADDRESS",
        "PRSM_ROYALTY_DISTRIBUTOR_ADDRESS",
        "PRSM_FOUNDATION_SAFE",
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS",
        "PRSM_SETTLEMENT_REGISTRY_ADDRESS",
        "PRSM_ESCROW_POOL_ADDRESS",
        "PRSM_STAKE_BOND_ADDRESS",
        "PRSM_EMISSION_CONTROLLER_ADDRESS",
        "PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS",
        "PRSM_STORAGE_SLASHING_ADDRESS",
        "PRSM_KEY_DISTRIBUTION_ADDRESS",
        "CREATOR_STAKE_REGISTRY_ADDRESS",
        "PRSM_CREATOR_STAKE_REGISTRY_ADDRESS",
    ]:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_default_network_is_mainnet(clean_env):
    e = resolve_endpoints()
    assert e.network_name == DEFAULT_NETWORK == "mainnet"
    assert e.chain_id == 8453
    assert e.rpc_url == "https://mainnet.base.org"
    assert e.ftns_token == MAINNET.ftns_token


def test_explicit_network_arg_overrides_env(clean_env):
    clean_env.setenv("PRSM_NETWORK", "testnet")
    # Explicit arg should win over the env var.
    e = resolve_endpoints("mainnet")
    assert e.network_name == "mainnet"
    assert e.chain_id == 8453


def test_prsm_network_testnet_picks_sepolia(clean_env):
    clean_env.setenv("PRSM_NETWORK", "testnet")
    e = resolve_endpoints()
    assert e.network_name == "testnet"
    assert e.chain_id == 84532
    assert e.rpc_url == "https://sepolia.base.org"
    assert e.ftns_token == TESTNET.ftns_token
    assert e.settlement_registry == TESTNET.settlement_registry


def test_base_sepolia_rpc_url_overrides_default_on_testnet(clean_env):
    clean_env.setenv("PRSM_NETWORK", "testnet")
    clean_env.setenv("BASE_SEPOLIA_RPC_URL", "https://my-private-sepolia.example.com")
    e = resolve_endpoints()
    assert e.rpc_url == "https://my-private-sepolia.example.com"


def test_base_rpc_url_overrides_default_on_mainnet(clean_env):
    clean_env.setenv("BASE_RPC_URL", "https://my-private-mainnet.example.com")
    e = resolve_endpoints()
    assert e.network_name == "mainnet"
    assert e.rpc_url == "https://my-private-mainnet.example.com"


def test_prsm_base_rpc_url_wins_on_either_network(clean_env):
    """`PRSM_BASE_RPC_URL` is the network-agnostic operator override."""
    clean_env.setenv("PRSM_NETWORK", "testnet")
    clean_env.setenv("PRSM_BASE_RPC_URL", "https://alchemy-pinned.example.com")
    clean_env.setenv("BASE_SEPOLIA_RPC_URL", "https://default-sepolia.example.com")
    e = resolve_endpoints()
    assert e.rpc_url == "https://alchemy-pinned.example.com"


def test_ftns_token_address_overrides_per_network_default(clean_env):
    clean_env.setenv("PRSM_NETWORK", "testnet")
    clean_env.setenv("FTNS_TOKEN_ADDRESS", "0x1111111111111111111111111111111111111111")
    e = resolve_endpoints()
    assert e.ftns_token == "0x1111111111111111111111111111111111111111"


def test_ftns_contract_address_legacy_alias_works(clean_env):
    """Backwards compatibility: `OnChainFTNSLedger` operators previously
    used `FTNS_CONTRACT_ADDRESS`. Keep that alias working."""
    clean_env.setenv("FTNS_CONTRACT_ADDRESS", "0x2222222222222222222222222222222222222222")
    e = resolve_endpoints()
    assert e.ftns_token == "0x2222222222222222222222222222222222222222"


def test_ftns_token_address_wins_over_legacy_alias(clean_env):
    clean_env.setenv("FTNS_TOKEN_ADDRESS", "0x3333333333333333333333333333333333333333")
    clean_env.setenv("FTNS_CONTRACT_ADDRESS", "0x4444444444444444444444444444444444444444")
    e = resolve_endpoints()
    assert e.ftns_token == "0x3333333333333333333333333333333333333333"


def test_provenance_registry_override(clean_env):
    clean_env.setenv("PRSM_NETWORK", "testnet")
    # Testnet has provenance_registry=None; an explicit override should
    # still flow through.
    clean_env.setenv(
        "PRSM_PROVENANCE_REGISTRY_ADDRESS",
        "0x5555555555555555555555555555555555555555",
    )
    e = resolve_endpoints()
    assert e.provenance_registry == "0x5555555555555555555555555555555555555555"


def test_testnet_provenance_is_none_by_default(clean_env):
    """Document the post-T1 reality: provenance/royalty are NOT on
    Base Sepolia; resolver returns None unless overridden."""
    clean_env.setenv("PRSM_NETWORK", "testnet")
    e = resolve_endpoints()
    assert e.provenance_registry is None
    assert e.royalty_distributor is None


def test_unknown_network_raises_keyerror(clean_env):
    clean_env.setenv("PRSM_NETWORK", "purple-frog")
    with pytest.raises(KeyError, match="unknown network"):
        resolve_endpoints()


# ── sp981 — creator-stake registry joins the canonical address registry ──────


def test_creator_stake_registry_none_by_default(clean_env):
    """Not yet deployed on any network — the resolver returns None until the
    commissioning ceremony records the address (the canonical home is networks.py,
    consistent with every other deployed contract)."""
    e = resolve_endpoints()
    assert e.creator_stake_registry is None
    assert MAINNET.creator_stake_registry is None
    assert TESTNET.creator_stake_registry is None


def test_creator_stake_registry_env_override_flows_through(clean_env):
    """Post-ceremony an operator may pin the address via the existing
    CREATOR_STAKE_REGISTRY_ADDRESS env (the name from_env + the deploy script
    already use), overriding the per-network default."""
    clean_env.setenv(
        "CREATOR_STAKE_REGISTRY_ADDRESS",
        "0x6666666666666666666666666666666666666666",
    )
    e = resolve_endpoints()
    assert e.creator_stake_registry == "0x6666666666666666666666666666666666666666"


def test_empty_env_value_treated_as_unset(clean_env):
    """`os.getenv` returns empty string for `export FOO=` syntax. The
    resolver should treat that as unset, not as an explicit empty
    address."""
    clean_env.setenv("PRSM_NETWORK", "")  # empty
    clean_env.setenv("FTNS_TOKEN_ADDRESS", "")  # empty
    e = resolve_endpoints()
    assert e.network_name == "mainnet"
    assert e.ftns_token == MAINNET.ftns_token


def test_resolved_endpoints_is_frozen(clean_env):
    """ResolvedEndpoints is a frozen dataclass — accidental mutation
    in a downstream consumer should raise."""
    e = resolve_endpoints()
    with pytest.raises(Exception):  # FrozenInstanceError on dataclass
        e.rpc_url = "https://hacked.example.com"  # type: ignore[misc]


def test_all_audit_bundle_addresses_resolve_on_testnet(clean_env):
    """Sanity-check that the post-T1 testnet config exposes the full
    audit-bundle + emission + storage surface."""
    clean_env.setenv("PRSM_NETWORK", "testnet")
    e = resolve_endpoints()
    for field in [
        "ftns_token",
        "settlement_registry",
        "escrow_pool",
        "stake_bond",
        "emission_controller",
        "compensation_distributor",
        "storage_slashing",
        "key_distribution",
        "foundation_safe",
    ]:
        assert getattr(e, field) is not None, f"testnet.{field} unexpectedly None"
        assert getattr(e, field).startswith("0x")
