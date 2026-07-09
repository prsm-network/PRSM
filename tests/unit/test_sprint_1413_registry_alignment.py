"""Sprint 1413 — the ContentAccessVerifier's registry must BE the network's canonical registry.

Base Sepolia had two live ProvenanceRegistryV2 contracts. ``contract_addresses.json`` advertised
``provenance_registry_v2 = 0xe75F0c24..`` while the deployed ContentAccessVerifier
(``0x99264Bca..``) was constructed against a different, fresher one (``0xCBe377Ae..``).

That is not a bookkeeping nit. ``ContentAccessVerifier.payForAccess`` does::

    (address creator, ) = registry.getCreatorAndRate(contentHash);
    if (creator == address(0)) revert ContentNotRegistered(contentHash);

So a publisher following the shipped config registered provenance into a registry the verifier
never reads, and every buyer's ``payForAccess`` reverted. The Tier B/C paid-content path — publish,
buy, decrypt — was broken on testnet, which is exactly the network whose job is to rehearse mainnet.
Base mainnet was always consistent (one registry serving both roles).

Evidence gathered before repointing (read-only event census, 2026-07-09): each registry held exactly
3 ``ContentRegistered`` events, all minted within ~60 blocks of its own deploy. The superseded one
carried ``ipfs://placeholder`` / ``ipfs://legacy`` fixtures and had seen zero activity in 64 days;
the CAV's carried the sp1364 ``smoke://cav`` money-path smoke. No real content was orphaned.

These tests pin the invariant that would have caught it: wherever a network declares BOTH a
content_access_verifier_registry and a provenance_registry_v2, they must be the same address.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from prsm.config.networks import TESTNET

_ADDRESSES = pathlib.Path(__file__).resolve().parents[2] / "prsm/deployments/contract_addresses.json"

SEPOLIA_V2 = "0xcbe377ae09fdd5f63875aa5313c65a3c8c073731"
SEPOLIA_SUPERSEDED_V2 = "0xe75f0c24a9e63b63456d170d99f03ab7fc3450a7"


def _addresses() -> dict:
    return json.loads(_ADDRESSES.read_text())


def _networks_declaring_both():
    out = []
    for net, cfg in _addresses().items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("content_access_verifier_registry") and cfg.get("provenance_registry_v2"):
            out.append(net)
    return out


def test_at_least_one_network_declares_both():
    """Guards the guard: if the keys get renamed, the parametrized test below would vacuously pass."""
    assert _networks_declaring_both(), "no network declares both keys — has the schema changed?"


@pytest.mark.parametrize("network", _networks_declaring_both())
def test_verifier_registry_is_the_canonical_registry(network):
    """THE invariant. payForAccess reverts ContentNotRegistered when the creator lookup misses, so a
    publisher registering into `provenance_registry_v2` must be registering into the very contract
    the verifier reads."""
    cfg = _addresses()[network]
    canonical = cfg["provenance_registry_v2"].lower()
    verifier_reads = cfg["content_access_verifier_registry"].lower()
    assert canonical == verifier_reads, (
        f"{network}: publishers register into provenance_registry_v2={canonical} but the deployed "
        f"ContentAccessVerifier reads {verifier_reads}. Every payForAccess will revert "
        f"ContentNotRegistered."
    )


def test_sepolia_points_at_the_registry_the_verifier_reads():
    cfg = _addresses()["base-sepolia"]
    assert cfg["provenance_registry_v2"].lower() == SEPOLIA_V2
    assert cfg["_provenance_registry_v2_superseded"].lower() == SEPOLIA_SUPERSEDED_V2


def test_networks_py_testnet_matches_the_deploy_manifest():
    """node.py resolves the V2 client from `endpoints.provenance_registry_v2`, NOT from
    contract_addresses.json — leaving TESTNET unset made a Sepolia node skip the V2 client entirely.
    The two sources of truth must agree."""
    assert TESTNET.provenance_registry_v2 is not None, (
        "TESTNET.provenance_registry_v2 is None → a Sepolia node builds no V2 client, so paid "
        "content can never be registered where the verifier reads it"
    )
    assert TESTNET.provenance_registry_v2.lower() == \
        _addresses()["base-sepolia"]["provenance_registry_v2"].lower()


def test_mainnet_was_always_consistent():
    """Regression anchor: mainnet's verifier and canonical registry always agreed, and must keep
    agreeing. This is the shape testnet now mirrors."""
    cfg = _addresses()["base"]
    assert cfg["provenance_registry_v2"].lower() == cfg["content_access_verifier_registry"].lower()
    assert cfg["provenance_registry_v2"].lower() == "0xe0cedda354f99526c7fbb9b9651e12adb2180dbf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
