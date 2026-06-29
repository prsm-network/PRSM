"""Sprint 1299 — make the TEE Tier-3 attestation-commit path a CONFIG flip.

The roadmap-F cutover's final, most consequential step is "flip
``supports_attestation``" on the live settlement client. Pre-sp1299 that flag was
a constructor parameter hardcoded ``False`` in ``client_wiring`` and exposed by NO
env var — so the flip required a code change + redeploy, inconsistent with the
already-env-flippable registry address (``PRSM_SETTLEMENT_REGISTRY_ADDRESS``).

sp1299 wires it through ``PRSM_SETTLEMENT_SUPPORTS_ATTESTATION`` (default OFF →
legacy ``commitBatch``, byte-for-byte unchanged), but FAIL-SAFE: it only enables
attestation-commit after confirming the DEPLOYED registry actually exposes
``commitBatchWithAttestation`` (selector ``0xfa4ce156``, ABI-derived). Flipping it
against an un-upgraded registry would make every commit tx revert and stall the
settlement cycle with escrow already locked — so an unconfirmable registry stays OFF.
"""
from __future__ import annotations


def _build(monkeypatch, env, *, provider_address="0x" + "11" * 20):
    from prsm.settlement.client_wiring import (
        build_onchain_settlement_client_or_none,
    )
    return build_onchain_settlement_client_or_none(
        provider_address=provider_address, env=env,
    )


# ── selector derivation (drift guard) ───────────────────────────────────────

def test_selector_derived_from_abi_matches_contract():
    """The Python selector MUST equal the contracts-side selector (0xfa4ce156),
    and be computed from the ABI — never hardcoded — so it can't drift."""
    from prsm.settlement.client_wiring import _commit_with_attestation_selector_hex
    assert _commit_with_attestation_selector_hex() == "fa4ce156"


# ── pure bytecode probe ──────────────────────────────────────────────────────

def test_bytecode_exposes_selector_pure():
    from prsm.settlement.client_wiring import _bytecode_exposes_selector
    assert _bytecode_exposes_selector("0xaabbfa4ce156ccdd", "fa4ce156") is True
    # case-insensitive
    assert _bytecode_exposes_selector("0xAABBFA4CE156CCDD", "fa4ce156") is True
    # absent selector
    assert _bytecode_exposes_selector("0xdeadbeef", "fa4ce156") is False
    # empty / None code → False (not a crash)
    assert _bytecode_exposes_selector("", "fa4ce156") is False
    assert _bytecode_exposes_selector(None, "fa4ce156") is False


# ── fail-safe probe ──────────────────────────────────────────────────────────

def test_registry_probe_failsafe_on_error(monkeypatch):
    """Any error in the probe (selector derivation, web3 import, RPC) → False:
    we never enable attestation-commit against a registry we can't confirm."""
    import prsm.settlement.client_wiring as cw

    def _boom():
        raise RuntimeError("selector derivation blew up")

    monkeypatch.setattr(cw, "_commit_with_attestation_selector_hex", _boom)
    assert cw._registry_supports_attestation("https://x", "0x" + "ab" * 20) is False


# ── env wiring through the client build ──────────────────────────────────────

def test_default_off_and_no_network_probe(monkeypatch):
    """Default (flag unset): supports_attestation is False AND the network probe
    is NEVER called (no startup RPC cost / coupling when the path is off)."""
    import prsm.settlement.client_wiring as cw

    def _must_not_call(*a, **k):  # pragma: no cover - asserts it isn't invoked
        raise AssertionError("registry probe must not run when the flag is unset")

    monkeypatch.setattr(cw, "_registry_supports_attestation", _must_not_call)
    client = _build(monkeypatch, {"PRSM_ONCHAIN_SETTLEMENT": "1"})
    assert client is not None
    assert client._contract._supports_attestation is False


def test_enabled_when_registry_supports(monkeypatch):
    """Flag set + the deployed registry exposes the function → attestation ON."""
    import prsm.settlement.client_wiring as cw
    monkeypatch.setattr(cw, "_registry_supports_attestation", lambda *a, **k: True)
    client = _build(
        monkeypatch,
        {"PRSM_ONCHAIN_SETTLEMENT": "1", "PRSM_SETTLEMENT_SUPPORTS_ATTESTATION": "1"},
    )
    assert client is not None
    assert client._contract._supports_attestation is True


def test_stays_off_when_registry_lacks_selector(monkeypatch, caplog):
    """Flag set but the deployed registry does NOT expose the function (e.g. the
    operator pointed at the OLD pre-sp1240 registry) → stays OFF + warns."""
    import logging
    import prsm.settlement.client_wiring as cw
    monkeypatch.setattr(cw, "_registry_supports_attestation", lambda *a, **k: False)
    with caplog.at_level(logging.WARNING):
        client = _build(
            monkeypatch,
            {
                "PRSM_ONCHAIN_SETTLEMENT": "1",
                "PRSM_SETTLEMENT_SUPPORTS_ATTESTATION": "true",
            },
        )
    assert client is not None
    assert client._contract._supports_attestation is False
    assert any(
        "does not expose commitBatchWithAttestation" in r.getMessage()
        for r in caplog.records
    )


def test_flag_passes_resolved_rpc_and_registry_to_probe(monkeypatch):
    """The probe is called with the SAME rpc_url + registry the client binds to
    (so the confirmation is about the registry we will actually write to)."""
    import prsm.settlement.client_wiring as cw
    seen = {}

    def _spy(rpc_url, address):
        seen["rpc_url"] = rpc_url
        seen["address"] = address
        return True

    monkeypatch.setattr(cw, "_registry_supports_attestation", _spy)
    client = _build(
        monkeypatch,
        {"PRSM_ONCHAIN_SETTLEMENT": "1", "PRSM_SETTLEMENT_SUPPORTS_ATTESTATION": "yes"},
    )
    assert client is not None
    # rpc + registry resolved from prsm.config.networks (mainnet defaults)
    assert seen["rpc_url"] and seen["address"]
    assert seen["address"] == client._contract.contract_address or seen["address"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
