"""Sprint 978 — creator-stake on-chain read backend + decision-A wiring.

Decision A: the §14 creator-stake gate keys on the creator's ETH ADDRESS. This
wires the real on-chain backend (CreatorStakeRegistry.creatorStakeOf) so the gate
has teeth, and removes the toothless in-memory free-stake bypass (stake/slash no
longer fall back to in-memory when a real backend is present; the read fail-closes
to 0).
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.creator_stake_registry_backend import (
    CreatorStakeRegistryBackend,
    CreatorStakeServerActionError,
)
from prsm.marketplace.creator_stake_client import (
    CreatorStakeClient,
    apply_stake_gate,
    MIN_HIGH_TIER_STAKE_WEI,
)
from prsm.marketplace.creator_reputation import TIER_HIGH, TIER_MEDIUM

ADDR = "0x" + "1" * 40


def _backend_with(balance, *, raises=False):
    class _Call:
        def call(self):
            if raises:
                raise RuntimeError("rpc down")
            return balance

    class _Fns:
        def creatorStakeOf(self, addr):
            return _Call()

    class _Contract:
        functions = _Fns()

    return CreatorStakeRegistryBackend(
        ADDR, "https://rpc.example", web3_factory=lambda: _Contract()
    )


# ── backend reads ──────────────────────────────────────────────────────────


def test_backend_balance_of_reads_onchain_stake():
    b = _backend_with(5000 * 10**18)
    assert b.balance_of(ADDR) == 5000 * 10**18


def test_backend_balance_of_invalid_address_is_zero():
    b = _backend_with(123)
    assert b.balance_of("not-an-eth-address") == 0
    assert b.balance_of("") == 0


def test_backend_balance_of_rpc_error_fails_closed_to_zero():
    b = _backend_with(0, raises=True)
    assert b.balance_of(ADDR) == 0


def test_backend_stake_and_slash_reject_as_server_actions():
    b = _backend_with(0)
    with pytest.raises(CreatorStakeServerActionError):
        b.stake(ADDR, 100)
    with pytest.raises(CreatorStakeServerActionError):
        b.slash(ADDR, 100, "spam")


# ── client: no in-memory fallback when a real backend is wired ──────────────


def test_client_stake_does_not_fall_back_to_in_memory_with_backend():
    b = _backend_with(0)
    client = CreatorStakeClient(backend=b)
    with pytest.raises(CreatorStakeServerActionError):
        client.stake(ADDR, MIN_HIGH_TIER_STAKE_WEI)
    # The toothless bypass is closed: the in-memory mirror was NOT credited.
    assert client._balances.get(ADDR, 0) == 0


def test_client_reads_balance_from_backend():
    b = _backend_with(MIN_HIGH_TIER_STAKE_WEI)
    client = CreatorStakeClient(backend=b)
    assert client.stake_balance(ADDR) == MIN_HIGH_TIER_STAKE_WEI
    assert client.is_high_tier_eligible(ADDR) is True


def test_in_memory_scaffold_still_works_when_uncommissioned():
    """backend=None (dev/uncommissioned) keeps the in-memory mirror."""
    client = CreatorStakeClient()  # no backend
    client.stake(ADDR, MIN_HIGH_TIER_STAKE_WEI)
    assert client.is_high_tier_eligible(ADDR) is True


# ── from_env constructs the real backend when commissioned ──────────────────


def test_from_env_wires_onchain_backend_when_addr_and_rpc_set(monkeypatch):
    monkeypatch.setenv("CREATOR_STAKE_REGISTRY_ADDRESS", ADDR)
    monkeypatch.setenv("BASE_RPC_URL", "https://rpc.example")
    client = CreatorStakeClient.from_env()
    assert isinstance(client._backend, CreatorStakeRegistryBackend)


def test_from_env_no_backend_when_unset(monkeypatch):
    monkeypatch.delenv("CREATOR_STAKE_REGISTRY_ADDRESS", raising=False)
    monkeypatch.delenv("BASE_RPC_URL", raising=False)
    client = CreatorStakeClient.from_env()
    assert client._backend is None  # in-memory dev scaffold


# ── apply_stake_gate keys on the eth address (decision A) ───────────────────


def test_gate_demotes_high_when_no_eth_address():
    client = CreatorStakeClient(backend=_backend_with(MIN_HIGH_TIER_STAKE_WEI))
    # None / empty eth address can't have bonded stake → demote.
    assert apply_stake_gate(TIER_HIGH, None, client) == TIER_MEDIUM
    assert apply_stake_gate(TIER_HIGH, "", client) == TIER_MEDIUM


def test_gate_keeps_high_when_eth_address_has_stake():
    client = CreatorStakeClient(backend=_backend_with(MIN_HIGH_TIER_STAKE_WEI))
    assert apply_stake_gate(TIER_HIGH, ADDR, client) == TIER_HIGH


def test_gate_demotes_high_when_eth_address_understaked():
    client = CreatorStakeClient(backend=_backend_with(MIN_HIGH_TIER_STAKE_WEI - 1))
    assert apply_stake_gate(TIER_HIGH, ADDR, client) == TIER_MEDIUM
