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


# ── sp995 (fix D): blip-tolerant caching / retry / last-known-good ──────────


def _toggle_backend(seq, clock):
    """Backend whose creatorStakeOf().call() yields outcomes from `seq` (an int
    returns it; the sentinel RAISE raises) and counts calls. `clock` is injected."""
    state = {"i": 0, "calls": 0}

    class _Call:
        def call(self):
            state["calls"] += 1
            v = seq[min(state["i"], len(seq) - 1)]
            state["i"] += 1
            if v == "RAISE":
                raise RuntimeError("rpc down")
            return v

    class _Fns:
        def creatorStakeOf(self, addr):
            return _Call()

    class _Contract:
        functions = _Fns()

    b = CreatorStakeRegistryBackend(
        ADDR, "https://rpc.example",
        web3_factory=lambda: _Contract(), clock=lambda: clock["t"],
    )
    return b, state


def test_balance_of_caches_within_ttl(monkeypatch):
    monkeypatch.setenv("PRSM_CREATOR_STAKE_CACHE_TTL_S", "60")
    monkeypatch.setenv("PRSM_CREATOR_STAKE_READ_ATTEMPTS", "3")
    clock = {"t": 1000.0}
    b, state = _toggle_backend([5 * 10**18], clock)
    assert b.balance_of(ADDR) == 5 * 10**18
    assert b.balance_of(ADDR) == 5 * 10**18  # within TTL → served from cache
    assert state["calls"] == 1  # only ONE chain call (per-request/cross-request dedup)


def test_balance_of_serves_last_known_good_on_blip(monkeypatch):
    """A transient RPC failure AFTER a good read serves the last-known-good
    value (within stale-grace), NOT a false 0 that would demote a staked creator."""
    monkeypatch.setenv("PRSM_CREATOR_STAKE_CACHE_TTL_S", "60")
    monkeypatch.setenv("PRSM_CREATOR_STAKE_STALE_GRACE_S", "600")
    monkeypatch.setenv("PRSM_CREATOR_STAKE_READ_ATTEMPTS", "2")
    clock = {"t": 1000.0}
    b, state = _toggle_backend([5 * 10**18, "RAISE", "RAISE"], clock)
    assert b.balance_of(ADDR) == 5 * 10**18  # first read: cached
    clock["t"] = 1100.0  # past TTL (60s) but within stale-grace (600s)
    # read now fails (RAISE x2 = exhausts 2 attempts) → last-known-good, NOT 0
    assert b.balance_of(ADDR) == 5 * 10**18


def test_balance_of_fail_closed_when_no_cache(monkeypatch):
    monkeypatch.setenv("PRSM_CREATOR_STAKE_READ_ATTEMPTS", "2")
    clock = {"t": 1000.0}
    b, _ = _toggle_backend(["RAISE"], clock)
    assert b.balance_of(ADDR) == 0  # no prior good read → fail-closed to 0


def test_balance_of_fail_closed_after_stale_grace(monkeypatch):
    """A SUSTAINED outage (cache older than stale-grace) fail-closes to 0 —
    preserving the documented fail-closed contract."""
    monkeypatch.setenv("PRSM_CREATOR_STAKE_CACHE_TTL_S", "60")
    monkeypatch.setenv("PRSM_CREATOR_STAKE_STALE_GRACE_S", "600")
    monkeypatch.setenv("PRSM_CREATOR_STAKE_READ_ATTEMPTS", "1")
    clock = {"t": 1000.0}
    b, _ = _toggle_backend([5 * 10**18, "RAISE"], clock)
    assert b.balance_of(ADDR) == 5 * 10**18
    clock["t"] = 2000.0  # 1000s later → beyond the 600s stale-grace
    assert b.balance_of(ADDR) == 0  # sustained outage → fail-closed


def test_balance_of_retries_transient_then_succeeds(monkeypatch):
    """A single dropped request self-heals via retry (no false 0 on a cold cache)."""
    monkeypatch.setenv("PRSM_CREATOR_STAKE_READ_ATTEMPTS", "3")
    clock = {"t": 1000.0}
    b, state = _toggle_backend(["RAISE", "RAISE", 7 * 10**18], clock)
    assert b.balance_of(ADDR) == 7 * 10**18
    assert state["calls"] == 3  # retried through the two transient failures


def test_balance_of_genuine_zero_is_cached_not_treated_as_failure(monkeypatch):
    """An on-chain 0 (creator never bonded) is a real value, cached + returned —
    NOT confused with a read failure."""
    monkeypatch.setenv("PRSM_CREATOR_STAKE_CACHE_TTL_S", "60")
    clock = {"t": 1000.0}
    b, state = _toggle_backend([0, 0], clock)
    assert b.balance_of(ADDR) == 0
    assert b.balance_of(ADDR) == 0  # served from cache (genuine 0)
    assert state["calls"] == 1


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
    monkeypatch.delenv("PRSM_CREATOR_STAKE_REGISTRY_ADDRESS", raising=False)
    monkeypatch.delenv("BASE_RPC_URL", raising=False)
    monkeypatch.delenv("PRSM_NETWORK", raising=False)
    client = CreatorStakeClient.from_env()
    assert client._backend is None  # in-memory dev scaffold


# ── sp981 — from_env resolves the address through the canonical registry ─────


def test_from_env_uses_networks_default_rpc_when_only_address_set(monkeypatch):
    """Post-ceremony the operator records ONLY the address (in networks.py, the
    canonical home, or via the env override). The RPC then resolves to the
    network default (Base mainnet) through resolve_endpoints — so the gate goes
    live without ALSO requiring a separate BASE_RPC_URL. This is the unified
    resolution every other deployed contract already uses."""
    monkeypatch.setenv("CREATOR_STAKE_REGISTRY_ADDRESS", ADDR)
    monkeypatch.delenv("BASE_RPC_URL", raising=False)
    monkeypatch.delenv("PRSM_BASE_RPC_URL", raising=False)
    monkeypatch.delenv("PRSM_NETWORK", raising=False)
    client = CreatorStakeClient.from_env()
    assert isinstance(client._backend, CreatorStakeRegistryBackend)
    assert client.is_commissioned() is True
    # The resolved RPC is the Base mainnet default (no explicit RPC env set).
    assert client._rpc_url == "https://mainnet.base.org"


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
