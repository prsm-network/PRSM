"""Sprint 1183 — wire a real read-only on-chain FTNS balance into the production wallet
API (kill the hardcoded $0 mock).

Day-one-live blocker #1 (from the readiness audit): after binding a wallet,
GET /api/v1/auth/wallet/balance always returned {ftns:0, usd:0.00} because
wallet_api_wiring injected _ZeroBalanceLookup(). This adds:
  - ReadOnlyFTNSBalanceReader (prsm/economy/web3/ftns_balance_reader.py): a no-key ERC-20
    balanceOf + decimals reader.
  - OnChainBalanceLookup (wallet_api.py): the BalanceLookup Protocol impl — wei→whole-FTNS
    Decimal, short TTL cache, FAIL-SOFT (returns last-good, then zero, never raises).
  - build_balance_lookup_or_zero (wallet_api_wiring): builds the on-chain lookup from the
    resolved network (token+rpc), falling back to _ZeroBalanceLookup when unresolved.
"""
from __future__ import annotations

from decimal import Decimal

import pytest


# ── OnChainBalanceLookup: conversion + cache + fail-soft (fake reader, no web3) ──────

class _FakeReader:
    """Duck-typed reader: balance_wei(addr)->int + .decimals. Counts calls + can raise."""
    def __init__(self, *, wei_by_addr=None, decimals=18, raises=False):
        self._wei = wei_by_addr or {}
        self.decimals = decimals
        self.raises = raises
        self.calls = 0

    def balance_wei(self, address: str) -> int:
        self.calls += 1
        if self.raises:
            raise RuntimeError("rpc down")
        return int(self._wei.get(address, 0))


def _lookup(reader, **kw):
    from prsm.interface.api.wallet_api import OnChainBalanceLookup
    return OnChainBalanceLookup(reader, **kw)


def test_converts_wei_to_whole_ftns():
    r = _FakeReader(wei_by_addr={"0xAbc": 1_500_000_000_000_000_000}, decimals=18)
    bal = _lookup(r).get_ftns_balance("0xAbc")
    assert bal == Decimal("1.5")


def test_respects_token_decimals_other_than_18():
    r = _FakeReader(wei_by_addr={"0xAbc": 2_500_000}, decimals=6)  # 2.5 @ 6dp
    assert _lookup(r).get_ftns_balance("0xAbc") == Decimal("2.5")


def test_zero_balance_is_zero_not_error():
    r = _FakeReader(wei_by_addr={}, decimals=18)
    assert _lookup(r).get_ftns_balance("0xUnknown") == Decimal("0")


def test_ttl_cache_avoids_re_reading_within_window():
    t = {"now": 1000.0}
    r = _FakeReader(wei_by_addr={"0xAbc": 10**18})
    lk = _lookup(r, cache_ttl_s=15.0, _now=lambda: t["now"])
    assert lk.get_ftns_balance("0xAbc") == Decimal("1")
    assert lk.get_ftns_balance("0xAbc") == Decimal("1")  # within TTL
    assert r.calls == 1


def test_cache_expires_after_ttl():
    t = {"now": 1000.0}
    r = _FakeReader(wei_by_addr={"0xAbc": 10**18})
    lk = _lookup(r, cache_ttl_s=15.0, _now=lambda: t["now"])
    lk.get_ftns_balance("0xAbc")
    t["now"] += 16.0
    lk.get_ftns_balance("0xAbc")
    assert r.calls == 2


def test_per_address_cache_isolation():
    r = _FakeReader(wei_by_addr={"0xA": 10**18, "0xB": 2 * 10**18})
    lk = _lookup(r)
    assert lk.get_ftns_balance("0xA") == Decimal("1")
    assert lk.get_ftns_balance("0xB") == Decimal("2")


def test_fail_soft_returns_last_good_then_never_raises():
    t = {"now": 1000.0}
    r = _FakeReader(wei_by_addr={"0xAbc": 3 * 10**18})
    lk = _lookup(r, cache_ttl_s=1.0, _now=lambda: t["now"])
    assert lk.get_ftns_balance("0xAbc") == Decimal("3")  # primes last-good
    r.raises = True
    t["now"] += 5.0  # cache expired → re-read raises → fail-soft to last-good
    assert lk.get_ftns_balance("0xAbc") == Decimal("3")


def test_fail_soft_returns_zero_when_no_prior_value():
    r = _FakeReader(raises=True)
    assert _lookup(r).get_ftns_balance("0xAbc") == Decimal("0")  # never raises


# ── ReadOnlyFTNSBalanceReader: web3 balanceOf + decimals (injected fake contract) ────

class _FakeFn:
    def __init__(self, val):
        self._val = val
    def call(self):
        return self._val

class _FakeContract:
    def __init__(self, *, bal=0, decimals=18):
        self._bal = bal
        self._decimals = decimals
        self.balanceof_calls = 0
        self.decimals_calls = 0
        class _F:
            def balanceOf(_s, a):
                self.balanceof_calls += 1
                return _FakeFn(self._bal)
            def decimals(_s):
                self.decimals_calls += 1
                return _FakeFn(self._decimals)
        self.functions = _F()


def test_reader_balance_wei_and_lazy_decimals():
    from prsm.economy.web3.ftns_balance_reader import ReadOnlyFTNSBalanceReader
    fc = _FakeContract(bal=7 * 10**18, decimals=18)
    r = ReadOnlyFTNSBalanceReader.__new__(ReadOnlyFTNSBalanceReader)
    r._contract = fc
    r._decimals = None
    assert r.balance_wei("0x" + "ab" * 20) == 7 * 10**18
    assert r.decimals == 18
    assert r.decimals == 18  # cached — not re-read
    assert fc.decimals_calls == 1


# ── wiring: real lookup when token resolves, fallback to zero otherwise ──────────────

def test_build_balance_lookup_uses_onchain_when_token_resolves(monkeypatch):
    import prsm.node.wallet_api_wiring as w
    from prsm.interface.api.wallet_api import OnChainBalanceLookup
    # token + rpc resolve → an OnChainBalanceLookup is built
    lk = w.build_balance_lookup_or_zero(
        rpc_url="https://example.invalid", token_address="0x" + "ab" * 20)
    assert isinstance(lk, OnChainBalanceLookup)


def test_build_balance_lookup_falls_back_to_zero_when_no_token():
    import prsm.node.wallet_api_wiring as w
    from prsm.interface.api.wallet_api import _ZeroBalanceLookup
    lk = w.build_balance_lookup_or_zero(rpc_url="https://x", token_address=None)
    assert isinstance(lk, _ZeroBalanceLookup)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
