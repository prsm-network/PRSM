"""Sprint 1151 — AerodromeSwapClient READ-ONLY dry-run (verify-before-broadcast).

The swap WRITE side (sp1069 ``_swap_sync``) checks balance/allowance, approves if
short, then BROADCASTS ``swapExactTokensForTokens`` — but it had NO dry-run, unlike
every other money/dispute path (the challenge submitters static-``eth_call`` a dry-run
BEFORE any broadcast). This brick adds ``dry_run_swap_usdc_for_ftns``: an operator can
VERIFY a swap WOULD succeed (liquidity / balance / allowance / slippage) without ever
broadcasting, signing, approving, or building a tx.

It is especially valuable pre-seeding: the Aerodrome USDC/FTNS pool is not yet seeded
(Foundation ceremony), so a dry-run today cleanly reports "would revert: insufficient
liquidity" — proving the swap is correctly wired and activates the moment the pool has
liquidity — instead of crashing.

These fakes mirror the sp1069 swap-client test fixtures, extended so the contract
``.call()`` understands ``getAmountsOut`` and so any broadcast/sign/approve/
build_transaction surface RAISES (the dry-run must touch ONLY ``.call()`` reads).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

ROUTER = "0x" + "a1" * 20
USDC = "0x" + "b2" * 20
FTNS = "0x" + "c3" * 20
FACTORY = "0x" + "d4" * 20
ME = "0x" + "11" * 20
OTHER = "0x" + "33" * 20

# Shared config the fakes read. ``amounts_out`` None => getAmountsOut .call() RAISES
# (the pre-seeding / no-pool revert).
_CFG = {}


class _Fn:
    def __init__(self, name, args, parent):
        self.name, self.args = name, args
        parent._calls.append((name, args))

    def build_transaction(self, overrides):  # pragma: no cover - must never run in dry-run
        raise AssertionError(
            f"dry-run must NOT build a transaction (called build_transaction for "
            f"{self.name})")

    def call(self, *call_args, **call_kwargs):
        if self.name == "allowance":
            return _CFG["allowance"]
        if self.name == "balanceOf":
            return _CFG["balance"]
        if self.name == "getAmountsOut":
            amounts = _CFG.get("amounts_out")
            if amounts is None:
                raise Exception("execution reverted: no pool / insufficient liquidity")
            return list(amounts)
        return None


class _Functions:
    def __init__(self, parent):
        self._p = parent

    def __getattr__(self, name):
        return lambda *args: _Fn(name, args, self._p)


class _FakeContract:
    def __init__(self, kind):
        self.kind = kind
        self._calls = []
        self.functions = _Functions(self)


class _FakeAccount:
    address = ME
    key = b"\x11" * 32

    @staticmethod
    def from_key(k):
        return _FakeAccount()

    @staticmethod
    def sign_transaction(tx, key):  # pragma: no cover - must never run in dry-run
        raise AssertionError("dry-run must NOT sign a transaction")


class _FakeEth:
    def __init__(self):
        self.account = _FakeAccount()

    def contract(self, address=None, abi=None):
        return _FakeContract("dummy")

    def get_transaction_count(self, addr, block):  # pragma: no cover
        raise AssertionError("dry-run must NOT read a nonce / prepare a tx")

    @property
    def gas_price(self):  # pragma: no cover
        raise AssertionError("dry-run must NOT read gas_price / prepare a tx")

    @property
    def chain_id(self):  # pragma: no cover
        raise AssertionError("dry-run must NOT read chain_id / prepare a tx")

    def send_raw_transaction(self, raw):  # pragma: no cover
        raise AssertionError("dry-run must NEVER broadcast (send_raw_transaction)")

    def wait_for_transaction_receipt(self, h, timeout=None):  # pragma: no cover
        raise AssertionError("dry-run must NEVER wait for a receipt")


class _FakeWeb3:
    def __init__(self, *args, **kwargs):
        self.eth = _FakeEth()

    class HTTPProvider:
        def __init__(self, url):
            pass

    @staticmethod
    def to_checksum_address(a):
        return a


def _client(*, private_key="0x" + "11" * 32, **over):
    cfg = {"allowance": 10**12, "balance": 10**9, "amounts_out": None}
    cfg.update(over)
    _CFG.clear()
    _CFG.update(cfg)
    router_c, usdc_c = _FakeContract("router"), _FakeContract("usdc")

    import prsm.economy.web3.aerodrome_swap_client as mod
    with patch.object(mod, "HAS_WEB3", True), \
         patch.object(mod, "Web3", _FakeWeb3), \
         patch.object(mod, "Account", _FakeAccount), \
         patch("prsm.economy.web3.tx_lock_registry.TX_LOCK_REGISTRY") as lockreg:
        import threading
        lockreg.get_lock.return_value = threading.Lock()
        c = mod.AerodromeSwapClient(
            rpc_url="http://rpc", router_address=ROUTER, usdc_address=USDC,
            ftns_address=FTNS, factory_address=FACTORY, private_key=private_key)
        c.router = router_c
        c.usdc = usdc_c
        return c, router_c, usdc_c


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# ── (a) happy path ───────────────────────────────────────────────────────────────

def test_dry_run_happy_path_would_succeed():
    expected_out = 6 * 10**18
    c, _, _ = _client(allowance=10**12, balance=10**9,
                      amounts_out=[10**6, expected_out])
    res = _run(c.dry_run_swap_usdc_for_ftns(amount_in=10**6, amount_out_min=5 * 10**18))
    assert res.would_succeed is True
    assert res.expected_out == expected_out
    assert res.amount_in == 10**6 and res.amount_out_min == 5 * 10**18
    assert res.sufficient_balance is True
    assert res.sufficient_allowance is True
    assert res.approve_needed is False
    assert res.revert_reason is None


# ── (b) no liquidity (pre-seeding) ─────────────────────────────────────────────────

def test_dry_run_no_liquidity_pre_seeding():
    # amounts_out=None => getAmountsOut .call() RAISES (no pool yet).
    c, _, _ = _client(allowance=10**12, balance=10**9, amounts_out=None)
    res = _run(c.dry_run_swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1))
    assert res.would_succeed is False
    assert res.expected_out is None
    rr = (res.revert_reason or "").lower()
    assert "liquidity" in rr or "pool" in rr


# ── (c) slippage ────────────────────────────────────────────────────────────────

def test_dry_run_slippage_floor_not_met():
    c, _, _ = _client(allowance=10**12, balance=10**9,
                      amounts_out=[10**6, 4 * 10**18])
    res = _run(c.dry_run_swap_usdc_for_ftns(amount_in=10**6, amount_out_min=5 * 10**18))
    assert res.would_succeed is False
    assert res.expected_out == 4 * 10**18
    rr = (res.revert_reason or "").lower()
    assert "slippage" in rr or "amount_out_min" in rr or "amountoutmin" in rr


# ── (d) insufficient balance ───────────────────────────────────────────────────────

def test_dry_run_insufficient_balance():
    c, _, _ = _client(allowance=10**12, balance=5,
                      amounts_out=[10**6, 6 * 10**18])
    res = _run(c.dry_run_swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1))
    assert res.would_succeed is False
    assert res.sufficient_balance is False
    rr = (res.revert_reason or "").lower()
    assert "balance" in rr


# ── (e) approve-needed is NOT a hard fail ───────────────────────────────────────────

def test_dry_run_approve_needed_is_not_a_hard_fail():
    # allowance short, but balance + liquidity + slippage all OK => the real swap
    # auto-approves, so the dry-run must still report would_succeed=True.
    c, _, _ = _client(allowance=0, balance=10**9,
                      amounts_out=[10**6, 6 * 10**18])
    res = _run(c.dry_run_swap_usdc_for_ftns(amount_in=10**6, amount_out_min=5 * 10**18))
    assert res.approve_needed is True
    assert res.sufficient_allowance is False
    assert res.would_succeed is True


# ── (f) NEVER broadcast / sign / approve / build_transaction ────────────────────────

def test_dry_run_never_broadcasts_signs_or_approves():
    c, router_c, usdc_c = _client(allowance=0, balance=10**9,
                                  amounts_out=[10**6, 6 * 10**18])
    # The fakes raise AssertionError on any broadcast/sign/build_transaction/nonce/
    # gas/chainId surface; completing proves the dry-run used ONLY .call() reads.
    res = _run(c.dry_run_swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1))
    assert res.would_succeed is True
    # No write methods were ever invoked on either contract.
    write_names = {"approve", "swapExactTokensForTokens"}
    assert not any(n in write_names for n, _ in usdc_c._calls)
    assert not any(n in write_names for n, _ in router_c._calls)


# ── (g) validation ────────────────────────────────────────────────────────────────

def test_dry_run_rejects_non_positive_amount_in():
    c, _, _ = _client(amounts_out=[10**6, 6 * 10**18])
    with pytest.raises(ValueError):
        _run(c.dry_run_swap_usdc_for_ftns(amount_in=0, amount_out_min=1))


def test_dry_run_rejects_negative_amount_out_min():
    c, _, _ = _client(amounts_out=[10**6, 6 * 10**18])
    with pytest.raises(ValueError):
        _run(c.dry_run_swap_usdc_for_ftns(amount_in=10**6, amount_out_min=-1))


# ── no-signer path: liquidity + slippage still verifiable, balance/allowance skipped ─

def test_dry_run_without_signer_or_owner_skips_balance_checks():
    c, _, _ = _client(private_key=None, amounts_out=[10**6, 6 * 10**18])
    res = _run(c.dry_run_swap_usdc_for_ftns(amount_in=10**6, amount_out_min=5 * 10**18))
    # Liquidity + slippage are verifiable without an owner; balance/allowance are not.
    assert res.expected_out == 6 * 10**18
    assert res.sufficient_balance is None
    assert res.sufficient_allowance is None
    assert "owner" in (res.revert_reason or "").lower() or res.would_succeed is True


def test_dry_run_owner_override_checks_that_owner():
    c, _, usdc_c = _client(private_key=None, allowance=10**12, balance=10**9,
                           amounts_out=[10**6, 6 * 10**18])
    res = _run(c.dry_run_swap_usdc_for_ftns(
        amount_in=10**6, amount_out_min=5 * 10**18, owner=OTHER))
    assert res.would_succeed is True
    assert res.sufficient_balance is True
    # balanceOf/allowance were called for the OVERRIDE owner.
    bal_calls = [a for n, a in usdc_c._calls if n == "balanceOf"]
    assert bal_calls and bal_calls[0][0] == OTHER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
