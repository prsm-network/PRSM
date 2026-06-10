"""Sprint 1069 — AerodromeSwapClient: the USDC→FTNS swap EXECUTION (Tier 4).

The fiat onramp loop (Coinbase Pay → USDC in the user's wallet → swap → FTNS) had
every piece EXCEPT the last mile: aerodrome_client.py is quote-only (it reads pool
state + computes a constant-product quote), and onramp_to_swap_orchestrator builds a
swap ENVELOPE — but nothing builds/signs/broadcasts the on-chain swap. This client
is that write side, mirroring EscrowPoolClient (sp1041): ensure the USDC allowance to
the Aerodrome router, then call swapExactTokensForTokens with the USDC→FTNS Route,
under the per-account tx lock + explicit gas + the three-tier error model.

Like the settlement rail, the live mainnet swap is gated on a SEEDED pool (Foundation
ceremony); this builds + proves the execution code so it activates the moment the
pool has liquidity.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError, OnChainRevertedError)

ROUTER = "0x" + "a1" * 20
USDC = "0x" + "b2" * 20
FTNS = "0x" + "c3" * 20
FACTORY = "0x" + "d4" * 20
ME = "0x" + "11" * 20
RECIP = "0x" + "22" * 20

_CFG = {"send_ok": True, "receipt_status": 1, "allowance": 0, "balance": 0}


class _Fn:
    def __init__(self, name, args, parent):
        self.name, self.args = name, args
        parent._calls.append((name, args))

    def build_transaction(self, overrides):
        return {"data": "0x", **overrides}

    def call(self):
        if self.name == "allowance":
            return _CFG["allowance"]
        if self.name == "balanceOf":
            return _CFG["balance"]
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


class _FakeReceipt:
    def __init__(self, status=1):
        self.status = status
        self.transactionHash = b"\xab" * 32


class _FakeAccount:
    address = ME
    key = b"\x11" * 32

    @staticmethod
    def from_key(k):
        return _FakeAccount()

    @staticmethod
    def sign_transaction(tx, key):
        class _S:
            raw_transaction = b"rawtx"
        return _S()


class _FakeEth:
    def __init__(self):
        self.account = _FakeAccount()

    def contract(self, address=None, abi=None):
        return _FakeContract("dummy")   # construction-time; the test overrides router/usdc

    def get_transaction_count(self, addr, block):
        return 7

    @property
    def gas_price(self):
        return 10**9

    @property
    def chain_id(self):
        return 8453

    def send_raw_transaction(self, raw):
        if not _CFG["send_ok"]:
            raise RuntimeError("rpc down")
        return b"\xcd" * 32

    def wait_for_transaction_receipt(self, h, timeout=None):
        seq = _CFG.get("receipt_statuses")
        if seq:
            return _FakeReceipt(status=seq.pop(0))   # per-tx status, in order
        return _FakeReceipt(status=_CFG["receipt_status"])


class _FakeWeb3:
    def __init__(self, *args, **kwargs):
        self.eth = _FakeEth()

    class HTTPProvider:
        def __init__(self, url):
            pass

    @staticmethod
    def to_checksum_address(a):
        return a


def _client(**over):
    cfg = {"send_ok": True, "receipt_status": 1, "allowance": 0, "balance": 10**9,
           "receipt_statuses": None}
    cfg.update(over)
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
            ftns_address=FTNS, factory_address=FACTORY, private_key="0x" + "11" * 32)
        # the constructor built dummy contracts off the fake web3; point at tracked ones
        c.router = router_c
        c.usdc = usdc_c
        return c, router_c, usdc_c


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_swap_with_sufficient_allowance_skips_approve_and_calls_router():
    c, router_c, usdc_c = _client(allowance=10**9)
    tx = _run(c.swap_usdc_for_ftns(amount_in=10**6, amount_out_min=5 * 10**18,
                                   deadline_unix=2_000_000_000))
    assert tx.startswith("0x")
    assert not any(n == "approve" for n, _ in usdc_c._calls)        # no approve needed
    swaps = [a for n, a in router_c._calls if n == "swapExactTokensForTokens"]
    assert len(swaps) == 1
    amount_in, amount_out_min, routes, to, deadline = swaps[0]
    assert amount_in == 10**6 and amount_out_min == 5 * 10**18
    assert to == ME and deadline == 2_000_000_000
    assert routes == [(USDC, FTNS, False, FACTORY)]               # USDC→FTNS volatile route


def test_swap_with_short_allowance_approves_first():
    c, router_c, usdc_c = _client(allowance=0)
    _run(c.swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1, deadline_unix=2_000_000_000))
    approves = [a for n, a in usdc_c._calls if n == "approve"]
    assert approves == [(ROUTER, 10**6)]                          # exact-amount approve to router


def test_recipient_override():
    c, router_c, _ = _client(allowance=10**9)
    _run(c.swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1,
                              deadline_unix=2_000_000_000, recipient=RECIP))
    to = [a for n, a in router_c._calls if n == "swapExactTokensForTokens"][0][3]
    assert to == RECIP


def test_zero_amount_rejected():
    c, _, _ = _client()
    with pytest.raises(ValueError):
        _run(c.swap_usdc_for_ftns(amount_in=0, amount_out_min=1, deadline_unix=2_000_000_000))


def test_negative_amount_out_min_rejected():
    c, _, _ = _client()
    with pytest.raises(ValueError):
        _run(c.swap_usdc_for_ftns(amount_in=10**6, amount_out_min=-1, deadline_unix=2_000_000_000))


def test_broadcast_failure_propagates():
    c, _, _ = _client(allowance=10**9, send_ok=False)
    with pytest.raises(BroadcastFailedError):
        _run(c.swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1, deadline_unix=2_000_000_000))


def test_revert_propagates():
    c, _, _ = _client(allowance=10**9, receipt_status=0)
    with pytest.raises(OnChainRevertedError):
        _run(c.swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1, deadline_unix=2_000_000_000))


def test_write_requires_private_key():
    import prsm.economy.web3.aerodrome_swap_client as mod
    with patch.object(mod, "HAS_WEB3", True), patch.object(mod, "Web3", _FakeWeb3), \
         patch.object(mod, "Account", _FakeAccount):
        c = mod.AerodromeSwapClient(
            rpc_url="http://rpc", router_address=ROUTER, usdc_address=USDC,
            ftns_address=FTNS, factory_address=FACTORY, private_key=None)
        c.web3 = _FakeWeb3(_FakeContract("router"), _FakeContract("usdc"))
        c.router = _FakeContract("router")
        c.usdc = _FakeContract("usdc")
        with pytest.raises(RuntimeError):
            _run(c.swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1, deadline_unix=2_000_000_000))


# ── execute_swap_envelope (bridge from the orchestrator's envelope) ─────────────

_NOW = 1_000_000_000   # built_at == now (fresh); deadline 2e9 is in the future


def _envelope(**over):
    args = {"amountIn": 8_000_000, "amountOutMin": 5 * 10**18,
            "routes": [{"from_token": "USDC", "to_token": "FTNS", "stable": False,
                        "factory": FACTORY}],
            "to": RECIP, "deadline": 2_000_000_000}
    env = {"status": "READY_FOR_SUBMISSION", "function": "swapExactTokensForTokens",
           "router_address": ROUTER, "envelope_built_at": _NOW,
           "args": args, "intent_id": "intent-1"}
    for k, v in over.items():
        if k in args:
            args[k] = v
        else:
            env[k] = v
    return env


def _exec(env, c, **kw):
    from prsm.economy.web3.aerodrome_swap_client import execute_swap_envelope
    kw.setdefault("now", _NOW)
    return _run(execute_swap_envelope(env, c, **kw))


def test_execute_envelope_happy_path():
    c, router_c, _ = _client(allowance=10**12)
    out = _exec(_envelope(), c)
    assert out["status"] == "SUBMITTED" and out["tx_hash"].startswith("0x")
    assert out["intent_id"] == "intent-1"
    swap = [a for n, a in router_c._calls if n == "swapExactTokensForTokens"][0]
    assert swap[0] == 8_000_000 and swap[3] == RECIP


def test_execute_envelope_rejects_wrong_function():
    c, _, _ = _client(allowance=10**12)
    with pytest.raises(ValueError):
        _exec(_envelope(function="transfer"), c)


def test_execute_envelope_rejects_non_usdc_ftns_route():
    c, _, _ = _client(allowance=10**12)
    env = _envelope()
    env["args"]["routes"] = [{"from_token": "USDC", "to_token": "WETH", "stable": False, "factory": FACTORY}]
    with pytest.raises(ValueError):
        _exec(env, c)


def test_execute_envelope_refuses_zero_slippage_floor():
    c, _, _ = _client(allowance=10**12)
    with pytest.raises(ValueError):
        _exec(_envelope(amountOutMin=0), c)


def test_execute_envelope_rejects_router_mismatch():
    """HIGH-1: an envelope quoted for a different router must not execute against
    this client's venue."""
    c, _, _ = _client(allowance=10**12)
    with pytest.raises(ValueError):
        _exec(_envelope(router_address="0x" + "ee" * 20), c)


def test_execute_envelope_rejects_stale_quote():
    """MEDIUM-1: a stale envelope (old quote) is refused — its slippage floor can't
    be trusted on a thin pool."""
    c, _, _ = _client(allowance=10**12)
    env = _envelope()
    env["envelope_built_at"] = _NOW - 4000   # > 600s old
    with pytest.raises(ValueError):
        _exec(env, c)


def test_execute_envelope_rejects_past_deadline():
    c, _, _ = _client(allowance=10**12)
    with pytest.raises(ValueError):
        _exec(_envelope(deadline=_NOW - 1), c)


def test_execute_envelope_requires_recipient():
    c, _, _ = _client(allowance=10**12)
    with pytest.raises(ValueError):
        _exec(_envelope(to=None), c)


def test_swap_insufficient_usdc_balance_rejected():
    """MEDIUM-3: fail fast (no approve) when the wallet can't cover amount_in."""
    c, _, usdc_c = _client(allowance=0, balance=5)
    with pytest.raises(ValueError):
        _run(c.swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1, deadline_unix=2_000_000_000))
    assert not any(n == "approve" for n, _ in usdc_c._calls)


def test_failed_swap_revokes_residual_allowance():
    """HIGH-2: when WE approved and the swap reverts, the residual router allowance
    is revoked (approve(router, 0))."""
    # approve succeeds (status 1), swap reverts (status 0), revoke succeeds (status 1)
    c, _, usdc_c = _client(allowance=0, balance=10**9, receipt_statuses=[1, 0, 1])
    with pytest.raises(OnChainRevertedError):
        _run(c.swap_usdc_for_ftns(amount_in=10**6, amount_out_min=1, deadline_unix=2_000_000_000))
    approves = [a for n, a in usdc_c._calls if n == "approve"]
    assert (ROUTER, 10**6) in approves and (ROUTER, 0) in approves   # approved then revoked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
