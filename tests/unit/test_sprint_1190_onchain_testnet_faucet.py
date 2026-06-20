"""Sprint 1190 — on-chain TESTNET FTNS faucet (day-one blocker #3, testnet path).

A brand-new user holds 0 FTNS and the only existing faucet credits the OFF-chain
LocalLedger — useless for the day-one on-chain flow (sp1183 balance + sp1189
pay_and_infer read/spend on-chain FTNS). This dispenses REAL on-chain testnet FTNS
(ERC-20 transfer from a faucet wallet → the user's address) so the full front door works
end-to-end on Base Sepolia. CRITICAL: a hard mainnet kill-switch — the faucet NEVER
dispenses on chainId != 84532 (never gives away real-value FTNS), fail-closed.
"""
from __future__ import annotations

import pytest
from eth_account import Account

from prsm.economy.web3.ftns_faucet import (
    FaucetMainnetRefusedError,
    OnChainFTNSFaucet,
    build_onchain_faucet_or_none,
)

_FAUCET_KEY = "0x" + "33" * 32
_RECIP = "0x" + "44" * 20


# ── fakes (no live chain) ─────────────────────────────────────────────────────────────

class _Fn:
    def __init__(self, val, parent=None, kind=None):
        self._val = val; self._p = parent; self._kind = kind
    def call(self):
        return self._val
    def build_transaction(self, ov):
        if self._p is not None:
            self._p.built = {"kind": self._kind, "overrides": ov}
        return {"to": "0xtoken", **ov}

class _Funcs:
    def __init__(self, parent):
        self._p = parent
    def balanceOf(self, a):
        self._p.balanceof_addr = a
        return _Fn(self._p.balances.get(a, 0))
    def decimals(self):
        return _Fn(self._p.decimals)
    def transfer(self, to, amount):
        self._p.transfer_args = (to, amount)
        return _Fn(True, self._p, "transfer")

class _Contract:
    def __init__(self, parent):
        self.functions = _Funcs(parent)

class _Receipt:
    def __init__(self, status): self.status = status

class _Eth:
    def __init__(self, parent): self._p = parent; self.account = _EthAcct()
    @property
    def chain_id(self): return self._p.chain_id
    def get_transaction_count(self, a, b): return 1
    @property
    def gas_price(self): return 10**9
    def send_raw_transaction(self, raw): return b"\xfa\xce"
    def wait_for_transaction_receipt(self, h, timeout=120): return _Receipt(self._p.tx_status)

class _EthAcct:
    def sign_transaction(self, tx, key):
        return type("S", (), {"raw_transaction": b"\x01"})()

class _Web3:
    def __init__(self, parent): self.eth = _Eth(parent)
    @staticmethod
    def to_checksum_address(a): return Account.from_key(_FAUCET_KEY).address if a == "self" else a


class _Harness:
    def __init__(self, *, chain_id=84532, tx_status=1, balances=None, decimals=18):
        self.chain_id = chain_id; self.tx_status = tx_status
        self.balances = balances or {}; self.decimals = decimals
        self.transfer_args = None; self.built = None; self.balanceof_addr = None
        self.account = type("A", (), {"address": "0xFaucetAddr", "key": b"k"})()
    def faucet(self, expected=84532):
        return OnChainFTNSFaucet(
            rpc_url="x", token_address="0xtoken", faucet_private_key=_FAUCET_KEY,
            expected_chain_id=expected, _web3=_Web3(self), _contract=_Contract(self),
            _account=self.account)


# ── the mainnet kill-switch (the critical safety) ────────────────────────────────────

def test_dispense_refuses_on_mainnet_chain_id():
    h = _Harness(chain_id=8453)  # Base mainnet — must NEVER dispense
    with pytest.raises(FaucetMainnetRefusedError):
        h.faucet().dispense(_RECIP, 10**18)
    assert h.transfer_args is None  # no transfer attempted


def test_dispense_refuses_on_unexpected_chain_id():
    h = _Harness(chain_id=1)  # Ethereum mainnet — also refused
    with pytest.raises(FaucetMainnetRefusedError):
        h.faucet().dispense(_RECIP, 10**18)


# ── testnet dispense ──────────────────────────────────────────────────────────────────

def test_dispense_transfers_on_testnet():
    h = _Harness(chain_id=84532, tx_status=1)
    tx = h.faucet().dispense(_RECIP, 5 * 10**18)
    assert tx == "0x" + b"\xfa\xce".hex()
    assert h.transfer_args == (_RECIP, 5 * 10**18)
    assert h.built["kind"] == "transfer" and h.built["overrides"]["gas"] > 0  # explicit gas


def test_balance_of_wei_and_decimals():
    h = _Harness(balances={_RECIP: 7 * 10**18}, decimals=18)
    f = h.faucet()
    assert f.balance_of_wei(_RECIP) == 7 * 10**18
    assert f.decimals == 18


def test_reverted_transfer_raises():
    from prsm.economy.web3.provenance_registry import OnChainRevertedError
    h = _Harness(chain_id=84532, tx_status=0)
    with pytest.raises(OnChainRevertedError):
        h.faucet().dispense(_RECIP, 10**18)


# ── build helper: testnet-only, key-gated ─────────────────────────────────────────────

def test_build_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("PRSM_FAUCET_PRIVATE_KEY", raising=False)
    assert build_onchain_faucet_or_none() is None


def test_build_returns_none_on_mainnet_config(monkeypatch):
    monkeypatch.setenv("PRSM_FAUCET_PRIVATE_KEY", _FAUCET_KEY)
    monkeypatch.setenv("PRSM_NETWORK", "mainnet")  # never build a faucet on mainnet
    assert build_onchain_faucet_or_none() is None


def test_build_constructs_on_testnet(monkeypatch):
    monkeypatch.setenv("PRSM_FAUCET_PRIVATE_KEY", _FAUCET_KEY)
    monkeypatch.setenv("PRSM_NETWORK", "testnet")
    f = build_onchain_faucet_or_none()
    assert isinstance(f, OnChainFTNSFaucet)


# ── HTTP endpoint: /ftns/faucet/onchain (day-one front door) ─────────────────────────

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


class _FakeFaucet:
    """Stand-in for OnChainFTNSFaucet at the HTTP layer (no chain)."""

    def __init__(self, *, decimals=18, balances=None, raises=None):
        self.decimals = decimals
        self._balances = balances or {}
        self._raises = raises
        self.faucet_address = "0xFaucetWallet"
        self.dispensed = None

    def balance_of_wei(self, addr):
        return self._balances.get(addr, 0)

    def dispense(self, recipient, amount_wei):
        if self._raises is not None:
            raise self._raises
        self.dispensed = (recipient, amount_wei)
        return "0x" + "ab" * 32


def _client(faucet, monkeypatch):
    # No PRSM_FAUCET_PRIVATE_KEY → the endpoint's lazy build_onchain_faucet_or_none()
    # fallback also returns None, so the only faucet is the one we inject on the node.
    monkeypatch.delenv("PRSM_FAUCET_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("PRSM_FAUCET_ENABLED", raising=False)
    monkeypatch.delenv("PRSM_FAUCET_MAX_PER_REQUEST", raising=False)
    monkeypatch.delenv("PRSM_FAUCET_MAX_PER_WALLET", raising=False)
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._onchain_faucet = faucet
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_endpoint_dispenses_on_testnet(monkeypatch):
    f = _FakeFaucet()
    resp = _client(f, monkeypatch).post(
        "/ftns/faucet/onchain", json={"destination_address": _RECIP, "amount": 50})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "dispensed"
    assert body["recipient"] == _RECIP
    assert body["dispensed_ftns"] == 50
    assert body["tx_hash"].startswith("0x")
    assert f.dispensed == (_RECIP, 50 * 10**18)


def test_endpoint_defaults_to_per_request_cap(monkeypatch):
    f = _FakeFaucet()
    resp = _client(f, monkeypatch).post(
        "/ftns/faucet/onchain", json={"destination_address": _RECIP})
    assert resp.status_code == 200
    assert resp.json()["dispensed_ftns"] == 100  # default PRSM_FAUCET_MAX_PER_REQUEST
    assert f.dispensed == (_RECIP, 100 * 10**18)


def test_endpoint_clamps_to_per_request_cap(monkeypatch):
    f = _FakeFaucet()
    resp = _client(f, monkeypatch).post(
        "/ftns/faucet/onchain", json={"destination_address": _RECIP, "amount": 99999})
    assert resp.status_code == 200
    assert resp.json()["dispensed_ftns"] == 100  # clamped
    assert f.dispensed == (_RECIP, 100 * 10**18)


def test_endpoint_403_when_no_faucet(monkeypatch):
    # No injected faucet AND no key → testnet-only faucet unavailable.
    resp = _client(None, monkeypatch).post(
        "/ftns/faucet/onchain", json={"destination_address": _RECIP})
    assert resp.status_code == 403
    assert "testnet" in resp.json()["detail"].lower()


def test_endpoint_422_on_bad_address(monkeypatch):
    resp = _client(_FakeFaucet(), monkeypatch).post(
        "/ftns/faucet/onchain", json={"destination_address": "nope"})
    assert resp.status_code == 422


def test_endpoint_429_when_wallet_at_cap(monkeypatch):
    # recipient already holds the per-wallet cap (1000 FTNS) → refuse.
    f = _FakeFaucet(balances={_RECIP: 1000 * 10**18})
    resp = _client(f, monkeypatch).post(
        "/ftns/faucet/onchain", json={"destination_address": _RECIP})
    assert resp.status_code == 429
    assert f.dispensed is None  # nothing transferred


def test_endpoint_502_on_mainnet_refusal(monkeypatch):
    # The chain-level kill-switch surfaces as a 502 (never a 200-with-error).
    f = _FakeFaucet(raises=FaucetMainnetRefusedError("nope, mainnet"))
    resp = _client(f, monkeypatch).post(
        "/ftns/faucet/onchain", json={"destination_address": _RECIP})
    assert resp.status_code == 502


def test_endpoint_403_when_disabled(monkeypatch):
    monkeypatch.setenv("PRSM_FAUCET_ENABLED", "0")
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._onchain_faucet = _FakeFaucet()
    resp = TestClient(
        create_api_app(node, enable_security=False), raise_server_exceptions=False,
    ).post("/ftns/faucet/onchain", json={"destination_address": _RECIP})
    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
