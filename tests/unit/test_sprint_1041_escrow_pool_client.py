"""Sprint 1041 — EscrowPoolClient (brick 3a): the requester-funding mechanism.

On-chain settlement (bricks 1-2.6) commits + finalizes batches, but finalizeBatch
calls EscrowPool.settleFromRequester(requester, recipient, amount) — which reverts
unless the requester has a funded escrow balance. There was no client to deposit
into EscrowPool, so the whole rail self-pays-of-zero (mainnet EscrowPool balance
== 0 → a live finalize reverts). This adds the missing deposit/balance/withdraw
client, mirroring Web3SettlementContractClient (sync web3 + Account signing +
three-tier error model, wrapped in asyncio.to_thread).

EscrowPool.deposit(amount) requires the caller to have approved FTNS for the pool
first, so deposit() ensures allowance >= amount (approve only when short) then
deposits. deposit moves the CALLER's own FTNS into the CALLER's own escrow balance
(recoverable via withdraw) — lower blast radius than the settlement client, but
still a real money path, so it is fail-fast + tested.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainRevertedError,
)

POOL_ADDR = "0x" + "ab" * 20
FTNS_ADDR = "0x" + "cd" * 20
ME = "0x" + "11" * 20


# ── Stub web3 surface (two contracts: pool + FTNS token) ──────────────────────

_CFG = {"send_ok": True, "receipt_status": 1, "allowance": 0, "balance": 0}


class _Fn:
    def __init__(self, name, args, parent):
        self.name, self.args = name, args
        self._p = parent
        parent._calls.append((name, args))

    def build_transaction(self, overrides):
        return {"to": self._p.kind, "data": "0x", **overrides}

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
        self.kind = kind            # "pool" or "ftns"
        self._calls = []
        self.functions = _Functions(self)


class _FakeReceipt:
    def __init__(self, status=1):
        self.status = status


class _FakeAccount:
    address = ME
    key = b"\x01" * 32


class _FakeEth:
    def __init__(self):
        self.chain_id = 8453
        self.gas_price = 10 ** 9
        self.account = MagicMock()
        signed = MagicMock()
        signed.raw_transaction = b"\xff" * 32
        self.account.sign_transaction.return_value = signed
        self._contracts = {}

    def get_transaction_count(self, addr, *_):
        return 7

    def contract(self, address, abi):
        # pool vs ftns disambiguated by address
        kind = "pool" if address == POOL_ADDR else "ftns"
        c = self._contracts.get(kind) or _FakeContract(kind)
        self._contracts[kind] = c
        return c

    def send_raw_transaction(self, raw):
        if not _CFG["send_ok"]:
            raise RuntimeError("network down")
        return b"\xab" * 32

    def wait_for_transaction_receipt(self, tx_hash, timeout=120):
        return _FakeReceipt(status=_CFG["receipt_status"])


class _FakeWeb3:
    def __init__(self, *a, **k):
        self.eth = _FakeEth()

    @staticmethod
    def to_checksum_address(addr):
        return addr

    @staticmethod
    def HTTPProvider(url):
        return object()


def _make_client(*, with_key=True):
    with patch("prsm.economy.web3.escrow_pool_client.Web3", _FakeWeb3), \
         patch("prsm.economy.web3.escrow_pool_client.Account") as mock_account:
        mock_account.from_key.return_value = _FakeAccount()
        from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
        return EscrowPoolClient(
            rpc_url="http://test",
            escrow_pool_address=POOL_ADDR,
            ftns_token_address=FTNS_ADDR,
            private_key="0x" + "01" * 32 if with_key else None,
        )


async def _run(coro):
    return await coro


def _reset():
    _CFG.update(send_ok=True, receipt_status=1, allowance=0, balance=0)


# ── balance ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_balance_of_reads_view():
    _reset()
    _CFG["balance"] = 4242
    client = _make_client(with_key=False)
    assert await client.balance_of(ME) == 4242


# ── deposit ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deposit_approves_then_deposits_when_allowance_short():
    _reset()
    _CFG["allowance"] = 0  # must approve first
    client = _make_client()
    tx = await client.deposit(1000)
    assert isinstance(tx, str) and tx.startswith("0x")
    pool = client.web3.eth._contracts["pool"]
    ftns = client.web3.eth._contracts["ftns"]
    fns = [c[0] for c in ftns._calls]
    pns = [c[0] for c in pool._calls]
    assert "approve" in fns           # approved the pool for FTNS
    assert "deposit" in pns           # then deposited


@pytest.mark.asyncio
async def test_deposit_skips_approve_when_allowance_sufficient():
    _reset()
    _CFG["allowance"] = 10_000        # already approved enough
    client = _make_client()
    await client.deposit(1000)
    ftns = client.web3.eth._contracts["ftns"]
    assert "approve" not in [c[0] for c in ftns._calls]
    pool = client.web3.eth._contracts["pool"]
    assert "deposit" in [c[0] for c in pool._calls]


@pytest.mark.asyncio
async def test_deposit_without_key_raises():
    _reset()
    client = _make_client(with_key=False)
    with pytest.raises(RuntimeError, match="private_key"):
        await client.deposit(1000)


@pytest.mark.asyncio
async def test_deposit_zero_or_negative_raises():
    _reset()
    client = _make_client()
    for bad in (0, -5):
        with pytest.raises(ValueError):
            await client.deposit(bad)


@pytest.mark.asyncio
async def test_deposit_broadcast_failure_raises():
    _reset()
    _CFG["allowance"] = 10_000        # skip approve so the failing tx is the deposit
    _CFG["send_ok"] = False
    client = _make_client()
    with pytest.raises(BroadcastFailedError):
        await client.deposit(1000)


@pytest.mark.asyncio
async def test_deposit_revert_raises():
    _reset()
    _CFG["allowance"] = 10_000
    _CFG["receipt_status"] = 0        # mined but reverted
    client = _make_client()
    with pytest.raises(OnChainRevertedError):
        await client.deposit(1000)


# ── withdraw ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_withdraw_happy_returns_tx():
    _reset()
    client = _make_client()
    tx = await client.withdraw(500)
    assert isinstance(tx, str) and tx.startswith("0x")
    pool = client.web3.eth._contracts["pool"]
    assert "withdraw" in [c[0] for c in pool._calls]


@pytest.mark.asyncio
async def test_withdraw_without_key_raises():
    _reset()
    client = _make_client(with_key=False)
    with pytest.raises(RuntimeError, match="private_key"):
        await client.withdraw(500)


# ── construction ──────────────────────────────────────────────────────────────


def test_address_property_reflects_key():
    _reset()
    assert _make_client().address == ME
    assert _make_client(with_key=False).address is None


@pytest.mark.asyncio
async def test_deposit_txs_carry_explicit_gas_no_estimate():
    """sp1047 — approve + deposit set an explicit gas limit so build_transaction
    SKIPS eth_estimateGas (which can spuriously revert against a lagging RPC
    replica that hasn't yet seen the just-mined approve — the mainnet failure)."""
    _reset()
    _CFG["allowance"] = 0          # forces both approve + deposit
    client = _make_client()
    await client.deposit(1000)
    signed_txs = [c.args[0] for c in
                  client.web3.eth.account.sign_transaction.call_args_list]
    assert len(signed_txs) >= 2    # approve + deposit
    assert all("gas" in tx and tx["gas"] > 0 for tx in signed_txs)


@pytest.mark.asyncio
async def test_withdraw_tx_carries_explicit_gas():
    _reset()
    client = _make_client()
    await client.withdraw(500)
    signed_txs = [c.args[0] for c in
                  client.web3.eth.account.sign_transaction.call_args_list]
    assert all("gas" in tx and tx["gas"] > 0 for tx in signed_txs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
