"""Sprint 931 — on-chain nonce-race between clients sharing a signing account
(settlement money-rail review, final finding).

OnChainFTNSLedger.transfer() and RoyaltyDistributorClient both sign from the same
FTNS_WALLET_PRIVATE_KEY but used INDEPENDENT locks (an asyncio.Lock and a private
threading.Lock). The per-account "pending" nonce read does NOT prevent collision:
both clients read the same pending nonce and each broadcasts a tx with it — the
second is silently dropped on-chain ("nonce too low" / "replacement underpriced"),
so a content-creator or royalty payout appears confirmed in logs but never lands.

Fix: both serialize their nonce-fetch→sign→send through the SAME process-wide
per-account lock (TX_LOCK_REGISTRY.get_lock(address)) — the mechanism a half-dozen
other web3 clients already use. RoyaltyDistributorClient swaps its private lock for
the registry lock; OnChainFTNSLedger.transfer acquires the registry lock (off the
event loop) around nonce→send, releasing it before the multi-second receipt wait.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from eth_account import Account

from prsm.economy import ftns_onchain
from prsm.economy.ftns_onchain import OnChainFTNSLedger
from prsm.economy.web3.royalty_distributor import RoyaltyDistributorClient
from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY

_KEY = "0x" + "1" * 64          # deterministic test signing key
_DEST = Account.from_key("0x" + "2" * 64).address


# ── RoyaltyDistributorClient now uses the shared registry lock ────────────


def test_royalty_distributor_uses_shared_account_lock():
    client = RoyaltyDistributorClient(
        rpc_url="http://localhost:8545",
        distributor_address="0x" + "2" * 40,
        ftns_token_address="0x" + "3" * 40,
        private_key=_KEY,
    )
    shared = TX_LOCK_REGISTRY.get_lock(client._account.address)
    assert client._tx_lock is shared, "royalty client must use the per-account registry lock"


def test_two_royalty_clients_same_key_share_one_lock():
    a = RoyaltyDistributorClient("http://x", "0x" + "2" * 40, "0x" + "3" * 40, private_key=_KEY)
    b = RoyaltyDistributorClient("http://y", "0x" + "4" * 40, "0x" + "5" * 40, private_key=_KEY)
    assert a._tx_lock is b._tx_lock


# ── OnChainFTNSLedger.transfer holds the shared lock + serializes nonces ──


def _fake_ledger(state, lock_held_log, shared_lock):
    """OnChainFTNSLedger with the minimum wired for transfer(), sharing `state`
    (the chain's nonce counter) with its siblings."""
    led = OnChainFTNSLedger.__new__(OnChainFTNSLedger)
    acct = Account.from_key(_KEY)
    led._account = acct
    led._connected_address = acct.address
    led._is_initialized = True
    led.chain_id = 8453
    led.contract_address = "0x" + "6" * 40
    led._decimals = 18
    led._transactions = []
    led._lock = asyncio.Lock()
    led._record_tx = AsyncMock()
    led._update_tx_status = AsyncMock()

    encoded = SimpleNamespace(_encode_transaction_data=lambda: "0x")
    led._token = SimpleNamespace(
        functions=SimpleNamespace(transfer=lambda to, wei: encoded)
    )

    class _Eth:
        def __init__(self):
            self.account = SimpleNamespace(
                sign_transaction=lambda tx, key: SimpleNamespace(raw_transaction=b"\x01")
            )

        def get_transaction_count(self, addr, block):
            # Record whether the shared per-account lock is held right now —
            # this is the deterministic proof the fix wired the lock in.
            lock_held_log.append(shared_lock.locked())
            return state["sent"]

        def send_raw_transaction(self, raw):
            state["sent"] += 1            # the nonce is consumed
            return SimpleNamespace(hex=lambda: "0x" + "f" * 64)

        def wait_for_transaction_receipt(self, h, timeout):
            return {"status": 1, "blockNumber": 1}

    led.w3 = SimpleNamespace(eth=_Eth())
    return led


@pytest.mark.asyncio
async def test_transfer_holds_shared_lock_during_nonce_fetch(monkeypatch):
    monkeypatch.setattr(ftns_onchain, "estimate_gas_price", lambda w3: 1)
    state = {"sent": 0}
    log = []
    shared = TX_LOCK_REGISTRY.get_lock(Account.from_key(_KEY).address)
    led = _fake_ledger(state, log, shared)

    rec = await led.transfer(job_id="j1", to_address=_DEST, amount_ftns=1.0)

    assert rec is not None and rec.status == "confirmed"
    assert log == [True], "the per-account lock must be HELD during the nonce fetch"
    assert not shared.locked(), "the lock must be released after send"


@pytest.mark.asyncio
async def test_concurrent_transfers_sharing_account_get_distinct_nonces(monkeypatch):
    # Two SEPARATE ledgers (distinct instances, so distinct asyncio self._locks)
    # signing from the SAME account, transferring concurrently. The shared
    # registry lock must serialize them so they consume DISTINCT nonces — the
    # cross-client collision the fix closes.
    monkeypatch.setattr(ftns_onchain, "estimate_gas_price", lambda w3: 1)
    state = {"sent": 0}
    log = []
    shared = TX_LOCK_REGISTRY.get_lock(Account.from_key(_KEY).address)
    nonces = []

    def _instrumented_ledger():
        led = _fake_ledger(state, log, shared)
        real_get = led.w3.eth.get_transaction_count

        def get(addr, block):
            n = real_get(addr, block)
            nonces.append(state["sent"])
            return n
        led.w3.eth.get_transaction_count = get
        return led

    a = _instrumented_ledger()
    b = _instrumented_ledger()
    await asyncio.gather(
        a.transfer(job_id="ja", to_address=_DEST, amount_ftns=1.0),
        b.transfer(job_id="jb", to_address=_DEST, amount_ftns=1.0),
    )

    assert sorted(nonces) == [0, 1], f"nonces collided/skipped: {nonces}"
    assert all(log), "the shared lock must be held during every nonce fetch"
