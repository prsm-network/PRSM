"""Sprint 1021 — Web3SettlementContractClient: the concrete on-chain settlement
client (Tier-1 audit gap).

The settlement orchestration (BatchSettlementClient + ReceiptAccumulator + Merkle)
already exists and drives commit→finalize against the SettlementContractClient
Protocol, but there was NO concrete web3 implementation — only the Protocol and
AsyncMock test doubles — so the signed inference receipt path could never reach the
deployed BatchSettlementRegistry on Base. This client closes that: a faithful
web3.py wrapper of the deployed registry, mirroring the compensation_distributor
client pattern, with the blocking sync web3 calls wrapped in asyncio.to_thread so it
satisfies the async Protocol without stalling the event loop.

Tests use a stub Web3 + contract surface (mirrors test_compensation_distributor_client.py)
to verify contract-call SHAPE, event decode, status decode, and error mapping — no
live chain. The critical correctness point: commitBatch on-chain takes `requester`
(NOT provider — provider is msg.sender), so the client must NOT pass provider_address
into the contract call.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.batch_settlement_contract_client import (
    Web3SettlementContractClient,
)
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainRevertedError,
)
from prsm.settlement.client import SettlementContractClient


# ── Stub Web3 surface ─────────────────────────────────────────────────────────

_EXPECTED_BATCH_ID = b"\xbb" * 32
_EXPECTED_COMMIT_TS = 1_777_000_000


class _Fn:
    def __init__(self, name, args, parent):
        self.name, self.args, self._p = name, args, parent
        parent._calls.append((name, args))

    def build_transaction(self, overrides):
        return {"to": "0xbsr", "data": "0x", **overrides}

    def call(self):
        return self._p._views.get(self.name)


class _Functions:
    def __init__(self, parent):
        self._p = parent

    def __getattr__(self, name):
        return lambda *args: _Fn(name, args, self._p)


class _BatchCommittedEvent:
    def __init__(self, parent):
        self._p = parent

    def process_receipt(self, receipt):
        return self._p._committed_logs


class _Events:
    def __init__(self, parent):
        self._p = parent

    def BatchCommitted(self):
        return _BatchCommittedEvent(self._p)


class _FakeContract:
    def __init__(self):
        self._calls = []
        self._views = {
            "isFinalizable": True,
            # batches() struct getter — status is field index 7 (uint8).
            "batches": (
                "0xprovider", "0xrequester", b"\x00" * 32, 3, 100, 0,
                _EXPECTED_COMMIT_TS, 2,  # <-- index 7 = status (FINALIZED)
                0, b"\x00" * 32, 0, 0, 0, "0x0", "0x0", "0x0", "",
            ),
        }
        self.functions = _Functions(self)
        self.events = _Events(self)
        self._committed_logs = [
            {"args": {"batchId": _EXPECTED_BATCH_ID, "commitTimestamp": _EXPECTED_COMMIT_TS}},
        ]


class _FakeReceipt:
    def __init__(self, status=1):
        self.status = status


class _FakeAccount:
    address = "0x" + "11" * 20
    key = b"\x01" * 32


class _FakeEth:
    def __init__(self, *, chain_id=8453, send_ok=True, receipt_status=1, contract=None):
        self.chain_id = chain_id
        self.gas_price = 10 ** 9
        self._send_ok = send_ok
        self._receipt_status = receipt_status
        self._contract = contract
        self.account = MagicMock()
        signed = MagicMock()
        signed.raw_transaction = b"\xff" * 32
        self.account.sign_transaction.return_value = signed

    def get_transaction_count(self, addr, *_):
        return 9

    def contract(self, address, abi):
        return self._contract

    def send_raw_transaction(self, raw):
        if not self._send_ok:
            raise RuntimeError("network down")
        return b"\xab" * 32

    def wait_for_transaction_receipt(self, tx_hash, timeout=120):
        return _FakeReceipt(status=self._receipt_status)

    def get_transaction_receipt(self, tx_hash):
        mode = _CFG.get("receipt_lookup", "landed")
        if mode == "notfound":
            raise Exception("TransactionNotFound")
        if mode == "reverted":
            return _FakeReceipt(status=0)
        return _FakeReceipt(status=1)  # landed


_CFG = {"send_ok": True, "receipt_status": 1, "contract": None, "receipt_lookup": "landed"}


class _FakeWeb3:
    def __init__(self, *a, **k):
        self.eth = _FakeEth(
            send_ok=_CFG["send_ok"],
            receipt_status=_CFG["receipt_status"],
            contract=_CFG["contract"],
        )

    @staticmethod
    def to_checksum_address(addr):
        return addr

    @staticmethod
    def HTTPProvider(url):
        return object()


def _make_client(*, send_ok=True, receipt_status=1, with_key=True, receipt_lookup="landed"):
    contract = _FakeContract()
    _CFG.update(send_ok=send_ok, receipt_status=receipt_status, contract=contract,
                receipt_lookup=receipt_lookup)
    with patch(
        "prsm.economy.web3.batch_settlement_contract_client.Web3", _FakeWeb3,
    ), patch(
        "prsm.economy.web3.batch_settlement_contract_client.Account",
    ) as mock_account:
        mock_account.from_key.return_value = _FakeAccount()
        client = Web3SettlementContractClient(
            rpc_url="http://test",
            contract_address="0x" + "ab" * 20,
            private_key="0x" + "01" * 32 if with_key else None,
        )
    return client, contract


# ── Protocol conformance ──────────────────────────────────────────────────────

def test_satisfies_settlement_contract_client_protocol():
    """Structural conformance: the client exposes every SettlementContractClient
    Protocol method as an async coroutine (stronger than a name-only
    isinstance check against the non-runtime-checkable Protocol)."""
    import inspect
    client, _ = _make_client(with_key=False)
    protocol_methods = ("commit_batch", "is_finalizable", "finalize_batch", "get_batch_status")
    # The Protocol itself defines exactly these (sanity-check the import is live).
    assert all(hasattr(SettlementContractClient, m) for m in protocol_methods)
    for name in protocol_methods:
        method = getattr(client, name, None)
        assert method is not None, f"missing Protocol method {name}"
        assert inspect.iscoroutinefunction(method), f"{name} must be async"


def test_construction_with_and_without_key():
    client, _ = _make_client()
    assert client.address == "0x" + "11" * 20
    client2, _ = _make_client(with_key=False)
    assert client2.address is None


# ── commit_batch ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_batch_passes_requester_not_provider_and_parses_event():
    client, contract = _make_client()
    batch_id, ts = await client.commit_batch(
        provider_address="0xPROVIDER",          # must NOT be sent (msg.sender on-chain)
        requester_address="0xREQUESTER",
        merkle_root=b"\xcd" * 32,
        receipt_count=3,
        total_value_ftns=12345,
        tier_slash_rate_bps=700,
        consensus_group_id=b"\x00" * 32,
        metadata_uri="ipfs://x",
    )
    assert batch_id == _EXPECTED_BATCH_ID
    assert ts == _EXPECTED_COMMIT_TS

    commit_calls = [c for c in contract._calls if c[0] == "commitBatch"]
    assert len(commit_calls) == 1
    args = commit_calls[0][1]
    # On-chain commitBatch(requester, merkleRoot, receiptCount, totalValueFTNS,
    # tierSlashRateBps, consensusGroupId, metadataURI) — 7 args, provider absent.
    assert args == ("0xREQUESTER", b"\xcd" * 32, 3, 12345, 700, b"\x00" * 32, "ipfs://x")
    assert "0xPROVIDER" not in args


@pytest.mark.asyncio
async def test_commit_batch_without_key_raises():
    client, _ = _make_client(with_key=False)
    with pytest.raises(RuntimeError, match="private_key"):
        await client.commit_batch(
            provider_address="0xP", requester_address="0xR", merkle_root=b"\x00" * 32,
            receipt_count=1, total_value_ftns=1, tier_slash_rate_bps=0,
            consensus_group_id=b"\x00" * 32, metadata_uri="",
        )


@pytest.mark.asyncio
async def test_commit_batch_broadcast_failure_raises():
    client, _ = _make_client(send_ok=False)
    with pytest.raises(BroadcastFailedError):
        await client.commit_batch(
            provider_address="0xP", requester_address="0xR", merkle_root=b"\x00" * 32,
            receipt_count=1, total_value_ftns=1, tier_slash_rate_bps=0,
            consensus_group_id=b"\x00" * 32, metadata_uri="",
        )


@pytest.mark.asyncio
async def test_commit_batch_revert_raises():
    client, _ = _make_client(receipt_status=0)
    with pytest.raises(OnChainRevertedError):
        await client.commit_batch(
            provider_address="0xP", requester_address="0xR", merkle_root=b"\x00" * 32,
            receipt_count=1, total_value_ftns=1, tier_slash_rate_bps=0,
            consensus_group_id=b"\x00" * 32, metadata_uri="",
        )


# ── is_finalizable / finalize_batch / get_batch_status ────────────────────────

@pytest.mark.asyncio
async def test_is_finalizable_reads_view():
    client, contract = _make_client(with_key=False)
    assert await client.is_finalizable(b"\xaa" * 32) is True
    contract._views["isFinalizable"] = False
    assert await client.is_finalizable(b"\xaa" * 32) is False


@pytest.mark.asyncio
async def test_get_batch_status_reads_struct_field_7():
    client, _ = _make_client(with_key=False)
    # The fake batches() tuple has status=2 (FINALIZED) at index 7.
    assert await client.get_batch_status(b"\xaa" * 32) == 2


@pytest.mark.asyncio
async def test_finalize_batch_happy_and_revert():
    client, contract = _make_client()
    await client.finalize_batch(b"\xaa" * 32)  # status 1 → no raise
    finalize_calls = [c for c in contract._calls if c[0] == "finalizeBatch"]
    assert finalize_calls and finalize_calls[0][1] == (b"\xaa" * 32,)

    client_revert, _ = _make_client(receipt_status=0)
    with pytest.raises(OnChainRevertedError):
        await client_revert.finalize_batch(b"\xaa" * 32)


@pytest.mark.asyncio
async def test_finalize_batch_without_key_raises():
    client, _ = _make_client(with_key=False)
    with pytest.raises(RuntimeError, match="private_key"):
        await client.finalize_batch(b"\xaa" * 32)


# ── get_committed_batch_for_tx (reconcile support, sp1022) ────────────────────

@pytest.mark.asyncio
async def test_get_committed_batch_for_tx_landed():
    """A mined commit tx that emitted BatchCommitted → recover (batch_id, ts)."""
    client, _ = _make_client(with_key=False, receipt_lookup="landed")
    result = await client.get_committed_batch_for_tx("0xLANDED")
    assert result == (_EXPECTED_BATCH_ID, _EXPECTED_COMMIT_TS)


@pytest.mark.asyncio
async def test_get_committed_batch_for_tx_not_found_returns_none():
    """Unknown / not-yet-mined tx (get_transaction_receipt raises) → None
    (the reconcile caller leaves it quarantined, never re-commits)."""
    client, _ = _make_client(with_key=False, receipt_lookup="notfound")
    assert await client.get_committed_batch_for_tx("0xUNKNOWN") is None


@pytest.mark.asyncio
async def test_get_committed_batch_for_tx_reverted_returns_none():
    """A reverted commit tx (receipt status != 1) → None."""
    client, _ = _make_client(with_key=False, receipt_lookup="reverted")
    assert await client.get_committed_batch_for_tx("0xREVERTED") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
