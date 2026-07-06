"""Sprint 1381 — per-batch challenge visibility (earlier signal than slash-status).

ReceiptChallenged indexes batchId (not provider), so get_challenges_for_provider cross-references the
operator's own BatchCommitted batchIds against ReceiptChallenged events. web3 mocked offline.
"""
import types

from click.testing import CliRunner

import prsm.economy.web3.batch_settlement_contract_client as bscc
from prsm.cli import main
from prsm.economy.web3.batch_settlement_contract_client import (
    Web3SettlementContractClient,
)


class _TxHash:
    def hex(self):
        return "0x" + "dd" * 32


def _chal(batch_id, reason=1, value=5 * 10 ** 17, block=50):
    return {
        "args": {"batchId": batch_id, "receiptLeafHash": b"\x22" * 32,
                 "challenger": "0x" + "cc" * 20, "reason": reason,
                 "invalidatedValueFTNS": value},
        "blockNumber": block, "transactionHash": _TxHash()}


def _client(head, committed_batches, challenges, status=1):
    c = Web3SettlementContractClient(
        rpc_url="http://localhost:8545", contract_address="0x" + "11" * 20)
    c.web3 = types.SimpleNamespace(eth=types.SimpleNamespace(block_number=head))

    class _F:
        def __init__(self, entries):
            self.entries = entries

        def get_all_entries(self):
            return self.entries

    class _CommittedEvent:
        def create_filter(self, from_block, to_block, argument_filters=None):
            return _F([{"args": {"batchId": b}} for b in committed_batches])

    class _ChallengedEvent:
        def create_filter(self, from_block, to_block, argument_filters=None):
            return _F(challenges)

    class _Batches:
        def call(self):
            row = [0] * 8
            row[7] = status
            return row

    c.contract = types.SimpleNamespace(
        events=types.SimpleNamespace(
            BatchCommitted=lambda: _CommittedEvent(),
            ReceiptChallenged=lambda: _ChallengedEvent()),
        functions=types.SimpleNamespace(batches=lambda _bid: _Batches()))
    return c


def test_returns_only_challenges_for_own_batches():
    mine = b"\x01" * 32
    c = _client(head=100, committed_batches=[mine],
                challenges=[_chal(mine, reason=1), _chal(b"\x99" * 32, reason=0)], status=1)
    evs = c.get_challenges_for_provider("0x" + "aa" * 20)
    assert len(evs) == 1                                    # the foreign batch's challenge dropped
    ev = evs[0]
    assert ev.batch_id == "0x" + mine.hex()
    assert ev.reason == "INVALID_SIGNATURE" and ev.reason_code == 1
    assert ev.invalidated_value_ftns == 0.5 and ev.batch_status == 1
    assert ev.challenger == "0x" + "cc" * 20 and ev.tx_hash == "0x" + "dd" * 32


def test_no_committed_batches_returns_empty():
    c = _client(head=100, committed_batches=[], challenges=[_chal(b"\x01" * 32)])
    assert c.get_challenges_for_provider("0x" + "aa" * 20) == []


def test_unknown_reason_code_labeled():
    mine = b"\x01" * 32
    c = _client(head=100, committed_batches=[mine], challenges=[_chal(mine, reason=9)])
    assert c.get_challenges_for_provider("0x" + "aa" * 20)[0].reason == "UNKNOWN(9)"


# ── CLI ──
class _CleanC:
    def __init__(self, *a, **k):
        pass

    def get_challenges_for_provider(self, _op, from_block=None):
        return []


class _ChalEv:
    batch_id = "0x" + "01" * 32
    reason = "INVALID_SIGNATURE"
    reason_code = 1
    invalidated_value_ftns = 0.5
    challenger = "0x" + "cc" * 20
    batch_status = 1
    receipt_leaf_hash = "0x" + "22" * 32
    block_number = 50
    tx_hash = "0x" + "dd" * 32


class _WithC(_CleanC):
    def get_challenges_for_provider(self, _op, from_block=None):
        return [_ChalEv()]


def _env(monkeypatch):
    monkeypatch.setenv("PRSM_OPERATOR_ADDRESS", "0x" + "aa" * 20)
    monkeypatch.setenv("PRSM_SETTLEMENT_REGISTRY_ADDRESS", "0x" + "11" * 20)
    monkeypatch.setenv("PRSM_BASE_RPC_URL", "http://localhost:8545")


def test_cli_requires_address(monkeypatch):
    monkeypatch.delenv("PRSM_OPERATOR_ADDRESS", raising=False)
    r = CliRunner().invoke(main, ["node", "challenge-status"])
    assert r.exit_code != 0 and "operator address" in r.output


def test_cli_clean(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bscc, "Web3SettlementContractClient", _CleanC)
    r = CliRunner().invoke(main, ["node", "challenge-status"])
    assert r.exit_code == 0, r.output
    assert "clean" in r.output


def test_cli_with_challenges(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bscc, "Web3SettlementContractClient", _WithC)
    r = CliRunner().invoke(main, ["node", "challenge-status"])
    assert r.exit_code == 0, r.output
    assert "challenge(s)" in r.output and "INVALID_SIGNATURE" in r.output


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
