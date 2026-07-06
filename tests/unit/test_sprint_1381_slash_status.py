"""Sprint 1381 — operator slash/challenge visibility.

get_slash_events decodes StakeBond Slashed logs for a provider (indexed → server-side filtered),
chunked under the Base public-RPC eth_getLogs cap. `prsm node slash-status` surfaces current stake +
those events. web3 is mocked so decode/chunking/CLI are verified offline.
"""
import types

from click.testing import CliRunner

import prsm.economy.web3.stake_manager as sm
from prsm.cli import main
from prsm.economy.web3.stake_manager import StakeManagerClient


class _TxHash:
    def hex(self):
        return "0x" + "ab" * 32


def _entry(block, amount):
    return (block, {
        "args": {"challenger": "0x" + "cc" * 20, "reasonId": b"\x11" * 32,
                 "slashAmount": amount, "challengerBounty": amount // 10,
                 "foundationShare": amount // 5},
        "blockNumber": block, "transactionHash": _TxHash()})


def _client_with_events(head, entries):
    c = StakeManagerClient(
        rpc_url="http://localhost:8545", contract_address="0x" + "11" * 20, private_key=None)
    c.web3 = types.SimpleNamespace(eth=types.SimpleNamespace(block_number=head))
    windows = []

    class _Filter:
        def __init__(self, lo, hi):
            self.lo, self.hi = lo, hi

        def get_all_entries(self):
            return [e for (b, e) in entries if self.lo <= b <= self.hi]

    class _Event:
        def create_filter(self, from_block, to_block, argument_filters):
            windows.append((from_block, to_block))
            return _Filter(from_block, to_block)

    c.contract = types.SimpleNamespace(events=types.SimpleNamespace(Slashed=lambda: _Event()))
    return c, windows


def test_get_slash_events_decodes_and_chunks():
    c, windows = _client_with_events(head=10_000, entries=[_entry(9_500, 10 ** 18)])
    evs = c.get_slash_events("0x" + "aa" * 20, lookback_blocks=15_000)   # from_block=0..10000
    assert len(windows) >= 2                                             # 0-8999, 9000-10000 chunked
    assert len(evs) == 1
    ev = evs[0]
    assert ev.block_number == 9_500 and ev.slash_amount_wei == 10 ** 18
    assert ev.challenger == "0x" + "cc" * 20
    assert ev.reason_id == "0x" + "11" * 32
    assert ev.tx_hash == "0x" + "ab" * 32


def test_get_slash_events_empty_window_is_clean():
    c, _ = _client_with_events(head=5_000, entries=[])
    assert c.get_slash_events("0x" + "aa" * 20, from_block=0) == []


# ── CLI ──
class _Rec:
    status = "BONDED"
    amount_wei = 10 ** 18
    tier_slash_rate_bps = 100


class _CleanClient:
    def __init__(self, *a, **k):
        pass

    def stake_of(self, _op):
        return _Rec()

    def get_slash_events(self, _op, from_block=None):
        return []


class _Ev:
    block_number = 9_500
    tx_hash = "0x" + "ab" * 32
    challenger = "0x" + "cc" * 20
    reason_id = "0x" + "11" * 32
    slash_amount_wei = 5 * 10 ** 17


class _SlashedClient(_CleanClient):
    def get_slash_events(self, _op, from_block=None):
        return [_Ev()]


def _env(monkeypatch):
    monkeypatch.setenv("PRSM_OPERATOR_ADDRESS", "0x" + "aa" * 20)
    monkeypatch.setenv("PRSM_STAKE_BOND_ADDRESS", "0x" + "11" * 20)
    monkeypatch.setenv("PRSM_BASE_RPC_URL", "http://localhost:8545")


def test_slash_status_requires_address(monkeypatch):
    monkeypatch.delenv("PRSM_OPERATOR_ADDRESS", raising=False)
    r = CliRunner().invoke(main, ["node", "slash-status"])
    assert r.exit_code != 0 and "operator address" in r.output


def test_slash_status_clean(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(sm, "StakeManagerClient", _CleanClient)
    r = CliRunner().invoke(main, ["node", "slash-status"])
    assert r.exit_code == 0, r.output
    assert "stake is clean" in r.output


def test_slash_status_with_events(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(sm, "StakeManagerClient", _SlashedClient)
    r = CliRunner().invoke(main, ["node", "slash-status"])
    assert r.exit_code == 0, r.output
    assert "slash event(s)" in r.output and "9500" in r.output


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
