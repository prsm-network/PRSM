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


def _entry(block, amount, provider="0x" + "aa" * 20):
    # sp1424: `provider` is an INDEXED param of the Slashed event, so real web3-decoded logs
    # always carry args["provider"]. The mock omitted it because the pre-1424 code read the
    # provider from the filter argument instead of the log; get_all_slash_events (no filter)
    # forced the shared scan to read it per-log, which is what real decoding does.
    return (block, {
        "args": {"provider": provider, "challenger": "0x" + "cc" * 20,
                 "reasonId": b"\x11" * 32, "slashAmount": amount,
                 "challengerBounty": amount // 10, "foundationShare": amount // 5},
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
    # sp1424 — provider is now decoded from the log itself (checksummed), not the filter arg.
    from web3 import Web3
    assert ev.provider == Web3.to_checksum_address("0x" + "aa" * 20)


def test_get_slash_events_empty_window_is_clean():
    c, _ = _client_with_events(head=5_000, entries=[])
    assert c.get_slash_events("0x" + "aa" * 20, from_block=0) == []


def test_get_all_slash_events_scans_without_a_provider_filter(monkeypatch):
    """sp1424 — the reputation bridge needs EVERY slashed provider, not just self, so
    get_all_slash_events must query with argument_filters=None and decode each log's own
    provider address."""
    filters_seen = []
    c, _ = _client_with_events(head=10_000, entries=[
        _entry(9_400, 10 ** 18, provider="0x" + "aa" * 20),
        _entry(9_600, 2 * 10 ** 18, provider="0x" + "bb" * 20),
    ])
    # capture the argument_filters the scan passes
    orig_event = c.contract.events.Slashed

    class _CapturingEvent:
        def __init__(self, inner):
            self._inner = inner

        def create_filter(self, from_block, to_block, argument_filters):
            filters_seen.append(argument_filters)
            return self._inner.create_filter(
                from_block=from_block, to_block=to_block, argument_filters=argument_filters,
            )

    c.contract.events.Slashed = lambda: _CapturingEvent(orig_event())

    evs = c.get_all_slash_events(lookback_blocks=15_000)
    from web3 import Web3
    providers = sorted(e.provider for e in evs)
    assert providers == sorted([
        Web3.to_checksum_address("0x" + "aa" * 20),
        Web3.to_checksum_address("0x" + "bb" * 20),
    ]), "get_all_slash_events must decode DIFFERENT providers from the logs, not a fixed one"
    assert filters_seen and all(f is None for f in filters_seen), (
        "get_all_slash_events must NOT filter by provider — it needs all of them"
    )


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
