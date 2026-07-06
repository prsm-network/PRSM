"""Sprint 1380 — stake unbond/withdraw CLIs (round out the StakeBond lifecycle after sp1379 bond).

bond -> request-unbond (BONDED->UNBONDING) -> (unbond delay) -> withdraw (UNBONDING->WITHDRAWN).
The client methods existed; these tests cover the new `prsm node stake-unbond` / `stake-withdraw`
wrappers — that they call the right method, report state, and refuse without a key. _resolve_stake_client
is patched so no chain is touched.
"""
from click.testing import CliRunner

import prsm.cli as cli_mod
from prsm.cli import main


class _FakeRec:
    def __init__(self, status, amount_wei=0, unbond_eligible_at=0):
        self.status = status
        self.amount_wei = amount_wei
        self.unbond_eligible_at = unbond_eligible_at


class _FakeClient:
    address = "0x" + "aa" * 20

    def __init__(self, calls):
        self.calls = calls

    def request_unbond(self):
        self.calls.append("request_unbond")
        return ("0xreqhash", "CONFIRMED")

    def withdraw(self):
        self.calls.append("withdraw")
        return ("0xwdhash", "CONFIRMED")

    def stake_of(self, _addr):
        return _FakeRec("UNBONDING", amount_wei=10 ** 18, unbond_eligible_at=123)


def _patch(monkeypatch, calls):
    monkeypatch.setattr(
        cli_mod, "_resolve_stake_client",
        lambda require_ftns=False: (_FakeClient(calls), {"stake_bond": "0xSB", "rpc": "r", "ftns": "f"}))


def test_stake_unbond_calls_request_unbond(monkeypatch):
    calls = []
    _patch(monkeypatch, calls)
    r = CliRunner().invoke(main, ["node", "stake-unbond"])
    assert r.exit_code == 0, r.output
    assert calls == ["request_unbond"]
    assert "unbond requested" in r.output and "0xreqhash" in r.output


def test_stake_withdraw_calls_withdraw(monkeypatch):
    calls = []
    _patch(monkeypatch, calls)
    r = CliRunner().invoke(main, ["node", "stake-withdraw"])
    assert r.exit_code == 0, r.output
    assert calls == ["withdraw"]
    assert "withdraw ->" in r.output and "0xwdhash" in r.output


def test_unbond_requires_key(monkeypatch):
    monkeypatch.delenv("FTNS_WALLET_PRIVATE_KEY", raising=False)
    r = CliRunner().invoke(main, ["node", "stake-unbond"])
    assert r.exit_code != 0 and "FTNS_WALLET_PRIVATE_KEY" in r.output


def test_withdraw_requires_key(monkeypatch):
    monkeypatch.delenv("FTNS_WALLET_PRIVATE_KEY", raising=False)
    r = CliRunner().invoke(main, ["node", "stake-withdraw"])
    assert r.exit_code != 0 and "FTNS_WALLET_PRIVATE_KEY" in r.output


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
