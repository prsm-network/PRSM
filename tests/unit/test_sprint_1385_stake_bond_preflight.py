"""Sprint 1385 — stake-bond funding preflight (`--dry-run`).

funding_preflight is read-only (FTNS balance / ETH gas / allowance + go/no-go); the CLI --dry-run
prints it and exits 1 on NO-GO without approving or bonding. web3 + client mocked offline.
"""
import types

from click.testing import CliRunner

import prsm.cli as cli_mod
from prsm.cli import main
from prsm.economy.web3.stake_manager import (
    StakeManagerClient,
    StakeRecord,
    StakeStatus,
)


def _pf_client(ftns_bal, allowance, eth_bal, stake_amt=0):
    c = StakeManagerClient(
        rpc_url="http://localhost:8545", contract_address="0x" + "11" * 20, private_key=None)
    c._account = types.SimpleNamespace(address="0x" + "aa" * 20, key=b"\x00" * 32)

    class _Fns:
        def balanceOf(self, _a):
            return types.SimpleNamespace(call=lambda: ftns_bal)

        def allowance(self, _o, _s):
            return types.SimpleNamespace(call=lambda: allowance)

    fake_erc20 = types.SimpleNamespace(functions=_Fns())
    c.web3 = types.SimpleNamespace(eth=types.SimpleNamespace(
        contract=lambda address, abi: fake_erc20,
        get_balance=lambda _a: eth_bal))
    c.stake_of = lambda _p: StakeRecord(
        amount_wei=stake_amt, bonded_at_unix=0, unbond_eligible_at=0,
        status=StakeStatus.BONDED if stake_amt else StakeStatus.NONE, tier_slash_rate_bps=0)
    return c


def test_preflight_go():
    pf = _pf_client(ftns_bal=2 * 10 ** 18, allowance=0, eth_bal=10 ** 15).funding_preflight(
        ftns_token_address="0x" + "22" * 20, amount_wei=10 ** 18)
    assert pf["sufficient_ftns"] and pf["has_gas"] and pf["needs_approve"] and pf["go"]


def test_preflight_nogo_insufficient_ftns():
    pf = _pf_client(ftns_bal=10 ** 17, allowance=0, eth_bal=10 ** 15).funding_preflight(
        ftns_token_address="0x" + "22" * 20, amount_wei=10 ** 18)
    assert not pf["sufficient_ftns"] and not pf["go"]


def test_preflight_nogo_no_gas():
    pf = _pf_client(ftns_bal=2 * 10 ** 18, allowance=2 * 10 ** 18, eth_bal=0).funding_preflight(
        ftns_token_address="0x" + "22" * 20, amount_wei=10 ** 18)
    assert not pf["has_gas"] and not pf["needs_approve"] and not pf["go"]


# ── CLI --dry-run ──
class _DryClient:
    address = "0x" + "aa" * 20

    def __init__(self, go):
        self._go = go
        self.bonded = False

    def funding_preflight(self, *, ftns_token_address, amount_wei):
        return {
            "operator": self.address, "amount_wei": amount_wei,
            "ftns_balance_wei": 2 * 10 ** 18, "sufficient_ftns": self._go,
            "eth_balance_wei": 10 ** 15, "has_gas": True, "allowance_wei": 0,
            "needs_approve": True, "current_stake_wei": 0, "current_stake_status": "NONE",
            "go": self._go}

    def approve_and_bond(self, **_k):
        self.bonded = True
        return ("0xtx", "CONFIRMED")


def _patch(monkeypatch, dc):
    monkeypatch.setattr(
        cli_mod, "_resolve_stake_client",
        lambda require_ftns=False: (dc, {"ftns": "0xf", "stake_bond": "0xsb", "rpc": "r"}))


def test_cli_dry_run_go_does_not_bond(monkeypatch):
    dc = _DryClient(go=True)
    _patch(monkeypatch, dc)
    r = CliRunner().invoke(main, ["node", "stake-bond", "1", "--dry-run"])
    assert r.exit_code == 0 and "[GO]" in r.output
    assert not dc.bonded                               # preflight never bonds


def test_cli_dry_run_nogo_exits_1(monkeypatch):
    dc = _DryClient(go=False)
    _patch(monkeypatch, dc)
    r = CliRunner().invoke(main, ["node", "stake-bond", "1", "--dry-run"])
    assert r.exit_code == 1 and "[NO-GO]" in r.output
    assert not dc.bonded


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
