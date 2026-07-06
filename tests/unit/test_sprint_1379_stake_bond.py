"""Sprint 1379 — one-call operator stake (approve FTNS + bond) for the PRODUCTION trust path.

The mainnet GPU canary ran under mock trust because production trust's filter_pool drops nodes
without on-chain stake. approve_and_bond + `prsm node stake-bond` let an operator post real
StakeBond stake in one command so the node passes the stake-eligibility filter. web3 is mocked so
the ordering/validation is verified offline.
"""
import threading
import types

import pytest

from prsm.economy.web3.stake_manager import StakeManagerClient


class _Call:
    def __init__(self, name, calls, ret=None):
        self.name = name
        self.calls = calls
        self.ret = ret
        calls.append(name)

    def call(self):
        return self.ret

    def build_transaction(self, _ov):
        return {"fn": self.name}


def _functions(calls, allowance_ret):
    class _F:
        def allowance(self, _owner, _spender):
            return _Call("allowance", calls, ret=allowance_ret)

        def approve(self, _spender, _amt):
            return _Call("approve", calls)

        def bond(self, _amt, _tier):
            return _Call("bond", calls)

    return _F()


def _client(allowance_ret, sent):
    c = StakeManagerClient(
        rpc_url="http://localhost:8545", contract_address="0x" + "11" * 20, private_key=None)
    c._account = types.SimpleNamespace(address="0x" + "aa" * 20, key=b"\x00" * 32)
    c._tx_lock = threading.Lock()
    calls = []
    c.contract = types.SimpleNamespace(functions=_functions(calls, None))       # StakeBond.bond
    c.web3.eth.contract = lambda address, abi: types.SimpleNamespace(            # FTNS approve/allowance
        functions=_functions(calls, allowance_ret))
    c._tx_overrides = lambda: {}
    c._sign_and_send = lambda tx: (sent.append(tx) or (f"0x{len(sent):064x}", "CONFIRMED"))
    return c, calls


def test_approve_then_bond_when_allowance_short():
    sent = []
    c, calls = _client(allowance_ret=0, sent=sent)                     # allowance 0 < amount
    _tx, status = c.approve_and_bond(
        ftns_token_address="0x" + "22" * 20, amount_wei=10 ** 18, tier_slash_rate_bps=100)
    assert calls == ["allowance", "approve", "bond"]                   # approved, then bonded
    assert len(sent) == 2 and status == "CONFIRMED"


def test_skips_approve_when_allowance_sufficient():
    sent = []
    c, calls = _client(allowance_ret=10 ** 18, sent=sent)             # allowance >= amount
    c.approve_and_bond(
        ftns_token_address="0x" + "22" * 20, amount_wei=10 ** 18, tier_slash_rate_bps=0)
    assert calls == ["allowance", "bond"]                             # no approve
    assert len(sent) == 1                                            # bond only


@pytest.mark.parametrize("amount,tier", [(0, 0), (-1, 0), (10 ** 18, 20000), (10 ** 18, -1)])
def test_rejects_bad_amount_or_tier(amount, tier):
    c, _ = _client(0, [])
    with pytest.raises(ValueError):
        c.approve_and_bond(ftns_token_address="0x" + "22" * 20, amount_wei=amount, tier_slash_rate_bps=tier)


def test_cli_requires_key_env(monkeypatch):
    from click.testing import CliRunner

    from prsm.cli import main
    monkeypatch.delenv("FTNS_WALLET_PRIVATE_KEY", raising=False)
    r = CliRunner().invoke(main, ["node", "stake-bond", "5"])
    assert r.exit_code != 0
    assert "FTNS_WALLET_PRIVATE_KEY" in r.output          # never touches the chain without a key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
