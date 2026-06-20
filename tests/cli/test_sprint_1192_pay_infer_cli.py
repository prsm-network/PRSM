"""Sprint 1192 — `prsm compute pay-infer` + `prsm wallet deposit` CLI.

The terminal front door for the sp1189 pay-for-inference SDK: a non-operator can
fund their on-chain EscrowPool (`wallet deposit`) and pay for one inference
(`compute pay-infer`, which signs an EIP-712 PaymentAuthorization with the wallet
key) without writing Python. Both wrap PRSMClient (sp1189); the wallet key comes
from PRIVATE_KEY / FTNS_WALLET_PRIVATE_KEY env, never the command line.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from prsm.cli import main

_TEST_KEY = "0x" + "11" * 32  # a valid 32-byte secp256k1 key (eth_account derives addr)


@pytest.fixture
def runner():
    return CliRunner()


def _fake_client_class(*, pay_result=None, deposit_result="0xtx", raises=None, captured=None):
    """Build a fake PRSMClient class whose async methods return canned values and
    optionally record the kwargs they were called with."""

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def pay_and_infer(self, prompt, **kwargs):
            if captured is not None:
                captured.update(prompt=prompt, **kwargs)
            if raises is not None:
                raise raises
            return pay_result

        async def deposit_escrow(self, **kwargs):
            if captured is not None:
                captured.update(**kwargs)
            if raises is not None:
                raise raises
            return deposit_result

        async def close(self):
            pass

    return _FakeClient


def _patch_client(monkeypatch, **kw):
    monkeypatch.setattr("prsm.sdk.client.PRSMClient", _fake_client_class(**kw))


# ── compute pay-infer ────────────────────────────────────────────────────────────────

def test_pay_infer_no_key_exits_1(runner, monkeypatch):
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    monkeypatch.delenv("FTNS_WALLET_PRIVATE_KEY", raising=False)
    r = runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi"])
    assert r.exit_code == 1
    assert "no signing key" in r.output.lower()


def test_pay_infer_success(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_client(monkeypatch, pay_result={
        "success": True, "output": " the", "ftns_charged": 0.5})
    r = runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi", "--budget", "1"])
    assert r.exit_code == 0, r.output
    assert "the" in r.output
    assert "charged" in r.output.lower()


def test_pay_infer_rejected_402_body(runner, monkeypatch):
    # A rejected authorization returns the server's 402 detail body (no output).
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_client(monkeypatch, pay_result={"detail": "insufficient escrow balance"})
    r = runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi"])
    assert r.exit_code == 1
    assert "rejected" in r.output.lower()
    assert "insufficient escrow" in r.output


def test_pay_infer_json_format(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_client(monkeypatch, pay_result={"success": True, "output": "x", "ftns_charged": 1})
    r = runner.invoke(
        main, ["compute", "pay-infer", "--prompt", "hi", "--format", "json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output.strip().splitlines()[-1])
    assert payload["output"] == "x"


def test_pay_infer_connection_error_exits_2(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_client(monkeypatch, raises=ConnectionError("Connection refused"))
    r = runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi"])
    assert r.exit_code == 2
    assert "cannot reach" in r.output.lower()


def test_pay_infer_chain_id_from_network(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    cap_testnet = {}
    _patch_client(monkeypatch, pay_result={"success": True, "output": "x"}, captured=cap_testnet)
    runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi", "--network", "testnet"])
    assert cap_testnet["chain_id"] == 84532

    cap_main = {}
    _patch_client(monkeypatch, pay_result={"success": True, "output": "x"}, captured=cap_main)
    runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi", "--network", "mainnet"])
    assert cap_main["chain_id"] == 8453


def test_pay_infer_max_spend_defaults_to_budget(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    cap = {}
    _patch_client(monkeypatch, pay_result={"success": True, "output": "x"}, captured=cap)
    runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi", "--budget", "3"])
    assert cap["max_spend_ftns"] == 3.0
    assert cap["budget_ftns"] == 3.0


# ── wallet deposit ─────────────────────────────────────────────────────────────────────

def test_wallet_deposit_no_key_exits_1(runner, monkeypatch):
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    monkeypatch.delenv("FTNS_WALLET_PRIVATE_KEY", raising=False)
    r = runner.invoke(main, ["wallet", "deposit", "--amount", "5", "--yes"])
    assert r.exit_code == 1
    assert "no signing key" in r.output.lower()


def test_wallet_deposit_success(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    cap = {}
    _patch_client(monkeypatch, deposit_result="0xdeadbeef", captured=cap)
    r = runner.invoke(
        main, ["wallet", "deposit", "--amount", "5", "--network", "testnet", "--yes"])
    assert r.exit_code == 0, r.output
    assert "0xdeadbeef" in r.output
    assert cap["amount_ftns"] == 5.0
    assert cap["network"] == "testnet"


def test_wallet_deposit_nonpositive_amount(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    r = runner.invoke(main, ["wallet", "deposit", "--amount", "0", "--yes"])
    assert r.exit_code == 1
    assert "> 0" in r.output


def test_wallet_deposit_abort_without_yes(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_client(monkeypatch, deposit_result="0xtx")
    r = runner.invoke(main, ["wallet", "deposit", "--amount", "5"], input="n\n")
    assert r.exit_code == 1
    assert "aborted" in r.output.lower()


def test_wallet_deposit_broadcast_failure_exits_1(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_client(monkeypatch, raises=RuntimeError("nonce too low"))
    r = runner.invoke(main, ["wallet", "deposit", "--amount", "5", "--yes"])
    assert r.exit_code == 1
    assert "deposit failed" in r.output.lower()


# ── compute pay-infer --dry-run (sp1194 preflight) ───────────────────────────────────

def _patch_info(monkeypatch, payload=None, raise_exc=None):
    import httpx

    def _get(u, timeout=None, headers=None):
        if raise_exc is not None:
            raise raise_exc

        class _R:
            def json(self_inner):
                return payload
        return _R()
    monkeypatch.setattr(httpx, "get", _get)


def _patch_escrow(monkeypatch, balance_wei):
    class _FakeEscrow:
        def __init__(self, *a, **k):
            pass

        async def balance_of(self, addr):
            return balance_wei
    monkeypatch.setattr(
        "prsm.economy.web3.escrow_pool_client.EscrowPoolClient", _FakeEscrow)


def test_dry_run_all_pass(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_info(monkeypatch, payload={"operator_address": "0xOperator"})
    _patch_escrow(monkeypatch, balance_wei=10 * 10**18)  # 10 FTNS ≥ budget 1
    # also assert pay_and_infer is NOT called (broadcast nothing)
    cap = {}
    _patch_client(monkeypatch, pay_result={"output": "x"}, captured=cap)
    r = runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output and "ready to pay" in r.output.lower()
    assert cap == {}, "dry-run must not call pay_and_infer"


def test_dry_run_underfunded_escrow_fails(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_info(monkeypatch, payload={"operator_address": "0xOperator"})
    _patch_escrow(monkeypatch, balance_wei=0)  # 0 FTNS < budget
    r = runner.invoke(
        main, ["compute", "pay-infer", "--prompt", "hi", "--budget", "2", "--dry-run"])
    assert r.exit_code == 1
    assert "FAIL" in r.output
    assert "wallet deposit" in r.output  # actionable next step


def test_dry_run_no_operator_payee_fails(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_info(monkeypatch, payload={})  # operator advertises no payee
    _patch_escrow(monkeypatch, balance_wei=10 * 10**18)
    r = runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi", "--dry-run"])
    assert r.exit_code == 1
    assert "no payment address" in r.output.lower()


def test_dry_run_explicit_provider_skips_info(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    # /info would raise if hit — explicit --provider-address must skip it
    _patch_info(monkeypatch, raise_exc=RuntimeError("should not be called"))
    _patch_escrow(monkeypatch, balance_wei=10 * 10**18)
    r = runner.invoke(main, [
        "compute", "pay-infer", "--prompt", "hi", "--dry-run",
        "--provider-address", "0xExplicit"])
    assert r.exit_code == 0, r.output
    assert "0xExplicit" in r.output


def test_dry_run_json_format(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_info(monkeypatch, payload={"operator_address": "0xOperator"})
    _patch_escrow(monkeypatch, balance_wei=10 * 10**18)
    r = runner.invoke(
        main, ["compute", "pay-infer", "--prompt", "hi", "--dry-run", "--format", "json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert any(c["level"] == "pass" for c in payload["checks"])


def test_dry_run_operator_rejects_payment_fails(runner, monkeypatch):
    # sp1196 — operator advertises an address but does NOT accept requester payment.
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_info(monkeypatch, payload={
        "operator_address": "0xOperator", "requester_payment_accepted": False})
    _patch_escrow(monkeypatch, balance_wei=10 * 10**18)
    r = runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi", "--dry-run"])
    assert r.exit_code == 1
    assert "does not accept requester payment" in r.output.lower()


def test_dry_run_operator_accepts_payment_passes(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    _patch_info(monkeypatch, payload={
        "operator_address": "0xOperator", "requester_payment_accepted": True})
    _patch_escrow(monkeypatch, balance_wei=10 * 10**18)
    r = runner.invoke(main, ["compute", "pay-infer", "--prompt", "hi", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "accepts requester payment" in r.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
