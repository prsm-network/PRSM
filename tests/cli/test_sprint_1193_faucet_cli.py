"""Sprint 1193 — `prsm wallet faucet` CLI (testnet self-funding, terminal front door).

Completes the end-to-end terminal flow on Base Sepolia: faucet → deposit →
pay-infer. sp1190 added the on-chain testnet faucet HTTP endpoint
(/ftns/faucet/onchain); this command calls it so a new user gets testnet FTNS to
their address without curl. No signing key needed (the faucet operator signs the
transfer) — the command only sends the destination address.
"""
from __future__ import annotations

import json

import httpx
import pytest
from click.testing import CliRunner

from prsm.cli import main

_TEST_KEY = "0x" + "11" * 32
_ADDR = "0x" + "44" * 20


@pytest.fixture
def runner():
    return CliRunner()


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _patch_post(monkeypatch, resp=None, raise_connect=False, captured=None):
    def _fake_post(url, json=None, timeout=None):
        if captured is not None:
            captured.update(url=url, body=json)
        if raise_connect:
            raise httpx.ConnectError("refused")
        return resp
    monkeypatch.setattr(httpx, "post", _fake_post)


def test_faucet_success_with_explicit_address(runner, monkeypatch):
    cap = {}
    _patch_post(monkeypatch, resp=_Resp(200, {
        "status": "dispensed", "tx_hash": "0xabc", "dispensed_ftns": 100,
        "recipient": _ADDR, "faucet_address": "0xFaucet", "network": "base-sepolia",
    }), captured=cap)
    r = runner.invoke(main, ["wallet", "faucet", "--address", _ADDR, "--network", "testnet"])
    assert r.exit_code == 0, r.output
    assert "0xabc" in r.output and "100" in r.output
    assert cap["body"]["destination_address"] == _ADDR
    assert "amount" not in cap["body"]  # omitted → operator default cap


def test_faucet_passes_amount_when_given(runner, monkeypatch):
    cap = {}
    _patch_post(monkeypatch, resp=_Resp(200, {
        "dispensed_ftns": 25, "tx_hash": "0x1", "recipient": _ADDR,
        "faucet_address": "0xF", "network": "base-sepolia"}), captured=cap)
    r = runner.invoke(
        main, ["wallet", "faucet", "--address", _ADDR, "--amount", "25"])
    assert r.exit_code == 0, r.output
    assert cap["body"]["amount"] == 25.0


def test_faucet_derives_address_from_key(runner, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", _TEST_KEY)
    cap = {}
    _patch_post(monkeypatch, resp=_Resp(200, {
        "dispensed_ftns": 100, "tx_hash": "0x2", "recipient": "x",
        "faucet_address": "0xF", "network": "base-sepolia"}), captured=cap)
    r = runner.invoke(main, ["wallet", "faucet", "--network", "testnet"])
    assert r.exit_code == 0, r.output
    # derived from the key, a 0x address was sent
    assert cap["body"]["destination_address"].startswith("0x")


def test_faucet_no_address_no_key_exits_1(runner, monkeypatch):
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    monkeypatch.delenv("FTNS_WALLET_PRIVATE_KEY", raising=False)
    # also no post should be attempted
    _patch_post(monkeypatch, resp=_Resp(200, {}))
    r = runner.invoke(main, ["wallet", "faucet", "--network", "testnet"])
    assert r.exit_code == 1
    assert "no destination address" in r.output.lower()


def test_faucet_403_unavailable_exits_1(runner, monkeypatch):
    _patch_post(monkeypatch, resp=_Resp(403, {
        "detail": "on-chain FTNS faucet is not available: it is TESTNET-ONLY"}))
    r = runner.invoke(main, ["wallet", "faucet", "--address", _ADDR])
    assert r.exit_code == 1
    assert "testnet-only" in r.output.lower()


def test_faucet_429_wallet_at_cap_exits_1(runner, monkeypatch):
    _patch_post(monkeypatch, resp=_Resp(429, {"detail": "address already holds 1000 FTNS"}))
    r = runner.invoke(main, ["wallet", "faucet", "--address", _ADDR])
    assert r.exit_code == 1
    assert "1000" in r.output


def test_faucet_connection_error_exits_2(runner, monkeypatch):
    _patch_post(monkeypatch, raise_connect=True)
    r = runner.invoke(main, ["wallet", "faucet", "--address", _ADDR])
    assert r.exit_code == 2
    assert "cannot connect" in r.output.lower()


def test_faucet_json_format(runner, monkeypatch):
    _patch_post(monkeypatch, resp=_Resp(200, {
        "status": "dispensed", "tx_hash": "0xJSON", "dispensed_ftns": 50,
        "recipient": _ADDR, "faucet_address": "0xF", "network": "base-sepolia"}))
    r = runner.invoke(
        main, ["wallet", "faucet", "--address", _ADDR, "--format", "json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output.strip().splitlines()[-1])
    assert payload["tx_hash"] == "0xJSON"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
