"""Sprint 1392 — `prsm wallet balance` on-chain balance command (F9).

The `wallet` group's help said "view balance" but had no balance command (only `ftns balance`, the
off-chain ledger). This reads live on-chain FTNS + ETH via /wallet/balance/by-address/{addr}. httpx
mocked.
"""
from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner

_ADDR = "0x" + "a1" * 20


def _invoke(args, monkeypatch):
    monkeypatch.setattr("prsm.cli._api_url_from_creds", lambda o: "http://x")
    monkeypatch.setattr("prsm.cli._node_api_key_headers", lambda: {})
    from prsm.cli import wallet as _wallet_group
    return CliRunner().invoke(_wallet_group, args)


def _resp(status_code, body):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = body
    r.headers = {"content-type": "application/json"}
    return r


def test_balance_reads_on_chain(monkeypatch):
    body = {"ftns": 0.4, "ftns_units": 4 * 10 ** 17, "native_eth": 0.0005, "native_eth_wei": 5 * 10 ** 14}
    with patch("httpx.get", return_value=_resp(200, body)) as mg:
        r = _invoke(["balance", "--address", _ADDR], monkeypatch)
    assert r.exit_code == 0, r.output
    called = mg.call_args.args[0] if mg.call_args.args else mg.call_args.kwargs.get("url", "")
    assert called.endswith(f"/wallet/balance/by-address/{_ADDR}")
    assert "0.4" in r.output and "FTNS" in r.output


def test_balance_json(monkeypatch):
    with patch("httpx.get", return_value=_resp(200, {"ftns": 1.5, "native_eth": 0.01})):
        r = _invoke(["balance", "--address", _ADDR, "--format", "json"], monkeypatch)
    assert r.exit_code == 0 and '"ftns": 1.5' in r.output


def test_balance_connect_error_exits_2(monkeypatch):
    with patch("httpx.get", side_effect=httpx.ConnectError("no conn")):
        r = _invoke(["balance", "--address", _ADDR], monkeypatch)
    assert r.exit_code == 2 and "Cannot connect" in r.output


def test_balance_no_address_no_key(monkeypatch):
    monkeypatch.setattr("prsm.cli._wallet_load_signer", lambda net: {})   # no key → no address
    r = _invoke(["balance"], monkeypatch)
    assert r.exit_code == 1 and "no address" in r.output


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
