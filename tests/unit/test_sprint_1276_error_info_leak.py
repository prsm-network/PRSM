"""Sprint 1276 — don't echo raw exceptions to HTTP clients (audit round 6).

The wallet-balance endpoints returned `detail=f"Base RPC balance read failed: {exc}"`. A
web3/httpx error carries the Base RPC URL, which embeds the operator's RPC API key — so an
UNAUTHENTICATED wallet-balance call that triggered an RPC error leaked that secret to the
client. Same `str(exc)` pattern on the coinbase-onramp + faucet endpoints. Fix: log the full
error server-side, return a STATIC client detail.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

_SECRET = "SECRET_RPC_KEY_abc123"
_RPC_URL = f"https://base-mainnet.example/v2/{_SECRET}"


class _RaisingReader:
    def get_balances(self, address):
        raise RuntimeError(f"connect to {_RPC_URL} failed")

    def close(self):
        pass


def _client():
    from prsm.node.api import create_api_app
    app = create_api_app(MagicMock(), enable_security=False)
    return TestClient(app, raise_server_exceptions=False)


def test_wallet_balance_by_address_does_not_leak_rpc_secret(monkeypatch):
    import prsm.economy.web3.wallet_balance_reader as wbr
    monkeypatch.setattr(wbr, "from_env", lambda: _RaisingReader())
    resp = _client().get("/wallet/balance/by-address/0x" + "a" * 40)
    assert resp.status_code == 502
    detail = str(resp.json().get("detail", ""))
    assert _SECRET not in detail            # the API key must not leak
    assert "base-mainnet.example" not in detail   # nor the RPC URL
    assert detail == "upstream Base RPC balance read failed"


def test_faucet_and_onramp_no_longer_interpolate_exception():
    # source-level pin: these endpoints must not f-string the exception into the client detail
    import inspect
    import prsm.node.api as api
    src = inspect.getsource(api)
    assert 'detail=f"faucet dispense failed: {e}"' not in src
    assert 'detail=f"coinbase onramp /token call failed: {exc}"' not in src
    assert 'detail=f"Base RPC balance read failed: {exc}"' not in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
