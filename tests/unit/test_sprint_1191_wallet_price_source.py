"""Sprint 1191 — wire a real (operator/governance-configured) FTNS→USD price into the
production wallet, replacing the hardcoded StaticPriceSource(0).

Day-one front-door follow-on to sp1183: after sp1183 the wallet shows a REAL on-chain
FTNS balance, but the price source was still StaticPriceSource(0) → GET
/api/v1/auth/wallet/balance rendered every real balance as $0.00. This resolves
FTNS_USD_RATE (the SAME env the FTNSOracle uses as its internal-rate fallback) into the
wallet price source, fail-soft to $0 when unset/invalid.

Honest scope: FTNS_USD_RATE IS the price source pre-market (no liquid FTNS market exists
until the gated Aerodrome pool ceremony); bridging the async FTNSOracle's live feeds into
this sync display path is the post-seeding follow-on.
"""
from __future__ import annotations

from decimal import Decimal

import pytest


# ── _resolve_price_source: config → price, fail-soft to 0 ────────────────────────────

def _resolve(monkeypatch, rate):
    import prsm.node.wallet_api_wiring as w
    if rate is None:
        monkeypatch.delenv("FTNS_USD_RATE", raising=False)
    else:
        monkeypatch.setenv("FTNS_USD_RATE", rate)
    return w._resolve_price_source()


def test_unset_rate_defaults_to_zero(monkeypatch):
    src = _resolve(monkeypatch, None)
    assert src.ftns_price_usd() == Decimal(0)


def test_configured_rate_is_used(monkeypatch):
    src = _resolve(monkeypatch, "0.05")
    assert src.ftns_price_usd() == Decimal("0.05")


def test_integer_rate(monkeypatch):
    src = _resolve(monkeypatch, "3")
    assert src.ftns_price_usd() == Decimal("3")


def test_garbage_rate_fails_soft_to_zero(monkeypatch):
    src = _resolve(monkeypatch, "not-a-number")
    assert src.ftns_price_usd() == Decimal(0)


def test_negative_rate_fails_soft_to_zero(monkeypatch):
    src = _resolve(monkeypatch, "-1.5")
    assert src.ftns_price_usd() == Decimal(0)


def test_blank_rate_defaults_to_zero(monkeypatch):
    src = _resolve(monkeypatch, "   ")
    assert src.ftns_price_usd() == Decimal(0)


def test_resolved_source_drives_usd_conversion(monkeypatch):
    # The whole point: a real balance now converts to a real USD figure.
    from prsm.interface.display import ftns_to_usd
    src = _resolve(monkeypatch, "0.05")
    assert ftns_to_usd(Decimal(100), src) == Decimal("5.00")  # 100 FTNS @ $0.05


def test_zero_source_still_converts_to_zero(monkeypatch):
    from prsm.interface.display import ftns_to_usd
    src = _resolve(monkeypatch, None)
    assert ftns_to_usd(Decimal(100), src) == Decimal("0.00")


# ── wiring: wire_wallet_api_services threads the resolved source through ──────────────

def test_wire_uses_resolved_price_source(monkeypatch, tmp_path):
    import prsm.node.wallet_api_wiring as w
    from prsm.interface.api import wallet_api

    monkeypatch.setenv("FTNS_USD_RATE", "0.25")
    # keep the binding store off the real ~/.prsm path
    monkeypatch.setenv("PRSM_WALLET_BINDINGS_DB", str(tmp_path / "bindings.db"))

    captured = {}
    monkeypatch.setattr(wallet_api, "set_services", lambda s: captured.update(s=s))

    class _Node:
        _receipt_store = None

    w.wire_wallet_api_services(_Node())

    assert "s" in captured, "set_services should have been called"
    assert captured["s"].price_source.ftns_price_usd() == Decimal("0.25")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
