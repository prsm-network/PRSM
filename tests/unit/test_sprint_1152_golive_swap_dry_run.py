"""Sprint 1152 — GO-LIVE READINESS GATE: swap EXECUTION dry-run check.

THE GAP. ``run_go_live_verification`` (the Foundation launch gate;
``report.go`` is True iff zero FAIL findings) verifies the swap with the
QUOTE-only client (``client.quote_swap`` → amount_out > 0) and the
onramp→swap ENVELOPE build. It does NOT exercise the EXECUTION client —
sp1151 added ``AerodromeSwapClient.dry_run_swap_usdc_for_ftns``, a
static-call verify of the REAL swap route/router (getAmountsOut
liquidity + slippage on the actual swap path). So the gate could report
go-live while the execution client (a DIFFERENT router/route config than
the quote client) would revert.

This brick adds a ``swap_dry_run`` GoLiveFinding that runs the sp1151
execution dry-run, catching exactly that quote-vs-execution divergence.

READ-ONLY: the new check only calls the sp1151 dry-run (static
``.call()``; never broadcasts/signs). ADDITIVE: ``swap_client`` defaults
to None; existing callers (client + network only) are unchanged except
for the new advisory WARN (which does NOT flip ``report.go`` — only FAIL
blocks). No network: fake quote client + fake async swap_client.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prsm.economy.web3.aerodrome_client import (
    AerodromePoolState,
    AerodromeQuote,
)
from prsm.economy.web3.aerodrome_pool_ceremony import (
    MAINNET_CONFIG,
    USDC_BASE_MAINNET,
    FTNS_BASE_MAINNET,
)
from prsm.economy.web3.aerodrome_swap_client import AerodromeSwapDryRun
from prsm.economy.web3.go_live_verification import (
    run_go_live_verification,
    GoLiveReport,
)

_SEED_USDC_UNITS = 500_000 * 10 ** 6
_SEED_FTNS_UNITS = 2_000_000 * 10 ** 18
_PROBE_USDC = 1_000_000  # $1.00 in USDC base units


def _seeded_state(
    *, token0=USDC_BASE_MAINNET, token1=FTNS_BASE_MAINNET,
    r0=_SEED_USDC_UNITS, r1=_SEED_FTNS_UNITS, stable=False,
):
    return AerodromePoolState(
        pool_address="0xPOOL",
        token0=token0, token1=token1,
        reserve0=r0, reserve1=r1,
        stable=stable, fee_bps=30,
        total_supply=1_000_000 * 10 ** 18, block_number=12345,
    )


def _quote(amount_out=3_960_000 * 10 ** 12):  # ~3.96 FTNS for $1
    return AerodromeQuote(
        amount_in=_PROBE_USDC, token_in=USDC_BASE_MAINNET,
        token_out=FTNS_BASE_MAINNET, amount_out=amount_out,
        price_impact_bps=15, route="aerodrome", fee_bps=30,
    )


class _FakeAero:
    def __init__(self, *, configured=True, state=None, quote=None):
        self._configured = configured
        self._state = state
        self._quote = quote
        self.pool_address = "0xPOOL" if configured else None

    def is_configured(self):
        return self._configured

    def get_pool_state(self, pool_address=None):
        return self._state

    def quote_swap(self, amount_in, token_in, pool_address=None):
        return self._quote


class _FakeSwapClient:
    """A fake AerodromeSwapClient exposing only the async dry-run the
    gate calls. Records the call args so the test can assert owner=None,
    amount_in=probe, and the envelope floor are threaded correctly."""

    def __init__(self, verdict=None, *, raises=None):
        self._verdict = verdict
        self._raises = raises
        self.calls = []

    async def dry_run_swap_usdc_for_ftns(
        self, amount_in, amount_out_min, *, owner=None,
    ):
        self.calls.append(
            {"amount_in": amount_in, "amount_out_min": amount_out_min,
             "owner": owner})
        if self._raises is not None:
            raise self._raises
        return self._verdict


def _run(client, **kw):
    return run_go_live_verification(
        client, MAINNET_CONFIG, probe_usdc_units=_PROBE_USDC, **kw,
    )


def _status(report, check):
    f = next((f for f in report.findings if f.check == check), None)
    return f.status if f else None


def _finding(report, check):
    return next((f for f in report.findings if f.check == check), None)


# ── (a) execution dry-run PASS ───────────────────────────────

def test_execution_dry_run_pass_keeps_go_true():
    swap_client = _FakeSwapClient(AerodromeSwapDryRun(
        would_succeed=True, amount_in=_PROBE_USDC, amount_out_min=1,
        expected_out=3_960_000 * 10 ** 12, approve_needed=True,
    ))
    report = _run(
        _FakeAero(state=_seeded_state(), quote=_quote()),
        swap_client=swap_client,
    )
    assert isinstance(report, GoLiveReport)
    assert _status(report, "swap_dry_run") == "PASS"
    assert report.go is True, report.to_dict()
    # The PASS detail surfaces expected_out + the approve note.
    detail = _finding(report, "swap_dry_run").detail
    assert "3960000000000000000" in detail or "expected" in detail.lower()


# ── (b) QUOTE-vs-EXECUTION DIVERGENCE — the gap this closes ──

def test_quote_passes_but_execution_dry_run_fails_blocks_go():
    """Quote returns amount_out > 0 (quote PASS) but the EXECUTION
    client's dry-run would revert (different router/route, no
    liquidity). The gate must now catch it: swap_dry_run FAIL and
    report.go False — the readiness blocker the quote missed."""
    swap_client = _FakeSwapClient(AerodromeSwapDryRun(
        would_succeed=False, amount_in=_PROBE_USDC, amount_out_min=1,
        expected_out=None,
        revert_reason="insufficient liquidity / no pool",
    ))
    report = _run(
        _FakeAero(state=_seeded_state(), quote=_quote()),
        swap_client=swap_client,
    )
    # Quote leg still PASSes (proving divergence).
    assert _status(report, "swap_quote") == "PASS"
    # Execution leg FAILs → go-live blocked.
    assert _status(report, "swap_dry_run") == "FAIL"
    assert report.go is False, report.to_dict()
    assert "insufficient liquidity" in _finding(
        report, "swap_dry_run").detail


# ── (c) no swap_client (from_env → None) → WARN, not blocking ─

def test_no_swap_client_warns_does_not_block(monkeypatch):
    # Force from_env() to return None (unconfigured execution client).
    import prsm.economy.web3.aerodrome_swap_client as swc
    monkeypatch.setattr(
        swc.AerodromeSwapClient, "from_env", classmethod(lambda cls: None))
    report = _run(
        _FakeAero(state=_seeded_state(), quote=_quote()),
        swap_client=None,
    )
    assert _status(report, "swap_dry_run") == "WARN"
    assert report.go is True, report.to_dict()


# ── (d) async-run guard: dry-run raises → WARN, no crash ─────

def test_dry_run_raises_degrades_to_warn():
    swap_client = _FakeSwapClient(raises=RuntimeError("boom in dry-run"))
    report = _run(
        _FakeAero(state=_seeded_state(), quote=_quote()),
        swap_client=swap_client,
    )
    # Does NOT crash the whole report; degrades to a non-blocking WARN.
    assert _status(report, "swap_dry_run") == "WARN"
    assert report.go is True, report.to_dict()


# ── (e) ADDITIVE: client+network only still works ────────────

def test_additive_client_network_only_still_go(monkeypatch):
    """Calling with ONLY (client, network) — no swap_client — still
    works; the new WARN (no client) does NOT flip an otherwise-go
    report to no-go."""
    import prsm.economy.web3.aerodrome_swap_client as swc
    monkeypatch.setattr(
        swc.AerodromeSwapClient, "from_env", classmethod(lambda cls: None))
    report = run_go_live_verification(
        _FakeAero(state=_seeded_state(), quote=_quote()), MAINNET_CONFIG,
        probe_usdc_units=_PROBE_USDC,
    )
    assert _status(report, "swap_dry_run") == "WARN"
    assert report.go is True, report.to_dict()


def test_additive_from_env_construction_error_warns(monkeypatch):
    """from_env() raising (construction error) → None → WARN, never a
    crashed report."""
    import prsm.economy.web3.aerodrome_swap_client as swc

    def _boom(cls):
        raise RuntimeError("env misconfigured")

    monkeypatch.setattr(
        swc.AerodromeSwapClient, "from_env", classmethod(_boom))
    report = run_go_live_verification(
        _FakeAero(state=_seeded_state(), quote=_quote()), MAINNET_CONFIG,
        probe_usdc_units=_PROBE_USDC,
    )
    assert _status(report, "swap_dry_run") == "WARN"
    assert report.go is True, report.to_dict()


# ── (f) owner=None + amount_in=probe + envelope floor ────────

def test_dry_run_called_with_owner_none_and_envelope_floor():
    swap_client = _FakeSwapClient(AerodromeSwapDryRun(
        would_succeed=True, amount_in=_PROBE_USDC, amount_out_min=1,
        expected_out=3_960_000 * 10 ** 12,
    ))
    report = _run(
        _FakeAero(state=_seeded_state(), quote=_quote()),
        swap_client=swap_client, slippage_bps=100,
    )
    assert len(swap_client.calls) == 1
    call = swap_client.calls[0]
    assert call["owner"] is None  # read-only, no signer
    assert call["amount_in"] == _PROBE_USDC
    # The envelope floor (quote.amount_out * (10000-100)//10000) is used.
    expected_floor = (_quote().amount_out * (10_000 - 100)) // 10_000
    assert call["amount_out_min"] == expected_floor
    # And the envelope itself was prepared with that same floor.
    assert report.prepared_envelope["args"]["amountOutMin"] == expected_floor
