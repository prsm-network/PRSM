"""Sprint 901 — Aerodrome pool go-live verification harness.

The final step of the pool-seed ceremony lifecycle (sp875/876 ship the
tx-batch builder + runbook; the seeding entity (Prismatica) multi-sig executes it — Option A: NOT the Foundation).
This module VERIFIES the seed actually worked and the fiat→FTNS swap
path is live — the missing go/no-go gate between "we signed the seed
tx" and "users can buy FTNS with fiat".

Run it the instant the multi-sig executes and the operator sets
``AERODROME_USDC_FTNS_POOL_ADDRESS``. It checks, read-only:

  pool_configured     — RPC + pool address present
  pool_seeded         — pool state readable + both reserves > 0
  token_pair          — the pool holds {USDC, FTNS} for the network
  volatile_pool       — stable=False (the route the orchestrator builds)
  opening_price       — implied FTNS price from reserves (INFO)
  reserves_match_seed — optional cross-check vs the seed amounts (WARN)
  swap_quote          — the live quoter returns amount_out > 0
  onramp_swap_envelope— the full onramp→swap envelope builds end-to-end

``go`` is True iff there are zero FAIL findings (WARN/INFO don't block).
On success the report carries ``prepared_envelope`` — the exact
executable swap envelope for the operator's probe amount, so the first
real onramp→swap→FTNS can be submitted immediately.

NO money moves here — verification + tx preparation only. Fully
offline-testable against a fake AerodromeClient; runs live unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# USDC has 6 decimals, FTNS 18 — for the implied-price calc.
_USDC_DECIMALS = 6
_FTNS_DECIMALS = 18

# Default probe size for the swap-path check ($1.00 in USDC base units).
_DEFAULT_PROBE_USDC_UNITS = 1_000_000

# Tolerance for the optional reserves-vs-seed cross-check (1%). The pool
# may legitimately have traded between seed and verification, so a
# mismatch is a WARN, not a FAIL — this just flags a LARGE divergence.
_RESERVE_MATCH_TOLERANCE = 0.01


@dataclass
class GoLiveFinding:
    check: str
    status: str  # PASS | FAIL | WARN | INFO
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class GoLiveReport:
    findings: List[GoLiveFinding] = field(default_factory=list)
    prepared_envelope: Optional[Dict[str, Any]] = None

    @property
    def go(self) -> bool:
        """Go-live iff no FAIL findings. WARN/INFO are advisory."""
        return not any(f.status == "FAIL" for f in self.findings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "go": self.go,
            "findings": [f.to_dict() for f in self.findings],
            "prepared_envelope": self.prepared_envelope,
        }


def _addr_eq(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a) and bool(b) and a.lower() == b.lower()


def _implied_ftns_price_usd(
    reserve_usdc_units: int, reserve_ftns_units: int,
) -> Optional[float]:
    """Opening FTNS price in USD from the pool reserves (decimal-
    adjusted). None if either reserve is zero."""
    if reserve_usdc_units <= 0 or reserve_ftns_units <= 0:
        return None
    usdc = reserve_usdc_units / (10 ** _USDC_DECIMALS)
    ftns = reserve_ftns_units / (10 ** _FTNS_DECIMALS)
    return usdc / ftns


def _verify_pool(
    client: Any,
    network: Any,
    *,
    expected_usdc_units: Optional[int],
    expected_ftns_units: Optional[int],
) -> List[GoLiveFinding]:
    findings: List[GoLiveFinding] = []

    if not client.is_configured():
        findings.append(GoLiveFinding(
            "pool_configured", "FAIL",
            "AerodromeClient not configured — set BASE_RPC_URL + "
            "AERODROME_USDC_FTNS_POOL_ADDRESS (the latter is populated "
            "the moment the seed ceremony executes). Pre-ceremony this "
            "FAIL is expected.",
        ))
        return findings  # nothing else is checkable
    findings.append(GoLiveFinding(
        "pool_configured", "PASS",
        f"pool address {client.pool_address}",
    ))

    state = client.get_pool_state()
    if state is None:
        findings.append(GoLiveFinding(
            "pool_seeded", "FAIL",
            "pool state unreadable (RPC error or pool address points "
            "at no deployed pool) — cannot confirm the seed.",
        ))
        return findings

    # Token pair — order-insensitive {USDC, FTNS} for this network.
    pool_tokens = {state.token0.lower(), state.token1.lower()}
    want_tokens = {
        network.usdc_address.lower(), network.ftns_address.lower(),
    }
    if pool_tokens == want_tokens:
        findings.append(GoLiveFinding(
            "token_pair", "PASS",
            "pool holds the USDC/FTNS pair",
        ))
    else:
        findings.append(GoLiveFinding(
            "token_pair", "FAIL",
            f"pool tokens {sorted(pool_tokens)} != expected USDC/FTNS "
            f"{sorted(want_tokens)} — wrong pool address configured.",
        ))

    # Seeded = both reserves non-zero.
    if state.reserve0 > 0 and state.reserve1 > 0:
        findings.append(GoLiveFinding(
            "pool_seeded", "PASS",
            f"reserves {state.reserve0} / {state.reserve1} "
            f"(block {state.block_number})",
        ))
    else:
        findings.append(GoLiveFinding(
            "pool_seeded", "FAIL",
            f"pool has zero reserve(s) ({state.reserve0} / "
            f"{state.reserve1}) — NOT seeded yet.",
        ))

    # Volatile vs stable — the orchestrator routes through the volatile
    # pool (stable=False). A stable pool means the swap route is wrong.
    if state.stable:
        findings.append(GoLiveFinding(
            "volatile_pool", "FAIL",
            "seeded pool is STABLE; the onramp→swap orchestrator builds "
            "a VOLATILE route (stable=False). Re-seed as volatile or "
            "update the route.",
        ))
    else:
        findings.append(GoLiveFinding(
            "volatile_pool", "PASS", "volatile pool (stable=False)",
        ))

    # Opening price (which reserve is USDC depends on token order).
    if _addr_eq(state.token0, network.usdc_address):
        usdc_res, ftns_res = state.reserve0, state.reserve1
    else:
        usdc_res, ftns_res = state.reserve1, state.reserve0
    price = _implied_ftns_price_usd(usdc_res, ftns_res)
    if price is not None:
        findings.append(GoLiveFinding(
            "opening_price", "INFO",
            f"implied opening price ~${price:.4f}/FTNS "
            f"({usdc_res / 10 ** _USDC_DECIMALS:.2f} USDC / "
            f"{ftns_res / 10 ** _FTNS_DECIMALS:.2f} FTNS)",
        ))

    # Optional cross-check against the seed amounts.
    if expected_usdc_units is not None and expected_ftns_units is not None:
        ok_usdc = _within(usdc_res, expected_usdc_units)
        ok_ftns = _within(ftns_res, expected_ftns_units)
        if ok_usdc and ok_ftns:
            findings.append(GoLiveFinding(
                "reserves_match_seed", "PASS",
                "reserves match the declared seed amounts",
            ))
        else:
            findings.append(GoLiveFinding(
                "reserves_match_seed", "WARN",
                f"reserves ({usdc_res} USDC / {ftns_res} FTNS) differ "
                f"from declared seed ({expected_usdc_units} / "
                f"{expected_ftns_units}) — the pool may have traded "
                "since seeding; confirm this is expected.",
            ))
    return findings


def _within(actual: int, expected: int) -> bool:
    if expected <= 0:
        return actual == expected
    return abs(actual - expected) / expected <= _RESERVE_MATCH_TOLERANCE


def _verify_swap_path(
    client: Any,
    network: Any,
    *,
    probe_usdc_units: int,
    ftns_address: str,
    slippage_bps: int,
    swap_client: Any = None,
) -> tuple[List[GoLiveFinding], Optional[Dict[str, Any]]]:
    findings: List[GoLiveFinding] = []

    # Direct quote (granular signal).
    try:
        quote = client.quote_swap(
            amount_in=probe_usdc_units,
            token_in=network.usdc_address,
        )
    except Exception as exc:  # noqa: BLE001
        quote = None
        logger.warning("go-live: quote_swap raised: %s", exc)
    if quote is not None and getattr(quote, "amount_out", 0) > 0:
        findings.append(GoLiveFinding(
            "swap_quote", "PASS",
            f"probe {probe_usdc_units} USDC units → "
            f"{quote.amount_out} FTNS units "
            f"(impact {getattr(quote, 'price_impact_bps', '?')}bps)",
        ))
    else:
        findings.append(GoLiveFinding(
            "swap_quote", "FAIL",
            "live swap quote unavailable or amount_out=0 — the pool "
            "cannot price a USDC→FTNS swap.",
        ))

    # Full onramp→swap envelope build (reuses the production path).
    from prsm.economy.web3.onramp_to_swap_orchestrator import (
        build_envelope_for_intent,
    )
    from types import SimpleNamespace
    intent = SimpleNamespace(
        intent_id="go-live-probe",
        user_id="go-live",
        destination_address="0x" + "00" * 20,
        usdc_received=probe_usdc_units / (10 ** _USDC_DECIMALS),
        usdc_received_units=probe_usdc_units,
    )
    envelope = None
    try:
        envelope = build_envelope_for_intent(
            intent,
            aerodrome_client=client,
            ftns_address=ftns_address,
            slippage_bps=slippage_bps,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("go-live: envelope build raised: %s", exc)
    if (
        envelope is not None
        and envelope.get("status") == "READY_FOR_SUBMISSION"
        and envelope.get("args", {}).get("amountIn") == probe_usdc_units
        and envelope.get("args", {}).get("amountOutMin", 0) >= 1
    ):
        findings.append(GoLiveFinding(
            "onramp_swap_envelope", "PASS",
            f"full onramp→swap envelope builds; amountIn="
            f"{probe_usdc_units} amountOutMin="
            f"{envelope['args']['amountOutMin']}",
        ))
    else:
        findings.append(GoLiveFinding(
            "onramp_swap_envelope", "FAIL",
            "onramp→swap envelope did not build to "
            "READY_FOR_SUBMISSION — the pool isn't seeded/quotable yet.",
        ))
        envelope = None

    # sp1152 — EXECUTION-path dry-run. The quote/envelope above use the
    # QUOTE client; this exercises the REAL swap route/router via the
    # sp1151 read-only static-call dry-run (getAmountsOut liquidity +
    # slippage on the actual execution path), catching a quote-vs-
    # execution divergence the gate previously missed. READ-ONLY: only
    # ``dry_run_swap_usdc_for_ftns`` (static .call(), never a broadcast).
    findings.append(_swap_dry_run_finding(
        swap_client,
        envelope=envelope,
        probe_usdc_units=probe_usdc_units,
        quote=quote,
        slippage_bps=slippage_bps,
    ))
    return findings, envelope


def _swap_dry_run_finding(
    swap_client: Any,
    *,
    envelope: Optional[Dict[str, Any]],
    probe_usdc_units: int,
    quote: Any,
    slippage_bps: int,
) -> GoLiveFinding:
    """sp1152 — run the sp1151 execution-client dry-run as a go-live
    check. No client → advisory WARN (never blocks). Client present →
    PASS/FAIL from ``would_succeed`` + ``revert_reason``. The async dry-
    run is invoked from this SYNC gate via a guarded ``asyncio.run`` — a
    running loop or ANY error degrades to a non-blocking WARN so the dry-
    run can never crash the whole go-live report."""
    if swap_client is None:
        return GoLiveFinding(
            "swap_dry_run", "WARN",
            "execution-path dry-run skipped: no swap client configured; "
            "verify the real AerodromeSwapClient path before going live.",
        )

    # Slippage floor for the dry-run: prefer the envelope's
    # amountOutMin (the exact floor the orchestrator computed); fall
    # back to a slippage_bps-derived floor from the quote when the
    # envelope is unavailable.
    amount_out_min: int = 0
    if (
        envelope is not None
        and isinstance(envelope.get("args"), dict)
        and envelope["args"].get("amountOutMin") is not None
    ):
        amount_out_min = int(envelope["args"]["amountOutMin"])
    elif quote is not None and getattr(quote, "amount_out", 0) > 0:
        _slip = max(0, min(int(slippage_bps), 9999))
        amount_out_min = (quote.amount_out * (10_000 - _slip)) // 10_000
        if amount_out_min < 1:
            amount_out_min = 1

    import asyncio

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no running loop — safe to asyncio.run below
        else:
            return GoLiveFinding(
                "swap_dry_run", "WARN",
                "execution-path dry-run skipped: a running event loop was "
                "detected (cannot asyncio.run the async dry-run from here); "
                "run the AerodromeSwapClient dry-run out-of-band before "
                "going live.",
            )
        verdict = asyncio.run(swap_client.dry_run_swap_usdc_for_ftns(
            amount_in=probe_usdc_units,
            amount_out_min=amount_out_min,
            owner=None,  # read-only: no signer → liquidity + slippage only
        ))
    except Exception as exc:  # noqa: BLE001 — never crash the report
        logger.warning("go-live: swap dry-run raised: %s", exc)
        return GoLiveFinding(
            "swap_dry_run", "WARN",
            f"execution-path dry-run could not run (error: {exc}); verify "
            "the real AerodromeSwapClient path before going live.",
        )

    if getattr(verdict, "would_succeed", False):
        approve_note = (
            " (approve_needed — the swap auto-approves at execution)"
            if getattr(verdict, "approve_needed", False) else ""
        )
        return GoLiveFinding(
            "swap_dry_run", "PASS",
            f"execution-path dry-run would succeed: expected_out="
            f"{getattr(verdict, 'expected_out', None)} FTNS units, "
            f"amount_out_min={amount_out_min}{approve_note}.",
        )
    return GoLiveFinding(
        "swap_dry_run", "FAIL",
        "execution-path dry-run would REVERT (the real swap route/router "
        "fails where the quote did not): "
        f"{getattr(verdict, 'revert_reason', None) or 'unknown reason'}.",
    )


def run_go_live_verification(
    client: Any,
    network: Any,
    *,
    probe_usdc_units: int = _DEFAULT_PROBE_USDC_UNITS,
    ftns_address: Optional[str] = None,
    expected_usdc_units: Optional[int] = None,
    expected_ftns_units: Optional[int] = None,
    slippage_bps: int = 100,
    swap_client: Any = None,
) -> GoLiveReport:
    """Run the full pool go-live verification + prepare the first-swap
    envelope. ``network`` is a CeremonyNetworkConfig (MAINNET_CONFIG /
    SEPOLIA_CONFIG). Returns a GoLiveReport; ``report.go`` is the
    operator's launch gate.

    ``swap_client`` (sp1152, optional) is an ``AerodromeSwapClient`` used
    for the read-only EXECUTION-path dry-run. When None, this falls back
    to ``AerodromeSwapClient.from_env()`` (lazy import, try/except →
    None); an unconfigured execution client yields an advisory WARN, not
    a FAIL — it must not block go-live in a test/dev env."""
    if swap_client is None:
        try:
            from prsm.economy.web3.aerodrome_swap_client import (
                AerodromeSwapClient,
            )
            swap_client = AerodromeSwapClient.from_env()
        except Exception as exc:  # noqa: BLE001 — construction error → WARN
            logger.warning(
                "go-live: AerodromeSwapClient.from_env() raised: %s", exc)
            swap_client = None

    report = GoLiveReport()
    report.findings.extend(_verify_pool(
        client, network,
        expected_usdc_units=expected_usdc_units,
        expected_ftns_units=expected_ftns_units,
    ))

    # Only probe the swap path once the pool is configured + seeded —
    # a quote against an unconfigured/empty pool is noise.
    pool_ready = not any(
        f.status == "FAIL"
        and f.check in ("pool_configured", "pool_seeded")
        for f in report.findings
    )
    if pool_ready:
        swap_findings, envelope = _verify_swap_path(
            client, network,
            probe_usdc_units=probe_usdc_units,
            ftns_address=ftns_address or network.ftns_address,
            slippage_bps=slippage_bps,
            swap_client=swap_client,
        )
        report.findings.extend(swap_findings)
        report.prepared_envelope = envelope
    else:
        report.findings.append(GoLiveFinding(
            "swap_quote", "FAIL",
            "skipped — pool not configured/seeded (see above).",
        ))
        report.findings.append(GoLiveFinding(
            "onramp_swap_envelope", "FAIL",
            "skipped — pool not configured/seeded (see above).",
        ))
    return report
