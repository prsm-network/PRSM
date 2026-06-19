"""Sprint 871 — onramp→swap auto-orchestrator.

When sp857's funnel sweep transitions an intent to CONFIRMED (USDC
arrived on-chain), this orchestrator immediately prepares an
Aerodrome USDC→FTNS swap envelope. The user (or their integration)
retrieves the prepared envelope via GET /wallet/onramp/funnel/
{intent_id} and executes the swap once the pool ceremony lands.

End-to-end fiat→FTNS becomes a single-shot user experience: complete
the Coinbase Pay buy, wait for sweep confirmation, swap envelope is
ready in the same intent record. No second manual quote step.

Architecture is intentionally minimal — just a callable that:
  1. Takes a CONFIRMED OnrampIntent
  2. Calls AerodromeClient.quote_swap to get the USDC→FTNS quote
  3. Builds the same READY_FOR_SUBMISSION envelope shape sp855's
     /wallet/swap/execute returns
  4. Attaches it to the intent's swap_envelope field

Fail-soft on:
  - Pool not seeded yet (POOL_NOT_CONFIGURED) → envelope stays None,
    intent stays CONFIRMED. Next sweep cycle will retry envelope
    build.
  - Pool quote raises (RPC down, malformed reserves, etc.) → log +
    leave envelope None.
"""
from __future__ import annotations

import logging
import time as _time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Default 1% slippage tolerance. Aggressive enough for liquid
# USDC↔FTNS pools, conservative enough to avoid sandwich attacks
# on the initial seeded pool (where liquidity is thin).
_DEFAULT_SLIPPAGE_BPS = 100
# 24h deadline — matches sp855's execute envelope.
_DEFAULT_DEADLINE_SECONDS = 86_400

# Canonical Base mainnet addresses (mirror sp855).
_USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_AERODROME_ROUTER_V2 = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
_AERODROME_POOL_FACTORY = (
    "0x420DD381b31aEf6683db6B902084cB0FFECe40Da"
)


def build_envelope_for_intent(
    intent: Any,
    *,
    aerodrome_client: Any,
    ftns_address: str,
    slippage_bps: int = _DEFAULT_SLIPPAGE_BPS,
) -> Optional[Dict[str, Any]]:
    """Build the Aerodrome USDC→FTNS swap envelope for a CONFIRMED
    intent.

    Returns None when:
      - aerodrome_client is unavailable
      - Pool is unconfigured (ceremony pending)
      - Pool state read fails
      - Quote returns no usable amount_out
    """
    if aerodrome_client is None:
        return None
    if not aerodrome_client.is_configured():
        # Pool ceremony pending. Caller can retry on next sweep.
        return None

    # Use the actual usdc_received as amount_in — not the original
    # expected_usd, since Coinbase's spread + price moves between
    # quote + buy mean the user got slightly less than expected.
    usdc_received = float(getattr(intent, "usdc_received", 0.0) or 0)
    if usdc_received <= 0:
        return None

    # Sp895 — amount_in is the EXACT on-chain USDC base units. The
    # funnel persists the authoritative integer (usdc_received_units,
    # from bal.usdc_units at the CONFIRMED transition); use it
    # directly. Computing `int(usdc_received * 1e6)` from the float
    # whole-token amount round-trips through float64 (units → /1e6 →
    # ×1e6 → int) and loses a base unit for ~1.2% of values (e.g.
    # 8_000_001 → 8.000001 → 8_000_000), making the swap under-spend
    # the received dust. Fall back to the float-derived value only
    # for legacy records that predate the units field.
    usdc_received_units = int(
        getattr(intent, "usdc_received_units", 0) or 0,
    )
    if usdc_received_units > 0:
        amount_in_units = usdc_received_units
    else:
        # USDC has 6 decimals — convert whole-token to base units.
        amount_in_units = int(usdc_received * (10 ** 6))

    try:
        quote = aerodrome_client.quote_swap(
            amount_in=amount_in_units,
            token_in=_USDC_BASE,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "onramp_to_swap_orchestrator: quote_swap failed for "
            "intent %s: %s",
            getattr(intent, "intent_id", "?"), exc,
        )
        return None
    if quote is None:
        return None

    # amount_out is in FTNS base units (18 decimals).
    # Sp890 — clamp slippage to [0, 10000) and enforce
    # amountOutMin >= 1 when amount_out > 0: a swap that accepts
    # zero output is an unbounded sandwich. (The user-facing
    # /wallet/swap/* endpoints reject high slippage outright;
    # this is the belt-and-suspenders for the auto-orchestrator
    # path where slippage_bps is operator/code-supplied.)
    _slip = max(0, min(int(slippage_bps), 9999))
    amount_out_min_units = (
        quote.amount_out * (10_000 - _slip) // 10_000
    )
    if quote.amount_out > 0 and amount_out_min_units < 1:
        amount_out_min_units = 1
    amount_out_whole = quote.amount_out / (10 ** 18)
    deadline = int(_time.time()) + _DEFAULT_DEADLINE_SECONDS

    return {
        "status": "READY_FOR_SUBMISSION",
        "router_address": _AERODROME_ROUTER_V2,
        "function": "swapExactTokensForTokens",
        "args": {
            "amountIn": amount_in_units,
            "amountOutMin": amount_out_min_units,
            "routes": [
                {
                    "from_token": "USDC",
                    "to_token": "FTNS",
                    "stable": False,
                    "factory": _AERODROME_POOL_FACTORY,
                },
            ],
            "to": intent.destination_address,
            "deadline": deadline,
        },
        "quote": {
            "amount_in_usdc": usdc_received,
            "amount_in_units": amount_in_units,
            "amount_out_ftns": amount_out_whole,
            "amount_out_units": quote.amount_out,
            "amount_out_min_units": amount_out_min_units,
            "price_impact_bps": quote.price_impact_bps,
            "fee_bps": quote.fee_bps,
            "slippage_bps": slippage_bps,
        },
        "destination_address": intent.destination_address,
        "intent_id": intent.intent_id,
        "user_id": intent.user_id,
        "envelope_built_at": _time.time(),
        "note": (
            "Sponsored by sp871 onramp→swap orchestrator at "
            "CONFIRMED transition. Aerodrome pool ceremony must "
            "have closed (Vision gantt 2026-06-15) before this "
            "envelope can be executed; until then envelope stays "
            "READY_FOR_SUBMISSION as a record."
        ),
    }


def make_on_confirmed_callback(
    *,
    funnel: Any,
    aerodrome_client: Any,
    ftns_address: str,
    slippage_bps: int = _DEFAULT_SLIPPAGE_BPS,
    completion_notifier: Any = None,
    compliance_ring: Any = None,
):
    """Factory that returns a callable suitable for OnrampFunnel
    .sweep(on_confirmed=...).

    The returned closure:
      - Builds the swap envelope (None-safe if pool not seeded yet)
      - Attaches to the intent's swap_envelope field
      - Persists the updated intent to disk via funnel._persist
      - Sp885: records onramp_execute to the compliance ring with
        the ACTUAL usdc_received, so the sp884 tier-limit rolling
        total reflects settled volume (no-op when ring not wired)
      - Sp874: fires the outbound completion webhook (no-op when
        notifier isn't configured)
    """
    def _on_confirmed(intent):
        envelope = build_envelope_for_intent(
            intent,
            aerodrome_client=aerodrome_client,
            ftns_address=ftns_address,
            slippage_bps=slippage_bps,
        )
        if envelope is not None:
            intent.swap_envelope = envelope
            funnel._persist(intent)
            logger.info(
                "sp871: built swap envelope for intent %s "
                "(amount_in_usdc=%.4f, amount_out_ftns=%.4f)",
                intent.intent_id,
                envelope["quote"]["amount_in_usdc"],
                envelope["quote"]["amount_out_ftns"],
            )
        else:
            logger.info(
                "sp871: pool not configured / quote unavailable "
                "for intent %s — envelope deferred",
                intent.intent_id,
            )
        # Sp885 — record settled onramp volume so the sp884 tier
        # rolling total accumulates against the user's limit. Uses
        # the ACTUAL usdc_received (not expected_usd). Fail-soft:
        # a ring error must not undo the CONFIRMED transition.
        if compliance_ring is not None:
            try:
                compliance_ring.record(
                    kind="onramp_execute",
                    user_id=intent.user_id or "",
                    usd_amount=float(
                        getattr(intent, "usdc_received", 0.0) or 0.0
                    ),
                    ftns_amount=0.0,
                    status="CONFIRMED",
                    address=intent.destination_address,
                    metadata={"intent_id": intent.intent_id},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sp885: compliance ring record failed for "
                    "intent %s: %s", intent.intent_id, exc,
                )
        # Sp874 — fire outbound completion webhook (fail-soft).
        if completion_notifier is not None:
            try:
                completion_notifier.notify(intent=intent)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sp874: completion notifier raised for "
                    "intent %s: %s", intent.intent_id, exc,
                )
    return _on_confirmed


def make_on_expired_callback(*, compliance_ring: Any = None):
    """Factory that returns a callable suitable for OnrampFunnel
    .sweep(on_expired=...).

    Fires on an intent's EXPIRED transition (no USDC arrived within
    the 24h window → abandoned). Records a TERMINAL EXPIRED
    onramp_execute compliance entry (usd_amount=0.0, same intent_id)
    so that:
      - the sp884 AML rolling total stops counting the sp968 PENDING
        reservation for this abandoned intent —
        FiatComplianceRing.total_usd_for_user dedups by intent_id and
        the latest-timestamp entry wins, so the $0 EXPIRED entry
        supersedes the reserved amount; and
      - the regulator-facing compliance audit trail (5-7yr retention)
        is COMPLETE: an abandoned reservation gets an explicit EXPIRED
        terminal record instead of a dangling PENDING.

    No-op when the ring isn't wired. The sp874 outbound completion
    notifier is intentionally NOT fired on expiry — it signals a
    settled conversion, not an abandonment. Fail-soft: a ring error
    must never abort the sweep (mirrors the on_confirmed contract).
    """
    def _on_expired(intent):
        if compliance_ring is None:
            return
        try:
            compliance_ring.record(
                kind="onramp_execute",
                user_id=intent.user_id or "",
                usd_amount=0.0,
                ftns_amount=0.0,
                status="EXPIRED",
                address=intent.destination_address,
                metadata={"intent_id": intent.intent_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sp1176: compliance ring EXPIRED record failed for "
                "intent %s: %s", intent.intent_id, exc,
            )
    return _on_expired
