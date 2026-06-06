"""Sprint 780 — settler-side credit-policy primitives.

Vision §4.5: "preemption should not result in operator slashing;
abandonment should". Sprints 772-779 closed the OPERATOR end of
the preemption arc (cloud-spot detection through to signed
partial-completion receipts). This module ships the FIRST two
SETTLER-side primitives that future settlement-flow integration
(sprint 781+) will call.

Two pure functions:

    effective_cost_for_receipt(receipt) -> Decimal
        Cost to credit the operator. For full-success receipts
        (no partial_completion): receipt.cost_ftns unchanged.
        For partial-completion receipts: scaled by
        tokens_completed / tokens_requested. Decimal arithmetic
        preserved (no float-precision drift).

    should_slash_for_receipt(receipt) -> bool
        Pure-semantic slash policy keyed on partial_completion.reason:
        - None partial_completion             → False (success path)
        - reason in {preempted, timeout}      → False (not operator's fault)
        - reason="error"                      → True (abandonment)
        - any other reason                    → True (defensive default;
          prevents a malicious operator from inventing a custom
          reason to dodge the rule)

Both functions are pure: no I/O, no side effects. Settlement-side
composes these with its OTHER slash triggers (Byzantine behavior,
attestation failures, etc.) — sprint 780 only ships the partial-
completion-aware rule.

Over-claim defense: if a tampered receipt has
tokens_completed > tokens_requested, effective_cost_for_receipt
clamps at full cost (no bonus credit). The signing-payload's
canonical hash (sprint 777) makes such tampering signature-
invalid before this code ever sees the receipt — this is belt-
and-suspenders.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, Optional, Union

from prsm.compute.inference.models import InferenceReceipt


# Reasons that are NOT operator-attributable (no slashing).
# Mirror Vision §4.5: cloud preemption + configurable timeout
# are not the operator's fault; runtime errors (OOM, crash) are.
_NO_SLASH_REASONS = frozenset({"preempted", "timeout"})


def effective_cost_for_receipt(receipt: InferenceReceipt) -> Decimal:
    """Cost amount to credit the operator for this receipt.

    Returns receipt.cost_ftns unchanged when there's no partial-
    completion marker. Otherwise scales proportionally by
    tokens_completed / tokens_requested, clamped to [0, full]."""
    info = getattr(receipt, "partial_completion", None)
    if info is None:
        return receipt.cost_ftns

    requested = int(getattr(info, "tokens_requested", 0))
    completed = int(getattr(info, "tokens_completed", 0))

    # Divide-by-zero defense: a malformed receipt with requested=0
    # MUST NOT pay out anything.
    if requested <= 0:
        return Decimal("0")

    # Clamp at full cost — no over-credit even if the operator
    # somehow signed a receipt claiming more completed than
    # requested (signature would normally invalidate this, but
    # defense-in-depth).
    if completed >= requested:
        return receipt.cost_ftns

    if completed <= 0:
        return Decimal("0")

    return (
        receipt.cost_ftns * Decimal(completed) / Decimal(requested)
    )


def should_slash_for_receipt(receipt: InferenceReceipt) -> bool:
    """Whether the partial-completion marker on this receipt
    should trigger a slash.

    Returns False (no slash) when:
    - receipt has no partial_completion (full-success path)
    - reason is "preempted" or "timeout" (not operator's fault)

    Returns True (slash) when:
    - reason is "error" (runtime fault — abandonment per §4.5)
    - reason is anything else (defensive default; an unknown
      reason cannot be allowed to escape the slash policy by
      virtue of being unknown)

    Settlement-side composes this with its OTHER slash triggers."""
    info = getattr(receipt, "partial_completion", None)
    if info is None:
        return False
    reason = getattr(info, "reason", "")
    return reason not in _NO_SLASH_REASONS


@dataclass(frozen=True)
class SettlementDecision:
    """Sprint 782 — composite policy output.

    Combines the proportional credit + slash flag into a single
    value that downstream escrow-flow code can act on with one
    call to compute_settlement_for_receipt(). The invariant
    `release_to_operator + refund_to_payer == escrow_amount`
    holds in all cases (no FTNS is created or destroyed by this
    decision — only routed)."""

    release_to_operator: Decimal
    refund_to_payer: Decimal
    should_slash: bool


def compute_settlement_for_receipt(
    receipt: InferenceReceipt,
    escrow_amount: Optional[Decimal] = None,
) -> SettlementDecision:
    """Sprint 782 — produce the full settlement decision for a
    receipt.

    Composes sprint-780 effective_cost_for_receipt +
    should_slash_for_receipt with the implied refund-remainder
    math.

    Parameters:
      receipt        — signed receipt with optional partial_completion
      escrow_amount  — the amount locked in escrow for this job.
                       Defaults to receipt.cost_ftns (the common
                       "user paid exactly the priced cost" case).
                       Pass explicitly when escrow > cost (over-
                       funded, dynamic-pricing differential, etc.)
                       so the refund math accounts for the excess.

    Returns SettlementDecision with the invariant:
        release_to_operator + refund_to_payer == escrow_amount

    Pure function — no I/O. Callers invoke PaymentEscrow APIs
    based on the decision."""
    if escrow_amount is None:
        escrow_amount = receipt.cost_ftns

    release = effective_cost_for_receipt(receipt)
    # Clamp release to escrow_amount (over-priced receipts can't
    # release more than was locked).
    if release > escrow_amount:
        release = escrow_amount

    refund = escrow_amount - release
    return SettlementDecision(
        release_to_operator=release,
        refund_to_payer=refund,
        should_slash=should_slash_for_receipt(receipt),
    )


def split_release_across_stages(
    receipt: InferenceReceipt,
    release: Decimal,
) -> Optional[list]:
    """Sprint 1031 — distribute ``release`` equally across the distinct stage
    node_ids carried in ``receipt.topology_assignment``.

    The Tier-1 cross-host bench proved a signed receipt naming each stage's
    node_id (stage0→head, stage1→worker, ...). Settlement previously credited
    only the single head/settler node, so a worker that ran half the model was
    paid nothing. This fans the release out across the nodes that actually ran
    a slice, reusing PaymentEscrow.release_escrow_split.

    Returns ``None`` — caller keeps the unchanged single-payee path — when the
    receipt carries no usable topology: topology_assignment absent, malformed
    (``positions`` not a non-empty dict), or naming fewer than 2 distinct
    node_ids (e.g. the single-node default path, where topology_assignment is
    None at runtime).

    Otherwise returns a list of ``(node_id, Decimal amount)`` splits whose
    amounts sum to ``release`` to within one Decimal ULP — no FTNS is created
    or lost; only the distribution changes, and the total stays bounded by
    ``release`` regardless of the topology contents (so a wrong/forged topology
    can mis-distribute but never over-pay), with any sub-ULP residual refunded
    to the payer by the escrow's remainder math. Equal-weight policy:
    layer-balanced stages do comparable work, so the release splits evenly; the
    non-terminating-division residual (e.g. 1/3) is folded into the first
    stage's node. Per-node FLOP/priced weighting + on-chain cross-node
    settlement are explicit follow-ons (this is the off-chain crediting half of
    the Tier-1 loop-closer).

    Known latent limit (not reachable at realistic inference pricing, fail-safe
    if it ever is): the caller float()-converts these Decimal shares and
    PaymentEscrow.release_escrow_split re-sums them against an absolute 1e-9
    tolerance, so at implausibly large single-job escrows (~1e6+ FTNS) the
    accumulated float error could exceed tolerance and strand the job (surfaced
    error, never an over-pay). The proper fix is a magnitude-relative tolerance
    in release_escrow_split — deferred to its own payment-escrow sprint.
    """
    topology = getattr(receipt, "topology_assignment", None)
    if topology is None:
        return None
    positions = getattr(topology, "positions", None)
    if not isinstance(positions, dict) or not positions:
        return None

    # Distinct node_ids in deterministic (stage, slot) order so the drift
    # remainder lands deterministically on the first stage's node.
    distinct: list = []
    try:
        ordered_keys = sorted(positions.keys())
    except TypeError:
        ordered_keys = list(positions.keys())
    for key in ordered_keys:
        node_id = positions[key]
        if isinstance(node_id, str) and node_id and node_id not in distinct:
            distinct.append(node_id)
    if len(distinct) < 2:
        return None

    n = len(distinct)
    per = release / Decimal(n)
    splits = [(node_id, per) for node_id in distinct]
    # Fold the Decimal-division residual (e.g. 1/3) into the first node so the
    # sum equals release to within one Decimal ULP; any sub-ULP residual is an
    # undershoot the escrow refunds to the payer (never an over-pay).
    drift = release - sum((amt for _, amt in splits), Decimal("0"))
    if drift != 0:
        first_id, first_amt = splits[0]
        splits[0] = (first_id, first_amt + drift)
    return splits


async def settle_inference_receipt(
    *,
    payment_escrow: "object",
    receipt: InferenceReceipt,
    operator_id: str,
    job_id: str,
    escrow_amount: Optional[Decimal] = None,
) -> SettlementDecision:
    """Sprint 783 — settle a job's escrow per the receipt's
    partial-completion marker.

    Branches:
      release > 0 + refund > 0  → release_escrow_split (operator
        gets proportional; payer auto-refunded the remainder by
        PaymentEscrow's existing split-API behavior)
      release > 0 + refund == 0 → release_escrow (full payout)
      release == 0 + refund > 0 → refund_escrow (no work credited)

    Returns the SettlementDecision so the caller can dispatch
    slash-emission based on decision.should_slash (slash hook
    itself is out of scope here — separate sprint).

    escrow_amount defaults to receipt.cost_ftns. Callers pass
    the actual locked escrow amount when it differs (over-funded
    escrow, dynamic-pricing differential)."""
    decision = compute_settlement_for_receipt(
        receipt, escrow_amount,
    )

    release = decision.release_to_operator
    refund = decision.refund_to_payer

    # Sprint 1031 — on a cross-host inference the receipt's signed
    # topology_assignment names every stage's node_id. Fan the release out
    # across the distinct stage nodes (each ran a slice) instead of paying only
    # the local settler. Gated on the receipt having been settled by THIS
    # operator (settler_node_id == operator_id, i.e. locally built + signed) so
    # the topology-driven split cannot be repurposed to credit a remote-supplied
    # topology if a future caller ever feeds a remote receipt into settlement.
    # None on the single-node default path (no topology) → unchanged.
    stage_splits = None
    if release > 0 and getattr(receipt, "settler_node_id", "") == operator_id:
        stage_splits = split_release_across_stages(receipt, release)

    if stage_splits is not None:
        # Multi-stage: split sums to `release`; release_escrow_split
        # auto-refunds the remainder (escrow_amount - release) to the payer.
        float_splits = [
            (node_id, float(amount)) for node_id, amount in stage_splits
        ]
        # Parity with the swarm/forge split path (api.py:6510): resolve each
        # stage node_id to the operator's registered FTNS wallet so a
        # multi-agent operator's per-stage credits consolidate onto one wallet.
        # Empty map (default) is a no-op — recipients fall through to
        # themselves; never let wallet-map resolution block payment.
        try:
            from prsm.node.compute_wallet_map import (
                ComputeWalletMap,
                resolve_splits,
            )
            float_splits = resolve_splits(
                float_splits, ComputeWalletMap.from_env(),
            )
        except Exception:
            pass
        await payment_escrow.release_escrow_split(job_id, float_splits)
    elif release > 0 and refund > 0:
        # Partial (single payee): split-API handles remainder refund.
        await payment_escrow.release_escrow_split(
            job_id,
            [(operator_id, float(release))],
        )
    elif release > 0:
        # Full payout (single payee) — existing escrow-flow path.
        await payment_escrow.release_escrow(job_id, operator_id)
    elif refund > 0:
        # No work credited.
        await payment_escrow.refund_escrow(
            job_id,
            reason=(
                f"settle_inference_receipt: zero release per "
                f"partial_completion marker"
            ),
        )

    return decision


def aggregate_earnings_by_node_id(
    receipts: Iterable[Union[Dict[str, Any], InferenceReceipt]],
) -> Dict[str, Decimal]:
    """Sprint 791 — sum effective_cost_for_receipt per settler_node_id.

    Multi-device operators want per-device earnings breakdowns —
    "device A earned X, device B earned Y" — so they can spot
    underperforming devices in their roster. This is the pure
    aggregator that any caller (CLI, dashboard, daemon endpoint)
    can drop in.

    Inputs:
      - Iterable of receipts. Accepts both persisted-shape dicts
        (`InferenceReceipt.to_dict()` output) and dataclass
        instances directly.

    Output:
      - Dict mapping settler_node_id → total credited Decimal.
        Uses sprint-780 effective_cost_for_receipt so
        partial_completion is honored (preempted partials credit
        proportionally; full successes credit full).

    Defensive:
      - Malformed receipts (missing settler_node_id, missing
        cost_ftns, empty node_id) are SKIPPED — don't crash an
        earnings summary on one bad row.
      - Never raises on per-row failures; per-row errors are
        silent (caller's earnings summary stays well-formed even
        with hostile/corrupt ReceiptStore rows).
    """
    totals: Dict[str, Decimal] = {}
    for r in receipts:
        try:
            if isinstance(r, InferenceReceipt):
                receipt = r
            elif isinstance(r, dict):
                # Skip dicts missing the load-bearing fields rather
                # than letting from_dict raise mid-aggregation.
                if not r.get("settler_node_id"):
                    continue
                if "cost_ftns" not in r:
                    continue
                receipt = InferenceReceipt.from_dict(r)
            else:
                continue
        except Exception:
            continue

        node_id = receipt.settler_node_id
        if not node_id:
            continue

        try:
            credit = effective_cost_for_receipt(receipt)
        except Exception:
            continue

        totals[node_id] = totals.get(node_id, Decimal("0")) + credit
    return totals
