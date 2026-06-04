"""Sprint 238 — PCU-weighted compute-participant split.

Closes the §4-step-6 deferred sub-item flagged by
`project_phase5_mcp_aggregate_source_2026_05_09.md`: the prompter's
compute budget is now distributed proportionally to actual PCU
consumed per participant when telemetry permits, falling back to
uniform-split for legacy callers mid-migration.

Strategy
--------

Given N compute participants (each carrying ``pcu_consumed: float``),
an aggregator coordination share `aggregator_share_bps` of the
total budget, and a `total_budget` FTNS:

  1. Carve off `aggregator_share = total_budget * bps / 10000`.
  2. The remainder = `compute_share_total`.
  3. If **all** participants have ``pcu_consumed > 0``: weight each
     participant's slice by ``pcu_consumed / sum(pcu_consumed)``.
  4. Otherwise: uniform split (`compute_share_total / N`).

The all-or-nothing trigger avoids punishing legacy callers in the
middle of a rollout — when even one participant's PCU is missing,
we don't know how to fairly weight, so uniform is the right
fallback rather than "treat missing as zero".

Return shape: ``(splits, mode)`` where:
  - ``splits`` is the ``[(recipient_id, amount), ...]`` list ready
    to pass to ``PaymentEscrow.release_escrow_split``.
  - ``mode`` is ``"pcu_weighted"`` or ``"uniform"`` — useful for
    telemetry / audit logging.

Aggregator entry is **omitted** when ``aggregator_share_bps == 0``
to avoid zero-amount transactions in the ledger.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _worker_recipient(p: Dict[str, Any]) -> str:
    """Payout recipient for a worker leg.

    sp998 — prefer the worker's node_id (the identity it self-assigns + the key
    the aggregator leg and the ledger-sync credit path already use), falling back
    to the raw pubkey hex only for legacy callers that don't supply it. The raw
    pubkey hex is NOT a payable identity: it is neither a 0x address (so the
    on-chain bridge's _resolve_address rejects it) nor a node_id (so a remote
    peer's `to_wallet == self.node_id` ledger-sync credit never matches), which
    stranded every worker's FTNS on a local-only dead wallet.
    """
    return p.get("source_agent_node_id") or p["source_agent_pubkey_hex"]


def compute_split_amounts(
    *,
    participants: List[Dict[str, Any]],
    aggregator_node_id: str,
    total_budget: float,
    aggregator_share_bps: int,
) -> Tuple[List[Tuple[str, float]], str]:
    """Compute the per-recipient split for a swarm query release.

    Each participant dict must carry ``source_agent_pubkey_hex``
    and may carry ``pcu_consumed`` (default 0). The function does
    not mutate input.

    Raises
    ------
    ValueError
        ``total_budget <= 0`` or ``aggregator_share_bps`` not in
        ``[0, 10000]``.
    """
    if total_budget <= 0:
        raise ValueError(
            f"total_budget must be > 0; got {total_budget}"
        )
    if not (0 <= aggregator_share_bps <= 10000):
        raise ValueError(
            f"aggregator_share_bps must be in [0, 10000]; "
            f"got {aggregator_share_bps}"
        )

    aggregator_share = total_budget * (aggregator_share_bps / 10000.0)
    compute_share_total = total_budget - aggregator_share

    splits: List[Tuple[str, float]] = []
    if aggregator_share > 0:
        splits.append((aggregator_node_id, aggregator_share))

    if not participants:
        return splits, "uniform"

    # Decide mode: PCU-weighted only when ALL participants have
    # pcu_consumed > 0. Mixed = uniform fallback.
    pcus = [
        float(p.get("pcu_consumed", 0.0) or 0.0)
        for p in participants
    ]
    if all(pcu > 0 for pcu in pcus):
        total_pcu = sum(pcus)
        for p, pcu in zip(participants, pcus):
            recipient = _worker_recipient(p)
            amount = compute_share_total * (pcu / total_pcu)
            splits.append((recipient, amount))
        return splits, "pcu_weighted"

    # Uniform fallback.
    n = len(participants)
    per_participant = compute_share_total / n
    for p in participants:
        splits.append((_worker_recipient(p), per_participant))
    return splits, "uniform"
