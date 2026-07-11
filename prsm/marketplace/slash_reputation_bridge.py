"""Sprint 1424 — bridge on-chain slashes into provider reputation.

THE HOLE: `ReputationTracker` is READ on every marketplace dispatch
(`MarketplaceCandidatePoolProvider.__call__` -> `score_for(provider_id)`), but on the live path
NOTHING ever writes to it. `record_success`/`record_failure` are only called from
`MarketplaceOrchestrator`, which is never instantiated in production, and `record_slash` had ZERO
production callers anywhere. So every provider's `score_for(...)` returned the constant NEUTRAL_SCORE
(0.5) forever, and `has_been_slashed(...)` returned False for everyone — including a provider caught
double-spending or forging signatures and **slashed on-chain by a successful challenge**. They kept
full selection weight, and `/marketplace/reputation/*` reported them clean.

The on-chain data already existed: `StakeManagerClient.get_all_slash_events()` (sp1424) decodes the
real `StakeBond.Slashed(provider, challenger, reasonId=batchId, slashAmount, ...)` events. Nothing
piped them into the tracker. This module is that pipe.

KEY-SPACE BRIDGE: the tracker is keyed by ``provider_id`` == a PRSM ``node_id``
(``sha256(pubkey)[:32]``), but `Slashed.provider` is the on-chain OPERATOR ETH ADDRESS. So each
slash must be mapped eth-address -> node_id before it can affect a score. The mapping is injected
(``resolve_node_id``); in production it comes from the same self-advertised
``hardware_profile.operator_address`` the sp1309 discovery resolver uses.

TRUST: a peer advertises its OWN operator_address in its OWN hardware_profile, so this mapping can
only ever apply a slash to the advertising peer's node_id (self-harm) or to the address's genuine
owner (correct). It cannot forge a penalty onto an innocent third party — a peer cannot make
someone ELSE's node_id absorb a slash. An unmappable slash (operator never seen in discovery) is
COUNTED and LOGGED, never silently dropped.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Recorded against the tracker as the human reason. The Slashed event carries the batchId (in its
# reasonId field), not the ReasonCode, so the specific fraud reason is not in THIS event — and it
# does not affect the score (a slash is SLASH_WEIGHT failures regardless). Kept explicit + honest.
_ONCHAIN_SLASH_REASON = "ONCHAIN_SLASH"


def slash_dedup_key(event: Any) -> Tuple[str, str]:
    """Idempotency key for a slash. (tx_hash, reasonId=batchId) uniquely identifies one on-chain
    Slashed log — re-scanning overlapping block windows must not double-count it into the deque."""
    return (
        str(getattr(event, "tx_hash", "")).lower(),
        str(getattr(event, "reason_id", "")).lower(),
    )


def apply_onchain_slashes_to_reputation(
    *,
    events: Iterable[Any],
    tracker: Any,
    resolve_node_id: Callable[[str], Optional[str]],
    already_recorded: Set[Tuple[str, str]],
) -> Tuple[int, int]:
    """Record each NEW on-chain slash into ``tracker`` under the provider's node_id.

    ``events``: iterable of ``stake_manager.SlashEvent``.
    ``tracker``: a ``ReputationTracker`` (calls ``record_slash``).
    ``resolve_node_id``: ``(operator_eth_address) -> node_id | None`` (injected; discovery-backed
        in prod). Returns None when the operator is not known — that slash is counted as unmapped.
    ``already_recorded``: a MUTABLE dedup set (see ``slash_dedup_key``), updated in place so the
        caller's next scan skips slashes already applied.

    Returns ``(recorded, unmapped)``. Never raises — a single bad event is logged and skipped so
    one malformed log cannot blind the whole sweep.
    """
    recorded = 0
    unmapped = 0
    for ev in events:
        try:
            key = slash_dedup_key(ev)
            if key in already_recorded:
                continue

            operator = str(getattr(ev, "provider", "") or "")
            node_id = None
            try:
                node_id = resolve_node_id(operator) if operator else None
            except Exception as exc:  # noqa: BLE001 — a resolver hiccup must not abort the sweep
                logger.debug("sp1424 slash node-id resolve failed for %s: %s", operator, exc)
                node_id = None

            if not node_id:
                # Mark it seen so we don't re-log it every cycle, but count it: an operator we
                # cannot map today may become mappable once it appears in discovery, and either
                # way the operator should know a slashed provider is out there unaccounted for.
                already_recorded.add(key)
                unmapped += 1
                logger.warning(
                    "sp1424: on-chain slash of operator %s (batch %s, %s wei) could NOT be mapped "
                    "to a known node_id — it is NOT reflected in reputation. If this provider is in "
                    "your pool, it currently keeps full selection weight despite being slashed.",
                    operator, getattr(ev, "reason_id", "?"), getattr(ev, "slash_amount_wei", "?"),
                )
                continue

            tracker.record_slash(
                provider_id=node_id,
                batch_id=str(getattr(ev, "reason_id", "") or "unknown"),
                slash_amount_wei=int(getattr(ev, "slash_amount_wei", 0) or 0),
                reason=_ONCHAIN_SLASH_REASON,
                tx_hash=str(getattr(ev, "tx_hash", "") or "") or None,
            )
            already_recorded.add(key)
            recorded += 1
            logger.critical(
                "sp1424: recorded on-chain SLASH against provider node %s (operator %s, batch %s, "
                "%s wei) — its reputation score is now penalized and has_been_slashed()=True.",
                node_id, operator, getattr(ev, "reason_id", "?"),
                getattr(ev, "slash_amount_wei", "?"),
            )
        except Exception as exc:  # noqa: BLE001 — never let one event abort the batch
            logger.error("sp1424: failed to apply a slash event (%s); skipped.", exc)

    return recorded, unmapped
