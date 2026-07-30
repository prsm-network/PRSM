"""Sprint 1481 — turn FINALIZED settlement batches into an emission epoch.

This is the middle link of the pool → earner rail::

    BatchSettlementRegistry.BatchFinalized events   (real, settled work)
        -> build_emission_epoch()                   (this module: who earned what)
        -> build_reward_epoch()                     (reward_epoch.py: root + proofs)
        -> OperatorRewardPool.publishEpoch()        (on chain: reserve + allow claims)

Two properties carry the money here, and both are enforced rather than documented:

**1. Exactly-once attribution.** A batch must contribute to exactly ONE epoch. If a
batch were included in two epochs the provider is paid twice for the same work, out
of a pot that is supposed to be shared. Every batch carries a unique on-chain
``batch_id``, so the builder takes the set of ids already consumed by previous
epochs, refuses to re-count them, and returns the ids it consumed so the caller can
persist the new watermark. This is the same shape as the deposit/withdraw
idempotency work (sp1472/sp1474): make the double-count unrepresentable rather than
relying on the caller to window correctly.

**2. Exact allocation.** ``sum(entitlements)`` MUST equal the epoch pot exactly.
Naive integer division loses dust; a pro-rata split that overshoots would publish a
root whose leaves sum to more than the reserved total, which the contract then
rejects mid-epoch (``EpochOverdrawn``) — i.e. the last earners cannot claim. The
largest-remainder (Hamilton) method below distributes every wei, and zero-share
providers are dropped and re-allocated (the contract rejects a zero-amount claim, so
a zero leaf would be permanently unclaimable dead weight inside the reserve).

Only FINALIZED batches are eligible: a PENDING batch can still be challenged and
its value reduced, so paying emissions on it would pay for work that may yet be
invalidated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from eth_utils import to_checksum_address

from prsm.settlement.reward_epoch import RewardEntry, RewardEpoch, build_reward_epoch


@dataclass(frozen=True)
class FinalizedBatch:
    """One BatchSettlementRegistry.BatchFinalized observation.

    ``final_value_wei`` is the event's ``finalValueFTNS`` — already net of
    successfully-challenged value, so it is the settled truth of what the provider
    actually delivered.
    """
    batch_id: str            # 0x… bytes32, unique on chain — the idempotency key
    provider: str            # 0x… EVM address, the earner
    final_value_wei: int
    finalize_timestamp: int  # unix seconds


@dataclass(frozen=True)
class EmissionEpochPlan:
    """The result of planning an epoch: what to publish and what it consumed."""
    epoch_id: int
    pot_wei: int
    entries: List[RewardEntry]
    consumed_batch_ids: List[str]
    skipped_already_consumed: List[str]
    contribution_wei: Dict[str, int]      # provider -> settled value in-window

    @property
    def total_allocated_wei(self) -> int:
        return sum(e.amount_wei for e in self.entries)


def allocate_pro_rata(weights: Mapping[str, int], pot_wei: int) -> Dict[str, int]:
    """Split ``pot_wei`` across ``weights`` so the parts sum EXACTLY to the pot.

    Largest-remainder (Hamilton) apportionment: floor each share, then hand the
    leftover wei one at a time to the largest fractional remainders. Ties break on
    the account key so the result is deterministic and independently reproducible —
    two parties building the same epoch must get byte-identical roots.

    Providers whose exact share floors to zero are dropped and the pot is
    re-apportioned among the rest, repeating until every remaining share is >= 1.
    (A zero-amount leaf is unclaimable on chain but would still sit inside the
    reserved total.) With a pot smaller than the number of providers this converges
    to at most ``pot_wei`` recipients.
    """
    if pot_wei < 0:
        raise ValueError(f"pot_wei must be non-negative, got {pot_wei}")
    live = {k: int(v) for k, v in weights.items() if int(v) > 0}
    if not live or pot_wei == 0:
        return {}

    while True:
        total_weight = sum(live.values())
        # Exact floor shares + fractional remainders as (numerator, denominator).
        base: Dict[str, int] = {}
        remainders: List[Tuple[int, str]] = []
        for acct, w in live.items():
            num = pot_wei * w
            base[acct] = num // total_weight
            remainders.append((num % total_weight, acct))

        leftover = pot_wei - sum(base.values())
        # Largest remainder first; ties -> lexicographic account for determinism.
        remainders.sort(key=lambda t: (-t[0], t[1]))
        for i in range(leftover):
            base[remainders[i][1]] += 1

        zeros = [a for a, v in base.items() if v == 0]
        if not zeros:
            assert sum(base.values()) == pot_wei, "apportionment lost or created wei"
            return base
        for a in zeros:
            del live[a]
        if not live:
            return {}


def build_emission_epoch(
    *,
    epoch_id: int,
    batches: Iterable[FinalizedBatch],
    pot_wei: int,
    consumed_batch_ids: Optional[Set[str]] = None,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
) -> EmissionEpochPlan:
    """Plan an emission epoch from finalized batches.

    :param pot_wei: the emission to distribute this epoch. The plan's entitlements
        sum to EXACTLY this (or the epoch is unpublishable and raises).
    :param consumed_batch_ids: batch ids already paid by a previous epoch. These are
        skipped — this is the exactly-once guarantee.
    :param window_start/window_end: optional inclusive/exclusive finalize-timestamp
        bounds. Windowing alone is NOT sufficient for exactly-once (a batch
        finalized on a boundary, or a re-scan with shifted bounds, would double
        count) — ``consumed_batch_ids`` is the authoritative guard.
    """
    consumed = {b.lower() for b in (consumed_batch_ids or set())}
    contribution: Dict[str, int] = {}
    newly_consumed: List[str] = []
    skipped: List[str] = []
    seen_this_run: Set[str] = set()

    for b in batches:
        bid = b.batch_id.lower()
        if window_start is not None and b.finalize_timestamp < window_start:
            continue
        if window_end is not None and b.finalize_timestamp >= window_end:
            continue
        if bid in consumed:
            skipped.append(b.batch_id)
            continue
        if bid in seen_this_run:
            # The same batch appearing twice in one scan (an RPC returning
            # overlapping ranges) must not be counted twice either.
            skipped.append(b.batch_id)
            continue
        if b.final_value_wei <= 0:
            # Fully invalidated by challenges — real work did not settle.
            seen_this_run.add(bid)
            newly_consumed.append(b.batch_id)
            continue
        seen_this_run.add(bid)
        newly_consumed.append(b.batch_id)
        acct = to_checksum_address(b.provider)
        contribution[acct] = contribution.get(acct, 0) + int(b.final_value_wei)

    if not contribution:
        raise ValueError(
            f"build_emission_epoch: epoch {epoch_id} has no eligible finalized "
            "batches (nothing to distribute — do not publish an empty epoch)"
        )

    allocation = allocate_pro_rata(contribution, pot_wei)
    if not allocation:
        raise ValueError(
            f"build_emission_epoch: epoch {epoch_id} pot ({pot_wei} wei) is too "
            "small to give any provider a non-zero share"
        )

    entries = [
        RewardEntry(account=a, amount_wei=v)
        for a, v in sorted(allocation.items(), key=lambda kv: int(kv[0], 16))
    ]
    return EmissionEpochPlan(
        epoch_id=int(epoch_id),
        pot_wei=int(pot_wei),
        entries=entries,
        consumed_batch_ids=newly_consumed,
        skipped_already_consumed=skipped,
        contribution_wei=contribution,
    )


def plan_to_reward_epoch(plan: EmissionEpochPlan) -> RewardEpoch:
    """Build the publishable Merkle epoch (root + proofs) from a plan."""
    return build_reward_epoch(plan.epoch_id, plan.entries)
