"""Sprint 296 — topology rotation per inference.

Vision §7 honest-limits names topology rotation as a defense
against "colluding minorities in the right positions" — a
coordinated subset of nodes that spans the pipeline can in
principle reconstruct activations end-to-end if they capture
their per-position slot data and pool it. Rotation per
inference forces the adversary to get lucky every time, not
just once.

This module ships the primitive that the streaming-inference
subsystem will adopt to randomize (stage_index, slot_index)
→ node_id assignment each inference:

  TopologyAssignment
    Frozen-style dataclass holding the position→node map.
    Provides stable_hash() for de-dup in history tracking.

  TopologySelector
    Pool + dims + seed → TopologyAssignment with a uniform-
    random shuffle. Enforces each node fills at most one
    position (defends operator double-duty attack).

  TopologyRotationPolicy
    Three strategies:
      "uniform_random" — independent random each call
      "beacon_seeded"  — deterministic from seed_hint; lets
                         on-chain verifiers replay rotation
      "anti_repeat"    — guarantees distinctness from the
                         last N topologies in history

  TopologyHistory
    Bounded ring of recently-issued TopologyAssignments.
    Used by anti_repeat strategy + by verify_topology_
    sequence.

  verify_topology_sequence
    Verifier predicate. Checks structural integrity (no
    duplicate node positions, all positions filled) +
    rotation distinctness against a recent-history window.

Sprint 297 will wire TopologyRotationPolicy into
RpcChainExecutor and surface the chosen TopologyAssignment
as a receipt field so verifiers can confirm rotation
actually happened.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any, Deque, Dict, List, Optional, Tuple,
)


_VALID_STRATEGIES = {
    "uniform_random",
    "beacon_seeded",
    "anti_repeat",
}

_ANTI_REPEAT_MAX_ATTEMPTS = 64


@dataclass
class TopologyAssignment:
    """Per-inference (stage, slot) → node_id assignment."""

    positions: Dict[Tuple[int, int], str]
    stage_count: int
    slots_per_stage: int

    def stable_hash(self) -> str:
        """Stable hex hash for de-dup + replay-verification.
        Sorting by (stage, slot) makes the hash invariant to
        dict iteration order."""
        sorted_pos = sorted(
            self.positions.items(),
            key=lambda kv: kv[0],
        )
        canonical = json.dumps(
            [
                [s, sl, node]
                for ((s, sl), node) in sorted_pos
            ],
            sort_keys=True,
        )
        return hashlib.sha256(
            canonical.encode("utf-8"),
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": [
                [s, sl, node]
                for ((s, sl), node) in sorted(
                    self.positions.items(),
                    key=lambda kv: kv[0],
                )
            ],
            "stage_count": self.stage_count,
            "slots_per_stage": self.slots_per_stage,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "TopologyAssignment":
        positions: Dict[Tuple[int, int], str] = {}
        for entry in d.get("positions", []):
            s, sl, node = entry
            positions[(int(s), int(sl))] = str(node)
        return cls(
            positions=positions,
            stage_count=int(d.get("stage_count", 0)),
            slots_per_stage=int(
                d.get("slots_per_stage", 0),
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TopologyAssignment):
            return False
        return (
            self.stage_count == other.stage_count
            and self.slots_per_stage == other.slots_per_stage
            and self.positions == other.positions
        )

    def __hash__(self) -> int:
        return int(self.stable_hash()[:16], 16)


def topology_from_chain_stages(stages: Any) -> "TopologyAssignment":
    """Sprint 1313 (S1) — build the canonical per-inference ``TopologyAssignment`` from a
    routed chain's ordered stage node_ids: position ``(stage_index, 0) → node_id``, one slot
    per stage.

    This is the SINGLE source for the stage→node positions, shared by
    ``TopologyAwareChainExecutor`` (the SERVE path — records it on the receipt) and
    ``ParallaxScheduledExecutor.plan_topology`` (the QUOTE path). Using one helper guarantees
    a quote's payee set matches what a real serve produces for the SAME routed chain (no
    quote/serve drift)."""
    stage_list = list(stages or [])
    positions = {(i, 0): node_id for i, node_id in enumerate(stage_list)}
    return TopologyAssignment(
        positions=positions,
        stage_count=len(stage_list),
        slots_per_stage=1,
    )


class TopologySelector:
    """Pool + dims + seed → TopologyAssignment via uniform
    random shuffle."""

    def select(
        self,
        *,
        node_pool: List[str],
        stage_count: int,
        slots_per_stage: int,
        seed: int,
    ) -> TopologyAssignment:
        if not isinstance(stage_count, int) or stage_count <= 0:
            raise ValueError(
                f"stage_count must be a positive integer, "
                f"got {stage_count!r}"
            )
        if (
            not isinstance(slots_per_stage, int)
            or slots_per_stage <= 0
        ):
            raise ValueError(
                f"slots_per_stage must be a positive "
                f"integer, got {slots_per_stage!r}"
            )
        if not node_pool:
            raise ValueError("node_pool must be non-empty")
        needed = stage_count * slots_per_stage
        if len(node_pool) < needed:
            raise ValueError(
                f"node_pool too small: need {needed} "
                f"(stage_count={stage_count} × "
                f"slots_per_stage={slots_per_stage}), "
                f"have {len(node_pool)}"
            )

        rng = random.Random(seed)
        shuffled = list(node_pool)
        rng.shuffle(shuffled)
        positions: Dict[Tuple[int, int], str] = {}
        idx = 0
        for s in range(stage_count):
            for sl in range(slots_per_stage):
                positions[(s, sl)] = shuffled[idx]
                idx += 1
        return TopologyAssignment(
            positions=positions,
            stage_count=stage_count,
            slots_per_stage=slots_per_stage,
        )


class TopologyHistory:
    """Bounded ring of recently-issued TopologyAssignments.
    Newest entries at the front; eviction is FIFO from the
    back."""

    def __init__(self, max_entries: int) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[TopologyAssignment] = deque(
            maxlen=max_entries,
        )

    def record(self, topology: TopologyAssignment) -> None:
        # Insert at front so recent_hashes() naturally returns
        # newest first.
        self._entries.appendleft(topology)

    def count(self) -> int:
        return len(self._entries)

    def recent_hashes(self) -> List[str]:
        return [t.stable_hash() for t in self._entries]

    def contains(
        self, topology: TopologyAssignment,
    ) -> bool:
        target = topology.stable_hash()
        return any(
            t.stable_hash() == target
            for t in self._entries
        )


@dataclass
class TopologyRotationPolicy:
    """Selects topologies under one of three strategies."""

    strategy: str = "uniform_random"
    anti_repeat_window: int = 3
    _selector: TopologySelector = field(
        default_factory=TopologySelector,
    )

    def __post_init__(self) -> None:
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of "
                f"{sorted(_VALID_STRATEGIES)}, "
                f"got {self.strategy!r}"
            )

    def next_topology(
        self,
        *,
        node_pool: List[str],
        stage_count: int,
        slots_per_stage: int,
        history: TopologyHistory,
        seed_hint: int,
    ) -> TopologyAssignment:
        if self.strategy in (
            "uniform_random", "beacon_seeded",
        ):
            # uniform_random + beacon_seeded both use the
            # seed_hint deterministically. Callers wanting
            # true randomness pass a time-derived seed.
            return self._selector.select(
                node_pool=node_pool,
                stage_count=stage_count,
                slots_per_stage=slots_per_stage,
                seed=seed_hint,
            )

        # anti_repeat: pick a topology not in the recent
        # history window. Retry with a derived seed up to
        # _ANTI_REPEAT_MAX_ATTEMPTS.
        recent_window = history.recent_hashes()[
            :self.anti_repeat_window
        ]
        for attempt in range(_ANTI_REPEAT_MAX_ATTEMPTS):
            seed = seed_hint * 7919 + attempt * 17
            candidate = self._selector.select(
                node_pool=node_pool,
                stage_count=stage_count,
                slots_per_stage=slots_per_stage,
                seed=seed,
            )
            if candidate.stable_hash() not in recent_window:
                return candidate
        raise ValueError(
            f"anti_repeat strategy could not find a distinct "
            f"topology in {_ANTI_REPEAT_MAX_ATTEMPTS} "
            f"attempts (pool size {len(node_pool)} may be too "
            f"small for anti_repeat_window="
            f"{self.anti_repeat_window})"
        )


def verify_topology_sequence(
    sequence: List[TopologyAssignment],
    *,
    expected_anti_repeat_window: int,
) -> Tuple[bool, str]:
    """Verifier for a sequence of topologies. Returns
    (ok, reason). Checks structural integrity per-topology
    + rotation distinctness across the sequence.

    Structural per-topology:
      - all (stage, slot) ∈ [0,stage_count)×[0,slots_per_stage)
        present
      - no node assigned to two positions in the same topology

    Sequence:
      - no topology appears within
        expected_anti_repeat_window of an earlier occurrence
    """
    for idx, topo in enumerate(sequence):
        # All positions filled
        expected_positions = {
            (s, sl)
            for s in range(topo.stage_count)
            for sl in range(topo.slots_per_stage)
        }
        actual_positions = set(topo.positions.keys())
        missing = expected_positions - actual_positions
        if missing:
            return (
                False,
                f"topology {idx} incomplete: missing "
                f"positions {sorted(missing)}",
            )
        extra = actual_positions - expected_positions
        if extra:
            return (
                False,
                f"topology {idx} has positions outside "
                f"the (stage_count, slots_per_stage) grid: "
                f"{sorted(extra)}",
            )
        # No duplicate node across positions
        node_counts: Dict[str, int] = {}
        for node in topo.positions.values():
            node_counts[node] = (
                node_counts.get(node, 0) + 1
            )
        dups = [
            node for node, n in node_counts.items() if n > 1
        ]
        if dups:
            return (
                False,
                f"topology {idx} has duplicated nodes "
                f"in multiple positions: {sorted(dups)}",
            )

    # Rotation distinctness across the sequence
    if expected_anti_repeat_window > 0:
        for i in range(len(sequence)):
            window_start = max(
                0, i - expected_anti_repeat_window,
            )
            for j in range(window_start, i):
                if (
                    sequence[i].stable_hash()
                    == sequence[j].stable_hash()
                ):
                    return (
                        False,
                        f"topology repeat within window: "
                        f"index {i} duplicates index {j} "
                        f"(window={expected_anti_repeat_window})",
                    )
    return (True, "")
