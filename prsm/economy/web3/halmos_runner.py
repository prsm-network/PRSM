"""Sprint 360 — halmos symbolic-execution runner.

Bridges the formal-invariants harness (sprint 302+356-359
runtime probe) with halmos symbolic execution. The runtime
probe answers "is the LIVE state in spec right now?"; halmos
answers "can ANY reachable state break the spec?"

Both layers consume the same Vision §14 item 4 commitment.

Module structure:
  SymbolicProofResult — typed outcome (passed/failed/error)
  SymbolicProofSuite  — collection result for one contract
  HalmosRunner        — invokes halmos via subprocess +
                        parses output. Fail-soft if halmos
                        or forge isn't installed (returns
                        a SKIPPED suite naming the missing
                        tool); never raises.

Halmos invocation is intentionally fail-soft: most operators
won't have halmos installed, and absence of symbolic proofs
should not crash the existing runtime-probe surface.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# Path constants — relative to repo root.
_SYMBOLIC_PROOFS_DIR = "contracts/symbolic-proofs"
_DEFAULT_TIMEOUT_SECONDS = 600


class SymbolicProofStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class SymbolicProofResult:
    name: str
    status: SymbolicProofStatus
    paths_explored: int = 0
    time_seconds: float = 0.0
    error: Optional[str] = None
    counterexample: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "paths_explored": self.paths_explored,
            "time_seconds": self.time_seconds,
            "error": self.error,
            "counterexample": self.counterexample,
        }


@dataclass
class SymbolicProofSuite:
    contract: str
    status: SymbolicProofStatus
    proofs: List[SymbolicProofResult] = field(
        default_factory=list,
    )
    error: Optional[str] = None
    halmos_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "status": self.status.value,
            "halmos_version": self.halmos_version,
            "proofs": [p.to_dict() for p in self.proofs],
            "error": self.error,
            "summary": {
                "passed": sum(
                    1 for p in self.proofs
                    if p.status == SymbolicProofStatus.PASSED
                ),
                "failed": sum(
                    1 for p in self.proofs
                    if p.status == SymbolicProofStatus.FAILED
                ),
                "errored": sum(
                    1 for p in self.proofs
                    if p.status == SymbolicProofStatus.ERROR
                ),
            },
        }


class HalmosRunner:
    """Runs halmos against the symbolic-proofs lane.

    Detects halmos + forge availability via PATH lookup;
    when either is missing, run() returns a SKIPPED suite
    with the missing tool named in the error field. Never
    raises — the caller can render the result as "halmos
    not installed" UX without crash handling.
    """

    def __init__(
        self,
        *,
        proofs_dir: str = _SYMBOLIC_PROOFS_DIR,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        halmos_bin: Optional[str] = None,
        forge_bin: Optional[str] = None,
        loop_bound: int = 32,
    ) -> None:
        self._proofs_dir = proofs_dir
        self._timeout = timeout_seconds
        # Allow explicit override (test injection); otherwise
        # resolve via PATH.
        self._halmos = halmos_bin
        self._forge = forge_bin
        # Halmos default loop unroll is 2 — too low for the
        # chunk-streaming bounded-iterator proofs (sprint
        # 368). 32 covers all current specs without solver
        # blow-up; override via constructor arg for proofs
        # that need deeper unrolling.
        self._loop_bound = loop_bound

    def _resolve_halmos(self) -> Optional[str]:
        if self._halmos:
            return self._halmos
        return shutil.which("halmos")

    def _resolve_forge(self) -> Optional[str]:
        if self._forge:
            return self._forge
        return shutil.which("forge")

    def is_available(self) -> bool:
        return (
            self._resolve_halmos() is not None
            and self._resolve_forge() is not None
        )

    def missing_tools(self) -> List[str]:
        missing: List[str] = []
        if self._resolve_halmos() is None:
            missing.append("halmos")
        if self._resolve_forge() is None:
            missing.append("forge")
        return missing

    def run(self, contract: str) -> SymbolicProofSuite:
        """Run halmos against a single Foundry contract.

        contract — the Solidity contract name (e.g.,
        FTNSSupplyCapSpec) NOT a file path. Halmos matches
        on contract name across all build artifacts.
        """
        halmos = self._resolve_halmos()
        forge = self._resolve_forge()
        if halmos is None or forge is None:
            missing = self.missing_tools()
            return SymbolicProofSuite(
                contract=contract,
                status=SymbolicProofStatus.SKIPPED,
                error=(
                    f"missing tools: {', '.join(missing)}. "
                    f"install halmos via 'pip install halmos' "
                    f"and forge via 'foundryup'."
                ),
            )
        if not os.path.isdir(self._proofs_dir):
            return SymbolicProofSuite(
                contract=contract,
                status=SymbolicProofStatus.SKIPPED,
                error=(
                    f"symbolic-proofs dir not found: "
                    f"{self._proofs_dir}"
                ),
            )
        cmd = [
            halmos,
            "--contract", contract,
            "--loop", str(self._loop_bound),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=self._proofs_dir,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return SymbolicProofSuite(
                contract=contract,
                status=SymbolicProofStatus.ERROR,
                error=(
                    f"halmos timed out after "
                    f"{self._timeout}s"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return SymbolicProofSuite(
                contract=contract,
                status=SymbolicProofStatus.ERROR,
                error=f"halmos invocation failed: {exc}",
            )
        # Defensive — subprocess can return bytes despite
        # text=True under some Python+OS combinations (seen
        # with halmos 0.3.3 on Python 3.14 / macOS).
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return _parse_halmos_output(
            contract, stdout + stderr,
        )


def _parse_halmos_output(
    contract: str, output: str,
) -> SymbolicProofSuite:
    """Parse halmos textual output into structured results.

    Halmos prints per-proof lines like:
      [PASS] check_name() (paths: 5, time: 0.02s, bounds: [])
      [FAIL] check_name() (paths: 3, time: 0.01s, bounds: [])
        Counterexample: ...

    Footer:
      Symbolic test result: 3 passed; 0 failed; time: 0.09s

    ANSI color codes are stripped before parsing.
    """
    import re
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    proofs: List[SymbolicProofResult] = []
    for line in clean.splitlines():
        m = re.match(
            r"^\s*\[(PASS|FAIL|ERROR)\]\s+"
            r"(\w+(?:\([^)]*\))?)\s+"
            r"\(paths:\s*(\d+),\s+"
            r"time:\s*([\d.]+)s",
            line,
        )
        if not m:
            continue
        status_str, name, paths, t = (
            m.group(1), m.group(2), m.group(3), m.group(4),
        )
        status = {
            "PASS": SymbolicProofStatus.PASSED,
            "FAIL": SymbolicProofStatus.FAILED,
            "ERROR": SymbolicProofStatus.ERROR,
        }[status_str]
        proofs.append(SymbolicProofResult(
            name=name,
            status=status,
            paths_explored=int(paths),
            time_seconds=float(t),
        ))
    # Aggregate status: PASSED iff all proofs passed and
    # we found at least one.
    if not proofs:
        return SymbolicProofSuite(
            contract=contract,
            status=SymbolicProofStatus.ERROR,
            error=(
                "halmos produced no parseable proof results"
                f"; output head: {clean[:500]!r}"
            ),
        )
    if any(
        p.status != SymbolicProofStatus.PASSED for p in proofs
    ):
        agg = SymbolicProofStatus.FAILED
    else:
        agg = SymbolicProofStatus.PASSED
    # Extract halmos version if present.
    halmos_version = None
    for line in clean.splitlines():
        m = re.match(r"^\s*halmos\s+(\S+)", line)
        if m:
            halmos_version = m.group(1)
            break
    return SymbolicProofSuite(
        contract=contract,
        status=agg,
        proofs=proofs,
        halmos_version=halmos_version,
    )


# Catalog of known symbolic-proof contracts that mirror the
# runtime-invariant registry. Operators / CI consume this
# to know which proofs to run when extending the harness.
SYMBOLIC_PROOF_CATALOG: Dict[str, Dict[str, Any]] = {
    "FTNSSupplyCapSpec": {
        "mirrors_runtime_contract": "ftns_token",
        "runtime_invariants": ["INV-FT-1", "INV-FT-2"],
        "description": (
            "Symbolic proof that no sequence of mintReward "
            "calls can break totalSupply() <= MAX_SUPPLY. "
            "Source-identity-mirrors FTNSTokenSimple."
            "mintReward at contracts/contracts/"
            "FTNSTokenSimple.sol:70-73."
        ),
    },
    "RoyaltyDistributorSolvencySpec": {
        "mirrors_runtime_contract": "royalty_distributor",
        "runtime_invariants": ["INV-RD-1", "INV-RD-4"],
        "description": (
            "Symbolic proof that distributeRoyalty + claim "
            "+ recoverStranded all preserve the solvency "
            "invariant balance >= totalClaimable. The "
            "canonical 'this is what halmos is for' proof "
            "— the runtime probe checks live state; this "
            "proves no symbolic input can ever break the "
            "invariant. Source-identity-mirrors "
            "RoyaltyDistributor.sol:111-193."
        ),
    },
    "AdminBoundedSettersSpec": {
        "mirrors_runtime_contract": "multiple",
        "runtime_invariants": [
            "INV-SS-1", "INV-SS-2",
            "INV-SB-1", "INV-SB-2", "INV-SB-3",
            "INV-CD-1",
            "INV-EC-1", "INV-EC-2",
        ],
        "description": (
            "Symbolic proof that admin-only bounded "
            "setters can NEVER produce out-of-range state "
            "for ANY symbolic input. Covers: "
            "StorageSlashing.setHeartbeatGrace bounds; "
            "StakeBond.setUnbondDelay bounds; "
            "StakeBond.CHALLENGER_BOUNTY_BPS constant; "
            "CompensationDistributor.updateWeights anti-"
            "rugpull 90-day delay; EmissionController "
            "mainnet-chainid halving rejection of off-"
            "cadence epoch. Proves the bounds ARE the "
            "runtime behavior, not just declarative."
        ),
    },
    "EscrowPoolSolvencySpec": {
        "mirrors_runtime_contract": "escrow_pool",
        "runtime_invariants": ["INV-EP-1"],
        "description": (
            "Symbolic proof that deposit + withdraw + "
            "settleFromRequester all preserve the "
            "solvency invariant balance >= "
            "totalEscrowedBalance. Sister proof to "
            "RoyaltyDistributorSolvencySpec — same "
            "algorithmic shape but on the Phase 3.1 "
            "per-requester escrow accumulator."
        ),
    },
    "ThreeBandRoutingSpec": {
        "mirrors_runtime_contract": "content_dedup",
        "runtime_invariants": [],
        "description": (
            "Symbolic proof of the PRSM-PROV-1 Item 6 three-"
            "band similarity-score routing invariants (audit-"
            "prep §7.19). Construction enforces band ordering "
            "(arbitration_floor <= derivative <= duplicate); "
            "routing is monotonic in score; each band-boundary "
            "is correctly inclusive on its lower bound. "
            "Source-identity-mirrors prsm/data/dedup/"
            "thresholds.py:55-99 (EffectiveThresholds + "
            "__post_init__) + the routing call-sites."
        ),
    },
    "StreamingEmitCapSpec": {
        "mirrors_runtime_contract": "streaming_inference",
        "runtime_invariants": [],
        "description": (
            "Symbolic proof of the Phase 3.x.8.1 SSE "
            "streaming-endpoint settle-on-tokens-emitted "
            "billing invariant (audit-prep §7.5). For ALL "
            "per-iteration accept counts + budget pre-"
            "states, tokens_emitted NEVER exceeds "
            "max_tokens. cap_hit_mid_emit fires iff actual "
            "truncation; cap_reached fires iff post-state "
            "hits or exceeds max. Source-identity-mirrors "
            "prsm/compute/chain_rpc/client.py:1359-1367 "
            "(round-1 MEDIUM-2 split remediation)."
        ),
    },
    "KVCacheLRUBoundSpec": {
        "mirrors_runtime_contract": "streaming_inference",
        "runtime_invariants": [],
        "description": (
            "Symbolic proof of the KVCacheManager LRU bound "
            "invariant (audit-prep §7.9 Phase 3.x.11). For "
            "ALL allocation sequences, cache size NEVER "
            "exceeds max_cached_requests — closes unbounded-"
            "cache-growth under concurrent request load. "
            "Source-identity-mirrors prsm/compute/chain_rpc/"
            "kv_cache.py:201-249 KVCacheManager.allocate "
            "pre-insert eviction loop."
        ),
    },
    "M1CadenceDrivenYieldSpec": {
        "mirrors_runtime_contract": "streaming_inference",
        "runtime_invariants": [],
        "description": (
            "Symbolic proof of the Phase 3.x.11.q M1 "
            "cadence-driven yield invariant (audit-prep "
            "§7.13). FixedRateShardedExecutor's pre-yield "
            "sleep gate guarantees inter-emission gap >= "
            "cadence regardless of inner-executor compute "
            "speed — closes the per-token timing leak where "
            "fast compute would reveal short-content tokens. "
            "Source-identity-mirrors prsm/compute/chain_rpc/"
            "tier_c_sharded_executors.py:328-351."
        ),
    },
    "EncryptedProbsCoSetSpec": {
        "mirrors_runtime_contract": "streaming_inference",
        "runtime_invariants": [],
        "description": (
            "Symbolic proof of Phase 3.x.11.q.y encrypted_"
            "proposed_token_probs co-set validators (audit-"
            "prep §7.14). Four invariants: mutual exclusion "
            "with plaintext probs, co-set with proposed_ids, "
            "non-empty when set, bytes length in [1, 1024]. "
            "Source-identity-mirrors prsm/compute/chain_rpc/"
            "protocol.py:956-998."
        ),
    },
    "M2ResponseSizePaddingSpec": {
        "mirrors_runtime_contract": "streaming_inference",
        "runtime_invariants": [],
        "description": (
            "Symbolic proof of the Phase 3.x.11.q.x M2 "
            "response-size padding invariant (audit-prep "
            "§7.15). For ALL input text byte lengths + ALL "
            "codepoint-boundary dropouts (0-3 bytes), the "
            "output is EXACTLY pad_to_bytes — closes the "
            "M2 response-size leak where un-padded responses "
            "leaked output-content length on the wire. Plus "
            "finish_reason invariants: preserve on under/"
            "exact-match branches, override to length_capped "
            "on truncate branch. Source-identity-mirrors "
            "prsm/compute/chain_rpc/tier_c_sharded_executors"
            ".py:_pad_or_truncate_utf8 (208-253)."
        ),
    },
    "ChunkStreamingBoundsSpec": {
        "mirrors_runtime_contract": "streaming_inference",
        "runtime_invariants": [],
        "description": (
            "Symbolic proof of the H1 round-1 remediation "
            "bounded-iterator invariant from the chunked-"
            "activation-streaming subsystem (audit-prep §7.3, "
            "Phase 3.x.7.1). For ALL peer-shipped frame "
            "counts, accepted chunk count is bounded above "
            "by expected_total_chunks — closes the unbounded-"
            "memory-growth threat where a malicious peer "
            "ships excess chunks past the manifest. Also "
            "covers the per-chunk request_id binding (relay-"
            "defense invariant) at server.py:1983. Source-"
            "identity-mirrors "
            "prsm/compute/chain_rpc/server.py:1960-1994."
        ),
    },
    "SpeculationRollbackMathSpec": {
        "mirrors_runtime_contract": "streaming_inference",
        "runtime_invariants": [],
        "description": (
            "Symbolic proof of two pure-arithmetic "
            "invariants from the speculative-decoding "
            "streaming-inference subsystem (audit-prep "
            "§7.11+§7.12). Covers: (a) the Phase 3.x.11.y.x "
            "critical rollback-math fix (cached_extra = "
            "(k_round + 1) - len(emitted); pre-fix formula "
            "(accepted_count + 1) - len(emitted) under-"
            "counted in cap_hit_mid_emit case); (b) adaptive-"
            "K state machine bounded transitions (halve <25%, "
            "double >75%, hold mid; floor 1, ceiling k_max). "
            "Source-identity-mirrors "
            "prsm/compute/chain_rpc/client.py:1431 + 1463-"
            "1466. No runtime invariant counterpart — "
            "streaming-inference subsystem isn't on-chain, "
            "so the runtime probe doesn't apply; symbolic "
            "verification IS the canonical layer."
        ),
    },
    "RoleDisarmAccessControlSpec": {
        "mirrors_runtime_contract": "ftns_token",
        "runtime_invariants": [
            "INV-FT-3", "INV-FT-4", "INV-FT-5",
        ],
        "description": (
            "Symbolic proof that OZ AccessControl is "
            "uncircumventable: role membership can only "
            "change via admin-gated grantRole / "
            "revokeRole. The PRSM-specific INV-FT-3/4/5 "
            "claims (Foundation Safe holds admin; "
            "disarmed hot key holds no role) are "
            "STATE-DEPENDENT — verified by the runtime "
            "probe against live mainnet state. This proof "
            "covers the STRUCTURAL guarantee that makes "
            "the disarm sticky: no symbolic caller can "
            "grant a role without first holding the "
            "admin role. Halmos caught a real proof bug "
            "during development (post-state vs pre-state "
            "assertion on self-revoke) — documented "
            "inline."
        ),
    },
    "CreatorStakeRegistrySpec": {
        "mirrors_runtime_contract": "creator_stake_registry",
        "runtime_invariants": [],
        "description": (
            "Symbolic proofs for the §14 anti-spam "
            "CreatorStakeRegistry (sp976; pre-deploy-"
            "audited + hardened sp979). Three families, "
            "all proven for ALL symbolic inputs across "
            "stake/requestUnbond/withdraw/slash/drain: "
            "(1) SOLVENCY — ftns.balanceOf(this) >= "
            "totalCustodied + foundationReserveBalance "
            "(sister to INV-EP-1 / INV-RD-4); (2) SLASH "
            "CONSERVATION — slash moves exactly `amount` "
            "from bonded stake to the reserve, never "
            "below zero; (3) ANTI-GAME ELIGIBILITY — "
            "creatorStakeOf drops to 0 the instant a "
            "creator begins to unbond, closing the "
            "stake->get-tier->spam->unstake game. "
            "Source-identity-mirrors CreatorStakeRegistry"
            ".sol (line ranges pinned). Runtime state-"
            "pins (owner == Foundation Safe; unbond-delay "
            "bounds) are added to the runtime registry at "
            "commission time, once the contract has an "
            "on-chain address."
        ),
    },
}
