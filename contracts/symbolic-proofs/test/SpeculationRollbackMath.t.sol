// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @title Speculation rollback + adaptive-K symbolic proof (sprint 367)
/// @notice Halmos-provable mirror of two pure-arithmetic invariants
///         from the speculative-decoding streaming-inference subsystem.
///         Both are algorithmic — they don't touch on-chain state but
///         govern the per-iteration KV-cache rollback distance and
///         the K state-machine bounds that protect against runaway
///         speculation depth.
///
///         Audit-prep cross-reference: §7.11 (Phase 3.x.11.y
///         speculative decoding) + §7.12 (Phase 3.x.11.y.x sampling-
///         correct speculation under T>0). The Phase 3.x.11.y.x
///         critical fix was specifically to the rollback math.
///
/// @dev STRUCTURAL EQUIVALENCE (audit-visible):
///   rollback math      → prsm/compute/chain_rpc/client.py:1448
///                        cached_extra = (k_round + 1) - len(emitted)
///   adaptive-K bounds  → prsm/compute/chain_rpc/client.py:1480-1483
///                        if rate < 0.25: k = max(1, k // 2)
///                        elif rate > 0.75: k = min(k_max, k * 2)
///
/// @dev The pre-fix BUG was `cached_extra = len(verified) - len(emitted)`.
///      In v2, len(verified) == accepted_count + 1 (NOT k_round + 1)
///      because v2's `verified_token_ids` is tail-shape-narrowed to
///      accepted_count + 1 (sprint 0.x.x.y.x §7.12 invariant). The
///      pre-fix formula UNDER-counts rollback in the cap_hit_mid_emit
///      case where len(emitted) < accepted_count + 1.
///
/// Run:
///   cd contracts/symbolic-proofs && \
///   halmos --contract SpeculationRollbackMathSpec

contract SpeculationRollbackMath {
    uint256 public constant MAX_VERIFY_BATCH_TOKENS = 8;

    /// Mirrors client.py:1448 — the POST-FIX correct formula.
    /// For any k_round (this round's K) and len_emitted (number of
    /// tokens we actually committed + showed to the user), the
    /// rollback distance is exactly (k_round + 1) - len_emitted.
    function cachedExtra(
        uint256 k_round,
        uint256 len_emitted
    ) public pure returns (uint256) {
        // INVARIANT (caller-asserted): len_emitted <= k_round + 1.
        // The cap_hit_mid_emit truncation reduces len_emitted from
        // accepted_count+1; both are bounded above by k_round + 1.
        return (k_round + 1) - len_emitted;
    }

    /// Mirrors client.py:1480-1483 — adaptive K state transition
    /// based on rolling-window accept rate. Rate input is in bps
    /// (0-10000) instead of float to keep the math integer-clean.
    function adaptiveK(
        uint256 k,
        uint256 k_max,
        uint256 rateBps
    ) public pure returns (uint256) {
        // Floor at 1, cap at k_max.
        if (rateBps < 2500) {
            // Halve, floor at 1
            uint256 halved = k / 2;
            return halved < 1 ? 1 : halved;
        }
        if (rateBps > 7500) {
            // Double, cap at k_max
            uint256 doubled = k * 2;
            return doubled > k_max ? k_max : doubled;
        }
        return k;  // Hold in [25%, 75%]
    }
}


contract SpeculationRollbackMathSpec {
    SpeculationRollbackMath internal m;

    function setUp() public {
        m = new SpeculationRollbackMath();
    }

    // ── Rollback math ───────────────────────────────────

    /// THE post-fix invariant: cached_extra is in [0, k_round + 1]
    /// for all reachable (k_round, len_emitted) pairs. Halmos
    /// explores symbolic inputs constrained by the source-side
    /// precondition len_emitted <= k_round + 1.
    function check_rollback_math_post_fix_bounded(
        uint256 k_round,
        uint256 len_emitted
    ) public {
        // Halmos cheatcode preconditions: bound to realistic
        // ranges to avoid solver blow-up on uint256-wide.
        vm_assume(k_round <= m.MAX_VERIFY_BATCH_TOKENS());
        vm_assume(len_emitted <= k_round + 1);

        uint256 extra = m.cachedExtra(k_round, len_emitted);

        // Invariant 1: rollback distance is non-negative.
        // (Trivial under uint256 — assert wouldn't catch
        // underflow because Solidity 0.8+ reverts on underflow,
        // and the precondition prevents underflow input.)
        assert(extra <= k_round + 1);

        // Invariant 2: post-rollback cache position recovery.
        // pre_cache = X (symbolic); after VERIFY, cache extended
        // by k_round + 1 to X + k_round + 1. After rollback of
        // `extra`, cache is at X + k_round + 1 - extra =
        // X + len_emitted. Which is the correct "keep emitted
        // prefix in cache" semantic.
        uint256 cached_after_verify = k_round + 1;
        uint256 cached_after_rollback = cached_after_verify - extra;
        assert(cached_after_rollback == len_emitted);
    }

    /// Boundary: at len_emitted == k_round + 1 (full accept,
    /// no cap_hit), rollback is exactly 0 (nothing to drop).
    function check_rollback_zero_on_full_accept(
        uint256 k_round
    ) public {
        vm_assume(k_round <= m.MAX_VERIFY_BATCH_TOKENS());
        uint256 extra = m.cachedExtra(k_round, k_round + 1);
        assert(extra == 0);
    }

    /// Boundary: at len_emitted == 0 (reject everything,
    /// extreme cap_hit), rollback is exactly k_round + 1
    /// (drop the full verify forward).
    function check_rollback_max_on_zero_accept(
        uint256 k_round
    ) public {
        vm_assume(k_round <= m.MAX_VERIFY_BATCH_TOKENS());
        uint256 extra = m.cachedExtra(k_round, 0);
        assert(extra == k_round + 1);
    }

    /// Demonstrates the pre-fix BUG. The bad formula was
    /// `cached_extra = (accepted_count + 1) - len_emitted`.
    /// In the cap_hit_mid_emit case where len_emitted is
    /// SHORTER than accepted_count + 1 (because the
    /// max_tokens cap truncated the emit), pre-fix would
    /// drop too few positions, leaving stale cache entries.
    ///
    /// This is a NEGATIVE proof — it CONSTRUCTS a counter-
    /// example demonstrating the pre-fix formula is wrong,
    /// motivating why the post-fix formula matters.
    function check_pre_fix_formula_undercounts(
    ) public {
        // Scenario: k_round = 4, accepted_count = 3, but
        // cap_hit_mid_emit truncates len_emitted to 1.
        uint256 k_round = 4;
        uint256 accepted_count = 3;
        uint256 len_emitted = 1;
        // accepted_count + 1 - len_emitted = 3
        // post-fix: k_round + 1 - len_emitted = 4
        uint256 post_fix = m.cachedExtra(
            k_round, len_emitted
        );
        uint256 pre_fix_buggy = (
            accepted_count + 1
        ) - len_emitted;
        // The pre-fix formula UNDER-counts by exactly
        // (k_round - accepted_count) = 1 in this scenario.
        assert(post_fix > pre_fix_buggy);
        assert(post_fix - pre_fix_buggy == 1);
    }

    // ── Adaptive K ──────────────────────────────────────

    /// THE adaptive-K bounded-state invariant: for ALL
    /// (k, rate) inputs, post-transition k is in [1, k_max].
    /// Halmos explores symbolic rate bps + symbolic
    /// pre-state k.
    function check_adaptive_k_stays_in_range(
        uint256 k,
        uint256 rateBps
    ) public {
        uint256 k_max = m.MAX_VERIFY_BATCH_TOKENS() - 1;
        vm_assume(k >= 1);
        vm_assume(k <= k_max);
        vm_assume(rateBps <= 10000);

        uint256 post = m.adaptiveK(k, k_max, rateBps);

        assert(post >= 1);
        assert(post <= k_max);
    }

    /// Low-rate halving: rate < 25% → k goes down (or stays
    /// at floor 1).
    function check_low_rate_halves(
        uint256 k,
        uint256 rateBps
    ) public {
        uint256 k_max = m.MAX_VERIFY_BATCH_TOKENS() - 1;
        vm_assume(k >= 2);  // k=1 already at floor, no change
        vm_assume(k <= k_max);
        vm_assume(rateBps < 2500);
        uint256 post = m.adaptiveK(k, k_max, rateBps);
        assert(post < k);  // Strictly decreased
        assert(post >= 1);
    }

    /// High-rate doubling: rate > 75% → k goes up (or stays
    /// at ceiling k_max).
    function check_high_rate_doubles(
        uint256 k,
        uint256 rateBps
    ) public {
        uint256 k_max = m.MAX_VERIFY_BATCH_TOKENS() - 1;
        vm_assume(k >= 1);
        vm_assume(k < k_max);  // Below ceiling → strict increase
        vm_assume(rateBps > 7500);
        uint256 post = m.adaptiveK(k, k_max, rateBps);
        assert(post > k);  // Strictly increased
        assert(post <= k_max);
    }

    /// Mid-rate hold: 25% <= rate <= 75% → k unchanged.
    function check_mid_rate_holds(
        uint256 k,
        uint256 rateBps
    ) public {
        uint256 k_max = m.MAX_VERIFY_BATCH_TOKENS() - 1;
        vm_assume(k >= 1);
        vm_assume(k <= k_max);
        vm_assume(rateBps >= 2500);
        vm_assume(rateBps <= 7500);
        uint256 post = m.adaptiveK(k, k_max, rateBps);
        assert(post == k);
    }

    /// Halmos cheatcode for symbolic pre-conditions.
    function vm_assume(bool cond) internal {
        // selector: assume(bool) = 0x4c63e562
        (bool ok,) = address(
            uint160(uint256(keccak256("hevm cheat code")))
        ).call(
            abi.encodeWithSelector(bytes4(0x4c63e562), cond)
        );
        ok;
    }
}
