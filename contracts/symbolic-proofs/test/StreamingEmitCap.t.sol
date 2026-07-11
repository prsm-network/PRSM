// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @title Streaming-emit cap invariant (sprint 372)
/// @notice Halmos-provable mirror of the Phase 3.x.8.1 SSE
///         streaming-endpoint settle-on-tokens-emitted
///         billing invariant from audit-prep §7.5. For ALL
///         per-iteration accept counts, the cumulative
///         tokens_emitted NEVER exceeds max_tokens — closes
///         a griefing vector where unbounded emission could
///         charge the requester past their budget.
///
/// @dev STRUCTURAL EQUIVALENCE (audit-visible):
///   prsm/compute/chain_rpc/client.py:1519-1527 — the
///   cap-bound + truncation block:
///
///     emitted = list(verified[: accepted_count + 1])
///     cap_hit_mid_emit = False
///     if tokens_emitted + len(emitted) > max_tokens:
///         emitted = emitted[: max_tokens - tokens_emitted]
///         cap_hit_mid_emit = True
///     cap_reached = (
///         tokens_emitted + len(emitted) >= max_tokens
///     )
///
///   The split of cap_hit_mid_emit (>, truncates) from
///   cap_reached (>=, terminates the loop) was the round-1
///   MEDIUM-2 remediation per §7.5. This proof verifies
///   both invariants symbolically.
///
/// Run:
///   cd contracts/symbolic-proofs && \
///   halmos --contract StreamingEmitCapSpec

contract StreamingEmitCap {
    /// Mirrors the cap-bound + truncation logic. Returns
    /// (emit_len, cap_hit_mid_emit, cap_reached).
    function applyCap(
        uint256 tokens_emitted_pre,
        uint256 max_tokens,
        uint256 accepted_count
    ) external pure returns (
        uint256 emit_len,
        bool cap_hit_mid_emit,
        bool cap_reached
    ) {
        // verified[: accepted_count + 1] semantically gives
        // accepted_count + 1 tokens.
        emit_len = accepted_count + 1;
        cap_hit_mid_emit = false;
        if (tokens_emitted_pre + emit_len > max_tokens) {
            // Truncate at the cap.
            emit_len = max_tokens - tokens_emitted_pre;
            cap_hit_mid_emit = true;
        }
        cap_reached = (
            tokens_emitted_pre + emit_len >= max_tokens
        );
    }
}


contract StreamingEmitCapSpec {
    StreamingEmitCap internal m;

    function setUp() public {
        m = new StreamingEmitCap();
    }

    /// THE settle-on-emit billing invariant: post-state
    /// tokens_emitted NEVER exceeds max_tokens for ANY
    /// (tokens_emitted_pre, max_tokens, accepted_count)
    /// input that satisfies the canonical precondition
    /// tokens_emitted_pre < max_tokens (loop-entry gate).
    function check_emitted_never_exceeds_max_tokens(
        uint256 tokens_emitted_pre,
        uint256 max_tokens,
        uint256 accepted_count
    ) public {
        vm_assume(max_tokens >= 1);
        vm_assume(max_tokens <= 128);
        vm_assume(tokens_emitted_pre < max_tokens);
        vm_assume(accepted_count <= 32);

        (uint256 emit_len, , ) = m.applyCap(
            tokens_emitted_pre, max_tokens, accepted_count
        );
        // Post-state tokens_emitted = pre + emit_len
        uint256 post = tokens_emitted_pre + emit_len;
        assert(post <= max_tokens);
    }

    /// The cap_hit_mid_emit flag fires IFF actual
    /// truncation happened (round-1 MEDIUM-2 split).
    function check_cap_hit_flag_correctness(
        uint256 tokens_emitted_pre,
        uint256 max_tokens,
        uint256 accepted_count
    ) public {
        vm_assume(max_tokens >= 1);
        vm_assume(max_tokens <= 128);
        vm_assume(tokens_emitted_pre < max_tokens);
        vm_assume(accepted_count <= 32);

        (
            uint256 emit_len,
            bool cap_hit,
        ) = m.applyCap(
            tokens_emitted_pre, max_tokens, accepted_count
        );
        // Pre-truncation length would have been
        // accepted_count + 1. If that's > the budget,
        // cap_hit MUST be true.
        uint256 natural_len = accepted_count + 1;
        if (
            tokens_emitted_pre + natural_len > max_tokens
        ) {
            assert(cap_hit);
            assert(emit_len < natural_len);
        } else {
            assert(!cap_hit);
            assert(emit_len == natural_len);
        }
    }

    /// cap_reached fires IFF the post-state hits or
    /// exceeds max_tokens. Critical for terminal-logic
    /// downstream — the loop MUST stop when cap_reached.
    function check_cap_reached_correctness(
        uint256 tokens_emitted_pre,
        uint256 max_tokens,
        uint256 accepted_count
    ) public {
        vm_assume(max_tokens >= 1);
        vm_assume(max_tokens <= 128);
        vm_assume(tokens_emitted_pre < max_tokens);
        vm_assume(accepted_count <= 32);

        (
            uint256 emit_len,
            ,
            bool cap_reached
        ) = m.applyCap(
            tokens_emitted_pre, max_tokens, accepted_count
        );
        uint256 post = tokens_emitted_pre + emit_len;
        if (post >= max_tokens) {
            assert(cap_reached);
        } else {
            assert(!cap_reached);
        }
    }

    /// Boundary: exact-cap-hit (post == max_tokens) DOES
    /// set cap_reached but doesn't necessarily set cap_hit
    /// (no truncation if natural fits exactly). Sprint
    /// 0.x.x.X round-1 MEDIUM-2 documented this case
    /// explicitly.
    function check_exact_cap_natural_fit(
        uint256 tokens_emitted_pre,
        uint256 max_tokens
    ) public {
        vm_assume(max_tokens >= 1);
        vm_assume(max_tokens <= 128);
        vm_assume(tokens_emitted_pre < max_tokens);
        // accepted_count chosen so natural_len exactly
        // fills the remaining budget.
        uint256 remaining = max_tokens - tokens_emitted_pre;
        if (remaining == 0) return;  // No budget left
        uint256 accepted_count = remaining - 1;
        (
            uint256 emit_len,
            bool cap_hit,
            bool cap_reached
        ) = m.applyCap(
            tokens_emitted_pre, max_tokens, accepted_count
        );
        assert(emit_len == remaining);
        // Natural fit — no truncation needed
        assert(!cap_hit);
        // But cap reached
        assert(cap_reached);
    }

    /// Boundary: zero accepted_count → emit_len = 1
    /// (the lookahead from target) UNLESS that overflows.
    function check_zero_accepted_emits_one(
        uint256 tokens_emitted_pre,
        uint256 max_tokens
    ) public {
        vm_assume(max_tokens >= 2);
        vm_assume(max_tokens <= 128);
        vm_assume(tokens_emitted_pre < max_tokens - 1);
        (uint256 emit_len, , ) = m.applyCap(
            tokens_emitted_pre, max_tokens, 0
        );
        assert(emit_len == 1);
    }

    function vm_assume(bool cond) internal {
        (bool ok,) = address(
            uint160(uint256(keccak256("hevm cheat code")))
        ).call(
            abi.encodeWithSelector(bytes4(0x4c63e562), cond)
        );
        ok;
    }
}
