// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @title encrypted_proposed_token_probs co-set validator (sprint 371)
/// @notice Halmos-provable mirror of the Phase 3.x.11.q.y
///         co-set validators from audit-prep §7.14. The
///         encrypted-probs wire field has four invariants:
///           (1) mutual exclusion with plaintext probs
///           (2) co-set with proposed_token_ids (both or
///               neither at the encrypted-probs site)
///           (3) bytes length in [1, 1024]
///           (4) non-empty when set
///
/// @dev STRUCTURAL EQUIVALENCE (audit-visible):
///   prsm/compute/chain_rpc/protocol.py:1069-1111 —
///     RunLayerSliceRequest.validate() encrypted-probs branch.
///   The Solidity mirror models the four boolean / length
///   checks as a single validate() function that reverts on
///   any violation, returns true on success.
///
/// Run:
///   cd contracts/symbolic-proofs && \
///   halmos --contract EncryptedProbsCoSetSpec

contract EncryptedProbsCoSet {
    uint256 public constant MAX_ENC_BYTES = 1024;

    /// Mirrors the four invariants. Halmos models the
    /// presence of each Optional field as a bool + length
    /// for the ciphertext bytes.
    function validate(
        bool has_plaintext_probs,
        bool has_encrypted_probs,
        bool has_proposed_ids,
        uint256 enc_byte_length
    ) external pure returns (bool) {
        if (has_encrypted_probs) {
            // Invariant (1): mutual exclusion
            if (has_plaintext_probs) {
                revert("mutually exclusive");
            }
            // Invariant (2): co-set with proposed_ids
            if (!has_proposed_ids) {
                revert("must be co-set with proposed_ids");
            }
            // Invariant (4): non-empty
            if (enc_byte_length == 0) {
                revert("must be non-empty");
            }
            // Invariant (3): bytes length cap
            if (enc_byte_length > MAX_ENC_BYTES) {
                revert("exceeds 1024-byte cap");
            }
        }
        return true;
    }
}


contract EncryptedProbsCoSetSpec {
    EncryptedProbsCoSet internal v;

    function setUp() public {
        v = new EncryptedProbsCoSet();
    }

    /// THE mutual-exclusion invariant: for ALL symbolic
    /// (plaintext_set, encrypted_set, ids_set, len) tuples,
    /// validate succeeds ONLY when both probs encodings
    /// aren't simultaneously set.
    function check_mutual_exclusion_holds(
        bool plaintext_set,
        bool encrypted_set,
        bool ids_set,
        uint256 enc_len
    ) public {
        vm_assume(enc_len <= 2048);
        try v.validate(
            plaintext_set, encrypted_set, ids_set, enc_len
        ) returns (bool ok) {
            // Success path: encrypted is unset OR (encrypted
            // is set AND plaintext is unset).
            if (encrypted_set) {
                assert(!plaintext_set);
            }
            ok;
        } catch {
            // Revert path: must violate at least one
            // invariant. The most common: both set.
            assert(true);
        }
    }

    /// Co-set invariant: encrypted + proposed_ids must be
    /// jointly present.
    function check_co_set_with_proposed_ids(
        bool plaintext_set,
        bool encrypted_set,
        bool ids_set,
        uint256 enc_len
    ) public {
        vm_assume(enc_len <= 2048);
        try v.validate(
            plaintext_set, encrypted_set, ids_set, enc_len
        ) returns (bool) {
            // Success: encrypted unset OR ids also set.
            if (encrypted_set) {
                assert(ids_set);
            }
        } catch {
            assert(true);
        }
    }

    /// Length-cap invariant: when encrypted is set,
    /// enc_byte_length is in [1, 1024] post-validate.
    function check_length_in_range_when_encrypted(
        bool plaintext_set,
        bool encrypted_set,
        bool ids_set,
        uint256 enc_len
    ) public {
        vm_assume(enc_len <= 2048);
        try v.validate(
            plaintext_set, encrypted_set, ids_set, enc_len
        ) returns (bool) {
            // Success: if encrypted is set, length is in
            // canonical range
            if (encrypted_set) {
                assert(enc_len >= 1);
                assert(enc_len <= v.MAX_ENC_BYTES());
            }
        } catch {
            assert(true);
        }
    }

    /// Negative: both probs set → must revert.
    function check_both_probs_set_rejected(
        bool ids_set,
        uint256 enc_len
    ) public {
        vm_assume(enc_len >= 1);
        vm_assume(enc_len <= 1024);
        try v.validate(true, true, ids_set, enc_len) {
            assert(false);
        } catch {
            assert(true);
        }
    }

    /// Negative: encrypted set without ids → must revert.
    function check_encrypted_without_ids_rejected(
        uint256 enc_len
    ) public {
        vm_assume(enc_len >= 1);
        vm_assume(enc_len <= 1024);
        try v.validate(false, true, false, enc_len) {
            assert(false);
        } catch {
            assert(true);
        }
    }

    /// Negative: encrypted set with empty bytes → reject.
    function check_encrypted_empty_rejected() public {
        try v.validate(false, true, true, 0) {
            assert(false);
        } catch {
            assert(true);
        }
    }

    /// Negative: encrypted bytes > 1024 → reject.
    function check_encrypted_oversized_rejected(
        uint256 enc_len
    ) public {
        vm_assume(enc_len > 1024);
        vm_assume(enc_len <= 1_000_000);
        try v.validate(false, true, true, enc_len) {
            assert(false);
        } catch {
            assert(true);
        }
    }

    /// Positive: canonical happy path passes.
    function check_canonical_happy_path(
        uint256 enc_len
    ) public {
        vm_assume(enc_len >= 1);
        vm_assume(enc_len <= 1024);
        bool ok = v.validate(false, true, true, enc_len);
        assert(ok);
    }

    /// Positive: no encrypted-probs at all passes (the
    /// validator is opt-in — pre-q.y traffic stays valid).
    function check_no_encrypted_probs_passes(
        bool plaintext_set,
        bool ids_set,
        uint256 enc_len
    ) public {
        vm_assume(enc_len <= 2048);
        bool ok = v.validate(
            plaintext_set, false, ids_set, enc_len
        );
        assert(ok);
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
