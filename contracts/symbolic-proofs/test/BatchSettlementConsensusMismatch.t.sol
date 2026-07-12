// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @title BatchSettlementRegistry consensus-mismatch false-slash proofs (sp986)
/// @notice Halmos-provable guarantee that the live-on-mainnet settlement
///         registry's CONSENSUS_MISMATCH challenge path can NEVER slash unless
///         a genuine, cross-provider, same-consensus-group disagreement is
///         proven. This is the machine-proof counterpart to the runtime pins
///         (sp985 INV-BSR-*) and the algorithmic backing for the protocol's
///         standing invariant: "a single node must NEVER falsely slash an
///         honest provider."
///
/// @dev The challenge returns true (→ stakeBond.slash fires) ONLY when EVERY
///      guard holds. These proofs show that each false-slash precondition,
///      taken alone, forces the validator to reject (return false) for ALL
///      symbolic inputs:
///        - same provider on both batches        (self-/honest-provider slash)
///        - mismatched consensus_group_id          (cross-group griefing)
///        - minority batch not opted into consensus (group_id == 0)
///        - no actual output disagreement          (no-op slash)
///        - self-referential challenge             (batch challenges itself)
///      A positive proof confirms the validator is not vacuously false: when
///      all guards hold + the majority leaf is proven, it does accept.
///
/// @dev STRUCTURAL EQUIVALENCE (audit-visible; line range pinned):
///     _handleConsensusMismatch → contracts/contracts/BatchSettlementRegistry.sol:997-1043
///   Simplifications (orthogonal to the false-slash guards):
///     - The Merkle proof of the majority leaf is substituted with a boolean
///       `majorityProofValid` (proving membership is MerkleProof.verify's job,
///       not this contract's; an invalid proof returns false either way).
///     - The "conflicting batch NONEXISTENT" case, which the real contract
///       REVERTS on, is modeled as returning false — strictly equivalent for
///       the false-slash property (a revert also fires no slash).
///     - Batch storage / status enum / abi.decode plumbing omitted; the decoded
///       values the guards compare are passed as a single struct (which also
///       keeps the EVM stack shallow — the guard chain has many operands).

struct MismatchInput {
    bytes32 batchId;
    bytes32 conflictingBatchId;
    bytes32 bGroupId;
    bytes32 otherGroupId;
    address bProvider;
    address otherProvider;
    bytes32 leafJobIdHash;
    bytes32 majJobIdHash;
    uint256 leafShardIndex;
    uint256 majShardIndex;
    bytes32 leafOutputHash;
    bytes32 majOutputHash;
    bool otherCommitted;
    bool majorityProofValid;
}

contract ConsensusMismatchValidator {
    /// Faithful model of _handleConsensusMismatch's guard chain. Returns true
    /// iff the challenge is valid (and therefore a slash may fire).
    function isValidMismatch(MismatchInput memory i)
        public pure returns (bool)
    {
        // L912: distinct batches — no self-referential challenge.
        if (i.conflictingBatchId == i.batchId) return false;
        // L917: minority batch must have opted into consensus.
        if (i.bGroupId == bytes32(0)) return false;
        // L920-921: same shard of same job.
        if (i.leafJobIdHash != i.majJobIdHash) return false;
        if (i.leafShardIndex != i.majShardIndex) return false;
        // L924: genuine disagreement on output bytes.
        if (i.leafOutputHash == i.majOutputHash) return false;
        // L928-930: conflicting batch must be committed (real contract REVERTS;
        // modeled as reject — equivalent for the no-slash property).
        if (!i.otherCommitted) return false;
        // L934-935: same non-zero group AND different providers.
        if (i.bGroupId != i.otherGroupId) return false;
        if (i.bProvider == i.otherProvider) return false;
        // L938-941: majority leaf provable in the conflicting batch.
        if (!i.majorityProofValid) return false;
        return true;
    }
}


contract BatchSettlementConsensusMismatchSpec {
    ConsensusMismatchValidator internal v;

    function setUp() public {
        v = new ConsensusMismatchValidator();
    }

    /// THE headline false-slash proof: a challenge can NEVER be valid when both
    /// batches are from the SAME provider — an honest provider cannot be slashed
    /// for "disagreeing with itself", and no single node can self-slash a peer
    /// it impersonates on both sides.
    function check_same_provider_never_valid(MismatchInput memory inp)
        public view
    {
        inp.otherProvider = inp.bProvider; // force same provider on both batches
        assert(!v.isValidMismatch(inp));
    }

    /// A challenge across two DIFFERENT consensus groups is never valid — a
    /// provider in group A cannot be slashed by a batch from group B.
    function check_cross_group_never_valid(MismatchInput memory inp)
        public view
    {
        if (inp.bGroupId == inp.otherGroupId) return; // constrain to cross-group
        assert(!v.isValidMismatch(inp));
    }

    /// A minority batch that never opted into consensus (group_id == 0) can
    /// never be the target of a CONSENSUS_MISMATCH slash.
    function check_zero_group_never_valid(MismatchInput memory inp)
        public view
    {
        inp.bGroupId = bytes32(0); // minority did not opt into consensus
        assert(!v.isValidMismatch(inp));
    }

    /// No disagreement, no slash: identical output hashes can never be valid.
    function check_no_disagreement_never_valid(MismatchInput memory inp)
        public view
    {
        inp.majOutputHash = inp.leafOutputHash; // equal outputs → no mismatch
        assert(!v.isValidMismatch(inp));
    }

    /// A batch cannot challenge itself.
    function check_self_referential_never_valid(MismatchInput memory inp)
        public view
    {
        inp.conflictingBatchId = inp.batchId; // challenge targets its own batch
        assert(!v.isValidMismatch(inp));
    }

    /// Sanity / non-vacuity: when EVERY guard holds (distinct committed batches,
    /// non-zero matching group, different providers, same job+shard, different
    /// outputs, valid majority proof) the validator DOES accept. Proves the
    /// rejections above are real guards, not a function that is always false.
    function check_all_guards_satisfied_accepts(MismatchInput memory inp)
        public view
    {
        // Force the all-guards-pass region.
        if (inp.conflictingBatchId == inp.batchId) return;
        if (inp.bGroupId == bytes32(0)) return;
        if (inp.bProvider == inp.otherProvider) return;
        inp.otherGroupId = inp.bGroupId;          // matching non-zero group
        inp.majJobIdHash = inp.leafJobIdHash;     // same job
        inp.majShardIndex = inp.leafShardIndex;   // same shard
        // distinct outputs:
        if (inp.leafOutputHash == inp.majOutputHash) return;
        inp.otherCommitted = true;
        inp.majorityProofValid = true;
        assert(v.isValidMismatch(inp));
    }
}
