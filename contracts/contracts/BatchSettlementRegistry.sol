// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

interface IEscrowPool {
    function settleFromRequester(address requester, address recipient, uint256 amount) external;
}

interface IStakeBond {
    /// @notice Slash `provider`'s stake for misbehavior; route 70% to
    /// `challenger` bounty, 30% to Foundation reserve (or 100%
    /// Foundation when challenger ∈ {provider, 0x0}). Slasher-only.
    function slash(
        address provider,
        address challenger,
        bytes32 reasonId
    ) external;
}

/// @notice Pluggable signature-verification surface used by the
/// INVALID_SIGNATURE challenge path. The production deployment
/// substitutes an audited Ed25519 verifier (per PRSM-PHASE3.1 §10.1
/// resolution). The interface is opaque to the cryptographic scheme
/// so a future migration (e.g., to BLS12-381 or secp256r1) doesn't
/// require contract surgery.
interface ISignatureVerifier {
    /// @return true iff `signature` is a valid signature of `messageHash`
    ///         under `publicKey` according to the implementing scheme.
    /// @dev L2 audit MEDIUM C-INT-02 fix: `pure` (was `view`). Signature
    ///      verification is a deterministic function of inputs only —
    ///      no chain state read should ever be needed. Forcing `pure`
    ///      at the interface level prevents a future implementation
    ///      from sneaking in state-dependent behavior (e.g., reading a
    ///      revocation list mid-verify) that would bypass on-chain
    ///      reasoning about challenge outcomes. If a stateful verifier
    ///      is ever genuinely needed (key rotation, revocation), it
    ///      should live behind a separate `IStatefulSignatureVerifier`
    ///      interface with explicit caller awareness.
    function verify(
        bytes32 messageHash,
        bytes calldata signature,
        bytes calldata publicKey
    ) external pure returns (bool);
}

/// @notice Canonical on-chain representation of a Phase 2 ShardExecutionReceipt,
/// ABI-encoded + keccak256'd to produce Merkle-tree leaves.
///
/// NOTE: canonical-form alignment between this struct and the Python-side
/// off-chain encoding is a Task 5 deliverable. Task 3 ships the struct
/// + on-chain challenge dispatch; Task 5 locks the parity.
struct ReceiptLeaf {
    bytes32 jobIdHash;             // keccak256(utf8(receipt.job_id))
    uint32 shardIndex;             // receipt.shard_index
    bytes32 providerIdHash;        // keccak256(utf8(receipt.provider_id))
    bytes32 providerPubkeyHash;    // keccak256(base64decode(provider_pubkey_b64))
    bytes32 outputHash;            // bytes32 of hex-decoded receipt.output_hash
    uint64 executedAtUnix;         // receipt.executed_at_unix
    uint128 valueFtns;             // per-receipt quoted price in FTNS base units
    bytes32 signatureHash;         // keccak256(base64decode(receipt.signature))
    bytes32 signingMessageHash;    // canonical signing-payload hash the provider committed
                                   // to and signed with their Ed25519 key. Off-chain:
                                   // keccak256("{job_id}||{shard_index}||{output_hash}||{executed_at_unix}")
                                   // per build_receipt_signing_payload(). Bound here so the
                                   // INVALID_SIGNATURE challenge cannot be re-targeted to an
                                   // attacker-chosen message — see L2 audit C-INT-01.
                                   //
                                   // L4 self-audit MED-5 (C-01) + INFO-2 (C-03) caller note:
                                   //   `signingMessageHash` is PROVIDER-supplied and not
                                   //   cross-validated against the other leaf fields
                                   //   on-chain (the off-chain
                                   //   `build_receipt_signing_payload` formula above is
                                   //   convention, not an on-chain invariant). A successful
                                   //   INVALID_SIGNATURE challenge therefore proves only:
                                   //     "the provider's pubkey did NOT sign the 32-byte
                                   //      preimage `signingMessageHash`."
                                   //   It does NOT prove that the receipt CONTENTS
                                   //   (job_id, shard_index, output_hash, executed_at_unix,
                                   //   value_ftns) faithfully reflect anything the provider
                                   //   actually signed. Receipt-content forgery is caught
                                   //   instead by:
                                   //     - NO_ESCROW (requester attests no matching authorization)
                                   //     - CONSENSUS_MISMATCH (k-of-n provider disagreement
                                   //       on a redundant-execution dispatch)
                                   //     - DOUBLE_SPEND (same receipt committed in two batches)
                                   //   Full on-chain binding of message-hash to leaf fields
                                   //   would require storing the variable-length `job_id`
                                   //   string, which is gas-prohibitive at batch scale and
                                   //   would also force an off-chain canonical-form rebuild;
                                   //   the on-chain protocol therefore relies on the three
                                   //   primitives above for content-forgery defense and
                                   //   uses INVALID_SIGNATURE only for the narrow
                                   //   "signature-doesn't-verify under declared pubkey" case.
                                   //   `bytes32` accepts any 32-byte value (INFO-2); the
                                   //   defense surface is not key-revocation-aware here —
                                   //   that is enforced upstream at the publisher-key
                                   //   anchor layer.
}

/**
 * @title BatchSettlementRegistry
 * @notice On-chain anchor for Phase 3.1 batched-settlement receipts.
 *
 * Providers accumulate Phase 2 ShardExecutionReceipts locally, build a
 * Merkle tree over them, and post the root here in a single commitBatch
 * transaction. After a challenge window elapses (default 3 days), the
 * provider calls finalizeBatch, which transitions state + emits the
 * BatchFinalized event that the Phase 3.1 settlement client consumes
 * to execute FTNS transfers.
 *
 * This file is the Phase 3.1 Task 1 deliverable: core batch-state machine
 * only. The actual FTNS transfer execution is wired in by Task 2
 * (EscrowPool.sol integration); the dispute/challenge surface is added in
 * Task 3. Both are additive — Task 1's state transitions remain the
 * authority on batch lifecycle.
 *
 * Key design choices (see docs/2026-04-21-phase3.1-batch-settlement-design.md):
 *   - Merkle-root-only commit. Individual receipt bytes are NOT on chain;
 *     challengers supply them inline (Task 3). Keeps commit gas ~100K.
 *   - Deterministic batchId = keccak256(provider || merkleRoot ||
 *     receiptCount || commitBlock || sequencePerProvider). Allows multiple
 *     simultaneous batches from the same provider without collision.
 *   - Challenge-window is a contract-level state variable (not per-batch)
 *     so Foundation governance can adjust it per PRSM-GOV-1 §4.2 without
 *     upgrading the contract. Default: 3 days.
 *   - Ownable for the governance-adjustable parameter only; batch
 *     operations are permissionless (anyone can commit; anyone can
 *     finalize a pending batch whose window has elapsed).
 */
contract BatchSettlementRegistry is Ownable2Step, Pausable {
    enum BatchStatus {
        NONEXISTENT, // default — batchId never committed
        PENDING,     // committed; within challenge window
        FINALIZED,   // past challenge window + finalizeBatch called
        VOIDED       // reserved for future governance voiding (e.g., if the
                     // whole batch is invalidated en bloc); not reachable in Task 1
    }

    struct Batch {
        address provider;             // who submitted + claims payment
        address requester;            // who owes payment (funds this batch from escrow)
        bytes32 merkleRoot;           // root of the receipt-hash tree
        uint256 receiptCount;         // number of receipts in the tree
        uint256 totalValueFTNS;       // sum of receipt values, pre-challenge
        uint256 invalidatedValueFTNS; // sum of successfully-challenged values
        uint64  commitTimestamp;      // block.timestamp at commit
        BatchStatus status;
        uint16  tier_slash_rate_bps;  // Phase 7: provider's stake tier slash rate at commit;
                                      // 0 means no-stake / no-slash. Snapshot at commit so
                                      // the provider can't dodge slashing via mid-batch downgrade.
        bytes32 consensus_group_id;   // Phase 7.1x §8.7: non-zero binds this batch to a k-of-n
                                      // consensus dispatch. CONSENSUS_MISMATCH challenges require
                                      // both batches to share the same non-zero group_id AND to be
                                      // committed by different providers — closes the sybil-requester
                                      // griefing vector by forcing an attacker to control k distinct
                                      // provider keys (not one provider + one requester) to trigger.
                                      // Zero means "not part of a consensus group" — DOUBLE_SPEND /
                                      // INVALID_SIGNATURE challenges still work unchanged.
        uint64  lookbackWindowSecondsAtCommit;   // L4 self-audit MED-3 (B-02) fix: snapshot of
                                      // `settlementLookbackWindowSeconds` at commit. Mirrors the
                                      // D-05 pattern for the EXPIRED-challenge path: prevents owner
                                      // from retroactively flipping receipt expiry eligibility on
                                      // already-PENDING batches via setSettlementLookbackWindow.
                                      // Pre-fix, owner could shorten the window to expire previously-
                                      // valid receipts (lose payment) or lengthen it to protect stale
                                      // receipts from EXPIRED challenges (overpay). Per-batch snapshot
                                      // closes both directions.
        uint64  totalPausedAtBatchOrigin;        // L4 self-audit HIGH-2 (B-01) fix: snapshot of
                                      // `totalPausedSeconds` at commit. Subtracting this from the
                                      // current `totalPausedSeconds` gives the seconds of paused-time
                                      // that have elapsed SINCE this batch was committed. The
                                      // effective-elapsed calculation subtracts that from the wall-clock
                                      // delta so pause does NOT consume the batch's challenge window.
        uint64  challengeWindowSecondsAtCommit;  // L2 audit MEDIUM D-05 fix: snapshot of
                                      // `challengeWindowSeconds` at commit time. finalizeBatch +
                                      // challengeReceipt + isPastChallengeWindow + secondsUntilFinalizable
                                      // all consult THIS field, NOT the live mutable storage value.
                                      // Pre-fix, owner could shorten the global window mid-flight and
                                      // retroactively allow already-PENDING batches to finalize before
                                      // their original window elapsed — robbing challengers of the time
                                      // they were promised at commit. Snapshotting per-batch closes
                                      // that retroactive-shrink primitive even if the owner is later
                                      // compromised; only the global `challengeWindowSeconds`
                                      // governance setter remains for FUTURE batches.
        // L2 audit MEDIUM D-03 fix: snapshot the 3 cross-wire pointers
        // (escrowPool, stakeBond, signatureVerifier) at commit time.
        // Pre-fix, mid-flight setEscrowPool / setStakeBond /
        // setSignatureVerifier soft-bricked in-flight batches —
        // finalizeBatch tried to settle from the new pool (empty),
        // challenge.slash hit the new bond (or reverted on EOA),
        // verification swapped to a malicious verifier mid-challenge.
        // Per-batch snapshots make these immutable for the batch's
        // lifecycle; setters affect only FUTURE commits, mirroring the
        // D-05 challengeWindowSecondsAtCommit pattern.
        address escrowPoolAtCommit;
        address stakeBondAtCommit;
        address signatureVerifierAtCommit;
        string  metadataURI;          // optional IPFS pointer (see §10.7 of design)
        // sp1240 (TEE Tier-3 roadmap F) — optional commitment to this batch's
        // TEE attestation tier/measurement (off-chain-computed, stored verbatim).
        // APPENDED at the struct end so the existing field layout is unchanged.
        // bytes32(0) = no attestation (legacy commitBatch path). NEVER part of
        // batchId or merkleRoot — pure metadata, zero consensus impact (sp1238).
        bytes32 attestationCommitment;
    }

    /// @dev Challenge window in seconds. Governance-adjustable.
    /// Default 3 days per design §5.1.
    uint256 public challengeWindowSeconds;

    /// @dev Reason codes for challengeReceipt. Ordering is stable; adding
    /// new codes appends to the enum. MALFORMED is reserved but not
    /// implementable on-chain without extra infrastructure; use of it in
    /// Task 3 reverts with NotYetImplemented.
    enum ReasonCode {
        DOUBLE_SPEND,        // 0: receipt present in two committed batches
        INVALID_SIGNATURE,   // 1: Ed25519 sig doesn't verify under declared pubkey
        NO_ESCROW,           // 2: batch.requester attests no matching authorization
        EXPIRED,             // 3: receipt.executed_at_unix beyond lookback window
        MALFORMED,           // 4: reserved (not yet implementable)
        CONSENSUS_MISMATCH   // 5: Phase 7.1 — provider's receipt disagrees with a majority in same batch
    }

    /// @dev EscrowPool contract authorized to execute settlement transfers.
    /// Set by owner via setEscrowPool; finalizeBatch invokes it to move
    /// FTNS from requester's escrow balance to provider's wallet.
    /// May be address(0) in Task 1 scope (no transfer happens); once set
    /// in Task 2, finalizeBatch requires it to be configured.
    IEscrowPool public escrowPool;

    /// @dev Phase 7: StakeBond contract used to slash providers on
    /// successful DOUBLE_SPEND or INVALID_SIGNATURE challenges.
    /// May be address(0) — in which case challenges succeed (receipt
    /// value invalidated) but no stake is slashed (Phase 3.1 behavior
    /// preserved when Phase 7 is not deployed).
    IStakeBond public stakeBond;

    /// @dev Pluggable Ed25519 verifier for INVALID_SIGNATURE challenges.
    /// May be address(0) until a production verifier is deployed;
    /// INVALID_SIGNATURE challenges revert with VerifierNotConfigured
    /// if it is unset.
    ISignatureVerifier public signatureVerifier;

    /// @dev Maximum receipt age (in seconds) that a provider may batch.
    /// EXPIRED challenges succeed if block.timestamp - leaf.executedAtUnix
    /// exceeds this window. Default 30 days per design §5.2 reason 3.
    /// Governance-adjustable within reasonable bounds.
    uint256 public settlementLookbackWindowSeconds;

    uint256 public constant MIN_LOOKBACK_SECONDS = 1 days;
    uint256 public constant MAX_LOOKBACK_SECONDS = 365 days;

    /// @dev invalidatedReceipts[batchId][receiptLeafHash] = true if this
    /// specific receipt has been successfully challenged. Prevents
    /// double-challenges that would double-subtract value.
    mapping(bytes32 batchId => mapping(bytes32 receiptLeafHash => bool))
        public invalidatedReceipts;

    /// @dev Per-provider monotonic counter used in batchId derivation to
    /// avoid collision when a provider commits multiple batches in the
    /// same block with identical merkle roots (pathological, but we want
    /// the contract to be correct under that case).
    mapping(address provider => uint256) public providerBatchSequence;

    /// @dev Primary state: all committed batches keyed by deterministic batchId.
    mapping(bytes32 batchId => Batch) public batches;

    /// @dev L4 self-audit HIGH-1 (A-01 ≡ D-01) fix: per-provider
    ///      monotonic tracker of `commitTimestamp + challengeWindowSecondsAtCommit`
    ///      across this provider's PENDING batches. Maintained by
    ///      commitBatch (max-update); never decremented (a stale value
    ///      below `block.timestamp` is naturally ignored by callers
    ///      because the unbond floor is `max(localFloor, thisValue)`
    ///      and a past timestamp is dominated).
    ///
    /// Read by `StakeBond.requestUnbond` via the
    /// `ISlasherWithProviderExpiry` interface to enforce that the
    /// unbond floor reflects the LONGEST-PINNED challenge window of any
    /// PENDING batch — NOT the live mutable `challengeWindowSeconds`.
    /// Closes the L4 self-audit HIGH-1 composition gap where a
    /// `setChallengeWindowSeconds` reduction after a high-window batch
    /// commit would let the provider unbond before that batch's pinned
    /// window elapsed (re-opening the original A-02/D-01 slash-evasion
    /// race).
    mapping(address provider => uint64) public lastPendingBatchExpiry;

    /// @dev L4 self-audit re-run A-06 fix: snapshot of `totalPausedSeconds`
    ///      at the moment `lastPendingBatchExpiry[provider]` was last
    ///      updated (i.e. for the SAME batch whose expiry currently
    ///      dominates the per-provider tracker). Read by
    ///      `StakeBond.requestUnbond` via the extended interface so the
    ///      unbond floor can be pause-extended:
    ///        pauseAdjustedExpiry =
    ///            lastPendingBatchExpiry[p]
    ///            + (totalPausedSeconds - lastPendingBatchPausedAtAccrual[p])
    ///
    ///      Without this, the wall-clock `lastPendingBatchExpiry` floor
    ///      diverges from the pause-extended actual challenge expiry
    ///      computed by `_effectiveElapsed` when ANY pause occurs during
    ///      the batch's window. Provider could request unbond at the
    ///      wall-clock boundary, withdraw `unbondDelaySeconds` later, and
    ///      a successful challenge in the `pausedSinceCommit`-shaped tail
    ///      of the effective challenge window would hit `WITHDRAWN` →
    ///      `SlashSwallowed`, re-opening the A-01/D-01 slash-evasion race.
    ///
    /// Maintained ATOMICALLY with `lastPendingBatchExpiry`: only written
    /// in `commitBatch` when the new expiry strictly dominates the old
    /// tracker. The pair is therefore always consistent (snapshot belongs
    /// to the same batch whose expiry is currently tracked).
    mapping(address provider => uint64) public lastPendingBatchPausedAtAccrual;

    /// @dev L4 self-audit HIGH-2 (B-01) fix: cumulative seconds the
    ///      contract has been paused since deployment. Incremented only
    ///      by `_unpause()` (so it reflects COMPLETED pauses; the
    ///      currently-active pause, if any, is not counted — and is
    ///      irrelevant for finalize/challenge math because both gate on
    ///      `whenNotPaused`).
    ///
    /// Each batch snapshots this at commit time as
    /// `b.totalPausedAtBatchOrigin`. The effective-elapsed calculation
    /// then subtracts `(totalPausedSeconds - b.totalPausedAtBatchOrigin)`
    /// from the wall-clock delta so that pause time does NOT consume
    /// the batch's challenge window. Closes the HIGH-2 vector where a
    /// compromised owner could pause through a batch's window to deny
    /// challengers their dispute period.
    uint256 public totalPausedSeconds;

    /// @dev Wall-clock timestamp of the most recent `_pause()`. Read by
    ///      `_unpause()` to compute the duration to add to
    ///      `totalPausedSeconds`. Cleared on unpause.
    uint256 public pauseStartedAt;

    /// @dev Minimum and maximum allowed challenge window values. Prevents
    /// governance from setting pathological values (e.g., 0 = no challenge
    /// period; 100 years = funds locked forever). Bounds themselves are
    /// owner-adjustable only via contract upgrade, not governance.
    uint256 public constant MIN_CHALLENGE_WINDOW_SECONDS = 1 hours;
    uint256 public constant MAX_CHALLENGE_WINDOW_SECONDS = 30 days;

    /// @dev Minimum `gasleft()` required immediately before the
    ///      stakeBond.slash call in challengeReceipt. Set conservatively
    ///      below the actual slash cost (~200K gas with storage writes +
    ///      event) but well above Solidity's 63/64 forwarding floor so
    ///      the nested slash cannot silently OOG via the try/catch.
    ///
    ///      Why this exists: without the floor, eth_estimateGas can
    ///      under-budget a CONSENSUS_MISMATCH / DOUBLE_SPEND /
    ///      INVALID_SIGNATURE challenge — the outer tx succeeds (receipt
    ///      invalidation lands) while the inner slash silently reverts
    ///      with out-of-gas, caught by the try/catch. A challenger who
    ///      under-paid gas gets a free burn of the receipt without the
    ///      economic penalty the provider owes. With this floor, the
    ///      outer tx reverts cleanly before any state change, forcing
    ///      the estimator (or a conscientious manual submission) to
    ///      allocate enough gas for the slash to actually complete.
    uint256 public constant MIN_SLASH_GAS = 150_000;

    event BatchCommitted(
        bytes32 indexed batchId,
        address indexed provider,
        bytes32 merkleRoot,
        uint256 receiptCount,
        uint256 totalValueFTNS,
        uint64 commitTimestamp,
        string metadataURI
    );

    /// @dev sp1240 (TEE Tier-3 roadmap F) — emitted ALONGSIDE BatchCommitted
    /// (a SEPARATE event so BatchCommitted's topic0 + decoders stay unchanged)
    /// only when a batch carries a non-zero TEE attestation commitment. Lets
    /// indexers/auditors correlate a batch with its committed attestation tier.
    event BatchAttestationCommitted(
        bytes32 indexed batchId,
        address indexed provider,
        bytes32 attestationCommitment
    );

    event BatchFinalized(
        bytes32 indexed batchId,
        address indexed provider,
        uint256 finalValueFTNS,
        uint256 invalidatedValueFTNS,
        uint64 finalizeTimestamp
    );

    event ChallengeWindowUpdated(uint256 oldSeconds, uint256 newSeconds);
    event EscrowPoolUpdated(address oldPool, address newPool);
    event StakeBondUpdated(address oldBond, address newBond);
    event SignatureVerifierUpdated(address oldVerifier, address newVerifier);
    event SettlementLookbackUpdated(uint256 oldSeconds, uint256 newSeconds);
    event ReceiptChallenged(
        bytes32 indexed batchId,
        bytes32 indexed receiptLeafHash,
        address indexed challenger,
        ReasonCode reason,
        uint128 invalidatedValueFTNS
    );

    /// @dev Forensic-observability event for the case where a successful
    /// challenge's stakeBond.slash() call reverted (e.g., provider already
    /// fully slashed in a prior challenge, stake withdrawn, or stakeBond
    /// itself misconfigured). The receipt-invalidation work is preserved
    /// (the try/catch swallows the revert) but no economic penalty was
    /// applied. Off-chain monitoring (Forta) should alert on this event
    /// because it indicates either (a) a benign double-challenge, or (b)
    /// the L2 audit HIGH-2 race condition resurfacing despite the
    /// StakeBond.requestUnbond floor fix. Per consolidated.md §3 HIGH-2.
    event SlashSwallowed(
        bytes32 indexed batchId,
        address indexed provider,
        address indexed challenger,
        ReasonCode reason
    );

    error InvalidChallengeWindow(uint256 provided);
    error InvalidLookbackWindow(uint256 provided);
    error BatchAlreadyCommitted(bytes32 batchId);
    error BatchNotFound(bytes32 batchId);
    error BatchNotPending(bytes32 batchId, BatchStatus current);
    error ChallengeWindowNotElapsed(bytes32 batchId, uint64 commitTimestamp, uint256 windowSeconds);
    error ChallengeWindowElapsed(bytes32 batchId);
    error EmptyMerkleRoot();
    error ZeroReceiptCount();
    error ZeroRequester();
    error EscrowPoolNotConfigured();
    error InvalidMerkleProof(bytes32 batchId, bytes32 receiptLeafHash);
    error ReceiptAlreadyInvalidated(bytes32 batchId, bytes32 receiptLeafHash);
    error ChallengeNotProven(ReasonCode reason);
    error VerifierNotConfigured();
    error MalformedReasonNotImplemented();
    error CallerNotRequester(address caller, address requester);
    error ReceiptNotExpired(uint64 executedAtUnix, uint256 lookbackSeconds);
    error ConflictingBatchNotCommitted(bytes32 conflictingBatchId);
    error InvalidSlashRateBps(uint16 provided);
    /// @dev L4 self-audit MED-7 (D-03): setters reject EOA / non-contract pointers for non-zero addresses.
    error SetterTargetNotContract(address provided);
    error InsufficientGasForSlash(uint256 available, uint256 required);

    constructor(address initialOwner, uint256 initialChallengeWindow) Ownable(initialOwner) {
        if (initialChallengeWindow < MIN_CHALLENGE_WINDOW_SECONDS ||
            initialChallengeWindow > MAX_CHALLENGE_WINDOW_SECONDS) {
            revert InvalidChallengeWindow(initialChallengeWindow);
        }
        challengeWindowSeconds = initialChallengeWindow;
        settlementLookbackWindowSeconds = 30 days; // default per design §5.2
    }

    /**
     * @notice Commit a batch of off-chain receipts as a single Merkle-root
     *         anchor. Permissionless: any provider can submit.
     * @param merkleRoot keccak256 Merkle root over the set of receipt-hash leaves
     * @param receiptCount number of receipts in the batch
     * @param totalValueFTNS sum of receipt values in FTNS base units
     * @param consensusGroupId Phase 7.1x: non-zero to mark this batch as
     *        part of a k-of-n consensus dispatch. Zero for single-provider
     *        batches. See Batch struct commentary for the sybil-griefing
     *        rationale (§8.7 fix).
     * @param metadataURI optional off-chain pointer (e.g., ipfs://...)
     * @return batchId deterministic identifier for the committed batch
     */
    // sp1240 — shared commit core. Both the legacy commitBatch (attestation =
    // bytes32(0)) and commitBatchWithAttestation delegate here so they can NEVER
    // drift. batchId derivation below is byte-identical regardless of
    // attestationCommitment — it is NOT in the keccak preimage.
    function _commitBatch(
        address requester,
        bytes32 merkleRoot,
        uint256 receiptCount,
        uint256 totalValueFTNS,
        uint16 tierSlashRateBps,
        bytes32 consensusGroupId,
        string calldata metadataURI,
        bytes32 attestationCommitment
    ) internal returns (bytes32 batchId) {
        if (requester == address(0)) revert ZeroRequester();
        if (merkleRoot == bytes32(0)) revert EmptyMerkleRoot();
        if (receiptCount == 0) revert ZeroReceiptCount();
        if (tierSlashRateBps > 10000) revert InvalidSlashRateBps(tierSlashRateBps);

        uint256 sequence = providerBatchSequence[msg.sender]++;

        batchId = keccak256(
            abi.encode(msg.sender, requester, merkleRoot, receiptCount, block.number, sequence)
        );

        // Defensive: deterministic derivation means this should never
        // collide within the same block, but check explicitly in case
        // the derivation ever changes.
        if (batches[batchId].status != BatchStatus.NONEXISTENT) {
            revert BatchAlreadyCommitted(batchId);
        }

        // Construct Batch via storage-pointer assignment rather than a
        // single struct literal — Solidity struct-literal codegen blows
        // the local-variable stack with this many fields (D-03 + D-05
        // snapshots pushed it over). Behaviour-equivalent.
        Batch storage b = batches[batchId];
        b.provider = msg.sender;
        b.requester = requester;
        b.merkleRoot = merkleRoot;
        b.receiptCount = receiptCount;
        b.totalValueFTNS = totalValueFTNS;
        // invalidatedValueFTNS defaults to 0
        b.commitTimestamp = uint64(block.timestamp);
        b.tier_slash_rate_bps = tierSlashRateBps;
        b.consensus_group_id = consensusGroupId;
        // L2 audit MEDIUM D-05 fix: snapshot the live window at commit
        // time so subsequent governance changes via
        // setChallengeWindowSeconds cannot retroactively shorten the
        // dispute period for batches already in PENDING.
        b.challengeWindowSecondsAtCommit = uint64(challengeWindowSeconds);
        // L4 self-audit HIGH-2 (B-01) fix: snapshot the cumulative
        // pause-time at commit. The effective-elapsed calc later
        // subtracts pauses that occur AFTER this commit so they
        // don't consume the challenge window.
        b.totalPausedAtBatchOrigin = uint64(totalPausedSeconds);
        // L4 self-audit MED-3 (B-02) fix: snapshot the lookback window
        // at commit. _handleExpired consults the per-batch snapshot
        // instead of the live mutable global so post-commit changes
        // via setSettlementLookbackWindow cannot retroactively flip
        // EXPIRED challenge eligibility on already-PENDING batches.
        b.lookbackWindowSecondsAtCommit = uint64(settlementLookbackWindowSeconds);
        // L2 audit MEDIUM D-03 fix: snapshot cross-wire pointers.
        b.escrowPoolAtCommit = address(escrowPool);
        b.stakeBondAtCommit = address(stakeBond);
        b.signatureVerifierAtCommit = address(signatureVerifier);
        b.metadataURI = metadataURI;
        b.attestationCommitment = attestationCommitment;  // sp1240 (default bytes32(0))
        // L4 self-audit INFO-5 (D-06) fix: write `status = PENDING` LAST
        // so any future addition that introduces an external call
        // (currently none) cannot observe a half-initialised batch in
        // the PENDING state. No present-day exploit — the function makes
        // no external call between `b.status` and the snapshot writes —
        // but the ordering is now defensively correct against future
        // edits that might add one.
        b.status = BatchStatus.PENDING;

        // L4 self-audit HIGH-1 (A-01 ≡ D-01) fix: maintain the
        // per-provider monotonic max-pending-expiry tracker so
        // StakeBond.requestUnbond can clamp against the LONGEST pinned
        // window of any PENDING batch (not the live mutable global).
        // Stale values are naturally ignored by callers via max-of-floors.
        //
        // L4 self-audit re-run A-06 fix: when expiry is updated, also
        // snapshot `totalPausedSeconds` so the reader can pause-extend
        // the floor in `_effectiveElapsed`-equivalent arithmetic. The
        // pair is written atomically: snapshot belongs to the same batch
        // whose expiry now dominates the tracker.
        uint64 newExpiry = uint64(block.timestamp + uint256(b.challengeWindowSecondsAtCommit));
        if (newExpiry > lastPendingBatchExpiry[msg.sender]) {
            lastPendingBatchExpiry[msg.sender] = newExpiry;
            lastPendingBatchPausedAtAccrual[msg.sender] = uint64(totalPausedSeconds);
        }

        emit BatchCommitted(
            batchId,
            msg.sender,
            merkleRoot,
            receiptCount,
            totalValueFTNS,
            uint64(block.timestamp),
            metadataURI
        );
        // sp1240 — separate additive event, only for attested batches, so
        // BatchCommitted's topic0 + decoders stay unchanged.
        if (attestationCommitment != bytes32(0)) {
            emit BatchAttestationCommitted(batchId, msg.sender, attestationCommitment);
        }
    }

    /**
     * @notice Commit a batch (no TEE attestation). Selector + behavior unchanged
     *         — un-upgraded clients keep working. Delegates to _commitBatch with
     *         a zero attestation commitment.
     */
    function commitBatch(
        address requester,
        bytes32 merkleRoot,
        uint256 receiptCount,
        uint256 totalValueFTNS,
        uint16 tierSlashRateBps,
        bytes32 consensusGroupId,
        string calldata metadataURI
    ) external whenNotPaused returns (bytes32 batchId) {
        return _commitBatch(
            requester, merkleRoot, receiptCount, totalValueFTNS,
            tierSlashRateBps, consensusGroupId, metadataURI, bytes32(0)
        );
    }

    /**
     * @notice sp1240 (TEE Tier-3 roadmap F) — commit a batch WITH an on-chain
     *         commitment to its TEE attestation tier/measurement. Identical to
     *         commitBatch in every consensus-relevant way (same batchId, same
     *         merkleRoot, same challenge/finalize/slash semantics); the only
     *         additions are the stored attestationCommitment + the
     *         BatchAttestationCommitted event. attestationCommitment is
     *         off-chain-computed + stored verbatim (the chain never re-hashes it).
     */
    function commitBatchWithAttestation(
        address requester,
        bytes32 merkleRoot,
        uint256 receiptCount,
        uint256 totalValueFTNS,
        uint16 tierSlashRateBps,
        bytes32 consensusGroupId,
        string calldata metadataURI,
        bytes32 attestationCommitment
    ) external whenNotPaused returns (bytes32 batchId) {
        return _commitBatch(
            requester, merkleRoot, receiptCount, totalValueFTNS,
            tierSlashRateBps, consensusGroupId, metadataURI, attestationCommitment
        );
    }

    /**
     * @notice Finalize a PENDING batch after its challenge window has elapsed.
     *         Permissionless: anyone may call (typically the provider does,
     *         since they benefit from settlement, but the contract accepts
     *         calls from any address so a watchdog can finalize on behalf
     *         of absent providers).
     *
     *         Task 1 scope: transitions state + emits event. Task 2 will
     *         add the FTNS transfer invocation here against EscrowPool;
     *         Task 3 will account for invalidated receipt values against
     *         the final payable amount.
     *
     * @param batchId identifier returned by commitBatch
     */
    function finalizeBatch(bytes32 batchId) external whenNotPaused {
        Batch storage b = batches[batchId];

        if (b.status == BatchStatus.NONEXISTENT) revert BatchNotFound(batchId);
        if (b.status != BatchStatus.PENDING) {
            revert BatchNotPending(batchId, b.status);
        }

        // L2 audit MEDIUM D-05 fix: read the per-batch snapshot, not the
        // live mutable global. Owner cannot retroactively shorten the
        // window for already-PENDING batches.
        // L4 self-audit HIGH-2 (B-01) fix: elapsed calc subtracts
        // pause-time-since-commit so a sustained pause cannot consume
        // the challenge window.
        uint256 elapsed = _effectiveElapsed(b);
        uint256 batchWindow = b.challengeWindowSecondsAtCommit;
        if (elapsed < batchWindow) {
            revert ChallengeWindowNotElapsed(
                batchId, b.commitTimestamp, batchWindow
            );
        }

        b.status = BatchStatus.FINALIZED;

        // Final payable = total - invalidated (Task 3 sets invalidatedValueFTNS).
        // In Task 2 scope, invalidatedValueFTNS is always 0, so finalValue ==
        // totalValueFTNS. Task 3 will track challenges and reduce finalValue.
        uint256 finalValue = b.totalValueFTNS - b.invalidatedValueFTNS;

        // Task 2: execute settlement via EscrowPool. A configured pool is
        // required once we've reached this code path — the contract is
        // meant to actually move value, not just emit events. Setting
        // finalValue=0 (pathological case where every receipt was
        // invalidated) skips the transfer but still finalizes state.
        if (finalValue > 0) {
            // L2 audit MEDIUM D-03 fix: settle against the per-batch
            // snapshot, not the live mutable pointer. Owner cannot
            // re-route in-flight settlements by calling setEscrowPool.
            address poolAtCommit = b.escrowPoolAtCommit;
            if (poolAtCommit == address(0)) {
                revert EscrowPoolNotConfigured();
            }
            IEscrowPool(poolAtCommit).settleFromRequester(b.requester, b.provider, finalValue);
        }

        emit BatchFinalized(
            batchId,
            b.provider,
            finalValue,
            b.invalidatedValueFTNS,
            uint64(block.timestamp)
        );
    }

    // ── Challenges (Task 3) ───────────────────────────────────────

    /**
     * @notice Challenge a specific receipt inside a PENDING batch. Reason-
     *         code-specific verification determines whether the challenge
     *         succeeds; on success, the receipt's value is subtracted from
     *         the batch's final payable amount at finalizeBatch time.
     *
     * @param batchId the pending batch
     * @param leaf canonical ReceiptLeaf being challenged
     * @param merkleProof proof that keccak256(abi.encode(leaf)) is in
     *                    batches[batchId].merkleRoot
     * @param reason which check the contract performs; see ReasonCode
     * @param auxData additional data per-reason-code. See individual
     *                _handle* functions for encoding.
     */
    function challengeReceipt(
        bytes32 batchId,
        ReceiptLeaf calldata leaf,
        bytes32[] calldata merkleProof,
        ReasonCode reason,
        bytes calldata auxData
    ) external whenNotPaused {
        Batch storage b = batches[batchId];
        if (b.status == BatchStatus.NONEXISTENT) revert BatchNotFound(batchId);
        if (b.status != BatchStatus.PENDING) {
            revert BatchNotPending(batchId, b.status);
        }
        // L2 audit MEDIUM D-05 fix: per-batch snapshot, not live global.
        // Owner cannot retroactively LENGTHEN the challenge window via
        // setChallengeWindowSeconds either — once committed, the
        // dispute period is fixed, in both directions.
        // L4 self-audit HIGH-2 (B-01) fix: elapsed subtracts paused
        // time so a sustained pause cannot consume the window.
        uint256 elapsed = _effectiveElapsed(b);
        if (elapsed >= b.challengeWindowSecondsAtCommit) {
            // Cannot challenge after window elapses — finalize is
            // eligible.  Challengers must act within the window.
            revert ChallengeWindowElapsed(batchId);
        }

        bytes32 leafHash = _hashLeaf(leaf);
        if (invalidatedReceipts[batchId][leafHash]) {
            revert ReceiptAlreadyInvalidated(batchId, leafHash);
        }

        if (!MerkleProof.verify(merkleProof, b.merkleRoot, leafHash)) {
            revert InvalidMerkleProof(batchId, leafHash);
        }

        bool proven;
        if (reason == ReasonCode.DOUBLE_SPEND) {
            proven = _handleDoubleSpend(batchId, b, leafHash, auxData);
        } else if (reason == ReasonCode.INVALID_SIGNATURE) {
            proven = _handleInvalidSignature(leaf, auxData, b.signatureVerifierAtCommit);
        } else if (reason == ReasonCode.NO_ESCROW) {
            proven = _handleNoEscrow(b);
        } else if (reason == ReasonCode.EXPIRED) {
            proven = _handleExpired(leaf, b.lookbackWindowSecondsAtCommit);
        } else if (reason == ReasonCode.CONSENSUS_MISMATCH) {
            proven = _handleConsensusMismatch(batchId, b, leaf, auxData);
        } else {
            // MALFORMED reserved — cannot prove on-chain in Task 3 scope.
            revert MalformedReasonNotImplemented();
        }

        if (!proven) revert ChallengeNotProven(reason);

        // Record + accumulate invalidated value.
        invalidatedReceipts[batchId][leafHash] = true;
        b.invalidatedValueFTNS += leaf.valueFtns;

        emit ReceiptChallenged(batchId, leafHash, msg.sender, reason, leaf.valueFtns);

        // Phase 7 + 7.1: DOUBLE_SPEND / INVALID_SIGNATURE / CONSENSUS_MISMATCH
        // challenges slash the provider's stake. NO_ESCROW is requester-
        // attestation-based (griefing risk if slash triggered) and EXPIRED
        // is protocol-hygiene rather than malice — neither triggers slashing.
        // Slashing is best-effort: if stakeBond is unconfigured, or if the
        // provider has no stake, the challenge still succeeds and the
        // receipt stays invalidated.
        // L2 audit MEDIUM D-03 fix: slash against the per-batch
        // snapshot of stakeBond, not the live mutable pointer.
        address bondAtCommit = b.stakeBondAtCommit;
        // sp1456 (slashing audit #1, batch-side) — do NOT gate the slash on the CALLER-supplied
        // `b.tier_slash_rate_bps > 0`. The provider sets that value at commit (validated only <=10000),
        // so committing the fraudulent batch with tierSlashRateBps=0 previously skipped slash()
        // entirely even for a properly-bonded provider. The slash AMOUNT is computed from the provider's
        // ON-CHAIN bonded rate inside StakeBond.slash (now floored per tier by bond()'s
        // minSlashRateForAmount), so caller input must not decide whether slashing fires. Always attempt
        // the slash on a proven fault; a genuinely unslashable stake (open-tier 0-rate bond, or already
        // withdrawn/slashed) is handled by StakeBond reverting into the try/catch below (SlashSwallowed).
        if (
            bondAtCommit != address(0)
            && (
                reason == ReasonCode.DOUBLE_SPEND
                || reason == ReasonCode.INVALID_SIGNATURE
                || reason == ReasonCode.CONSENSUS_MISMATCH
            )
        ) {
            // Gas-floor guard — revert cleanly BEFORE the try/catch if
            // the challenger hasn't funded enough gas for the slash to
            // complete. Without this, the try/catch could swallow a
            // nested out-of-gas revert and leave the receipt invalidated
            // but the provider unslashed — see MIN_SLASH_GAS commentary.
            // Reverting here rolls back the receipt invalidation too, so
            // the challenger's tx is all-or-nothing: either they fund
            // the slash or they get nothing.
            if (gasleft() < MIN_SLASH_GAS) {
                revert InsufficientGasForSlash(gasleft(), MIN_SLASH_GAS);
            }
            // Wrapped in try/catch so a StakeBond-level revert (e.g.,
            // provider already fully slashed, stake already withdrawn)
            // doesn't undo the challenge's receipt-invalidation work.
            // OOG is excluded by the gas floor above; the only catchable
            // reverts now are legitimate slash-ineligibility conditions
            // (NotSlashable, NothingToSlash, CallerNotSlasher) that
            // should NOT unwind the challenge.
            try IStakeBond(bondAtCommit).slash(b.provider, msg.sender, batchId) {
                // success; bounty credited + foundation reserve updated
            } catch {
                // swallow — receipt is already invalidated; slash is
                // best-effort under non-OOG conditions. Emit a forensic
                // event so Forta + off-chain monitoring can alert. See
                // L2 audit HIGH-2 commentary on the SlashSwallowed event.
                emit SlashSwallowed(batchId, b.provider, msg.sender, reason);
            }
        }
    }

    /// @dev Compute the canonical keccak256 leaf hash for a ReceiptLeaf.
    function _hashLeaf(ReceiptLeaf calldata leaf) internal pure returns (bytes32) {
        return keccak256(abi.encode(leaf));
    }

    /**
     * @dev DOUBLE_SPEND: the same receipt was committed in a DIFFERENT,
     *      EARLIER batch. auxData layout:
     *        abi.encode(bytes32 conflictingBatchId, bytes32[] conflictingProof)
     *      Succeeds iff the receipt's leafHash is provable in both `this
     *      batch` (already verified by caller) and the conflicting batch,
     *      AND the conflicting batch is a distinct batch committed strictly
     *      BEFORE this one (first-committer-wins).
     *
     *      Audit fix (money-path #1): this handler previously had NEITHER
     *      guard its sibling _handleConsensusMismatch enforces, so:
     *        (A) self-reference — passing THIS batch as its own conflict made
     *            the caller-verified proof re-verify against the same root, so
     *            ANY receipt in ANY pending batch could be "double-spend"
     *            challenged against itself; and
     *        (B) copycat framing — an attacker commits a throwaway batch
     *            (commitBatch is permissionless) copying the victim's leaf,
     *            then challenges the HONEST provider's batch citing it.
     *      Either invalidated the honest provider's payment, slashed their
     *      stake, and paid the challenger the 70% bounty. The two guards below
     *      (distinct batch + first-committer-wins) close both, mirroring the
     *      protections _handleConsensusMismatch already had.
     */
    function _handleDoubleSpend(
        bytes32 batchId,
        Batch storage b,
        bytes32 leafHash,
        bytes calldata auxData
    ) internal view returns (bool) {
        (bytes32 conflictingBatchId, bytes32[] memory conflictingProof) =
            abi.decode(auxData, (bytes32, bytes32[]));

        // (A) Self-referential challenges are blocked — a batch cannot be its
        //     own conflicting double-spend. (Mirrors _handleConsensusMismatch.)
        if (conflictingBatchId == batchId) return false;

        Batch storage other = batches[conflictingBatchId];
        if (other.status == BatchStatus.NONEXISTENT) {
            revert ConflictingBatchNotCommitted(conflictingBatchId);
        }

        // (B) First-committer-wins: the cited conflict must have been committed
        //     STRICTLY EARLIER than the challenged batch, so the honest first
        //     committer is protected and only a later duplicate is slashable. A
        //     copycat batch committed at or after the victim's batch cannot be
        //     used to slash the victim. Equal timestamps (same block, ambiguous
        //     order) fail closed — no slash.
        if (other.commitTimestamp >= b.commitTimestamp) return false;

        // The receipt must also be provable in the (earlier) conflicting batch.
        return MerkleProof.verify(conflictingProof, other.merkleRoot, leafHash);
    }

    /**
     * @dev INVALID_SIGNATURE: the receipt's signature doesn't verify under
     *      its declared pubkey for the leaf-committed signing-message hash.
     *      auxData layout:
     *        abi.encode(bytes publicKey, bytes signature)
     *      Contract verifies:
     *        - keccak256(publicKey) == leaf.providerPubkeyHash
     *        - keccak256(signature) == leaf.signatureHash
     *        - ISignatureVerifier.verify(leaf.signingMessageHash, ...) returns FALSE
     *      If verify returns TRUE, the signature is genuinely valid over the
     *      committed message → challenge is not proven.
     *
     *      The signing-message hash is consumed directly from the leaf rather
     *      than being supplied by the challenger. This closes L2 audit
     *      C-INT-01: previously the challenger could supply an arbitrary
     *      signingMessage that the receipt's signature was NOT over, causing
     *      the verifier to correctly return FALSE and the contract to slash
     *      the provider. Binding signingMessageHash in the leaf forces the
     *      verifier to check the actual committed message.
     */
    function _handleInvalidSignature(
        ReceiptLeaf calldata leaf,
        bytes calldata auxData,
        address verifierAtCommit
    ) internal view returns (bool) {
        // L2 audit MEDIUM D-03 fix: verify against the per-batch
        // snapshot of signatureVerifier, not the live mutable pointer.
        // Owner cannot rotate to a malicious verifier mid-flight to
        // forge or invalidate signatures on already-PENDING batches.
        if (verifierAtCommit == address(0)) {
            revert VerifierNotConfigured();
        }
        (bytes memory publicKey, bytes memory signature) =
            abi.decode(auxData, (bytes, bytes));

        // Bind the submitted pubkey + signature to the leaf-committed hashes.
        if (keccak256(publicKey) != leaf.providerPubkeyHash) return false;
        if (keccak256(signature) != leaf.signatureHash) return false;

        // Verify against the leaf-committed signingMessageHash, NOT a
        // challenger-supplied value. Closes C-INT-01.
        bool valid = ISignatureVerifier(verifierAtCommit).verify(
            leaf.signingMessageHash, signature, publicKey
        );
        return !valid; // challenge succeeds iff verification fails
    }

    /**
     * @dev NO_ESCROW: the batch's named requester attests (via msg.sender)
     *      that they did not authorize this receipt. No cryptographic
     *      proof of a negative is possible on-chain, so this reduces to
     *      an authorization check — only the batch.requester can invoke.
     *      Their signed transaction IS the attestation.
     */
    function _handleNoEscrow(Batch storage b) internal view returns (bool) {
        if (msg.sender != b.requester) {
            revert CallerNotRequester(msg.sender, b.requester);
        }
        return true;
    }

    /**
     * @dev EXPIRED: receipt older than settlementLookbackWindowSeconds.
     *      Pure time check against leaf.executedAtUnix.
     *
     *      L4 self-audit MED-3 (B-02) fix: takes the per-batch
     *      `lookbackWindowSecondsAtCommit` snapshot, NOT the live
     *      mutable global. Owner cannot retroactively flip EXPIRED
     *      eligibility on already-PENDING batches by mutating
     *      `settlementLookbackWindowSeconds` mid-flight.
     */
    function _handleExpired(
        ReceiptLeaf calldata leaf,
        uint64 lookbackWindowAtCommit
    ) internal view returns (bool) {
        uint256 age = block.timestamp - uint256(leaf.executedAtUnix);
        uint256 lookback = uint256(lookbackWindowAtCommit);
        if (age <= lookback) {
            revert ReceiptNotExpired(leaf.executedAtUnix, lookback);
        }
        return true;
    }

    /**
     * @dev CONSENSUS_MISMATCH (Phase 7.1 + 7.1x §8.7): the provider's
     *      receipt disagrees with a majority receipt from a DIFFERENT
     *      provider's batch for the same shard of the same job under a
     *      k-of-n redundant-execution dispatch. auxData layout:
     *        abi.encode(
     *          bytes32 conflictingBatchId,  // the majority provider's batch
     *          bytes32[] majorityProof,     // proof for the majority leaf
     *          ReceiptLeaf majorityLeaf     // the majority receipt
     *        )
     *      The challenged leaf (passed into challengeReceipt) is the
     *      minority leaf. The challenge slashes b.provider — the
     *      committer of THIS (minority) batch.
     *
     *      Authorization (Phase 7.1x §8.7 update): the challenge is open
     *      to ANY caller — third-party bounty hunters and the original
     *      requester alike. What makes this safe is the consensus_group_id
     *      binding: both batches must carry the same non-zero group_id
     *      (set at commit time), AND they must be committed by DIFFERENT
     *      providers. Without the group_id binding, a sybil attacker
     *      controlling provider P + requester R could commit two P-signed
     *      batches with conflicting outputs and have R challenge to
     *      collect the 70% bounty (griefing, not farming — negative EV
     *      with the 30% Foundation skim, but still possible). With the
     *      binding, an attacker must control k DIFFERENT provider keys
     *      to trigger a slash, multiplying the cost k×.
     *
     *      Checks performed (all must pass):
     *        1. conflictingBatchId != batchId   (distinct batches)
     *        2. b.consensus_group_id != 0       (minority batch opted
     *           into consensus — non-consensus batches can't be
     *           targeted by CONSENSUS_MISMATCH; use DOUBLE_SPEND)
     *        3. b.consensus_group_id == other.consensus_group_id
     *           (both batches belong to the SAME k-of-n dispatch)
     *        4. b.provider != other.provider   (distinct providers —
     *           same-provider disagreement is DOUBLE_SPEND territory,
     *           not CONSENSUS_MISMATCH)
     *        5. leaf.jobIdHash == majorityLeaf.jobIdHash
     *        6. leaf.shardIndex == majorityLeaf.shardIndex
     *        7. leaf.outputHash != majorityLeaf.outputHash
     *        8. conflictingBatch exists
     *        9. MerkleProof.verify(majorityProof, conflictingBatch.root,
     *             hash(majorityLeaf))
     *
     *      Returns true iff all pass.
     */
    function _handleConsensusMismatch(
        bytes32 batchId,
        Batch storage b,
        ReceiptLeaf calldata leaf,
        bytes calldata auxData
    ) internal view returns (bool) {
        (
            bytes32 conflictingBatchId,
            bytes32[] memory majorityProof,
            ReceiptLeaf memory majorityLeaf
        ) = abi.decode(auxData, (bytes32, bytes32[], ReceiptLeaf));

        // Distinct batches. Self-referential challenges are blocked here
        // regardless of what else matches.
        if (conflictingBatchId == batchId) return false;

        // Consensus-group opt-in: the minority batch must have been
        // committed with a non-zero group_id. Batches that didn't opt in
        // to consensus can't be targeted by CONSENSUS_MISMATCH.
        if (b.consensus_group_id == bytes32(0)) return false;

        // Same shard of same job — otherwise "disagreement" is meaningless.
        if (leaf.jobIdHash != majorityLeaf.jobIdHash) return false;
        if (leaf.shardIndex != majorityLeaf.shardIndex) return false;

        // Genuine disagreement on output bytes.
        if (leaf.outputHash == majorityLeaf.outputHash) return false;

        // The conflicting batch must be committed.
        Batch storage other = batches[conflictingBatchId];
        if (other.status == BatchStatus.NONEXISTENT) {
            revert ConflictingBatchNotCommitted(conflictingBatchId);
        }

        // Sybil-griefing mitigation: both batches must share the same
        // non-zero group_id AND be committed by different providers.
        if (b.consensus_group_id != other.consensus_group_id) return false;
        if (b.provider == other.provider) return false;

        // Majority leaf must be provable in the conflicting batch.
        bytes32 majorityLeafHash = keccak256(abi.encode(majorityLeaf));
        if (!MerkleProof.verify(majorityProof, other.merkleRoot, majorityLeafHash)) {
            return false;
        }

        return true;
    }

    // ── Governance surface ────────────────────────────────────────

    /**
     * @notice Set the EscrowPool contract that executes settlement transfers.
     *         Owner-only. The pool must be deployed separately; this function
     *         registers its address with the registry. Setting to address(0)
     *         effectively disables finalization of non-zero-value batches.
     * @param newPool EscrowPool contract address
     */
    function setEscrowPool(address newPool) external onlyOwner {
        // L4 self-audit MED-7 (D-03) fix: non-zero values must be
        // contracts. Zero is permitted (documented "disable" mode).
        // The D-03 per-batch snapshot protects in-flight batches from
        // mid-flight rotation; this guard prevents new batches from
        // being committed against an EOA / non-conforming target.
        if (newPool != address(0) && newPool.code.length == 0) {
            revert SetterTargetNotContract(newPool);
        }
        address old = address(escrowPool);
        escrowPool = IEscrowPool(newPool);
        emit EscrowPoolUpdated(old, newPool);
    }

    /**
     * @notice Set the StakeBond contract used to slash providers on
     *         DOUBLE_SPEND / INVALID_SIGNATURE challenges. Owner-only.
     *         Address(0) disables slashing (receipt invalidation still
     *         works — Phase 3.1 behavior preserved).
     */
    function setStakeBond(address newBond) external onlyOwner {
        // L4 self-audit MED-7 (D-03) fix: non-zero values must be contracts.
        if (newBond != address(0) && newBond.code.length == 0) {
            revert SetterTargetNotContract(newBond);
        }
        address old = address(stakeBond);
        stakeBond = IStakeBond(newBond);
        emit StakeBondUpdated(old, newBond);
    }

    /**
     * @notice Set the pluggable Ed25519 verifier used by INVALID_SIGNATURE
     *         challenges. Owner-only. May be address(0) to effectively
     *         disable INVALID_SIGNATURE challenges (they revert with
     *         VerifierNotConfigured).
     */
    function setSignatureVerifier(address newVerifier) external onlyOwner {
        // L4 self-audit MED-7 (D-03) fix: non-zero values must be contracts.
        if (newVerifier != address(0) && newVerifier.code.length == 0) {
            revert SetterTargetNotContract(newVerifier);
        }
        address old = address(signatureVerifier);
        signatureVerifier = ISignatureVerifier(newVerifier);
        emit SignatureVerifierUpdated(old, newVerifier);
    }

    /**
     * @notice Set the EXPIRED-receipt lookback window. Owner-only.
     *         Bounded [1 day, 365 days] to prevent pathological values.
     */
    function setSettlementLookbackWindow(uint256 newSeconds) external onlyOwner {
        if (newSeconds < MIN_LOOKBACK_SECONDS || newSeconds > MAX_LOOKBACK_SECONDS) {
            revert InvalidLookbackWindow(newSeconds);
        }
        uint256 old = settlementLookbackWindowSeconds;
        settlementLookbackWindowSeconds = newSeconds;
        emit SettlementLookbackUpdated(old, newSeconds);
    }

    /**
     * @notice Update the challenge-window duration. Owner-only.
     *         Affects ONLY future commitBatch calls. L2 audit MEDIUM
     *         D-05 fix: every Batch struct snapshots
     *         `challengeWindowSecondsAtCommit` at commit time;
     *         finalization + challenge eligibility consult that
     *         per-batch field. Pre-fix, owner could shorten the live
     *         global mid-flight and retroactively allow already-PENDING
     *         batches to finalize before their original window
     *         elapsed — robbing challengers of promised time. Post-fix,
     *         the only effect of this setter is on batches committed
     *         AFTER the change.
     *
     *         Operational note: PRSM-GOV-1 §10.3 (14-day advance notice
     *         for non-emergency on-chain governance) still applies, but
     *         the on-chain enforcement is now a per-batch immutable
     *         field rather than a notice-period social commitment.
     *
     * @param newSeconds new window duration in seconds
     */
    function setChallengeWindowSeconds(uint256 newSeconds) external onlyOwner {
        if (newSeconds < MIN_CHALLENGE_WINDOW_SECONDS ||
            newSeconds > MAX_CHALLENGE_WINDOW_SECONDS) {
            revert InvalidChallengeWindow(newSeconds);
        }
        uint256 old = challengeWindowSeconds;
        challengeWindowSeconds = newSeconds;
        emit ChallengeWindowUpdated(old, newSeconds);
    }

    // ── Pause control (L2 audit HIGH-3 / D-02) ────────────────────

    /**
     * @notice Pause commitBatch / finalizeBatch / challengeReceipt.
     *         Owner-only; intended for incident response per
     *         docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md. Admin setters
     *         (setEscrowPool, setStakeBond, setSignatureVerifier,
     *         setChallengeWindowSeconds, setSettlementLookbackWindow)
     *         remain accessible while paused so the owner can perform
     *         emergency rotation.
     *
     *         Note: pausing BSR also implicitly halts EscrowPool
     *         settlement (finalizeBatch is the only path that calls it).
     *         If EscrowPool is paused independently, finalizeBatch on
     *         this contract will revert at the settle step. Both
     *         pauses are operationally fine and do not lock funds.
     */
    function pause() external onlyOwner {
        _pause();
    }

    /// @notice Resume normal operations after pause.
    function unpause() external onlyOwner {
        _unpause();
    }

    // ── Views ────────────────────────────────────────────────────

    /// @notice Read the full Batch struct for a given batchId.
    function getBatch(bytes32 batchId) external view returns (Batch memory) {
        return batches[batchId];
    }

    /// @notice True iff the batch exists + is PENDING + window has elapsed.
    /// @dev Lightweight pre-check for would-be finalizers.
    /// @dev L2 audit MEDIUM D-05 fix: reads per-batch snapshot, not
    ///      live mutable global. Mirrors finalizeBatch logic.
    function isFinalizable(bytes32 batchId) external view returns (bool) {
        Batch storage b = batches[batchId];
        if (b.status != BatchStatus.PENDING) return false;
        return _effectiveElapsed(b) >= b.challengeWindowSecondsAtCommit;
    }

    /// @notice Seconds remaining until a PENDING batch can be finalized.
    ///         Returns 0 if the window has elapsed or the batch isn't PENDING.
    /// @dev L2 audit MEDIUM D-05 fix: per-batch snapshot, not live global.
    /// @dev L4 self-audit HIGH-2 (B-01) fix: pause-aware elapsed.
    function secondsUntilFinalizable(bytes32 batchId) external view returns (uint256) {
        Batch storage b = batches[batchId];
        if (b.status != BatchStatus.PENDING) return 0;
        uint256 elapsed = _effectiveElapsed(b);
        uint256 batchWindow = b.challengeWindowSecondsAtCommit;
        if (elapsed >= batchWindow) return 0;
        return batchWindow - elapsed;
    }

    /// @dev L4 self-audit HIGH-2 (B-01) helper: compute the elapsed
    ///      time since `b.commitTimestamp`, MINUS pause-time that
    ///      occurred since the commit. Used by finalizeBatch,
    ///      challengeReceipt, isFinalizable, and secondsUntilFinalizable
    ///      to ensure that pausing the contract during a batch's
    ///      challenge window does NOT consume that window.
    ///
    /// `totalPausedSeconds` only increments on `_unpause()` (so it
    /// reflects COMPLETED pauses; an in-progress pause is not counted).
    /// Both finalizeBatch and challengeReceipt are gated on
    /// `whenNotPaused`, so they cannot fire during an in-progress pause
    /// anyway. The view functions (isFinalizable, secondsUntilFinalizable)
    /// will momentarily UNDER-report during an in-progress pause, but
    /// callers can't act on that information until the contract is
    /// unpaused — at which point `totalPausedSeconds` catches up
    /// atomically.
    function _effectiveElapsed(Batch storage b) internal view returns (uint256) {
        uint256 wall = block.timestamp - uint256(b.commitTimestamp);
        uint256 pausedSinceCommit = totalPausedSeconds - uint256(b.totalPausedAtBatchOrigin);
        // pausedSinceCommit is bounded by wall (every paused second
        // counted in `totalPausedSeconds - totalPausedAtBatchOrigin`
        // has to have elapsed between commit and now), so the
        // subtraction is safe in Solidity 0.8 checked arithmetic. We
        // still defensive-clamp to handle any future code path that
        // could violate the invariant.
        return wall > pausedSinceCommit ? wall - pausedSinceCommit : 0;
    }

    /// @dev L4 self-audit HIGH-2 (B-01) fix: override OZ Pausable's
    ///      `_pause` to record the wall-clock at which pause started.
    ///      The accumulator update happens on `_unpause` — that's when
    ///      we know the duration.
    function _pause() internal override {
        pauseStartedAt = block.timestamp;
        super._pause();
    }

    /// @dev L4 self-audit HIGH-2 (B-01) fix: override OZ Pausable's
    ///      `_unpause` to add the just-completed pause's duration to
    ///      `totalPausedSeconds`. Subsequent commits' batches will
    ///      record this incremented value as their
    ///      `totalPausedAtBatchOrigin`, so they correctly start with
    ///      "0 paused-time since their commit."
    function _unpause() internal override {
        totalPausedSeconds += block.timestamp - pauseStartedAt;
        pauseStartedAt = 0;
        super._unpause();
    }
}
