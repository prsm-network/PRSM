// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title IStakeBondSlasher
 * @notice Minimal interface this contract uses to slash storage providers.
 *         Matches the Phase 7 StakeBond.sol `slash` function signature —
 *         StakeBond itself handles the 70/30 bounty split + challenger
 *         self-slash (100% to Foundation) per §3.4.
 */
interface IStakeBondSlasher {
    function slash(
        address provider,
        address challenger,
        bytes32 reasonId
    ) external;
}

/**
 * @title StorageSlashing
 * @notice Phase 7-storage slashing enforcement.
 *
 * Per docs/2026-04-22-phase7-storage-design-plan.md §6 Task 3. Acts as the
 * StakeBond `slasher` for storage-proof failures and heartbeat-missing
 * events. The 70/30 bounty split, challenger-self-slash convention, and
 * FTNS custody all live in StakeBond — this contract is the event source.
 *
 * Authorization model:
 *
 *   - `submitProofFailure` is VERIFIER-gated. Storage-proof challenge/
 *     response (Task 4) is a Foundation-operated service today. The
 *     verifier address is the public key of that service; decentralising
 *     the challenger is a Phase 7-storage.x concern. The verifier can
 *     credit any `challenger` address as the bounty recipient, including
 *     the actual off-chain consumer who submitted the failed request.
 *
 *   - `slashForMissingHeartbeat` is PERMISSIONLESS. The heartbeat-grace
 *     math is self-evident from block timestamps — anyone can trigger
 *     the slash once the grace window has elapsed, and the caller
 *     receives the challenger bounty. Incentivises distributed
 *     monitoring without requiring Foundation gating.
 *
 * Double-slash prevention:
 *
 *   - Every slash is keyed by a keccak256-derived slashId. The
 *     `slashRecorded` mapping blocks a second slash for the same
 *     (type, provider, evidence) triple. Prevents replay and
 *     prevents two verifiers crediting the same failure to different
 *     challengers.
 */
contract StorageSlashing is Ownable2Step, ReentrancyGuard {
    // -------------------------------------------------------------------------
    // Immutable wiring
    // -------------------------------------------------------------------------

    /// @notice Phase 7 StakeBond instance — receives our `slash` calls.
    IStakeBondSlasher public immutable stakeBond;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    uint256 public constant MIN_HEARTBEAT_GRACE = 1 hours;
    uint256 public constant MAX_HEARTBEAT_GRACE = 30 days;

    // sp940 — a provider must miss `slashGraceMultiplier` whole grace windows
    // (not one) before a permissionless heartbeat slash applies. This gives an
    // honest operator a real recovery buffer for a transient RPC/connectivity
    // blip, instead of being slashable the instant one grace window lapses (the
    // honest-operator griefing the storage-slashing review surfaced). Bounded +
    // governance-settable so the Foundation can tune it without another redeploy.
    uint256 public constant MIN_SLASH_GRACE_MULTIPLIER = 1;
    uint256 public constant MAX_SLASH_GRACE_MULTIPLIER = 10;

    uint256 public heartbeatGraceSeconds;
    uint256 public slashGraceMultiplier;
    address public authorizedVerifier;

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    /// @notice Most recent heartbeat timestamp per provider.
    /// Zero means the provider has never heartbeated — required to
    /// heartbeat at least once before a heartbeat-missing slash can
    /// apply, so new providers are not vulnerable immediately after
    /// bonding.
    mapping(address => uint64) public lastHeartbeat;

    /// @notice Prevents the same slash from executing twice.
    mapping(bytes32 => bool) public slashRecorded;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event HeartbeatRecorded(address indexed provider, uint64 timestamp);
    event ProofFailureSlashed(
        address indexed provider,
        address indexed challenger,
        bytes32 indexed shardId,
        bytes32 evidenceHash,
        bytes32 slashId
    );
    event HeartbeatMissingSlashed(
        address indexed provider,
        address indexed challenger,
        uint64 lastHeartbeatAt,
        bytes32 slashId
    );
    event HeartbeatGraceUpdated(uint256 oldGrace, uint256 newGrace);
    event SlashGraceMultiplierUpdated(uint256 oldMultiplier, uint256 newMultiplier);
    event AuthorizedVerifierUpdated(
        address indexed oldVerifier,
        address indexed newVerifier
    );

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error InvalidAddress();
    error GraceOutOfRange(uint256 grace, uint256 minAllowed, uint256 maxAllowed);
    error MultiplierOutOfRange(uint256 multiplier, uint256 minAllowed, uint256 maxAllowed);
    error NotAuthorizedVerifier();
    error AlreadySlashed(bytes32 slashId);
    error HeartbeatNotRecorded();
    error HeartbeatNotExpired(uint256 nowTs, uint256 expiryTs);

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    constructor(
        address _stakeBond,
        address _authorizedVerifier,
        uint256 _heartbeatGraceSeconds,
        address _initialOwner
    ) Ownable(_initialOwner) {
        if (_stakeBond == address(0)) revert InvalidAddress();
        if (_authorizedVerifier == address(0)) revert InvalidAddress();
        if (
            _heartbeatGraceSeconds < MIN_HEARTBEAT_GRACE ||
            _heartbeatGraceSeconds > MAX_HEARTBEAT_GRACE
        ) {
            revert GraceOutOfRange(
                _heartbeatGraceSeconds,
                MIN_HEARTBEAT_GRACE,
                MAX_HEARTBEAT_GRACE
            );
        }
        stakeBond = IStakeBondSlasher(_stakeBond);
        authorizedVerifier = _authorizedVerifier;
        heartbeatGraceSeconds = _heartbeatGraceSeconds;
        // sp940 — default to requiring TWO missed grace windows before a
        // heartbeat slash (honest-operator recovery buffer). Tunable via
        // setSlashGraceMultiplier within [MIN, MAX].
        slashGraceMultiplier = 2;
    }

    // -------------------------------------------------------------------------
    // Heartbeats
    // -------------------------------------------------------------------------

    /// @notice Provider self-reports liveness. No access control —
    /// heartbeats from non-providers are harmless (no associated stake).
    function recordHeartbeat() external {
        uint64 ts = uint64(block.timestamp);
        lastHeartbeat[msg.sender] = ts;
        emit HeartbeatRecorded(msg.sender, ts);
    }

    // -------------------------------------------------------------------------
    // Slash — proof failure
    // -------------------------------------------------------------------------

    /// @notice Slash a provider for a failed storage-proof challenge.
    ///         Caller must be the authorised verifier.
    ///
    /// @param provider     address whose stake is slashed
    /// @param shardId      identifier of the shard whose proof failed
    /// @param evidenceHash opaque hash of the off-chain proof-failure
    ///                     evidence; included in the slashId so identical
    ///                     (provider, shardId) pairs with different
    ///                     evidence each slash independently
    /// @param challenger   credited as challenger on StakeBond (receives
    ///                     the 70% bounty). If equal to provider,
    ///                     StakeBond routes 100% to Foundation.
    function submitProofFailure(
        address provider,
        bytes32 shardId,
        bytes32 evidenceHash,
        address challenger
    ) external nonReentrant {
        if (msg.sender != authorizedVerifier) revert NotAuthorizedVerifier();
        if (provider == address(0)) revert InvalidAddress();

        bytes32 slashId = keccak256(
            abi.encode("proof", provider, shardId, evidenceHash)
        );
        if (slashRecorded[slashId]) revert AlreadySlashed(slashId);
        slashRecorded[slashId] = true;

        stakeBond.slash(provider, challenger, slashId);
        emit ProofFailureSlashed(
            provider, challenger, shardId, evidenceHash, slashId
        );
    }

    // -------------------------------------------------------------------------
    // Slash — missing heartbeat (permissionless)
    // -------------------------------------------------------------------------

    /// @notice Slash a provider whose last heartbeat is older than the
    ///         grace period. Anyone can call — the caller receives the
    ///         challenger bounty.
    function slashForMissingHeartbeat(address provider)
        external
        nonReentrant
    {
        if (provider == address(0)) revert InvalidAddress();

        uint64 last = lastHeartbeat[provider];
        if (last == 0) revert HeartbeatNotRecorded();

        // sp940 — slashable only after `slashGraceMultiplier` whole grace
        // windows have elapsed, so a single missed window (transient RPC /
        // connectivity blip) does NOT slash an honest operator.
        uint256 expiry = uint256(last) + heartbeatGraceSeconds * slashGraceMultiplier;
        if (block.timestamp <= expiry) {
            revert HeartbeatNotExpired(block.timestamp, expiry);
        }

        bytes32 slashId = keccak256(
            abi.encode("heartbeat", provider, last)
        );
        if (slashRecorded[slashId]) revert AlreadySlashed(slashId);
        slashRecorded[slashId] = true;

        stakeBond.slash(provider, msg.sender, slashId);
        emit HeartbeatMissingSlashed(provider, msg.sender, last, slashId);
    }

    // -------------------------------------------------------------------------
    // Governance
    // -------------------------------------------------------------------------

    function setAuthorizedVerifier(address newVerifier) external onlyOwner {
        if (newVerifier == address(0)) revert InvalidAddress();
        address old = authorizedVerifier;
        authorizedVerifier = newVerifier;
        emit AuthorizedVerifierUpdated(old, newVerifier);
    }

    function setHeartbeatGrace(uint256 newGrace) external onlyOwner {
        if (newGrace < MIN_HEARTBEAT_GRACE || newGrace > MAX_HEARTBEAT_GRACE) {
            revert GraceOutOfRange(
                newGrace, MIN_HEARTBEAT_GRACE, MAX_HEARTBEAT_GRACE
            );
        }
        uint256 old = heartbeatGraceSeconds;
        heartbeatGraceSeconds = newGrace;
        emit HeartbeatGraceUpdated(old, newGrace);
    }

    /// @notice sp940 — set how many whole grace windows a provider must miss
    ///         before a permissionless heartbeat slash applies. Bounded to
    ///         [MIN_SLASH_GRACE_MULTIPLIER, MAX_SLASH_GRACE_MULTIPLIER].
    function setSlashGraceMultiplier(uint256 newMultiplier) external onlyOwner {
        if (
            newMultiplier < MIN_SLASH_GRACE_MULTIPLIER ||
            newMultiplier > MAX_SLASH_GRACE_MULTIPLIER
        ) {
            revert MultiplierOutOfRange(
                newMultiplier,
                MIN_SLASH_GRACE_MULTIPLIER,
                MAX_SLASH_GRACE_MULTIPLIER
            );
        }
        uint256 old = slashGraceMultiplier;
        slashGraceMultiplier = newMultiplier;
        emit SlashGraceMultiplierUpdated(old, newMultiplier);
    }
}
