// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

/**
 * @title OperatorRewardPool
 * @notice The missing pool -> earner rail (sp1481).
 *
 * WHY THIS EXISTS
 * ---------------
 * EmissionController mints protocol FTNS and CompensationDistributor splits it
 * across three pool ADDRESSES. Until now there was no contract, service or CLI
 * anywhere that routed those pooled funds onward to the individual operators who
 * actually earned them — so `setPoolAddresses` had no meaningful address to point
 * at (all three pools were copies of the Foundation Safe) and emitted rewards
 * could not reach a participant. This contract is that address: fund it, publish a
 * per-epoch Merkle root of (account, amount) entitlements, and each earner claims
 * permissionlessly with a proof.
 *
 * TRUST MODEL
 * -----------
 *  - `owner` (the Foundation Safe, via Ownable2Step) configures the publisher and
 *    can pause. It CANNOT rewrite a published epoch and cannot withdraw funds that
 *    are already reserved for earners (see `sweepSurplus`).
 *  - `rootPublisher` is a separate, lower-privilege role: an automated key that
 *    publishes epoch roots. Splitting it from `owner` means the hot key that runs
 *    the epoch job can never move funds or change ownership.
 *  - Anyone may `claim` with a valid proof. Claiming is permissionless and the
 *    recipient is the leaf's `account`, never `msg.sender`, so a third party (or a
 *    relayer) can submit a claim on an earner's behalf without being able to
 *    redirect it.
 *
 * SOLVENCY (the property that makes this safe to fund incrementally)
 * -----------------------------------------------------------------
 * A Merkle-root airdrop whose pool is under-funded pays the fastest claimers and
 * strands the rest. `publishEpoch` therefore REFUSES a root whose `totalAmount`
 * is not already covered by the contract's uncommitted balance:
 *
 *     balanceOf(this) >= totalReserved + totalAmount
 *
 * `totalReserved` is the outstanding (published-but-unclaimed) liability across
 * all epochs. So every published epoch is fully backed at publication time, and
 * `sweepSurplus` can only ever move the balance ABOVE that liability. An earner's
 * claim can never fail for lack of funds.
 *
 * LEAF ENCODING (must match prsm/settlement/reward_epoch.py)
 * ---------------------------------------------------------
 *     leaf = keccak256(bytes.concat(keccak256(abi.encode(epochId, account, amount))))
 *
 * The DOUBLE hash is deliberate: it is OpenZeppelin's standard defence against a
 * second-preimage attack in which a 64-byte internal tree node is presented as if
 * it were a leaf. `epochId` is bound INTO the leaf so a proof for epoch N can
 * never be replayed against epoch M, even if the same root were ever republished.
 * Pair hashing uses OZ's sorted-pair convention (MerkleProof.verify), which the
 * Python builder in prsm/settlement/merkle.py already implements.
 */
contract OperatorRewardPool is Ownable2Step, ReentrancyGuard {
    // ── Immutable config ────────────────────────────────────────────────
    IERC20 public immutable ftns;

    /// @notice Hard floor on how long an epoch's funds stay claimable before the
    ///         owner may reclaim the remainder. Immutable and long on purpose: a
    ///         short/owner-tunable window would let the Foundation reclaim funds
    ///         out from under a slow earner, which is exactly the rug this rail
    ///         must not enable.
    uint64 public constant MIN_CLAIM_WINDOW = 365 days;

    // ── Roles ───────────────────────────────────────────────────────────
    /// @notice Authorized to publish epoch roots. Separate from `owner` so the
    ///         automated epoch job runs on a key that cannot move funds.
    address public rootPublisher;

    /// @notice When true, publishing and claiming are halted (incident response).
    bool public paused;

    // ── Epoch state ─────────────────────────────────────────────────────
    struct Epoch {
        bytes32 merkleRoot;    // immutable once published
        uint256 totalAmount;   // funds reserved for this epoch
        uint256 claimedAmount; // running total claimed
        uint64 publishedAt;    // timestamp; gates reclaimUnclaimed
        bool reclaimed;        // remainder returned to the pool surplus
    }

    mapping(uint256 => Epoch) public epochs;

    /// @notice epochId => account => claimed. Enforces exactly-once per earner.
    mapping(uint256 => mapping(address => bool)) public hasClaimed;

    /// @notice Outstanding liability: published-but-unclaimed across all epochs.
    ///         Everything at or below this is spoken for and cannot be swept.
    uint256 public totalReserved;

    // ── Events ──────────────────────────────────────────────────────────
    event EpochPublished(
        uint256 indexed epochId,
        bytes32 merkleRoot,
        uint256 totalAmount,
        uint64 publishedAt
    );
    event RewardClaimed(
        uint256 indexed epochId,
        address indexed account,
        uint256 amount
    );
    event RootPublisherUpdated(address indexed previous, address indexed current);
    event PausedSet(bool paused);
    event SurplusSwept(address indexed to, uint256 amount);
    event UnclaimedReclaimed(uint256 indexed epochId, uint256 amount);

    // ── Errors ──────────────────────────────────────────────────────────
    error ZeroAddress();
    error NotPublisher();
    error IsPaused();
    error EpochAlreadyPublished(uint256 epochId);
    error EmptyMerkleRoot();
    error ZeroAmount();
    error InsufficientBacking(uint256 available, uint256 required);
    error EpochNotPublished(uint256 epochId);
    error AlreadyClaimed(uint256 epochId, address account);
    error InvalidProof();
    error EpochOverdrawn(uint256 epochId, uint256 requested, uint256 remaining);
    error TransferFailed();
    error ClaimWindowOpen(uint64 reclaimableAt);
    error AlreadyReclaimed(uint256 epochId);
    error NoSurplus();

    modifier onlyPublisher() {
        if (msg.sender != rootPublisher) revert NotPublisher();
        _;
    }

    modifier whenNotPaused() {
        if (paused) revert IsPaused();
        _;
    }

    constructor(address _ftns, address _initialOwner, address _rootPublisher)
        Ownable(_initialOwner)
    {
        if (_ftns == address(0) || _initialOwner == address(0)
            || _rootPublisher == address(0)) revert ZeroAddress();
        ftns = IERC20(_ftns);
        rootPublisher = _rootPublisher;
        emit RootPublisherUpdated(address(0), _rootPublisher);
    }

    // ── Leaf hashing ────────────────────────────────────────────────────

    /// @notice The canonical leaf hash. Exposed so the off-chain builder and any
    ///         integrator can assert byte-for-byte parity against the chain rather
    ///         than re-deriving the convention (a mismatch silently produces
    ///         unverifiable proofs).
    function leafHash(uint256 epochId, address account, uint256 amount)
        public
        pure
        returns (bytes32)
    {
        return keccak256(
            bytes.concat(keccak256(abi.encode(epochId, account, amount)))
        );
    }

    // ── Publishing ──────────────────────────────────────────────────────

    /**
     * @notice Publish an epoch's entitlement root. Reverts unless the pool already
     *         holds enough uncommitted FTNS to cover EVERY leaf in the tree, so a
     *         published epoch is always fully backed.
     * @param epochId    Monotonic epoch identifier (also bound into each leaf).
     * @param merkleRoot Root over leafHash(epochId, account, amount) leaves.
     * @param totalAmount Sum of all leaf amounts. This is the reserved liability.
     */
    function publishEpoch(uint256 epochId, bytes32 merkleRoot, uint256 totalAmount)
        external
        onlyPublisher
        whenNotPaused
    {
        if (epochs[epochId].publishedAt != 0) revert EpochAlreadyPublished(epochId);
        if (merkleRoot == bytes32(0)) revert EmptyMerkleRoot();
        if (totalAmount == 0) revert ZeroAmount();

        // Solvency gate — the whole point. Uncommitted balance must cover this
        // epoch in full, otherwise early claimers would strand later ones.
        uint256 balance = ftns.balanceOf(address(this));
        uint256 required = totalReserved + totalAmount;
        if (balance < required) revert InsufficientBacking(balance, required);

        epochs[epochId] = Epoch({
            merkleRoot: merkleRoot,
            totalAmount: totalAmount,
            claimedAmount: 0,
            publishedAt: uint64(block.timestamp),
            reclaimed: false
        });
        totalReserved = required;

        emit EpochPublished(epochId, merkleRoot, totalAmount, uint64(block.timestamp));
    }

    // ── Claiming ────────────────────────────────────────────────────────

    /**
     * @notice Claim an epoch entitlement. Permissionless: anyone may submit a valid
     *         proof, but funds always go to `account` (never msg.sender), so a
     *         relayer cannot redirect an earner's reward.
     */
    function claim(
        uint256 epochId,
        address account,
        uint256 amount,
        bytes32[] calldata proof
    ) external nonReentrant whenNotPaused {
        Epoch storage e = epochs[epochId];
        if (e.publishedAt == 0) revert EpochNotPublished(epochId);
        if (hasClaimed[epochId][account]) revert AlreadyClaimed(epochId, account);
        if (amount == 0) revert ZeroAmount();

        if (!MerkleProof.verify(proof, e.merkleRoot, leafHash(epochId, account, amount))) {
            revert InvalidProof();
        }

        // Defence in depth: a root whose leaves sum to more than the declared
        // totalAmount must not drain other epochs' reserved funds.
        uint256 remaining = e.totalAmount - e.claimedAmount;
        if (amount > remaining) revert EpochOverdrawn(epochId, amount, remaining);

        // Effects before interaction (CEI) — mirrors the audited
        // ContentAccessVerifier.claim() pull-payment shape.
        hasClaimed[epochId][account] = true;
        e.claimedAmount += amount;
        totalReserved -= amount;

        if (!ftns.transfer(account, amount)) revert TransferFailed();
        emit RewardClaimed(epochId, account, amount);
    }

    // ── Owner operations ────────────────────────────────────────────────

    function setRootPublisher(address newPublisher) external onlyOwner {
        if (newPublisher == address(0)) revert ZeroAddress();
        emit RootPublisherUpdated(rootPublisher, newPublisher);
        rootPublisher = newPublisher;
    }

    function setPaused(bool p) external onlyOwner {
        paused = p;
        emit PausedSet(p);
    }

    /**
     * @notice Move funds NOT reserved for any published epoch. This is the only
     *         owner path out of the contract, and it can never touch an earner's
     *         reserved entitlement.
     */
    function sweepSurplus(address to, uint256 amount) external onlyOwner nonReentrant {
        if (to == address(0)) revert ZeroAddress();
        uint256 balance = ftns.balanceOf(address(this));
        uint256 available = balance > totalReserved ? balance - totalReserved : 0;
        if (available == 0 || amount > available) revert NoSurplus();
        if (!ftns.transfer(to, amount)) revert TransferFailed();
        emit SurplusSwept(to, amount);
    }

    /**
     * @notice After MIN_CLAIM_WINDOW, release an epoch's unclaimed remainder back
     *         into the sweepable surplus so funds are not stranded forever. The
     *         window is an immutable constant so this can never be shortened to
     *         reclaim funds out from under a slow earner.
     */
    function reclaimUnclaimed(uint256 epochId) external onlyOwner {
        Epoch storage e = epochs[epochId];
        if (e.publishedAt == 0) revert EpochNotPublished(epochId);
        if (e.reclaimed) revert AlreadyReclaimed(epochId);

        uint64 reclaimableAt = e.publishedAt + MIN_CLAIM_WINDOW;
        if (block.timestamp < reclaimableAt) revert ClaimWindowOpen(reclaimableAt);

        uint256 remaining = e.totalAmount - e.claimedAmount;
        e.reclaimed = true;
        if (remaining > 0) {
            totalReserved -= remaining;
        }
        emit UnclaimedReclaimed(epochId, remaining);
    }

    // ── Views ───────────────────────────────────────────────────────────

    /// @notice FTNS held but not reserved for any published epoch.
    function surplus() external view returns (uint256) {
        uint256 balance = ftns.balanceOf(address(this));
        return balance > totalReserved ? balance - totalReserved : 0;
    }

    /// @notice Whether `account` can still claim `amount` from `epochId` with `proof`.
    ///         Pure read — lets the CLI tell an operator "you have X claimable"
    ///         without simulating a transaction.
    function isClaimable(
        uint256 epochId,
        address account,
        uint256 amount,
        bytes32[] calldata proof
    ) external view returns (bool) {
        Epoch storage e = epochs[epochId];
        if (e.publishedAt == 0 || hasClaimed[epochId][account] || amount == 0) {
            return false;
        }
        if (amount > e.totalAmount - e.claimedAmount) return false;
        return MerkleProof.verify(proof, e.merkleRoot, leafHash(epochId, account, amount));
    }
}
