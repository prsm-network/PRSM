// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title CreatorStakeRegistry
 * @notice Sprint 976 — on-chain teeth for the Vision §14 anti-spam creator
 *         stake. A creator bonds real FTNS as collateral ("uploading requires
 *         collateral that can be slashed"); spam → the bonded stake is slashed,
 *         making spam-uploading economically unattractive. The Python
 *         CreatorStakeClient reads `creatorStakeOf(creator)` to gate HIGH
 *         creator-tier eligibility; pre-deploy it used an in-memory mirror with
 *         NO economic teeth (sp290 scaffold), which sp970/the marketplace review
 *         flagged. This contract is that real backend.
 *
 * @dev Mirrors StakeBond's security posture: Ownable2Step (owner = Foundation
 *      Safe, accepts via acceptOwnership), ReentrancyGuard, Pausable, an
 *      immutable FTNS token + immutable slasher, transferFrom-on-stake, an
 *      unbond DELAY before withdraw, and a slasher-gated slash whose proceeds
 *      accrue to a Foundation reserve. Stake counts toward eligibility ONLY
 *      while BONDED — requesting unbond drops `creatorStakeOf` to 0 immediately,
 *      closing the stake→get-tier→spam→unstake game.
 *
 *      Commissioning (the irreversible mainnet steps, NOT done here): deploy
 *      with owner = Foundation Safe + slasher = the governance/Foundation slash
 *      authority, Sepolia-rehearse the full stake/unbond/withdraw/slash
 *      round-trip, then accept ownership via the 2-of-3 multisig. See the
 *      Sepolia rehearsal runbook accompanying this sprint.
 */
contract CreatorStakeRegistry is Ownable2Step, ReentrancyGuard, Pausable {
    enum StakeStatus {
        NONE,        // never staked / fully withdrawn
        BONDED,      // active collateral; counts toward HIGH-tier eligibility
        UNBONDING,   // requestUnbond fired; waiting for the delay to elapse
        WITHDRAWN    // funds returned to the creator
    }

    struct CreatorStake {
        uint256 amount;             // FTNS wei currently locked
        uint64 bondedAt;
        uint64 unbondEligibleAt;    // 0 while BONDED; set when requestUnbond fires
        StakeStatus status;
    }

    IERC20 public immutable ftns;

    /// @dev Seconds a creator must wait between requestUnbond() and withdraw().
    uint256 public unbondDelaySeconds;
    uint256 public constant MIN_UNBOND_DELAY_SECONDS = 1 days;
    uint256 public constant MAX_UNBOND_DELAY_SECONDS = 30 days;

    /// @dev Authorized slasher. IMMUTABLE (mirrors StakeBond HIGH-7): set once
    /// at construction, no setter. Production = the governance/Foundation slash
    /// authority. address(0) is rejected at construction.
    address public immutable slasher;

    mapping(address creator => CreatorStake) public stakes;

    /// @dev Accrued slash proceeds, drained by the owner to the reserve wallet.
    uint256 public foundationReserveBalance;
    /// @dev Destination for drainFoundationReserve (owner-set; non-zero required).
    address public foundationReserveWallet;

    event CreatorStaked(address indexed creator, uint256 amount, uint256 newTotal);
    event UnbondRequested(address indexed creator, uint64 eligibleAt);
    event Withdrawn(address indexed creator, uint256 amount);
    event CreatorSlashed(
        address indexed creator, uint256 amount, uint256 remaining, string reason
    );
    event UnbondDelayUpdated(uint256 oldDelay, uint256 newDelay);
    event FoundationReserveWalletUpdated(address oldWallet, address newWallet);
    event FoundationReserveDrained(address indexed to, uint256 amount);

    error ZeroAddress();
    error ZeroAmount();
    error InvalidUnbondDelay(uint256 delay);
    error NotSlasher();
    error NotBonded();
    error NotUnbonding();
    error UnbondNotElapsed(uint64 eligibleAt);
    error InsufficientStake(uint256 staked, uint256 requested);
    error TransferFailed();
    error NoReserveWallet();
    error WalletNotContract(address provided);

    constructor(
        address initialOwner,
        address ftnsAddress,
        uint256 initialUnbondDelay,
        address initialSlasher
    ) Ownable(initialOwner) {
        if (ftnsAddress == address(0)) revert ZeroAddress();
        if (initialUnbondDelay < MIN_UNBOND_DELAY_SECONDS ||
            initialUnbondDelay > MAX_UNBOND_DELAY_SECONDS) {
            revert InvalidUnbondDelay(initialUnbondDelay);
        }
        // Immutable slasher must be non-zero (else slash() is permanently
        // bricked with no setter recovery — loud-fail at deploy, like StakeBond).
        if (initialSlasher == address(0)) revert ZeroAddress();
        ftns = IERC20(ftnsAddress);
        unbondDelaySeconds = initialUnbondDelay;
        slasher = initialSlasher;
    }

    // ── Creator lifecycle ──────────────────────────────────────────

    /**
     * @notice Bond FTNS as anti-spam collateral. Caller must have approved this
     *         contract for at least `amount`. Re-staking while UNBONDING flips
     *         the record back to BONDED (the creator changed their mind).
     */
    function stake(uint256 amount) external nonReentrant whenNotPaused {
        if (amount == 0) revert ZeroAmount();
        bool ok = ftns.transferFrom(msg.sender, address(this), amount);
        if (!ok) revert TransferFailed();
        CreatorStake storage s = stakes[msg.sender];
        s.amount += amount;
        s.status = StakeStatus.BONDED;
        s.unbondEligibleAt = 0;
        if (s.bondedAt == 0) s.bondedAt = uint64(block.timestamp);
        emit CreatorStaked(msg.sender, amount, s.amount);
    }

    /**
     * @notice Begin unbonding. Eligibility (creatorStakeOf) drops to 0
     *         immediately so a creator cannot keep HIGH tier while exiting.
     */
    function requestUnbond() external whenNotPaused {
        // sp979 — whenNotPaused: pause freezes ALL state transitions (mirrors
        // StakeBond). NOTE (audit, discretionary-slasher SLA): unlike StakeBond
        // (whose slasher is an async-challenge registry needing a window floor),
        // this slasher is a discretionary governance/Foundation authority and
        // slash() works throughout the full unbondDelaySeconds window on both
        // BONDED + UNBONDING. Set unbondDelaySeconds >= the spam-detection SLA so
        // a creator can't unbond+withdraw before a warranted slash can land.
        CreatorStake storage s = stakes[msg.sender];
        if (s.status != StakeStatus.BONDED || s.amount == 0) revert NotBonded();
        s.status = StakeStatus.UNBONDING;
        s.unbondEligibleAt = uint64(block.timestamp + unbondDelaySeconds);
        emit UnbondRequested(msg.sender, s.unbondEligibleAt);
    }

    /// @notice Withdraw bonded FTNS after the unbond delay has elapsed.
    function withdraw() external nonReentrant whenNotPaused {
        CreatorStake storage s = stakes[msg.sender];
        if (s.status != StakeStatus.UNBONDING) revert NotUnbonding();
        if (block.timestamp < s.unbondEligibleAt) {
            revert UnbondNotElapsed(s.unbondEligibleAt);
        }
        uint256 amount = s.amount;
        s.amount = 0;
        s.status = StakeStatus.WITHDRAWN;
        s.unbondEligibleAt = 0;
        bool ok = ftns.transfer(msg.sender, amount);
        if (!ok) revert TransferFailed();
        emit Withdrawn(msg.sender, amount);
    }

    // ── Slashing ───────────────────────────────────────────────────

    /**
     * @notice Slash a creator's bonded collateral for spam/abuse. Slasher-only.
     *         Proceeds accrue to the Foundation reserve (owner-drained). Works
     *         whether the creator is BONDED or UNBONDING (a spammer can't dodge
     *         a slash by starting to unbond — the funds are still custodied here
     *         until withdraw, and withdraw can only move what remains).
     */
    function slash(
        address creator, uint256 amount, string calldata reason
    ) external whenNotPaused {
        if (msg.sender != slasher) revert NotSlasher();
        if (amount == 0) revert ZeroAmount();
        CreatorStake storage s = stakes[creator];
        if (s.amount < amount) revert InsufficientStake(s.amount, amount);
        s.amount -= amount;
        foundationReserveBalance += amount;
        emit CreatorSlashed(creator, amount, s.amount, reason);
    }

    // ── Views ──────────────────────────────────────────────────────

    /// @notice Bonded collateral counting toward HIGH-tier eligibility. Returns
    /// 0 unless the creator is actively BONDED (UNBONDING/WITHDRAWN → 0).
    function creatorStakeOf(address creator) external view returns (uint256) {
        CreatorStake storage s = stakes[creator];
        return s.status == StakeStatus.BONDED ? s.amount : 0;
    }

    // ── Owner / Foundation ─────────────────────────────────────────

    function setUnbondDelay(uint256 newDelay) external onlyOwner {
        if (newDelay < MIN_UNBOND_DELAY_SECONDS ||
            newDelay > MAX_UNBOND_DELAY_SECONDS) {
            revert InvalidUnbondDelay(newDelay);
        }
        emit UnbondDelayUpdated(unbondDelaySeconds, newDelay);
        unbondDelaySeconds = newDelay;
    }

    function setFoundationReserveWallet(address wallet) external onlyOwner whenNotPaused {
        if (wallet == address(0)) revert ZeroAddress();
        // sp979 (audit MEDIUM, mirrors StakeBond L505): the reserve must be a
        // CONTRACT (the Foundation Safe), never an EOA. A typo or compromised
        // owner setting an EOA here would let drainFoundationReserve() misroute
        // accrued slash proceeds to a key-controlled wallet. code.length > 0
        // rejects EOAs (note: it cannot catch a not-yet-deployed CREATE2 addr,
        // but blocks the realistic typo/EOA-misroute case).
        if (wallet.code.length == 0) revert WalletNotContract(wallet);
        emit FoundationReserveWalletUpdated(foundationReserveWallet, wallet);
        foundationReserveWallet = wallet;
    }

    function drainFoundationReserve() external onlyOwner nonReentrant whenNotPaused {
        if (foundationReserveWallet == address(0)) revert NoReserveWallet();
        uint256 amount = foundationReserveBalance;
        foundationReserveBalance = 0;
        bool ok = ftns.transfer(foundationReserveWallet, amount);
        if (!ok) revert TransferFailed();
        emit FoundationReserveDrained(foundationReserveWallet, amount);
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }
}
