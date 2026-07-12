// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title CompensationDistributor
 * @notice Phase 8 distributor — pulls FTNS emission from EmissionController
 * and splits it across three compensation pools under governance-set weights.
 *
 * Per docs/2026-04-22-phase8-design-plan.md §3.5, §4.2, §6 Task 2.
 *
 * Design commitments:
 *
 *  - Permissionless `pullAndDistribute`. Anyone can trigger distribution —
 *    the contract imposes no access control on the flow itself. Operator
 *    economics depend on this being called frequently enough that accrued
 *    allowance is drained; monitoring alerts on call-gap > 7 days per
 *    Phase 8 plan §8.2 + EmissionController design.
 *
 *  - Two-phase weight updates with 90-day advance notice on-chain.
 *    `updateWeights(newWeights, scheduledAt)` requires
 *    `scheduledAt >= now + 90 days` — the 90-day stakeholder-notice period
 *    from PRSM-GOV-1 §4.1 is enforced at the contract level rather than
 *    purely operationally. Pending weights activate lazily on the next
 *    `pullAndDistribute` after `scheduledAt`; no separate activation tx is
 *    required, which avoids a governance-drift window where old weights
 *    outlive their schedule.
 *
 *  - Integer-exact split with dust accumulation in the grant pool. Creator
 *    and operator shares are computed via integer bps math; any rounding
 *    remainder accrues to grant. Since bps-sum is validated == 10000, the
 *    remainder is never negative and is bounded by 2 wei per call.
 *
 *  - Pool weight + address updates are owner-only (Foundation multi-sig).
 *    Both emit events so governance history is reconstructable from logs.
 */
interface IEmissionControllerView {
    function currentEpochRate() external view returns (uint256);
    function lastMintTimestamp() external view returns (uint64);
    function mintedToDate() external view returns (uint256);
    function mintCap() external view returns (uint256);
    function paused() external view returns (bool);
    function mintAuthorized(uint256 amount) external;
}

contract CompensationDistributor is Ownable2Step, ReentrancyGuard {
    // -------------------------------------------------------------------------
    // Types
    // -------------------------------------------------------------------------

    struct PoolWeights {
        uint16 creatorPoolBps;
        uint16 operatorPoolBps;
        uint16 grantPoolBps;
    }

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    /// @notice 90-day stakeholder-notice period required before a weight
    /// change takes effect. Enforced on-chain, matching PRSM-GOV-1 §4.1.
    uint256 public constant MIN_WEIGHT_SCHEDULE_DELAY = 90 days;

    uint256 internal constant BPS_DENOMINATOR = 10_000;

    // -------------------------------------------------------------------------
    // Immutable wiring
    // -------------------------------------------------------------------------

    IERC20 public immutable ftnsToken;
    IEmissionControllerView public immutable emissionController;

    // -------------------------------------------------------------------------
    // Mutable state
    // -------------------------------------------------------------------------

    address public creatorPool;
    address public operatorPool;
    address public grantPool;

    PoolWeights public currentWeights;

    PoolWeights public scheduledWeights;
    uint64 public scheduledAt;
    bool public hasScheduledWeights;

    uint64 public lastDistributionTimestamp;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event Distributed(uint256 toCreator, uint256 toOperator, uint256 toGrant);
    event WeightsScheduled(PoolWeights newWeights, uint64 effectiveTimestamp);
    event WeightsActivated(PoolWeights newWeights);
    event PoolAddressesUpdated(
        address indexed creator,
        address indexed operator,
        address indexed grant
    );

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error InvalidAddress();
    error InvalidWeights();
    error ScheduleTooSoon(uint256 scheduledAt, uint256 minAllowed);
    error TransferFailed();

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    constructor(
        address _ftnsToken,
        address _emissionController,
        address _creatorPool,
        address _operatorPool,
        address _grantPool,
        PoolWeights memory _initialWeights,
        address _initialOwner
    ) Ownable(_initialOwner) {
        if (
            _ftnsToken == address(0) ||
            _emissionController == address(0) ||
            _creatorPool == address(0) ||
            _operatorPool == address(0) ||
            _grantPool == address(0)
        ) revert InvalidAddress();

        _validateWeights(_initialWeights);

        ftnsToken = IERC20(_ftnsToken);
        emissionController = IEmissionControllerView(_emissionController);
        creatorPool = _creatorPool;
        operatorPool = _operatorPool;
        grantPool = _grantPool;
        currentWeights = _initialWeights;
    }

    // -------------------------------------------------------------------------
    // Permissionless distribution
    // -------------------------------------------------------------------------

    /// @notice Mint the currently-allowed emission from the controller and
    /// distribute this contract's full balance across the three pools per
    /// the current weights.
    ///
    /// Callable by anyone — this is a utility call, not a privileged one.
    /// Activates scheduled weights if their `scheduledAt` has been reached.
    function pullAndDistribute() external nonReentrant {
        _applyScheduledWeightsIfActive();
        _tryPull();
        _distribute();
    }

    function _tryPull() internal {
        // If the controller is paused, skip pull — distributing any residual
        // balance is still valid behaviour.
        if (emissionController.paused()) return;

        uint256 rate = emissionController.currentEpochRate();
        if (rate == 0) return;

        uint64 lastMint = emissionController.lastMintTimestamp();
        if (block.timestamp <= lastMint) return;

        uint256 elapsed = block.timestamp - lastMint;
        uint256 toMint = rate * elapsed;

        uint256 remaining =
            emissionController.mintCap() - emissionController.mintedToDate();
        if (toMint > remaining) toMint = remaining;
        if (toMint == 0) return;

        emissionController.mintAuthorized(toMint);
    }

    // slither-disable-start reentrancy-balance
    // TRIAGE (SAST baseline): FALSE POSITIVE. _distribute is internal and reached only through
    // pullAndDistribute()/distribute(), which are `nonReentrant` (this contract is a
    // ReentrancyGuard). ftnsToken is the project's own OZ-based FTNSTokenSimple — a plain ERC20
    // with no ERC777/transfer-callback hooks, so transfer() cannot re-enter. The transfers all
    // check their return and revert TransferFailed, and `available` is read once and consumed, not
    // re-read across the calls. No exploitable reentrancy. Suppressed at the site so `fail-on:high`
    // still catches any NEW reentrancy-balance elsewhere.
    function _distribute() internal {
        uint256 available = ftnsToken.balanceOf(address(this));
        if (available == 0) return;

        PoolWeights memory w = currentWeights;
        uint256 toCreator = (available * w.creatorPoolBps) / BPS_DENOMINATOR;
        uint256 toOperator = (available * w.operatorPoolBps) / BPS_DENOMINATOR;
        // Any rounding dust (bounded by 2 wei) accrues to grant pool.
        uint256 toGrant = available - toCreator - toOperator;

        if (toCreator > 0 && !ftnsToken.transfer(creatorPool, toCreator)) {
            revert TransferFailed();
        }
        if (toOperator > 0 && !ftnsToken.transfer(operatorPool, toOperator)) {
            revert TransferFailed();
        }
        if (toGrant > 0 && !ftnsToken.transfer(grantPool, toGrant)) {
            revert TransferFailed();
        }

        lastDistributionTimestamp = uint64(block.timestamp);
        emit Distributed(toCreator, toOperator, toGrant);
    }
    // slither-disable-end reentrancy-balance

    // -------------------------------------------------------------------------
    // Weight scheduling
    // -------------------------------------------------------------------------

    /// @notice Schedule a weight update. `scheduledAt` must be at least 90
    /// days in the future. The new weights replace the current weights on
    /// the first `pullAndDistribute` call at or after `scheduledAt`.
    function updateWeights(PoolWeights calldata newWeights, uint64 scheduledAtTs)
        external
        onlyOwner
    {
        _validateWeights(newWeights);
        uint256 minAllowed = block.timestamp + MIN_WEIGHT_SCHEDULE_DELAY;
        if (scheduledAtTs < minAllowed) {
            revert ScheduleTooSoon(scheduledAtTs, minAllowed);
        }
        scheduledWeights = newWeights;
        scheduledAt = scheduledAtTs;
        hasScheduledWeights = true;
        emit WeightsScheduled(newWeights, scheduledAtTs);
    }

    function _applyScheduledWeightsIfActive() internal {
        if (hasScheduledWeights && block.timestamp >= scheduledAt) {
            currentWeights = scheduledWeights;
            hasScheduledWeights = false;
            scheduledAt = 0;
            // scheduledWeights left as-is — the delete would cost gas and
            // a stale-but-superseded value carries no semantic meaning;
            // `hasScheduledWeights` is the authority.
            emit WeightsActivated(currentWeights);
        }
    }

    // -------------------------------------------------------------------------
    // Pool address rotation
    // -------------------------------------------------------------------------

    function setPoolAddresses(
        address creator,
        address operator,
        address grant
    ) external onlyOwner {
        if (creator == address(0) || operator == address(0) || grant == address(0)) {
            revert InvalidAddress();
        }
        creatorPool = creator;
        operatorPool = operator;
        grantPool = grant;
        emit PoolAddressesUpdated(creator, operator, grant);
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    function _validateWeights(PoolWeights memory w) internal pure {
        uint256 sum =
            uint256(w.creatorPoolBps) +
            uint256(w.operatorPoolBps) +
            uint256(w.grantPoolBps);
        if (sum != BPS_DENOMINATOR) revert InvalidWeights();
    }
}
