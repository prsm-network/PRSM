// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @title Admin bounded-setter symbolic proofs (sprint 362)
/// @notice For each protocol contract that exposes an
///         owner-only setter on a parameterized rate bound,
///         this file proves that the setter can NEVER
///         produce out-of-range state — for ANY symbolic
///         caller input. The bounds ARE the runtime
///         behavior, not just declarative comments.
///
///         Runtime probe (formal_invariants.py) verifies
///         the CURRENT live MIN/MAX constants on Base
///         mainnet. These symbolic proofs verify that the
///         constants are LOAD-BEARING — that the bounds
///         check on every mutating path actually enforces
///         the documented range.
///
/// @dev Coverage:
///   StorageSlashing.setHeartbeatGrace      → INV-SS-1, INV-SS-2
///   StorageSlashing.setSlashGraceMultiplier→ INV-SS-3, INV-SS-4 (sp940)
///   StakeBond.setUnbondDelay             → INV-SB-1, INV-SB-2
///   StakeBond.CHALLENGER_BOUNTY_BPS      → INV-SB-3 (constant)
///   CompensationDistributor.updateWeights→ INV-CD-1 (anti-rugpull
///                                          90-day weight-schedule delay)
///   EmissionController.constructor       → INV-EC-1, INV-EC-2
///                                          (chainid-8453 enforces
///                                          4-year halving cadence)
///
/// @dev STRUCTURAL EQUIVALENCE:
///   Mirrors source line-by-line for the bounds-check
///   logic. Orthogonal complexity (events, onlyOwner,
///   ReentrancyGuard) omitted — they don't affect the
///   value-range arithmetic.
///
///   StorageSlashing.setHeartbeatGrace →
///     contracts/contracts/StorageSlashing.sol:258-267
///   StorageSlashing.setSlashGraceMultiplier →
///     contracts/contracts/StorageSlashing.sol:272-286
///   StakeBond.setUnbondDelay →
///     contracts/contracts/StakeBond.sol:464-470

contract StorageSlashingBounded {
    uint256 public constant MIN_HEARTBEAT_GRACE = 1 hours;
    uint256 public constant MAX_HEARTBEAT_GRACE = 30 days;
    // sp940 — whole grace windows a provider must miss before a
    // permissionless heartbeat slash applies. Bounded [1, 10].
    uint256 public constant MIN_SLASH_GRACE_MULTIPLIER = 1;
    uint256 public constant MAX_SLASH_GRACE_MULTIPLIER = 10;
    uint256 public heartbeatGraceSeconds;
    uint256 public slashGraceMultiplier;

    constructor(uint256 initial) {
        require(
            initial >= MIN_HEARTBEAT_GRACE
            && initial <= MAX_HEARTBEAT_GRACE
        );
        heartbeatGraceSeconds = initial;
        // Constructor seeds a valid mid-range default (2),
        // mirroring StorageSlashing.sol's constructor.
        slashGraceMultiplier = 2;
    }

    function setHeartbeatGrace(uint256 newGrace) external {
        if (
            newGrace < MIN_HEARTBEAT_GRACE
            || newGrace > MAX_HEARTBEAT_GRACE
        ) {
            revert("OutOfRange");
        }
        heartbeatGraceSeconds = newGrace;
    }

    function setSlashGraceMultiplier(uint256 newMultiplier) external {
        if (
            newMultiplier < MIN_SLASH_GRACE_MULTIPLIER
            || newMultiplier > MAX_SLASH_GRACE_MULTIPLIER
        ) {
            revert("OutOfRange");
        }
        slashGraceMultiplier = newMultiplier;
    }
}


contract StakeBondBounded {
    uint256 public constant MIN_UNBOND_DELAY_SECONDS = 1 days;
    uint256 public constant MAX_UNBOND_DELAY_SECONDS = 30 days;
    uint16 public constant CHALLENGER_BOUNTY_BPS = 7000;
    uint256 public unbondDelaySeconds;

    constructor(uint256 initial) {
        require(
            initial >= MIN_UNBOND_DELAY_SECONDS
            && initial <= MAX_UNBOND_DELAY_SECONDS
        );
        unbondDelaySeconds = initial;
    }

    function setUnbondDelay(uint256 newDelay) external {
        if (
            newDelay < MIN_UNBOND_DELAY_SECONDS
            || newDelay > MAX_UNBOND_DELAY_SECONDS
        ) {
            revert("OutOfRange");
        }
        unbondDelaySeconds = newDelay;
    }
}


/// Mirrors CompensationDistributor.updateWeights bound
/// (CompensationDistributor.sol:215-228). The anti-rugpull
/// invariant: scheduledAt is NEVER set to a value less
/// than (block.timestamp + MIN_WEIGHT_SCHEDULE_DELAY). The
/// 90-day delay is what makes a unilateral weight-flip by
/// a compromised owner key publicly observable for a full
/// quarter before it takes effect.
contract CompensationDistributorBounded {
    uint256 public constant MIN_WEIGHT_SCHEDULE_DELAY =
        90 days;
    uint64 public scheduledAt;
    bool public hasScheduledWeights;

    function updateWeights(uint64 scheduledAtTs) external {
        uint256 minAllowed = block.timestamp
            + MIN_WEIGHT_SCHEDULE_DELAY;
        if (scheduledAtTs < minAllowed) {
            revert("ScheduleTooSoon");
        }
        scheduledAt = scheduledAtTs;
        hasScheduledWeights = true;
    }
}


/// Mirrors EmissionController constructor's chainid-8453
/// halving enforcement (EmissionController.sol:175-196).
/// The constructor chooses between MAINNET_EPOCH_DURATION
/// (4 years) and MIN_EPOCH_DURATION (1 hour) based on
/// chainid. On mainnet, the value is structurally pinned.
contract EmissionControllerBounded {
    uint256 public constant MAINNET_EPOCH_DURATION_SECONDS =
        4 * 365 days;
    uint256 public constant MIN_EPOCH_DURATION_SECONDS =
        1 hours;
    uint256 public constant BASE_MAINNET_CHAIN_ID = 8453;
    uint256 public immutable EPOCH_DURATION_SECONDS;

    constructor(uint256 _epochDurationSeconds) {
        uint256 minRequired = block.chainid
            == BASE_MAINNET_CHAIN_ID
                ? MAINNET_EPOCH_DURATION_SECONDS
                : MIN_EPOCH_DURATION_SECONDS;
        if (
            _epochDurationSeconds
            != MAINNET_EPOCH_DURATION_SECONDS
            && block.chainid == BASE_MAINNET_CHAIN_ID
        ) {
            revert("MainnetEpochMismatch");
        }
        if (_epochDurationSeconds < minRequired) {
            revert("EpochTooShort");
        }
        EPOCH_DURATION_SECONDS = _epochDurationSeconds;
    }
}


/// Halmos spec — bounds-check enforcement
contract AdminBoundedSettersSpec {
    StorageSlashingBounded internal ss;
    StakeBondBounded internal sb;
    CompensationDistributorBounded internal cd;
    EmissionControllerBounded internal ec;

    function setUp() public {
        // Seed each with a valid mid-range initial value.
        // Halmos test names are constants → symbolic.
        ss = new StorageSlashingBounded(1 hours);
        sb = new StakeBondBounded(1 days);
        cd = new CompensationDistributorBounded();
        // Halmos default block.chainid = 1 (non-mainnet
        // branch). 1 hour is the testnet-acceptable epoch.
        ec = new EmissionControllerBounded(1 hours);
    }

    /// THE bounded-setter invariant for StorageSlashing.
    /// For ALL symbolic `newGrace` inputs, the post-state
    /// `heartbeatGraceSeconds` MUST be in
    /// [MIN_HEARTBEAT_GRACE, MAX_HEARTBEAT_GRACE].
    ///
    /// Halmos explores both paths:
    ///   (a) newGrace in range → state updated, post-state
    ///       is newGrace which is in range
    ///   (b) newGrace out of range → revert, state
    ///       unchanged, post-state is initial which is in
    ///       range (by setUp guarantee)
    function check_storage_slashing_grace_always_in_range(
        uint256 newGrace
    ) public {
        try ss.setHeartbeatGrace(newGrace) {}
        catch {}
        uint256 post = ss.heartbeatGraceSeconds();
        assert(post >= ss.MIN_HEARTBEAT_GRACE());
        assert(post <= ss.MAX_HEARTBEAT_GRACE());
    }

    /// MIN/MAX constants pinned (mirrors INV-SS-1+2)
    function check_storage_slashing_bounds_constants()
        public view
    {
        assert(ss.MIN_HEARTBEAT_GRACE() == 1 hours);
        assert(ss.MAX_HEARTBEAT_GRACE() == 30 days);
    }

    /// sp940 — THE bounded-setter invariant for the new
    /// slashGraceMultiplier. For ALL symbolic inputs, the
    /// post-state MUST stay in [MIN, MAX] — so an honest
    /// operator's anti-griefing grace can never be set to 0
    /// (which would re-enable single-window slashing) nor to
    /// an absurd value. Mirrors INV-SS-3.
    function check_storage_slashing_multiplier_always_in_range(
        uint256 newMultiplier
    ) public {
        try ss.setSlashGraceMultiplier(newMultiplier) {}
        catch {}
        uint256 post = ss.slashGraceMultiplier();
        assert(post >= ss.MIN_SLASH_GRACE_MULTIPLIER());
        assert(post <= ss.MAX_SLASH_GRACE_MULTIPLIER());
    }

    /// MIN/MAX constants pinned for the multiplier. Mirrors
    /// INV-SS-4. The lower bound of 1 is load-bearing: it
    /// forbids a 0-multiplier that would zero out the grace
    /// window and re-open the one-blip slash hole sp940 closed.
    function check_storage_slashing_multiplier_constants()
        public view
    {
        assert(ss.MIN_SLASH_GRACE_MULTIPLIER() == 1);
        assert(ss.MAX_SLASH_GRACE_MULTIPLIER() == 10);
    }

    /// Negative direction for the multiplier: an out-of-range
    /// (below-min, i.e. 0) input MUST revert — proves the
    /// revert path actually fires, not just that state stays
    /// in range by luck.
    function check_storage_slashing_multiplier_rejects_below_min(
        uint256 newMultiplier
    ) public {
        vm_assume(newMultiplier < ss.MIN_SLASH_GRACE_MULTIPLIER());
        try ss.setSlashGraceMultiplier(newMultiplier) {
            assert(false);
        } catch {
            assert(true);
        }
    }

    /// Bounded-setter invariant for StakeBond.
    function check_stake_bond_delay_always_in_range(
        uint256 newDelay
    ) public {
        try sb.setUnbondDelay(newDelay) {}
        catch {}
        uint256 post = sb.unbondDelaySeconds();
        assert(post >= sb.MIN_UNBOND_DELAY_SECONDS());
        assert(post <= sb.MAX_UNBOND_DELAY_SECONDS());
    }

    /// MIN/MAX constants pinned (mirrors INV-SB-1+2)
    function check_stake_bond_bounds_constants() public view {
        assert(sb.MIN_UNBOND_DELAY_SECONDS() == 1 days);
        assert(sb.MAX_UNBOND_DELAY_SECONDS() == 30 days);
    }

    /// Anti-confiscation invariant: CHALLENGER_BOUNTY_BPS
    /// is canonical 7000 (70%). Mirrors INV-SB-3.
    function check_stake_bond_challenger_bounty_pinned()
        public view
    {
        assert(sb.CHALLENGER_BOUNTY_BPS() == 7000);
    }

    /// Negative direction: out-of-range MUST revert. This
    /// is the dual of check_*_always_in_range — proves the
    /// revert path actually fires.
    function check_storage_slashing_rejects_below_min(
        uint256 newGrace
    ) public {
        // Constrain to clearly-out-of-range space.
        vm_assume(newGrace < ss.MIN_HEARTBEAT_GRACE());
        try ss.setHeartbeatGrace(newGrace) {
            // Should NOT reach — revert was expected.
            assert(false);
        } catch {
            // Correct path.
            assert(true);
        }
    }

    function check_stake_bond_rejects_above_max(
        uint256 newDelay
    ) public {
        vm_assume(newDelay > sb.MAX_UNBOND_DELAY_SECONDS());
        try sb.setUnbondDelay(newDelay) {
            assert(false);
        } catch {
            assert(true);
        }
    }

    /// CompensationDistributor anti-rugpull invariant:
    /// scheduledAt is NEVER set to a timestamp closer than
    /// 90 days from now. For ALL symbolic scheduledAtTs
    /// inputs:
    ///   (a) If accepted, post-state scheduledAt is at
    ///       least block.timestamp + 90 days
    ///   (b) If rejected, scheduledAt unchanged (zero
    ///       from setUp); trivially in bounds
    function check_compensation_weight_schedule_delay(
        uint64 scheduledAtTs
    ) public {
        try cd.updateWeights(scheduledAtTs) {
            // Accepted → must honor 90-day delay
            assert(
                cd.scheduledAt()
                >= block.timestamp
                    + cd.MIN_WEIGHT_SCHEDULE_DELAY()
            );
        } catch {
            // Rejected → state unchanged
            assert(cd.scheduledAt() == 0);
        }
    }

    /// MIN_WEIGHT_SCHEDULE_DELAY constant pin (INV-CD-1)
    function check_compensation_delay_constant()
        public view
    {
        assert(cd.MIN_WEIGHT_SCHEDULE_DELAY() == 90 days);
    }

    /// EmissionController halving constants pinned at
    /// source level. Mirrors INV-EC-1 + INV-EC-2.
    function check_emission_controller_constants()
        public view
    {
        assert(
            ec.MAINNET_EPOCH_DURATION_SECONDS()
                == 4 * 365 days
        );
        assert(ec.BASE_MAINNET_CHAIN_ID() == 8453);
    }

    /// Structural proof: under chainid 8453 (mainnet), the
    /// constructor REVERTS unless `_epochDurationSeconds`
    /// is exactly MAINNET_EPOCH_DURATION_SECONDS. Halmos
    /// explores symbolic epoch + simulates mainnet chainid.
    ///
    /// We can't set block.chainid symbolically inside
    /// halmos without cheatcodes; instead we prove the
    /// constructor logic structurally by direct call on a
    /// helper.
    function check_emission_controller_mainnet_rejects_off_cadence(
        uint256 attemptedEpoch
    ) public {
        vm_assume(
            attemptedEpoch != ec.MAINNET_EPOCH_DURATION_SECONDS()
        );
        vm_assume(attemptedEpoch >= 1 hours);
        // Set chainid to mainnet via halmos cheatcode.
        // selector: chainId(uint256) = 0x4049ddd2
        (bool ok,) = address(
            uint160(uint256(keccak256("hevm cheat code")))
        ).call(
            abi.encodeWithSelector(bytes4(0x4049ddd2), uint256(8453))
        );
        ok;
        // Now attempt to deploy with the wrong epoch on
        // mainnet — MUST revert.
        try new EmissionControllerBounded(attemptedEpoch) {
            assert(false);
        } catch {
            assert(true);
        }
    }

    /// Halmos cheatcode for symbolic-input pre-condition.
    /// Invokes hevm.assume via the canonical cheat-code
    /// address.
    function vm_assume(bool cond) internal {
        // selector: assume(bool) = 0x4c63e562
        (bool ok,) = address(
            uint160(uint256(keccak256("hevm cheat code")))
        ).call(
            abi.encodeWithSelector(0x4c63e562, cond)
        );
        ok;
    }
}
