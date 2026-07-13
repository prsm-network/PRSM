// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @dev Minimal read-only view of the slasher (BatchSettlementRegistry)
 *      that StakeBond consults to compute the unbond delay floor. Per
 *      L2 audit HIGH-2 (A-02/D-01): if `unbondDelay < challengeWindow`,
 *      a provider can `commitBatch` + `requestUnbond` in the same block,
 *      `withdraw` after MIN unbondDelay (1 day), and a successful later
 *      challenge silently no-ops because state is WITHDRAWN. Reading the
 *      challenge window from the slasher and clamping
 *      `unbond_eligible_at` to AT LEAST `now + challengeWindowSeconds`
 *      closes the race regardless of governance parameter pairings.
 */
interface ISlasherWithChallengeWindow {
    function challengeWindowSeconds() external view returns (uint256);
}

/**
 * @dev L4 self-audit HIGH-1 (A-01 ≡ D-01) fix interface: per-provider
 *      maximum challenge-window expiry across PENDING batches. The
 *      LIVE `challengeWindowSeconds` (above) is insufficient as an
 *      unbond floor because the L2 D-05 fix per-batch snapshots the
 *      window at commit time; a subsequent governance reduction would
 *      let a provider unbond before any high-window batch's pinned
 *      window elapses, re-opening the original A-02 slash-evasion
 *      race. This interface exposes the BSR-side tracker so StakeBond
 *      can clamp against the LONGEST-PINNED PENDING expiry.
 *
 * Stale values (where `lastPendingBatchExpiry(provider) < block.timestamp`)
 * are naturally ignored by StakeBond's max-of-floors logic: a past
 * timestamp is dominated by the local floor `now + unbondDelay`.
 */
interface ISlasherWithProviderExpiry {
    function lastPendingBatchExpiry(address provider) external view returns (uint64);
}

/**
 * @dev L4 self-audit re-run A-06 fix interface: pause-extended unbond
 *      floor. The HIGH-1 `lastPendingBatchExpiry` tracker is wall-clock
 *      (`commitTimestamp + windowAtCommit`); the HIGH-2
 *      `_effectiveElapsed` math makes the actual challenge expiry
 *      pause-extended (wall-clock + pausedSinceCommit). Unbond floor
 *      diverges from effective challenge expiry by `pausedSinceCommit`
 *      under the canonical A-01 trigger condition (governance
 *      `setChallengeWindowSeconds` reduction).
 *
 * Closes the gap by exposing both:
 *   - `lastPendingBatchPausedAtAccrual(provider)` — `totalPausedSeconds`
 *     value at the moment the per-provider expiry was last updated.
 *   - `totalPausedSeconds()` — current cumulative paused time.
 *
 * The reader computes `pauseAdjustedExpiry =
 *   lastPendingBatchExpiry(p) + (totalPausedSeconds() - lastPendingBatchPausedAtAccrual(p))`
 * and uses it as the slasher floor in `requestUnbond`.
 */
interface ISlasherWithProviderExpiryAndPause {
    function lastPendingBatchExpiry(address provider) external view returns (uint64);
    function lastPendingBatchPausedAtAccrual(address provider) external view returns (uint64);
    function totalPausedSeconds() external view returns (uint256);
}

/**
 * @title StakeBond
 * @notice Per-provider FTNS collateral for Phase 7 Tier-C verification.
 *
 * Providers post FTNS stake to back their claimed tier. On-chain
 * balance determines effective tier; successful Phase 3.1 DOUBLE_SPEND
 * or INVALID_SIGNATURE challenges automatically slash the stake.
 *
 * Scope (Task 1): bond / requestUnbond / withdraw lifecycle + read-only
 * queries. Slashing + bounty accrual are Task 2. Registry wiring is
 * Task 3.
 *
 * Lifecycle:
 *   [none] →  bond(amount, tierSlashRateBps)  →  BONDED
 *   BONDED →  requestUnbond()                  →  UNBONDING
 *   UNBONDING (after unbond_delay) → withdraw() → WITHDRAWN
 *   BONDED OR UNBONDING → slash() (Task 2) → stake reduced
 *
 * Unbonding delay (default 7 days) prevents flash-stake attacks where
 * a provider bonds right before a dispatch, collects payment, then
 * unbonds and disappears before a challenge can fire.
 *
 * Slashing during UNBONDING is permitted: a provider who initiated
 * unbonding is still accountable for misbehavior caught within the
 * delay window. Prevents the challenge-then-unbond race escape.
 *
 * Slasher authorization: L2 audit HIGH-7 (B-CROSS-3) fix — the slasher
 * address is now IMMUTABLE, set once at construction. In production it
 * is the BatchSettlementRegistry's address. L4 self-audit MED-6 (D-02)
 * additionally hard-rejects address(0) in the constructor: rotation is
 * impossible after deploy, so the production setup must wire the BSR
 * address at construction time. Closes the post-handoff drain vector
 * where a compromised owner could re-point slasher to an attacker EOA
 * and call slash() against arbitrary providers (50%-100% bond capture
 * + 70% bounty net to the attacker).
 *
 * L4 self-audit INFO-4 (D-05) — cross-contract pause coordination is
 * IMPLICIT, not enforced. StakeBond and BatchSettlementRegistry each
 * carry independent OpenZeppelin Pausable state; the operational
 * invariant the protocol relies on is:
 *
 *   "Pause both contracts together, in either order, before triaging
 *    an exploit. Unpause both contracts together once safe to resume."
 *
 * The contracts do NOT enforce this via cross-calls. The relevant
 * interaction surfaces:
 *   - BSR.challengeReceipt → StakeBond.slash (try/catch). If StakeBond
 *     is paused but BSR is not, the slash silently reverts and BSR's
 *     `SlashSwallowed` event fires. This is observable by Forta but is
 *     a degraded state for the duration of the asymmetric pause.
 *   - StakeBond.requestUnbond → BSR.lastPendingBatchExpiry (view).
 *     Read-only, not affected by either pause.
 *
 * Operational defense: PRSM-EXPLOIT-PLAYBOOK §11 lists the contract-pair
 * pause sequence; the multisig signing UI presents both pause txs as a
 * batched bundle to reduce the asymmetric-pause window. Future work
 * could move this guarantee on-chain via a shared `PauseCoordinator`,
 * but the operational defense is currently sufficient given the small
 * Foundation Safe + 14-day public review window.
 */
contract StakeBond is Ownable2Step, ReentrancyGuard, Pausable {
    enum StakeStatus {
        NONE,        // never bonded (or fully withdrawn + re-bonded happens via new Stake)
        BONDED,      // active stake backing a tier
        UNBONDING,   // requestUnbond fired; waiting for delay to elapse
        WITHDRAWN    // funds returned to provider
    }

    struct Stake {
        uint128 amount;               // FTNS wei currently locked
        uint64 bonded_at_unix;
        uint64 unbond_eligible_at;    // 0 while BONDED; set when requestUnbond fires
        StakeStatus status;
        uint16 tier_slash_rate_bps;   // snapshot at bond time; survives slashes
    }

    IERC20 public immutable ftns;

    /// @dev Seconds a provider must wait between requestUnbond() and
    /// withdraw(). Governance-adjustable within sane bounds.
    uint256 public unbondDelaySeconds;
    uint256 public constant MIN_UNBOND_DELAY_SECONDS = 1 days;
    uint256 public constant MAX_UNBOND_DELAY_SECONDS = 30 days;

    /// @dev Authorized slasher address. L2 audit HIGH-7 (B-CROSS-3) fix:
    /// IMMUTABLE — set once at construction, no setter. address(0) =
    /// slashing intentionally disabled (test/dev only); production
    /// MUST set this to the BatchSettlementRegistry.
    address public immutable slasher;

    /// @dev sp1456 (slashing audit #2) — a SECOND authorized slasher for the STORAGE-fault path
    /// (StorageSlashing). `slasher` (the BatchSettlementRegistry) is immutable per HIGH-7, so
    /// StorageSlashing calling slash() as itself reverts CallerNotSlasher — the ENTIRE storage-fault
    /// slashing class is dead. This is SET ONCE by the owner (from address(0), then never re-pointable),
    /// which preserves the HIGH-7 anti-repoint intent while allowing post-deploy wiring (StakeBond and
    /// StorageSlashing reference each other, so one can't be constructor-immutable in the other).
    address public storageSlasher;

    event StorageSlasherSet(address indexed storageSlasher);

    /// @dev Per-provider stake record. Re-bonding after a full withdraw
    /// overwrites the prior record.
    mapping(address provider => Stake) public stakes;

    /// @dev Per-challenger claimable bounty balance. Accrues 70% of
    /// slashed FTNS when the challenger is NOT the slashed provider.
    /// Claimed via claimBounty(); zeroed on claim.
    mapping(address challenger => uint256) public slashedBountyPayable;

    /// @dev Accumulated Foundation-reserve balance from slashing
    /// (30% normal case; 100% on self-slash / anonymous challenger).
    /// Drained via drainFoundationReserve() by owner to the
    /// foundationReserveWallet address.
    uint256 public foundationReserveBalance;

    /// @dev Destination for drainFoundationReserve. Owner-set; must
    /// be non-zero before drain can succeed. Typically the Foundation's
    /// 2-of-3 multi-sig treasury wallet.
    address public foundationReserveWallet;

    /// @dev Bounty split in basis points. 7000 = 70% of slash amount
    /// goes to challenger; remainder (3000 = 30%) to Foundation reserve.
    /// When the challenger equals the provider or is address(0), the
    /// entire slash goes to the Foundation reserve (prevents self-slash
    /// schemes that would net-profit by capturing the challenger bounty).
    uint16 public constant CHALLENGER_BOUNTY_BPS = 7000;

    event Bonded(
        address indexed provider,
        uint128 amount,
        uint16 tierSlashRateBps,
        uint64 bondedAt
    );
    event UnbondRequested(
        address indexed provider,
        uint64 eligibleAt
    );
    event Withdrawn(
        address indexed provider,
        uint128 amount
    );
    event UnbondDelayUpdated(uint256 oldDelay, uint256 newDelay);
    event FoundationReserveWalletUpdated(address oldWallet, address newWallet);
    event Slashed(
        address indexed provider,
        address indexed challenger,
        bytes32 indexed reasonId,
        uint256 slashAmount,
        uint256 challengerBounty,
        uint256 foundationShare
    );
    event BountyClaimed(address indexed challenger, uint256 amount);
    event FoundationReserveDrained(address indexed recipient, uint256 amount);

    error ZeroAmount();
    error ZeroAddress();
    error InvalidUnbondDelay(uint256 provided);
    error AlreadyBonded(address provider, StakeStatus status);
    error NotBonded(address provider, StakeStatus status);
    error NotUnbonding(address provider, StakeStatus status);
    error UnbondDelayNotElapsed(uint64 eligibleAt);
    error InvalidSlashRateBps(uint16 provided);
    /// @dev sp1456 — bond() rejects a slash rate below the tier-derived floor (see minSlashRateForAmount).
    error SlashRateBelowTierFloor(uint16 provided, uint16 floor);
    /// @dev sp1456 — the storage-fault slasher may be wired exactly once (never re-pointable, per HIGH-7).
    error StorageSlasherAlreadySet(address current);
    error TransferFailed();
    error CallerNotSlasher(address caller, address expectedSlasher);
    error NotSlashable(address provider, StakeStatus status);
    error NothingToSlash(address provider);
    error NothingToClaim(address challenger);
    error FoundationReserveEmpty();
    error FoundationReserveWalletNotSet();
    /// @dev L4 self-audit MED-4 (B-03) fix: foundation reserve wallet must be a contract (typically Foundation Safe).
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
        // L4 self-audit MED-6 (D-02) fix: reject address(0) for the
        // immutable slasher. Same rationale as EscrowPool: post-HIGH-7
        // the slasher is immutable, so a misconfigured deploy
        // permanently bricks slash() with no setter recovery. Hard
        // reject here so the misconfiguration is loud at deploy time.
        // Tests that previously relied on address(0) deploys must now
        // wire a stub slasher (any non-zero address; the slash gate
        // ignores EOA slashers via slasher.code.length > 0 check).
        if (initialSlasher == address(0)) revert ZeroAddress();
        ftns = IERC20(ftnsAddress);
        unbondDelaySeconds = initialUnbondDelay;
        slasher = initialSlasher;
    }

    // ── Provider lifecycle ────────────────────────────────────────

    /**
     * @notice Post FTNS stake. Caller must have approved this contract
     *         for at least `amount` of FTNS. The `tierSlashRateBps`
     *         snapshot is stored at bond time and used on subsequent
     *         slashes — so a provider cannot dodge slashing by
     *         downgrading their claimed tier mid-flight.
     * @param amount FTNS in base units (wei)
     * @param tierSlashRateBps basis-points slash rate (0-10000). In
     *        Phase 7 tier mapping: standard=5000 (50%), premium=
     *        critical=10000 (100%). 0 = no-slash tier (== "open"
     *        tier, typically wouldn't bond).
     */
    function bond(uint128 amount, uint16 tierSlashRateBps) external nonReentrant whenNotPaused {
        if (amount == 0) revert ZeroAmount();
        if (tierSlashRateBps > 10000) revert InvalidSlashRateBps(tierSlashRateBps);
        // sp1456 (slashing audit wu7qhwsz3 #1) — bind the slash rate to a tier-derived FLOOR. Without
        // this a provider bonds a large amount at rate 0 (or a tiny nonzero rate), qualifies for a high
        // stake TIER (effectiveTier grades on AMOUNT alone), clears the marketplace/orchestrator stake
        // gates, yet on a proven DOUBLE_SPEND/INVALID_SIGNATURE slash() computes amount*0/10000 = 0 →
        // NothingToSlash → the whole 50-100% forfeit is nullified. Flooring the rate at the tier's
        // required exposure makes tier eligibility and slash exposure inseparable. Governance-tunable
        // policy (see _minSlashRateForAmount).
        uint16 minRate = _minSlashRateForAmount(amount);
        if (tierSlashRateBps < minRate) {
            revert SlashRateBelowTierFloor(tierSlashRateBps, minRate);
        }

        Stake storage s = stakes[msg.sender];
        if (s.status == StakeStatus.BONDED || s.status == StakeStatus.UNBONDING) {
            revert AlreadyBonded(msg.sender, s.status);
        }

        // Effects before interaction.
        stakes[msg.sender] = Stake({
            amount: amount,
            bonded_at_unix: uint64(block.timestamp),
            unbond_eligible_at: 0,
            status: StakeStatus.BONDED,
            tier_slash_rate_bps: tierSlashRateBps
        });

        bool ok = ftns.transferFrom(msg.sender, address(this), amount);
        if (!ok) revert TransferFailed();

        emit Bonded(
            msg.sender, amount, tierSlashRateBps,
            uint64(block.timestamp)
        );
    }

    /**
     * @notice Begin the unbonding process. Transitions BONDED →
     *         UNBONDING and starts the delay timer. During UNBONDING
     *         the provider's effectiveTier drops to "open" (view-
     *         layer concern, see effectiveTier).
     *         Slashing remains active during UNBONDING — a provider
     *         caught in a challenge after requesting unbond still
     *         forfeits stake.
     */
    function requestUnbond() external whenNotPaused {
        Stake storage s = stakes[msg.sender];
        if (s.status != StakeStatus.BONDED) {
            revert NotBonded(msg.sender, s.status);
        }
        s.status = StakeStatus.UNBONDING;

        // L2 audit HIGH-2 fix: unbond_eligible_at is the MAX of (the local
        // unbondDelay floor) AND (the slasher's current challenge window).
        // Closes the slash-evasion race when governance has set
        // unbondDelaySeconds < challengeWindowSeconds.
        //
        // The slasher is the BSR in production. We gate the call on
        // `slasher.code.length > 0` because a Solidity try/catch does NOT
        // catch the "returned unexpected amount of data" decode error that
        // fires when an external call lands on an EOA or non-conforming
        // contract. If slasher is 0, an EOA, or a contract without
        // challengeWindowSeconds(), we fall back to the local floor.
        uint256 localFloor = block.timestamp + unbondDelaySeconds;
        uint256 effectiveEligibleAt = localFloor;
        if (slasher != address(0) && slasher.code.length > 0) {
            try ISlasherWithChallengeWindow(slasher).challengeWindowSeconds() returns (uint256 windowSecs) {
                uint256 slasherFloor = block.timestamp + windowSecs;
                if (slasherFloor > effectiveEligibleAt) {
                    effectiveEligibleAt = slasherFloor;
                }
            } catch {
                // slasher contract doesn't expose challengeWindowSeconds() —
                // keep local floor. Operationally this branch only fires if
                // a governance change re-points slasher to a non-BSR target.
            }

            // L4 self-audit HIGH-1 (A-01 ≡ D-01) fix: clamp against the
            // BSR-side `lastPendingBatchExpiry` tracker. This reads the
            // MAXIMUM expiry across this provider's currently-PENDING
            // batches, which is the correct floor regardless of any
            // post-commit `setChallengeWindowSeconds` reduction.
            // Without this, the L2 HIGH-2 + D-05 fixes compose
            // incorrectly: HIGH-2 reads the LIVE window (small after
            // reduction); D-05 keeps each PENDING batch's PINNED window
            // (still large). A provider could unbond + withdraw before
            // their pinned window elapsed, then a successful challenge
            // hits WITHDRAWN status and the slash silently swallows.
            // Clamping against the per-provider max-pending-expiry
            // closes that composition gap.
            //
            // Stale values (past timestamps from already-finalized
            // batches) are naturally dominated by `localFloor` (which
            // is always strictly in the future). No explicit
            // stale-check needed.
            //
            // L4 self-audit re-run A-06 fix: pause-extend the per-provider
            // expiry. The wall-clock `lastPendingBatchExpiry` diverges
            // from the pause-extended actual challenge expiry computed by
            // BSR's `_effectiveElapsed` whenever ANY pause occurred
            // during the batch's window. We mirror that arithmetic by
            // adding `(totalPausedSeconds - lastPendingBatchPausedAtAccrual)`
            // to the wall-clock expiry. Without this, a provider could
            // unbond at the wall-clock boundary and withdraw `unbondDelay`
            // later, while a successful challenge in the
            // `pausedSinceCommit`-shaped tail of the effective challenge
            // window would hit WITHDRAWN → SlashSwallowed.
            // sp1456 — refactored into _pendingExpiryFloor() so withdraw() can re-apply the SAME
            // pause-adjusted per-provider expiry floor at withdraw time, closing the
            // commit-after-requestUnbond evasion (a batch committed AFTER this one-shot snapshot).
            uint256 pendingFloor = _pendingExpiryFloor(msg.sender);
            if (pendingFloor > effectiveEligibleAt) {
                effectiveEligibleAt = pendingFloor;
            }
        }
        s.unbond_eligible_at = uint64(effectiveEligibleAt);
        emit UnbondRequested(msg.sender, s.unbond_eligible_at);
    }

    /// @dev sp1456 — the pause-adjusted MAXIMUM challenge expiry across `provider`'s currently-PENDING
    /// batches, read LIVE from the slasher. Mirrors BSR `_effectiveElapsed` pause arithmetic
    /// (A-01/A-06). Returns 0 when the slasher can't be read or exposes no such tracker — a stale/past
    /// value is naturally dominated by the caller's other floors and never wrongly blocks. Called by
    /// requestUnbond (to SET unbond_eligible_at) AND withdraw (to RE-CLAMP against batches committed
    /// after the requestUnbond snapshot).
    function _pendingExpiryFloor(address provider) internal view returns (uint256) {
        if (slasher == address(0) || slasher.code.length == 0) return 0;
        try ISlasherWithProviderExpiryAndPause(slasher).lastPendingBatchExpiry(provider) returns (uint64 maxExpiry) {
            uint256 pauseAdjusted = uint256(maxExpiry);
            try ISlasherWithProviderExpiryAndPause(slasher).lastPendingBatchPausedAtAccrual(provider) returns (uint64 pausedAtAccrual) {
                try ISlasherWithProviderExpiryAndPause(slasher).totalPausedSeconds() returns (uint256 currentPaused) {
                    if (currentPaused > uint256(pausedAtAccrual)) {
                        pauseAdjusted += currentPaused - uint256(pausedAtAccrual);
                    }
                } catch {}  // partial-fix BSR without totalPausedSeconds — no-pause case still correct
            } catch {}  // pre-A-06 BSR without lastPendingBatchPausedAtAccrual — no-pause case correct
            return pauseAdjusted;
        } catch {
            return 0;  // slasher exposes no lastPendingBatchExpiry — no per-provider floor available
        }
    }

    /**
     * @notice After the unbonding delay has elapsed, withdraw the
     *         remaining staked FTNS to the provider's wallet. If the
     *         stake was partially slashed during UNBONDING, the
     *         current `amount` is what gets returned.
     */
    function withdraw() external nonReentrant whenNotPaused {
        Stake storage s = stakes[msg.sender];
        if (s.status != StakeStatus.UNBONDING) {
            revert NotUnbonding(msg.sender, s.status);
        }
        if (block.timestamp < s.unbond_eligible_at) {
            revert UnbondDelayNotElapsed(s.unbond_eligible_at);
        }
        // sp1456 (slashing audit #3) — RE-CLAMP against the LIVE per-provider pending-expiry floor.
        // unbond_eligible_at is a one-shot snapshot taken at requestUnbond; a batch committed AFTER it
        // (BSR._commitBatch has no BONDED gate) raises the slasher's lastPendingBatchExpiry but never
        // re-clamps this snapshot. Withdrawing before that later challenge window closes would let the
        // batch be challenged post-WITHDRAWN and swallow the 50-100% slash. Block withdraw until every
        // currently-pending batch's (pause-adjusted) window has elapsed.
        uint256 pendingFloor = _pendingExpiryFloor(msg.sender);
        if (block.timestamp < pendingFloor) {
            revert UnbondDelayNotElapsed(uint64(pendingFloor));
        }

        uint128 payout = s.amount;
        // Effects before interaction.
        s.amount = 0;
        s.status = StakeStatus.WITHDRAWN;

        if (payout > 0) {
            bool ok = ftns.transfer(msg.sender, payout);
            if (!ok) revert TransferFailed();
        }

        emit Withdrawn(msg.sender, payout);
    }

    // ── Views ─────────────────────────────────────────────────────

    /// @notice Return the full Stake record for a provider.
    function stakeOf(address provider) external view returns (Stake memory) {
        return stakes[provider];
    }

    /// @notice Return the provider's currently-effective tier as a
    ///         short string. Consumed by the marketplace orchestrator
    ///         when enforcing DispatchPolicy.min_stake_tier.
    ///         Returns "open" during UNBONDING or WITHDRAWN states
    ///         even if the on-chain amount is high — a provider
    ///         who requested unbond is not reliably backing a tier.
    function effectiveTier(address provider) external view returns (string memory) {
        Stake storage s = stakes[provider];
        if (s.status != StakeStatus.BONDED) return "open";
        uint128 amt = s.amount;
        // Thresholds per Phase 7 design §3.2.
        if (amt >= 50_000 * 1e18) return "critical";
        if (amt >= 25_000 * 1e18) return "premium";
        if (amt >= 5_000 * 1e18) return "standard";
        return "open";
    }

    /// @notice sp1456 — the MINIMUM slash rate (bps) a bond of `amount` must post, mirroring the
    ///         effectiveTier thresholds so tier eligibility and slash exposure cannot diverge (the
    ///         rate-0-at-critical-tier evasion). GOVERNANCE-TUNABLE POLICY: >=50k FTNS (critical) must
    ///         post 100% (10000bps); >=5k (standard/premium) must post >=50% (5000bps); <5k (open tier,
    ///         which clears no stake gate and earns no tier privileges) has no floor. A future version
    ///         may make these values a governance-settable table; hard-coded here per the approved policy.
    function minSlashRateForAmount(uint128 amount) public pure returns (uint16) {
        return _minSlashRateForAmount(amount);
    }

    function _minSlashRateForAmount(uint128 amount) internal pure returns (uint16) {
        if (amount >= 50_000 * 1e18) return 10000;  // critical tier → full 100% forfeit
        if (amount >= 5_000 * 1e18) return 5000;    // standard/premium → 50% floor
        return 0;                                   // open tier → untrusted, no tier, no floor
    }

    // ── Governance surface ────────────────────────────────────────

    /**
     * @notice Update the unbonding delay. Owner-only. Does NOT retro-
     *         actively affect already-UNBONDING stakes (they retain
     *         their `unbond_eligible_at` from requestUnbond time).
     *         Per PRSM-GOV-1 §10.3, governance changes to unbond
     *         delay are subject to the 14-day advance-notice period.
     */
    function setUnbondDelay(uint256 newDelay) external onlyOwner {
        if (newDelay < MIN_UNBOND_DELAY_SECONDS ||
            newDelay > MAX_UNBOND_DELAY_SECONDS) {
            revert InvalidUnbondDelay(newDelay);
        }
        uint256 old = unbondDelaySeconds;
        unbondDelaySeconds = newDelay;
        emit UnbondDelayUpdated(old, newDelay);
    }

    /**
     * @notice sp1456 — wire the storage-fault slasher (StorageSlashing) EXACTLY ONCE. Owner-only, and
     *         only permitted while `storageSlasher == address(0)`, so it can never be re-pointed to an
     *         attacker (the same anti-repoint guarantee HIGH-7 gave `slasher` by making it immutable).
     *         Needed because StakeBond and StorageSlashing reference each other and cannot both be
     *         constructor-immutable; without it every storage-fault slash reverts CallerNotSlasher.
     */
    function setStorageSlasherOnce(address newStorageSlasher) external onlyOwner {
        if (storageSlasher != address(0)) revert StorageSlasherAlreadySet(storageSlasher);
        if (newStorageSlasher == address(0)) revert ZeroAddress();
        storageSlasher = newStorageSlasher;
        emit StorageSlasherSet(newStorageSlasher);
    }

    // L2 audit HIGH-7 (B-CROSS-3) fix: setSlasher was removed. The
    // slasher field is now immutable — set in the constructor.
    // Rotation requires re-deployment of StakeBond; this is the
    // intended trade-off documented at the field declaration above.

    /**
     * @notice Set the destination wallet for drainFoundationReserve.
     *         Owner-only. Must be non-zero before drain can succeed.
     *
     * @dev L4 self-audit MED-4 (B-03) fix: reject address(0) explicitly
     *      and require the destination to be a contract (typically the
     *      Foundation Safe). Pre-fix, an operator typo or compromised
     *      owner could route the entire reserve stream to an arbitrary
     *      EOA on the next drain. Post-fix, only contract destinations
     *      are accepted; combined with the deploy-script canonical-Safe
     *      pin (out-of-contract concern), a misconfigured destination
     *      can't silently absorb slashes. Pre-existing operators who
     *      previously set the wallet to address(0) to "disable" drain
     *      can still do so via the existing FoundationReserveWalletNotSet
     *      revert path in drainFoundationReserve — this fix makes the
     *      disable-by-zero path explicit rather than implicit.
     *
     * @dev L4 self-audit LOW-3 (D-04) fix: gated by `whenNotPaused`.
     *      Pre-fix, a compromised owner could (during pause)
     *      setFoundationReserveWallet(attacker) + drainFoundationReserve
     *      to extract the reserve while users believed pause was
     *      protecting state. Post-fix, drain-destination changes
     *      respect the global pause invariant.
     */
    function setFoundationReserveWallet(address newWallet) external onlyOwner whenNotPaused {
        if (newWallet == address(0)) revert ZeroAddress();
        if (newWallet.code.length == 0) revert WalletNotContract(newWallet);
        address old = foundationReserveWallet;
        foundationReserveWallet = newWallet;
        emit FoundationReserveWalletUpdated(old, newWallet);
    }

    // ── Pause control (L2 audit HIGH-3 / D-02) ────────────────────

    /**
     * @notice Pause bond / requestUnbond / withdraw / slash / claimBounty.
     *         Owner-only; intended for incident response per
     *         docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md. Admin setters
     *         (setUnbondDelay, setFoundationReserveWallet) and
     *         drainFoundationReserve remain accessible while paused so
     *         the owner can perform emergency rotation. Note: slasher
     *         is immutable post-HIGH-7 — bond re-deploy required for
     *         slasher rotation.
     *
     *         Unbond-timer-during-pause semantics: timestamp-based,
     *         clock CONTINUES during pause (no reset). When unpaused,
     *         providers whose unbond_eligible_at has elapsed can
     *         withdraw immediately. A 30-day pause does NOT extend the
     *         unbond delay; it freezes the action of withdrawal.
     */
    function pause() external onlyOwner {
        _pause();
    }

    /// @notice Resume normal operations after pause.
    function unpause() external onlyOwner {
        _unpause();
    }

    // ── Slashing (Task 2) ─────────────────────────────────────────

    /**
     * @notice Slash a provider's stake at their bonded slash rate.
     *         Slasher-only (the authorized BatchSettlementRegistry).
     *
     *         Amount slashed = current stake × tier_slash_rate_bps / 10000.
     *         Split:
     *           - If challenger ∈ {provider, address(0)}: 100% → Foundation
     *             reserve. Prevents self-slash schemes where a sophisticated
     *             adversary operates both the provider and the challenger
     *             to net-profit from the 70% bounty.
     *           - Otherwise: 70% → challenger's claimable bounty balance,
     *             30% → Foundation reserve.
     *
     *         Slashing is permitted in BOTH BonDED and UNBONDING states —
     *         closes the challenge-then-unbond race escape.
     *
     * @param provider  provider address whose stake is slashed
     * @param challenger address that submitted the successful challenge
     * @param reasonId   opaque identifier (typically the batchId that
     *                   contained the invalid receipt) for audit trail
     */
    function slash(
        address provider,
        address challenger,
        bytes32 reasonId
    ) external nonReentrant whenNotPaused {
        // sp1456 — authorize BOTH the immutable settlement slasher (BSR) AND the set-once storage
        // slasher (StorageSlashing). storageSlasher==address(0) (unwired) leaves only `slasher`
        // authorized (a real caller can never be address(0)), so this cannot widen access before it is
        // deliberately wired once.
        if (msg.sender != slasher && msg.sender != storageSlasher) {
            revert CallerNotSlasher(msg.sender, slasher);
        }
        Stake storage s = stakes[provider];
        if (s.status != StakeStatus.BONDED && s.status != StakeStatus.UNBONDING) {
            revert NotSlashable(provider, s.status);
        }
        if (s.amount == 0) revert NothingToSlash(provider);

        // slashAmount = amount * tier_slash_rate_bps / 10000, capped at amount.
        uint256 slashAmount = (uint256(s.amount) * uint256(s.tier_slash_rate_bps)) / 10000;
        if (slashAmount == 0) revert NothingToSlash(provider);
        if (slashAmount > s.amount) slashAmount = s.amount;  // defensive

        // Effects before interactions.
        s.amount -= uint128(slashAmount);

        // Split + credit to claimable pools. No ERC-20 transfer here —
        // FTNS stays in this contract until claimed via claimBounty /
        // drainFoundationReserve. Keeps the slash operation lightweight
        // and decouples the challenger's claim timing.
        uint256 challengerShare;
        uint256 foundationShare;
        if (challenger == provider || challenger == address(0)) {
            challengerShare = 0;
            foundationShare = slashAmount;
        } else {
            challengerShare = (slashAmount * uint256(CHALLENGER_BOUNTY_BPS)) / 10000;
            foundationShare = slashAmount - challengerShare;
            slashedBountyPayable[challenger] += challengerShare;
        }
        foundationReserveBalance += foundationShare;

        emit Slashed(
            provider, challenger, reasonId,
            slashAmount, challengerShare, foundationShare
        );
    }

    /**
     * @notice Claim accumulated bounty from successful challenges.
     *         Permissionless for the caller's own balance. Idempotent —
     *         a zero-balance claim reverts rather than emitting a
     *         noise event.
     */
    function claimBounty() external nonReentrant whenNotPaused {
        uint256 amount = slashedBountyPayable[msg.sender];
        if (amount == 0) revert NothingToClaim(msg.sender);

        // Effects before interaction.
        slashedBountyPayable[msg.sender] = 0;

        bool ok = ftns.transfer(msg.sender, amount);
        if (!ok) revert TransferFailed();

        emit BountyClaimed(msg.sender, amount);
    }

    /**
     * @notice Move the accumulated Foundation-reserve balance to the
     *         configured foundationReserveWallet. Owner-only. Reverts
     *         if the wallet is unset or the reserve is empty.
     */
    function drainFoundationReserve() external onlyOwner nonReentrant whenNotPaused {
        // L4 self-audit LOW-3 (D-04) fix: gated by `whenNotPaused`. Pre-fix,
        // a compromised owner could drain the reserve while users
        // believed pause was protecting state, violating Stated
        // Invariant #3 ("when paused, no value can move in any
        // direction"). To drain during a real incident, owner must
        // unpause first — which itself creates an on-chain breadcrumb.
        if (foundationReserveWallet == address(0)) {
            revert FoundationReserveWalletNotSet();
        }
        uint256 amount = foundationReserveBalance;
        if (amount == 0) revert FoundationReserveEmpty();

        // Effects before interaction.
        foundationReserveBalance = 0;

        bool ok = ftns.transfer(foundationReserveWallet, amount);
        if (!ok) revert TransferFailed();

        emit FoundationReserveDrained(foundationReserveWallet, amount);
    }
}
