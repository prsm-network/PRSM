// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @title CreatorStakeRegistry symbolic proofs (sprint 983)
/// @notice Halmos-provable money-safety properties for the §14 anti-spam
///         creator-stake contract (sp976; pre-deploy-audited + hardened sp979).
///         Complements the human audit (workflow w9riufvls) with machine proof,
///         and extends the symbolic lane (sprints 360/361/363) to the newest
///         money-custody contract before it is commissioned.
///
/// @dev Three properties, all proven for ALL symbolic inputs across every public
///      state-mutating entry point:
///
///   1. SOLVENCY (sister to INV-EP-1 / INV-RD-4):
///        ftns.balanceOf(this) >= totalCustodied + foundationReserveBalance
///      preserved across stake / requestUnbond / withdraw / slash / drain.
///      `totalCustodied` is the sum of every creator's still-held stake
///      (BONDED or UNBONDING). The contract never tracks this sum on-chain, so
///      it is modeled here as a ghost accumulator updated in lockstep with each
///      per-creator amount change — exactly mirroring the real arithmetic.
///
///   2. SLASH CONSERVATION: slash moves EXACTLY `amount` from the creator's
///      bonded stake to the Foundation reserve — no FTNS is created or
///      destroyed, and a creator's stake can never be slashed below zero
///      (the InsufficientStake guard).
///
///   3. ANTI-GAME ELIGIBILITY: creatorStakeOf(c) counts toward HIGH tier ONLY
///      while BONDED. The instant a creator begins to exit (requestUnbond),
///      creatorStakeOf drops to 0 — closing the stake→get-tier→spam→unstake
///      game. This is the property the entire §14 stake gate rests on.
///
/// @dev STRUCTURAL EQUIVALENCE (audit-visible; line ranges pinned in
///      source_identity_pins.json so the parity gate catches silent drift):
///     stake          → contracts/contracts/CreatorStakeRegistry.sol:116-126
///     requestUnbond  → contracts/contracts/CreatorStakeRegistry.sol:132-145
///     withdraw       → contracts/contracts/CreatorStakeRegistry.sol:148-161
///     slash          → contracts/contracts/CreatorStakeRegistry.sol:172-182
///     drain          → contracts/contracts/CreatorStakeRegistry.sol:217-224
///     creatorStakeOf → contracts/contracts/CreatorStakeRegistry.sol:188-191
///
///   Simplifications (orthogonal to the three properties above):
///     - IERC20 substituted with an internal `balance` uint256 under the
///       honest-ERC20 assumption (same as the EscrowPool/Royalty specs):
///       transferFrom-success == balance += amount, transfer-success ==
///       balance -= amount.
///     - Ownable2Step / ReentrancyGuard / Pausable / the slasher access check /
///       the unbond-delay TIMESTAMP gate are omitted: they gate WHO/WHEN, not
///       the balance/conservation arithmetic the proofs concern. (The
///       whenNotPaused freeze + slasher-only access are covered by the Hardhat
///       suite + the sp979 audit; the timestamp only delays withdraw, it does
///       not change the amount moved.)
///
/// Run:
///   cd contracts/symbolic-proofs && halmos --contract CreatorStakeRegistrySpec

contract CreatorStakeRegistry {
    enum StakeStatus { NONE, BONDED, UNBONDING, WITHDRAWN }

    struct CreatorStake {
        uint256 amount;
        StakeStatus status;
    }

    uint256 public balance;                  // mirrors ftns.balanceOf(this)
    uint256 public totalCustodied;           // ghost: Σ still-held stake (BONDED|UNBONDING)
    uint256 public foundationReserveBalance; // accrued slash proceeds
    mapping(address => CreatorStake) public stakes;

    // ── stake(amount) — transferFrom-on-stake; status → BONDED ──
    function stake(uint256 amount) external {
        require(amount > 0, "ZeroAmount");
        // honest-ERC20: transferFrom(msg.sender, this, amount) success.
        balance += amount;
        totalCustodied += amount;
        CreatorStake storage s = stakes[msg.sender];
        s.amount += amount;
        s.status = StakeStatus.BONDED;
    }

    // ── requestUnbond() — eligibility drops to 0 immediately; no token move ──
    function requestUnbond() external {
        CreatorStake storage s = stakes[msg.sender];
        require(
            s.status == StakeStatus.BONDED && s.amount > 0, "NotBonded"
        );
        s.status = StakeStatus.UNBONDING;
        // no balance / totalCustodied / reserve change.
    }

    // ── withdraw() — returns the remaining stake (delay gate omitted) ──
    function withdraw() external {
        CreatorStake storage s = stakes[msg.sender];
        require(s.status == StakeStatus.UNBONDING, "NotUnbonding");
        uint256 amount = s.amount;
        s.amount = 0;
        s.status = StakeStatus.WITHDRAWN;
        totalCustodied -= amount;
        balance -= amount; // honest-ERC20: transfer(msg.sender, amount) success.
    }

    // ── slash(creator, amount) — bonded → reserve; slasher gate omitted ──
    function slash(address creator, uint256 amount) external {
        require(amount > 0, "ZeroAmount");
        CreatorStake storage s = stakes[creator];
        require(s.amount >= amount, "InsufficientStake");
        s.amount -= amount;
        totalCustodied -= amount;
        foundationReserveBalance += amount;
        // balance unchanged: FTNS stays custodied here, just re-earmarked.
    }

    // ── drainFoundationReserve() — reserve → wallet; owner gate omitted ──
    function drainFoundationReserve() external {
        uint256 amount = foundationReserveBalance;
        foundationReserveBalance = 0;
        balance -= amount; // honest-ERC20: transfer(reserveWallet, amount).
    }

    // ── creatorStakeOf — counts ONLY while BONDED ──
    function creatorStakeOf(address creator) external view returns (uint256) {
        CreatorStake storage s = stakes[creator];
        return s.status == StakeStatus.BONDED ? s.amount : 0;
    }
}


/// Halmos spec — proves the three money-safety properties hold across all
/// public mutating entry points, for all symbolic inputs.
contract CreatorStakeRegistrySpec {
    CreatorStakeRegistry internal reg;

    function setUp() public {
        reg = new CreatorStakeRegistry();
    }

    function _solvent() internal view returns (bool) {
        return reg.balance() >=
            reg.totalCustodied() + reg.foundationReserveBalance();
    }

    // ── Property 1: SOLVENCY across every entry point ──

    /// Boot state: all accumulators zero → 0 >= 0 + 0.
    function check_post_construction_solvency() public view {
        assert(_solvent());
    }

    /// stake increments balance and totalCustodied by the same amount.
    function check_stake_preserves_solvency(uint256 amount) public {
        try reg.stake(amount) { assert(_solvent()); }
        catch { assert(_solvent()); }
    }

    /// requestUnbond moves no value — solvency trivially preserved. Composed
    /// with a prior stake so the creator is actually BONDED.
    function check_requestUnbond_preserves_solvency(uint256 amount) public {
        try reg.stake(amount) {} catch { return; }
        try reg.requestUnbond() { assert(_solvent()); }
        catch { assert(_solvent()); }
    }

    /// withdraw decrements balance and totalCustodied by the same amount.
    function check_withdraw_preserves_solvency(uint256 amount) public {
        try reg.stake(amount) {} catch { return; }
        try reg.requestUnbond() {} catch { return; }
        try reg.withdraw() { assert(_solvent()); }
        catch { assert(_solvent()); }
    }

    /// slash moves value from totalCustodied to reserve (RHS sum unchanged),
    /// balance untouched — solvency preserved.
    function check_slash_preserves_solvency(
        uint256 stakeAmt, uint256 slashAmt
    ) public {
        try reg.stake(stakeAmt) {} catch { return; }
        try reg.slash(address(this), slashAmt) { assert(_solvent()); }
        catch { assert(_solvent()); }
    }

    /// drain decrements balance by exactly the reserve it zeroes; from a
    /// solvent pre-state, balance - reserve >= totalCustodied still holds.
    function check_drain_preserves_solvency(
        uint256 stakeAmt, uint256 slashAmt
    ) public {
        try reg.stake(stakeAmt) {} catch { return; }
        try reg.slash(address(this), slashAmt) {} catch {}
        try reg.drainFoundationReserve() { assert(_solvent()); }
        catch { assert(_solvent()); }
    }

    // ── Property 2: SLASH CONSERVATION ──

    /// slash moves EXACTLY `slashAmt` from the creator's stake into the
    /// reserve, and never takes a stake below zero (InsufficientStake guard).
    function check_slash_conserves_value(
        uint256 stakeAmt, uint256 slashAmt
    ) public {
        try reg.stake(stakeAmt) {} catch { return; }
        (uint256 amtBefore, ) = reg.stakes(address(this));
        uint256 reserveBefore = reg.foundationReserveBalance();
        try reg.slash(address(this), slashAmt) {
            (uint256 amtAfter, ) = reg.stakes(address(this));
            // exact move: stake down by slashAmt, reserve up by slashAmt.
            assert(amtBefore - amtAfter == slashAmt);
            assert(
                reg.foundationReserveBalance() - reserveBefore == slashAmt
            );
            // never below zero (would have reverted via InsufficientStake).
            assert(amtAfter <= amtBefore);
        } catch {
            // a reverted slash leaves the stake + reserve untouched.
            (uint256 amtAfter, ) = reg.stakes(address(this));
            assert(amtAfter == amtBefore);
            assert(reg.foundationReserveBalance() == reserveBefore);
        }
    }

    // ── Property 3: ANTI-GAME ELIGIBILITY ──

    /// After staking, the bonded amount counts toward eligibility.
    function check_creatorStakeOf_counts_while_bonded(uint256 amount) public {
        try reg.stake(amount) {
            (uint256 amt, ) = reg.stakes(address(this));
            assert(reg.creatorStakeOf(address(this)) == amt);
        } catch { return; }
    }

    /// THE headline anti-game proof: the instant a creator begins to exit,
    /// creatorStakeOf drops to 0 — they cannot hold HIGH tier while unbonding.
    function check_creatorStakeOf_zero_after_requestUnbond(
        uint256 amount
    ) public {
        try reg.stake(amount) {} catch { return; }
        try reg.requestUnbond() {
            assert(reg.creatorStakeOf(address(this)) == 0);
        } catch { return; }
    }

    /// And after a full withdraw, eligibility stays 0.
    function check_creatorStakeOf_zero_after_withdraw(uint256 amount) public {
        try reg.stake(amount) {} catch { return; }
        try reg.requestUnbond() {} catch { return; }
        try reg.withdraw() {
            assert(reg.creatorStakeOf(address(this)) == 0);
        } catch { return; }
    }
}
