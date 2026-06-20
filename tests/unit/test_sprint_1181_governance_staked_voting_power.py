"""Sprint 1181 — governance voting power routed through the STAKE (Option 1).

The governance sweep confirmed a MAJOR Sybil-vote-inflation exploit: voting
power read the voter's LIVE, transferable FTNS balance, with per-voter_id-only
dedup, so an attacker could shuttle the same tokens across N self-owned
identities to vote N times on one proposal — and the anti-whale sqrt curve
AMPLIFIED splitting (√N more power: Σ√xᵢ > √Σxᵢ).

sp1181 routes voting power through the existing stake-to-vote model:
  (a) STAKED BASIS — power is computed from the non-transferable
      governance stake (staked_balance), which stake_for_governance moves
      OUT of `balance`, so the same tokens can't be transferred to a second
      identity and voted again.
  (b) LINEAR formula — split-neutral (Σxᵢ == Σxᵢ), so fragmenting a stake
      across Sybil identities yields NO amplification (vs the old √N).

These tests prove both properties + that delegation (which derives its power
from calculate_voting_power) inherits the staked basis, and that the read
fails safe to 0 power.
"""
from __future__ import annotations

import pytest

from prsm.economy.governance.voting import (
    MIN_VOTING_BALANCE,
    TokenWeightedVoting,
)


class _FakeStakingService:
    """Fake of the staking FTNS service: get_user_balance returns the
    governance stake (staked_balance) per user. Models the post-stake
    ledger state (tokens moved balance→staked_balance)."""

    def __init__(self, staked_by_user):
        self._staked = dict(staked_by_user)
        self.raise_error = False

    async def get_user_balance(self, user_id):
        if self.raise_error:
            raise RuntimeError("simulated staking-ledger read failure")
        return {
            "balance": 0.0,
            "locked_balance": 0.0,
            "staked_balance": self._staked.get(user_id, 0.0),
            "available_balance": 0.0,
            "total_balance": self._staked.get(user_id, 0.0),
        }


def _voting_with_stakes(staked_by_user):
    vs = TokenWeightedVoting()
    vs._staking_ftns_service = _FakeStakingService(staked_by_user)
    return vs


async def _power(vs, user):
    calc = await vs.calculate_voting_power(user)
    return calc.total_voting_power


class TestStakedBasis:
    async def test_power_from_staked_not_live_balance(self):
        """Voting power tracks the staked balance (the live balance is
        never read for power)."""
        vs = _voting_with_stakes({"alice": 10 * MIN_VOTING_BALANCE})
        assert await _power(vs, "alice") > 0

    async def test_zero_stake_zero_power(self):
        """A holder with tokens but NO governance stake has no voting
        power — closes the 'transfer liquid tokens to a fresh identity
        then vote' vector (the recipient has 0 stake)."""
        vs = _voting_with_stakes({"alice": 0.0})
        assert await _power(vs, "alice") == 0.0

    async def test_below_min_stake_zero_power(self):
        vs = _voting_with_stakes({"alice": MIN_VOTING_BALANCE - 1})
        assert await _power(vs, "alice") == 0.0

    async def test_staked_read_failure_fails_safe_to_zero(self):
        """A staking-ledger read error denies power (secure default) —
        it must NOT fall back to the transferable live balance."""
        vs = TokenWeightedVoting()
        fake = _FakeStakingService({"alice": 10 * MIN_VOTING_BALANCE})
        fake.raise_error = True
        vs._staking_ftns_service = fake
        assert await _power(vs, "alice") == 0.0


class TestSybilResistance:
    async def test_split_is_not_amplified_linear(self):
        """THE FIX: splitting a stake across N identities yields the SAME
        summed power as keeping it whole (linear / split-neutral). Under
        the old sqrt curve, two half-stakes summed to √2× the whole — the
        amplification this closes."""
        whole = 4 * MIN_VOTING_BALANCE
        vs_whole = _voting_with_stakes({"whale": whole})
        p_whole = await _power(vs_whole, "whale")

        vs_split = _voting_with_stakes(
            {"sybil_a": whole / 2, "sybil_b": whole / 2},
        )
        p_split = await _power(vs_split, "sybil_a") + await _power(
            vs_split, "sybil_b",
        )
        assert p_split == pytest.approx(p_whole), (
            f"splitting amplified power: whole={p_whole} split={p_split} "
            "(must be equal under the linear formula)"
        )

    async def test_many_way_split_no_advantage(self):
        """N-way split is also split-neutral (no √N amplification)."""
        total = 10 * MIN_VOTING_BALANCE
        vs_whole = _voting_with_stakes({"whale": total})
        p_whole = await _power(vs_whole, "whale")

        n = 5
        per = total / n
        vs_split = _voting_with_stakes(
            {f"s{i}": per for i in range(n)},
        )
        p_split = sum(
            [await _power(vs_split, f"s{i}") for i in range(n)]
        )
        assert p_split == pytest.approx(p_whole)

    async def test_transfer_then_revote_cannot_double_count(self):
        """The end-to-end exploit, modeled at the ledger level: an
        attacker's tokens, once staked on A, are OUT of `balance` and
        cannot be transferred to B to stake+vote again. With a fixed
        total stake T, the aggregate voting power across any number of
        identities never exceeds the power of staking T on one identity
        (linear), so re-voting via transfer yields no gain."""
        T = 6 * MIN_VOTING_BALANCE
        # One identity stakes the full T.
        single = await _power(_voting_with_stakes({"a": T}), "a")
        # Or the SAME T is split across three identities (the most an
        # attacker could field — staked tokens are locked, not reused).
        vs = _voting_with_stakes(
            {"a": T / 3, "b": T / 3, "c": T / 3},
        )
        aggregate = (
            await _power(vs, "a")
            + await _power(vs, "b")
            + await _power(vs, "c")
        )
        assert aggregate == pytest.approx(single)


class TestDelegationInheritsStakedBasis:
    async def test_delegated_power_is_staked_based(self):
        """delegate_voting_power derives power from calculate_voting_power,
        so a delegator with NO stake delegates ZERO power (closes the
        delegation Sybil vector too)."""
        vs = _voting_with_stakes({"delegator": 0.0, "delegate": 0.0})
        ok = await vs.delegate_voting_power("delegator", "delegate")
        # Delegation may be rejected or record 0 power; either way the
        # delegate gains nothing.
        assert vs.delegation_pool.get("delegate", 0.0) == 0.0

    async def test_delegated_power_nonzero_when_delegator_staked(self):
        vs = _voting_with_stakes(
            {"delegator": 10 * MIN_VOTING_BALANCE, "delegate": 0.0},
        )
        await vs.delegate_voting_power("delegator", "delegate")
        assert vs.delegation_pool.get("delegate", 0.0) > 0.0
