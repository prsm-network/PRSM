"""Sprint 1275 — governance: reject votes on concluded proposals + un-hardcode quorum (round 6).

#3 (MEDIUM): cast_vote checked only the voting-time WINDOW, not proposal.status — but a
proposal can conclude EARLY (_should_conclude_voting_early → _conclude_voting sets
status=approved/rejected) while still inside the window, after which more votes were accepted
on the already-decided proposal. Fix: reject votes unless proposal.status == "active".

#5 (LOW): _calculate_total_eligible_voting_power (the quorum denominator) returned a hardcoded
1_000_000.0 — a fixed quorum a proposer can size their vote against. Fix: operator-configurable
via PRSM_GOVERNANCE_TOTAL_ELIGIBLE_POWER (the in-memory governance object can't enumerate
on-chain holders), defaulting to the legacy value.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from prsm.economy.governance.voting import TokenWeightedVoting


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["approved", "rejected", "expired", "draft"])
async def test_vote_rejected_on_non_active_proposal(status):
    v = TokenWeightedVoting()
    pid = uuid4()
    v.proposals[pid] = SimpleNamespace(status=status)
    v.votes[pid] = []
    v._validate_vote_eligibility = AsyncMock(return_value=True)  # isolate the status guard
    # a vote on a concluded/non-active proposal must be refused
    assert await v.cast_vote("voter-1", pid, True) is False


@pytest.mark.asyncio
async def test_total_eligible_power_env_override(monkeypatch):
    v = TokenWeightedVoting()
    monkeypatch.setenv("PRSM_GOVERNANCE_TOTAL_ELIGIBLE_POWER", "5000000")
    assert await v._calculate_total_eligible_voting_power() == 5_000_000.0


@pytest.mark.asyncio
async def test_total_eligible_power_default(monkeypatch):
    v = TokenWeightedVoting()
    monkeypatch.delenv("PRSM_GOVERNANCE_TOTAL_ELIGIBLE_POWER", raising=False)
    assert await v._calculate_total_eligible_voting_power() == 1_000_000.0


@pytest.mark.asyncio
async def test_total_eligible_power_invalid_env_falls_back(monkeypatch):
    v = TokenWeightedVoting()
    monkeypatch.setenv("PRSM_GOVERNANCE_TOTAL_ELIGIBLE_POWER", "not-a-number")
    assert await v._calculate_total_eligible_voting_power() == 1_000_000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
