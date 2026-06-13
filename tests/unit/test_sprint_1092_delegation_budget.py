"""Sprint 1092 — relayer delegation budget store (requester-payment relayer, brick 2).

A funder's PaymentDelegation (sp1091) caps the relayer's CUMULATIVE spend
(max_total_spend_wei) across many per-request authorizations. This store tracks consumed
budget per delegation_nonce and reserves conservatively (the per-request ceiling) so the
relayer can never authorize more than the funder delegated — even across requests/restarts.
Mirrors the sp1055 nonce store: InMemory for tests/single-process, Durable (sp1039 atomic
JSON) so a restart can't reopen budget the funder already spent.
"""
from __future__ import annotations

import pytest

from prsm.settlement.delegation_budget import (
    DurableDelegationBudgetStore,
    InMemoryDelegationBudgetStore,
)

_NONCE = "0x" + "cd" * 32
_CAP = 10**18  # 1 FTNS total


def test_reserve_under_cap_succeeds_and_accumulates():
    s = InMemoryDelegationBudgetStore()
    assert s.consumed(_NONCE) == 0
    assert s.reserve(_NONCE, 3 * 10**17, _CAP) is True
    assert s.consumed(_NONCE) == 3 * 10**17
    assert s.reserve(_NONCE, 4 * 10**17, _CAP) is True
    assert s.consumed(_NONCE) == 7 * 10**17


def test_reserve_over_cap_rejected_and_does_not_consume():
    s = InMemoryDelegationBudgetStore()
    assert s.reserve(_NONCE, 9 * 10**17, _CAP) is True
    # the next request would push cumulative to 1.4 FTNS > 1 FTNS cap → rejected
    assert s.reserve(_NONCE, 5 * 10**17, _CAP) is False
    assert s.consumed(_NONCE) == 9 * 10**17   # unchanged — a rejected reserve adds nothing


def test_reserve_exactly_to_cap_succeeds():
    s = InMemoryDelegationBudgetStore()
    assert s.reserve(_NONCE, _CAP, _CAP) is True
    assert s.reserve(_NONCE, 1, _CAP) is False


def test_distinct_delegations_have_independent_budgets():
    s = InMemoryDelegationBudgetStore()
    other = "0x" + "ee" * 32
    assert s.reserve(_NONCE, _CAP, _CAP) is True
    assert s.reserve(other, _CAP, _CAP) is True   # independent
    assert s.consumed(_NONCE) == _CAP
    assert s.consumed(other) == _CAP


def test_nonce_case_insensitive():
    s = InMemoryDelegationBudgetStore()
    assert s.reserve(_NONCE.upper(), 5 * 10**17, _CAP) is True
    assert s.consumed(_NONCE.lower()) == 5 * 10**17


def test_durable_survives_restart(tmp_path):
    path = tmp_path / "budget.json"
    s1 = DurableDelegationBudgetStore(path)
    assert s1.reserve(_NONCE, 6 * 10**17, _CAP) is True

    # a fresh store at the same path rehydrates the consumed budget
    s2 = DurableDelegationBudgetStore(path)
    assert s2.consumed(_NONCE) == 6 * 10**17
    # ...so a restart can't reopen already-spent budget
    assert s2.reserve(_NONCE, 5 * 10**17, _CAP) is False
    assert s2.reserve(_NONCE, 4 * 10**17, _CAP) is True


def test_durable_corrupt_file_refuses_to_start_empty(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text("{ this is not valid json")
    with pytest.raises(Exception):
        DurableDelegationBudgetStore(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
