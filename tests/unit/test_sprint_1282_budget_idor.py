"""Sprint 1282 — budget endpoints enforce ownership (audit round 7, MED IDOR).

get_budget_status / spend_from_budget / request_budget_expansion were authenticated but the
ownership check was a TODO comment ("Verify user ownership ... For now, simple check would go
here") — so any authenticated user could read, spend, or expand ANY budget_id (horizontal
privilege escalation / IDOR). Fix: _assert_budget_owner verifies the budget's user_id matches
the caller (404 missing, 403 other-owner), enforced on all three endpoints.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from prsm.interface.api.budget_api import _assert_budget_owner


def _mgr(user_id):
    m = type("M", (), {})()
    m.get_budget_status = AsyncMock(return_value={"user_id": user_id, "budget_id": "b"} if user_id else None)
    return m


@pytest.mark.asyncio
async def test_owner_allowed():
    mgr = _mgr("alice")
    status = await _assert_budget_owner(mgr, uuid4(), {"user_id": "alice"})
    assert status["user_id"] == "alice"


@pytest.mark.asyncio
async def test_other_users_budget_forbidden():
    mgr = _mgr("victim")
    with pytest.raises(HTTPException) as ei:
        await _assert_budget_owner(mgr, uuid4(), {"user_id": "attacker"})
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_missing_budget_404():
    mgr = _mgr(None)
    with pytest.raises(HTTPException) as ei:
        await _assert_budget_owner(mgr, uuid4(), {"user_id": "alice"})
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_spend_endpoint_guards_before_mutation(monkeypatch):
    # spend_from_budget must reject another user's budget WITHOUT calling spend_budget_amount
    import prsm.interface.api.budget_api as api
    from prsm.interface.api.budget_api import spend_from_budget, SpendingRequest
    from prsm.economy.tokenomics.ftns_budget_manager import SpendingCategory

    mgr = _mgr("victim")
    mgr.spend_budget_amount = AsyncMock(return_value=True)
    monkeypatch.setattr(api, "get_ftns_budget_manager", lambda: mgr)
    req = SpendingRequest(amount=1, category=list(SpendingCategory)[0], description="x")
    with pytest.raises(HTTPException) as ei:
        await spend_from_budget(uuid4(), req, current_user={"user_id": "attacker"})
    assert ei.value.status_code == 403
    mgr.spend_budget_amount.assert_not_awaited()   # mutation blocked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
