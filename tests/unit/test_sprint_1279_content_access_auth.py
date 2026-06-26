"""Sprint 1279 — content-economy /access must authenticate + bind the payer (audit round 7, HIGH).

POST /content-economy/access had NO auth and took accessor_id (who pays) + creator_id (royalty
recipient) straight from the request body, so any caller could charge ANOTHER user's FTNS
(accessor_id spoofing) and/or steer royalties. Fix: authenticate the caller and OVERRIDE
accessor_id with the authenticated user id — the payer is always the caller, never a
body-supplied value.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from prsm.api.content_economy_routes import ContentAccessRequest, process_content_access


def _payment():
    return SimpleNamespace(
        payment_id="pay-1", content_id="cidX", status=SimpleNamespace(value="completed"),
        amount=0.01, royalty_distributions=[], error=None,
    )


@pytest.mark.asyncio
async def test_accessor_id_bound_to_authenticated_user():
    economy = SimpleNamespace(process_content_access=AsyncMock(return_value=_payment()))
    me = SimpleNamespace(id=uuid4())
    # attacker tries to bill a victim by putting their id in the body
    req = ContentAccessRequest(cid="cidX", accessor_id="victim-user",
                               royalty_rate=0.01, creator_id="someone")
    await process_content_access(req, economy=economy, current_user=me)

    economy.process_content_access.assert_awaited_once()
    kwargs = economy.process_content_access.await_args.kwargs
    # the charge is bound to the AUTHENTICATED caller, not the spoofed body value
    assert kwargs["accessor_id"] == str(me.id)
    assert kwargs["accessor_id"] != "victim-user"


@pytest.mark.asyncio
async def test_endpoint_requires_auth_dependency():
    # the route must declare a get_current_user dependency (no longer anonymous)
    import inspect
    sig = inspect.signature(process_content_access)
    assert "current_user" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
