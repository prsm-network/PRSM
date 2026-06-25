"""Sprint 1272 — authenticate the /integrations/* router (audit round 5 follow-on, HIGH).

integration_api.get_current_user was a stub returning "default_user", so the whole
/integrations/* router (mounted in router_registry) was effectively UNAUTHENTICATED — any
caller could register connectors, submit imports, and read import history. Fix: wire the
canonical JWT verifier (signature + exp + revocation + access-token-type) and return the real
authenticated user id (preserving the endpoints' str contract).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from prsm.core.auth.jwt_handler import JWTHandler, TokenData
from prsm.core.integrations.api.integration_api import get_current_user


def _creds(tok="tok"):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=tok)


def _td(token_type="access"):
    now = datetime.now(timezone.utc)
    return TokenData(
        user_id=uuid4(), username="alice", email="a@x.io", role="user",
        permissions=[], token_type=token_type, issued_at=now,
        expires_at=now + timedelta(minutes=15),
    )


@pytest.mark.asyncio
async def test_valid_access_token_returns_user_id(monkeypatch):
    td = _td("access")
    monkeypatch.setattr(JWTHandler, "verify_token", AsyncMock(return_value=td))
    assert await get_current_user(_creds()) == str(td.user_id)


@pytest.mark.asyncio
async def test_refresh_token_rejected(monkeypatch):
    monkeypatch.setattr(JWTHandler, "verify_token", AsyncMock(return_value=_td("refresh")))
    with pytest.raises(HTTPException) as ei:
        await get_current_user(_creds())
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_invalid_or_revoked_rejected(monkeypatch):
    monkeypatch.setattr(JWTHandler, "verify_token", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as ei:
        await get_current_user(_creds())
    assert ei.value.status_code == 401


def test_no_longer_returns_default_user():
    # the stub is gone — the dependency must require credentials + verify the token, not hand
    # out a static "default_user". (The docstring still mentions the old stub, so check the
    # function does NOT `return "default_user"` and DOES verify a token.)
    import inspect
    import prsm.core.integrations.api.integration_api as m
    src = inspect.getsource(m.get_current_user)
    assert 'return "default_user"' not in src
    assert "verify_token" in src
    assert "credentials" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
