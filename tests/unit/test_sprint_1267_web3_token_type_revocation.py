"""Sprint 1267 — web3 wallet API auth: enforce token-type + revocation (audit round 5, HIGH).

The /api/v1/web3/* router authenticated via an ad-hoc pyjwt.decode that verified
signature/exp but NEVER checked token_type or revocation. The core JWTHandler mints 7-day
REFRESH tokens (sub+exp, same HS256 secret), so a captured refresh token worked as a full
bearer credential on wallet connect/transfer endpoints, and a logged-out/revoked token still
authenticated. Fix: route through the canonical jwt_handler.verify_token (signature + exp +
required claims + revocation) and reject any non-access token.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from prsm.core.auth.jwt_handler import JWTHandler, TokenData
from prsm.economy.web3.frontend_integration import get_current_user


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
async def test_access_token_accepted(monkeypatch):
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
async def test_invalid_or_revoked_token_rejected(monkeypatch):
    # verify_token returns None for an invalid OR revoked token → reject
    monkeypatch.setattr(JWTHandler, "verify_token", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as ei:
        await get_current_user(_creds())
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_uses_canonical_verifier(monkeypatch):
    # the ad-hoc decode is gone — auth must go through jwt_handler.verify_token
    spy = AsyncMock(return_value=_td("access"))
    monkeypatch.setattr(JWTHandler, "verify_token", spy)
    await get_current_user(_creds("the-token"))
    spy.assert_awaited_once_with("the-token")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
