"""Sprint 1278 — authenticate the integration config + security routers (audit round 7, HIGH).

config_api.py (credential vault: store/list/validate/DELETE credentials, settings import/export)
and security_api.py (e.g. POST /policies/update, which can globally disable security scanning)
both used the same placeholder `get_current_user()->'default_user'` stub → effectively
UNAUTHENTICATED, just like the integration_api router fixed in sp1272. Fix: wire the canonical
JWT verifier (signature + exp + revocation + access-token-type) into both.
"""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from prsm.core.auth.jwt_handler import JWTHandler, TokenData

_MODULES = [
    "prsm.core.integrations.api.config_api",
    "prsm.core.integrations.api.security_api",
]


def _creds(tok="tok"):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=tok)


def _td(token_type="access"):
    now = datetime.now(timezone.utc)
    return TokenData(
        user_id=uuid4(), username="alice", email="a@x.io", role="user",
        permissions=[], token_type=token_type, issued_at=now,
        expires_at=now + timedelta(minutes=15),
    )


@pytest.mark.parametrize("modname", _MODULES)
@pytest.mark.asyncio
async def test_valid_access_token_returns_user_id(modname, monkeypatch):
    mod = importlib.import_module(modname)
    td = _td("access")
    monkeypatch.setattr(JWTHandler, "verify_token", AsyncMock(return_value=td))
    assert await mod.get_current_user(_creds()) == str(td.user_id)


@pytest.mark.parametrize("modname", _MODULES)
@pytest.mark.asyncio
async def test_refresh_token_rejected(modname, monkeypatch):
    mod = importlib.import_module(modname)
    monkeypatch.setattr(JWTHandler, "verify_token", AsyncMock(return_value=_td("refresh")))
    with pytest.raises(HTTPException) as ei:
        await mod.get_current_user(_creds())
    assert ei.value.status_code == 401


@pytest.mark.parametrize("modname", _MODULES)
@pytest.mark.asyncio
async def test_invalid_or_revoked_rejected(modname, monkeypatch):
    mod = importlib.import_module(modname)
    monkeypatch.setattr(JWTHandler, "verify_token", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as ei:
        await mod.get_current_user(_creds())
    assert ei.value.status_code == 401


@pytest.mark.parametrize("modname", _MODULES)
def test_no_longer_default_user(modname):
    import inspect
    mod = importlib.import_module(modname)
    src = inspect.getsource(mod.get_current_user)
    assert 'return "default_user"' not in src
    assert "verify_token" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
