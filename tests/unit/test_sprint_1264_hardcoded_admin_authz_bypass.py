"""Sprint 1264 — close the hardcoded-ADMIN authorization bypass (security audit round 4).

Three "admin only" call sites passed `user_role=UserRole.ADMIN` HARDCODED into
EnhancedAuthorizationManager.check_permission (with the comment "Would fetch actual role
from database"). check_permission short-circuits `if user_role == UserRole.ADMIN: return
True`, so every gate evaluated as if the caller were an admin → ANY authenticated user
passed:
  - websocket_auth._check_admin_conversation_access → read ANY user's conversation (IDOR)
  - credential_api.get_system_credential_status / initialize_secure_configuration_endpoint
    → admin credential endpoints reachable by any authenticated user

Fix: thread the caller's REAL role. credential_api has the authenticated User
(get_current_user → User) so it uses current_user.role; the WebSocket path only has a
user_id string, so a new fail-safe resolve_user_role(user_id) fetches the real role and
defaults to the LOWEST privilege (UserRole.USER → deny) on unknown user / bad id / error.
The permission matrix grants `conversations`/`system_credentials` to ADMIN only, so a
non-admin role is correctly denied.
"""
from __future__ import annotations

import importlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

# Use importlib to get the REAL submodule object — `import prsm.core.auth.auth_manager as am`
# binds to the singleton INSTANCE that the package __init__ re-exports under the same name
# (a classic shadowing gotcha), not the module. We need the module to patch its functions.
am = importlib.import_module("prsm.core.auth.auth_manager")
from prsm.core.auth.auth_manager import resolve_user_role
from prsm.core.auth.models import UserRole
from prsm.core.security.enhanced_authorization import get_enhanced_auth_manager


# ── resolve_user_role: real role + fail-safe to USER ─────────────────────────────

@pytest.mark.asyncio
async def test_resolve_user_role_returns_real_role(monkeypatch):
    fake = MagicMock()
    fake.role = UserRole.ADMIN
    monkeypatch.setattr(am.auth_manager, "_get_user_by_id", AsyncMock(return_value=fake))
    assert await resolve_user_role(str(uuid.uuid4())) == UserRole.ADMIN


@pytest.mark.asyncio
async def test_resolve_user_role_unknown_user_defaults_user(monkeypatch):
    monkeypatch.setattr(am.auth_manager, "_get_user_by_id", AsyncMock(return_value=None))
    assert await resolve_user_role(str(uuid.uuid4())) == UserRole.USER


@pytest.mark.asyncio
async def test_resolve_user_role_bad_id_defaults_user():
    # a non-UUID id makes UUID(...) raise → fail-safe to the lowest privilege
    assert await resolve_user_role("not-a-uuid") == UserRole.USER


@pytest.mark.asyncio
async def test_resolve_user_role_error_defaults_user(monkeypatch):
    monkeypatch.setattr(am.auth_manager, "_get_user_by_id",
                        AsyncMock(side_effect=RuntimeError("db down")))
    assert await resolve_user_role(str(uuid.uuid4())) == UserRole.USER


# ── the permission matrix denies non-admins for the abused resources ─────────────

@pytest.mark.asyncio
async def test_non_admin_denied_for_abused_resources():
    auth = get_enhanced_auth_manager()
    assert await auth.check_permission(user_id="u", user_role=UserRole.USER,
                                       resource_type="conversations", action="read_all") is False
    assert await auth.check_permission(user_id="u", user_role=UserRole.USER,
                                       resource_type="system_credentials", action="read") is False
    # a genuine admin is still granted
    assert await auth.check_permission(user_id="a", user_role=UserRole.ADMIN,
                                       resource_type="conversations", action="read_all") is True


# ── WebSocket admin-conversation check now uses the REAL role ─────────────────────

@pytest.mark.asyncio
async def test_ws_admin_conversation_denies_non_admin(monkeypatch):
    import prsm.interface.api.websocket_auth as wsa
    monkeypatch.setattr(am, "resolve_user_role", AsyncMock(return_value=UserRole.USER))
    mgr = wsa.WebSocketAuthManager()
    assert await mgr._check_admin_conversation_access(str(uuid.uuid4())) is False


@pytest.mark.asyncio
async def test_ws_admin_conversation_allows_real_admin(monkeypatch):
    import prsm.interface.api.websocket_auth as wsa
    monkeypatch.setattr(am, "resolve_user_role", AsyncMock(return_value=UserRole.ADMIN))
    mgr = wsa.WebSocketAuthManager()
    assert await mgr._check_admin_conversation_access(str(uuid.uuid4())) is True


# ── credential endpoints deny a non-admin (403) ──────────────────────────────────

@pytest.mark.asyncio
async def test_credential_status_denies_non_admin():
    from fastapi import HTTPException
    from prsm.interface.api.credential_api import get_system_credential_status
    fake = MagicMock()
    fake.role = UserRole.USER
    fake.id = uuid.uuid4()
    with pytest.raises(HTTPException) as ei:
        await get_system_credential_status(current_user=fake)
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_credential_initialize_denies_non_admin():
    from fastapi import HTTPException
    from prsm.interface.api.credential_api import initialize_secure_configuration_endpoint
    fake = MagicMock()
    fake.role = UserRole.USER
    fake.id = uuid.uuid4()
    with pytest.raises(HTTPException) as ei:
        await initialize_secure_configuration_endpoint(current_user=fake)
    assert ei.value.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
