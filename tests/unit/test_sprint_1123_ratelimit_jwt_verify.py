"""Sprint 1123 (Domain-08 review LOW-8) — the per-user rate-limit bucket key is taken
from a SIGNATURE-VERIFIED token only. A forged/unsigned `sub` no longer keys a bucket
(it falls back to per-IP limiting), closing the evade-limit + grief-a-victim vectors.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import jwt
import pytest

from prsm.interface.api.middleware import _extract_user_id_from_token
from prsm.core.config import get_settings


def _request(token: str):
    req = MagicMock()
    req.headers = {"Authorization": f"Bearer {token}"}
    return req


def _secret():
    s = get_settings()
    return (s.secret_key if s else "test-secret-key") or "test-secret-key"


@pytest.mark.asyncio
async def test_correctly_signed_token_yields_sub():
    token = jwt.encode({"sub": "user-123"}, _secret(), algorithm="HS256")
    assert await _extract_user_id_from_token(_request(token)) == "user-123"


@pytest.mark.asyncio
async def test_forged_token_signed_with_wrong_key_is_rejected():
    forged = jwt.encode({"sub": "victim-or-evader"}, "attacker-key", algorithm="HS256")
    assert await _extract_user_id_from_token(_request(forged)) is None


@pytest.mark.asyncio
async def test_unsigned_alg_none_token_is_rejected():
    unsigned = jwt.encode({"sub": "spoofed"}, "", algorithm="none")
    assert await _extract_user_id_from_token(_request(unsigned)) is None


@pytest.mark.asyncio
async def test_no_bearer_header_is_none():
    req = MagicMock()
    req.headers = {}
    assert await _extract_user_id_from_token(req) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
