"""Sprint 547 — wire ``onboarding_router`` into the node FastAPI app.

User-perspective dogfood arc (sprint 424) surfaced **F6**:
``node start`` logs an "/onboarding/" URL but the path returns 404.
``prsm/interface/api/onboarding_router.py`` defines a 6-step
interactive wizard (welcome → API keys → backend → network →
identity → launch) and templates for each step under
``prsm/interface/api/templates/onboarding/``, but
``create_api_app`` never includes the router. The wizard is dead
code until sprint 547 plumbs it in.

Pin tests assert:
  - GET /onboarding/ returns 200 (was 404 pre-sprint)
  - The router is registered in the FastAPI app's route table
  - At least the welcome + launch step paths are reachable
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _stub_node():
    n = MagicMock()
    n.identity = MagicMock(node_id="stub-node-id")
    return n


def _make_app():
    from prsm.node.api import create_api_app
    return create_api_app(_stub_node(), enable_security=False)


def _all_paths(routes) -> set:
    """Every registered path, recursing into FastAPI's ``_IncludedRouter`` wrappers.

    FastAPI >= ~0.13x no longer FLATTENS ``include_router()`` into ``app.routes``: it appends one
    ``_IncludedRouter`` delegate that holds the sub-router in ``.original_router`` and dispatches to
    it at match time. The routes are mounted and serve normally (verified: /onboarding/* → 200) —
    they are just one level down, and the delegate has no ``.path``. Older FastAPI copied them in
    flat. This handles both, and fails CLOSED: it can only find FEWER paths, never invent one, so a
    genuinely un-mounted router still trips the assertions below.
    """
    found = set()
    for r in routes:
        p = getattr(r, "path", None)
        if p:
            found.add(p)
        inner = getattr(r, "original_router", None)
        if inner is not None:
            found |= _all_paths(inner.routes)
    return found


def test_onboarding_welcome_returns_200():
    """The dogfood arc F6 symptom: /onboarding/ returned 404
    pre-sprint. After wiring the router, the welcome step must
    serve a 200."""
    app = _make_app()
    client = TestClient(app)
    response = client.get("/onboarding/")
    assert response.status_code == 200, (
        f"GET /onboarding/ returned {response.status_code} — "
        f"router not wired. Body: {response.text[:200]}"
    )


def test_onboarding_router_in_app_routes():
    """Defensive structural check: the route table contains at
    least one route under the ``/onboarding`` prefix."""
    app = _make_app()
    onboarding_routes = [
        p for p in _all_paths(app.routes)
        if p.startswith("/onboarding")
    ]
    assert onboarding_routes, (
        "create_api_app did not include the onboarding router. "
        f"App has {len(app.routes)} routes total."
    )


def test_onboarding_router_has_all_six_steps():
    """The router defines 6 steps + their POST handlers — total
    paths registered should be the full set, not a subset."""
    app = _make_app()
    paths = {
        p for p in _all_paths(app.routes)
        if p.startswith("/onboarding")
    }
    expected_get_paths = {
        "/onboarding/",
        "/onboarding/api-keys",
        "/onboarding/backend",
        "/onboarding/network",
        "/onboarding/identity",
        "/onboarding/launch",
    }
    missing = expected_get_paths - paths
    assert not missing, (
        f"onboarding router missing {missing} from registered paths"
    )


def test_onboarding_launch_get_is_reachable():
    """End-of-wizard step is reachable (i.e., not just step 1)."""
    app = _make_app()
    client = TestClient(app)
    response = client.get("/onboarding/launch")
    # Either 200 (renders launch page) or 3xx (redirects to an
    # earlier step due to incomplete pending config) — both are
    # WIRED behavior; 404 means the router isn't wired.
    assert response.status_code != 404, (
        f"GET /onboarding/launch returned 404 — router not wired"
    )
