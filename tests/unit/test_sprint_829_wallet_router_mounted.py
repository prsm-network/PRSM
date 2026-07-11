"""Sprint 829 — F28 fix: wallet_api router mounted in daemon.

Live dogfood 2026-05-24 surfaced F28: `prsm wallet devices add
--register` returned HTTP 404 from the daemon's
`/api/v1/auth/wallet/bind` endpoint. Root cause: the wallet_api
router was only mounted in `prsm.interface.api.app_factory`'s
router_registry path, but `prsm node start` builds its app via
`prsm.node.api.create_api_app` which never called
`app.include_router(wallet_router)`. Sprint 793 wired the
SERVICES (set_services for shared state) but the actual HTTP
routes never reached the running daemon — every multi-device
endpoint (bind, devices, devices/earnings — sprints 786-794)
returned 404 in production.

Sprint 829 fix: add the include_router call to create_api_app
next to sprint-547's onboarding_router include. Same fail-soft
try/except pattern so a broken import doesn't crash the daemon —
only the multi-device flow degrades.

Pin tests:
- After create_api_app, /api/v1/auth/wallet/bind is registered
  (regression guard against future router-removal).
- After create_api_app, /api/v1/auth/wallet/bindings is
  registered.
- Source-shape pin: create_api_app imports + includes
  wallet_api.router (defends the wiring against accidental
  removal during refactors).
"""
from __future__ import annotations

import inspect


def _node_stub():
    """Minimal node stub for create_api_app (most callbacks are
    optional / fail-soft)."""
    from unittest.mock import MagicMock
    n = MagicMock()
    n.node_id = "test-node"
    n.display_name = "test"
    n._receipt_store = None
    n._audit_log = None
    return n


def _all_paths(routes) -> set:
    """Every registered path, recursing into FastAPI's ``_IncludedRouter`` wrappers.

    FastAPI >= ~0.13x no longer FLATTENS ``include_router()`` into ``app.routes``: it appends one
    ``_IncludedRouter`` delegate that holds the sub-router in ``.original_router`` and dispatches to
    it at match time. The routes are mounted and serve normally (verified: POST
    /api/v1/auth/wallet/bind dispatches into the handler) — they are just one level down, and the
    delegate has no ``.path`` (which is the AttributeError this replaces). Older FastAPI copied them
    in flat. This handles both, and fails CLOSED: it can only find FEWER paths, never invent one, so
    a genuinely un-mounted router still trips the assertions below.
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


def test_wallet_router_mounted_bind_endpoint_exists():
    """After create_api_app, /api/v1/auth/wallet/bind is a
    registered POST route — F28 regression guard."""
    from prsm.node.api import create_api_app
    app = create_api_app(_node_stub(), enable_security=False)
    paths = _all_paths(app.routes)
    assert "/api/v1/auth/wallet/bind" in paths, (
        f"wallet_api router not mounted; got paths sample: "
        f"{sorted(p for p in paths if 'wallet' in p)}"
    )


def test_wallet_router_mounted_bindings_list_exists():
    """Sprint 790's /api/v1/auth/wallet/bindings (devices list)
    must also be registered."""
    from prsm.node.api import create_api_app
    app = create_api_app(_node_stub(), enable_security=False)
    paths = _all_paths(app.routes)
    assert "/api/v1/auth/wallet/bindings" in paths, (
        f"bindings-list not mounted; wallet paths: "
        f"{sorted(p for p in paths if 'wallet' in p)}"
    )


def test_create_api_app_source_includes_wallet_router():
    """Source-shape pin: defends the include_router call against
    accidental removal during refactors. The string match is
    intentionally loose so the test survives reformatting."""
    from prsm.node import api as _api_mod
    src = inspect.getsource(_api_mod.create_api_app)
    assert "wallet_api" in src, (
        "create_api_app no longer references wallet_api — "
        "F28 fix regressed; reinstate the include_router call."
    )
    assert "include_router" in src, (
        "create_api_app no longer calls include_router — "
        "F28 wiring path is broken."
    )
