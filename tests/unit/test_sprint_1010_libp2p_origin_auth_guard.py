"""Sprint 1010 — startup guard for the un-authenticated libp2p transport (Residual A).

The P2P-substrate hunt (workflow wbu7u2ftm) hardened the live WebSocket stack but
left the libp2p gossip/discovery path carrying NONE of the origin-authentication
(sp934/937/941/1005). That path is latent today (its native library is absent on
the Linux fleet), but transport_backend defaults to "libp2p" in config, so a
future operator who builds + ships the .so and deploys libp2p inherits every
eclipse / index-poisoning vector the WebSocket path was just fixed against.

sp1246 UPDATE: origin-authentication IS now ported to libp2p (discovery
sp1086/1087/1097 + the gossip-layer sp934 envelope auth shipped in sp1246), so the
guard no longer overstates the risk — it logs WARNING (not CRITICAL) and its message
states origin-auth is ported while two honest residuals (no authenticated PEX relay
path; deferred direct-message sender binding) keep libp2p OPT-IN. The hard-refuse on
PRSM_FORBID_UNAUTHENTICATED_LIBP2P is unchanged.
"""
from __future__ import annotations

import logging

import pytest

from prsm.node.node import _check_libp2p_origin_auth_gap


def test_warns_by_default(monkeypatch, caplog):
    # sp1246 — a SECURITY advisory (WARNING) still fires whenever libp2p is selected,
    # but it is no longer CRITICAL now that origin-auth is ported. It must not raise.
    monkeypatch.delenv("PRSM_FORBID_UNAUTHENTICATED_LIBP2P", raising=False)
    with caplog.at_level(logging.WARNING):
        _check_libp2p_origin_auth_gap()  # must not raise
    advisories = [r for r in caplog.records if "libp2p" in r.message.lower()
                  and r.levelno >= logging.WARNING]
    assert advisories, "expected a libp2p security advisory"
    # the advisory must NOT be CRITICAL anymore (origin-auth is ported)
    assert all(r.levelno < logging.CRITICAL for r in advisories)


def test_refuses_when_enforcement_enabled(monkeypatch):
    monkeypatch.setenv("PRSM_FORBID_UNAUTHENTICATED_LIBP2P", "1")
    with pytest.raises(RuntimeError, match="libp2p"):
        _check_libp2p_origin_auth_gap()


def test_enforcement_false_value_does_not_refuse(monkeypatch):
    monkeypatch.setenv("PRSM_FORBID_UNAUTHENTICATED_LIBP2P", "0")
    _check_libp2p_origin_auth_gap()  # must not raise
