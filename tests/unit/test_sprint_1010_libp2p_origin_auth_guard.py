"""Sprint 1010 — startup guard for the un-authenticated libp2p transport (Residual A).

The P2P-substrate hunt (workflow wbu7u2ftm) hardened the live WebSocket stack but
left the libp2p gossip/discovery path carrying NONE of the origin-authentication
(sp934/937/941/1005). That path is latent today (its native library is absent on
the Linux fleet), but transport_backend defaults to "libp2p" in config, so a
future operator who builds + ships the .so and deploys libp2p inherits every
eclipse / index-poisoning vector the WebSocket path was just fixed against.

This is the cheap interim from docs/2026-06-04-p2p-substrate-hunt-residuals.md
(Residual A): a startup guard that warns CRITICAL whenever libp2p is selected, and
hard-refuses when the operator opts into enforcement
(PRSM_FORBID_UNAUTHENTICATED_LIBP2P), so the unhardened path cannot ship silently.
"""
from __future__ import annotations

import logging

import pytest

from prsm.node.node import _check_libp2p_origin_auth_gap


def test_warns_critical_by_default(monkeypatch, caplog):
    monkeypatch.delenv("PRSM_FORBID_UNAUTHENTICATED_LIBP2P", raising=False)
    with caplog.at_level(logging.CRITICAL):
        _check_libp2p_origin_auth_gap()  # must not raise
    assert any("libp2p" in r.message.lower() and r.levelno >= logging.CRITICAL
               for r in caplog.records)


def test_refuses_when_enforcement_enabled(monkeypatch):
    monkeypatch.setenv("PRSM_FORBID_UNAUTHENTICATED_LIBP2P", "1")
    with pytest.raises(RuntimeError, match="libp2p"):
        _check_libp2p_origin_auth_gap()


def test_enforcement_false_value_does_not_refuse(monkeypatch):
    monkeypatch.setenv("PRSM_FORBID_UNAUTHENTICATED_LIBP2P", "0")
    _check_libp2p_origin_auth_gap()  # must not raise
