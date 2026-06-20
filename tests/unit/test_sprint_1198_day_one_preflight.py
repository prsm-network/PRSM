"""Sprint 1198 — day-one-live readiness checks in the node startup preflight.

`_node_preflight_diagnostics` (run by `prsm node start`) checked Python/config/
API-bind/bootstrap/wallet but NONE of the front-door readiness items built this
session. These 4 additions make the preflight answer "will a real user get real
results?": inference serving, network+contracts resolution, public-bind auth
posture (predicts the sp1011 refuse-to-start), and requester-payment readiness.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.cli import _node_preflight_diagnostics


def _config(api_host="127.0.0.1"):
    cfg = MagicMock()
    cfg.config_path = MagicMock()
    cfg.config_path.exists = MagicMock(return_value=False)
    cfg.api_port = 0
    cfg.bootstrap_nodes = []
    cfg.p2p_port = 9001
    cfg.api_host = api_host
    return cfg


def _check(checks, name):
    return next((c for c in checks if c.name == name), None)


def _run(env, api_host="127.0.0.1"):
    with patch.dict(os.environ, env, clear=False):
        for k in ("PRSM_INFERENCE_EXECUTOR", "PRSM_NODE_API_KEY",
                  "PRSM_AUTO_PROVISION_API_KEY", "PRSM_ALLOW_INSECURE_PUBLIC_BIND",
                  "PRSM_REQUESTER_PAYMENT", "FTNS_WALLET_PRIVATE_KEY",
                  "PRSM_AUTO_INIT_LOGGING"):
            if k not in env:
                os.environ.pop(k, None)
        return _node_preflight_diagnostics(_config(api_host))


# ── inference serving ────────────────────────────────────────────────────────────────

def test_inference_check_present():
    c = _check(_run({}), "Inference serving")
    assert c is not None
    assert c.status in ("PASS", "WARN")  # PASS iff .[ml] installed in this env


def test_inference_disabled_warns():
    c = _check(_run({"PRSM_INFERENCE_EXECUTOR": "none"}), "Inference serving")
    assert c.status == "WARN"
    assert "503" in c.details


# ── network + contracts ──────────────────────────────────────────────────────────────

def test_network_check_present_and_resolves_mainnet():
    c = _check(_run({"PRSM_NETWORK": "mainnet"}), "Network + contracts")
    assert c is not None
    assert c.status == "PASS"  # mainnet has pinned FTNS + EscrowPool
    assert "chainId 8453" in c.details


# ── public-bind auth posture (the refuse-to-start predictor) ─────────────────────────

def test_auth_posture_loopback_pass():
    c = _check(_run({}, api_host="127.0.0.1"), "API auth posture")
    assert c.status == "PASS"


def test_auth_posture_public_no_key_fails_required():
    c = _check(_run({}, api_host="0.0.0.0"), "API auth posture")
    assert c.status == "FAIL"
    assert c.required is True
    assert "REFUSE to start" in c.details


def test_auth_posture_public_with_key_pass():
    c = _check(_run({"PRSM_NODE_API_KEY": "k"}, api_host="0.0.0.0"), "API auth posture")
    assert c.status == "PASS"


def test_auth_posture_public_auto_provision_pass():
    c = _check(_run({"PRSM_AUTO_PROVISION_API_KEY": "1"}, api_host="0.0.0.0"),
               "API auth posture")
    assert c.status == "PASS"  # provisioning will mint a key → posture ok


def test_auth_posture_public_insecure_ack_warns():
    c = _check(_run({"PRSM_ALLOW_INSECURE_PUBLIC_BIND": "1"}, api_host="0.0.0.0"),
               "API auth posture")
    assert c.status == "WARN"


# ── requester-payment readiness (only when opted in) ─────────────────────────────────

def test_requester_payment_check_absent_when_off():
    assert _check(_run({}), "Requester-payment readiness") is None


def test_requester_payment_on_without_key_warns():
    c = _check(_run({"PRSM_REQUESTER_PAYMENT": "1"}), "Requester-payment readiness")
    assert c is not None
    assert c.status == "WARN"
    assert "402" in c.details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
