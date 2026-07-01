"""Sprint 1337 (S4) — big-model MULTI-STAGE on-chain settlement go-live preflight.

Read-only/offline go/no-go for a Design-A multi-stage operator, extending the single-stage
sp1301 preflight per-node. In Design A each stage node self-commits its own share with its own
funded settler key, so the base preflight IS the per-node gate; this adds the multi-stage config
(PRSM_MULTISTAGE_SETTLEMENT on, node_id→payee wallet map, endpoint resolution). Fully offline via
the injectable reader + a tmp wallet-map file.
"""
from __future__ import annotations

import json

import pytest

F_REGISTRY = "0x12a01F6C487d765af389bC7D95D90b3136a391F2"
F_ESCROW = "0x4e93a04b3A0C5063FE584980e6c2B1429495EEa1"
CODE_WITH_SEL = "0x60806040" + "fa4ce156" + "00" * 12

_ENV_KEYS = (
    "PRSM_ONCHAIN_SETTLEMENT", "FTNS_WALLET_PRIVATE_KEY", "PRSM_OPERATOR_ADDRESS",
    "PRSM_SETTLEMENT_REGISTRY_ADDRESS", "PRSM_SETTLEMENT_SUPPORTS_ATTESTATION",
    "PRSM_NETWORK", "PRSM_ESCROW_POOL_ADDRESS", "BASE_RPC_URL", "PRSM_BASE_RPC_URL",
    "PRSM_MULTISTAGE_SETTLEMENT", "PRSM_COMPUTE_WALLET_MAP_FILE",
    "PRSM_MULTISTAGE_ENDPOINT_MAP", "PRSM_MULTISTAGE_SETTLEMENT_STATE_FILE",
)


class FakeReader:
    def __init__(self, *, paused=False, code=CODE_WITH_SEL, balance=10 ** 18):
        self._paused, self._code, self._balance = paused, code, balance

    def paused(self, addr):
        return self._paused

    def code_hex(self, addr):
        return self._code

    def eth_balance_wei(self, addr):
        return self._balance


def _eth_keypair():
    from eth_account import Account
    acct = Account.create()
    return acct.key.hex(), acct.address


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PRSM_NETWORK", "mainnet")


def _status(report, check):
    for f in report.findings:
        if f.check == check:
            return f.status
    return None


def _base_green_env(monkeypatch):
    """The single-stage GO scenario (mirrors sp1301's _all_green_env)."""
    key, addr = _eth_keypair()
    monkeypatch.setenv("PRSM_ONCHAIN_SETTLEMENT", "1")
    monkeypatch.setenv("FTNS_WALLET_PRIVATE_KEY", key)
    monkeypatch.setenv("PRSM_OPERATOR_ADDRESS", addr)
    monkeypatch.setenv("PRSM_SETTLEMENT_REGISTRY_ADDRESS", F_REGISTRY)
    monkeypatch.setenv("PRSM_ESCROW_POOL_ADDRESS", F_ESCROW)
    monkeypatch.setenv("PRSM_SETTLEMENT_SUPPORTS_ATTESTATION", "1")
    monkeypatch.setattr(
        "prsm.settlement.client_wiring._registry_supports_attestation",
        lambda *a, **k: True)
    return addr


def _wallet_map_file(tmp_path, entries):
    p = tmp_path / "wallet_map.json"
    p.write_text(json.dumps(entries))
    return str(p)


def _run(monkeypatch, addr, node=None):
    from prsm.settlement.go_live_preflight import run_multistage_go_live_preflight
    return run_multistage_go_live_preflight(
        reader=FakeReader(), provider_address=addr, node=node)


# ── happy path ────────────────────────────────────────────────────────────────

def test_multistage_all_green_is_go(monkeypatch, tmp_path):
    addr = _base_green_env(monkeypatch)
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_COMPUTE_WALLET_MAP_FILE",
                       _wallet_map_file(tmp_path, {"nodeA": "0x" + "a1" * 20,
                                                   "nodeB": "0x" + "b2" * 20}))
    report = _run(monkeypatch, addr)
    assert report.go is True
    assert _status(report, "multistage_settlement") == "PASS"
    assert _status(report, "compute_wallet_map") == "PASS"


# ── the two multi-stage FAILs ─────────────────────────────────────────────────

def test_multistage_gate_off_is_no_go(monkeypatch, tmp_path):
    addr = _base_green_env(monkeypatch)
    # gate OFF, everything else green
    monkeypatch.setenv("PRSM_COMPUTE_WALLET_MAP_FILE",
                       _wallet_map_file(tmp_path, {"nodeA": "0x" + "a1" * 20}))
    report = _run(monkeypatch, addr)
    assert report.go is False
    assert _status(report, "multistage_settlement") == "FAIL"


def test_empty_wallet_map_is_no_go(monkeypatch):
    addr = _base_green_env(monkeypatch)
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    # no PRSM_COMPUTE_WALLET_MAP_FILE → splitter would fail-close
    report = _run(monkeypatch, addr)
    assert report.go is False
    assert _status(report, "compute_wallet_map") == "FAIL"


# ── composition + advisory findings ───────────────────────────────────────────

def test_includes_base_single_stage_findings(monkeypatch, tmp_path):
    addr = _base_green_env(monkeypatch)
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_COMPUTE_WALLET_MAP_FILE",
                       _wallet_map_file(tmp_path, {"nodeA": "0x" + "a1" * 20}))
    report = _run(monkeypatch, addr)
    # the per-node base gate is included (Design A each node self-commits)
    assert _status(report, "settler_key") == "PASS"
    assert _status(report, "onchain_settlement") == "PASS"


def test_endpoint_map_absent_warns_not_fails(monkeypatch, tmp_path):
    addr = _base_green_env(monkeypatch)
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_COMPUTE_WALLET_MAP_FILE",
                       _wallet_map_file(tmp_path, {"nodeA": "0x" + "a1" * 20}))
    report = _run(monkeypatch, addr)
    # discovery is the runtime fallback, so a missing static map is advisory, not a blocker
    assert _status(report, "endpoint_resolution") == "WARN"
    assert report.go is True


def test_node_injection_adds_per_stage_wiring_checks(monkeypatch, tmp_path):
    from types import SimpleNamespace
    addr = _base_green_env(monkeypatch)
    monkeypatch.setenv("PRSM_MULTISTAGE_SETTLEMENT", "1")
    monkeypatch.setenv("PRSM_COMPUTE_WALLET_MAP_FILE",
                       _wallet_map_file(tmp_path, {"nodeA": "0x" + "a1" * 20}))
    # no node → the per-stage client check is absent
    assert _status(_run(monkeypatch, addr), "per_stage_client") is None
    # node injected → the check appears (any status; fail-soft if it can't build)
    report = _run(monkeypatch, addr, node=SimpleNamespace(transport=None, identity=None))
    assert _status(report, "per_stage_client") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
