"""Sprint 1301 — on-chain settlement go-live preflight.

Read-only/offline go/no-go gate for the FIRST production on-chain settlement
activation (the funded settler-key ceremony that has never run). Mirrors the
Aerodrome go_live_verification harness: PASS/FAIL/WARN/INFO findings, ``go`` iff
zero FAIL. On-chain reads go through an injectable reader so the gate is fully
offline-testable.
"""
from __future__ import annotations

import pytest

F_REGISTRY = "0x12a01F6C487d765af389bC7D95D90b3136a391F2"
RETIRED_REGISTRY = "0x48fFab641b9D638F312FFA776818756a326F995B"
F_ESCROW = "0x4e93a04b3A0C5063FE584980e6c2B1429495EEa1"
# bytecode containing the commitBatchWithAttestation selector (sp1241/sp1299)
CODE_WITH_SEL = "0x60806040" + "fa4ce156" + "00" * 12
CODE_NO_SEL = "0xdeadbeefcafe"

_ENV_KEYS = (
    "PRSM_ONCHAIN_SETTLEMENT", "FTNS_WALLET_PRIVATE_KEY", "PRSM_OPERATOR_ADDRESS",
    "PRSM_SETTLEMENT_REGISTRY_ADDRESS", "PRSM_SETTLEMENT_SUPPORTS_ATTESTATION",
    "PRSM_NETWORK", "PRSM_ESCROW_POOL_ADDRESS", "BASE_RPC_URL", "PRSM_BASE_RPC_URL",
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
    """Deterministic env regardless of the dev machine."""
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PRSM_NETWORK", "mainnet")


def _run(monkeypatch, reader=None, provider_address=None):
    from prsm.settlement.go_live_preflight import run_settlement_go_live_preflight
    return run_settlement_go_live_preflight(
        reader=reader or FakeReader(), provider_address=provider_address,
    )


def _status(report, check):
    for f in report.findings:
        if f.check == check:
            return f.status
    return None


def _all_green_env(monkeypatch):
    key, addr = _eth_keypair()
    monkeypatch.setenv("PRSM_ONCHAIN_SETTLEMENT", "1")
    monkeypatch.setenv("FTNS_WALLET_PRIVATE_KEY", key)
    monkeypatch.setenv("PRSM_OPERATOR_ADDRESS", addr)
    monkeypatch.setenv("PRSM_SETTLEMENT_REGISTRY_ADDRESS", F_REGISTRY)
    monkeypatch.setenv("PRSM_ESCROW_POOL_ADDRESS", F_ESCROW)
    monkeypatch.setenv("PRSM_SETTLEMENT_SUPPORTS_ATTESTATION", "1")
    # the client build's own attestation probe would hit the network — stub it
    monkeypatch.setattr(
        "prsm.settlement.client_wiring._registry_supports_attestation",
        lambda *a, **k: True,
    )
    return addr


# ── happy path ───────────────────────────────────────────────────────────────

def test_all_green_is_go(monkeypatch):
    addr = _all_green_env(monkeypatch)
    report = _run(monkeypatch, provider_address=addr)
    assert report.go is True
    for check in ("provider_address", "onchain_settlement", "settler_key",
                  "client_build", "registry_resolved", "registry_active",
                  "attestation_surface", "settler_funded", "attestation_flag"):
        assert _status(report, check) == "PASS", f"{check}: {_status(report, check)}"


# ── off by default ───────────────────────────────────────────────────────────

def test_off_by_default_is_no_go(monkeypatch):
    """Bare env → multiple FAILs (no opt-in, no key, view-only build)."""
    report = _run(monkeypatch, provider_address=None)
    assert report.go is False
    assert _status(report, "onchain_settlement") == "FAIL"
    assert _status(report, "settler_key") == "FAIL"
    assert _status(report, "client_build") == "FAIL"
    assert _status(report, "provider_address") == "FAIL"


# ── key control ──────────────────────────────────────────────────────────────

def test_key_not_controlling_provider_is_no_go(monkeypatch):
    key, _addr = _eth_keypair()
    monkeypatch.setenv("PRSM_ONCHAIN_SETTLEMENT", "1")
    monkeypatch.setenv("FTNS_WALLET_PRIVATE_KEY", key)
    monkeypatch.setenv("PRSM_SETTLEMENT_REGISTRY_ADDRESS", F_REGISTRY)
    other = "0x" + "22" * 20  # NOT the key's address
    report = _run(monkeypatch, provider_address=other)
    assert report.go is False
    assert _status(report, "settler_key") == "FAIL"


# ── retired registry ─────────────────────────────────────────────────────────

def test_retired_registry_is_no_go(monkeypatch):
    addr = _all_green_env(monkeypatch)
    monkeypatch.setenv("PRSM_SETTLEMENT_REGISTRY_ADDRESS", RETIRED_REGISTRY)
    report = _run(monkeypatch, provider_address=addr)
    assert report.go is False
    assert _status(report, "registry_resolved") == "FAIL"


# ── paused registry ──────────────────────────────────────────────────────────

def test_paused_registry_is_no_go(monkeypatch):
    addr = _all_green_env(monkeypatch)
    report = _run(monkeypatch, reader=FakeReader(paused=True), provider_address=addr)
    assert report.go is False
    assert _status(report, "registry_active") == "FAIL"


# ── attestation surface absent (WARN, not FAIL) ──────────────────────────────

def test_missing_attestation_surface_warns_not_blocks(monkeypatch):
    """A registry without commitBatchWithAttestation is still go-able (legacy
    commitBatch); the F surface is a WARN, and with the flag on the flip is held
    OFF by the sp1299 fail-safe (also WARN) — neither blocks go-live."""
    addr = _all_green_env(monkeypatch)
    report = _run(monkeypatch, reader=FakeReader(code=CODE_NO_SEL),
                  provider_address=addr)
    assert _status(report, "attestation_surface") == "WARN"
    assert _status(report, "attestation_flag") == "WARN"
    assert report.go is True  # WARN does not block


# ── unfunded settler (WARN, not FAIL) ────────────────────────────────────────

def test_unfunded_settler_warns_not_blocks(monkeypatch):
    addr = _all_green_env(monkeypatch)
    report = _run(monkeypatch, reader=FakeReader(balance=0), provider_address=addr)
    assert _status(report, "settler_funded") == "WARN"
    assert report.go is True


# ── attestation flag off → INFO (legacy commit) ──────────────────────────────

def test_attestation_flag_off_is_info(monkeypatch):
    addr = _all_green_env(monkeypatch)
    monkeypatch.delenv("PRSM_SETTLEMENT_SUPPORTS_ATTESTATION", raising=False)
    report = _run(monkeypatch, provider_address=addr)
    assert _status(report, "attestation_flag") == "INFO"
    assert report.go is True


# ── no reader → on-chain checks degrade to WARN, never crash ──────────────────

def test_no_reader_degrades_to_warn(monkeypatch):
    addr = _all_green_env(monkeypatch)
    from prsm.settlement.go_live_preflight import run_settlement_go_live_preflight
    # reader=None + a bogus RPC so _Web3Reader builds but reads fail-soft → WARN
    monkeypatch.setenv("PRSM_BASE_RPC_URL", "http://127.0.0.1:1")  # unreachable
    report = run_settlement_go_live_preflight(reader=None, provider_address=addr)
    # registry_active degrades to WARN (paused() unreadable), not FAIL/crash
    assert _status(report, "registry_active") in ("WARN", "PASS")
    # the offline FAILs are unaffected; with all-green offline env it's still go
    assert isinstance(report.to_dict(), dict)


def test_report_to_dict_shape(monkeypatch):
    addr = _all_green_env(monkeypatch)
    d = _run(monkeypatch, provider_address=addr).to_dict()
    assert set(d) == {"go", "findings"}
    assert all(set(f) == {"check", "status", "detail"} for f in d["findings"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
