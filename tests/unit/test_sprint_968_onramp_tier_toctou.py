"""Sprint 968 — close the onramp tier rolling-total TOCTOU.

Phase-5 fiat review finding #2 (MEDIUM). The AML rolling-total (KYC tier limit)
counted only `onramp_execute`/`offramp_execute` ring entries, and `onramp_execute`
was recorded ONLY at CONFIRMED (during the on-chain sweep, minutes-to-hours after
the user hits /wallet/onramp/execute). The execute endpoint recorded to the FUNNEL,
not the ring. So a user could fire /wallet/onramp/execute repeatedly; each
tier-check saw the same stale total (no in-flight onramps) and every session
passed — bypassing the tier limit.

Fix:
1. total_usd_for_user dedups by metadata.intent_id (latest entry per intent wins),
   so an execute-time PENDING reservation + the later CONFIRMED settle for the
   SAME onramp count once (no double-count / over-restriction).
2. The onramp execute path records an `onramp_execute` PENDING reservation (with
   the intent_id + expected USD) at execute-time, so the next tier-check sees the
   in-flight onramp.

(The remaining concurrent-same-instant + cross-process-replica race needs a
shared/atomic store — the in-memory ring is per-process — documented follow-on.)
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.economy.web3.fiat_compliance_ring import FiatComplianceRing
from prsm.economy.web3.kyc_client import KYCClient
from prsm.node.api import create_api_app

_NOW = time.time()  # recent base so entries fall inside the 24h rolling window


def _ring():
    return FiatComplianceRing()


def _rec(ring, *, user="u1", usd, status, intent_id=None, ts):
    ring.record(
        kind="onramp_execute", user_id=user, usd_amount=usd, ftns_amount=0.0,
        status=status, metadata=({"intent_id": intent_id} if intent_id else {}),
        timestamp=ts,
    )


# ── dedup-by-intent_id (no double-count of reservation + settle) ───────────


def test_same_intent_pending_then_confirmed_counts_once_latest():
    ring = _ring()
    _rec(ring, usd=600.0, status="PENDING", intent_id="i1", ts=_NOW - 200)
    _rec(ring, usd=620.0, status="CONFIRMED", intent_id="i1", ts=_NOW - 100)  # actual settle
    # One onramp → counts ONCE, the latest (settled) amount, not 1220.
    assert ring.total_usd_for_user("u1") == 620.0


def test_distinct_intents_are_summed():
    ring = _ring()
    _rec(ring, usd=600.0, status="PENDING", intent_id="i1", ts=_NOW - 200)
    _rec(ring, usd=700.0, status="PENDING", intent_id="i2", ts=_NOW - 150)
    assert ring.total_usd_for_user("u1") == 1300.0


def test_entries_without_intent_id_are_summed_each():
    """Offramp / legacy entries carry no intent_id and must each count (the
    dedup only collapses entries sharing an intent_id)."""
    ring = _ring()
    ring.record(kind="offramp_execute", user_id="u1", usd_amount=400.0,
                ftns_amount=0.0, status="CONFIRMED", timestamp=_NOW - 200)
    ring.record(kind="offramp_execute", user_id="u1", usd_amount=500.0,
                ftns_amount=0.0, status="CONFIRMED", timestamp=_NOW - 150)
    assert ring.total_usd_for_user("u1") == 900.0


def test_in_flight_pending_reservation_counts_toward_limit():
    """The core TOCTOU fix: a PENDING (not-yet-confirmed) onramp reservation
    counts immediately, so a second execute sees it."""
    ring = _ring()
    _rec(ring, usd=800.0, status="PENDING", intent_id="i1", ts=_NOW - 200)
    # No CONFIRMED yet — still counts (was 0.0 before the fix).
    assert ring.total_usd_for_user("u1") == 800.0


def test_window_still_applies_to_deduped_intents():
    ring = _ring()
    _rec(ring, usd=900.0, status="CONFIRMED", intent_id="old", ts=_NOW - 90000)  # >24h
    _rec(ring, usd=300.0, status="PENDING", intent_id="new", ts=_NOW - 10)
    assert ring.total_usd_for_user("u1") == 300.0


# ── endpoint: execute-time reservation closes the sequential TOCTOU ─────────


def _onramp_execute_client(monkeypatch, ring):
    """A TestClient whose onramp-execute path actually reaches the session mint
    (onramp session client mocked) + records to a REAL compliance ring."""
    class _FakeWaasBackend:
        def create_wallet(self, user_id, email):
            return {"wallet_id": f"w-{user_id}", "address": "0x" + "11" * 20,
                    "network": "base-mainnet"}

    class _FakeKYCBackend:
        def initiate_session(self, user_id, email, level):
            return {"vendor_ref": f"p-{user_id}",
                    "session_url": f"https://p.example/{user_id}",
                    "status": "INITIATED"}

    waas = CoinbaseWaaSClient(api_key_name="k", api_key_private="p",
                              backend=_FakeWaasBackend())
    waas.provision_wallet(user_id="alice", email="a@x.io")
    kyc = KYCClient(vendor="persona", api_key="k", backend=_FakeKYCBackend())
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    kyc.update_status("alice", "VERIFIED")

    class _FakeOnrampSession:
        def build_buy_url(self, address, usd_amount, partner_user_ref=None):
            return {"token": "tok-123", "session_url": "https://pay.coinbase/x",
                    "channel_id": "ch-1"}
        def close(self):
            pass

    monkeypatch.setattr(
        "prsm.economy.web3.coinbase_onramp_session.from_env",
        lambda: _FakeOnrampSession(),
    )

    node = MagicMock()
    node.identity.node_id = "test-node"
    node._coinbase_waas_client = waas
    node._kyc_client = kyc
    node._aerodrome_client = None
    node._fiat_compliance_ring = ring
    node._onramp_funnel = None  # let the endpoint create one
    return TestClient(create_api_app(node, enable_security=False),
                      raise_server_exceptions=False)


def test_execute_records_pending_reservation_and_second_execute_blocked(monkeypatch):
    """basic tier limit is $1000. First $700 onramp mints a session + records a
    PENDING reservation; the second $700 onramp's tier-check now sees the
    in-flight $700 → blocked (was bypassable before sp968)."""
    ring = _ring()
    client = _onramp_execute_client(monkeypatch, ring)

    r1 = client.post("/wallet/onramp/execute",
                     json={"usd_amount": 700.0, "destination_user_id": "alice"})
    assert r1.status_code == 200, r1.text
    assert r1.json()["status"] == "SESSION_READY"
    # The in-flight onramp is now counted.
    assert ring.total_usd_for_user("alice") == 700.0

    r2 = client.post("/wallet/onramp/execute",
                     json={"usd_amount": 700.0, "destination_user_id": "alice"})
    assert r2.status_code == 403
    assert r2.json()["detail"]["error"] == "tier_limit_exceeded"
