"""Sprint 871 — onramp→swap auto-orchestrator pin tests."""
from __future__ import annotations

import pytest

from prsm.economy.web3.onramp_to_swap_orchestrator import (
    build_envelope_for_intent,
    make_on_confirmed_callback,
)
from prsm.economy.web3.onramp_funnel import (
    OnrampFunnel,
    OnrampIntent,
    STATUS_CONFIRMED,
    STATUS_PENDING_SETTLEMENT,
)


class _FakeIntent:
    """Lightweight intent shim (full OnrampIntent dataclass not
    needed for the orchestrator's read-only access pattern)."""
    def __init__(
        self, *, intent_id="ix_1", user_id="alice",
        destination_address="0x" + "11" * 20,
        usdc_received=4.92, expected_usd=5.0,
    ):
        self.intent_id = intent_id
        self.user_id = user_id
        self.destination_address = destination_address
        self.usdc_received = usdc_received
        self.expected_usd = expected_usd
        self.swap_envelope = None


class _FakeQuote:
    """Mirrors AerodromeQuote duck type."""
    def __init__(
        self, *, amount_out_units=100 * 10**18,
        price_impact_bps=10, fee_bps=30,
    ):
        self.amount_out = amount_out_units
        self.price_impact_bps = price_impact_bps
        self.fee_bps = fee_bps
        self.route = "aerodrome"


class _FakeAerodrome:
    def __init__(
        self, *, configured=True, quote=None, raises=False,
    ):
        self._configured = configured
        self._quote = quote or _FakeQuote()
        self._raises = raises

    def is_configured(self): return self._configured

    def quote_swap(self, *, amount_in, token_in, **_):
        if self._raises:
            raise RuntimeError("simulated pool RPC down")
        return self._quote


# ── Pool-state gating ────────────────────────────────────────

def test_returns_none_when_aerodrome_client_none():
    r = build_envelope_for_intent(
        _FakeIntent(),
        aerodrome_client=None,
        ftns_address="0x" + "ff" * 20,
    )
    assert r is None


def test_returns_none_when_pool_not_configured():
    """Pool ceremony pending — envelope build deferred."""
    r = build_envelope_for_intent(
        _FakeIntent(),
        aerodrome_client=_FakeAerodrome(configured=False),
        ftns_address="0x" + "ff" * 20,
    )
    assert r is None


def test_returns_none_when_usdc_received_zero():
    """If usdc_received=0 (intent transitioning straight from
    INTENT_RECORDED without real arrival), don't build a quote
    for 0 USDC — meaningless."""
    intent = _FakeIntent(usdc_received=0)
    r = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ff" * 20,
    )
    assert r is None


def test_returns_none_when_quote_swap_raises():
    """RPC down or malformed pool — log + return None, sweep
    callback won't propagate the error."""
    intent = _FakeIntent()
    r = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(raises=True),
        ftns_address="0x" + "ff" * 20,
    )
    assert r is None


# ── Envelope shape ───────────────────────────────────────────

def test_envelope_has_canonical_fields():
    intent = _FakeIntent(usdc_received=4.92)
    env = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ff" * 20,
    )
    assert env is not None
    assert env["status"] == "READY_FOR_SUBMISSION"
    assert env["function"] == "swapExactTokensForTokens"
    assert env["intent_id"] == "ix_1"
    assert env["user_id"] == "alice"


def test_envelope_amount_in_units_from_usdc_received():
    """amount_in_units = usdc_received × 10^6 (6 decimals)."""
    intent = _FakeIntent(usdc_received=4.92)
    env = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ff" * 20,
    )
    assert env["args"]["amountIn"] == int(4.92 * 10**6)
    assert env["quote"]["amount_in_usdc"] == 4.92


def test_envelope_uses_received_not_expected():
    """Critical: if the user expected $5 but only received $4.92
    (Coinbase 1.5% spread), the swap envelope must use $4.92,
    not $5 — otherwise the swap reverts with INSUFFICIENT_BALANCE."""
    intent = _FakeIntent(
        usdc_received=4.92, expected_usd=5.0,
    )
    env = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ff" * 20,
    )
    # 4.92 USDC → 4_920_000 base units, NOT 5_000_000
    assert env["args"]["amountIn"] == 4_920_000


def test_envelope_amount_out_min_honors_slippage():
    """1% slippage default → amount_out_min = amount_out × 0.99."""
    quote = _FakeQuote(amount_out_units=100 * 10**18)
    intent = _FakeIntent(usdc_received=4.92)
    env = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(quote=quote),
        ftns_address="0x" + "ff" * 20,
    )
    expected_min = (100 * 10**18) * (10_000 - 100) // 10_000
    assert env["args"]["amountOutMin"] == expected_min
    assert env["quote"]["amount_out_min_units"] == expected_min


def test_envelope_amount_out_min_custom_slippage():
    """Higher slippage allowed for thin-liquidity pools."""
    quote = _FakeQuote(amount_out_units=100 * 10**18)
    intent = _FakeIntent(usdc_received=4.92)
    env = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(quote=quote),
        ftns_address="0x" + "ff" * 20,
        slippage_bps=500,  # 5%
    )
    expected_min = (100 * 10**18) * (10_000 - 500) // 10_000
    assert env["args"]["amountOutMin"] == expected_min
    assert env["quote"]["slippage_bps"] == 500


def test_envelope_routes_volatile_pool():
    """USDC↔FTNS goes through a volatile (not stable) pool —
    pool ceremony will deploy as volatile."""
    intent = _FakeIntent(usdc_received=4.92)
    env = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ff" * 20,
    )
    routes = env["args"]["routes"]
    assert len(routes) == 1
    assert routes[0]["from_token"] == "USDC"
    assert routes[0]["to_token"] == "FTNS"
    assert routes[0]["stable"] is False


def test_envelope_deadline_is_24h_in_future():
    import time
    intent = _FakeIntent(usdc_received=4.92)
    env = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ff" * 20,
    )
    now = int(time.time())
    assert env["args"]["deadline"] > now
    assert env["args"]["deadline"] - now > 86_000  # ~24h


def test_envelope_router_canonical_address():
    """Pin: must match sp855's canonical Aerodrome v2 router."""
    intent = _FakeIntent(usdc_received=4.92)
    env = build_envelope_for_intent(
        intent, aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ff" * 20,
    )
    assert env["router_address"] == (
        "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
    )


# ── on_confirmed callback integration ────────────────────────

def test_callback_attaches_envelope_to_intent(tmp_path):
    """End-to-end: funnel sweep → CONFIRMED → callback fires →
    intent.swap_envelope populated → persisted to disk."""
    funnel = OnrampFunnel(persist_dir=tmp_path)
    rec = funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "11" * 20,
        expected_usd=5.0,
        session_token="tk",
    )

    class _FakeBal:
        def __init__(self, usdc): self.usdc = usdc; self.usdc_units = int(usdc * 1e6)

    class _FakeReader:
        def get_balances(self, addr):
            return _FakeBal(usdc=4.92)  # arrival triggers CONFIRMED

    cb = make_on_confirmed_callback(
        funnel=funnel,
        aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ff" * 20,
    )
    summary = funnel.sweep(
        balance_reader=_FakeReader(), on_confirmed=cb,
    )
    assert summary["confirmed_new"] == 1
    confirmed = funnel.get_intent(rec.intent_id)
    assert confirmed.status == STATUS_CONFIRMED
    assert confirmed.swap_envelope is not None
    assert (
        confirmed.swap_envelope["status"] == "READY_FOR_SUBMISSION"
    )
    assert confirmed.swap_envelope["intent_id"] == rec.intent_id


def test_callback_fail_soft_when_pool_unconfigured(tmp_path):
    """Pool ceremony pending — CONFIRMED transition still happens,
    swap_envelope just stays None until next sweep retries."""
    funnel = OnrampFunnel(persist_dir=tmp_path)
    rec = funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="tk",
    )

    class _FakeBal:
        usdc = 4.92; usdc_units = 4_920_000

    class _Reader:
        def get_balances(self, addr): return _FakeBal()

    cb = make_on_confirmed_callback(
        funnel=funnel,
        aerodrome_client=_FakeAerodrome(configured=False),
        ftns_address="0x" + "ff" * 20,
    )
    funnel.sweep(balance_reader=_Reader(), on_confirmed=cb)
    confirmed = funnel.get_intent(rec.intent_id)
    assert confirmed.status == STATUS_CONFIRMED  # still flipped
    assert confirmed.swap_envelope is None  # but envelope deferred


def test_callback_fail_soft_when_callback_raises(tmp_path):
    """Even if the callback itself raises (e.g., bug in envelope
    builder), funnel CONFIRMED transition must hold — operator
    can investigate the failed envelope build separately."""
    funnel = OnrampFunnel(persist_dir=tmp_path)
    rec = funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="tk",
    )

    class _FakeBal:
        usdc = 4.92; usdc_units = 4_920_000

    class _Reader:
        def get_balances(self, addr): return _FakeBal()

    def _bad_callback(intent):
        raise RuntimeError("simulated callback bug")

    summary = funnel.sweep(
        balance_reader=_Reader(), on_confirmed=_bad_callback,
    )
    # CONFIRMED still counted, transition still applied
    assert summary["confirmed_new"] == 1
    assert funnel.get_intent(rec.intent_id).status == (
        STATUS_CONFIRMED
    )


# ── Sp1176: on_expired terminal compliance record ────────────

def _zero_balance_reader():
    class _ZeroBal:
        usdc = 0.0
        usdc_units = 0

    class _Reader:
        def get_balances(self, addr):
            return _ZeroBal()
    return _Reader()


def test_on_expired_records_terminal_entry_and_clears_rolling_total(
    tmp_path,
):
    """Abandoned onramp → sweep past 24h → EXPIRED. The on_expired
    callback writes a terminal EXPIRED $0 onramp_execute entry for
    the same intent_id, so (a) the audit trail is complete and (b)
    the sp968 PENDING reservation stops counting against the AML
    rolling total (dedup-by-intent, latest-wins)."""
    from prsm.economy.web3.fiat_compliance_ring import (
        FiatComplianceRing,
    )
    from prsm.economy.web3.onramp_funnel import (
        _EXPIRY_SECONDS, STATUS_EXPIRED,
    )
    from prsm.economy.web3.onramp_to_swap_orchestrator import (
        make_on_expired_callback,
    )

    ring = FiatComplianceRing()
    funnel = OnrampFunnel(persist_dir=tmp_path)
    rec = funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "11" * 20,
        expected_usd=500.0, session_token="tk",
    )
    # Mirror the api.py sp968 PENDING reservation at session mint.
    ring.record(
        kind="onramp_execute", user_id="alice", usd_amount=500.0,
        ftns_amount=0.0, status="PENDING",
        metadata={"intent_id": rec.intent_id},
    )
    assert ring.total_usd_for_user("alice") == 500.0  # reserved

    on_expired = make_on_expired_callback(compliance_ring=ring)
    summary = funnel.sweep(
        balance_reader=_zero_balance_reader(),
        on_expired=on_expired,
        now=rec.created_at + _EXPIRY_SECONDS + 1,  # past the window
    )
    assert summary["expired_new"] == 1
    assert funnel.get_intent(rec.intent_id).status == STATUS_EXPIRED

    expired = [
        e for e in ring.recent(limit=100, user_id="alice")
        if e.status == "EXPIRED"
    ]
    assert len(expired) == 1
    assert expired[0].usd_amount == 0.0
    assert expired[0].metadata["intent_id"] == rec.intent_id
    # The abandoned reservation no longer counts against the limit.
    assert ring.total_usd_for_user("alice") == 0.0


def test_on_expired_fires_at_most_once(tmp_path):
    """A second sweep does NOT re-record EXPIRED: the intent is
    already terminal, so the loop skips it (the at-most-once
    guarantee, same as the sp891 CONFIRMED path)."""
    from prsm.economy.web3.fiat_compliance_ring import (
        FiatComplianceRing,
    )
    from prsm.economy.web3.onramp_funnel import _EXPIRY_SECONDS
    from prsm.economy.web3.onramp_to_swap_orchestrator import (
        make_on_expired_callback,
    )

    ring = FiatComplianceRing()
    funnel = OnrampFunnel(persist_dir=tmp_path)
    rec = funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "11" * 20,
        expected_usd=500.0, session_token="tk",
    )
    on_expired = make_on_expired_callback(compliance_ring=ring)
    when = rec.created_at + _EXPIRY_SECONDS + 1
    funnel.sweep(
        balance_reader=_zero_balance_reader(),
        on_expired=on_expired, now=when,
    )
    summary2 = funnel.sweep(
        balance_reader=_zero_balance_reader(),
        on_expired=on_expired, now=when + 5,
    )
    assert summary2["expired_new"] == 0  # already terminal, skipped
    expired = [
        e for e in ring.recent(limit=100, user_id="alice")
        if e.status == "EXPIRED"
    ]
    assert len(expired) == 1  # still exactly one terminal record


def test_on_expired_fail_soft_when_callback_raises(tmp_path):
    """A misbehaving on_expired callback must NOT undo the EXPIRED
    transition (mirrors the on_confirmed fail-soft contract)."""
    from prsm.economy.web3.onramp_funnel import (
        _EXPIRY_SECONDS, STATUS_EXPIRED,
    )
    funnel = OnrampFunnel(persist_dir=tmp_path)
    rec = funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "11" * 20,
        expected_usd=500.0, session_token="tk",
    )

    def _bad(intent):
        raise RuntimeError("simulated on_expired bug")

    summary = funnel.sweep(
        balance_reader=_zero_balance_reader(),
        on_expired=_bad,
        now=rec.created_at + _EXPIRY_SECONDS + 1,
    )
    assert summary["expired_new"] == 1
    assert funnel.get_intent(rec.intent_id).status == STATUS_EXPIRED


def test_make_on_expired_callback_noop_when_ring_none():
    """No compliance ring wired → the callback is a safe no-op."""
    from prsm.economy.web3.onramp_to_swap_orchestrator import (
        make_on_expired_callback,
    )
    cb = make_on_expired_callback(compliance_ring=None)
    cb(_FakeIntent())  # must not raise


def test_intent_dataclass_includes_swap_envelope_field():
    """The dataclass roundtrip preserves swap_envelope."""
    intent = OnrampIntent(
        intent_id="x", user_id="a",
        destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="tk", created_at=0.0,
        swap_envelope={"status": "READY_FOR_SUBMISSION"},
    )
    d = intent.to_dict()
    assert d["swap_envelope"]["status"] == "READY_FOR_SUBMISSION"
    rebuilt = OnrampIntent.from_dict(d)
    assert rebuilt.swap_envelope == intent.swap_envelope


def test_intent_dataclass_defaults_envelope_none():
    """Existing on-disk records (sp857 era) without swap_envelope
    must load cleanly with envelope = None."""
    d = {
        "intent_id": "x", "user_id": "a",
        "destination_address": "0x" + "11" * 20,
        "expected_usd": 5.0, "session_token": "tk",
        "created_at": 0.0,
        # No swap_envelope field
    }
    intent = OnrampIntent.from_dict(d)
    assert intent.swap_envelope is None
