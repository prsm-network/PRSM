"""Sprint 1410 — the settlement commit thresholds must be operator-configurable.

``AccumulatorConfig`` decides when a provider's accumulated receipts become a committable
batch: 1000 receipts, OR 1 hour, OR 100 FTNS. ``client_wiring`` always constructed
``ReceiptAccumulator()`` with those defaults — there was NO env knob. An operator running
a small node, a canary, or a first ceremony therefore waited an hour for their earnings to
commit, or hand-POSTed ``/admin/settlement/onchain/commit-ready?force=1`` (which is exactly
why sp1407 had to add ``force``).

Failure mode matters here. ``build_onchain_settlement_client_or_none`` swallows ANY
exception into ``logger.debug`` and returns None — i.e. settlement silently OFF. So a
malformed threshold must NEVER raise out of the resolver: it warns loudly and falls back to
the default for that field, and settlement stays ON. Misconfiguring a *tuning knob* must not
disable the money path.

Note on rate-limiting: a low threshold cannot commit a batch per receipt. Commit frequency
is bounded by the settlement poll loop (``PRSM_SETTLEMENT_POLL_INTERVAL_S``, default 600s,
floored at 5s), which is what actually drives ``commit_ready_batches``. So the resolver
enforces only ``AccumulatorConfig``'s own positivity invariants, not an invented floor.
"""
from __future__ import annotations

import pytest

from prsm.settlement import client_wiring
from prsm.settlement.accumulator import AccumulatorConfig


ONE_FTNS = 10**18
DEFAULTS = AccumulatorConfig()


def _resolve(**env):
    return client_wiring._resolve_accumulator_config(env)


# ── resolver ─────────────────────────────────────────────────────────────


def test_no_env_returns_none_so_defaults_are_byte_identical():
    assert _resolve() is None
    assert _resolve(PRSM_SETTLEMENT_COUNT_THRESHOLD="", PRSM_SETTLEMENT_TIME_THRESHOLD_S="  ") is None


def test_count_threshold_from_env():
    cfg = _resolve(PRSM_SETTLEMENT_COUNT_THRESHOLD="5")
    assert cfg.count_threshold == 5
    assert cfg.time_threshold_seconds == DEFAULTS.time_threshold_seconds
    assert cfg.value_threshold_ftns == DEFAULTS.value_threshold_ftns


def test_time_threshold_from_env():
    cfg = _resolve(PRSM_SETTLEMENT_TIME_THRESHOLD_S="120")
    assert cfg.time_threshold_seconds == 120
    assert cfg.count_threshold == DEFAULTS.count_threshold


def test_value_threshold_is_expressed_in_FTNS_and_stored_as_wei():
    """The operator thinks in FTNS; AccumulatorConfig stores wei. Exact, via Decimal —
    a float would make 0.1 FTNS off by a few wei."""
    assert _resolve(PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS="0.5").value_threshold_ftns == 5 * 10**17
    assert _resolve(PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS="100").value_threshold_ftns == 100 * ONE_FTNS
    assert _resolve(PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS="0.1").value_threshold_ftns == 10**17
    # sub-wei precision truncates rather than exploding
    assert _resolve(PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS="1e-18").value_threshold_ftns == 1


def test_all_three_from_env():
    cfg = _resolve(
        PRSM_SETTLEMENT_COUNT_THRESHOLD="3",
        PRSM_SETTLEMENT_TIME_THRESHOLD_S="60",
        PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS="2.5",
    )
    assert (cfg.count_threshold, cfg.time_threshold_seconds, cfg.value_threshold_ftns) \
        == (3, 60, 25 * 10**17)


# ── malformed input: warn + fall back, never raise, never disable settlement ──


def test_malformed_field_is_ignored_and_the_others_still_apply(caplog):
    cfg = _resolve(PRSM_SETTLEMENT_COUNT_THRESHOLD="banana", PRSM_SETTLEMENT_TIME_THRESHOLD_S="90")
    assert cfg.time_threshold_seconds == 90
    assert cfg.count_threshold == DEFAULTS.count_threshold   # default, not crashed
    assert "PRSM_SETTLEMENT_COUNT_THRESHOLD" in caplog.text


def test_malformed_value_threshold_is_ignored():
    cfg = _resolve(PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS="not-a-number",
                   PRSM_SETTLEMENT_COUNT_THRESHOLD="7")
    assert cfg.count_threshold == 7
    assert cfg.value_threshold_ftns == DEFAULTS.value_threshold_ftns


@pytest.mark.parametrize("env", [
    {"PRSM_SETTLEMENT_COUNT_THRESHOLD": "0"},
    {"PRSM_SETTLEMENT_COUNT_THRESHOLD": "-1"},
    {"PRSM_SETTLEMENT_TIME_THRESHOLD_S": "0"},
    {"PRSM_SETTLEMENT_TIME_THRESHOLD_S": "-30"},
    {"PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS": "0"},
    {"PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS": "-2"},
])
def test_non_positive_thresholds_fall_back_to_defaults_without_raising(env, caplog):
    """AccumulatorConfig.__post_init__ rejects these. The resolver must catch that:
    an escaping ValueError would be swallowed by the builder as 'settlement OFF'."""
    cfg = _resolve(**env)
    assert cfg is None                       # all defaults
    assert caplog.text                       # and it said so


def test_resolver_never_raises_on_hostile_input():
    for env in ({"PRSM_SETTLEMENT_COUNT_THRESHOLD": "1e999"},
                {"PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS": "nan"},
                {"PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS": "inf"},
                {"PRSM_SETTLEMENT_TIME_THRESHOLD_S": "1.5.2"},
                {"PRSM_SETTLEMENT_COUNT_THRESHOLD": "0x10"}):
        client_wiring._resolve_accumulator_config(env)   # must not raise


# ── builder integration ──────────────────────────────────────────────────


def _build(env, **kw):
    base = {"PRSM_ONCHAIN_SETTLEMENT": "1", "PRSM_SETTLEMENT_STATE_FILE": ":memory:"}
    base.update(env)
    return client_wiring.build_onchain_settlement_client_or_none(
        provider_address="0x" + "11" * 20, env=base, **kw)


def test_env_thresholds_reach_the_live_clients_accumulator():
    client = _build({"PRSM_SETTLEMENT_COUNT_THRESHOLD": "2",
                     "PRSM_SETTLEMENT_TIME_THRESHOLD_S": "30"})
    assert client is not None
    cfg = client._accumulator.config
    assert cfg.count_threshold == 2
    assert cfg.time_threshold_seconds == 30


def test_default_build_is_unchanged():
    client = _build({})
    assert client._accumulator.config == DEFAULTS


def test_explicit_accumulator_config_wins_over_env():
    """The per-stage commit path (sp1329) passes count_threshold=1 deliberately; an env
    var must not override an explicit programmatic choice."""
    client = _build({"PRSM_SETTLEMENT_COUNT_THRESHOLD": "999"},
                    accumulator_config=AccumulatorConfig(count_threshold=1))
    assert client._accumulator.config.count_threshold == 1


def test_garbage_env_does_not_disable_settlement():
    """THE safety property: a typo'd tuning knob must not silently turn the money path off."""
    client = _build({"PRSM_SETTLEMENT_COUNT_THRESHOLD": "banana",
                     "PRSM_SETTLEMENT_TIME_THRESHOLD_S": "-5"})
    assert client is not None, "a malformed threshold silently disabled on-chain settlement"
    assert client._accumulator.config == DEFAULTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
