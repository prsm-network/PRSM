"""Sprint 1491 — one unrecognised PRSM_* env var must not disable ALL config.

FOUND LIVE on the sfo operator. PRSMConfig is extra="forbid" (right, for typos),
but the env loader maps every PRSM_FOO_BAR into a nested key — so setting any env
var the schema does not model made the WHOLE PRSMConfig fail to construct, and
get_config() then returned None PROCESS-WIDE. Of its callers, exactly one guards
against None.

The trigger was self-inflicted: the sp1482 marketplace drop-in
(PRSM_MARKETPLACE_ADVERTISE etc.) meant every get_config() on that node returned
None while the node otherwise looked perfectly healthy — readyz 200, no traceback.

The operator's real problem is not the exception. It is not knowing their setting
did nothing. So the fix drops only the rejected keys, keeps everything else, and
NAMES the ignored keys in the log.
"""
from __future__ import annotations

import logging

import pytest

from prsm.core.config.manager import (
    _build_config_dropping_unknown_keys,
    _prune_keys,
)


def test_a_clean_config_is_unaffected():
    cfg = _build_config_dropping_unknown_keys({})
    assert cfg is not None


def test_an_unknown_key_no_longer_takes_down_the_whole_config(caplog):
    """★ THE regression. Previously this raised, get_config() returned None, and
    every config-driven behaviour in the process silently lost its settings."""
    with caplog.at_level(logging.WARNING):
        cfg = _build_config_dropping_unknown_keys(
            {"marketplace": {"advertise": True}})
    assert cfg is not None, "one unknown key must not disable all config"


def test_the_ignored_keys_are_NAMED_in_the_log(caplog):
    """★ The point of the fix. A setting that silently does nothing is the actual
    harm — the operator must be told which one."""
    with caplog.at_level(logging.WARNING):
        _build_config_dropping_unknown_keys({
            "marketplace": {"advertise": True, "ttl": {"seconds": 300}},
        })
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "IGNORED" in msg
    assert "NO EFFECT" in msg
    assert "marketplace.advertise" in msg
    assert "marketplace.ttl" in msg


def test_the_exact_live_sfo_config_loads(caplog):
    """The real drop-in that broke the live node — all four keys."""
    with caplog.at_level(logging.WARNING):
        cfg = _build_config_dropping_unknown_keys({
            "marketplace": {
                "advertise": True,
                "price": {"per": {"shard": {"ftns": 0.25}}},
                "capacity": {"shards": {"per": {"sec": 1.0}}},
                "ttl": {"seconds": 300},
            },
        })
    assert cfg is not None
    msg = " ".join(r.getMessage() for r in caplog.records)
    for k in ("marketplace.advertise", "marketplace.price",
              "marketplace.capacity", "marketplace.ttl"):
        assert k in msg, f"{k} must be reported as ignored"


def test_known_settings_SURVIVE_alongside_an_unknown_one():
    """Dropping must be surgical — an unrelated bad key must not silently discard
    the operator's real settings."""
    clean = _build_config_dropping_unknown_keys({})
    env = getattr(clean, "environment", None)

    mixed = _build_config_dropping_unknown_keys(
        {"marketplace": {"advertise": True}})
    assert getattr(mixed, "environment", None) == env


def test_a_REAL_validation_error_is_still_RAISED():
    """★ Only extra_forbidden is tolerated. A KNOWN field with an unusable value is
    a genuine misconfiguration — masking it would hide a real operator mistake
    behind a config that looks fine. api.port is a real int field."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _build_config_dropping_unknown_keys({"api": {"port": "not-a-number"}})


def test_a_real_error_is_raised_even_ALONGSIDE_an_unknown_key():
    """The prune path must not become a way to swallow genuine errors just because
    an unrelated unknown key is also present."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _build_config_dropping_unknown_keys({
            "api": {"port": "not-a-number"},
            "marketplace": {"advertise": True},
        })


# ── the pruner itself ───────────────────────────────────────────────

def test_prune_removes_only_the_named_path():
    data = {"a": {"b": 1, "c": 2}, "d": 3}
    out = _prune_keys(data, ["a.b"])
    assert out == {"a": {"c": 2}, "d": 3}


def test_prune_does_not_mutate_the_input():
    data = {"a": {"b": 1}}
    _prune_keys(data, ["a.b"])
    assert data == {"a": {"b": 1}}, "input must not be mutated"


def test_prune_tolerates_a_missing_path():
    assert _prune_keys({"a": 1}, ["x.y.z"]) == {"a": 1}


def test_prune_handles_a_top_level_key():
    assert _prune_keys({"a": 1, "b": 2}, ["a"]) == {"b": 2}


# ── the manager actually uses it ────────────────────────────────────

def test_load_config_ACTUALLY_routes_through_the_tolerant_builder():
    """★ Binding test. A tolerant builder nothing calls is worthless — load_config
    would still construct PRSMConfig directly and return None process-wide.
    Verified necessary: reverting the manager to `PRSMConfig(**config_data)` left
    every other test in this file green."""
    import inspect

    from prsm.core.config import manager as mgr

    src = inspect.getsource(mgr.ConfigManager.load_config)
    assert "_build_config_dropping_unknown_keys(config_data)" in src
    assert "PRSMConfig(**config_data)" not in src


def test_get_config_returns_a_config_despite_an_unknown_env_key(monkeypatch):
    """★ END TO END, through the real public entry point: the live failure was
    get_config() returning None. Set an unrecognised PRSM_* var and it must still
    hand back a usable config."""
    from prsm.core.config import manager as mgr

    monkeypatch.setenv("PRSM_MARKETPLACE_ADVERTISE", "true")
    monkeypatch.setenv("PRSM_MARKETPLACE_TTL_SECONDS", "300")
    mgr.get_config_manager.cache_clear()
    mgr._config_manager = None
    try:
        cfg = mgr.get_config()
        assert cfg is not None, (
            "get_config() returned None because of an unrelated env var — this is "
            "the exact live sfo failure this sprint fixes")
    finally:
        mgr.get_config_manager.cache_clear()
        mgr._config_manager = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
