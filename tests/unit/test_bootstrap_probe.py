"""Sprint 385 — bootstrap probe helper + CLI tests.

The probe helper at prsm/cli_helpers/bootstrap_probe.py
exists so `prsm node bootstrap-test` has a unit-testable
core. Tests here cover:

  - parse_bootstrap_url across input formats
  - HostProbe + FleetProbe serialization + aggregate logic
  - canonical_bootstrap_urls dedups + honors env vars
  - probe_host failure paths (DNS / TCP / TLS / WSS)
  - probe_fleet parallelism
"""
from __future__ import annotations

import asyncio
import os
import socket
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.cli_helpers.bootstrap_probe import (
    FleetProbe,
    HostProbe,
    ProbeStatus,
    canonical_bootstrap_urls,
    parse_bootstrap_url,
    probe_fleet,
    probe_host,
)


@pytest.fixture(autouse=True)
def _restore_node_config_module():
    """sp948 — cross-file leak guard. Two tests here (canonical_urls_*)
    call importlib.reload(prsm.node.config) INSIDE a patch.dict(os.environ,
    {"BOOTSTRAP_PRIMARY": ...}) block to rebuild DEFAULT_BOOTSTRAP_NODES under
    the patched env. reload re-executes the module body, but patch.dict only
    restores os.environ — the already-executed module global stays polluted in
    sys.modules for the rest of the suite, so unrelated tests (e.g. the
    legacy-bootstrap-migration canaries) later read a leaked DEFAULT and fail.
    This teardown reloads the config module after each test; because the
    patch.dict context managers have already exited by teardown, the reload
    rebuilds DEFAULT_BOOTSTRAP_NODES from the real (un-patched) env. The probe
    tests still reload-under-patch and assert override/dedup as before — no
    coverage lost; only the leak is closed."""
    yield
    import importlib
    import prsm.node.config as _cfg
    importlib.reload(_cfg)


# ── parse_bootstrap_url ──────────────────────────────


def test_parse_wss_url_with_port():
    host, port = parse_bootstrap_url(
        "wss://bootstrap-eu.prsm-network.com:8765"
    )
    assert host == "bootstrap-eu.prsm-network.com"
    assert port == 8765


def test_parse_ws_url_with_port():
    host, port = parse_bootstrap_url(
        "ws://localhost:8765"
    )
    assert host == "localhost"
    assert port == 8765


def test_parse_host_port_form():
    host, port = parse_bootstrap_url("example.com:8765")
    assert host == "example.com"
    assert port == 8765


def test_parse_bare_host_defaults_to_8765():
    host, port = parse_bootstrap_url("example.com")
    assert host == "example.com"
    assert port == 8765


def test_parse_wss_url_without_explicit_port_defaults_443():
    """wss:// without explicit port → TLS default 443."""
    host, port = parse_bootstrap_url("wss://example.com")
    assert host == "example.com"
    assert port == 443


# ── HostProbe + FleetProbe serialization ─────────────


def test_host_probe_to_dict_contains_all_fields():
    p = HostProbe(
        url="wss://x:8765", host="x", port=8765,
        status=ProbeStatus.OK,
        tcp_ok=True, tls_ok=True, wss_ok=True,
        latency_ms=42.5,
        cert_subject="x", cert_issuer="Let's Encrypt",
    )
    d = p.to_dict()
    assert d["url"] == "wss://x:8765"
    assert d["status"] == "ok"
    assert d["tcp_ok"] is True
    assert d["latency_ms"] == 42.5
    assert d["cert_issuer"] == "Let's Encrypt"


def test_fleet_probe_healthy_count():
    fleet = FleetProbe(hosts=[
        HostProbe(url="a", host="a", port=8765,
                  status=ProbeStatus.OK),
        HostProbe(url="b", host="b", port=8765,
                  status=ProbeStatus.TCP_FAIL),
        HostProbe(url="c", host="c", port=8765,
                  status=ProbeStatus.OK),
    ])
    assert fleet.healthy_count == 2
    assert fleet.total_count == 3
    assert fleet.all_healthy is False
    assert fleet.any_healthy is True


def test_fleet_probe_all_healthy_when_every_host_ok():
    fleet = FleetProbe(hosts=[
        HostProbe(url="a", host="a", port=8765,
                  status=ProbeStatus.OK),
        HostProbe(url="b", host="b", port=8765,
                  status=ProbeStatus.OK),
    ])
    assert fleet.all_healthy is True


def test_fleet_probe_any_healthy_false_when_all_down():
    fleet = FleetProbe(hosts=[
        HostProbe(url="a", host="a", port=8765,
                  status=ProbeStatus.TCP_FAIL),
        HostProbe(url="b", host="b", port=8765,
                  status=ProbeStatus.DNS_FAIL),
    ])
    assert fleet.any_healthy is False
    assert fleet.all_healthy is False


def test_fleet_probe_empty_treated_as_not_healthy():
    """Empty fleet (no URLs to probe) should NOT report
    all_healthy=True — that would be a misleading default."""
    fleet = FleetProbe(hosts=[])
    assert fleet.all_healthy is False


def test_fleet_probe_to_dict_includes_summary():
    fleet = FleetProbe(hosts=[
        HostProbe(url="a", host="a", port=8765,
                  status=ProbeStatus.OK),
        HostProbe(url="b", host="b", port=8765,
                  status=ProbeStatus.TCP_FAIL),
    ])
    d = fleet.to_dict()
    assert d["summary"]["total"] == 2
    assert d["summary"]["healthy"] == 1
    assert d["summary"]["degraded"] == 1
    assert d["summary"]["all_healthy"] is False
    assert d["summary"]["any_healthy"] is True


# ── canonical_bootstrap_urls ─────────────────────────


def test_canonical_urls_returns_3_default_hosts():
    """Canonical fleet has primary (US) + EU + APAC
    fallbacks defined in prsm/node/config.py."""
    with patch.dict(os.environ, {}, clear=False):
        for v in (
            "BOOTSTRAP_PRIMARY",
            "BOOTSTRAP_FALLBACK_EU",
            "BOOTSTRAP_FALLBACK_APAC",
        ):
            os.environ.pop(v, None)
        # Force reimport so the os.getenv-based defaults
        # re-evaluate against the cleared env.
        import importlib
        import prsm.node.config as cfg
        importlib.reload(cfg)
        urls = canonical_bootstrap_urls()
    # Sprint 575 F29: bootstrap1 → bootstrap-us DNS rename. Primary
    # must be bootstrap-us; EU + APAC fallbacks unchanged. The
    # retired bootstrap1 hostname must NOT appear in defaults.
    assert any(
        "bootstrap-us.prsm-network.com" in u for u in urls
    )
    assert any(
        "bootstrap-eu.prsm-network.com" in u for u in urls
    )
    assert any(
        "bootstrap-apac.prsm-network.com" in u for u in urls
    )
    for u in urls:
        assert "bootstrap1.prsm-network.com" not in u


def test_canonical_urls_honors_env_override():
    """BOOTSTRAP_PRIMARY env var replaces the US default."""
    with patch.dict(
        os.environ,
        {"BOOTSTRAP_PRIMARY": "wss://custom.example.com:8765"},
        clear=False,
    ):
        import importlib
        import prsm.node.config as cfg
        importlib.reload(cfg)
        urls = canonical_bootstrap_urls()
    assert "wss://custom.example.com:8765" in urls


def test_canonical_urls_dedups():
    """If primary == fallback (operator override quirk),
    don't probe the same URL twice."""
    with patch.dict(
        os.environ,
        {
            "BOOTSTRAP_PRIMARY": "wss://same.example.com:8765",
            "BOOTSTRAP_FALLBACK_EU": "wss://same.example.com:8765",
            "BOOTSTRAP_FALLBACK_APAC": "wss://different.example.com:8765",
        },
        clear=False,
    ):
        import importlib
        import prsm.node.config as cfg
        importlib.reload(cfg)
        urls = canonical_bootstrap_urls()
    # Same URL appears once
    assert urls.count("wss://same.example.com:8765") == 1


# ── probe_host failure paths ─────────────────────────


def test_probe_host_dns_fail():
    """DNS resolution failure → status=dns_fail, all layers
    marked not-ok."""
    async def run():
        with patch(
            "asyncio.open_connection",
            new=AsyncMock(side_effect=socket.gaierror("DNS")),
        ):
            return await probe_host(
                "wss://nonexistent.invalid:8765",
                timeout_seconds=1.0,
            )
    result = asyncio.run(run())
    assert result.status == ProbeStatus.DNS_FAIL
    assert result.tcp_ok is False
    assert result.tls_ok is False
    assert result.wss_ok is False
    assert "DNS" in (result.error or "")


def test_probe_host_tcp_fail():
    async def run():
        with patch(
            "asyncio.open_connection",
            new=AsyncMock(
                side_effect=ConnectionRefusedError("refused"),
            ),
        ):
            return await probe_host(
                "wss://x.example.com:8765",
                timeout_seconds=1.0,
            )
    result = asyncio.run(run())
    assert result.status == ProbeStatus.TCP_FAIL
    assert result.tcp_ok is False


def test_probe_host_timeout():
    async def run():
        with patch(
            "asyncio.open_connection",
            new=AsyncMock(
                side_effect=asyncio.TimeoutError,
            ),
        ):
            return await probe_host(
                "wss://slow.example.com:8765",
                timeout_seconds=0.5,
            )
    result = asyncio.run(run())
    assert result.status == ProbeStatus.TIMEOUT
    assert result.tcp_ok is False
    assert "timeout" in (result.error or "").lower()


# ── probe_fleet parallel + aggregate ─────────────────


def test_probe_fleet_empty_urls():
    fleet = asyncio.run(probe_fleet([], timeout_seconds=1.0))
    assert fleet.total_count == 0
    assert fleet.healthy_count == 0


def test_probe_fleet_runs_all_hosts():
    async def fake_probe(url, **kwargs):
        h = HostProbe(
            url=url, host=url, port=8765,
            status=ProbeStatus.OK,
            tcp_ok=True, tls_ok=True, wss_ok=True,
            latency_ms=10.0,
        )
        return h
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_host",
        new=fake_probe,
    ):
        fleet = asyncio.run(probe_fleet(
            ["a", "b", "c"], timeout_seconds=1.0,
        ))
    assert fleet.total_count == 3
    assert fleet.all_healthy is True
    assert [h.url for h in fleet.hosts] == ["a", "b", "c"]
