"""Sprint 1268 — SSRF guard on the Ollama connector base_url (audit round 5, HIGH).

OllamaConnector took base_url verbatim from user-supplied config.custom_settings["base_url"]
and issued every request to it, with no scheme/host validation. Combined with the
(separately-tracked) unauthenticated /integrations/* router, an attacker could register an
Ollama connector pointed at cloud metadata (169.254.169.254) or internal services and turn
the node into an SSRF proxy / internal port scanner.

Fix: assert_safe_outbound_url validates base_url — http/https only, and ALWAYS rejects
link-local (incl. 169.254.169.254 cloud metadata), reserved, multicast, and unspecified
addresses; private LAN is rejected unless explicitly allowlisted; loopback (the legit local
Ollama default) is allowed. Validated at connector construction AND before each request
(re-resolve → defeats DNS rebinding).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.core.integrations.connectors.ollama_connector import assert_safe_outbound_url


def _addrinfo(ip):
    # shape of socket.getaddrinfo entries: (family, type, proto, canonname, sockaddr)
    return [(2, 1, 6, "", (ip, 0))]


def test_loopback_allowed_by_default():
    # the default local Ollama (http://localhost:11434 / 127.0.0.1) must work
    assert_safe_outbound_url("http://127.0.0.1:11434")
    with patch("socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")):
        assert_safe_outbound_url("http://localhost:11434")


def test_cloud_metadata_blocked():
    with pytest.raises(ValueError, match="blocked|internal|link-local|metadata"):
        assert_safe_outbound_url("http://169.254.169.254/latest/meta-data/")


@pytest.mark.parametrize("ip", ["10.0.0.5", "192.168.1.10", "172.16.0.1"])
def test_private_lan_blocked_by_default(ip):
    with patch("socket.getaddrinfo", return_value=_addrinfo(ip)):
        with pytest.raises(ValueError):
            assert_safe_outbound_url(f"http://internal-host:11434")


def test_private_lan_allowed_when_allowlisted(monkeypatch):
    monkeypatch.setenv("PRSM_CONNECTOR_ALLOW_INTERNAL_HOSTS", "internal-host,other")
    with patch("socket.getaddrinfo", return_value=_addrinfo("10.0.0.5")):
        assert_safe_outbound_url("http://internal-host:11434")  # no raise


def test_non_http_scheme_blocked():
    with pytest.raises(ValueError, match="scheme"):
        assert_safe_outbound_url("file:///etc/passwd")
    with pytest.raises(ValueError, match="scheme"):
        assert_safe_outbound_url("gopher://x/")


def test_unresolvable_host_blocked():
    import socket
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("nope")):
        with pytest.raises(ValueError):
            assert_safe_outbound_url("http://does-not-exist.invalid/")


def test_dns_rebind_any_resolved_private_blocks():
    # if a host resolves to BOTH a public and a private IP, the private one must block it
    infos = _addrinfo("8.8.8.8") + _addrinfo("10.0.0.5")
    with patch("socket.getaddrinfo", return_value=infos):
        with pytest.raises(ValueError):
            assert_safe_outbound_url("http://rebind.example/")


def test_ollama_connector_rejects_metadata_base_url():
    from prsm.core.integrations.connectors.ollama_connector import OllamaConnector
    from prsm.core.integrations.models.integration_models import ConnectorConfig, IntegrationPlatform
    cfg = ConnectorConfig(
        platform=IntegrationPlatform.OLLAMA,
        user_id="attacker",
        custom_settings={"base_url": "http://169.254.169.254/latest/meta-data/"},
    )
    with pytest.raises(ValueError):
        OllamaConnector(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
