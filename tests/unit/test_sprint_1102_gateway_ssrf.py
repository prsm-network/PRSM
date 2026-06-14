"""Sprint 1102 — fold the Domain-06 review HIGH (WIRED): SSRF in the gateway fetch path.

A malicious peer that gossiped a provider advertisement for a CID can reply with a
``ContentResponseMessage`` carrying transfer_mode=GATEWAY and a ``gateway_url`` pointing
at an INTERNAL address (http://169.254.169.254/... AWS metadata, http://127.0.0.1:9944/
a local admin/RPC). ContentProvider._fetch_from_url only checked the URL SCHEME, then
issued the GET — so the retriever's node made a request to an attacker-chosen internal
host. The CID/anchor check rejects the returned BYTES, but the SSRF request (and any
leak via timing/error) already fired.

Fix: validate the gateway URL before fetching — reject any host that resolves to a
private / loopback / link-local / reserved / multicast / unspecified IP, pin the
connection to the validated IP(s) (DNS-rebinding defense), and disable redirects (so a
public URL can't 302 to an internal one).
"""
from __future__ import annotations

import pytest

from prsm.node.content_provider import _classify_gateway_url


def _fake_resolver(ip):
    """Return a getaddrinfo-shaped result resolving any host to `ip`."""
    import socket
    fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
    def _resolve(host, port, *a, **k):
        return [(fam, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]
    return _resolve


# ── numeric internal IPs are blocked (no DNS needed) ─────────────────────────────

@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",  # AWS metadata
    "http://127.0.0.1:9944/",          # loopback admin/RPC
    "http://[::1]/x",                  # IPv6 loopback
    "http://10.0.0.5/x",              # private
    "http://192.168.1.1/x",          # private
    "http://172.16.5.5/x",           # private
    "http://0.0.0.0/x",              # unspecified
])
def test_internal_ip_urls_blocked(url):
    safe, reason, ips = _classify_gateway_url(url)
    assert safe is False, f"{url} should be blocked, got {reason}"


def test_non_http_scheme_blocked():
    assert _classify_gateway_url("file:///etc/passwd")[0] is False
    assert _classify_gateway_url("gopher://127.0.0.1/")[0] is False


def test_public_numeric_ip_allowed():
    safe, reason, ips = _classify_gateway_url("http://93.184.216.34/file")
    assert safe is True, reason
    assert ips == ["93.184.216.34"]


# ── hostname resolution (rebinding / internal-hostname) via injected resolver ────

def test_hostname_resolving_to_private_blocked():
    safe, reason, ips = _classify_gateway_url(
        "http://evil.example.com/x", resolve=_fake_resolver("10.1.2.3"))
    assert safe is False
    assert "blocked-ip" in reason


def test_hostname_resolving_to_metadata_blocked():
    safe, reason, ips = _classify_gateway_url(
        "https://rebind.example.com/x", resolve=_fake_resolver("169.254.169.254"))
    assert safe is False


def test_hostname_resolving_to_public_allowed():
    safe, reason, ips = _classify_gateway_url(
        "http://cdn.example.com/x", resolve=_fake_resolver("93.184.216.34"))
    assert safe is True
    assert ips == ["93.184.216.34"]


# ── the live fetch path refuses a blocked URL WITHOUT issuing a request ──────────

@pytest.mark.asyncio
async def test_fetch_from_url_refuses_internal_without_requesting():
    from prsm.node.content_provider import ContentProvider

    provider = ContentProvider.__new__(ContentProvider)
    provider.default_timeout = 5.0

    class _ExplodingSession:
        async def get(self, *a, **k):
            raise AssertionError("SSRF: a request was issued to a blocked URL")

    # if the guard fails, the code would reach the session and explode the test
    provider._http_session = _ExplodingSession()

    out = await provider._fetch_from_url(
        "http://169.254.169.254/latest/meta-data/", max_bytes=1024, timeout=1.0)
    assert out is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
