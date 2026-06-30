"""Sprint 1309 — automatic producer-endpoint discovery for the cross-provider audit.

sp1308's cross-provider audit needed a manual PRSM_PEER_RECEIPT_ENDPOINTS map. sp1309
makes the resolver AUTOMATIC: it matches a provider eth-address against the live peer
set (a peer's hardware_profile.operator_address, sp690/sp788) to find its receipt-serve
endpoint, with the env map kept as a manual override pin.
"""
from __future__ import annotations

from types import SimpleNamespace

from prsm.settlement.settlement_audit_wiring import build_discovery_endpoint_resolver


def _peer(address, operator_address=None, hw=None):
    profile = dict(hw or {})
    if operator_address is not None:
        profile["operator_address"] = operator_address
    return SimpleNamespace(address=address, hardware_profile=(profile or None))


def _discovery(peers):
    return SimpleNamespace(get_known_peers=lambda: peers)


# ── inert when nothing available ─────────────────────────────────────────────

def test_none_when_no_discovery_and_no_env():
    assert build_discovery_endpoint_resolver(None, environ={}) is None


def test_resolver_built_with_discovery_only():
    r = build_discovery_endpoint_resolver(_discovery([]), environ={})
    assert r is not None
    assert r("0xANY") is None   # no peers match yet


# ── discovery match ──────────────────────────────────────────────────────────

def test_resolves_via_operator_address_match():
    peers = [
        _peer("9.9.9.9:8000", operator_address="0xOTHER"),
        _peer("1.2.3.4:8000", operator_address="0xPrOvIdEr"),  # mixed case
    ]
    r = build_discovery_endpoint_resolver(_discovery(peers), environ={})
    assert r("0xprovider") == "http://1.2.3.4:8000"   # case-insensitive eth compare
    assert r("0xPROVIDER") == "http://1.2.3.4:8000"


def test_already_schemed_address_not_double_prefixed():
    peers = [_peer("https://node.example:443", operator_address="0xP")]
    r = build_discovery_endpoint_resolver(_discovery(peers), environ={})
    assert r("0xp") == "https://node.example:443"


def test_no_match_returns_none():
    peers = [_peer("1.2.3.4:8000", operator_address="0xOTHER")]
    r = build_discovery_endpoint_resolver(_discovery(peers), environ={})
    assert r("0xprovider") is None


def test_peer_without_operator_address_skipped():
    peers = [_peer("1.2.3.4:8000", hw={"tflops_fp16": 10}),  # no operator_address
             _peer("5.6.7.8:8000")]                          # no hardware_profile
    r = build_discovery_endpoint_resolver(_discovery(peers), environ={})
    assert r("0xprovider") is None


# ── env override pin wins ────────────────────────────────────────────────────

def test_env_pin_overrides_discovery():
    peers = [_peer("1.2.3.4:8000", operator_address="0xP")]
    env = {"PRSM_PEER_RECEIPT_ENDPOINTS": '{"0xP":"http://pinned:9000"}'}
    r = build_discovery_endpoint_resolver(_discovery(peers), environ=env)
    assert r("0xp") == "http://pinned:9000"          # pin wins over discovery's 1.2.3.4
    # a provider NOT in the pin still resolves via discovery
    peers.append(_peer("2.2.2.2:8000", operator_address="0xQ"))
    assert r("0xq") == "http://2.2.2.2:8000"


def test_env_only_works_without_discovery():
    env = {"PRSM_PEER_RECEIPT_ENDPOINTS": '{"0xP":"http://pinned:9000"}'}
    r = build_discovery_endpoint_resolver(None, environ=env)
    assert r is not None
    assert r("0xp") == "http://pinned:9000"


# ── fail-soft ────────────────────────────────────────────────────────────────

def test_failsoft_on_discovery_error():
    def _boom():
        raise RuntimeError("discovery down")
    r = build_discovery_endpoint_resolver(
        SimpleNamespace(get_known_peers=_boom), environ={})
    assert r("0xprovider") is None   # never raises


def test_empty_provider_returns_none():
    r = build_discovery_endpoint_resolver(_discovery([_peer("1.2.3.4:8000", "0xP")]),
                                          environ={})
    assert r("") is None
    assert r(None) is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
