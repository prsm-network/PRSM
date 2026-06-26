"""Sprint 1283 — bind KV-cache eviction to the request owner (audit round 7 #2, HIGH).

EvictCacheRequest carried no auth and keyed on a request_id transmitted in cleartext to every
chain stage, so any peer that observed it could wipe another session's in-flight KV-cache
(cross-session DoS / forced recompute). Fix: EvictCacheRequest now optionally carries the
settler-signed upstream_token; with PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER on (and an anchor
wired), an evict is honored ONLY if it presents a valid token bound to the request_id.
Default OFF preserves the prior behavior on the proven multi-host path. (RollbackCacheRequest
has the same gap — same-pattern follow-on.)
"""
from __future__ import annotations

import pytest

from prsm.compute.chain_rpc.protocol import (
    EvictCacheRequest,
    HandoffToken,
    parse_message,
    encode_message,
)
from prsm.compute.chain_rpc.server import _enforce_cache_owner


# ── enforcement flag ─────────────────────────────────────────────────────────

def test_enforce_flag_default_off(monkeypatch):
    monkeypatch.delenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", raising=False)
    assert _enforce_cache_owner() is False


def test_enforce_flag_on(monkeypatch):
    monkeypatch.setenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", "1")
    assert _enforce_cache_owner() is True


# ── protocol: token round-trips, backward-compatible ─────────────────────────

def test_evict_request_without_token_roundtrips():
    msg = EvictCacheRequest(request_id="infer-abc123")
    back = parse_message(encode_message(msg))
    assert isinstance(back, EvictCacheRequest)
    assert back.request_id == "infer-abc123"
    assert back.upstream_token is None   # backward-compatible default


def test_evict_request_with_token_roundtrips():
    from prsm.node.identity import generate_node_identity
    idy = generate_node_identity()
    tok = HandoffToken.sign(
        identity=idy, request_id="infer-abc123",
        chain_stage_index=0, chain_total_stages=2, deadline_unix=1.0e12,
    )
    msg = EvictCacheRequest(request_id="infer-abc123", upstream_token=tok)
    back = parse_message(encode_message(msg))
    assert isinstance(back.upstream_token, HandoffToken)
    assert back.upstream_token.request_id == "infer-abc123"


def test_evict_request_rejects_bad_token_type():
    from prsm.compute.chain_rpc.protocol import ChainRpcMalformedError
    with pytest.raises(ChainRpcMalformedError):
        EvictCacheRequest(request_id="x", upstream_token="not-a-token")


# ── server handler authorization (the security-critical path) ────────────────

def _server_and_cache():
    from prsm.compute.chain_rpc.server import LayerStageServer
    from prsm.compute.chain_rpc.kv_cache import KVCacheManager
    from prsm.node.identity import generate_node_identity
    idy = generate_node_identity()
    srv = LayerStageServer.__new__(LayerStageServer)   # bypass the heavy constructor
    srv._kv_cache_manager = KVCacheManager()
    srv._anchor = type("A", (), {"lookup": staticmethod(lambda nid: idy.public_key_b64)})()
    return srv, srv._kv_cache_manager, idy


def _valid_token(idy, request_id):
    return HandoffToken.sign(
        identity=idy, request_id=request_id,
        chain_stage_index=0, chain_total_stages=1, deadline_unix=1.0e12,
    )


def test_enforced_evict_rejects_tokenless(monkeypatch):
    monkeypatch.setenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", "1")
    srv, kvm, _idy = _server_and_cache()
    kvm.allocate("infer-X", n_layers=2)
    srv._handle_evict_cache(EvictCacheRequest(request_id="infer-X"))
    assert "infer-X" in kvm   # NOT evicted — unauthorized


def test_enforced_evict_rejects_wrong_request_token(monkeypatch):
    monkeypatch.setenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", "1")
    srv, kvm, idy = _server_and_cache()
    kvm.allocate("infer-X", n_layers=2)
    # a valid token, but bound to a DIFFERENT request_id → rejected
    srv._handle_evict_cache(
        EvictCacheRequest(request_id="infer-X", upstream_token=_valid_token(idy, "infer-OTHER")))
    assert "infer-X" in kvm


def test_enforced_evict_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", "1")
    srv, kvm, idy = _server_and_cache()
    kvm.allocate("infer-X", n_layers=2)
    srv._handle_evict_cache(
        EvictCacheRequest(request_id="infer-X", upstream_token=_valid_token(idy, "infer-X")))
    assert "infer-X" not in kvm   # evicted by the legitimate owner


def test_disabled_evict_allows_tokenless(monkeypatch):
    monkeypatch.delenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", raising=False)
    srv, kvm, _idy = _server_and_cache()
    kvm.allocate("infer-X", n_layers=2)
    srv._handle_evict_cache(EvictCacheRequest(request_id="infer-X"))
    assert "infer-X" not in kvm   # legacy behavior preserved (default OFF)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
