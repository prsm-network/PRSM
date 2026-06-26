"""Sprint 1284 — bind KV-cache rollback to the request owner (audit round 7 #2, rollback half).

RollbackCacheRequest had the SAME no-auth gap sp1283 closed for EvictCacheRequest: no auth +
keyed on a cleartext request_id, so an observer could truncate another session's KV-cache.
Fix mirrors sp1283: RollbackCacheRequest optionally carries the settler-signed upstream_token;
under PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER (+ anchor) the rollback is honored only with a valid
token bound to the request_id. Default OFF preserves prior behavior; protocol field is
backward-compatible.
"""
from __future__ import annotations

import pytest

from prsm.compute.chain_rpc.protocol import (
    HandoffToken,
    RollbackCacheRequest,
    encode_message,
    parse_message,
)


def _valid_token(idy, request_id):
    return HandoffToken.sign(
        identity=idy, request_id=request_id,
        chain_stage_index=0, chain_total_stages=1, deadline_unix=1.0e12,
    )


# ── protocol round-trip (backward-compatible) ────────────────────────────────

def test_rollback_without_token_roundtrips():
    msg = RollbackCacheRequest(request_id="infer-r1", n_positions_to_drop=3)
    back = parse_message(encode_message(msg))
    assert isinstance(back, RollbackCacheRequest)
    assert back.request_id == "infer-r1" and back.n_positions_to_drop == 3
    assert back.upstream_token is None


def test_rollback_with_token_roundtrips():
    from prsm.node.identity import generate_node_identity
    idy = generate_node_identity()
    msg = RollbackCacheRequest(
        request_id="infer-r1", n_positions_to_drop=2,
        upstream_token=_valid_token(idy, "infer-r1"),
    )
    back = parse_message(encode_message(msg))
    assert isinstance(back.upstream_token, HandoffToken)
    assert back.upstream_token.request_id == "infer-r1"


def test_rollback_rejects_bad_token_type():
    from prsm.compute.chain_rpc.protocol import ChainRpcMalformedError
    with pytest.raises(ChainRpcMalformedError):
        RollbackCacheRequest(request_id="x", n_positions_to_drop=1, upstream_token="nope")


# ── server handler authorization ─────────────────────────────────────────────

def _server_and_cache():
    from prsm.compute.chain_rpc.server import LayerStageServer
    from prsm.compute.chain_rpc.kv_cache import KVCacheManager
    from prsm.node.identity import generate_node_identity
    idy = generate_node_identity()
    srv = LayerStageServer.__new__(LayerStageServer)
    kvm = KVCacheManager()
    srv._kv_cache_manager = kvm
    srv._anchor = type("A", (), {"lookup": staticmethod(lambda nid: idy.public_key_b64)})()
    # rollback handler also checks _sharded_runner — give it a truncating one
    class _Runner:
        def rollback_cache(self, request_id, n_positions_to_drop, replay_accepted_prefix=None,
                           target_stage_index=None):
            return kvm.rollback(request_id, n_positions_to_drop, lambda payload, n: payload)
    srv._sharded_runner = _Runner()
    return srv, kvm, idy


def _seed(kvm, request_id, positions=5):
    h = kvm.allocate(request_id, n_layers=2)
    h.cached_positions = positions
    return h


def test_enforced_rollback_rejects_tokenless(monkeypatch):
    monkeypatch.setenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", "1")
    srv, kvm, _idy = _server_and_cache()
    h = _seed(kvm, "infer-R")
    srv._handle_rollback_cache(RollbackCacheRequest(request_id="infer-R", n_positions_to_drop=2))
    assert h.cached_positions == 5   # untouched — unauthorized


def test_enforced_rollback_rejects_wrong_request_token(monkeypatch):
    monkeypatch.setenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", "1")
    srv, kvm, idy = _server_and_cache()
    h = _seed(kvm, "infer-R")
    srv._handle_rollback_cache(RollbackCacheRequest(
        request_id="infer-R", n_positions_to_drop=2, upstream_token=_valid_token(idy, "OTHER")))
    assert h.cached_positions == 5


def test_enforced_rollback_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", "1")
    srv, kvm, idy = _server_and_cache()
    h = _seed(kvm, "infer-R")
    srv._handle_rollback_cache(RollbackCacheRequest(
        request_id="infer-R", n_positions_to_drop=2, upstream_token=_valid_token(idy, "infer-R")))
    assert h.cached_positions == 3   # truncated by the legitimate owner


def test_disabled_rollback_allows_tokenless(monkeypatch):
    monkeypatch.delenv("PRSM_CHAIN_RPC_ENFORCE_CACHE_OWNER", raising=False)
    srv, kvm, _idy = _server_and_cache()
    h = _seed(kvm, "infer-R")
    srv._handle_rollback_cache(RollbackCacheRequest(request_id="infer-R", n_positions_to_drop=2))
    assert h.cached_positions == 3   # legacy behavior preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
