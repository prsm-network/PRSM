"""Sprint 1213 — off-chain static publisher-key anchor for dev/bench parallax fleets.

The 2-A10 multi-host bench got past pool gating but the chain's per-stage HandoffToken
failed anchor verification: the chain executor's anchor (_build_anchor_or_none) is the
on-chain PublisherKeyAnchorClient, and verify_with_anchor needs each settler's REAL
pubkey — which our un-registered nodes don't have on-chain. StaticFileAnchor resolves
node_id->pubkey from a local file so the fleet runs off-chain (opt-in
PRSM_PUBLISHER_KEY_ANCHOR_KIND=static), same lookup contract as the on-chain client.
"""
from __future__ import annotations

import json

import pytest

from prsm.security.publisher_key_anchor.static_anchor import StaticFileAnchor


def _write(tmp_path, obj):
    p = tmp_path / "anchor.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_lookup_returns_pubkey(tmp_path):
    a = StaticFileAnchor(_write(tmp_path, {"aa" * 16: "PUBKEY_A", "bb" * 16: "PUBKEY_B"}))
    assert a.lookup("aa" * 16) == "PUBKEY_A"
    assert a.lookup("bb" * 16) == "PUBKEY_B"
    assert len(a) == 2


def test_lookup_case_insensitive_and_absent(tmp_path):
    a = StaticFileAnchor(_write(tmp_path, {"abc123": "PK"}))
    assert a.lookup("ABC123") == "PK"          # case-insensitive (mirrors on-chain client)
    assert a.lookup("notpresent") is None
    assert a.lookup("") is None


def test_accepts_nodes_wrapper_shape(tmp_path):
    a = StaticFileAnchor(_write(tmp_path, {"nodes": {"n1": "PK1"}}))
    assert a.lookup("n1") == "PK1"


def test_build_anchor_static_kind(tmp_path, monkeypatch):
    from prsm.node.inference_wiring import _build_anchor_or_none
    f = _write(tmp_path, {"n1": "PK1"})
    monkeypatch.setenv("PRSM_PUBLISHER_KEY_ANCHOR_KIND", "static")
    monkeypatch.setenv("PRSM_PUBLISHER_KEY_ANCHOR_FILE", f)
    anchor = _build_anchor_or_none()
    assert anchor is not None and anchor.lookup("n1") == "PK1"


def test_build_anchor_static_kind_without_file_returns_none(monkeypatch):
    from prsm.node.inference_wiring import _build_anchor_or_none
    monkeypatch.setenv("PRSM_PUBLISHER_KEY_ANCHOR_KIND", "static")
    monkeypatch.delenv("PRSM_PUBLISHER_KEY_ANCHOR_FILE", raising=False)
    assert _build_anchor_or_none() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
