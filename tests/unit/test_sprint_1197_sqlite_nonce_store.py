"""Sprint 1197 — SqliteNonceStore: SIWE nonces correct across multiple workers.

The default InMemoryNonceStore is per-process: in a multi-worker deployment a
nonce issued by worker A is invisible to worker B → SIWE login fails
intermittently (siwe_nonce_invalid_or_consumed). SqliteNonceStore shares a SQLite
file so issue/consume work across all workers on one host, with atomic single-use.
"""
from __future__ import annotations

import pytest

from prsm.interface.onboarding.siwe import SqliteNonceStore


def _store(tmp_path, now=None, name="siwe_nonces.db"):
    return SqliteNonceStore(tmp_path / name, _now=now)


def test_issue_then_consume_succeeds(tmp_path):
    s = _store(tmp_path)
    nonce = s.issue(ttl_seconds=300)
    assert isinstance(nonce, str) and len(nonce) >= 8
    assert s.consume(nonce) is True


def test_single_use_second_consume_fails(tmp_path):
    s = _store(tmp_path)
    nonce = s.issue()
    assert s.consume(nonce) is True
    assert s.consume(nonce) is False  # already consumed


def test_unknown_nonce_consume_fails(tmp_path):
    s = _store(tmp_path)
    assert s.consume("never-issued") is False


def test_expired_nonce_cannot_be_consumed(tmp_path):
    clock = {"t": 1000.0}
    s = _store(tmp_path, now=lambda: clock["t"])
    nonce = s.issue(ttl_seconds=300)
    clock["t"] += 301  # past TTL
    assert s.consume(nonce) is False


def test_cross_instance_sharing_the_multiworker_proof(tmp_path):
    # Two SqliteNonceStore instances over the SAME file = two workers. A nonce
    # issued by "worker A" MUST be consumable by "worker B".
    worker_a = _store(tmp_path)
    worker_b = _store(tmp_path)
    nonce = worker_a.issue()
    assert worker_b.consume(nonce) is True
    # and single-use holds across workers: A can't re-consume it
    assert worker_a.consume(nonce) is False


def test_unique_nonces_issued(tmp_path):
    s = _store(tmp_path)
    nonces = {s.issue() for _ in range(50)}
    assert len(nonces) == 50


def test_durable_across_reopen(tmp_path):
    # A nonce survives a store re-open within its TTL (restart safety).
    s1 = _store(tmp_path)
    nonce = s1.issue(ttl_seconds=300)
    s2 = _store(tmp_path)  # "restart"
    assert s2.consume(nonce) is True


# ── wiring resolver ──────────────────────────────────────────────────────────────────

def test_resolver_uses_sqlite_by_default(tmp_path, monkeypatch):
    import prsm.node.wallet_api_wiring as w
    monkeypatch.setenv("PRSM_WALLET_NONCE_DB", str(tmp_path / "nonces.db"))
    store = w._resolve_nonce_store()
    assert isinstance(store, SqliteNonceStore)
    # functional: issue+consume round-trips
    n = store.issue()
    assert store.consume(n) is True


def test_resolver_falls_back_to_in_memory_on_bad_path(tmp_path, monkeypatch):
    import prsm.node.wallet_api_wiring as w
    from prsm.interface.onboarding.siwe import InMemoryNonceStore
    # point at a path whose parent is a FILE → mkdir/connect fails → fallback
    bad_parent = tmp_path / "afile"
    bad_parent.write_text("not a dir")
    monkeypatch.setenv("PRSM_WALLET_NONCE_DB", str(bad_parent / "nonces.db"))
    store = w._resolve_nonce_store()
    assert isinstance(store, InMemoryNonceStore)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
