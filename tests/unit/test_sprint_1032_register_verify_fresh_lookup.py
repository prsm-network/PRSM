"""Sprint 1032 — register-operator-pubkey verify step bypasses the stale
negative cache.

Live bug hit during the Tier-1 cross-host bench registrations: the script's
pre-tx idempotency check (anchor.lookup) negative-caches the 'not registered'
miss for the client's TTL (~1h). The post-tx verify reused the SAME client, so
it read that stale cached miss and printed
'✗ anchor.lookup returned empty after confirmation' + exited 1 on a SUCCESSFUL
on-chain registration. Both Lambda nodes WERE registered (confirmed
independently with a fresh client on two RPCs; tx 786e3442 / 0933e93a landed
with status=1, ~92k gas), so the script's own verdict was a false negative that
would scare any operator into thinking the ceremony failed.

Fix: the verify step invalidates the cache entry (PublisherKeyAnchorClient
exposes invalidate() for exactly this) + a short retry for RPC read-replica
lag, so it re-reads the chain instead of the stale miss.
"""
import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "sprint_675_register_operator_pubkey.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("sprint_675_reg", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeAnchor:
    """Models the PublisherKeyAnchorClient negative cache: a node_id whose
    pre-tx miss was cached returns None from lookup() until invalidate()."""

    def __init__(self, on_chain=None):
        self._chain = {k.lower(): v for k, v in (on_chain or {}).items()}
        self._neg = set()
        self.invalidate_calls = []

    def lookup(self, node_id):
        nid = node_id.lower()
        if nid in self._neg:
            return None          # stale negative cache
        return self._chain.get(nid)

    def invalidate(self, node_id):
        nid = node_id.lower()
        self.invalidate_calls.append(nid)
        self._neg.discard(nid)

    def cache_negative(self, node_id):
        self._neg.add(node_id.lower())


def test_helper_exists():
    mod = _load_script_module()
    assert hasattr(mod, "_fresh_anchor_lookup")


def test_fresh_lookup_bypasses_stale_negative_cache():
    """The whole bug: the chain HAS the pubkey, but the client cached the
    pre-tx miss. A plain lookup returns None; the fix invalidates → reads it."""
    mod = _load_script_module()
    nid = "d148f913c7d8a46fe528486888278c47"
    pub = "uRG2SldIutMvoPGfB6BjzgiMok+vvF2/1FnM+AiK870="
    anchor = _FakeAnchor(on_chain={nid: pub})
    anchor.cache_negative(nid)               # pre-tx miss cached
    assert anchor.lookup(nid) is None        # the stale read (the bug)
    got = mod._fresh_anchor_lookup(anchor, nid, attempts=1)
    assert got == pub                        # fixed: invalidate → fresh read
    assert nid in anchor.invalidate_calls    # it actually invalidated


def test_fresh_lookup_returns_none_when_truly_unregistered():
    """Not a false-positive machine: if the chain genuinely has no pubkey it
    still returns None, so a real failed registration is still caught."""
    mod = _load_script_module()
    nid = "ab" * 16
    anchor = _FakeAnchor(on_chain={})        # nothing on chain
    got = mod._fresh_anchor_lookup(anchor, nid, attempts=1)
    assert got is None


def test_fresh_lookup_retries_then_succeeds_on_replica_lag():
    """RPC read-replica lag: the pubkey appears only on a later read. The
    helper retries (invalidating each time) and eventually returns it."""
    mod = _load_script_module()
    nid = "ee" * 16
    pub = "somebase64pubkey=="
    anchor = _FakeAnchor(on_chain={})

    calls = {"n": 0}
    real_lookup = anchor.lookup

    def lagging_lookup(node_id):
        calls["n"] += 1
        if calls["n"] >= 2:               # lands on the 2nd attempt
            anchor._chain[nid.lower()] = pub
        return real_lookup(node_id)

    anchor.lookup = lagging_lookup
    got = mod._fresh_anchor_lookup(anchor, nid, attempts=3, sleep_s=0)
    assert got == pub
    assert calls["n"] >= 2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
