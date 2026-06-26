"""Sprint 1280 — bound the client's TokenFrame consumption from an untrusted tail (round 7, HIGH).

In the trustless cross-host streaming path the requester consumes TokenFrames from an UNTRUSTED
tail stage. The consume loop enforced only monotonic sequence + request_id match — NO absolute
cap — so a malicious tail could stream TokenFrames forever (unbounded memory / non-terminating
generator). The honest server caps locally (while seq < max_tokens), but that cap doesn't bind
the untrusted peer. Fix: the requester re-enforces an absolute ceiling
(_stream_token_cap = the requester's max_tokens, bounded by the operator inference ceiling,
never unbounded) and raises if the tail exceeds it.
"""
from __future__ import annotations

import importlib

from prsm.compute.chain_rpc.client import _stream_token_cap


def test_requester_max_tokens_used_when_small():
    assert _stream_token_cap(10) == 10


def test_none_falls_back_to_ceiling():
    import prsm.compute.inference.autoregressive_runner as ar
    assert _stream_token_cap(None) == ar._resolve_max_tokens_ceiling()


def test_huge_request_clamped_to_ceiling():
    import prsm.compute.inference.autoregressive_runner as ar
    assert _stream_token_cap(10**9) == ar._resolve_max_tokens_ceiling()


def test_zero_or_negative_falls_back_to_ceiling():
    import prsm.compute.inference.autoregressive_runner as ar
    ceiling = ar._resolve_max_tokens_ceiling()
    assert _stream_token_cap(0) == ceiling
    assert _stream_token_cap(-5) == ceiling


def test_never_unbounded():
    # whatever the input, the cap is a finite positive int (never None / inf)
    for v in (None, 0, -1, "x", 10, 10**12):
        cap = _stream_token_cap(v)
        assert isinstance(cap, int) and cap > 0


def test_consume_loop_enforces_the_cap():
    # source-pin: the streaming-tail consume loop raises when expected_seq reaches the cap
    import inspect
    import prsm.compute.chain_rpc.client as client
    src = inspect.getsource(client)
    assert "_stream_token_cap" in src
    assert "expected_seq >= _token_cap" in src


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
