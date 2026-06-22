"""Sprint 1226 — PRSM_PARALLAX_CHAIN_DEADLINE_S overrides the per-stage
HandoffToken deadline.

The default 30s deadline is fine for WARM inference but expires mid-load when a
node COLD-slice-loads a big model (sp1220-1225: reading GBs of safetensors +
building the skeleton + moving to GPU takes minutes on the first request →
DEADLINE_EXCEEDED). Operators serving large models set this env higher. Found
live on the 2-A10 7B slice-load bench.
"""
from __future__ import annotations

import pytest

from prsm.compute.chain_rpc.factories import make_rpc_chain_executor
from prsm.node.identity import generate_node_identity


class _Anchor:
    def lookup(self, node_id):  # noqa: ARG002
        return None


def _executor(monkeypatch, env_val):
    if env_val is None:
        monkeypatch.delenv("PRSM_PARALLAX_CHAIN_DEADLINE_S", raising=False)
    else:
        monkeypatch.setenv("PRSM_PARALLAX_CHAIN_DEADLINE_S", env_val)
    # wrap_*=False → returns the bare RpcChainExecutor whose
    # _default_deadline_seconds is the minted-token budget.
    return make_rpc_chain_executor(
        settler_identity=generate_node_identity(),
        send_message=lambda addr, data: data,
        anchor=_Anchor(),
        wrap_topology_aware=False,
        wrap_activation_dp_aware=False,
    )


def test_default_deadline_is_30s(monkeypatch):
    ex = _executor(monkeypatch, None)
    assert ex._default_deadline_seconds == 30.0


def test_env_override_raises_deadline(monkeypatch):
    ex = _executor(monkeypatch, "600")
    assert ex._default_deadline_seconds == 600.0


def test_invalid_env_keeps_default(monkeypatch):
    ex = _executor(monkeypatch, "not-a-number")
    assert ex._default_deadline_seconds == 30.0


def test_nonpositive_env_keeps_default(monkeypatch):
    ex = _executor(monkeypatch, "0")
    assert ex._default_deadline_seconds == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
