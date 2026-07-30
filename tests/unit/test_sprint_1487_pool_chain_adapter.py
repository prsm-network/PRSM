"""Sprint 1487 — OperatorRewardPoolClient must really satisfy PoolChain.

The epoch job is written against a Protocol so its ordering logic is testable
without a chain. That buys nothing if the real client silently fails to implement
the Protocol — the job would then be exercised only against a fake and break the
first time it touched mainnet. These tests bind the two together.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.web3.operator_reward_pool_client import (
    OPERATOR_REWARD_POOL_ABI,
    OperatorRewardPoolClient,
)
from prsm.settlement.epoch_runner import PoolChain


def _bare_client():
    """A client without __init__, so no web3/RPC is required."""
    c = object.__new__(OperatorRewardPoolClient)
    c.pool = MagicMock()
    c.web3 = MagicMock()
    c._account = None
    import threading
    c._tx_lock = threading.Lock()
    return c


def test_client_structurally_satisfies_the_pool_chain_protocol():
    """★ The binding test: the job's Protocol and the real client must agree."""
    c = _bare_client()
    assert isinstance(c, PoolChain)
    for name in ("epoch_exists", "unreserved_balance_wei", "publish_epoch"):
        assert callable(getattr(c, name)), f"client is missing {name}"


def test_client_signatures_match_how_the_job_actually_calls_them():
    """isinstance() on a Protocol compares method NAMES only — a client whose
    publish_epoch took different arguments would still pass it and then fail at the
    one moment that is irreversible. Check the call shapes the job really uses."""
    import inspect
    c = _bare_client()

    # run_epoch calls these positionally, with no extra required arguments.
    def required_positional(fn):
        sig = inspect.signature(fn)
        return [n for n, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]

    assert required_positional(c.epoch_exists) == ["epoch_id"]
    assert required_positional(c.unreserved_balance_wei) == []
    assert required_positional(c.publish_epoch) == [
        "epoch_id", "merkle_root", "total_amount_wei"]


def test_abi_exposes_the_two_calls_the_job_needs():
    names = {e["name"] for e in OPERATOR_REWARD_POOL_ABI}
    assert "surplus" in names and "publishEpoch" in names


def test_epoch_exists_reflects_published_at():
    c = _bare_client()
    # epochs() -> (root, total, claimed, publishedAt, reclaimed)
    c.pool.functions.epochs.return_value.call.return_value = (
        b"\x00" * 32, 0, 0, 0, False)
    assert c.epoch_exists(1) is False
    c.pool.functions.epochs.return_value.call.return_value = (
        b"\x11" * 32, 100, 0, 1234567, False)
    assert c.epoch_exists(1) is True


def test_unreserved_balance_reads_surplus_in_ONE_call(monkeypatch):
    """★ Not balanceOf minus totalReserved: a load-balanced RPC can serve those two
    reads from replicas at different heights, producing a number that was never
    true (the sp1474 class). One call at one height cannot be inconsistent."""
    c = _bare_client()
    c.pool.functions.surplus.return_value.call.return_value = 42
    assert c.unreserved_balance_wei() == 42
    c.pool.functions.surplus.assert_called_once_with()
    # ...and it must never fall back to the two-read form.
    c.pool.functions.balanceOf.assert_not_called()


def test_publish_refuses_without_a_publisher_key():
    c = _bare_client()
    with pytest.raises(RuntimeError, match="no private key"):
        c.publish_epoch(1, b"\x11" * 32, 100)


def test_publish_rejects_a_malformed_root():
    """A short root would be right-padded into a DIFFERENT valid-looking root,
    against which no earner's proof verifies — the epoch id is then burned."""
    c = _bare_client()
    c._account = MagicMock()
    with pytest.raises(ValueError, match="32 bytes"):
        c.publish_epoch(1, b"\x11" * 31, 100)


def test_publish_rejects_a_non_positive_total():
    c = _bare_client()
    c._account = MagicMock()
    with pytest.raises(ValueError, match="must be positive"):
        c.publish_epoch(1, b"\x11" * 32, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
