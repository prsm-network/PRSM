"""Sprint 1494 — escrow releases must reach the PAYEE's own node, without double-paying.

The last item from the 26-agent money-path trace. PaymentEscrow had no cross-node
credit rail: a release to a remote payee moved funds on THIS node's ledger only.
The payee's node never learned of it, and broadcast_tx cannot help — it routes to
BatchSettlementManager, whose _resolve_address rejects a 32-hex node_id (sp1492).
It was not even reconcilable: the tx's parties are escrow-<uuid> and the payee,
neither being the settling node's own id, so get_transaction_history() misses it.

The gating is the whole subtlety. `broadcast` is not an "on-chain?" switch — it
answers "does something ELSE settle this payee?":

  broadcast=False  -> per-stage on-chain commit settles them (credit_policy.py with
                      PRSM_MULTISTAGE_SETTLEMENT). Gossiping too = DOUBLE-PAY.
  broadcast=True   -> nothing else pays a remote node_id at all. That is the strand.

So gossip follows `broadcast` exactly, and is additionally restricted to payees
that broadcast_tx provably cannot pay.
"""
from __future__ import annotations

import logging

import pytest

from prsm.node.payment_escrow import PaymentEscrow

PEER = "d437aa67d99cff4a6a17179f5c731b77"          # 32 hex — a real node_id shape
OTHER_PEER = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
ADDR = "0x" + "a1" * 20
SELF_ID = "beef" * 8                                # 32 hex, this node


class _Gossip:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    async def __call__(self, tx):
        if self.fail:
            raise RuntimeError("gossip down")
        self.sent.append(tx)


class _Tx:
    def __init__(self, to_wallet=PEER, tx_id="tx-1"):
        self.to_wallet, self.tx_id = to_wallet, tx_id


def _esc(gossip=None):
    e = object.__new__(PaymentEscrow)
    e.node_id = SELF_ID
    e.gossip_tx = gossip
    return e


# ── the gate ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gossips_to_a_REMOTE_peer_when_nothing_else_pays_them():
    """★ THE strand this closes."""
    g = _Gossip()
    assert await _esc(g)._maybe_gossip(_Tx(PEER), PEER, broadcast=True) is True
    assert len(g.sent) == 1


@pytest.mark.asyncio
async def test_does_NOT_gossip_when_broadcast_is_False():
    """★ THE double-pay guard. broadcast=False means per-stage on-chain commit
    already settles this payee (credit_policy.py under PRSM_MULTISTAGE_SETTLEMENT);
    gossiping too would credit them twice — once withdrawable off-chain, once on
    chain."""
    g = _Gossip()
    assert await _esc(g)._maybe_gossip(_Tx(PEER), PEER, broadcast=False) is False
    assert g.sent == []


@pytest.mark.asyncio
async def test_does_NOT_gossip_to_an_0x_address():
    """broadcast_tx CAN mirror a 0x payee on chain, so gossiping as well would
    double-pay."""
    g = _Gossip()
    assert await _esc(g)._maybe_gossip(_Tx(ADDR), ADDR, broadcast=True) is False
    assert g.sent == []


@pytest.mark.asyncio
async def test_does_NOT_gossip_a_self_release():
    """A release to our own node is already local — nothing to tell anyone."""
    g = _Gossip()
    assert await _esc(g)._maybe_gossip(_Tx(SELF_ID), SELF_ID, broadcast=True) is False
    assert g.sent == []


@pytest.mark.asyncio
async def test_does_NOT_gossip_internal_wallets():
    g = _Gossip()
    for internal in ("escrow-6f1b2c3d-4e5f-6789-abcd-ef01", "system", ""):
        assert await _esc(g)._maybe_gossip(_Tx(internal), internal, broadcast=True) is False
    assert g.sent == []


@pytest.mark.asyncio
async def test_no_rail_configured_is_a_safe_noop():
    assert await _esc(None)._maybe_gossip(_Tx(), PEER, broadcast=True) is False


@pytest.mark.asyncio
async def test_a_none_tx_is_a_noop():
    """release_escrow returns None on an already-released escrow — must not gossip."""
    g = _Gossip()
    assert await _esc(g)._maybe_gossip(None, PEER, broadcast=True) is False
    assert g.sent == []


@pytest.mark.asyncio
async def test_a_failed_gossip_is_LOUD_and_non_fatal(caplog):
    """The local release already committed, so do not raise — but this is the ONLY
    rail that tells the payee they were paid, so it must not be silent (sp1489)."""
    g = _Gossip(fail=True)
    with caplog.at_level(logging.ERROR):
        assert await _esc(g)._maybe_gossip(_Tx(PEER), PEER, broadcast=True) is False
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "GOSSIP FAILED" in msg and "will NOT see this credit" in msg


# ── wiring ──────────────────────────────────────────────────────────

def test_release_escrow_invokes_the_rail():
    """★ Binding test — a rail nothing calls leaves the strand live."""
    import inspect
    src = inspect.getsource(PaymentEscrow._release_escrow_locked)
    assert "_maybe_gossip(tx, provider_id, broadcast=True)" in src


def test_split_release_invokes_the_rail_and_honours_broadcast():
    """★ The split path is the one reachable TODAY (api.py:7281 swarm workers), and
    it is also the path where broadcast=False must suppress gossip."""
    import inspect
    src = inspect.getsource(PaymentEscrow._release_escrow_split_locked)
    assert "_maybe_gossip(" in src
    assert "broadcast=broadcast" in src, (
        "the split path must pass broadcast through, or multistage double-pays")


def test_split_pairs_each_tx_with_its_OWN_payee():
    """Pairing by index into `splits` would misattribute credits when a split is
    partially applied — each tx must name its own to_wallet."""
    import inspect
    src = inspect.getsource(PaymentEscrow._release_escrow_split_locked)
    guard = src[src.index("for tx in txs:"):]
    assert 'getattr(tx, "to_wallet"' in guard


def test_node_wires_the_gossip_rail():
    """★ Binding test on the live wiring."""
    import inspect
    import prsm.node.node as node_mod
    src = inspect.getsource(node_mod)
    assert "_payment_escrow.gossip_tx = self.ledger_sync.broadcast_transaction" in src


def test_split_broadcast_failure_is_no_longer_silent():
    """sp1489 loudened two broadcast sites and missed this third one."""
    import inspect
    src = inspect.getsource(PaymentEscrow._release_escrow_split_locked)
    assert "escrow-split broadcast FAILED" in src
    assert "except Exception:\n                    pass" not in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
