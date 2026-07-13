"""sp1455 (money-audit w6x780nqe) — on-chain content-access royalty is funded ONLY for an access this
node actually COLLECTED payment for.

The royalty-distribution audit found a CRITICAL, unanimously-confirmed drain: process_content_access
fired _try_onchain_distribute UNCONDITIONALLY, and distribute_royalty pulls `gross` from the OPERATOR's
own wallet (FTNS_WALLET_PRIVATE_KEY, via the RoyaltyDistributor transferFrom). But Step 1 debits/escrows
ONLY when accessor_id == this node — a REMOTE accessor "pays via their own node" and is never charged
here, yet Step 2 still distributed. With no per-access dedup (and each request minting a fresh random
payment_id), any peer replaying the same content request N times triggered N operator-funded on-chain
distributions → drains the operator's FTNS wallet + over-credits creators, at zero cost to the attacker
(once PRSM_ONCHAIN_PROVENANCE is enabled).

Fix (collection gate): the operator funds the on-chain royalty ONLY for an access it collected
(accessor_id == our node). A remote (uncollected) access falls through to LOCAL royalty bookkeeping
(return None); its real on-chain royalty settles via the settlement/forge path, which claims a STABLE
per-settlement idempotency key (royalty_dispatch_key requires a stable key, NOT a random payment_id).

Money assertion — never weaken.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from prsm.node.content_economy import ContentEconomy

_META = {"provenance_hash": "0x" + "11" * 32}


def _run(coro):
    return asyncio.run(coro)


def _content_economy(node_id: str = "operator-node"):
    ce = object.__new__(ContentEconomy)  # bypass the heavy constructor; set only what the method reads
    ce.identity = SimpleNamespace(node_id=node_id)
    dist = MagicMock()
    dist.preview_split.return_value = SimpleNamespace(
        creator_amount=8 * 10 ** 17, network_amount=2 * 10 ** 16, serving_node_amount=98 * 10 ** 16)
    dist.distribute_royalty.return_value = ("0x" + "ab" * 32, SimpleNamespace(value="confirmed"))
    ce._get_royalty_distributor = lambda: dist
    ce._serving_node_address = lambda: "0x" + "cc" * 20
    return ce, dist


def _payment(accessor_id: str):
    # _try_onchain_distribute reads only these four attributes off `payment`.
    return SimpleNamespace(
        payment_id=f"pay-{accessor_id}", content_id="cid-abc", accessor_id=accessor_id,
        amount=Decimal("1.0"))


def test_remote_uncollected_access_does_not_fund_onchain_royalty():
    ce, dist = _content_economy(node_id="operator-node")
    result = _run(ce._try_onchain_distribute(_payment("remote-peer-xyz"), _META))
    assert result is None, (
        "a remote (uncollected) content access funded an operator-paid on-chain royalty — a peer "
        "replaying content requests would drain the operator's wallet")
    dist.distribute_royalty.assert_not_called()


def test_repeated_remote_accesses_never_fund_onchain_royalty():
    """The replay drain: N identical remote requests must fire ZERO operator-funded distributions."""
    ce, dist = _content_economy(node_id="operator-node")
    for _ in range(5):
        assert _run(ce._try_onchain_distribute(_payment("remote-peer-xyz"), _META)) is None
    assert dist.distribute_royalty.call_count == 0


def test_own_collected_access_still_distributes_onchain():
    """The gate must NOT break the legitimate path: an access THIS node collected still distributes."""
    ce, dist = _content_economy(node_id="operator-node")
    result = _run(ce._try_onchain_distribute(_payment("operator-node"), _META))
    assert result is not None
    assert dist.distribute_royalty.call_count == 1
    assert any(d.get("type") == "original_creator" for d in result)
