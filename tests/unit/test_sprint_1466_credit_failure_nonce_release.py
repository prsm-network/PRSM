"""Sprint 1466 — a failed recipient credit must RELEASE the claimed nonce so the payment retries.

`_on_ftns_transaction` (ledger_sync) CLAIMS the nonce via record_nonce (the sp898 double-credit
serialization point, which COMMITS) BEFORE crediting the recipient. The credit was not
exception-guarded and no release_nonce ran on failure, so if credit() raised (e.g. the sp910
concurrent-write savepoint collision under load) the nonce stayed claimed forever: every re-gossiped
copy is rejected at the has_seen_nonce fast-path BEFORE the has_transaction retry can fire → the
recipient is PERMANENTLY never credited (a stranded, money-losing payment). Fix: release the nonce on
credit failure so a re-gossip re-claims + retries. sp898's happy path (successful credit → nonce stays
claimed) is preserved, so there is no double-credit regression.
"""
from __future__ import annotations

import contextlib
import json
import time
import uuid
from unittest.mock import MagicMock

import pytest

from prsm.node.identity import generate_node_identity
from prsm.node.ledger_sync import LedgerSync
from prsm.node.local_ledger import LocalLedger, Transaction, TransactionType


async def _make_ledger():
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    return ledger


def _signed(sender, recipient, amount):
    tx = Transaction(
        tx_id=str(uuid.uuid4()), tx_type=TransactionType.TRANSFER,
        from_wallet=sender.node_id, to_wallet=recipient.node_id, amount=amount,
        description="cross-node xfer", timestamp=time.time(), signature="")
    canonical = LedgerSync._canonical_tx_payload(tx, tx.tx_id)
    sig = sender.sign(json.dumps(canonical, sort_keys=True).encode())
    return {**canonical, "signature": sig, "origin_public_key": sender.public_key_b64}


def _ls(recipient, ledger):
    return LedgerSync(identity=recipient, gossip=MagicMock(), ledger=ledger, transport=MagicMock())


@pytest.mark.asyncio
async def test_credit_failure_releases_nonce_and_payment_is_retried():
    ledger = await _make_ledger()
    recipient = generate_node_identity("recipient")
    sender = generate_node_identity("sender")
    ls = _ls(recipient, ledger)
    data = _signed(sender, recipient, 10.0)

    # Make the FIRST credit fail (transient — e.g. a concurrent-write savepoint collision), then heal.
    real_credit = ledger.credit
    n = {"calls": 0}

    async def flaky_credit(*a, **k):
        n["calls"] += 1
        if n["calls"] == 1:
            raise RuntimeError("simulated transient credit failure")
        return await real_credit(*a, **k)

    ledger.credit = flaky_credit

    # Delivery 1: the credit raises. The claimed nonce MUST be released (not strand the payment).
    with contextlib.suppress(Exception):
        await ls._on_ftns_transaction("transfer", data, sender.node_id)
    assert await ledger.get_balance(recipient.node_id) == 0.0            # credit did not apply
    assert await ledger.has_seen_nonce(data["nonce"]) is False           # ★ nonce released for retry

    # Delivery 2: gossip re-delivers the SAME signed tx → credit now succeeds → payment delivered.
    await ls._on_ftns_transaction("transfer", data, sender.node_id)
    assert await ledger.get_balance(recipient.node_id) == 10.0           # ★ no longer stranded


@pytest.mark.asyncio
async def test_successful_credit_keeps_nonce_claimed_no_double_credit():
    # The fix must NOT weaken sp898: a SUCCESSFUL credit leaves the nonce claimed so a re-gossiped copy
    # is rejected (no double-credit). This guard passes before + after the fix.
    ledger = await _make_ledger()
    recipient = generate_node_identity("recipient")
    sender = generate_node_identity("sender")
    ls = _ls(recipient, ledger)
    data = _signed(sender, recipient, 7.0)

    await ls._on_ftns_transaction("transfer", data, sender.node_id)
    assert await ledger.get_balance(recipient.node_id) == 7.0
    assert await ledger.has_seen_nonce(data["nonce"]) is True            # claimed, not released
    await ls._on_ftns_transaction("transfer", data, sender.node_id)      # re-delivery
    assert await ledger.get_balance(recipient.node_id) == 7.0            # NOT double-credited


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
