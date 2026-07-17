"""Sprint 1468 — DAGLedger money-audit: value-integrity guards (non-finite + negative amounts).

Adversarial audit of the core money primitive (workflow wf_6ceaaeff) found:
  • CRITICAL: the P2P gossip credit path guarded only `amount <= 0`, which NaN/Infinity bypass
    (inf<=0 and nan<=0 are both False) → Infinity mints an unbounded, restart-surviving spendable
    balance; NaN freezes any targeted node's wallet. Remote, unauthenticated.
  • MEDIUM: submit_transaction (the shared primitive under credit/debit/transfer AND the gossip
    credit) did ZERO amount validation, so a negative amount is a durable reverse-transfer (adds to
    the sender, subtracts from the receiver) with no last-line defense.
  • wire: P2PMessage.from_json used default json.loads, so Infinity/NaN JSON tokens round-trip as
    real floats.

Fix (defense in depth): reject non-finite/negative IN the primitive (submit_transaction); keep the
early gossip guard but make it finiteness-aware; and reject the NaN/Infinity tokens at JSON ingress.
"""
from __future__ import annotations

import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.dag_ledger import DAGLedger
from prsm.node.identity import generate_node_identity
from prsm.node.ledger_sync import LedgerSync
from prsm.node.local_ledger import Transaction, TransactionType

pytestmark = pytest.mark.asyncio


async def _ledger():
    lg = DAGLedger(":memory:", verify_signatures=False)
    await lg.initialize()
    return lg


# ── #3: the shared primitive rejects non-finite + negative ─────────────────────

@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), -1.0, -0.5])
async def test_credit_rejects_non_finite_or_negative(bad):
    lg = await _ledger()
    with pytest.raises(ValueError):
        await lg.credit("wallet-a", bad, TransactionType.REWARD)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -100.0])
async def test_transfer_rejects_non_finite_or_negative(bad):
    lg = await _ledger()
    await lg.credit("sender", 1000.0, TransactionType.REWARD)
    with pytest.raises(ValueError):
        await lg.transfer("sender", "receiver", bad)
    # sender's balance is intact — the reverse-transfer never applied.
    assert await lg.get_balance("sender") == 1000.0


async def test_infinity_credit_does_not_mint_unbounded_balance():
    lg = await _ledger()
    with pytest.raises(ValueError):
        await lg.credit("attacker", float("inf"), TransactionType.REWARD)
    bal = await lg.get_balance("attacker")
    assert bal == 0.0                      # no inf balance minted
    import math
    assert math.isfinite(bal)


async def test_legit_positive_amount_still_works():
    lg = await _ledger()
    await lg.credit("w", 42.5, TransactionType.REWARD)
    assert await lg.get_balance("w") == 42.5


# ── #2: the gossip credit path rejects non-finite (no mint / no freeze) ─────────

def _signed_incoming(sender, recipient, amount):
    tx = Transaction(
        tx_id=str(uuid.uuid4()), tx_type=TransactionType.TRANSFER,
        from_wallet=sender.node_id, to_wallet=recipient.node_id, amount=amount,
        description="x", timestamp=time.time(), signature="")
    canonical = LedgerSync._canonical_tx_payload(tx, tx.tx_id)
    sig = sender.sign(json.dumps(canonical, sort_keys=True).encode())
    return {**canonical, "signature": sig, "origin_public_key": sender.public_key_b64}


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
async def test_gossip_credit_rejects_non_finite_amount(bad):
    from prsm.node.local_ledger import LocalLedger
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    recipient = generate_node_identity("recipient")
    sender = generate_node_identity("sender")
    ls = LedgerSync(identity=recipient, gossip=MagicMock(), ledger=ledger, transport=MagicMock())

    data = _signed_incoming(sender, recipient, bad)
    await ls._on_ftns_transaction("transfer", data, sender.node_id)

    bal = await ledger.get_balance(recipient.node_id)
    import math
    assert bal == 0.0 and math.isfinite(bal)         # NOT credited inf/nan → no mint, no NULL-freeze


# ── wire: JSON ingress rejects NaN/Infinity tokens ─────────────────────────────

@pytest.mark.parametrize("token", ["Infinity", "-Infinity", "NaN"])
async def test_p2p_from_json_rejects_non_finite_tokens(token):
    from prsm.node.transport import P2PMessage
    raw = json.dumps({"msg_type": "direct", "sender_id": "s", "payload": {}})
    # splice a non-finite token into the payload amount
    raw_bad = raw.replace('"payload": {}', f'"payload": {{"amount": {token}}}')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        P2PMessage.from_json(raw_bad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
