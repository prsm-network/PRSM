"""Sprint 1471 — LocalLedger parity hardening (from adversarial audit wf_2bd224fd).

The audit found LocalLedger (the NON-default sibling of DAGLedger) never received the DAGLedger
sp1468/sp1469 primitive-level hardening. All findings were LOW / latent (no reachable attacker-
controlled caller — every ingress is guarded upstream), but three are real, cheap fail-closed /
correctness gaps worth closing for parity:

  #1 credit() lacked math.isfinite: SQLite CHECK(amount>0) admits positive Infinity (inf>0 is True),
     so a non-finite credit would mint an unbounded, restart-surviving balance. Guard added at the
     _insert_tx choke point (covers credit/debit/transfer/agent_debit).
  #2 credit()'s idempotency branch swallowed ALL aiosqlite.IntegrityErrors and returned fake-success
     — a real (non-PK) violation was masked as an idempotent replay (silent under-credit). Now it
     re-verifies the row exists (a true PK collision) and re-raises otherwise.

(Finding #3 — an sp1466-class nonce stranding in content_uploader — is covered separately.)
"""
from __future__ import annotations

import math
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from prsm.node.local_ledger import LocalLedger, TransactionType

pytestmark = pytest.mark.asyncio


async def _ledger():
    lg = LocalLedger(":memory:")
    await lg.initialize()
    return lg


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), -1.0, 0.0])
async def test_credit_rejects_non_finite_or_non_positive(bad):
    lg = await _ledger()
    with pytest.raises(ValueError):
        await lg.credit("w", bad, TransactionType.REWARD)
    bal = await lg.get_balance("w")
    assert bal == 0.0 and math.isfinite(bal)          # no inf mint, no bad row


async def test_transfer_and_debit_reject_infinity_at_primitive():
    lg = await _ledger()
    await lg.credit("src", 1000.0, TransactionType.REWARD)
    with pytest.raises(ValueError):
        await lg.transfer("src", "dst", float("inf"))
    assert await lg.get_balance("src") == 1000.0      # intact
    with pytest.raises(ValueError):
        await lg.debit("src", float("inf"), TransactionType.TRANSFER)


async def test_legit_positive_credit_and_transfer_work():
    lg = await _ledger()
    await lg.credit("w", 42.5, TransactionType.REWARD)
    assert await lg.get_balance("w") == 42.5
    await lg.transfer("w", "x", 2.5)
    assert await lg.get_balance("w") == 40.0
    assert await lg.get_balance("x") == 2.5


async def test_idempotency_reraises_a_non_pk_integrity_error():
    # A genuine IntegrityError (not a same-tx_id PK collision) must NOT be masked as an idempotent
    # replay + returned as fake-success — otherwise a caller records the event PAID while nothing
    # was inserted (silent under-credit).
    lg = await _ledger()
    lg._insert_tx = AsyncMock(side_effect=aiosqlite.IntegrityError("simulated CHECK/FK violation"))
    with pytest.raises(aiosqlite.IntegrityError):
        await lg.credit("w", 5.0, TransactionType.REWARD, idempotency_key="key-1")


async def test_idempotency_still_dedups_a_true_replay():
    # The real exactly-once path is preserved: crediting the same idempotency_key twice applies once.
    lg = await _ledger()
    await lg.credit("w", 5.0, TransactionType.REWARD, idempotency_key="dep-42")
    await lg.credit("w", 5.0, TransactionType.REWARD, idempotency_key="dep-42")   # replay → no-op
    assert await lg.get_balance("w") == 5.0


# ── #3: content_uploader source-royalty credit failure releases the nonce (sp1466 parity) ──

async def test_content_royalty_credit_failure_releases_nonce():
    from unittest.mock import MagicMock, patch

    from prsm.node.content_uploader import ContentUploader

    identity = MagicMock()
    identity.node_id = "me-node"
    ledger = AsyncMock()
    ledger.record_nonce = AsyncMock(return_value=True)          # nonce claim succeeds (committed)
    ledger.credit = AsyncMock(side_effect=RuntimeError("transient credit failure"))
    ledger.release_nonce = AsyncMock(return_value=True)

    up = ContentUploader(
        identity=identity, gossip=AsyncMock(), ledger=ledger, transport=AsyncMock())
    up._platform_royalty_transfer = AsyncMock()                 # isolate the credit path
    parent = "parent-cid"
    up.uploaded_content[parent] = MagicMock(royalty_rate=0.05, total_royalties=0.0)

    data = {
        "content_id": "deriv-cid", "accessor_id": "acc", "creator_id": "other-creator",
        "parent_cids": [parent], "access_nonce": "n1",
        "signature": "sig", "origin_public_key": "pk",
    }
    with patch("prsm.node.content_uploader.verify_signature", return_value=True):
        await up._on_content_access("access", data, origin="origin-node")

    ledger.credit.assert_awaited()                              # attempted the royalty credit
    # ★ sp1466 parity: on credit failure the claimed nonce is RELEASED so a re-gossip can retry,
    # instead of stranding the royalty forever (every re-gossip would short-circuit at `if not claimed`).
    ledger.release_nonce.assert_awaited_once_with("content_access:origin-node:n1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
