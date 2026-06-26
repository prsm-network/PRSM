"""Sprint 1277 — settler multi-sig signature verification must be mandatory (audit round 7, HIGH).

SettlerRegistry.sign_batch only verified a signature `if getattr(settler, 'public_key_b64',
None)` — but the Settler dataclass had NO such field and register_settler never set one, so
the getattr was ALWAYS None, the verify branch was DEAD CODE, and every signature was accepted
("backward compatibility"). Any caller could submit 3 distinct active settler_ids with garbage
signatures, reach the threshold, and fire the on-chain settlement callback — the 3-of-N
multi-sig was purely cosmetic.

Fix: Settler carries public_key_b64; sign_batch ALWAYS verifies when a key is registered and
FAILS CLOSED when one isn't (a settler with no key cannot approve a batch), with a DEV-ONLY
PRSM_ALLOW_UNSIGNED_SETTLER_BATCH opt-in for legacy/local testing.
"""
from __future__ import annotations

import pytest

from prsm.node.identity import generate_node_identity
from prsm.node.settler_registry import SettlerRegistry


def _registry():
    return SettlerRegistry(min_settler_bond=1000.0, settlement_threshold=2, max_settlers=5)


def _sign(identity, batch, settler_id):
    return identity.sign(f"PRSM:{batch.batch_hash}:{settler_id}".encode())


@pytest.fixture(autouse=True)
def _no_dev_flag(monkeypatch):
    monkeypatch.delenv("PRSM_ALLOW_UNSIGNED_SETTLER_BATCH", raising=False)


@pytest.mark.asyncio
async def test_valid_signature_accepted():
    reg = _registry()
    idy = generate_node_identity()
    await reg.register_settler("s1", "0x1" + "0" * 39, 1000.0, public_key_b64=idy.public_key_b64)
    batch = await reg.propose_batch([{"to": "0xabc", "amount": 1.0, "tx_id": "t1"}])
    sig = await reg.sign_batch(batch.batch_id, "s1", _sign(idy, batch, "s1"))
    assert sig is not None
    assert batch.signature_count == 1


@pytest.mark.asyncio
async def test_forged_signature_rejected():
    reg = _registry()
    idy = generate_node_identity()
    await reg.register_settler("s1", "0x1" + "0" * 39, 1000.0, public_key_b64=idy.public_key_b64)
    batch = await reg.propose_batch([{"to": "0xabc", "amount": 1.0, "tx_id": "t1"}])
    with pytest.raises(ValueError, match="[Ii]nvalid signature"):
        await reg.sign_batch(batch.batch_id, "s1", "garbage-signature")
    assert batch.signature_count == 0


@pytest.mark.asyncio
async def test_keyless_settler_fails_closed_by_default():
    # the forge path: a settler with NO registered public key can no longer approve a batch
    reg = _registry()
    await reg.register_settler("s1", "0x1" + "0" * 39, 1000.0)   # no public_key_b64
    batch = await reg.propose_batch([{"to": "0xabc", "amount": 1.0, "tx_id": "t1"}])
    with pytest.raises(ValueError, match="no registered public key|cannot be verified"):
        await reg.sign_batch(batch.batch_id, "s1", "anything")
    assert batch.signature_count == 0


@pytest.mark.asyncio
async def test_forge_threequorum_no_longer_triggers_settlement():
    # the full exploit: 3 keyless settlers + garbage sigs must NOT reach threshold/settlement
    fired = {"n": 0}

    async def _on_ready(batch):
        fired["n"] += 1

    reg = SettlerRegistry(min_settler_bond=1000.0, settlement_threshold=2, max_settlers=5)
    reg._on_settlement_ready = _on_ready
    for i in (1, 2, 3):
        await reg.register_settler(f"s{i}", f"0x{i}" + "0" * 39, 1000.0)  # keyless
    batch = await reg.propose_batch([{"to": "0xabc", "amount": 1.0, "tx_id": "t1"}])
    for i in (1, 2, 3):
        with pytest.raises(ValueError):
            await reg.sign_batch(batch.batch_id, f"s{i}", "garbage")
    assert batch.signature_count == 0
    assert fired["n"] == 0   # settlement never triggered


@pytest.mark.asyncio
async def test_dev_flag_allows_unsigned(monkeypatch):
    monkeypatch.setenv("PRSM_ALLOW_UNSIGNED_SETTLER_BATCH", "1")
    reg = _registry()
    await reg.register_settler("s1", "0x1" + "0" * 39, 1000.0)
    batch = await reg.propose_batch([{"to": "0xabc", "amount": 1.0, "tx_id": "t1"}])
    sig = await reg.sign_batch(batch.batch_id, "s1", "unsigned-ok-in-dev")
    assert sig is not None and batch.signature_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
