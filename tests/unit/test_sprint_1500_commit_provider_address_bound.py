"""Sprint 1500 — the write-time provider-address check the docs said MUST exist.

`prsm/settlement/client_wiring.py`'s module docstring states it plainly:

    Brick 2 (the commit/finalize poll loop) MUST re-verify the signing key
    controls provider_address AT WRITE TIME — a view-only build binding does not
    prove key control, so before commitBatch the commit path must assert the
    key's eth address == provider_address again.

It was never implemented. `_commit_one`'s own docstring said the opposite: "the
Python client trusts the accumulator's keyed batches."

Why accumulate-time checking is not sufficient:
  * `_restore_pending` rehydrates batches from disk, so a batch can reach commit
    without passing through this process's accumulate path at all;
  * the accumulate gate compares against a configured STRING, which does not
    prove the signing key controls that address. The build-time guard in
    client_wiring does prove it — but only when a key was supplied at build.

commitBatch records `provider = msg.sender`. Committing a batch whose
provider_address is not this signer records the REAL provider as having done
nothing (paid zero on chain) while exposing THIS key's bond to a challenge for
work it never performed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.settlement.client import (
    BatchSettlementClient,
    ProviderAddressMismatchError,
)

MINE = "0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2"
THEIRS = "0x2aF5E504C6735fF1Fa07f19c145aF79B8AF459e9"


def _client(signer):
    c = object.__new__(BatchSettlementClient)
    contract = MagicMock()
    contract.address = signer
    contract.commit_batch = AsyncMock(return_value=(b"\x11" * 32, 1700000000))
    c._contract = contract
    return c


def _ready(provider):
    r = MagicMock()
    r.key = ("0xreq", provider, b"\x00" * 32, 500)
    r.batch.total_value_ftns = 10**18
    r.batch.receipts = [MagicMock(receipt=MagicMock(tee_attestation=None))]
    return r


async def _commit(client, ready):
    return await BatchSettlementClient._commit_one(
        client, ready, leaf_hashes=[b"\x22" * 32], root=b"\x33" * 32)


# ── the guard ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_committing_ANOTHER_providers_batch_is_REFUSED():
    """★ THE fix. commitBatch records provider = msg.sender, so this would credit
    the wrong party on chain and expose our bond to their challenges."""
    c = _client(MINE)
    with pytest.raises(ProviderAddressMismatchError, match="msg.sender"):
        await _commit(c, _ready(THEIRS))
    c._contract.commit_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_refusal_names_both_addresses():
    c = _client(MINE)
    with pytest.raises(ProviderAddressMismatchError) as e:
        await _commit(c, _ready(THEIRS))
    assert THEIRS in str(e.value) and MINE in str(e.value)


@pytest.mark.asyncio
async def test_our_OWN_batch_still_commits():
    c = _client(MINE)
    await _commit(c, _ready(MINE))
    c._contract.commit_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_compare_is_checksum_insensitive():
    """EIP-55 is case-only, so a lowercase address is the SAME address and must
    not be refused."""
    c = _client(MINE.lower())
    await _commit(c, _ready(MINE.upper().replace("0X", "0x")))
    c._contract.commit_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_0x_less_provider_address_is_tolerated():
    """client_wiring's build-time guard tolerates a 0x-less operator address; the
    write-time guard must agree, or the two disagree about the same config."""
    c = _client(MINE)
    await _commit(c, _ready(MINE[2:]))
    c._contract.commit_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_view_only_client_is_not_blocked():
    """No signing key => contract.address is None => nothing to verify against.
    Such a client cannot broadcast anyway; refusing here would break accumulation
    -only deployments that client_wiring explicitly allows."""
    c = _client(None)
    await _commit(c, _ready(THEIRS))
    c._contract.commit_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_non_address_signer_does_not_block_commits():
    """★ Deliberate scoping, and the reason this is not over-broad: a test double
    or an adapter that does not expose a signer reports something that is not an
    address. Comparing against it would refuse EVERY commit rather than the wrong
    ones. Only a real 0x address is enforceable."""
    from unittest.mock import AsyncMock as _AM
    c = _client(_AM(name="mock.address"))          # what a MagicMock yields
    await _commit(c, _ready(THEIRS))
    c._contract.commit_batch.assert_awaited_once()


# ── the documented requirement is now real ──────────────────────────

def test_client_wiring_still_documents_this_as_a_MUST():
    """If the doc requirement is ever dropped, this guard's rationale goes with
    it — keep them married."""
    from pathlib import Path
    src = Path("prsm/settlement/client_wiring.py").read_text()
    assert "AT WRITE TIME" in src


def test_commit_one_no_longer_claims_to_TRUST_the_accumulator():
    """★ The docstring asserted the very thing that was wrong."""
    import inspect

    # The DOCSTRING is the claim; the method body legitimately quotes the old
    # wording in a comment explaining what changed.
    doc = BatchSettlementClient._commit_one.__doc__ or ""
    assert "trusts the accumulator" not in doc
    assert "msg.sender" in doc, "the docstring should say what decides the payee"
    assert "ProviderAddressMismatchError" in inspect.getsource(
        BatchSettlementClient._commit_one)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
