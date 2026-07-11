"""Sprint 1421 — wire the requester's ONLY defense against an escrow drain.

THE LIVE HOLE (verified against the deployed Base-mainnet contracts):

  * `BatchSettlementRegistry.commitBatch(address requester, ...)` takes the requester as a
    PLAIN, UNVERIFIED argument. No signature. No authorization. No bond — the only `stakeBond`
    line in `_commitBatch` is `b.stakeBondAtCommit = address(stakeBond)`, an address snapshot.
    `provider` is simply `msg.sender`.
  * `finalizeBatch` is callable by ANYONE after the window, and calls
    `EscrowPool.settleFromRequester(requester, provider, value)`, which checks only (a) the
    caller is the registry and (b) the victim HAS the balance — then transfers.
  * So ANY address can commit a batch naming ANY requester and drain their escrow for gas.
  * The victim's SOLE defense is a NO_ESCROW challenge, and `_handleNoEscrow` reverts unless
    `msg.sender == b.requester` — nobody else can defend them.

And that defense was DEAD end-to-end. Raising NO_ESCROW means proving "I never authorized this
batch", which requires knowing what you DID authorize — the IssuedAuthorizationStore. But every
SDK payment path called the plain `build_payment_authorization`, never the recording wrapper, so
the store was **empty**: `build_and_record_payment_authorization`, `match_unauthorized_batches`
and `assemble_no_escrow_challenges` all had ZERO production callers.

An empty store does not merely disable the defense — it makes it USELESS EVEN IF WIRED, because
the matcher cannot then tell the requester's own legitimate batches from an attacker's forgery.
`test_an_empty_store_cannot_defend_at_all` pins exactly that, so nobody "optimizes away" the
recording later without the suite going red.

These tests do NOT hardcode any private key: each generates an ephemeral one.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest
from eth_account import Account

from prsm.sdk.client import PRSMClient
from prsm.settlement.issued_authorization_store import (
    AUTHORIZED,
    UNAUTHORIZED,
    CommittedBatchView,
    IssuedAuthorizationStore,
    match_unauthorized_batches,
)

PROVIDER = "0x000000000000000000000000000000000000BEEF"
ATTACKER = "0x00000000000000000000000000000000000ATTACK".replace("ATTACK", "0dead")


@pytest.fixture
def victim():
    """An ephemeral requester identity — never a real or hardcoded key."""
    return Account.create()


@pytest.fixture
def client(tmp_path, victim):
    c = PRSMClient("http://node", issued_auth_store_path=tmp_path / "issued.json")
    c._get = AsyncMock(return_value={"operator_address": PROVIDER})
    c._post = AsyncMock(return_value={"ok": True, "output": "hi"})
    return c


async def _pay(client, victim, *, max_spend=5.0, budget=5.0):
    return await client.pay_and_infer(
        prompt="hello",
        model_id="gpt2",
        requester_key=victim.key.hex(),
        budget_ftns=budget,
        max_spend_ftns=max_spend,
        provider_address=PROVIDER,
        expiry_unix=int(time.time()) + 300,
        chain_id=8453,
    )


class TestTheStoreIsActuallyPopulated:
    async def test_pay_and_infer_records_the_issued_authorization(self, client, victim):
        """The wiring itself. Pre-1421 this recorded NOTHING, so the store stayed empty."""
        store = client._issued_auth_store()
        assert store is not None, "retention is ON by default"
        assert store.all_authorizations() == [], "precondition: store starts empty"

        await _pay(client, victim)

        recorded = store.all_authorizations()
        assert len(recorded) == 1, (
            "the SDK paid but recorded no authorization — the requester now has NO evidence "
            "with which to dispute a forged batch against its escrow"
        )
        assert recorded[0].provider.lower() == PROVIDER.lower()
        assert recorded[0].job_nonce, "recorded auth must be keyed by its job_nonce"

    async def test_retention_survives_a_fresh_client_same_path(self, tmp_path, victim):
        """The evidence must outlive the process — a drain is disputed LATER, not in-process."""
        path = tmp_path / "issued.json"
        c1 = PRSMClient("http://node", issued_auth_store_path=path)
        c1._get = AsyncMock(return_value={"operator_address": PROVIDER})
        c1._post = AsyncMock(return_value={"ok": True})
        await _pay(c1, victim)

        c2 = PRSMClient("http://node", issued_auth_store_path=path)
        assert len(c2._issued_auth_store().all_authorizations()) == 1, (
            "issued authorizations did not persist — the requester loses its evidence on restart"
        )


class TestTheNoEscrowDefenseCanNowFire:
    """The point of the sprint: the matcher can finally tell a real batch from a forged one."""

    async def test_forged_batch_is_flagged_unauthorized_and_the_real_one_is_not(
        self, client, victim,
    ):
        await _pay(client, victim, max_spend=5.0)
        store = client._issued_auth_store()
        me = victim.address

        now = int(time.time())
        legit = CommittedBatchView(
            batch_id=b"\x01" * 32, requester=me, provider=PROVIDER,
            total_value_wei=int(2e18), commit_timestamp=now,
        )
        # The attack: someone the victim never paid, naming the victim as requester.
        forged = CommittedBatchView(
            batch_id=b"\x02" * 32, requester=me, provider=ATTACKER,
            total_value_wei=int(2e18), commit_timestamp=now,
        )

        by_id = {
            bytes(c.batch_id): c
            for c in match_unauthorized_batches([legit, forged], store, my_address=me)
        }

        assert by_id[b"\x01" * 32].classification == AUTHORIZED, (
            "the requester's OWN authorized batch was flagged unauthorized — a false NO_ESCROW "
            "would grief an honest provider and waste the requester's gas"
        )
        assert by_id[b"\x02" * 32].classification == UNAUTHORIZED, (
            "the FORGED batch was not flagged — the escrow-drain defense still cannot fire"
        )

    async def test_an_empty_store_cannot_defend_at_all(self, victim):
        """Pin the pre-1421 state so the recording can never be quietly removed.

        With no recorded authorizations the matcher flags the requester's own LEGITIMATE batch
        as UNAUTHORIZED too — it has nothing to compare against. So an unwired store doesn't
        just disable the defense, it makes it indistinguishable from noise.
        """
        empty = IssuedAuthorizationStore(":memory:")
        me = victim.address
        legit = CommittedBatchView(
            batch_id=b"\x01" * 32, requester=me, provider=PROVIDER,
            total_value_wei=int(2e18), commit_timestamp=int(time.time()),
        )
        [result] = match_unauthorized_batches([legit], empty, my_address=me)
        assert result.classification == UNAUTHORIZED, (
            "an empty store should flag even a legitimate batch — proving the matcher is blind "
            "without the recording this sprint wires up"
        )

    async def test_batches_naming_someone_else_are_ignored(self, client, victim):
        """R can only self-dispute its OWN escrow (_handleNoEscrow requires msg.sender == requester)."""
        await _pay(client, victim)
        other = Account.create().address
        someone_elses = CommittedBatchView(
            batch_id=b"\x03" * 32, requester=other, provider=ATTACKER,
            total_value_wei=int(9e18), commit_timestamp=int(time.time()),
        )
        assert match_unauthorized_batches(
            [someone_elses], client._issued_auth_store(), my_address=victim.address,
        ) == [], "must not classify a batch drawn against someone else's escrow"


class TestRecordingIsFailSafe:
    async def test_a_broken_store_never_breaks_the_payment(self, tmp_path, victim, monkeypatch):
        """Retention guards no funds directly; it must NEVER take down the money path."""
        c = PRSMClient("http://node", issued_auth_store_path=tmp_path / "issued.json")
        c._get = AsyncMock(return_value={"operator_address": PROVIDER})
        c._post = AsyncMock(return_value={"ok": True, "output": "hi"})

        store = c._issued_auth_store()
        monkeypatch.setattr(
            store, "record",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )
        result = await _pay(c, victim)  # must not raise
        assert result["ok"] is True, "a store failure broke the payment — unacceptable"

    async def test_retention_can_be_disabled(self, tmp_path, victim, monkeypatch):
        monkeypatch.setenv("PRSM_ISSUED_AUTH_STORE", "0")
        c = PRSMClient("http://node", issued_auth_store_path=tmp_path / "issued.json")
        c._get = AsyncMock(return_value={"operator_address": PROVIDER})
        c._post = AsyncMock(return_value={"ok": True})
        assert c._issued_auth_store() is None
        await _pay(c, victim)  # still pays

    async def test_the_signed_auth_is_unchanged_by_recording(self, client, victim):
        """Recording must not perturb the SIGNED money payload in any way."""
        from prsm.settlement.payment_client import build_payment_authorization

        kwargs = dict(
            requester_key=victim.key.hex(), provider_address=PROVIDER, model_id="gpt2",
            prompt="hello", max_tokens=0, privacy_tier="none", content_tier="none",
            max_spend_ftns=5.0, expiry_unix=int(time.time()) + 300, chain_id=8453,
            job_nonce="0x" + "ab" * 32,  # job_nonce is a bytes32 — must be hex
        )
        plain = build_payment_authorization(**kwargs)

        from prsm.settlement.issued_authorization_store import (
            build_and_record_payment_authorization,
        )
        recorded = build_and_record_payment_authorization(
            store=client._issued_auth_store(), **kwargs,
        )
        assert recorded["payload"] == plain["payload"]
        assert recorded["signature"] == plain["signature"]
