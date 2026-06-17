"""Sprint 1146 — requester-side issued-authorization store + CONSERVATIVE
NO_ESCROW matcher (requester self-dispute foundation).

NO_ESCROW (reason code 2) is a REQUESTER self-dispute: on chain ``_handleNoEscrow``
only checks ``msg.sender == batch.requester``. So the requester R needs a LOCAL way
to recognize a committed batch (naming R as requester, against R's escrow) that R
never authorized — by comparing committed batches to the auths R issued.

These tests pin:
  (a) IssuedAuthorizationStore round-trips an issued auth byte-exact; link_job
      updates; prune/evict bound; never-raises on a malformed on-disk entry.
  (b) build_and_record_payment_authorization with store=None is byte-identical to
      build_payment_authorization; with a store it ALSO records the issued auth.
  (c) MATCHER SOUNDNESS (critical): a batch covered by a recorded auth (provider +
      value <= max_spend + not-expired) -> AUTHORIZED; an exact job-link -> AUTHORIZED.
      NEVER a false UNAUTHORIZED for an authorized batch.
  (d) UNAUTHORIZED cases: no auth to that provider; value exceeds every auth; auth
      expired before commit_timestamp (and no other covering auth).
  (e) batches where requester != my_address are skipped.
  (f) end-to-end: record several auths, classify a mix, exactly the unauthorized
      one is the NO_ESCROW candidate.
"""
from __future__ import annotations

import json

import pytest
from eth_account import Account
from eth_utils import keccak

from prsm.settlement.payment_client import build_payment_authorization
from prsm.settlement.issued_authorization_store import (
    AUTHORIZED,
    UNAUTHORIZED,
    BatchAuthorizationClassification,
    CommittedBatchView,
    IssuedAuthorization,
    IssuedAuthorizationStore,
    build_and_record_payment_authorization,
    match_unauthorized_batches,
)

WEI = 10 ** 18


def _key():
    return Account.create()


def _auth_payload(*, requester_key, provider, max_spend_ftns, expiry_unix,
                  job_nonce=None, model_id="m", prompt="p", max_tokens=10,
                  privacy_tier="standard", content_tier="standard"):
    return build_payment_authorization(
        requester_key=requester_key,
        provider_address=provider,
        model_id=model_id,
        prompt=prompt,
        max_tokens=max_tokens,
        privacy_tier=privacy_tier,
        content_tier=content_tier,
        max_spend_ftns=max_spend_ftns,
        expiry_unix=expiry_unix,
        job_nonce=job_nonce,
        chain_id=8453,
    )


# ── (a) store round-trip / link / prune / evict / malformed ──────────


def test_store_record_get_round_trips_byte_exact(tmp_path):
    rk = _key().key
    prov = _key().address
    auth = _auth_payload(requester_key=rk, provider=prov,
                         max_spend_ftns="5", expiry_unix=2_000_000_000)

    store = IssuedAuthorizationStore(tmp_path / "auths.json", now=lambda: 1000.0)
    rec = store.record(auth, linked_job_id=None)
    assert isinstance(rec, IssuedAuthorization)

    got = store.get(auth["payload"]["job_nonce"])
    assert got is not None
    # byte-exact on the load-bearing auth fields
    assert got.requester == auth["payload"]["requester"]
    assert got.provider == auth["payload"]["provider"]
    assert got.max_spend_wei == auth["payload"]["max_spend_wei"]
    assert got.job_nonce == auth["payload"]["job_nonce"]
    assert got.expiry_unix == auth["payload"]["expiry_unix"]
    assert got.request_hash == auth["payload"]["request_hash"]

    # survives a reload from disk byte-exact
    store2 = IssuedAuthorizationStore(tmp_path / "auths.json")
    got2 = store2.get(auth["payload"]["job_nonce"])
    assert got2 == got


def test_authorizations_for_provider_filters(tmp_path):
    rk = _key().key
    prov_a = _key().address
    prov_b = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    store.record(_auth_payload(requester_key=rk, provider=prov_a,
                               max_spend_ftns="1", expiry_unix=2_000_000_000))
    store.record(_auth_payload(requester_key=rk, provider=prov_a,
                               max_spend_ftns="2", expiry_unix=2_000_000_000))
    store.record(_auth_payload(requester_key=rk, provider=prov_b,
                               max_spend_ftns="3", expiry_unix=2_000_000_000))
    assert len(store.authorizations_for_provider(prov_a)) == 2
    assert len(store.authorizations_for_provider(prov_b)) == 1
    # case-insensitive address match
    assert len(store.authorizations_for_provider(prov_a.lower())) == 2


def test_link_job_updates(tmp_path):
    rk = _key().key
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    auth = _auth_payload(requester_key=rk, provider=prov,
                         max_spend_ftns="1", expiry_unix=2_000_000_000)
    store.record(auth)
    nonce = auth["payload"]["job_nonce"]
    assert store.get(nonce).linked_job_id is None
    store.link_job(nonce, "job-xyz")
    assert store.get(nonce).linked_job_id == "job-xyz"
    # persisted
    store2 = IssuedAuthorizationStore(tmp_path / "a.json")
    assert store2.get(nonce).linked_job_id == "job-xyz"


def test_prune_drops_expired_issued(tmp_path):
    rk = _key().key
    prov = _key().address
    clock = {"t": 1000.0}
    store = IssuedAuthorizationStore(tmp_path / "a.json", now=lambda: clock["t"])
    auth = _auth_payload(requester_key=rk, provider=prov,
                         max_spend_ftns="1", expiry_unix=2_000_000_000)
    store.record(auth)
    nonce = auth["payload"]["job_nonce"]
    assert store.get(nonce) is not None
    clock["t"] = 1000.0 + 100_000
    dropped = store.prune(retention_seconds=10)
    assert dropped == 1
    assert store.get(nonce) is None


def test_bounded_evicts_oldest(tmp_path):
    rk = _key().key
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json", max_entries=2)
    a1 = _auth_payload(requester_key=rk, provider=prov, max_spend_ftns="1",
                       expiry_unix=2_000_000_000)
    a2 = _auth_payload(requester_key=rk, provider=prov, max_spend_ftns="1",
                       expiry_unix=2_000_000_000)
    a3 = _auth_payload(requester_key=rk, provider=prov, max_spend_ftns="1",
                       expiry_unix=2_000_000_000)
    store.record(a1)
    store.record(a2)
    store.record(a3)
    assert store.get(a1["payload"]["job_nonce"]) is None  # evicted oldest
    assert store.get(a2["payload"]["job_nonce"]) is not None
    assert store.get(a3["payload"]["job_nonce"]) is not None


def test_never_raises_on_malformed_disk_entry(tmp_path):
    rk = _key().key
    prov = _key().address
    path = tmp_path / "a.json"
    store = IssuedAuthorizationStore(path)
    good = _auth_payload(requester_key=rk, provider=prov, max_spend_ftns="1",
                         expiry_unix=2_000_000_000)
    store.record(good)
    nonce = good["payload"]["job_nonce"]

    # corrupt the file: inject a malformed entry alongside the good one
    raw = json.loads(path.read_text())
    raw["authorizations"]["0xdeadbeef"] = {"not": "a valid auth"}
    path.write_text(json.dumps(raw))

    # reload must NOT raise; good entry survives, bad one skipped
    store2 = IssuedAuthorizationStore(path)
    assert store2.get(nonce) is not None
    assert store2.get("0xdeadbeef") is None


def test_never_raises_on_unparseable_file(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("{ this is not json")
    store = IssuedAuthorizationStore(path)  # must not raise
    assert store.get("0x00") is None


# ── (b) build_and_record pass-through / record ───────────────────────


def test_build_and_record_store_none_byte_identical():
    rk = _key().key
    prov = _key().address
    nonce = "0x" + "11" * 32
    direct = build_payment_authorization(
        requester_key=rk, provider_address=prov, model_id="m", prompt="p",
        max_tokens=10, privacy_tier="standard", content_tier="standard",
        max_spend_ftns="5", expiry_unix=2_000_000_000, job_nonce=nonce,
        chain_id=8453,
    )
    wrapped = build_and_record_payment_authorization(
        store=None, requester_key=rk, provider_address=prov, model_id="m",
        prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5",
        expiry_unix=2_000_000_000, job_nonce=nonce, chain_id=8453,
    )
    assert wrapped == direct  # byte-identical


def test_build_and_record_with_store_records(tmp_path):
    rk = _key().key
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    auth = build_and_record_payment_authorization(
        store=store, requester_key=rk, provider_address=prov, model_id="m",
        prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5",
        expiry_unix=2_000_000_000,
    )
    nonce = auth["payload"]["job_nonce"]
    rec = store.get(nonce)
    assert rec is not None
    assert rec.provider == prov
    assert rec.max_spend_wei == auth["payload"]["max_spend_wei"]


def test_build_and_record_with_linked_job(tmp_path):
    rk = _key().key
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    auth = build_and_record_payment_authorization(
        store=store, linked_job_id="job-1", requester_key=rk,
        provider_address=prov, model_id="m", prompt="p", max_tokens=10,
        privacy_tier="standard", content_tier="standard", max_spend_ftns="5",
        expiry_unix=2_000_000_000,
    )
    assert store.get(auth["payload"]["job_nonce"]).linked_job_id == "job-1"


# ── (c) MATCHER SOUNDNESS — never false-positive ─────────────────────


def test_matcher_heuristic_match_is_authorized(tmp_path):
    rk_acct = _key()
    me = rk_acct.address
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5", expiry_unix=2_000_000_000,
    )
    batch = CommittedBatchView(
        batch_id=b"\x01" * 32, requester=me, provider=prov,
        total_value_wei=3 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert len(out) == 1
    assert out[0].classification == AUTHORIZED
    assert out[0].match_basis == "provider-value-heuristic"


def test_matcher_exact_job_link_is_authorized(tmp_path):
    rk_acct = _key()
    me = rk_acct.address
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    auth = build_and_record_payment_authorization(
        store=store, linked_job_id="job-77", requester_key=rk_acct.key,
        provider_address=prov, model_id="m", prompt="p", max_tokens=10,
        privacy_tier="standard", content_tier="standard",
        max_spend_ftns="5", expiry_unix=2_000_000_000,
    )
    # value EXCEEDS max_spend so the heuristic would NOT cover it — only the
    # exact job-link can authorize it.
    job_id_hash = keccak(text="job-77")
    batch = CommittedBatchView(
        batch_id=b"\x02" * 32, requester=me, provider=prov,
        total_value_wei=99 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=[job_id_hash],
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert out[0].classification == AUTHORIZED
    assert out[0].match_basis == "exact-job-link"


def test_matcher_value_equal_to_max_spend_is_authorized(tmp_path):
    # boundary: value == max_spend must be AUTHORIZED (>= is the rule)
    rk_acct = _key()
    me = rk_acct.address
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5", expiry_unix=2_000_000_000,
    )
    batch = CommittedBatchView(
        batch_id=b"\x03" * 32, requester=me, provider=prov,
        total_value_wei=5 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert out[0].classification == AUTHORIZED


def test_matcher_expiry_equal_to_commit_is_authorized(tmp_path):
    # boundary: not-expired means expiry_unix >= commit_timestamp
    rk_acct = _key()
    me = rk_acct.address
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5", expiry_unix=1_000_000,
    )
    batch = CommittedBatchView(
        batch_id=b"\x04" * 32, requester=me, provider=prov,
        total_value_wei=1 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert out[0].classification == AUTHORIZED


# ── (d) UNAUTHORIZED cases ───────────────────────────────────────────


def test_matcher_no_auth_to_provider_is_unauthorized(tmp_path):
    rk_acct = _key()
    me = rk_acct.address
    prov_authed = _key().address
    prov_attacker = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov_authed,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5", expiry_unix=2_000_000_000,
    )
    batch = CommittedBatchView(
        batch_id=b"\x05" * 32, requester=me, provider=prov_attacker,
        total_value_wei=1 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert out[0].classification == UNAUTHORIZED
    assert out[0].match_basis == "none"


def test_matcher_value_exceeds_every_auth_is_unauthorized(tmp_path):
    rk_acct = _key()
    me = rk_acct.address
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5", expiry_unix=2_000_000_000,
    )
    batch = CommittedBatchView(
        batch_id=b"\x06" * 32, requester=me, provider=prov,
        total_value_wei=100 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert out[0].classification == UNAUTHORIZED


def test_matcher_expired_before_commit_is_unauthorized(tmp_path):
    rk_acct = _key()
    me = rk_acct.address
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5", expiry_unix=999_999,
    )
    batch = CommittedBatchView(
        batch_id=b"\x07" * 32, requester=me, provider=prov,
        total_value_wei=1 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert out[0].classification == UNAUTHORIZED


def test_matcher_covering_auth_among_noncovering_is_authorized(tmp_path):
    # one expired + one low-value + one COVERING auth to same provider -> AUTHORIZED
    rk_acct = _key()
    me = rk_acct.address
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    # expired
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="100", expiry_unix=999_999,
    )
    # too small
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="1", expiry_unix=2_000_000_000,
    )
    # covering
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="50", expiry_unix=2_000_000_000,
    )
    batch = CommittedBatchView(
        batch_id=b"\x08" * 32, requester=me, provider=prov,
        total_value_wei=10 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert out[0].classification == AUTHORIZED


# ── (e) requester != my_address is skipped ───────────────────────────


def test_matcher_skips_batches_for_other_requesters(tmp_path):
    me = _key().address
    other = _key().address
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")  # empty
    batch = CommittedBatchView(
        batch_id=b"\x09" * 32, requester=other, provider=prov,
        total_value_wei=1 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert out == []  # not the requester's concern


def test_matcher_requester_match_case_insensitive(tmp_path):
    rk_acct = _key()
    me = rk_acct.address
    prov = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")
    build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5", expiry_unix=2_000_000_000,
    )
    batch = CommittedBatchView(
        batch_id=b"\x0a" * 32, requester=me.lower(), provider=prov,
        total_value_wei=1 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    out = match_unauthorized_batches([batch], store, my_address=me)
    assert len(out) == 1
    assert out[0].classification == AUTHORIZED


# ── (f) end-to-end ───────────────────────────────────────────────────


def test_end_to_end_exactly_one_unauthorized(tmp_path):
    rk_acct = _key()
    me = rk_acct.address
    prov_good = _key().address
    prov_attacker = _key().address
    store = IssuedAuthorizationStore(tmp_path / "a.json")

    a1 = build_and_record_payment_authorization(
        store=store, requester_key=rk_acct.key, provider_address=prov_good,
        model_id="m", prompt="p", max_tokens=10, privacy_tier="standard",
        content_tier="standard", max_spend_ftns="5", expiry_unix=2_000_000_000,
    )
    a2 = build_and_record_payment_authorization(
        store=store, linked_job_id="job-link-1", requester_key=rk_acct.key,
        provider_address=prov_good, model_id="m2", prompt="p2", max_tokens=20,
        privacy_tier="standard", content_tier="standard", max_spend_ftns="2",
        expiry_unix=2_000_000_000,
    )

    authorized_heuristic = CommittedBatchView(
        batch_id=b"\xa1" * 32, requester=me, provider=prov_good,
        total_value_wei=4 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    authorized_exact = CommittedBatchView(
        batch_id=b"\xa2" * 32, requester=me, provider=prov_good,
        total_value_wei=1 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=[keccak(text="job-link-1")],
    )
    unauthorized = CommittedBatchView(
        batch_id=b"\xbb" * 32, requester=me, provider=prov_attacker,
        total_value_wei=7 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )
    other_requester = CommittedBatchView(
        batch_id=b"\xcc" * 32, requester=_key().address, provider=prov_attacker,
        total_value_wei=7 * WEI, commit_timestamp=1_000_000,
        leaf_job_id_hashes=None,
    )

    out = match_unauthorized_batches(
        [authorized_heuristic, authorized_exact, unauthorized, other_requester],
        store, my_address=me,
    )
    # other_requester skipped -> 3 results
    assert len(out) == 3
    candidates = [c for c in out if c.classification == UNAUTHORIZED]
    assert len(candidates) == 1
    assert candidates[0].batch_id == b"\xbb" * 32
    assert candidates[0].provider == prov_attacker
