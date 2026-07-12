"""sp1438 (audit B5 findings #1 + #2) — the two stranded-key fund-loss bugs on the LIVE Tier B/C
paid-decrypt path.

An adversarial audit of the paid-decrypt money+crypto path found the release gate, the on-chain
verifyPayment, the crypto, and the consumer commitment-substitution defense all SOUND, and exactly
two defects — both the same class: a buyer pays on-chain (ContentAccessVerifier.payForAccess pulls
FTNS, credits the creator, sets paid=true) but the wrapped key was never durably retained, so the
serve endpoint returns 404. The live CAV has NO refund function, so the loss is permanent.

  #1  node.py wired PaidKeyStore(... or None): an unset PRSM_PAID_KEY_STORE_FILE silently degraded
      the "DURABLE" store to in-memory, so a node restart wiped every retained key. Fixed with
      resolve_paid_key_store_path (durable ~/.prsm default, mirroring the settlement store).
  #2  deposit_commitment_and_retain deposited the on-chain gate BEFORE retaining the key — the
      reverse of the sibling publish_paid_content's sp1361 F5 invariant. A pending-then-mined
      deposit (or a crash/409 in between) left a live gate with no key. Fixed by retaining first.

These are money assertions (CLAUDE.md: never weaken to pass).
"""
from __future__ import annotations

import pytest

from prsm.economy.paid_content import (
    SquatMismatchError,
    deposit_commitment_and_retain,
)
from prsm.node.paid_key_serve import PaidKeyStore, resolve_paid_key_store_path


_CH = b"\x11" * 32
_COMMIT = b"\x22" * 32
_WRAPPED = b"the-wrapped-content-key-bytes"
_FEE = 10**18
_VERIFIER = "0x" + "33" * 20


# ── Finding #2 — retain the key BEFORE the on-chain deposit ───────────────────


class _RaisingKeyClient:
    """deposit_key raises like a pending-then-mined deposit (OnChainPendingError on the receipt
    wait). address=None so the anti-squat precondition short-circuits harmlessly."""
    address = None

    def deposit_key(self, *a, **k):
        raise RuntimeError("OnChainPendingError: receipt wait timed out (tx may still mine)")


class _OkKeyClient:
    address = None

    def deposit_key(self, ch, commitment, verifier, fee):
        return (b"\x01" * 32, "confirmed")


def test_key_is_retained_before_a_failing_deposit_so_it_is_not_stranded():
    """The money shot: deposit_key raises (pending-then-mined), yet the wrapped key was already
    retained — so a subsequently-mined deposit can't leave a live gate with no key."""
    store = PaidKeyStore()
    with pytest.raises(RuntimeError):
        deposit_commitment_and_retain(
            key_client=_RaisingKeyClient(), paid_key_store=store,
            content_hash=_CH, commitment=_COMMIT, wrapped_key=_WRAPPED,
            fee_wei=_FEE, verifier_address=_VERIFIER)
    entry = store.get(_CH)
    assert entry is not None, "key was NOT retained before the deposit — a paid buyer would 404"
    assert entry["wrapped_key"] == _WRAPPED


def test_success_path_still_retains_and_returns_tx():
    store = PaidKeyStore()
    tx, status = deposit_commitment_and_retain(
        key_client=_OkKeyClient(), paid_key_store=store,
        content_hash=_CH, commitment=_COMMIT, wrapped_key=_WRAPPED,
        fee_wei=_FEE, verifier_address=_VERIFIER)
    assert store.get(_CH)["wrapped_key"] == _WRAPPED
    assert (tx, status) == (b"\x01" * 32, "confirmed")


class _SquatKeyClient:
    """Someone ELSE already deposited this content_hash (a squat)."""
    address = "0x" + "cc" * 20

    class _Dep:
        publisher = "0x" + "dd" * 20

    def get_deposit(self, ch):
        return self._Dep()

    def deposit_key(self, *a, **k):
        raise AssertionError("must not deposit when the hash is squatted")


def test_squat_precondition_still_runs_first_and_does_not_retain():
    """Retain-first must not defeat the anti-squat guard: a squatted hash raises BEFORE retaining,
    so we never stash a key for content we can't legitimately gate."""
    store = PaidKeyStore()
    with pytest.raises(SquatMismatchError):
        deposit_commitment_and_retain(
            key_client=_SquatKeyClient(), paid_key_store=store,
            content_hash=_CH, commitment=_COMMIT, wrapped_key=_WRAPPED,
            fee_wei=_FEE, verifier_address=_VERIFIER)
    assert store.get(_CH) is None


# ── Finding #1 — the store path defaults DURABLE, never in-memory ─────────────


def test_resolve_path_defaults_durable_when_env_unset():
    """The bug: an unset PRSM_PAID_KEY_STORE_FILE yielded None (in-memory). The resolver must
    instead return a durable path — NEVER None/empty on the production path."""
    p = resolve_paid_key_store_path({})
    assert p, "resolver returned an empty/None path — the store would be in-memory (fund loss)"
    assert p.endswith("paid_key_store.json")
    assert ".prsm" in p


def test_resolve_path_honors_explicit_override():
    assert resolve_paid_key_store_path(
        {"PRSM_PAID_KEY_STORE_FILE": "/data/keys.json"}) == "/data/keys.json"
    # whitespace-only is treated as unset → durable default, not a broken "" path
    assert resolve_paid_key_store_path({"PRSM_PAID_KEY_STORE_FILE": "   "}).endswith(
        "paid_key_store.json")


def test_default_path_type_is_actually_durable_across_a_restart(tmp_path):
    """The resolved path drives a store that survives a node restart: put → reconstruct → get."""
    path = resolve_paid_key_store_path(
        {"PRSM_PAID_KEY_STORE_FILE": str(tmp_path / "ks.json")})
    s1 = PaidKeyStore(path)
    s1.put(_CH, _WRAPPED, _FEE)
    s2 = PaidKeyStore(path)  # "restart" — fresh store over the same file
    entry = s2.get(_CH)
    assert entry is not None and entry["wrapped_key"] == _WRAPPED
