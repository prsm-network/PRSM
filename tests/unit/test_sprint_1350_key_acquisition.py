"""Sprint 1350 — Tier B/C paid-decrypt consumer arc, brick 2: on-chain key acquisition.

A consumer who has PAID the release fee triggers KeyDistribution.release + reads the KeyReleased
event to get the deposited encrypted_key. Idempotent (already-released → no re-trigger), fail-loud
(unpaid / not-deposited / no-event → KeyNotReleasedError). The end-to-end test ties brick 2's
acquired key into brick 1's reconstruct → plaintext. Mocked client (the live chain is user-gated).
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.key_acquisition import KeyNotReleasedError, acquire_released_key
from prsm.economy.web3.key_distribution import KeyNotFoundError, PaymentNotVerifiedError
from prsm.economy.web3.provenance_registry import OnChainRevertedError

_CH = bytes.fromhex("ab" * 32)
_BUYER = "0x" + "11" * 20


class _Ev:
    def __init__(self, content_hash, recipient, encrypted_key):
        self.content_hash = content_hash
        self.recipient = recipient
        self.encrypted_key = encrypted_key


class _FakeClient:
    def __init__(self, *, released=None, release_error=None, emit_on_release=None):
        self._released = list(released or [])
        self._release_error = release_error
        self._emit = emit_on_release
        self.release_calls = []

    def latest_block(self):
        return 100

    def get_key_released_events(self, from_block, to_block, *, argument_filters=None):
        return list(self._released)

    def release(self, content_hash, recipient):
        self.release_calls.append((content_hash, recipient))
        if self._release_error is not None:
            raise self._release_error
        if self._emit is not None:
            self._released.append(self._emit)
        return ("0xtx", None)


# ── idempotent read + release-then-read ───────────────────────────────────────

def test_already_released_returns_key_without_re_triggering():
    client = _FakeClient(released=[_Ev(_CH, _BUYER, b"wrapped-key-A")])
    assert acquire_released_key(client, _CH, _BUYER) == b"wrapped-key-A"
    assert client.release_calls == []                      # idempotent — no re-release


def test_not_released_triggers_release_then_reads():
    client = _FakeClient(emit_on_release=_Ev(_CH, _BUYER, b"wrapped-key-B"))
    assert acquire_released_key(client, _CH, _BUYER) == b"wrapped-key-B"
    assert client.release_calls == [(_CH, _BUYER)]


def test_matches_on_content_and_recipient_not_a_different_buyer():
    client = _FakeClient(released=[_Ev(_CH, "0x" + "99" * 20, b"someone-elses-key")],
                         emit_on_release=_Ev(_CH, _BUYER, b"my-key"))
    assert acquire_released_key(client, _CH, _BUYER) == b"my-key"   # ignores the other recipient
    assert client.release_calls == [(_CH, _BUYER)]                  # had to trigger for us


# ── fail-loud ──────────────────────────────────────────────────────────────────

def test_unpaid_release_reverts_to_pay_first():
    client = _FakeClient(release_error=PaymentNotVerifiedError("PaymentNotVerified"))
    with pytest.raises(KeyNotReleasedError, match="PAY the release fee"):
        acquire_released_key(client, _CH, _BUYER)


def test_not_deposited_surfaces_clearly():
    client = _FakeClient(release_error=KeyNotFoundError("KeyNotFound"))
    with pytest.raises(KeyNotReleasedError, match="never called deposit_key"):
        acquire_released_key(client, _CH, _BUYER)


def test_generic_revert_surfaces():
    client = _FakeClient(release_error=OnChainRevertedError("reverted", "0xtx"))
    with pytest.raises(KeyNotReleasedError, match="reverted"):
        acquire_released_key(client, _CH, _BUYER)


def test_trigger_release_false_when_not_released_raises():
    client = _FakeClient()
    with pytest.raises(KeyNotReleasedError, match="trigger_release=False"):
        acquire_released_key(client, _CH, _BUYER, trigger_release=False)
    assert client.release_calls == []


def test_release_sent_but_no_event_raises():
    client = _FakeClient()  # release succeeds but emits nothing
    with pytest.raises(KeyNotReleasedError, match="no matching KeyReleased event"):
        acquire_released_key(client, _CH, _BUYER)


# ── end-to-end: brick 2 acquire → brick 1 reconstruct → plaintext ─────────────

def test_acquired_key_reconstructs_plaintext_end_to_end():
    from prsm.enterprise.recipient_encryption import (
        EnterpriseRecipient, generate_recipient_keypair,
    )
    from prsm.storage.encryption import encrypt, generate_key
    from prsm.storage.paid_unlock import (
        reconstruct_paid_content, wrap_content_key_for_deposit,
    )

    plaintext = b"paywalled research dataset"
    content_key = generate_key()
    content = encrypt(plaintext, content_key)                       # served free (ciphertext)
    buyer_priv, buyer_pub = generate_recipient_keypair()
    wrapped = wrap_content_key_for_deposit(                          # the deposited encrypted_key
        content_key, [EnterpriseRecipient(identifier="buyer", x25519_pubkey_b64=buyer_pub)])

    # the paid release surfaces `wrapped` in the KeyReleased event
    client = _FakeClient(released=[_Ev(_CH, _BUYER, wrapped)])
    released_key = acquire_released_key(client, _CH, _BUYER)         # B2
    assert reconstruct_paid_content(released_key, buyer_priv, content) == plaintext   # B1


# ── sp1457: the on-chain KeyReleased scan must stay under the RPC eth_getLogs range cap ──
# A one-shot get_key_released_events(0, latest) HARD-FAILS on Base mainnet, whose RPC rejects any
# eth_getLogs wider than 10k blocks (error -32614 "eth_getLogs is limited to a 10,000 range"). The
# acquire path must walk the range in windows; these tests use a stub that enforces the same cap.

_RPC_RANGE_CAP = 10_000


class _RangeCappedClient:
    """A KeyDistribution client stub that rejects an eth_getLogs query wider than the RPC cap
    (mimicking Base mainnet's -32614), and range-filters returned events by block — so a scan
    only succeeds if it chunks under the cap AND queries the window holding the event."""

    def __init__(self, *, released_at=None, emit_on_release=None, emit_block=None, latest=30_000_000):
        # released_at: list of (block_number, _Ev); emit_on_release/_block: event to add on release()
        self._events = list(released_at or [])
        self._emit = emit_on_release
        self._emit_block = emit_block
        self._latest = int(latest)
        self.release_calls = []
        self.window_queries = []

    def latest_block(self):
        return self._latest

    def get_key_released_events(self, from_block, to_block, *, argument_filters=None):
        span = int(to_block) - int(from_block) + 1
        if span > _RPC_RANGE_CAP:
            raise ValueError(
                f"eth_getLogs is limited to a {_RPC_RANGE_CAP} range (-32614): got {span} blocks "
                f"[{from_block}, {to_block}]")
        self.window_queries.append((int(from_block), int(to_block)))
        return [ev for (blk, ev) in self._events if from_block <= blk <= to_block]

    def release(self, content_hash, recipient):
        self.release_calls.append((content_hash, recipient))
        if self._emit is not None:
            self._events.append((self._emit_block, self._emit))
        return ("0xtx", None)


def test_range_capped_stub_rejects_oversized_query_sanity():
    # Discriminating: prove the stub actually enforces the cap, so the tests below are meaningful.
    client = _RangeCappedClient()
    with pytest.raises(ValueError, match="10000 range"):
        client.get_key_released_events(0, 30_000_000)


def test_recent_release_found_under_range_cap():
    # Event near the chain tip; a one-shot [0, latest] scan would exceed the cap and raise.
    client = _RangeCappedClient(released_at=[(29_999_000, _Ev(_CH, _BUYER, b"wrapped-recent"))])
    assert acquire_released_key(client, _CH, _BUYER) == b"wrapped-recent"
    assert client.release_calls == []                                  # idempotent — already released
    assert all(hi - lo + 1 <= _RPC_RANGE_CAP for (lo, hi) in client.window_queries)


def test_release_then_read_stays_under_range_cap():
    # Not-yet-released → trigger release() → the re-read must ALSO chunk (line-132 scan).
    client = _RangeCappedClient(emit_on_release=_Ev(_CH, _BUYER, b"wrapped-fresh"),
                                emit_block=29_999_500)
    assert acquire_released_key(client, _CH, _BUYER) == b"wrapped-fresh"
    assert client.release_calls == [(_CH, _BUYER)]
    assert all(hi - lo + 1 <= _RPC_RANGE_CAP for (lo, hi) in client.window_queries)


def test_older_release_found_several_chunks_back():
    # Event ~20k blocks below the tip → the newest-first walk must continue past the first window.
    client = _RangeCappedClient(released_at=[(29_980_000, _Ev(_CH, _BUYER, b"wrapped-older"))])
    assert acquire_released_key(client, _CH, _BUYER) == b"wrapped-older"
    assert len(client.window_queries) >= 2                             # took more than one window
    assert all(hi - lo + 1 <= _RPC_RANGE_CAP for (lo, hi) in client.window_queries)


def test_chunked_scan_still_ignores_other_recipient():
    # A different recipient's release in the same window must not be mistaken for ours.
    client = _RangeCappedClient(
        released_at=[(29_999_000, _Ev(_CH, "0x" + "99" * 20, b"not-mine"))],
        emit_on_release=_Ev(_CH, _BUYER, b"mine"), emit_block=29_999_900)
    assert acquire_released_key(client, _CH, _BUYER) == b"mine"
    assert client.release_calls == [(_CH, _BUYER)]                     # had to trigger for us


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
