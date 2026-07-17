"""Sprint 556 — enforce user-sig on /wallet/withdraw when flag is on.

Third sprint in the user-sig arc (554/555/556/557). Wires sprint
555's EIP-712 primitive into POST /wallet/withdraw. When a wallet
has ``requires_user_signature=True`` (sprint 554 flag), the
handler enforces:

  1. Signature + nonce + expiry_unix must be present in the body.
  2. Payload must not be expired (`expiry_unix >= now`).
  3. Signature must recover to the wallet's linked eth_address
     (sprint 540).
  4. Nonce must match ``ledger.get_next_withdraw_nonce(wallet_id)``.
  5. Nonce is consumed BEFORE broadcast, even if broadcast fails
     (replay safety — a captured signature can't be reused after
     a refund).

Each failure mode maps to a specific status code so callers can
self-diagnose. When the flag is OFF, the legacy daemon-mediated
flow is preserved byte-for-byte.
"""
from __future__ import annotations

import time

from unittest.mock import MagicMock

import pytest
from eth_account import Account
from fastapi.testclient import TestClient


# ── shared test fixtures ──────────────────────────────────


_USER_PK = "0x" + "ab" * 32  # deterministic test signer


def _user_acct():
    return Account.from_key(_USER_PK)


class _StubLedger:
    """In-memory stand-in for both ftns_ledger + local_ledger.
    Implements only the surface the withdraw handler uses."""

    class _DebitTx:
        def __init__(self, tx_id):
            self.tx_id = tx_id

    class _BroadcastTx:
        def __init__(self, status="confirmed", tx_hash="0xfeed",
                     block_number=42):
            self.status = status
            self.tx_hash = tx_hash
            self.block_number = block_number

    def __init__(
        self,
        wallet_id: str,
        balance: float = 100.0,
        linked_eth: str = None,
        requires_sig: bool = False,
        broadcast: "_StubLedger._BroadcastTx" = None,
    ):
        self.wallet_id = wallet_id
        self._balance = balance
        self._linked = linked_eth
        self._requires = requires_sig
        self._next_nonce = 0
        self._broadcast = broadcast or _StubLedger._BroadcastTx()
        self.debits: list = []
        self.credits: list = []

    async def eth_address_for_wallet(self, wallet_id):
        return self._linked if wallet_id == self.wallet_id else None

    async def get_requires_user_signature(self, wallet_id):
        return self._requires if wallet_id == self.wallet_id else False

    async def get_next_withdraw_nonce(self, wallet_id):
        return self._next_nonce

    async def bump_withdraw_nonce(self, wallet_id):
        old = self._next_nonce
        self._next_nonce += 1
        return old

    async def compare_and_bump_withdraw_nonce(self, wallet_id, expected):
        # sp1474 — atomic check-and-consume mirror of the real ledgers.
        if self._next_nonce != int(expected):
            return None
        old = self._next_nonce
        self._next_nonce += 1
        return old

    async def get_balance(self, wallet_id):
        return self._balance if wallet_id == self.wallet_id else 0.0

    async def debit(self, *, wallet_id, amount, tx_type, description):
        self._balance -= amount
        rec = _StubLedger._DebitTx(
            tx_id=f"debit-{len(self.debits)}",
        )
        self.debits.append({
            "wallet_id": wallet_id, "amount": amount,
            "description": description,
        })
        return rec

    async def credit(self, *, wallet_id, amount, tx_type, description,
                     idempotency_key=None):
        self._balance += amount
        self.credits.append({
            "wallet_id": wallet_id, "amount": amount,
            "description": description, "idempotency_key": idempotency_key,
        })

    async def transfer(self, *, job_id, to_address, amount_ftns):
        return self._broadcast


def _stub_node(ledger):
    n = MagicMock()
    n.identity = MagicMock(node_id=ledger.wallet_id)
    n.ledger = ledger
    n.ftns_ledger = ledger
    return n


def _make_app(ledger):
    from prsm.node.api import create_api_app
    return create_api_app(_stub_node(ledger), enable_security=False)


def _signed_body(
    *,
    amount_ftns: float,
    to_eth_address: str,
    nonce: int = 0,
    expiry_unix: int = None,
    wallet_id: str = "w1",
    private_key: str = _USER_PK,
    tamper_amount: bool = False,
):
    """Build a request body with a valid EIP-712 signature."""
    from prsm.economy.withdraw_signature import (
        sign_withdraw_payload,
    )
    if expiry_unix is None:
        expiry_unix = int(time.time()) + 300
    amount_wei = int(amount_ftns * 1e18)
    payload = {
        "wallet_id": wallet_id,
        "amount_ftns_wei": amount_wei,
        "to_eth_address": to_eth_address,
        "nonce": nonce,
        "expiry_unix": expiry_unix,
    }
    sig = sign_withdraw_payload(payload, private_key)
    body = {
        "amount_ftns": amount_ftns * 2 if tamper_amount else amount_ftns,
        "wallet_id": wallet_id,
        "to_eth_address": to_eth_address,
        "signature": "0x" + sig.hex(),
        "nonce": nonce,
        "expiry_unix": expiry_unix,
    }
    return body


# ── flag-off path: legacy unchanged ───────────────────────


def test_flag_off_legacy_path_unchanged():
    """``requires_user_signature=False`` (sprint 541 default): the
    daemon-mediated flow works exactly as before. No signature on
    the request body is fine."""
    acct = _user_acct()
    ledger = _StubLedger(
        wallet_id="w1", balance=10.0,
        linked_eth=acct.address.lower(),
        requires_sig=False,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/withdraw",
        json={
            "amount_ftns": 1.0,
            "wallet_id": "w1",
            "to_eth_address": acct.address,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "confirmed"
    # Sprint 541 invariant preserved: legacy flow does NOT bump
    # the nonce.
    assert ledger._next_nonce == 0


# ── flag-on path: enforcement ─────────────────────────────


def test_flag_on_missing_signature_returns_401():
    """Flag on + no signature in body → 401 with actionable detail."""
    acct = _user_acct()
    ledger = _StubLedger(
        wallet_id="w1", balance=10.0,
        linked_eth=acct.address.lower(),
        requires_sig=True,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/withdraw",
        json={
            "amount_ftns": 1.0,
            "wallet_id": "w1",
            "to_eth_address": acct.address,
        },
    )
    assert response.status_code == 401
    assert "signature" in response.json()["detail"].lower()


def test_flag_on_valid_signature_succeeds():
    """Flag on + valid sig + matching nonce + fresh expiry → 200,
    nonce bumped to 1."""
    acct = _user_acct()
    ledger = _StubLedger(
        wallet_id="w1", balance=10.0,
        linked_eth=acct.address.lower(),
        requires_sig=True,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/withdraw",
        json=_signed_body(
            amount_ftns=1.0,
            to_eth_address=acct.address,
        ),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "confirmed"
    assert body.get("nonce_consumed") == 0
    # Nonce was advanced.
    assert ledger._next_nonce == 1


def test_flag_on_wrong_signer_returns_401():
    """Sig recovers to a DIFFERENT address than the linked one →
    401 (signer mismatch)."""
    # User has one key; an attacker has a different key.
    real_acct = _user_acct()
    attacker_pk = "0x" + "cd" * 32
    ledger = _StubLedger(
        wallet_id="w1", balance=10.0,
        linked_eth=real_acct.address.lower(),
        requires_sig=True,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/withdraw",
        json=_signed_body(
            amount_ftns=1.0,
            to_eth_address=real_acct.address,
            private_key=attacker_pk,
        ),
    )
    assert response.status_code == 401
    detail = response.json()["detail"].lower()
    assert "signer" in detail or "address" in detail


def test_flag_on_tampered_amount_returns_401():
    """Signed payload claims 1 FTNS but the request body says 2 FTNS
    → signature recovers to a different address → 401."""
    acct = _user_acct()
    ledger = _StubLedger(
        wallet_id="w1", balance=10.0,
        linked_eth=acct.address.lower(),
        requires_sig=True,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/withdraw",
        json=_signed_body(
            amount_ftns=1.0,
            to_eth_address=acct.address,
            tamper_amount=True,
        ),
    )
    assert response.status_code == 401


def test_flag_on_wrong_nonce_returns_401():
    """Ledger expects nonce=0 but request sends nonce=5 → 401."""
    acct = _user_acct()
    ledger = _StubLedger(
        wallet_id="w1", balance=10.0,
        linked_eth=acct.address.lower(),
        requires_sig=True,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/withdraw",
        json=_signed_body(
            amount_ftns=1.0,
            to_eth_address=acct.address,
            nonce=5,
        ),
    )
    assert response.status_code == 401
    assert "nonce" in response.json()["detail"].lower()


def test_flag_on_expired_payload_returns_401():
    """Expiry in the past → 401 expired."""
    acct = _user_acct()
    ledger = _StubLedger(
        wallet_id="w1", balance=10.0,
        linked_eth=acct.address.lower(),
        requires_sig=True,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/withdraw",
        json=_signed_body(
            amount_ftns=1.0,
            to_eth_address=acct.address,
            expiry_unix=int(time.time()) - 1,
        ),
    )
    assert response.status_code == 401
    assert "expir" in response.json()["detail"].lower()


def test_flag_on_unlinked_wallet_returns_401():
    """Flag is on but wallet has no linked eth_address — there's
    nothing to compare the recovered signer against → 401 with
    actionable hint pointing at /wallet/deposit/link."""
    acct = _user_acct()
    ledger = _StubLedger(
        wallet_id="w1", balance=10.0,
        linked_eth=None,  # no linkage
        requires_sig=True,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/withdraw",
        json=_signed_body(
            amount_ftns=1.0,
            to_eth_address=acct.address,
        ),
    )
    assert response.status_code == 401
    assert "/wallet/deposit/link" in response.json()["detail"]


# ── replay safety ─────────────────────────────────────────


def test_flag_on_nonce_consumed_on_broadcast_failure():
    """Replay safety: even if the on-chain broadcast FAILS and the
    debit is refunded, the nonce MUST stay bumped — otherwise a
    captured signature could be replayed for free after a refund.

    Failure mode: broadcast returns status='rejected' → debit is
    refunded (sprint 541) AND 502 raised, BUT the nonce stays at 1.
    A subsequent signed request with nonce=0 would now fail.
    """
    acct = _user_acct()
    ledger = _StubLedger(
        wallet_id="w1", balance=10.0,
        linked_eth=acct.address.lower(),
        requires_sig=True,
        broadcast=_StubLedger._BroadcastTx(status="rejected"),
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/withdraw",
        json=_signed_body(
            amount_ftns=1.0,
            to_eth_address=acct.address,
        ),
    )
    assert response.status_code == 502
    # Nonce consumed despite broadcast failure.
    assert ledger._next_nonce == 1
    # Debit was refunded — net-zero balance.
    assert abs(ledger._balance - 10.0) < 1e-9
