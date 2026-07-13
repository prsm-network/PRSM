"""Sprint 1092 — relayer delegation budget store (requester-payment relayer, brick 2).

A funder's ``PaymentDelegation`` (sp1091) caps the relayer's CUMULATIVE spend
(``max_total_spend_wei``) across many per-request authorizations. This store tracks the
consumed budget per ``delegation_nonce`` and reserves conservatively — the per-request
ceiling (``max_spend_wei``) — at admission, so the relayer can never authorize more than
the funder delegated, even across many requests and across restarts.

Mirrors the sp1055 nonce store: ``InMemoryDelegationBudgetStore`` for tests / a single
process, ``DurableDelegationBudgetStore`` (atomic JSON via the sp1039 state store) so a
restart cannot reopen budget the funder has already spent. Reserve is conservative
(reserve the authorized ceiling); releasing the unused remainder after the actual settle
is a documented additive refinement, not required for the no-overspend guarantee.

Concurrency: like the nonce store, these are synchronous and expected to be called under
the relayer verifier's lock (sp1053-style), so the check-then-add is atomic per process.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from prsm.settlement.state_store import SettlementStateStore

logger = logging.getLogger(__name__)


def _key(nonce: str) -> str:
    """sp1452 (money-audit wuh7kwl7i) — canonicalize the delegation_nonce to its bytes32 form so
    signature-equivalent spellings collapse to ONE budget bucket.

    The EIP-712 delegation signature binds the nonce via ``_to_bytes32`` (strips a leading '0x',
    hex-case-insensitive), so '0x<hex>', bare '<hex>' and case variants share ONE digest — i.e. ONE
    funder signature verifies all of them. Keying the budget on the raw ``.lower()``'d string let a
    malicious relayer (no funder key) alias one funder-signed delegation into N distinct cap buckets
    and drain N×C past the signed cumulative cap. This is the SINGLE choke point for
    consumed/reserve/release on BOTH stores (and the durable on-load re-key), so canonicalizing here
    makes every signature-equivalent alias map to the same bucket. A malformed nonce — which can never
    pass the signature gate anyway — falls back to case-folding so it still can't alias on case alone.
    """
    s = str(nonce).strip()
    try:
        from prsm.settlement.payment_authorization import _to_bytes32
        return "0x" + _to_bytes32("delegation_nonce", s).hex()
    except (ValueError, TypeError):
        return s.lower()


class InMemoryDelegationBudgetStore:
    """Process-local consumed-budget map keyed by delegation_nonce. Fine for tests / a
    single process that never restarts; use ``DurableDelegationBudgetStore`` on a live
    node so already-spent budget survives a restart."""

    def __init__(self) -> None:
        self._consumed: Dict[str, int] = {}

    def consumed(self, nonce: str) -> int:
        return self._consumed.get(_key(nonce), 0)

    def reserve(self, nonce: str, amount: int, cap: int) -> bool:
        """Atomically reserve ``amount`` against the delegation's cumulative budget.
        Returns True (and records the reservation) iff consumed + amount <= cap; False
        otherwise (a rejected reserve records NOTHING). ``amount`` must be >= 0."""
        amount = int(amount)
        cap = int(cap)
        if amount < 0:
            return False
        k = _key(nonce)
        new_total = self._consumed.get(k, 0) + amount
        if new_total > cap:
            return False
        self._consumed[k] = new_total
        return True

    def release(self, nonce: str, amount: int) -> None:
        """sp1095 — return ``amount`` of previously-reserved budget (the unused remainder
        after a cheaper-than-ceiling settle, or a full hold whose job never settled).
        Floors at 0 so an over-release can never make budget go negative."""
        amount = int(amount)
        if amount <= 0:
            return
        k = _key(nonce)
        self._consumed[k] = max(0, self._consumed.get(k, 0) - amount)


class DurableDelegationBudgetStore:
    """Atomic-JSON-backed consumed-budget map so a restart can't reopen budget the funder
    already spent (the relayer parallel to the sp1055 DurableNonceStore). A corrupt file
    raises on load — LOUD-but-safe: refusing to start with a forgotten budget is correct
    (forgetting it would let the relayer re-spend up to the full cap again)."""

    def __init__(self, path: Any):
        self._store = SettlementStateStore(path)
        # A corrupt file raises SettlementStateCorruptError here (must NOT silently
        # forget consumed budget → that would reopen the funder's spent cap).
        loaded = self._store.load()
        self._consumed: Dict[str, int] = {
            _key(k): int(v) for k, v in (loaded or {}).get("consumed", {}).items()
        }

    def consumed(self, nonce: str) -> int:
        return self._consumed.get(_key(nonce), 0)

    def reserve(self, nonce: str, amount: int, cap: int) -> bool:
        amount = int(amount)
        cap = int(cap)
        if amount < 0:
            return False
        k = _key(nonce)
        new_total = self._consumed.get(k, 0) + amount
        if new_total > cap:
            return False
        self._consumed[k] = new_total
        # Persist BEFORE returning True so a crash can't lose a reservation we then act
        # on (the no-overspend invariant must survive a restart).
        self._persist()
        return True

    def release(self, nonce: str, amount: int) -> None:
        """sp1095 — durably return ``amount`` of reserved budget (unused remainder or a
        non-settled hold). Floors at 0. Persists so the reclaim survives a restart."""
        amount = int(amount)
        if amount <= 0:
            return
        k = _key(nonce)
        self._consumed[k] = max(0, self._consumed.get(k, 0) - amount)
        self._persist()

    def _persist(self) -> None:
        self._store.save({
            "version": 1,
            "consumed": {kk: str(vv) for kk, vv in sorted(self._consumed.items())},
        })
