"""Sprint 1055 — PaymentAuthorizationVerifier (requester-payment, brick 2).

Fail-closed gate a provider runs before serving + settling a PAID inference. Given
a signed ``PaymentAuthorization`` (sprint 1054), it verifies — in cheap-to-costly
order — that the auth is genuine, pins this provider + this request, is unexpired,
within the authorized ceiling, not replayed, and (when a balance reader is wired)
backed by enough on-chain EscrowPool balance. Any failure raises
``AuthorizationRejected(reason)``; only on full success is the nonce remembered and
the authenticated requester address returned for the settle path (brick 3).

The nonce store is pluggable: ``InMemoryNonceStore`` for tests / single-process and
``DurableNonceStore`` (atomic JSON via the sprint-1039 state store) so a restart
can't reopen a replay window. The balance reader is an injected async callable
(``EscrowPoolClient.balance_of``); when absent the pre-flight is skipped (the real
enforcement is ``finalizeBatch`` reverting on insufficient escrow — the pre-flight
is a courtesy that fails fast before doing the work).

See docs/2026-06-10-requester-payment-model-proposal.md.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Optional

from prsm.settlement.payment_authorization import (
    InvalidSignatureFormat,
    _to_bytes32,
    is_expired,
    verify_payment_authorization_signature,
)
from prsm.settlement.state_store import (
    SettlementStateStore,
    SettlementStateCorruptError,
)

logger = logging.getLogger(__name__)

BalanceReader = Callable[[str], Awaitable[int]]


class AuthorizationRejected(Exception):
    """A PaymentAuthorization failed verification. ``reason`` is a stable
    machine-readable string the caller can log / return."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}{': ' + detail if detail else ''}")


class InMemoryNonceStore:
    """Process-local seen-nonce set. Fine for tests / a single process that never
    restarts; use ``DurableNonceStore`` on a live node."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    @staticmethod
    def _key(nonce: str) -> str:
        return nonce.lower()

    def seen(self, nonce: str) -> bool:
        return self._key(nonce) in self._seen

    def remember(self, nonce: str) -> None:
        self._seen.add(self._key(nonce))


class DurableNonceStore:
    """Atomic-JSON-backed seen-nonce set so a restart can't reopen a replay
    window. Reuses the sprint-1039 atomic write (tmp + fsync + rename + dir
    fsync). Small by design — nonces are 32-byte hex; a busy node can prune by
    rotating the file out of band if it ever grows large (a paid-job nonce is only
    replay-relevant until its auth expires)."""

    def __init__(self, path: Any):
        self._store = SettlementStateStore(path)
        try:
            loaded = self._store.load()
        except SettlementStateCorruptError:
            # LOUD-but-safe: a corrupt nonce file must NOT silently forget seen
            # nonces (that would reopen replays). Refuse to start empty.
            logger.error(
                "payment nonce store at %s is corrupt — refusing to start with "
                "an empty seen-set (would reopen replay window)", path,
            )
            raise
        self._seen: set[str] = set((loaded or {}).get("seen_nonces", []))

    @staticmethod
    def _key(nonce: str) -> str:
        return nonce.lower()

    def seen(self, nonce: str) -> bool:
        return self._key(nonce) in self._seen

    def remember(self, nonce: str) -> None:
        self._seen.add(self._key(nonce))
        self._store.save({"version": 1, "seen_nonces": sorted(self._seen)})


class PaymentAuthorizationVerifier:
    """Fail-closed verifier for signed PaymentAuthorizations."""

    def __init__(
        self,
        *,
        provider_address: str,
        nonce_store: Any = None,
        balance_reader: Optional[BalanceReader] = None,
        chain_id: Optional[int] = None,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        if not provider_address:
            raise ValueError("provider_address is required")
        self._provider = provider_address
        self._nonce_store = nonce_store if nonce_store is not None else InMemoryNonceStore()
        self._balance_reader = balance_reader
        self._chain_id = chain_id
        self._now = now or time.time

    async def verify(
        self,
        payload: dict,
        signature: Any,
        *,
        request_hash: Any,
        quoted_price_wei: int,
    ) -> str:
        """Verify ``payload``/``signature`` against ``request_hash`` and the
        provider's quote. Returns the authenticated requester checksum address on
        success (and remembers the nonce); raises ``AuthorizationRejected``
        otherwise. Checks run cheap-to-costly so a forged/expired/replayed auth is
        rejected before the network balance read.
        """
        chain_kw = {} if self._chain_id is None else {"chain_id": self._chain_id}

        # 1. Signature recovers to a real address (cheapest hard failure).
        try:
            recovered = verify_payment_authorization_signature(
                payload, signature, **chain_kw
            )
        except InvalidSignatureFormat as exc:
            raise AuthorizationRejected("bad-signature", str(exc))

        # 2. Recovered signer must be the named requester (no provider-forged auth).
        requester = str(payload.get("requester", ""))
        if recovered.lower() != requester.lower():
            raise AuthorizationRejected(
                "signer-mismatch",
                f"recovered {recovered} != requester {requester}",
            )

        # 3. Provider pin — auth can't be replayed against a different node.
        if str(payload.get("provider", "")).lower() != self._provider.lower():
            raise AuthorizationRejected(
                "provider-mismatch",
                f"auth provider {payload.get('provider')} != this {self._provider}",
            )

        # 4. Request-hash binding — auth covers THIS work, not arbitrary work.
        try:
            auth_rh = _to_bytes32("request_hash", payload["request_hash"])
            want_rh = _to_bytes32("request_hash", request_hash)
        except (KeyError, ValueError) as exc:
            raise AuthorizationRejected("request-hash-mismatch", str(exc))
        if auth_rh != want_rh:
            raise AuthorizationRejected("request-hash-mismatch")

        # 5. Expiry / staleness.
        if is_expired(payload, now=self._now):
            raise AuthorizationRejected(
                "expired", f"expiry_unix={payload.get('expiry_unix')}"
            )

        # 6. Price must be within the signed ceiling.
        if int(quoted_price_wei) > int(payload["max_spend_wei"]):
            raise AuthorizationRejected(
                "exceeds-max-spend",
                f"price {quoted_price_wei} > max_spend {payload['max_spend_wei']}",
            )

        # 7. Nonce must be unseen (anti-replay). Checked before the balance read so
        # a replayed auth never triggers an RPC call; remembered only after the
        # balance check passes so a transient underfund stays retryable.
        nonce = str(payload["job_nonce"])
        if self._nonce_store.seen(nonce):
            raise AuthorizationRejected("nonce-replayed", nonce)

        # 8. On-chain escrow pre-flight (when a reader is wired). The authoritative
        # check is finalizeBatch reverting; this fails fast + avoids serving work
        # the requester demonstrably can't pay for.
        if self._balance_reader is not None:
            balance = int(await self._balance_reader(requester))
            if balance < int(quoted_price_wei):
                raise AuthorizationRejected(
                    "insufficient-escrow-balance",
                    f"balance {balance} < price {quoted_price_wei}",
                )
        else:
            logger.debug(
                "payment auth verified without escrow pre-flight (no balance "
                "reader wired); relying on finalizeBatch to enforce funding"
            )

        # All checks passed — consume the nonce + return the authenticated address.
        self._nonce_store.remember(nonce)
        return recovered
