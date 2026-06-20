"""EIP-4361 Sign-In With Ethereum backend verifier for PRSM wallet onboarding.

Per docs/2026-04-22-phase4-wallet-sdk-design-plan.md §6 Task 1.

The `siwe` PyPI package handles the EIP-4361 parsing + signature recovery +
domain/timing checks. This module wraps it with PRSM-specific invariants:

  - expected-chain-id enforcement (the message-level chain_id must match the
    chain we issued a nonce for),
  - single-use nonce storage (replay protection),
  - normalised PRSM exception types so callers can render friendly errors
    without branching on the upstream exception taxonomy.
"""

from __future__ import annotations

import secrets
import sqlite3
import string
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol, Union

from siwe import (
    DomainMismatch,
    ExpiredMessage,
    InvalidSignature,
    MalformedSession,
    NotYetValidMessage,
    SiweMessage,
    VerificationError,
)


__all__ = [
    "InMemoryNonceStore",
    "NonceStore",
    "SiweChainIdError",
    "SiweDomainError",
    "SiweError",
    "SiweExpiredError",
    "SiweMalformedError",
    "SiweNonceError",
    "SiweNotYetValidError",
    "SiweSignatureError",
    "VerifiedSiwe",
    "verify",
]


class SiweError(Exception):
    """Base class for SIWE verification failures."""


class SiweMalformedError(SiweError):
    """Raw message did not parse as an EIP-4361 SIWE message."""


class SiweSignatureError(SiweError):
    """Signature did not recover to the address stated in the message."""


class SiweDomainError(SiweError):
    """Message's `domain` field did not match the server's expected domain."""


class SiweChainIdError(SiweError):
    """Message's `chain_id` did not match the server's expected chain."""


class SiweExpiredError(SiweError):
    """`now` is after the message's `expiration_time`."""


class SiweNotYetValidError(SiweError):
    """`now` is before the message's `not_before`."""


class SiweNonceError(SiweError):
    """Nonce was not registered by the server, or has already been consumed."""


@dataclass(frozen=True)
class VerifiedSiwe:
    """Result of a successful SIWE verification.

    `address` is EIP-55 checksummed — safe to use as a primary key for
    wallet-to-identity bindings.
    """

    address: str
    domain: str
    chain_id: int
    nonce: str
    issued_at: str
    statement: Optional[str]


class NonceStore(Protocol):
    """Single-use nonce store. Production deployments should use Redis or
    an equivalent TTL-capable store; see Phase 4 plan §8.4."""

    def issue(self, ttl_seconds: int = 300) -> str: ...
    def consume(self, nonce: str) -> bool: ...


_NONCE_ALPHABET = string.ascii_letters + string.digits


def _random_nonce(length: int = 17) -> str:
    # EIP-4361 requires alphanumeric, at least 8 chars. 17 → ~101 bits.
    return "".join(secrets.choice(_NONCE_ALPHABET) for _ in range(length))


@dataclass
class InMemoryNonceStore:
    """Process-local nonce store. Suitable for tests and single-process
    deployments; use a shared backend (Redis) for any multi-worker rollout."""

    _now: Callable[[], float] = field(default_factory=lambda: time.time)
    _nonces: Dict[str, float] = field(default_factory=dict)

    def issue(self, ttl_seconds: int = 300) -> str:
        self._purge_expired()
        nonce = _random_nonce()
        # In the extremely unlikely event of collision, regenerate.
        while nonce in self._nonces:
            nonce = _random_nonce()
        self._nonces[nonce] = self._now() + ttl_seconds
        return nonce

    def consume(self, nonce: str) -> bool:
        self._purge_expired()
        if nonce in self._nonces:
            del self._nonces[nonce]
            return True
        return False

    def _purge_expired(self) -> None:
        now = self._now()
        expired = [n for n, exp in self._nonces.items() if exp <= now]
        for n in expired:
            del self._nonces[n]


_NONCE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS siwe_nonces (
    nonce      TEXT PRIMARY KEY,
    expires_at REAL NOT NULL
);
"""


class SqliteNonceStore:
    """Sprint 1197 — SQLite-backed single-use SIWE nonce store.

    The default ``InMemoryNonceStore`` keeps nonces in a per-process dict, so a
    multi-worker deployment (the realistic production posture: uvicorn/gunicorn
    with N workers) issues a nonce on worker A that worker B can't see → SIWE
    login fails intermittently with ``siwe_nonce_invalid_or_consumed``. A shared
    SQLite file makes ``issue``/``consume`` correct across all workers on one host
    WITHOUT a Redis dependency.

    Single-use is atomic across workers: ``consume`` is a single
    ``DELETE ... WHERE nonce=? AND expires_at>now`` — SQLite serializes writes, so
    of two concurrent consumes of the same nonce exactly one sees ``rowcount==1``.
    A short ``busy_timeout`` rides out cross-worker write contention. The Protocol
    is identical to InMemoryNonceStore so it's a drop-in.
    """

    def __init__(
        self,
        db_path: Union[Path, str],
        *,
        _now: Optional[Callable[[], float]] = None,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._path = Path(db_path)
        self._now = _now or time.time
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_NONCE_SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=self._busy_timeout_ms / 1000.0)
        conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        return conn

    def issue(self, ttl_seconds: int = 300) -> str:
        now = self._now()
        with self._conn() as conn:
            conn.execute("DELETE FROM siwe_nonces WHERE expires_at <= ?", (now,))
            expires_at = now + ttl_seconds
            for _ in range(8):  # regenerate on the astronomically-unlikely collision
                nonce = _random_nonce()
                try:
                    conn.execute(
                        "INSERT INTO siwe_nonces (nonce, expires_at) VALUES (?, ?)",
                        (nonce, expires_at),
                    )
                    conn.commit()
                    return nonce
                except sqlite3.IntegrityError:
                    continue
            raise RuntimeError("could not allocate a unique SIWE nonce")

    def consume(self, nonce: str) -> bool:
        now = self._now()
        with self._conn() as conn:
            conn.execute("DELETE FROM siwe_nonces WHERE expires_at <= ?", (now,))
            cur = conn.execute(
                "DELETE FROM siwe_nonces WHERE nonce = ? AND expires_at > ?",
                (nonce, now),
            )
            conn.commit()
            return cur.rowcount == 1


def verify(
    raw_message: str,
    signature: str,
    *,
    expected_domain: str,
    expected_chain_id: int,
    nonce_store: NonceStore,
    now: Optional[datetime] = None,
) -> VerifiedSiwe:
    """Verify an EIP-4361 message + signature.

    Order of checks matters: we validate every invariant BEFORE consuming the
    nonce, so a failed request (wrong domain, bad signature, etc.) does not
    invalidate a still-fresh nonce — the client can re-submit against the
    same nonce after fixing the failure.

    The final `nonce_store.consume` call is the commit point; after it
    returns True, the same (message, signature) pair cannot succeed again.
    """

    try:
        msg = SiweMessage.from_message(raw_message)
    except MalformedSession as exc:
        raise SiweMalformedError(str(exc)) from exc
    except Exception as exc:  # pydantic ValidationError, grammar errors, etc.
        raise SiweMalformedError(f"parse failed: {exc}") from exc

    if msg.chain_id != expected_chain_id:
        raise SiweChainIdError(
            f"chain_id mismatch: message={msg.chain_id}, expected={expected_chain_id}"
        )

    try:
        msg.verify(signature, domain=expected_domain, timestamp=now)
    except DomainMismatch as exc:
        raise SiweDomainError(str(exc)) from exc
    except ExpiredMessage as exc:
        raise SiweExpiredError(str(exc)) from exc
    except NotYetValidMessage as exc:
        raise SiweNotYetValidError(str(exc)) from exc
    except InvalidSignature as exc:
        raise SiweSignatureError(str(exc)) from exc
    except VerificationError as exc:
        # Any other siwe verification failure — surface as signature error.
        raise SiweSignatureError(str(exc)) from exc

    if not nonce_store.consume(msg.nonce):
        raise SiweNonceError(
            f"nonce not registered or already consumed: {msg.nonce}"
        )

    return VerifiedSiwe(
        address=msg.address,
        domain=msg.domain,
        chain_id=msg.chain_id,
        nonce=msg.nonce,
        issued_at=msg.issued_at,
        statement=msg.statement,
    )
