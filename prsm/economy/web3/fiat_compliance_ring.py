"""Sprint 282 — Fiat compliance audit ring.

Records every fiat-flow event (onramp/offramp/gasless quotes
+ executes + KYC lifecycle events) for AUSTRAC / FinCEN / IRS
reporting once Phase 5 ramps go live. Single queryable log
across all fiat surfaces so operators have one source of
truth for regulatory inquiries.

Persistence is required, not opt-in — regulators expect 5-7
year retention. When PRSM_FIAT_COMPLIANCE_LOG_DIR is set the
ring writes a JSON file per entry. Without the env var, the
ring still operates (bounded in-memory ≥ 100K entries) so
audit data isn't silently dropped on a misconfigured node,
but operators running fiat flows in production MUST set the
env var (operationally enforced by `prsm node start`
health-check in a follow-on sprint).

This sprint records events from the existing quote endpoints
(every artifact return). Execute paths (gated on CDP
commission) auto-record once they ship — same hook point.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


_VALID_KINDS = {
    "onramp_quote", "onramp_execute",
    "offramp_quote", "offramp_execute",
    "gasless_transfer_quote", "gasless_transfer_execute",
    "kyc_initiate", "kyc_status_change",
}

_DEFAULT_MAX_ENTRIES = 100_000


@dataclass
class FiatComplianceEntry:
    entry_id: str
    timestamp: float
    kind: str
    user_id: str
    usd_amount: float
    ftns_amount: float
    status: str
    kyc_status: Optional[str] = None
    tx_hash: Optional[str] = None
    vendor_ref: Optional[str] = None
    address: Optional[str] = None
    jurisdiction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "FiatComplianceEntry":
        return cls(
            entry_id=d["entry_id"],
            timestamp=float(d.get("timestamp", 0.0)),
            kind=d["kind"],
            user_id=d.get("user_id", ""),
            usd_amount=float(d.get("usd_amount", 0.0)),
            ftns_amount=float(d.get("ftns_amount", 0.0)),
            status=d.get("status", "UNKNOWN"),
            kyc_status=d.get("kyc_status"),
            tx_hash=d.get("tx_hash"),
            vendor_ref=d.get("vendor_ref"),
            address=d.get("address"),
            jurisdiction=d.get("jurisdiction"),
            metadata=d.get("metadata", {}) or {},
        )


class RedisRollingTotal:
    """Sp1179 — a SHARED (cross-replica) backend for the fiat AML
    rolling total, so a multi-replica deployment ENFORCES the per-user
    tier limit GLOBALLY instead of per-pod (the sp1175 gap — where the
    in-memory ring let a user fanned across N replicas transact N×).
    Opt-in: constructed only when PRSM_FIAT_TIER_LIMIT_MODE=shared_redis
    and a Redis URL is configured.

    Faithfully mirrors FiatComplianceRing.total_usd_for_user's
    semantics on a Redis sorted-set per user: each settled/reserved
    fiat-USD event is a member scored by its event timestamp; the query
    dedups by intent_id (taking the latest-timestamp event per intent —
    the sp968 in-memory dedup) and sums over the rolling window. A
    differential test asserts byte-equal totals vs the in-memory ring
    for identical event sequences (incl. PENDING→CONFIRMED→EXPIRED
    supersession + window expiry).

    Read errors PROPAGATE so the caller (the tier check) fails CLOSED —
    deny rather than under-enforce when the shared store is unreachable.
    Write errors are logged loudly but do NOT raise (record() is
    best-effort; a lost write under-counts, surfaced by the loud log)."""

    _SEP = "\x1f"  # unit separator — won't appear in intent/entry ids.

    def __init__(
        self,
        redis_client: Any,
        *,
        window_sec: int = 86400,
        key_prefix: str = "prsm:fiat:roll:",
    ) -> None:
        self._redis = redis_client
        self._window_sec = int(window_sec)
        self._prefix = key_prefix

    def _key(self, user_id: str) -> str:
        return f"{self._prefix}{user_id}"

    def record(
        self,
        *,
        user_id: str,
        dedup_key: str,
        entry_id: str,
        usd_amount: float,
        timestamp: float,
    ) -> None:
        """Append an event (best-effort; logs but never raises)."""
        if not user_id:
            return
        key = self._key(user_id)
        # member = dedup_key | entry_id | amount. The entry_id keeps
        # distinct events for the SAME intent as distinct members (so
        # the query can dedup by intent taking the latest score); the
        # amount rides in the member so the read needs no second lookup.
        member = (
            f"{dedup_key}{self._SEP}{entry_id}{self._SEP}"
            f"{repr(float(usd_amount))}"
        )
        try:
            self._redis.zadd(key, {member: float(timestamp)})
            # TTL so an inactive user's key self-expires (bounded mem).
            self._redis.expire(key, self._window_sec * 2 + 3600)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sp1179: shared rolling-total WRITE failed for user %s "
                "(the AML rolling total may UNDER-count until the next "
                "successful write — investigate Redis): %s",
                user_id, exc,
            )

    @classmethod
    def _latest_amount_by_dedup(cls, rows: Any) -> Dict[str, float]:
        """From ``(member, score)`` rows → {dedup_key: latest-score amount}. The intent-dedup: for
        each dedup_key keep the amount of its highest-scored (newest) member, so a PENDING→CONFIRMED
        pair for one intent counts once. Malformed members are skipped."""
        latest: Dict[str, tuple] = {}  # dedup_key -> (score, amount)
        for member, score in rows:
            parts = str(member).split(cls._SEP)
            if len(parts) != 3:
                continue
            dk, _entry, amount_str = parts
            try:
                amount = float(amount_str)
            except (ValueError, TypeError):
                continue
            prev = latest.get(dk)
            if prev is None or score >= prev[0]:
                latest[dk] = (score, amount)
        return {dk: amt for dk, (_s, amt) in latest.items()}

    def total_usd_for_user(
        self, user_id: str, *, window_sec: int, now: float,
    ) -> float:
        """Sum the windowed, intent-deduped rolling total. RAISES on a
        Redis error so the tier check fails closed."""
        if not user_id:
            return 0.0
        key = self._key(user_id)
        cutoff = now - window_sec
        # Best-effort prune of events older than the window.
        try:
            self._redis.zremrangebyscore(key, "-inf", f"({cutoff}")
        except Exception:  # noqa: BLE001
            pass  # the windowed read below still excludes stale events
        # ts >= cutoff (inclusive) — matches in-memory `ts < cutoff` skip.
        rows = self._redis.zrangebyscore(
            key, cutoff, "+inf", withscores=True,
        )
        return sum(self._latest_amount_by_dedup(rows).values())

    def try_reserve(
        self,
        *,
        user_id: str,
        dedup_key: str,
        entry_id: str,
        usd_amount: float,
        timestamp: float,
        limit_usd: float,
        window_sec: int,
        now: float,
        max_retries: int = 8,
    ) -> bool:
        """sp1464 — ATOMIC check-and-reserve. Returns True iff the windowed, intent-deduped total for
        ``user_id`` PLUS ``usd_amount`` stays within ``limit_usd`` — and, when True, atomically records
        the reservation so a concurrent reserver (another replica) immediately sees it. Returns False
        WITHOUT recording when it would breach. RAISES on a Redis error (caller must fail closed).

        Closes the check→record race: ``total_usd_for_user`` (read) and ``record`` (zadd) were two
        separate round-trips, so N concurrent onramps across replicas all read the same under-limit
        total and all recorded → structuring past the AML limit. Here the read+compare+zadd run under
        a WATCH on the user's key; any concurrent modification aborts the EXEC → retry with the fresh
        total. Dedup-aware: re-reserving the SAME dedup_key replaces its contribution (idempotent),
        so it can't inflate the total — for a fresh intent_id (the onramp case) that contribution is 0
        so prospective = current + amount."""
        if not user_id:
            return True
        from redis.exceptions import WatchError  # redis is present whenever a shared total exists

        key = self._key(user_id)
        member = (
            f"{dedup_key}{self._SEP}{entry_id}{self._SEP}"
            f"{repr(float(usd_amount))}"
        )
        cutoff = now - window_sec
        for _ in range(max(1, int(max_retries))):
            with self._redis.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    # After watch() the pipe runs commands IMMEDIATELY until multi().
                    pipe.zremrangebyscore(key, "-inf", f"({cutoff}")
                    rows = pipe.zrangebyscore(key, cutoff, "+inf", withscores=True)
                    latest = self._latest_amount_by_dedup(rows)
                    # Replace this dedup_key's current contribution with the new amount (fresh key → 0).
                    prospective = (
                        sum(latest.values())
                        - latest.get(str(dedup_key), 0.0)
                        + float(usd_amount)
                    )
                    if prospective > limit_usd:
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.zadd(key, {member: float(timestamp)})
                    pipe.expire(key, self._window_sec * 2 + 3600)
                    pipe.execute()
                    return True
                except WatchError:
                    continue  # the key changed under us → retry with the fresh total
        # Exhausted retries under sustained contention: fail closed (caller denies) rather than
        # record without a fresh consistency check.
        raise RuntimeError(
            "sp1464: try_reserve exhausted retries under contention for "
            f"user {user_id}"
        )


class FiatComplianceRing:
    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        *,
        persist_dir: Optional[Path] = None,
        default_jurisdiction: Optional[str] = None,
        shared_total: Optional[RedisRollingTotal] = None,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[FiatComplianceEntry] = deque(
            maxlen=max_entries,
        )
        self._by_id: Dict[str, FiatComplianceEntry] = {}
        # Sp896 — the ring is shared between the asyncio event loop
        # (tier check on /wallet/onramp/execute reads it) and the
        # sp894 auto-sweep WORKER THREAD (sp885 records onramp_execute
        # on every CONFIRMED intent). Concurrent deque append + iterate
        # raises "deque mutated during iteration", so all access to
        # _entries / _by_id is serialized through this mutex. Readers
        # snapshot under the lock then iterate the snapshot lock-free
        # to keep the hot-path hold short.
        self._lock = threading.Lock()
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        self._default_jurisdiction = default_jurisdiction
        # Sp1179 — optional SHARED cross-replica rolling-total backend
        # (Redis). When present, total_usd_for_user reads from it (the
        # AML limit is enforced globally) and record() mirrors writes to
        # it. None → process-local in-memory (the default).
        self._shared_total: Optional[RedisRollingTotal] = shared_total
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    @classmethod
    def from_env(cls) -> "FiatComplianceRing":
        persist_raw = os.environ.get(
            "PRSM_FIAT_COMPLIANCE_LOG_DIR",
        )
        persist_dir = Path(persist_raw) if persist_raw else None
        jurisdiction = (
            os.environ.get("PRSM_OPERATOR_JURISDICTION") or None
        )
        # Sp1179 — wire the shared Redis rolling total ONLY when the
        # operator opts into shared_redis mode AND a Redis URL resolves.
        # Construction failure (no URL / redis import / connect) leaves
        # shared_total=None; the tier check then FAILS CLOSED in
        # shared_redis mode rather than silently under-enforcing.
        shared_total = None
        mode = (
            os.environ.get("PRSM_FIAT_TIER_LIMIT_MODE", "process_local")
            .strip()
            .lower()
        )
        if mode == "shared_redis":
            shared_total = cls._build_shared_total_from_env()
        return cls(
            persist_dir=persist_dir,
            default_jurisdiction=jurisdiction,
            shared_total=shared_total,
        )

    @staticmethod
    def _build_shared_total_from_env() -> Optional["RedisRollingTotal"]:
        url = (
            os.environ.get("PRSM_FIAT_COMPLIANCE_REDIS_URL")
            or os.environ.get("PRSM_REDIS_URL")
            or os.environ.get("REDIS_URL")
        )
        if not url:
            logger.warning(
                "sp1179: PRSM_FIAT_TIER_LIMIT_MODE=shared_redis but no "
                "Redis URL (PRSM_FIAT_COMPLIANCE_REDIS_URL / "
                "PRSM_REDIS_URL / REDIS_URL) — the tier check will FAIL "
                "CLOSED (deny fiat) until a shared backend is configured."
            )
            return None
        try:
            import redis as _redis  # sync client; the ring is sync

            client = _redis.Redis.from_url(url, decode_responses=True)
            return RedisRollingTotal(client)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sp1179: shared Redis rolling-total construction failed "
                "(%s) — the tier check will FAIL CLOSED until Redis is "
                "reachable.", exc,
            )
            return None

    def record(
        self,
        *,
        kind: str,
        user_id: str,
        usd_amount: float,
        ftns_amount: float,
        status: str,
        kyc_status: Optional[str] = None,
        tx_hash: Optional[str] = None,
        vendor_ref: Optional[str] = None,
        address: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> FiatComplianceEntry:
        entry = self._build_entry(
            kind=kind, user_id=user_id, usd_amount=usd_amount,
            ftns_amount=ftns_amount, status=status, kyc_status=kyc_status,
            tx_hash=tx_hash, vendor_ref=vendor_ref, address=address,
            jurisdiction=jurisdiction, metadata=metadata, timestamp=timestamp,
        )
        with self._lock:
            self._entries.append(entry)
            self._by_id[entry.entry_id] = entry
        # Disk write outside the lock: it's a per-entry independent
        # file (entry_id.json), so it can't race another writer, and
        # keeping I/O out of the mutex avoids serializing the hot path.
        self._write_to_disk(entry)
        # Sp1179 — mirror fiat-USD events to the SHARED rolling-total
        # backend so a multi-replica deployment enforces the AML limit
        # globally. Only the kinds that burn the limit (executes) +
        # identified users; dedup_key = intent_id (PENDING→CONFIRMED→
        # EXPIRED supersede) else the unique entry_id (offramp/legacy
        # each count individually) — matching total_usd_for_user.
        if (
            self._shared_total is not None
            and kind in self._FIAT_USD_KINDS
            and entry.user_id
        ):
            self._shared_total.record(
                user_id=entry.user_id,
                dedup_key=str(
                    (entry.metadata or {}).get("intent_id")
                    or entry.entry_id
                ),
                entry_id=entry.entry_id,
                usd_amount=entry.usd_amount,
                timestamp=entry.timestamp,
            )
        return entry

    def _build_entry(
        self,
        *,
        kind: str,
        user_id: str,
        usd_amount: float,
        ftns_amount: float,
        status: str,
        kyc_status: Optional[str] = None,
        tx_hash: Optional[str] = None,
        vendor_ref: Optional[str] = None,
        address: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> FiatComplianceEntry:
        """Validate + construct a FiatComplianceEntry (no side effects). Shared by record() and
        try_reserve() so both apply the identical kind/amount guards."""
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_KINDS)}, "
                f"got {kind!r}"
            )
        # Sp889 — reject non-finite amounts FIRST. `NaN < 0` and
        # `inf < 0` are both False, so the negative-guard below
        # would let them through — and a single NaN poisons
        # total_usd_for_user (sum → NaN), making the sp884 tier
        # check `(NaN + requested) > limit` always False and thus
        # permanently disabling tier enforcement for that user.
        if not math.isfinite(usd_amount):
            raise ValueError(
                f"usd_amount must be finite, got {usd_amount}"
            )
        if not math.isfinite(ftns_amount):
            raise ValueError(
                f"ftns_amount must be finite, got {ftns_amount}"
            )
        if usd_amount < 0:
            raise ValueError(
                f"usd_amount must be >= 0, got {usd_amount}"
            )
        if ftns_amount < 0:
            raise ValueError(
                f"ftns_amount must be >= 0, got {ftns_amount}"
            )
        return FiatComplianceEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=(
                timestamp if timestamp is not None else time.time()
            ),
            kind=kind,
            user_id=user_id or "",
            usd_amount=float(usd_amount),
            ftns_amount=float(ftns_amount),
            status=status,
            kyc_status=kyc_status,
            tx_hash=tx_hash,
            vendor_ref=vendor_ref,
            address=address,
            jurisdiction=(
                jurisdiction
                if jurisdiction is not None
                else self._default_jurisdiction
            ),
            metadata=metadata or {},
        )

    def try_reserve(
        self,
        *,
        user_id: str,
        dedup_key: str,
        usd_amount: float,
        limit_usd: float,
        window_sec: int = 86400,
        timestamp: Optional[float] = None,
        address: Optional[str] = None,
        kyc_status: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """sp1464 — atomic check-and-reserve backing the AML tier limit on onramp/execute. Returns
        True (reservation recorded) iff the user's windowed, intent-deduped total + ``usd_amount`` <=
        ``limit_usd``; False (nothing recorded) on breach. RAISES on a shared-Redis error so the
        caller FAILS CLOSED. ``dedup_key`` is the intent_id (dedups this PENDING with the later
        CONFIRMED settle, matching total_usd_for_user).

        shared_redis mode: the check+reserve is ATOMIC on the shared store (WATCH/MULTI), closing the
        cross-replica structuring race that a separate read-then-record left open. process_local mode
        has no cross-replica store, so this preserves the existing behavior (record + True; the API's
        pre-mint tier check gates that mode) rather than changing single-replica semantics."""
        ts = timestamp if timestamp is not None else time.time()
        md = dict(metadata or {})
        md.setdefault("intent_id", str(dedup_key))
        if self._shared_total is None:
            # process_local — unchanged behavior: record the PENDING reservation, report success.
            self.record(
                kind="onramp_execute", user_id=user_id, usd_amount=usd_amount,
                ftns_amount=0.0, status="PENDING", address=address,
                kyc_status=kyc_status, jurisdiction=jurisdiction, metadata=md,
                timestamp=ts,
            )
            return True
        # shared_redis — ATOMIC check-and-reserve on the authoritative shared store.
        entry = self._build_entry(
            kind="onramp_execute", user_id=user_id, usd_amount=usd_amount,
            ftns_amount=0.0, status="PENDING", address=address,
            kyc_status=kyc_status, jurisdiction=jurisdiction, metadata=md,
            timestamp=ts,
        )
        reserved = self._shared_total.try_reserve(
            user_id=entry.user_id,
            dedup_key=str(dedup_key),
            entry_id=entry.entry_id,
            usd_amount=entry.usd_amount,
            timestamp=entry.timestamp,
            limit_usd=float(limit_usd),
            window_sec=window_sec,
            now=entry.timestamp,
        )
        if reserved:
            # Mirror to the local deque for audit/export ONLY (the shared zadd already happened
            # atomically above — going through record() would double-write to the shared store).
            with self._lock:
                self._entries.append(entry)
                self._by_id[entry.entry_id] = entry
            self._write_to_disk(entry)
        return reserved

    def get(
        self, entry_id: str,
    ) -> Optional[FiatComplianceEntry]:
        with self._lock:
            return self._by_id.get(entry_id)

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[FiatComplianceEntry]:
        if not isinstance(limit, int) or limit <= 0 or limit > 10000:
            raise ValueError(
                f"limit must be in [1, 10000], got {limit}"
            )
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(
                f"offset must be >= 0, got {offset}"
            )
        if kind is not None and kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_KINDS)}, "
                f"got {kind!r}"
            )
        with self._lock:  # Sp896 — snapshot under lock, then iterate.
            snap = list(self._entries)
        snap.reverse()
        if kind:
            snap = [e for e in snap if e.kind == kind]
        if user_id:
            snap = [e for e in snap if e.user_id == user_id]
        return snap[offset:offset + limit]

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    # Sprint 285 / 885 — sum SETTLED fiat USD volume per user over
    # a rolling window. Backs the tier-limit enforcement on
    # onramp/execute (sp884).
    #
    # Sp885 correction: count only EXECUTES, NOT quotes. Quotes are
    # non-binding price checks ("nothing has moved on-chain or via
    # fiat rails") — counting them let a user exhaust their tier
    # limit by price-shopping without transacting, AND executes
    # (the real settled volume) recorded nothing, so the rolling
    # total counted the wrong event entirely. Regulatory limits
    # (FinCEN MSB) are on transacted volume, so settled executes
    # are the correct denominator. Quotes are still recorded for
    # the audit trail (compliance export / summary_by_kind) — they
    # just don't burn the limit. Gasless (FTNS-denominated) + KYC
    # (zero-amount) remain excluded.
    _FIAT_USD_KINDS = frozenset({
        "onramp_execute", "offramp_execute",
    })

    def total_usd_for_user(
        self,
        user_id: str,
        window_sec: int = 86400,  # 24h default
    ) -> float:
        """Sum usd_amount across fiat-surface events for this
        user_id within the rolling window. Empty user_id
        returns 0.0 (explicit-address flows aren't aggregated
        — there's no stable identity to aggregate against)."""
        if not user_id:
            return 0.0
        # Sp1179 — in shared_redis mode, read the GLOBAL (cross-replica)
        # total from the shared backend. Errors PROPAGATE so the tier
        # check fails CLOSED (deny) rather than silently under-enforcing
        # off a stale per-pod view.
        if self._shared_total is not None:
            return self._shared_total.total_usd_for_user(
                user_id, window_sec=window_sec, now=time.time(),
            )
        cutoff = time.time() - window_sec
        with self._lock:  # Sp896 — snapshot, then sum lock-free.
            snap = list(self._entries)
        # sp968 — dedup by intent_id so an onramp's execute-time PENDING
        # reservation + its later CONFIRMED settle (same intent) count ONCE
        # (the latest entry by timestamp wins → reflects the settled amount once
        # confirmed, the reserved amount while in-flight). Entries WITHOUT an
        # intent_id (offramp / legacy) each count individually. The PENDING
        # reservation is what closes the rolling-total TOCTOU: an in-flight
        # onramp is counted immediately, not only minutes later at CONFIRMED.
        total = 0.0
        latest_by_intent: Dict[str, Any] = {}  # intent_id -> (timestamp, usd_amount)
        for e in snap:
            if e.user_id != user_id:
                continue
            if e.kind not in self._FIAT_USD_KINDS:
                continue
            if e.timestamp < cutoff:
                continue
            intent_id = (e.metadata or {}).get("intent_id")
            if intent_id:
                prev = latest_by_intent.get(intent_id)
                if prev is None or e.timestamp >= prev[0]:
                    latest_by_intent[intent_id] = (e.timestamp, e.usd_amount)
            else:
                total += e.usd_amount
        total += sum(amount for _ts, amount in latest_by_intent.values())
        return total

    def summary_by_kind(self) -> Dict[str, Dict[str, float]]:
        """Aggregate count + total USD volume per kind. Empty
        ring → empty dict."""
        out: Dict[str, Dict[str, float]] = {}
        with self._lock:  # Sp896 — snapshot, then aggregate lock-free.
            snap = list(self._entries)
        for e in snap:
            bucket = out.setdefault(
                e.kind, {"count": 0, "total_usd": 0.0},
            )
            bucket["count"] += 1
            bucket["total_usd"] += e.usd_amount
        return out

    # ── Persistence ──────────────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        loaded: List[Dict[str, Any]] = []
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                loaded.append(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FiatComplianceRing: skipping corrupt %s: %s",
                    path, exc,
                )
        loaded.sort(key=lambda d: d.get("timestamp", 0))
        for d in loaded:
            try:
                entry = FiatComplianceEntry.from_dict(d)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "FiatComplianceRing: malformed entry %r: %s",
                    d, exc,
                )
                continue
            self._entries.append(entry)
            self._by_id[entry.entry_id] = entry

    def _write_to_disk(
        self, entry: FiatComplianceEntry,
    ) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / f"{entry.entry_id}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(entry.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FiatComplianceRing: disk write failed "
                "for %s: %s",
                entry.entry_id, exc,
            )
