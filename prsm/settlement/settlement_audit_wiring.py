"""Sprint 1140 — settlement-receipt data plane BRICK E.2: the live-daemon WIRING logic.

Bricks A-E.1 produced PURE, injectable, offline-verified components (see
``docs/2026-06-16-settlement-receipt-data-plane-design.md``):
  - A ``PublishedBatchStore`` (producer receipt retention, already wired into
    ``BatchSettlementClient`` via an optional ``published_batch_store=`` arg);
  - B ``receipt_set_blob`` (serialize / cid / deserialize / on-chain-root cross-check);
  - C ``settlement_gossip`` (``SettlementBatchAnnouncer`` / ``SettlementBatchAdIndex``);
  - D ``settlement_audit_engine`` (selectors / ``VerifiedBatchCache`` / dry-run-gated
    ``SettlementAuditEngine``);
  - E.1 ``settlement_audit_loop`` (the PURE ``run_once`` observe pass + the read-only
    ``Web3SettlementContractClient.enumerate_committed_batches`` enumerator).

Brick E.2 is the OPT-IN daemon glue. The hard constraint is that ``node.initialize()``
behavior is BYTE-FOR-BYTE unchanged when the single env flag ``PRSM_SETTLEMENT_AUDIT`` is
OFF (the default). So ALL the real logic lives here, fully unit-tested with fakes, and
node.py gets only thin ``if audit_enabled():`` glue:

  - ``audit_enabled(env)`` — the single gate (1/true/yes/on).
  - ``BlockCursor`` — a tiny persisted last-fully-scanned-block cursor (reuses the
    ``SettlementStateStore`` atomic-write discipline; ``:memory:`` disables durability).
  - ``build_settlement_audit_components(...)`` — the FACTORY: returns None when disabled;
    else assembles the ad-index + a composite (OwnStake ∪ ConsensusGroup) selector +
    cache + engine + loop + announcer + scheduler into one ``SettlementAuditBundle``.
  - ``SettlementAuditScheduler`` — the block-cursor scheduler: each tick scans
    [cursor+1, head]; on SUCCESS advances the cursor to head; on ANY failure (chain-head
    error OR run_once raise) does NOT advance (retry the same window next tick) and NEVER
    raises out of the loop (fail-closed — the audit guards no funds, must not crash the
    daemon).
  - ``announce_committed_batches(announcer, committed)`` — best-effort announce-after-
    commit; never raises.

NEVER-BROADCAST end-to-end: the scheduler only ever drives the dry-run-gated engine via
``audit_loop.run_once`` (which itself only fetches + cross-checks + dry-runs). The
``dry_run_client`` is the read-only one; no submit/broadcast/finalize is ever reached
from this module.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Set

from prsm.settlement.settlement_audit_engine import (
    ConsensusGroupSelector,
    ObservedBatch,
    OwnStakeSelector,
    SettlementAuditEngine,
    VerifiedBatchCache,
)
from prsm.settlement.settlement_audit_loop import SettlementAuditLoop
from prsm.settlement.settlement_audit_report import summarize_audit_result
from prsm.settlement.settlement_gossip import (
    SettlementBatchAdIndex,
    SettlementBatchAnnouncer,
)

logger = logging.getLogger(__name__)

_AUDIT_ENV_FLAG = "PRSM_SETTLEMENT_AUDIT"
_TRUTHY = ("1", "true", "yes", "on")
_CURSOR_VERSION = 1


# ── the gate ──────────────────────────────────────────────────────────


def audit_enabled(env=None) -> bool:
    """True iff ``PRSM_SETTLEMENT_AUDIT`` is one of 1/true/yes/on (case-insensitive,
    whitespace-trimmed). The SINGLE flag that gates the entire data-plane audit wiring;
    default OFF leaves node.initialize() byte-for-byte unchanged."""
    environ = os.environ if env is None else env
    return str(environ.get(_AUDIT_ENV_FLAG, "")).strip().lower() in _TRUTHY


# ── the persisted block cursor ────────────────────────────────────────


class BlockCursor:
    """A tiny persisted last-fully-scanned-block cursor.

    ``load()`` returns the last block fully scanned (default ``-1`` so the first scan
    starts at block 0 via ``load()+1``); ``save(block)`` records it atomically. Reuses
    the ``SettlementStateStore`` atomic-write (.tmp + fsync + os.replace) discipline so a
    crash mid-write can't corrupt the cursor. ``path=":memory:"`` disables durability
    (in-memory only — for tests / ephemeral nodes).

    The cursor is the ONLY thing that makes the scheduler gap-free + overlap-free across
    restarts: it advances ONLY on a successful scan, so a crash before ``save`` simply
    re-scans the same window (idempotent — enumeration is read-only and the
    ``VerifiedBatchCache`` dedups by batch_id)."""

    _KEY = "last_scanned_block"

    def __init__(self, *, path: Any, default: int = -1):
        self._default = int(default)
        self._memory_only = str(path) == ":memory:"
        self._value: Optional[int] = None
        if self._memory_only:
            self._store = None
            self._path = None
        else:
            # Imported lazily to keep this module importable without the (heavier)
            # state-store dependency on the pure offline test path.
            from prsm.settlement.state_store import SettlementStateStore

            self._path = Path(path)
            self._store = SettlementStateStore(self._path)

    def load(self) -> int:
        """The last fully-scanned block, or ``default`` (-1) if none recorded. Never
        raises: a corrupt/unreadable cursor file falls back to the default (the audit
        re-scans from the start — it guards no funds, so a re-scan is safe, not a strand)."""
        if self._memory_only:
            return self._value if self._value is not None else self._default
        if self._value is not None:
            return self._value
        try:
            raw = self._store.load()
        except Exception as exc:  # noqa: BLE001 — a bad cursor must not crash the daemon
            logger.warning(
                "BlockCursor: state at %s unreadable (%s); starting from default %d "
                "(re-scan is safe — audit guards no funds)",
                self._path, exc, self._default,
            )
            self._value = self._default
            return self._value
        if raw is None:
            self._value = self._default
            return self._value
        try:
            self._value = int(raw.get(self._KEY, self._default))
        except (TypeError, ValueError):
            self._value = self._default
        return self._value

    def save(self, block: int) -> None:
        """Record ``block`` as the last fully-scanned block. Atomic + durable (unless
        memory-only). Never raises — a persist failure is logged; the in-memory value is
        still advanced so the running process doesn't re-scan, and the worst case after a
        crash is an idempotent re-scan."""
        self._value = int(block)
        if self._memory_only:
            return
        try:
            self._store.save({"version": _CURSOR_VERSION, self._KEY: int(block)})
        except Exception as exc:  # noqa: BLE001 — best-effort; never crash the daemon
            logger.warning(
                "BlockCursor: persist of block %d to %s failed (%s); in-memory only "
                "this run", block, self._path, exc,
            )


# ── the composite selector ────────────────────────────────────────────


class _CompositeSelector:
    """Union the kept sets of several ``AuditSelector``s, preserving first-seen order and
    de-duplicating by ``batch_id`` (a batch matched by more than one sub-selector is kept
    once). Pure — no network/chain."""

    def __init__(self, selectors: List[Any]):
        self._selectors = list(selectors)

    def select(self, observed: Iterable[ObservedBatch]) -> List[ObservedBatch]:
        observed = list(observed)
        seen: Set[bytes] = set()
        out: List[ObservedBatch] = []
        for sel in self._selectors:
            for b in sel.select(observed):
                key = bytes(b.batch_id)
                if key in seen:
                    continue
                seen.add(key)
                out.append(b)
        return out


# ── the assembled bundle ──────────────────────────────────────────────


@dataclass(frozen=True)
class SettlementAuditBundle:
    """Everything the daemon needs to run + shut down the audit capability. node.py
    holds this opaquely: it launches ``scheduler.run_forever()`` and, on the commit hook,
    uses ``announcer`` via ``announce_committed_batches``."""

    ad_index: SettlementBatchAdIndex
    selector: Any
    cache: VerifiedBatchCache
    engine: SettlementAuditEngine
    audit_loop: SettlementAuditLoop
    announcer: Optional[SettlementBatchAnnouncer]
    scheduler: "SettlementAuditScheduler"
    cursor: BlockCursor


# ── the factory ───────────────────────────────────────────────────────


def build_settlement_audit_components(
    *,
    enabled: bool,
    gossip: Any,
    fetcher: Any,
    contract_client: Any,
    my_address: Optional[str],
    group_ids: Optional[Set[bytes]],
    dry_run_client: Any,
    cursor: BlockCursor,
    content_provider: Any = None,  # accepted for symmetry; fetcher is pre-wrapped
    store: Any = None,
    content_publisher: Any = None,
    inference_receipt_store: Any = None,
    chain_head_fn: Optional[Callable[[], Awaitable[int]]] = None,
    interval_s: float = 600.0,
    max_fetches_per_run: int = 256,
    max_cached_batches: int = 10_000,
    findings_store: Any = None,
    on_findings: Optional[Callable[[Any], None]] = None,
) -> Optional[SettlementAuditBundle]:
    """Assemble the audit bundle, or None when ``enabled`` is False.

    Returning None when disabled is the keystone of default-off safety: node.py calls
    this only inside ``if audit_enabled():`` AND this returns None for a False ``enabled``,
    so nothing (no ad-index subscription, no scheduler, no store) is constructed.

    The selector is a composite: ``OwnStakeSelector(my_address)`` (always, when an address
    is known) UNION ``ConsensusGroupSelector(group_ids)`` (only when ``group_ids`` is
    non-empty). The engine gets the (read-only) ``dry_run_client``; the loop's enumerator
    is ``contract_client.enumerate_committed_batches``."""
    if not enabled:
        return None

    ad_index = SettlementBatchAdIndex(gossip)

    selectors: List[Any] = []
    if my_address:
        selectors.append(OwnStakeSelector(my_address))
    if group_ids:
        selectors.append(ConsensusGroupSelector(set(group_ids)))
    selector = _CompositeSelector(selectors)

    cache = VerifiedBatchCache(max_batches=max_cached_batches)
    engine = SettlementAuditEngine(cache, dry_run_client=dry_run_client)

    audit_loop = SettlementAuditLoop(
        enumerator=contract_client,
        selector=selector,
        ad_index=ad_index,
        fetcher=fetcher,
        cache=cache,
        engine=engine,
        max_fetches_per_run=max_fetches_per_run,
    )

    # sp1144 (§7 producer wiring): when a content publisher is wired the announcer
    # PUBLISHES the receipt-set blob (so the announced CID resolves, ad-cid ==
    # published-content-cid) and, when an InferenceReceiptStore is wired, ENRICHES the blob
    # with the retained §7 receipts. Both default None => the legacy plain-blob /
    # legacy-cid announcer (byte-for-byte the prior behavior).
    announcer = (
        SettlementBatchAnnouncer(
            store,
            gossip,
            content_publisher=content_publisher,
            inference_receipt_store=inference_receipt_store,
        )
        if store is not None
        else None
    )

    scheduler = SettlementAuditScheduler(
        audit_loop,
        chain_head_fn if chain_head_fn is not None else _null_chain_head,
        cursor=cursor,
        interval_s=interval_s,
        # sp1149 — ADDITIVE operator surface; both default None => WARNING-log-only (no
        # persist, no callback), preserving the prior behavior when unwired.
        findings_store=findings_store,
        on_findings=on_findings,
    )

    return SettlementAuditBundle(
        ad_index=ad_index,
        selector=selector,
        cache=cache,
        engine=engine,
        audit_loop=audit_loop,
        announcer=announcer,
        scheduler=scheduler,
        cursor=cursor,
    )


async def _null_chain_head() -> int:
    """A chain-head function placeholder that always errs (so a bundle built without a
    real head function never silently scans a bogus window). node.py injects the real
    ``w3.eth.block_number`` wrapper before launching the scheduler."""
    raise RuntimeError("chain_head_fn not configured")


# ── the scheduler ─────────────────────────────────────────────────────


class SettlementAuditScheduler:
    """Block-cursor-driven scheduler for the (read-only, never-broadcast) audit loop.

    Each tick: read the chain head; compute ``from_block = cursor.load()+1`` and
    ``to_block = head``; if ``to_block >= from_block`` run ``audit_loop.run_once(...)``
    and, ONLY on success, advance the cursor to ``to_block`` (so the next tick starts at
    ``to_block+1`` — contiguous, no gap, no overlap). On ANY exception (chain-head read OR
    run_once) the cursor is NOT advanced (the same window retries next tick) and the tick
    NEVER raises out — the audit guards no funds and must not crash the daemon.

    ``run_once_tick`` is the testable single iteration (inject a fake clock/head + a
    tmp_path cursor); ``run_forever`` is the daemon loop (tick → sleep, surviving
    per-iteration errors, exiting cleanly on cancellation)."""

    def __init__(
        self,
        audit_loop: Any,
        chain_head_fn: Callable[[], Awaitable[int]],
        *,
        cursor: BlockCursor,
        interval_s: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        start_block: Optional[int] = None,
        findings_store: Any = None,
        on_findings: Optional[Callable[[Any], None]] = None,
    ):
        self._audit_loop = audit_loop
        self._chain_head_fn = chain_head_fn
        self._cursor = cursor
        self._interval_s = max(0.0, float(interval_s))
        self._sleep = sleep
        self._start_block = start_block
        # sp1149 — OPTIONAL, ADDITIVE operator-facing findings surface. Both default None =>
        # only the WARNING log fires on a finding; no persist, no callback (byte-for-byte the
        # prior idle behavior on a no-finding tick). NEITHER ever broadcasts/signs.
        self._findings_store = findings_store
        self._on_findings = on_findings

    async def run_once_tick(self) -> None:
        """One scan iteration. Never raises (fail-closed)."""
        try:
            head = await self._chain_head_fn()
        except Exception as exc:  # noqa: BLE001 — chain-head read failed; retry next tick
            logger.warning(
                "SettlementAuditScheduler: chain-head read failed (%s); "
                "skipping tick (cursor unchanged)", exc,
            )
            return

        try:
            head = int(head)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "SettlementAuditScheduler: bogus chain head %r (%s); skipping tick",
                head, exc,
            )
            return

        last = self._cursor.load()
        from_block = (last + 1) if self._start_block is None else max(
            last + 1, int(self._start_block)
        )
        to_block = head
        if to_block < from_block:
            # Nothing new on chain since the last fully-scanned block. No scan, no
            # advance — no gap, no re-scan/overlap.
            logger.debug(
                "SettlementAuditScheduler: no new blocks (head=%d, last=%d); idle tick",
                head, last,
            )
            return

        try:
            result = await self._audit_loop.run_once(from_block, to_block)
        except Exception as exc:  # noqa: BLE001 — a failed scan must NOT advance the cursor
            logger.warning(
                "SettlementAuditScheduler: audit run_once failed for window [%d, %d] "
                "(%s); cursor NOT advanced (retry next tick)",
                from_block, to_block, exc,
            )
            return

        # SUCCESS — advance the cursor so the next tick is contiguous. (Advance FIRST: the
        # operator surface below is best-effort and must NEVER block or undo the advance.)
        self._cursor.save(to_block)

        # sp1149 — surface any actionable dispute candidates to the operator. On a tick WITH
        # findings: WARNING log (operator-visible) + best-effort persist + optional callback.
        # On a no-finding tick: keep the EXISTING debug idle behavior (byte-for-byte). All of
        # this is READ-ONLY — it NEVER broadcasts/signs; the user-gated submit path is
        # untouched. A persist/callback failure NEVER breaks the loop or undoes the cursor.
        summary = summarize_audit_result(result)
        if summary.total > 0:
            logger.warning(
                "SettlementAuditScheduler: settlement audit found %d dispute candidate(s) "
                "in window [%d, %d]\n%s",
                summary.total, from_block, to_block, summary.render(),
            )
            if self._findings_store is not None:
                try:
                    self._findings_store.record_summary(summary)
                except Exception as exc:  # noqa: BLE001 — best-effort; never break the loop
                    logger.warning(
                        "SettlementAuditScheduler: persisting audit findings failed (%s); "
                        "WARNING log above is authoritative this run", exc,
                    )
            if self._on_findings is not None:
                try:
                    self._on_findings(summary)
                except Exception as exc:  # noqa: BLE001 — best-effort; never break the loop
                    logger.warning(
                        "SettlementAuditScheduler: on_findings callback raised (%s); "
                        "ignored", exc,
                    )
        else:
            logger.debug(
                "SettlementAuditScheduler: scanned [%d, %d] ok (%s); cursor -> %d",
                from_block, to_block, result, to_block,
            )

    async def run_forever(self) -> None:
        """Daemon loop: tick → sleep, forever. Per-iteration errors are swallowed by
        ``run_once_tick``; exits cleanly on cancellation."""
        while True:
            try:
                await self.run_once_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — belt-and-suspenders; never die
                logger.warning(
                    "SettlementAuditScheduler: unexpected tick error (%s); continuing",
                    exc,
                )
            await self._sleep(self._interval_s)


# ── announce-after-commit ─────────────────────────────────────────────


def _batch_id_of(item: Any) -> Optional[bytes]:
    """Accept either a CommittedBatch-like object (``.batch_id``) or a raw 32-byte id."""
    if item is None:
        return None
    bid = getattr(item, "batch_id", item)
    try:
        return bytes(bid)
    except (TypeError, ValueError):
        return None


async def announce_committed_batches(
    announcer: Any, committed_batches: Iterable[Any],
) -> int:
    """Best-effort announce-after-commit. For each committed batch (a CommittedBatch-like
    object or a raw batch_id), call ``announcer.announce(batch_id)``. Returns the count of
    successful announces. NEVER raises: a None announcer / empty list / per-batch announce
    error is swallowed (announce carries no funds; the next commit re-announces and the
    on-chain commit is unaffected)."""
    if announcer is None or committed_batches is None:
        return 0
    n = 0
    for item in committed_batches:
        batch_id = _batch_id_of(item)
        if batch_id is None:
            continue
        try:
            ok = await announcer.announce(batch_id)
        except Exception as exc:  # noqa: BLE001 — best-effort; never raise into the caller
            logger.warning(
                "announce_committed_batches: announce of %s failed (%s); skipped",
                bytes(batch_id).hex(), exc,
            )
            continue
        if ok:
            n += 1
    return n
