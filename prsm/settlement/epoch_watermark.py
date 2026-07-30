"""Sprint 1486 — durable watermark for emission-epoch attribution.

`build_emission_epoch` guarantees exactly-once attribution ONLY if the caller
remembers which batches previous epochs already paid. sp1481 returned that set and
left persistence to a caller — and no caller existed, so the guarantee was
theoretical: the first real epoch job would have kept it in memory and double-paid
every provider after a restart.

WHY DOUBLE-PAY IS THE FAILURE TO ENGINEER AGAINST
-------------------------------------------------
The two ways to be wrong are not symmetric:

  * Lose the watermark  -> batches counted twice -> providers paid twice from a
    SHARED pot -> everyone else is underpaid. Unrecoverable without manual
    reconciliation, and invisible until someone audits the pot.
  * Skip a batch        -> a provider is paid late. Recoverable: the batch simply
    isn't in the consumed set, so the NEXT epoch picks it up.

So every ordering decision here favours "might pay late" over "might pay twice".

ORDERING: PUBLISH FIRST, THEN PERSIST
-------------------------------------
Persisting before publishing would mark batches consumed for an epoch that might
never reach the chain — silent, permanent non-payment. So the root is published
first and the watermark advanced only after it lands.

That leaves the crash window (published, not yet persisted), which would otherwise
double-count on restart. It is closed by making the CHAIN authoritative rather than
the file: `reconcile_from_chain` asks the contract whether an epoch id is already
published and, if so, advances the watermark from the plan that produced it. This
is the same lesson as sp1472/sp1474 — when a durable external system already knows
the answer, do not keep a second private opinion of it.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


class WatermarkIntegrityError(RuntimeError):
    """The on-disk watermark is unreadable or malformed.

    Raised rather than silently starting empty: an empty watermark is
    indistinguishable from "nothing has ever been paid", so treating a corrupt
    file as empty would re-pay every historical batch.
    """


class EpochWatermarkStore:
    """Durable record of which finalized batches have already been paid.

    Atomic on every write (tmp file + os.replace) so a crash mid-write cannot
    truncate the set into a partial state that silently re-pays the missing tail.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._consumed: Set[str] = set()
        self._last_epoch_id: Optional[int] = None
        self._loaded = False

    # ── load / save ─────────────────────────────────────────────────────

    def load(self) -> "EpochWatermarkStore":
        """Load the watermark. A MISSING file is a legitimate cold start; a
        malformed one is not, and raises."""
        if not self.path.exists():
            self._loaded = True
            return self
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise WatermarkIntegrityError(
                f"watermark at {self.path} is unreadable ({exc}). Refusing to start "
                "with an empty set — that would re-pay every historical batch. "
                "Restore it from backup or reconcile against the chain."
            ) from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("consumed_batch_ids"), list):
            raise WatermarkIntegrityError(
                f"watermark at {self.path} is malformed (expected "
                "{{'consumed_batch_ids': [...]}}). Refusing to start empty."
            )
        self._consumed = {str(b).lower() for b in raw["consumed_batch_ids"]}
        last = raw.get("last_epoch_id")
        self._last_epoch_id = int(last) if last is not None else None
        self._loaded = True
        return self

    def _save(self) -> None:
        payload = {
            "consumed_batch_ids": sorted(self._consumed),
            "last_epoch_id": self._last_epoch_id,
            "count": len(self._consumed),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: a crash mid-write leaves the previous complete file intact rather
        # than a truncated set whose missing tail would be re-paid.
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ── queries ─────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise WatermarkIntegrityError(
                "watermark used before load() — refusing to build an epoch against "
                "an unknown consumed set (it would re-pay historical batches)"
            )

    @property
    def consumed(self) -> Set[str]:
        self._require_loaded()
        return set(self._consumed)

    @property
    def last_epoch_id(self) -> Optional[int]:
        self._require_loaded()
        return self._last_epoch_id

    def next_epoch_id(self) -> int:
        self._require_loaded()
        return 1 if self._last_epoch_id is None else self._last_epoch_id + 1

    def has_consumed(self, batch_id: str) -> bool:
        self._require_loaded()
        return str(batch_id).lower() in self._consumed

    # ── advance ─────────────────────────────────────────────────────────

    def commit_epoch(self, epoch_id: int, consumed_batch_ids: Iterable[str]) -> int:
        """Record that ``epoch_id`` was PUBLISHED and consumed these batches.

        Call this only AFTER the root is on chain (see the module docstring on
        ordering). Idempotent: committing the same epoch twice is a no-op rather
        than an error, so a retry after an ambiguous publish is safe.
        """
        self._require_loaded()
        if self._last_epoch_id is not None and epoch_id <= self._last_epoch_id:
            # Already recorded (or older) — do not rewind the watermark.
            new = {str(b).lower() for b in consumed_batch_ids} - self._consumed
            if not new:
                return 0
            # A genuinely new batch attributed to an already-committed epoch means
            # the caller is replaying with different inputs; absorb it rather than
            # leaving it to be double-counted later.
            self._consumed |= new
            self._save()
            return len(new)
        before = len(self._consumed)
        self._consumed |= {str(b).lower() for b in consumed_batch_ids}
        self._last_epoch_id = int(epoch_id)
        self._save()
        return len(self._consumed) - before

    def reconcile_from_chain(
        self, epoch_id: int, plan_batch_ids: Iterable[str], *, published_on_chain: bool,
    ) -> bool:
        """Close the publish-then-persist crash window.

        If the chain says ``epoch_id`` is already published but our watermark has
        not advanced, we crashed between broadcast and persist. The CHAIN is
        authoritative: adopt the plan's batches so the next epoch does not
        re-attribute them. Returns True when a recovery was applied.
        """
        self._require_loaded()
        if not published_on_chain:
            return False
        if self._last_epoch_id is not None and epoch_id <= self._last_epoch_id:
            return False
        self.commit_epoch(epoch_id, plan_batch_ids)
        return True

    def stats(self) -> Dict[str, Any]:
        self._require_loaded()
        return {
            "path": str(self.path),
            "consumed_batches": len(self._consumed),
            "last_epoch_id": self._last_epoch_id,
            "next_epoch_id": self.next_epoch_id(),
        }
