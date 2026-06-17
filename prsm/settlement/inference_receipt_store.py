"""Sprint 1141 — §7-Brick-1: producer retention of the §7 InferenceReceipt.

The settlement-receipt data plane (Bricks A-E.2) detects ON-CHAIN settlement fraud
(INVALID_SIGNATURE / DOUBLE_SPEND): it retains the committed ``BatchedReceipt`` set
(``PublishedBatchStore``, Brick A) so an observer can recompute the merkle root and
challenge a bad leaf. The §7 follow-on extends fraud detection to COMPUTE-INTEGRITY
(faked / spliced distributed inference) via the EXISTING ChallengeWatcher (sp1129) +
``verify_inference_receipt_for_challenge`` (``challenge_verifier.py``) — which need the
§7 ``InferenceReceipt`` (the settler signature over the WHOLE receipt + the
``stage_activation_chain`` + ``topology_assignment`` + ``request_id`` + ``prompt_hash``),
NOT just the settlement ``BatchedReceipt``.

THE GAP this brick closes: the adapter
(``inference_adapter.inference_receipt_to_batched_receipt``) RE-SIGNS the SHARD payload
(the on-chain leaf binds a DIFFERENT preimage than the §7 ``settler_signature``) and
produces a ``BatchedReceipt``. The §7-specific fields do NOT survive into settlement, so
an observer holding a committed batch leaf has no channel to fetch the §7 receipt to run
the §7 verifier on it.

THE LINKING-KEY CRUX (the reason this store is keyed the way it is): an observer sees a
committed batch with ``leaf_hashes`` (Brick A/E.1). To run the §7 verifier on the receipt
at a leaf, it must look the §7 ``InferenceReceipt`` up by THAT leaf hash. The producer, at
adapt time, has the ``BatchedReceipt`` -> its leaf -> ``hash_leaf(batched_receipt_to_leaf
(batched))`` (``merkle.py``). So this store retains the §7 ``InferenceReceipt`` keyed by
THAT leaf_hash — IDENTICAL to what the committed batch will carry — so the observer's
``get(leaf_hash)`` works.

Pattern: mirrors Brick A ``PublishedBatchStore`` — bounded (evict oldest), GC-able via
``prune(retention_seconds)``, atomic persist via ``SettlementStateStore``, NEVER raises on
a malformed/oversized on-disk entry (skip + log). Like Brick A, this store is read/written
only OUTSIDE the money path; a write happens after the ``BatchedReceipt`` is produced,
wrapped in a try/except, and a failure here must NEVER break adapt/accumulate (§7
retention guards no funds — the money path is sacred).
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from prsm.compute.inference.models import InferenceReceipt
from prsm.settlement.state_store import SettlementStateStore

logger = logging.getLogger(__name__)

_STORE_VERSION = 1
# Defensive cap: a settler public key b64 is ~44 chars; a §7 receipt's serialized form is
# bounded by its (optional) stage chain. Reject anything absurd so a forged/oversized
# on-disk entry can't blow up memory on reload.
_MAX_PUBKEY_B64_LEN = 4096


# ── the retained record ───────────────────────────────────────────────


@dataclass(frozen=True)
class RetainedInferenceReceipt:
    """A §7 ``InferenceReceipt`` retained for compute-integrity challenge, keyed by the
    committed batch's leaf hash.

    ``leaf_hash`` == ``hash_leaf(batched_receipt_to_leaf(adapted))`` — the SAME 32-byte
    leaf an observer reads off the committed batch (Brick A/E.1), so the observer's
    ``get(leaf_hash)`` returns this record. ``settler_public_key_b64`` is the settler's
    Ed25519 pubkey (derivable from the adapter identity), sufficient for the §7 verifier's
    INVALID_SETTLER_SIGNATURE ground. ``stage_public_keys`` ({node_id: pubkey_b64}) is
    OPTIONAL — when None the verifier still runs internal-link / topology / request_id
    checks (per-stage signature authenticity is just skipped)."""

    leaf_hash: bytes
    inference_receipt: InferenceReceipt
    settler_public_key_b64: str
    stage_public_keys: Optional[Dict[str, str]]
    retained_at: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leaf_hash": self.leaf_hash.hex(),
            # Reuse InferenceReceipt.to_dict: preserves settler_signature +
            # stage_activation_chain + topology_assignment + prompt_hash byte-exact.
            "inference_receipt": self.inference_receipt.to_dict(),
            "settler_public_key_b64": self.settler_public_key_b64,
            "stage_public_keys": (
                dict(self.stage_public_keys)
                if self.stage_public_keys is not None
                else None
            ),
            "retained_at": int(self.retained_at),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetainedInferenceReceipt":
        """Inverse of ``to_dict``. Raises on a malformed entry (the store catches +
        skips per its never-raise-on-load contract)."""
        pubkey = d["settler_public_key_b64"]
        if not isinstance(pubkey, str) or not pubkey:
            raise ValueError("settler_public_key_b64 must be a non-empty string")
        if len(pubkey) > _MAX_PUBKEY_B64_LEN:
            raise ValueError(
                f"settler_public_key_b64 length {len(pubkey)} exceeds cap "
                f"{_MAX_PUBKEY_B64_LEN}"
            )
        stage_raw = d.get("stage_public_keys")
        if stage_raw is not None and not isinstance(stage_raw, dict):
            raise ValueError("stage_public_keys must be a dict or null")
        return cls(
            leaf_hash=bytes.fromhex(d["leaf_hash"]),
            inference_receipt=InferenceReceipt.from_dict(d["inference_receipt"]),
            settler_public_key_b64=pubkey,
            stage_public_keys=(dict(stage_raw) if stage_raw is not None else None),
            retained_at=int(d["retained_at"]),
        )


# ── the store (mirrors PublishedBatchStore) ────────────────────────────


class InferenceReceiptStore:
    """Bounded, GC-able, atomically-persisted store of §7 InferenceReceipts keyed by
    committed-batch leaf hash.

    Keyed by ``leaf_hash`` (hex). Insertion order is the eviction order (oldest-first), so
    ``max_entries`` overflow evicts the oldest. Persistence mirrors ``SettlementStateStore``
    (atomic .tmp write + fsync + os.replace; bytes hex). Re-putting an existing leaf_hash
    refreshes it (moved to newest).

    Never raises on a malformed/oversized on-disk entry: such an entry is skipped + logged
    on load. A corrupt JSON *file* is treated the same way — start empty + log — because
    §7 retention is best-effort and must never block a producer from committing money
    (contrast ``SettlementStateStore``, which is loud because IT guards escrowed funds)."""

    def __init__(
        self,
        path: Any,
        *,
        max_entries: int = 50_000,
        now: Callable[[], float] = time.time,
    ):
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive (got {max_entries})")
        self._path = Path(path)
        self._max_entries = max_entries
        self._now = now
        self._store = SettlementStateStore(self._path)
        # OrderedDict[leaf_hash_hex -> RetainedInferenceReceipt], insertion-ordered
        # (oldest first) so popitem(last=False) evicts the oldest.
        self._receipts: "OrderedDict[str, RetainedInferenceReceipt]" = OrderedDict()
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    # ── public API ────────────────────────────────────────────────

    def put(
        self,
        leaf_hash: bytes,
        inference_receipt: InferenceReceipt,
        settler_public_key_b64: str,
        stage_public_keys: Optional[Dict[str, str]] = None,
    ) -> None:
        """Retain the §7 ``inference_receipt`` under ``leaf_hash`` (the committed batch
        leaf). Re-putting an existing leaf_hash refreshes it (moved to newest). Evicts the
        oldest if over ``max_entries``. Persisted atomically."""
        rec = RetainedInferenceReceipt(
            leaf_hash=bytes(leaf_hash),
            inference_receipt=inference_receipt,
            settler_public_key_b64=settler_public_key_b64,
            stage_public_keys=(
                dict(stage_public_keys) if stage_public_keys is not None else None
            ),
            retained_at=int(self._now()),
        )
        key = rec.leaf_hash.hex()
        # Re-put moves to newest (refresh recency).
        if key in self._receipts:
            del self._receipts[key]
        self._receipts[key] = rec
        # Bound: evict oldest until within cap.
        while len(self._receipts) > self._max_entries:
            evicted_key, _ = self._receipts.popitem(last=False)
            logger.debug(
                "InferenceReceiptStore: evicted oldest receipt %s (over "
                "max_entries=%d)", evicted_key, self._max_entries,
            )
        self._persist()

    def get(self, leaf_hash: bytes) -> Optional[RetainedInferenceReceipt]:
        return self._receipts.get(bytes(leaf_hash).hex())

    def prune(self, retention_seconds: int) -> int:
        """Drop every entry whose ``retained_at + retention_seconds < now``. Returns the
        number dropped. Persists iff anything was dropped."""
        cutoff = self._now() - retention_seconds
        expired = [
            k for k, rec in self._receipts.items() if rec.retained_at < cutoff
        ]
        for k in expired:
            del self._receipts[k]
        if expired:
            self._persist()
        return len(expired)

    # ── persistence (mirrors SettlementStateStore discipline) ─────

    def _persist(self) -> None:
        state = {
            "version": _STORE_VERSION,
            # dict preserves insertion order in Py3.7+; the load path also restores
            # order, so eviction order survives a restart.
            "receipts": {k: rec.to_dict() for k, rec in self._receipts.items()},
        }
        try:
            self._store.save(state)
        except Exception as exc:  # pragma: no cover - disk-failure path
            # Retention is best-effort; a persist failure is logged, never raised (the
            # in-memory set is still usable this run, and the money path that called us
            # must never see an exception).
            logger.warning(
                "InferenceReceiptStore: persist to %s failed (%s); retention "
                "in-memory only this run", self._path, exc,
            )

    def _load(self) -> None:
        try:
            raw = self._store.load()
        except Exception as exc:
            # A corrupt retention file must NOT block the producer (it guards no money).
            logger.warning(
                "InferenceReceiptStore: state at %s unreadable (%s); starting "
                "empty — retention only, no funds at risk", self._path, exc,
            )
            return
        if raw is None:
            return
        receipts = raw.get("receipts")
        if not isinstance(receipts, dict):
            logger.warning(
                "InferenceReceiptStore: %s has no 'receipts' object; starting empty",
                self._path,
            )
            return
        for key, entry in receipts.items():
            try:
                rec = RetainedInferenceReceipt.from_dict(entry)
            except Exception as exc:
                # Skip the single bad entry, keep the rest. Never raise.
                logger.warning(
                    "InferenceReceiptStore: skipping malformed entry %s in %s (%s)",
                    key, self._path, exc,
                )
                continue
            # Re-key by the record's own leaf_hash hex (authoritative), not the raw map
            # key, so a tampered key can't shadow a real entry.
            self._receipts[rec.leaf_hash.hex()] = rec


# ── the additive, never-break-the-money-path adapt+retain helper ──────


def adapt_and_retain_inference_receipt(
    *,
    receipt: Any,
    identity: Any,
    store: Optional[InferenceReceiptStore] = None,
    settler_public_key_b64: Optional[str] = None,
    stage_public_keys: Optional[Dict[str, str]] = None,
    **adapter_kwargs: Any,
) -> Any:
    """Adapt a §7 ``InferenceReceipt`` to a settlement ``BatchedReceipt`` (the EXISTING
    money-path math, unchanged) and — ADDITIVELY, when a ``store`` is provided — retain the
    ORIGINAL §7 receipt keyed by the produced batch's leaf hash.

    Default-off safety: ``store=None`` (the default) makes this a pure pass-through to
    ``inference_receipt_to_batched_receipt`` — the returned ``BatchedReceipt`` is
    byte-for-byte identical and NOTHING §7 is touched.

    Money-path sacredness: the retention write is wrapped in try/except. A retention
    failure (store.put raises, leaf-hash recompute raises, anything) is logged and
    swallowed — the produced ``BatchedReceipt`` is ALWAYS returned. §7 retention guards no
    funds; it must never break adapt/accumulate.

    ``settler_public_key_b64`` defaults to ``identity.public_key_b64`` (the settler IS the
    adapter identity). ``stage_public_keys`` may be None (the §7 verifier still runs
    internal-link / topology / request_id checks)."""
    from prsm.settlement.inference_adapter import (
        inference_receipt_to_batched_receipt,
    )

    # The money path: NO change to the adapter math.
    batched = inference_receipt_to_batched_receipt(
        receipt=receipt, identity=identity, **adapter_kwargs
    )

    # ADDITIVE: retain the §7 receipt keyed by the committed-batch leaf hash. Wrapped so a
    # retention failure NEVER unwinds the produced BatchedReceipt.
    if store is not None:
        try:
            from prsm.settlement.merkle import (
                batched_receipt_to_leaf,
                hash_leaf,
            )

            leaf_hash = hash_leaf(batched_receipt_to_leaf(batched))
            pubkey = (
                settler_public_key_b64
                if settler_public_key_b64 is not None
                else getattr(identity, "public_key_b64", None)
            )
            if pubkey:
                store.put(
                    leaf_hash,
                    receipt,
                    settler_public_key_b64=pubkey,
                    stage_public_keys=stage_public_keys,
                )
            else:
                logger.debug(
                    "adapt_and_retain_inference_receipt: no settler pubkey "
                    "derivable; skipping §7 retention (BatchedReceipt unaffected)"
                )
        except Exception as exc:  # noqa: BLE001 — retention must NEVER break the money path
            logger.warning(
                "adapt_and_retain_inference_receipt: §7 retention failed "
                "(non-fatal; settlement BatchedReceipt stands): %s: %s",
                type(exc).__name__, exc,
            )

    return batched
