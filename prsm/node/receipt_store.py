"""Sprint 242 — LRU-bounded ReceiptStore for signed inference
receipts.

Mirrors the JobHistoryStore design pattern (in-memory LRU with
optional filesystem persistence) to give /compute/inference a
durable audit trail. End-users + auditors can query receipts by
job_id post-hoc instead of relying on the caller having saved
the HTTP response.

Filesystem persistence opt-in via ``PRSM_RECEIPT_STORE_DIR``.
Filenames are SHA-256 of the job_id (path-traversal-proof).
Corrupt files fail-soft at constructor scan time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ENTRIES = 1024
_DEFAULT_LIST_MAX = 10000   # sp925 — cap on list_for_node_ids results


def _receipt_list_max() -> int:
    """sp925 — max receipts returned by list_for_node_ids (bounds the disk
    scan reachable via the public /devices/earnings). PRSM_RECEIPT_LIST_MAX."""
    try:
        return max(1, int(os.environ.get("PRSM_RECEIPT_LIST_MAX", _DEFAULT_LIST_MAX)))
    except (TypeError, ValueError):
        return _DEFAULT_LIST_MAX


class ReceiptStore:
    """LRU-bounded receipt cache keyed by job_id. Optional
    filesystem persistence under ``PRSM_RECEIPT_STORE_DIR``.
    """

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._max_entries = max_entries
        self._persist_dir = persist_dir
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._scan_disk()

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> "ReceiptStore":
        env = env if env is not None else os.environ
        raw = (env.get("PRSM_RECEIPT_STORE_DIR") or "").strip()
        if not raw:
            return cls(max_entries=max_entries)
        return cls(max_entries=max_entries, persist_dir=Path(raw))

    def _filename(self, job_id: str) -> str:
        return hashlib.sha256(
            job_id.encode("utf-8")
        ).hexdigest() + ".json"

    def _scan_disk(self) -> None:
        assert self._persist_dir is not None
        for f in sorted(self._persist_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "ReceiptStore: skip corrupt %s (%s)", f.name, e,
                )
                continue
            jid = data.get("job_id")
            if not isinstance(jid, str) or not jid:
                continue
            self._cache[jid] = data

    def put(self, job_id: str, receipt: Dict[str, Any]) -> None:
        if not job_id:
            raise ValueError("job_id must be non-empty")
        if not isinstance(receipt, dict):
            raise TypeError(
                f"receipt must be a dict; got {type(receipt).__name__}"
            )
        if job_id in self._cache:
            self._cache.move_to_end(job_id)
        self._cache[job_id] = receipt
        # Eviction
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        if self._persist_dir is not None:
            self._write_file(job_id, receipt)

    def _write_file(
        self, job_id: str, receipt: Dict[str, Any],
    ) -> None:
        assert self._persist_dir is not None
        path = self._persist_dir / self._filename(job_id)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(receipt, fh)
        except OSError as e:
            logger.warning(
                "ReceiptStore: failed to persist receipt for "
                "job_id=%s: %s", job_id[:8], e,
            )

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id in self._cache:
            self._cache.move_to_end(job_id)
            return self._cache[job_id]
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def count(self) -> int:
        """Symmetric with JobHistoryStore.count() — useful where
        callers want a method instead of len()."""
        return len(self._cache)

    def list_for_node_ids(
        self,
        node_ids: "Iterable[str]",
        *,
        max_results: Optional[int] = None,
    ) -> "List[Dict[str, Any]]":
        """Sprint 801 — return ALL receipts whose
        ``settler_node_id`` is in ``node_ids``.

        Persistence-aware: when ``persist_dir`` is set, iterates
        the on-disk file glob (uncapped by the in-memory cache).
        Without persist_dir, falls back to scanning the cache.

        Closes the sprint-793 limitation in
        ``_ReceiptStoreAdapter`` where queries were bounded by
        the 1024-entry LRU cap even when persistence was on —
        long-running operators silently lost earnings history
        for receipts that had aged out of cache.

        Empty ``node_ids`` → ``[]`` (defense against unbounded
        "give me all" scans).
        """
        ids = list(node_ids)
        if not ids:
            return []
        allowed = set(ids)
        # sp925 — hard cap so this O(N) scan can't be amplified into a DoS via
        # the PUBLIC, zero-cost /devices/earnings endpoint when persist_dir
        # holds a huge receipt history. Default PRSM_RECEIPT_LIST_MAX (10000);
        # truncation is logged (never silent).
        if max_results is None:
            max_results = _receipt_list_max()

        # Persistence path: scan disk, capped.
        if self._persist_dir is not None:
            results: List[Dict[str, Any]] = []
            for path in self._persist_dir.glob("*.json"):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("settler_node_id") in allowed:
                    results.append(data)
                    if len(results) >= max_results:
                        logger.warning(
                            "ReceiptStore.list_for_node_ids hit cap %d — "
                            "results truncated (raise PRSM_RECEIPT_LIST_MAX "
                            "if the operator needs more)", max_results,
                        )
                        break
            return results

        # In-memory fallback: scan the cache (also capped).
        out: List[Dict[str, Any]] = []
        for r in self._cache.values():
            if isinstance(r, dict) and r.get("settler_node_id") in allowed:
                out.append(r)
                if len(out) >= max_results:
                    break
        return out

    def list(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        model_id: Optional[str] = None,
    ) -> list:
        """Sprint 250 — paginated enumeration of stored receipts,
        most-recently-put first. Optional ``model_id`` filter.

        Raises ``ValueError`` for out-of-range limit / offset.
        """
        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise ValueError(
                f"limit must be in [1, 1000]; got {limit}"
            )
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(
                f"offset must be >= 0; got {offset}"
            )
        # OrderedDict insertion order — most recent is at the
        # end. Reverse for newest-first.
        items = list(self._cache.values())
        items.reverse()
        if model_id:
            items = [
                r for r in items
                if r.get("model_id") == model_id
            ]
        return items[offset:offset + limit]
