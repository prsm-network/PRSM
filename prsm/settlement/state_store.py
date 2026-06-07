"""Sprint 1039 — durable settlement state store (brick 2.5).

A small, dataclass-agnostic atomic JSON dict store. ``BatchSettlementClient``
owns the dataclass<->dict translation and hands this store a plain JSON-able
dict to persist; keeping the store dumb avoids a circular import on the client's
dataclasses and makes it independently testable.

Durability matters here because the client's post-commit state (committed
batches awaiting finalization + broadcast-but-unconfirmed quarantine) represents
money ALREADY escrowed on chain. Losing it on a restart strands that money.

Two deliberate safety choices:
  - Atomic write: serialize to a sibling ``.tmp``, fsync, then ``os.replace``
    (atomic on POSIX). A crash mid-write never leaves a half-written state file
    in place — the previous good file survives.
  - LOUD on corruption: ``load`` raises ``SettlementStateCorruptError`` on
    unparseable JSON rather than returning ``None``. A silent empty rehydrate
    would make the client forget every committed batch and strand all of it; the
    caller (the client build) treats the raise as "settlement OFF + logged" so
    an operator notices and restores the file, instead of silently double-losing.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SettlementStateCorruptError(RuntimeError):
    """The on-disk settlement state file exists but could not be parsed.

    Raised (not swallowed) so the caller refuses to start settlement with an
    empty in-memory state — which would strand every committed batch — and an
    operator is alerted to restore/inspect the file instead.
    """


class SettlementStateStore:
    """Atomic JSON dict persistence for BatchSettlementClient state."""

    def __init__(self, path: Any):
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Optional[dict]:
        """Return the persisted dict, or ``None`` if no file exists yet.

        Raises ``SettlementStateCorruptError`` if the file exists but is not
        valid JSON OR is valid JSON that is not an object (e.g. truncated to
        ``null`` / ``0`` / ``[]``). Both are loud, never silent-empty — an empty
        rehydrate would forget every committed batch and strand the escrow. The
        caller additionally validates the schema version (sp1039 review #1/#4)."""
        if not self._path.exists():
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SettlementStateCorruptError(
                f"settlement state at {self._path} is corrupt: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise SettlementStateCorruptError(
                f"settlement state at {self._path} is not a JSON object "
                f"(got {type(parsed).__name__}) — refusing to start empty"
            )
        return parsed

    def save(self, state: dict) -> None:
        """Atomically persist ``state``. Parent dirs are created as needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path)  # atomic on POSIX
        # sp1039 review #3 — fsync the parent directory so the rename's
        # directory-entry update is itself committed to stable storage; without
        # this the os.replace can be lost on a crash even though the file's data
        # was fsynced. Best-effort: some platforms can't fsync a directory.
        try:
            dir_fd = os.open(str(self._path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):  # pragma: no cover - platform-specific
            pass
