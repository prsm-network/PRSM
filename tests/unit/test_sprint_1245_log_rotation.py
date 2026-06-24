"""Sprint 1245 — opt-in bounded/rotated FILE logging for a long-running node.

The proven deployment posture (`setsid -f bash start_node.sh ... >log 2>&1`)
redirects stdout to a logfile that grows UNBOUNDED → eventually fills the disk
and kills the node. configure_structlog accepted a log_file_path it never used,
so there was no file logging at all. This adds an opt-in size-based
RotatingFileHandler on the stdlib ROOT logger (the runtime path — node /
settlement / p2p / monitoring all use logging.getLogger). Default (PRSM_LOG_FILE
unset) → no-op, stdout-only behavior byte-identical. Fail-soft + idempotent.
"""
from __future__ import annotations

import logging
import os

import pytest

from prsm.core.logging_config import configure_rotating_file_logging


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """The helper mutates the global root logger — snapshot + restore so a test
    can't leak a file handler (open fd) into the rest of the suite."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for h in list(root.handlers):
        if h not in saved_handlers:
            try:
                h.close()
            finally:
                root.removeHandler(h)
    root.setLevel(saved_level)


def _rotating_handlers(root=None):
    root = root or logging.getLogger()
    return [h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)]


def test_noop_when_unset(monkeypatch):
    monkeypatch.delenv("PRSM_LOG_FILE", raising=False)
    before = list(logging.getLogger().handlers)
    assert configure_rotating_file_logging() is None
    assert logging.getLogger().handlers == before   # default behavior untouched


def test_attaches_rotating_handler_when_set(monkeypatch, tmp_path):
    logpath = tmp_path / "node.log"
    monkeypatch.setenv("PRSM_LOG_FILE", str(logpath))
    monkeypatch.setenv("PRSM_LOG_MAX_BYTES", "1048576")
    monkeypatch.setenv("PRSM_LOG_BACKUP_COUNT", "3")

    assert configure_rotating_file_logging() == str(logpath)
    handlers = _rotating_handlers()
    assert len(handlers) == 1
    assert handlers[0].maxBytes == 1048576
    assert handlers[0].backupCount == 3


def test_log_record_lands_in_file(monkeypatch, tmp_path):
    logpath = tmp_path / "node.log"
    monkeypatch.setenv("PRSM_LOG_FILE", str(logpath))
    logging.getLogger().setLevel(logging.INFO)
    configure_rotating_file_logging()

    logging.getLogger("prsm.test.sp1245").info("hello-rotated-logfile")
    for h in _rotating_handlers():
        h.flush()
    assert logpath.exists()
    assert "hello-rotated-logfile" in logpath.read_text()


def test_idempotent_no_double_attach(monkeypatch, tmp_path):
    logpath = tmp_path / "node.log"
    monkeypatch.setenv("PRSM_LOG_FILE", str(logpath))
    assert configure_rotating_file_logging() == str(logpath)
    assert configure_rotating_file_logging() == str(logpath)   # second call
    assert len(_rotating_handlers()) == 1                       # still one


def test_actually_rotates_past_max_bytes(monkeypatch, tmp_path):
    logpath = tmp_path / "node.log"
    monkeypatch.setenv("PRSM_LOG_FILE", str(logpath))
    monkeypatch.setenv("PRSM_LOG_MAX_BYTES", "512")   # tiny so it rolls fast
    monkeypatch.setenv("PRSM_LOG_BACKUP_COUNT", "2")
    logging.getLogger().setLevel(logging.INFO)
    configure_rotating_file_logging()

    log = logging.getLogger("prsm.test.sp1245.rot")
    for i in range(200):
        log.info("filler line %d %s", i, "x" * 40)
    for h in _rotating_handlers():
        h.flush()
    # rotation produced at least one backup, and total is bounded by
    # (backupCount+1) files — not one unbounded file.
    assert (tmp_path / "node.log.1").exists()
    rolled = list(tmp_path.glob("node.log*"))
    assert len(rolled) <= 3   # node.log + .1 + .2 (backupCount=2)


def test_zero_max_bytes_falls_back_to_bounded_default(monkeypatch, tmp_path):
    # maxBytes=0 in RotatingFileHandler means NEVER rotate (unbounded) — the exact
    # footgun this feature exists to prevent. A non-positive value must coerce to
    # the bounded default, not disable rotation.
    logpath = tmp_path / "node.log"
    monkeypatch.setenv("PRSM_LOG_FILE", str(logpath))
    monkeypatch.setenv("PRSM_LOG_MAX_BYTES", "0")
    configure_rotating_file_logging()
    assert _rotating_handlers()[0].maxBytes > 0


def test_fail_soft_on_unwritable_path(monkeypatch, tmp_path):
    # parent is an existing FILE → makedirs raises → must return None, not crash,
    # and must not leave a half-attached handler.
    blocker = tmp_path / "iam_a_file"
    blocker.write_text("x")
    monkeypatch.setenv("PRSM_LOG_FILE", str(blocker / "sub" / "node.log"))
    before = len(logging.getLogger().handlers)
    assert configure_rotating_file_logging() is None
    assert len(logging.getLogger().handlers) == before   # nothing attached
