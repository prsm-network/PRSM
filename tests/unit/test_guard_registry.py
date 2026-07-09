"""Sprint 1415 — enforce the money/security guard registry.

The registry (prsm/security/guard_registry.py) is the answer to this session's recurring failure:
a defense that exists, is wired, and yet cannot fire, with no test that would notice
(sp1412/sp1178/sp1172/sp1411). Each registered guard names the anchor line that IS the guard and the
test that must die with it.

Enforcement (this file, in the normal suite, CI-blocking): every guard's anchor line is still present
and unique at its cited file, and every named killing test exists and targets it. So deleting or
moving a guard line — the sp1412 failure mode — or deleting its killing test trips CI immediately.

This is the durable, cheap half. The other half — proving each killing test actually goes RED when the
guard is removed, not merely that it exists — is NOT automated here. A generic "delete the line and
re-run" mutation is the unbounded mutation-testing problem: a guard that spans a multi-line condition
mutates into a syntax error, which "fails" for the wrong reason and proves nothing. Instead, each
guard's non-vacuity is proven AT SHIP TIME and re-provable by hand:

    # temporarily neutralize the guard, then run its killing test — it MUST fail:
    git stash push -- <guard.file>            # or hand-edit the anchor to a no-op
    python -m pytest <guard.killed_by>::<guard.kills_test_id>   # expect FAIL
    git stash pop                             # restore; confirm `git status` is clean

Every guard below was shipped with exactly that demonstration recorded in its sprint commit. When you
ADD a guard, do the same and note it — an entry whose killing test passes with the guard removed is a
lie the registry cannot detect on its own.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prsm.security.guard_registry import GUARDS

REPO_ROOT = Path(__file__).resolve().parents[2]

# Parametrization by guard id keeps failures legible (the id names the broken guard).
_BY_ID = {g.id: g for g in GUARDS}


def test_registry_is_non_empty_and_ids_unique():
    assert GUARDS, "the guard registry is empty — it must list the load-bearing money/security guards"
    ids = [g.id for g in GUARDS]
    assert len(ids) == len(set(ids)), f"duplicate guard ids: {ids}"


@pytest.mark.parametrize("gid", list(_BY_ID))
def test_guard_anchor_is_present_and_unique(gid):
    """The anchor line that IS the guard must still exist — and exactly once — in its file. A silent
    deletion, or a move that drops the anchor, trips here (the sp1412 failure mode)."""
    g = _BY_ID[gid]
    path = REPO_ROOT / g.file
    assert path.exists(), f"{gid}: guard file {g.file} does not exist"
    occurrences = path.read_text().count(g.anchor)
    assert occurrences == 1, (
        f"{gid}: anchor {g.anchor!r} appears {occurrences}x in {g.file} (expected exactly 1). "
        f"If the guard moved, update the registry; if it was deleted, THAT is the bug this catches."
    )


@pytest.mark.parametrize("gid", list(_BY_ID))
def test_killing_test_exists_and_targets_the_guard(gid):
    """The named test must exist in the named file. A guard whose killing test was deleted is a guard
    that can silently die next."""
    g = _BY_ID[gid]
    test_path = REPO_ROOT / g.killed_by
    assert test_path.exists(), f"{gid}: killing-test file {g.killed_by} does not exist"
    body = test_path.read_text()
    assert f"def {g.kills_test_id}" in body, (
        f"{gid}: {g.killed_by} has no test named {g.kills_test_id!r} — a renamed/deleted killing test"
    )


def test_every_guard_field_is_populated():
    """A half-filled entry is a silent hole — every field is load-bearing."""
    for g in GUARDS:
        for field in ("id", "sprint", "file", "anchor", "protects", "killed_by", "kills_test_id"):
            assert getattr(g, field), f"{g.id}: empty {field}"
        assert g.file.startswith("prsm/"), f"{g.id}: file should be a repo-relative prsm/ path"
        assert g.killed_by.startswith("tests/"), f"{g.id}: killed_by should be a tests/ path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
