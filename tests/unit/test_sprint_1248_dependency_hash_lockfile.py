"""Sprint 1248 — dependency hash-pinned lockfile (supply-chain audit #8).

requirements.txt declares dependencies with RANGE specifiers (`fastapi>=0.104.0`), so
`pip install -r requirements.txt` resolves whatever satisfies the range at install
time and verifies NO hashes — a dependency-confusion / typosquat / compromised-mirror
substitution (including in a transitive dep) is installed silently.

The fix is an ADDITIVE, opt-in lockfile: requirements.lock pins every package in the
fully-resolved transitive tree to an exact version + sha256 hash(es), generated with
`uv pip compile requirements.txt --universal --generate-hashes`. Production / CI can
then `pip install --require-hashes -r requirements.lock` for a verified, reproducible
install. The default `pip install -r requirements.txt` flow is unchanged (non-disruptive);
adopting the lock as the enforced install path is the operator's clean-rebuild step.

This validates the lockfile stays hash-complete and in sync with requirements.txt — if
someone adds a top-level dependency without regenerating the lock, this fails. (It was
validated to install cleanly under `--require-hashes` in a fresh venv when shipped.)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQS = _REPO_ROOT / "requirements.txt"
_LOCK = _REPO_ROOT / "requirements.lock"


def _normalize(name: str) -> str:
    """PEP 503 name normalization: lowercase, runs of -_. → single -."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _top_level_requirements() -> list[str]:
    """Normalized package names declared in requirements.txt (extras/specifiers/markers stripped)."""
    names = []
    for raw in _REQS.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # strip env marker, then extras, then version specifier
        line = line.split(";", 1)[0]
        line = re.sub(r"\[.*?\]", "", line)
        m = re.match(r"^[A-Za-z0-9._-]+", line)
        if m:
            names.append(_normalize(m.group(0)))
    return names


def _locked_names() -> set[str]:
    names = set()
    for line in _LOCK.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9._-]+)==", line)
        if m:
            names.add(_normalize(m.group(1)))
    return names


def test_lockfile_exists():
    assert _LOCK.is_file(), "requirements.lock missing — regenerate with uv pip compile --generate-hashes"


def test_every_pin_is_hashed():
    """Every `pkg==version` pin in the lock must carry at least one sha256 hash."""
    text = _LOCK.read_text()
    # split into per-package blocks at each `name==` at column 0
    blocks = re.split(r"\n(?=[A-Za-z0-9._-]+==)", text)
    unhashed = []
    for block in blocks:
        m = re.match(r"^([A-Za-z0-9._-]+==[^\s\\]+)", block)
        if not m:
            continue
        if "--hash=sha256:" not in block:
            unhashed.append(m.group(1))
    assert not unhashed, f"lockfile pins without a sha256 hash (supply-chain #8 regression): {unhashed}"


def test_lock_covers_every_top_level_requirement():
    """Every top-level requirement must appear in the lock (lock stays in sync with requirements.txt)."""
    locked = _locked_names()
    missing = [n for n in _top_level_requirements() if n not in locked]
    assert not missing, (
        f"requirements.txt deps absent from requirements.lock (regenerate the lock): {missing}"
    )


def test_lock_is_fully_pinned_not_ranged():
    """The lock must use == exact pins, never range specifiers."""
    ranged = [
        line.strip()
        for line in _LOCK.read_text().splitlines()
        if re.match(r"^[A-Za-z0-9._-]+\s*(>=|<=|~=|>|<|!=)", line)
    ]
    assert not ranged, f"lockfile contains non-exact (ranged) pins: {ranged[:5]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
