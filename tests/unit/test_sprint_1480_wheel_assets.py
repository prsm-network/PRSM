"""Sprint 1480 — the shipped wheel must contain every asset the code loads at runtime.

User-readiness re-assessment (wf_58eae474) found the #1 blocker for a real user is
that the DOCUMENTED install produces a broken node: pyproject sets
``include-package-data = false`` with a package-data allowlist that covered only
``ui_mockup/``, so ``pip install prsm-network`` shipped **zero** dashboard
templates, **zero** onboarding templates, **zero** ``.wasm`` executors and **zero**
model-catalog ``.yaml``. The node advertises the dashboard + wizard URLs in its own
startup log and both are dead on the wheel.

The defect class is invisible to the rest of the suite because CI runs from the
SOURCE TREE, where every asset is present. Only a built-wheel check catches it.
Two layers here:

  1. ``test_every_runtime_asset_is_matched_by_a_package_data_glob`` (fast, always
     runs) — walks the asset dirs the code actually loads and asserts a glob in
     ``[tool.setuptools.package-data]`` matches each file. This also catches a glob
     that matches NOTHING (the old ``skills/*.yaml`` matched no file at all — the
     real paths are ``skills/builtins/<name>/SKILL.yaml``).
  2. ``test_built_wheel_contains_runtime_assets`` (builds a real wheel) — the
     ground truth: no config reasoning, just the wheel's namelist.
"""
from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"

# Assets loaded at runtime by prsm/ code. Each entry: (repo-relative dir, glob,
# the code that loads it). If you add a runtime asset dir, add it here.
RUNTIME_ASSETS = [
    ("prsm/dashboard/templates", "*.html", "prsm/dashboard/app.py Jinja2Templates"),
    ("prsm/dashboard/static/css", "*.css", "prsm/node/api.py _DASHBOARD_STATIC mount"),
    ("prsm/dashboard/static/js", "*.js", "prsm/node/api.py _DASHBOARD_STATIC mount"),
    ("prsm/interface/api/templates", "*.html", "onboarding_router.py + docs_ui.py"),
    ("prsm/compute/wasm/binaries", "*.wasm", "prsm/compute/wasm executor"),
    ("prsm/core/config/models", "*.yaml", "model catalog / pricing / providers"),
    ("prsm/data", "*.yaml", "dedup thresholds"),
]


def _package_data_globs() -> list[str]:
    """The prsm = [...] entry of [tool.setuptools.package-data]."""
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import tomli as tomllib  # type: ignore
    data = tomllib.loads(PYPROJECT.read_text())
    globs = data["tool"]["setuptools"]["package-data"]["prsm"]
    assert globs, "package-data for prsm is empty"
    return globs


def _glob_matches(rel_posix: str, pattern: str) -> bool:
    """setuptools package-data glob semantics: ``*`` does NOT cross a path
    separator. ``fnmatch`` is WRONG here — it translates ``*`` to ``.*``, so
    ``interface/api/templates/*.html`` would appear to match the nested
    ``templates/onboarding/welcome.html`` when setuptools ships no such file.
    That false pass is exactly how the wizard templates were missed."""
    parts = []
    for ch in pattern:
        if ch == "*":
            parts.append("[^/]*")
        elif ch == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(ch))
    return re.fullmatch("".join(parts), rel_posix) is not None


def _runtime_asset_files() -> list[Path]:
    """RECURSIVE on purpose — nested asset dirs (templates/onboarding/*.html) are
    precisely the ones a top-level glob silently drops."""
    out: list[Path] = []
    for rel_dir, pattern, _why in RUNTIME_ASSETS:
        d = REPO / rel_dir
        if not d.is_dir():
            continue
        out.extend(sorted(
            p for p in d.rglob(pattern)
            if p.is_file() and "__pycache__" not in p.parts
        ))
    return out


def test_runtime_asset_dirs_exist_in_source():
    """Guard the guard: if an asset dir is renamed, this list must be updated or the
    wheel check silently degrades to asserting nothing."""
    missing = [d for d, _p, _w in RUNTIME_ASSETS if not (REPO / d).is_dir()]
    assert not missing, (
        f"runtime asset dirs vanished from the source tree: {missing}. "
        "Either the code no longer loads them (drop the entry) or they moved "
        "(update RUNTIME_ASSETS + the package-data globs)."
    )


def test_every_runtime_asset_is_matched_by_a_package_data_glob():
    """★ Every runtime-loaded asset must be matched by some package-data glob.
    Pre-fix this failed on the dashboard/onboarding/wasm/yaml trees."""
    globs = _package_data_globs()
    files = _runtime_asset_files()
    assert files, "no runtime assets discovered — the check would be vacuous"

    unmatched = []
    for f in files:
        rel = f.relative_to(REPO / "prsm").as_posix()   # globs are package-relative
        if not any(_glob_matches(rel, g) for g in globs):
            unmatched.append(rel)
    assert not unmatched, (
        "these runtime-loaded assets are NOT matched by any "
        "[tool.setuptools.package-data] glob, so `pip install` ships a broken "
        f"node: {unmatched}"
    )


def test_no_package_data_glob_matches_nothing():
    """A glob that matches no file is a silent lie (the old `skills/*.yaml`)."""
    globs = _package_data_globs()
    pkg = REPO / "prsm"
    all_rel = [p.relative_to(pkg).as_posix()
               for p in pkg.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    dead = [g for g in globs if not any(_glob_matches(r, g) for r in all_rel)]
    assert not dead, (
        f"package-data globs that match NO file (typo or moved asset): {dead}"
    )


@pytest.mark.slow
@pytest.mark.requires_real_subprocess
def test_built_wheel_contains_runtime_assets(tmp_path):
    """★ Ground truth: build a real wheel and inspect its namelist. This is the only
    check that survives a change to include-package-data / MANIFEST semantics."""
    try:
        import build  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("`build` not installed — cannot construct a wheel to inspect")

    # setuptools STAGES package-data into <repo>/build/lib*/ and REUSES that
    # staging on the next build. A stale staging dir from an earlier (correct)
    # build makes this test pass even when package-data is broken — verified:
    # reverting pyproject to the defective allowlist still produced a green
    # wheel until this cleanup was added. Remove the staging dir so the wheel is
    # built from the CURRENT config. (build/ is a generated artifact, not source.)
    import shutil
    for staging in REPO.glob("build/lib*"):
        shutil.rmtree(staging, ignore_errors=True)

    # Prefer --no-isolation (fast: reuses this env). It fails when a build
    # requirement (setuptools-scm) is absent, so fall back to an isolated build
    # rather than skipping — skipping here would silently disable the ONLY
    # ground-truth check of the defect this sprint exists to fix.
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(tmp_path), str(REPO)],
        capture_output=True, text=True, timeout=900,
    )
    if not list(tmp_path.glob("*.whl")):
        proc = subprocess.run(
            [sys.executable, "-m", "build", "--wheel",
             "--outdir", str(tmp_path), str(REPO)],
            capture_output=True, text=True, timeout=1800,
        )
    wheels = list(tmp_path.glob("*.whl"))
    if not wheels:
        pytest.skip(
            "wheel build unavailable in this env (no network for build "
            f"isolation?): {proc.stderr[-400:]}"
        )

    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())

    missing = []
    for f in _runtime_asset_files():
        rel = f.relative_to(REPO).as_posix()   # wheel stores prsm/<...>
        if rel not in names:
            missing.append(rel)
    assert not missing, (
        f"built wheel {wheels[0].name} is MISSING runtime assets (a "
        f"`pip install` user gets a broken node): {missing}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
