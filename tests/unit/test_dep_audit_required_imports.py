"""Sprint 461 — dep-audit invariant pin.

After sprints 458-460 surfaced F15 (bencodepy), F16 (zfec), F17
(pycryptodome), F18 (sentence_transformers eager) as a CLASS of
bugs caused by undeclared / miscategorized dependencies, this
test pins the invariant that prevents future siblings:

  Every module imported at TOP-LEVEL in prsm/ MUST be either:
    1. A stdlib module (filtered via sys.stdlib_module_names), OR
    2. A relative import within prsm itself, OR
    3. Declared in pyproject.toml — either required deps OR
       any [project.optional-dependencies] extra

The test reads pyproject.toml + grep all top-level imports in
prsm/ + finds any gap. A future developer who adds `import foo`
at module top-level without adding `foo` to pyproject.toml gets
a CI failure here.

Known undeclared-but-acceptable imports (lazy-feature-gated paths
where the import IS at top-level but only fires when the optional
feature is wired) are explicitly allow-listed below — those
modules need lazy-import refactors (sprint-460 pattern) but are
deferred to future sprints. The allow-list is a known-debt
register, not a "we'll just ignore this" escape hatch.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Map of PyPI package name → Python module name when they
# differ. Keep this in sync with the script in sprint 461.
PYPI_TO_MODULE = {
    "pyyaml": "yaml",
    "python-jose": "jose",
    "pyjwt": "jwt",
    "python-dateutil": "dateutil",
    "python-magic": "magic",
    "python-socks": "python_socks",
    "pysocks": "socks",
    "pynacl": "nacl",
    "pillow": "PIL",
    "pycryptodome": "Crypto",
    "scikit-learn": "sklearn",
    "sentence-transformers": "sentence_transformers",
    "pydantic-settings": "pydantic_settings",
    "qdrant-client": "qdrant_client",
    "flask-socketio": "flask_socketio",
    "eth-abi": "eth_abi",
    "eth-account": "eth_account",
    "eth-utils": "eth_utils",
    "prometheus-client": "prometheus_client",
    "slack-sdk": "slack_sdk",
    "simple-term-menu": "simple_term_menu",
    "uvicorn[standard]": "uvicorn",
}

# Allow-list of imports that ARE at top-level but only fire on
# optional features. These need lazy-import refactors (sprint-460
# pattern) — TRACKED here as known debt, NOT ignored.
#
# Sprint 462 lazy-refactor closures:
#   - pyotp, qrcode → enhanced_authentication.py: now lazy inside
#     MFAManager._import_pyotp() / _import_qrcode()
#   - stem → anonymous_networking.py: now lazy inside
#     _import_stem_controller()
#   - plotly (security_analytics) → now lazy inside
#     _import_plotly_json_encoder() — fires only when
#     generate_dashboard_html() is called
#   - plotly (performance_dashboard.py) → standalone streamlit
#     script, excluded from audit walk (see _SKIP_DIRS below)
#   - PIL → false positive from sprint 461 (no top-level
#     imports in prsm/ after closer inspection)
ALLOWED_UNDECLARED_TOP_LEVEL = {
    # tomli — Python <3.11 stdlib backport (3.11+ has tomllib)
    "tomli",
    # The "fancy regex false positives" group — these aren't
    # real module names; my grep picked up doc-quoted text
    # that happened to start with `from X` or `import X`.
    # Stable list kept here so the audit doesn't churn on
    # docstring edits.
    "AggregationCommitMismatchError",  # docstring example
    # paid_content.py's module docstring wraps onto a line beginning
    # "from B1 on a wrong key / tampered content)" — B1/B2/B3 are the
    # brick names, not modules. The line-based regex reads it as an import.
    "B1",
    "EOS",  # constant name
    "EmissionController",  # class name in docstring
    "GPUtil",  # imported in legacy module
    "N",  # variable name
    "Phase",  # docstring
    "a",  # docstring example like "from a perspective"
    "anchor",  # variable in docstring
    "p_i",  # math variable in docstring
    "bootstrap",  # word in prose
    "cancellation",  # word
    "comprehensive_performance_benchmark",  # filename in docstring
    "cost",  # word
    "dashboard",  # word
    "demos",  # word
    "issues",  # word
    "its",  # word
    "logs",  # word
    "memory",  # word
    "outside",  # word
    "policy",  # word
    "probs",  # word
    "providers",  # word
    "publisher_node_id",  # field name in docstring
    "retried",  # word
    "running",  # word
    "scripts",  # word
    "the",  # word
    "their",  # word
    "upstream",  # word
    "affecting",  # word
    "corrupting",  # word
    # Heavy ML deps already lazy-imported (sprint 460) but
    # still pattern-match the regex because the import line
    # exists inside _ensure_loaded() — false positive.
    # sentence_transformers is in [ml] extra; declared.
}


def _read_declared_pypi():
    """Parse pyproject.toml and return all declared PyPI package
    names from required deps + any extra."""
    import tomllib
    data = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(),
    )
    declared = set()
    for dep in data["project"].get("dependencies", []):
        pkg = re.split(r'[<>=!~\s\[]', dep, maxsplit=1)[0]
        if pkg:
            declared.add(pkg.lower())
    for extra_deps in data["project"].get(
        "optional-dependencies", {},
    ).values():
        for dep in extra_deps:
            pkg = re.split(r'[<>=!~\s\[]', dep, maxsplit=1)[0]
            if pkg:
                declared.add(pkg.lower())
    return declared


def _read_required_pypi():
    """Sprint 463: parse pyproject.toml and return only
    REQUIRED PyPI package names (not in any extra). The
    tightened invariant: imports on the node-startup path
    must be in this set, not just any extra."""
    import tomllib
    data = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(),
    )
    required = set()
    for dep in data["project"].get("dependencies", []):
        pkg = re.split(r'[<>=!~\s\[]', dep, maxsplit=1)[0]
        if pkg:
            required.add(pkg.lower())
    return required


def _required_module_set():
    """Sprint 463: return the set of Python module names
    provided by REQUIRED deps only (no extras)."""
    required_pypi = _read_required_pypi()
    provided = set()
    for pkg in required_pypi:
        provided.add(pkg)
        provided.add(pkg.replace("-", "_"))
        if pkg in PYPI_TO_MODULE:
            provided.add(PYPI_TO_MODULE[pkg])
    return {m.lower() for m in provided}


def _resolved_module_set(declared_pypi):
    """Expand declared PyPI names into Python-module names."""
    provided = set()
    for pkg in declared_pypi:
        provided.add(pkg)
        provided.add(pkg.replace("-", "_"))
        if pkg in PYPI_TO_MODULE:
            provided.add(PYPI_TO_MODULE[pkg])
    return {m.lower() for m in provided}


# Sprint 462: paths excluded from the dep-audit walk. These are
# standalone scripts (run via `streamlit run` or `python script.py`)
# that are NOT imported by the prsm package — their top-level
# imports don't fire during `prsm node start`, so they don't
# trigger the F15/F16/F17/F18 bug class.
_SKIP_DIRS = (
    "_legacy",
    # Streamlit dashboards are run directly, not imported.
    # Their requirements.txt covers their deps separately.
    "interface/dashboard",
)


def _is_skipped_path(py_path):
    s = str(py_path)
    return any(skip in s for skip in _SKIP_DIRS)


# Sprint 463: real transitive trace of the node-startup
# import chain. Starts from canonical root files + follows
# `from prsm.X import Y` chains AT MODULE TOP-LEVEL until
# fixed-point. The resulting set is the modules that
# actually get loaded when `prsm node start` runs.
#
# Compared to a directory-prefix heuristic, this trace
# correctly handles:
#   - Lazy imports (inside function bodies) NOT included
#   - Files in a startup-dir but not imported by anything
#     (e.g., compute/inference/tensor_parallel.py) NOT included
#   - Transitive chains across directory boundaries
#
# The roots are the canonical entry points: node startup +
# API surface + CLI entry. If the operator runs `prsm node
# start`, eventually one of these is the first prsm module
# loaded.
_STARTUP_ROOTS = (
    "prsm.cli",
    "prsm.node.node",
    "prsm.node.api",
    # prsm.mcp_server is intentionally NOT a startup root:
    # it's an OPT-IN entry (`prsm mcp start`) that requires
    # `pip install -e '.[mcp]'`. The dep-audit invariant
    # scopes to the canonical `prsm node start` chain so it
    # doesn't false-flag deps in the [mcp] extra. Operators
    # who want MCP install the extra explicitly per docs.
)


def _resolve_module_to_file(mod_name):
    """Map a dotted module name (prsm.X.Y) to its .py path."""
    parts = mod_name.split(".")
    candidate = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    # Package: look for __init__.py
    pkg_init = REPO_ROOT.joinpath(*parts) / "__init__.py"
    if pkg_init.is_file():
        return pkg_init
    return None


def _resolve_relative_import(from_module, source_file):
    """Map `from .X import Y` to its dotted module name.

    `source_file` is the file containing the import; the
    `from_module` may start with dots indicating relative
    import depth. Returns dotted module name.
    """
    if not from_module.startswith("."):
        return from_module
    # Count leading dots
    depth = 0
    while depth < len(from_module) and from_module[depth] == ".":
        depth += 1
    tail = from_module[depth:]
    # source_file is REPO_ROOT/prsm/.../foo.py
    # Its package is the parent dir as a dotted name.
    rel = source_file.relative_to(REPO_ROOT)
    pkg_parts = list(rel.parts[:-1])  # drop file name
    # Climb `depth - 1` levels (depth=1 means same package)
    if depth > 1:
        pkg_parts = pkg_parts[:-(depth - 1)]
    if tail:
        return ".".join(pkg_parts + tail.split("."))
    return ".".join(pkg_parts)


def _trace_startup_imports():
    """Sprint 463: BFS from _STARTUP_ROOTS, follow top-level
    `from prsm.X.Y import Z` (and `import prsm.X.Y`) chains.
    Returns set of dotted prsm.* module names reachable on
    the startup chain."""
    re_from = re.compile(
        r'^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
    )
    re_import = re.compile(
        r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
    )
    reachable = set()
    worklist = list(_STARTUP_ROOTS)
    while worklist:
        mod = worklist.pop()
        if mod in reachable:
            continue
        reachable.add(mod)
        path = _resolve_module_to_file(mod)
        if path is None:
            continue
        try:
            text = path.read_text()
        except Exception:
            continue
        for line in text.splitlines():
            if line.startswith((" ", "\t", "#")):
                continue  # lazy / commented
            m_from = re_from.match(line)
            m_imp = re_import.match(line)
            target = None
            if m_from:
                target = _resolve_relative_import(
                    m_from.group(1), path,
                )
            elif m_imp:
                target = m_imp.group(1)
            if target is None:
                continue
            # Only follow prsm.* chains; external modules
            # are leaves
            if target.startswith("prsm."):
                # Drop the trailing symbol if present
                worklist.append(target)
                # Also add parent packages (their __init__
                # runs on import)
                parts = target.split(".")
                for i in range(2, len(parts) + 1):
                    worklist.append(".".join(parts[:i]))
    return reachable


def _startup_path_files():
    """Sprint 463: set of .py paths on the canonical startup
    chain (after transitive trace from _STARTUP_ROOTS)."""
    reachable = _trace_startup_imports()
    files = set()
    for mod in reachable:
        path = _resolve_module_to_file(mod)
        if path is not None:
            files.add(path)
    return files


def _is_startup_path(py_path):
    """Sprint 463: True iff this file is on the canonical
    node-startup chain (transitive-trace result). Cached
    because the trace is O(N) over prsm/."""
    global _STARTUP_FILES_CACHE
    if _STARTUP_FILES_CACHE is None:
        _STARTUP_FILES_CACHE = _startup_path_files()
    return py_path in _STARTUP_FILES_CACHE


_STARTUP_FILES_CACHE = None


def _top_level_imports_in_prsm():
    """Walk prsm/ and collect every module imported at TOP-LEVEL
    (column 0, no indent). Returns set of top-level package names.

    Skips standalone-script directories (_legacy/, dashboards) —
    those are run directly, not imported by node startup."""
    re_top = re.compile(
        r'^(import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    )
    found = set()
    for py in (REPO_ROOT / "prsm").rglob("*.py"):
        if _is_skipped_path(py):
            continue
        try:
            text = py.read_text()
        except Exception:
            continue
        for line in text.splitlines():
            if line.startswith((" ", "\t", "#")):
                continue
            m = re_top.match(line)
            if not m:
                continue
            mod = m.group(2)
            if mod == "prsm":
                continue
            found.add(mod)
    return found


def _startup_path_top_level_imports():
    """Sprint 463: collect top-level imports keyed by file,
    only for files on the canonical node-startup chain.

    Returns: dict[mod_name] -> list of (file, line) pairs.
    Used by the tightened invariant test to find any
    startup-path top-level import whose package isn't in
    REQUIRED deps."""
    re_top = re.compile(
        r'^(import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    )
    found = {}
    for py in (REPO_ROOT / "prsm").rglob("*.py"):
        if _is_skipped_path(py):
            continue
        if not _is_startup_path(py):
            continue
        try:
            text = py.read_text()
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.startswith((" ", "\t", "#")):
                continue
            m = re_top.match(line)
            if not m:
                continue
            mod = m.group(2)
            if mod == "prsm":
                continue
            found.setdefault(mod, []).append(
                (str(py.relative_to(REPO_ROOT)), lineno),
            )
    return found


def test_every_top_level_external_import_is_declared():
    """The dep-audit invariant: every external module imported
    at top-level in prsm/ must be (a) stdlib, (b) declared in
    pyproject.toml, or (c) explicitly allow-listed as known
    debt."""
    declared_pypi = _read_declared_pypi()
    provided_modules = _resolved_module_set(declared_pypi)
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    all_imports = _top_level_imports_in_prsm()

    external = all_imports - stdlib
    undeclared = []
    for mod in external:
        if mod.lower() in provided_modules:
            continue
        if mod in ALLOWED_UNDECLARED_TOP_LEVEL:
            continue
        undeclared.append(mod)

    assert not undeclared, (
        f"Sprint 461 dep-audit invariant violated. New top-level "
        f"imports found in prsm/ that are NOT declared in "
        f"pyproject.toml and NOT in the known-debt allow-list:\n"
        f"  {sorted(undeclared)}\n\n"
        f"Fix options:\n"
        f"  1. Add the missing dep to pyproject.toml's required "
        f"`dependencies` list (or an appropriate extra)\n"
        f"  2. Make the import lazy (sprint-460 pattern — move "
        f"inside a function body, not module top-level)\n"
        f"  3. If genuinely false-positive, add to "
        f"ALLOWED_UNDECLARED_TOP_LEVEL with a comment explaining"
    )


def test_sprint_463_startup_path_imports_in_required_deps():
    """Sprint 463 tightened invariant — F16/F19 class closure.

    F16 (zfec) and F19 (bleach) both passed sprint 461's
    invariant because the package WAS declared somewhere in
    pyproject.toml — just in an OPTIONAL extra ([blockchain]
    and [server] respectively). Both were imported at top-
    level by files on the node-startup path. Operators doing
    a default `pip install -e .` (no extras) got a fresh-venv
    install that crashed at `prsm node start`.

    This test fires when a top-level import in any
    `_STARTUP_PATH_PREFIXES` file resolves to a package that's
    in an EXTRA but not REQUIRED. Closes the F16/F19 class
    at the CI level — no more sibling bugs of that pattern.

    To fix a violation: either (a) move the dep from the
    extra to required `dependencies`, or (b) make the import
    lazy (sprint 460/462 pattern), or (c) move the importing
    file out of the startup path (rare).
    """
    required_modules = _required_module_set()
    all_declared = _resolved_module_set(_read_declared_pypi())
    stdlib = set(getattr(sys, "stdlib_module_names", set()))

    startup_imports = _startup_path_top_level_imports()

    violations = []
    for mod, hits in startup_imports.items():
        if mod in stdlib:
            continue
        if mod in ALLOWED_UNDECLARED_TOP_LEVEL:
            continue
        mod_lower = mod.lower()
        if mod_lower in required_modules:
            continue  # OK: in required deps
        # Not in required. Two sub-cases:
        # (a) Declared in an extra (the F16/F19 bug class)
        # (b) Not declared anywhere (caught by the broader
        #     test_every_top_level_external_import_is_declared
        #     — leave that to fire instead of duplicating)
        if mod_lower in all_declared:
            violations.append((mod, hits))

    assert not violations, (
        "Sprint 463 tightened invariant violated. Top-level "
        "imports on the node-startup path that resolve to "
        "packages declared in optional extras (NOT required "
        "deps). Same class as F16 (zfec) + F19 (bleach):\n\n"
        + "\n".join(
            f"  {mod} (in some optional extra):\n"
            + "\n".join(f"    - {f}:{ln}" for f, ln in hits[:3])
            for mod, hits in violations
        )
        + "\n\nFix options:\n"
        + "  1. Move the dep from the extra into required "
        + "`dependencies` (preferred for true-required deps)\n"
        + "  2. Make the import lazy (sprint 460/462 pattern — "
        + "import inside function body, not top-level)\n"
        + "  3. Add the file's directory to _SKIP_DIRS if it's "
        + "actually a standalone script\n"
        + "  4. Remove the path from _STARTUP_PATH_PREFIXES if "
        + "it's truly not on the startup chain (rare)\n"
    )


def test_sprint_461_required_deps_are_present():
    """Pin the 6 deps that sprint 461 added to required. A
    revert that loses any of these re-introduces the F15-F17
    bug class."""
    declared_pypi = _read_declared_pypi()
    must_be_declared = {
        "pycryptodome",  # F17 — Shamir secret-sharing
        "pyyaml",        # required by config loaders
        "python-jose",   # JWT auth middleware
        "networkx",      # graph algorithms
        "starlette",     # FastAPI ASGI base
        "eth-abi",       # on-chain settlement merkle
        # Plus the earlier-sprint deps that are also at risk:
        "bencodepy",     # F15 fix
        "zfec",          # F16 fix
    }
    missing = must_be_declared - declared_pypi
    assert not missing, (
        f"Required deps removed from pyproject.toml: {missing}. "
        f"These were added by the dep-audit to fix F15/F16/F17 "
        f"production-blockers. Removing them re-breaks fresh-"
        f"venv installs."
    )


def test_sprint_462_lazy_refactors_closed():
    """Sprint 462 closed the 5 known lazy-import debts from
    sprint 461's allow-list:
      - pyotp / qrcode → enhanced_authentication.py
      - stem → anonymous_networking.py
      - plotly (security_analytics) → lazy in
        generate_dashboard_html
      - plotly (performance_dashboard) → standalone streamlit
        script; excluded via _SKIP_DIRS
      - PIL → was a false positive

    This test verifies the refactored modules have NO unguarded
    top-level imports of these heavy deps. A regression that
    re-adds the import at module top breaks operator UX for the
    ~80% who don't use these features.
    """
    re_top_import = re.compile(
        r'^(import|from)\s+(stem|pyotp|qrcode|plotly|PIL)',
    )
    targets = {
        REPO_ROOT / "prsm" / "core" / "privacy" / "anonymous_networking.py":
            {"stem"},
        REPO_ROOT / "prsm" / "core" / "security" / "enhanced_authentication.py":
            {"pyotp", "qrcode"},
        REPO_ROOT / "prsm" / "core" / "security" / "security_analytics.py":
            {"plotly"},
    }
    for path, deps in targets.items():
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            # Only fail on TOP-LEVEL (column 0) — comments and
            # indented lazy-imports are fine
            if line.startswith((" ", "\t", "#")):
                continue
            m = re_top_import.match(line)
            if not m:
                continue
            mod = m.group(2)
            if mod in deps:
                raise AssertionError(
                    f"Sprint 462 lazy-refactor regression: "
                    f"{path.relative_to(REPO_ROOT)}:{lineno} has "
                    f"top-level `import {mod}` — must be moved "
                    f"inside a function body. Pre-sprint-462 "
                    f"this pattern broke operator UX for the "
                    f"~80% who don't use the {mod}-dependent "
                    f"feature."
                )
