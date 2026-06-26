"""Sprint 1287 — GitHub Actions supply-chain pin hygiene (supply-chain audit #9).

A workflow that does `uses: <third-party>@master` runs whatever that branch HEAD is
*at run time*, inside CI, with the job's secrets/permissions. A compromised or
retagged upstream action therefore executes in our pipeline. The fix (sp1287) pins
every third-party action to an immutable 40-hex commit SHA (with a `# version`
comment so Dependabot — already configured for the github-actions ecosystem — can
propose bumps).

This is the regression guard: a mutable BRANCH ref must never reappear, and the
specific third-party actions sp1287 pinned must stay SHA-pinned.

(Sibling supply-chain items: #7 AWS-bootstrap binary checksum, #8 dependency
hash-lockfile remain — see the audit topic memory.)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"

# A `uses:` value, capturing the ref after the final `@` (ignoring any trailing
# `# comment`). Matches both `- uses: x@y` and `uses: x@y` forms.
_USES_RE = re.compile(r"""uses:\s*['"]?(?P<action>[^\s'"@]+)@(?P<ref>[^\s'"#]+)""")

# Branch refs are the egregious case — they float with every upstream push.
_BRANCH_REFS = {"master", "main", "develop", "latest", "HEAD"}

# `release/v1`-style branch refs (a `/` in the ref and not a SHA) are also mutable.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# First-party GitHub-owned actions are accepted at major-version tags (the
# community-standard posture; GitHub controls these orgs). Everyone else must
# be SHA-pinned.
_FIRST_PARTY_OWNERS = {"actions", "github"}

# The specific third-party actions sp1287 pinned — assert they stay pinned.
_MUST_BE_SHA_PINNED = {
    "aquasecurity/trivy-action",
    "orhun/git-cliff-action",
    "crytic/slither-action",
    "pypa/gh-action-pypi-publish",
}


def _iter_uses():
    """Yield (workflow_path, action, ref) for every `uses:` in every workflow."""
    assert _WORKFLOW_DIR.is_dir(), f"missing {_WORKFLOW_DIR}"
    for wf in sorted(_WORKFLOW_DIR.glob("*.yml")) + sorted(_WORKFLOW_DIR.glob("*.yaml")):
        for line in wf.read_text().splitlines():
            m = _USES_RE.search(line)
            if not m:
                continue
            action = m.group("action")
            # Skip local (`./...`) and docker (`docker://...`) refs — not registry actions.
            if action.startswith(".") or action.startswith("docker://"):
                continue
            yield wf.name, action, m.group("ref")


def test_no_action_uses_a_mutable_branch_ref():
    """No workflow may pin ANY action to a floating branch (master/main/...)."""
    offenders = [
        f"{wf}: {action}@{ref}"
        for wf, action, ref in _iter_uses()
        if ref in _BRANCH_REFS
    ]
    assert not offenders, (
        "mutable branch refs found (supply-chain risk — pin to a SHA):\n  "
        + "\n  ".join(offenders)
    )


def test_third_party_actions_with_slash_refs_are_sha_pinned():
    """A ref containing `/` (e.g. release/v1) is a branch — must be SHA-pinned for third parties."""
    offenders = []
    for wf, action, ref in _iter_uses():
        owner = action.split("/", 1)[0]
        if owner in _FIRST_PARTY_OWNERS:
            continue
        if "/" in ref and not _SHA_RE.match(ref):
            offenders.append(f"{wf}: {action}@{ref}")
    assert not offenders, (
        "third-party actions on a branch-style ref (pin to a SHA):\n  "
        + "\n  ".join(offenders)
    )


def test_sp1287_pinned_actions_stay_sha_pinned():
    """The four third-party actions sp1287 hardened must remain SHA-pinned everywhere they appear."""
    seen = {name: [] for name in _MUST_BE_SHA_PINNED}
    for wf, action, ref in _iter_uses():
        if action in _MUST_BE_SHA_PINNED:
            seen[action].append((wf, ref))
    for action, occurrences in seen.items():
        assert occurrences, f"expected {action} to be referenced by a workflow"
        for wf, ref in occurrences:
            assert _SHA_RE.match(ref), (
                f"{wf}: {action}@{ref} is not SHA-pinned (sp1287 regression)"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
