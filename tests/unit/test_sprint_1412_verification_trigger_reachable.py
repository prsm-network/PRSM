"""Sprint 1412 — the sp928 optimistic-verification defense was silently dead code in production.

sp1390 (commit be9528cc) added ``ComputeRequester.deliver_local_result``. Its diff anchored on the
``job._result_event.set()`` context line at the end of ``_on_job_result``'s success path, so the new
method was inserted BETWEEN that line and the method's tail. Everything below — the completion
``logger.info`` and the entire sp928 trigger::

    if self.sampler is not None and not job.verification_run \\
            and provider_id != self.identity.node_id and self.sampler.should_sample():
        asyncio.create_task(self._run_verification(...))

— ended up after ``deliver_local_result``'s ``return True``: unreachable.

The consequence is not cosmetic. sp928 is the OPTIMISTIC VERIFICATION defense against
"paid-on-signature-alone mis-pay": a valid signature proves WHO produced a result, not that any work
was done, so with probability ``sample_rate`` (5% by default, no opt-in flag) the requester
re-executes the job on an independent provider and penalizes the outlier. With the trigger dead, a
remote provider could return a well-formed fabricated result — NOT tagged ``source: "mock"``, so the
sp1408 guard does not catch it either — and be paid with zero chance of being sampled,
reputation-penalized, or routed a CONSENSUS_MISMATCH slash challenge.

Two suites failed for that whole time (test_sprint_928_compute_result_sampler) and were dismissed as
"pre-existing failures". They were the defense telling us it was switched off.

This file pins the trigger's REACHABILITY, and adds a structural guard for the whole class: no
statement may follow a return/raise/continue/break in the same block, anywhere in the core packages.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TERMINATORS = (ast.Return, ast.Raise, ast.Continue, ast.Break)

#: Packages where dead code after a terminator would silently disable real behavior.
_GUARDED_PACKAGES = ("prsm/node", "prsm/settlement", "prsm/compute", "prsm/economy")


def _unreachable_statements(path: pathlib.Path):
    """Yield (enclosing_name, terminator_line, dead_line, dead_kind) for every statement that
    directly follows a return/raise/continue/break in the SAME block."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block[:-1]):
                if isinstance(stmt, _TERMINATORS):
                    dead = block[i + 1]
                    yield (getattr(node, "name", type(node).__name__),
                           stmt.lineno, dead.lineno, type(dead).__name__)
                    break


@pytest.mark.parametrize("package", _GUARDED_PACKAGES)
def test_no_unreachable_code_after_a_terminator(package):
    """The structural guard. Against the pre-sp1412 tree this reports exactly:
    'DEAD CODE in deliver_local_result(): Return at line 686, unreachable Expr at line 688'."""
    findings = []
    for path in sorted((REPO_ROOT / package).rglob("*.py")):
        for name, term_line, dead_line, kind in _unreachable_statements(path):
            rel = path.relative_to(REPO_ROOT)
            findings.append(f"{rel}:{dead_line} — unreachable {kind} after a terminator "
                            f"at line {term_line}, in {name}()")
    assert not findings, "unreachable code (a whole feature can hide here):\n" + "\n".join(findings)


def test_the_sp928_verification_trigger_lives_inside_on_job_result():
    """Directly pins the regression: ``should_sample`` must be called from ``_on_job_result``,
    not from some method it was accidentally relocated into."""
    src = (REPO_ROOT / "prsm/node/compute_requester.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "ComputeRequester")

    def _calls_should_sample(fn: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Attribute) and n.attr == "should_sample"
            for n in ast.walk(fn)
        )

    callers = [fn.name for fn in cls.body
               if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
               and _calls_should_sample(fn)]

    assert callers == ["_on_job_result"], (
        f"the sp928 sampler trigger must fire from _on_job_result; found it in {callers}. "
        "If it moved, the optimistic-verification defense is inert on the pay path."
    )


def test_deliver_local_result_is_a_sibling_method_not_a_tail():
    """sp1390's method must be defined AFTER _on_job_result completes, so nothing of
    _on_job_result's body can be stranded behind its `return`."""
    tree = ast.parse((REPO_ROOT / "prsm/node/compute_requester.py").read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "ComputeRequester")
    fns = {fn.name: fn for fn in cls.body
           if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))}

    assert "deliver_local_result" in fns and "_on_job_result" in fns
    deliver = fns["deliver_local_result"]
    # its body must END at the return — nothing after it
    assert isinstance(deliver.body[-1], ast.Return), (
        "deliver_local_result has statements after its final return — the sp1390 bug shape"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
