"""sp1449 — the per-stage node-side gate binds the ON-CHAIN settled amount (batched_receipt.value_ftns)
to the requester-AUTHORIZED amount (share_wei).

The per-stage authorization commits to a payee set of (address, share_wei) and the gate
(verify_per_stage_authorization) enforces membership + the cumulative cap over share_wei. But the
amount that actually settles on-chain is batched_receipt.value_ftns (accumulate() sums value_ftns →
commitBatch(totalValueFTNS) → settleFromRequester). The honest splitter sets value_ftns = share_wei
(per_stage_settlement_split.py:486), so they are equal by construction — but nothing RE-ASSERTED it at
the gate. A task whose value_ftns was inflated above its authorized share_wei (a malformed/tampered
routed task) would pass the gate (share_wei still a valid member) yet settle MORE than the requester
authorized on-chain — over-drawing the requester's escrow and getting the committing (honest) node
challenged + slashed.

The adversarial per-stage-authz money audit (workflow wgktvg9wk) flagged this share_wei-vs-value_ftns
decoupling (one verifier confirmed high-confidence). This closes it: verify_routed_settlement_task now
rejects any task whose settled value_ftns != its authorized share_wei — giving the per-stage gate the
same "authorize exactly what settles" binding the single-payee sibling already has.

Money assertion — never weaken.
"""
from __future__ import annotations

import dataclasses

from prsm.settlement.per_stage_routing import verify_routed_settlement_task

# Reuse the fully-signed authorized staged-task fixture from the sp1446 suite; staged.task is the
# routable PerStageSettlementTask and staged.payees is the full signed payee set.
from tests.unit.test_sprint_1446_per_stage_double_settle_closed import _staged_task


def test_honest_task_value_equals_authorized_share_is_authorized(tmp_path):
    _store, staged = _staged_task(tmp_path)
    task = staged.task
    # Sanity: the honest splitter set value_ftns == share_wei.
    assert int(task.batched_receipt.value_ftns) == int(task.share_wei)
    v = verify_routed_settlement_task(task, payees=staged.payees)
    assert v.authorized, v.reason


def test_gate_rejects_value_ftns_exceeding_authorized_share(tmp_path):
    _store, staged = _staged_task(tmp_path)
    task = staged.task
    # Tamper: inflate the on-chain-settled value ABOVE the requester-authorized share. share_wei (what
    # the gate checks for membership/cap) is unchanged, so the OLD gate would still authorize it.
    inflated = dataclasses.replace(
        task.batched_receipt, value_ftns=int(task.share_wei) + 10 ** 18)
    tampered = dataclasses.replace(task, batched_receipt=inflated)
    v = verify_routed_settlement_task(tampered, payees=staged.payees)
    assert not v.authorized, (
        "gate authorized a task whose on-chain settled value_ftns EXCEEDS the requester-authorized "
        "share_wei — the node would over-draw the requester's escrow and be slashed")
    assert "value" in v.reason.lower() or "share" in v.reason.lower(), v.reason


def test_gate_rejects_value_ftns_below_authorized_share(tmp_path):
    """Also reject an UNDER-value (settles less than authorized) — the gate authorizes EXACTLY the
    amount that settles, in either direction, so the authorization means what it says."""
    _store, staged = _staged_task(tmp_path)
    task = staged.task
    deflated = dataclasses.replace(
        task.batched_receipt, value_ftns=max(0, int(task.share_wei) - 1))
    tampered = dataclasses.replace(task, batched_receipt=deflated)
    v = verify_routed_settlement_task(tampered, payees=staged.payees)
    assert not v.authorized, "gate authorized a task whose settled value_ftns != authorized share_wei"
