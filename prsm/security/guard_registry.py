"""Sprint 1415 — the load-bearing money/security guard registry.

Today's session found FOUR defenses that existed, were wired, and yet could never fire:
sp1412 (the anti-fabrication sampler pushed after a `return`), sp1178 (PCU-weighting inert on a
0-default), sp1172 (a payment-authorization primitive with zero non-test callers), sp1411 (a slash
alarm whose condition the real event never met). The common shape: **a guard whose deletion would
break no test.** A comment cannot prevent that; a registry can.

Each entry below names a guard that protects money or security, the exact anchor line that IS the
guard, and the test that MUST go red if the guard is deleted. `test_guard_registry.py` asserts every
anchor is still present at its cited location — so silently deleting or moving a guard trips CI — and
that every named killing test exists. `verify` (the make/CI target) goes further: it deletes each
anchor and requires the killing test to actually fail, the deletion-gate the memo recommended.

This is a CURATED set, not every guard in the tree. Add an entry when you ship a money/security guard
whose silent removal would be catastrophic and invisible. The bar: "if someone deleted this line in a
refactor, would any existing test notice?" If the honest answer is no, it belongs here with a test
that makes the answer yes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Guard:
    id: str                 # stable identifier
    sprint: str             # where it shipped
    file: str               # repo-relative path
    anchor: str             # a substring that uniquely identifies the guard line; must stay present
    protects: str           # one line: what its absence would allow
    killed_by: str          # test file (tests/unit/...) that fails if the anchor is removed
    kills_test_id: str      # a specific test in that file that targets this guard


#: The load-bearing guards. Ordered by subsystem. Anchors are matched as exact substrings of a
#: single source line, so keep them specific enough to be unique within their file.
GUARDS: List[Guard] = [
    Guard(
        id="requester-no-pay-for-mock",
        sprint="sp1408",
        file="prsm/node/compute_requester.py",
        anchor='and result.get("source") == "mock"',
        protects="paying a remote provider for a self-declared fabricated (mock) result",
        killed_by="tests/unit/test_sprint_1408_no_pay_for_mock_result.py",
        kills_test_id="test_signed_remote_mock_result_is_never_paid",
    ),
    Guard(
        id="requester-signature-reject-gate",
        sprint="sp924/sp1416",
        file="prsm/node/compute_requester.py",
        anchor="if not verified and provider_id != self.identity.node_id:",
        protects="paying a remote provider for a result whose signature does not verify under the "
                 "accepted provider's key (an unsigned or forged result from the right ids)",
        killed_by="tests/unit/test_sprint_1416_live_defense_killing_tests.py",
        kills_test_id="test_unsigned_result_from_accepted_provider_is_rejected",
    ),
    Guard(
        id="requester-verification-sampler-trigger",
        sprint="sp1412",
        file="prsm/node/compute_requester.py",
        anchor="and self.sampler.should_sample()",
        protects="the sp928 optimistic-verification defense against paid-on-signature-alone mis-pay "
                 "(the trigger was dead code after a misplaced return)",
        killed_by="tests/unit/test_sprint_1412_verification_trigger_reachable.py",
        kills_test_id="test_the_sp928_verification_trigger_lives_inside_on_job_result",
    ),
    Guard(
        id="requester-wrong-payee-guard",
        sprint="sp924",
        file="prsm/node/compute_requester.py",
        anchor="if job.provider_id and provider_id != job.provider_id:",
        protects="redirecting a job's payment to a third party who published a JOB_RESULT signed "
                 "with their own key",
        killed_by="tests/unit/test_sprint_924_job_result_payment_guards.py",
        kills_test_id="test_wrong_payee_attacker_result_is_rejected",
    ),
    Guard(
        id="settlement-intent-wins-on-restore",
        sprint="sp1409",
        file="prsm/settlement/client.py",
        anchor="if key in intent_keys:",
        protects="double-settle: restoring a pending batch a commit intent already owns lets the "
                 "commit phase re-commit those receipts as a second on-chain batchId",
        killed_by="tests/unit/test_sprint_1409_accumulator_durability.py",
        kills_test_id="test_restore_skips_a_batch_owned_by_a_commit_intent",
    ),
    Guard(
        id="autodefense-classify-by-reason",
        sprint="sp1411",
        file="prsm/settlement/challenge_auto_defense.py",
        anchor="if code in SLASHING_REASONS:",
        protects="the operator's slash alarm firing on DOUBLE_SPEND / CONSENSUS_MISMATCH (a "
                 "validly-signed receipt verifies, so a receipt_ok-based verdict never fired)",
        killed_by="tests/unit/test_sprint_1411_auto_defense_reason_code.py",
        kills_test_id="test_double_spend_with_a_verifying_receipt_alerts_as_slashing",
    ),
    Guard(
        id="escrow-per-job-serialization-lock",
        sprint="sp907/sp1416",
        file="prsm/node/payment_escrow.py",
        anchor="return self._job_locks.setdefault(job_id, asyncio.Lock())",
        protects="concurrent release/refund/split on one escrow both paying (the escrow wallet goes "
                 "negative — FTNS minted from nothing) — the per-job lock must be the SAME object per "
                 "job_id, so this setdefault is the lock-identity primitive",
        killed_by="tests/unit/test_sprint_1416_live_defense_killing_tests.py",
        kills_test_id="test_concurrent_release_pays_the_provider_exactly_once",
    ),
    Guard(
        id="paid-unlock-fee-guard-invoked",
        sprint="sp1361/sp1417",
        file="prsm/sdk/client.py",
        anchor="assert_fee_matches_deposit(key_client, ch, int(fee_wei))",
        protects="pay_and_unlock_content paying a non-refundable fee that doesn't match the on-chain "
                 "deposit (pure buyer fund loss) — the pure function is tested by sp1361, but this "
                 "is the CALL SITE: deleting it silently bypasses the guard on the live path",
        killed_by="tests/unit/test_sprint_1417_paid_unlock_guard_wiring.py",
        kills_test_id="test_pay_and_unlock_invokes_the_fee_guard",
    ),
    Guard(
        id="paid-unlock-squat-guard-invoked",
        sprint="sp1365/sp1417",
        file="prsm/sdk/client.py",
        anchor="assert_publisher_controls_payee(key_client, creator_reader, ch)",
        protects="pay_and_unlock_content paying a squatter (fee payee != key depositor, so the fee "
                 "can't reach whoever can unlock) — the CALL SITE for the sp1365 guard, whose "
                 "invocation the sp1417 _creator_reader seam makes testable",
        killed_by="tests/unit/test_sprint_1417_paid_unlock_guard_wiring.py",
        kills_test_id="test_pay_and_unlock_invokes_the_squat_guard",
    ),
    Guard(
        id="discovery-capability-announce-cap",
        sprint="sp1414",
        file="prsm/node/discovery.py",
        anchor="dropping new capability announce",
        protects="unbounded known_peers growth (memory DoS + eclipse-by-magnitude) via an "
                 "authenticated capability-announce flood",
        killed_by="tests/unit/test_sprint_1414_capability_announce_cap.py",
        kills_test_id="test_capability_announce_respects_the_known_peers_cap",
    ),
]


def guard_ids() -> List[str]:
    return [g.id for g in GUARDS]
