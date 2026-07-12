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
    Guard(
        id="ledger-reconciliation-tx-ids-on-default-ledger",
        sprint="sp1419",
        file="prsm/node/dag_ledger.py",
        # Anchor on the method BODY (not `async def get_recent_tx_ids`): the body line is the
        # distinctive, behaviour-bearing part. (A dead DAGLedgerAdapter used to define a rival
        # get_recent_tx_ids that made the signature non-unique and the default ledger look
        # covered; the adapter was removed in sp1427, but the body anchor remains the right pin.)
        anchor="history = await self.get_transaction_history(wallet_id, limit)",
        protects="balance reconciliation silently dying on EVERY default node. LedgerSync calls "
                 "self.ledger.get_recent_tx_ids() to build its balance proof, but Node builds a "
                 "bare DAGLedger by default (config.ledger_type='dag') and DAGLedger never had the "
                 "method — so every cycle raised AttributeError into a bare `except Exception: "
                 "logger.error(...)`. The node never sent a balance proof and never answered a "
                 "peer's balance_request: cross-node ledger divergence went undetected network-wide",
        killed_by="tests/unit/test_sprint_1419_reconciliation_dead_on_default_ledger.py",
        kills_test_id="test_reconciliation_sends_a_balance_request_on_the_default_dag_ledger",
    ),
    Guard(
        id="requester-records-issued-payment-authorizations",
        sprint="sp1421",
        file="prsm/sdk/client.py",
        anchor="def _issued_auth_store(self):",
        protects="the requester's ONLY defense against an escrow drain. On-chain, commitBatch("
                 "address requester, ...) takes the requester as a PLAIN, UNVERIFIED argument — no "
                 "signature, no authorization, NO BOND — and finalizeBatch (callable by anyone) "
                 "then drains that requester's escrow via settleFromRequester. The victim's sole "
                 "defense is a NO_ESCROW challenge, which _handleNoEscrow lets ONLY the victim "
                 "raise, and which requires proving 'I never authorized this batch' — i.e. it "
                 "requires knowing what you DID authorize. That is this store. Un-wire it and "
                 "the store goes empty, the matcher goes blind (it then flags even the "
                 "requester's OWN legitimate batches), and any address can drain any funded "
                 "escrow for the cost of gas",
        killed_by="tests/unit/test_sprint_1421_issued_auth_recording_wired.py",
        kills_test_id="test_forged_batch_is_flagged_unauthorized_and_the_real_one_is_not",
    ),
    Guard(
        id="onchain-slash-penalizes-reputation",
        sprint="sp1424",
        file="prsm/marketplace/slash_reputation_bridge.py",
        anchor="tracker.record_slash(",
        protects="a provider slashed on-chain for fraud (double-spend / forged signature) keeping "
                 "FULL aggregator-selection weight. The ReputationTracker is read on every "
                 "dispatch but was written never on the live path, so score_for() was a constant "
                 "0.5 and has_been_slashed() was False for everyone. This bridge is the only thing "
                 "that carries a real StakeBond.Slashed event into the tracker (mapped operator "
                 "eth-address -> node_id); without this call the slash never affects selection",
        killed_by="tests/unit/test_sprint_1424_slash_reputation_bridge.py",
        kills_test_id="test_a_slashed_provider_loses_reputation_and_is_flagged",
    ),
    Guard(
        id="escrow-settle-books-only-paid-creators",
        sprint="sp1426",
        file="prsm/node/multi_party_escrow.py",
        anchor="if acc.onchain_address and creator_id in self._pending:",
        protects="MultiPartyEscrow booking an UNPAID creator as settled. The atomic on-chain "
                 "settlement branch used to delete EVERY creator from _pending + add the full "
                 "batch total to _total_settled, but _execute_onchain_settlement silently drops "
                 "address-less creators from the transfer — so in a mixed batch each address-less "
                 "creator's royalty was booked-settled, deleted (no retry record), and never paid. "
                 "This gate books/clears ONLY creators actually paid on-chain; drop it and the "
                 "fund loss returns the moment the on-chain royalty path carries value",
        killed_by="tests/unit/test_sprint_1426_escrow_settle_only_pays_addressed.py",
        kills_test_id="test_addressless_creator_stays_pending_and_is_not_booked",
    ),
    Guard(
        id="reconciliation-balance-response-tx-cap",
        sprint="sp1428",
        file="prsm/node/ledger_sync.py",
        anchor="[:_MAX_RECONCILIATION_TX_IDS]",
        protects="a resource-exhaustion DoS of the money event loop. sp1419 activated "
                 "reconciliation on every default node, turning on _handle_balance_response, which "
                 "iterated a peer-supplied recent_tx_ids list with up to 2 awaited SQLite lookups "
                 "per element on the shared ledger connection. Without this cap a hostile peer "
                 "sends one oversized balance_response (a 256MB frame ~= millions of ids) and "
                 "monopolizes the ledger connection, starving concurrent transfers/credits",
        killed_by="tests/unit/test_sprint_1428_balance_response_dos.py",
        kills_test_id="test_a_giant_tx_id_list_does_not_produce_unbounded_db_lookups",
    ),
    Guard(
        id="settlement-durable-committed-escrow-dedup",
        sprint="sp1436",
        file="prsm/settlement/client.py",
        anchor="br.local_escrow_id and br.local_escrow_id in self._committed_escrow_ids",
        protects="double-settle by RE-DELIVERY. sp973's dedup lives on the per-batch "
                 "PendingBatch.seen_escrow_ids set, which is discarded the instant a batch commits "
                 "and pops. A receipt re-delivered AFTER its batch committed (a crash in the "
                 "commit->discard window, or an upstream retry) therefore sailed past the per-batch "
                 "dedup, built a FRESH batch, and committed a SECOND time under a distinct on-chain "
                 "batchId — the registry has no content dedup, so the provider is paid / the escrow "
                 "is drained twice for one unit of work. This reject consults the DURABLE "
                 "_committed_escrow_ids ledger (persisted across restarts) and drops an "
                 "already-settled escrow id. Delete it and the re-delivery double-settle returns",
        killed_by="tests/unit/test_sprint_1436_committed_escrow_dedup.py",
        kills_test_id="test_redelivery_after_commit_is_dropped_no_second_settle",
    ),
    Guard(
        id="delegation-provider-binding",
        sprint="sp1437",
        file="prsm/settlement/payment_delegation.py",
        anchor="auth_provider.lower() != deleg_provider.lower()",
        protects="a relayer draining a funder's escrow N× the signed cap. A PaymentDelegation's "
                 "cumulative cap (max_total_spend_wei) is enforced by a PER-NODE budget store keyed "
                 "only by delegation_nonce; the per-request auth is provider-pinned but the "
                 "delegation was NOT, so one delegation with cap C could be presented (with a fresh "
                 "provider-matched auth) at each of N providers, each reserving up to C against its "
                 "own store from consumed=0 → C×N drain. This check binds the delegation to the ONE "
                 "provider named in its signed struct; combined with the verifier's auth.provider== "
                 "this_node pin, a delegation is spendable at exactly one node. Delete it and the "
                 "cross-provider drain returns",
        killed_by="tests/unit/test_sprint_1437_delegation_provider_binding.py",
        kills_test_id="test_delegation_bound_to_one_provider_cannot_be_spent_at_another",
    ),
    Guard(
        id="paid-key-store-durable-default",
        sprint="sp1438",
        file="prsm/node/node.py",
        anchor="resolve_paid_key_store_path(os.environ)",
        protects="a paid buyer stranded (permanent 404, no refund) after a node restart. The node "
                 "used to wire PaidKeyStore(PRSM_PAID_KEY_STORE_FILE or None): an unset env var "
                 "silently degraded the 'DURABLE' retained-key store to IN-MEMORY, so a restart "
                 "wiped every wrapped key while the on-chain payment gate persists forever. A buyer "
                 "who then pays via ContentAccessVerifier.payForAccess (FTNS pulled, creator "
                 "credited, paid=true) GETs /content/paid-key and hits a 404 — the live CAV has no "
                 "refund. resolve_paid_key_store_path defaults to a durable ~/.prsm path (mirroring "
                 "the settlement store); reverting to `or None` re-opens the fund loss",
        killed_by="tests/unit/test_sprint_1438_paid_decrypt_fund_loss.py",
        kills_test_id="test_resolve_path_defaults_durable_when_env_unset",
    ),
    Guard(
        id="paid-publish-retain-before-deposit",
        sprint="sp1438",
        file="prsm/economy/paid_content.py",
        anchor="paid_key_store.put(content_hash, wrapped_key, int(fee_wei))",
        protects="a paid buyer stranded via a live gate with no key. deposit_commitment_and_retain "
                 "(the sole HTTP-wired paid-publish path) used to deposit_key (create the on-chain "
                 "payment gate) BEFORE retaining the wrapped key — the reverse of the sibling "
                 "publish_paid_content's sp1361 F5 invariant. deposit_key raises OnChainPendingError "
                 "on a receipt-wait timeout while the tx still MINES, so put() was skipped yet the "
                 "gate went live: a buyer pays and gets a permanent 404 (no refund). This put() must "
                 "stay BEFORE the deposit_key call below it; a retained key with no gate is harmless "
                 "(no one can pay for it), a live gate with no key is the fund loss",
        killed_by="tests/unit/test_sprint_1438_paid_decrypt_fund_loss.py",
        kills_test_id="test_key_is_retained_before_a_failing_deposit_so_it_is_not_stranded",
    ),
]


def guard_ids() -> List[str]:
    return [g.id for g in GUARDS]
