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
    Guard(
        id="withdraw-refund-exactly-once-credit",
        sprint="sp1439",
        file="prsm/node/pending_withdraw_reconciler.py",
        anchor='idempotency_key=f"withdraw-refund:{intent.job_id}"',
        protects="a reverted/dropped withdraw NEVER being refunded (permanent lost-debit). The old "
                 "_refund claimed a separate record_nonce marker in its OWN durable commit BEFORE "
                 "crediting; a crash in the gap burned the claim while no credit landed, and on "
                 "restart the burned claim made _refund return without ever crediting. Folding "
                 "idempotency into the credit itself (deterministic tx_id → PRIMARY-KEY dedup) makes "
                 "the refund exactly-once and crash-safe: a crash before the commit just re-runs the "
                 "same idempotent credit. Reverting to claim-before-credit re-opens the lost refund",
        killed_by="tests/unit/test_sprint_1439_withdraw_reconciler_lost_debit.py",
        kills_test_id="test_refund_credits_despite_a_burned_claim_from_a_crashed_prior_run",
    ),
    Guard(
        id="withdraw-dropped-tx-nonce-advance-gate",
        sprint="sp1439",
        file="prsm/node/pending_withdraw_reconciler.py",
        anchor="int(confirmed_nonce) > int(intent.nonce)",
        protects="TWO opposite fund-loss failures. (a) A dropped/evicted withdraw tx never produces "
                 "a receipt, so without this the reconciler polls 'pending' forever and the "
                 "off-chain debit is stranded (lost-debit). (b) This STRICT-greater gate is also the "
                 "double-pay guard: a tx is refunded ONLY once the escrow's CONFIRMED nonce has "
                 "advanced strictly PAST this tx's nonce (a different tx took the slot → this "
                 "tx_hash can never mine). Weakening `>` to `>=`, or refunding on age/absence alone, "
                 "would refund a tx that could still confirm → double-pay",
        killed_by="tests/unit/test_sprint_1439_withdraw_reconciler_lost_debit.py",
        kills_test_id="test_is_dropped_true_only_when_confirmed_nonce_advanced_past_tx_nonce",
    ),
    Guard(
        id="deposit-link-no-cross-wallet-steal",
        sprint="sp1441",
        file="prsm/node/local_ledger.py",
        anchor="existing is not None and existing != wallet_id",
        protects="deposit THEFT via the eth_address→wallet link. _credit_deposit resolves the "
                 "destination wallet via wallet_for_eth_address at SCAN time, and link_eth_address "
                 "used to SILENTLY MOVE an address already bound to another wallet (last-write-wins) "
                 "— so whoever re-linked a known deposit address LAST before the scan received its "
                 "next inbound on-chain transfer. Safe today only because the /wallet/* API is "
                 "loopback-bound + API-key-gated (single-tenant); this guard makes the protection "
                 "boundary-INDEPENDENT (refuses a cross-wallet re-link) so the theft primitive stays "
                 "closed if the daemon is ever exposed multi-tenant. Deleting it restores the silent "
                 "move (the deposit-audit-flagged latent theft)",
        killed_by="tests/unit/test_sprint_540_bridge_deposit_pattern_a.py",
        kills_test_id="test_link_eth_address_refuses_cross_wallet_relink",
    ),
    Guard(
        id="transport-concurrent-connection-cap",
        sprint="sp1442",
        file="prsm/node/transport.py",
        anchor="is_new and len(self.peers) >= self._max_peers",
        protects="an eclipse + fd/memory DoS of the whole node. node_id = sha256(pubkey)[:32] is "
                 "free to mint, so without a ceiling one host opens UNLIMITED authenticated "
                 "WebSocket slots: each is an fd + PeerConnection + read-loop coroutine + rate "
                 "bucket (exhaustion), and because gossip() fans out by random-sampling self.peers, "
                 "attacker-dominated slots black-hole honest publishes (eclipse/partition). The "
                 "sp936 bucket caps messages-per-peer not peer COUNT; sp1414 caps a DIFFERENT dict "
                 "(discovery.known_peers); sp1326 only collapses duplicate ids — none bounds this. "
                 "This is the global concurrent-connection ceiling (+ a per-IP cap); deleting it "
                 "re-opens the Sybil-flood eclipse/DoS",
        killed_by="tests/unit/test_sprint_1442_p2p_substrate_hardening.py",
        kills_test_id="test_global_max_peers_rejects_a_new_peer_when_full",
    ),
    Guard(
        id="gossip-digest-response-solicitation-gate",
        sprint="sp1442",
        file="prsm/node/gossip.py",
        anchor='"unsolicited_digest_response"',
        protects="a ~200x CPU/DB amplifier AND a signed-frame replay-injection. _handle_digest_"
                 "response processed ANY inbound digest_response with no check we solicited it, and "
                 "each of up to 200 entries drove a full-window gossip-log scan — one rate-limited "
                 "frame → ~200 unfiltered ledger scans/writes, past the sp936 per-frame limit; and "
                 "because the digest path runs BEFORE the sp1008 replay barrier, a validly-signed "
                 "frame older than the 24h window could be re-injected as authentic. This "
                 "solicitation gate (keyed on the authenticated peer.peer_id, single-use) drops a "
                 "response from any peer we never sent a request to — and we only request from "
                 "OUTBOUND peers we dialed, so an inbound attacker can never trigger it. Deleting it "
                 "re-opens both the amplifier and the replay-injection",
        killed_by="tests/unit/test_sprint_1442_p2p_substrate_hardening.py",
        kills_test_id="test_unsolicited_digest_response_is_dropped_before_processing",
    ),
    Guard(
        id="sandbox-exec-fail-closed",
        sprint="sp1443",
        file="prsm/core/integrations/security/sandbox_manager.py",
        anchor="if not _unisolated_exec_enabled():",
        protects="arbitrary code execution / host-secret exfiltration / SSRF on the node. "
                 "SandboxManager.execute_safely runs untrusted .py as a PLAIN child of the daemon "
                 "user — no netns/seccomp/userns/chroot, full host filesystem read + unrestricted "
                 "network (the block_network/allowed_domains flags are advisory JSON no execution "
                 "code reads). Reachable by design from the authenticated /integrations/import + "
                 "/integrations/security/scan flows (enable_sandbox defaults True, _should_run_"
                 "sandbox auto-runs any payload that evades the regex vuln scan). This gate FAILS "
                 "CLOSED: it refuses to EXECUTE untrusted content unless the operator explicitly "
                 "opts in (behind real external isolation). Delete it and downloaded code auto-runs "
                 "as the daemon user again",
        killed_by="tests/unit/test_sprint_1443_sandbox_exec_fail_closed.py",
        kills_test_id="test_execute_safely_does_not_run_untrusted_code_by_default",
    ),
    Guard(
        id="sandbox-exec-process-group-reap",
        sprint="sp1443",
        file="prsm/core/integrations/security/sandbox_manager.py",
        anchor="preexec_fn=_sandbox_preexec(timeout, _sandbox_mem_limit_bytes())",
        protects="an unbounded runaway-process DoS on the (opt-in) subprocess exec path. sp.run(..., "
                 "timeout) SIGKILLs only the direct child, so a forked grandchild (double-fork / "
                 "os.fork) survives the timeout and spins a core / grows memory forever — bounded "
                 "requester effort, unbounded host CPU/memory. _sandbox_preexec starts a NEW SESSION "
                 "(so the timeout handler's os.killpg reaps the whole process GROUP) and clamps "
                 "per-process CPU/address-space rlimits. Delete it and a timed-out job leaks runaways",
        killed_by="tests/unit/test_sprint_1443_sandbox_exec_fail_closed.py",
        kills_test_id="test_opt_in_exec_wires_new_session_rlimits_and_drops_host_path",
    ),
    Guard(
        id="authz-protect-dashboard-ftns-drain",
        sprint="sp1444",
        file="prsm/api/auth_middleware.py",
        anchor='"/api/ftns/",',
        protects="an unkeyed DRAIN of the operator's off-chain FTNS. NodeAuthMiddleware is a "
                 "deny-list (is_protected_path), and the web dashboard sub-app mounted at "
                 "app.mount(\"\",…) exposes POST /api/ftns/transfer → node.ledger_sync.signed_"
                 "transfer (debits the operator's own wallet) — the whole dashboard /api/* surface "
                 "was invisible to the deny-list, so on a KEYED public node an attacker with no API "
                 "key could transfer the operator's FTNS to themselves (+ read /api/ftns/balance|"
                 "history PII). This prefix gates the dashboard money subtree; deleting it re-opens "
                 "the drain (a blanket /api/ is NOT usable — it would gate the dashboard's own "
                 "/api/auth/login)",
        killed_by="tests/unit/test_sprint_1444_api_authz_denylist_gaps.py",
        kills_test_id="test_keyed_node_rejects_unkeyed_ftns_transfer",
    ),
    Guard(
        id="authz-protect-content-paid-publish",
        sprint="sp1444",
        file="prsm/api/auth_middleware.py",
        anchor='"/content/paid/",',
        protects="an unkeyed spend of the operator's gas + on-chain provenance poisoning. POST "
                 "/content/paid/publish (sp1367) calls key_client.deposit_key signed by the "
                 "operator's PRSM_PAID_PUBLISHER_KEY hot wallet (burns gas per call), registers the "
                 "operator as the on-chain creator, and injects an attacker-chosen wrapped key into "
                 "the paid-key store — yet the /content/* deny-list prefixes (/content/upload, "
                 "/content/arbitration/, /content/mine, .../pin) never covered /content/paid/, so it "
                 "was reachable with no API key on a keyed node. Delete this prefix and the gas "
                 "drain / provenance-poison returns",
        killed_by="tests/unit/test_sprint_1444_api_authz_denylist_gaps.py",
        kills_test_id="test_sensitive_route_is_now_protected",
    ),
    Guard(
        id="authz-default-deny",
        sprint="sp1445",
        file="prsm/api/auth_middleware.py",
        anchor="return True  # default-deny: every unenumerated path requires the operator key",
        protects="the WHOLE node API from the recurring deny-list-gap class. NodeAuthMiddleware was "
                 "inverted from a deny-list (protect only enumerated prefixes; everything else OPEN "
                 "even on a keyed node — which leaked a new gap every time a sensitive route shipped, "
                 "sp138/183/1012/1103/1444) to DEFAULT-DENY: a path is protected unless on the explicit "
                 "PUBLIC allowlist, so a NEW route is fail-closed the moment it is added. This final "
                 "`return True` IS the inversion. Flip it to `return False` (or remove it) and every "
                 "unenumerated path — including any future operator/money route — is served "
                 "unauthenticated again",
        killed_by="tests/unit/test_sprint_1445_authz_default_deny.py",
        kills_test_id="test_a_new_unlisted_route_is_protected_by_default",
    ),
    Guard(
        id="per-stage-stable-escrow-id-for-dedup",
        sprint="sp1446",
        file="prsm/settlement/per_stage_settlement_split.py",
        anchor='local_escrow_id=f"{job_id}::stage::{node_id}"',
        protects="a double-settle on the big-model paid MULTI-STAGE path. Each stage node "
                 "self-commits its own share (Design A); the ONLY thing that stops a crash-before-"
                 "discard re-drain from committing the same share TWICE on-chain is sp1436's durable "
                 "committed-escrow-id dedup, which keys on this share-batch's local_escrow_id. That "
                 "id MUST be STABLE across re-deliveries of the same (job, stage, node) — a "
                 "non-deterministic id (e.g. a random/per-delivery value) would get a fresh key each "
                 "drain, sail past the dedup, and double-settle. Keep it derived from (job, stage, "
                 "node); deleting/randomizing it re-opens the double-settle sp1436 closed",
        killed_by="tests/unit/test_sprint_1446_per_stage_double_settle_closed.py",
        kills_test_id="test_re_commit_of_a_committed_stage_task_does_not_double_settle",
    ),
    Guard(
        id="per-stage-arm-dedup-before-broadcast",
        sprint="sp1448",
        file="prsm/settlement/client.py",
        anchor="self._arm_committing_escrow_ids(ready)",
        protects="a DOUBLE escrow release on the big-model per-stage path. sp1436's dedup ledger was "
                 "armed ONLY on the clean-commit success tail, so a commit that BROADCAST but did not "
                 "cleanly confirm — OnChainPendingError (receipt-wait timed out) or BroadcastFailedError "
                 "(send threw) yet actually mined — left the escrow id UNARMED. The receiver store then "
                 "re-injected the still-staged share and committed a SECOND on-chain batch for it → the "
                 "requester is charged twice / the payee paid twice. Arming the id BEFORE the "
                 "irreversible broadcast closes the quarantine/broadcast-failed/crash-in-window gaps. "
                 "Delete this and every uncertain-fate per-stage commit that lands double-settles",
        killed_by="tests/unit/test_sprint_1448_per_stage_commit_failure_double_settle.py",
        kills_test_id="test_onchain_pending_commit_does_not_double_settle_on_redrain",
    ),
    Guard(
        id="per-stage-revert-unarms-escrow-id",
        sprint="sp1448",
        file="prsm/settlement/client.py",
        anchor="self._discard_committed_escrow_id(br.local_escrow_id)",
        protects="STRANDED escrow on the per-stage path. sp1448 arms the dedup ledger BEFORE broadcast; "
                 "a commit that MINES-AND-REVERTS (OnChainRevertedError) landed nothing, so its share "
                 "MUST stay re-committable. This unarm (revert branch only — pending/broadcast-failed "
                 "keep the arm because they MAY have landed) is the ONLY thing that lets a reverted "
                 "share retry; delete it and the pre-broadcast arming permanently blocks the share from "
                 "ever committing → funds locked, work never paid",
        killed_by="tests/unit/test_sprint_1448_per_stage_commit_failure_double_settle.py",
        kills_test_id="test_reverted_commit_is_retryable_not_stranded",
    ),
    Guard(
        id="per-stage-discard-owned-share",
        sprint="sp1448",
        file="prsm/settlement/per_stage_receiver_store.py",
        anchor="client.has_committed_escrow_id(staged.local_escrow_id)",
        protects="a DOUBLE escrow release via receiver-store re-injection. Once a per-stage commit has "
                 "broadcast (armed in the client's durable ledger), the client's WAL/quarantine + the "
                 "recover/reconcile phases own settling it. If the staged task is NOT discarded it "
                 "re-injects the same share every drain cycle → a second on-chain batch (and an "
                 "unbounded stuck-task leak). This discard-on-ownership stops the re-injection; delete "
                 "it and a broadcast-but-unconfirmed share is re-committed on the next drain",
        killed_by="tests/unit/test_sprint_1448_per_stage_commit_failure_double_settle.py",
        kills_test_id="test_owned_share_task_is_discarded_not_re_injected",
    ),
    Guard(
        id="per-stage-recover-adoption-phases",
        sprint="sp1448",
        file="prsm/settlement/client_wiring.py",
        anchor="for status_key, method_name in _PER_STAGE_RECOVER_PHASES:",
        protects="STRANDED escrow on the per-stage path. The per-stage cycle omitted the pending-commit "
                 "recovery/adoption phases the single-stage poll loop runs (_POLL_PHASES). A per-stage "
                 "commit that broadcast-but-unconfirmed and then LANDED is parked in _pending_commits / "
                 "_committing and is NEVER adopted into _tracked, so run_per_stage_finalize_cycle can't "
                 "finalize it → the escrow it locked on-chain is never released to the payee. This loop "
                 "runs recover_committing_intents + reconcile_pending_commits on the per-stage client so "
                 "landed batches are adopted + finalizable; delete it and landed-but-unconfirmed shares "
                 "strand forever",
        killed_by="tests/unit/test_sprint_1448_per_stage_commit_failure_double_settle.py",
        kills_test_id="test_per_stage_commit_cycle_runs_recovery_phases",
    ),
    Guard(
        id="per-stage-value-bound-to-authorized-share",
        sprint="sp1449",
        file="prsm/settlement/per_stage_routing.py",
        anchor="if settled_value != authorized_share:",
        protects="a per-stage node settling a DIFFERENT amount than the requester authorized. The "
                 "per-stage authorization commits to (payee, share_wei) and the gate enforces "
                 "membership + the cumulative cap over share_wei — but the amount that settles on-chain "
                 "is batched_receipt.value_ftns (accumulate→commitBatch(totalValueFTNS)→"
                 "settleFromRequester). The honest splitter sets value_ftns == share_wei, so nothing "
                 "re-asserted it; a malformed/tampered routed task with value_ftns > share_wei would "
                 "pass the gate yet over-draw the requester's escrow (and get the committing node "
                 "challenged + slashed). This binds settled==authorized both directions; delete it and "
                 "the gate authorizes a share while a different value settles",
        killed_by="tests/unit/test_sprint_1449_per_stage_value_bound_to_authorized_share.py",
        kills_test_id="test_gate_rejects_value_ftns_exceeding_authorized_share",
    ),
    Guard(
        id="per-stage-no-escrow-recording",
        sprint="sp1450",
        file="prsm/settlement/issued_authorization_store.py",
        anchor="def record_per_stage(",
        protects="the requester-side NO_ESCROW defense being BLIND to per-stage batches. commitBatch "
                 "has NO on-chain requester-auth check, so the PRIMARY boundary against an unauthorized "
                 "escrow drain is the requester challenging a batch that matches no issued authorization "
                 "(scan_for_unauthorized_batches → match_unauthorized_batches). The requester signs ONE "
                 "per-stage auth over a SET of (payee, share) and each stage node commits its own batch; "
                 "this records ONE entry per payee (provider=payee, max_spend=share) so the matcher "
                 "classifies an HONEST per-stage batch AUTHORIZED and an INFLATED/FOREIGN one "
                 "UNAUTHORIZED. Delete it and the store stays EMPTY for per-stage — the matcher either "
                 "griefs every honest stage node or misses an escrow-draining unauthorized batch",
        killed_by="tests/unit/test_sprint_1450_per_stage_no_escrow_coverage.py",
        kills_test_id="test_honest_per_stage_batches_are_authorized",
    ),
    Guard(
        id="per-stage-published-batch-store-for-observer-challenge",
        sprint="sp1451",
        file="prsm/settlement/client_wiring.py",
        anchor='published_batch_store=getattr(node, "_settlement_published_batch_store", None))',
        protects="the observer/watchdog challenge data plane being BLIND to per-stage fraud. The "
                 "settlement audit engine's double-spend/invalid-sig/expired scanners read the "
                 "VerifiedBatchCache, populated by ingesting receipts fetched via a gossiped announce "
                 "CID; the announce step advertises every batch in the node's PublishedBatchStore. "
                 "Passing the node's store to the per-stage client is what makes per-stage committed "
                 "batches RETAINED + ANNOUNCED (parity with single-stage). Delete it and the per-stage "
                 "client keeps published_batch_store=None → per-stage batches are never announced → the "
                 "observer scanners never see them → per-stage fraud by an authorized node goes "
                 "unsurfaced (the requester's sp1450 NO_ESCROW defense covers only UNauthorized batches)",
        killed_by="tests/unit/test_sprint_1451_per_stage_observer_challenge_coverage.py",
        kills_test_id="test_per_stage_client_receives_the_nodes_published_batch_store",
    ),
    Guard(
        id="delegation-budget-key-canonical-nonce",
        sprint="sp1452",
        file="prsm/settlement/delegation_budget.py",
        anchor='_to_bytes32("delegation_nonce", s).hex()',
        protects="a relayer draining a funder's escrow past the signed delegation cap by ALIASING one "
                 "delegation into N budget buckets. The EIP-712 delegation signature canonicalizes the "
                 "nonce via _to_bytes32 ('0x<hex>' and bare '<hex>' and case variants share ONE digest "
                 "→ ONE funder signature verifies all), but the budget cap is enforced OFF-CHAIN keyed "
                 "by delegation_nonce. Keying on the raw .lower()'d string let a malicious relayer (no "
                 "funder key) spend cap C, then strip '0x' from the nonce STRING (same signature), land "
                 "a DISTINCT bucket, and drain another C → N×C. Canonicalizing the budget key the SAME "
                 "way the signature does collapses every alias to one bucket. Revert it to .lower() and "
                 "the cumulative cap becomes a per-spelling cap → N×C escrow drain",
        killed_by="tests/unit/test_sprint_1452_delegation_nonce_alias_cap_multiplication.py",
        kills_test_id="test_aliased_delegation_nonce_cannot_double_the_cap_end_to_end",
    ),
    Guard(
        id="kyc-webhook-replay-token-shared-parser",
        sprint="sp1453",
        file="prsm/node/api.py",
        anchor='parse_persona_signature_header(persona_sig_header).get("v1", "")',
        protects="the KYC-webhook replay ring being SILENTLY SKIPPED for a whitespace variant of the "
                 "Persona-Signature header. The signature verifier parses the header tolerantly "
                 "(partition('=')+strip(), so `v1 = <hex>` verifies), so the replay-token extractor MUST "
                 "parse it the SAME way — else a signature-valid spaced header leaves replay_token='' and "
                 "the `if replay_token:` guard skips replay_ring.record(), letting a captured KYC-approval "
                 "webhook be REPLAYED (bounded only by the ±300s freshness window). Reverting this to a "
                 "strict inline startswith('v1=') re-drifts the two parsers and re-opens the ring bypass",
        killed_by="tests/unit/test_sprint_1453_kyc_webhook_replay_parser_parity.py",
        kills_test_id="test_replay_token_extraction_agrees_with_the_verifier_on_a_spaced_header",
    ),
]


def guard_ids() -> List[str]:
    return [g.id for g in GUARDS]
