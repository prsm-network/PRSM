# Settlement-Receipt Data Plane — Design Scope (2026-06-16)

Status: **design scope** (not yet implemented). Grounds the runtime layer that feeds the
already-complete, already-tested dispute modules so a node can actually *detect and challenge*
another provider's settlement fraud. Produced from a 5-investigator code survey of the live
substrate; synthesis + adversarial review folded inline (the workflow's synthesis/review agents
were lost to a transient API overload, so this was authored from the raw findings + first-hand
knowledge of the dispute modules built in sp1129–1132).

Companion precedent: `docs/2026-06-04-content-data-plane-trust-anchors.md` (the content data plane
used the same "gossiped data is untrusted until a chain anchor binds it" framing).

## 1. The gap

The dispute mechanism MODULES are complete + TDD'd + adversarially verified (sp1116–1132), and
all are **pure / injectable**:

| Module | Input it needs | File |
|--------|----------------|------|
| `verify_inference_receipt_for_challenge(receipt, settler_public_key_b64, stage_public_keys)` | a signed `InferenceReceipt` + the settler pubkey | `prsm/settlement/challenge_verifier.py:92` |
| `ChallengeWatcher(source, dry_run_client).scan()` | `WatchUnit(batch_id, inference_receipt, settler_public_key_b64, batch_receipts, target_index, stage_public_keys)` from a pluggable async `ReceiptSource` | `prsm/settlement/challenge_watcher.py:61,148` |
| `detect_double_spends(batches)` | an iterable of `CommittedBatch` (reads `batch_id`, `provider_address`, `leaf_hashes`) | `prsm/settlement/double_spend_detector.py:87` |
| `assemble_invalid_signature_challenge(batch_id, batch_receipts, target_index)` | the **order-preserved receipt preimages** | `prsm/settlement/challenge_assembler.py:115` |
| `assemble_double_spend_challenge(...)` | target preimages + the conflicting batch's ordered `leaf_hashes` | `prsm/settlement/double_spend_assembler.py:110` |

**Nothing feeds them at runtime.** Concretely:

1. **The producer discards the data.** After `commit_ready_batches` builds the merkle root, the
   accumulated `BatchedReceipt`s are popped and never persisted. `CommittedBatch` retains only
   `leaf_hashes` (and the root) — *not* the receipt preimages (`client.py:289-379`, `:145-163`).
2. **The chain has only the root.** `BatchCommitted` emits `batchId, provider (indexed),
   merkleRoot, receiptCount, totalValueFTNS, commitTimestamp, metadataURI`
   (`BatchSettlementRegistry.sol:355-363`). The leaf set + receipt bytes are **not** on-chain;
   `receiptCount`/`totalValueFTNS` are provider-submitted and **unvalidated**. A node can
   *enumerate* committed batches by provider (indexed topic) by scanning `BatchCommitted` logs in
   9k-block windows over a ~300k-block lookback (`batch_settlement_contract_client.py:232-309`,
   `_SCAN_MAX_WINDOW=9000`).
3. **No network channel carries receipts/leaves.** The gossip mesh has 27 subtypes (jobs, storage,
   capability ads, provenance) but **none** for inference receipts, batch commitments, leaf sets,
   or challenges (`gossip.py:145-220`). The accumulator never broadcasts (`accumulator.py`).

So an observing node can see *that* provider P committed batch B with root R, but cannot obtain
B's leaf set or receipts to verify it. That sourcing layer is the **data plane**.

## 2. What is available where

| Layer | Holds | Trust |
|-------|-------|-------|
| **On-chain** (`BatchCommitted` + `Batch` struct) | `batch_id`, `provider`, `merkleRoot`, `receiptCount`, `totalValueFTNS`, `commitTimestamp`, per-batch governance snapshots | **Trusted anchor.** The merkle root binds the leaf set; pubkey/sig *hashes* are bound into each leaf on-chain (`_handleInvalidSignature`). |
| **Producer local-at-commit** | full ordered `BatchedReceipt`s + leaf order + `InferenceReceipt` | Authoritative, but **discarded** post-commit today. |
| **Gossip** (`gossip.py`) | sp934 origin-authenticated messages (signer signs `subtype+data+origin+nonce`; receiver derives node_id from pubkey + verifies), sp1008 24h replay dedup, digest exchange for late joiners | Origin authenticated, **content NOT** — a peer can sign a message carrying a false receipt set. |
| **P2P content** (`ContentProvider`, sp1020) | INLINE ≤1 MiB / CHUNKED ≤64 MiB blobs, CID = content hash, re-verified on fetch | Content-addressed, self-verifying on fetch. |

## 3. The trust model (the property that makes this feasible)

**Observed receipt/leaf data is never trusted.** Three independent gates make forgery harmless:

1. **Re-verification in the modules.** The verifier re-checks the settler signature
   (`receipt.py:71`) and the assembler re-checks the shard signature
   (`challenge_assembler.py:44`); leaf hashes are deterministically recomputed from preimages
   (`merkle.py:98-185`). Forged signatures fail closed.
2. **Chain-root cross-check (the anchor).** An observer recomputes the merkle root from a fetched
   leaf set and compares it to the **on-chain `merkleRoot`** for that `batch_id`. A leaf set that
   doesn't reproduce the committed root is discarded before any work. Because the on-chain
   `_handleInvalidSignature` binds `keccak(pubkey)==leaf.providerPubkeyHash` and
   `keccak(sig)==leaf.signatureHash`, even the settler pubkey is chain-anchored via the leaf — a
   challenger cannot substitute an arbitrary key.
3. **Dry-run gate.** Every assembled challenge is dry-run (static `eth_call`) against live chain
   state before any (user-gated) broadcast (`invalid_signature_submitter.dry_run`). A challenge
   that wouldn't succeed never broadcasts.

**Net security property:** a Sybil/malicious peer feeding forged or inconsistent receipts can at
worst waste a *bounded* amount of an observer's local compute (a DoS, mitigated by §6). It can
**never** cause a wrongful slash — the chain root + the dry-run reject anything that isn't a true,
on-chain-provable fraud. This is what lets the data plane consume untrusted gossip safely.

## 4. Privacy stance

Receipts carry **hashes only** — `output_hash`, `prompt_hash`, `signature_hash`,
`signing_message_hash` (`merkle.py:98-159`, `models.py:114-183`) — never prompt/output plaintext.
The only plaintext-ish fields are `requester_address`, `provider_address`, `value_ftns`, which are
**already on-chain**. So propagating a batch's receipt set leaks **no new content**. (Confidential
Tier B/C content is a separate concern and is not in the settlement receipt at all.) Stance:
publish the hashes-only receipt set; never add plaintext.

## 5. Proposed architecture — pull-on-demand, chain-anchored

Prefer **pull-on-demand over flood**: a lightweight authenticated *availability* announcement plus
fetch-the-blob-when-you-want-to-audit, rather than gossiping every receipt set (which would flood
the mesh and amplify). Three roles:

**Producer (the committing node):**
- On commit, **retain** the batch's ordered receipt set + leaf order in a bounded local
  "published-batch store" (keyed by `batch_id`, GC'd after the challenge window + margin).
- Serialize the set into a canonical content blob (CID = hash), reusing the sp1074 artifact-bundle
  + `ContentProvider` pattern; serve it on demand (a ~1000-receipt batch ≈ 1 MB, well under the
  64 MiB CHUNKED cap).
- **Announce availability** via a new sp934-authenticated gossip subtype
  `settlement_batch_available` carrying `(batch_id, provider, merkle_root, cid, leaf_hashes?)`.
  Including `leaf_hashes` inline (cheap, ~32 KB for 1000 leaves) lets observers run **double-spend
  detection without any fetch**; the full receipt set is fetched only for INVALID_SIGNATURE
  assembly.

**Observer (any node):**
- Builds a worklist from `BatchCommitted` events (trusted anchor) cross-referenced with received
  availability ads.
- For audited batches: recompute root from the inline `leaf_hashes` (double-spend) or the fetched
  blob (invalid-signature) and **cross-check against the on-chain root** (§3 gate 2). Cache only
  verified, chain-anchored sets in a bounded store.

**Feed:** a concrete `ReceiptSource` over the verified cache yields `WatchUnit`s to
`ChallengeWatcher`, and the cached `CommittedBatch` list (with leaf_hashes) feeds
`detect_double_spends`. The dry-run gate (already built) is the final pre-broadcast check.

## 6. Bounding DoS / spam / replay

- **Replay:** `batch_id` is the natural idempotency key (immutable per batch) — stronger than the
  generic sp1008 nonce window; an availability ad for a known `batch_id` is deduped.
- **Rate / amplification:** reuse the existing sp936 per-peer token bucket; cap concurrent fetches
  and add a per-observer **audit budget** (sample/triage rather than audit-all — see §8).
- **Cheap rejection first:** recompute-root is O(n) hashes and rejects a forged set *before* any
  signature work or fetch of the full blob.
- **libp2p origin-auth gap (sp1010 Residual A):** the libp2p gossip path is not yet
  origin-authenticated. Settlement availability ads must **fail closed** there (only trust ads on
  the sp934-authenticated WebSocket path) until sp1010 lands.

## 7. Phased brick sequence

Four of five bricks are testable **without a live network** (pure/injectable); only the last
touches the daemon/network.

| Brick | Scope | Depends on | Offline-testable |
|-------|-------|------------|------------------|
| **A — producer retention store** | After commit, persist the ordered receipt set + leaf order to a bounded, GC'd local store keyed by `batch_id`. The prerequisite for everything (receipts are discarded today). | — | ✅ |
| **B — serialize + root cross-check** | Canonical receipt-set blob (CID = hash) + a pure `verify_against_onchain_root(leaves, root)` that recomputes + compares. The §3 gate-2 primitive. | A | ✅ |
| **C — `settlement_batch_available` subtype** | New sp934-authenticated gossip subtype `(batch_id, provider, merkle_root, cid, leaf_hashes?)` + retention + `batch_id` dedup; producer announces on commit. | B | ✅ (build/parse/auth/dedup unit-tested like sp934) |
| **D — observer cache + `ReceiptSource`** | Bounded cache of chain-anchored verified sets; a concrete `ReceiptSource` (yields `WatchUnit`s) + `CommittedBatch` list feeding the detector. Chain client + fetcher **injected**. First end-to-end run of the modules on cross-node data (fetch mocked). | B, C | ✅ |
| **E — daemon wiring (network-gated)** | Producer announce-on-commit + observer fetch-via-`ContentProvider` + the audit scheduler/sampler, wired into `node.initialize()`. | A–D | ⚠️ integration (needs a live mesh to fully exercise) |

## 8. Open questions

1. **Which batches does an observer audit?** Auditing every batch network-wide is O(batches)
   fetches. Options: bounded random sampling, targeted (batches touching the observer's own
   requesters/jobs), or staking-weighted. A "completeness vs. cost" knob — needs a policy decision.
2. **Incentive to publish.** Why would a provider publish receipts that let others challenge it?
   Honest providers' receipts verify (nothing to hide), and a *refusal* to publish is itself a weak
   negative signal. The strongest incentive-aligned source is **consensus-group co-execution**
   (`consensus_group_id`): peers who independently hold the same receipts can challenge without the
   target's cooperation. Worth tying to reputation/staking later. Genuinely open.
3. **Retention window.** Producer store GC after `challengeWindowSecondsAtCommit` + margin; confirm
   the margin against reconcile/finalize timing.
4. **Inline leaf_hashes vs. fetch-only.** Carrying `leaf_hashes` in the ad (cheap) enables
   no-fetch double-spend scanning; confirm the ~32 KB ad size is acceptable on the mesh.

## 9. Risks + alternative considered

- **Scale** — audit-all doesn't scale; mitigated by §8.1 sampling/triage. If un-bounded, `log()`
  what was skipped (no silent coverage gaps).
- **Incentive** — a non-publishing provider is only challengeable if someone else holds the leaves;
  the consensus-group path (§8.2) is the robust answer.
- **libp2p auth gap** — settlement ads fail closed on the unauthenticated path until sp1010.
- **Alternative (rejected): push every receipt set via gossip.** Floods the mesh, amplification
  risk, and forces all peers to ingest data they'll never audit. Pull-on-demand + a small
  authenticated availability ad dominates on bandwidth, privacy, and DoS surface.

## 10. Recommended first step

**Brick A (producer retention store).** It's the true prerequisite (receipts are discarded today),
fully offline-testable, and de-risks the data shapes for everything downstream. It is *not* a
half-bridge: combined with the existing modules it already lets a node re-derive and self-verify
its own committed roots from retained receipts; bricks B–D then generalize that to cross-node,
chain-anchored auditing, and E wires it to the live mesh.
