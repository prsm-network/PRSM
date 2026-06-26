# Fable 5 Review — Domain 01: Money path + Trust path

Adversarial, cross-cutting review. See `00_INDEX.md` for shared framing + output format
(skeptical; cross-module not per-function; concrete `file:line` + exploit; verdict per
invariant; top fix + needs-live-test at the end).

## Context

PRSM runs distributed AI inference with an on-chain economy on Base mainnet (chainId 8453).
Requesters pay FTNS (ERC-20) via an EscrowPool; settlement batches inference receipts and
finalizes them on-chain (`settleFromRequester` moves FTNS from a requester's escrow to a
provider). Confidential ("Tier B/C") inference must run only inside hardware TEEs (Intel SGX
/ AMD SEV-SNP), gated by cryptographic attestation. This domain is the money + trust surface
— a flaw is high-consequence (lost/stolen funds, or confidential work on an unattested box).

---

## SURFACE A — On-chain money path

### Read (trace value flow ACROSS these, not in isolation)
- `prsm/settlement/accumulator.py`, `client.py`, `state_store.py`, `client_wiring.py`,
  `inference_adapter.py`
- `prsm/settlement/payment_authorization.py`, `payment_authorization_verifier.py`
  (`PaymentAuthorizationVerifier` + `RelayerAuthorizationVerifier`), `payment_delegation.py`,
  `delegation_budget.py`
- `prsm/economy/web3/escrow_pool_client.py`, `batch_settlement_contract_client.py`
- `prsm/economy/credit_policy.py` (`settle_inference_receipt`, per-stage release split)
- `prsm/node/api.py` — `_resolve_paid_requester_or_402`, `_capture_relayer_budget`,
  `_record_paid_requester`/`_take_paid_requester`, and the TWO settle sites (unary + streaming)

### Invariants — confirm or break
1. **No double-settle.** A batch/receipt-set can never settle twice on-chain. Stress the
   broadcast-but-unconfirmed path (`BroadcastFailedError` — RPC response dropped, tx MAY have
   landed): confirm the quarantine + recover-by-merkle-root adoption never re-commits the same
   receipts as a new batchId, across restarts (durable state) and concurrent poll cycles (lock).
2. **No overspend (relayer).** Trace the **reserve → settle → capture/release** seam: verifier
   reserves the per-request ceiling (`max_spend_wei`); settle clamps the booked value at
   `max_spend_wei`; `_capture_relayer_budget` releases `(reserved − actual)`. Can cumulative
   on-chain settlement across many requests EVER exceed the funder-signed
   `PaymentDelegation.max_total_spend_wei`? Hunt: wei-vs-FTNS-Decimal conversions; capture
   firing only on success while a failed job's reservation lingers; two delegations to one
   relayer; one auth paired with a different delegation.
3. **Correct payee.** `settleFromRequester` draws from the funding requester (the FUNDER for
   the relayer path), never the provider/relayer; multi-stage release credits the signed
   topology workers. Find a path that charges a third party or mis-credits.
4. **No fund-strand.** A committed-on-chain batch must always finalize or surface as
   funds-in-flight. Find a crash/restart window (durable state + commit-intent WAL +
   chain-scan recovery) that strands escrowed FTNS with no recovery handle.
5. **Replay + binding.** job_nonce single-use; EIP-712 sigs bind chainId+provider+request_hash;
   expiry enforced. Replay an auth/delegation against another node/request/after-expiry? TOCTOU
   between nonce `seen()` and `remember()` across the `await` escrow read (the relayer verifier
   has a lock — does `PaymentAuthorizationVerifier`?).
6. **Fail-closed verify, fail-open settle.** Verification rejects (closed); the on-chain settle
   hook is best-effort (never unwinds the completed off-chain settlement, never double-charges,
   never crashes the request). Confirm a verify error can't serve a paid job.

---

## SURFACE B — TEE attestation / trust path

### Read
- `prsm/compute/inference/x509_path.py` (shared chain validator: CA-flag/keyCertSign/pathLen/
  name-chaining/EC+RSA-PSS+EdDSA/CRL), `intel_dcap.py`, `amd_sev_snp.py`, `sgx_tcb.py`,
  `qe_identity.py`, `amd_tcb.py`, `amd_vmpl.py`, `vendor_anchors.py`, `collateral_refresh.py`,
  `attestation_backends.py`
- `prsm/compute/parallax_scheduling/trust_adapter.py` (`verified_tier_attestation`,
  `is_hardware_attestation`, `TierGateAdapter`)
- `prsm/node/dht_backed_pool_provider.py` (per-peer `node_id_authenticated`, pool builder)
- `prsm/node/discovery.py` + `libp2p_discovery.py` (origin-auth + node_id authentication only —
  the broader transport correctness is Domain 02)

### Invariants — confirm or break
1. **No unverified/forged attestation earns a hardware tier.** Confidential routing requires
   `vendor_verified==true` (real DCAP/KDS chain to a pinned vendor root) AND the quote's
   REPORT_DATA bound to the node's node_id. Try: software node advertising `tier-sgx`;
   structurally-valid-unsigned quote; chain to an attacker root; downlevel/revoked TEE;
   debug-enabled SEV-SNP guest.
2. **No quote replay across nodes.** node_id = sha256(pubkey); the gate requires REPORT_DATA to
   commit to node_id, and work delivery is signature-gated against the registered pubkey.
   Confirm a poisoned `(victim_node_id, victim_quote)` entry can't get a software attacker
   routed confidential work — across BOTH transports and all peer-learning paths (gossip
   capability/shard, bootstrap-relay credential); is `node_id_authenticated` set correctly
   per-peer and honored only for authenticated peers?
3. **Collateral freshness can't silently lapse.** An expired CRL must not be treated as "no CRL"
   (revocation silently off); a stale TCB-Info must not keep passing a now-OutOfDate TEE.
   Confirm validate-before-swap can never cache stale/forged/wrong-issuer collateral, and a MITM
   on the plain-HTTP PCS/KDS response can't substitute its own CRL+chain+root.
4. **Only the documented escape hatches exist.** Known residuals: `PRSM_PARALLAX_TIER_GATE=
   advisory` (warns per-request); the node-binding needing the real PublisherKeyAnchor wired
   (mock-anchor configs); the work-delivery signature gate as backstop. Find any *undocumented*
   bypass of the gate.

---

Follow the `00_INDEX.md` output format: severity + `file:line` + invariant + exploit + why it
survives; a HOLDS/VIOLATED/CONDITIONAL verdict per invariant; the top pre-mainnet fix; and what
needs a live test. Report only.
