# On-Chain Settlement Go-Live Runbook

**The first real production activation of on-chain settlement.** All the machinery —
the F-capable audit bundle (BatchSettlementRegistry + EscrowPool + StakeBond +
Ed25519Verifier), the settlement client, the §7 attestation chain — is deployed,
Foundation-owned, Basescan-verified, and the canonical config points at the F bundle
(sp1300 cutover). What has **never** happened in production: a funded settler key
committing a real batch on-chain. This runbook is that activation.

It is GATED on operator action: funding a settler key + opting in. The assistant runs
the read-only preflight + verifies; the operator funds the key, sets the env, and
starts the node.

---

## 0. What this activates

- `PRSM_ONCHAIN_SETTLEMENT=1` turns the settlement client from VIEW-ONLY (local
  accumulation) into a WRITE-CAPABLE client that commits/finalizes batches on the
  live BatchSettlementRegistry.
- With `PRSM_SETTLEMENT_SUPPORTS_ATTESTATION=1` (sp1299, fail-safe), each batch
  commits a **real AMD SEV-SNP attestation** via `commitBatchWithAttestation`
  (roadmap F). Off → legacy `commitBatch(bytes32(0))`, no on-chain attestation.

---

## 1. Prerequisites (operator)

1. **A settler key** that controls the node's `provider_address` — the same eth
   address the node advertises. `commitBatch` settles to `msg.sender`, so the key
   MUST equal `provider_address` or funds settle to the wrong party (the client
   refuses to build on a mismatch). Set `FTNS_WALLET_PRIVATE_KEY` (0x-prefixed) and
   either let `provider_address` derive from it or pin `PRSM_OPERATOR_ADDRESS`.
2. **Base ETH in that key** for gas (commit + finalize are two txs per batch).
3. **The F bundle resolved** — default since sp1300; no env needed unless overriding.
   A PAYG RPC is recommended (`BASE_RPC_URL`) for reliable reads/writes.

> Never paste the private key into chat or shared history. Set it in the node's
> environment directly (e.g. a root-owned env file).

---

## 2. Preflight (read-only — the go/no-go gate)

Run the sp1301 preflight; it never moves money or broadcasts a tx. `go` is true iff
zero FAIL findings (WARN/INFO are advisory).

```bash
PRSM_ONCHAIN_SETTLEMENT=1 \
  FTNS_WALLET_PRIVATE_KEY=0x…           # the settler key (or set in the node env) \
  PRSM_OPERATOR_ADDRESS=0x…             # the node's provider_address \
  PRSM_SETTLEMENT_SUPPORTS_ATTESTATION=1 \
  BASE_RPC_URL=<PAYG endpoint> \
  python -m prsm.settlement.go_live_preflight
# exit 0 = GO; exit 1 = NO-GO (one line per FAIL)
```

It checks: `provider_address` resolves · `PRSM_ONCHAIN_SETTLEMENT` on · settler key
**controls** provider_address · the client builds **write-capable** · the resolved
registry is the F bundle (not the retired pre-F one) · the registry is **active**
(`paused()==false`) · it exposes `commitBatchWithAttestation` · the settler is
**funded** with Base ETH · the EscrowPool resolves · the attestation flag vs the
on-chain surface (so you know whether the first batch commits a real attestation).

Do not proceed while any FAIL stands.

---

## 3. Activation

1. Set the env from §1 on the node (persisted), confirm preflight = **GO**.
2. **Start the node** with settlement on. The poll loop
   (`run_settlement_poll_cycle`) now commits ready batches + finalizes past the
   challenge window.
3. **First-batch canary.** Drive one real inference→settle cycle, then watch the
   poll loop commit + finalize it:
   - confirm a `BatchCommitted` event on the F registry (the new batchId),
   - after the challenge window, confirm `finalizeBatch` lands,
   - with the attestation flag on, confirm `BatchAttestationCommitted`
     (topic0 `0xec923112ccc386fa91e7116abfe5da0211d8908195bb5d41e644c8a0c79222e3`)
     and that the committed measurement is a **real** SEV-SNP quote (not dev-only).
4. **Soak.** Let several batches commit + finalize cleanly before scaling up.

---

## 4. Attestation flip ordering

For the very first activation it is safe to set
`PRSM_SETTLEMENT_SUPPORTS_ATTESTATION=1` from the start (there is no legacy traffic
to migrate). If you prefer maximum caution, go live with it OFF, confirm a clean
legacy commit+finalize, then flip it on and re-run the preflight (the `attestation_flag`
check should read PASS). The sp1299 fail-safe will keep it OFF if the registry
surface can't be confirmed, so a misconfiguration can't send reverting commits.

---

## 5. Abort / rollback

- Preflight NO-GO → fix the flagged item; do not start settlement.
- A committed batch won't finalize / repeated commit errors → set
  `PRSM_ONCHAIN_SETTLEMENT=0` and restart (reverts to VIEW-ONLY local accumulation;
  durable state preserves any committed-but-unconfirmed batch for reconcile on
  re-enable). Investigate before re-enabling.
- The TEE attestation is wrong/unavailable → set
  `PRSM_SETTLEMENT_SUPPORTS_ATTESTATION=0` (legacy commitBatch keeps settling) while
  the TEE runtime is fixed; no settlement downtime.

---

## 6. References

- `prsm/settlement/go_live_preflight.py` (sp1301) — the preflight harness.
- `docs/2026-06-26-tee-tier3-f-activation-deploy-runbook.md` — the F bundle deploy +
  cutover (Phases 1–4, complete 2026-06-29).
- `prsm/settlement/client_wiring.py` — the build + the sp1299 attestation config flip.
