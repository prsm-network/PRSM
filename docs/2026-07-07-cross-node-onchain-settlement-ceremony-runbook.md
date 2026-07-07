# Cross-Node On-Chain Settlement — Funded Ceremony Runbook

**Goal:** settle a real cross-node compute job **on-chain** — us (`484f003c`, requester/payer)
submits a job, sfo (`d437aa67`, provider/payee) serves it, and the payment is committed +
finalized on the live **BatchSettlementRegistry** (Base mainnet, chainId 8453), so sfo earns
**on-chain FTNS**. This is the first on-chain settlement of a cross-node gossip job.

The code path is complete + verified off-chain (sp1401/1402) and bridged to the on-chain
accumulator (sp1403) with an operator-triggered commit/finalize driver (sp1404). What remains is a
**funded ceremony** — it is gated on operator action.

## Operator / assistant division (do not blur)

- **Operator (you):** fund the settler key, deposit FTNS, set the env, run the two POST triggers,
  sign nothing in chat. The private key lives ONLY in `us`'s systemd env / shell — never in a
  command argument, a file that gets committed, or this doc.
- **Assistant:** runs the read-only preflight + Basescan/`eth_call` verification, reads balances,
  and confirms each step. Never funds, never signs, never holds the key.

## Live F-bundle (Base mainnet, chainId 8453)

| Contract | Address |
|---|---|
| BatchSettlementRegistry | `0x12a01F6C487d765af389bC7D95D90b3136a391F2` |
| EscrowPool | `0x4e93a04b3A0C5063FE584980e6c2B1429495EEa1` |
| StakeBond | `0x21B5de0f65B9273A715C6a02b7085a8ABE8adA72` |
| Ed25519Verifier | `0x9d369312bf3b502Bc07c5859a18f7158c22A31e1` |

These are already the canonical config (`prsm/config/networks.py`, sp1300 cutover). The registry is
Foundation-owned + not retired. `commitBatch` settles to `msg.sender`, so the **settler key IS the
provider-of-record** — see §4.

---

## 0. Preconditions (assistant, read-only)

- [ ] us + sfo peered and settling **off-chain** (baseline works): a `/compute/submit` on us is
      served by sfo and sfo's `/balance` increases (sp1401/1402). Confirm before going on-chain.
- [ ] The result carries a signed `shard_receipt` (sp1403) — `GET /compute/job/{id}` shows
      `result.shard_receipt` with `provider_id == d437aa67` and a signature. This is what settles.
- [ ] Registry live + not retired: `paused() == false`, `owner() == Foundation Safe`. (Assistant
      verifies via `eth_call`.)

## 1. Fund the settler key (operator)

The settler key is an **Ethereum key** (the payer/requester = us's on-chain identity). It needs:
- **Base ETH** for gas (a few cents' worth covers commit + finalize).
- **FTNS** to deposit into the EscrowPool (the batch draws from this at finalize).

Fund the key's address on Base mainnet. Record only the **address** (public) here: `PRSM_OPERATOR_ADDRESS = 0x…`.

## 2. Deposit FTNS into the EscrowPool (operator)

The requester's escrow must cover the batch value before finalize. Deposit ≥ the job budget (e.g.
1 FTNS) for the settler key into EscrowPool `0x4e93a04b…EEa1`. Use the same deposit path as the
first Tier-A canary (see `docs/2026-06-29-onchain-settlement-go-live-runbook.md` §3). Assistant
verifies the deposit landed via `eth_call balanceOf`.

## 3. Configure the nodes (operator)

On **us** (`159.203.129.218`), add to the operator drop-in (`zzz-compute.conf`) — the key goes in the
systemd env, never on a command line:

```
Environment=PRSM_ONCHAIN_SETTLEMENT=1
Environment=PRSM_OPERATOR_ADDRESS=0x<settler key's address>
# FTNS_WALLET_PRIVATE_KEY: put in a root-only EnvironmentFile (chmod 600), NOT here in a world-readable unit.
EnvironmentFile=/etc/prsm-operator/settler.env      # contains: FTNS_WALLET_PRIVATE_KEY=0x...
```

On **sfo** (`146.190.175.239`), advertise its payee address so us can resolve it (sp1403 reads it
from the peer's `hardware_profile.operator_address`):

```
Environment=PRSM_OPERATOR_ADDRESS=0x<sfo's payee eth address>
```

`systemctl daemon-reload && systemctl restart prsm-operator.service` on both. Confirm they re-peer.

## 4. Read-only preflight (assistant)

```
PRSM_NETWORK=mainnet PRSM_ONCHAIN_SETTLEMENT=1 PRSM_OPERATOR_ADDRESS=0x… \
  python -m prsm.settlement.go_live_preflight
```

Expect GO on: onchain-settlement-enabled, F-registry-not-retired, paused==false, key-controls-provider
(the key's address == `PRSM_OPERATOR_ADDRESS`), settler funding (ETH + escrow). Any NO-GO here stops
the ceremony — fix before committing. `GET /admin/settlement/onchain/status` on us should now read
`onchain_settlement: on`.

## 5. Run a cross-node job → verify accumulation (operator + assistant)

```
curl -sX POST http://127.0.0.1:8002/compute/submit -H 'content-type: application/json' \
  -d '{"job_type":"inference","payload":{"prompt":"on-chain canary","max_tokens":5},"ftns_budget":1.0}'
```

Confirm it completed with `provider_id == d437aa67` (sfo served it) and `result.shard_receipt` is
present. Then `GET /admin/settlement/onchain/status` — the stats show a pending/ready batch
accumulated (the sp1403 bridge fed it). Off-chain balances still move (sfo +1) — the on-chain commit
is additive.

## 6. Commit the batch on-chain (operator triggers; assistant verifies)

```
curl -sX POST http://127.0.0.1:8002/admin/settlement/onchain/commit-ready
```

Returns the committed batch: `batch_id`, `tx_hash`, `provider_address` (sfo's), `requester_address`
(us's), `receipt_count`, `total_value_ftns`. **Capture the `batch_id` + `tx_hash`.**

Assistant verifies read-only: the `tx_hash` is mined on Base (status 1); on the registry,
`getBatch(batch_id)` shows `provider == msg.sender == settler address`, the merkle root, and status
PENDING. Basescan-confirm the tx targets `0x12a01F6C…391F2`.

## 7. Finalize after the challenge window (operator triggers; assistant verifies)

The batch is finalizable only after its challenge window elapses. Assistant polls read-only:
`GET /admin/settlement/onchain/status` (or the finalize script's `--check`:
`SETTLER_KEY=… PRSM_NETWORK=mainnet python scripts/finalize_batch_base_sepolia.py <batch_id> --check`
prints `isFinalizable` + `secondsUntilFinalizable`, no key needed for `--check`).

When finalizable:

```
curl -sX POST http://127.0.0.1:8002/admin/settlement/onchain/finalize-ready
```

Returns the finalized `batch_id` + `tx_submitted`. Assistant verifies: the EscrowPool drew the
requester's escrow and paid the provider — `FTNS.Transfer(EscrowPool → sfo's payee, value)`, sfo's
on-chain FTNS increased by the batch value, escrow decreased. Conservation: escrow-out == provider-in.

## 8. Success criteria

- [ ] A batch committed on `0x12a01F6C…391F2` naming sfo's payee as provider, us as requester.
- [ ] Finalized after the window; EscrowPool paid sfo's payee on-chain (verified `Transfer`).
- [ ] Off-chain ledgers still consistent (sp1401/1402 unaffected); no double-pay.
- [ ] `getBatch(batch_id)` status FINALIZED.

## Rollback / safety

- **Before commit:** revert is trivial — remove the on-chain env (`PRSM_ONCHAIN_SETTLEMENT`,
  `FTNS_WALLET_PRIVATE_KEY`) + restart; the node returns to off-chain-only. No on-chain state touched.
- **After commit, before finalize:** a committed batch can be left unfinalized (no harm) or
  challenged within the window if wrong; escrow is not drawn until finalize.
- **After finalize:** on-chain + irreversible. This is why §6–§7 are operator-triggered one-shots
  (there is deliberately no auto-commit loop) and each is verified before the next.
- Deposited-but-unused escrow is recoverable per the EscrowPool withdraw path; it is never stranded.

## Notes

- There is **no auto-commit loop** by design — a first ceremony must be controlled. The two POST
  triggers (`commit-ready`, `finalize-ready`, sp1404) each drive exactly one cycle.
- `PRSM_SETTLEMENT_SUPPORTS_ATTESTATION` (sp1299) stays **off** here — these are software nodes with
  no TEE; leave it off so commits use legacy `commitBatch(bytes32(0))`. Real SEV-SNP attestation
  carriage is the parallax-path story, not this local-inference canary.
