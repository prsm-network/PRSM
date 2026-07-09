# Cross-Node On-Chain Settlement — Funded Ceremony Runbook

**Goal:** settle a real cross-node compute job **on-chain**. us (`484f003c`, requester/**payer**)
submits a job; sfo (`d437aa67`, provider/**earner**) serves it, **commits** its earning to the live
**BatchSettlementRegistry** (Base mainnet, chainId 8453), and after the challenge window **finalizes**
— drawing us's on-chain escrow to pay sfo in **on-chain FTNS**. First on-chain settlement of a
cross-node gossip job.

The code path is complete + verified off-chain (sp1401/1402), the provider accumulates its own earning
(sp1405), and the commit/finalize driver endpoints exist (sp1404). What remains is a **funded
ceremony**, gated on operator action.

> **Direction matters (sp1405):** `commitBatch` sets `provider = msg.sender`, and the commit client
> asserts `committer_key_address == provider_address`. So the **provider (sfo) commits** — sfo holds
> the funded settler key. The **requester (us) pays** — us deposits the on-chain escrow the batch draws
> at finalize. Do NOT put the settler key on us.

## Operator / assistant division (do not blur)

- **Operator (you):** fund sfo's settler key + us's escrow, set the env, run the two POST triggers on
  sfo, sign every mainnet tx. The private key lives ONLY in sfo's systemd env / a `chmod 600`
  EnvironmentFile — never in a command argument, a committed file, or this doc.
- **Assistant:** runs the read-only preflight + Basescan/`eth_call` verification, reads balances,
  confirms each step. Never funds, never signs, never holds a key.

## Live F-bundle (Base mainnet, chainId 8453) — verified §0

| Contract | Address |
|---|---|
| BatchSettlementRegistry | `0x12a01F6C487d765af389bC7D95D90b3136a391F2` |
| EscrowPool | `0x4e93a04b3A0C5063FE584980e6c2B1429495EEa1` |
| StakeBond | `0x21B5de0f65B9273A715C6a02b7085a8ABE8adA72` |
| Ed25519Verifier | `0x9d369312bf3b502Bc07c5859a18f7158c22A31e1` |

Registry: **not paused**, owner = Foundation Safe `0x91b0e6f8…5791`, chainId 8453 (assistant-verified).

---

## 0. Preconditions (assistant, read-only) — ✅ done

- [x] Registry live: `paused()==false`, `owner()==Foundation Safe`, chainId 8453.
- [x] us + sfo peered and settling **off-chain** (baseline): a `/compute/submit` on us is served by
      sfo and sfo's `/balance` rises (sp1401/1402/1405 verified: us 91→90, sfo 101→102).
- [x] The served result carries a signed `shard_receipt` (`provider_id == d437aa67`) — sp1403.

## 1. Fund (operator)

- **sfo's settler key** — an Ethereum key. It needs **Base ETH** for gas (commit + finalize; cents).
  This key IS sfo's on-chain payee. Record only the address: `SFO_OPERATOR_ADDRESS = 0x…`.
- **us's escrow** — us is the payer, so **us must have FTNS deposited in the EscrowPool** ≥ the job
  budget (e.g. 1 FTNS), under us's payer address `US_OPERATOR_ADDRESS = 0x…`. Deposit path: see
  `scripts/deposit_escrow.py` (key from env ONLY):
  `PRSM_NETWORK=mainnet python scripts/deposit_escrow.py --check` (read-only) then
  `PRSM_NETWORK=mainnet ESCROW_DEPOSIT_KEY=0x<us payer key> python scripts/deposit_escrow.py 1.0`.
  Assistant verifies via `--check` / `eth_call`.

## 2. Configure the nodes (operator)

On **sfo** (`146.190.175.239`) — the earner/committer. Key in a root-only EnvironmentFile, never a unit:

```
# /etc/systemd/system/prsm-operator.service.d/zzz-compute.conf
Environment=PRSM_ONCHAIN_SETTLEMENT=1
Environment=PRSM_OPERATOR_ADDRESS=0x<SFO_OPERATOR_ADDRESS>   # == the settler key's address
EnvironmentFile=/etc/prsm-operator/settler.env               # chmod 600: FTNS_WALLET_PRIVATE_KEY=0x...
```

On **us** (`159.203.129.218`) — the payer. It only advertises its payer address (no settler key):

```
Environment=PRSM_OPERATOR_ADDRESS=0x<US_OPERATOR_ADDRESS>    # == the escrow depositor
```

`systemctl daemon-reload && systemctl restart prsm-operator.service` on both; confirm they re-peer.

## 3. Read-only preflight on sfo (assistant)

```
PRSM_NETWORK=mainnet PRSM_ONCHAIN_SETTLEMENT=1 PRSM_OPERATOR_ADDRESS=0x<SFO_OPERATOR_ADDRESS> \
  python -m prsm.settlement.go_live_preflight
```

Expect GO: onchain-settlement-enabled, F-registry-not-retired, paused==false, **key-controls-provider**
(sfo's key address == `PRSM_OPERATOR_ADDRESS`), settler funding (sfo ETH gas). Any NO-GO stops the
ceremony. `GET http://127.0.0.1:8002/admin/settlement/onchain/status` on **sfo** → `onchain_settlement: on`.

## 4. Run a cross-node job → verify accumulation (operator + assistant)

Submit on **us** (the payer):

```
curl -sX POST http://127.0.0.1:8002/compute/submit -H 'content-type: application/json' \
  -d '{"job_type":"inference","payload":{"prompt":"on-chain canary","max_tokens":5},"ftns_budget":1.0}'
```

Confirm `provider_id == d437aa67` (sfo served it). On **sfo**, `GET /admin/settlement/onchain/status`
shows a ready batch accumulated (sp1405: sfo accumulated its OWN earning, requester = us's payer
address from the offer). Off-chain balances still move (sfo +1) — the on-chain commit is additive.

## 5. Commit on sfo (operator triggers; assistant verifies)

> **What actually drives a commit (sp1410 correction).** With `PRSM_ONCHAIN_SETTLEMENT=1` the node
> ALWAYS runs the sp1038 settlement poll loop (`Node._settlement_poll_loop`, every
> `PRSM_SETTLEMENT_POLL_INTERVAL_S`, default 600s) — there is no flag that disables it. Each cycle
> commits every batch that has crossed an accumulator threshold and finalizes every batch past its
> challenge window. It is NOT a no-op you opt into.
>
> A ceremony batch stays put only because it crosses no threshold: the defaults are **1000 receipts
> OR 1 hour OR 100 FTNS**, and a lone 1-FTNS receipt hits none of them — for the first hour. After
> that the poll loop commits it unattended. The two POSTs below force an *immediate* cycle; they are
> not what makes a commit possible.
>
> So there are two clean ways to run this canary:
> - **Deterministic (preferred, sp1410):** set `PRSM_SETTLEMENT_COUNT_THRESHOLD=1` on sfo so the
>   single share is ready at once, and let the poll loop commit it (or POST to go now).
> - **Explicit one-shot:** keep the defaults and POST `commit-ready?force=1` (sp1407), which commits
>   every pending batch regardless of thresholds.
>
> Either way, do the ceremony inside the first hour if you want the commit to be *yours* rather than
> the loop's.

```
curl -sX POST http://127.0.0.1:8002/admin/settlement/onchain/commit-ready   # ON SFO
# ...or, for a single small canary batch that has crossed no threshold:
curl -sX POST 'http://127.0.0.1:8002/admin/settlement/onchain/commit-ready?force=1'
```

Returns the committed batch: `batch_id`, `tx_hash`, `provider_address` (sfo's), `requester_address`
(us's), `receipt_count`, `total_value_ftns`. **Capture `batch_id` + `tx_hash`.**

Assistant verifies read-only: `tx_hash` mined (status 1), targets `0x12a01F6C…391F2`; `getBatch(batch_id)`
shows `provider == msg.sender == SFO_OPERATOR_ADDRESS`, `requester == US_OPERATOR_ADDRESS`, status PENDING.

## 6. Finalize after the challenge window (operator triggers on sfo; assistant verifies)

Assistant polls read-only until finalizable:
`SETTLER_KEY unset PRSM_NETWORK=mainnet python scripts/finalize_batch_base_sepolia.py <batch_id> --check`
prints `isFinalizable` + `secondsUntilFinalizable` (no key for `--check`). When finalizable:

```
curl -sX POST http://127.0.0.1:8002/admin/settlement/onchain/finalize-ready   # ON SFO
```

Assistant verifies: `FTNS.Transfer(EscrowPool → SFO_OPERATOR_ADDRESS, value)`; sfo's on-chain FTNS up
by the batch value; us's escrow down by the same. Conservation: escrow-out == provider-in.

## 7. Success criteria

- [ ] Batch committed on `0x12a01F6C…391F2` with `provider == sfo`, `requester == us`.
- [ ] Finalized after the window; EscrowPool paid sfo on-chain (verified `Transfer`).
- [ ] Off-chain ledgers still consistent (sp1401/1402 unaffected); no double-pay.
- [ ] `getBatch(batch_id)` status FINALIZED.

## Rollback / safety

- **Before commit:** remove sfo's on-chain env (`PRSM_ONCHAIN_SETTLEMENT`, `FTNS_WALLET_PRIVATE_KEY`)
  + restart → off-chain-only. No on-chain state touched.
- **After commit, before finalize:** a committed batch can be left unfinalized (no harm) or challenged
  within the window; escrow is NOT drawn until finalize.
- **After finalize:** on-chain + irreversible — why §5–§6 are operator-triggered one-shots (no auto
  loop) and each is verified before the next.
- us's deposited-but-unused escrow is recoverable via the EscrowPool withdraw path; never stranded.

## Settlement env quick-ref (sfo)

| Var | Default | Effect |
|---|---|---|
| `PRSM_SETTLEMENT_POLL_INTERVAL_S` | `600` | How often the poll loop commits/finalizes (floor 5s). |
| `PRSM_SETTLEMENT_COUNT_THRESHOLD` | `1000` | Receipts before a batch is committable. `1` = commit each. |
| `PRSM_SETTLEMENT_TIME_THRESHOLD_S` | `3600` | Age before a batch is committable. |
| `PRSM_SETTLEMENT_VALUE_THRESHOLD_FTNS` | `100` | Accrued FTNS before a batch is committable. |
| `PRSM_SETTLEMENT_STATE_FILE` | `~/.prsm/settlement_state.json` | Durable state; `:memory:` disables. |

A malformed threshold is WARNed and ignored (that field keeps its default); settlement stays ON.
Pending receipts are durable across restarts (sp1409), so a redeploy mid-accumulation no longer
loses the provider's un-committed earnings.

## Notes

- **The poll loop is always on** whenever `PRSM_ONCHAIN_SETTLEMENT=1` (sp1038): it commits any
  threshold-ready batch and finalizes any batch past the challenge window, every
  `PRSM_SETTLEMENT_POLL_INTERVAL_S`. The two POSTs (`commit-ready`, `finalize-ready`, sp1404) force
  an immediate cycle on **sfo**; they do not gate whether settlement happens. A first ceremony stays
  controlled only because the default thresholds keep a small batch pending for its first hour — see §5.
- `PRSM_SETTLEMENT_SUPPORTS_ATTESTATION` stays **off** — software nodes, no TEE → legacy
  `commitBatch(bytes32(0))`. Real SEV-SNP attestation carriage is the parallax-path story.
