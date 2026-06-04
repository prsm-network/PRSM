# CreatorStakeRegistry — rehearsal + mainnet commissioning runbook

**Sprints 976–981.** Commissioning the Vision §14 anti-spam creator stake with real
on-chain teeth. Pre-deploy, `CreatorStakeClient` (prsm/marketplace/creator_stake_client.py)
ran on an in-memory mirror with **no economic teeth** (sp290 scaffold; flagged by
sp970 + the marketplace money review). `CreatorStakeRegistry.sol` is the real
backend: a creator bonds real FTNS as collateral; spam → the stake is slashed; the
stake stops counting toward tier the instant the creator begins to withdraw.

This is the **single authoritative commissioning doc**. The whole engineering side
is done and verified; what remains is the **irreversible ceremony** (governance
resolution + Foundation 2-of-3 multisig), which is **operator-run with guidance** —
nothing below the GATE markers in §7 is run autonomously. Read §7 (the consolidated
go/no-go) end to end before starting; §1–§6 are reference, §8 is the resolution to
ratify.

---

## 1. Artifact status — engineering side COMPLETE

| Piece | Where | Status |
|---|---|---|
| Money-custody contract | `contracts/contracts/CreatorStakeRegistry.sol` | ✅ sp976; hardened sp979 |
| Contract tests | `contracts/test/CreatorStakeRegistry.test.js` | ✅ **19 Hardhat green** |
| Deploy script | `contracts/scripts/deploy-creator-stake-registry.js` | ✅ sp977; base/mainnet **blocked** |
| Rehearsal driver | `contracts/scripts/rehearse-creator-stake-registry.js` | ✅ sp980; **14/14 invariants on hardhat** |
| Pre-deploy security audit | workflow w9riufvls (FIX-FIRST) | ✅ 6 findings fixed sp979 |
| Python read-backend | `prsm/economy/web3/creator_stake_registry_backend.py` | ✅ sp978 |
| Stake-gate wiring | `prsm/marketplace/creator_stake_client.py` | ✅ sp978 (decision A) |
| Canonical address slot | `prsm/config/networks.py` (`creator_stake_registry`) | ✅ sp981 |

**Contract posture** (mirrors the battle-tested StakeBond): Ownable2Step (owner =
Foundation Safe via `acceptOwnership`), ReentrancyGuard, Pausable with
`whenNotPaused` on **every** value/state mutation (stake / requestUnbond / withdraw
/ slash / setFoundationReserveWallet / drainFoundationReserve — the sp979 audit
caught that these were missing), immutable FTNS + immutable slasher (non-zero
enforced at construction), transferFrom-on-stake, 1–30d unbond delay, slasher-gated
slash → owner-drained Foundation reserve, and a `WalletNotContract` guard so the
reserve wallet must be a contract (the Foundation Safe), never an EOA.
`creatorStakeOf` returns the bonded amount **only while BONDED** (drops to 0 on
`requestUnbond`), which closes the stake→get-tier→spam→unstake game.

**sp979 audit findings (all fixed, pinned by tests):** 5 HIGH — every
`whenNotPaused` guard had been dropped, so a paused (emergency-frozen) contract
still allowed unbond/withdraw/slash/reserve-redirect/drain; 1 MEDIUM —
`setFoundationReserveWallet` accepted an EOA (one-typo path to misrouting slashed
funds). The audit-first gate is the reason these never reached mainnet.

---

## 2. Prerequisites for the Sepolia rehearsal

- **Base Sepolia** ETH in the deployer EOA (recommended — it's the same OP-stack L2
  as Base mainnet, so the closest bytecode/opcode/gas analog. Ethereum Sepolia also
  works since the Solidity is identical; just swap the network + RPC var below).
- `BASE_SEPOLIA_RPC_URL` (defaults to `https://sepolia.base.org` if unset),
  `PRIVATE_KEY`, and — for explorer verification — `ETHERSCAN_API_KEY`.
- **Nothing else.** The rehearsal driver is self-contained: it deploys a
  `MockERC20` (testnet FTNS stand-in) + a fresh 1-day-delay registry (deployer =
  owner = slasher) and mints/approves itself. To rehearse against an
  already-deployed registry instead, pass `REGISTRY_ADDRESS` (+ `FTNS_TOKEN_ADDRESS`);
  to rehearse the role separation, pass a distinct `CREATOR_PRIVATE_KEY`.

## 3. Rehearsal — run the driver (replaces the old manual lifecycle)

The `rehearse-creator-stake-registry.js` driver exercises the full value-movement
lifecycle and **asserts every economic invariant with an explicit PASS/FAIL line**,
aborting with "do NOT proceed to the mainnet ceremony" on any failure. It proves the
**deployed bytecode** behaves as the audited source claims, against a real RPC.

Each command below is a SINGLE line (copy-paste whole). Fill in the `<...>`.

3.1 — Local dry-run (one shot, full lifecycle incl. post-delay withdraw via time-travel). Expect `REHEARSAL RESULT: 14 passed, 0 failed`:
```
npx hardhat run scripts/rehearse-creator-stake-registry.js --network hardhat
```

3.2 — Live testnet, PHASE 1 (bond / slash / drain / unbond + all reverts). Prints the registry + token addresses and the exact PHASE-2 command + the eligible-at unix:
```
PRIVATE_KEY=0x<deployer> BASE_SEPOLIA_RPC_URL=https://sepolia.base.org npx hardhat run scripts/rehearse-creator-stake-registry.js --network base-sepolia
```

3.3 — Live testnet, PHASE 2 (after the unbond delay elapses — ~24h with the 1-day min). Expect `withdraw returns the remaining bonded FTNS` + status flips to WITHDRAWN:
```
PRIVATE_KEY=0x<deployer> BASE_SEPOLIA_RPC_URL=https://sepolia.base.org REGISTRY_ADDRESS=0x<from phase 1> FTNS_TOKEN_ADDRESS=0x<from phase 1> REHEARSAL_PHASE=withdraw npx hardhat run scripts/rehearse-creator-stake-registry.js --network base-sepolia
```

(On Ethereum Sepolia instead: use `SEPOLIA_RPC_URL=...` and `--network sepolia`.)
A live chain cannot fast-forward time, so the time-gated `withdraw` is the only step
split into a second phase. Record the testnet registry address + the phase-1/phase-2
tx hashes — this is the "live exercise" evidence the mainnet resolution (§8) cites,
mirroring the ProvenanceRegistryV2 / A-08 ceremonies.

## 4. Mainnet commissioning ceremony — see §7 for the gated, ordered guide.

## 5. Design decision (creator_id ↔ wallet) — ✅ DECIDED: (A) creator ETH address

The contract is address-keyed (`creatorStakeOf(address)`). The stake gate keys on
the creator's **ETH address** — the same canonical identity §14's fingerprint
registry and content-royalty layer already use — so `balance_of` is a direct
`creatorStakeOf(addr)`, with no second delegation scheme. Implemented in sp978:
`apply_stake_gate(tier, creator_eth_address, client)` (a falsy address → demote, as
an unbonded creator can't have stake); `/content/search` tier-filter passes
`r.creator_eth_address`. The rejected alternative (a node_id → wallet binding) was
heavier and is not needed.

## 6. Python wiring — ✅ DONE (sp978 + sp981)

- `CreatorStakeRegistryBackend` (web3): `balance_of(eth_addr)` → `creatorStakeOf`,
  degrade-to-0 on RPC error (fail-closed eligibility). `stake()` and `slash()` are
  **not server-executable** (stake is a creator-wallet action; slash is
  slasher-only) — the backend raises `CreatorStakeServerActionError` rather than
  mutate, so the in-memory scaffold's credit-anyone semantics never reach a
  commissioned node.
- `CreatorStakeClient.from_env` resolves the address + RPC through
  `prsm.config.networks.resolve_endpoints` (sp981) — the same unified path every
  other deployed contract uses. The toothless in-memory free-stake fallback is
  closed: when a real backend is wired it is authoritative; in-memory is strictly
  the uncommissioned dev path.
- **Activation is a one-line edit** (see §7 step 7): record the address in
  `networks.py`; the gate goes live against the network's default Base RPC with no
  separate RPC env var.

---

## 7. ★ CONSOLIDATED CEREMONY GUIDE (go / no-go at each gate)

Ordered, top to bottom. **GATE** lines are hard stops: do not pass until the
checklist is satisfied. Steps marked **[OPERATOR SIGNS]** are the irreversible /
multisig actions — the operator (Foundation) performs them; the assistant only
prepares and verifies. Everything above the first GATE is reversible / autonomous.

**Step 0 — confirm the engineering gate (autonomous, already true).**
- ✅ 19 Hardhat contract tests green; ✅ pre-deploy audit FIX-FIRST findings all
  closed (sp979); ✅ rehearsal driver 14/14 on hardhat; ✅ Python wiring + canonical
  address slot landed (sp978/981). Re-run to confirm before proceeding:
  `cd contracts && npx hardhat test test/CreatorStakeRegistry.test.js` (expect 19
  passing) and `npx hardhat run scripts/rehearse-creator-stake-registry.js --network hardhat`
  (expect 14 passed, 0 failed).

**Step 1 — Sepolia rehearsal (autonomous, no mainnet risk).** Run §3.2 then §3.3.
- **GATE 1:** every rehearsal invariant PASSED on the live testnet (phase 1 + phase
  2). Record the testnet address + tx hashes. If any invariant failed, STOP — the
  deployed bytecode does not match the audited source; do not proceed.

**Step 2 — ratify the council resolution.** Take the draft in §8 to the council /
Foundation. It authorizes the Base deploy on the rehearsal evidence and fixes the
constructor parameters (owner, FTNS, slasher, unbond delay).
- **GATE 2:** resolution ratified + recorded (assign a PRSM-CR-… id). No resolution
  → no deploy.

**Step 3 — unblock the deploy script in a ratified PR.** The deploy + rehearsal
scripts hard-block `base`/`mainnet` by design. In a PR that cites the GATE-2
resolution id, make this EXACT one-clause edit in
`contracts/scripts/deploy-creator-stake-registry.js` (the `if (network === "mainnet" || network === "base")` block, ~line 55):

> change `if (network === "mainnet" || network === "base") {`
> to&nbsp;&nbsp;&nbsp;`if (network === "mainnet") {`

i.e. delete ONLY the ` || network === "base"` substring. Leave the throw body and
the Ethereum-`mainnet` clause intact (Ethereum mainnet is never a target). Do NOT
edit `rehearse-creator-stake-registry.js` — the rehearsal must stay blocked on base.
This PR is the auditable record that the block was lifted deliberately.
- **GATE 3:** PR merged; the deploy params exactly match the ratified resolution
  (`CREATOR_STAKE_OWNER`, `FTNS_TOKEN_ADDRESS` = canonical Base FTNS
  `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`, `CREATOR_STAKE_SLASHER` = the
  governance/Foundation slash authority, `CREATOR_STAKE_UNBOND_DELAY`).

**Step 4 — deploy to Base mainnet. [OPERATOR SIGNS]**

⚠ **SIGNING KEY — read first.** The `base` hardhat network signs via
`MAINNET_PRIVATE_KEY || PRIVATE_KEY` (hardhat.config.js). `contracts/.env` may carry
a stale `MAINNET_PRIVATE_KEY` that would silently WIN over an inline `PRIVATE_KEY` —
deploying this money-custody contract from the wrong EOA. So set
**`MAINNET_PRIVATE_KEY`** (the var `base` actually reads) to your deployer key, and
make sure `CREATOR_STAKE_OWNER` equals that same EOA. (sp990 added a guard that
ABORTS the deploy if the resolved signer ≠ `CREATOR_STAKE_OWNER`, so a mismatch
fails loud before any tx — but set the right key anyway.)

Single-line command (fill in the `<...>`; `MAINNET_PRIVATE_KEY` and
`CREATOR_STAKE_OWNER` are the SAME deployer EOA):

```
MAINNET_PRIVATE_KEY=0x<deployer> CREATOR_STAKE_OWNER=0x<deployer> BASE_RPC_URL=https://mainnet.base.org FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5 CREATOR_STAKE_SLASHER=0x<slash-authority per resolution> CREATOR_STAKE_UNBOND_DELAY=<seconds per resolution> AUTO_VERIFY=1 ETHERSCAN_API_KEY=<key> npx hardhat run scripts/deploy-creator-stake-registry.js --network base
```

The script prints `Deployer: 0x...`, runs post-deploy invariant checks (owner /
ftns / slasher / delay / fresh `creatorStakeOf == 0`), and verifies on Basescan.
Deploy as the deployer EOA first (owner = deployer); ownership transfers to the
Safe in steps 5–6.
- **GATE 4:** BEFORE relying on it, confirm on Basescan that the deploy tx `from`
  address == the printed `Deployer:` == `CREATOR_STAKE_OWNER` == `owner()` on the
  deployed contract (the post-deploy invariant checks do NOT catch a wrong signer).
  Then: contract verified on Basescan; post-deploy invariants printed OK.

**Step 5 — transfer ownership to the Foundation Safe. [OPERATOR SIGNS]** From the
deployer EOA, call `transferOwnership(<Foundation Safe>)`. Ownable2Step: this only
*nominates* (sets `pendingOwner`); authority does NOT move until the Safe accepts,
so a fat-finger here is recoverable (re-call `transferOwnership`, or the wrong
nominee would have to actively accept). Two ways:
- Basescan (simplest): on the verified contract's **Write Contract** tab, connect
  the deployer wallet → `transferOwnership` → `newOwner` = `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`. (selector `0xf2fde38b`)
- Or cast: `cast send <registry> "transferOwnership(address)" 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 --rpc-url https://mainnet.base.org --private-key 0x<deployer>`

**Step 6 — Foundation Safe accepts ownership (2-of-3 multisig). [OPERATOR SIGNS]**
In the Safe UI **Transaction Builder**, queue one tx against the deployed registry
and collect 2-of-3 signatures (Ledger + Trezor + OneKey), as the audit-bundle +
publisher-key-anchor ceremonies did:
- **To:** `<deployed registry address>`
- **Method:** `acceptOwnership()` — **Data:** `0x79ba5097` — **Value:** `0`

- **GATE 5:** `owner()` on-chain == the Foundation Safe (NOT `pendingOwner()` — that
  is pre-acceptance state); the deployer EOA retains zero authority. This is the
  irreversibility line — the contract is now Foundation-governed.

**Step 7 — record the address + go live (autonomous, one-line edit).** Set
`creator_stake_registry="0x<deployed>"` in the `MAINNET` config in
`prsm/config/networks.py` (the placeholder line is already there with a pointer to
this runbook), with a comment citing the GATE-2 resolution id + deploy tx + block,
in the style of the existing `publisher_key_anchor` line. `CreatorStakeClient.from_env`
then resolves it automatically and the gate goes live against the Base default RPC —
no extra env var. (Optionally also pin via `CREATOR_STAKE_REGISTRY_ADDRESS` env.)
- **GATE 6:** `is_commissioned()` returning True is **necessary but NOT sufficient**
  — it is True from address+RPC alone even if the on-chain read backend never built
  (e.g. web3 missing), in which case the gate silently falls back to the in-memory
  scaffold with no teeth. Confirm a real LIVE READ: on a node,
  `GET /admin/formal-verification/check?contract=creator_stake_registry` must return
  non-503 with **INV-CSR-3 (owner() == Foundation Safe) = PASS** (this executes
  against live chain state via the wired checker). That proves both that the address
  resolved AND that the backend reads the real contract — the §14 HIGH-tier gate now
  requires a real on-chain bond.

**Step 8 — set the Foundation reserve wallet (optional, [OPERATOR SIGNS]).** When
slash proceeds need to be drainable, the Safe calls
`setFoundationReserveWallet(<a contract address — the Safe itself>)`. The
`WalletNotContract` guard rejects an EOA. Not required for the gate to function;
do it before the first `drainFoundationReserve`.

## 8. ★ DRAFT COUNCIL RESOLUTION (ratify at GATE 2)

> **PRSM-CR-2026-__-__ — Authorization to deploy CreatorStakeRegistry to Base mainnet**
>
> **Preamble.** Vision §14 requires that high-tier content creators bond slashable
> collateral so that spam-uploading is economically unattractive. The Python
> creator-stake gate has shipped since sprint 290 but ran on an in-memory mirror
> with no on-chain teeth (sprints 970 + the 2026-06-03 marketplace money review).
> `CreatorStakeRegistry.sol` (sprint 976) supplies the real backend.
>
> **Findings.** The council notes that, prior to this authorization:
> 1. The contract passed 19 Hardhat unit tests and a dedicated pre-deploy security
>    audit (FIX-FIRST), whose 6 findings (5 HIGH dropped-pause-guards + 1 MEDIUM
>    EOA-reserve) were all fixed and pinned by tests (sprint 979).
> 2. The full value-movement lifecycle (bond / slash / drain / unbond / time-gated
>    withdraw) was rehearsed on a live testnet via
>    `rehearse-creator-stake-registry.js`, with every economic invariant asserted
>    PASS. Evidence: testnet address `0x____`, tx hashes `____` (attach the GATE-1
>    record).
> 3. The contract mirrors the security posture of the audit-bundle StakeBond
>    contract already governed by the Foundation Safe since 2026-05-07.
>
> **Resolution.** The council authorizes a single deployment of
> `CreatorStakeRegistry` to Base mainnet (chainId 8453) with constructor parameters
> (listed in the contract's positional order — `initialOwner, ftnsAddress,
> initialUnbondDelay, initialSlasher` — so this is a 1:1 cross-check against the
> Solidity signature at GATE 3):
> 1. `initialOwner` = the deployer EOA, to be transferred to the Foundation Safe
>    `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` via Ownable2Step within the same
>    ceremony;
> 2. `ftnsAddress` = the canonical Base FTNS `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`;
> 3. `initialUnbondDelay` = `____` seconds (1–30 days; RECOMMEND ≥ the spam-detection
>    SLA so a creator cannot unbond + withdraw before a warranted slash lands — see
>    the slasher-SLA note in `requestUnbond`). **Correctable in place** post-deploy
>    via `setUnbondDelay` (owner-only, same 1–30d bounds) — a wrong value here is a
>    one-tx fix.
> 4. `initialSlasher` = `0x____` (the governance/Foundation slash authority — RECOMMEND
>    the Foundation Safe or a council-controlled slasher, never a single hot EOA).
>    ⚠ **CONSTRUCTOR-IMMUTABLE — choose deliberately.** There is NO `setSlasher()`.
>    Unlike `initialUnbondDelay` (re-settable above), a wrong `initialSlasher` can
>    ONLY be corrected by REDEPLOYING the contract and migrating every staked
>    creator's collateral. This parameter is fixed FOREVER at deploy and controls
>    who can confiscate creator collateral — treat it with the same care as the
>    ownership-accept (GATE 5). address(0) is rejected at construction.
>
> **Conditions.** (a) The deploy is performed only after this resolution is recorded;
> (b) the `base` guard in the deploy script is removed in a PR citing this
> resolution id; (c) ownership is transferred to and accepted by the Foundation Safe
> via the 2-of-3 hardware multisig before the address is recorded in `networks.py`;
> (d) the foundation reserve wallet, if set, is a contract (the Safe), per the
> `WalletNotContract` guard. No other CreatorStakeRegistry deployment is authorized.

---

### Appendix — why each gate exists (one line each)

- GATE 1 (rehearsal green): deployed bytecode == audited source, proven on a real chain.
- GATE 2 (resolution): a money-custody contract reaches mainnet only by governance act.
- GATE 3 (unblock PR): the `base` block is lifted deliberately + auditably, not silently.
- GATE 4 (verified + invariants): the right bytecode is on-chain with the right params.
- GATE 5 (Safe owns it): no single hot key can pause/slash/drain — the irreversibility line.
- GATE 6 (gate live): the §14 anti-spam stake now has real economic teeth.
