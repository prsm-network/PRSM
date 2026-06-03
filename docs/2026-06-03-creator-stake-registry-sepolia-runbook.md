# CreatorStakeRegistry — Sepolia rehearsal + mainnet commissioning runbook

**Sprint 976/977.** Commissioning the Vision §14 anti-spam creator stake with real
on-chain teeth. Pre-deploy, `CreatorStakeClient` (prsm/marketplace/creator_stake_client.py)
ran on an in-memory mirror with **no economic teeth** (sp290 scaffold; flagged by
sp970 + the marketplace money review). `CreatorStakeRegistry.sol` is the real
backend: a creator bonds real FTNS as collateral; spam → the stake is slashed.

This runbook covers the **Sepolia rehearsal** (autonomous, no mainnet risk) and the
**mainnet commissioning ceremony** (irreversible — Foundation/multisig). It also
documents the one **design decision** that gates the Python wiring (§5).

---

## 1. Artifact status (done)

- `contracts/contracts/CreatorStakeRegistry.sol` — deploy-ready. Ownable2Step
  (owner = Foundation Safe via acceptOwnership), ReentrancyGuard, Pausable,
  immutable FTNS + immutable slasher, transferFrom-on-stake, 1–30d unbond delay,
  slasher-gated slash → owner-drained Foundation reserve. `creatorStakeOf` returns
  the bonded amount ONLY while BONDED (drops to 0 on requestUnbond).
- `contracts/test/CreatorStakeRegistry.test.js` — 13 Hardhat tests green.
- `contracts/scripts/deploy-creator-stake-registry.js` — env-driven deploy with
  post-deploy invariant checks; mainnet/base **blocked** pending a ratified
  resolution; dry-run validated on the local Hardhat network.

## 2. Prerequisites for the Sepolia rehearsal

- Sepolia ETH in the deployer EOA.
- A testnet FTNS ERC-20 (deploy `MockERC20` via `scripts/deploy-mock-ftns.js`, or
  reuse an existing testnet token) → `FTNS_TOKEN_ADDRESS`.
- A slasher address (any non-zero EOA you control for the rehearsal) →
  `CREATOR_STAKE_SLASHER`.
- `SEPOLIA_RPC_URL`, `PRIVATE_KEY`, and (for verify) `ETHERSCAN_API_KEY`.

## 3. Rehearsal steps

```
# 3.1 Dry-run on the local Hardhat node first (validates script + invariants, no gas):
FTNS_TOKEN_ADDRESS=0x...01 CREATOR_STAKE_SLASHER=0x...02 \
  npx hardhat run scripts/deploy-creator-stake-registry.js --network hardhat

# 3.2 Deploy to Sepolia:
PRIVATE_KEY=0x... SEPOLIA_RPC_URL=https://... \
FTNS_TOKEN_ADDRESS=0x<testnet-ftns> CREATOR_STAKE_SLASHER=0x<your-slasher> \
AUTO_VERIFY=1 ETHERSCAN_API_KEY=... \
  npx hardhat run scripts/deploy-creator-stake-registry.js --network sepolia
```

Then exercise the full lifecycle against the deployed Sepolia contract (via a
Hardhat console script or cast):

1. **stake** — creator approves FTNS, calls `stake(amount)`; confirm
   `creatorStakeOf(creator) == amount` and the contract's FTNS balance rose.
2. **eligibility-drops-on-exit** — `requestUnbond()`; confirm `creatorStakeOf`
   immediately returns `0` (the anti-game property).
3. **withdraw-delay** — `withdraw()` reverts before the delay; after
   `unbondDelaySeconds`, it returns the funds.
4. **slash** — from the slasher, `slash(creator, amount, "spam")`; confirm the
   bonded amount dropped and `foundationReserveBalance` rose; non-slasher reverts.
5. **reserve drain** — `setFoundationReserveWallet(w)` then
   `drainFoundationReserve()`; confirm `w` received the proceeds.

Record the Sepolia address + tx hashes (this is the "live exercise" evidence the
mainnet resolution will cite, mirroring the ProvenanceRegistryV2 / A-08 ceremonies).

## 4. Mainnet commissioning ceremony (GATED — do NOT run autonomously)

Irreversible; requires a ratified resolution + the Foundation 2-of-3 multisig.

1. Obtain a council resolution authorizing the Base mainnet deploy on the Sepolia
   evidence (mirror PRSM-CR-2026-05-06-2's pattern for ProvenanceRegistryV2).
2. Remove the `base` guard in the deploy script in that ratified PR.
3. Deploy to Base with `CREATOR_STAKE_SLASHER` = the governance/Foundation slash
   authority and `FTNS_TOKEN_ADDRESS` = the canonical Base FTNS.
4. `transferOwnership(<Foundation Safe>)`; the Safe calls `acceptOwnership()`
   (Ownable2Step 2-step handover).
5. Copy the address into `prsm/deployments/contract_addresses.json` under
   `base.creator_stake_registry`; set `CREATOR_STAKE_REGISTRY_ADDRESS` in operator env.

## 5. ★ DESIGN DECISION required before the Python wiring (creator_id ↔ wallet)

The contract is **address-keyed** (`msg.sender` bonds; `creatorStakeOf(address)`),
but `CreatorStakeClient.is_high_tier_eligible(creator_id)` is called with the
reputation/provenance **`creator_id`**, which today is the uploader's **node_id**
(sp964), while §14's fingerprint registry keys the canonical creator by
**`creator_eth_address`**. So `creator_id → ETH address` is currently
under-determined. **Decide one before wiring the Python backend:**

- **(A) Adopt `creator_eth_address` as the stake-gate creator identity** — pass the
  eth address (already threaded through uploads, sp245) to `is_high_tier_eligible`,
  so the web3 backend's `balance_of` is a direct `creatorStakeOf(addr)`. Simplest;
  aligns the stake gate with the §14 fingerprint registry's canonical-creator key.
- **(B) Add a verified `node_id → wallet` binding** (mirror sp788 operator
  delegation) and resolve it in `balance_of`. Heavier; keeps node_id as the
  reputation key.

Recommendation: **(A)** — the §14 surfaces (fingerprint dedup, content royalty)
already treat the eth address as the canonical creator, and it avoids a second
delegation scheme.

## 6. Python wiring follow-on (after §5 is decided)

- A web3 read-backend implementing `_StakeBackend.balance_of(creator)` →
  `CreatorStakeRegistry.creatorStakeOf(<addr per §5>)` (read-only; degrade-to-0 on
  RPC error, mirroring OnChainStakeReader). `stake()` is a **creator-wallet**
  action (not server-executable) and `slash()` is **slasher-only** — the web3
  backend should reject those as non-server actions rather than mutate (the
  in-memory scaffold's mutate-anything semantics must NOT carry to the real
  backend).
- Wire `CreatorStakeClient.from_env` to construct that backend when
  `CREATOR_STAKE_REGISTRY_ADDRESS` + `BASE_RPC_URL` are set.
- Tests: mock-web3 backend balance read; the from_env wiring; the
  reject-server-mutation behavior.
