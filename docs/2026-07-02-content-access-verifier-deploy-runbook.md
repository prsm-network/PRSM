# ContentAccessVerifier deploy runbook (Tier B/C paid-decrypt)

Deploys the production `ContentAccessVerifier` (the IRoyaltyPaymentVerifier that gates the paid-key
flow). **The user signs every transaction; the assistant is read-only.** Keys live in the operator's
own env, never in chat/argv.

**Why this is low-risk:** `ContentAccessVerifier` is **not Ownable**, is **not upgradeable**, and
**holds no funds at deploy** — so there is no ownership ceremony, no multisig, no migration. It's a
single plain deploy tx, verified read-only. The code passed 8 Hardhat tests (incl. the end-to-end
through real KeyDistribution) and two adversarial reviews; the deploy script dry-ran green on a local
node.

## 0. Prerequisites

- A funded deployer key (Base ETH for gas) exported as `PRIVATE_KEY` in the operator's shell.
- An RPC URL (`BASE_RPC_URL` for mainnet, `BASE_SEPOLIA_RPC_URL` for testnet).
- The constructor addresses (pass **explicitly** — a wrong registry sends fees to the wrong creator):
  - **Base mainnet (chain 8453):** `FTNS_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5`,
    `REGISTRY_ADDRESS=0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf` (ProvenanceRegistry **V2** — the
    registry the RoyaltyDistributor uses, so access fees credit the same authoritative creator).
  - **Base Sepolia:** `FTNS_ADDRESS=0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a`; there is **no
    ProvenanceRegistry on Sepolia yet**, so step 1 deploys one first.

## 1. (Recommended) Testnet dress-rehearse the live flow on Base Sepolia

```bash
# 1a. deploy a ProvenanceRegistry V2 to Sepolia (needed as the fee-payee lookup):
PRIVATE_KEY=0x… BASE_SEPOLIA_RPC_URL=https://… \
  npx hardhat run scripts/deploy-provenance-registry-v2.js --network base-sepolia
#   → note the printed address as SEPOLIA_REGISTRY

# 1b. deploy ContentAccessVerifier against it:
PRIVATE_KEY=0x… BASE_SEPOLIA_RPC_URL=https://… \
FTNS_ADDRESS=0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a REGISTRY_ADDRESS=$SEPOLIA_REGISTRY \
AUTO_VERIFY=1 ETHERSCAN_API_KEY=… \
  npx hardhat run scripts/deploy-content-access-verifier.js --network base-sepolia

# 1c. live smoke (publisher + consumer, small dataset): a publisher registers content in the Sepolia
#     registry, publish_paid_content deposits the commitment (naming the CAV as the royalty verifier)
#     + serves the key via a PRSM_PAID_KEY_SERVE node; a consumer runs `prsm content unlock` and gets
#     the plaintext. Confirms pay → gated serve → fetch+verify → decrypt against a real chain.
```

## 2. Base mainnet deploy (the ceremony)

```bash
PRIVATE_KEY=0x… BASE_RPC_URL=https://… \
FTNS_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
REGISTRY_ADDRESS=0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf \
AUTO_VERIFY=1 ETHERSCAN_API_KEY=… \
  npx hardhat run scripts/deploy-content-access-verifier.js --network base
```

The script fails-closed if `ftns()`/`registry()` don't match, `verifyPayment(unknown)` isn't false,
or `totalClaimable()` isn't 0. Record the printed `ContentAccessVerifier` address + the manifest.

## 3. Verify (read-only — assistant can run this)

```bash
BASE_RPC_URL=https://… CAV_ADDRESS=0x<deployed> \
FTNS_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
REGISTRY_ADDRESS=0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf \
  npx hardhat run scripts/verify-content-access-verifier.js --network base
```

Confirm the contract is verified on Basescan (source published).

## 4. Wire config

- Add the address to `prsm/deployments/contract_addresses.json` under `base.content_access_verifier`
  and commit.
- **Consumers/CLI:** `PRSM_CONTENT_ACCESS_VERIFIER=0x<deployed>` (or `--verifier-address`).
- **Serve operators** (publishers running the paid-key endpoint): also
  `PRSM_PAID_KEY_SERVE=1` and `PRSM_PAID_KEY_STORE_FILE=/var/lib/prsm/paid_keys.json` (durable, so
  paid buyers survive a restart — R5 HIGH fix).

## 5. Production smoke

A publisher publishes one small Tier-B dataset (registers it in the V2 registry, publishes with the
CAV as the royalty verifier); a consumer runs `prsm content unlock <hash> --fee <F>` and gets the
plaintext. Then the paid-decrypt path is live.

## Deferred (not blocking)

F9/F2 on-chain squatting (bind `depositKey`/fee-payee to the publisher) — the Option-B fast-follow;
confirmed low-severity by both reviews. Track before broad third-party publishing.
