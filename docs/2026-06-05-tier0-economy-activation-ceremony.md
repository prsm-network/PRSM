# Tier 0 — Economy Activation Ceremony (grant MINTER_ROLE to the EmissionController)

**Status: READY for the Foundation 2-of-3 multisig. One transaction.**
Prepared + verified by the assistant on 2026-06-05 against live Base-mainnet state.
The assistant does **not** sign — the irreversible step is the Foundation Safe
execution, marked **[OPERATOR SIGNS]** below.

## Why this is the whole of Tier 0 (corrected from the readiness audit)

The 2026-06-05 readiness audit said the economy was inert because of "an
un-executed multi-sig `acceptOwnership` **and** `MINTER_ROLE` never granted."
Verified live on Base mainnet (`base-rpc.publicnode.com`), **only the second half
is true**:

| Check (live, 2026-06-05) | Result |
|---|---|
| FTNS token — Foundation Safe holds `DEFAULT_ADMIN_ROLE` | ✅ true (Safe *can* grant) |
| EmissionController / CompensationDistributor / EscrowPool / StakeBond / SettlementRegistry — `owner()` | ✅ Foundation Safe (all) |
| …`pendingOwner()` | ✅ `0x0` (no pending — **`acceptOwnership` already done; the audit's "pending" claim was wrong**) |
| EmissionController `ftnsToken()` / `authorizedDistributor()` / `paused()` | ✅ correct FTNS · already set to the CompensationDistributor · `false` |
| CompensationDistributor `ftnsToken()` / `emissionController()` | ✅ both correct (the audit's "misconfigured" claim was a bad storage-slot guess) |
| **EmissionController has `MINTER_ROLE`** | ❌ **false — the one and only blocker** |

So every prerequisite is already in place. The **single missing thing** is that
the EmissionController cannot mint, because it lacks `MINTER_ROLE` on the FTNS
token. The mint path is `CompensationDistributor.pullAndDistribute()` →
`EmissionController.mintAuthorized()` → `FTNS.mintReward(distributor, amount)`;
`mintReward` is `onlyRole(MINTER_ROLE)` and its **caller is the EmissionController**
— so `MINTER_ROLE` must be granted to the **EmissionController** (NOT the
CompensationDistributor, which never calls `mintReward`).

## The transaction (exactly one)

**Foundation Safe → `FTNS.grantRole(MINTER_ROLE, EmissionController)`**

| Field | Value |
|---|---|
| To (FTNS token) | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` |
| Value | `0` |
| Operation | **CALL** (NOT DelegateCall) |
| `role` (MINTER_ROLE) | `0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a6` |
| `account` (EmissionController) | `0x13A0D76895c196B795b94fe843F76B6e145AeaAE` |
| Raw calldata | `0x2f2ff15d9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a600000000000000000000000013a0d76895c196b795b94fe843f76b6e145aeaae` |
| Signer | Foundation Safe `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` (must hold `DEFAULT_ADMIN_ROLE` — ✅) |

Regenerate / re-verify the artifact any time (offline, no RPC needed for prep):
`EMISSION_CONTROLLER_ADDRESS=0x13A0D76895c196B795b94fe843F76B6e145AeaAE python3 scripts/safe-tx-grant-minter-role.py prep`
→ writes the Safe Transaction Builder import JSON to `/tmp/safe-tx-grant-minter-0x13A0D768.json`.

## The gated procedure

### GATE 1 — Pre-flight (re-verify on YOUR machine; abort on any mismatch)
Set a working RPC first (the public `mainnet.base.org` 403s from some networks): `export BASE_RPC_URL=https://base-rpc.publicnode.com`
1. Safe holds admin (must print `true`): `python3 scripts/foundation-safe-health-check.py` (or the `hasRole(DEFAULT_ADMIN_ROLE, Safe)` read).
2. EmissionController does NOT yet have the role (must print role **not** granted / exit 1): `EMISSION_CONTROLLER_ADDRESS=0x13A0D76895c196B795b94fe843F76B6e145AeaAE python3 scripts/safe-tx-grant-minter-role.py verify`
3. Confirm the Safe threshold is **2** (not 1) in the Safe UI settings before relying on the multisig safety model.

### GATE 2 — Load + eyeball the tx in the Safe UI
1. https://app.safe.global → Foundation Safe `0x91b0e6F8…5791` → Apps → **Transaction Builder**.
2. **Use a STRUCTURED entry mode — do NOT hand-paste the raw calldata hex** (a fat-fingered/truncated paste is the one operator error that could mis-target the role; an adversarial review pass flagged exactly this hypothetical). Pick ONE:
   - **(preferred)** drag-drop `/tmp/safe-tx-grant-minter-0x13A0D768.json` — it carries `role` + `account` as separate, human-readable `contractInputsValues`; OR
   - add a Contract Interaction with the ABI fragment `function grantRole(bytes32 role, address account)` and fill `role` / `account` as the two separate fields from the table above.
3. **Confirm what the Safe UI DECODES + displays before signing** (the UI shows the decoded `grantRole(role, account)`): To = `0x5276…16e5`; `role` = `0x9f2df0…56a6` (MINTER_ROLE — **NOT** the all-zeros `DEFAULT_ADMIN_ROLE`); `account` = the **full 42-char** `0x13A0D76895c196B795b94fe843F76B6e145AeaAE` (the EmissionController — **NOT** the Safe, **NOT** the CompensationDistributor); chainId **8453**; CALL not DelegateCall.
4. If you must use raw-calldata mode anyway: the `Data` field is **exactly 138 characters including `0x`** (4-byte selector + 32-byte role + 32-byte left-padded account), starts `0x2f2ff15d9f2df0…` and ends `…145aeaae`. A shorter string = truncated = WRONG; abort.

### GATE 3 — 2-of-3 hardware signing  **[OPERATOR SIGNS — irreversible]**
Two of {Ledger, Trezor} sign in the Safe UI (OneKey is the reserve third). Then **Execute**.

### GATE 4 — Post-execution verification (the economy is now live)
`EMISSION_CONTROLLER_ADDRESS=0x13A0D76895c196B795b94fe843F76B6e145AeaAE python3 scripts/safe-tx-grant-minter-role.py verify` → must print role **granted** / exit 0. (Wait ~1-2 blocks; Base RPC state can lag a few seconds.)

## ★ What happens next — read before triggering distribution

- **Granting the role mints nothing.** It only enables minting.
- **The first `CompensationDistributor.pullAndDistribute()` (permissionless) mints the accrued backlog.** Emissions accrue from `epochZeroStartTimestamp` (≈ 28 days ago as of 2026-06-05) at 1 FTNS/sec → the first distribute will mint **≈ 2.5M FTNS** (0.28% of the 900M `mintCap`) in one shot, growing ~86,400/day until triggered. This is by design (the schedule's accrued emission), but decide deliberately *when* to make the first trigger.
- **All three reward pools (creator/operator/grant) currently point at the Foundation Safe.** So the first distribution sends the ~2.5M to the Safe, not to split creator/operator/grant accounts. If you want real pool splits, the Safe should call `CompensationDistributor.setPoolAddresses(creator, operator, grant)` **before** the first `pullAndDistribute()` (owner-only; a separate, reversible Safe tx).
- Optional hardening (separate decision): the Safe currently *also* holds `MINTER_ROLE` directly. Leaving it grants an emergency override; revoking it (a second Safe tx, AFTER confirming the EmissionController has the role) makes the EmissionController the sole minter. Recommend leaving it for now (emergency pause/override capability).

## Reversibility note
`grantRole` is reversible (`revokeRole(MINTER_ROLE, EmissionController)` by the
Safe) — but its *effect* (the economy goes live + minting becomes possible) is
consequential, hence the 2-of-3 gate. There is no irreversible state change in
this single tx itself.
