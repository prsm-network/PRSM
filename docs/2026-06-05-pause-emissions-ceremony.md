# Pause FTNS Emissions Ceremony (pre-launch) — `EmissionController.pauseMinting()`

**Status: READY for the Foundation 2-of-3 multisig. One transaction.**
Prepared + verified by the assistant on 2026-06-05 against live Base-mainnet state.
The assistant does **not** sign — the on-chain step is the Foundation Safe
execution, marked **[OPERATOR SIGNS]** below.

## Why pause now

Tier 0 granted `MINTER_ROLE` to the EmissionController (see
`docs/2026-06-05-tier0-economy-activation-ceremony.md`), so the economy *can* now
mint. But **the network has no participants yet** — no creators, operators, or
grant recipients to reward. Minting emissions into the Foundation's own wallet
before launch serves no economic purpose and ticks up circulating supply for
nothing.

Two facts make a pause the right pre-launch posture:

1. **`CompensationDistributor.pullAndDistribute()` is permissionless** — *anyone*
   can trigger the accrued ~2.5M-and-growing mint at any time. With all three
   reward pools currently pointed at the Foundation Safe, an unwanted trigger
   would mint that new supply straight into the Safe.
2. **Bootstrapping doesn't need emissions** — the Foundation already holds the
   100M genesis treasury (`INITIAL_SUPPLY`) for AMM-seeding / grants / ops. The
   separate 900M emission bucket should stay pristine to reward *actual*
   participation once the network is live.

`pauseMinting()` makes `mintAuthorized()` revert (`whenNotPaused`), so no caller
can mint while paused. Circulating supply stays at the genesis **100M**. At launch
you call `resumeMinting()` (owner-only) — at which point you split the pools to
real recipients and turn emissions on deliberately.

**Honest caveat (not a reason to delay):** pausing *defers* but does not *reduce*
the schedule's accrued entitlement. The emission clock has an immutable
`epochZeroStartTimestamp` (≈ 28 days before 2026-06-05) and keeps accruing on its
timeline regardless of the pause. Whenever you resume and first-mint at launch,
that call captures the full backlog accrued from epoch-zero to then. Where that
backlog lands is a launch-time decision (likely an ecosystem/grant pool), not a
reason to leave minting un-paused now.

## The transaction (exactly one)

**Foundation Safe → `EmissionController.pauseMinting()`**

| Field | Value |
|---|---|
| To (EmissionController) | `0x13A0D76895c196B795b94fe843F76B6e145AeaAE` |
| Value | `0` |
| Operation | **CALL** (NOT DelegateCall) |
| Method | `pauseMinting()` — **no arguments** |
| Raw calldata | `0xda8fbf2a` (the 4-byte selector; exactly 10 chars incl. `0x`) |
| Signer | Foundation Safe `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` (must be `owner()` — ✅ verified) |

`0xda8fbf2a` = `keccak256("pauseMinting()")[:4]`, proven from source by the prep
script (not hand-copied).

Regenerate / re-verify the artifact any time (offline, no RPC for prep):
`python3 scripts/safe-tx-pause-emissions.py prep`
→ writes the Safe Transaction Builder import JSON to `/tmp/safe-tx-pause-emissions-0x13A0D768.json`.

## Review scope (scaled to the risk surface)

This is the **least consequential** of the economy ceremonies and did **not** get a
full multi-agent adversarial panel — deliberately. It is a no-argument, owner-only,
**reversible** CALL that **halts** minting (a conservative action). It carries no
role, address, or amount that could be mis-targeted — the entire attack/error
surface is "is the target the EmissionController and the method `pauseMinting()`?",
which the prep script proves from source and the Safe UI decodes before signing.
Worst-case failure modes are benign: a wrong target / wrong method just reverts (no
state change), and `pauseMinting()` while already paused reverts `AlreadyPaused`.
Direct on-chain verification (selector, `owner()==Safe`, `paused()==false`) covers
the whole surface; a skeptic panel would add nothing.

## The gated procedure

### GATE 1 — Pre-flight (re-verify on YOUR machine; abort on any FAIL)
Set a working RPC first (the public `mainnet.base.org` 403s from some networks): `export BASE_RPC_URL=https://base-rpc.publicnode.com`
1. Both preconditions must pass (`owner()==Safe` and `paused()==false`): `python3 scripts/safe-tx-pause-emissions.py preflight` → must print both `[PASS]` / exit 0.
2. Confirm the Safe threshold is **2** (not 1) in the Safe UI settings before relying on the multisig safety model.

### GATE 2 — Load + eyeball the tx in the Safe UI
1. https://app.safe.global → Foundation Safe `0x91b0e6F8…5791` → Apps → **Transaction Builder**.
2. Use a **structured entry mode**. Pick ONE:
   - **(preferred)** drag-drop `/tmp/safe-tx-pause-emissions-0x13A0D768.json`; OR
   - add a Contract Interaction with the ABI fragment `function pauseMinting()` (no fields to fill).
3. **Confirm what the Safe UI DECODES + displays before signing:** To = `0x13A0D768…AeaAE` (the **EmissionController**, NOT the FTNS token, NOT the Safe); method = **`pauseMinting`** (NOT `resumeMinting`, NOT `grantRole`); no arguments; ETH value `0`; chainId **8453**; CALL not DelegateCall.
4. If you must use raw-calldata mode: the `Data` field is **exactly `0xda8fbf2a`** (10 chars incl. `0x`). Anything else = wrong; abort.

### GATE 3 — 2-of-3 hardware signing  **[OPERATOR SIGNS]**
Two of {Ledger, Trezor} sign in the Safe UI (OneKey is the reserve third). Then **Execute**.

### GATE 4 — Post-execution verification (emissions now paused)
`python3 scripts/safe-tx-pause-emissions.py verify` → must print `paused() == true` / exit 0. (Wait ~1-2 blocks; Base RPC state can lag a few seconds.)

## Reversibility note
Fully reversible: `EmissionController.resumeMinting()` (owner-only Safe tx) turns
emissions back on at launch. This single tx makes **no** irreversible state change —
it sets a `paused` boolean. Pause is the conservative default; you resume
deliberately when the network has participants and the reward pools have been split
to real recipient addresses.

## What this does NOT do
- It does **not** revoke `MINTER_ROLE` (the EmissionController keeps the role; it
  just can't *use* it while paused).
- It does **not** touch the 100M genesis treasury or change `totalSupply`.
- It does **not** set pool addresses — that is a separate launch-time decision
  (`CompensationDistributor.setPoolAddresses(creator, operator, grant)`), to be made
  *before* the first `resumeMinting()` + `pullAndDistribute()` at launch.
