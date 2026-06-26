# Fable 5 Review — Domain 05: Economy, tokenomics & on-chain contracts

Adversarial, cross-cutting review. See `00_INDEX.md` for shared framing + output format.

## Context

FTNS is PRSM's ERC-20 on Base mainnet (100M genesis pre-mined to a Foundation Safe; a 900M
emission bucket; 1B max supply). The economy layer holds: tokenomics (emission curve,
staking, slashing, rewards), the FTNS / EmissionController / ProvenanceRegistry / staking /
royalty on-chain CONTRACT CLIENTS (web3), payments, marketplace pricing, and governance. The
settlement/escrow money-path is Domain 01; **this domain is the BROADER economy** — supply
integrity, emission/reward math, staking/slashing, royalty distribution, the on-chain
clients' transaction safety, marketplace pricing, and governance actions. A flaw here =
mint/inflation bugs, mis-paid rewards, double-claims, or an unsafe on-chain write.

### Read
- `prsm/economy/tokenomics/` (emission curve, staking, slashing, rewards, supply accounting)
- `prsm/economy/web3/` EXCEPT the settlement/escrow clients in Domain 01 (i.e. FTNS token
  client, EmissionController client, ProvenanceRegistry client, staking/royalty clients,
  `aerodrome_swap_client.py` (Tier-4 fiat→FTNS swap), the tx-lock registry, gas handling)
- `prsm/economy/blockchain/`, `prsm/economy/payments/`, `prsm/economy/governance/`,
  `prsm/economy/economics/`, `prsm/economy/pricing/`
- `prsm/emission/`, `prsm/governance/`, `prsm/marketplace/`

### Invariants — confirm or break
1. **Supply integrity.** No code path mints beyond the emission bucket / max supply, double-
   counts emission, or lets a non-MINTER_ROLE caller mint. The pause/resume emission controls
   behave (paused → mint reverts). Reward accounting can't pay more than was emitted.
2. **Reward / royalty math.** Per-stage / per-contributor / royalty splits sum to ≤ the amount
   released, are non-negative, round safely (no rounding leak that over-pays), and credit the
   correct addresses. Royalty rate bounds (e.g. ≤ 9800 bps) enforced; the authenticated
   on-chain creator (not a forgeable advertised value) drives royalty.
3. **Staking / slashing correctness.** Stake can't be double-counted for eligibility; a
   claim-rewards can't be replayed or claim more than accrued; slashing math + timing
   (tz-aware datetimes) is correct; unstake/withdraw can't drain more than staked.
4. **On-chain client tx safety.** Every web3 WRITE client (mirroring the reviewed escrow/
   settlement clients): exact-amount approvals, per-account tx lock (no nonce races),
   explicit gas where replica-lag false-reverts estimateGas, idempotent re-runs (a landed-
   but-lagged tx isn't re-sent → no double-spend), and three-tier error handling
   (broadcast-failed vs reverted vs pending). Find a client that can double-submit or strand.
5. **Fiat onramp (Aerodrome swap).** The USDC→FTNS swap enforces the quoted floor / slippage,
   revokes residual allowance on failure, validates the router + route vs config, rejects
   stale envelopes / past deadlines, and never defaults the recipient to the swap wallet.
6. **Governance actions.** Privileged actions (parameter changes, recovery, pool seeding) are
   gated to the Foundation multisig / the right authority and can't be triggered by an
   unprivileged caller; the Foundation must not unilaterally seed AMM pools (a stated invariant).

### Hunt list
- Integer/Decimal mixing in money math (wei vs FTNS vs bps); rounding direction; truncation.
- A "advertised"/self-reported economic value (rate, price, stake, cost) trusted without an
  on-chain / signed authoritative source.
- Reentrancy-style ordering in the client logic (state updated before/after the irreversible
  broadcast); a permissionless trigger (e.g. `pullAndDistribute`) that an attacker front-runs.
- Off-chain bookkeeping that can diverge from on-chain truth (local-memory balances).

Follow the `00_INDEX.md` output format. Report only.
