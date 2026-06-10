# Requester-Payment Model — Design Proposal

**Status:** DRAFT for review (2026-06-10)
**Author:** engineering (Claude)
**Decision owner:** Ryne / Foundation
**Scope:** how a *paying requester* authenticates + funds a `/compute/inference` job
so on-chain settlement moves **real cross-party value** instead of the current
self-pay. Unblocks a meaningful mainnet activation of the settlement rail
(sp1031, 1035–1053).

---

## 1. Problem

The on-chain settlement rail is built, reviewed, live-on-mainnet (Phase-1 batch
`7b6490c9…` committed), observable, and hardened. But it currently **self-pays**:
the settle path (`prsm/node/api.py` unary + streaming) calls
`accumulate_settled_inference_receipt(...)` with **no `requester_address`**, which
`prsm/settlement/client_wiring.py` defaults to `provider_address` — so the on-chain
`EscrowPool.settleFromRequester(requester, provider, amount)` draws from and pays
the *same* address. Net economic value transferred: **zero**.

To settle real value, a job must carry a **distinct paying requester** whose
**on-chain `EscrowPool` balance** funds the settlement, and the provider must be
able to **trust** that the named requester actually authorized the spend.

Two escrow concepts exist today and must not be conflated:
- **Off-chain `PaymentEscrow`** (`node._payment_escrow`, per-job `escrow_entry`):
  the *current* node-local payment gate. Fast, node-internal, not trustless.
- **On-chain `EscrowPool`** (`0x526D40C0…` on Base): the *settlement* rail's
  per-eth-address escrow. `finalizeBatch → settleFromRequester` draws from the
  requester's on-chain balance. Trustless, self-custodied, the source of truth for
  cross-party settlement.

This proposal makes the **on-chain `EscrowPool`** the canonical requester-funding
mechanism for settled inference, and defines the per-request authorization that
lets a provider safely name a remote requester in the settled receipt.

---

## 2. Design

Three independent pieces: **funding** (on-chain, self-custodied), **authorization**
(per-request, signed), and **settle wiring** (use the authenticated address).

### 2.1 Funding — requester self-custodies FTNS in the on-chain EscrowPool

The requester deposits FTNS into `EscrowPool` **under their own eth address**
(`EscrowPool.deposit(amount)` — the same `EscrowPoolClient.deposit` we built in
sp1041, callable by anyone for their own balance). No PRSM-side custody, no trust:
the escrow is the requester's; `settleFromRequester` can only move it to the
provider named in a *finalized* batch, and the requester can `withdraw` any
unspent balance.

- The requester maintains a working balance ≥ the inference prices they intend to
  spend. Topping up is a normal on-chain deposit.
- No change to `EscrowPool`; this is exactly what sp1041's client already does.

### 2.2 Authorization — a signed per-request payment authorization

The inference request gains a **payment authorization** the requester signs with
the eth key that controls their EscrowPool balance. The provider verifies the
signature and binds the receipt to the authenticated `requester_address`.

Proposed authorization payload (EIP-712 typed data — wallet-friendly, replay-safe):

```
PaymentAuthorization {
  requester:     address   // the EscrowPool depositor; settleFromRequester source
  provider:      address   // the node being paid (binds the auth to THIS provider)
  max_spend_wei: uint256   // ceiling the requester authorizes for this job
  job_nonce:     bytes32   // unique per job (anti-replay)
  expiry_unix:   uint64    // auth invalid after this (anti-replay / staleness)
  request_hash:  bytes32   // keccak of the canonical inference-request params
}
```

The provider:
1. Recovers the signer from the EIP-712 signature; requires `signer == requester`.
2. Requires `provider == this node's settler/provider address` (the auth can't be
   replayed against a different provider).
3. Requires `now < expiry_unix` and `job_nonce` unseen (a small durable
   seen-nonce set — we already have the `webhook_replay_defense` /
   `seen_nonces` patterns to reuse).
4. Requires `request_hash` matches the actual request (the auth covers *this* work).
5. Pre-flight: `EscrowPool.balanceOf(requester) ≥ quoted_price ≤ max_spend_wei`.

Only if all pass does the provider serve the job and, on completion, settle on-chain
against `requester`.

### 2.3 Settle wiring — use the authenticated requester address

Minimal, already-supported plumbing: the settle sites pass the authenticated
`requester_address` into `accumulate_settled_inference_receipt(...,
requester_address=<authenticated requester>, ...)` (the parameter has existed since
sp1037; it just defaults to self today). The sp1035 adapter already accepts a
distinct requester; the BatchSettlementClient already commits with the requester;
`finalizeBatch` already draws from the requester's escrow. **No settlement-core
change is needed** — only: (a) carry + verify the authorization, (b) thread the
authenticated address into the existing settle call.

Note the funds-safety invariant flips for cross-party: the **settler key**
(`provider`/`msg.sender` of `commitBatch`) is the *provider*, and `requester` is a
*distinct* address. The sp1036 build-time check (`key.address == provider_address`)
stays correct (the node settles as itself = provider); the requester is supplied
per-receipt, not from the key.

---

## 3. Security analysis

| Threat | Mitigation |
|---|---|
| **Auth replay** (provider re-uses a signed auth for extra settlements) | `job_nonce` (durable seen-set) + `expiry_unix` + `request_hash` binding; one auth settles one job, once. |
| **Cross-provider replay** | `provider` field in the signed payload pins the auth to one node. |
| **Forged requester** | EIP-712 signature recovery; `signer == requester` required. The provider can't name a requester who didn't sign. |
| **Over-commit / double-spend across providers** | The challenge window (3 days) means escrow is *drawn at finalize*, not reserved at commit. A requester could authorize more than their balance across N providers → some `finalize`s revert (provider unpaid). **v1 mitigation:** provider pre-flight checks `balanceOf(requester) ≥ price` at *commit* time + keeps batch values modest; an over-committed requester is the *provider's* counterparty risk (bounded, observable). A true reservation (escrow lock at commit) is a **future EscrowPool change**, out of v1 scope. |
| **Requester withdraws mid-window** | Same class as over-commit — finalize reverts if balance < value. Provider pre-flight + small batches bound the exposure. A `pendingSettlement` lock on EscrowPool is the durable fix (future). |
| **Provider serves but never settles** | Off-chain `PaymentEscrow` already gates the *result* delivery; on-chain settlement is the provider claiming payment. A provider that skips settlement simply doesn't get paid. |

The honest residual: **v1 has no escrow reservation at commit**, so a requester can
under-fund relative to outstanding authorizations and cause finalize-reverts. This
is bounded provider counterparty risk (not a protocol exploit — no one is paid twice
or paid from someone else's escrow). The protocol-level fix (reserve-at-commit) is a
contract change tracked as a follow-on.

---

## 4. API changes

- **`/compute/inference` request** (`prsm/node/api.py`): optional
  `payment_authorization` object (the EIP-712 payload + signature) and
  `requester_address`. Absent → current behavior (off-chain PaymentEscrow gate;
  on-chain settlement self-pays or is skipped). Present + valid → settled on-chain
  against the requester.
- **Receipt**: `requester_address` already flows through the adapter; no shape
  change.
- **A small `PaymentAuthorizationVerifier`** (new, ~1 module): EIP-712 verify +
  nonce/expiry/request-hash/balance checks. Reuses `eth_account` + the existing
  seen-nonce durable pattern.
- **Client/SDK helper**: build + sign a `PaymentAuthorization` (so a requester CLI
  can produce one). Optional for v1 (can hand-construct).

---

## 5. Implementation plan (bricks, each TDD + reviewed)

1. **`PaymentAuthorization` type + EIP-712 signing/verification** (pure, hermetic
   tests with generated eth keys — like the attestation engines). No network.
2. **Verifier**: nonce durability (reuse `seen_nonces`), expiry, provider-pin,
   request-hash binding, `EscrowPool.balanceOf` pre-flight. Fail-closed.
3. **Wire into `/compute/inference`** (unary + streaming): if a valid auth is
   present, thread `requester_address` into `accumulate_settled_inference_receipt`;
   else unchanged. Fail-open on the on-chain path (settlement is best-effort,
   post-release), fail-closed on the *authorization* (a bad auth → reject the paid
   job, fall back to the existing free/own-escrow path or 402).
4. **Requester-side helper** to build/sign an authorization (+ deposit via the
   sp1041 `EscrowPoolClient`).
5. **Sepolia two-party proof**: requester A deposits + signs; provider B serves +
   commits + finalizes → A's escrow drops, B's wallet rises. (The harness already
   supports two keys; this is the cross-party analogue of the self-pay proof.)
6. *(Future)* EscrowPool reserve-at-commit to remove the over-commit residual.

---

## 6. Decisions needed from you

1. **EIP-712 vs a simpler canonical-payload signature?** EIP-712 is wallet-native
   (MetaMask `signTypedData`) and the recommended default; a raw canonical-bytes
   ECDSA is simpler but worse UX. **Recommendation: EIP-712.**
2. **Is the on-chain `EscrowPool` the canonical requester-funding mechanism**, with
   the off-chain `PaymentEscrow` retained only as the local result-delivery gate?
   Or should the two be unified? **Recommendation: on-chain EscrowPool canonical;
   keep off-chain PaymentEscrow as the fast local gate for now.**
3. **Accept the v1 over-commit residual** (provider counterparty risk, mitigated by
   pre-flight + small batches), deferring reserve-at-commit to a contract follow-on?
   **Recommendation: yes — ship v1 without a contract change; reserve-at-commit
   later if real usage shows over-commit is a problem.**
4. **Who is the "requester"** in PRSM's product — an end user with a wallet, a
   gateway/relayer paying on users' behalf, or both? This shapes the auth UX (#1)
   and whether a relayer-style delegated authorization is needed. **Needs your
   product call** — the rest of the design is largely independent of it.

On your answers (especially #4), I'll build bricks 1–5.

---

## 7. What this does NOT change

- The settlement core (adapter, client, commit/finalize, durable state, WAL
  recovery, double-commit + concurrency guards) — all unchanged; this rides on top.
- The free / self-escrow path — absent an authorization, behavior is exactly today.
- Trust model — funding is self-custodied on-chain; PRSM never holds requester FTNS.
