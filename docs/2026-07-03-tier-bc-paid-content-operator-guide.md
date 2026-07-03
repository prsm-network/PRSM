# Tier B/C paid content — operator + user guide

How to **publish** and **buy** paywalled datasets on PRSM. The `ContentAccessVerifier` is live on
Base mainnet (`0xF32A9049EeB1Fa3eE9f76A085a6b8662d5c9aE59`) and Base Sepolia
(`0x99264Bca75d63DB9b8B5C7C1e2ECBf78d133905a`), and the address auto-resolves from network config —
no per-user setup for the verifier.

## The model (one paragraph)

The content ciphertext is served **freely** (a node holding it sees only random bytes). The
decryption key is wrapped to the **buyer's X25519 key** and served **only after on-chain payment**,
through a gated endpoint — and the served key is verified against an **on-chain commitment**, so a
malicious serving node can't hand over a wrong key. Reading public commons is free (Tier A); this is
the paywalled tier (Tier B/C).

## Roles

- **Buyer** — generates an X25519 keypair, gives the publisher the public key, pays the fee, unlocks.
- **Publisher / serve operator** — runs a node that publishes paid content and serves the key.

---

## Buyer

No node required for the keypair. Set your ETH key (pays + signs the key fetch) and X25519 key
(decrypts) in your **own shell**, never on the command line.

```bash
# 1. Generate your X25519 keypair (local, no node):
prsm content buyer-keygen
#    → PUBLIC  (give to the publisher)
#    → PRIVATE (keep secret)

# 2. Give the publisher your PUBLIC key. They publish and send you back a content_hash + commitment.

# 3. Unlock (keys from the ENVIRONMENT, never argv):
export PRSM_REQUESTER_KEY=0x...          # your ETH key (pays payForAccess + signs the key fetch)
export PRSM_X25519_PRIVKEY=<your-private> # from buyer-keygen
prsm content unlock <content_hash> --fee <FTNS> --commitment <commitment> [--output out.bin]
```

Your ETH key needs FTNS (≥ the fee) + a little ETH for gas. The unlock pays the fee, fetches the
key, verifies it against the commitment, and decrypts.

---

## Publisher / serve operator

The node both **serves** the payment-gated key and (optionally) **publishes** paid content.

### Node env

```bash
export PRSM_PAID_KEY_SERVE=1                         # enable the paid-key serve endpoint
export PRSM_PAID_KEY_STORE_FILE=/var/lib/prsm/paid_keys.json   # DURABLE — see below
export PRSM_PAID_PUBLISHER_KEY=0x...                 # publisher ETH key (signs the deposit); only
                                                     #   needed if THIS node publishes
# PRSM_CONTENT_ACCESS_VERIFIER auto-resolves from network config; override only for a custom deploy.
prsm node start
```

**`PRSM_PAID_KEY_STORE_FILE` MUST be a durable path.** The on-chain payment gate persists forever;
if the retained wrapped key is lost (e.g. an in-memory-only store on restart), buyers who pay after
the restart get a 404 with no refund. The file store atomically flushes each publish and rehydrates
on startup (fix from the R5 review). The wrapped key is already ciphertext (sealed to the buyer), so
file storage adds little exposure — still, protect the file.

The publisher ETH key needs to be a **registered creator's** identity (the normal upload registers
it in the ProvenanceRegistry) and hold ETH for gas. Rotate it if exposed; it is not a contract
owner.

### Publish

```bash
prsm content publish-paid ./dataset.bin \
    --buyer-pubkey <buyer1-x25519-public> [--buyer-pubkey <buyer2-...>] \
    --fee 0.5     # FTNS
```

It encrypts + wraps the content to the buyer(s), serves the ciphertext, deposits the sha256
commitment on-chain (naming the verifier), and retains the key. It prints the `content_hash`,
`commitment`, `deposit_tx`, and the exact `prsm content unlock ...` command to hand each buyer.

---

## Trust model + honest scope

- **The paywall binds** (F1 fix): the wrapped key never lives in world-readable on-chain storage —
  only the commitment does, and the key is served off-chain only to a paid + authenticated fetcher.
- **The commitment defeats a rogue serving node / MITM**, when read from the authoritative on-chain
  source (the unlock path does this). It does **not** defend against a malicious *publisher*, who
  controls both the deposit and the content and whom you already trust for the dataset itself.
- **Squatting (F9/F2)** is mitigated at the client layer: the unlock refuses to pay if the fee payee
  (registered creator) isn't the key depositor, and publish refuses a content_hash another party
  already deposited. This is a mitigation, not a front-run-proof on-chain guarantee.
- **Fair exchange** (publisher takes payment then withholds) is out of scope for v1 beyond the
  commitment check; keep serve nodes durable + reputable.

## Production smoke checklist

Before relying on a paid dataset in production, prove the full loop once on the target network:

1. Buyer: `prsm content buyer-keygen` → note the public key + fund the buyer ETH key with FTNS+ETH.
2. Publisher node up with `PRSM_PAID_KEY_SERVE=1` + `PRSM_PAID_PUBLISHER_KEY` + a durable store file.
3. Publisher: `prsm content publish-paid <small-file> --buyer-pubkey <pub> --fee <F>` → note the
   printed `content_hash` + `commitment`.
4. Buyer: `PRSM_REQUESTER_KEY` + `PRSM_X25519_PRIVKEY` set → `prsm content unlock <hash> --fee <F>
   --commitment <C> --output out.bin` → confirm `out.bin` matches the original.
5. Restart the publisher node → repeat step 4 with a fresh buyer/payment → confirm the durable store
   still serves (no 404).
