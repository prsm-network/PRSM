# Fable 5 Review — Domain 06: Content, storage & data layer

Adversarial, cross-cutting review. See `00_INDEX.md` for shared framing + output format.

## Context

PRSM stores + serves content over a BitTorrent-derived data layer. Content identity is a v1
(BEP-3) infohash = sha1(bencode(info-dict)); the canonical computation is pure-Python so it's
reproducible without libtorrent. Content tiers: A = public single-file; B/C = encrypted
multi-file (a manifest + AES-256-GCM-encrypted shards + Shamir-split key shares), bundled into
a single blob. The data layer also includes provenance, fingerprints/dedup, embeddings, and
vector store. The security-critical properties are **content-addressing integrity** (the bytes
you get match the CID you asked for) and **content cryptography** (Tier B/C confidentiality +
correct key handling).

### Read
- The content/torrent/encryption code (mostly in `node` + `core`): `prsm/core/torrent_infohash.py`,
  `prsm/core/bittorrent_client.py`, `prsm/node/content_provider.py`, `content_uploader.py`,
  `local_content_publisher.py`, `artifact_bundle.py`, and the content-tier encrypt/decrypt +
  Shamir keyshare code
- `prsm/data/` (provenance, fingerprints, dedup, embeddings, vector_store, context,
  content_processing), `prsm/storage/`, storage backends in `prsm/core/integrations/`

### Invariants — confirm or break
1. **Content-address integrity (anti-substitution).** A fetched object's bytes must match the
   requested CID. For a 40-hex/v1 CID the infohash is recomputed and a mismatch is rejected;
   for a bundle the inner content is re-verified against the CID. Find a path where substituted
   bytes are accepted (CID/infohash not re-verified, `verify_hash=false`, a bundle whose inner
   CID isn't checked, a decode-before-verify ordering).
2. **Canonical infohash reproducibility.** The pure-Python v1 infohash must be byte-identical
   to what the real libtorrent client produces (piece_length, name=basename, BEP-3 dict
   ordering), so the network doesn't fragment / mis-route. Find an input where they diverge.
3. **Tier B/C confidentiality + key handling.** AES-256-GCM is used correctly (unique nonce
   per encryption, auth tag verified on decrypt, no nonce reuse); Shamir key shares reconstruct
   only with threshold and aren't leaked; the manifest can't be tampered to point at attacker
   content; decryption happens only where authorized (cross-ref the TEE decrypt gate, Domain 03).
4. **Bundle parsing is hardened against hostile blobs.** The bundle format enforces magic +
   bounds (max part size, max shards, max keyshares, count×size coherence) so a malicious
   bundle can't OOM or integer-overflow the parser. Find an unbounded/overflowing field.
5. **Provenance / dedup / fingerprint integrity.** The provenance record (creator, royalty) is
   the authenticated on-chain value, not a forgeable local claim (cross-ref Domain 05);
   dedup/fingerprint false-positives can't cause one user's content to satisfy another's
   request (content-confusion).

### Hunt list
- Decode/decrypt-before-verify ordering anywhere in the fetch path.
- Hash/CID algorithm confusion (sha1 vs sha256, hex vs bytes, truncation).
- Nonce/IV generation for AES-GCM (counter reuse, predictable, reused across shards).
- Path traversal / zip-slip in manifest/shard/bundle extraction to disk.
- Untrusted-size fields driving allocation (the classic decompression/realloc bomb).
- Vector-store / embedding inputs from untrusted sources reaching a query without bounds.

Follow the `00_INDEX.md` output format. Report only.
