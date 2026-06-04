# P2P network-substrate hunt — fixes and residual decisions

**Sprints 1005–1009.** The P2P-substrate adversarial integrity hunt (workflow
`wbu7u2ftm`, 5 dimensions — transport-identity / gossip-mesh-DoS-replay /
discovery-DHT-poisoning-eclipse / bootstrap-trust / message-validation — with
default-refute + reachability discipline) confirmed 14 findings about whether a
malicious peer can spoof transport identity, eclipse/poison peer discovery,
amplify or replay the gossip mesh, or over-trust the bootstrap server. The
live/high-impact ones are fixed; this doc records the residuals that are either
latent (an unwired transport) or tied to semi-trusted infra, with their
recommended fixes — mirroring the sp1000 storage-slashing-gap pattern.

## Reachability anchor — which transport is LIVE

NodeConfig defaults `transport_backend="libp2p"`, but the Go shared library is
**absent** on the Linux fleet (`.so` is gitignored + only `darwin_arm64` is
built), and `Libp2pTransport._load_library` raises with no fallback — so a Linux
node cannot start in libp2p mode. Operators pin `PRSM_TRANSPORT_BACKEND=websocket`
(canonical systemd unit + deploy docs). **The live substrate is
WebSocketTransport + PeerDiscovery + GossipProtocol**, so the fixes target that
stack; the libp2p findings are latent until/unless the `.so` is built + deployed.

## Shipped (autonomous, on the live WebSocket substrate)

| Sprint | Finding(s) | Fix |
|--------|-----------|-----|
| **1005** | 1, 2, 7, 12 (HIGH, eclipse) | Authenticate the PEX peer-exchange response: each `_handle_peer_response` entry must carry a portable, self-verifying credential (the sp937 announce attestation made portable); unauthenticated/forged/tampered/impersonating entries are dropped. Adds a `known_peers` cap (`PRSM_MAX_KNOWN_PEERS`) + a `max_peers` response bound (also closes finding 3's amplification). |
| **1006** | 9 (HIGH, MITM) | Bootstrap WSS client verifies TLS by default (was `CERT_NONE` unconditionally). Verified the live fleet uses valid Let's Encrypt certs that pass full verification. Dev/self-signed opt in via `PRSM_BOOTSTRAP_TLS_CA_FILE` (pin) / `PRSM_BOOTSTRAP_TLS_INSECURE=1` (disable, warned). |
| **1007** | 6 (HIGH, eclipse/DoS) | Cap the per-CID provider set (`PRSM_MAX_PROVIDERS_PER_CID`, default 64) so an attacker can't grow `ContentRecord.providers` without bound via unauthenticated `provider_id`s. |
| **1008** | 4 (replay/amplification) | Gossip-layer replay barrier: a TTL'd (≥ log retention) + LRU-bounded seen-nonce set in `_handle_gossip` drops a frame replayed past the transport's ~300s window before re-delivery/re-fan. |
| **1009** | 14 (DoS) | Cap bootstrap-client peer ingestion (`_ingest_bootstrap_peers` honors `PRSM_MAX_KNOWN_PEERS`) so a compromised bootstrap can't flood `known_peers`. |

## ★ RESIDUAL A — the libp2p transport has NO origin-auth (findings 5, 11, 13)

`libp2p_gossip` / `libp2p_discovery` carry none of the sp934/937/941/1005
attestation: capability/shard-index entries and peer records are trusted from the
raw payload, and `_capability_index` / `_shard_cache` are uncapped. **Latent**
today (libp2p can't start on the fleet), but the config DEFAULT is libp2p, so a
future operator who builds the `.so` and deploys libp2p inherits every eclipse /
index-poisoning / node-impersonation vector the WebSocket path was just hardened
against.

**Recommendation (decision):**
- **Port the attestation primitives to the libp2p handlers** (the sp934
  origin-auth + sp937 announce attestation + sp1005 PEX credential + the sp1007/1008
  caps) before libp2p is ever made live. This is the correct long-term fix but a
  meaningful chunk, best done when libp2p deployment is actually on the roadmap.
- **Interim, cheap, do-now:** a **startup guard** — refuse to start (or emit a
  CRITICAL) when `transport_backend=libp2p`, until the attestation is ported.
  This prevents silently shipping the unhardened path. ✅ **SHIPPED (sp1010):**
  `_check_libp2p_origin_auth_gap` in `node.py` logs CRITICAL whenever libp2p is
  selected and hard-refuses startup when `PRSM_FORBID_UNAUTHENTICATED_LIBP2P` is
  set. The full attestation port to libp2p remains the long-term fix.

## ★ RESIDUAL B — bootstrap register accepts a caller-chosen peer_id (finding 10)

`bootstrap/server.py` register handler trusts the caller-supplied `peer_id` and
overwrites the registry/connection slot unconditionally — registry-griefing /
eclipse-assist (a peer registers as another's id and displaces it from the
bootstrap's peer-list view). **Impersonation harm is backstopped**: the actual
P2P WebSocket handshake binds node identity (sp937), and PEX entries are now
credential-verified (sp1005), so the registry slot is only the bootstrap's
*view*, not real node identity. The bootstrap is Foundation-run, semi-trusted
infra.

**Recommendation (decision):** require the register to prove key ownership —
e.g. the registrant signs a challenge/nonce with the key whose
`sha256(pubkey)[:32]` equals the claimed `peer_id` (the same binding used
everywhere else). Server-side change to a live service → a small ceremony
(deploy the bootstrap server), so it is grouped with the next bootstrap-server
maintenance rather than shipped piecemeal.

## ★ RESIDUAL C — single-bootstrap-anchor cold-start trust (finding 8)

A cold-starting node trusts one bootstrap anchor's peer payloads with no
cross-verification. Now substantially mitigated: the connection is TLS-verified
(sp1006, no MITM), the ingested peer list is capped (sp1009), and any peer it
later relays via PEX must be credential-verified (sp1005). The irreducible part —
trusting a single anchor's *selection* of peers at cold start — is partly inherent
to bootstrapping.

**Recommendation:** multi-bootstrap quorum / peer-exchange diversity (cross-check
peers across the US/EU/APAC anchors before trusting) — a discovery-policy
enhancement, lower priority now that MITM + flooding are closed.

## Not-a-bug / already-backstopped (recorded for audit)

- **Transport identity spoofing of a DIRECT message** — the WebSocket handshake
  (transport.py) binds the connection to a verified node_id; `_dispatch` excludes
  MSG_GOSSIP from sender_id re-binding, which is exactly why gossip/PEX needed
  payload-level attestation (now provided). Direct (MSG_DIRECT) sender_id is
  handshake-bound.
- **Money/state gossip handlers** (job_result sp924, escrow, ftns_transaction
  sp898/899, content_advertise sp1004) — hunted in prior sprints; the substrate
  hunt deliberately did not re-cover them.
