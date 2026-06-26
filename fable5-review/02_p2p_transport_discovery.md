# Fable 5 Review — Domain 02: P2P networking, transport & discovery

Adversarial, cross-cutting review. See `00_INDEX.md` for shared framing + output format.

## Context

PRSM nodes form a peer-to-peer network for discovery, content transfer, and cross-host
inference. Two transports exist: a WebSocket stack (`prsm/node/transport.py`, the more
hardened path) and a libp2p stack (the default; `prsm/node/libp2p_transport.py` /
`libp2p_gossip.py` / `libp2p_discovery.py`). Peers learn each other via a bootstrap server
(`prsm/bootstrap/`), gossip announces, and peer-exchange (PEX). Origin-authentication
(signed announces / portable credentials) and node_id = sha256(pubkey) are load-bearing
because downstream security decisions (which node runs confidential work, who gets paid)
trust node_id. **This domain owns the broader transport + discovery correctness + the
network trust boundary; the attestation-specific slice of node_id auth is also reviewed in
Domain 01 — focus here on the wire protocol, connection management, propagation, NAT,
framing, and eclipse/poisoning/DoS resistance.**

### Read
- `prsm/node/transport.py`, `libp2p_transport.py`, `libp2p_gossip.py`, `libp2p_discovery.py`,
  `discovery.py`, `bootstrap_transport.py`, `heartbeat*.py`, anything `*nat*` / `*pex*` /
  `*gossip*` / `*peer*` in `prsm/node/`
- `prsm/network/` (the network primitives), `prsm/bootstrap/` (server + client + peer DB)

### Invariants — confirm or break
1. **node_id is unspoofable on every ingest path.** Every place a peer's claimed node_id
   enters local state (gossip capability/shard announce, bootstrap-server peer list +
   join/leave announcements, PEX/peer-response relay, the handshake) must authenticate it —
   `sha256(origin_pubkey)[:32] == claimed` + a valid signature, or a portable credential the
   receiver re-verifies. Find an ingest path that trusts a payload-supplied node_id/address
   unverified (→ eclipse / routing-table or capability-index poisoning).
2. **No eclipse / unbounded table growth.** Peer tables, capability index, shard cache, and
   any seen-nonce / dedup sets are bounded; a hostile peer can't flood a victim into evicting
   honest peers or exhausting memory. Confirm the caps + eviction policy can't be gamed.
3. **Replay / freshness on announces.** A captured genuine announce can't be replayed to
   re-assert a stale address/capability after the transport's nonce-dedup window (monotonic
   announce_time / nonce). Find a replay that re-poisons an entry.
4. **Address integrity.** Advertised addresses (observed-address from the bootstrap server,
   declared listen port in the handshake, `host:port` operator advertise) are combined without
   corruption (no double-port, no own-IP, no using a relayer's source endpoint for a relayed
   peer). Find an address-construction path that yields an undialable or attacker-chosen target.
5. **Content-transfer integrity + DoS.** Chunked/inline content transfer over the substrate
   re-verifies integrity per frame and bounds reassembly memory; a hostile sender can't
   corrupt-without-detection or OOM the receiver. Confirm the frame size + total caps hold.
6. **Transport auth parity.** The libp2p path is the default but historically less
   origin-authenticated than WS. Enumerate every libp2p gossip subscription + ingest and
   confirm each authenticates (or is correctly gated downstream). Flag any libp2p ingest that
   feeds a security/money decision while unauthenticated.

### Hunt list
- A malicious bootstrap server (operator-chosen, semi-trusted) — what can it forge into a
  connecting node's view, and what backstops catch it?
- Gossip message parsing: malformed/oversized/duplicate-key payloads, type confusion.
- Connection lifecycle: dedup-by-node_id races, half-open connections, handshake downgrade.
- Any `verify_signature` / credential check that can be bypassed with an empty/None/format-trick.

Follow the `00_INDEX.md` output format. Report only.
