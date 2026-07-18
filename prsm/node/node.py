"""
PRSM Node Runtime
=================

Main orchestrator that wires together identity, ledger, transport,
discovery, gossip, compute, storage, and the management API into
a single running node.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional
from decimal import Decimal

from contextlib import suppress
from pathlib import Path

from prsm.node.config import NodeConfig, NodeRole
from prsm.node.identity import (
    NodeIdentity,
    generate_node_identity,
    load_node_identity,
    save_node_identity,
)

try:
    from prsm.data.embeddings.real_embedding_api import RealEmbeddingAPI
    _HAS_EMBEDDING_API = True
except Exception:
    _HAS_EMBEDDING_API = False
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.dag_ledger import DAGLedger
from prsm.node.transport import WebSocketTransport
from prsm.node.discovery import PeerDiscovery
from prsm.node.gossip import GossipProtocol
from prsm.node.compute_provider import ComputeProvider
from prsm.node.compute_requester import ComputeRequester
from prsm.node.storage_provider import StorageProvider
from prsm.node.content_uploader import ContentUploader
from prsm.node.content_index import ContentIndex
from prsm.node.content_provider import ContentProvider
from prsm.node.ledger_sync import LedgerSync
from prsm.node.agent_registry import AgentRegistry
from prsm.node.agent_collaboration import AgentCollaboration, BidStrategy
from prsm.economy.tokenomics.staking_manager import StakingManager, StakingConfig
from prsm.config.networks import resolve_endpoints as _resolve_endpoints
from prsm.economy.ftns_onchain import OnChainFTNSLedger
from prsm.node.content_economy import ContentEconomy, RoyaltyModel

# BitTorrent integration
from prsm.core.bittorrent_client import BitTorrentClient, BitTorrentConfig
from prsm.core.bittorrent_manifest import TorrentManifestStore
from prsm.node.bittorrent_provider import BitTorrentProvider, BitTorrentProviderConfig
from prsm.node.bittorrent_requester import BitTorrentRequester, BitTorrentRequesterConfig

logger = logging.getLogger(__name__)


def _check_libp2p_origin_auth_gap() -> None:
    """sp1010/sp1246 — startup advisory for the libp2p transport (Residual A).

    Origin-authentication is now PORTED to libp2p: discovery announces self-verify
    their data (sp1086/1087 attestation + sp1097 replay/cap) and the gossip layer
    authenticates the message origin (sp1246: the same sp934 envelope attestation +
    fallback the WebSocket GossipProtocol uses), so the eclipse / index-poisoning
    vectors the WebSocket stack defends against are now covered on libp2p too.

    libp2p nonetheless remains OPT-IN: transport_backend defaults to "websocket"
    (config.py, sp1188) and the live fleet runs WebSocket exclusively; the native
    library is also absent on the Linux fleet, so the path is latent. Two honest
    residuals keep it opt-in until a live cross-host validation: there is still no
    authenticated PEER-EXCHANGE relay path on libp2p (no DISCOVERY_PEER_REQUEST/
    RESPONSE equivalent), and direct-message sender binding (the Noise-authenticated
    PeerID → node_id cross-check) is a deferred follow-on. This advisory logs WARNING
    and still hard-refuses when an operator opts into strict enforcement via
    PRSM_FORBID_UNAUTHENTICATED_LIBP2P. See
    docs/2026-06-04-p2p-substrate-hunt-residuals.md (Residual A).
    """
    msg = (
        "transport_backend=libp2p selected. Origin-authentication IS ported "
        "(discovery sp1086/1087/1097 + gossip-layer sp1246), so eclipse / index "
        "poisoning are covered — but libp2p is still OPT-IN: no authenticated "
        "peer-exchange relay path yet and direct-message sender-binding is a "
        "deferred follow-on, so prefer PRSM_TRANSPORT_BACKEND=websocket until a live "
        "cross-host validation. See docs/2026-06-04-p2p-substrate-hunt-residuals.md "
        "(Residual A)."
    )
    logger.warning("SECURITY: %s", msg)
    if os.environ.get("PRSM_FORBID_UNAUTHENTICATED_LIBP2P", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        raise RuntimeError(
            "Refusing to start the libp2p transport: " + msg
            + " (PRSM_FORBID_UNAUTHENTICATED_LIBP2P is set)."
        )


def build_persistent_privacy_budget(data_dir, identity, max_epsilon: float = 100.0):
    """Construct a PersistentPrivacyBudgetTracker rooted at <data_dir>/privacy_budget/.

    Phase 3.x.4 wiring factory. Imported by Node.__init__ AND by the
    integration test ``tests/integration/test_node_privacy_budget_persistence.py``
    so any drift between production wiring and the test harness is a
    compile-time error rather than a silent integration gap.

    Raises ``JournalCorruptionError`` (from prsm.security.privacy_budget_persistence)
    if the existing journal at the configured path fails verify_chain
    on construction. Caller MUST let this propagate — silently falling
    back to in-memory loses the audit trail.
    """
    from pathlib import Path

    from prsm.security.privacy_budget_persistence import (
        FilesystemPrivacyBudgetStore,
        PersistentPrivacyBudgetTracker,
    )

    budget_dir = Path(data_dir) / "privacy_budget"
    budget_dir.mkdir(parents=True, exist_ok=True)
    store = FilesystemPrivacyBudgetStore(budget_dir, identity.public_key_b64)
    return PersistentPrivacyBudgetTracker(
        max_epsilon=max_epsilon, store=store, identity=identity
    )


def _is_valid_eth_address(addr: Optional[str]) -> bool:
    """Cheap format check for 0x-prefixed 20-byte Ethereum address."""
    if not isinstance(addr, str):
        return False
    if not addr.startswith("0x") or len(addr) != 42:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in addr[2:])


def _redact_rpc_url(url: str) -> str:
    """Sprint 171 — redact RPC URLs for logging.

    Returns the URL with the path/query stripped so logs never
    carry Alchemy / Infura / Quicknode API keys (which live in
    the URL path). Falls back to ``"<rpc>"`` for unparseable
    input rather than echoing potentially-sensitive material.

    Examples
    --------
    >>> _redact_rpc_url("https://base-mainnet.g.alchemy.com/v2/SECRET")
    'https://base-mainnet.g.alchemy.com'
    >>> _redact_rpc_url("https://mainnet.base.org")
    'https://mainnet.base.org'
    >>> _redact_rpc_url("")
    '<rpc>'
    """
    if not url:
        return "<rpc>"
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        if p.scheme and p.hostname:
            host = p.hostname
            if p.port:
                host = f"{host}:{p.port}"
            return f"{p.scheme}://{host}"
        return "<rpc>"
    except Exception:  # noqa: BLE001
        return "<rpc>"


def _derive_creator_address(ftns_ledger: Optional[Any]) -> Optional[str]:
    """Resolve the on-chain creator 0x address for this node.

    Priority:
      1. ftns_ledger._connected_address — canonical on-chain identity
         derived from FTNS_WALLET_PRIVATE_KEY in OnChainFTNSLedger.__init__.
      2. PRSM_CREATOR_ADDRESS env var — for nodes without the full
         on-chain stack that still want to register content on-chain.
      3. None — backward compat; on-chain routing silently skips and
         local fallback handles payments.

    Invalid addresses (bad format, empty string) log a warning and
    fall through rather than poisoning the on-chain registry with a
    hash bound to a garbage address.

    Phase 1.3 Task 3a.
    """
    if ftns_ledger is not None:
        addr = getattr(ftns_ledger, "_connected_address", None)
        if addr:
            if _is_valid_eth_address(addr):
                return addr
            logger.warning(
                f"ftns_ledger._connected_address has invalid format: "
                f"{addr!r}; falling through to env var."
            )

    env_addr = os.environ.get("PRSM_CREATOR_ADDRESS")
    if env_addr:
        if _is_valid_eth_address(env_addr):
            return env_addr
        logger.warning(
            f"PRSM_CREATOR_ADDRESS env var has invalid format: "
            f"{env_addr!r}; on-chain routing disabled for this node. "
            f"Local royalty fallback will be used."
        )

    return None


def _build_provenance_client_or_none():
    """T6 (2026-05-05): construct an on-chain ProvenanceRegistryClient
    if all required env vars are set. Returns None on any miss — the
    caller treats None as "skip on-chain registration."

    Required env vars:
      PRSM_ONCHAIN_PROVENANCE=1
      PRSM_PROVENANCE_REGISTRY_ADDRESS=<0x...>
      FTNS_WALLET_PRIVATE_KEY=<0x...>
      PRSM_BASE_RPC_URL=<https://...>  (optional; defaults to Base mainnet)

    Mirrors content_economy.py's _get_provenance_client() pattern but
    constructed eagerly at node-startup so the resulting client is
    available to ContentUploader at upload-time without a lazy-init
    race.
    """
    if os.getenv("PRSM_ONCHAIN_PROVENANCE", "").lower() not in ("1", "true", "yes"):
        return None
    addr = os.getenv("PRSM_PROVENANCE_REGISTRY_ADDRESS", "").strip()
    pk = os.getenv("FTNS_WALLET_PRIVATE_KEY", "").strip()
    # Sprint 526 — F42 fix: V2 ProvenanceRegistry routing. Detect whether
    # the configured address matches V2 (canonical going forward) and
    # dispatch to ProvenanceRegistryV2Client. The V2 client exposes a
    # V1-compatible `register_content` shim (sprint 526) so the auto-register
    # caller is contract-agnostic.
    v2_addr = ""
    if os.getenv("PRSM_NETWORK", "").strip():
        try:
            ep = _resolve_endpoints()
            v2_addr = (getattr(ep, "provenance_registry_v2", None) or "").strip()
            if not addr:
                # Canonical fallback: prefer V2 when wired; else V1.
                addr = (v2_addr or ep.provenance_registry or "").strip()
        except Exception:  # noqa: BLE001
            addr = addr or ""
    if not addr or not pk:
        if not addr:
            logger.info(
                "PRSM_ONCHAIN_PROVENANCE=1 but PRSM_PROVENANCE_REGISTRY_ADDRESS "
                "not set — uploads will not register on-chain."
            )
        if not pk:
            logger.info(
                "PRSM_ONCHAIN_PROVENANCE=1 but FTNS_WALLET_PRIVATE_KEY not "
                "set — cannot sign on-chain registerContent calls."
            )
        return None
    try:
        rpc_url = _resolve_endpoints().rpc_url
        # Pick V2 client if the wired address is the V2 contract.
        if v2_addr and addr.lower() == v2_addr.lower():
            from prsm.economy.web3.provenance_registry_v2 import (
                ProvenanceRegistryV2Client,
            )
            client = ProvenanceRegistryV2Client(
                rpc_url=rpc_url,
                contract_address=addr,
                private_key=pk,
            )
            logger.info(
                f"on-chain ProvenanceRegistry V2 wired: {addr} via "
                f"{_redact_rpc_url(rpc_url)}"
            )
        else:
            from prsm.economy.web3.provenance_registry import (
                ProvenanceRegistryClient,
            )
            client = ProvenanceRegistryClient(
                rpc_url=rpc_url,
                contract_address=addr,
                private_key=pk,
            )
            logger.info(
                f"on-chain ProvenanceRegistry V1 wired: {addr} via "
                f"{_redact_rpc_url(rpc_url)}"
            )
        return client
    except Exception as exc:
        logger.warning(
            f"failed to construct ProvenanceRegistryClient: "
            f"{type(exc).__name__}: {exc} — uploads will not register on-chain."
        )
        return None


# ──────────────────────────────────────────────────────────────────────
# PRSM-PROV-1 Item 6 — three-band dedup component builders.
# All three return None on any failure; the upload path falls back
# to legacy 2-band auto-attribute behavior when any component is None.
# ──────────────────────────────────────────────────────────────────────


def _build_threshold_resolver_or_none():
    """Construct the canonical ``ThresholdResolver`` from the
    project's ``prsm/data/dedup_thresholds.yaml``. Returns None on
    any IO/parse failure — uploads fall back to ``_SemanticIndex``
    class-constant thresholds.
    """
    try:
        from prsm.data.dedup.thresholds import ThresholdResolver
        return ThresholdResolver.from_default_path()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failed to load ThresholdResolver: %s — uploads will use "
            "legacy 2-band class-constant thresholds",
            exc,
        )
        return None


def _build_arbitration_queue_or_none():
    """Construct a ``FilesystemArbitrationQueue`` rooted at
    ``~/.prsm/arbitration_queue/``. Returns None on any IO failure
    — uploads then run without disputed-band recording (legacy 2-band).
    """
    try:
        from prsm.data.dedup.arbitration import FilesystemArbitrationQueue
        queue_dir = Path.home() / ".prsm" / "arbitration_queue"
        return FilesystemArbitrationQueue(queue_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failed to construct FilesystemArbitrationQueue: %s — "
            "uploads will not record disputed-band records",
            exc,
        )
        return None


def _build_arbitration_proposal_sink_or_none():
    """Construct a ``TokenWeightedVotingProposalSink`` if the operator
    has configured a system proposer address. Returns None otherwise.

    Required env var:
      ``PRSM_ARBITRATION_PROPOSER_ID`` — system-level proposer (the
      Foundation Safe address or a delegate). Must hold sufficient
      FTNS to cover proposal submission fees. Without this, the
      arbitration queue still runs (records persist + are retrievable
      via ``list_pending``), but no governance proposals are auto-
      created — councils may author proposals by hand from queue
      entries.

    Disable explicitly with ``PRSM_ARBITRATION_PROPOSER_ID=""``.
    """
    proposer_id = os.getenv("PRSM_ARBITRATION_PROPOSER_ID", "").strip()
    if not proposer_id:
        return None
    try:
        from prsm.economy.governance.arbitration_sink import (
            TokenWeightedVotingProposalSink,
        )
        from prsm.economy.governance.voting import TokenWeightedVoting
        voting = TokenWeightedVoting()
        sink = TokenWeightedVotingProposalSink(
            voting=voting,
            proposer_id=proposer_id,
        )
        logger.info(
            "TokenWeightedVotingProposalSink wired with proposer_id=%s "
            "(disputed-band records will surface as ARBITRATION_DISPUTE "
            "proposals)",
            proposer_id,
        )
        return sink
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failed to construct TokenWeightedVotingProposalSink "
            "(proposer_id=%s): %s — disputed-band records still queued, "
            "but no governance proposals auto-created",
            proposer_id,
            exc,
        )
        return None


class _ContentProviderBlobFetcher:
    """Sprint 1140 (Brick E.2) — adapts ``ContentProvider.request_content`` to the audit
    loop's ``BlobFetcher`` protocol (``async fetch(cid) -> bytes``). ``request_content``
    returns ``Optional[bytes]`` (None on a miss/timeout); this fetcher RAISES on None so
    the loop's per-batch fail-closed path treats a missing blob as a rejection rather than
    feeding ``deserialize_receipt_set(None)``."""

    def __init__(self, content_provider: Optional[Any]):
        self._cp = content_provider

    async def fetch(self, cid: str) -> bytes:
        if self._cp is None:
            raise RuntimeError("content_provider unavailable for audit fetch")
        blob = await self._cp.request_content(cid)
        if blob is None:
            raise RuntimeError(f"content {cid} not found")
        return bytes(blob)


class _FailClosedAnchor:
    """Pre-T3c stub: every lookup returns None, so cross-node manifest
    verification refuses to trust. Used when no production
    PublisherKeyAnchor address is configured. The fail-closed default
    is intentional: better to refuse than to accept unverified bytes.

    Hoisted to module scope (was previously nested in
    ``_build_dht_components_or_none``) so test code can detect the
    pre-anchor-wired path via isinstance.
    """

    def lookup(self, node_id):  # noqa: ARG002
        return None


def _fail_closed_creator_pubkey_for(content_hash):  # noqa: ARG001
    """Fallback creator-pubkey resolver. Returns None for every input.

    Used when no LocalEmbeddingIndex is configured (no T3d resolver
    can be built) or when the T3d wiring fails. Every embedding-DHT
    signature verification then returns None → SignatureVerification
    Error → reject. Cold-start correctness preserved at the cost of
    cross-node embedding fetch.
    """
    return None


def _make_creator_pubkey_for(embedding_index, anchor):
    """T3d (option (a)) — content_hash → creator_node_id → pubkey resolver.

    The verifier hands ``creator_pubkey_for`` a content_hash. We:
      1. look up the local LocalEmbeddingIndex for any record with that
         content_hash → creator_node_id (populated at upload time on
         the publisher node, or after a verified cross-node fetch on a
         relay node)
      2. anchor.lookup(creator_node_id) → base64 pubkey on-chain
      3. base64-decode → bytes for ed25519 verify

    Limitation (intentional): only resolves for content this node has
    previously seen. Cold-start cross-node embedding fetch — content
    B has never seen — still returns None → reject. This is the
    correct fail-closed behavior pre-(b)-extension. Once any node in
    the swarm has cached the (content_hash, creator_id) record, that
    node serves as a relay for verification.

    Returns the same fail-closed stub if either dependency is missing
    (defensive — the caller is responsible for not constructing the
    resolver in that case, but we don't trust the caller).
    """
    if embedding_index is None or anchor is None:
        return _fail_closed_creator_pubkey_for

    def _resolve(content_hash):
        try:
            creator_id = embedding_index.lookup_creator_by_content_hash(
                content_hash,
            )
        except Exception:  # noqa: BLE001
            return None
        if not creator_id:
            return None
        try:
            pubkey_b64 = anchor.lookup(creator_id)
        except Exception:  # noqa: BLE001
            return None
        if not pubkey_b64 or not isinstance(pubkey_b64, str):
            return None
        try:
            import base64
            return base64.b64decode(pubkey_b64, validate=True)
        except Exception:  # noqa: BLE001
            return None

    return _resolve


def _build_publisher_key_anchor_client_or_none():
    """T3c — construct a PublisherKeyAnchorClient from env vars.

    Mirrors ``_build_provenance_client_or_none``: env-driven, fail-soft.
    Returns None when any required piece is missing OR when the web3
    construction itself fails — the caller falls back to the
    fail-closed anchor in that case.

    Required env vars:
      PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS=<0x…>
      PRSM_BASE_RPC_URL=<https://…>  (optional; defaults to Base mainnet
                                       so the same env var that drives
                                       the provenance client also drives
                                       the anchor client)

    The anchor is read-only on the verifier side — no private_key is
    passed. Read-only mode supports lookup() but rejects register_self()
    (the publisher side has its own anchor instance for that with
    PRSM_FTNS_WALLET_PRIVATE_KEY).
    """
    addr = os.getenv("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "").strip()
    if not addr and os.getenv("PRSM_NETWORK", "").strip():
        # Sprint 146 — canonical fallback when PRSM_NETWORK declared.
        try:
            addr = (
                _resolve_endpoints().publisher_key_anchor or ""
            ).strip()
        except Exception:  # noqa: BLE001
            addr = ""
    if not addr:
        logger.debug(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS not set — DHT manifest "
            "verification will use fail-closed anchor."
        )
        return None
    rpc_url = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
    try:
        from prsm.security.publisher_key_anchor.client import (
            PublisherKeyAnchorClient,
        )
        client = PublisherKeyAnchorClient(
            contract_address=addr,
            rpc_url=rpc_url,
        )
        logger.info(
            f"PublisherKeyAnchorClient wired: {addr} via {_redact_rpc_url(rpc_url)}"
        )
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"PublisherKeyAnchorClient construction failed: "
            f"{type(exc).__name__}: {exc} — falling back to "
            f"fail-closed anchor."
        )
        return None


# ──────────────────────────────────────────────────────────────────────
# Phase 7-storage + Phase 8 client + scheduler builders.
#
# All four return None on any failure / missing env var. Operators
# opt in explicitly via address env var (constructs client) AND
# scheduler-enable env var (launches daemon). Either component
# missing → no-op; node still serves its other functions.
# ──────────────────────────────────────────────────────────────────────


def _build_compensation_distributor_client_or_none():
    """Construct a CompensationDistributorClient if env-driven config
    is complete. Closes the §6.2 first-deferred item from the 2026-05
    exploit-response annex.

    Required env vars:
      PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS=<0x...>
      FTNS_WALLET_PRIVATE_KEY=<0x...>
      PRSM_BASE_RPC_URL=<https://...>  (optional; defaults to Base mainnet)

    Returns None on any miss or construction failure — the caller
    treats None as "no compensation distributor client wired."
    """
    addr = os.getenv("PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS", "").strip()
    pk = os.getenv("FTNS_WALLET_PRIVATE_KEY", "").strip()
    if not addr and os.getenv("PRSM_NETWORK", "").strip():
        # Sprint 144 — canonical fallback when network resolved.
        try:
            addr = (
                _resolve_endpoints().compensation_distributor or ""
            ).strip()
        except Exception:  # noqa: BLE001
            addr = ""
    if not addr or not pk:
        return None
    try:
        from prsm.economy.web3.compensation_distributor import (
            CompensationDistributorClient,
        )
        rpc_url = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
        client = CompensationDistributorClient(
            rpc_url=rpc_url,
            contract_address=addr,
            private_key=pk,
        )
        logger.info(
            f"CompensationDistributorClient wired: {addr} via {_redact_rpc_url(rpc_url)}"
        )
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"CompensationDistributorClient construction failed: "
            f"{type(exc).__name__}: {exc} — pull-and-distribute "
            f"surface unavailable."
        )
        return None


def _build_key_distribution_client_or_none():
    """Construct a KeyDistributionClient if env-driven config is
    complete.

    Required env vars:
      PRSM_KEY_DISTRIBUTION_ADDRESS=<0x...>
      FTNS_WALLET_PRIVATE_KEY=<0x...>  (optional for read-only paths;
        required for deposit_key / release / deauthorize writes)
      PRSM_BASE_RPC_URL=<https://...>  (optional; defaults to Base mainnet)

    Returns None on any miss / construction failure. Without it the
    node cannot drive Tier C key deposit / release-on-payment, and
    the KeyDistributionWatcher cannot launch.
    """
    addr = os.getenv("PRSM_KEY_DISTRIBUTION_ADDRESS", "").strip()
    if not addr and os.getenv("PRSM_NETWORK", "").strip():
        # Sprint 144 — canonical fallback when network resolved.
        try:
            addr = (
                _resolve_endpoints().key_distribution or ""
            ).strip()
        except Exception:  # noqa: BLE001
            addr = ""
    if not addr:
        return None
    pk = os.getenv("FTNS_WALLET_PRIVATE_KEY", "").strip() or None
    try:
        from prsm.economy.web3.key_distribution import KeyDistributionClient
        rpc_url = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
        client = KeyDistributionClient(
            rpc_url=rpc_url,
            contract_address=addr,
            private_key=pk,
        )
        logger.info(
            f"KeyDistributionClient wired: {addr} via {_redact_rpc_url(rpc_url)}"
            f"{' (read-only)' if pk is None else ''}"
        )
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"KeyDistributionClient construction failed: "
            f"{type(exc).__name__}: {exc} — Tier C surface unavailable."
        )
        return None


def _build_royalty_distributor_client_or_none():
    """Construct a RoyaltyDistributorClient if env-driven config is
    complete.

    Required env vars:
      PRSM_ROYALTY_DISTRIBUTOR_ADDRESS=<0x...>
      FTNS_TOKEN_ADDRESS=<0x...>  (or legacy FTNS_CONTRACT_ADDRESS)
        — RoyaltyDistributorClient constructor needs this for
        balanceOf reads of the distributor's own FTNS holdings
        (per-claim accounting).

    Optional env vars:
      FTNS_WALLET_PRIVATE_KEY=<0x...>  (write paths require this;
        read paths — claimable() — work without it)
      PRSM_BASE_RPC_URL=<https://...>  (defaults to Base mainnet)

    Read-only mode (no private key) is supported because the
    aggregate-source quoting path in `prsm_balance_check` only
    reads `claimable()`. Operators wanting to actually `claim()`
    their royalties must also configure the private key.

    Returns None when distributor address is unset or construction
    fails (e.g., RPC unreachable). The endpoint falls back to
    treating claimable_royalties as unavailable.
    """
    addr = os.getenv("PRSM_ROYALTY_DISTRIBUTOR_ADDRESS", "").strip()
    if not addr:
        # Sprint 144 — fall back to canonical for explicitly-resolved
        # network. Operators who set PRSM_NETWORK shouldn't ALSO have
        # to paste each per-contract address into env vars.
        if os.getenv("PRSM_NETWORK", "").strip():
            try:
                addr = (
                    _resolve_endpoints().royalty_distributor or ""
                ).strip()
            except Exception:  # noqa: BLE001
                addr = ""
        if not addr:
            return None
    pk = os.getenv("FTNS_WALLET_PRIVATE_KEY", "").strip() or None
    # FTNS token address is required for the RoyaltyDistributor
    # client's constructor. Fall back to the canonical Base mainnet
    # FTNS address if the env var isn't set — operators running
    # mainnet without override use the canonical pin.
    ftns_token = (
        os.getenv("FTNS_TOKEN_ADDRESS", "").strip()
        or os.getenv("FTNS_CONTRACT_ADDRESS", "").strip()
        or "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"  # canonical Base mainnet
    )
    try:
        from prsm.economy.web3.royalty_distributor import (
            RoyaltyDistributorClient,
        )
        rpc_url = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
        client = RoyaltyDistributorClient(
            rpc_url=rpc_url,
            distributor_address=addr,
            ftns_token_address=ftns_token,
            private_key=pk,
        )
        logger.info(
            f"RoyaltyDistributorClient wired: {addr} via {_redact_rpc_url(rpc_url)}"
            f"{' (read-only)' if pk is None else ''}"
        )
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"RoyaltyDistributorClient construction failed: "
            f"{type(exc).__name__}: {exc} — claimable-royalties surface "
            f"unavailable for aggregate balance quoting."
        )
        return None


def _build_storage_slashing_client_or_none():
    """Construct a StorageSlashingClient if env-driven config is
    complete.

    Required env vars:
      PRSM_STORAGE_SLASHING_ADDRESS=<0x...>
      FTNS_WALLET_PRIVATE_KEY=<0x...>
      PRSM_BASE_RPC_URL=<https://...>  (optional; defaults to Base mainnet)

    Returns None on any miss or construction failure. Without a
    StorageSlashingClient the node cannot heartbeat — providers will
    eventually become slashable via permissionless
    slash_for_missing_heartbeat. Operators running storage providers
    SHOULD set this; non-storage operator nodes can omit it safely.
    """
    addr = os.getenv("PRSM_STORAGE_SLASHING_ADDRESS", "").strip()
    pk = os.getenv("FTNS_WALLET_PRIVATE_KEY", "").strip()
    if not addr and os.getenv("PRSM_NETWORK", "").strip():
        # Sprint 144 — canonical fallback when network resolved.
        try:
            addr = (
                _resolve_endpoints().storage_slashing or ""
            ).strip()
        except Exception:  # noqa: BLE001
            addr = ""
    if not addr or not pk:
        return None
    try:
        from prsm.economy.web3.storage_slashing import (
            StorageSlashingClient,
        )
        rpc_url = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
        client = StorageSlashingClient(
            rpc_url=rpc_url,
            contract_address=addr,
            private_key=pk,
        )
        logger.info(
            f"StorageSlashingClient wired: {addr} via {_redact_rpc_url(rpc_url)}"
        )
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"StorageSlashingClient construction failed: "
            f"{type(exc).__name__}: {exc} — heartbeat surface unavailable."
        )
        return None


def _build_formal_invariant_backend_or_none(endpoints):
    """Construct a web3-backed FormalBackend for the sprint
    302 invariant checker, or None if unwirable.

    Returns None when:
      - web3 is not importable
      - RPC URL is unset
      - any backend bootstrap step raises

    The endpoint surface degrades cleanly to 503 for /check
    when this returns None, while the public /invariants
    list endpoint stays available (no backend needed for
    spec readout).
    """
    rpc = (endpoints.rpc_url or "").strip()
    if not rpc:
        return None
    try:
        from web3 import Web3, HTTPProvider
    except Exception:  # noqa: BLE001
        return None
    try:
        w3 = Web3(HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        # Smoke-test the connection. If it errors we report
        # None and let the checker surface SKIPPED rather
        # than crashing on every probe.
        try:
            w3.eth.chain_id  # noqa: B018
        except Exception:  # noqa: BLE001
            return None
        return _Web3FormalBackend(w3)
    except Exception:  # noqa: BLE001
        return None


class _Web3FormalBackend:
    """FormalBackend impl over web3.py — read-only EVM calls.

    Each method fail-softs to None on RPC error so the
    InvariantChecker can mark the result SKIPPED rather
    than crash. Backed by eth_call (low-level — no ABI
    objects needed) for the static selector probes, and
    by a token-balance helper that issues
    `balanceOf(holder)` against an ERC-20 contract.
    """

    _BALANCE_OF_SELECTOR = "0x70a08231"  # balanceOf(address)
    _HAS_ROLE_SELECTOR = "0x91d14854"    # hasRole(bytes32,address)

    def __init__(self, w3) -> None:
        self._w3 = w3

    def _eth_call(self, addr, data_hex):
        try:
            return self._w3.eth.call({
                "to": self._w3.to_checksum_address(addr),
                "data": data_hex,
            })
        except Exception:  # noqa: BLE001
            return None

    def call_uint256(self, addr, selector):
        raw = self._eth_call(addr, selector)
        if raw is None or len(raw) < 32:
            return None
        return int.from_bytes(raw[-32:], "big")

    def call_uint256_at_word(self, addr, selector, word_index):
        """Decode the `word_index`-th 32-byte word of a return
        value (e.g. one field of a struct getter). Returns None
        if the return is too short to contain that word."""
        raw = self._eth_call(addr, selector)
        start = word_index * 32
        end = start + 32
        if raw is None or len(raw) < end:
            return None
        return int.from_bytes(raw[start:end], "big")

    def call_address(self, addr, selector):
        raw = self._eth_call(addr, selector)
        if raw is None or len(raw) < 32:
            return None
        # Last 20 bytes of the 32-byte return word.
        return "0x" + raw[-20:].hex()

    def call_bool(self, addr, selector):
        raw = self._eth_call(addr, selector)
        if raw is None or len(raw) < 32:
            return None
        return int.from_bytes(raw[-32:], "big") != 0

    def token_balance_of(self, token, holder):
        try:
            holder_bytes = bytes.fromhex(
                holder.removeprefix("0x").rjust(64, "0"),
            )
        except ValueError:
            return None
        data = self._BALANCE_OF_SELECTOR + holder_bytes.hex()
        return self.call_uint256(token, data)

    def call_has_role(self, addr, role_hash, account):
        """ABI-encode hasRole(bytes32 role, address account)
        + dispatch eth_call. Returns bool or None on failure.

        Calldata layout:
          selector (4 bytes) || role_hash (32 bytes; bytes32
          is encoded as-is) || account (32 bytes; address is
          left-padded with 12 zero bytes).
        """
        try:
            role_clean = role_hash.removeprefix("0x").rjust(
                64, "0",
            )
            account_clean = account.removeprefix("0x").rjust(
                64, "0",
            )
            # Validate hex
            bytes.fromhex(role_clean)
            bytes.fromhex(account_clean)
        except ValueError:
            return None
        data = (
            self._HAS_ROLE_SELECTOR + role_clean
            + account_clean
        )
        return self.call_bool(addr, data)


def _build_heartbeat_scheduler_or_none(*, client):
    """Construct a HeartbeatScheduler if the operator opted in AND
    the underlying StorageSlashingClient is non-None.

    Activation env vars:
      PRSM_HEARTBEAT_SCHEDULER_ENABLED=1     (required to enable)
      PRSM_HEARTBEAT_SCHEDULER_INTERVAL_SECONDS=<seconds>   (optional;
        if set: explicit interval. If unset (default): the
        HeartbeatScheduler auto-tunes from
        client.heartbeat_grace_seconds() per its own internal
        DEFAULT/AUTO_TUNE_DIVISOR/MIN_INTERVAL constants.)

    Invalid interval (non-numeric / zero / negative) silently
    falls back to auto-tune rather than failing — the operator
    clearly wants the scheduler to run.
    """
    if client is None:
        return None
    if os.getenv("PRSM_HEARTBEAT_SCHEDULER_ENABLED", "").lower() not in (
        "1", "true", "yes",
    ):
        return None
    # Operator-explicit interval if env is set + valid; otherwise
    # None (which triggers auto-tune in HeartbeatScheduler.__init__).
    interval = None
    raw = os.getenv("PRSM_HEARTBEAT_SCHEDULER_INTERVAL_SECONDS", "").strip()
    if raw:
        try:
            parsed = float(raw)
            if parsed > 0:
                interval = parsed
        except ValueError:
            pass  # keep None → auto-tune
    try:
        from prsm.economy.web3.heartbeat_scheduler import HeartbeatScheduler
        scheduler = HeartbeatScheduler(
            client=client, interval_seconds=interval,
        )
        # scheduler.interval_seconds is now resolved (either operator-
        # explicit or auto-tuned).
        logger.info(
            f"HeartbeatScheduler wired (interval={scheduler.interval_seconds}s, "
            f"auto-tuned={'no' if interval is not None else 'yes'})"
        )
        return scheduler
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"HeartbeatScheduler construction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def _build_compensation_scheduler_or_none(*, client):
    """Construct a PullAndDistributeScheduler if the operator opted
    in AND the underlying CompensationDistributorClient is non-None.

    Activation env vars:
      PRSM_COMPENSATION_SCHEDULER_ENABLED=1                 (required)
      PRSM_COMPENSATION_SCHEDULER_INTERVAL_SECONDS=86400    (optional; default 86400 = 24h)

    Invalid interval (non-numeric / zero / negative / above the
    contract's 7-day monitoring threshold) silently falls back to
    default 86400s. The 7-day cap matches PullAndDistributeScheduler's
    own constructor invariant — falling back rather than raising
    keeps the scheduler running on operator misconfiguration.
    """
    if client is None:
        return None
    if os.getenv("PRSM_COMPENSATION_SCHEDULER_ENABLED", "").lower() not in (
        "1", "true", "yes",
    ):
        return None
    interval = 86400.0
    raw = os.getenv(
        "PRSM_COMPENSATION_SCHEDULER_INTERVAL_SECONDS", "",
    ).strip()
    if raw:
        try:
            parsed = float(raw)
            # Must satisfy PullAndDistributeScheduler's [0, 7days] band.
            if 0 < parsed <= 7 * 24 * 60 * 60:
                interval = parsed
        except ValueError:
            pass  # keep default
    try:
        from prsm.economy.web3.pull_and_distribute_scheduler import (
            PullAndDistributeScheduler,
        )
        scheduler = PullAndDistributeScheduler(
            client=client, interval_seconds=interval,
        )
        logger.info(
            f"PullAndDistributeScheduler wired (interval={interval}s)"
        )
        return scheduler
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"PullAndDistributeScheduler construction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


# ──────────────────────────────────────────────────────────────────────
# Phase 7-storage + Phase 8 event-watcher builders (2026-05-08).
#
# Watchers poll on-chain event logs and fire callbacks. Without a
# callback wired, the watcher does no polling (per its contract);
# the builders ship default INFO/WARNING-log callbacks so the watcher
# launches with out-of-the-box visibility. Operators wanting custom
# behavior can replace `node.<watcher>._on_<event>` post-construction
# OR construct the watcher directly with their own callbacks.
# ──────────────────────────────────────────────────────────────────────


def _parse_poll_interval(env_name: str, default: float) -> float:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
        if parsed > 0:
            return parsed
    except ValueError:
        pass
    return default


def _build_watcher_state_store_or_none():
    """Construct a shared FilesystemLastProcessedBlockStore for the
    3 event watchers if operator opted in.

    Activation:
      PRSM_WATCHER_STATE_PERSISTENCE_ENABLED=1   (required to enable)
      PRSM_WATCHER_STATE_DIR=<path>              (optional override
        of default ~/.prsm/watchers/)

    When the store is wired, each watcher's `last_processed_block`
    baseline persists across process restarts — events that landed
    during downtime get replayed when the watcher comes back online,
    instead of being silently skipped. Without it, watchers fall
    back to chain-tip baselining (legacy behavior).

    Returns None when persistence is disabled OR when filesystem
    construction fails (e.g., permission denied on the configured
    base_dir). The caller passes None through to the watcher
    builders, which preserves legacy behavior.
    """
    if os.getenv("PRSM_WATCHER_STATE_PERSISTENCE_ENABLED", "").lower() not in (
        "1", "true", "yes",
    ):
        return None
    try:
        from prsm.economy.web3.last_processed_block_store import (
            FilesystemLastProcessedBlockStore,
        )
        override = os.getenv("PRSM_WATCHER_STATE_DIR", "").strip()
        if override:
            from pathlib import Path
            store = FilesystemLastProcessedBlockStore(
                base_dir=Path(override),
            )
            logger.info(
                f"FilesystemLastProcessedBlockStore wired: {override}"
            )
        else:
            store = FilesystemLastProcessedBlockStore()
            logger.info(
                f"FilesystemLastProcessedBlockStore wired (default "
                f"{store.base_dir})"
            )
        return store
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"FilesystemLastProcessedBlockStore construction failed: "
            f"{type(exc).__name__}: {exc} — watchers will fall back "
            f"to chain-tip baseline (legacy behavior)."
        )
        return None


def _build_watcher_event_dedup_store_or_none():
    """Sprint 549 — sibling to the block-store builder. Returns an
    ``EventDedupStore`` so watchers can persistently dedup
    ``(tx_hash, log_index)`` event identifiers across restart.

    Without this, restart catch-up re-dispatches every event the
    previous run handled between callback dispatch and the post-loop
    baseline persist — duplicating distribution-log rows + webhook
    fires (CompensationDistributorWatcher) and similar in
    KeyDistribution + StorageSlashing (deferred follow-ons).

    Activation: piggybacks on the same env var as the block-store
    so operators get watcher persistence holistically:
      PRSM_WATCHER_STATE_PERSISTENCE_ENABLED=1   (required)
      PRSM_WATCHER_EVENT_DEDUP_DB=<path>         (optional override;
        default ~/.prsm/watcher_event_dedup.db)
    """
    if os.getenv(
        "PRSM_WATCHER_STATE_PERSISTENCE_ENABLED", "",
    ).lower() not in ("1", "true", "yes"):
        return None
    try:
        from prsm.economy.web3.last_processed_block_store import (
            EventDedupStore,
        )
        override = os.getenv("PRSM_WATCHER_EVENT_DEDUP_DB", "").strip()
        if override:
            db_path = override
        else:
            db_path = str(
                Path.home() / ".prsm" / "watcher_event_dedup.db"
            )
        store = EventDedupStore(db_path)
        logger.info(f"EventDedupStore wired ({db_path})")
        return store
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"EventDedupStore construction failed: "
            f"{type(exc).__name__}: {exc} — watcher event dedup "
            f"disabled (restart catch-up may double-dispatch)."
        )
        return None


_LOOPBACK_BIND_HOSTS = frozenset({
    "127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1",
})


def assess_public_bind_auth_posture(*, listen_host, api_key_present):
    """Sprint 960 — classify the API server's network-exposure auth posture.

    The protected money prefixes (/wallet/, /compute/, /transactions/) are only
    authenticated by NodeAuthMiddleware when PRSM_NODE_API_KEY is set. With the
    default listen_host of 0.0.0.0 and no key, those endpoints are reachable
    UNAUTHENTICATED by anyone who can route to the host.

    Returns ``(level, message)`` where level is ``"ok"`` or ``"insecure"``.
    ``"insecure"`` = bound to a non-loopback interface AND no API key. Loopback
    binds (local dev / reverse-proxy-fronted) and any bind WITH a key are ``"ok"``.
    Pure + side-effect-free so the caller decides whether to warn or fail-closed.
    """
    host = (listen_host or "").strip().lower()
    is_loopback = host in _LOOPBACK_BIND_HOSTS
    if is_loopback or api_key_present:
        return ("ok", "")
    msg = (
        f"SECURITY: node bound to non-loopback host {listen_host!r} with no "
        f"PRSM_NODE_API_KEY set — the protected money endpoints (/wallet/, "
        f"/compute/, /transactions/) are UNAUTHENTICATED and reachable by anyone "
        f"who can route to this host. Set PRSM_NODE_API_KEY=... (or bind 127.0.0.1 "
        f"behind an authenticating reverse proxy). Set "
        f"PRSM_REQUIRE_AUTH_ON_PUBLIC_BIND=1 to refuse to start in this posture."
    )
    return ("insecure", msg)


def should_refuse_insecure_public_bind(posture_level, *, allow_insecure):
    """sp1011 — fail-closed-by-default decision for the public-bind posture.

    Returns True (refuse to start) when ``posture_level`` is ``"insecure"`` (a
    non-loopback bind with no API key — money + KYC endpoints unauthenticated)
    AND the operator has NOT explicitly acknowledged the risk. The ack
    (PRSM_ALLOW_INSECURE_PUBLIC_BIND) is the single escape hatch for genuinely
    firewall-fronted deployments that intentionally run unauthenticated.

    Secure-by-default: the common case (CLI loopback bind, or any bind with a
    key) is ``"ok"`` and never refuses; only a deliberate non-loopback bind
    without a key is stopped.
    """
    if posture_level != "insecure":
        return False
    return not allow_insecure


def decide_api_key_provisioning(*, api_host, env_key, persisted_key, auto_provision):
    """Sprint 1195 — pure decision for self-provisioning the node API key.

    Decides what to do with the management-API auth key BEFORE the fail-closed
    public-bind posture gate runs, so a public-bind operator has a smooth,
    secure-by-default path instead of just hitting the refuse-to-start wall.
    Side-effect-free; the IO wrapper (``ensure_public_bind_api_key``) acts on it.

    Returns ``(action, reason)`` where action ∈:
      - ``"use_env"``      operator set PRSM_NODE_API_KEY explicitly — always wins.
      - ``"use_persisted"`` a public bind has no env key but a persisted key exists
                            → reuse it (STABLE across restarts so clients keep working).
      - ``"generate"``     a public bind has no key at all AND the operator opted into
                            PRSM_AUTO_PROVISION_API_KEY → mint + persist + log one.
      - ``"none"``         loopback dev bind (auth optional), or a public bind with no
                            key and no opt-in → leave it to the reviewed fail-closed gate.

    Loopback is deliberately untouched (the documented dev posture); we never
    silently flip a public bind from fail-closed to authenticated unless the
    operator opted in OR a key was already provisioned."""
    if (env_key or "").strip():
        return ("use_env", "operator-supplied PRSM_NODE_API_KEY")
    host = (api_host or "").strip().lower()
    if host in _LOOPBACK_BIND_HOSTS:
        return ("none", "loopback bind — API auth optional (dev posture)")
    if (persisted_key or "").strip():
        return ("use_persisted", "reusing the persisted node API key")
    if auto_provision:
        return ("generate", "auto-provisioning a node API key (PRSM_AUTO_PROVISION_API_KEY)")
    return ("none", "public bind with no key — fail-closed gate applies")


def _default_node_api_key_path():
    """~/.prsm/node_api_key, overridable via PRSM_NODE_API_KEY_FILE (mirrors the
    settlement-state-file convention)."""
    import os
    from pathlib import Path
    override = (os.environ.get("PRSM_NODE_API_KEY_FILE") or "").strip()
    if override:
        return Path(override)
    return Path.home() / ".prsm" / "node_api_key"


def ensure_public_bind_api_key(*, api_host, key_path=None, env=None):
    """Sprint 1195 — self-provision / reuse the node API key for a public bind.

    Runs before the public-bind posture gate. Returns the resolved key (and sets
    ``PRSM_NODE_API_KEY`` in ``env`` so the gate sees it), or None when no key
    applies (loopback dev, or a public bind the operator hasn't opted into
    provisioning for → the fail-closed gate then stops startup as designed).

    Fail-SOFT on IO: a key-file read error is treated as "no persisted key"; a
    write error on generation logs + still sets the in-process env key (so the
    node comes up authenticated this run even if persistence failed) — it never
    raises, so it can't break startup or weaken the gate."""
    import os
    import secrets as _secrets
    e = env if env is not None else os.environ
    path = key_path if key_path is not None else _default_node_api_key_path()

    persisted = None
    try:
        if path.exists():
            persisted = path.read_text(encoding="utf-8").strip() or None
    except Exception as exc:  # noqa: BLE001 — unreadable file → treat as absent
        logger.warning("Sprint 1195 — could not read node API key file %s (%s); "
                       "treating as unset.", path, exc)

    auto_provision = (e.get("PRSM_AUTO_PROVISION_API_KEY") or "").strip().lower() in {
        "1", "true", "yes", "on"}
    action, reason = decide_api_key_provisioning(
        api_host=api_host,
        env_key=e.get("PRSM_NODE_API_KEY", ""),
        persisted_key=persisted,
        auto_provision=auto_provision,
    )

    if action == "use_env":
        return (e.get("PRSM_NODE_API_KEY") or "").strip()
    if action == "none":
        return None
    if action == "use_persisted":
        e["PRSM_NODE_API_KEY"] = persisted
        logger.info("Sprint 1195 — %s (from %s); management API is authenticated.",
                    reason, path)
        return persisted
    # action == "generate"
    import hashlib as _hashlib
    key = _secrets.token_urlsafe(32)
    persisted = False
    try:
        from prsm.node.identity import write_owner_only_file
        # sp1266 — atomic 0o600 write (no world-readable window for the API key file).
        write_owner_only_file(path, key)
        persisted_note = f"persisted to {path} (0600)"
        persisted = True
    except Exception as exc:  # noqa: BLE001 — persist failed → still authenticate this run
        logger.warning("Sprint 1195 — could not persist the auto-provisioned API key to "
                       "%s (%s); using it for THIS process only (will regenerate on "
                       "restart).", path, exc)
        persisted_note = "NOT persisted (write failed) — ephemeral for this process"
    e["PRSM_NODE_API_KEY"] = key
    if persisted:
        # sp1266 — do NOT log the raw key (it leaks the credential to log sinks). Point the
        # operator to the 0600 file + a short fingerprint to confirm which key is active.
        _fp = _hashlib.sha256(key.encode()).hexdigest()[:12]
        logger.warning(
            "Sprint 1195 — auto-provisioned a node API key for the public bind (%s). "
            "Read it from %s (0600); fingerprint sha256:%s. Use it as the Bearer token / "
            "X-API-Key for protected endpoints.", reason, path, _fp)
    else:
        # Ephemeral fallback: the file write failed, so the log is the ONLY way to surface the
        # key this run. Emit it with a strong warning (still better than an unusable node).
        logger.warning(
            "Sprint 1195 — auto-provisioned an EPHEMERAL node API key for the public bind "
            "(%s); could not persist it, so it is shown ONCE here (set PRSM_NODE_API_KEY "
            "explicitly to avoid logging secrets):\n    PRSM_NODE_API_KEY=%s\n  (%s)",
            reason, key, persisted_note)
    return key


def _build_key_distribution_watcher_or_none(
    *, client, state_store=None, webhook_deliverer=None,
    webhook_url=None, webhook_secret=None, dedup_store=None,
):
    """Construct a KeyDistributionWatcher if operator opted in AND
    underlying client is non-None.

    Activation:
      PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED=1     (required)
      PRSM_KEY_DISTRIBUTION_WATCHER_POLL_SECONDS=30   (optional)

    Default callbacks log each event at INFO level. Closes annex §5.4
    detection-actionability gap (KeyReleased monitoring).
    """
    if client is None:
        return None
    if os.getenv("PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED", "").lower() not in (
        "1", "true", "yes",
    ):
        return None
    interval = _parse_poll_interval(
        "PRSM_KEY_DISTRIBUTION_WATCHER_POLL_SECONDS", 30.0,
    )
    try:
        from prsm.economy.web3.key_distribution_watcher import (
            KeyDistributionWatcher,
        )

        async def _on_released(event):
            logger.info(
                "KeyDistributionWatcher: KeyReleased "
                "content_hash=0x%s recipient=%s",
                event.content_hash.hex(), event.recipient,
            )
            if webhook_deliverer is not None and webhook_url:
                try:
                    payload = {
                        "event": "key.released",
                        "content_hash": (
                            "0x" + event.content_hash.hex()
                        ),
                        "recipient": event.recipient,
                    }
                    await webhook_deliverer.deliver(
                        url=webhook_url,
                        event="key.released",
                        payload=payload,
                        secret=webhook_secret,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "key.released webhook dispatch raised: %s", exc,
                    )

        def _on_deposited(event):
            logger.info(
                "KeyDistributionWatcher: KeyDeposited "
                "content_hash=0x%s publisher=%s release_fee=%d",
                event.content_hash.hex(), event.publisher,
                event.release_fee_ftns_wei,
            )

        def _on_deauthorized(event):
            logger.info(
                "KeyDistributionWatcher: KeyDeauthorized "
                "content_hash=0x%s publisher=%s",
                event.content_hash.hex(), event.publisher,
            )

        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=_on_released,
            on_key_deposited=_on_deposited,
            on_key_deauthorized=_on_deauthorized,
            poll_interval_sec=interval,
            state_store=state_store,
            dedup_store=dedup_store,
        )
        logger.info(
            f"KeyDistributionWatcher wired (interval={interval}s, "
            f"persistence={'on' if state_store is not None else 'off'}, "
            f"dedup={'on' if dedup_store is not None else 'off'})"
        )
        return watcher
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"KeyDistributionWatcher construction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def _build_settlement_challenge_watcher_or_none(
    operator_address, receipt_store_getter=None, defense_stats=None):
    """Sprint 1382 — construct a SettlementChallengeWatcher if the operator opted in AND has an
    address to watch. Env:
      PRSM_SETTLEMENT_CHALLENGE_WATCHER_ENABLED=1         (required)
      PRSM_SETTLEMENT_CHALLENGE_WATCHER_POLL_SECONDS=60   (optional)
    Read-only (no key); settlement registry + RPC resolve from PRSM_NETWORK. Returns None on
    opt-out / no address / unresolvable config / any build error, so startup never breaks.

    sp1383 — when ``receipt_store_getter`` is provided, each new challenge is auto-defended:
    the node self-verifies the retained §7 receipt for the challenged leaf and logs whether the
    challenge is BAD-FAITH (receipt verifies) or LEGITIMATE (receipt fails).
    """
    import os

    if os.getenv("PRSM_SETTLEMENT_CHALLENGE_WATCHER_ENABLED", "").lower() not in (
        "1", "true", "yes",
    ):
        return None
    if not operator_address:
        logger.info("SettlementChallengeWatcher: no operator address resolved — not started.")
        return None
    try:
        from prsm.config.networks import resolve_endpoints
        e = resolve_endpoints()
        registry = (
            os.getenv("PRSM_SETTLEMENT_REGISTRY_ADDRESS")
            or getattr(e, "settlement_registry", "") or "").strip()
        rpc = (os.getenv("PRSM_BASE_RPC_URL") or getattr(e, "rpc_url", "") or "").strip()
        if not registry or not rpc:
            logger.warning(
                "SettlementChallengeWatcher: could not resolve registry/RPC — not started.")
            return None
        poll = float(os.getenv("PRSM_SETTLEMENT_CHALLENGE_WATCHER_POLL_SECONDS", "") or 60.0)
        from prsm.economy.web3.batch_settlement_contract_client import (
            Web3SettlementContractClient,
        )
        from prsm.economy.web3.settlement_challenge_watcher import (
            SettlementChallengeWatcher,
        )
        client = Web3SettlementContractClient(rpc_url=rpc, contract_address=registry)
        on_new_challenge = None
        if receipt_store_getter is not None:
            from prsm.settlement.challenge_auto_defense import (
                build_challenge_auto_defense,
            )
            on_new_challenge = build_challenge_auto_defense(
                receipt_store_getter,
                on_verdict=(defense_stats.record if defense_stats is not None else None))
        logger.info(
            "SettlementChallengeWatcher: watching challenges against %s (poll %.0fs, auto-defense %s)",
            operator_address, poll, "on" if on_new_challenge else "off")
        return SettlementChallengeWatcher(
            client, operator_address, poll_interval_sec=poll,
            on_new_challenge=on_new_challenge)
    except Exception as exc:  # noqa: BLE001 — never break startup on an optional watcher
        logger.warning("SettlementChallengeWatcher build failed (%s) — not started.", exc)
        return None


def _build_storage_slashing_watcher_or_none(
    *, client, state_store=None, slash_event_log=None,
    heartbeat_log=None, webhook_deliverer=None,
    webhook_url=None, webhook_secret=None, dedup_store=None,
):
    """Construct a StorageSlashingWatcher if operator opted in AND
    underlying client is non-None.

    Activation:
      PRSM_STORAGE_SLASHING_WATCHER_ENABLED=1     (required)
      PRSM_STORAGE_SLASHING_WATCHER_POLL_SECONDS=30   (optional)

    Default callbacks: HeartbeatRecorded → INFO; ProofFailureSlashed
    + HeartbeatMissingSlashed → WARNING (own-provider monitoring is
    higher-attention than fleet-liveness observation).
    """
    if client is None:
        return None
    if os.getenv("PRSM_STORAGE_SLASHING_WATCHER_ENABLED", "").lower() not in (
        "1", "true", "yes",
    ):
        return None
    interval = _parse_poll_interval(
        "PRSM_STORAGE_SLASHING_WATCHER_POLL_SECONDS", 30.0,
    )
    try:
        from prsm.economy.web3.storage_slashing_watcher import (
            StorageSlashingWatcher,
        )

        def _on_recorded(event):
            logger.info(
                "StorageSlashingWatcher: HeartbeatRecorded "
                "provider=%s timestamp=%d",
                event.provider, event.timestamp,
            )
            if heartbeat_log is not None:
                try:
                    heartbeat_log.append(
                        provider=event.provider,
                        onchain_timestamp=event.timestamp,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "heartbeat_log.append raised: %s", exc,
                    )

        async def _on_proof(event):
            logger.warning(
                "StorageSlashingWatcher: ProofFailureSlashed "
                "provider=%s challenger=%s shard_id=0x%s slash_id=0x%s",
                event.provider, event.challenger,
                event.shard_id.hex(), event.slash_id.hex(),
            )
            if slash_event_log is not None:
                try:
                    slash_event_log.append(
                        kind="proof_failure_slashed",
                        provider=event.provider,
                        challenger=event.challenger,
                        slash_id=event.slash_id,
                        extras={
                            "shard_id": "0x" + event.shard_id.hex(),
                            "evidence_hash": (
                                "0x" + event.evidence_hash.hex()
                            ),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "slash_event_log.append (proof) raised: %s",
                        exc,
                    )
            if webhook_deliverer is not None and webhook_url:
                try:
                    payload = {
                        "event": "slash.proof_failure_slashed",
                        "provider": event.provider,
                        "challenger": event.challenger,
                        "shard_id": "0x" + event.shard_id.hex(),
                        "evidence_hash": (
                            "0x" + event.evidence_hash.hex()
                        ),
                        "slash_id": "0x" + event.slash_id.hex(),
                    }
                    await webhook_deliverer.deliver(
                        url=webhook_url,
                        event="slash.proof_failure_slashed",
                        payload=payload,
                        secret=webhook_secret,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "slash.proof_failure webhook dispatch raised: %s",
                        exc,
                    )

        async def _on_missing(event):
            logger.warning(
                "StorageSlashingWatcher: HeartbeatMissingSlashed "
                "provider=%s challenger=%s last_heartbeat_at=%d "
                "slash_id=0x%s",
                event.provider, event.challenger,
                event.last_heartbeat_at, event.slash_id.hex(),
            )
            if slash_event_log is not None:
                try:
                    slash_event_log.append(
                        kind="heartbeat_missing_slashed",
                        provider=event.provider,
                        challenger=event.challenger,
                        slash_id=event.slash_id,
                        extras={
                            "last_heartbeat_at": event.last_heartbeat_at,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "slash_event_log.append (missing) raised: %s",
                        exc,
                    )
            if webhook_deliverer is not None and webhook_url:
                try:
                    payload = {
                        "event": "slash.heartbeat_missing_slashed",
                        "provider": event.provider,
                        "challenger": event.challenger,
                        "last_heartbeat_at": event.last_heartbeat_at,
                        "slash_id": "0x" + event.slash_id.hex(),
                    }
                    await webhook_deliverer.deliver(
                        url=webhook_url,
                        event="slash.heartbeat_missing_slashed",
                        payload=payload,
                        secret=webhook_secret,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "slash.heartbeat_missing webhook dispatch raised: "
                        "%s", exc,
                    )

        watcher = StorageSlashingWatcher(
            client=client,
            on_heartbeat_recorded=_on_recorded,
            on_proof_failure_slashed=_on_proof,
            on_heartbeat_missing_slashed=_on_missing,
            poll_interval_sec=interval,
            state_store=state_store,
            dedup_store=dedup_store,
        )
        logger.info(
            f"StorageSlashingWatcher wired (interval={interval}s, "
            f"persistence={'on' if state_store is not None else 'off'}, "
            f"dedup={'on' if dedup_store is not None else 'off'})"
        )
        return watcher
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"StorageSlashingWatcher construction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def _build_compensation_distributor_watcher_or_none(
    *, client, state_store=None, webhook_deliverer=None,
    webhook_url=None, webhook_secret=None, distribution_log=None,
    dedup_store=None,
):
    """Construct a CompensationDistributorWatcher if operator opted in
    AND underlying client is non-None.

    Activation:
      PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED=1     (required)
      PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_POLL_SECONDS=30   (optional)

    Default callback: Distributed → INFO (operator-side accounting
    visibility; not P0).
    """
    if client is None:
        return None
    if os.getenv(
        "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED", "",
    ).lower() not in ("1", "true", "yes"):
        return None
    interval = _parse_poll_interval(
        "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_POLL_SECONDS", 30.0,
    )
    try:
        from prsm.economy.web3.compensation_distributor_watcher import (
            CompensationDistributorWatcher,
        )

        async def _on_distributed(event):
            logger.info(
                "CompensationDistributorWatcher: Distributed "
                "to_creator=%d to_operator=%d to_grant=%d",
                event.to_creator, event.to_operator, event.to_grant,
            )
            if distribution_log is not None:
                try:
                    distribution_log.append(
                        to_creator=event.to_creator,
                        to_operator=event.to_operator,
                        to_grant=event.to_grant,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "distribution_log.append raised: %s", exc,
                    )
            if webhook_deliverer is not None and webhook_url:
                try:
                    payload = {
                        "event": "distribution.distributed",
                        "to_creator": event.to_creator,
                        "to_operator": event.to_operator,
                        "to_grant": event.to_grant,
                    }
                    await webhook_deliverer.deliver(
                        url=webhook_url,
                        event="distribution.distributed",
                        payload=payload,
                        secret=webhook_secret,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "distribution.distributed webhook dispatch "
                        "raised: %s", exc,
                    )

        watcher = CompensationDistributorWatcher(
            client=client,
            on_distributed=_on_distributed,
            poll_interval_sec=interval,
            state_store=state_store,
            dedup_store=dedup_store,
        )
        logger.info(
            f"CompensationDistributorWatcher wired (interval={interval}s, "
            f"persistence={'on' if state_store is not None else 'off'}, "
            f"dedup={'on' if dedup_store is not None else 'off'})"
        )
        return watcher
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"CompensationDistributorWatcher construction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def _build_dht_components_or_none(
    *, identity, listen_host, dht_listen_port,
    manifest_index, embedding_index,
    local_fingerprint_index=None,
):
    """PRSM-DHT-TRANSPORT T3b/T3c — opt-in construction of
    :class:`DHTNodeComponents`.

    Returns ``None`` when DHT is disabled or when the prerequisites
    can't be satisfied. The node continues to function without the
    DHT (FilesystemModelRegistry falls back to local-only lookup,
    ContentUploader skips cross-node embedding gossip) — same
    fail-soft behavior as ``_build_provenance_client_or_none`` above.

    Enabled when EITHER:
      - ``NodeConfig.dht_enabled == True`` (caller passes already
        through the indexes / identity), OR
      - ``PRSM_DHT_ENABLED=1`` env var is set (operator-side override).

    Trust inputs:
      - ``anchor`` for ManifestDHT verification — T3c wires
        :class:`PublisherKeyAnchorClient` from
        ``PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS`` + ``PRSM_BASE_RPC_URL``.
        When the address env var is unset, falls back to
        :class:`_FailClosedAnchor` so the node still boots, but
        cross-node manifest verification refuses every signature.
      - ``creator_pubkey_for`` for EmbeddingDHT — still
        :func:`_fail_closed_creator_pubkey_for`. Production wiring
        needs a content-hash → creator-node-id mapping that isn't
        ratified yet (the on-chain ProvenanceRegistry stores
        creator-as-EVM-address; the corresponding PRSM node_id mapping
        is the gap). Tracked as a follow-on; T3c lights up the
        manifest path without it.
      - ``verify_signature`` is real Ed25519 from cryptography.hazmat.
    """
    if manifest_index is None and embedding_index is None:
        return None
    try:
        from prsm.network.dht_components import DHTNodeComponents
        from prsm.node.transport_adapter import DirectAdapter
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            f"DHT components unavailable (import failed): "
            f"{type(exc).__name__}: {exc}",
        )
        return None

    def _real_verify_signature(pubkey_bytes, message, signature) -> bool:
        if not pubkey_bytes:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(pubkey_bytes).verify(
                signature, message,
            )
        except InvalidSignature:
            return False
        except Exception:  # noqa: BLE001
            return False
        return True

    # T3c+T3d: build a single anchor instance shared between both
    # DHT paths. The anchor itself is constructed once — production
    # PublisherKeyAnchorClient when env vars are set, _FailClosedAnchor
    # otherwise. Both downstream paths inherit the same trust posture.
    shared_anchor = (
        _build_publisher_key_anchor_client_or_none()
        or _FailClosedAnchor()
    )

    anchor_for_manifest = shared_anchor if manifest_index is not None else None

    # T3d (option (a)): when only the embedding DHT is enabled, the
    # local-index resolver still needs an anchor for the on-chain
    # creator-pubkey lookup. Reuse the shared anchor.
    if embedding_index is not None:
        creator_pubkey_for = _make_creator_pubkey_for(
            embedding_index=embedding_index,
            anchor=shared_anchor,
        )
    else:
        creator_pubkey_for = None

    try:
        components = DHTNodeComponents.build(
            my_node_id=identity.node_id,
            my_host=listen_host or "127.0.0.1",
            dht_listen_port=dht_listen_port,
            transport_adapter=DirectAdapter(),
            listen_host=listen_host or "0.0.0.0",
            local_manifest_index=manifest_index,
            local_embedding_index=embedding_index,
            local_fingerprint_index=local_fingerprint_index,
            anchor=anchor_for_manifest,
            creator_pubkey_for=creator_pubkey_for,
            verify_signature=(
                _real_verify_signature if embedding_index is not None else None
            ),
        )
        # Stash the verifier inputs on the instance so start() can
        # forward them — keeps build() pure of trust state, while
        # avoiding a second hop through env. (Field name kept as
        # _t3b_* so the existing test suite continues to introspect
        # via stable attribute names; the underlying anchor is now
        # the production PublisherKeyAnchorClient when configured.)
        components._t3b_anchor = anchor_for_manifest  # noqa: SLF001
        components._t3b_creator_pubkey_for = creator_pubkey_for  # noqa: SLF001
        components._t3b_verify_signature = (  # noqa: SLF001
            _real_verify_signature if embedding_index is not None else None
        )
        return components
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"DHTNodeComponents.build failed: "
            f"{type(exc).__name__}: {exc} — node will run without DHT.",
        )
        return None


class _StakingFTNSAdapter:
    """Bridges node ledger to the FTNS interface expected by StakingManager."""

    def __init__(self, ledger: Any, node_id: str) -> None:
        self._ledger = ledger
        self._node_id = node_id
        self._locked_balances: Dict[str, Decimal] = {}  # user_id -> locked amount

    async def get_available_balance(self, user_id: str) -> Decimal:
        """Get available (unlocked) balance for a user."""
        balance = await self._ledger.get_balance(user_id)
        locked = self._locked_balances.get(user_id, Decimal('0'))
        return max(Decimal('0'), Decimal(str(balance)) - locked)

    async def lock_tokens(self, user_id: str, amount: Decimal, reason: str = "") -> bool:
        """Lock tokens for staking.

        Sprint 492 (F32 fix) — pre-fix this method swallowed
        ALL exceptions (including the InsufficientBalance
        ValueError it raised internally) and returned False
        silently. SettlerRegistry.register_settler didn't
        check the return value, so a settler could register
        a HUGE bond (live-tested: 10^12 FTNS against a 1083
        FTNS wallet) without any FTNS actually being locked.
        Anti-Sybil completely broken.

        Now: InsufficientBalance + ValueError propagate up
        so callers see the failure. Other unexpected errors
        still convert to False for backwards compat with
        unit tests that may pass odd ledger mocks."""
        available = await self.get_available_balance(user_id)
        if available < amount:
            raise ValueError(
                f"Insufficient available balance: "
                f"{available} < {amount}"
            )
        self._locked_balances[user_id] = (
            self._locked_balances.get(user_id, Decimal('0')) + amount
        )
        return True

    async def unlock_tokens(self, user_id: str, amount: Decimal, reason: str = "") -> bool:
        """Unlock tokens when unstaking."""
        try:
            current_locked = self._locked_balances.get(user_id, Decimal('0'))
            self._locked_balances[user_id] = max(Decimal('0'), current_locked - amount)
            return True
        except Exception:
            return False

    async def burn_tokens(self, user_id: str, amount: Decimal, reason: str = "") -> bool:
        """Burn tokens (for slashing)."""
        try:
            await self._ledger.debit(
                wallet_id=user_id,
                amount=float(amount),
                tx_type=TransactionType.PENALTY,
                description=reason or "Slashing penalty",
            )
            # Also reduce locked balance
            current_locked = self._locked_balances.get(user_id, Decimal('0'))
            self._locked_balances[user_id] = max(Decimal('0'), current_locked - amount)
            return True
        except Exception:
            return False

    async def mint_tokens(self, user_id: str, amount: Decimal, reason: str = "") -> bool:
        """Mint tokens (for rewards or appeal refunds)."""
        try:
            await self._ledger.credit(
                wallet_id=user_id,
                amount=float(amount),
                tx_type=TransactionType.REWARD,
                description=reason or "Staking reward",
            )
            return True
        except Exception:
            return False


async def _drain_task_bounded(
    task: "Optional[asyncio.Task]",
    timeout: float,
    name: str = "task",
) -> bool:
    """Sprint 955 — bounded drain of a daemon task during node shutdown.

    Awaits ``task`` for up to ``timeout`` seconds via ``asyncio.wait`` (which
    returns at the deadline regardless of the task's state). If it finishes,
    its result is reaped (a non-cancel exception is logged + swallowed —
    shutdown is best-effort) and True is returned. If it does NOT finish in
    time, the task is cancel-REQUESTED but ABANDONED (we never await the
    cancellation) and False is returned.

    This is the load-bearing difference from ``asyncio.wait_for(task, timeout)``:
    wait_for, on timeout, cancels the task AND THEN AWAITS the cancellation —
    so a task stuck in an uncancellable await (a blocking executor RPC, a
    subprocess teardown) makes the shutdown itself hang forever. Abandoning the
    task guarantees node.stop() returns within the bound (the process is
    stopping; an orphaned task is acceptable). Sibling of sp953's bounded
    transport shutdown, one layer up in the node's own stop sequence."""
    if task is None:
        return True
    done, _pending = await asyncio.wait({task}, timeout=timeout)
    if task in done:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 — best-effort shutdown
            logger.warning("node.stop: %s raised on stop: %s", name, exc)
        return True
    logger.warning(
        "node.stop: %s did not wind down within %.1fs — cancel-requested and "
        "abandoned (avoids hanging shutdown).", name, timeout,
    )
    task.cancel()  # request only — do NOT await (a stuck task would re-hang us)
    return False


# sp956 — generous per-subsystem shutdown bound. Legitimate subsystem stops
# complete in well under this; it only bites a genuinely-stuck subsystem (an
# uncancellable external dep — libp2p subprocess teardown, a wedged SQLite
# close, a libtorrent shutdown, a chain RPC). Generous so a merely-slow stop
# (e.g. a large SQLite WAL checkpoint) is not abandoned prematurely.
_STOP_TIMEOUT = 10.0


async def _await_bounded(coro: "Awaitable", timeout: float, name: str) -> bool:
    """Sprint 956 — coroutine form of `_drain_task_bounded` for node.stop()'s
    subsystem stops. Runs ``coro`` as a task and drains it up to ``timeout``;
    if it doesn't finish, cancel-REQUESTS and ABANDONS it (never awaits the
    cancellation), guaranteeing node shutdown can't hang on a stuck subsystem.
    Returns True if the stop finished within the bound, False if abandoned."""
    return await _drain_task_bounded(asyncio.ensure_future(coro), timeout, name)


class PRSMNode:
    """A fully operational PRSM network node.

    Orchestrates all subsystems:
    - Identity (Ed25519 keypair)
    - Local FTNS ledger (SQLite)
    - WebSocket P2P transport
    - Peer discovery (bootstrap + gossip)
    - Gossip protocol
    - Compute provider (accept jobs)
    - Compute requester (submit jobs)
    - Storage provider (ContentStore pins)
    - Content uploader (provenance + royalties)
    - Management API (FastAPI)
    """

    def __init__(self, config: Optional[NodeConfig] = None) -> None:
        self.config = config or NodeConfig()
        self.config.ensure_dirs()

        # Subsystems (initialized in self.initialize())
        self.identity: Optional[NodeIdentity] = None
        self.ledger: Optional[LocalLedger] = None
        self.transport: Optional[WebSocketTransport] = None
        self.discovery: Optional[PeerDiscovery] = None
        self.gossip: Optional[GossipProtocol] = None
        self.compute_provider: Optional[ComputeProvider] = None
        self.compute_requester: Optional[ComputeRequester] = None
        self.storage_provider: Optional[StorageProvider] = None
        self.content_uploader: Optional[ContentUploader] = None
        self.content_index: Optional[ContentIndex] = None
        self.content_provider: Optional[ContentProvider] = None
        self.ledger_sync: Optional[LedgerSync] = None
        self.agent_registry: Optional[AgentRegistry] = None
        self.agent_collaboration: Optional[AgentCollaboration] = None
        self.staking_manager: Optional[StakingManager] = None
        # On-chain FTNS ledger (Base mainnet)
        self.ftns_ledger: Optional[OnChainFTNSLedger] = None
        # Content economy manager (Phase 4)
        self.content_economy: Optional[ContentEconomy] = None
        
        # BitTorrent components
        self.bt_client: Optional[BitTorrentClient] = None
        self.bt_manifest_store: Optional[TorrentManifestStore] = None
        self.bt_provider: Optional[BitTorrentProvider] = None
        self.bt_requester: Optional[BitTorrentRequester] = None

        # Native-storage migration PR 2c: ContentPublisher/Retriever
        # composed from the BT layer above. Set in initialize() once
        # the BT layer is live; None when libtorrent is unavailable.
        self.content_publisher: Optional[Any] = None
        self.content_retriever: Optional[Any] = None

        # PRSM-DHT-TRANSPORT T3b: opt-in DHT stack (Manifest + Embedding)
        # Constructed in initialize() iff dht_enabled or PRSM_DHT_ENABLED=1.
        self.dht_components: Optional[Any] = None

        self._started = False
        self._start_time: Optional[float] = None
        self._api_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Load or generate identity, initialize all subsystems."""
        # ── Identity ─────────────────────────────────────────────
        self.identity = load_node_identity(self.config.identity_path)
        if self.identity is None:
            self.identity = generate_node_identity(self.config.display_name)
            save_node_identity(self.identity, self.config.identity_path)
            logger.info(f"Generated new node identity: {self.identity.node_id}")
        else:
            logger.info(f"Loaded node identity: {self.identity.node_id}")

        # ── Local Ledger (DAG-based or legacy) ─────────────────────
        if self.config.ledger_type == "dag":
            self.ledger = DAGLedger(
                str(self.config.ledger_path),
                verify_signatures=False,
            )
        else:
            self.ledger = LocalLedger(str(self.config.ledger_path))
        await self.ledger.initialize()
        await self.ledger.create_wallet(self.identity.node_id, self.config.display_name)
        await self.ledger.create_wallet("system", "PRSM Network")

        # Issue welcome grant if this is a new wallet
        try:
            await self.ledger.issue_welcome_grant(
                self.identity.node_id, self.config.welcome_grant
            )
            logger.info(f"Welcome grant: {self.config.welcome_grant} FTNS")
        except ValueError:
            pass  # Already received welcome grant

        # ── Transport / Gossip / Discovery ───────────────────────
        # Derive local capabilities from node roles (used by both backends)
        local_capabilities: list[str] = []
        for role in self.config.roles:
            if role in (NodeRole.FULL, NodeRole.COMPUTE):
                if "compute" not in local_capabilities:
                    local_capabilities.append("compute")
            if role in (NodeRole.FULL, NodeRole.STORAGE):
                if "storage" not in local_capabilities:
                    local_capabilities.append("storage")
        self._local_capabilities = local_capabilities

        # Sprint 681 — best-effort load of local hardware profile for
        # advertisement in DISCOVERY_ANNOUNCE (sprint 680 plumbing).
        # None on any failure → peer simply doesn't advertise hardware
        # and is excluded from the sprint 682 DHT-backed pool.
        try:
            from prsm.node.hardware_profile_loader import (
                load_local_hardware_profile,
            )
            self._local_hardware_profile = load_local_hardware_profile()
        except Exception:  # noqa: BLE001
            self._local_hardware_profile = None

        # sp1114 (Domain-08 review MEDIUM-7) — load THIS node's raw TEE attestation quote
        # so the dispatch-time TEE-policy gate (api._enforce_tee_policy) can actually
        # admit confidential work when the node has real hardware. Was never populated in
        # production → every hardware policy failed (fail-safe but unusable). None when
        # the node has no quote (stays tier-none = ineligible for confidential work).
        try:
            from prsm.node.hardware_profile_loader import (
                load_tee_attestation_blob,
            )
            self._tee_node_attestation_blob = load_tee_attestation_blob()
        except Exception:  # noqa: BLE001
            self._tee_node_attestation_blob = None

        if self.config.transport_backend == "libp2p":
            # sp1010 — the libp2p gossip/discovery path carries none of the
            # WebSocket stack's origin-auth (Residual A). Warn (or refuse, if
            # enforcement is enabled) before instantiating it.
            _check_libp2p_origin_auth_gap()
            from prsm.node.libp2p_transport import Libp2pTransport
            from prsm.node.libp2p_gossip import Libp2pGossip
            from prsm.node.libp2p_discovery import Libp2pDiscovery

            self.transport = Libp2pTransport(
                identity=self.identity,
                host=self.config.listen_host,
                port=self.config.p2p_port,
                library_path=self.config.libp2p_library_path,
            )
            self.gossip = Libp2pGossip(transport=self.transport)
            self.discovery = Libp2pDiscovery(
                transport=self.transport,
                bootstrap_nodes=self.config.bootstrap_nodes,
                gossip=self.gossip,
                # Sprint 375 — thread the multi-region fallback
                # list from NodeConfig so libp2p-mode operators
                # get the same SPOF protection that WebSocket-
                # mode operators have had since the original
                # PeerDiscovery design.
                bootstrap_fallback_nodes=(
                    self.config.bootstrap_fallback_nodes
                ),
                bootstrap_fallback_enabled=(
                    self.config.bootstrap_fallback_enabled
                ),
                # Sprint 838 — advertise local hardware_profile
                # via bootstrap so cold-start joiners (NAT'd
                # operators) see the fleet's real capacity
                # without waiting on direct DISCOVERY_ANNOUNCE.
                local_hardware_profile=self._local_hardware_profile,
            )
            logger.info("Using libp2p transport backend")
        else:
            # ── WebSocket transport (fallback) ────────────────────
            self.transport = WebSocketTransport(
                identity=self.identity,
                host=self.config.listen_host,
                port=self.config.p2p_port,
                nonce_window=self.config.nonce_window,
                ws_ping_interval=self.config.ws_ping_interval,
                ws_ping_timeout=self.config.ws_ping_timeout,
                handshake_timeout=self.config.handshake_timeout,
                nonce_cleanup_interval=self.config.nonce_cleanup_interval,
            )

            self.gossip = GossipProtocol(
                transport=self.transport,
                fanout=self.config.gossip_fanout,
                default_ttl=self.config.gossip_ttl,
                heartbeat_interval=self.config.heartbeat_interval,
            )

            self.discovery = PeerDiscovery(
                transport=self.transport,
                bootstrap_nodes=self.config.bootstrap_nodes,
                bootstrap_connect_timeout=self.config.bootstrap_connect_timeout,
                bootstrap_retry_attempts=self.config.bootstrap_retry_attempts,
                bootstrap_fallback_enabled=self.config.bootstrap_fallback_enabled,
                bootstrap_fallback_nodes=self.config.bootstrap_fallback_nodes,
                bootstrap_validate_addresses=self.config.bootstrap_validate_addresses,
                bootstrap_backoff_base=self.config.bootstrap_backoff_base,
                bootstrap_backoff_max=self.config.bootstrap_backoff_max,
                target_peers=self.config.target_peers,
                announce_interval=self.config.announce_interval,
                maintenance_interval=self.config.maintenance_interval,
                peer_stale_timeout=self.config.peer_stale_timeout,
                local_capabilities=local_capabilities,
                local_hardware_profile=self._local_hardware_profile,
            )
            logger.info("Using WebSocket transport backend (fallback)")

        # ── Compute ──────────────────────────────────────────────
        self._marketplace_advertiser = None  # sp1459 — set below iff opted-in + a compute provider
        if NodeRole.FULL in self.config.roles or NodeRole.COMPUTE in self.config.roles:
            self.compute_provider = ComputeProvider(
                identity=self.identity,
                transport=self.transport,
                gossip=self.gossip,
                ledger=self.ledger,
                cpu_allocation_pct=self.config.cpu_allocation_pct,
                memory_allocation_pct=self.config.memory_allocation_pct,
                max_concurrent_jobs=self.config.max_concurrent_jobs,
                gpu_allocation_pct=self.config.gpu_allocation_pct,
                config=self.config,
            )
            self.compute_provider.allow_self_compute = self.config.allow_self_compute

            # sp1459 — OPT-IN marketplace advertising (default OFF: PRSM_MARKETPLACE_ADVERTISE).
            # Completes the long-dormant wiring — the advertiser had a full lifecycle but was never
            # constructed in prod, so no node ever broadcast a listing and the whole selection
            # machinery (directory → candidate pool → stake-weighted aggregator selection, sp1457
            # binding, the price-quote handler) fired on an empty directory. When an operator opts
            # in, this node broadcasts its signed compute listing (+ any stake binding) so it becomes
            # selectable supply. Fail-safe: None when not opted in / not a compute provider.
            try:
                from prsm.marketplace.advertiser import (
                    build_marketplace_advertiser_from_env,
                )
                self._marketplace_advertiser = build_marketplace_advertiser_from_env(
                    identity=self.identity,
                    gossip=self.gossip,
                    compute_provider=self.compute_provider,
                )
                if self._marketplace_advertiser is not None:
                    # The Task-5 price-quote handler reads the live price off this handle.
                    self.compute_provider._marketplace_advertiser = self._marketplace_advertiser
                    logger.info(
                        "marketplace advertising ENABLED (price=%.4f FTNS/shard, tier=%s, "
                        "verified_stake_binding=%s)",
                        self._marketplace_advertiser.price_per_shard_ftns,
                        self._marketplace_advertiser.stake_tier,
                        self._marketplace_advertiser.stake_eth_address is not None,
                    )
            except Exception as exc:  # noqa: BLE001 — advertising must never break node bring-up
                logger.warning("marketplace advertiser wiring skipped: %s", exc)
                self._marketplace_advertiser = None

        self.compute_requester = ComputeRequester(
            identity=self.identity,
            transport=self.transport,
            gossip=self.gossip,
            ledger=self.ledger,
            discovery=self.discovery,
        )

        # ── Storage ──────────────────────────────────────────────
        if NodeRole.FULL in self.config.roles or NodeRole.STORAGE in self.config.roles:
            self.storage_provider = StorageProvider(
                identity=self.identity,
                gossip=self.gossip,
                ledger=self.ledger,
                pledged_gb=self.config.storage_gb,
                config=self.config,
                transport=self.transport,
                discovery=self.discovery,
            )
            # Initialize bandwidth limits from config
            if self.config.upload_mbps_limit > 0 or self.config.download_mbps_limit > 0:
                # Update the bandwidth limiter with config values
                # Note: This is synchronous initialization, the async update happens in start()
                self.storage_provider.upload_mbps_limit = self.config.upload_mbps_limit
                self.storage_provider.download_mbps_limit = self.config.download_mbps_limit

        # ── Content Index ─────────────────────────────────────────
        self.content_index = ContentIndex(
            gossip=self.gossip,
            max_indexed_cids=self.config.max_indexed_cids,
            ledger=self.ledger,
        )

        # Optionally attach semantic embedding for near-duplicate detection
        _embedding_fn = None
        _embedding_model_id: Optional[str] = None
        if _HAS_EMBEDDING_API:
            try:
                _embed_api = RealEmbeddingAPI()
                # Sprint 431 (F9 fix) — embedding-dimension parity
                # invariant. The query orchestrator's embedder is
                # pinned to `sentence-transformers/all-MiniLM-L6-v2`
                # (384-dim) at the production wiring site. If the
                # upload-side RealEmbeddingAPI falls through to a
                # different provider (notably OpenAI ada-002 at
                # 1536-dim when `OPENAI_API_KEY` is set), stored
                # shard embeddings live in a different vector space
                # than query embeddings → numpy dot-product raises
                # "shapes (384,) and (1536,) not aligned" → forge
                # pipeline blows up entirely.
                #
                # Fix: pin the upload-side to the local
                # sentence_transformers provider via functools.partial
                # so both lanes share one model. To opt back into
                # OpenAI uploads (currently breaks query-side parity
                # until the orchestrator grows an OpenAI-sync path),
                # set PRSM_UPLOAD_EMBEDDING_PROVIDER explicitly.
                #
                # F9 was surfaced 2026-05-15 during sprint 431's
                # forge E2E verification. See dogfood-findings doc.
                import functools
                _pref_provider = os.environ.get(
                    "PRSM_UPLOAD_EMBEDDING_PROVIDER",
                    "sentence_transformers",
                )
                _embedding_fn = functools.partial(
                    _embed_api.generate_embedding,
                    preferred_provider=_pref_provider,
                )
                # T3.6 (PRSM-PROV-1): the model_id used to key the
                # cross-node EmbeddingDHT. Sourced from the same
                # RealEmbeddingAPI that produces vectors here so the
                # (vector, model_id) tuple is internally consistent.
                # Falls back to the env-configured local model name
                # when no remote provider is wired (Item 1 fallback
                # path).
                _embedding_model_id = getattr(
                    _embed_api, "_st_model_name", None,
                )
                # Sprint 431 — warn loudly if the operator has
                # OPENAI_API_KEY set but didn't override the upload
                # provider. The dim mismatch is silent at upload
                # time; surfaces only at forge time as a cryptic
                # numpy shape error. Make the trade-off explicit.
                if (
                    os.environ.get("OPENAI_API_KEY")
                    and _pref_provider == "sentence_transformers"
                ):
                    logger.info(
                        "Upload-side embeddings pinned to "
                        "sentence_transformers (384-dim) for "
                        "parity with the query orchestrator, "
                        "even though OPENAI_API_KEY is set. "
                        "Override with PRSM_UPLOAD_EMBEDDING_"
                        "PROVIDER=openai if you've also wired an "
                        "OpenAI-compatible orchestrator embedder "
                        "(currently not supported — would break "
                        "forge queries)."
                    )
            except Exception as _e:
                logger.debug(f"Embedding API unavailable, semantic dedup disabled: {_e}")

        _semantic_index_path = Path.home() / ".prsm" / "semantic_index.json"
        # T4.9.next3: persist the per-kind binary fingerprint index next
        # to the semantic index. Same lifecycle: persists across node
        # restarts so warm-cache dedup survives a process bounce.
        _fingerprint_index_path = Path.home() / ".prsm" / "fingerprint_index.json"

        # PRSM-PROV-1 Item 6 — three-band dedup wiring. All three
        # components degrade to None on failure; uploads still work
        # without arbitration (legacy 2-band behavior).
        _threshold_resolver = _build_threshold_resolver_or_none()
        _arbitration_queue = _build_arbitration_queue_or_none()
        # Expose on self so /content/arbitration/queue + the
        # prsm_arbitration_status MCP tool can surface pending
        # disputes for operator review.
        self._arbitration_queue = _arbitration_queue
        _arbitration_proposal_sink = _build_arbitration_proposal_sink_or_none()

        # Phase 7-storage + Phase 8 client/scheduler wiring (2026-05-08).
        # All four optional; node functions without them, just without
        # the corresponding contract-call surface. Schedulers depend on
        # their underlying clients so are constructed in pairs.
        self._compensation_distributor_client = (
            _build_compensation_distributor_client_or_none()
        )
        self._storage_slashing_client = _build_storage_slashing_client_or_none()
        self._compensation_scheduler = _build_compensation_scheduler_or_none(
            client=self._compensation_distributor_client,
        )
        self._heartbeat_scheduler = _build_heartbeat_scheduler_or_none(
            client=self._storage_slashing_client,
        )
        # Event watchers — same client-sharing as schedulers; activation
        # is independent (operator can want watching without scheduling
        # or vice versa). KeyDistributionClient is constructed here for
        # the watcher; it does NOT auto-launch any heartbeat-style
        # daemon (KeyDistribution is event-driven on the operator side,
        # not cadence-driven).
        self._key_distribution_client = _build_key_distribution_client_or_none()
        # Aggregate-source quoting (audit-prep §7.23 honest-scope
        # closure): RoyaltyDistributor read surface for
        # `prsm_balance_check` to surface claimable royalties
        # alongside on-chain FTNS balance.
        self._royalty_distributor_client = (
            _build_royalty_distributor_client_or_none()
        )
        # In-memory ring buffer of on-chain slash events. Wired to
        # StorageSlashingWatcher callbacks below; visible at
        # GET /admin/slash-history. Distinct from contract event log
        # (authoritative on-chain) — this is a fast operator
        # dashboard view.
        from prsm.node.slash_event_log import SlashEventRing
        from prsm.node.heartbeat_log import HeartbeatRecordedRing
        from prsm.node.distribution_log import DistributedEventRing
        # Slash event ring: opt-in filesystem persistence so slash
        # history survives node restart. Authoritative on-chain;
        # this is operator-dashboard convenience.
        from pathlib import Path as _PathForSlash
        _slash_dir_raw = os.environ.get(
            "PRSM_SLASH_EVENT_LOG_DIR", "",
        ).strip()
        _slash_persist_dir = (
            _PathForSlash(_slash_dir_raw) if _slash_dir_raw else None
        )
        self._slash_event_log = SlashEventRing(
            persist_dir=_slash_persist_dir,
        )
        # sp957 — CONSENSUS_MISMATCH evidence log for the single-provider
        # compute pay path (sp928 optimistic verification). When the sampler
        # catches a bonded provider returning a fabricated result, the evidence
        # lands here (opt-in persistent, visible at
        # GET /admin/consensus-mismatch-evidence). NOT an on-chain slash — an
        # autonomous slash from one node's re-execution is unsound; this is the
        # corpus a future authority-gated bridge consumes.
        from prsm.node.consensus_mismatch_log import ConsensusMismatchLog
        from prsm.node.onchain_stake_reader import OnChainStakeReader
        _cm_dir_raw = os.environ.get(
            "PRSM_CONSENSUS_MISMATCH_LOG_DIR", "",
        ).strip()
        self._consensus_mismatch_log = ConsensusMismatchLog(
            persist_dir=(
                _PathForSlash(_cm_dir_raw) if _cm_dir_raw else None
            ),
        )
        # Shared on-chain stake reader (graceful-degrade to 0 off-chain) used to
        # resolve a caught provider's bond posture for the evidence record.
        self._compute_stake_reader = OnChainStakeReader()
        _hb_dir_raw = os.environ.get(
            "PRSM_HEARTBEAT_LOG_DIR", "",
        ).strip()
        _dist_dir_raw = os.environ.get(
            "PRSM_DISTRIBUTION_LOG_DIR", "",
        ).strip()
        self._heartbeat_log = HeartbeatRecordedRing(
            persist_dir=(
                _PathForSlash(_hb_dir_raw) if _hb_dir_raw else None
            ),
        )
        self._distribution_log = DistributedEventRing(
            persist_dir=(
                _PathForSlash(_dist_dir_raw) if _dist_dir_raw else None
            ),
        )

        # Pre-construct webhook deliverer + log + URL + secret so
        # the StorageSlashingWatcher built below can fire
        # slash.proof_failure_slashed / slash.heartbeat_missing_slashed
        # webhooks. The DaemonWatchdog initialization further down
        # reuses the same _webhook_deliverer + _webhook_log when
        # PRSM_WEBHOOK_URL is set; otherwise these stay None.
        self._webhook_deliverer = None
        self._webhook_log = None
        _early_webhook_url = os.environ.get(
            "PRSM_WEBHOOK_URL", "",
        ).strip() or None
        _early_webhook_secret = (
            os.environ.get("PRSM_WEBHOOK_SECRET", "").strip() or None
        )
        if _early_webhook_url:
            try:
                from prsm.node.webhook_delivery import WebhookDeliverer
                from prsm.node.webhook_log import WebhookLogRing
                self._webhook_log = WebhookLogRing()
                self._webhook_deliverer = WebhookDeliverer(
                    log_ring=self._webhook_log,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Webhook deliverer construction failed: %s — "
                    "slash + daemon webhooks disabled",
                    exc,
                )

        # Shared state store for the 3 watchers — single instance,
        # 3 watcher_key namespaces inside. None when persistence is
        # disabled (legacy chain-tip-baseline behavior preserved).
        self._watcher_state_store = _build_watcher_state_store_or_none()
        # Sprint 549: shared event-dedup store, sibling to the block
        # store. Without it, restart catch-up re-dispatches every
        # event between the previous run's last successful baseline-
        # persist and the crash. KeyDistribution + StorageSlashing
        # watchers don't consume this yet (deferred to sprint 550 +
        # 551 follow-ons) — only CompensationDistributorWatcher.
        self._watcher_event_dedup_store = (
            _build_watcher_event_dedup_store_or_none()
        )
        self._key_distribution_watcher = (
            _build_key_distribution_watcher_or_none(
                client=self._key_distribution_client,
                state_store=self._watcher_state_store,
                webhook_deliverer=self._webhook_deliverer,
                webhook_url=_early_webhook_url,
                webhook_secret=_early_webhook_secret,
                dedup_store=self._watcher_event_dedup_store,
            )
        )
        self._storage_slashing_watcher = (
            _build_storage_slashing_watcher_or_none(
                client=self._storage_slashing_client,
                state_store=self._watcher_state_store,
                slash_event_log=self._slash_event_log,
                heartbeat_log=self._heartbeat_log,
                webhook_deliverer=self._webhook_deliverer,
                webhook_url=_early_webhook_url,
                webhook_secret=_early_webhook_secret,
                dedup_store=self._watcher_event_dedup_store,
            )
        )
        self._compensation_distributor_watcher = (
            _build_compensation_distributor_watcher_or_none(
                client=self._compensation_distributor_client,
                state_store=self._watcher_state_store,
                webhook_deliverer=self._webhook_deliverer,
                webhook_url=_early_webhook_url,
                webhook_secret=_early_webhook_secret,
                distribution_log=self._distribution_log,
                dedup_store=self._watcher_event_dedup_store,
            )
        )
        # Operator on-chain address for /admin/earnings-summary
        # heartbeat-status lookup. Resolution order:
        # PRSM_OPERATOR_ADDRESS explicit > FTNS_WALLET_PRIVATE_KEY
        # derived > None. Auto-derivation eliminates the
        # double-config error of setting PK but forgetting
        # PRSM_OPERATOR_ADDRESS.
        from prsm.node.operator_address import resolve_operator_address
        self._operator_address = resolve_operator_address()
        # sp1382 — background settlement-challenge watcher (opt-in, read-only). sp1383 — the
        # store-getter is resolved lazily (the receipt store is built later in init) so a challenge
        # auto-defends against the live retained-receipt store. sp1384 — verdict tallies for metrics.
        from prsm.settlement.challenge_auto_defense import ChallengeDefenseStats
        self._challenge_defense_stats = ChallengeDefenseStats()
        self._settlement_challenge_watcher = (
            _build_settlement_challenge_watcher_or_none(
                self._operator_address,
                receipt_store_getter=lambda: getattr(
                    self, "_settlement_inference_receipt_store", None),
                defense_stats=self._challenge_defense_stats))
        # Tasks created on start() — None until then.
        self._compensation_scheduler_task = None
        self._content_readvertise_task = None  # sp1343 — late-joiner catalog re-advertise
        self._settlement_poll_task = None  # sp1038 brick 2
        self._settlement_audit_task = None  # sp1140 brick E.2 (opt-in audit loop)
        self._escrow_drain_watch_task = None  # sp1422 — unauthorized-batch watch (opt-in)
        self._reputation_slash_watch_task = None  # sp1424 — on-chain slash -> reputation
        self._compute_integrity_watcher = None  # sp1307 — §7 ChallengeWatcher (opt-in)
        self._compute_integrity_watcher_task = None  # sp1307
        self._peer_compute_integrity_watcher = None  # sp1308 — cross-provider audit
        self._peer_integrity_watcher_task = None  # sp1308
        self._settlement_audit_bundle = None  # sp1140 — None unless PRSM_SETTLEMENT_AUDIT
        self._collateral_refresh_task = None  # sp1081 — attestation collateral refresh
        self._metrics_collector = None  # sp1217 — opt-in runtime MetricsCollector
        self._alert_manager = None  # sp1217 — opt-in runtime AlertManager
        # sp1219 — always-on, cheap inference serving counters (the metrics
        # only surface when the opt-in MetricsCollector reads them). Fail-soft
        # so observability can never block node construction.
        try:
            from prsm.core.monitoring.node_runtime_metrics import (
                InferenceServingCounters,
            )
            self._inference_serving_counters = InferenceServingCounters()
        except Exception:  # noqa: BLE001
            self._inference_serving_counters = None
        self._heartbeat_scheduler_task = None
        self._key_distribution_watcher_task = None
        self._storage_slashing_watcher_task = None
        self._settlement_challenge_watcher_task = None  # sp1382
        self._compensation_distributor_watcher_task = None

        # T3.6 (PRSM-PROV-1): LocalEmbeddingIndex backs the
        # EmbeddingDHT — every successful upload + embedding gets a
        # creator-signed record persisted here, so peers querying us
        # via the DHT can verify and serve them. Stored under
        # ~/.prsm/embedding_index/. None disables the registration
        # path (existing behavior preserved when the embedding_dht
        # subpackage is unavailable).
        _embedding_index = None
        try:
            from prsm.network.embedding_dht.local_index import (
                LocalEmbeddingIndex,
            )
            _embedding_index_path = Path.home() / ".prsm" / "embedding_index"
            _embedding_index_path.mkdir(parents=True, exist_ok=True)
            _embedding_index = LocalEmbeddingIndex(_embedding_index_path)
        except Exception as _e:  # noqa: BLE001
            logger.debug(
                f"EmbeddingDHT local index unavailable, cross-node "
                f"dedup-serve disabled: {_e}"
            )

        # T4.9.next5: parallel server-side store for the binary
        # fingerprint lane. Same lifecycle + failure-mode posture as
        # _embedding_index above. Without this, peers can ASK us for
        # fingerprints but we'd have nothing to serve.
        _local_fingerprint_index = None
        try:
            from prsm.network.embedding_dht.local_fingerprint_index import (
                LocalFingerprintIndex,
            )
            _local_fp_index_path = (
                Path.home() / ".prsm" / "local_fingerprint_index"
            )
            _local_fp_index_path.mkdir(parents=True, exist_ok=True)
            _local_fingerprint_index = LocalFingerprintIndex(
                _local_fp_index_path,
            )
        except Exception as _e:  # noqa: BLE001
            logger.debug(
                f"FingerprintDHT local index unavailable, cross-node "
                f"fingerprint-serve disabled: {_e}"
            )

        # PRSM-DHT-TRANSPORT T3b — construct DHTNodeComponents if the
        # operator opted in. Off-by-default; enable via NodeConfig.dht_enabled
        # or PRSM_DHT_ENABLED=1. The components run their own asyncio loop
        # in a daemon thread (DHTLoopRunner) so the Node's existing event
        # loop is unaffected.
        _dht_enabled = (
            self.config.dht_enabled
            or os.getenv("PRSM_DHT_ENABLED", "").lower() in ("1", "true", "yes")
        )
        if _dht_enabled:
            _manifest_index = None
            try:
                from prsm.network.manifest_dht.local_index import (
                    LocalManifestIndex,
                )
                _manifest_index_path = Path.home() / ".prsm" / "manifest_index"
                _manifest_index_path.mkdir(parents=True, exist_ok=True)
                _manifest_index = LocalManifestIndex(_manifest_index_path)
            except Exception as _e:  # noqa: BLE001
                logger.debug(
                    f"ManifestDHT local index unavailable, cross-node "
                    f"manifest-serve disabled: {_e}"
                )
            self.dht_components = _build_dht_components_or_none(
                identity=self.identity,
                listen_host=self.config.listen_host,
                dht_listen_port=self.config.dht_listen_port,
                manifest_index=_manifest_index,
                embedding_index=_embedding_index,
                local_fingerprint_index=_local_fingerprint_index,
            )
            if self.dht_components is not None:
                logger.info(
                    "DHT components constructed "
                    f"(manifest={_manifest_index is not None}, "
                    f"embedding={_embedding_index is not None})"
                )

        # ── Content Provider (Cross-Node Retrieval) ───────────────────────
        # Phase 1.3: ContentProvider is constructed BEFORE ContentUploader so
        # the uploader can hold a reference and populate provider._local_content
        # on every successful upload / DB hydration. Previously the provider was
        # built after the uploader, leaving register_local_content with zero
        # production callers — the serve path returned not_found for every CID
        # and the on-chain royalty payment never fired end-to-end.
        _bandwidth_limiter = None
        if self.storage_provider:
            _bandwidth_limiter = self.storage_provider.bandwidth_limiter

        self.content_provider = ContentProvider(
            identity=self.identity,
            transport=self.transport,
            gossip=self.gossip,
            content_index=self.content_index,
            bandwidth_limiter=_bandwidth_limiter,
        )

        # Phase 1.3 Task 3e: wire content_provider back into
        # storage_provider so its _on_direct_content_request can
        # defer to the canonical serve path when the provider also
        # has the CID. Without this, both MSG_DIRECT handlers race
        # and the legacy-shape response from storage_provider wins
        # (arrives first because it skips payment), which the
        # canonical ContentResponseMessage.from_payload() parser on
        # the requester side downgrades to ERROR.
        if self.storage_provider is not None:
            self.storage_provider._content_provider = self.content_provider

        # ── On-Chain FTNS Ledger (Base mainnet) ────────────────────
        # Phase 1.3 Task 3a: instantiated BEFORE ContentUploader so the
        # uploader bootstrap can derive creator_address from the ledger's
        # _connected_address. Previously this was constructed ~200 lines
        # later, which left creator_address=None at upload-time and
        # silently bypassed provenance_hash computation / on-chain
        # royalty routing for every production upload.
        # Sprint 501: persist on-chain TX audit trail to SQLite so
        # operators don't lose history on daemon restart. Defaults to
        # ~/.prsm/onchain_tx.db; PRSM_ONCHAIN_TX_DB=":memory:" disables.
        _onchain_db = os.environ.get("PRSM_ONCHAIN_TX_DB")
        if _onchain_db == ":memory:" or _onchain_db == "":
            _onchain_db = None
        elif _onchain_db is None:
            _onchain_db = str(
                Path.home() / ".prsm" / "onchain_tx.db"
            )
            try:
                Path(_onchain_db).parent.mkdir(
                    parents=True, exist_ok=True,
                )
            except OSError:
                _onchain_db = None
        self.ftns_ledger = OnChainFTNSLedger(
            node_id=self.identity.node_id,
            db_path=_onchain_db,
        )

        # T6 (2026-05-05): on-chain ProvenanceRegistry client. Lazy-init at
        # node construction so ContentUploader can register content on-chain
        # at upload-time. Same env-var contract as content_economy.py:
        # PRSM_ONCHAIN_PROVENANCE=1 + PRSM_PROVENANCE_REGISTRY_ADDRESS +
        # FTNS_WALLET_PRIVATE_KEY. Returns None gracefully when any required
        # piece is missing — the upload still succeeds locally.
        provenance_client = _build_provenance_client_or_none()
        # Expose on self so /health/detailed can surface the
        # provenance_registry subsystem + its canonical-match check
        # against networks.py.
        self._provenance_client = provenance_client

        # Native-storage migration PR 2c: content_publisher / content_retriever
        # are attached AFTER the BT layer is initialised below (~line 950).
        # The uploader's internal _publish_content / _fetch_content helpers
        # log + return None until that attachment runs.
        self.content_uploader = ContentUploader(
            identity=self.identity,
            gossip=self.gossip,
            ledger=self.ledger,
            transport=self.transport,
            content_index=self.content_index,
            embedding_fn=_embedding_fn,
            semantic_index_path=_semantic_index_path,
            content_provider=self.content_provider,
            creator_address=_derive_creator_address(self.ftns_ledger),
            provenance_client=provenance_client,
            # T3.6 (PRSM-PROV-1): cross-node embedding gossip wiring.
            # embedding_dht_client stays None until the Phase 6 P2P
            # transport is wired into a Kademlia routing table at the
            # node level — at which point peer-side fetch will engage
            # without further uploader changes (T3.5's escalation
            # path is gated on a real client). Local-side store is
            # active now: every signed embedding lands in
            # _embedding_index so future peer fetches succeed.
            embedding_model_id=_embedding_model_id,
            embedding_dht_client=None,
            embedding_index=_embedding_index,
            # T4.9.next3: persist FingerprintIndex to disk + share the
            # embedding-lane DHT wiring. ``embedding_dht_client`` above
            # is still None (Phase 6 P2P transport hasn't lit it yet),
            # so fingerprint escalation is also dormant until the same
            # client switch is flipped — at which point both lanes
            # engage simultaneously without further uploader changes.
            fingerprint_index_path=_fingerprint_index_path,
            # T4.9.next5: serve-side fingerprint storage. Same instance
            # is also passed to DHTNodeComponents above, so the
            # uploader's _register_local_fingerprint and the
            # EmbeddingDHTServer's fetch handler share a single store.
            local_fingerprint_index=_local_fingerprint_index,
            # PRSM-PROV-1 Item 6 (T6.3 + T6.5 + T6.5.gov.next):
            # disputed-band three-tier wiring. All three may be None
            # — when so, uploads fall back to legacy 2-band auto-
            # attribute behavior.
            threshold_resolver=_threshold_resolver,
            arbitration_queue=_arbitration_queue,
            arbitration_proposal_sink=_arbitration_proposal_sink,
        )

        # ── Ledger Sync ──────────────────────────────────────────
        self.ledger_sync = LedgerSync(
            identity=self.identity,
            gossip=self.gossip,
            ledger=self.ledger,
            transport=self.transport,
            reconciliation_interval=self.config.reconciliation_interval,
        )

        # ── Agent Registry & Collaboration ────────────────────────
        self.agent_registry = AgentRegistry(
            gossip=self.gossip,
            transport=self.transport,
            node_id=self.identity.node_id,
        )
        self.agent_collaboration = AgentCollaboration(
            gossip=self.gossip,
            node_id=self.identity.node_id,
            ledger=self.ledger,
            bid_strategy=BidStrategy(self.config.bid_strategy),
            bid_window_seconds=self.config.bid_window_seconds,
            min_bids=self.config.min_bids,
            task_timeout=self.config.task_timeout,
            review_timeout=self.config.review_timeout,
            query_timeout=self.config.query_timeout,
            max_completed_records=self.config.max_completed_records,
            cleanup_interval=self.config.collab_cleanup_interval,
        )

        # ── Staking Manager ─────────────────────────────────────────
        # Create FTNS adapter for staking operations
        staking_ftns_adapter = _StakingFTNSAdapter(self.ledger, self.identity.node_id)
        self.staking_manager = StakingManager(
            db_session=None,  # session unused; StakingManager calls get_async_session() internally
            ftns_service=staking_ftns_adapter,
            config=StakingConfig(),
        )
        logger.info("Staking manager initialized")

        # ── BitTorrent Integration ───────────────────────────────────
        # Initialize BitTorrent client
        bt_config = BitTorrentConfig(
            port_range_start=getattr(self.config, 'bt_port_start', 6881),
            port_range_end=getattr(self.config, 'bt_port_end', 6891),
            download_dir=str(Path(self.config.data_dir) / "torrents"),
            dht_enabled=getattr(self.config, 'bt_dht_enabled', True),
        )
        self.bt_client = BitTorrentClient(config=bt_config)
        bt_available = await self.bt_client.initialize()
        if bt_available:
            logger.info("BitTorrent client initialized")

            # Initialize manifest store
            self.bt_manifest_store = TorrentManifestStore(
                database_url=f"sqlite:///{self.config.data_dir}/torrent_manifests.db"
            )
            await self.bt_manifest_store.initialize()

            # Initialize provider (seeder).
            # Sprint 179 — coerce data_dir to pathlib.Path so `/`
            # join works regardless of whether config supplies str
            # or Path (NodeConfig defaults to str).
            from pathlib import Path as _Path
            _data_dir = _Path(self.config.data_dir)
            bt_provider_config = BitTorrentProviderConfig(
                max_torrents=getattr(self.config, 'bt_max_torrents', 50),
                data_dir=str(_data_dir / "torrents"),
                seeder_reward_per_gb=getattr(self.config, 'bt_seeder_reward_per_gb', Decimal("0.10")),
            )
            self.bt_provider = BitTorrentProvider(
                identity=self.identity,
                transport=self.transport,
                gossip=self.gossip,
                ledger=self.ledger,
                bt_client=self.bt_client,
                manifest_store=self.bt_manifest_store,
                config=bt_provider_config,
                node_config=self.config,
            )

            # Initialize requester (downloader)
            bt_requester_config = BitTorrentRequesterConfig(
                max_concurrent_downloads=getattr(self.config, 'bt_max_downloads', 10),
                data_dir=str(_data_dir / "torrents"),
                download_cost_per_gb=getattr(self.config, 'bt_download_cost_per_gb', Decimal("0.05")),
            )
            self.bt_requester = BitTorrentRequester(
                identity=self.identity,
                gossip=self.gossip,
                bt_client=self.bt_client,
                manifest_store=self.bt_manifest_store,
                ledger=self.ledger,
                config=bt_requester_config,
            )
            logger.info("BitTorrent provider and requester initialized")

            # Native-storage migration PR 2c (2026-05-07): wire
            # ContentPublisher / ContentRetriever onto the already-
            # constructed ContentUploader. The uploader was initialised
            # earlier with content_publisher=None; production uploads
            # logged "content_publisher is None — cannot publish" until
            # this attachment runs. With these set, prsm_upload_dataset
            # actually distributes content via the proprietary
            # BitTorrent layer instead of returning a placeholder.
            from prsm.node.content_publisher import (
                ContentPublisher,
                ContentRetriever,
            )

            staging_dir = _data_dir / "content_publish_staging"
            cache_dir = _data_dir / "content_fetch_cache"

            self.content_publisher = ContentPublisher(
                bt_provider=self.bt_provider,
                staging_dir=staging_dir,
            )
            self.content_retriever = ContentRetriever(
                bt_requester=self.bt_requester,
                cache_dir=cache_dir,
            )
            self.content_uploader.content_publisher = self.content_publisher
            self.content_uploader.content_retriever = self.content_retriever
            # Sprint 427 (F7) — wire the retriever into the
            # ContentProvider so _fetch_local can fall back to the BT
            # swarm for cids that are structurally invalid for
            # ContentHash.from_hex (e.g., 40-char BT v1 infohashes
            # produced by Tier A publishes).
            if self.content_provider is not None:
                self.content_provider.content_retriever = (
                    self.content_retriever
                )
                # sp1073 — also wire the publisher so ContentProvider._fetch_local can
                # serve self-published Tier-A content via the local_publish_path
                # shortcut (no BT swarm needed even with libtorrent present).
                self.content_provider.content_publisher = self.content_publisher
            # Sprint 428 (F8) — wire the publisher back into the
            # retriever so it can short-circuit the BT swarm for
            # locally-published content. Closes single-node Vision §4
            # step-8 self-fetch without touching the BT layer's
            # session-isolation design.
            self.content_retriever.content_publisher = (
                self.content_publisher
            )
            logger.info(
                "ContentUploader wired through ContentPublisher (Tier A) — "
                "uploads now distribute via the BitTorrent layer."
            )
        else:
            # sp1071 — libtorrent absent: instead of leaving content_publisher=None
            # (every upload then failed "content_publisher is None — cannot publish"),
            # use the pure-Python LocalContentPublisher. It computes the SAME canonical
            # v1 CID a libtorrent v1_only node would (sp1070, byte-identical) and stages
            # the bytes; ContentUploader then registers the content with ContentProvider,
            # which serves it over the P2P substrate (INLINE/CHUNKED, no libtorrent), and
            # cross-node fetch uses ContentProvider.request_content (also libtorrent-free).
            # So a default operator's Tier-A (public) uploads now actually publish +
            # become retrievable. Tier B/C encrypted still needs the BitTorrent/
            # ContentStore path (LocalContentPublisher raises a clear error for it).
            try:
                from prsm.node.local_content_publisher import LocalContentPublisher
                staging_dir = _data_dir / "content_publish_staging"
                self.content_publisher = LocalContentPublisher(
                    staging_dir=staging_dir,
                    node_id=self.identity.node_id if self.identity else "",
                )
                self.content_retriever = None   # BT retriever needs libtorrent; fetch
                #   goes through ContentProvider.request_content (libtorrent-free)
                self.content_uploader.content_publisher = self.content_publisher
                # sp1073 — wire the publisher into ContentProvider so its _fetch_local
                # local_publish_path shortcut can serve this node's own Tier-A uploads
                # (self-fetch + serving to peers), all without libtorrent.
                if self.content_provider is not None:
                    self.content_provider.content_publisher = self.content_publisher
                logger.info(
                    "libtorrent not available — using LocalContentPublisher (Tier A, "
                    "pure-Python v1 infohash). Uploads publish + serve over the P2P "
                    "substrate without libtorrent; install libtorrent for the BT swarm "
                    "+ Tier B/C encrypted content."
                )
            except Exception as _lcp_exc:  # noqa: BLE001 — never block startup
                logger.warning(
                    "LocalContentPublisher wiring failed (%s) — uploads disabled until "
                    "libtorrent is installed", _lcp_exc)
                self.content_publisher = None
                self.content_retriever = None

        # ── Inference Executor (Sprint 438) ───────────────────────
        # /compute/inference returns 503 "Inference executor not
        # initialized" by default because production-grade inference
        # requires operator-supplied executor (model files + TEE
        # attestation backend). For verification-campaign + dogfood
        # paths, operators opt in to the deterministic mock via
        # PRSM_INFERENCE_EXECUTOR=mock. The mock zero-fills the
        # cryptographic fields and MUST NOT be trusted by real
        # verifiers — that's the explicit honest-scope.
        # Sprint 1184 (day-one-live blocker #2) — resolve the effective executor kind.
        # UNSET now DEFAULTS to 'local' real single-node inference WHEN the .[ml] extra
        # (torch+transformers) is installed, so a properly-installed node serves real
        # inference with NO env var (was: unset → None → 503 on every request). Without
        # .[ml] it stays off and the 503 diagnostic points at the install. An explicit
        # none/off/disabled opts out.
        from prsm.node.inference_wiring import (
            ml_inference_available,
            resolve_inference_executor_kind,
        )
        _exec_raw = os.environ.get("PRSM_INFERENCE_EXECUTOR", "")
        _ml_available = ml_inference_available()
        _exec_kind = resolve_inference_executor_kind(
            _exec_raw, ml_available=_ml_available,
        )
        if _exec_kind == "mock":
            from prsm.compute.inference import (
                MockInferenceExecutor,
            )
            self.inference_executor = MockInferenceExecutor()
            logger.info(
                "Inference executor: MockInferenceExecutor "
                "(opt-in via PRSM_INFERENCE_EXECUTOR=mock). "
                "Synthetic outputs only — zero-filled crypto "
                "fields. Real production deployments must wire "
                "their own InferenceExecutor."
            )
        elif _exec_kind == "mock-streaming":
            # Sprint 1016 — single-node mock STREAMING executor. Unlike the
            # plain `mock` (non-streaming MockInferenceExecutor), this builds a
            # real ParallaxScheduledExecutor from mock components so
            # /compute/inference/stream works end-to-end on one node (no real
            # GPUs / model files / swarm). Synthetic tokens flow through the
            # genuine gates + allocation + receipt-signing path. Honest scope:
            # zero/software-tier crypto — MUST NOT be trusted by real verifiers.
            # Streaming requests must use model_id="mock-model".
            from prsm.compute.inference.mock_streaming import (
                build_mock_streaming_executor,
            )
            self.inference_executor = build_mock_streaming_executor(self.identity)
            logger.info(
                "Inference executor: mock-streaming "
                "ParallaxScheduledExecutor (opt-in via "
                "PRSM_INFERENCE_EXECUTOR=mock-streaming). Single-node "
                "synthetic streaming — zero-filled crypto. Use "
                "model_id='mock-model'. Real deployments must wire their "
                "own InferenceExecutor."
            )
        elif _exec_kind == "local":
            # Sprint 1019 (Tier 2) — REAL single-node inference. Runs a real
            # HuggingFace causal-LM (gpt2/distilgpt2) in-process via transformers
            # model.generate, wired into a real ParallaxScheduledExecutor — no
            # RPC, no swarm, no ~8-env-var PRSM_PARALLAX_* ceremony, no Base
            # anchor. One opt-in env var. Optional overrides:
            #   PRSM_LOCAL_INFERENCE_MODEL   (default distilgpt2)
            #   PRSM_LOCAL_INFERENCE_MAX_TOKENS (default 32)
            # Honest scope: the MODEL output is real, but trust is software-tier
            # (TEEType.SOFTWARE, vendor_verified=false) — real hardware TEE is
            # a separate (Tier 3) item. Not for confidential Tier B/C content.
            from prsm.compute.inference.local_inference import (
                DEFAULT_LOCAL_MODEL,
                build_local_inference_executor,
                start_local_prewarm,
            )
            _local_model = (
                os.environ.get("PRSM_LOCAL_INFERENCE_MODEL", "").strip()
                or DEFAULT_LOCAL_MODEL
            )
            _local_max_raw = os.environ.get(
                "PRSM_LOCAL_INFERENCE_MAX_TOKENS", "",
            ).strip()
            try:
                _local_max = int(_local_max_raw) if _local_max_raw else 32
            except ValueError:
                _local_max = 32
            self.inference_executor = build_local_inference_executor(
                self.identity, model_id=_local_model, max_tokens=_local_max,
            )
            logger.info(
                "Inference executor: LOCAL real inference "
                "(PRSM_INFERENCE_EXECUTOR=local, model=%s). Runs a real "
                "HuggingFace model in-process, single-node — no GPU swarm / "
                "no PRSM_PARALLAX_* ceremony. Trust is SOFTWARE-tier "
                "(vendor_verified=false): real model output, but NOT "
                "hardware-confidential — do not use for Tier B/C secrets.",
                _local_model,
            )
            # sp1302 — pre-warm the model OFF the request path so the FIRST
            # /compute/inference returns real tokens fast instead of paying the
            # cold-load cost (~350MB download + load on a fresh install) inline,
            # which looks like a hang/timeout. Background daemon thread, fail-soft;
            # PRSM_LOCAL_INFERENCE_PREWARM=0 disables (lazy first-request load).
            try:
                if start_local_prewarm(self.inference_executor) is not None:
                    logger.info(
                        "local inference: pre-warming model %s in the background "
                        "so the first request skips the cold-load cost "
                        "(PRSM_LOCAL_INFERENCE_PREWARM=0 to disable).",
                        _local_model)
            except Exception as _pw_exc:  # noqa: BLE001 — never block startup
                logger.debug("local inference pre-warm dispatch skipped: %s", _pw_exc)
        elif _exec_kind == "parallax":
            # Sprint 558 — opt-in production wiring path for the
            # real ParallaxScheduledExecutor. The builder reads
            # PRSM_PARALLAX_* env vars for operator-supplied
            # components (model catalog, trust stack, gpu pool).
            # Each missing/invalid piece logs a structured warning
            # naming the env var; result is None and the daemon
            # surfaces the same 503 it does today (sprint 438).
            # Sprints 559/560/561+ progressively replace mock-kind
            # components with real production kinds.
            from prsm.node.inference_wiring import (
                build_parallax_executor_or_none,
            )
            built = build_parallax_executor_or_none(self)
            self.inference_executor = built
            # sp1377 — auto-stage the configured HF model into the local model registry so the
            # layer-stage server can build (else the worker stage-executor is a stub → chain fails).
            # Uses the num_layers from the catalog (auto-derived by sp1376). Fail-soft; only when the
            # operator opted into a registry root + an HF model. Removes the manual stage_hf_model.py.
            if built is not None:
                _reg_root = (os.environ.get("PRSM_MODEL_REGISTRY_ROOT", "") or "").strip()
                _hf_mid = (os.environ.get("PRSM_PARALLAX_HF_MODEL_ID", "") or "").strip()
                if _reg_root and _hf_mid and getattr(self, "identity", None) is not None:
                    try:
                        _mi = (getattr(built, "_catalog", {}) or {}).get(_hf_mid)
                        _nl = getattr(_mi, "num_layers", None)
                        if _nl:
                            from prsm.node.chain_executor_adapters import (
                                ensure_hf_model_registered,
                            )
                            ensure_hf_model_registered(
                                registry_root=_reg_root, model_id=_hf_mid,
                                num_layers=_nl, identity=self.identity)
                    except Exception as _stg_exc:  # noqa: BLE001 — never block startup
                        logger.debug("sp1377 auto-stage skipped: %s", _stg_exc)
            if built is not None:
                logger.info(
                    "Inference executor: ParallaxScheduledExecutor "
                    "wired via sprint-558 opt-in. Verify each "
                    "PRSM_PARALLAX_*_KIND env var matches your "
                    "deployment posture (mock kinds MUST NOT be "
                    "trusted by real verifiers)."
                )
            else:
                logger.info(
                    "Inference executor: PRSM_INFERENCE_EXECUTOR="
                    "parallax requested but build_parallax_executor"
                    "_or_none returned None (see preceding warnings "
                    "for the missing PRSM_PARALLAX_* component). "
                    "/compute/inference will surface 503 until the "
                    "wiring is complete."
                )
            # Sprint 1034 — auto-stage the configured HF model into the local
            # model registry so the layer_stage chain server can serve it
            # WITHOUT the operator running scripts/stage_hf_model.py by hand on
            # every node (the last manual step the Tier-1 bench exposed: an
            # unstaged model → MODEL_NOT_FOUND at inference). Idempotent +
            # publisher-aware (won't clobber another publisher). FAIL-OPEN: this
            # branch is not otherwise try/except-wrapped, so a staging failure
            # must log + continue, never crash daemon initialize().
            try:
                from prsm.node.inference_wiring import ensure_hf_model_staged
                _stage_status = ensure_hf_model_staged(self.identity)
                logger.info(
                    "Sprint 1034 auto-stage HF model: %s", _stage_status,
                )
            except Exception as _stage_exc:  # noqa: BLE001
                logger.warning(
                    "Sprint 1034 auto-stage failed (non-fatal; run "
                    "scripts/stage_hf_model.py manually if needed): %s",
                    _stage_exc,
                )
        else:
            self.inference_executor = None
            # sp1184 — actionable startup signal so a non-operator isn't left to discover
            # the 503 reason from a request. Distinguish "off by choice" from "missing deps".
            _raw_norm = (_exec_raw or "").strip().lower()
            if _raw_norm in ("none", "off", "disabled", "false", "0"):
                logger.info(
                    "Inference executor: DISABLED (PRSM_INFERENCE_EXECUTOR=%s). "
                    "/compute/inference will return 503 by request.", _raw_norm,
                )
            elif not _raw_norm:
                logger.warning(
                    "Inference executor: NONE — PRSM_INFERENCE_EXECUTOR is unset and the "
                    ".[ml] extra (torch+transformers) is NOT installed, so real "
                    "single-node inference could not be enabled by default. Run "
                    "`pip install -e '.[ml]'` to serve real inference automatically, set "
                    "PRSM_INFERENCE_EXECUTOR=parallax for a GPU pool, or =none to silence "
                    "this. /compute/inference returns 503 until then.",
                )
            else:
                logger.warning(
                    "Inference executor: NONE — PRSM_INFERENCE_EXECUTOR=%r is not a "
                    "recognized kind (local|parallax|mock|mock-streaming|none). "
                    "/compute/inference returns 503.", _exec_raw,
                )

        # Sprint 1046 — activate REAL Intel SGX DCAP attestation verification on
        # the default registry (the one backing /compute/receipt/verify) when an
        # Intel SGX Root CA is configured (PRSM_INTEL_SGX_ROOT_CA_PEM / _FILE).
        # Without the anchor: unchanged structural behavior (vendor_verified=False).
        # FAIL-OPEN: a misconfig must never crash daemon initialize().
        try:
            # Sprint 1081 — default the attestation-collateral cache dir (where the
            # CollateralRefresher writes auto-refreshed CRLs/QE-Identity, read by the
            # DCAP backend) to <data_dir>/attestation_collateral unless the operator
            # set it. Set it BEFORE configuring the registry so the first build reads
            # any already-cached (persisted-across-restart) collateral.
            import os as _os_collat
            if not (_os_collat.environ.get("PRSM_ATTESTATION_COLLATERAL_DIR", "") or "").strip():
                _collat_dir = _Path(self.config.data_dir) / "attestation_collateral"
                try:
                    _collat_dir.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
                _os_collat.environ["PRSM_ATTESTATION_COLLATERAL_DIR"] = str(_collat_dir)
        except Exception as _collat_exc:  # noqa: BLE001
            logger.debug("collateral cache dir default skipped: %s", _collat_exc)

        try:
            from prsm.compute.inference.attestation_backends import (
                configure_default_registry_from_env,
            )
            if configure_default_registry_from_env():
                logger.info("Sprint 1046: Intel SGX DCAP attestation verification "
                            "ACTIVE (root CA configured) — vendor_verified is now "
                            "cryptographically enforced for SGX quotes")
        except Exception as _dcap_exc:  # noqa: BLE001
            logger.warning("Sprint 1046 DCAP activation failed (non-fatal; "
                           "structural fallback): %s", _dcap_exc)

        # ── Payment Escrow & Result Consensus ─────────────────────
        from prsm.node.payment_escrow import PaymentEscrow
        from prsm.node.result_consensus import ResultConsensus
        self._payment_escrow = PaymentEscrow(
            ledger=self.ledger,
            node_id=self.identity.node_id,
        )
        # B8 async-dispatch follow-on: JobHistoryStore for
        # /compute/forge → /compute/status traceability. In-memory
        # LRU-bounded with optional filesystem persistence (v2
        # 2026-05-09): when PRSM_JOB_HISTORY_DIR is set, records
        # survive node restart so /compute/status can serve
        # late-arriving status queries from prior runs. Best-effort
        # wiring — construction failure degrades to escrow-only
        # /compute/status (legacy behavior).
        try:
            import os as _os
            from pathlib import Path as _Path
            from prsm.node.job_history import JobHistoryStore
            persist_raw = _os.getenv("PRSM_JOB_HISTORY_DIR", "").strip()
            persist_dir = _Path(persist_raw) if persist_raw else None
            self._job_history = JobHistoryStore(persist_dir=persist_dir)
            logger.info(
                "JobHistoryStore wired (in-memory LRU 1024%s)",
                f", persisted to {persist_dir}" if persist_dir else "",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "JobHistoryStore construction failed: %s — "
                "/compute/status will fall back to escrow-only.",
                exc,
            )
            self._job_history = None

        # Sprint 272 — TakedownNoticeRing for foundation-side
        # intake of DMCA / legal / content moderation notices.
        # Per R9-SCOPING-1 §8 invariant: this RING does not
        # enforce; it logs received notices for distribution.
        # Each operator runs their own compliance analysis and
        # voluntarily updates their ContentFilterStore (sprint
        # 269) if they decide to act on a given notice.
        # Opt-in disk persistence via PRSM_TAKEDOWN_NOTICE_LOG_DIR.
        try:
            import os as _os
            from pathlib import Path as _Path
            from prsm.node.takedown_notice_log import (
                TakedownNoticeRing,
            )
            persist_raw = _os.getenv(
                "PRSM_TAKEDOWN_NOTICE_LOG_DIR", "",
            ).strip()
            persist_dir = _Path(persist_raw) if persist_raw else None
            self._takedown_notice_ring = TakedownNoticeRing(
                persist_dir=persist_dir,
            )
            logger.info(
                "TakedownNoticeRing wired%s",
                f", persisted to {persist_dir}" if persist_dir else "",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TakedownNoticeRing construction failed: %s — "
                "/admin/takedown-notices will 503.",
                exc,
            )
            self._takedown_notice_ring = None

        # Sprint 269 — ContentFilterStore for the operator's
        # self-managed content blocklist (Vision §14 "content
        # moderation" mitigation; R9-SCOPING-1 §7-8 operator-side
        # filter, not Foundation-curated). Opt-in filesystem
        # persistence via PRSM_CONTENT_FILTER_DIR. Failure-soft.
        try:
            from prsm.node.content_filter_store import (
                ContentFilterStore,
            )
            self._content_filter_store = ContentFilterStore.from_env()
            logger.info("ContentFilterStore wired")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ContentFilterStore construction failed: %s — "
                "/content/retrieve will not enforce blocklist.",
                exc,
            )
            self._content_filter_store = None

        # Sprint 276 — Coinbase Wallet-as-a-Service adapter.
        # Per Vision §14 "Crypto-UX adoption barrier"
        # mitigation: embedded MPC wallets with email-only
        # onboarding. Failure-soft (None when scaffold not
        # importable). The client always constructs; the
        # commission gate is internal — it returns
        # PENDING_COMMISSION records when CDP env keys are
        # absent rather than failing.
        try:
            from prsm.economy.web3.coinbase_waas_client import (
                CoinbaseWaaSClient,
            )
            self._coinbase_waas_client = (
                CoinbaseWaaSClient.from_env()
            )
            commissioned = (
                self._coinbase_waas_client
                and self._coinbase_waas_client.is_commissioned()
            )
            logger.info(
                "CoinbaseWaaSClient wired (commissioned=%s)",
                commissioned,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CoinbaseWaaSClient construction failed: %s — "
                "/wallet/waas/* endpoints will return 503.",
                exc,
            )
            self._coinbase_waas_client = None

        # Sprint 277 — Coinbase paymaster adapter for gasless
        # FTNS transfers. Same PENDING_COMMISSION pattern as
        # WaaS: client always constructs; commission gate is
        # internal via env-key presence. /wallet/transfer/gasless
        # returns PENDING_COMMISSION preview records until
        # paymaster env keys land.
        try:
            from prsm.economy.web3.paymaster_client import (
                PaymasterClient,
            )
            self._paymaster_client = PaymasterClient.from_env()
            paymaster_commissioned = (
                self._paymaster_client.is_commissioned()
            )
            logger.info(
                "PaymasterClient wired (commissioned=%s)",
                paymaster_commissioned,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PaymasterClient construction failed: %s — "
                "/wallet/transfer/gasless will return 503.",
                exc,
            )
            self._paymaster_client = None

        # Sprint 279 — Aerodrome read-only pool quoter. Real
        # production code (no commission gate). Pre-seeding,
        # is_configured() returns False and the endpoints
        # surface NOT_CONFIGURED. Post-seeding (env vars
        # pasted), endpoints return live pool state.
        try:
            from prsm.economy.web3.aerodrome_client import (
                AerodromeClient,
            )
            self._aerodrome_client = AerodromeClient.from_env()
            aerodrome_configured = (
                self._aerodrome_client.is_configured()
            )
            logger.info(
                "AerodromeClient wired (configured=%s)",
                aerodrome_configured,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AerodromeClient construction failed: %s — "
                "/wallet/pool/* endpoints will return 503.",
                exc,
            )
            self._aerodrome_client = None

        # Sprint 280 — KYC vendor adapter. Pluggable backend
        # (Persona / Onfido / Plaid). PENDING_COMMISSION
        # records when KYC_VENDOR_API_KEY absent.
        try:
            from prsm.economy.web3.kyc_client import KYCClient
            self._kyc_client = KYCClient.from_env()
            kyc_commissioned = self._kyc_client.is_commissioned()
            logger.info(
                "KYCClient wired (commissioned=%s, vendor=%s)",
                kyc_commissioned,
                getattr(self._kyc_client, "_vendor", None),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "KYCClient construction failed: %s — "
                "/wallet/kyc/* endpoints will return 503.",
                exc,
            )
            self._kyc_client = None

        # Sprint 282 — Fiat compliance audit ring. Records
        # quotes + executes across all Phase 5 fiat surfaces
        # (onramp/offramp/gasless/KYC) for regulatory
        # reporting. Persistence via
        # PRSM_FIAT_COMPLIANCE_LOG_DIR; jurisdiction tag via
        # PRSM_OPERATOR_JURISDICTION.
        try:
            from prsm.economy.web3.fiat_compliance_ring import (
                FiatComplianceRing,
            )
            self._fiat_compliance_ring = (
                FiatComplianceRing.from_env()
            )
            logger.info(
                "FiatComplianceRing wired "
                "(persist_dir=%s, jurisdiction=%s)",
                getattr(
                    self._fiat_compliance_ring,
                    "_persist_dir", None,
                ),
                getattr(
                    self._fiat_compliance_ring,
                    "_default_jurisdiction", None,
                ),
            )
            # Sp1175 — the AML tier-limit rolling total is PROCESS-LOCAL
            # (this ring is an in-memory deque). On a MULTI-REPLICA
            # deployment each replica enforces the limit independently,
            # so a user fanned across N replicas can transact up to N×
            # their tier limit. Single-replica deployments are unaffected.
            # Operators running fiat on >1 replica without a shared
            # real-time backend should set PRSM_FIAT_TIER_LIMIT_MODE=
            # strict_shared to FAIL CLOSED (deny the gated fiat surface)
            # until the shared-rolling-total follow-on ships.
            _tier_mode = (
                os.environ.get(
                    "PRSM_FIAT_TIER_LIMIT_MODE", "process_local",
                )
                .strip()
                .lower()
            )
            if _tier_mode == "strict_shared":
                logger.warning(
                    "Fiat tier-limit mode=strict_shared — the gated fiat "
                    "onramp surface FAILS CLOSED (503) because the rolling "
                    "AML total is process-local and cannot be enforced "
                    "across replicas. Unset PRSM_FIAT_TIER_LIMIT_MODE for "
                    "single-replica deployments."
                )
            else:
                logger.warning(
                    "Fiat AML tier-limit rolling total is PROCESS-LOCAL "
                    "(per-replica). On a multi-replica deployment a user "
                    "can transact up to N× their limit. Single-replica is "
                    "safe; multi-replica operators should set "
                    "PRSM_FIAT_TIER_LIMIT_MODE=strict_shared until the "
                    "shared-rolling-total backend ships."
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FiatComplianceRing construction failed: %s — "
                "/admin/fiat-compliance/* will return 503.",
                exc,
            )
            self._fiat_compliance_ring = None

        # Sp1180 — optional SHARED (cross-replica) idempotency claim
        # store for /compute/forge. When PRSM_FORGE_IDEMPOTENCY_REDIS_URL
        # (or PRSM_REDIS_URL / REDIS_URL) is set, the forge endpoint
        # atomically claims each Idempotency-Key via SETNX across
        # replicas so a same-key request landing on a DIFFERENT replica
        # can't lock a second escrow (the sp1177 fix only covered the
        # same-replica window). Default None → in-process guarantee only.
        self._forge_idempotency_redis = None
        _forge_idem_url = (
            os.environ.get("PRSM_FORGE_IDEMPOTENCY_REDIS_URL")
            or os.environ.get("PRSM_REDIS_URL")
            or os.environ.get("REDIS_URL")
        )
        if _forge_idem_url:
            try:
                import redis as _redis_sync

                self._forge_idempotency_redis = _redis_sync.Redis.from_url(
                    _forge_idem_url, decode_responses=True,
                )
                logger.info(
                    "Sp1180 — cross-replica forge idempotency claim "
                    "store wired (SETNX); same-key requests are deduped "
                    "across replicas."
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Sp1180 — forge idempotency Redis construction "
                    "failed (%s); /compute/forge falls back to the "
                    "sp1177 in-process idempotency guarantee.", exc,
                )
                self._forge_idempotency_redis = None

        # Sprint 284 — webhook replay defense ring. Bounded set of
        # recently-seen signatures; second occurrence of the same
        # signature → 409.
        #
        # Sp893 — the ring is now PERSISTENT across restarts. The
        # sp284 "restart = fresh replay window" rationale only held
        # for Persona (which carries a `t=` timestamp → the 300s
        # freshness window is a second layer). ONFIDO carries no
        # timestamp, so the ring is its ONLY replay defense; an
        # in-memory-only ring let a captured Onfido webhook be
        # replayed across any restart. Persist dir defaults to
        # ~/.prsm/kyc-webhook-replay/ (env override
        # PRSM_KYC_WEBHOOK_REPLAY_DIR; ":memory:" opts back into the
        # old in-memory-only behavior). Disk bounded by FIFO cap +
        # PRSM_KYC_WEBHOOK_REPLAY_RETENTION_SEC time window.
        try:
            from prsm.economy.web3.webhook_replay_defense import (
                WebhookReplayRing,
            )
            _replay_dir_raw = os.environ.get(
                "PRSM_KYC_WEBHOOK_REPLAY_DIR",
            )
            if _replay_dir_raw == ":memory:":
                _replay_persist_dir = None
            elif _replay_dir_raw:
                _replay_persist_dir = Path(_replay_dir_raw)
            else:
                _replay_persist_dir = (
                    Path.home() / ".prsm" / "kyc-webhook-replay"
                )
            try:
                _replay_retention = int(os.environ.get(
                    "PRSM_KYC_WEBHOOK_REPLAY_RETENTION_SEC", "86400",
                ))
            except (ValueError, TypeError):
                _replay_retention = 86400
            self._kyc_webhook_replay_ring = WebhookReplayRing(
                persist_dir=_replay_persist_dir,
                retention_sec=_replay_retention,
            )
            logger.info(
                "KYC webhook replay ring wired (persist=%s, "
                "retention=%ds)",
                _replay_persist_dir or ":memory:", _replay_retention,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "WebhookReplayRing construction failed: %s — "
                "replay defense disabled.",
                exc,
            )
            self._kyc_webhook_replay_ring = None

        # Sprint 299 — insurance fund tracker (Vision §14
        # mitigation item 2). Public read surface for the
        # "5% treasury reserve for exploit recovery" promise.
        # Recovery transfer composer-only — Foundation Safe
        # 2-of-3 multisig gates execution.
        try:
            from prsm.economy.web3.insurance_fund_tracker import (
                InsuranceFundTracker,
            )
            self._insurance_fund_tracker = (
                InsuranceFundTracker.from_env()
            )
            logger.info(
                "InsuranceFundTracker wired "
                "(fund_addr=%s, target_bps=%d)",
                self._insurance_fund_tracker.fund_address,
                self._insurance_fund_tracker.target_bps,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "InsuranceFundTracker construction failed: "
                "%s — /admin/insurance-fund/* will return "
                "503.",
                exc,
            )
            self._insurance_fund_tracker = None

        # Sprint 298 — emergency pause composer (Vision §14
        # smart-contract exploit response). Reads pausable-
        # contract addresses from prsm.config.networks per
        # PRSM_NETWORK env; backend wires to BASE_RPC_URL.
        # Composer-only: never executes pause; produces tx
        # payload for Foundation Safe multi-sig upload.
        try:
            from prsm.economy.web3.emergency_pause_client import (
                EmergencyPauseClient,
            )
            self._emergency_pause_client = (
                EmergencyPauseClient.from_env()
            )
            logger.info("EmergencyPauseClient wired")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EmergencyPauseClient construction failed: "
                "%s — /admin/emergency-pause/* will return "
                "503.",
                exc,
            )
            self._emergency_pause_client = None

        # Sprint 300 — responsible-disclosure intake (Vision
        # §14 mitigation item 3). Filesystem-persisted via
        # PRSM_DISCLOSURE_INTAKE_DIR; bounty payouts are
        # composer-only (Foundation Safe multisig gates
        # execution). Also pins FTNS token address + chain_id
        # for the payout composer.
        try:
            from prsm.economy.web3.disclosure_intake import (
                DisclosureIntake,
            )
            from prsm.config.networks import resolve_endpoints
            self._disclosure_intake = DisclosureIntake.from_env()
            endpoints = resolve_endpoints()
            self._disclosure_ftns_token_address = (
                endpoints.ftns_token
            )
            self._disclosure_chain_id = endpoints.chain_id
            logger.info(
                "DisclosureIntake wired "
                "(records=%d, ftns_token=%s, chain_id=%s)",
                self._disclosure_intake.count(),
                self._disclosure_ftns_token_address,
                self._disclosure_chain_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DisclosureIntake construction failed: %s — "
                "/admin/disclosure/* will return 503.", exc,
            )
            self._disclosure_intake = None
            self._disclosure_ftns_token_address = None
            self._disclosure_chain_id = None

        # Sprint 301 — incident response playbook (Vision
        # §14 item 5). Filesystem-persisted via
        # PRSM_INCIDENT_RESPONSE_DIR. Pre-committed decision-
        # tree + comms templates live in the module itself
        # (public). The /admin/incident/playbook endpoint
        # surfaces the playbook unauthenticated for §14
        # transparency promise.
        try:
            from prsm.economy.web3.incident_response import (
                IncidentResponse,
            )
            self._incident_response = (
                IncidentResponse.from_env()
            )
            logger.info(
                "IncidentResponse wired (records=%d)",
                self._incident_response.count(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "IncidentResponse construction failed: "
                "%s — /admin/incident/* will return 503.",
                exc,
            )
            self._incident_response = None

        # Sprint 302 — formal-invariant checker (Vision §14
        # item 4). Pinned invariants live in
        # formal_invariants.INVARIANT_REGISTRY (public per
        # §14 transparency promise). Checker is wired only
        # when an RPC-capable backend is available; without
        # one, /admin/formal-verification/check returns 503
        # but /invariants stays public.
        try:
            from prsm.economy.web3.formal_invariants import (
                InvariantChecker,
            )
            from prsm.config.networks import resolve_endpoints
            endpoints = resolve_endpoints()
            backend = _build_formal_invariant_backend_or_none(
                endpoints,
            )
            if backend is not None:
                self._formal_invariant_checker = (
                    InvariantChecker(backend=backend)
                )
            else:
                self._formal_invariant_checker = None
            self._formal_invariant_addresses = {
                "royalty_distributor": (
                    endpoints.royalty_distributor
                ),
                "ftns_token": endpoints.ftns_token,
                "escrow_pool": endpoints.escrow_pool,
                "emission_controller": (
                    endpoints.emission_controller
                ),
                "compensation_distributor": (
                    endpoints.compensation_distributor
                ),
                "storage_slashing": (
                    endpoints.storage_slashing
                ),
                "stake_bond": endpoints.stake_bond,
                # sp984 — §14 CreatorStakeRegistry (PENDING_COMMISSION; None
                # until the address is recorded in networks.py post-ceremony,
                # at which point the runtime probe covers it automatically).
                "creator_stake_registry": (
                    endpoints.creator_stake_registry
                ),
                # sp985 — BatchSettlementRegistry (settlement + consensus-slash).
                # LIVE on mainnet → these runtime pins execute against real state.
                "settlement_registry": endpoints.settlement_registry,
                # sp987 — ProvenanceRegistry V2 (active) + V1 (legacy); both
                # deployed + read by the RoyaltyDistributor. Royalty-rate ceiling.
                "provenance_registry_v2": (
                    endpoints.provenance_registry_v2
                ),
                "provenance_registry": endpoints.provenance_registry,
                # sp988 — fleet governance sweep: the last deployed contracts
                # with governance state (immutable admin / Ownable2Step owner).
                "publisher_key_anchor": (
                    endpoints.publisher_key_anchor
                ),
                "key_distribution": endpoints.key_distribution,
            }
            logger.info(
                "Formal-invariant checker wired "
                "(backend=%s, 13 contracts: rd=%s ftns=%s "
                "ep=%s ec=%s cd=%s ss=%s sb=%s csr=%s bsr=%s "
                "prv2=%s prv=%s pka=%s kd=%s)",
                bool(backend),
                self._formal_invariant_addresses.get(
                    "royalty_distributor",
                ),
                self._formal_invariant_addresses.get(
                    "ftns_token",
                ),
                self._formal_invariant_addresses.get(
                    "escrow_pool",
                ),
                self._formal_invariant_addresses.get(
                    "emission_controller",
                ),
                self._formal_invariant_addresses.get(
                    "compensation_distributor",
                ),
                self._formal_invariant_addresses.get(
                    "storage_slashing",
                ),
                self._formal_invariant_addresses.get(
                    "stake_bond",
                ),
                self._formal_invariant_addresses.get(
                    "creator_stake_registry",
                ),
                self._formal_invariant_addresses.get(
                    "settlement_registry",
                ),
                self._formal_invariant_addresses.get(
                    "provenance_registry_v2",
                ),
                self._formal_invariant_addresses.get(
                    "provenance_registry",
                ),
                self._formal_invariant_addresses.get(
                    "publisher_key_anchor",
                ),
                self._formal_invariant_addresses.get(
                    "key_distribution",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Formal-invariant checker construction "
                "failed: %s — /admin/formal-verification/check "
                "will return 503 (invariant LIST stays "
                "public).", exc,
            )
            self._formal_invariant_checker = None
            self._formal_invariant_addresses = {}

        # Sprint 364 — halmos symbolic-execution runner.
        # Wired unconditionally; fail-soft when halmos/forge
        # isn't installed (runner.is_available() returns
        # False, endpoint returns 503 with named missing
        # tools). Same posture as the runtime probe above —
        # symbolic surface stays operational when tools are
        # absent, just returns 503 explaining what's missing.
        try:
            from prsm.economy.web3.halmos_runner import (
                HalmosRunner,
            )
            self._halmos_runner = HalmosRunner()
            logger.info(
                "Halmos symbolic-execution runner wired "
                "(available=%s; missing=%s)",
                self._halmos_runner.is_available(),
                self._halmos_runner.missing_tools(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Halmos runner construction failed: %s "
                "— /admin/formal-verification/symbolic/check "
                "will return 503.", exc,
            )
            self._halmos_runner = None

        # Sprint 306 — $CORP authorization capability store
        # (Vision §7 Enterprise Confidentiality Mode layer 2).
        # Filesystem-persisted via PRSM_CORP_CAPABILITY_DIR.
        # NOT the security gate (encryption is, sprint 304;
        # TEE policy is, sprint 305) — this is the ergonomics
        # + accounting + audit layer.
        try:
            from prsm.enterprise.corp_capability import (
                CorpCapabilityStore,
            )
            self._corp_capability_store = (
                CorpCapabilityStore.from_env()
            )
            logger.info(
                "$CORP capability store wired "
                "(issuers=%d)",
                len(self._corp_capability_store.list_issuers()),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CorpCapabilityStore construction failed: "
                "%s — /admin/corp/* will return 503.", exc,
            )
            self._corp_capability_store = None

        # Sprint 308 — federated-learning orchestrator
        # (Vision §7 Enterprise Confidentiality Mode
        # capstone). Coordinates round-by-round training
        # across TEE-attested workers that see only
        # gradients, never plaintext. Filesystem-persisted
        # via PRSM_FEDERATED_LEARNING_DIR.
        try:
            from prsm.enterprise.federated_learning import (
                FederatedLearningOrchestrator,
            )
            self._federated_learning_orchestrator = (
                FederatedLearningOrchestrator.from_env()
            )
            logger.info(
                "FederatedLearningOrchestrator wired "
                "(jobs=%d)",
                len(
                    self._federated_learning_orchestrator.list_jobs(),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FederatedLearningOrchestrator construction "
                "failed: %s — /admin/federated/* will return "
                "503.", exc,
            )
            self._federated_learning_orchestrator = None

        # Sprint 308b — worker-side federated training key.
        # PRSM_FEDERATED_WORKER_PRIVKEY (Ed25519, b64, 32B)
        # gates /compute/train. Reading the env directly
        # keeps the value out of any persisted state — only
        # in-memory.
        try:
            import os as _os
            raw = (
                _os.environ.get(
                    "PRSM_FEDERATED_WORKER_PRIVKEY", "",
                ).strip()
            )
            self._federated_worker_privkey_b64 = (
                raw or None
            )
            if self._federated_worker_privkey_b64:
                logger.info(
                    "Federated worker privkey wired "
                    "(/compute/train enabled)",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Federated worker privkey wiring failed: "
                "%s — /compute/train returns 503.", exc,
            )
            self._federated_worker_privkey_b64 = None

        # Sprint 308c — orchestrator transport privkey for
        # unsealing worker gradients. X25519, b64, 32B.
        # Read from PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY
        # env; in-memory only, never persisted.
        try:
            import os as _os
            raw = (
                _os.environ.get(
                    "PRSM_FEDERATED_ORCHESTRATOR_"
                    "TRANSPORT_PRIVKEY", "",
                ).strip()
            )
            self._federated_orchestrator_transport_privkey_b64 = (
                raw or None
            )
            if raw:
                logger.info(
                    "Federated orchestrator transport "
                    "privkey wired (encrypted-gradient "
                    "transport enabled)",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Federated transport privkey wiring "
                "failed: %s", exc,
            )
            self._federated_orchestrator_transport_privkey_b64 = None

        # Sprint 312 — pipeline inference orchestrator.
        # Coordinates multi-stage TEE-attested inference.
        # Privkey from PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY
        # (Ed25519, b64, 32B); in-memory only, never
        # persisted. Filesystem state via
        # PRSM_PIPELINE_ORCHESTRATOR_DIR.
        try:
            import os as _os
            raw = (
                _os.environ.get(
                    "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY",
                    "",
                ).strip()
            )
            if raw:
                from pathlib import Path as _Path
                from prsm.compute.inference.pipeline_orchestrator import (
                    PipelineInferenceOrchestrator,
                )
                pdir = (
                    _os.environ.get(
                        "PRSM_PIPELINE_ORCHESTRATOR_DIR",
                        "",
                    ).strip()
                )
                self._pipeline_inference_orchestrator = (
                    PipelineInferenceOrchestrator(
                        orchestrator_privkey_b64=raw,
                        persist_dir=(
                            _Path(pdir) if pdir else None
                        ),
                    )
                )
                logger.info(
                    "Pipeline inference orchestrator wired"
                    " (jobs=%d)",
                    len(
                        self._pipeline_inference_orchestrator.list_jobs(),
                    ),
                )
            else:
                self._pipeline_inference_orchestrator = None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PipelineInferenceOrchestrator construction "
                "failed: %s — /admin/inference/pipeline/* "
                "will return 503.", exc,
            )
            self._pipeline_inference_orchestrator = None

        # Sprint 313 — pipeline stage runner. Enables this
        # node to serve as a REMOTE STAGE WORKER for some
        # other node's PipelineInferenceOrchestrator. Set
        # PRSM_PIPELINE_STAGE_RUNNER_ENABLED=1 to enable
        # with the default stub runner. Real PyTorch
        # per-stage runner = sprint 314 via the same hook.
        try:
            import os as _os
            enabled = (
                _os.environ.get(
                    "PRSM_PIPELINE_STAGE_RUNNER_ENABLED",
                    "",
                ).strip().lower()
                in ("1", "true", "yes")
            )
            if enabled:
                from prsm.compute.inference.pipeline_stage import (
                    deterministic_stub_stage_runner,
                )
                self._pipeline_stage_runner = (
                    deterministic_stub_stage_runner()
                )
                logger.info(
                    "Pipeline stage runner wired "
                    "(default stub)",
                )
            else:
                self._pipeline_stage_runner = None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Pipeline stage runner wiring failed: %s",
                exc,
            )
            self._pipeline_stage_runner = None

        # Sprint 303 — UUPS upgrade orchestrator (Vision §14
        # item 7). Filesystem-persisted via
        # PRSM_UPGRADE_ORCHESTRATOR_DIR. All upgrade +
        # rollback execution is composer-only (Foundation
        # Safe gates).
        try:
            from prsm.economy.web3.upgrade_orchestrator import (
                UpgradeOrchestrator,
            )
            from prsm.config.networks import resolve_endpoints
            self._upgrade_orchestrator = (
                UpgradeOrchestrator.from_env()
            )
            self._upgrade_chain_id = (
                resolve_endpoints().chain_id
            )
            logger.info(
                "UpgradeOrchestrator wired "
                "(proposals=%d, chain_id=%s)",
                self._upgrade_orchestrator.count(),
                self._upgrade_chain_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "UpgradeOrchestrator construction failed: "
                "%s — /admin/upgrade/* will return 503.",
                exc,
            )
            self._upgrade_orchestrator = None
            self._upgrade_chain_id = None

        # Sprint 286 — fiat-surface health check at startup.
        # Loud-but-non-blocking: ERROR findings surface in the
        # log so operators see misconfigs (e.g., KYC
        # commissioned without webhook secret) before vendor
        # traffic arrives. Does not refuse to start — operator
        # may be mid-deploy. The /admin/fiat-surface/health
        # endpoint + prsm_fiat_surface_health MCP tool give
        # the same data on demand.
        try:
            import os as _os
            from prsm.economy.web3.fiat_surface_health import (
                check_fiat_surface_health, FindingSeverity,
            )
            _findings = check_fiat_surface_health(_os.environ)
            for _f in _findings:
                if _f.severity == FindingSeverity.ERROR:
                    logger.error(
                        "FIAT-SURFACE HEALTH ⚠ ERROR  "
                        "%s — %s",
                        _f.cause, _f.remediation,
                    )
                elif _f.severity == FindingSeverity.WARN:
                    logger.warning(
                        "FIAT-SURFACE HEALTH △ WARN  "
                        "%s — %s",
                        _f.cause, _f.remediation,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FiatSurfaceHealth check failed: %s — "
                "/admin/fiat-surface/health will still run "
                "if endpoint is hit.",
                exc,
            )

        # Sprint 249 — RoyaltyDispatchRing for the on-chain
        # content-royalty audit trail (sprint 248). Opt-in
        # filesystem persistence via PRSM_ROYALTY_DISPATCH_LOG_DIR.
        try:
            import os as _os
            from pathlib import Path as _Path
            from prsm.node.royalty_dispatch_log import (
                RoyaltyDispatchRing,
            )
            persist_raw = _os.getenv(
                "PRSM_ROYALTY_DISPATCH_LOG_DIR", "",
            ).strip()
            persist_dir = _Path(persist_raw) if persist_raw else None
            self._royalty_dispatch_ring = RoyaltyDispatchRing(
                persist_dir=persist_dir,
            )
            logger.info(
                "RoyaltyDispatchRing wired%s",
                f", persisted to {persist_dir}" if persist_dir else "",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RoyaltyDispatchRing construction failed: %s — "
                "/admin/royalty-dispatch-history will 503.",
                exc,
            )
            self._royalty_dispatch_ring = None

        # Sprint 242 — ReceiptStore for signed InferenceReceipts.
        # Opt-in filesystem persistence via PRSM_RECEIPT_STORE_DIR.
        # Powers GET /compute/receipt/{job_id} for post-hoc audit.
        try:
            from prsm.node.receipt_store import ReceiptStore
            self._receipt_store = ReceiptStore.from_env()
            logger.info(
                "ReceiptStore wired (in-memory LRU)"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ReceiptStore construction failed: %s — "
                "/compute/receipt/{job_id} will return 503.",
                exc,
            )
            self._receipt_store = None

        # Operator audit log (in-memory ring buffer of state-changing
        # API requests). Optional filesystem persistence via
        # PRSM_AUDIT_LOG_DIR. Best-effort wiring — failure-soft.
        try:
            import os as _os
            from pathlib import Path as _Path
            from prsm.node.audit_log import AuditLogRing
            audit_dir_raw = _os.getenv("PRSM_AUDIT_LOG_DIR", "").strip()
            audit_persist_dir = _Path(audit_dir_raw) if audit_dir_raw else None
            retention_raw = _os.getenv(
                "PRSM_AUDIT_LOG_RETENTION_DAYS", "",
            ).strip()
            retention_days = None
            if retention_raw:
                try:
                    retention_days = float(retention_raw)
                    if retention_days <= 0:
                        retention_days = None
                except ValueError:
                    logger.warning(
                        "PRSM_AUDIT_LOG_RETENTION_DAYS=%r not "
                        "numeric; retention disabled", retention_raw,
                    )
            self._audit_log = AuditLogRing(
                persist_dir=audit_persist_dir,
                retention_days=retention_days,
            )
            logger.info(
                "AuditLogRing wired (in-memory ring 1024%s%s)",
                f", persisted to {audit_persist_dir}"
                if audit_persist_dir else "",
                f", retention={retention_days}d"
                if retention_days else "",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AuditLogRing construction failed: %s — "
                "/audit/recent will return 503.", exc,
            )
            self._audit_log = None

        # JobReaper for per-job duration cap (PRSM_FORGE_MAX_DURATION_SEC).
        # Decoupled from /compute/forge body — runs as separate background
        # task scanning JobHistoryStore for stale IN_PROGRESS records.
        # Disabled when env var unset (v1 behavior preserved).
        try:
            import os as _os
            from prsm.node.job_reaper import JobReaper
            duration_raw = _os.getenv(
                "PRSM_FORGE_MAX_DURATION_SEC", "",
            ).strip()
            interval_raw = _os.getenv(
                "PRSM_FORGE_REAPER_INTERVAL_SEC", "",
            ).strip()
            self._job_reaper = None
            self._job_reaper_task = None
            if duration_raw and self._job_history is not None:
                try:
                    cap = float(duration_raw)
                    if cap > 0:
                        kwargs = {
                            "job_history": self._job_history,
                            "payment_escrow": getattr(
                                self, "_payment_escrow", None,
                            ),
                            "max_duration_seconds": cap,
                        }
                        if interval_raw:
                            try:
                                ival = float(interval_raw)
                                if ival > 0:
                                    kwargs["interval_seconds"] = ival
                            except ValueError:
                                logger.warning(
                                    "PRSM_FORGE_REAPER_INTERVAL_SEC=%r "
                                    "not numeric; using default",
                                    interval_raw,
                                )
                        self._job_reaper = JobReaper(**kwargs)
                        logger.info(
                            "JobReaper wired (cap=%ss, interval=%ss)",
                            cap,
                            kwargs.get(
                                "interval_seconds",
                                self._job_reaper.interval_seconds,
                            ),
                        )
                except ValueError:
                    logger.warning(
                        "PRSM_FORGE_MAX_DURATION_SEC=%r not numeric; "
                        "duration cap disabled", duration_raw,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "JobReaper construction failed: %s — "
                "duration cap disabled.", exc,
            )
            self._job_reaper = None
            self._job_reaper_task = None

        # DaemonWatchdog: dispatches webhook event when any daemon
        # task crashes silently. Operators wire PRSM_WEBHOOK_URL
        # (+ optional PRSM_WEBHOOK_SECRET for HMAC signing) to
        # opt in. Disabled when URL unset.
        self._daemon_watchdog = None
        self._daemon_watchdog_task = None
        try:
            import os as _os
            webhook_url = _os.getenv("PRSM_WEBHOOK_URL", "").strip()
            if webhook_url:
                from prsm.node.daemon_watchdog import DaemonWatchdog
                webhook_secret = _os.getenv(
                    "PRSM_WEBHOOK_SECRET", "",
                ).strip() or None
                # Reuse early-constructed deliverer + log (built
                # before the StorageSlashingWatcher so slash events
                # could fire webhooks). Re-construct here only if
                # early init failed (env was set during run rather
                # than startup, or import error skipped early init).
                if self._webhook_deliverer is None:
                    from prsm.node.webhook_delivery import WebhookDeliverer
                    from prsm.node.webhook_log import WebhookLogRing
                    self._webhook_log = WebhookLogRing()
                    self._webhook_deliverer = WebhookDeliverer(
                        log_ring=self._webhook_log,
                    )
                deliverer = self._webhook_deliverer
                interval_raw = _os.getenv(
                    "PRSM_DAEMON_WATCHDOG_INTERVAL_SEC", "",
                ).strip()
                wd_kwargs = {
                    "node": self,
                    "webhook_deliverer": deliverer,
                    "webhook_url": webhook_url,
                    "webhook_secret": webhook_secret,
                }
                if interval_raw:
                    try:
                        ival = float(interval_raw)
                        if ival > 0:
                            wd_kwargs["interval_seconds"] = ival
                    except ValueError:
                        logger.warning(
                            "PRSM_DAEMON_WATCHDOG_INTERVAL_SEC=%r "
                            "not numeric; using default",
                            interval_raw,
                        )
                # Optional canonical-pin drift detection.
                # PRSM_WEBHOOK_WATCH_CANONICAL=1 enables per-sweep
                # check; transitions from match to drift fire
                # canonical.drifted webhook events.
                if _os.getenv(
                    "PRSM_WEBHOOK_WATCH_CANONICAL", "",
                ).strip() == "1":
                    def _canonical_check_fn():
                        from prsm.config.networks import (
                            get_network_config, _resolve_network_name,
                        )
                        try:
                            cfg = get_network_config(
                                _resolve_network_name(),
                            )
                        except Exception:
                            return {}
                        result = {}
                        # ftns_ledger
                        ftns = getattr(self, "ftns_ledger", None)
                        if ftns is not None:
                            wired = getattr(
                                ftns, "contract_address", None,
                            )
                            canonical = cfg.ftns_token
                            if wired and canonical:
                                result["ftns_ledger"] = (wired, canonical)
                        # royalty_distributor
                        royalty = getattr(
                            self, "_royalty_distributor_client", None,
                        )
                        if royalty is not None:
                            wired = getattr(
                                royalty, "distributor_address", None,
                            )
                            canonical = cfg.royalty_distributor
                            if wired and canonical:
                                result["royalty_distributor"] = (
                                    wired, canonical,
                                )
                        # provenance_registry (V2 canonical)
                        provenance = getattr(
                            self, "_provenance_client", None,
                        )
                        if provenance is not None:
                            wired = getattr(
                                provenance, "contract_address", None,
                            )
                            canonical = cfg.provenance_registry_v2
                            if wired and canonical:
                                result["provenance_registry"] = (
                                    wired, canonical,
                                )
                        # Phase 7-storage + Phase 8 contract pins.
                        # Each client exposes `.address`; mismatch
                        # fires canonical.drifted same as the 3 above.
                        for attr_name, networks_field, label in (
                            (
                                "_storage_slashing_client",
                                "storage_slashing",
                                "storage_slashing",
                            ),
                            (
                                "_compensation_distributor_client",
                                "compensation_distributor",
                                "compensation_distributor",
                            ),
                            (
                                "_key_distribution_client",
                                "key_distribution",
                                "key_distribution",
                            ),
                        ):
                            client = getattr(self, attr_name, None)
                            if client is None:
                                continue
                            # Sprint 142 — read CONTRACT address,
                            # not signer. `.address` returns the
                            # operator wallet on phase 7-storage /
                            # phase 8 clients. Sprint 86 used the
                            # wrong attribute, producing false
                            # canonical.drifted webhooks at every
                            # sweep. Mirror the /health/detailed
                            # fix.
                            wired = (
                                getattr(client, "contract_address", None)
                                or getattr(client, "address", None)
                            )
                            canonical = getattr(
                                cfg, networks_field, None,
                            )
                            if wired and canonical:
                                result[label] = (wired, canonical)
                        return result
                    wd_kwargs["check_canonical_pins"] = True
                    wd_kwargs["canonical_check_fn"] = _canonical_check_fn
                    logger.info(
                        "DaemonWatchdog: canonical-pin drift detection enabled",
                    )
                self._daemon_watchdog = DaemonWatchdog(**wd_kwargs)
                logger.info(
                    "DaemonWatchdog wired (url=%s%s)",
                    webhook_url,
                    " [signed]" if webhook_secret else "",
                )
                # Wire escrow.leaked webhook event source.
                # When PaymentEscrow.periodic_cleanup reaps any
                # stale escrows, this callback dispatches an
                # escrow.leaked event with count + node_id.
                if hasattr(self, "_payment_escrow") and (
                    self._payment_escrow is not None
                ):
                    import time as _time
                    async def _on_cleanup(count: int) -> None:
                        if count <= 0:
                            return  # only fire when leaked > 0
                        try:
                            await deliverer.deliver(
                                url=webhook_url,
                                event="escrow.leaked",
                                payload={
                                    "event": "escrow.leaked",
                                    "node_id": self.identity.node_id,
                                    "count": count,
                                    "timestamp": _time.time(),
                                },
                                secret=webhook_secret,
                            )
                        except Exception as cb_exc:
                            logger.warning(
                                "escrow.leaked dispatch raised: %s",
                                cb_exc,
                            )
                    # Best-effort: existing PaymentEscrow may not
                    # have the on_cleanup_callback hook (older
                    # impl). Set if attribute exists.
                    if hasattr(
                        self._payment_escrow,
                        "_on_cleanup_callback",
                    ):
                        self._payment_escrow._on_cleanup_callback = (
                            _on_cleanup
                        )
                        logger.info(
                            "escrow.leaked webhook event source wired",
                        )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DaemonWatchdog construction failed: %s — "
                "crash webhooks disabled.", exc,
            )
            self._daemon_watchdog = None

        self._result_consensus = ResultConsensus(
            epsilon=0.01,
            timeout_seconds=300.0,
        )

        # ── Remote Shard Dispatcher (Phase 2) ─────────────────────
        # Plugs into TensorParallelExecutor's remote_dispatcher slot
        # (wired below). Tier A verification (receipt-only) in Phase 2;
        # Tiers B/C plug in at Phase 7 via the same VerificationStrategy
        # protocol without changing the dispatch protocol.
        from prsm.compute.remote_dispatcher import RemoteShardDispatcher
        from prsm.compute.shard_receipt import ReceiptOnlyVerification

        self.remote_shard_dispatcher = RemoteShardDispatcher(
            identity=self.identity,
            transport=self.transport,
            payment_escrow=self._payment_escrow,
            verification_strategy=ReceiptOnlyVerification(),
            default_timeout=30.0,
            max_retries=1,
            max_shard_bytes=10 * 1024 * 1024,
            local_fallback=None,
        )

        async def _tensor_remote_dispatch(shard, input_data, assignment):
            """Adapter from TensorParallelExecutor's (shard, bytes,
            assignment) contract to RemoteShardDispatcher.dispatch's
            (shard, np.ndarray, node_id, job_id, stake_tier, amount)
            contract. Wraps output in the dict shape _execute_local
            returns.
            """
            import numpy as _np

            from prsm.compute.model_sharding.models import PipelineStakeTier

            node_id = assignment.get("node_id", "")
            job_id = assignment.get("job_id", "")
            tier_label = assignment.get("stake_tier", "standard")
            tier_map = {t.label: t for t in PipelineStakeTier}
            stake_tier = tier_map.get(tier_label, PipelineStakeTier.STANDARD)
            escrow_amount = float(assignment.get("escrow_amount_ftns", 1.0))

            input_arr = _np.frombuffer(input_data, dtype=_np.float64)
            if input_arr.size == 0:
                input_arr = _np.ones(
                    shard.tensor_shape[-1] if len(shard.tensor_shape) > 1
                    else shard.tensor_shape[0]
                )

            output = await self.remote_shard_dispatcher.dispatch(
                shard=shard,
                input_tensor=input_arr,
                node_id=node_id,
                job_id=job_id,
                stake_tier=stake_tier,
                escrow_amount_ftns=escrow_amount,
            )

            return {
                "shard_index": shard.shard_index,
                "node_id": node_id,
                "output_array": output.tolist(),
                "execution_mode": "remote",
            }

        self._tensor_remote_dispatch = _tensor_remote_dispatch

        # ── Mobile Agent Dispatch (Ring 2) ────────────────────────────
        try:
            from prsm.compute.agents.dispatcher import AgentDispatcher
            from prsm.compute.agents.executor import AgentExecutor

            self.agent_dispatcher = AgentDispatcher(
                identity=self.identity,
                gossip=self.gossip,
                transport=self.transport,
                escrow=self._payment_escrow,
            )

            self.agent_executor = AgentExecutor(
                identity=self.identity,
                gossip=self.gossip,
            )
            logger.info("Mobile agent dispatch (Ring 2) initialized")
        except ImportError:
            self.agent_dispatcher = None
            self.agent_executor = None
            logger.debug("Mobile agent dispatch not available")

        # ── Swarm Compute (Ring 3) ────────────────────────────────────
        try:
            from prsm.compute.swarm.coordinator import SwarmCoordinator

            self.swarm_coordinator = SwarmCoordinator(
                dispatcher=self.agent_dispatcher,
                result_consensus=getattr(self, '_result_consensus', None),
            )
            logger.info("Swarm compute (Ring 3) initialized")
        except (ImportError, AttributeError):
            self.swarm_coordinator = None
            logger.debug("Swarm compute not available")

        # ── Economy Engine (Ring 4) ───────────────────────────────────
        try:
            from prsm.economy.pricing.engine import PricingEngine
            from prsm.economy.prosumer import ProsumerManager

            self.pricing_engine = PricingEngine()
            self.prosumer_manager = ProsumerManager(
                node_id=self.identity.node_id,
                ledger=self.ledger,
            )

            from prsm.economy.pricing.revenue_split import RevenueSplitEngine
            from prsm.economy.pricing.data_listing import DataListingManager
            from prsm.economy.pricing.spot_arbitrage import SpotArbitrage

            self.revenue_split = RevenueSplitEngine()
            self.data_listing_manager = DataListingManager()
            self.spot_arbitrage = SpotArbitrage(pricing_engine=self.pricing_engine)

            logger.info("Economy engine (Ring 4) initialized")
        except ImportError:
            self.pricing_engine = None
            self.prosumer_manager = None
            self.revenue_split = None
            self.data_listing_manager = None
            self.spot_arbitrage = None
            logger.debug("Economy engine not available")

        # ── Content Economy (Phase 4) ──────────────────────────────────────
        # Determine royalty model from config
        royalty_model = RoyaltyModel.PHASE4
        if getattr(self.config, 'royalty_model', 'phase4') == 'legacy':
            royalty_model = RoyaltyModel.LEGACY
        
        # Initialize vector store backend (optional)
        _vector_store = None
        vector_backend = getattr(self.config, 'vector_backend', 'memory')
        if vector_backend and vector_backend != 'disabled':
            try:
                from prsm.node.vector_store_backend import create_vector_store
                _vector_store = create_vector_store(
                    backend=vector_backend,
                    postgres_host=getattr(self.config, 'postgres_host', 'localhost'),
                    postgres_port=getattr(self.config, 'postgres_port', 5432),
                    postgres_database=getattr(self.config, 'postgres_database', 'prsm'),
                    postgres_user=getattr(self.config, 'postgres_user', 'prsm'),
                    postgres_password=getattr(self.config, 'postgres_password', ''),
                )
                await _vector_store.initialize()
                logger.info(f"Vector store initialized: {vector_backend}")
            except Exception as e:
                logger.warning(f"Vector store initialization failed: {e}")
                _vector_store = None
        
        self.content_economy = ContentEconomy(
            identity=self.identity,
            ledger=self.ledger,
            gossip=self.gossip,
            content_index=self.content_index,
            ftns_ledger=self.ftns_ledger,
            royalty_model=royalty_model,
            min_replicas=getattr(self.config, 'min_replicas', 3),
            vector_store=_vector_store,
            embedding_fn=_embedding_fn if '_embedding_fn' in dir() else None,
        )
        
        # Register content economy with API routes
        try:
            from prsm.api.content_economy_routes import set_content_economy
            set_content_economy(self.content_economy)
        except ImportError:
            pass  # API routes not available
        
        # Wire ContentEconomy to StorageProvider for replication tracking (Phase 4)
        if self.storage_provider:
            self.storage_provider.content_economy = self.content_economy
        
        # Initialize multi-party escrow for batch settlements (Phase 4)
        from prsm.node.multi_party_escrow import MultiPartyEscrow, EscrowConfig
        self._mp_escrow = MultiPartyEscrow(
            ftns_ledger=self.ftns_ledger,
            config=EscrowConfig(
                min_batch_size=getattr(self.config, 'escrow_min_batch_size', 5),
                min_batch_value=getattr(self.config, 'escrow_min_batch_value', 0.1),
                settlement_interval=getattr(self.config, 'escrow_settlement_interval', 300.0),
            ),
        )
        if self.content_economy:
            self.content_economy.set_escrow(self._mp_escrow)
        
        self.db_initialized = False
        self._broadcast_sent = set()

        # ── Batch Settlement (gas-efficient on-chain broadcasting) ──
        from prsm.economy.batch_settlement import BatchSettlementManager, SettlementMode
        # sp1476 — durable so a restart (every deploy) resumes owed on-chain
        # payouts instead of dropping the queue + in-flight tracker. Mirrors the
        # PendingWithdrawStore wiring; override dir via PRSM_BATCH_SETTLEMENT_DIR
        # ("" / ":memory:" → in-memory only).
        _bs_dir = os.environ.get("PRSM_BATCH_SETTLEMENT_DIR")
        if _bs_dir in ("", ":memory:"):
            _bs_dir = None
        elif _bs_dir is None:
            _bs_dir = str(Path.home() / ".prsm")
        self._batch_settlement = BatchSettlementManager(
            ftns_ledger=self.ftns_ledger,
            node_id=self.identity.node_id,
            connected_address=(
                self.ftns_ledger._connected_address
                if hasattr(self.ftns_ledger, '_connected_address')
                else None
            ),
            mode=SettlementMode.PERIODIC,
            flush_interval=600.0,     # 10 minutes
            flush_threshold=1.0,      # or when pending ≥ 1.0 FTNS
            persist_dir=_bs_dir,
        )
        
        # ── On-chain cross-node settlement client (sprint 1036, opt-in) ──
        # OFF by default. PRSM_ONCHAIN_SETTLEMENT=1 + a provider address opt in
        # to LOCAL accumulation (the receipt->BatchedReceipt adapter feeds it,
        # sprint 1035); commit/finalize additionally need a funded
        # FTNS_WALLET_PRIVATE_KEY (brick 2 + the key ceremony). Fail-open: a
        # settlement-wiring problem must never crash daemon initialize().
        self._onchain_settlement_client = None
        try:
            from prsm.settlement.client_wiring import (
                build_onchain_settlement_client_or_none,
            )
            # sp1140 (Brick E.2) — OPT-IN producer receipt retention. Only when
            # PRSM_SETTLEMENT_AUDIT is set do we build a PublishedBatchStore and hand
            # it to the client; OFF (the default) leaves published_batch_store=None so
            # the commit path's retention put is a strict no-op (byte-for-byte unchanged).
            _published_batch_store = None
            # sp1144 (§7 producer wiring) — the §7 InferenceReceipt retention store, built
            # under the SAME audit gate. When set it is threaded into the accumulate adapter
            # (so the original §7 receipt is retained keyed by the committed leaf hash) and
            # into the announcer (so the published blob is ENRICHED with the §7 receipts).
            # OFF (the default) leaves it None => no §7 retention, plain published blob,
            # money path byte-for-byte unchanged.
            _inference_receipt_store = None
            # sp1164 — the observer's verified-foreign-batch retention store (built under the
            # SAME audit gate). OFF (the default) leaves it None => the audit loop retains
            # nothing for out-of-daemon challenge prep (behavior unchanged).
            _verified_batch_store = None
            try:
                from prsm.settlement.settlement_audit_wiring import audit_enabled
                if audit_enabled():
                    from pathlib import Path as _Path
                    from prsm.settlement.published_batch_store import (
                        PublishedBatchStore,
                    )
                    from prsm.settlement.inference_receipt_store import (
                        InferenceReceiptStore,
                    )
                    _audit_store_path = (
                        os.environ.get("PRSM_SETTLEMENT_AUDIT_STORE_FILE", "").strip()
                        or str(_Path.home() / ".prsm" / "published_batches.json")
                    )
                    _published_batch_store = PublishedBatchStore(_audit_store_path)
                    _ir_store_path = (
                        os.environ.get(
                            "PRSM_SETTLEMENT_INFERENCE_RECEIPT_STORE_FILE", "",
                        ).strip()
                        or str(_Path.home() / ".prsm" / "inference_receipts.json")
                    )
                    _inference_receipt_store = InferenceReceiptStore(_ir_store_path)
                    # sp1164 — the observer's VERIFIED-FOREIGN-BATCH store: the audit loop
                    # retains cross-check-passed third-party batches here so an operator can
                    # later PREPARE a challenge for them (scripts/settlement_challenge_prepare.py).
                    _verified_batches_path = (
                        os.environ.get(
                            "PRSM_SETTLEMENT_VERIFIED_BATCHES_FILE", "",
                        ).strip()
                        or str(_Path.home() / ".prsm" / "settlement_verified_batches.json")
                    )
                    _verified_batch_store = PublishedBatchStore(_verified_batches_path)
            except Exception as _audit_store_exc:  # noqa: BLE001 — never crash startup
                logger.warning(
                    "Settlement audit retention store build failed (audit OFF for "
                    "retention): %s", _audit_store_exc,
                )
                _published_batch_store = None
                _inference_receipt_store = None
                _verified_batch_store = None
            self._settlement_published_batch_store = _published_batch_store
            self._settlement_inference_receipt_store = _inference_receipt_store
            self._settlement_verified_batch_store = _verified_batch_store

            # sp1360 (F1 redesign, R4) — Tier B/C paid-key serve data plane (default OFF). When
            # PRSM_PAID_KEY_SERVE is enabled, hold a retained-key store the publish path populates
            # and, if PRSM_CONTENT_ACCESS_VERIFIER is set, a read-only verify_payment callable the
            # GET /content/paid-key endpoint gates on. Fail-soft: any wiring error leaves the serve
            # disabled (the endpoint returns 503) rather than breaking node startup.
            self._paid_key_store = None
            self._paid_key_verify_payment = None
            self._paid_publish_key_client = None      # sp1367 — publisher deposit client
            self._paid_publish_provenance_client = None  # sp1458 — publisher-key-signed registry client
            if (os.environ.get("PRSM_PAID_KEY_SERVE") or "").strip().lower() in (
                    "1", "true", "yes", "on"):
                try:
                    from prsm.node.paid_key_serve import (
                        PaidKeyStore,
                        build_paid_key_verify_payment,
                        resolve_paid_key_store_path,
                    )
                    # sp1362 (R5 HIGH) / sp1438 (audit B5 #1): durable store so paid buyers aren't
                    # stranded on restart. resolve_paid_key_store_path defaults to a durable
                    # ~/.prsm path when PRSM_PAID_KEY_STORE_FILE is unset — the bare `or None` here
                    # silently degraded to in-memory, so a paid buyer got a permanent 404 (no
                    # refund) after any restart.
                    self._paid_key_store = PaidKeyStore(
                        resolve_paid_key_store_path(os.environ))
                    _cav = (os.environ.get("PRSM_CONTENT_ACCESS_VERIFIER") or "").strip()
                    if _cav:
                        from prsm.config.networks import resolve_endpoints
                        from prsm.economy.web3.content_access_verifier import (
                            ContentAccessVerifierClient,
                        )
                        _ep = resolve_endpoints(None)
                        # sp1362 (R5 low): pin the chain on the serve-side gate too — the gate
                        # decides whether to release the retained key, so a hostile/mis-chained RPC
                        # reporting a bogus verifyPayment==true would leak it.
                        _vc = ContentAccessVerifierClient(
                            _ep.rpc_url, _cav, _ep.ftns_token,  # read-only (no signer)
                            expected_chain_id=_ep.chain_id)
                        self._paid_key_verify_payment = build_paid_key_verify_payment(_vc)
                    # sp1367 — publisher key client, so this operator node can PUBLISH paid content
                    # (deposit the commitment). Optional: only when PRSM_PAID_PUBLISHER_KEY is set.
                    _pubkey = (os.environ.get("PRSM_PAID_PUBLISHER_KEY") or "").strip()
                    if _pubkey and _ep.key_distribution:
                        from prsm.economy.web3.key_distribution import KeyDistributionClient
                        self._paid_publish_key_client = KeyDistributionClient(
                            _ep.rpc_url, _ep.key_distribution, private_key=_pubkey)
                        # sp1458 — a ProvenanceRegistry client signed with the SAME paid-publisher key,
                        # so a STREAMING large-paid-content publish registers creator == depositor
                        # on-chain (the sp1365 anti-squat invariant — else the buyer's fee doesn't
                        # credit the publisher). The node's default _provenance_client signs with a
                        # DIFFERENT key (FTNS_WALLET_PRIVATE_KEY), so it can't be reused here. Only when
                        # the V2 ProvenanceRegistry is configured; the streaming route fails closed
                        # (503) without it.
                        _prov_addr = (
                            (os.environ.get("PRSM_PROVENANCE_REGISTRY_ADDRESS") or "").strip()
                            or getattr(_ep, "provenance_registry", None))
                        if _prov_addr:
                            from prsm.economy.web3.provenance_registry_v2 import (
                                ProvenanceRegistryV2Client)
                            self._paid_publish_provenance_client = ProvenanceRegistryV2Client(
                                rpc_url=_ep.rpc_url, contract_address=_prov_addr,
                                private_key=_pubkey)
                except Exception as _exc:  # noqa: BLE001
                    logger.warning("paid-key serve wiring failed (disabled): %s", _exc)
                    self._paid_key_store = None
                    self._paid_key_verify_payment = None
                    self._paid_publish_key_client = None
                    self._paid_publish_provenance_client = None

            self._onchain_settlement_client = (
                build_onchain_settlement_client_or_none(
                    provider_address=self._operator_address,
                    published_batch_store=_published_batch_store,
                )
            )
            if self._onchain_settlement_client is not None:
                logger.info(
                    "On-chain settlement client wired (provider=%s, "
                    "write_capable=%s). Accumulation is local; commit/finalize "
                    "need a funded settler key.",
                    self._operator_address,
                    bool(
                        (os.getenv("FTNS_WALLET_PRIVATE_KEY", "") or "").strip()
                    ),
                )
            else:
                logger.info(
                    "On-chain settlement client OFF (opt in with "
                    "PRSM_ONCHAIN_SETTLEMENT=1 + PRSM_OPERATOR_ADDRESS; add "
                    "FTNS_WALLET_PRIVATE_KEY for on-chain commit)."
                )
        except Exception as _settle_exc:  # noqa: BLE001
            logger.warning(
                "On-chain settlement client build failed (non-fatal): %s",
                _settle_exc,
            )

        # Sprint 1056 (requester-payment brick 3) — build the per-request
        # PaymentAuthorization verifier (OFF unless PRSM_REQUESTER_PAYMENT=1). When
        # the on-chain settlement client carries a contract with an EscrowPool
        # client, wire its balance_of as the pre-flight balance reader; else the
        # pre-flight is skipped (finalizeBatch still enforces funding). Fail-open.
        self._payment_verifier = None
        self._relayer_verifier = None   # sp1094 — delegated-auth (relayer) verifier
        self._paid_requester_by_job = {}
        try:
            from prsm.settlement.client_wiring import (
                build_payment_verifier_or_none,
            )
            _balance_reader = None
            _sc = self._onchain_settlement_client
            _escrow = getattr(getattr(_sc, "_contract", None), "escrow_pool", None)
            if _escrow is not None and hasattr(_escrow, "balance_of"):
                _balance_reader = _escrow.balance_of
            self._payment_verifier = build_payment_verifier_or_none(
                provider_address=self._operator_address,
                balance_reader=_balance_reader,
            )
            if self._payment_verifier is not None:
                logger.info(
                    "Requester-payment verifier wired (provider=%s, "
                    "escrow_preflight=%s).",
                    self._operator_address, _balance_reader is not None,
                )
            # sp1094 (relayer brick 4) — also build the relayer (delegated-auth) verifier
            # under the same gate, sharing the escrow pre-flight reader. A request that
            # carries a payment_delegation routes here (a gateway signing on a funder's
            # behalf); a plain payment_authorization stays on the self-signed verifier.
            from prsm.settlement.client_wiring import build_relayer_verifier_or_none
            self._relayer_verifier = build_relayer_verifier_or_none(
                provider_address=self._operator_address,
                balance_reader=_balance_reader,
            )
            if self._relayer_verifier is not None:
                logger.info(
                    "Relayer (delegated-auth) payment verifier wired (provider=%s).",
                    self._operator_address,
                )
        except Exception as _pay_exc:  # noqa: BLE001
            logger.warning(
                "Requester-payment verifier build failed (non-fatal): %s",
                _pay_exc,
            )

        # Sprint 1190 (day-one blocker #3) — build the on-chain TESTNET FTNS faucet
        # ONCE here (so /ftns/faucet/onchain doesn't rebuild a web3 client per request).
        # build_onchain_faucet_or_none returns None unless PRSM_FAUCET_PRIVATE_KEY is set
        # AND the active network is Base Sepolia — it NEVER constructs on mainnet, the
        # config-time half of the kill-switch. Fail-open (None disables the endpoint).
        self._onchain_faucet = None
        try:
            from prsm.economy.web3.ftns_faucet import build_onchain_faucet_or_none
            self._onchain_faucet = build_onchain_faucet_or_none()
            if self._onchain_faucet is not None:
                logger.info(
                    "On-chain TESTNET FTNS faucet wired (faucet=%s, Base Sepolia only).",
                    self._onchain_faucet.faucet_address,
                )
        except Exception as _faucet_exc:  # noqa: BLE001
            logger.warning(
                "On-chain faucet build failed (non-fatal): %s", _faucet_exc,
            )

        # ── Settler Registry (Phase 6: L2-style staking for batch security) ──
        from prsm.node.settler_registry import SettlerRegistry
        self._settler_registry = SettlerRegistry(
            min_settler_bond=10_000.0,    # 10K FTNS to become a settler
            settlement_threshold=3,        # 3-of-N multi-sig for batch approval
            max_settlers=10,
            ftns_service=_StakingFTNSAdapter(self.ledger, self.identity.node_id),
            staking_manager=self.staking_manager,
        )
        
        # Wire: When batch gets multi-sig approval, trigger settlement
        async def _on_batch_approved(batch):
            """Callback when batch reaches multi-sig threshold."""
            logger.info(
                "Batch approved via multi-sig, executing settlement",
                batch_id=batch.batch_id,
                signatures=len(batch.signatures),
            )
            # The batch settlement manager handles the actual on-chain tx
            result = await self._batch_settlement.flush()
            batch.settled = True
            batch.settlement_tx = result.tx_hashes[0] if result.tx_hashes else None
        
        self._settler_registry.on_settlement_ready(_on_batch_approved)

        # Agent Forge (Ring 5) removed in v1.6.0 — legacy NWTN AGI framework.
        #
        # Replacement = QueryOrchestrator (data-query path per Vision §4).
        # Default-disabled. Operators opt in via PRSM_QUERY_ORCHESTRATOR_ENABLED=1
        # AFTER verifying their deployment delivers the canonical workflow
        # end-to-end. Disabled-by-default keeps `agent_forge = None` so the
        # MCP `BROKEN_TOOLS_HIDDEN` gate stays effective until B8 lands.
        #
        # See:
        #   docs/2026-05-08-query-orchestrator-wiring-readiness.md
        #   docs/2026-05-07-aggregator-selector-threat-model.md
        self.agent_forge = self._build_query_orchestrator_or_none()

        # Sprint 488 (F26 fix) — ContentFingerprintRegistry must
        # be wired UNCONDITIONALLY, not gated behind
        # QueryOrchestrator. Pre-fix: the registry was constructed
        # inside `_build_query_orchestrator_or_none` which
        # early-returns when `PRSM_QUERY_ORCHESTRATOR_ENABLED` is
        # unset (the default). Result: every daemon without QO had
        # `_content_fingerprint_registry = None` → §14 anti-Sybil
        # first-creator-wins NEVER fired → every duplicate upload
        # returned `duplicate_of_creator: null` regardless of race
        # conditions. Move it out so the registry is wired on
        # every daemon. (The duplicate init in
        # `_build_query_orchestrator_or_none` is left in place for
        # QO-enabled deployments; the second from_env() call
        # constructs a fresh instance, harmlessly overwriting the
        # earlier reference — a follow-on can consolidate.)
        try:
            from prsm.marketplace.content_fingerprint_registry import (  # noqa: E501
                ContentFingerprintRegistry,
            )
            self._content_fingerprint_registry = (
                ContentFingerprintRegistry.from_env()
            )
            logger.info(
                "ContentFingerprintRegistry wired "
                "(persist_dir=%s)",
                getattr(
                    self._content_fingerprint_registry,
                    "_persist_dir", None,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ContentFingerprintRegistry "
                "construction failed: %s — fingerprint "
                "dedup will not be enforced.",
                exc,
            )
            self._content_fingerprint_registry = None

        # Sprint 494 (F34 fix) — CreatorReputationTracker
        # and CreatorStakeClient are siblings of the F26
        # fingerprint registry: all three were originally
        # init'd inside `_build_query_orchestrator_or_none`
        # and therefore None on every non-QO daemon. Sprint
        # 488 moved only ContentFingerprintRegistry out;
        # sprint 494 moves the other two so the
        # cross-feature integration chain
        # (upload → retrieve → creator-reputation update →
        # auto-record-access → royalty dispatch) works on
        # default daemons. Pre-fix:
        # `/marketplace/creator-reputation/{id}` returned
        # 503 "tracker not initialized" on every default
        # daemon → cross-feature §14 reputation surface
        # silently inert.
        try:
            from prsm.marketplace.creator_reputation import (
                CreatorReputationTracker,
            )
            self._creator_reputation_tracker = (
                CreatorReputationTracker()
            )
            logger.info("CreatorReputationTracker wired")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CreatorReputationTracker construction "
                "failed: %s — /marketplace/creator-"
                "reputation/* will return 503.",
                exc,
            )
            self._creator_reputation_tracker = None
        try:
            from prsm.marketplace.creator_stake_client import (
                CreatorStakeClient,
            )
            self._creator_stake_client = (
                CreatorStakeClient.from_env()
            )
            logger.info(
                "CreatorStakeClient wired (commissioned=%s)",
                self._creator_stake_client.is_commissioned(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CreatorStakeClient construction failed: "
                "%s — /marketplace/creator-stake/* will "
                "return 503.",
                exc,
            )
            self._creator_stake_client = None

        # ── Confidential Compute (Ring 7) ─────────────────────────────
        try:
            from prsm.compute.tee.confidential_executor import ConfidentialExecutor
            from prsm.compute.tee.models import PrivacyLevel

            self.confidential_executor = ConfidentialExecutor(
                privacy_level=PrivacyLevel.STANDARD,
            )
            logger.info("Confidential compute (Ring 7) initialized")
        except ImportError:
            self.confidential_executor = None
            logger.debug("Confidential compute not available")

        # ── Model Sharding (Ring 8) ───────────────────────────────────
        try:
            from prsm.compute.model_sharding.executor import TensorParallelExecutor
            from prsm.compute.model_sharding.models import PipelineConfig

            self.tensor_executor = TensorParallelExecutor(
                confidential_executor=self.confidential_executor,
                pipeline_config=PipelineConfig(),
                remote_dispatcher=self._tensor_remote_dispatch,
            )
            logger.info("Model sharding (Ring 8) initialized")
        except ImportError:
            self.tensor_executor = None
            logger.debug("Model sharding not available")

        # ── NWTN Model Service (Ring 9) ───────────────────────────────
        try:
            from prsm.compute.nwtn.training.model_service import NWTNModelService

            self.nwtn_model_service = NWTNModelService(
                tensor_executor=self.tensor_executor,
            )
            logger.info("NWTN model service (Ring 9) initialized")
        except ImportError:
            self.nwtn_model_service = None
            logger.debug("NWTN model service not available")

        # ── Security Hardening (Ring 10) ──────────────────────────────
        try:
            from prsm.security import IntegrityVerifier, PrivacyBudgetTracker, PipelineAuditLog

            self.integrity_verifier = IntegrityVerifier()

            # Persistent privacy-budget tracker (Phase 3.x.4): wraps
            # the in-memory PrivacyBudgetTracker in a signed, append-only
            # journal at <data_dir>/privacy_budget/. Restart-survival +
            # tamper-evidence; verify_chain on construction refuses to
            # silently reconstitute possibly-wrong cumulative ε state.
            #
            # JournalCorruptionError propagates out of the try block —
            # it indicates an existing journal is broken, which an
            # operator must investigate (vs. silently falling back to
            # in-memory and losing the audit trail).
            self.privacy_budget = build_persistent_privacy_budget(
                self.config.data_dir, self.identity
            )

            self.pipeline_audit_log = PipelineAuditLog()
            logger.info(
                "Security hardening (Ring 10) initialized; privacy-budget "
                "journal at %s/privacy_budget",
                self.config.data_dir,
            )
        except ImportError:
            self.integrity_verifier = None
            self.privacy_budget = None
            self.pipeline_audit_log = None
            logger.debug("Security hardening not available")

        # ── Local Discovery (mDNS fallback) ───────────────────────────
        try:
            from prsm.node.mdns_discovery import MDNSDiscovery

            self.mdns_discovery = MDNSDiscovery(
                node_id=self.identity.node_id,
                p2p_port=self.config.p2p_port,
                display_name=self.config.display_name,
            )
            logger.info("mDNS local discovery available")
        except ImportError:
            self.mdns_discovery = None

        # Wire ledger_sync and agent_registry into subsystems
        self.content_uploader.ledger_sync = self.ledger_sync
        # Wire content_economy into content_uploader for replication tracking
        if self.content_economy:
            self.content_uploader.content_economy = self.content_economy
        if self.compute_provider:
            self.compute_provider.ledger_sync = self.ledger_sync
            # sp1387 — give self-compute the real local model (so a single-node node's compute
            # jobs return a REAL answer via _run_inference instead of a mock string).
            self.compute_provider.inference_executor = getattr(self, "inference_executor", None)
            # sp1397 — deliver a self-requested job's result straight to the local requester (gossip
            # has no loopback, so a job this node requests from itself would stall "accepted").
            if getattr(self, "compute_requester", None) is not None:
                self.compute_provider.local_result_sink = (
                    self.compute_requester.deliver_local_result)
            # Wire escrow and consensus into compute provider
            self.compute_provider.escrow = self._payment_escrow
            self.compute_provider.consensus = self._result_consensus
            # Wire batch settlement as the on-chain broadcast handler
            if self.ftns_ledger is not None:
                self._payment_escrow.broadcast_tx = self._on_chain_ftns_transfer
        self.compute_requester.ledger_sync = self.ledger_sync
        # sp1405 — on-chain settlement is PROVIDER-side (the earner commits: commitBatch →
        # provider=msg.sender). The node that SERVES a remote job accumulates its OWN earning, so the
        # settlement client goes on compute_provider; the requester only advertises its payer address
        # in the offer. commit/finalize still need a funded settler key. None → off-chain-only, no-op.
        self.compute_requester.operator_address = getattr(self, "_operator_address", None)
        if self.compute_provider is not None:
            self.compute_provider.settlement_client = getattr(
                self, "_onchain_settlement_client", None)
            self.compute_provider.operator_address = getattr(self, "_operator_address", None)
        if hasattr(self.compute_requester, 'escrow'):
            self.compute_requester.escrow = self._payment_escrow
        # sp957 — wire CONSENSUS_MISMATCH evidence routing into the requester's
        # sampler (built lazily in compute_requester.start(), which runs after
        # this). stake_reader resolves a caught provider's bond posture; the
        # mismatch_log.record becomes the sampler's challenge_sink.
        if hasattr(self.compute_requester, 'stake_reader'):
            self.compute_requester.stake_reader = self._compute_stake_reader
            self.compute_requester.mismatch_log = self._consensus_mismatch_log
        if self.storage_provider:
            self.storage_provider.ledger_sync = self.ledger_sync
        if self.bt_provider:
            # sp1423 — bt_provider was the ONE reward-earning subsystem this block forgot.
            # BitTorrentProvider.__init__ sets `self.ledger_sync = None  # Set by node.py` and
            # node.py never did, so `if self.ledger_sync:` in its _reward_loop was ALWAYS False:
            # seeding rewards were credited to the local ledger and NEVER broadcast. The network
            # saw none of those mints (LedgerSync gossips self-credits precisely "for
            # transparency"), and record_nonce was skipped with them. Every other reward path
            # broadcasts — this asymmetry was an omission, not a design choice. It also poisons
            # the sp1419 balance reconciliation, which compares our balance against peers who
            # never saw the mint.
            self.bt_provider.ledger_sync = self.ledger_sync
        self.agent_collaboration.ledger_sync = self.ledger_sync
        self.agent_collaboration.agent_registry = self.agent_registry

        # Wire ledger into gossip for persistence / catch-up
        self.gossip.ledger = self.ledger

        # NWTN orchestrator removed in v1.6.0 — legacy AGI framework replaced
        # by third-party LLMs invoked via MCP

        # Hydrate content uploader from DB (restores provenance across restarts)
        if self.content_uploader:
            hydrated = await self.content_uploader._hydrate_from_db()
            if hydrated > 0:
                logger.info(f"Restored {hydrated} provenance record(s) from DB")

        logger.info("Node initialized — all subsystems ready")

    async def _start_marketplace_advertiser_if_present(self) -> None:
        """sp1459 — begin broadcasting this compute node's marketplace listing.

        No-op when advertising is not opted into (`_marketplace_advertiser is None`, the default),
        so an existing node is unchanged. When enabled, emits the first signed listing immediately
        and then re-broadcasts on a jittered loop so the node becomes selectable supply (+ carries
        its sp1457 stake binding). Fail-soft: a start error is logged but never aborts node bring-up.
        """
        if getattr(self, "_marketplace_advertiser", None) is None:
            return
        try:
            await self._marketplace_advertiser.start()
            logger.info("marketplace advertiser started (broadcasting compute listing)")
        except Exception as exc:  # noqa: BLE001 — advertising must never break node bring-up
            logger.warning("marketplace advertiser failed to start: %s", exc)

    def _start_dht_components_if_present(self) -> None:
        """T3b — start the DHT components stack on its own loop thread.

        Idempotent. Logs the bound port at INFO so operators can verify
        the listener is reachable. A failed start is logged at WARNING
        but does NOT fail the Node.start() — the rest of the node
        continues to run with the DHT in degraded "library-only" mode.
        """
        if self.dht_components is None:
            return
        try:
            port = self.dht_components.start(
                anchor=getattr(self.dht_components, "_t3b_anchor", None),
                creator_pubkey_for=getattr(
                    self.dht_components, "_t3b_creator_pubkey_for", None,
                ),
                verify_signature=getattr(
                    self.dht_components, "_t3b_verify_signature", None,
                ),
            )
            logger.info(
                f"DHT listener bound on "
                f"{self.config.listen_host}:{port}"
            )
            # T4.9.next4: late-bind the EmbeddingDHTClient that
            # DHTNodeComponents.start() just constructed into the
            # ContentUploader so cross-node dedup engages on both
            # lanes (text-vector + binary fingerprint). Skipped when
            # the embedding lane wasn't enabled at build() time
            # (manifest-only DHT setups).
            embedding_client = getattr(
                self.dht_components, "embedding_client", None,
            )
            if embedding_client is not None and self.content_uploader is not None:
                self.content_uploader.set_embedding_dht_client(embedding_client)
                logger.info(
                    "EmbeddingDHTClient late-bound into ContentUploader — "
                    "cross-node dedup is live (semantic + fingerprint lanes)"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"DHT components start failed: "
                f"{type(exc).__name__}: {exc} — "
                f"continuing without inbound DHT listener.",
            )
            self.dht_components = None

    async def start(self) -> None:
        """Start all subsystems concurrently."""
        if self._started:
            return

        # Sprint 1245 — opt-in bounded/rotated FILE logging. The proven deploy
        # posture (`setsid -f bash start_node.sh … >log 2>&1`) grows an unbounded
        # logfile → fills the disk → kills a long-running node. PRSM_LOG_FILE
        # attaches a size-capped RotatingFileHandler to the root logger instead.
        # No-op (stdout-only, byte-identical) when unset; fail-soft so logging
        # setup can never block startup. Done FIRST so the node's own startup
        # logs land in the rotated file from the very beginning.
        try:
            from prsm.core.logging_config import configure_rotating_file_logging
            _log_file = configure_rotating_file_logging()
            if _log_file:
                logger.info("Sprint 1245: node logs → rotating file %s", _log_file)
        except Exception:  # noqa: BLE001 — logging setup must never block startup
            pass

        # Sprint 762 — operator-facing CPU politeness via os.nice().
        # Consumer-device operators (MacBook, gaming PC) often want
        # the daemon to YIELD CPU to their interactive workloads
        # (browser, editor, game) so PRSM doesn't compete with their
        # primary use. Linux/macOS `nice(N)` adds N to the process's
        # priority value — HIGHER nice value = LOWER scheduling
        # priority. Non-root processes can only increase (lower
        # their own priority); reasonable values are 1-19. Set via
        # `PRSM_NODE_NICE=10` in systemd unit. Default unset = 0 =
        # no change (backward-compat).
        #
        # Set BEFORE event-loop capture so the priority change
        # applies to every coroutine the loop schedules. Safe-fail
        # on platforms without os.nice (Windows): log + continue
        # with default priority.
        import os as _os_nice
        _nice_raw = _os_nice.environ.get(
            "PRSM_NODE_NICE", "",
        ).strip()
        if _nice_raw:
            try:
                _nice_increment = int(_nice_raw)
            except ValueError:
                logger.warning(
                    "PRSM_NODE_NICE=%r is not an int; ignoring "
                    "(daemon stays at default priority).",
                    _nice_raw,
                )
            else:
                try:
                    actual = _os_nice.nice(_nice_increment)
                    logger.info(
                        "Sprint 762 — daemon process priority "
                        "adjusted: PRSM_NODE_NICE=%d → effective "
                        "nice=%d", _nice_increment, actual,
                    )
                except AttributeError:
                    # Windows: os.nice not available.
                    logger.warning(
                        "PRSM_NODE_NICE set but os.nice() not "
                        "available on this platform; daemon stays "
                        "at default priority.",
                    )
                except OSError as exc:
                    logger.warning(
                        "PRSM_NODE_NICE=%d rejected by OS "
                        "(non-root processes can only INCREASE "
                        "nice, i.e., LOWER priority — positive "
                        "values only): %s",
                        _nice_increment, exc,
                    )

        # Sprint 960/1011 — network-exposure auth posture check. The protected
        # money + KYC endpoints are only authenticated when PRSM_NODE_API_KEY is
        # set; binding a non-loopback interface without that key exposes them
        # unauthenticated. sp1011 makes this FAIL-CLOSED BY DEFAULT (was
        # warn-only): the node refuses to start in the insecure posture unless
        # the operator explicitly acknowledges running unauthenticated via
        # PRSM_ALLOW_INSECURE_PUBLIC_BIND (e.g. a firewall-fronted deployment).
        # Non-disruptive to the common case — the CLI binds 127.0.0.1 (an "ok"
        # loopback posture), so only a deliberate non-loopback bind without a
        # key is stopped. The legacy opt-in PRSM_REQUIRE_AUTH_ON_PUBLIC_BIND is
        # now the default and remains honored.
        # sp1017 — assess the MANAGEMENT-API bind (config.api_host, served by
        # _run_api), NOT the P2P transport bind (listen_host). The money/KYC
        # endpoints live on the API server; listen_host=0.0.0.0 is the P2P
        # listener and was a false-positive signal that refused every default
        # `prsm node start` even though the API was on loopback.
        # sp1195 — self-provision / reuse the node API key BEFORE the posture gate
        # so a public-bind operator gets a smooth, secure-by-default path: an
        # explicit PRSM_NODE_API_KEY always wins; else a persisted key is reused
        # (stable across restarts); else, opt-in via PRSM_AUTO_PROVISION_API_KEY, a
        # key is minted + persisted + logged. Loopback is untouched; without a key
        # and without opt-in, the fail-closed gate below still stops startup.
        try:
            ensure_public_bind_api_key(
                api_host=getattr(self.config, "api_host", None) or "127.0.0.1",
                env=_os_nice.environ,
            )
        except Exception as _apikey_exc:  # noqa: BLE001 — never block startup on this
            logger.warning("Sprint 1195 — API-key provisioning skipped (%s).", _apikey_exc)
        _posture, _posture_msg = assess_public_bind_auth_posture(
            listen_host=getattr(self.config, "api_host", None) or "127.0.0.1",
            api_key_present=bool(
                _os_nice.environ.get("PRSM_NODE_API_KEY", "").strip()
            ),
        )
        if _posture == "insecure":
            logger.warning(_posture_msg)
            _allow_insecure = _os_nice.environ.get(
                "PRSM_ALLOW_INSECURE_PUBLIC_BIND", "",
            ).strip().lower() in {"1", "true", "yes", "on"}
            if should_refuse_insecure_public_bind(
                _posture, allow_insecure=_allow_insecure,
            ):
                raise RuntimeError(
                    "Refusing to start: insecure public-bind/no-auth posture — "
                    "the money + KYC endpoints would be UNAUTHENTICATED. Set "
                    "PRSM_NODE_API_KEY, bind 127.0.0.1 behind an authenticating "
                    "reverse proxy, or set PRSM_ALLOW_INSECURE_PUBLIC_BIND=1 to "
                    "run unauthenticated (NOT recommended). " + _posture_msg
                )

        # Sprint 595 (Phase 2D) — capture the running event loop +
        # initialize the chain-executor pending-requests dict. Used
        # by sprint-594's run_async_on_loop primitive to bridge the
        # sync SendMessage contract over async transport calls
        # (sprint-596+ wiring). Reversible: pure attribute init; if
        # nothing reads these, no behavior change.
        import asyncio as _asyncio
        try:
            self._loop = _asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        # Maps chain-executor request_id → Future awaiting response
        # bytes. Sprint-597 wires the message-handler to resolve.
        self._chain_executor_pending = {}

        # Sprint 686 — rebuild inference_executor now that _loop is
        # set. _build_chain_executor needs node._loop to wire the
        # async-to-sync bridge (sprint 595+); when build runs at
        # __init__ time _loop is None → chain_executor falls back to
        # the sprint-558 stub. Rebuilding here uses the freshly-set
        # _loop so the real RPC chain executor wires correctly.
        # Only re-runs when the operator opted into the parallax
        # kind; mock/None paths keep their __init__-time decision.
        if (
            os.environ.get("PRSM_INFERENCE_EXECUTOR", "")
            .strip().lower() == "parallax"
        ):
            try:
                from prsm.node.inference_wiring import (
                    build_parallax_executor_or_none,
                )
                rebuilt = build_parallax_executor_or_none(self)
                if rebuilt is not None:
                    self.inference_executor = rebuilt
                    logger.info(
                        "Sprint 686 — inference_executor rebuilt "
                        "after _loop assignment; real RPC chain "
                        "executor now wired (was sprint-558 stub "
                        "at __init__ time)."
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Sprint 686 — inference_executor rebuild after "
                    "_loop assignment raised: %s. Keeping "
                    "__init__-time executor.", exc,
                )

        await self.transport.start()

        # Sprint 599 (Phase 2D step 5) — register the chain-executor
        # response handler on the transport's MSG_DIRECT dispatch.
        # Co-exists with content_provider's existing MSG_DIRECT
        # handler (transport.on_message uses append, not replace).
        # Sprint 601 (Phase 2E-1) — also register the REQUEST-side
        # handler so this node can respond to incoming chain-executor
        # requests from peers (currently with a "not yet implemented"
        # error response; Phase 2E-2+ adds real stage execution).
        try:
            from prsm.node.transport import MSG_DIRECT
            from prsm.node.chain_executor_adapters import (
                handle_chain_executor_response,
                handle_chain_executor_request,
                handle_chain_stream_request,
                handle_chain_stream_response,
            )
            _self = self

            # Sprint 730 F63 — bind msg.sender_id to the
            # handshake-authenticated peer.peer_id before any
            # handler sees the message. Pre-730, the transport
            # only verified signatures at handshake time; any
            # subsequent MSG_DIRECT message arrived with a
            # `sender_id` field that was wire-trusted but not
            # cryptographically rebound. A peer that established
            # a valid handshake could then send messages claiming
            # to be a DIFFERENT peer — defeating sprint-726/723
            # per-peer caps (open many under fake ids) and sprint-
            # 727/719 sender-binding checks (forge responses by
            # claiming to be the victim's expected peer). Overwriting
            # sender_id with peer.peer_id at the dispatch boundary
            # protects ALL chain-executor handlers in one place.
            def _bind_sender(msg, peer):
                if peer is not None:
                    authentic = getattr(peer, "peer_id", None)
                    if authentic:
                        msg.sender_id = authentic

            async def _chain_executor_response_dispatch(msg, peer):
                _bind_sender(msg, peer)
                handle_chain_executor_response(_self, msg)

            async def _chain_executor_request_dispatch(msg, peer):
                _bind_sender(msg, peer)
                await handle_chain_executor_request(_self, msg)

            # Sprint 711 F40 — token-stream wire protocol dispatch.
            # Stream requests + frames + ends all ride MSG_DIRECT
            # alongside unary chain-exec messages. The two response
            # handlers' return-False fall-through lets each handler
            # ignore messages destined for the other.
            async def _chain_stream_request_dispatch(msg, peer):
                _bind_sender(msg, peer)
                await handle_chain_stream_request(_self, msg)

            async def _chain_stream_response_dispatch(msg, peer):
                _bind_sender(msg, peer)
                handle_chain_stream_response(_self, msg)

            self.transport.on_message(
                MSG_DIRECT, _chain_executor_response_dispatch,
            )
            self.transport.on_message(
                MSG_DIRECT, _chain_executor_request_dispatch,
            )
            self.transport.on_message(
                MSG_DIRECT, _chain_stream_request_dispatch,
            )
            self.transport.on_message(
                MSG_DIRECT, _chain_stream_response_dispatch,
            )
            logger.info(
                "Sprint 599 chain-executor response dispatch wired "
                "+ sprint 601 request-side handler wired on "
                "MSG_DIRECT — Phase 2 RPC client + server scaffolding "
                "now complete (server-side stage exec still Phase 2E-2+)."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Sprint 599/601 chain-executor wiring failed: "
                "%s. RPC chain-executor (sprint 598) will time out "
                "on dispatch. Operators on PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=rpc "
                "should fall back to stub until this lands.",
                exc,
            )
        await self.gossip.start()
        await self.discovery.start()
        # T3b: bring up the DHT listener + clients on their own
        # asyncio loop thread. Idempotent + non-fatal on failure.
        self._start_dht_components_if_present()

        # Initialize native content storage.
        # Sprint 168 — thread self.identity.node_id so manifests
        # carry correct owner attribution. Pre-fix init was called
        # with no args, defaulting node_id="" — every manifest
        # claimed ownership by empty string.
        try:
            from prsm.storage import init_content_store
            init_content_store(
                node_id=self.identity.node_id if self.identity else "",
            )
            logger.info(
                "ContentStore initialized (node_id=%s)",
                self.identity.node_id if self.identity else "<unset>",
            )
        except Exception as exc:
            logger.warning("ContentStore initialization failed: %s", exc)

        # Initialize on-chain FTNS ledger (best-effort)
        if self.ftns_ledger:
            ft_initialized = await self.ftns_ledger.initialize()
            if ft_initialized:
                logger.info("FTNS on-chain ledger connected to Base mainnet")
            else:
                logger.info("FTNS on-chain ledger unavailable — running local mode only")
                self.ftns_ledger = None

        # Initialize SQLAlchemy database for NWTN features (best-effort)
        if not self.db_initialized:
            try:
                from prsm.core.database import init_database
                await init_database()
                self.db_initialized = True
                logger.info("SQLAlchemy database tables initialized")
            except Exception as e:
                logger.warning(f"SQLAlchemy DB init failed: {e} — NWTN features unavailable")
                self.db_initialized = False

        # Seed welcome grant if the node has no balance
        await self._seed_welcome_grant()

        if self.compute_provider:
            await self.compute_provider.start()
            await self._start_marketplace_advertiser_if_present()
        await self.compute_requester.start()

        if self.storage_provider:
            await self.storage_provider.start()

        # ── Capability Announcement ──────────────────────────────────
        if hasattr(self.discovery, 'set_local_capabilities'):
            cap_list = list(self._local_capabilities)
            backends_list = []
            gpu_available = False
            if self.compute_provider:
                if self.compute_provider.resources.gpu_available:
                    gpu_available = True
                    if "gpu" not in cap_list:
                        cap_list.append("gpu")
                # NWTN backends subsystem removed in v1.6.0 — third-party LLMs
                # are now dispatched directly by MCP clients; no local backend
                # advertisement is required.
            self.discovery.set_local_capabilities(
                capabilities=cap_list,
                backends=backends_list,
                gpu_available=gpu_available,
            )
            await self.discovery.announce_capabilities()

            async def _periodic_capability_announce():
                while self._started:
                    await asyncio.sleep(300)
                    try:
                        await self.discovery.announce_capabilities()
                    except Exception as exc:
                        logger.debug("Capability re-announcement failed: %s", exc)

            self._capability_announce_task = asyncio.create_task(
                _periodic_capability_announce()
            )

        if self.content_index:
            self.content_index.start()
        if self.content_uploader:
            self.content_uploader.start()
            # sp1343 — periodic content RE-ADVERTISE so LATE-JOINING nodes can discover
            # already-hosted content. Gossip catch-up (digest) only spans the ~24h log
            # retention; without a re-advertise cadence, content goes dark after 24h and new
            # nodes can never learn it exists. Interval kept well under the retention window;
            # opt out with PRSM_CONTENT_READVERTISE_INTERVAL_S=0. Fail-soft.
            import os as _os
            try:
                _readv_interval = float(
                    _os.environ.get("PRSM_CONTENT_READVERTISE_INTERVAL_S", "3600"))
            except (TypeError, ValueError):
                _readv_interval = 3600.0
            if _readv_interval > 0:
                async def _periodic_content_readvertise():
                    while self._started:
                        await asyncio.sleep(_readv_interval)
                        try:
                            await self.content_uploader.readvertise_all()
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("content re-advertise failed: %s", exc)
                self._content_readvertise_task = asyncio.create_task(
                    _periodic_content_readvertise())
        if self.content_provider:
            self.content_provider.start()
        # Wire content_economy into content_provider for payment processing (Phase 4)
        if self.content_economy and self.content_provider:
            self.content_provider.content_economy = self.content_economy
        # Start content economy (Phase 4)
        if self.content_economy:
            await self.content_economy.start()
        # Start multi-party escrow (Phase 4)
        if hasattr(self, '_mp_escrow') and self._mp_escrow:
            self._mp_escrow.start()
        if self.ledger_sync:
            self.ledger_sync.start()
        if self._payment_escrow:
            self._escrow_cleanup_task = asyncio.create_task(self._payment_escrow.periodic_cleanup())
        # Sprint 506: periodic gas-status monitor. Logs on transitions
        # ok↔low↔critical so operators get continuous signal between
        # the startup log (sprint 504) and active polling. Interval
        # configurable via PRSM_GAS_MONITOR_INTERVAL_SECONDS.
        if (
            self.ftns_ledger is not None
            and getattr(self.ftns_ledger, "w3", None) is not None
            and self.ftns_ledger._connected_address is not None
        ):
            from prsm.economy.ftns_onchain import GasStatusMonitor
            try:
                _gas_interval = float(
                    os.environ.get(
                        "PRSM_GAS_MONITOR_INTERVAL_SECONDS",
                        "300",  # default: 5 min
                    )
                )
            except ValueError:
                _gas_interval = 300.0
            # Sprint 507: also fire webhook on transitions if
            # operator has set PRSM_WEBHOOK_URL (reuses the
            # deliverer the early-init created at __init__).
            self._gas_status_monitor = GasStatusMonitor(
                self.ftns_ledger,
                interval_seconds=_gas_interval,
                webhook_deliverer=getattr(
                    self, "_webhook_deliverer", None,
                ),
                webhook_url=os.environ.get(
                    "PRSM_WEBHOOK_URL", "",
                ).strip() or None,
                webhook_secret=os.environ.get(
                    "PRSM_WEBHOOK_SECRET", "",
                ).strip() or None,
            )
            self._gas_status_monitor_task = asyncio.create_task(
                self._gas_status_monitor.run_forever(),
            )
            logger.info(
                "GasStatusMonitor launched (interval=%.0fs)",
                _gas_interval,
            )
            # Sprint 514: inbound monitor — periodic Transfer
            # event scan for the operator wallet. Push signal
            # complementing sprint 512/513 pull surfaces.
            from prsm.economy.ftns_onchain import (
                InboundMonitor, InboundCheckpointStore,
            )
            try:
                _inbound_interval = float(
                    os.environ.get(
                        "PRSM_INBOUND_MONITOR_INTERVAL_SECONDS",
                        "60",
                    )
                )
            except ValueError:
                _inbound_interval = 60.0
            # Sprint 543: checkpoint store persists scan window
            # across restart so deposits arriving during downtime
            # get caught on next boot. Default path mirrors
            # sprint-501's onchain_tx.db sibling. PRSM_INBOUND_CHECKPOINT_DB
            # opt-out: set to ":memory:" or "" to disable.
            _ck_db = os.environ.get("PRSM_INBOUND_CHECKPOINT_DB")
            if _ck_db == "" or _ck_db == ":memory:":
                _checkpoint_store = None
            else:
                if _ck_db is None:
                    _ck_db = str(
                        Path.home() / ".prsm"
                        / "inbound_checkpoint.db"
                    )
                try:
                    _checkpoint_store = InboundCheckpointStore(_ck_db)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Sprint 543: checkpoint store init failed "
                        "(%s); monitor falling back to in-memory "
                        "checkpoint (deposits during downtime will "
                        "be missed).", exc,
                    )
                    _checkpoint_store = None
            self._inbound_monitor = InboundMonitor(
                self.ftns_ledger,
                interval_seconds=_inbound_interval,
                webhook_deliverer=getattr(
                    self, "_webhook_deliverer", None,
                ),
                webhook_url=os.environ.get(
                    "PRSM_WEBHOOK_URL", "",
                ).strip() or None,
                webhook_secret=os.environ.get(
                    "PRSM_WEBHOOK_SECRET", "",
                ).strip() or None,
                # Sprint 540 Pattern A: pass local_ledger so detected
                # inbound transfers from linked addresses
                # automatically credit off-chain balances. Bridge
                # deposit flow operates as a daemon-side hook on the
                # existing InboundMonitor — no new contract needed.
                local_ledger=getattr(self, "ledger", None),
                # Sprint 543: persistent checkpoint store.
                checkpoint_store=_checkpoint_store,
            )
            self._inbound_monitor_task = asyncio.create_task(
                self._inbound_monitor.run_forever(),
            )
            logger.info(
                "InboundMonitor launched (interval=%.0fs)",
                _inbound_interval,
            )
        # sp916 — pending-withdraw reconciler. The /wallet/withdraw handler
        # records each broadcast-but-unconfirmed withdraw into the store; this
        # worker polls the tx receipt and, on a REVERT, refunds the off-chain
        # debit idempotently (the debit was taken BEFORE broadcast — sp914 —
        # so a revert otherwise loses the user's FTNS with no recovery). Cheap
        # when idle (no-ops on an empty store). Enabled by default; disable via
        # PRSM_PENDING_WITHDRAW_RECONCILER_ENABLED=0.
        if self.ftns_ledger is not None and getattr(self, "ledger", None) is not None:
            from prsm.node.pending_withdraw_reconciler import (
                PendingWithdrawStore,
                PendingWithdrawReconciler,
                resolve_pending_withdraw_reconciler_config_from_env,
            )
            _pw_enabled, _pw_interval = (
                resolve_pending_withdraw_reconciler_config_from_env()
            )
            _pw_dir = os.environ.get("PRSM_PENDING_WITHDRAW_DIR")
            if _pw_dir in ("", ":memory:"):
                _pw_dir = None
            elif _pw_dir is None:
                _pw_dir = str(Path.home() / ".prsm")
            try:
                self._pending_withdraw_store = PendingWithdrawStore(
                    persist_dir=_pw_dir,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PendingWithdrawStore init failed (%s); using in-memory",
                    exc,
                )
                self._pending_withdraw_store = PendingWithdrawStore(
                    persist_dir=None,
                )
            if _pw_enabled:
                self._pending_withdraw_reconciler = PendingWithdrawReconciler(
                    store=self._pending_withdraw_store,
                    ftns_ledger=self.ftns_ledger,
                    local_ledger=self.ledger,
                    interval_seconds=_pw_interval,
                )
                self._pending_withdraw_reconciler_task = asyncio.create_task(
                    self._pending_withdraw_reconciler.run_forever(),
                )
                logger.info(
                    "PendingWithdrawReconciler launched (interval=%.0fs)",
                    _pw_interval,
                )
        if hasattr(self, '_batch_settlement') and self._batch_settlement:
            # Update connected_address now that ftns_ledger may have initialized
            if self.ftns_ledger and hasattr(self.ftns_ledger, '_connected_address'):
                self._batch_settlement._connected_address = self.ftns_ledger._connected_address
            self._batch_settlement.start()
        if self.agent_registry:
            self.agent_registry.start()
        if self.agent_collaboration:
            self.agent_collaboration.start()
            await self.agent_collaboration.load_state()

        # Start BitTorrent components
        if self.bt_provider:
            await self.bt_provider.start()
        if self.bt_requester:
            await self.bt_requester.start()

        # Phase 7-storage + Phase 8 daemons (2026-05-08). Launched only
        # if the operator opted into both the client AND the scheduler
        # via env vars (see _build_*_scheduler_or_none). Each survives
        # any one tick failure; restart-resilience is in the daemon.
        if self._heartbeat_scheduler is not None:
            self._heartbeat_scheduler_task = asyncio.create_task(
                self._heartbeat_scheduler.run_forever(),
            )
            logger.info("HeartbeatScheduler launched")
        if self._compensation_scheduler is not None:
            self._compensation_scheduler_task = asyncio.create_task(
                self._compensation_scheduler.run_forever(),
            )
            logger.info("PullAndDistributeScheduler launched")
        # JobReaper for per-job duration cap.
        if getattr(self, "_job_reaper", None) is not None:
            self._job_reaper_task = asyncio.create_task(
                self._job_reaper.run_forever(),
            )
            logger.info("JobReaper launched")
        # DaemonWatchdog (active-push of daemon-crash events).
        if getattr(self, "_daemon_watchdog", None) is not None:
            self._daemon_watchdog_task = asyncio.create_task(
                self._daemon_watchdog.watch(),
            )
            logger.info("DaemonWatchdog launched")

        # Phase 7-storage + Phase 8 event watchers. Same opt-in shape
        # as the schedulers; default-callback wired in the builders so
        # the watcher polls and logs events out-of-the-box.
        if self._key_distribution_watcher is not None:
            self._key_distribution_watcher_task = asyncio.create_task(
                self._key_distribution_watcher.run_forever(),
            )
            logger.info("KeyDistributionWatcher launched")
        if self._storage_slashing_watcher is not None:
            self._storage_slashing_watcher_task = asyncio.create_task(
                self._storage_slashing_watcher.run_forever(),
            )
            logger.info("StorageSlashingWatcher launched")
        if self._settlement_challenge_watcher is not None:
            self._settlement_challenge_watcher_task = asyncio.create_task(
                self._settlement_challenge_watcher.run_forever(),
            )
            logger.info("SettlementChallengeWatcher launched")
        if self._compensation_distributor_watcher is not None:
            self._compensation_distributor_watcher_task = asyncio.create_task(
                self._compensation_distributor_watcher.run_forever(),
            )
            logger.info("CompensationDistributorWatcher launched")

        # Sprint 1038 (brick 2) — on-chain settlement commit/finalize/reconcile
        # poll loop. Only when settlement is opted in; inert (commit/finalize
        # raise) until a funded settler key is provisioned.
        if getattr(self, "_onchain_settlement_client", None) is not None:
            self._settlement_poll_task = asyncio.create_task(
                self._settlement_poll_loop(),
            )
            logger.info(
                "Sprint 1038 settlement poll loop launched (commit/finalize "
                "inert until a funded settler key is provisioned)."
            )

        # Sprint 1140 (Brick E.2) — OPT-IN settlement-receipt audit loop. Gated by
        # PRSM_SETTLEMENT_AUDIT (default OFF). Builds the audit bundle (ad-index +
        # composite selector + verified-cache + dry-run-gated engine + observe loop +
        # announcer + block-cursor scheduler) and launches the scheduler. Read-only +
        # NEVER broadcasts (the engine only dry-runs); fail-closed (a tick never crashes
        # the daemon). OFF => none of this runs, no task, no gossip subscription.
        try:
            from prsm.settlement.settlement_audit_wiring import audit_enabled
            if (
                audit_enabled()
                and getattr(self, "_onchain_settlement_client", None) is not None
            ):
                self._launch_settlement_audit_loop()
        except Exception as _audit_launch_exc:  # noqa: BLE001 — never crash startup
            logger.warning(
                "Sprint 1140 settlement audit loop launch failed (audit OFF): %s",
                _audit_launch_exc,
            )

        # sp1424 — REPUTATION SLASH WATCH. The provider ReputationTracker is READ on every
        # marketplace dispatch but was WRITTEN never on the live path, so every provider scored a
        # constant 0.5 forever and a provider slashed on-chain for fraud kept full selection
        # weight. This loop bridges the real StakeBond.Slashed events into the tracker. Launched
        # only when reputation-based selection is actually in play (the tracker exists, i.e. QO is
        # enabled) and a stake client + discovery are available to read/resolve. Read-only on
        # chain (never signs/broadcasts); fail-closed.
        try:
            if getattr(self, "reputation_tracker", None) is not None:
                self._reputation_slash_watch_task = asyncio.create_task(
                    self._reputation_slash_watch_loop(),
                )
        except Exception as _slash_launch_exc:  # noqa: BLE001 — never crash startup
            logger.warning(
                "sp1424 reputation slash-watch launch failed (OFF): %s", _slash_launch_exc,
            )

        # Sprint 1081 — attestation collateral (Intel PCK CRL + QE-Identity) auto
        # refresh. Opt-in via PRSM_COLLATERAL_AUTO_REFRESH so a long-running node keeps
        # revocation enforcement current instead of silently lapsing when a static CRL
        # expires. Off by default (no outbound calls unless asked).
        import os as _os_cr
        if str(_os_cr.environ.get("PRSM_COLLATERAL_AUTO_REFRESH", "")).strip().lower() in (
                "1", "true", "yes", "on"):
            self._collateral_refresh_task = asyncio.create_task(
                self._collateral_refresh_loop(),
            )
            logger.info("Sprint 1081 attestation collateral auto-refresh loop launched.")

        # Sprint 1217 — OPT-IN node runtime observability (default OFF). A
        # MetricsCollector populated by NodeRuntimeMetrics (readiness +
        # on-chain settlement funds-in-flight) and, when also enabled, an
        # AlertManager evaluating node-runtime rules over it. Both collectors
        # own their internal async tasks (started here, drained in stop()), so
        # we instantiate+start rather than create_task ourselves. Fully fenced:
        # any failure leaves observability OFF and never crashes startup. With
        # both flags unset this block is a single falsy `if` — startup is
        # byte-for-byte unchanged.
        import os as _os_obs

        def _obs_on(_name: str) -> bool:
            return str(_os_obs.environ.get(_name, "")).strip().lower() in (
                "1", "true", "yes", "on",
            )

        def _obs_interval(_name: str, _default: float) -> float:
            try:
                v = float(str(_os_obs.environ.get(_name, "")).strip())
                return v if v > 0 else _default
            except (TypeError, ValueError):
                return _default

        if _obs_on("PRSM_RUNTIME_METRICS_ENABLED"):
            try:
                from prsm.core.monitoring.metrics import MetricsCollector
                from prsm.core.monitoring.node_runtime_metrics import (
                    NodeRuntimeMetrics,
                )
                _m_interval = _obs_interval("PRSM_RUNTIME_METRICS_INTERVAL_S", 30.0)
                self._metrics_collector = MetricsCollector(
                    collection_interval=_m_interval,
                )
                self._metrics_collector.registry.register_metric(
                    NodeRuntimeMetrics(self),
                )
                await self._metrics_collector.start_collection()
                logger.info(
                    "Sprint 1217 runtime MetricsCollector started "
                    "(interval=%ss).", _m_interval,
                )
                if _obs_on("PRSM_RUNTIME_ALERTS_ENABLED"):
                    from prsm.core.monitoring.alerts import AlertManager
                    _a_interval = _obs_interval(
                        "PRSM_RUNTIME_ALERTS_INTERVAL_S", 30.0,
                    )
                    self._alert_manager = AlertManager(
                        evaluation_interval=_a_interval,
                    )
                    self._alert_manager.set_metrics_collector(
                        self._metrics_collector,
                    )
                    # sp1218 — opt-in webhook delivery. PRSM_ALERT_WEBHOOK_URL
                    # set → register a WebhookAlertChannel + route the
                    # node-runtime rules to it (Slack/PagerDuty/etc. accept a
                    # generic JSON webhook). Unset → channels=[] (alerts still
                    # register in active_alerts + log; just no push).
                    _alert_channels: list = []
                    _webhook_url = str(
                        _os_obs.environ.get("PRSM_ALERT_WEBHOOK_URL", ""),
                    ).strip()
                    if _webhook_url:
                        from prsm.core.monitoring.alerts import (
                            WebhookAlertChannel,
                        )
                        self._alert_manager.add_channel(
                            WebhookAlertChannel(
                                "node-webhook", {"url": _webhook_url},
                            ),
                        )
                        _alert_channels.append("node-webhook")
                    self._alert_manager.setup_node_runtime_rules(
                        channels=_alert_channels,
                    )
                    await self._alert_manager.start_monitoring()
                    logger.info(
                        "Sprint 1217 runtime AlertManager started "
                        "(node-runtime rules, interval=%ss).", _a_interval,
                    )
            except Exception as _obs_exc:  # noqa: BLE001 — never crash startup
                logger.warning(
                    "Sprint 1217 runtime observability launch failed "
                    "(observability OFF): %s", _obs_exc,
                )
                self._metrics_collector = None
                self._alert_manager = None

        # Start management API in background
        self._api_task = asyncio.create_task(self._run_api())

        # Sprint 766 — wire AutoClaimWorker into the daemon
        # lifecycle. Only constructed when staking_manager +
        # identity are both present (defensive — staking is
        # optional in some test configs). Worker reads env at
        # construction; .start() short-circuits when disabled
        # (threshold = 0), so it's safe to always invoke.
        self._auto_claim_worker = None
        if (
            self.staking_manager is not None
            and self.identity is not None
        ):
            try:
                from prsm.node.auto_claim import AutoClaimWorker
                self._auto_claim_worker = AutoClaimWorker(
                    staking_manager=self.staking_manager,
                    user_id=self.identity.node_id,
                )
                await self._auto_claim_worker.start()
                if self._auto_claim_worker.config.enabled:
                    logger.info(
                        "Sprint 766 — AutoClaimWorker started: "
                        "threshold=%s FTNS, interval=%ss",
                        self._auto_claim_worker.config.threshold_ftns,
                        self._auto_claim_worker.config.interval_seconds,
                    )
            except Exception as exc:
                logger.warning(
                    "AutoClaimWorker construction failed (auto-"
                    "claim disabled this session): %s", exc,
                )

        # Sprint 878 — wire FunnelAutoSweepWorker into the daemon
        # lifecycle. Periodically sweeps the onramp conversion
        # funnel (sp857) with the same on_confirmed callback the
        # manual /wallet/onramp/sweep endpoint uses — sp871
        # envelope build + sp874 outbound completion notify. Opt-in
        # via PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S; .start()
        # short-circuits when unset (interval=0), so safe to always
        # invoke.
        self._funnel_auto_sweep_worker = None
        try:
            from prsm.node.funnel_auto_sweep import (
                FunnelAutoSweepWorker,
            )

            def _do_sweep():
                # Build the same callback chain the manual sweep
                # endpoint uses. Imports deferred so unrelated
                # daemon configs don't pay the cost.
                from prsm.economy.web3.onramp_funnel import (
                    OnrampFunnel,
                )
                from prsm.economy.web3.wallet_balance_reader import (
                    from_env as _wbr_from_env,
                )
                from prsm.economy.web3.onramp_to_swap_orchestrator import (  # noqa: E501
                    make_on_confirmed_callback,
                    make_on_expired_callback,
                )
                from prsm.economy.web3.onramp_completion_notifier import (  # noqa: E501
                    from_env as _notifier_from_env,
                )
                from prsm.config.networks import get_network_config

                funnel = getattr(self, "_onramp_funnel", None)
                if funnel is None:
                    funnel = OnrampFunnel()
                    self._onramp_funnel = funnel
                reader = _wbr_from_env()
                aero = getattr(self, "_aerodrome_client", None)
                net = get_network_config("mainnet")
                notifier = _notifier_from_env()
                # Sp885 — always build the callback (NOT gated on
                # aero): envelope build is None-safe when Aerodrome
                # is unwired, but the compliance-ring record (tier
                # limit) + completion webhook MUST fire on every
                # CONFIRMED regardless of pool status.
                on_confirmed = make_on_confirmed_callback(
                    funnel=funnel,
                    aerodrome_client=aero,
                    ftns_address=net.ftns_token or "",
                    completion_notifier=notifier,
                    compliance_ring=getattr(
                        self, "_fiat_compliance_ring", None,
                    ),
                )
                # Sp1176 — terminal EXPIRED $0 compliance entry on an
                # abandoned intent (AML rolling-total correctness +
                # complete audit trail); same callback the manual
                # /wallet/onramp/sweep endpoint wires.
                on_expired = make_on_expired_callback(
                    compliance_ring=getattr(
                        self, "_fiat_compliance_ring", None,
                    ),
                )
                try:
                    return funnel.sweep(
                        balance_reader=reader,
                        on_confirmed=on_confirmed,
                        on_expired=on_expired,
                    )
                finally:
                    reader.close()
                    notifier.close()

            self._funnel_auto_sweep_worker = FunnelAutoSweepWorker(
                sweep_fn=_do_sweep,
            )
            await self._funnel_auto_sweep_worker.start()
            if self._funnel_auto_sweep_worker.config.enabled:
                logger.info(
                    "Sprint 878 — FunnelAutoSweepWorker started: "
                    "interval=%ss",
                    self._funnel_auto_sweep_worker
                    .config.interval_seconds,
                )
        except Exception as exc:
            logger.warning(
                "FunnelAutoSweepWorker construction failed (auto-"
                "sweep disabled this session): %s", exc,
            )

        # Sprint 775 — wire PreemptionDetector into the daemon
        # lifecycle. resolve_detector_from_env() returns None when
        # PRSM_PREEMPTION_DETECTOR is unset (safe default for non-
        # cloud nodes — no metadata polling, no behavior change).
        # When set, the detector is started + globally registered
        # so sprint-773's announce gate + sprint-774's dispatch
        # gate read a live flag. MUST happen BEFORE _started=True
        # so an inbound request at the boundary sees a queryable
        # flag (not a None-detector race).
        self._preemption_detector = None
        try:
            from prsm.node.preemption import (
                resolve_detector_from_env,
                register_detector,
            )
            det = resolve_detector_from_env()
            if det is not None:
                det.start()
                register_detector(det)
                self._preemption_detector = det
                logger.info(
                    "Sprint 775 — PreemptionDetector started "
                    "(backend=%s, interval=%ss)",
                    det.backend.__class__.__name__,
                    det.poll_interval_s,
                )
        except Exception as exc:
            logger.warning(
                "PreemptionDetector wire-up failed (preemption "
                "awareness disabled this session): %s", exc,
            )

        # Sprint 793 — register WalletApiServices with a
        # production ReceiptStore adapter so the
        # /devices/earnings endpoint (sprint 792) actually
        # returns per-device earnings instead of falling back
        # to the no-op default. Idempotent + fail-soft inside
        # the helper.
        try:
            from prsm.node.wallet_api_wiring import (
                wire_wallet_api_services,
            )
            wire_wallet_api_services(self)
        except Exception as exc:
            logger.warning(
                "wallet_api production wire-up failed "
                "(devices/earnings endpoint will return empty): "
                "%s", exc,
            )

        # Sprint 799 — construct the partial-completion event
        # ring so sprint-784/785 settle-path appends + sprint-799
        # /admin/partial-completion-events endpoint both see a
        # live instance. Bounded in-memory (256 entries default).
        try:
            from prsm.node.partial_completion_event_log import (
                PartialCompletionEventRing,
            )
            self._partial_completion_event_log = (
                PartialCompletionEventRing()
            )
        except Exception as exc:
            logger.warning(
                "PartialCompletionEventRing construction "
                "failed (/admin/partial-completion-events will "
                "return 503): %s", exc,
            )
            self._partial_completion_event_log = None

        self._started = True
        self._start_time = time.time()
        bootstrap_status = self.discovery.get_bootstrap_status() if self.discovery else {}
        if bootstrap_status.get("degraded_mode"):
            logger.warning(
                "Node startup in DEGRADED local mode: no bootstrap peers reachable. "
                "Limited features: remote peer discovery and cross-node collaboration may be unavailable "
                "until peers connect or bootstrap targets recover."
            )
            # Try local mDNS discovery as fallback
            if hasattr(self, 'mdns_discovery') and self.mdns_discovery:
                self.mdns_discovery.start()
                logger.info("Started mDNS local discovery as bootstrap fallback")
        elif bootstrap_status.get("success_node"):
            logger.info(
                "Node startup bootstrap path: connected via %s",
                bootstrap_status.get("success_node"),
            )

        # Emit bootstrap decision telemetry (additive, best-effort)
        if self.discovery:
            bt = self.discovery.get_bootstrap_telemetry()
            if bt.get("fallback_activated"):
                logger.info(
                    "Bootstrap decision: fallback activated, "
                    "fallback_attempted=%d, fallback_succeeded=%s, "
                    "addresses_rejected=%d, source_policy=%s",
                    bt.get("fallback_attempted", 0),
                    bt.get("fallback_succeeded", False),
                    bt.get("addresses_rejected", 0),
                    bt.get("source_policy", "unknown"),
                )
            elif bt.get("addresses_rejected", 0) > 0:
                logger.warning(
                    "Bootstrap decision: %d address(es) rejected during validation",
                    bt["addresses_rejected"],
                )

        logger.info(
            f"PRSM node started — "
            f"P2P: ws://{self.config.listen_host}:{self.config.p2p_port}, "
            f"API: http://127.0.0.1:{self.config.api_port}, "
            f"Dashboard: http://127.0.0.1:{self.config.api_port}/"
        )
        logger.info(
            "Node onboarding UI available: http://127.0.0.1:%s/onboarding/",
            self.config.api_port,
        )

    def _build_query_orchestrator_or_none(self):
        """Construct QueryOrchestrator from this node's primitives, or
        return None if the operator hasn't opted in via
        `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` OR if any required adapter
        cannot be constructed against current node state.

        Default-disabled. Behavior identical to v1.6.0
        (`agent_forge = None`) until the operator explicitly enables.

        On any wiring failure with the env var set, logs the reason
        and falls back to None — the operator gets a clear signal
        that their deployment is missing something + the canonical
        workflow stays gated rather than half-broken.

        See `prsm/compute/query_orchestrator/node_wiring.py` for the
        factory contract + `docs/2026-05-08-query-orchestrator-wiring-readiness.md`
        for the wiring program.
        """
        from prsm.compute.query_orchestrator.node_wiring import (
            is_query_orchestrator_enabled,
        )
        # Sprint 173 — diagnostic state for /info. Reset on each
        # call so a stale value never lingers across re-wiring.
        self._query_orchestrator_state = "disabled"
        self._query_orchestrator_error = None
        if not is_query_orchestrator_enabled():
            return None
        self._query_orchestrator_state = "enabled_constructing"

        try:
            from prsm.compute.query_orchestrator import (
                FoundationBeaconProvider,
                MarketplaceCandidatePoolProvider,
                SemanticIndexAdapter,
                SentenceTransformerEmbedder,
                SwarmDispatcherAdapter,
            )
            from prsm.compute.query_orchestrator.node_wiring import (
                build_query_orchestrator_for_node,
            )
            from prsm.marketplace.directory import MarketplaceDirectory
            from prsm.marketplace.reputation import ReputationTracker
        except ImportError as exc:
            logger.warning(
                "QueryOrchestrator wiring unavailable: %s — falling back "
                "to agent_forge=None",
                exc,
            )
            return None

        # All 5 adapter dependencies required. Each construction step
        # raises clearly if a node-side primitive is missing.
        try:
            if self.content_uploader is None:
                raise RuntimeError(
                    "content_uploader not initialized — cannot wire "
                    "SemanticIndexAdapter"
                )
            if self.agent_dispatcher is None:
                raise RuntimeError(
                    "agent_dispatcher not initialized — cannot wire "
                    "SwarmDispatcherAdapter"
                )
            if self.gossip is None:
                raise RuntimeError(
                    "gossip not initialized — cannot wire "
                    "MarketplaceDirectory"
                )

            # Marketplace + reputation primitives are constructed here
            # because node.py doesn't currently own them. Once the
            # marketplace orchestrator becomes a top-level node
            # subsystem (separate sprint), pull these from self.* instead.
            marketplace_directory = MarketplaceDirectory(self.gossip)
            reputation_tracker = ReputationTracker()
            # Sprint 275 — expose tracker on Node so operator
            # endpoints (/marketplace/reputation/*) can read it.
            # Lifetime is tied to the QO instance.
            self.reputation_tracker = reputation_tracker
            # Sprint 287 — creator-side reputation tracker
            # (Vision §14 data quality / Sybil resistance).
            # Built alongside the provider-side tracker; same
            # lifetime semantics.
            try:
                from prsm.marketplace.creator_reputation import (
                    CreatorReputationTracker,
                )
                self._creator_reputation_tracker = (
                    CreatorReputationTracker()
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CreatorReputationTracker construction "
                    "failed: %s — /marketplace/creator-"
                    "reputation/* will return 503.",
                    exc,
                )
                self._creator_reputation_tracker = None
            # Sprint 290 — creator stake client (Vision §14
            # item 2). PENDING_COMMISSION pattern: in-memory
            # mirror when no contract address configured;
            # real contract delegation post-deploy.
            try:
                from prsm.marketplace.creator_stake_client import (
                    CreatorStakeClient,
                )
                self._creator_stake_client = (
                    CreatorStakeClient.from_env()
                )
                logger.info(
                    "CreatorStakeClient wired "
                    "(commissioned=%s)",
                    self._creator_stake_client.is_commissioned(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CreatorStakeClient construction "
                    "failed: %s — /marketplace/creator-"
                    "stake/* will return 503.",
                    exc,
                )
                self._creator_stake_client = None
            # Sprint 291 — content fingerprint registry
            # (Vision §14 item 3). First-creator-wins dedup
            # for content_hash. Opt-in disk persistence via
            # PRSM_FINGERPRINT_REGISTRY_DIR.
            try:
                from prsm.marketplace.content_fingerprint_registry import (  # noqa: E501
                    ContentFingerprintRegistry,
                )
                self._content_fingerprint_registry = (
                    ContentFingerprintRegistry.from_env()
                )
                logger.info(
                    "ContentFingerprintRegistry wired "
                    "(persist_dir=%s)",
                    getattr(
                        self._content_fingerprint_registry,
                        "_persist_dir", None,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ContentFingerprintRegistry "
                    "construction failed: %s — fingerprint "
                    "dedup will not be enforced.",
                    exc,
                )
                self._content_fingerprint_registry = None

            semantic_index = SemanticIndexAdapter(
                embedder=SentenceTransformerEmbedder(),
                index=self.content_uploader._semantic_index,
            )
            # Per-shard FTNS budget default. The orchestrator's
            # retry-loop owns the actual per-call budget at request
            # time; this constructor default applies only to call
            # sites that don't override. Operators tune via env
            # without forking.
            try:
                _per_shard_default = int(
                    os.environ.get(
                        "PRSM_PER_SHARD_FTNS_DEFAULT", "100",
                    )
                )
                if _per_shard_default <= 0:
                    raise ValueError("non-positive")
            except (ValueError, TypeError):
                logger.warning(
                    "PRSM_PER_SHARD_FTNS_DEFAULT not a positive int; "
                    "using 100"
                )
                _per_shard_default = 100
            # Sprint 173 + 174 — load the WASM executor binary.
            # Priority:
            #   1. PRSM_WASM_EXECUTOR_PATH env (operator-supplied custom
            #      executor — required for production query execution)
            #   2. Bundled minimal stub at
            #      `prsm/compute/wasm/binaries/minimal_executor.wasm`
            #      (smoke-test only — exports `run` → i32(42), does NOT
            #      interpret InstructionManifest)
            #
            # Sprint 174 added the bundled fallback so a fresh node can
            # construct SwarmDispatcherAdapter for dispatch-pipeline
            # validation without first building a real executor.
            # Production deployments MUST supply a real binary via the
            # env var; the stub returns a fixed value for any input.
            _wasm_path = os.environ.get(
                "PRSM_WASM_EXECUTOR_PATH", "",
            ).strip()
            if _wasm_path:
                try:
                    with open(_wasm_path, "rb") as f:
                        _wasm_binary = f.read()
                except OSError as exc:
                    raise RuntimeError(
                        f"PRSM_WASM_EXECUTOR_PATH points at unreadable "
                        f"file: {_wasm_path!r}: {exc}"
                    )
                if not _wasm_binary:
                    raise RuntimeError(
                        f"PRSM_WASM_EXECUTOR_PATH binary is empty: "
                        f"{_wasm_path!r}"
                    )
                logger.info(
                    "QueryOrchestrator: using operator-supplied WASM "
                    "executor from %s (%d bytes)",
                    _wasm_path, len(_wasm_binary),
                )
            else:
                # Sprint 177 — bundled real executor (~188 KB Rust
                # binary, compiled to wasm32-wasip1). Interprets all
                # 11 AgentOps against CSV / JSON / JSONL data via
                # WASI stdin/stdout ABI. Operator can still override
                # via PRSM_WASM_EXECUTOR_PATH for custom workflows.
                from prsm.compute.wasm.binaries import (
                    load_bundled_executor,
                )
                _wasm_binary = load_bundled_executor()
                logger.info(
                    "QueryOrchestrator: using bundled prsm_executor.wasm "
                    "(%d bytes) — real instruction interpreter. Override "
                    "with PRSM_WASM_EXECUTOR_PATH if you need a custom "
                    "build.",
                    len(_wasm_binary),
                )
            dispatcher = SwarmDispatcherAdapter(
                agent_dispatcher=self.agent_dispatcher,
                wasm_executor_binary=_wasm_binary,
                per_shard_budget_ftns=_per_shard_default,
            )
            # AggregatorClient + beacon need a Foundation Safe address
            # that this deployment trusts. Default to mainnet Safe;
            # operators on other networks override via constructor
            # extension (separate ratification + tooling sprint).
            from prsm.compute.query_orchestrator import (
                AggregatorClientAdapter,
                ChainedEndpointResolver,
                HttpAggregateTransport,
                StaticMapEndpointResolver,
                TransportPeerEndpointResolver,
            )
            from prsm.compute.query_orchestrator.foundation_safe_resolver import (
                resolve_foundation_safe_address,
            )
            beacon_provider = FoundationBeaconProvider(
                foundation_safe_address=resolve_foundation_safe_address(),
            )
            # Endpoint resolver: ordered list of backends. Operators
            # supply a static map via PRSM_AGGREGATOR_ENDPOINT_MAP
            # (JSON of {node_id: base_url}) for known aggregators;
            # unknown node_ids fall back to the WS transport peer
            # registry (host:port from the live connection).
            import json as _json_for_endpoints
            import os as _os_for_endpoints
            _static_map_raw = _os_for_endpoints.environ.get(
                "PRSM_AGGREGATOR_ENDPOINT_MAP", "",
            ).strip()
            _static_map = {}
            if _static_map_raw:
                try:
                    _static_map = _json_for_endpoints.loads(_static_map_raw)
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "PRSM_AGGREGATOR_ENDPOINT_MAP malformed JSON: %s — "
                        "ignoring static map, using transport-peer fallback only",
                        exc,
                    )
            _endpoint_resolver = ChainedEndpointResolver([
                StaticMapEndpointResolver(_static_map),
                TransportPeerEndpointResolver(self.transport),
            ])
            aggregator_client = AggregatorClientAdapter(
                prompter_pubkey=self.identity.public_key_bytes,
                prompter_node_id=self.identity.node_id,
                prompter_signer=self.identity.sign,
                prompter_privkey=self.identity.private_key_bytes,
                beacon_provider=beacon_provider,
                transport=HttpAggregateTransport(
                    endpoint_resolver=_endpoint_resolver,
                ),
            )
            # sp1457 — weight aggregator selection by REAL on-chain stake when a listing carries a
            # verified stake binding (else the self-declared stake_tier, or exclusion under
            # enforcement). Reuses the node's cached OnChainStakeReader. require_stake_binding gates
            # exclusion of unverifiable-stake providers (opt-in via PRSM_REQUIRE_STAKE_BINDING) so
            # operators enable it once their providers advertise bindings.
            from prsm.compute.query_orchestrator.marketplace_candidate_pool_provider import (
                make_ftns_stake_reader as _make_ftns_stake_reader,
            )
            _onchain_reader = getattr(self, "_compute_stake_reader", None)
            _pool_stake_reader = (
                _make_ftns_stake_reader(_onchain_reader)
                if _onchain_reader is not None else None
            )
            _require_binding = os.environ.get(
                "PRSM_REQUIRE_STAKE_BINDING", "").strip().lower() in ("1", "true", "yes")
            candidate_pool_provider = MarketplaceCandidatePoolProvider(
                directory=marketplace_directory,
                reputation=reputation_tracker,
                stake_reader=_pool_stake_reader,
                require_stake_binding=_require_binding,
            )

            orchestrator = build_query_orchestrator_for_node(
                semantic_index=semantic_index,
                dispatcher=dispatcher,
                aggregator_client=aggregator_client,
                candidate_pool_provider=candidate_pool_provider,
                beacon_provider=beacon_provider,
            )
            logger.info(
                "QueryOrchestrator wired (env-enabled). agent_forge live."
            )
            self._query_orchestrator_state = "wired"
            return orchestrator
        except Exception as exc:  # noqa: BLE001
            # Sprint 173 — log the FULL traceback so the operator
            # can see the missing primitive immediately. Pre-fix
            # the exception was logged as a single line which
            # truncated useful chained-error context (e.g. which
            # AttributeError on which sub-object).
            import traceback as _tb
            logger.warning(
                "QueryOrchestrator construction failed: %s — falling "
                "back to agent_forge=None. (Operator must wire missing "
                "primitive before re-enabling.)\nTraceback:\n%s",
                exc,
                _tb.format_exc(),
            )
            # Surface the failure reason on the node so /info can
            # show it without requiring log-file scraping.
            self._query_orchestrator_state = "construction_failed"
            self._query_orchestrator_error = (
                f"{type(exc).__name__}: {exc}"
            )
            return None

    async def _settlement_poll_loop(self) -> None:
        """Sprint 1038 (brick 2) — periodically drive the on-chain settlement
        commit/finalize/reconcile cycle. Runs only when settlement is opted in
        (self._onchain_settlement_client is not None); inert (commit/finalize
        raise 'private_key required') until a funded settler key is provisioned.
        Each cycle is isolated + never raises; the loop survives iteration errors
        and exits cleanly on cancellation."""
        import os as _os
        from prsm.settlement.client_wiring import (
            run_per_stage_commit_cycle,
            run_per_stage_finalize_cycle,
            run_settlement_poll_cycle,
        )
        try:
            interval = float(
                _os.environ.get("PRSM_SETTLEMENT_POLL_INTERVAL_S", "600"),
            )
        except (TypeError, ValueError):
            interval = 600.0
        interval = max(5.0, interval)
        while True:
            try:
                results = await run_settlement_poll_cycle(
                    self._onchain_settlement_client,
                )
                logger.debug("Sprint 1038 settlement poll cycle: %s", results)
                # sp1322 (S3b-3b) — per-stage commit cycle: drain the receiver store + commit
                # each staged share on this node's own client. GATED (PRSM_MULTISTAGE_SETTLEMENT,
                # default-off → skipped:disabled) + fail-soft; inert on a view-only client.
                try:
                    ps_results = await run_per_stage_commit_cycle(self)
                    logger.debug("Sprint 1322 per-stage commit cycle: %s", ps_results)
                except Exception as _ps_exc:  # noqa: BLE001 — never break the loop
                    logger.debug("sp1322 per-stage commit cycle skipped (%s)", _ps_exc)
                # sp1447 — per-stage FINALIZE cycle: finalize this node's own committed share-batches
                # whose challenge window elapsed (releasing escrow to the payee). Without it a
                # self-committed share stays PENDING forever. Same gate/fail-soft as the commit cycle.
                try:
                    psf_results = await run_per_stage_finalize_cycle(self)
                    logger.debug("Sprint 1447 per-stage finalize cycle: %s", psf_results)
                except Exception as _psf_exc:  # noqa: BLE001 — never break the loop
                    logger.debug("sp1447 per-stage finalize cycle skipped (%s)", _psf_exc)
                # sp1140 (Brick E.2) — best-effort announce-after-commit. Only when
                # the audit data plane is opted in AND an announcer exists; never
                # raises, never broadcasts (gossips an UNTRUSTED availability ad only).
                bundle = getattr(self, "_settlement_audit_bundle", None)
                if bundle is not None and getattr(bundle, "announcer", None) is not None:
                    try:
                        from prsm.settlement.settlement_audit_wiring import (
                            announce_committed_batches,
                        )
                        store = getattr(
                            self, "_settlement_published_batch_store", None,
                        )
                        retained = store.all_batches() if store is not None else []
                        await announce_committed_batches(bundle.announcer, retained)
                    except Exception as _ann_exc:  # noqa: BLE001 — best-effort
                        logger.debug(
                            "sp1140 announce-after-commit skipped (%s)", _ann_exc,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — keep the loop alive
                logger.warning(
                    "Sprint 1038 settlement poll iteration error: %s", exc,
                )
            await asyncio.sleep(interval)

    def _launch_settlement_audit_loop(self) -> None:
        """Sprint 1140 (Brick E.2) — build + launch the OPT-IN settlement-receipt audit
        loop. Caller has already checked ``audit_enabled()`` + a settlement client exists.

        Assembles the audit bundle via the (fully-unit-tested) factory:
          - fetcher = a thin wrapper over ``content_provider.request_content`` (raises on
            a None/miss so the loop fails closed per batch);
          - chain_head_fn = ``w3.eth.block_number`` off the settlement contract client's
            web3 (read-only);
          - dry_run_client = the settlement client (read-only dry-run path; the engine
            NEVER broadcasts);
          - my_address = the node's settler/operator address; group_ids default empty.
        Then creates the scheduler task (added to the shutdown drain list). Never raises."""
        import os as _os
        from prsm.settlement.settlement_audit_wiring import (
            BlockCursor,
            build_settlement_audit_components,
        )

        client = self._onchain_settlement_client
        contract_client = getattr(client, "_contract_client", None) or getattr(
            client, "contract_client", None,
        )
        if contract_client is None:
            logger.warning(
                "sp1140 audit loop: no contract client on settlement client; OFF.",
            )
            return

        content_provider = getattr(self, "content_provider", None)
        fetcher = _ContentProviderBlobFetcher(content_provider)

        async def _chain_head() -> int:
            web3 = getattr(contract_client, "web3", None)
            if web3 is None:
                raise RuntimeError("no web3 on contract client for chain head")
            import asyncio as _asyncio
            return int(await _asyncio.to_thread(lambda: web3.eth.block_number))

        cursor_path = (
            _os.environ.get("PRSM_SETTLEMENT_AUDIT_CURSOR_FILE", "").strip()
            or str(Path.home() / ".prsm" / "settlement_audit_cursor.json")
        )
        try:
            interval = float(
                _os.environ.get("PRSM_SETTLEMENT_AUDIT_INTERVAL_S", "600"),
            )
        except (TypeError, ValueError):
            interval = 600.0

        # sp1149 — OPTIONAL operator-facing findings surface. When
        # ``PRSM_SETTLEMENT_AUDIT_FINDINGS_FILE`` is set, construct a bounded/atomic/
        # never-raising store so detected dispute candidates are PERSISTED + reviewable (in
        # addition to the always-on WARNING log). Unset => None => WARNING-log-only (no
        # behavior change). The store is READ-ONLY; it never broadcasts/signs.
        findings_store = None
        findings_path = _os.environ.get(
            "PRSM_SETTLEMENT_AUDIT_FINDINGS_FILE", "",
        ).strip()
        if findings_path:
            try:
                from prsm.settlement.settlement_audit_report import (
                    SettlementAuditFindingsStore,
                )

                findings_store = SettlementAuditFindingsStore(findings_path)
            except Exception as exc:  # noqa: BLE001 — best-effort; never block startup
                logger.warning(
                    "sp1149 audit findings store at %s unavailable (%s); "
                    "WARNING-log-only surface this run", findings_path, exc,
                )
                findings_store = None

        # sp1171 — the audit engine's dry_run_client must implement READ-ONLY dry_run(challenge).
        # Previously this passed the BatchSettlementClient (commit client), which has NO dry_run
        # at all, so EVERY surfaced finding's dry-run verdict was a silent no-op (dry_run_ok
        # unknown). Build a KEYLESS ChallengeSubmitter (static eth_call needs only a FROM address,
        # not a key) so the in-daemon pre-flight is real for the four challengeReceipt reasons
        # (invalid_signature/double_spend/no_escrow/expired). (The consensus_mismatch engine path
        # passes a ChallengeAttempt, not a to_call_args challenge, so its in-daemon dry-run stays
        # advisory-None — the operator's prepare CLI does the real consensus dry-run; findings are
        # surfaced regardless. NEVER broadcasts — no key is held.)
        dry_run_client = client
        try:
            from web3 import Web3 as _Web3

            from prsm.settlement.invalid_signature_submitter import (
                CHALLENGE_RECEIPT_ABI as _CR_ABI,
            )
            from prsm.settlement.invalid_signature_submitter import (
                ChallengeSubmitter as _ChallengeSubmitter,
            )
            _w3 = getattr(contract_client, "web3", None)
            _reg_addr = getattr(contract_client, "contract_address", None)
            if _w3 is not None and _reg_addr and self._operator_address:
                _reg = _w3.eth.contract(
                    address=_Web3.to_checksum_address(_reg_addr), abi=_CR_ABI)

                class _FromOnly:
                    address = _Web3.to_checksum_address(self._operator_address)

                dry_run_client = _ChallengeSubmitter(
                    web3=_w3, registry=_reg, account=_FromOnly())
        except Exception as _dr_exc:  # noqa: BLE001 — best-effort; fall back to advisory-None
            logger.warning(
                "sp1171 keyless audit dry-run client unavailable (%s); dry-run verdicts "
                "advisory-only this run", _dr_exc,
            )
            dry_run_client = client

        bundle = build_settlement_audit_components(
            enabled=True,
            gossip=self.gossip,
            fetcher=fetcher,
            contract_client=contract_client,
            my_address=self._operator_address,
            group_ids=set(),
            dry_run_client=dry_run_client,
            cursor=BlockCursor(path=cursor_path),
            store=getattr(self, "_settlement_published_batch_store", None),
            # sp1144 (§7 producer wiring) — publish the receipt-set blob (so the announced
            # CID resolves) and enrich it with the retained §7 receipts. Both gated by the
            # same audit flag; default-off leaves them None (legacy plain-blob announcer).
            content_publisher=getattr(self, "content_publisher", None),
            inference_receipt_store=getattr(
                self, "_settlement_inference_receipt_store", None,
            ),
            chain_head_fn=_chain_head,
            interval_s=max(5.0, interval),
            findings_store=findings_store,
            # sp1164 — retain verified foreign batches so an operator can later prepare a
            # challenge for them (scripts/settlement_challenge_prepare.py). Audit-gated; None
            # when unbuilt => not retained (behavior unchanged).
            verified_batch_store=getattr(
                self, "_settlement_verified_batch_store", None,
            ),
        )
        if bundle is None:
            return
        self._settlement_audit_bundle = bundle
        self._settlement_audit_task = asyncio.create_task(
            bundle.scheduler.run_forever(),
        )
        logger.info(
            "Sprint 1140 settlement-receipt audit loop launched (read-only, "
            "never-broadcast; cursor=%s, interval=%.0fs).",
            cursor_path, max(5.0, interval),
        )

        # sp1422 — ESCROW-DRAIN WATCH. The audit engine above hunts fraud in batches this node
        # PRODUCED. This is the other direction: batches someone else committed AGAINST THIS
        # NODE'S ESCROW. On-chain, commitBatch(address requester, ...) takes the requester as a
        # plain, unverified, un-bonded argument and finalizeBatch (callable by anyone) pays out
        # of that requester's escrow — so anyone can name us and drain us for the cost of gas.
        # Our only defense is a NO_ESCROW challenge that ONLY WE may raise (_handleNoEscrow
        # reverts unless msg.sender == b.requester). If we are not looking, nobody is.
        self._escrow_drain_watch_task = asyncio.create_task(
            self._escrow_drain_watch_loop(contract_client, max(5.0, interval)),
        )

    async def _escrow_drain_watch_loop(
        self, contract_client: Any, interval: float,
    ) -> None:
        """sp1422 — periodically ask the chain: has anyone committed a batch against MY escrow
        that I never authorized?

        Reads only. NEVER submits a challenge — raising NO_ESCROW is a signed, user-gated tx.
        This loop's job is to make sure the operator FINDS OUT while the challenge window is
        still open, because after it closes the funds are simply gone.

        Scans a rolling lookback rather than a cursor: a batch is only defensible while PENDING,
        so re-examining the recent window every pass is both correct and self-healing (a missed
        pass is picked up by the next one). PRSM_ESCROW_DRAIN_LOOKBACK_BLOCKS sizes it — the
        default 50k blocks is ~24h+ on Base's ~2s blocks, comfortably covering the challenge
        window. The scan is O(batches in window) because BatchCommitted does not carry (let alone
        index) the requester, so we cannot filter for "batches against me" — see
        no_escrow_assembler.scan_for_unauthorized_batches.
        """
        import os as _os

        from prsm.settlement.issued_authorization_store import IssuedAuthorizationStore
        from prsm.settlement.no_escrow_assembler import scan_for_unauthorized_batches

        me = self._operator_address
        if not me:
            logger.warning(
                "sp1422 escrow-drain watch: no operator address; OFF. If this node ever funds "
                "escrow it will be undefended against an unauthorized-batch drain.",
            )
            return

        store_path = (
            _os.environ.get("PRSM_ISSUED_AUTH_STORE_FILE", "").strip()
            or str(Path.home() / ".prsm" / "issued_authorizations.json")
        )
        try:
            store = IssuedAuthorizationStore(store_path)
        except Exception as exc:  # noqa: BLE001 — never take down the node
            logger.error(
                "sp1422 escrow-drain watch: cannot open the issued-auth store at %s (%s); OFF. "
                "This node cannot prove an unauthorized batch against its escrow.",
                store_path, exc,
            )
            return

        try:
            lookback = int(_os.environ.get("PRSM_ESCROW_DRAIN_LOOKBACK_BLOCKS", "50000"))
        except (TypeError, ValueError):
            lookback = 50_000

        logger.info(
            "sp1422 escrow-drain watch launched (read-only, never-submits; requester=%s, "
            "lookback=%d blocks, interval=%.0fs).", me, lookback, interval,
        )

        while True:
            try:
                await asyncio.sleep(interval)
                web3 = getattr(contract_client, "web3", None)
                if web3 is None:
                    continue
                head = int(await asyncio.to_thread(lambda: web3.eth.block_number))
                alerts = await asyncio.to_thread(
                    scan_for_unauthorized_batches,
                    enumerate_batches=lambda f, t: (
                        contract_client._enumerate_committed_batches_sync(f, t, None)
                    ),
                    read_batch=contract_client._get_batch_sync,
                    store=store,
                    my_address=me,
                    from_block=max(0, head - lookback),
                    to_block=head,
                )
                # scan_for_unauthorized_batches already logs each hit at CRITICAL; this is the
                # roll-up an operator's alerting can key on.
                if alerts:
                    savable = [a for a in alerts if a.challengeable]
                    logger.critical(
                        "sp1422: %d UNAUTHORIZED batch(es) drawn against this node's escrow "
                        "(%d still challengeable). Raise NO_ESCROW challenges NOW — only THIS "
                        "key can. See scripts/settlement_challenge_prepare.py.",
                        len(alerts), len(savable),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a bad pass must not kill the watch
                logger.error(
                    "sp1422 escrow-drain watch pass failed (%s); retrying next interval. "
                    "This node is BLIND to an escrow drain until a pass succeeds.", exc,
                )

    async def _reputation_slash_watch_loop(self) -> None:
        """sp1424 — periodically bridge on-chain StakeBond.Slashed events into the provider
        ReputationTracker, so a provider slashed for fraud stops scoring a clean 0.5 and stops
        keeping full aggregator-selection weight.

        Read-only on chain (never signs, never broadcasts). Resolves each slashed operator
        eth-address -> node_id via the SAME self-advertised hardware_profile.operator_address the
        sp1309 discovery resolver uses; a peer can only ever map a slash onto its own node_id or
        the address's genuine owner, so this cannot forge a penalty onto an innocent third party.
        Dedup persists across passes so overlapping block windows never double-count a slash.
        """
        import os as _os

        from prsm.marketplace.slash_reputation_bridge import (
            apply_onchain_slashes_to_reputation,
        )

        tracker = getattr(self, "reputation_tracker", None)
        if tracker is None:
            return

        reader = getattr(self, "_compute_stake_reader", None)
        get_client = getattr(reader, "_get_client", None) if reader is not None else None
        if get_client is None:
            logger.info(
                "sp1424 reputation slash-watch: no on-chain stake reader; OFF. Slashed providers "
                "will keep full selection weight on this node.",
            )
            return

        try:
            interval = float(_os.environ.get("PRSM_SLASH_WATCH_INTERVAL_S", "600"))
        except (TypeError, ValueError):
            interval = 600.0

        seen: set = set()

        def _resolve_node_ids(operator_address: str):
            """operator eth-address -> ALL node_ids advertising it, via discovery hardware_profile.

            sp1430 (audit #4) — returns EVERY match, not just the first: a slashed provider could
            otherwise run a decoy peer advertising its operator_address to soak the one-shot slash
            while its real dispatch node kept full selection weight.
            """
            disc = getattr(self, "discovery", None)
            if disc is None or not operator_address:
                return []
            try:
                peers = disc.get_known_peers()
            except Exception:  # noqa: BLE001 — discovery hiccup: no resolution this cycle
                return []
            target = str(operator_address).lower()
            out = []
            for p in (peers or []):
                hw = getattr(p, "hardware_profile", None)
                if not isinstance(hw, dict):
                    continue
                op = hw.get("operator_address")
                if op and str(op).lower() == target:
                    nid = getattr(p, "node_id", None)
                    if nid:
                        out.append(nid)
            return out

        logger.info(
            "sp1424 reputation slash-watch launched (read-only, never-signs; interval=%.0fs).",
            interval,
        )

        while True:
            try:
                await asyncio.sleep(interval)
                client = get_client()
                if client is None:
                    continue
                events = await asyncio.to_thread(client.get_all_slash_events)
                recorded, unmapped = await asyncio.to_thread(
                    apply_onchain_slashes_to_reputation,
                    events=events,
                    tracker=tracker,
                    resolve_node_ids=_resolve_node_ids,
                    already_recorded=seen,
                )
                if recorded or unmapped:
                    logger.info(
                        "sp1424 slash-watch: recorded %d new on-chain slash(es) into reputation, "
                        "%d unmapped (operator not yet in discovery).", recorded, unmapped,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a bad pass must not kill the watch
                logger.error(
                    "sp1424 slash-watch pass failed (%s); retrying next interval.", exc,
                )

        # sp1307 — also launch the COMPUTE-INTEGRITY watcher (the §7 counterpart to the
        # settlement-fraud audit engine above): self-audit this node's own committed
        # batches by running the §7 verifier on each retained receipt + DRY-RUNNING any
        # actionable challenge. Read-only, never-broadcast; fail-soft (a build error never
        # blocks the audit loop). Reuses the SAME dry_run_client + the two audit stores.
        try:
            from prsm.settlement.settlement_audit_wiring import (
                build_compute_integrity_watcher,
                build_discovery_endpoint_resolver,
                build_peer_compute_integrity_watcher,
            )
            watcher = build_compute_integrity_watcher(
                published_batch_store=getattr(
                    self, "_settlement_published_batch_store", None),
                inference_receipt_store=getattr(
                    self, "_settlement_inference_receipt_store", None),
                dry_run_client=dry_run_client,
            )
            if watcher is not None:
                self._compute_integrity_watcher = watcher
                self._compute_integrity_watcher_task = asyncio.create_task(
                    self._compute_integrity_watcher_loop(
                        watcher, max(5.0, interval), "self"),
                )
                logger.info(
                    "Sprint 1307 compute-integrity watcher launched (read-only, "
                    "never-broadcast; self-audits committed batches every %.0fs).",
                    max(5.0, interval),
                )
            # sp1308 — CROSS-PROVIDER audit: same watcher machinery, but the source pulls
            # FOREIGN producers' §7 receipts (verified_batch_store + the sp1305 serve
            # endpoint, fetched via sp1306) through an INJECTED endpoint resolver.
            # sp1309 — the resolver is now AUTOMATIC: it matches each provider eth-address
            # against the live peer set (a peer's hardware_profile.operator_address, sp690/
            # sp788) to find its receipt-serve endpoint, with PRSM_PEER_RECEIPT_ENDPOINTS as
            # a manual override pin. Inert only if there's neither discovery nor the env map.
            resolver = build_discovery_endpoint_resolver(
                getattr(self, "discovery", None), environ=_os.environ)
            peer_watcher = build_peer_compute_integrity_watcher(
                verified_batch_store=getattr(
                    self, "_settlement_verified_batch_store", None),
                endpoint_resolver=resolver,
                dry_run_client=dry_run_client,
            )
            if peer_watcher is not None:
                self._peer_compute_integrity_watcher = peer_watcher
                self._peer_integrity_watcher_task = asyncio.create_task(
                    self._compute_integrity_watcher_loop(
                        peer_watcher, max(5.0, interval), "peer"),
                )
                logger.info(
                    "Sprint 1308 CROSS-PROVIDER compute-integrity watcher launched "
                    "(read-only; audits foreign producers' §7 receipts via "
                    "PRSM_PEER_RECEIPT_ENDPOINTS every %.0fs).", max(5.0, interval),
                )
        except Exception as _ciw_exc:  # noqa: BLE001 — never block the audit loop
            logger.debug("compute-integrity watcher(s) not launched: %s", _ciw_exc)

    async def _compute_integrity_watcher_loop(
        self, watcher: Any, interval: float, label: str = "self",
    ) -> None:
        """Sprint 1307/1308 — periodically scan for COMPUTE-INTEGRITY fraud (a §7 receipt
        that fails ``verify_inference_receipt_for_challenge``). ``label`` distinguishes the
        'self' audit (sp1307 — this node's OWN committed batches, a TRIPWIRE on producing
        slashable receipts) from the 'peer' audit (sp1308 — FOREIGN producers' receipts, a
        surfaced cross-provider challenge). Read-only + NEVER broadcasts (dry-run + surface
        only). Isolated: never raises, survives iteration errors, exits clean on cancel."""
        if watcher is None:
            return
        while True:
            try:
                result = await watcher.scan()
                if result.actionable:
                    logger.warning(
                        "Sprint 1307/1308 compute-integrity watcher [%s]: %d ACTIONABLE "
                        "challenge(s) over %d unit(s) — %s (dry-run only, NOT broadcast): "
                        "%s", label, len(result.actionable), result.units_scanned,
                        ("this node may have produced INVALID §7 receipt(s)"
                         if label == "self"
                         else "a FOREIGN producer's §7 receipt failed verification"),
                        [a.to_dict() for a in result.actionable],
                    )
                elif result.errors:
                    logger.info(
                        "Sprint 1307/1308 compute-integrity watcher [%s]: %d scanned, "
                        "0 actionable, %d per-unit error(s).",
                        label, result.units_scanned, len(result.errors),
                    )
                else:
                    logger.debug(
                        "Sprint 1307/1308 compute-integrity watcher [%s]: %d unit(s) "
                        "clean.", label, result.units_scanned,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — keep the loop alive
                logger.warning(
                    "Sprint 1307/1308 compute-integrity watcher [%s] iteration error: %s",
                    label, exc,
                )
            await asyncio.sleep(interval)

    async def _collateral_refresh_loop(self) -> None:
        """Sprint 1081 — periodically refresh the Intel SGX attestation collateral
        (PCK CRLs + QE-Identity) from Intel PCS into the cache dir the DCAP backend
        reads, then reload the backend so a long-running node picks up fresh revocation
        data without a restart. Each item is validated (issuer chain to the Intel root +
        signature + freshness) BEFORE the atomic swap, so a refresh can never downgrade
        good cached collateral. The loop never raises and survives iteration errors."""
        import os as _os
        from prsm.compute.inference.collateral_refresh import (
            CollateralRefresher, collateral_cache_dir)
        try:
            interval = float(_os.environ.get("PRSM_COLLATERAL_REFRESH_INTERVAL_S", "86400"))
        except (TypeError, ValueError):
            interval = 86400.0
        interval = max(60.0, interval)
        cache = collateral_cache_dir()
        if cache is None:
            logger.warning("Sprint 1081 collateral refresh: no cache dir — loop idle")
            return
        # sp1081 review M1 — pass the operator-configured TCB signer so the refreshed
        # QE-Identity is validated under the SAME cert the verify path uses.
        _signer_pem = (_os.environ.get("PRSM_INTEL_SGX_TCB_SIGNING_PEM", "") or "").strip()
        _signer_bytes = _signer_pem.encode() if _signer_pem else None
        if not _signer_bytes:
            _sf = (_os.environ.get("PRSM_INTEL_SGX_TCB_SIGNING_FILE", "") or "").strip()
            if _sf:
                try:
                    with open(_sf, "rb") as _f:
                        _signer_bytes = _f.read()
                except OSError:
                    _signer_bytes = None
        # sp1089 — pass the platform FMSPC (if configured) so the per-FMSPC TCB-Info is
        # refreshed alongside the CRLs + QE-Identity.
        _fmspc = (_os.environ.get("PRSM_INTEL_SGX_FMSPC", "") or "").strip() or None
        refresher = CollateralRefresher(cache, tcb_signer_pem=_signer_bytes,
                                        intel_fmspc=_fmspc)
        while True:
            try:
                results = await refresher.refresh_all()
                if any(r.ok for r in results.values()):
                    from prsm.compute.inference.attestation_backends import (
                        reload_amd_sev_snp, reload_intel_dcap)
                    # Reload only the vendor(s) whose collateral actually changed.
                    if any(k.startswith("intel") and v.ok for k, v in results.items()):
                        try:
                            reload_intel_dcap()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Sprint 1081 DCAP reload error: %s", exc)
                    if any(k.startswith("amd") and v.ok for k, v in results.items()):
                        try:
                            reload_amd_sev_snp()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Sprint 1082 SEV-SNP reload error: %s", exc)
                    logger.info("Sprint 1081/1082: reloaded attestation backend(s) with "
                                "refreshed collateral (%s)",
                                {k: v.ok for k, v in results.items()})
                else:
                    logger.debug("Sprint 1081 collateral refresh: no items updated")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — keep the loop alive
                logger.warning("Sprint 1081 collateral refresh iteration error: %s", exc)
            await asyncio.sleep(interval)

    async def stop(self) -> None:
        """Gracefully shut down all subsystems."""
        if not self._started:
            return

        logger.info("Shutting down PRSM node...")

        # sp1459 — stop the marketplace advertiser's re-broadcast loop first so a shutting-down
        # node stops advertising itself as available supply. Safe when None / never started.
        if getattr(self, "_marketplace_advertiser", None) is not None:
            try:
                await _await_bounded(
                    self._marketplace_advertiser.stop(), _STOP_TIMEOUT, "marketplace_advertiser")
            except Exception as exc:  # noqa: BLE001
                logger.warning("marketplace advertiser stop raised: %s", exc)

        # Sprint 766 — stop AutoClaimWorker before tearing down the
        # staking_manager it depends on. Safe to call .stop() even
        # when worker is None or was never started.
        if getattr(self, "_auto_claim_worker", None) is not None:
            try:
                await _await_bounded(self._auto_claim_worker.stop(), _STOP_TIMEOUT, "auto_claim_worker")
            except Exception as exc:
                logger.warning(
                    "AutoClaimWorker stop raised: %s", exc,
                )

        # Sprint 878 — stop FunnelAutoSweepWorker. Safe to call
        # even when None / never started.
        if getattr(
            self, "_funnel_auto_sweep_worker", None,
        ) is not None:
            try:
                await _await_bounded(self._funnel_auto_sweep_worker.stop(), _STOP_TIMEOUT, "funnel_auto_sweep_worker")
            except Exception as exc:
                logger.warning(
                    "FunnelAutoSweepWorker stop raised: %s", exc,
                )

        # Sprint 775 — stop the PreemptionDetector polling loop +
        # clear the global registration so a process-recycle test
        # or daemon restart doesn't see stale state. Happens BEFORE
        # api_task cancel so a final in-flight dispatch gate
        # decision sees a stable flag.
        if getattr(self, "_preemption_detector", None) is not None:
            try:
                await _await_bounded(self._preemption_detector.stop(), _STOP_TIMEOUT, "preemption_detector")
            except Exception as exc:
                logger.warning(
                    "PreemptionDetector stop raised: %s", exc,
                )
            try:
                from prsm.node.preemption import reset_for_testing
                reset_for_testing()
            except Exception:
                pass
            self._preemption_detector = None

        if self._api_task:
            self._api_task.cancel()
            self._api_task = None

        # Phase 7-storage + Phase 8 daemons + watchers — graceful stop
        # signals the loop to exit at next iteration; await the task
        # to ensure any in-flight tick completes before we tear down
        # further.
        if self._heartbeat_scheduler is not None:
            await self._heartbeat_scheduler.stop()
        if self._compensation_scheduler is not None:
            await self._compensation_scheduler.stop()
        if self._key_distribution_watcher is not None:
            await self._key_distribution_watcher.stop()
        if self._storage_slashing_watcher is not None:
            await self._storage_slashing_watcher.stop()
        if getattr(self, "_settlement_challenge_watcher", None) is not None:
            await self._settlement_challenge_watcher.stop()
        if self._compensation_distributor_watcher is not None:
            await self._compensation_distributor_watcher.stop()
        if getattr(self, "_job_reaper", None) is not None:
            await self._job_reaper.stop()
        if getattr(self, "_daemon_watchdog", None) is not None:
            await self._daemon_watchdog.stop()
        if getattr(self, "_pending_withdraw_reconciler", None) is not None:
            await self._pending_withdraw_reconciler.stop()  # sp916
        # sp1217 — stop runtime observability. AlertManager FIRST (its eval loop
        # reads the collector's registry); then the collector. Both own their
        # internal async tasks via these coroutines; bounded so a wedged stop
        # can't hang shutdown.
        if getattr(self, "_alert_manager", None) is not None:
            await _await_bounded(
                self._alert_manager.stop_monitoring(), _STOP_TIMEOUT, "alert_manager",
            )
            self._alert_manager = None
        if getattr(self, "_metrics_collector", None) is not None:
            await _await_bounded(
                self._metrics_collector.stop_collection(), _STOP_TIMEOUT, "metrics_collector",
            )
            self._metrics_collector = None
        for task_attr in (
            "_heartbeat_scheduler_task",
            "_job_reaper_task",
            "_daemon_watchdog_task",
            "_compensation_scheduler_task",
            "_key_distribution_watcher_task",
            "_storage_slashing_watcher_task",
            "_settlement_challenge_watcher_task",  # sp1382
            "_compensation_distributor_watcher_task",
            "_pending_withdraw_reconciler_task",  # sp916
            "_settlement_poll_task",  # sp1038
            "_settlement_audit_task",  # sp1140 (Brick E.2)
            "_escrow_drain_watch_task",  # sp1422
            "_reputation_slash_watch_task",  # sp1424
            "_compute_integrity_watcher_task",  # sp1307
            "_peer_integrity_watcher_task",  # sp1308
            "_collateral_refresh_task",  # sp1081
        ):
            task = getattr(self, task_attr, None)
            if task is not None:
                # sp955 — bounded drain. `asyncio.wait_for(task, 5)` cancels then
                # AWAITS the cancellation on timeout, so a task stuck in an
                # uncancellable await hangs shutdown forever (observed: node.stop
                # parked on PendingWithdrawReconciler). `_drain_task_bounded`
                # abandons a non-finishing task instead, guaranteeing we return.
                await _drain_task_bounded(task, 5.0, name=task_attr)
                setattr(self, task_attr, None)

        if self.agent_collaboration:
            # sp956 — every awaitable subsystem stop below is bounded via
            # _await_bounded so a single stuck subsystem (libp2p subprocess
            # teardown, a wedged SQLite close, a libtorrent shutdown, a chain
            # RPC) cannot hang node shutdown. Normal stops finish fast and are
            # reaped immediately; only a genuinely-stuck one is abandoned.
            await _await_bounded(self.agent_collaboration.stop(), _STOP_TIMEOUT, "agent_collaboration")
        # Stop content economy (Phase 4)
        if self.content_economy:
            await _await_bounded(self.content_economy.stop(), _STOP_TIMEOUT, "content_economy")
        # Stop multi-party escrow (Phase 4) — synchronous stop (fast flag-set).
        if hasattr(self, '_mp_escrow') and self._mp_escrow:
            self._mp_escrow.stop()
        # Stop BitTorrent components (libtorrent teardown can hang → bounded)
        if self.bt_requester:
            await _await_bounded(self.bt_requester.stop(), _STOP_TIMEOUT, "bt_requester")
        if self.bt_provider:
            await _await_bounded(self.bt_provider.stop(), _STOP_TIMEOUT, "bt_provider")
        if self.bt_client:
            await _await_bounded(self.bt_client.shutdown(), _STOP_TIMEOUT, "bt_client")
        if self.ledger_sync:
            await _await_bounded(self.ledger_sync.stop(), _STOP_TIMEOUT, "ledger_sync")
        if hasattr(self, '_escrow_cleanup_task') and self._escrow_cleanup_task:
            self._escrow_cleanup_task.cancel()
            await _drain_task_bounded(
                self._escrow_cleanup_task, _STOP_TIMEOUT, "escrow_cleanup_task",
            )

        if self.content_uploader:
            await _await_bounded(self.content_uploader.close(), _STOP_TIMEOUT, "content_uploader")
        if self.storage_provider:
            await _await_bounded(self.storage_provider.stop(), _STOP_TIMEOUT, "storage_provider")
        if self.compute_provider:
            await _await_bounded(self.compute_provider.stop(), _STOP_TIMEOUT, "compute_provider")
        if self.compute_requester:
            await _await_bounded(self.compute_requester.stop(), _STOP_TIMEOUT, "compute_requester")
        if hasattr(self, '_capability_announce_task'):
            self._capability_announce_task.cancel()
            await _drain_task_bounded(
                self._capability_announce_task, _STOP_TIMEOUT, "capability_announce_task",
            )

        if getattr(self, '_content_readvertise_task', None):  # sp1343
            self._content_readvertise_task.cancel()
            await _drain_task_bounded(
                self._content_readvertise_task, _STOP_TIMEOUT, "content_readvertise_task",
            )

        if self.discovery:
            await _await_bounded(self.discovery.stop(), _STOP_TIMEOUT, "discovery")
        # T3b: stop the DHT components before transport so any in-flight
        # outbound DHT RPC has the underlying transport adapter still
        # available during teardown. Idempotent. (Synchronous stop — fast.)
        if self.dht_components is not None:
            try:
                self.dht_components.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"DHT components stop failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        if self.gossip:
            await _await_bounded(self.gossip.stop(), _STOP_TIMEOUT, "gossip")
        if self.transport:
            await _await_bounded(self.transport.stop(), _STOP_TIMEOUT, "transport")
        if self.ledger:
            await _await_bounded(self.ledger.close(), _STOP_TIMEOUT, "ledger")

        # Close content storage
        try:
            from prsm.storage import close_content_store
            close_content_store()
            logger.info("ContentStore closed")
        except Exception as e:
            logger.warning(f"ContentStore close failed: {e}")

        self._started = False
        logger.info("PRSM node stopped")

    async def _seed_welcome_grant(self) -> None:
        """Rebuild wallet_balances cache from dag_transactions on startup.

        wallet_balances is just a performance cache — the source of truth
        is dag_transactions. We rebuild it from scratch once per startup
        to avoid any stale version counters from previous buggy runs.
        """
        try:
            # ── Step 1: Nuke and rebuild wallet_balances from truth ─────
            if hasattr(self.ledger, "_db"):
                # Delete all rows first (safe — cache only)
                await self.ledger._db.execute("DELETE FROM wallet_balances")
                # Rebuild from dag_transactions
                await self.ledger._db.execute(
                    """INSERT INTO wallet_balances (wallet_id, balance, version, last_updated)
                       SELECT w.wallet_id,
                              COALESCE((SELECT SUM(amount) FROM dag_transactions WHERE to_wallet = w.wallet_id), 0) -
                              COALESCE((SELECT SUM(amount) FROM dag_transactions WHERE from_wallet = w.wallet_id), 0),
                              1,
                              COALESCE((SELECT MAX(timestamp) FROM dag_transactions WHERE to_wallet = w.wallet_id OR from_wallet = w.wallet_id), 0)
                       FROM wallets w"""
                )
                await self.ledger._db.commit()
                # Reset the in-memory version cache
                if hasattr(self.ledger, "_balance_version_cache"):
                    self.ledger._balance_version_cache.clear()
                    cursor = await self.ledger._db.execute(
                        "SELECT wallet_id, version FROM wallet_balances"
                    )
                    async for row in cursor:
                        self.ledger._balance_version_cache[row[0]] = row[1]

            # ── Step 2: Check balance and grant if needed ──
            balance = await self.ledger.get_balance(self.identity.node_id)
            if balance <= 0:
                await self.ledger.credit(
                    wallet_id=self.identity.node_id,
                    amount=100.0,
                    tx_type=TransactionType.WELCOME_GRANT,
                    description="Welcome grant for new node",
                )
                # Also prime wallet_balances for the fresh grant
                if hasattr(self.ledger, "_db"):
                    # Ensure wallet exists first
                    if not await self.ledger.wallet_exists(self.identity.node_id):
                        await self.ledger.create_wallet(self.identity.node_id, "node")
                    await self.ledger._db.execute(
                        """INSERT INTO wallet_balances (wallet_id, balance, version, last_updated)
                           VALUES (?, ?, 1, ?)
                           ON CONFLICT(wallet_id) DO UPDATE SET balance = excluded.balance""",
                        (self.identity.node_id, 100.0, time.time()),
                    )
                    await self.ledger._db.commit()
                    if hasattr(self.ledger, "_balance_version_cache"):
                        self.ledger._balance_version_cache[self.identity.node_id] = 1
                logger.info(f"Seeded welcome grant: 100 FTNS to {self.identity.node_id[:12]}...")
            else:
                logger.debug(f"Node already has balance: {balance:.6f}")
        except Exception as e:
            logger.warning(f"Welcome-grant reconciliation failed: {e}")

    # ── On-Chain FTNS Transfer Handler ────────────────────────
    async def _on_chain_ftns_transfer(self, transaction) -> None:
        """Queue a transaction for batch settlement on Base mainnet.

        KEY SAFETY: This must only be called AFTER the local ledger
        transaction has successfully committed. Never broadcast before
        local commit — otherwise we burn gas on transactions that get
        rolled back by TOCTOU/ConcurrentModification failures.

        Transactions are queued in BatchSettlementManager and flushed
        periodically (default: every 10 min or when pending >= 1.0 FTNS).
        This saves gas by netting opposing transfers and batching commits.
        """
        # Route through batch settlement if available
        if hasattr(self, '_batch_settlement') and self._batch_settlement:
            await self._batch_settlement.enqueue(transaction)
            return

        # Legacy fallback: direct on-chain transfer (no batch settlement)
        await self._direct_on_chain_transfer(transaction)

    async def _direct_on_chain_transfer(self, transaction) -> None:
        """Direct on-chain transfer (legacy fallback, no batching).

        Used when batch settlement is not initialized.
        """
        if not self.ftns_ledger or not self.ftns_ledger._is_initialized:
            return
        if not hasattr(transaction, "from_wallet") or not hasattr(transaction, "to_wallet"):
            return

        # Dedup: skip if we've already broadcast this transaction
        tx_key = (transaction.tx_id if hasattr(transaction, "tx_id") else "",
                  transaction.to_wallet if hasattr(transaction, "to_wallet") else "")
        if not hasattr(self, "_broadcast_sent"):
            self._broadcast_sent = set()
        if tx_key in self._broadcast_sent:
            logger.debug(
                f"Skipping duplicate FTNS broadcast for {tx_key[0][:12]}…"
            )
            return
        self._broadcast_sent.add(tx_key)

        to_addr = transaction.to_wallet
        target_address = None
        if to_addr.startswith("0x") and len(to_addr) >= 40:
            target_address = to_addr
        elif to_addr == self.identity.node_id and self.ftns_ledger._connected_address:
            target_address = self.ftns_ledger._connected_address
            logger.info(
                f"Bridging local payment to on-chain: "
                f"{self.identity.node_id[:12]}... -> {target_address}"
            )
        else:
            logger.debug(
                f"Skipping on-chain FTNS transfer for named wallet: {to_addr[:20]}…"
            )
            return

        amount = float(transaction.amount) if hasattr(transaction, "amount") else 0
        if amount <= 0:
            return

        try:
            tx_record = await self.ftns_ledger.transfer(
                job_id=tx_key[0],
                to_address=target_address,
                amount_ftns=amount,
            )
            if tx_record and tx_record.status == "confirmed":
                logger.info(
                    f"FTNS on-chain: {amount:.6f} confirmed "
                    f"(tx: {tx_record.tx_hash[:16]}…)"
                )
            elif tx_record and tx_record.status == "rejected":
                logger.warning(
                    f"FTNS on-chain transfer rejected: "
                    f"tx={tx_record.tx_hash[:16] if tx_record.tx_hash else 'N/A'}..."
                )
        except Exception as e:
            logger.error(f"FTNS on-chain transfer failed: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Comprehensive node status."""
        balance = 0.0
        if self.ledger and self.identity:
            balance = await self.ledger.get_balance(self.identity.node_id)

        uptime = time.time() - self._start_time if self._start_time else 0.0

        status = {
            "node_id": self.identity.node_id if self.identity else None,
            "display_name": self.config.display_name,
            "roles": [r.value for r in self.config.roles],
            "ledger_type": self.config.ledger_type,
            "started": self._started,
            "uptime_seconds": round(uptime, 1),
            "p2p_address": f"ws://{self.config.listen_host}:{self.config.p2p_port}",
            "api_address": f"http://127.0.0.1:{self.config.api_port}",
            "peers": {
                "connected": self.transport.peer_count if self.transport else 0,
                # Use get_known_peers() (the documented API on both
                # legacy Discovery + new Libp2pDiscovery) instead
                # of the .known_peers attribute (legacy-only). The
                # attribute path crashed get_status() when running
                # with libp2p discovery.
                "known": (
                    len(self.discovery.get_known_peers())
                    if self.discovery else 0
                ),
                "bootstrap": (
                    self.discovery.get_bootstrap_status()
                    if self.discovery
                    and hasattr(self.discovery, "get_bootstrap_status")
                    else {}
                ),
                "bootstrap_telemetry": (
                    self.discovery.get_bootstrap_telemetry()
                    if self.discovery
                    and hasattr(self.discovery, "get_bootstrap_telemetry")
                    else {}
                ),
            },
            "ftns_balance": balance,
            "dag_stats": {
                "note": "DAG ledger in async mode",
                "mode": "dag" if hasattr(self.ledger, '_dag') else "sql"
            },
            "compute": self.compute_provider.get_stats() if self.compute_provider else None,
            "compute_requester": self.compute_requester.get_stats() if self.compute_requester else None,
            "storage": self.storage_provider.get_stats() if self.storage_provider else None,
            "content": self.content_uploader.get_stats() if self.content_uploader else None,
            "content_index": self.content_index.get_stats() if self.content_index else None,
            "content_provider": self.content_provider.get_stats() if self.content_provider else None,
            "ledger_sync": self.ledger_sync.get_stats() if self.ledger_sync else None,
            "escrow": self._payment_escrow.get_stats() if hasattr(self, '_payment_escrow') and self._payment_escrow else None,
            "consensus": self._result_consensus.get_stats() if hasattr(self, '_result_consensus') and self._result_consensus else None,
            "batch_settlement": self._batch_settlement.get_stats() if hasattr(self, '_batch_settlement') and self._batch_settlement else None,
            "ftns_onchain": (
                self.ftns_ledger.get_summary()
                if self.ftns_ledger and self.ftns_ledger._is_initialized
                else None
            ),
            "agents": self.agent_registry.get_stats() if self.agent_registry else None,
            "collaboration": self.agent_collaboration.get_stats() if self.agent_collaboration else None,
            "bittorrent": {
                "available": self.bt_client.available if self.bt_client else False,
                "provider": self.bt_provider.get_stats() if self.bt_provider else None,
                "requester": self.bt_requester.get_stats() if self.bt_requester else None,
            },
        }
        return status

    async def _run_api(self) -> None:
        """Run the management API server."""
        try:
            from prsm.node.api import create_api_app
            import uvicorn

            app = create_api_app(self)
            # sp1017 — bind the configured management-API host (default
            # 127.0.0.1 loopback; PRSM_API_HOST / config.api_host to expose).
            # The sp1011 posture check assesses this same host.
            config = uvicorn.Config(
                app,
                host=getattr(self.config, "api_host", None) or "127.0.0.1",
                port=self.config.api_port,
                log_level="warning",
            )
            if self.config.tls_enabled and self.config.tls_cert_path:
                config.ssl_certfile = self.config.tls_cert_path
                config.ssl_keyfile = self.config.tls_key_path
            server = uvicorn.Server(config)
            await server.serve()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"API server error: {e}")
