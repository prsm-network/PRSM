"""
Node Configuration
==================

Dataclass-based configuration for a PRSM network node.
Defaults work out of the box for local development.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional
import json
import os


# Multi-region bootstrap server configuration.
# Sprint 533 F57 fix: write ALL canonical regional bootstraps to the
# default list so fresh installs get full failover wired into
# `~/.prsm/config.yaml`, not just the (currently DNS-failing)
# primary. Previously DEFAULT contained only `bootstrap1.prsm-
# network.com` — when its DNS A record drifted, every fresh user
# got an unreachable primary. The daemon's runtime FALLBACK list
# already tries EU/APAC, but `prsm config show` and the setup
# wizard only persisted the primary, making the failure invisible
# at config-inspection time. Listing all 3 here means:
#   - `prsm config show` displays the full fleet (no surprises)
#   - daemon's connect loop tries them in order
#   - operators can edit/remove regions without env-var hacking
DEFAULT_BOOTSTRAP_NODES = [
    # Sprint 575 F29 — bootstrap1 → bootstrap-us DNS rename
    # (2026-05-19). Old hostname no longer resolves; defaulting
    # to it would make every new operator fail initial bootstrap.
    os.getenv("BOOTSTRAP_PRIMARY", "wss://bootstrap-us.prsm-network.com:8765"),
    os.getenv("BOOTSTRAP_FALLBACK_EU", "wss://bootstrap-eu.prsm-network.com:8765"),
    os.getenv("BOOTSTRAP_FALLBACK_APAC", "wss://bootstrap-apac.prsm-network.com:8765"),
]

# FALLBACK_BOOTSTRAP_NODES retained for backwards compat with
# callers that explicitly distinguish primary vs fallbacks (e.g.
# health-probe display). DEFAULT_BOOTSTRAP_NODES is the canonical
# operator-config source.
FALLBACK_BOOTSTRAP_NODES = [
    os.getenv("BOOTSTRAP_FALLBACK_EU", "wss://bootstrap-eu.prsm-network.com:8765"),
    os.getenv("BOOTSTRAP_FALLBACK_APAC", "wss://bootstrap-apac.prsm-network.com:8765"),
]


class NodeRole(str, Enum):
    """Operating mode for the node."""
    FULL = "full"           # Compute + storage + routing
    COMPUTE = "compute"     # Compute jobs only
    STORAGE = "storage"     # Storage contribution only


@dataclass
class NodeConfig:
    """Configuration for a PRSM network node.

    Sensible defaults allow a node to start with zero configuration.
    """
    # Identity
    display_name: str = "prsm-node"
    roles: List[NodeRole] = field(default_factory=lambda: [NodeRole.FULL])

    # Ledger type: "dag" for DAG-based (IOTA-style) or "legacy" for linear
    ledger_type: str = "dag"

    # Network
    listen_host: str = "0.0.0.0"  # P2P transport bind (peers must reach it)
    p2p_port: int = 9001
    api_port: int = 8000
    # sp1017 — management-API bind. Defaults to loopback (the money/KYC
    # endpoints are not network-exposed out of the box; the sp1011 fail-closed
    # posture check assesses THIS host, not listen_host). Operators who front
    # the API with a reverse proxy keep this loopback; those who intentionally
    # expose it (e.g. Docker port-mapping) set api_host=0.0.0.0 — and then must
    # also set PRSM_NODE_API_KEY or the node fail-closes (sp1011). Override via
    # PRSM_API_HOST.
    api_host: str = "127.0.0.1"

    # DHT transport (PRSM-DHT-TRANSPORT T1-T5; T3b wiring)
    # Off by default — opt in via PRSM_DHT_ENABLED=1 or by setting
    # dht_enabled=True. When enabled, the node binds an additional TCP
    # listener for inbound ManifestDHT + EmbeddingDHT requests and
    # constructs the corresponding sync clients for upload-critical-path
    # code. Port 0 lets the kernel assign one — operators who need a
    # stable port (firewall rule, NAT mapping) should set explicitly.
    dht_enabled: bool = False
    dht_listen_port: int = 0
    bootstrap_nodes: List[str] = field(default_factory=lambda: list(DEFAULT_BOOTSTRAP_NODES))
    bootstrap_connect_timeout: float = 5.0
    bootstrap_retry_attempts: int = 2
    bootstrap_fallback_enabled: bool = True
    bootstrap_fallback_nodes: List[str] = field(
        default_factory=lambda: list(FALLBACK_BOOTSTRAP_NODES)
    )
    bootstrap_validate_addresses: bool = True
    bootstrap_backoff_base: float = 1.0
    bootstrap_backoff_max: float = 8.0
    max_peers: int = 50

    # Compute behavior
    allow_self_compute: bool = True        # Execute own jobs when no peers (single-node mode)

    # Resources
    storage_gb: float = 10.0
    cpu_allocation_pct: int = 50       # % of CPU to offer for jobs
    memory_allocation_pct: int = 50    # % of RAM to offer

    # Compute limits
    max_concurrent_jobs: int = 3          # Parallel job slots
    gpu_allocation_pct: int = 80          # % of GPU VRAM to offer (if GPU detected)

    # Network/bandwidth
    upload_mbps_limit: float = 0.0        # 0 = unlimited; non-zero = cap in Mbps
    download_mbps_limit: float = 0.0      # 0 = unlimited

    # Scheduling
    active_hours_start: Optional[int] = None  # Hour 0-23 (None = always on)
    active_hours_end: Optional[int] = None    # Hour 0-23 (None = always on)
    active_days: List[int] = field(          # 0=Mon ... 6=Sun (empty = every day)
        default_factory=list
    )

    # Gossip protocol
    gossip_fanout: int = 3
    gossip_ttl: int = 5
    heartbeat_interval: float = 30.0   # seconds

    # Storage paths
    data_dir: str = field(default_factory=lambda: str(Path.home() / ".prsm"))

    # FTNS
    welcome_grant: float = 100.0

    # TLS (production)
    tls_enabled: bool = False
    tls_cert_path: str = ""
    tls_key_path: str = ""

    # Content Economy (Phase 4)
    min_replicas: int = 3
    royalty_model: str = "phase4"  # "phase4" or "legacy"

    # WASM Runtime (Ring 1)
    wasm_enabled: bool = True
    wasm_max_memory_bytes: int = 256 * 1024 * 1024  # 256 MB default sandbox
    wasm_max_execution_seconds: int = 30
    wasm_max_module_size: int = 5 * 1024 * 1024  # 5 MB

    # On-chain resilience (Ring 6)
    base_rpc_urls: List[str] = field(default_factory=lambda: [
        "https://mainnet.base.org",
    ])
    gas_price_multiplier: float = 1.2
    max_gas_gwei: int = 50

    # Transport configuration. sp1188 (day-one-live #7) — default to the hardened
    # WebSocket path: it carries the origin-auth / replay / table-bound defenses
    # (sp941/1005/1026), it's what the live fleet + bootstrap server actually run
    # (operators were all overriding the old "libp2p" default with
    # PRSM_TRANSPORT_BACKEND=websocket), and the libp2p data-plane still lacks per-message
    # auth (sp1010 Residual A). libp2p remains available via PRSM_TRANSPORT_BACKEND=libp2p.
    transport_backend: str = "websocket"       # "websocket" or "libp2p"
    libp2p_library_path: str = ""              # Auto-detected if empty
    enable_relay: bool = True                  # Circuit Relay v2
    enable_nat_traversal: bool = True          # AutoNAT + hole punching
    dht_mode: str = "auto"                     # "server", "client", or "auto"

    # Discovery tuning
    target_peers: int = 8
    announce_interval: float = 60.0
    maintenance_interval: float = 30.0
    peer_stale_timeout: float = 600.0          # 10 minutes

    # Transport tuning
    nonce_window: float = 300.0                # 5 minutes
    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 10.0
    handshake_timeout: float = 10.0
    nonce_cleanup_interval: float = 60.0

    # Collaboration tuning
    task_timeout: float = 3600.0               # 1 hour
    review_timeout: float = 3600.0             # 1 hour
    query_timeout: float = 1800.0              # 30 minutes
    max_completed_records: int = 500
    collab_cleanup_interval: float = 60.0

    # Bid selection tuning
    bid_strategy: str = "best_score"       # "lowest_cost", "fastest", "best_score"
    bid_window_seconds: float = 30.0
    min_bids: int = 1

    # Content index tuning
    max_indexed_cids: int = 10000

    # Ledger sync tuning
    reconciliation_interval: float = 300.0     # 5 minutes

    def __post_init__(self):
        # Allow override via PRSM_BOOTSTRAP_NODES env var
        env_bootstrap = os.environ.get("PRSM_BOOTSTRAP_NODES", "")
        if env_bootstrap:
            self.bootstrap_nodes = [s.strip() for s in env_bootstrap.split(",") if s.strip()]

        # Allow override via PRSM_TRANSPORT_BACKEND env var
        if os.getenv("PRSM_TRANSPORT_BACKEND"):
            self.transport_backend = os.getenv("PRSM_TRANSPORT_BACKEND")

        # sp1017 — allow override of the management-API bind via PRSM_API_HOST.
        _api_host = os.environ.get("PRSM_API_HOST", "").strip()
        if _api_host:
            self.api_host = _api_host

    @property
    def identity_path(self) -> Path:
        return Path(self.data_dir) / "identity.json"

    @property
    def ledger_path(self) -> Path:
        return Path(self.data_dir) / "ledger.db"

    @property
    def config_path(self) -> Path:
        return Path(self.data_dir) / "node_config.json"

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Persist config to disk."""
        self.ensure_dirs()
        data = {
            "display_name": self.display_name,
            "roles": [r.value for r in self.roles],
            "listen_host": self.listen_host,
            "p2p_port": self.p2p_port,
            "api_port": self.api_port,
            "api_host": self.api_host,
            "bootstrap_nodes": self.bootstrap_nodes,
            "bootstrap_connect_timeout": self.bootstrap_connect_timeout,
            "bootstrap_retry_attempts": self.bootstrap_retry_attempts,
            "bootstrap_fallback_enabled": self.bootstrap_fallback_enabled,
            "bootstrap_fallback_nodes": self.bootstrap_fallback_nodes,
            "bootstrap_validate_addresses": self.bootstrap_validate_addresses,
            "bootstrap_backoff_base": self.bootstrap_backoff_base,
            "bootstrap_backoff_max": self.bootstrap_backoff_max,
            "max_peers": self.max_peers,
            "allow_self_compute": self.allow_self_compute,
            "storage_gb": self.storage_gb,
            "cpu_allocation_pct": self.cpu_allocation_pct,
            "memory_allocation_pct": self.memory_allocation_pct,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "gpu_allocation_pct": self.gpu_allocation_pct,
            "upload_mbps_limit": self.upload_mbps_limit,
            "download_mbps_limit": self.download_mbps_limit,
            "active_hours_start": self.active_hours_start,
            "active_hours_end": self.active_hours_end,
            "active_days": self.active_days,
            "gossip_fanout": self.gossip_fanout,
            "gossip_ttl": self.gossip_ttl,
            "heartbeat_interval": self.heartbeat_interval,
            "data_dir": self.data_dir,
            "welcome_grant": self.welcome_grant,
            "target_peers": self.target_peers,
            "announce_interval": self.announce_interval,
            "maintenance_interval": self.maintenance_interval,
            "peer_stale_timeout": self.peer_stale_timeout,
            "nonce_window": self.nonce_window,
            "ws_ping_interval": self.ws_ping_interval,
            "ws_ping_timeout": self.ws_ping_timeout,
            "handshake_timeout": self.handshake_timeout,
            "nonce_cleanup_interval": self.nonce_cleanup_interval,
            "task_timeout": self.task_timeout,
            "review_timeout": self.review_timeout,
            "query_timeout": self.query_timeout,
            "max_completed_records": self.max_completed_records,
            "collab_cleanup_interval": self.collab_cleanup_interval,
            "bid_strategy": self.bid_strategy,
            "bid_window_seconds": self.bid_window_seconds,
            "min_bids": self.min_bids,
            "max_indexed_cids": self.max_indexed_cids,
            "reconciliation_interval": self.reconciliation_interval,
        }
        self.config_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NodeConfig":
        """Load config from disk, falling back to defaults.

        Resolution order:
          1. Explicit path argument (test-friendly injection)
          2. ~/.prsm/node_config.json (legacy; backwards-compat)
          3. ~/.prsm/config.yaml (new PRSMConfig format —
             post-migration users land here)
          4. Defaults

        Pre-fix dueling-config bug: NodeConfig only checked (2),
        but cli_modules/migration.py renames node_config.json →
        .bak after writing config.yaml. Wizard-configured ports +
        bootstrap settings were silently ignored at runtime.
        Sprint 134 closes that loop.
        """
        if path is not None:
            if not path.exists():
                return cls()
            return cls._load_from_json_path(path)

        legacy_path = Path.home() / ".prsm" / "node_config.json"
        if legacy_path.exists():
            return cls._load_from_json_path(legacy_path)

        yaml_path = Path.home() / ".prsm" / "config.yaml"
        if yaml_path.exists():
            return cls._load_from_yaml_path(yaml_path)

        return cls()

    @classmethod
    def _load_from_json_path(cls, path: Path) -> "NodeConfig":
        data = json.loads(path.read_text())
        roles = [NodeRole(r) for r in data.pop("roles", ["full"])]
        # Sprint 575 F29 — bootstrap1 → bootstrap-us DNS rename.
        # Migrate stored JSON configs the same way _load_from_yaml_path
        # does, so legacy JSON-on-disk operators get redirected to the
        # live hostname instead of stranded on dead DNS.
        bs = data.get("bootstrap_nodes")
        if isinstance(bs, list):
            data["bootstrap_nodes"] = [
                (entry.replace(
                    "bootstrap1.prsm-network.com",
                    "bootstrap-us.prsm-network.com",
                ) if isinstance(entry, str) else entry)
                for entry in bs
            ]
        return cls(roles=roles, **data)

    @classmethod
    def _load_from_yaml_path(cls, path: Path) -> "NodeConfig":
        """Load + map PRSMConfig (config.yaml) → NodeConfig fields.

        Field-name renames:
          cpu_pct       -> cpu_allocation_pct
          memory_pct    -> memory_allocation_pct
          gpu_pct       -> gpu_allocation_pct
          node_role     -> roles (single str -> [NodeRole])

        Direct passthroughs (same name on both):
          display_name, p2p_port, api_port, bootstrap_nodes,
          storage_gb, max_concurrent_jobs, upload_mbps_limit,
          active_hours_start, active_hours_end, active_days

        PRSMConfig-only fields (has_openai_key, wallet_address,
        mcp_server_*, setup_*, etc.) are silently dropped.
        """
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        ncfg: dict = {}
        _renames = {
            "cpu_pct": "cpu_allocation_pct",
            "memory_pct": "memory_allocation_pct",
            "gpu_pct": "gpu_allocation_pct",
        }
        for src, dst in _renames.items():
            if src in raw:
                ncfg[dst] = raw[src]

        for field_name in (
            "display_name", "p2p_port", "api_port", "api_host",
            "bootstrap_nodes", "storage_gb",
            "max_concurrent_jobs", "upload_mbps_limit",
            "active_hours_start", "active_hours_end", "active_days",
        ):
            if field_name in raw:
                ncfg[field_name] = raw[field_name]

        # Sprint 149 — silent migration of pre-148 wizard's broken
        # bootstrap defaults. Only an EXACT match of the legacy list
        # is migrated; anything else is operator-customized and
        # preserved as-is.
        _LEGACY_BROKEN_BOOTSTRAP = [
            "/dns4/bootstrap1.prsm.network/tcp/9001/p2p/QmPRSM1",
            "/dns4/bootstrap2.prsm.network/tcp/9001/p2p/QmPRSM2",
        ]
        if ncfg.get("bootstrap_nodes") == _LEGACY_BROKEN_BOOTSTRAP:
            ncfg["bootstrap_nodes"] = list(DEFAULT_BOOTSTRAP_NODES)

        # Sprint 575 F29 — bootstrap1 → bootstrap-us DNS rename
        # (2026-05-19). Auto-migrate stored configs whose primary
        # bootstrap still references the dead hostname so existing
        # operators don't get stranded on a non-resolving URL after
        # `prsm upgrade`. Per-entry replacement preserves any
        # operator-customized fallbacks.
        bs = ncfg.get("bootstrap_nodes")
        if isinstance(bs, list):
            ncfg["bootstrap_nodes"] = [
                (entry.replace(
                    "bootstrap1.prsm-network.com",
                    "bootstrap-us.prsm-network.com",
                ) if isinstance(entry, str) else entry)
                for entry in bs
            ]

        if "node_role" in raw:
            try:
                ncfg["roles"] = [NodeRole(raw["node_role"])]
            except (ValueError, KeyError):
                pass

        return cls(**ncfg)


def is_active_now(config: NodeConfig) -> bool:
    """Check if the node should be active based on configured schedule.
    
    Returns True if:
    - No schedule configured (always on), OR
    - Current time is within active hours AND current day is in active_days
    
    Args:
        config: NodeConfig with active_hours_start, active_hours_end, active_days
        
    Returns:
        True if node should accept work, False otherwise
    """
    # Always on if no schedule configured
    if config.active_hours_start is None or config.active_hours_end is None:
        return True
    
    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()  # 0=Monday, 6=Sunday
    
    # Check if today is an active day
    if config.active_days and current_day not in config.active_days:
        return False
    
    start, end = config.active_hours_start, config.active_hours_end
    
    # Handle wrap-around (e.g., 22:00 - 06:00)
    if start <= end:
        # Normal range (e.g., 09:00 - 17:00)
        return start <= current_hour < end
    else:
        # Wraps midnight (e.g., 22:00 - 06:00)
        return current_hour >= start or current_hour < end
