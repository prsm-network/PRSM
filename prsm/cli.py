"""
PRSM Command Line Interface
Main entry point for PRSM CLI commands
"""

import asyncio
import socket
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Forward references for type annotations — the actual NodeConfig class
    # is imported inside functions that need it at runtime (setup wizard /
    # configure flow) to avoid loading the full node stack at module import.
    from prsm.node.config import NodeConfig  # noqa: F401
from urllib.parse import urlparse

import os

import click
import uvicorn
from rich.console import Console
from rich.table import Table


def detect_available_backends() -> dict:
    """
    Check which LLM backends have valid API keys configured.

    Inlined in v1.6.0 from the removed ``prsm.compute.nwtn.backends.config``
    module — the legacy backends subsystem was deleted as part of the scope
    alignment refactor (PR 2). PRSM users now rely on third-party LLMs
    (local or via OAuth/API); this helper only needs to inspect environment
    variables to decide whether a real backend is available.
    """
    anthropic_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
    openai_available = bool(os.environ.get("OPENAI_API_KEY"))
    local_available = bool(os.environ.get("PRSM_LOCAL_MODEL_URL"))
    return {
        "anthropic": anthropic_available,
        "openai": openai_available,
        "local": local_available,
        "any_real_backend": anthropic_available or openai_available or local_available,
    }


console = Console()

# ---------------------------------------------------------------------------
# AI-agent-centric output helpers (Sub-phase 10.6)
# ---------------------------------------------------------------------------

def _agent_output(data: dict, format: str = "json") -> None:
    """
    Write structured output to stdout.

    In ``json`` mode (the default for agent callers) the output is a single
    valid JSON document — no colour, no Rich formatting, no spinners.
    In ``text`` mode the caller should render with Rich as normal.

    All agent-callable commands should call this helper so the same command
    works for both human inspection and programmatic consumption.

    Conventions:
    - Success:  exit 0,  ``{"ok": true, ...}``
    - Failure:  exit 1,  ``{"ok": false, "error": "..."}``
    """
    import json
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
    sys.stdout.flush()


def _agent_error(message: str, code: int = 1) -> None:
    """Write a JSON error to stdout and exit with *code*."""
    import json
    sys.stdout.write(json.dumps({"ok": False, "error": message}) + "\n")
    sys.stdout.flush()
    raise SystemExit(code)


def _run_async(coro):
    """
    Run *coro* and return its result regardless of whether an event loop is
    already running in the current thread (e.g. inside pytest-asyncio).

    Strategy:
    - Try ``asyncio.run()`` in the current thread first (fast path, works in
      production where no loop is running).
    - If a loop is already running, spawn a daemon thread with a fresh loop
      and block until completion (safe, no nesting required).
    """
    import threading

    try:
        asyncio.get_running_loop()
        # A loop IS running — use a separate thread
        result_box: list = [None]
        exc_box:    list = [None]

        def _worker():
            try:
                result_box[0] = asyncio.run(coro)
            except Exception as e:
                exc_box[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if exc_box[0]:
            raise exc_box[0]
        return result_box[0]

    except RuntimeError:
        # No loop running — safe to call asyncio.run() directly
        return asyncio.run(coro)


PREFLIGHT_PASS = "PASS"
PREFLIGHT_WARN = "WARN"
PREFLIGHT_FAIL = "FAIL"
PREFLIGHT_BOOTSTRAP_TIMEOUT_SECONDS = 1.5


# ── Credential storage ──────────────────────────────────────────────────────
# Credentials are stored at ~/.prsm/credentials.json (chmod 600).
# They contain the JWT access token obtained by `prsm login`.

_CREDENTIALS_FILE = Path.home() / ".prsm" / "credentials.json"


def _load_credentials() -> Optional[dict]:
    """Return stored credentials dict, or None if not logged in."""
    if _CREDENTIALS_FILE.exists():
        try:
            import json
            return json.loads(_CREDENTIALS_FILE.read_text())
        except Exception:
            return None
    return None


def _save_credentials(data: dict) -> None:
    """Write credentials to disk with restricted permissions (owner-only read)."""
    import json
    from prsm.node.identity import write_owner_only_file
    # sp1266 — atomic 0o600 write (no world-readable window between create + chmod, the
    # TOCTOU the round-4 audit flagged for this JWT credentials file).
    write_owner_only_file(_CREDENTIALS_FILE, json.dumps(data, indent=2))


def _clear_credentials() -> None:
    """Remove stored credentials file."""
    if _CREDENTIALS_FILE.exists():
        _CREDENTIALS_FILE.unlink()


def _auth_headers() -> dict:
    """
    Return a dict with Authorization header, or empty dict if not logged in.

    Used by all commands that call authenticated API endpoints. Empty dict
    causes the request to proceed without auth (resulting in a 401 which
    the command then surfaces as "Session expired. Run: prsm login").
    """
    creds = _load_credentials()
    if creds and creds.get("access_token"):
        return {"Authorization": f"Bearer {creds['access_token']}"}
    return {}


def _api_url_from_creds(override: Optional[str]) -> str:
    """
    Return the API URL to use: explicit override → stored credential → default.
    """
    if override:
        return override.rstrip("/")
    creds = _load_credentials()
    if creds and creds.get("api_url"):
        return creds["api_url"].rstrip("/")
    return "http://127.0.0.1:8000"


def _node_api_key_headers() -> dict:
    """Sprint 1199 — the node-API-key auth header for daemon calls against an
    AUTHENTICATED node.

    NodeAuthMiddleware protects /compute/, /ftns/, /wallet/, ... when the operator
    sets PRSM_NODE_API_KEY (the common day-one public-bind posture, now easy via the
    sp1195 auto-provision). The CLI must therefore present that key or every inference
    / faucet call 401s. Sent as ``X-API-Key`` (the middleware accepts it) so it never
    collides with a user-login ``Authorization: Bearer`` JWT (_auth_headers). Empty
    when PRSM_NODE_API_KEY is unset (dev / loopback no-key node → no header needed)."""
    key = (os.environ.get("PRSM_NODE_API_KEY") or "").strip()
    return {"X-API-Key": key} if key else {}


@dataclass
class PreflightCheckResult:
    """Node startup preflight check result."""

    name: str
    status: str
    required: bool
    details: str
    remediation: str


def _probe_tcp_endpoint(host: str, port: int, timeout_seconds: float) -> tuple[bool, str]:
    """Attempt a bounded TCP connection probe to a host:port endpoint."""
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True, f"reachable at {host}:{port}"
    except Exception as exc:
        return False, f"unreachable at {host}:{port} ({exc})"


def _parse_endpoint(address: str, default_port: int) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Parse host/port endpoint from either URL or host:port style strings."""
    if not address or not address.strip():
        return None, None, "empty address"

    address = address.strip()

    try:
        if "://" in address:
            parsed = urlparse(address)
            if not parsed.hostname:
                return None, None, f"invalid URL: {address}"
            return parsed.hostname, int(parsed.port or default_port), None

        host, sep, port_text = address.rpartition(":")
        if sep and host:
            return host, int(port_text), None

        return address, default_port, None
    except Exception as exc:
        return None, None, f"unable to parse '{address}' ({exc})"


def parse_active_hours(value: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse active hours string like '22-8' or 'off'.
    
    Returns:
        Tuple of (start_hour, end_hour) where None values mean "always on".
    
    Raises:
        click.BadParameter: If the format is invalid.
    """
    if value is None:
        return None, None
    if value.lower() == "off":
        return None, None  # Always on
    try:
        parts = value.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            if 0 <= start <= 23 and 0 <= end <= 23:
                return start, end
    except (ValueError, AttributeError):
        pass
    raise click.BadParameter(f"Invalid active hours format: {value}. Use '22-8' or 'off'")


def parse_active_days(value: Optional[str]) -> List[int]:
    """Parse active days string like 'mon,tue,wed' or 'weekdays'.
    
    Returns:
        List of day numbers (0=Mon, 6=Sun). Empty list means every day.
    """
    if value is None:
        return []
    value = value.lower()
    if value == "weekdays":
        return [0, 1, 2, 3, 4]  # Mon-Fri
    if value == "weekends":
        return [5, 6]  # Sat-Sun
    day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    days = []
    for part in value.split(","):
        part = part.strip().lower()[:3]  # Take first 3 chars
        if part in day_map:
            days.append(day_map[part])
    return sorted(days)


def _should_warn_no_inference_backend(any_real_backend: bool, executor: Any) -> bool:
    """Sprint 1209 — decide whether to print the 'no real inference backend' startup
    warning. Pre-fix the node always cried 'inference will return mock responses' when no
    Anthropic/OpenAI key was set — MISLEADING for a node running PRSM_INFERENCE_EXECUTOR=
    local|parallax, which serves REAL /compute/inference with no API key (it confused the
    GPU bench). Warn only when there's NEITHER a real LLM-API backend NOR a real
    (non-mock) inference executor wired."""
    if any_real_backend:
        return False
    if executor is not None and "mock" not in type(executor).__name__.lower():
        return False
    return True


def _operator_wallet_preflight() -> PreflightCheckResult:
    """Wallet-config diagnostic — reports the resolved operator on-chain address.

    Mirrors ``resolve_operator_address()``'s real precedence so the preflight tells the
    truth for BOTH operator models: explicit ``PRSM_OPERATOR_ADDRESS`` (requester-payment /
    read-only operator) wins, else derive from ``FTNS_WALLET_PRIVATE_KEY``. Catches stale
    env / wrong-key misconfigurations before any on-chain action. Optional + fail-soft.
    """
    explicit = os.environ.get("PRSM_OPERATOR_ADDRESS", "").strip()
    pk = os.environ.get("FTNS_WALLET_PRIVATE_KEY", "").strip()
    try:
        from prsm.node.operator_address import resolve_operator_address
        addr = resolve_operator_address()
    except Exception as exc:  # noqa: BLE001 — resolve is fail-soft, but never crash preflight
        return PreflightCheckResult(
            name="Wallet config (optional)",
            status=PREFLIGHT_WARN,
            required=False,
            details=f"Address derivation failed: {exc}",
            remediation="Check FTNS_WALLET_PRIVATE_KEY / PRSM_OPERATOR_ADDRESS format.",
        )

    if addr:
        source = (
            "PRSM_OPERATOR_ADDRESS"
            if explicit
            else "derived from FTNS_WALLET_PRIVATE_KEY"
        )
        return PreflightCheckResult(
            name="Wallet config (optional)",
            status=PREFLIGHT_PASS,
            required=False,
            details=f"Operator address: {addr[:8]}...{addr[-6:]} ({source})",
            remediation="None",
        )

    if pk:
        # Key present but resolve_operator_address() returned None → unparseable key
        # (and no explicit override to fall back on).
        return PreflightCheckResult(
            name="Wallet config (optional)",
            status=PREFLIGHT_WARN,
            required=False,
            details="FTNS_WALLET_PRIVATE_KEY set but malformed",
            remediation=(
                "Verify private key is 0x-prefixed 32 bytes of hex, or set "
                "PRSM_OPERATOR_ADDRESS explicitly. Bad key won't crash startup but "
                "all on-chain operations will fail."
            ),
        )

    # Neither source configured.
    return PreflightCheckResult(
        name="Wallet config (optional)",
        status=PREFLIGHT_WARN,
        required=False,
        details="operator address not configured",
        remediation=(
            "Set PRSM_OPERATOR_ADDRESS (explicit on-chain EOA — requester-payment / "
            "read-only operators) or FTNS_WALLET_PRIVATE_KEY (derives the address and "
            "enables royalty claim / heartbeat / distribution triggers). Read-only "
            "paths work without either."
        ),
    )


def _node_preflight_diagnostics(config: "NodeConfig") -> List[PreflightCheckResult]:
    """Run non-breaking node startup diagnostics with required/optional classification."""

    checks: List[PreflightCheckResult] = []

    # Required: Python runtime basics.
    py_ok = sys.version_info >= (3, 10)
    checks.append(
        PreflightCheckResult(
            name="Python runtime",
            status=PREFLIGHT_PASS if py_ok else PREFLIGHT_FAIL,
            required=True,
            details=f"detected {sys.version.split()[0]}",
            remediation="Install Python 3.10+ and re-run 'prsm node start'.",
        )
    )

    # Required but non-blocking by itself: config presence (defaults are supported).
    config_exists = config.config_path.exists()
    checks.append(
        PreflightCheckResult(
            name="Node config file",
            status=PREFLIGHT_PASS if config_exists else PREFLIGHT_WARN,
            required=False,
            details=(
                f"found at {config.config_path}"
                if config_exists
                else f"missing at {config.config_path}; defaults will be used"
            ),
            remediation=(
                "Run 'prsm node start --wizard' to generate explicit node settings."
                if not config_exists
                else "None"
            ),
        )
    )

    # Required hard precondition: local API bind availability.
    api_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    api_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        api_sock.bind(("127.0.0.1", config.api_port))
        api_status = PREFLIGHT_PASS
        api_details = f"127.0.0.1:{config.api_port} is available"
    except Exception as exc:
        api_status = PREFLIGHT_FAIL
        api_details = f"127.0.0.1:{config.api_port} unavailable ({exc})"
    finally:
        api_sock.close()

    checks.append(
        PreflightCheckResult(
            name="Local API bind",
            status=api_status,
            required=True,
            details=api_details,
            remediation="Choose a free --api-port or stop the process using this port.",
        )
    )

    # Optional: quick bootstrap reachability probe (bounded timeout).
    bootstrap_nodes = list(config.bootstrap_nodes or [])
    if not bootstrap_nodes:
        checks.append(
            PreflightCheckResult(
                name="Bootstrap target reachability",
                status=PREFLIGHT_WARN,
                required=False,
                details="no bootstrap targets configured",
                remediation="Configure --bootstrap or node_config bootstrap_nodes for peer discovery.",
            )
        )
    else:
        any_reachable = False
        probe_notes: List[str] = []
        for address in bootstrap_nodes:
            host, port, parse_error = _parse_endpoint(address, default_port=config.p2p_port)
            if parse_error:
                probe_notes.append(f"{address}: {parse_error}")
                continue
            ok, detail = _probe_tcp_endpoint(host, port, PREFLIGHT_BOOTSTRAP_TIMEOUT_SECONDS)
            probe_notes.append(f"{address}: {detail}")
            if ok:
                any_reachable = True
                break

        checks.append(
            PreflightCheckResult(
                name="Bootstrap target reachability",
                status=PREFLIGHT_PASS if any_reachable else PREFLIGHT_WARN,
                required=False,
                details=(
                    "at least one target reachable"
                    if any_reachable
                    else "; ".join(probe_notes[:2]) or "no reachable targets"
                ),
                remediation=(
                    "None"
                    if any_reachable
                    else "Startup will continue in degraded mode; verify DNS/network or update bootstrap targets."
                ),
            )
        )

    # Optional dependency: ContentStore availability.
    try:
        from prsm.storage import get_content_store
        store_available = get_content_store() is not None
    except Exception:
        store_available = False
    checks.append(
        PreflightCheckResult(
            name="ContentStore (optional)",
            status=PREFLIGHT_PASS if store_available else PREFLIGHT_WARN,
            required=False,
            details="Native ContentStore initialized" if store_available else "ContentStore not initialized",
            remediation=(
                "None"
                if store_available
                else "ContentStore will be auto-initialized on node start."
            ),
        )
    )

    # Wallet config diagnostic. Operators see the resolved on-chain operator address
    # pre-startup (from PRSM_OPERATOR_ADDRESS or derived from FTNS_WALLET_PRIVATE_KEY)
    # so they can confirm they're running with the wallet they expect, and catch stale
    # env / wrong-key misconfigurations before any on-chain action.
    checks.append(_operator_wallet_preflight())

    # ── Sprint 1198 — day-one-live readiness: will a REAL user get REAL results? ──
    # These aggregate the front-door work (sp1183-1197) into the pre-start verdict so
    # `prsm node start` (and the preflight) tells the operator whether the node will
    # actually serve, resolve its chain, refuse to start on an insecure bind, and
    # honor paid requests — BEFORE it boots. Each check is cheap + fail-soft.

    # 1. Inference serving by default (sp1184).
    try:
        from prsm.node.inference_wiring import (
            ml_inference_available, resolve_inference_executor_kind,
        )
        _raw = (os.environ.get("PRSM_INFERENCE_EXECUTOR") or "").strip() or None
        _kind = resolve_inference_executor_kind(
            _raw, ml_available=ml_inference_available())
        checks.append(PreflightCheckResult(
            name="Inference serving",
            status=PREFLIGHT_PASS if _kind else PREFLIGHT_WARN,
            required=False,
            details=(f"executor '{_kind}' will serve /compute/inference"
                     if _kind else
                     "no inference executor (ML deps absent or executor disabled) — "
                     "/compute/inference will 503"),
            remediation=("None" if _kind else
                         "pip install -e '.[ml]' for real local inference by default, "
                         "or set PRSM_INFERENCE_EXECUTOR=local|parallax|...."),
        ))
    except Exception as exc:  # noqa: BLE001
        checks.append(PreflightCheckResult(
            name="Inference serving", status=PREFLIGHT_WARN, required=False,
            details=f"readiness check failed: {exc}",
            remediation="Verify the prsm.node.inference_wiring import."))

    # 2. Network + contract addresses resolve (sp1183).
    try:
        from prsm.config.networks import resolve_endpoints
        _ep = resolve_endpoints()
        _have = bool(getattr(_ep, "ftns_token", None) and getattr(_ep, "escrow_pool", None))
        checks.append(PreflightCheckResult(
            name="Network + contracts",
            status=PREFLIGHT_PASS if _have else PREFLIGHT_WARN,
            required=False,
            details=(f"{_ep.network_name} (chainId {_ep.chain_id}): FTNS + EscrowPool resolved"
                     if _have else
                     f"{_ep.network_name} (chainId {_ep.chain_id}): FTNS/EscrowPool address missing"),
            remediation=("None" if _have else
                         "Set PRSM_NETWORK=mainnet|testnet to a network with deployed contracts."),
        ))
    except Exception as exc:  # noqa: BLE001
        checks.append(PreflightCheckResult(
            name="Network + contracts", status=PREFLIGHT_WARN, required=False,
            details=f"network resolution failed: {exc}",
            remediation="Set PRSM_NETWORK to a known network."))

    # 3. Public-bind auth posture (sp1011 fail-closed gate + sp1195 provisioning).
    #    This is the one that predicts a REFUSE-TO-START before it happens.
    try:
        from prsm.node.node import (
            assess_public_bind_auth_posture, should_refuse_insecure_public_bind,
            decide_api_key_provisioning, _default_node_api_key_path,
        )
        _truthy = {"1", "true", "yes", "on"}
        _api_host = str(getattr(config, "api_host", None) or "127.0.0.1")
        try:
            _persisted = "x" if _default_node_api_key_path().exists() else None
        except Exception:  # noqa: BLE001
            _persisted = None
        _auto = (os.environ.get("PRSM_AUTO_PROVISION_API_KEY") or "").strip().lower() in _truthy
        _action, _ = decide_api_key_provisioning(
            api_host=_api_host, env_key=os.environ.get("PRSM_NODE_API_KEY", ""),
            persisted_key=_persisted, auto_provision=_auto)
        _will_have_key = _action in ("use_env", "use_persisted", "generate")
        _posture, _ = assess_public_bind_auth_posture(
            listen_host=_api_host, api_key_present=_will_have_key)
        _allow = (os.environ.get("PRSM_ALLOW_INSECURE_PUBLIC_BIND") or "").strip().lower() in _truthy
        if _posture == "ok":
            checks.append(PreflightCheckResult(
                name="API auth posture", status=PREFLIGHT_PASS, required=True,
                details=f"{_api_host}: loopback or authenticated", remediation="None"))
        elif should_refuse_insecure_public_bind(_posture, allow_insecure=_allow):
            checks.append(PreflightCheckResult(
                name="API auth posture", status=PREFLIGHT_FAIL, required=True,
                details=(f"{_api_host}: public bind with no API key — the node will "
                         "REFUSE to start (money/KYC endpoints would be unauthenticated)"),
                remediation=("Set PRSM_NODE_API_KEY, or PRSM_AUTO_PROVISION_API_KEY=1 to "
                             "self-provision one, or bind 127.0.0.1 behind a reverse proxy.")))
        else:
            checks.append(PreflightCheckResult(
                name="API auth posture", status=PREFLIGHT_WARN, required=False,
                details=f"{_api_host}: public + unauthenticated (PRSM_ALLOW_INSECURE_PUBLIC_BIND ack'd)",
                remediation="Front with an authenticating reverse proxy in production."))
    except Exception as exc:  # noqa: BLE001
        checks.append(PreflightCheckResult(
            name="API auth posture", status=PREFLIGHT_WARN, required=False,
            details=f"posture check failed: {exc}", remediation="None"))

    # 4. Requester-payment readiness — ONLY when the operator opted in (sp1056/1196).
    if (os.environ.get("PRSM_REQUESTER_PAYMENT") or "").strip().lower() in {
            "1", "true", "yes", "on"}:
        try:
            from prsm.node.operator_address import resolve_operator_address
            _addr = resolve_operator_address()
            _settler = (os.environ.get("FTNS_WALLET_PRIVATE_KEY") or "").strip()
            if _addr and _settler:
                checks.append(PreflightCheckResult(
                    name="Requester-payment readiness", status=PREFLIGHT_PASS,
                    required=False,
                    details="PRSM_REQUESTER_PAYMENT on; operator payee + settler key present",
                    remediation="None"))
            else:
                _missing = []
                if not _addr:
                    _missing.append("operator payee address")
                if not _settler:
                    _missing.append("funded settler key (FTNS_WALLET_PRIVATE_KEY)")
                checks.append(PreflightCheckResult(
                    name="Requester-payment readiness", status=PREFLIGHT_WARN,
                    required=False,
                    details=("PRSM_REQUESTER_PAYMENT on but missing: " + ", ".join(_missing)
                             + " — paid requests will be rejected (402) or won't settle"),
                    remediation="Set FTNS_WALLET_PRIVATE_KEY (derives the payee + signs settle)."))
        except Exception as exc:  # noqa: BLE001
            checks.append(PreflightCheckResult(
                name="Requester-payment readiness", status=PREFLIGHT_WARN, required=False,
                details=f"readiness check failed: {exc}", remediation="None"))

    return checks


def _render_preflight_summary(results: List[PreflightCheckResult]) -> None:
    """Render startup diagnostics summary with remediation hints."""
    table = Table(title="Node Startup Preflight Diagnostics")
    table.add_column("Check", style="cyan")
    table.add_column("Req", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="green")
    table.add_column("Remediation", style="yellow")

    for result in results:
        status_style = {
            PREFLIGHT_PASS: "green",
            PREFLIGHT_WARN: "yellow",
            PREFLIGHT_FAIL: "red",
        }.get(result.status, "white")
        table.add_row(
            result.name,
            "required" if result.required else "optional",
            f"[{status_style}]{result.status}[/{status_style}]",
            result.details,
            result.remediation,
        )

    console.print()
    console.print(table)
    console.print()


def _has_hard_preflight_failures(results: List[PreflightCheckResult]) -> bool:
    """True when a required preflight check has failed."""
    return any(r.required and r.status == PREFLIGHT_FAIL for r in results)


def _should_announce_degraded_mode(results: List[PreflightCheckResult]) -> bool:
    """Determine whether startup should explicitly mention degraded mode continuation."""
    for result in results:
        if result.name == "Bootstrap target reachability" and result.status in (PREFLIGHT_WARN, PREFLIGHT_FAIL):
            return True
    return False


def _init_config():
    """Initialize config manager with defaults so get_settings() works.

    This must be called before starting uvicorn so that when the app
    module is imported, get_settings()/get_config() return valid objects.
    """
    from prsm.core.config import ConfigManager
    manager = ConfigManager()
    if manager.get_config() is None:
        manager.load_config()


def _get_debug() -> bool:
    """Get debug setting, defaulting to False if config unavailable."""
    try:
        _init_config()
        from prsm.core.config import get_settings
        s = get_settings()
        return getattr(s, 'debug', False) if s else False
    except Exception:
        return False


def _get_version():
    """Read version from package metadata (single source of truth: pyproject.toml).

    Sprint 150 — fallback now reads `prsm.__version__` instead of a
    stale literal. The hardcoded "0.24.0" leaked into bootstrap
    server registrations + assorted UA strings on every importlib
    metadata-miss path.
    """
    try:
        from importlib.metadata import version as pkg_version
        return pkg_version("prsm-network")
    except Exception:
        try:
            import prsm as _prsm_pkg
            return _prsm_pkg.__version__
        except Exception:
            return "unknown"


def _prsm_appears_configured() -> bool:
    """sp1258 — True once the user is past the brand-new, never-touched state, so the
    "Run: prsm setup" nudge is suppressed.

    ``~/.prsm/config.yaml`` is the canonical marker, BUT it is not the only way to be
    configured: a node identity (``~/.prsm/identity.json``, written on first
    ``node start``) or explicit PRSM_*/wallet env means the user is actively running
    PRSM — env-driven deploys and SSH-driven node runs never write config.yaml yet work
    fine, and the nudge was firing on EVERY working command (faucet/deposit/pay-infer/
    infer), undermining confidence. Genuinely-fresh users (no config, no identity, no
    env) still get nudged."""
    import os
    home = Path.home() / ".prsm"
    if (home / "config.yaml").exists() or (home / "identity.json").exists():
        return True
    return any(
        (os.environ.get(v) or "").strip()
        for v in (
            "PRSM_NETWORK", "PRSM_INFERENCE_EXECUTOR", "FTNS_WALLET_PRIVATE_KEY",
            "PRIVATE_KEY", "PRSM_NODE_API_KEY", "PRSM_OPERATOR_ADDRESS",
        )
    )


@click.group()
@click.version_option(version=_get_version(), prog_name="PRSM")
@click.pass_context
def main(ctx):
    """
    PRSM: Protocol for Research, Storage, and Modeling

    A peer-to-peer protocol unifying data, compute, and economic layers.
    Frontier LLMs (Claude, GPT, Gemini, or local) call PRSM via MCP as
    a retrieval + heavy-compute substrate. Contributors earn FTNS for
    sharing storage, compute, and data.
    """
    # Auto-migrate old node_config.json -> config.yaml for existing users
    try:
        from prsm.cli_modules.migration import migrate_if_needed
        migrate_if_needed()
    except Exception:
        pass  # never block CLI startup on migration issues

    # First-run auto-detection: nudge GENUINELY-unconfigured users toward setup.
    # sp1258 — suppress once the user is past the brand-new state (node identity or
    # PRSM_*/wallet env present), so the nudge stops firing on working commands.
    if not _prsm_appears_configured():
        # Don't nag on setup, help, or version invocations
        invoked = ctx.invoked_subcommand or ""
        safe_commands = {"setup", None}  # None = no subcommand (shows help)
        if invoked not in safe_commands:
            click.echo("  ◇ PRSM is not configured yet. Run: prsm setup")
            click.echo()


# ── Theme + icons (used by config/mcp subcommands below) ─────────────
from prsm.cli_modules.theme import THEME, ICONS  # noqa: E402, F401

# ── Skills CLI (Phase 4.3) ───────────────────────────────────────────
from prsm.cli_modules.skills_cli import skills as skills_group  # noqa: E402
main.add_command(skills_group, "skills")


# ── On-chain Provenance CLI (Phase 1) ────────────────────────────────
from prsm.cli_modules.provenance import provenance as provenance_group  # noqa: E402
main.add_command(provenance_group, "provenance")


@main.command()
def version():
    """Print the installed PRSM package version.

    Reads from importlib.metadata.version("prsm-network") so
    it stays in sync with pyproject.toml across releases.
    """
    try:
        from importlib.metadata import version as _pkg_version
        v = _pkg_version("prsm-network")
        console.print(f"PRSM {v}")
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]Could not resolve PRSM version: {exc}[/yellow]\n"
            f"[dim]Install via `pip install -e .` or check "
            f"`pip show prsm-network`[/dim]"
        )
        sys.exit(1)


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: localhost for security)")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--workers", default=1, help="Number of worker processes")
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the PRSM API server.

    DEPRECATED: Use `prsm node start` for full P2P connectivity,
    or `prsm daemon start` for background operation.
    """
    console.print()
    console.print("  prsm serve is deprecated.", style="yellow")
    console.print("  Use one of these instead:", style="yellow")
    console.print("    prsm node start         -- Full P2P node with dashboard", style="dim")
    console.print("    prsm node start --no-dashboard  -- Full P2P, static output", style="dim")
    console.print("    prsm daemon start       -- Background daemon mode", style="dim")
    console.print()

    console.print(f"🚀 Starting PRSM server on {host}:{port}", style="bold green")
    _init_config()

    if reload and workers > 1:
        console.print("⚠️  Cannot use --reload with multiple workers", style="yellow")
        workers = 1

    uvicorn.run(
        "prsm.interface.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info" if not _get_debug() else "debug"
    )


@main.command()
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format: 'text' (human) or 'json' (agent-parseable)")
def status(output_format: str):
    """Show PRSM system status.

    Use --format json for machine-readable output (AI agent consumption).
    """
    _init_config()
    from prsm.core.config import get_settings
    settings = get_settings()

    # Sprint 533 F53 fix: when ~/.prsm/config.yaml doesn't exist,
    # the global "PRSM is not configured" nudge fires above, but
    # then the table below showed "Configuration: ✅ Loaded" using
    # defaults. Contradictory signals confuse new users. Detect
    # the unconfigured state + reflect it in the table.
    config_path = Path.home() / ".prsm" / "config.yaml"
    config_status = "loaded" if config_path.exists() else "not_configured"

    if settings:
        data = {
            "ok": True,
            "components": {
                "configuration": {
                    "status": config_status,
                    "environment": getattr(settings, "environment", "unknown"),
                },
                "database": {
                    "status": "configured",
                    "url": str(getattr(settings, "database_url", "sqlite (default)")),
                },
                "storage": {
                    "status": "configured",
                    "data_dir": getattr(settings, "storage_data_dir", "~/.prsm/storage"),
                },
                "nwtn": {
                    "enabled": getattr(settings, "nwtn_enabled", True),
                    "model": getattr(settings, "nwtn_default_model", "default"),
                },
                "ftns": {
                    "enabled": getattr(settings, "ftns_enabled", True),
                    "initial_grant": getattr(settings, "ftns_initial_grant", 100),
                },
            },
        }
    else:
        data = {
            "ok": True,
            "components": {
                "configuration": {"status": "defaults", "note": "No .env file found"},
                "nwtn": {"enabled": True, "model": "default"},
                "ftns": {"enabled": True, "initial_grant": 100},
            },
        }

    if output_format == "json":
        _agent_output(data)
        return

    table = Table(title="PRSM System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")
    for name, info in data["components"].items():
        enabled = info.get("enabled", True)
        status_text = "✅ Loaded" if info.get("status") == "loaded" else \
                      "⚠️ Defaults" if info.get("status") == "defaults" else \
                      "⚠️ Not configured" if info.get("status") == "not_configured" else \
                      "✅ Enabled" if enabled else "❌ Disabled"
        detail = info.get("url", info.get("model", info.get("note", "")))
        table.add_row(name.capitalize(), status_text, str(detail))
    console.print(table)


@main.command()
@click.option("--port", default=8501, help="Port for the Streamlit dashboard")
@click.option("--api-port", default=8000, help="Port for the PRSM API")
def dashboard(port: int, api_port: int):
    """Launch the high-fidelity PRSM Dashboard and API"""
    import subprocess
    import time
    import socket

    _init_config()

    console.print("🚀 Starting PRSM Command Center...", style="bold green")

    # Sprint 537 F67 fix: detect if a daemon is already listening on
    # api_port. If yes, reuse it (the dashboard talks to /rings/status
    # etc. which the daemon's prsm.node.api serves). If no, spawn the
    # interface API as a fallback. Pre-fix unconditionally spawned a
    # second uvicorn that conflicted with the daemon on :8000 and died
    # silently, leaving the dashboard talking to the wrong API surface.
    api_process = None
    api_already_running = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            api_already_running = (
                s.connect_ex(("127.0.0.1", api_port)) == 0
            )
    except Exception:
        api_already_running = False

    if api_already_running:
        console.print(
            f"  ✓ API already running on port {api_port} — reusing existing daemon",
            style="dim",
        )
    else:
        console.print(
            f"📡 No daemon detected — launching fallback API on port {api_port}...",
            style="dim",
        )
        api_process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "prsm.interface.api.main:app",
                "--port", str(api_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give the API a moment to spin up
        time.sleep(2)
    
    # 2. Launch Streamlit
    console.print(f"🎨 Launching Dashboard UI on port {port}...", style="dim")
    
    # Find the correct Python executable (use venv if available)
    # Try python3.14 first, then python, then sys.executable
    venv_base = Path(__file__).parent.parent / ".venv" / "bin"
    for py_name in ["python3.14", "python"]:
        venv_python = venv_base / py_name
        if venv_python.exists():
            python_exe = str(venv_python)
            break
    else:
        python_exe = sys.executable
    
    dashboard_path = Path(__file__).parent / "interface" / "dashboard" / "streamlit_app.py"
    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", 
        str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "false"
    ]
    
    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        console.print("\n👋 Closing Command Center...", style="bold yellow")
    finally:
        # Sprint 537 F67 fix: only kill the API process if WE spawned
        # it. Pre-fix: api_process.terminate() always ran — when a
        # daemon was already running and we reused it, this line
        # would AttributeError on None (post-fix) or wrongly kill the
        # operator's daemon (pre-fix never had this branch).
        if api_process is not None:
            api_process.terminate()


@main.command()
@click.argument("query", required=True)
@click.option("--context", "-c", default=100, help="FTNS context allocation")
@click.option("--user-id", default="cli-user", help="User ID for the query")
@click.option("--api-url", default="http://127.0.0.1:8000", help="PRSM API URL")
def query(query: str, context: int, user_id: str, api_url: str):
    """Submit a query to a running PRSM node (dev / testing convenience).

    The recommended end-user flow is: run ``prsm mcp-server`` and configure
    your LLM client (Claude Desktop / Gemini CLI / local MCP-compatible
    tool) to point at it. Your LLM will then invoke PRSM tools directly
    whenever a query needs retrieval, heavy compute, or provenance.

    This ``query`` command hits the legacy ``/query`` REST endpoint
    directly without going through an LLM. Useful for development and
    integration testing; not the primary user flow.
    """
    import httpx
    from rich.panel import Panel

    console.print(f"Submitting query to PRSM node at {api_url}...", style="bold blue")
    console.print(f"Query: {query}")
    console.print(f"Context allocation: {context} FTNS")

    payload = {
        "prompt": query,
        "context_allocation": str(context),
        "user_id": user_id
    }

    try:
        with console.status("[bold green]PRSM node processing query..."):
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{api_url}/query", json=payload)

        if response.status_code == 200:
            data = response.json()
            answer = data.get("final_answer", "No answer provided.")

            console.print("\n")
            console.print(Panel(answer, title="[bold green]PRSM Response[/bold green]", border_style="green"))

            trace = data.get("reasoning_trace", [])
            if trace:
                console.print("\n[bold cyan]Reasoning Trace:[/bold cyan]")
                for i, step in enumerate(trace, 1):
                    action = step.get("input_data", {}).get("action", str(step.get("agent_type", "Process")))
                    console.print(f"  [dim]{i}.[/dim] {action}")

            conf = data.get('confidence_score', 0)
            ctx = data.get('context_used', 0)
            console.print(f"\n[dim]Confidence Score: {conf:.2f} | Context Used: {ctx} tokens[/dim]")
        elif response.status_code == 404:
            # Sprint 533 F47 fix: /query was retired in favor of
            # /compute/forge. Surface actionable hint instead of
            # opaque "Not Found".
            console.print(
                "\n[bold yellow]⚠️  Legacy `/query` endpoint not found.[/bold yellow]"
            )
            console.print(
                "The `/query` route was retired. Use one of:\n"
                "  • [cyan]prsm compute run --query \"...\"[/cyan]  — full forge pipeline (Rings 1-10)\n"
                "  • [cyan]prsm compute submit --prompt \"...\"[/cyan]  — single-shot inference\n"
                "  • [cyan]prsm mcp-server[/cyan] + configure your LLM (Claude/Gemini)\n"
                "  • Direct: [cyan]curl -X POST {api}/compute/forge[/cyan]"
                .format(api=api_url)
            )
            raise SystemExit(1)
        else:
            console.print(f"\n[bold red]Error ({response.status_code}):[/bold red] {response.text}")
            raise SystemExit(1)

    except httpx.RequestError as e:
        console.print(f"\n[bold red]Connection Error:[/bold red] Could not connect to {api_url}.")
        console.print("Make sure the PRSM API server is running (`prsm serve`).")
        console.print(f"[dim]Details: {e}[/dim]")
        raise SystemExit(1)


@main.command()
def init():
    """Initialize PRSM configuration and database.

    DEPRECATED: Use `prsm setup` for the full interactive setup experience.
    """
    console.print()
    console.print("  prsm init is deprecated.", style="yellow")
    console.print("  Use one of these instead:", style="yellow")
    console.print("    prsm setup              -- Interactive setup wizard", style="dim")
    console.print("    prsm setup --minimal    -- Quick setup with defaults", style="dim")
    console.print("    prsm daemon start       -- Start the daemon directly", style="dim")
    console.print()

    console.print("🔧 Initializing PRSM...", style="bold blue")

    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            console.print("📄 Copying .env.example to .env")
            env_file.write_text(env_example.read_text())
        else:
            console.print("❌ No .env.example found", style="red")
            return

    console.print("✅ Configuration ready")
    console.print("Next steps:")
    console.print("1. Edit .env with your configuration")
    console.print("2. Run: prsm db-upgrade")
    console.print("3. Run: prsm daemon start")


@main.command()
@click.option("--username", "-u", prompt="Username", help="PRSM account username")
@click.option(
    "--password", "-p",
    prompt="Password", hide_input=True,
    help="PRSM account password"
)
@click.option(
    "--api-url",
    default="http://127.0.0.1:8000",
    show_default=True,
    help="PRSM API base URL"
)
def login(username: str, password: str, api_url: str) -> None:
    """
    Log in to PRSM and save credentials for subsequent commands.

    Credentials (JWT access token) are stored at ~/.prsm/credentials.json
    with owner-only read permissions (chmod 600). Run `prsm logout` to
    remove them.
    """
    import httpx

    api_url = api_url.rstrip("/")
    console.print(f"🔑 Logging in to {api_url} as '{username}'...", style="bold blue")

    try:
        response = httpx.post(
            f"{api_url}/api/v1/auth/login",
            json={"username": username, "password": password},
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {api_url}", style="red")
        console.print("💡 Make sure the PRSM server is running: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        _save_credentials({
            "access_token":  data["access_token"],
            "refresh_token": data["refresh_token"],
            "api_url":       api_url,
            "username":      username,
        })
        expires_min = data.get("expires_in", 1800) // 60
        console.print(f"✅ Logged in as '{username}'", style="bold green")
        console.print(
            f"   Credentials saved to {_CREDENTIALS_FILE} "
            f"(token expires in {expires_min} min)",
            style="dim"
        )
    elif response.status_code == 401:
        console.print("❌ Invalid username or password", style="red")
        raise SystemExit(1)
    else:
        console.print(
            f"❌ Login failed: HTTP {response.status_code}",
            style="red"
        )
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@main.command()
def logout() -> None:
    """Remove stored PRSM credentials."""
    creds = _load_credentials()
    if creds:
        username = creds.get("username", "unknown")
        _clear_credentials()
        console.print(f"✅ Logged out (was: '{username}')", style="green")
        console.print(f"   Removed {_CREDENTIALS_FILE}", style="dim")
    else:
        console.print("Not currently logged in.", style="dim")


def _find_alembic_ini() -> Optional[Path]:
    """
    Locate alembic.ini, checking two locations in priority order:
    1. Parent of the directory containing cli.py (works for editable installs
       and direct invocation from the source tree)
    2. Current working directory (works when running `prsm` from the project root
       after a standard `pip install`)
    Returns the Path if found, None otherwise.
    """
    candidates = [
        Path(__file__).parent.parent / "alembic.ini",  # source tree
        Path.cwd() / "alembic.ini",                    # cwd fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@main.command()
@click.option(
    "--revision",
    default="head",
    show_default=True,
    help="Target revision label or ID. Use 'head' for latest, '+1'/'-1' for relative steps.",
)
def db_upgrade(revision: str) -> None:
    """Upgrade database schema to the target revision (default: head)."""
    console.print(f"🗄️   Upgrading database schema → '{revision}'...", style="bold blue")

    alembic_ini = _find_alembic_ini()
    if alembic_ini is None:
        console.print(
            "❌ alembic.ini not found. Run this command from the PRSM project root.",
            style="red",
        )
        raise SystemExit(1)

    try:
        from alembic.config import Config
        from alembic import command as alembic_command

        cfg = Config(str(alembic_ini))
        alembic_command.upgrade(cfg, revision)
        console.print(f"✅ Schema upgraded to '{revision}' successfully.", style="green")
    except Exception as exc:
        console.print(f"❌ Migration failed: {exc}", style="red")
        console.print(
            "💡 Verify PRSM_DATABASE_URL is set and the database is reachable.",
            style="yellow",
        )
        raise SystemExit(1)


@main.command()
@click.option(
    "--revision",
    default="-1",
    show_default=True,
    help="Target revision. Use '-1' for one step back, 'base' to revert everything.",
)
def db_downgrade(revision: str) -> None:
    """Downgrade database schema (default: one step back).

    \b
    WARNING: downgrading drops columns and tables. Back up your data first.
    Common values:
      -1              one revision back (default)
      -2              two revisions back
      base            revert all migrations
      <revision-id>   specific revision ID from `prsm db-status`
    """
    console.print(f"🗄️   Downgrading database schema → '{revision}'...", style="bold blue")
    console.print("⚠️   This will drop columns/tables. Ensure you have a backup.", style="yellow")

    alembic_ini = _find_alembic_ini()
    if alembic_ini is None:
        console.print(
            "❌ alembic.ini not found. Run this command from the PRSM project root.",
            style="red",
        )
        raise SystemExit(1)

    try:
        from alembic.config import Config
        from alembic import command as alembic_command

        cfg = Config(str(alembic_ini))
        alembic_command.downgrade(cfg, revision)
        console.print(f"✅ Schema downgraded to '{revision}' successfully.", style="green")
    except Exception as exc:
        console.print(f"❌ Migration failed: {exc}", style="red")
        console.print(
            "💡 Verify PRSM_DATABASE_URL is set and the database is reachable.",
            style="yellow",
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Version management — update command
# ---------------------------------------------------------------------------


def _is_installed_via_pipx() -> bool:
    """Check if prsm-network was installed via pipx."""
    import subprocess as _sub
    result = _sub.run(
        ["pipx", "list", "--json"], capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return False
    import json
    try:
        data = json.loads(result.stdout)
        return "prsm-network" in data.get("venvs", {})
    except (json.JSONDecodeError, KeyError):
        return False


def _run_pipx_upgrade() -> None:
    """Upgrade prsm-network via pipx."""
    import subprocess as _sub
    console.print("[dim]Detected pipx installation. Upgrading via pipx...[/dim]")
    result = _sub.run(
        ["pipx", "upgrade", "prsm-network"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        console.print("[green]✓ PRSM upgraded successfully.[/green]")
        if result.stdout.strip():
            console.print(f"[dim]{result.stdout.strip()}[/dim]")
    else:
        console.print(f"[red]Upgrade failed:[/red]")
        console.print(f"[dim]{result.stderr or result.stdout}[/dim]")
        raise SystemExit(1)


def _run_pip_upgrade() -> None:
    """Upgrade prsm-network via pip."""
    import subprocess as _sub
    console.print("[dim]Detected pip installation. Upgrading via pip...[/dim]")
    result = _sub.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "prsm-network"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        console.print("[green]✓ PRSM upgraded successfully.[/green]")
    else:
        console.print(f"[red]Upgrade failed:[/red]")
        console.print(f"[dim]{result.stderr or result.stdout}[/dim]")
        raise SystemExit(1)


@main.command()
@click.option("--dry-run", is_flag=True, help="Show what would be updated without installing")
def update(dry_run: bool):
    """Check for and install PRSM updates.

    Automatically detects how PRSM was installed (pipx vs pip) and
    upgrades accordingly.
    """
    import importlib.metadata
    from packaging.version import Version as _V
    from packaging.version import InvalidVersion

    # Get installed version
    try:
        current_ver = importlib.metadata.version("prsm-network")
    except importlib.metadata.PackageNotFoundError:
        console.print("[red]PRSM is not installed as a package.[/red]")
        console.print("Install with:  pip install prsm-network")
        raise SystemExit(1)

    # Fetch latest from PyPI
    try:
        import httpx
        resp = httpx.get("https://pypi.org/pypi/prsm-network/json", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        latest_ver = data["info"]["version"]
    except Exception as e:
        console.print(f"[red]Could not check for updates: {e}[/red]")
        raise SystemExit(1)

    try:
        current = _V(current_ver)
        latest = _V(latest_ver)
    except InvalidVersion:
        console.print(f"[yellow]Cannot compare versions: installed={current_ver}, latest={latest_ver}[/yellow]")
        raise SystemExit(1)

    if current >= latest:
        console.print(f"[green]✓ PRSM is up to date (v{current_ver})[/green]")
        return

    console.print(f"Update available: v{current_ver} → v{latest_ver}")

    if dry_run:
        console.print("[dim]Dry run — no changes will be made.[/dim]")
        return

    # Detect install method and upgrade
    is_pipx = _is_installed_via_pipx()
    if is_pipx:
        _run_pipx_upgrade()
    else:
        _run_pip_upgrade()


# ---------------------------------------------------------------------------
# Database management
# ---------------------------------------------------------------------------


@main.command()
def db_status() -> None:
    """Show current database migration revision and pending migrations."""
    alembic_ini = _find_alembic_ini()
    if alembic_ini is None:
        console.print(
            "❌ alembic.ini not found. Run this command from the PRSM project root.",
            style="red",
        )
        raise SystemExit(1)

    try:
        from alembic.config import Config
        from alembic import command as alembic_command

        cfg = Config(str(alembic_ini))
        console.print("🗄️   Current database revision:", style="bold blue")
        alembic_command.current(cfg, verbose=True)
        console.print("\n📋 Migration history (latest first):", style="bold blue")
        alembic_command.history(cfg, indicate_current=True)
    except Exception as exc:
        console.print(f"❌ Could not check migration status: {exc}", style="red")
        console.print(
            "💡 Verify PRSM_DATABASE_URL is set and the database is reachable.",
            style="yellow",
        )
        raise SystemExit(1)


@main.command()
@click.option("--dry-run", is_flag=True, help="Walk through setup without saving")
@click.option("--minimal", is_flag=True, help="Quick setup with smart defaults")
@click.option("--reset", is_flag=True, help="Reset all settings and re-run setup")
def setup(dry_run, minimal, reset):
    """Interactive first-run setup wizard for PRSM."""
    from prsm.cli_modules.setup_wizard import run_setup_wizard
    run_setup_wizard(dry_run=dry_run, minimal=minimal, reset=reset)


@main.group()
def node():
    """P2P node management commands"""
    pass


def _run_node_wizard() -> "NodeConfig":
    """Deprecated: redirects to the new `prsm setup` wizard.

    Kept for backward compatibility with `prsm node start --wizard`.
    Saves a minimal NodeConfig so the existing node start flow continues
    to work with any CLI overrides (ports, resources, etc.).
    """
    from prsm.node.config import NodeConfig

    console.print()
    console.print("  prsm node start --wizard is deprecated.")
    console.print("  Redirecting to the new setup wizard.", style="yellow")
    console.print("  Tip: Run `prsm setup` directly for the full experience.", style="dim")

    # Run the new setup wizard
    from prsm.cli_modules.setup_wizard import run_setup_wizard
    run_setup_wizard(minimal=False)

    # Also try to run migration so the legacy NodeConfig picks up the
    # settings written by the new wizard (~/.prsm/config.yaml → node_config.json).
    try:
        from prsm.cli_modules.migration import migrate_if_needed
        migrate_if_needed()
    except Exception:
        pass

    # Return a loaded NodeConfig so the caller (node start) continues normally
    return NodeConfig.load()


@node.command()
@click.option("--wizard", is_flag=True, help="Run interactive setup wizard")
@click.option("--background", "-b", is_flag=True, help="Start as background daemon")
@click.option("--p2p-port", default=None, type=int, help="P2P listen port (default: 9001)")
@click.option("--api-port", default=None, type=int, help="API listen port (default: 8000)")
@click.option("--bootstrap", default=None, help="Bootstrap node address (host:port)")
@click.option("--no-dashboard", is_flag=True, help="Disable live dashboard (static output)")
@click.option("--cpu", default=None, type=int, help="CPU allocation % (10-90)")
@click.option("--memory", default=None, type=int, help="RAM allocation % (10-90)")
@click.option("--storage", default=None, type=float, help="Storage to pledge in GB")
@click.option("--jobs", default=None, type=int, help="Max concurrent compute jobs")
def start(wizard: bool, background: bool, p2p_port: int, api_port: int, bootstrap: str, no_dashboard: bool,
          cpu: Optional[int], memory: Optional[int], storage: Optional[float], jobs: Optional[int]):
    """Start a PRSM network node with real P2P connectivity.

    Runs in the foreground by default. Use --background to run as a
    background daemon (equivalent to 'prsm daemon start').
    """
    from prsm.node.config import NodeConfig

    host = "127.0.0.1"
    port = 8000

    # Background mode — route to daemon
    if background:
        from prsm.cli_modules.daemon import daemon_start as _dstart
        _dstart(host=host, port=api_port or 8000)
        return

    if wizard:
        config = _run_node_wizard()
    else:
        # Load existing config or use defaults
        config = NodeConfig.load()

    # CLI overrides
    if p2p_port is not None:
        config.p2p_port = p2p_port
    if api_port is not None:
        config.api_port = api_port
    if bootstrap:
        config.bootstrap_nodes = [b.strip() for b in bootstrap.split(",")]
    
    # Resource CLI overrides
    if cpu is not None:
        if not 10 <= cpu <= 90:
            raise click.BadParameter("CPU allocation must be between 10-90%")
        config.cpu_allocation_pct = cpu
    if memory is not None:
        if not 10 <= memory <= 90:
            raise click.BadParameter("Memory allocation must be between 10-90%")
        config.memory_allocation_pct = memory
    if storage is not None:
        if storage <= 0:
            raise click.BadParameter("Storage must be a positive value in GB")
        config.storage_gb = storage
    if jobs is not None:
        if jobs < 1:
            raise click.BadParameter("Max concurrent jobs must be at least 1")
        config.max_concurrent_jobs = jobs
    
    # Save config if any resource overrides were provided
    if any(v is not None for v in [cpu, memory, storage, jobs]):
        config.save()

    # Run preflight diagnostics before starting the node
    results = _node_preflight_diagnostics(config)
    _render_preflight_summary(results)

    if _has_hard_preflight_failures(results):
        print("\n❌ Preflight checks failed. Cannot start node.")
        print("   Fix the issues above and try again.")
        sys.exit(1)

    # When using the live dashboard, suppress startup noise so the
    # terminal is clean.  The dashboard captures prsm.node.* activity
    # via its own log handler.
    if not no_dashboard:
        import logging as _logging
        import warnings as _warnings

        # Suppress Python logging below ERROR during startup
        _logging.root.setLevel(_logging.ERROR)
        # Suppress structlog info messages (MCP tool registrations)
        _logging.getLogger("prsm.compute").setLevel(_logging.ERROR)
        _logging.getLogger("prsm.core").setLevel(_logging.ERROR)
        # Suppress Python warnings (optional dependency notices)
        _warnings.filterwarnings("ignore")

        # Suppress structlog output (uses its own rendering pipeline)
        try:
            import structlog
            structlog.configure(
                wrapper_class=structlog.make_filtering_bound_logger(_logging.ERROR),
            )
        except ImportError:
            pass

        # Suppress uvicorn's CancelledError traceback on shutdown
        _logging.getLogger("uvicorn.error").setLevel(_logging.CRITICAL)

        console.print()
        console.print("  Starting PRSM Node...", style="bold green")
        console.print(f"  🖥️   Dashboard:  http://localhost:{config.api_port}/", style="bold cyan")

    else:
        console.print()
        console.print("=" * 60, style="bold green")
        console.print("  Starting PRSM Node", style="bold green")
        console.print("=" * 60, style="bold green")

    async def _run():
        import logging as _logging
        from prsm.node.node import PRSMNode

        prsm_node = PRSMNode(config)
        await prsm_node.initialize()
        await prsm_node.start()

        # Check for inference backends and warn only if NONE is real (sp1209).
        backends = detect_available_backends()
        if _should_warn_no_inference_backend(
            bool(backends.get("any_real_backend")),
            getattr(prsm_node, "inference_executor", None),
        ):
            print()
            print("⚠️   No real inference backend — /compute/inference will be "
                  "unavailable or mock.")
            print("    Enable real inference either way:")
            print("      • local model:  PRSM_INFERENCE_EXECUTOR=local "
                  "(after: pip install -e '.[ml]')")
            print("      • or an LLM API: set ANTHROPIC_API_KEY / OPENAI_API_KEY in .env")
            print()

        try:
            if no_dashboard:
                # Static table + sleep loop (original behavior)
                status = await prsm_node.get_status()
                console.print()
                table = Table(title="PRSM Node Status")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Node ID", status["node_id"])
                table.add_row("Display Name", status["display_name"])
                table.add_row("Roles", ", ".join(status["roles"]))
                table.add_row("P2P Address", status["p2p_address"])
                table.add_row("API Address", status["api_address"])
                table.add_row("Dashboard", f"http://127.0.0.1:{config.api_port}/")
                table.add_row("FTNS Balance", f"{status['ftns_balance']:.2f}")
                bootstrap = status.get("peers", {}).get("bootstrap", {})
                if bootstrap.get("degraded_mode"):
                    table.add_row("Bootstrap", "DEGRADED local mode")
                    table.add_row(
                        "Limited Features",
                        "Remote peer discovery/collaboration may be unavailable until peers connect",
                    )
                elif bootstrap.get("success_node"):
                    table.add_row("Bootstrap", f"connected via {bootstrap['success_node']}")
                elif bootstrap.get("configured_nodes") == []:
                    table.add_row("Bootstrap", "none configured (first node/local mode)")
                if status.get("compute"):
                    res = status["compute"]["resources"]
                    table.add_row("CPU", f"{res['cpu_count']} cores")
                    table.add_row("RAM", f"{res['memory_total_gb']} GB")
                    table.add_row("GPU", res.get("gpu_name", "none") if res.get("gpu_available") else "none")
                if status.get("storage"):
                    st = status["storage"]
                    table.add_row("Storage", "connected" if st.get("storage_available", False) else "not available")
                    table.add_row("Storage Pledged", f"{st['pledged_gb']} GB")
                console.print(table)
                console.print()
                console.print("Node is running. Press Ctrl+C to stop.", style="bold")
                console.print()
                while True:
                    await asyncio.sleep(1)
            else:
                # Restore logging for prsm.node.* so the dashboard
                # activity log captures events during operation.
                _logging.getLogger("prsm.node").setLevel(_logging.INFO)

                # Clear startup messages before showing dashboard
                console.clear()

                from prsm.node.dashboard import NodeDashboard
                dashboard = NodeDashboard(prsm_node)
                await dashboard.run()
        except asyncio.CancelledError:
            pass
        finally:
            await prsm_node.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\nNode stopped.", style="bold")


@node.command("stop")
@click.option("--timeout", default=10, help="Seconds to wait for graceful shutdown")
def node_stop(timeout: int):
    """Stop the background node daemon."""
    from prsm.cli_modules.daemon import daemon_stop as _stop
    _stop(timeout=timeout)


@node.command("restart")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--timeout", default=10, help="Seconds to wait for graceful shutdown")
def node_restart(host: str, port: int, timeout: int):
    """Restart the background node daemon."""
    from prsm.cli_modules.daemon import daemon_restart as _restart
    _restart(host=host, port=port, timeout=timeout)


@node.command("status")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def node_status(output_format: str):
    """Show background node daemon status."""
    from prsm.cli_modules.daemon import daemon_status as _status
    _status(output_format=output_format)


@node.command("logs")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
def node_logs(lines: int, follow: bool):
    """Show background node daemon logs."""
    from prsm.cli_modules.daemon import daemon_logs as _logs
    _logs(lines=lines, follow=follow)


@node.command("earnings")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_earnings(api_port: int, output_format: str):
    """Show this node's earnings dashboard.

    Aggregates 3 streams: royalty (claimable_wei) + heartbeat
    status + distribution timing. Backed by GET
    /admin/earnings-summary on the running node daemon.
    """
    import json
    import httpx

    url = f"http://127.0.0.1:{api_port}/admin/earnings-summary"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code != 200:
        console.print(
            f"[red]/admin/earnings-summary returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)

    body = resp.json()

    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    op = body.get("operator_address") or "(PRSM_OPERATOR_ADDRESS unset)"
    console.print(f"[bold]PRSM Operator Earnings[/bold]")
    console.print(f"  Operator: {op}")
    console.print()

    royalty = body.get("royalty", {})
    if royalty.get("available"):
        wei = royalty.get("claimable_wei", 0)
        ftns = wei / 1e18
        console.print(
            f"  [green]Royalty:[/green]      {ftns:.6f} FTNS claimable"
        )
    else:
        err = royalty.get("error")
        suffix = f" — error: {err}" if err else ""
        console.print(f"  [yellow]Royalty:[/yellow]      not wired{suffix}")

    hb = body.get("heartbeat", {})
    if hb.get("available"):
        if hb.get("never_recorded"):
            console.print(
                f"  [red]Heartbeat:[/red]    never recorded — "
                f"slashing imminent"
            )
        elif hb.get("expired"):
            console.print(
                f"  [red]Heartbeat:[/red]    EXPIRED — slashing window open"
            )
        elif hb.get("at_risk"):
            console.print(
                f"  [yellow]Heartbeat:[/yellow]    at-risk — "
                f"{hb['grace_remaining']}s grace remaining"
            )
        else:
            console.print(
                f"  [green]Heartbeat:[/green]    ok — "
                f"{hb['grace_remaining']}s grace "
                f"(of {hb['grace_seconds']}s)"
            )
    else:
        err = hb.get("error")
        suffix = f" — error: {err}" if err else ""
        console.print(f"  [yellow]Heartbeat:[/yellow]    not wired{suffix}")

    dist = body.get("distribution", {})
    if dist.get("available"):
        if dist.get("never_distributed"):
            console.print(f"  [yellow]Distribution:[/yellow] never run yet")
        else:
            secs = dist.get("seconds_since", 0)
            hrs = secs // 3600
            console.print(
                f"  [green]Distribution:[/green] last run {hrs}h ago"
            )
    else:
        err = dist.get("error")
        suffix = f" — error: {err}" if err else ""
        console.print(
            f"  [yellow]Distribution:[/yellow] not wired{suffix}"
        )


def _node_admin_history(
    *, api_port: int, path: str, label: str,
    output_format: str, limit: int = 20,
    row_renderer=None, provider: Optional[str] = None,
):
    """Shared helper for the admin-history CLI commands.

    `row_renderer` is a callable(entry: dict) -> str that
    each command supplies for typed rendering. Without it the
    raw entry dict is printed (debug fallback). Optional `provider`
    narrows the server-side query to one provider/address.
    """
    import json
    import datetime
    import httpx
    from urllib.parse import quote

    url = (
        f"http://127.0.0.1:{api_port}{path}?limit={limit}"
    )
    if provider:
        url += f"&provider={quote(str(provider), safe='')}"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code != 200:
        if resp.status_code == 503:
            console.print(
                f"[yellow]{label} log not configured.[/yellow]\n"
                f"[dim]{resp.json().get('detail', 'unknown')}[/dim]"
            )
            sys.exit(0)  # 503 is not an error — just unwired
        console.print(
            f"[red]{path} returned {resp.status_code}[/red]: "
            f"{resp.text}"
        )
        sys.exit(1)

    body = resp.json()
    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    entries = body.get("entries", [])
    total = body.get("total", 0)
    console.print(
        f"[bold]PRSM {label}[/bold] (showing {len(entries)} of {total}):"
    )
    if not entries:
        console.print(f"  [dim]No entries[/dim]")
        return
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(ts).strftime(
                "%H:%M:%S",
            )
        except Exception:
            t = "????"
        if row_renderer is not None:
            console.print(f"  {t}  {row_renderer(e)}")
        else:
            console.print(f"  {t}  {e}")


def _short_addr(addr: str, *, head: int = 8, tail: int = 6) -> str:
    """Truncate 0x... addresses for column display."""
    if not addr or len(addr) <= head + tail + 2:
        return addr or "?"
    return f"{addr[:head]}..{addr[-tail:]}"


# Sp861 — `prsm node phase5-status` terminal-friendly readiness grid.
@node.command("phase5-status")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_phase5_status(api_port: int, output_format: str):
    """Show the Phase 5 fiat-surface readiness grid.

    Aggregates KYC + WaaS + Paymaster + Onramp + Aerodrome status
    via GET /wallet/phase5/status on the running daemon. Rolls up
    to READY / PARTIAL / NOT_READY for at-a-glance triage.
    """
    import json
    import httpx

    url = f"http://127.0.0.1:{api_port}/wallet/phase5/status"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)
    if resp.status_code != 200:
        console.print(
            f"[red]/wallet/phase5/status returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)
    body = resp.json()

    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    overall = body.get("overall", "UNKNOWN")
    live = body.get("live_surface_count", 0)
    total = body.get("total_surface_count", 0)
    color = {
        "READY": "green", "PARTIAL": "yellow", "NOT_READY": "red",
    }.get(overall, "white")
    console.print(
        f"[bold]PRSM Phase 5 Readiness[/bold] — "
        f"[{color}]{overall}[/{color}] "
        f"({live}/{total} live)"
    )
    console.print()

    table = Table()
    table.add_column("Surface", style="bold")
    table.add_column("Commissioned", justify="center")
    table.add_column("Adapter Wired", justify="center")
    table.add_column("Live Exec", justify="center")
    table.add_column("Notes", overflow="fold")

    def _tick(b: bool) -> str:
        return "[green]✓[/green]" if b else "[red]✗[/red]"

    # Order surfaces by user-onboarding flow sequence (more
    # intuitive than dict-iteration order).
    surface_order = [
        "kyc", "waas", "onramp", "paymaster", "aerodrome",
    ]
    surfaces = body.get("surfaces", {})
    for name in surface_order:
        s = surfaces.get(name) or {}
        table.add_row(
            name,
            _tick(s.get("commissioned", False)),
            _tick(s.get("adapter_wired", False)),
            _tick(s.get("live_exec", False)),
            s.get("notes", "") or "—",
        )

    console.print(table)


# Sp863 — `prsm node wallet-balance` terminal-friendly USDC/FTNS/ETH
# readout backed by sp862's /wallet/balance/* endpoints.
@node.command("wallet-balance")
@click.argument("identifier")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_wallet_balance(
    identifier: str, api_port: int, output_format: str,
):
    """Show live Base mainnet USDC + FTNS + ETH balances.

    IDENTIFIER is either a WaaS user_id (resolved via local store)
    or a raw 0x address. Auto-detects which based on shape.
    Backed by GET /wallet/balance/{user_id} or
    /wallet/balance/by-address/{address} on the running daemon.
    """
    import json
    import httpx

    # Auto-detect: anything matching 0x + 40 hex chars = address;
    # anything else = user_id lookup.
    is_address = (
        identifier.startswith("0x")
        and len(identifier) == 42
    )
    path = (
        f"/wallet/balance/by-address/{identifier}"
        if is_address
        else f"/wallet/balance/{identifier}"
    )
    url = f"http://127.0.0.1:{api_port}{path}"
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)
    if resp.status_code == 404:
        console.print(
            f"[red]No wallet found for "
            f"{identifier!r}[/red]\n"
            f"[dim]If this is a user_id, provision first via: "
            f"POST /wallet/waas/provision[/dim]"
        )
        sys.exit(1)
    if resp.status_code != 200:
        console.print(
            f"[red]{path} returned {resp.status_code}[/red]: "
            f"{resp.text}"
        )
        sys.exit(1)
    body = resp.json()

    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    address = body.get("address") or "(none)"
    user_id = body.get("user_id")
    wallet_id = body.get("wallet_id")

    title = "[bold]Wallet Balance[/bold]"
    if user_id:
        title += f" — user_id=[cyan]{user_id}[/cyan]"
    console.print(title)
    console.print(f"  Address:  {address}")
    if wallet_id:
        console.print(f"  Wallet:   {wallet_id}")
    network = body.get("network") or "base"
    console.print(f"  Network:  {network}")
    console.print(
        f"  Block:    {body.get('block_number', '?')} "
        f"[dim](via {body.get('rpc_url', '?')})[/dim]"
    )
    console.print()

    table = Table()
    table.add_column("Asset", style="bold")
    table.add_column("Balance", justify="right")
    table.add_column("Base Units", justify="right", style="dim")

    usdc = body.get("usdc", 0.0)
    ftns = body.get("ftns", 0.0)
    eth = body.get("native_eth", 0.0)

    def _fmt(val: float, name: str) -> str:
        color = "green" if val > 0 else "dim"
        return f"[{color}]{val:.6f} {name}[/{color}]"

    table.add_row(
        "USDC", _fmt(usdc, "USDC"),
        str(body.get("usdc_units", 0)),
    )
    table.add_row(
        "FTNS", _fmt(ftns, "FTNS"),
        str(body.get("ftns_units", 0)),
    )
    table.add_row(
        "ETH (native)", _fmt(eth, "ETH"),
        str(body.get("native_eth_wei", 0)),
    )
    console.print(table)


# Sp865 — `prsm node treasury` fleet-wide rollup readout backed by
# sp864's /wallet/treasury endpoint.
@node.command("treasury")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--max-wallets", default=100, type=int,
    help="Max wallets to query (default 100)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_treasury(
    api_port: int, max_wallets: int, output_format: str,
):
    """Show fleet-wide treasury: aggregated balances + per-wallet breakdown.

    Backed by GET /wallet/treasury on the running daemon.
    """
    import json
    import httpx

    url = (
        f"http://127.0.0.1:{api_port}/wallet/treasury"
        f"?max_wallets={max_wallets}"
    )
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)
    if resp.status_code != 200:
        console.print(
            f"[red]/wallet/treasury returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)
    body = resp.json()

    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    overall = body.get("overall") or {}
    note = body.get("note")
    if note:
        console.print(f"[yellow]{note}[/yellow]")
        console.print()

    console.print(f"[bold]PRSM Fleet Treasury[/bold]")
    total_w = overall.get("wallet_count_total", 0)
    with_addr = overall.get("wallet_count_with_address", 0)
    funded = overall.get("wallet_count_funded", 0)
    console.print(
        f"  Wallets:   {total_w} total · {with_addr} provisioned "
        f"· {funded} funded"
    )
    block = overall.get("block_number", 0)
    rpc = overall.get("rpc_url") or "—"
    console.print(
        f"  Block:     {block} [dim](via {rpc})[/dim]"
    )
    console.print()

    # Aggregated totals table
    totals = Table(title="Aggregate Holdings")
    totals.add_column("Asset", style="bold")
    totals.add_column("Total Balance", justify="right")
    totals.add_column("Base Units", justify="right", style="dim")

    def _fmt(val: float, name: str) -> str:
        color = "green" if val > 0 else "dim"
        return f"[{color}]{val:.6f} {name}[/{color}]"

    totals.add_row(
        "USDC", _fmt(overall.get("total_usdc", 0.0), "USDC"),
        str(overall.get("total_usdc_units", 0)),
    )
    totals.add_row(
        "FTNS", _fmt(overall.get("total_ftns", 0.0), "FTNS"),
        str(overall.get("total_ftns_units", 0)),
    )
    totals.add_row(
        "ETH (native)",
        _fmt(overall.get("total_native_eth", 0.0), "ETH"),
        str(overall.get("total_native_eth_wei", 0)),
    )
    console.print(totals)
    console.print()

    # Per-wallet breakdown
    wallets = body.get("wallets") or []
    if not wallets:
        console.print("[dim]No wallets to display.[/dim]")
        return

    per_wallet = Table(title="Per-Wallet Breakdown")
    per_wallet.add_column("User ID", style="cyan")
    per_wallet.add_column("Address", overflow="fold")
    per_wallet.add_column("USDC", justify="right")
    per_wallet.add_column("FTNS", justify="right")
    per_wallet.add_column("ETH", justify="right")
    per_wallet.add_column("Status")

    for w in wallets:
        addr = w.get("address") or "—"
        short = (
            f"{addr[:8]}..{addr[-6:]}" if len(addr) > 20 else addr
        )
        bal = w.get("balances")
        if bal is None:
            err = w.get("error") or "?"
            per_wallet.add_row(
                w.get("user_id", "?"), short,
                "[red]err[/red]", "[red]err[/red]",
                "[red]err[/red]", err[:30],
            )
            continue
        usdc_v = bal.get("usdc", 0.0)
        ftns_v = bal.get("ftns", 0.0)
        eth_v = bal.get("native_eth", 0.0)
        per_wallet.add_row(
            w.get("user_id", "?"), short,
            (
                f"[green]{usdc_v:.4f}[/green]" if usdc_v > 0
                else f"[dim]{usdc_v:.4f}[/dim]"
            ),
            (
                f"[green]{ftns_v:.4f}[/green]" if ftns_v > 0
                else f"[dim]{ftns_v:.4f}[/dim]"
            ),
            (
                f"[green]{eth_v:.4f}[/green]" if eth_v > 0
                else f"[dim]{eth_v:.4f}[/dim]"
            ),
            w.get("status", "?"),
        )
    console.print(per_wallet)


# Sp877 — `prsm node onramp-notifications` terminal viewer for sp874's
# outbound webhook delivery history.
@node.command("onramp-notifications")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--limit", default=50, type=int,
    help="Max deliveries to show (default 50)",
)
@click.option(
    "--success-only", is_flag=True,
    help="Only show successful deliveries (status 2xx)",
)
@click.option(
    "--failures-only", is_flag=True,
    help="Only show failed deliveries (non-2xx or transport error)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_onramp_notifications(
    api_port: int,
    limit: int,
    success_only: bool,
    failures_only: bool,
    output_format: str,
):
    """Show outbound onramp-completion webhook delivery history.

    Backed by GET /wallet/onramp/notifications (sp874). Persistent
    cross-restart audit trail — useful for operators investigating
    "did the customer's downstream system get notified when their
    onramp confirmed?"
    """
    import json as _json
    import httpx
    from datetime import datetime, timezone

    if success_only and failures_only:
        console.print(
            "[red]Pass at most one of --success-only / "
            "--failures-only[/red]"
        )
        sys.exit(2)

    url = (
        f"http://127.0.0.1:{api_port}/wallet/onramp/notifications"
        f"?limit={limit}"
    )
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)
    if resp.status_code != 200:
        console.print(
            f"[red]/wallet/onramp/notifications returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)
    body = resp.json()

    deliveries = body.get("deliveries") or []
    if success_only:
        deliveries = [
            d for d in deliveries if d.get("success")
        ]
    if failures_only:
        deliveries = [
            d for d in deliveries if not d.get("success")
        ]

    if output_format == "json":
        body["deliveries"] = deliveries  # apply filter
        console.print(_json.dumps(body, indent=2))
        return

    configured = body.get("configured", False)
    if not configured:
        console.print(
            "[yellow]⚠ PRSM_ONRAMP_COMPLETION_WEBHOOK_URL is "
            "unset — notifier is no-op.[/yellow]"
        )
        console.print(
            "[dim]Set the env var + restart the daemon to enable "
            "outbound delivery.[/dim]"
        )
        console.print()

    total = body.get("count", 0)
    succ = body.get("success_count", 0)
    fail = body.get("failure_count", 0)
    rate = (succ / total) if total > 0 else 0.0
    rate_color = (
        "green" if rate >= 0.99 else
        "yellow" if rate >= 0.5 else "red"
    )

    console.print("[bold]PRSM Onramp Completion Notifications[/bold]")
    console.print(
        f"  Total:         {total} · "
        f"[green]{succ} successes[/green] · "
        f"[red]{fail} failures[/red]"
    )
    if total > 0:
        console.print(
            f"  Success rate:  "
            f"[{rate_color}]{rate * 100:.1f}%[/{rate_color}]"
        )
    console.print()

    if not deliveries:
        msg = "No deliveries recorded yet."
        if success_only:
            msg = "No successful deliveries match the filter."
        if failures_only:
            msg = "No failed deliveries match the filter."
        console.print(f"[dim]{msg}[/dim]")
        return

    table = Table()
    table.add_column("Timestamp (UTC)", style="dim")
    table.add_column("Intent", style="cyan")
    table.add_column("URL", overflow="fold")
    table.add_column("Status", justify="right")
    table.add_column("Sig")
    table.add_column("Error", overflow="fold")

    for d in deliveries:
        ts = d.get("timestamp", 0)
        ts_str = (
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S",
            ) if ts else "?"
        )
        intent = (d.get("intent_id") or "?")[:18]
        durl = d.get("url") or ""
        # Show shortened URL — full one in --format json
        durl_short = durl[:40] + ("…" if len(durl) > 40 else "")
        status = d.get("status_code", 0)
        if status == 0:
            status_cell = "[red]transport[/red]"
        elif 200 <= status < 300:
            status_cell = f"[green]{status}[/green]"
        elif 400 <= status < 500:
            status_cell = f"[yellow]{status}[/yellow]"
        else:
            status_cell = f"[red]{status}[/red]"
        sig = (
            "[green]✓[/green]"
            if d.get("signature_attached") else "[dim]—[/dim]"
        )
        err = d.get("error") or ""
        err_short = err[:60] + ("…" if len(err) > 60 else "")
        table.add_row(
            ts_str, intent, durl_short, status_cell,
            sig, err_short,
        )
    console.print(table)


# Sp876 — `prsm node aerodrome-ceremony` Safe TX batch + runbook
# generator backed by sp875's pure-payload builders. Lets operators
# generate ceremony artifacts from terminal without writing Python.
@node.command("aerodrome-ceremony")
@click.option(
    "--network", type=click.Choice(["mainnet", "sepolia"]),
    default="mainnet",
    help="Target network — default mainnet; use sepolia for rehearsal",
)
@click.option(
    "--seeder-safe", required=True, type=str,
    help="Seeding entity's Safe address (0x...) — Prismatica or an "
         "independent third party. Option A: NOT the Foundation Safe.",
)
@click.option(
    "--seed-usdc", required=True, type=float,
    help="USDC seed amount in whole tokens (e.g., 50000 = 50k USDC)",
)
@click.option(
    "--seed-ftns", required=True, type=float,
    help="FTNS seed amount in whole tokens (e.g., 50000 = 50k FTNS)",
)
@click.option(
    "--slippage-bps", default=100, type=int,
    help="Slippage tolerance in basis points (default 100 = 1%)",
)
@click.option(
    "--deadline-seconds", default=3600, type=int,
    help="Deadline from now in seconds (default 3600 = 1h)",
)
@click.option(
    "--output-json", "-j", type=click.Path(), default=None,
    help="Write batch JSON to this path (Safe TX Builder format)",
)
@click.option(
    "--output-runbook", "-r", type=click.Path(), default=None,
    help="Write co-signer runbook markdown to this path",
)
def node_aerodrome_ceremony(
    network: str,
    seeder_safe: str,
    seed_usdc: float,
    seed_ftns: float,
    slippage_bps: int,
    deadline_seconds: int,
    output_json: Optional[str],
    output_runbook: Optional[str],
):
    """Generate Aerodrome pool seeding ceremony artifacts.

    Produces Safe-Transaction-Builder-compatible JSON (3-tx batch:
    USDC.approve + FTNS.approve + Router.addLiquidity) and a
    co-signer runbook markdown. Pure-payload — never signs or
    submits. Upload JSON via wallet.safe.global → TX Builder.

    The seed amounts encode the OPENING MARKET PRICE for FTNS:
    price = USDC / FTNS. Choose deliberately.
    """
    import json as _json
    from prsm.economy.web3.aerodrome_pool_ceremony import (
        MAINNET_CONFIG, SEPOLIA_CONFIG,
        build_ceremony_batch, build_runbook_markdown,
    )

    if not seeder_safe.startswith("0x") or len(
        seeder_safe,
    ) != 42:
        console.print(
            f"[red]seeder_safe must be 0x + 40 hex chars[/red]"
        )
        sys.exit(2)
    if seed_usdc <= 0 or seed_ftns <= 0:
        console.print(
            "[red]Both --seed-usdc and --seed-ftns must be > 0[/red]"
        )
        sys.exit(2)

    config = (
        MAINNET_CONFIG if network == "mainnet"
        else SEPOLIA_CONFIG
    )

    # Convert whole tokens to base units.
    seed_usdc_units = int(seed_usdc * 10**6)
    seed_ftns_units = int(seed_ftns * 10**18)
    opening_price = seed_usdc / seed_ftns if seed_ftns > 0 else 0

    # Color-code mainnet differently — visual signal that this is
    # REAL MONEY, not rehearsal.
    net_color = "red" if network == "mainnet" else "yellow"

    console.print(
        f"[bold]Aerodrome USDC↔FTNS Ceremony Generator[/bold]"
    )
    console.print(
        f"  Network:           [{net_color}]{network}[/{net_color}] "
        f"(chain_id {config.chain_id})"
    )
    console.print(f"  Seeding Safe:      {seeder_safe}")
    console.print(
        f"  Seed USDC:         {seed_usdc:.6f} "
        f"({seed_usdc_units} base units, 6 decimals)"
    )
    console.print(
        f"  Seed FTNS:         {seed_ftns:.6f} "
        f"({seed_ftns_units} base units, 18 decimals)"
    )
    console.print(
        f"  Opening price:     "
        f"[bold]${opening_price:.6f} per FTNS[/bold]"
    )
    console.print(f"  Slippage:          {slippage_bps} bps")
    console.print(f"  Deadline:          +{deadline_seconds}s from now")
    console.print()

    if network == "mainnet":
        console.print(
            "[red]⚠ MAINNET — this batch will move real money "
            "when executed.[/red]"
        )
        console.print(
            "[dim]Strongly recommend a Sepolia rehearsal first "
            "(--network sepolia) with throwaway amounts.[/dim]"
        )
        console.print()

    try:
        batch = build_ceremony_batch(
            network=config,
            seeder_safe=seeder_safe,
            seed_usdc_units=seed_usdc_units,
            seed_ftns_units=seed_ftns_units,
            slippage_bps=slippage_bps,
            deadline_seconds=deadline_seconds,
        )
        runbook = build_runbook_markdown(
            network=config,
            seeder_safe=seeder_safe,
            seed_usdc_units=seed_usdc_units,
            seed_ftns_units=seed_ftns_units,
            slippage_bps=slippage_bps,
        )
    except ValueError as exc:
        console.print(f"[red]Validation error: {exc}[/red]")
        sys.exit(1)

    # Output handling: either to files (with confirmation) or
    # stdout (JSON to stdout, runbook to stderr-ish console).
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            _json.dump(batch, f, indent=2)
        console.print(
            f"[green]✓[/green] Wrote batch JSON to "
            f"[cyan]{output_json}[/cyan] "
            f"({len(batch['transactions'])} transactions)"
        )
    if output_runbook:
        with open(output_runbook, "w", encoding="utf-8") as f:
            f.write(runbook)
        console.print(
            f"[green]✓[/green] Wrote runbook to "
            f"[cyan]{output_runbook}[/cyan] "
            f"({runbook.count(chr(10)) + 1} lines)"
        )

    if not output_json and not output_runbook:
        # No outputs — print summary + suggest next step
        console.print(
            "[yellow]No --output-json / --output-runbook[/yellow] "
            "specified — nothing written."
        )
        console.print(
            "[dim]Example:[/dim]\n"
            "[dim]  prsm node aerodrome-ceremony "
            "--network sepolia --seeder-safe 0x... "
            "--seed-usdc 1 --seed-ftns 1 "
            "-j /tmp/sepolia-batch.json "
            "-r /tmp/sepolia-runbook.md[/dim]"
        )
        sys.exit(1)

    console.print()
    console.print("[bold]Next steps[/bold]")
    console.print(
        "  1. Distribute the runbook to all co-signers BEFORE "
        "ceremony"
    )
    console.print(
        "  2. Co-signers verify every address in the runbook "
        "matches what their hardware wallet shows during signing"
    )
    console.print(
        "  3. Upload batch JSON: wallet.safe.global → "
        f"connect to seeding Safe {seeder_safe[:8]}... → Apps → "
        "Transaction Builder → Load from JSON"
    )
    console.print(
        "  4. Threshold co-signers sign via hardware wallets + "
        "execute"
    )
    console.print(
        "  5. After landing, set "
        "[yellow]AERODROME_USDC_FTNS_POOL_ADDRESS[/yellow] in "
        "operator env to the resulting pool address"
    )
    console.print(
        "  6. Verify go-live: [yellow]prsm node aerodrome-go-live"
        "[/yellow] — confirms the pool is seeded + the onramp→swap "
        "path is live, and prepares the first real swap envelope"
    )


# Sp901 — `prsm node aerodrome-go-live` post-seed verification harness.
# The final step of the ceremony lifecycle: confirm the seed actually
# worked + the fiat→FTNS swap path is live before opening the taps.
@node.command("aerodrome-go-live")
@click.option(
    "--network", type=click.Choice(["mainnet", "sepolia"]),
    default="mainnet",
    help="Target network — default mainnet",
)
@click.option(
    "--probe-usd", default=1.0, type=float,
    help="Probe swap size in whole USDC for the path check (default 1)",
)
@click.option(
    "--expected-seed-usdc", default=None, type=float,
    help="Optional: declared USDC seed (whole tokens) to cross-check "
         "reserves (mismatch → WARN, not a block)",
)
@click.option(
    "--expected-seed-ftns", default=None, type=float,
    help="Optional: declared FTNS seed (whole tokens) to cross-check",
)
@click.option(
    "--slippage-bps", default=100, type=int,
    help="Slippage tolerance for the prepared envelope (default 100)",
)
@click.option(
    "--output-json", "-j", type=click.Path(), default=None,
    help="Write the full report (incl. prepared swap envelope) here",
)
def node_aerodrome_go_live(
    network: str,
    probe_usd: float,
    expected_seed_usdc: Optional[float],
    expected_seed_ftns: Optional[float],
    slippage_bps: int,
    output_json: Optional[str],
):
    """Verify the Aerodrome pool is live + ready for fiat→FTNS.

    Run this the instant the seed ceremony executes and you've set
    AERODROME_USDC_FTNS_POOL_ADDRESS (+ BASE_RPC_URL). It confirms the
    pool is configured, seeded (non-zero reserves), holds the USDC/FTNS
    pair, is volatile, reports the opening price, quotes a live swap,
    and builds the full onramp→swap envelope end-to-end. On success it
    prepares the exact executable first-swap envelope. Read-only — no
    money moves.
    """
    import json as _json
    from prsm.economy.web3.aerodrome_client import AerodromeClient
    from prsm.economy.web3.aerodrome_pool_ceremony import (
        MAINNET_CONFIG, SEPOLIA_CONFIG,
    )
    from prsm.economy.web3.go_live_verification import (
        run_go_live_verification,
    )

    cfg = MAINNET_CONFIG if network == "mainnet" else SEPOLIA_CONFIG
    client = AerodromeClient.from_env()
    exp_usdc = (
        int(expected_seed_usdc * 10 ** 6)
        if expected_seed_usdc else None
    )
    exp_ftns = (
        int(expected_seed_ftns * 10 ** 18)
        if expected_seed_ftns else None
    )
    report = run_go_live_verification(
        client, cfg,
        probe_usdc_units=int(probe_usd * 10 ** 6),
        expected_usdc_units=exp_usdc,
        expected_ftns_units=exp_ftns,
        slippage_bps=slippage_bps,
    )

    _glyph = {
        "PASS": "[green]✓[/green]", "FAIL": "[red]✗[/red]",
        "WARN": "[yellow]![/yellow]", "INFO": "[dim]·[/dim]",
    }
    console.print()
    console.print(f"[bold]Aerodrome go-live verification ({network})[/bold]")
    for f in report.findings:
        console.print(
            f"  {_glyph.get(f.status, '?')} "
            f"[bold]{f.check}[/bold]: {f.detail}"
        )
    console.print()
    if report.go:
        console.print(
            "[bold green]GO[/bold green] — pool is seeded + the "
            "onramp→swap path is live."
        )
        if report.prepared_envelope is not None:
            console.print(
                "[dim]Prepared first-swap envelope is in the report "
                "JSON (--output-json) for submission.[/dim]"
            )
    else:
        console.print(
            "[bold red]NO-GO[/bold red] — resolve the ✗ findings above "
            "before opening the fiat→FTNS path. (Pre-ceremony, "
            "pool_configured FAIL is expected.)"
        )

    if output_json:
        with open(output_json, "w") as fh:
            _json.dump(report.to_dict(), fh, indent=2)
        console.print(f"[green]Report written to {output_json}[/green]")
    sys.exit(0 if report.go else 1)


# Sp873 — `prsm node compliance-export` terminal export of sp872's
# /admin/fiat-compliance/export.csv. Lets operators run quarterly /
# annual exports from a single command without needing curl +
# shell redirect.
@node.command("compliance-export")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--since", type=float, default=None,
    help="Include entries with timestamp ≥ (Unix seconds)",
)
@click.option(
    "--until", "until_ts", type=float, default=None,
    help="Include entries with timestamp < (Unix seconds)",
)
@click.option(
    "--user-id", type=str, default=None,
    help="Filter to a single user_id (exact match)",
)
@click.option(
    "--kind", type=str, default=None,
    help="Filter to a specific entry kind (e.g., onramp_quote)",
)
@click.option(
    "--min-usd", type=float, default=None,
    help=(
        "Filter to entries above a USD threshold "
        "(FinCEN $10k CTR typical)"
    ),
)
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Output file path. If omitted, writes to stdout.",
)
def node_compliance_export(
    api_port: int,
    since: Optional[float],
    until_ts: Optional[float],
    user_id: Optional[str],
    kind: Optional[str],
    min_usd: Optional[float],
    output: Optional[str],
):
    """Export fiat compliance ring entries as CSV.

    Backed by GET /admin/fiat-compliance/export.csv. Filters
    compose for AUSTRAC TTR / FinCEN CTR / IRS 1099 use cases.
    Operators transform downstream to regulator-specific formats.
    """
    import httpx

    params = {}
    if since is not None: params["since"] = since
    if until_ts is not None: params["until"] = until_ts
    if user_id: params["user_id"] = user_id
    if kind: params["kind"] = kind
    if min_usd is not None: params["min_usd"] = min_usd

    url = f"http://127.0.0.1:{api_port}/admin/fiat-compliance/export.csv"
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.get(url, params=params)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)
    if resp.status_code != 200:
        console.print(
            f"[red]Export returned {resp.status_code}[/red]: "
            f"{resp.text}"
        )
        sys.exit(1)
    csv_text = resp.text

    # Count rows for the operator summary line.
    row_count = max(csv_text.count("\n") - 1, 0)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(csv_text)
        console.print(
            f"[green]✓[/green] Wrote {row_count} row(s) to "
            f"[cyan]{output}[/cyan]"
        )
        # Show filter context for archive audit trail
        if any([since, until_ts, user_id, kind, min_usd]):
            filt_parts = []
            if since: filt_parts.append(f"since={since}")
            if until_ts: filt_parts.append(f"until={until_ts}")
            if user_id: filt_parts.append(f"user_id={user_id}")
            if kind: filt_parts.append(f"kind={kind}")
            if min_usd: filt_parts.append(f"min_usd={min_usd}")
            console.print(
                f"  [dim]Filters: {' · '.join(filt_parts)}[/dim]"
            )
    else:
        # stdout: emit the CSV directly so the operator can pipe
        # (`prsm node compliance-export | column -ts,` etc.)
        click.echo(csv_text, nl=False)


# Sp856 — `prsm node phase5-dashboard` unified operator view.
# Combines phase5-status + treasury + onramp-funnel into one
# command for fast operator triage. Calls each endpoint
# independently so a partial failure in one surface doesn't blank
# the rest (sibling pattern to sp864's per-wallet fail-soft).
@node.command("phase5-dashboard")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_phase5_dashboard(api_port: int, output_format: str):
    """Comprehensive Phase 5 operator dashboard.

    Combines readiness grid + fleet treasury + conversion funnel
    in one command. Each surface fails independently so partial
    daemon outages still show what IS available.
    """
    import json
    import httpx

    base = f"http://127.0.0.1:{api_port}"

    def _fetch(path):
        try:
            with httpx.Client(timeout=60.0) as client:
                r = client.get(f"{base}{path}")
            if r.status_code == 200:
                return r.json(), None
            return None, f"{r.status_code} {r.text[:100]}"
        except httpx.RequestError as exc:
            return None, str(exc)

    phase5, p5_err = _fetch("/wallet/phase5/status")
    treasury, t_err = _fetch("/wallet/treasury?max_wallets=100")
    funnel, f_err = _fetch("/wallet/onramp/funnel?limit=20")

    if output_format == "json":
        out = {
            "phase5_status": phase5 or {"error": p5_err},
            "treasury": treasury or {"error": t_err},
            "onramp_funnel": funnel or {"error": f_err},
        }
        console.print(json.dumps(out, indent=2))
        return

    if p5_err and t_err and f_err:
        console.print(
            f"[red]All 3 endpoints unreachable[/red]\n"
            f"[dim]phase5: {p5_err}[/dim]\n"
            f"[dim]treasury: {t_err}[/dim]\n"
            f"[dim]funnel: {f_err}[/dim]\n"
            f"[dim]Start with: prsm node start[/dim]"
        )
        sys.exit(2)

    console.print("[bold]═══ PRSM Phase 5 Dashboard ═══[/bold]")
    console.print()

    # ── Section 1: Surface readiness ──
    if phase5:
        overall = phase5.get("overall", "UNKNOWN")
        live = phase5.get("live_surface_count", 0)
        total = phase5.get("total_surface_count", 0)
        color = {
            "READY": "green", "PARTIAL": "yellow",
            "NOT_READY": "red",
        }.get(overall, "white")
        console.print(
            f"[bold]§ Readiness[/bold]   "
            f"[{color}]{overall}[/{color}] ({live}/{total} live)"
        )

        rt = Table(show_header=True, padding=(0, 1))
        rt.add_column("Surface", style="bold")
        rt.add_column("Live")
        rt.add_column("Notes", overflow="fold")
        for sn in [
            "kyc", "waas", "onramp", "paymaster", "aerodrome",
        ]:
            s = (phase5.get("surfaces") or {}).get(sn, {})
            live_exec = s.get("live_exec", False)
            marker = (
                "[green]✓[/green]" if live_exec
                else "[red]✗[/red]"
            )
            rt.add_row(sn, marker, s.get("notes", "—") or "—")
        console.print(rt)
    elif p5_err:
        console.print(f"[red]§ Readiness ERR: {p5_err}[/red]")
    console.print()

    # ── Section 2: Fleet treasury ──
    if treasury:
        overall = treasury.get("overall") or {}
        total_w = overall.get("wallet_count_total", 0)
        with_addr = overall.get("wallet_count_with_address", 0)
        funded = overall.get("wallet_count_funded", 0)
        u = overall.get("total_usdc", 0.0)
        ft = overall.get("total_ftns", 0.0)
        eth = overall.get("total_native_eth", 0.0)
        block = overall.get("block_number", 0)
        u_color = "green" if u > 0 else "dim"
        ft_color = "green" if ft > 0 else "dim"
        eth_color = "green" if eth > 0 else "dim"
        console.print(
            f"[bold]§ Treasury[/bold]    "
            f"{total_w} wallets · {with_addr} provisioned · "
            f"{funded} funded · block {block}"
        )
        console.print(
            f"              "
            f"[{u_color}]{u:.6f} USDC[/{u_color}] · "
            f"[{ft_color}]{ft:.6f} FTNS[/{ft_color}] · "
            f"[{eth_color}]{eth:.6f} ETH[/{eth_color}]"
        )
    elif t_err:
        console.print(f"[red]§ Treasury ERR: {t_err}[/red]")
    console.print()

    # ── Section 3: Onramp funnel ──
    if funnel:
        summary = funnel.get("summary") or {}
        total_i = summary.get("total_intents", 0)
        rate = summary.get("conversion_rate", 0.0)
        expected = summary.get("total_expected_usd", 0.0)
        conf_usdc = summary.get("total_confirmed_usdc", 0.0)
        rate_color = (
            "green" if rate >= 0.5 else
            "yellow" if rate > 0 else "dim"
        )
        console.print(
            f"[bold]§ Onramp[/bold]      "
            f"{total_i} intents · "
            f"[{rate_color}]{rate * 100:.1f}% conv rate"
            f"[/{rate_color}] · "
            f"${expected:.2f} expected → "
            f"[green]${conf_usdc:.2f}[/green] confirmed"
        )
        counts = summary.get("status_counts") or {}
        breakdown_parts = []
        for s in [
            "INTENT_RECORDED", "PENDING_SETTLEMENT",
            "CONFIRMED", "EXPIRED",
        ]:
            n = counts.get(s, 0)
            if n > 0:
                short = {
                    "INTENT_RECORDED": "recorded",
                    "PENDING_SETTLEMENT": "pending",
                    "CONFIRMED": "confirmed",
                    "EXPIRED": "expired",
                }[s]
                breakdown_parts.append(f"{short}={n}")
        if breakdown_parts:
            console.print(
                f"              {' · '.join(breakdown_parts)}"
            )
    elif f_err:
        console.print(f"[red]§ Onramp ERR: {f_err}[/red]")


# Sp866 — `prsm node onramp-funnel` conversion-tracker readout
# backed by sp857's /wallet/onramp/funnel + /sweep endpoints.
@node.command("onramp-funnel")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--status", default=None,
    type=click.Choice([
        "INTENT_RECORDED", "PENDING_SETTLEMENT",
        "CONFIRMED", "EXPIRED",
    ]),
    help="Filter intents by status",
)
@click.option(
    "--sweep", is_flag=True,
    help="Trigger an on-chain sweep before showing funnel",
)
@click.option(
    "--limit", default=50, type=int,
    help="Max intents to show (default 50)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_onramp_funnel(
    api_port: int,
    status: Optional[str],
    sweep: bool,
    limit: int,
    output_format: str,
):
    """Show onramp conversion funnel: intents + sweep results.

    Backed by GET /wallet/onramp/funnel (and optionally POST
    /wallet/onramp/sweep if --sweep is passed). Conversion rate
    is the ratio of CONFIRMED intents to total — operator-visible
    proxy for fiat-onramp UX health.
    """
    import json
    import httpx

    base = f"http://127.0.0.1:{api_port}"

    sweep_result = None
    if sweep:
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{base}/wallet/onramp/sweep")
        except httpx.RequestError as exc:
            console.print(
                f"[red]Cannot reach PRSM node at {base}[/red]\n"
                f"[dim]Start with: prsm node start[/dim]\n"
                f"[dim]Details: {exc}[/dim]"
            )
            sys.exit(2)
        if resp.status_code != 200:
            console.print(
                f"[red]Sweep returned {resp.status_code}[/red]: "
                f"{resp.text}"
            )
            sys.exit(1)
        sweep_result = resp.json()

    url = f"{base}/wallet/onramp/funnel?limit={limit}"
    if status:
        url += f"&status={status}"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)
    if resp.status_code != 200:
        console.print(
            f"[red]/wallet/onramp/funnel returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)
    body = resp.json()

    if output_format == "json":
        if sweep_result is not None:
            body["_sweep"] = sweep_result
        console.print(json.dumps(body, indent=2))
        return

    if sweep_result is not None:
        console.print(
            f"[bold]Sweep:[/bold] checked={sweep_result['checked']} "
            f"confirmed_new=[green]{sweep_result['confirmed_new']}[/green] "
            f"expired_new=[yellow]{sweep_result['expired_new']}[/yellow]"
        )
        console.print()

    summary = body.get("summary") or {}
    total = summary.get("total_intents", 0)
    rate = summary.get("conversion_rate", 0.0)
    counts = summary.get("status_counts") or {}
    expected = summary.get("total_expected_usd", 0.0)
    confirmed_usdc = summary.get("total_confirmed_usdc", 0.0)

    rate_color = (
        "green" if rate >= 0.5 else
        "yellow" if rate > 0 else "dim"
    )

    console.print("[bold]PRSM Onramp Conversion Funnel[/bold]")
    console.print(
        f"  Intents:           {total} total "
        f"([{rate_color}]{rate * 100:.1f}% conversion rate[/{rate_color}])"
    )
    console.print(
        f"  Expected USD:      ${expected:.2f}"
    )
    console.print(
        f"  Confirmed USDC:    "
        f"[green]${confirmed_usdc:.2f}[/green]"
    )
    console.print()

    if counts:
        ct = Table(title="Status Distribution")
        ct.add_column("Status", style="bold")
        ct.add_column("Count", justify="right")
        status_color = {
            "INTENT_RECORDED": "cyan",
            "PENDING_SETTLEMENT": "yellow",
            "CONFIRMED": "green",
            "EXPIRED": "red",
        }
        order = [
            "INTENT_RECORDED", "PENDING_SETTLEMENT",
            "CONFIRMED", "EXPIRED",
        ]
        for s in order:
            n = counts.get(s, 0)
            color = status_color.get(s, "white")
            ct.add_row(
                f"[{color}]{s}[/{color}]",
                str(n) if n > 0 else f"[dim]{n}[/dim]",
            )
        console.print(ct)
        console.print()

    intents = body.get("intents") or []
    if not intents:
        console.print(
            "[dim]No intents to display (try without "
            "--status filter).[/dim]"
        )
        return

    table = Table(
        title=(
            "Intents"
            + (f" (status={status})" if status else "")
        ),
    )
    table.add_column("Intent", style="cyan")
    table.add_column("User", overflow="fold")
    table.add_column("Address", overflow="fold")
    table.add_column("Expected USD", justify="right")
    table.add_column("USDC In", justify="right")
    table.add_column("Status")
    table.add_column("Age")

    import time as _time
    now = _time.time()

    for intent in intents:
        intent_short = intent.get("intent_id", "")[:18]
        user = intent.get("user_id") or "—"
        addr = intent.get("destination_address") or ""
        addr_short = (
            f"{addr[:8]}..{addr[-6:]}" if len(addr) > 20 else addr
        )
        usd = intent.get("expected_usd", 0.0)
        usdc = intent.get("usdc_received", 0.0)
        st = intent.get("status", "?")
        st_color = {
            "INTENT_RECORDED": "cyan",
            "PENDING_SETTLEMENT": "yellow",
            "CONFIRMED": "green",
            "EXPIRED": "red",
        }.get(st, "white")
        age_seconds = now - intent.get("created_at", now)
        if age_seconds < 60:
            age = f"{int(age_seconds)}s"
        elif age_seconds < 3600:
            age = f"{int(age_seconds / 60)}m"
        else:
            age = f"{age_seconds / 3600:.1f}h"
        table.add_row(
            intent_short, user, addr_short,
            f"${usd:.2f}",
            (
                f"[green]${usdc:.2f}[/green]" if usdc > 0
                else f"[dim]${usdc:.2f}[/dim]"
            ),
            f"[{st_color}]{st}[/{st_color}]",
            age,
        )
    console.print(table)


def _render_webhook_row(e: dict) -> str:
    success = e.get("success")
    # Escape brackets so rich doesn't interpret as markup tags
    marker = r"\[ok]" if success else r"\[!]"
    code = e.get("status_code", "?")
    detail = (
        "delivered" if success
        else (e.get("error") or "no error msg")
    )
    return (
        f"{marker} {e.get('event', '?'):<28}  "
        f"status={code:<5} {detail}"
    )


def _render_slash_row(e: dict) -> str:
    return (
        f"{e.get('kind', '?'):<28}  "
        f"provider={_short_addr(e.get('provider', '?'))}  "
        f"slash_id={e.get('slash_id', '?')[:14]}..."
    )


def _render_heartbeat_row(e: dict) -> str:
    on_ts = e.get("onchain_timestamp", 0)
    return (
        f"provider={_short_addr(e.get('provider', '?'))}  "
        f"onchain_ts={on_ts}"
    )


def _render_consensus_mismatch_row(e: dict) -> str:
    bonded = "bonded" if e.get("accused_bonded") else "unbonded"
    stake = (e.get("accused_stake_wei", 0) or 0) / 1e18
    return (
        f"job={str(e.get('job_id', '?'))[:18]:<18}  "
        f"accused={_short_addr(e.get('accused_provider_id', '?'))}  "
        f"{bonded} stake={stake:.4f}  "
        f"out={str(e.get('accused_output_hash', '?'))[:10]}.."
        f" vs majority={str(e.get('majority_output_hash', '?'))[:10]}.."
    )


def _render_distribution_row(e: dict) -> str:
    creator = e.get("to_creator", 0) / 1e18
    operator = e.get("to_operator", 0) / 1e18
    grant = e.get("to_grant", 0) / 1e18
    return (
        f"creator={creator:.4f} operator={operator:.4f} "
        f"grant={grant:.4f} FTNS"
    )


@node.command("slash-history")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def node_slash_history(api_port, output_format, limit):
    """Show recent on-chain slash events."""
    _node_admin_history(
        api_port=api_port,
        path="/admin/slash-history",
        label="Slash Events",
        output_format=output_format,
        limit=limit,
        row_renderer=_render_slash_row,
    )


@node.group("consensus-mismatch", invoke_without_command=False)
def node_consensus_mismatch():
    """sp957 — CONSENSUS_MISMATCH evidence (single-provider compute pay path).

    Read-only operator triage over GET /admin/consensus-mismatch-evidence:
    the bonded providers a sampled re-execution caught returning an output that
    disagreed with the re-run majority. This is NOT an on-chain slash — it is
    the operator-reviewable corpus a future authority-gated bridge consumes.
    """


@node_consensus_mismatch.command("list")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
@click.option(
    "--provider", default=None,
    help="Filter to one accused provider id / address.",
)
def node_consensus_mismatch_list(api_port, output_format, limit, provider):
    """Show recent CONSENSUS_MISMATCH evidence."""
    _node_admin_history(
        api_port=api_port,
        path="/admin/consensus-mismatch-evidence",
        label="Consensus Mismatch Evidence",
        output_format=output_format,
        limit=limit,
        row_renderer=_render_consensus_mismatch_row,
        provider=provider,
    )


@node_consensus_mismatch.command("summary")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
def node_consensus_mismatch_summary(api_port, output_format):
    """Show which providers this node is excluding from dispatch + why."""
    import json as _json
    import httpx as _httpx
    url = f"http://127.0.0.1:{api_port}/admin/consensus-mismatch-evidence/summary"
    try:
        with _httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except _httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)
    if resp.status_code == 503:
        console.print(
            "[yellow]Dispatch-exclusion policy not available.[/yellow]\n"
            f"[dim]{resp.json().get('detail', 'compute requester not wired')}[/dim]"
        )
        sys.exit(0)
    if resp.status_code != 200:
        console.print(f"[red]{url} returned {resp.status_code}[/red]: {resp.text}")
        sys.exit(1)
    body = resp.json()
    if output_format == "json":
        console.print(_json.dumps(body, indent=2))
        return
    threshold = body.get("threshold", 0)
    window = body.get("window_sec", 0)
    providers = body.get("providers", [])
    console.print(
        f"[bold]Dispatch Exclusion Policy[/bold] "
        f"(threshold={threshold} confirmed mismatches, "
        f"window={'all-time' if not window else f'{window:.0f}s'}):"
    )
    if not providers:
        console.print("  [dim]No flagged providers[/dim]")
        return
    for p in providers:
        excluded = p.get("excluded")
        tag = "[red]EXCLUDED[/red]" if excluded else "[green]eligible[/green]"
        console.print(
            f"  {tag}  {_short_addr(p.get('provider_id', '?'))}  "
            f"mismatches={p.get('mismatch_count', 0)}"
        )


@node.command("doctor")
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_doctor_cli(
    api_url_override: Optional[str], output_format: str,
) -> None:
    """Sprint 818 — composite triage. ONE command tells you
    what's wrong.

    Runs 4 daemon checks (daemon up, preemption status, output
    cache, peer count) and reports PASS / WARN / FAIL per check
    + an overall verdict. Exit codes:
      0 — overall PASS or WARN (warnings are advisory)
      1 — overall FAIL (one or more FAIL-class issues)
      2 — daemon unreachable at /health
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)

    # ── /health probe — special case: failure → exit 2 ──
    try:
        h = _httpx.get(f"{url}/health", timeout=5.0)
    except Exception as exc:
        msg = f"daemon unreachable at {url}/health: {exc}"
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "error": msg,
                "checks": [], "overall": "FAIL",
            }))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(2)

    checks = []
    if h.status_code == 200:
        try:
            hdata = h.json()
            node_id = hdata.get("node_id", "unknown")
        except Exception:
            node_id = "unknown"
        checks.append({
            "name": "daemon",
            "status": "PASS",
            "detail": f"ok (node_id={(node_id or 'unknown')[:16]}…)",
        })
    else:
        checks.append({
            "name": "daemon",
            "status": "FAIL",
            "detail": f"/health returned {h.status_code}",
        })

    # ── preemption status ──
    try:
        r = _httpx.get(
            f"{url}/admin/preemption/status", timeout=5.0,
        )
        if r.status_code == 200:
            pdata = r.json()
            if pdata.get("preempted"):
                checks.append({
                    "name": "preemption",
                    "status": "WARN",
                    "detail": (
                        f"PREEMPTED — node draining "
                        f"(backend={pdata.get('backend', '?')})"
                    ),
                })
            else:
                checks.append({
                    "name": "preemption",
                    "status": "PASS",
                    "detail": (
                        f"clear (backend="
                        f"{pdata.get('backend', '?')})"
                    ),
                })
        elif r.status_code == 503:
            checks.append({
                "name": "preemption",
                "status": "WARN",
                "detail": "detector not configured "
                "(PRSM_PREEMPTION_DETECTOR unset)",
            })
        else:
            checks.append({
                "name": "preemption",
                "status": "FAIL",
                "detail": (
                    f"/admin/preemption/status returned "
                    f"{r.status_code}"
                ),
            })
    except Exception as exc:
        checks.append({
            "name": "preemption",
            "status": "FAIL",
            "detail": f"probe raised: {exc}",
        })

    # ── output cache stats ──
    try:
        r = _httpx.get(
            f"{url}/admin/output-cache-stats", timeout=5.0,
        )
        if r.status_code == 200:
            cdata = r.json()
            hits = int(cdata.get("hits", 0))
            misses = int(cdata.get("misses", 0))
            total = hits + misses
            if total == 0:
                checks.append({
                    "name": "output_cache",
                    "status": "PASS",
                    "detail": "configured (no traffic yet)",
                })
            else:
                rate = round((hits / total) * 100, 1)
                checks.append({
                    "name": "output_cache",
                    "status": "PASS",
                    "detail": (
                        f"hit_rate={rate}% "
                        f"(hits={hits} misses={misses})"
                    ),
                })
        elif r.status_code == 503:
            checks.append({
                "name": "output_cache",
                "status": "WARN",
                "detail": "not configured "
                "(PRSM_INFERENCE_OUTPUT_CACHE_ENABLED unset)",
            })
        else:
            checks.append({
                "name": "output_cache",
                "status": "FAIL",
                "detail": (
                    f"/admin/output-cache-stats returned "
                    f"{r.status_code}"
                ),
            })
    except Exception as exc:
        checks.append({
            "name": "output_cache",
            "status": "FAIL",
            "detail": f"probe raised: {exc}",
        })

    # ── peers ──
    try:
        r = _httpx.get(f"{url}/peers", timeout=5.0)
        if r.status_code == 200:
            pdata = r.json()
            connected = pdata.get("connected", []) or []
            n = len(connected)
            if n == 0:
                checks.append({
                    "name": "peers",
                    "status": "WARN",
                    "detail": (
                        "0 connected (node is isolated — check "
                        "bootstrap reachability)"
                    ),
                })
            else:
                checks.append({
                    "name": "peers",
                    "status": "PASS",
                    "detail": f"{n} connected",
                })
        else:
            checks.append({
                "name": "peers",
                "status": "FAIL",
                "detail": f"/peers returned {r.status_code}",
            })
    except Exception as exc:
        checks.append({
            "name": "peers",
            "status": "FAIL",
            "detail": f"probe raised: {exc}",
        })

    # ── aggregate overall ──
    any_fail = any(c["status"] == "FAIL" for c in checks)
    any_warn = any(c["status"] == "WARN" for c in checks)
    if any_fail:
        overall = "FAIL"
        exit_code = 1
    elif any_warn:
        overall = "WARN"
        exit_code = 0
    else:
        overall = "PASS"
        exit_code = 0

    if output_format == "json":
        click.echo(_json.dumps({
            "ok": overall != "FAIL",
            "checks": checks,
            "overall": overall,
        }, indent=2))
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    # text mode
    console.print("[bold]PRSM Node Doctor[/bold]")
    console.print("[dim]" + "─" * 16 + "[/dim]")
    color_map = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    for c in checks:
        color = color_map.get(c["status"], "white")
        console.print(
            f"[{color}][{c['status']}][/{color}] "
            f"[bold]{c['name']}[/bold]: {c['detail']}"
        )
    console.print("[dim]" + "─" * 16 + "[/dim]")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    overall_color = color_map[overall]
    console.print(
        f"[{overall_color}]Overall: {overall}[/{overall_color}] "
        f"({n_warn} warning{'s' if n_warn != 1 else ''}, "
        f"{n_fail} failure{'s' if n_fail != 1 else ''})"
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


@node.command("inference-status")
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_inference_status_cli(
    api_url_override: Optional[str], output_format: str,
) -> None:
    """Sprint 1304 — is real inference ready to serve? ONE command.

    Reads the daemon's readiness probe (/readyz) and surfaces whether the node can
    serve inference and — for the local executor — whether the model is actually
    LOADED yet (sp1302 pre-warms it in the background, so there's a window where the
    executor is wired but still warming). Exit codes:
      0 — ready to serve inference
      1 — NOT ready (no executor wired / inference disabled)
      2 — daemon unreachable at /readyz
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)

    try:
        r = _httpx.get(f"{url}/readyz", timeout=5.0)
    except Exception as exc:
        msg = f"daemon unreachable at {url}/readyz: {exc}"
        if output_format == "json":
            click.echo(_json.dumps({"ready": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(2)

    try:
        detail = r.json()
    except Exception:
        detail = {}
    ready = bool(detail.get("ready"))
    inf = detail.get("inference_detail") or {}

    if output_format == "json":
        click.echo(_json.dumps(detail))
    else:
        console.print(
            f"Inference: [{'green' if ready else 'red'}]"
            f"{'READY' if ready else 'NOT READY'}[/]"
        )
        if inf.get("enabled"):
            loaded = bool(inf.get("loaded"))
            console.print(f"  executor: {inf.get('kind', 'local')}")
            console.print(f"  model:    {inf.get('model_id', '?')}")
            console.print(
                f"  loaded:   [{'green' if loaded else 'yellow'}]{loaded}[/]"
                + ("" if loaded else "  (pre-warming — first request will load it)")
            )
            if loaded and inf.get("device"):
                console.print(f"  device:   {inf['device']}")
        else:
            # non-local executor (parallax/mock) or none — readiness 'ready' still
            # authoritative; the loaded-state surface is local-executor specific.
            kind = inf.get("kind")
            console.print(
                f"  executor: {kind if kind else '(none — not a local executor)'}"
            )
        if not ready and detail.get("reason"):
            # escape: the reason can contain bracketed text (e.g. ".[ml]") that rich
            # would otherwise parse as a style tag and strip.
            from rich.markup import escape as _rich_escape
            console.print(f"  [dim]{_rich_escape(str(detail['reason']))}[/dim]")

    raise SystemExit(0 if ready else 1)


@node.command("output-cache-stats")
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_output_cache_stats_cli(
    api_url_override: Optional[str], output_format: str,
) -> None:
    """Sprint 815 — show the output cache hit/miss/evict snapshot.

    Wraps GET /admin/output-cache-stats. Useful for tuning
    PRSM_INFERENCE_OUTPUT_CACHE_TTL_S + _MAX_ENTRIES against
    actual workload.

    Exit 0 success, 1 cache unconfigured (503), 2 unreachable.
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/admin/output-cache-stats"
    try:
        resp = _httpx.get(endpoint, timeout=10.0)
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]cache-stats query failed "
                f"({resp.status_code}):[/red] {resp.text}"
            )
            if resp.status_code == 503:
                console.print(
                    "[dim]Hint: set [bold]PRSM_INFERENCE_OUTPUT_CACHE_ENABLED"
                    "=1[/bold] in your env + restart the "
                    "daemon to enable cache.[/dim]"
                )
        raise SystemExit(1)
    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return

    hits = int(data.get("hits", 0))
    misses = int(data.get("misses", 0))
    total = hits + misses
    hit_rate_pct = (
        round((hits / total) * 100, 1) if total > 0 else 0.0
    )
    console.print(
        f"[bold]Output cache stats:[/bold]"
    )
    console.print(
        f"  hits=[green]{hits}[/green]  "
        f"misses=[yellow]{misses}[/yellow]  "
        f"hit_rate=[cyan]{hit_rate_pct}%[/cyan]"
    )
    console.print(
        f"  puts={data.get('puts', 0)}  "
        f"evictions={data.get('evictions', 0)}  "
        f"ttl_evictions={data.get('ttl_evictions', 0)}"
    )
    console.print(
        f"  size=[bold]{data.get('size', 0)}[/bold] / "
        f"max_entries={data.get('max_entries', '?')}  "
        f"ttl_seconds={data.get('ttl_seconds', '?')}"
    )


@node.command("partial-completion-history")
@click.option(
    "--limit", "limit", default=50, type=int,
    help="Max entries to return (default 50, max 1000).",
)
@click.option(
    "--offset", "offset", default=0, type=int,
    help="Pagination offset (default 0).",
)
@click.option(
    "--operator-node-id", "operator_node_id", default=None,
    help="Filter to a single operator_node_id (32-char hex).",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
def node_partial_completion_history_cli(
    limit: int, offset: int, operator_node_id: Optional[str],
    output_format: str, api_url_override: Optional[str],
) -> None:
    """Sprint 799 — show recent partial-completion slash events.

    Queries /admin/partial-completion-events. Each entry records
    a settle-time `should_slash=True` decision (sprints 784/785)
    persisted by the sprint-798 ring.

    Exit 0 on success, 1 on daemon error, 2 on unreachable.
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/admin/partial-completion-events"
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if operator_node_id:
        params["operator_node_id"] = operator_node_id
    try:
        resp = _httpx.get(endpoint, params=params, timeout=10.0)
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]history query failed "
                f"({resp.status_code}):[/red] {resp.text}"
            )
        raise SystemExit(1)
    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return

    entries = data.get("entries", [])
    total = data.get("total", 0)
    if not entries:
        console.print(
            "[dim]No partial-completion events recorded.[/dim] "
            "Operator-attributable receipt errors (reason=error) "
            "would appear here. Healthy operators see zero."
        )
        return
    console.print(
        f"[bold]Partial-completion events[/bold] "
        f"({len(entries)} of {total} shown):"
    )
    for e in entries:
        console.print(
            f"  [dim]ts={e.get('timestamp'):.0f}[/dim]  "
            f"job=[cyan]{e.get('job_id')}[/cyan]  "
            f"reason=[yellow]{e.get('reason')}[/yellow]  "
            f"tokens={e.get('tokens_completed')}/"
            f"{e.get('tokens_requested')}"
        )


@node.command("heartbeats")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def node_heartbeats(api_port, output_format, limit):
    """Show recent on-chain heartbeats."""
    _node_admin_history(
        api_port=api_port,
        path="/admin/heartbeat-history",
        label="Heartbeats",
        output_format=output_format,
        limit=limit,
        row_renderer=_render_heartbeat_row,
    )


@node.command("distributions")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def node_distributions(api_port, output_format, limit):
    """Show recent on-chain Distributed events."""
    _node_admin_history(
        api_port=api_port,
        path="/admin/distribution-history",
        label="Distributions",
        output_format=output_format,
        limit=limit,
        row_renderer=_render_distribution_row,
    )


@node.command("webhooks")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def node_webhooks(api_port, output_format, limit):
    """Show recent webhook dispatch attempts."""
    _node_admin_history(
        api_port=api_port,
        path="/admin/webhook-history",
        label="Webhooks",
        output_format=output_format,
        limit=limit,
        row_renderer=_render_webhook_row,
    )


@node.command("bootstrap")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_bootstrap(api_port: int, output_format: str):
    """Show P2P bootstrap status: primary, fallback,
    active URL, SPOF posture.

    Sprint 380 — third surface for the sprint-375 multi-
    bootstrap fields. Same data as /bootstrap/status JSON +
    prsm_bootstrap_status MCP tool; the CLI completes the
    operator-trifecta (REST / MCP / shell).
    """
    import json
    import httpx

    url = f"http://127.0.0.1:{api_port}/bootstrap/status"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at "
            f"{url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code == 503:
        detail = "(not wired)"
        try:
            detail = resp.json().get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        console.print(
            f"[yellow]Bootstrap discovery not wired.[/yellow]"
            f"\n[dim]{detail}[/dim]"
        )
        sys.exit(1)
    if resp.status_code != 200:
        console.print(
            f"[red]/bootstrap/status returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)

    body = resp.json()

    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    # Health summary marker — operators triage by color.
    connected = int(body.get("connected", 0) or 0)
    degraded = bool(body.get("degraded", False))
    client_state = body.get("client_state", "?")
    if connected > 0 and not degraded and (
        client_state == "connected"
    ):
        marker = "[green]✓ healthy[/green]"
    elif degraded or client_state == "dead":
        marker = "[red]⚠ degraded[/red]"
    else:
        marker = "[yellow]⚠ disconnected[/yellow]"

    console.print(f"[bold]PRSM Bootstrap Status[/bold] — {marker}")
    console.print(f"  client_state:           {client_state}")
    console.print(f"  connected:              {connected}")
    console.print(f"  attempted:              {body.get('attempted', 0)}")
    console.print(
        f"  discovered peers:       "
        f"{body.get('discovered_peer_count', 0)}"
    )

    # Sprint 375/376 — active_url + fallback config
    active_url = body.get("active_url")
    if active_url:
        # Strip scheme for compact rendering — same pattern
        # as the prsm_node_health MCP wrapper.
        short = active_url
        if "://" in short:
            short = short.split("://", 1)[1]
        console.print(
            f"  [bold]active URL:[/bold]             {short}"
        )
    else:
        console.print(
            "  [bold]active URL:[/bold]             "
            "[dim](none — all candidates failed)[/dim]"
        )

    fb_enabled = body.get("bootstrap_fallback_enabled")
    if fb_enabled is not None:
        if fb_enabled:
            console.print(
                "  fallback enabled:       [green]yes[/green]"
            )
        else:
            console.print(
                "  fallback enabled:       "
                "[yellow]no (single-host posture)[/yellow]"
            )

    # Counter snapshot (sprint 324)
    console.print()
    console.print(
        "  peer_join_events:       "
        f"{body.get('peer_join_events', 0)}"
    )
    console.print(
        "  peer_leave_events:      "
        f"{body.get('peer_leave_events', 0)}"
    )
    console.print(
        "  stale_evictions:        "
        f"{body.get('stale_evictions', 0)}"
    )
    console.print(
        "  reconnect_attempts:     "
        f"{body.get('reconnect_attempts', 0)}"
    )
    console.print(
        "  reconnect_successes:    "
        f"{body.get('reconnect_successes', 0)}"
    )

    # Candidate URLs (primary + fallback)
    bnodes = body.get("bootstrap_nodes") or []
    if bnodes:
        console.print()
        console.print(
            f"  bootstrap_nodes ({len(bnodes)} primary):"
        )
        for n in bnodes:
            marker = (
                "[green]●[/green]" if n == active_url
                else "[dim]○[/dim]"
            )
            console.print(f"    {marker} {n}")

    fb_nodes = body.get("bootstrap_fallback_nodes") or []
    if fb_nodes:
        console.print(
            f"  fallback_nodes ({len(fb_nodes)}):"
        )
        for n in fb_nodes:
            marker = (
                "[green]●[/green]" if n == active_url
                else "[dim]○[/dim]"
            )
            console.print(f"    {marker} {n}")


@node.command("bootstrap-test")
@click.option(
    "--url", "urls", multiple=True,
    help=(
        "Bootstrap URL(s) to test. Repeatable. When unset, "
        "tests the canonical fleet (US + EU + APAC defaults "
        "from prsm/node/config.py)."
    ),
)
@click.option(
    "--timeout", default=10.0, type=float,
    help="Per-host probe timeout in seconds (default 10)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
def node_bootstrap_test(urls, timeout, output_format):
    """Probe canonical bootstrap fleet from this machine.

    Sprint 385 — operator-trifecta complement to
    `prsm node bootstrap` (sprint 380). That one shows
    THIS node's bootstrap-registration state. This one
    probes ALL canonical bootstraps from wherever the
    operator is standing and reports TCP / TLS / WSS
    health for each. Diagnostic for "is my regional
    bootstrap up, or is something local broken?"

    Doesn't require a running PRSM node.
    """
    import asyncio
    import json
    from prsm.cli_helpers.bootstrap_probe import (
        ProbeStatus,
        canonical_bootstrap_urls,
        probe_fleet,
    )

    target_urls = list(urls) if urls else (
        canonical_bootstrap_urls()
    )
    if not target_urls:
        console.print(
            "[red]No bootstrap URLs to test.[/red]\n"
            "[dim]Pass --url <wss://...> or check your "
            "BOOTSTRAP_PRIMARY env vars.[/dim]"
        )
        sys.exit(2)

    fleet = asyncio.run(
        probe_fleet(target_urls, timeout_seconds=timeout),
    )

    if output_format == "json":
        console.print(json.dumps(fleet.to_dict(), indent=2))
        sys.exit(0 if fleet.all_healthy else 1)

    # Header marker
    if fleet.all_healthy:
        marker = "[green]✓ all healthy[/green]"
    elif fleet.any_healthy:
        marker = "[yellow]⚠ partial[/yellow]"
    else:
        marker = "[red]⚠ all degraded[/red]"
    console.print(
        f"[bold]PRSM Bootstrap Fleet Probe[/bold] — "
        f"{marker} "
        f"({fleet.healthy_count}/{fleet.total_count} reachable)"
    )
    console.print()

    for h in fleet.hosts:
        if h.status == ProbeStatus.OK:
            status_str = "[green]✓ ok[/green]"
        elif h.status == ProbeStatus.TIMEOUT:
            status_str = "[yellow]⚠ timeout[/yellow]"
        else:
            status_str = (
                f"[red]✗ {h.status.value}[/red]"
            )
        url_short = h.url
        if "://" in url_short:
            url_short = url_short.split("://", 1)[1]
        latency_str = (
            f"{h.latency_ms:.0f}ms"
            if h.latency_ms is not None else "-"
        )
        console.print(
            f"  {status_str:<22}  {url_short:<50}  "
            f"{latency_str}"
        )
        # Per-layer detail line
        layer_marks = []
        for label, ok in (
            ("TCP", h.tcp_ok),
            ("TLS", h.tls_ok),
            ("WSS", h.wss_ok),
        ):
            if ok:
                layer_marks.append(
                    f"[green]{label}[/green]"
                )
            else:
                layer_marks.append(f"[dim]{label}[/dim]")
        console.print(
            "    " + " · ".join(layer_marks)
            + (
                f"  ([dim]cert: {h.cert_subject} "
                f"issued by {h.cert_issuer}[/dim])"
                if h.cert_subject else ""
            )
        )
        # Sprint 591 — surface SAN-mismatch warning in text output
        # (sprint 590 already populates h.san_mismatch + JSON; text
        # output had only the subject CN before this sprint).
        if getattr(h, "san_mismatch", False):
            san_list = ", ".join(h.cert_san_dns) if h.cert_san_dns else "<empty>"
            console.print(
                f"    [yellow]⚠ SAN mismatch[/yellow] — "
                f"cert covers [{san_list}], "
                f"not [cyan]{h.host}[/cyan]. "
                f"Strict-TLS clients will fail."
            )
        if h.error:
            console.print(f"    [red]error:[/red] {h.error}")

    # Exit code mirrors the CI-friendly contract:
    #   0 = all reachable
    #   1 = some degraded
    #   2 = all degraded
    if fleet.all_healthy:
        sys.exit(0)
    elif fleet.any_healthy:
        sys.exit(1)
    else:
        sys.exit(2)


def _node_admin_trigger(*, api_port: int, path: str, label: str):
    """Shared helper for action triggers — POSTs to admin
    endpoints, renders tx_hash + status."""
    import httpx

    url = f"http://127.0.0.1:{api_port}{path}"
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code == 503:
        detail = resp.json().get("detail", "not wired")
        console.print(
            f"[yellow]{label} unavailable.[/yellow]\n"
            f"[dim]{detail}[/dim]"
        )
        sys.exit(1)
    if resp.status_code == 502:
        detail = resp.json().get("detail", "chain error")
        console.print(
            f"[red]{label} failed on-chain.[/red]\n"
            f"[dim]{detail}[/dim]"
        )
        sys.exit(1)
    if resp.status_code == 202:
        # sp915 — broadcast SUCCEEDED but the receipt is unconfirmed. NOT a
        # failure: re-triggering would race/revert + burn gas. Render the
        # pending signal + tx_hash and exit 0 so the operator reconciles.
        body = resp.json()
        console.print(
            f"[yellow]{label} broadcast but UNCONFIRMED — do NOT "
            f"re-trigger.[/yellow]\n"
            f"  tx_hash: {body.get('tx_hash', '?')}\n"
            f"  [dim]{body.get('detail', 'Reconcile via tx_hash.')}[/dim]"
        )
        return
    if resp.status_code != 200:
        console.print(
            f"[red]{path} returned {resp.status_code}[/red]: "
            f"{resp.text}"
        )
        sys.exit(1)

    body = resp.json()
    tx_hash = body.get("tx_hash", "?")
    status = body.get("status", "?")
    console.print(
        f"[green]{label} submitted on-chain.[/green]\n"
        f"  tx_hash: {tx_hash}\n"
        f"  status:  {status}"
    )


@node.command("trigger-heartbeat")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--yes", "-y", is_flag=True,
    help="Skip confirmation prompt",
)
def node_trigger_heartbeat(api_port, yes):
    """Manually record a heartbeat on-chain.

    Use when the HeartbeatScheduler crashed/paused and you
    want to avoid the slashing window opening. Caller pays gas.
    """
    if not yes:
        click.confirm(
            "This will submit an on-chain transaction. Continue?",
            abort=True,
        )
    _node_admin_trigger(
        api_port=api_port,
        path="/admin/heartbeat/trigger",
        label="Heartbeat",
    )


@node.command("trigger-distribution")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--yes", "-y", is_flag=True,
    help="Skip confirmation prompt",
)
def node_trigger_distribution(api_port, yes):
    """Manually invoke pull_and_distribute on-chain.

    Use when the PullAndDistributeScheduler crashed/paused or
    to force an emission round before the next cadence tick.
    Caller pays gas.
    """
    if not yes:
        click.confirm(
            "This will submit an on-chain transaction. Continue?",
            abort=True,
        )
    _node_admin_trigger(
        api_port=api_port,
        path="/admin/distribution/trigger",
        label="Distribution",
    )


@node.command("claim-royalty")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--execute", is_flag=True,
    help="Execute the claim on-chain (default: dry-run only)",
)
def node_claim_royalty(api_port, execute):
    """Claim accumulated royalties from RoyaltyDistributor.

    Default: dry-run that shows claimable amount without
    on-chain action. Pass --execute to send the tx.
    """
    import json
    import httpx

    url = f"http://127.0.0.1:{api_port}/wallet/royalty/claim"
    payload = {"dry_run": not execute}
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code == 503:
        detail = resp.json().get("detail", "not wired")
        console.print(
            f"[yellow]RoyaltyDistributor not wired.[/yellow]\n"
            f"[dim]{detail}[/dim]"
        )
        sys.exit(1)

    if resp.status_code == 202:
        # sp915 — claim broadcast SUCCEEDED but unconfirmed. Re-claiming
        # would race the first tx + revert ZeroClaim. Render pending + exit 0.
        body = resp.json()
        console.print(
            f"[yellow]Royalty claim broadcast but UNCONFIRMED — do NOT "
            f"re-claim.[/yellow]\n"
            f"  tx_hash: {body.get('tx_hash', '?')}\n"
            f"  [dim]{body.get('detail', 'Reconcile via tx_hash.')}[/dim]"
        )
        return
    if resp.status_code != 200:
        console.print(
            f"[red]/wallet/royalty/claim returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)

    body = resp.json()
    status = body.get("status", "?")
    if status == "DRY_RUN":
        wei = body.get("claimable_wei", 0)
        ftns = wei / 1e18
        console.print(
            f"[bold]Dry run — no on-chain action[/bold]\n"
            f"  Claimable: {ftns:.6f} FTNS ({wei} wei)\n"
            f"  Re-run with --execute to claim on-chain."
        )
    elif status == "SKIPPED_ZERO":
        console.print(
            "[yellow]Nothing to claim — claimable balance is 0.[/yellow]"
        )
    elif status == "EXECUTED":
        tx_hash = body.get("tx_hash", "?")
        wei = body.get("amount_claimed_wei", 0)
        ftns = wei / 1e18
        console.print(
            f"[green]Royalty claim submitted on-chain.[/green]\n"
            f"  Amount:  {ftns:.6f} FTNS\n"
            f"  tx_hash: {tx_hash}"
        )
    else:
        console.print(json.dumps(body, indent=2))


@node.command("install")
@click.option("--dry-run", is_flag=True, help="Print service file without installing")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def node_install(dry_run: bool, host: str, port: int):
    """Install node as a system service (launchd / systemd)."""
    from prsm.cli_modules.daemon import daemon_service_install as _install
    _install(dry_run=dry_run, host=host, port=port)


@node.command("uninstall")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def node_uninstall(yes: bool):
    """Remove the node system service."""
    from prsm.cli_modules.daemon import daemon_service_uninstall as _uninstall
    _uninstall(yes=yes)


@node.command()
def peers():
    """List connected + known peers (sprint 574 enhanced)."""
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity

    config = NodeConfig.load()
    identity = load_node_identity(config.identity_path)

    if not identity:
        console.print("No node identity found. Run 'prsm setup' first.", style="yellow")
        return

    console.print(f"Node ID: {identity.node_id}", style="bold cyan")
    console.print(f"P2P address: ws://{config.listen_host}:{config.p2p_port}", style="cyan")
    console.print()

    # Try connecting to the running node's API to get live peer data
    # Sprint 574: honor PRSM_API_PORT env override (operators on
    # non-default ports e.g. droplet at 8002)
    try:
        import httpx, os as _os
        _env_port = (_os.environ.get("PRSM_API_PORT", "") or "").strip()
        try:
            _api_port = int(_env_port) if _env_port else config.api_port
        except ValueError:
            _api_port = config.api_port
        resp = httpx.get(f"http://127.0.0.1:{_api_port}/peers", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            connected = data.get("connected", [])
            known = data.get("known", [])

            connected_ids = {p["peer_id"] for p in connected}

            if connected:
                table = Table(title="Connected Peers")
                table.add_column("Peer ID", style="cyan")
                table.add_column("Address", style="green")
                table.add_column("Name", style="magenta")
                table.add_column("Direction", style="blue")
                for p in connected:
                    table.add_row(
                        p["peer_id"][:16] + "...",
                        p["address"],
                        p.get("display_name", ""),
                        "outbound" if p.get("outbound") else "inbound",
                    )
                console.print(table)
            else:
                console.print("No peers connected.", style="dim")

            # Sprint 574 — also show known-but-unconnected so operators
            # see the gap between bootstrap-discovered and actually-dialed
            known_only = [
                k for k in known if k.get("node_id") not in connected_ids
            ]
            if known_only:
                console.print()
                tbl2 = Table(title="Known (not connected)")
                tbl2.add_column("Peer ID", style="cyan")
                tbl2.add_column("Address", style="green")
                tbl2.add_column("Name", style="magenta")
                tbl2.add_column("Capabilities", style="dim")
                for k in known_only:
                    tbl2.add_row(
                        (k.get("node_id", "") or "")[:16] + "...",
                        k.get("address", ""),
                        k.get("display_name", "") or "",
                        ", ".join(k.get("capabilities", []) or []),
                    )
                console.print(tbl2)

            console.print(f"\nConnected: {data.get('connected_count', 0)}  "
                          f"Known: {data.get('known_count', 0)}")
            return
    except Exception:
        pass

    console.print("Node is not running. Start it with 'prsm node start'.", style="yellow")


# ── Sprint 585 — §7 readiness aggregate ──────────────────────────


def _probe_anchor_outcome():
    """Returns (outcome, error_str_or_None) for anchor construction."""
    import os as _os
    addr = (_os.environ.get("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "") or "").strip()
    rpc = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    if not addr:
        return "unset", None
    try:
        from prsm.security.publisher_key_anchor.client import (
            PublisherKeyAnchorClient,
        )
        PublisherKeyAnchorClient(contract_address=addr, rpc_url=rpc)
        return "ok", None
    except Exception as exc:  # noqa: BLE001
        return "construction_failed", f"{type(exc).__name__}: {exc}"


def _probe_stake_bond_outcome():
    import os as _os
    addr = (_os.environ.get("PRSM_STAKE_BOND_ADDRESS", "") or "").strip()
    rpc = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    if not addr:
        return "unset", None
    try:
        from prsm.economy.web3.stake_manager import StakeManagerClient
        StakeManagerClient(contract_address=addr, rpc_url=rpc)
        return "ok", None
    except Exception as exc:  # noqa: BLE001
        return "construction_failed", f"{type(exc).__name__}: {exc}"


def _probe_rpc_outcome():
    import os as _os
    import httpx as _httpx
    rpc = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    try:
        resp = _httpx.post(
            rpc,
            json={
                "jsonrpc": "2.0", "method": "eth_chainId",
                "params": [], "id": 1,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            return "error", f"HTTP {resp.status_code}"
        body = resp.json()
        if body.get("result") is None:
            return "error", f"no result: {body!r}"[:200]
        return "ok", None
    except _httpx.HTTPError as exc:
        return "unreachable", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "error", f"{type(exc).__name__}: {exc}"


@node.command("section7-readiness")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def section7_readiness_cli(output_format: str):
    """Aggregate §7 production-readiness check.

    Sprint 585 — runs anchor-probe + stake-bond-probe + rpc-probe
    in one shot, reports overall readiness. Exit 0 only when ALL
    three probes return ok (suitable for CI gating).
    """
    import json
    components = {
        "anchor": dict(zip(("outcome", "error"), _probe_anchor_outcome())),
        "stake_bond": dict(zip(("outcome", "error"), _probe_stake_bond_outcome())),
        "rpc": dict(zip(("outcome", "error"), _probe_rpc_outcome())),
    }
    overall = (
        "ready"
        if all(c["outcome"] == "ok" for c in components.values())
        else "not_ready"
    )
    payload = {"overall": overall, "components": components}

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2))
        raise SystemExit(0 if overall == "ready" else 1)

    table = Table(title="§7 production-readiness (sprint 585)")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Outcome", style="green", no_wrap=True)
    table.add_column("Error", style="red", no_wrap=False)
    for name, c in components.items():
        out = c["outcome"]
        out_style = "[green]✓ ok[/green]" if out == "ok" else f"[red]{out}[/red]"
        table.add_row(name, out_style, c.get("error") or "")
    console.print(table)
    if overall == "ready":
        console.print(
            "\n[green]✓ ready[/green] — all three §7 components "
            "pass preflight. Safe to set "
            "PRSM_PARALLAX_TRUST_STACK_KIND=production."
        )
    else:
        console.print(
            "\n[yellow]not_ready[/yellow] — fix the failing "
            "component(s) above before flipping production."
        )
        raise SystemExit(1)


# ── Sprint 696 — Parallax env-var readiness ─────────────────────


_PARALLAX_ENV_REGISTRY = [
    # (env_var, required, valid_values_or_None, description)
    ("PRSM_INFERENCE_EXECUTOR", True, ["parallax", "mock", ""],
     "Inference executor kind. 'parallax' wires the real ParallaxScheduledExecutor."),
    ("PRSM_PARALLAX_GPU_POOL_KIND", True, ["dht-backed", "static-empty"],
     "GPU pool provider. 'dht-backed' reads peers from discovery."),
    ("PRSM_PARALLAX_TRUST_STACK_KIND", True, ["production", "mock"],
     "Trust stack kind. 'production' uses real anchor + stake lookups."),
    ("PRSM_PARALLAX_MODEL_CATALOG_FILE", True, None,
     "Path to model catalog JSON (e.g., config/parallax/model_catalog.json)."),
    ("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", False, None,
     "PublisherKeyAnchor contract address (required when TRUST_STACK_KIND=production)."),
    ("PRSM_PARALLAX_CHAIN_EXECUTOR_KIND", False, ["rpc", "stub", ""],
     "Chain executor. 'rpc' = real cross-host dispatch; default 'stub'."),
    ("PRSM_PARALLAX_STAGE_EXECUTOR_KIND", False, ["layer_stage", "stub", "echo", ""],
     "Stage executor. 'layer_stage' = real HF model forward pass."),
    ("PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND", False,
     ["huggingface", "identity", ""],
     "Layer runner. 'huggingface' loads a real HF model."),
    ("PRSM_PARALLAX_HF_MODEL_ID", False, None,
     "HF model id (e.g., 'gpt2'). Required when LAYER_SLICE_RUNNER_KIND=huggingface."),
    ("PRSM_PARALLAX_HF_DEVICE", False, ["cpu", "cuda", "mps", "auto", ""],
     "HF compute device. 'cpu' default; 'cuda' for GPU; 'auto' detects."),
    ("PRSM_PARALLAX_PROMPT_ENCODER_KIND", False, ["huggingface", ""],
     "Prompt encoder. 'huggingface' uses real tokenizer; default is utf8 byte-passthrough."),
    ("PRSM_PARALLAX_OUTPUT_DECODER_KIND", False, ["huggingface", ""],
     "Output decoder. Set together with PROMPT_ENCODER_KIND for real generation."),
    ("PRSM_PARALLAX_STREAMING_RUNNER_KIND", False,
     ["embedder_backed", "synthetic", ""],
     "SSE streaming runner. 'embedder_backed' = real autoregressive (default)."),
    ("PRSM_PARALLAX_KV_CACHE_ENABLED", False, ["1", "0", "true", "false", ""],
     "KV cache toggle for streaming. Default off; '1' enables."),
    ("PRSM_PARALLAX_STAKE_ELIGIBILITY", False, ["enforced", "advisory", ""],
     "Stake eligibility mode. 'enforced' = production; 'advisory' = pre-stake-ceremony bypass."),
    ("PRSM_PARALLAX_TIER_GATE", False, ["enforced", "advisory", ""],
     "Tier gate mode (Adapter B). 'enforced' = require hardware-TEE for tier>=standard (production); 'advisory' = software-only operators can serve tier=standard for DP-injection live-attest."),
    ("PRSM_STAKE_BOND_ADDRESS", False, None,
     "StakeBond contract address (used by PoolBackedStakeLookup)."),
    ("PRSM_OPERATOR_ADDRESS", False, None,
     "Operator's EOA address (advertised in hardware_profile for stake lookup)."),
    ("PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE", False, None,
     "Override per-GPU layer_capacity (int). Useful for small models on small GPUs."),
    ("PRSM_PARALLAX_MEMORY_GB_OVERRIDE", False, None,
     "Override advertised memory_gb (float). Forces multi-stage allocation when needed."),
    ("PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE", False, None,
     "Override advertised tflops_fp16 (float). Useful when CPU benchmark is too low."),
    ("PRSM_PARALLAX_DEFAULT_RTT_MS", False, None,
     "Default inter-peer RTT in ms. Required for Phase-2 routing on pools without profile measurements."),
    ("PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE", False, ["1", "true", "yes", "0", "false", "no", ""],
     "Sprint 836 (F31) — admit peers with no hardware_profile under conservative synthetic defaults (1vCPU/1GB). Closes cold-start gossip gap: bootstrap doesn't propagate hardware_profile, so a fresh joiner sees known peers but no hw → DHT pool reports 0 GPUs. Default unset = strict (legacy behavior). Set to 1/true/yes for dogfood + multi-host testing."),
    ("PRSM_ANCHOR_LOOKUP_RETRY_ATTEMPTS", False, None,
     "Sprint 837 — PublisherKeyAnchor.lookup() retry count for transient RPC errors. Default 3 (1 initial + 2 retries with exp backoff). Multi-host dispatch on free mainnet.base.org RPC hits rate-limits — pre-837 a single 429 bricked chain inference. Cached lookups unaffected; only contract-call path retries. <1 → clamped to 1."),
    ("PRSM_ANCHOR_LOOKUP_RETRY_BACKOFF_S", False, None,
     "Sprint 837 — base backoff seconds for anchor.lookup() retries. Default 0.5s; exponential progression (0.5s → 1.0s → 2.0s). <0 → defaults to 0.5. Pairs with PRSM_ANCHOR_LOOKUP_RETRY_ATTEMPTS."),
    ("PRSM_PARALLAX_SEND_MESSAGE_TIMEOUT_S", False, None,
     "Per-stage dispatch timeout in seconds (default 30)."),
    ("PRSM_PARALLAX_REGION", False, None,
     "Region tag for ParallaxGPU (default 'default'). Allocator never spans regions."),
    ("PRSM_INFERENCE_CONCURRENCY_LIMIT", False, None,
     "Cap on concurrent in-flight inference requests (default unset = no cap). Set to 1 on memory-tight nodes (e.g., 2GB droplets) to prevent OOM under simultaneous cold-load."),
    ("PRSM_CHAIN_STREAM_QUEUE_MAXSIZE", False, None,
     "Sprint 713 — bounded receive queue for remote token-stream back-pressure. Default 64 frames; <=0 = unbounded (pre-713 behavior); non-int safely defaults to 64. Producer blocks at queue.put when full → real back-pressure (not lossy drop)."),
    ("PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES", False, None,
     "Sprint 721 — max bytes accepted by server-side stream request handler before b64-decode (memory-DoS defense). Default 16 MiB (covers a real gpt2 activation blob with 100-token context); <=0 = unbounded; non-int safely defaults to 16 MiB."),
    ("PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY", False, None,
     "Sprint 723 — max concurrent CHAIN_STREAM_REQs from a single peer (per-peer DoS defense). Default 8 (covers realistic multi-stream coordinator workloads); <=0 = unbounded; non-int safely defaults to 8."),
    ("PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES", False, None,
     "Sprint 725 — max bytes accepted by server-side unary CHAIN_REQ handler before b64-decode (memory-DoS defense; F55 sibling on unary path). Default 16 MiB (matches streaming limit); <=0 = unbounded; non-int safely defaults to 16 MiB."),
    ("PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY", False, None,
     "Sprint 726 — max concurrent unary CHAIN_REQs from a single peer (per-peer DoS defense; F56 sibling on unary path). Default 8 (matches streaming cap); <=0 = unbounded; non-int safely defaults to 8."),
    ("PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S", False, None,
     "Sprint 728 — max wall-clock seconds for executor.execute() on a single unary CHAIN_REQ. Defends against hung executor holding per-peer cap slot indefinitely. Default 60s (covers gpt2/100-token on CPU); <=0 = no timeout; non-float safely defaults to 60s."),
    ("PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S", False, None,
     "Sprint 729 — max wall-clock seconds for a server-side streaming response (cumulative across all yielded frames). Defends against slow/malicious StageExecutor holding sprint-723 per-peer cap slot. Default 300s (5 minutes; covers gpt2/200-token CPU streaming); <=0 = no timeout; non-float safely defaults to 300s."),
    ("PRSM_ADMIN_REMOTE_ALLOWED", False, ["1", "true", "yes", "0", "false", "no", ""],
     "Sprint 734 — allow remote (non-loopback) access to /admin/* endpoints. Default unset = SAFE DENY (only 127.0.0.1/::1/localhost can hit /admin/*). Set to 1/true/yes only when behind reverse-proxy auth or a VPN; otherwise leaks expected_sender peer IDs + KYC records + moderation state to any peered network client."),
    ("PRSM_HTTP_MAX_BODY_BYTES", False, None,
     "Sprint 742 — max HTTP request body size (memory-DoS defense at the HTTP layer; sibling of F55/F58 on the wire protocol). Default 1 MiB (covers reasonable prompts + metadata); <=0 = disabled; non-int safely defaults to 1 MiB. Returns 413 Payload Too Large when Content-Length exceeds limit BEFORE reading the body."),
    ("PRSM_API_DOCS_ENABLED", False, ["1", "true", "yes", "0", "false", "no", ""],
     "Sprint 744 — enable /docs + /redoc + /openapi.json surface. Default unset/0 = HIDDEN (production-safe; attackers don't get a free API-surface map). Set to 1/true/yes for dev so operators can use the interactive Swagger docs in a browser."),
    ("PRSM_ACTIVE_HOURS", False, None,
     "Sprint 755-756 — operator-controlled active window as 'HH:MM-HH:MM' (e.g., '22:00-08:00' for overnight only). Outside the window the daemon refuses inference dispatch (503 with Retry-After) AND skips discovery announces (peers evict from routing pool). Default unset = always-active (backward-compat). Cross-midnight ranges OK."),
    ("PRSM_ACTIVE_TIMEZONE", False, None,
     "Sprint 755-756 — IANA timezone name for PRSM_ACTIVE_HOURS (e.g., 'America/New_York', 'Europe/London'). Default 'UTC'. Operator's wall-clock time is what matters — pick a tz that matches their schedule."),
    ("PRSM_STORAGE_UPLOAD_MBPS", False, None,
     "Sprint 761 — operator-facing upload bandwidth cap for the storage provider (content serving + shard transfer). Float Mbps; default 0 = unlimited (pre-761 behavior). Useful on metered/capped consumer ISPs and gaming PCs where the daemon shouldn't saturate upload during peak hours."),
    ("PRSM_STORAGE_DOWNLOAD_MBPS", False, None,
     "Sprint 761 — operator-facing download bandwidth cap for the storage provider. Float Mbps; default 0 = unlimited."),
    ("PRSM_NODE_NICE", False, None,
     "Sprint 762 — process-priority increment via os.nice(). Positive int (1-19); higher = lower CPU priority. Daemon yields CPU to operator's interactive workloads (browser, editor, game). Default unset = 0 (no change). Non-root processes can only INCREASE nice (lower priority); negative values are silently rejected by the OS with a warning log. Not available on Windows (os.nice not provided)."),
    ("PRSM_ACTIVE_ONLY_ON_AC", False, ["1", "true", "yes", "0", "false", "no", ""],
     "Sprint 763 — on laptops, only activate when plugged in (AC power). Set to 1/true/yes → daemon refuses inference + skips announces while on battery. Combines with PRSM_ACTIVE_HOURS (sprint 755) — both gates must pass. Fail-safe: when psutil sensor is unavailable OR no battery (desktop), treats as 'on AC' → active. Default unset = always-active regardless of power source."),
    ("PRSM_AUTO_CLAIM_THRESHOLD_FTNS", False, None,
     "Sprint 765 — auto-claim accumulated FTNS rewards when total reaches this threshold. Decimal value (e.g., '100' = claim at 100 FTNS). Default unset/0 = disabled. Operator opts in for set-and-forget earnings claiming. Combined with PRSM_AUTO_CLAIM_INTERVAL_S to control claim cadence."),
    ("PRSM_AUTO_CLAIM_INTERVAL_S", False, None,
     "Sprint 765 — seconds between auto-claim checks. Default 3600 (1 hour). Clamped to >= 60s. Only effective when PRSM_AUTO_CLAIM_THRESHOLD_FTNS is set."),
    ("PRSM_INFERENCE_OUTPUT_CACHE_ENABLED", False,
     ["", "0", "1", "true", "false", "yes", "no"],
     "Sprint 810/811 — opt-in output cache for deterministic "
     "inference. PRIVACY INVARIANT: only privacy_tier=none "
     "requests are cached; standard/high/maximum tiers require "
     "DP per-request and never enter the cache. Default unset = "
     "no cache (pre-811 behavior). Set to 1 to enable."),
    ("PRSM_INFERENCE_OUTPUT_CACHE_TTL_S", False, None,
     "Sprint 810 — per-entry TTL in seconds for the output "
     "cache. Default 3600 (1h). Lower values trade hit rate "
     "for freshness; higher values amortize compute across "
     "more requests but risk staleness when models change."),
    ("PRSM_INFERENCE_OUTPUT_CACHE_MAX_ENTRIES", False, None,
     "Sprint 810 — max entries in the LRU output cache. "
     "Default 1024. Each entry holds prompt+output bytes; "
     "memory cost scales with average output size."),
    ("PRSM_PREEMPTION_DETECTOR", False, ["", "aws", "gcp"],
     "Sprint 772 — cloud-spot preemption detector backend. 'aws' polls EC2 instance-action metadata (169.254.169.254). 'gcp' polls GCE preemptible-metadata. Unset/'' = disabled (safe default for non-cloud nodes). Future sprints wire the flag into discovery + dispatch gates. Fail-safe: metadata endpoint unreachable → flag stays clear."),
    ("PRSM_PREEMPTION_POLL_INTERVAL_S", False, None,
     "Sprint 772 — seconds between preemption-metadata polls. Default 10s. Lower = faster detection but more metadata-endpoint traffic. AWS/GCP preemption notices give ~2min warning so 10s is plenty."),
]


# Sprint 809 — recommended production defaults for `node init`.
# Only includes vars with STABLE recommended values that work for
# most operators. Operator-specific values (wallet addresses,
# model paths) deliberately stay placeholder.
_RECOMMENDED_DEFAULTS = {
    "PRSM_INFERENCE_EXECUTOR": "parallax",
    "PRSM_PARALLAX_GPU_POOL_KIND": "dht-backed",
    "PRSM_PARALLAX_TRUST_STACK_KIND": "production",
    "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND": "rpc",
    "PRSM_PARALLAX_STAGE_EXECUTOR_KIND": "layer_stage",
    "PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND": "huggingface",
    "PRSM_PARALLAX_HF_DEVICE": "cpu",
    "PRSM_PARALLAX_PROMPT_ENCODER_KIND": "huggingface",
}


@node.command("init")
@click.option(
    "--output", "output_path", default=None,
    help="Write template to PATH. Default: ~/.prsm/operator.env.",
)
@click.option(
    "--force", "force_overwrite", is_flag=True, default=False,
    help="Overwrite existing file (default refuses).",
)
@click.option(
    "--dry-run", "dry_run", is_flag=True, default=False,
    help="Print to stdout instead of writing.",
)
@click.option(
    "--recommended", "recommended_defaults",
    is_flag=True, default=False,
    help="Sprint 809 — pre-fill sensible production defaults "
    "for vars with stable recommended values "
    "(PRSM_INFERENCE_EXECUTOR=parallax, etc.). Operator-"
    "specific values (PRSM_OPERATOR_ADDRESS, "
    "PRSM_PARALLAX_HF_MODEL_ID) stay placeholder. Filled "
    "lines carry a '# recommended:' annotation.",
)
def node_init_env_template_cli(
    output_path: Optional[str], force_overwrite: bool,
    dry_run: bool, recommended_defaults: bool,
) -> None:
    """Sprint 800 — write a starter operator.env template.

    Pairs with `prsm node parallax-readiness` (sprint 696):
    readiness reports MISSING env vars; init writes a TEMPLATE
    with every documented var, descriptions, and placeholder
    values so the operator has a single file to fill out + drop
    into systemd via `EnvironmentFile=`.

    Default destination: ~/.prsm/operator.env (chmod 600 since
    the file may eventually hold PRIVATE_KEY).

    Exit codes:
      0 — written / dry-run printed
      1 — file exists + no --force
    """
    from pathlib import Path as _Path

    # Build template content
    lines = [
        "# ───────────────────────────────────────────────────",
        "# PRSM operator config — generated by `prsm node init`",
        "# ",
        "# Use:",
        "#   systemd:   EnvironmentFile=/path/to/operator.env",
        "#   shell:     set -a; source operator.env; set +a",
        "# ",
        "# Fill in the values you need. Required vars are marked",
        "# '# REQUIRED' above. Run `prsm node parallax-readiness`",
        "# after editing to validate.",
        "# ───────────────────────────────────────────────────",
        "",
    ]
    for name, required, valid_values, description in (
        _PARALLAX_ENV_REGISTRY
    ):
        if required:
            lines.append("# REQUIRED")
        # Wrap long descriptions for readability.
        for chunk in _wrap_for_template(description):
            lines.append(f"# {chunk}")
        if valid_values:
            # Only show non-empty allowed values
            allowed = [v for v in valid_values if v]
            if allowed:
                lines.append(
                    f"#   Allowed: {', '.join(allowed)}"
                )
        # Sprint 809 — recommended-defaults fill
        rec_val = (
            _RECOMMENDED_DEFAULTS.get(name)
            if recommended_defaults else None
        )
        if rec_val:
            lines.append(f"# recommended: {rec_val}")
            lines.append(f"{name}={rec_val}")
        else:
            lines.append(f"{name}=")
        lines.append("")
    content = "\n".join(lines)

    if dry_run:
        click.echo(content)
        return

    if output_path is None:
        resolved = _Path.home() / ".prsm" / "operator.env"
    else:
        resolved = _Path(output_path)

    if resolved.exists() and not force_overwrite:
        console.print(
            f"[red]{resolved} exists.[/red] Use --force to "
            "overwrite, or pass --output to write elsewhere."
        )
        raise SystemExit(1)

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        try:
            resolved.chmod(0o600)
        except OSError:
            pass  # non-posix
    except OSError as exc:
        console.print(f"[red]Write failed:[/red] {exc}")
        raise SystemExit(2)

    console.print(
        f"[green]Wrote operator env template[/green] to "
        f"[bold]{resolved}[/bold]. Edit it + drop into systemd "
        "via [bold]EnvironmentFile=[/bold] or source it in your "
        "shell. Then run [bold]prsm node parallax-readiness[/bold] "
        "to validate."
    )


def _wrap_for_template(text: str, width: int = 70) -> list[str]:
    """Wrap a description string into chunks ≤ `width` chars for
    multi-line `#` comments. Pure-functional; no I/O."""
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


@node.command("parallax-readiness")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def parallax_readiness_cli(output_format: str):
    """Sprint 696 — operator preflight for the 22 PRSM_PARALLAX_*
    env vars accumulated across sprints 558-695.

    Reports which env vars are set, validates values against
    known-good enumerations where applicable, and flags missing
    required vars + bad values. Read-only — does not start any
    daemon or touch on-chain.

    Exit 0 only when all REQUIRED vars are set + all set vars
    have valid values (suitable for CI gating).
    """
    import json as _json
    import os as _os
    rows = []
    missing_required: List[str] = []
    bad_values: List[Tuple[str, str]] = []
    for env_var, required, valid_values, desc in _PARALLAX_ENV_REGISTRY:
        raw = _os.environ.get(env_var, "").strip()
        if not raw:
            status = "unset" if not required else "MISSING (required)"
            if required:
                missing_required.append(env_var)
            rows.append((env_var, status, "", desc))
            continue
        if valid_values is not None and raw not in valid_values:
            status = f"INVALID: must be one of {valid_values}"
            bad_values.append((env_var, raw))
        else:
            status = "set"
        rows.append((env_var, status, raw, desc))

    overall = "ready" if not missing_required and not bad_values else "not_ready"
    payload = {
        "overall": overall,
        "missing_required": missing_required,
        "bad_values": [{"env": e, "value": v} for e, v in bad_values],
        "vars": [
            {"env": r[0], "status": r[1], "value": r[2], "description": r[3]}
            for r in rows
        ],
    }
    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        raise SystemExit(0 if overall == "ready" else 1)

    table = Table(title="Parallax readiness (sprint 696)")
    table.add_column("Env var", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Value", style="yellow", no_wrap=False, max_width=30)
    for env_var, status, value, _ in rows:
        if status == "set":
            s = "[green]✓ set[/green]"
        elif status == "unset":
            s = "[dim]unset[/dim]"
        elif status.startswith("MISSING"):
            s = f"[red]{status}[/red]"
        else:
            s = f"[red]{status}[/red]"
        table.add_row(env_var, s, value)
    console.print(table)
    if overall == "ready":
        console.print(
            "\n[green]✓ ready[/green] — all required parallax env "
            "vars are set with valid values."
        )
    else:
        if missing_required:
            console.print(
                f"\n[red]missing required:[/red] "
                f"{', '.join(missing_required)}"
            )
        if bad_values:
            console.print(
                f"\n[red]invalid values:[/red] "
                f"{', '.join(f'{e}={v}' for e, v in bad_values)}"
            )
        raise SystemExit(1)


# ── Sprint 722 — stream observability ────────────────────────────


@node.command("schedule")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_schedule_cli(output_format: str):
    """Sprint 755-756 — show the operator's active-window schedule.

    The daemon reads `PRSM_ACTIVE_HOURS` (e.g., '22:00-08:00') +
    optional `PRSM_ACTIVE_TIMEZONE` (default UTC). Outside the
    window, the daemon refuses inference dispatch (503) AND skips
    discovery announces (peers evict from routing pool).

    This CLI reads the same env and reports:
      - The configured window (or "always-active" if env unset)
      - Whether we're currently inside the window
      - The current time in the schedule's timezone (for sanity)

    Read-only. To change the schedule, edit your systemd unit's
    `Environment=PRSM_ACTIVE_HOURS=...` line and restart the
    daemon. File-based config can be added in a follow-up sprint
    if operators ask for an interactive set/clear CLI.
    """
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from prsm.node.schedule import (
        resolve_active_window_from_env,
    )

    try:
        window = resolve_active_window_from_env()
    except ValueError as exc:
        # Malformed env at startup-equivalent — surface the same
        # error the daemon would crash with.
        if output_format == "json":
            click.echo(_json.dumps({
                "configured": False,
                "error": str(exc),
            }))
            raise SystemExit(1)
        console.print(f"[red]Schedule config error:[/red] {exc}")
        console.print(
            "[yellow]Fix PRSM_ACTIVE_HOURS in your systemd unit, "
            "then restart the daemon.[/yellow]"
        )
        raise SystemExit(1)

    if window is None:
        if output_format == "json":
            click.echo(_json.dumps({
                "configured": False,
                "mode": "always-active",
                "is_currently_active": True,
            }))
            return
        console.print("[bold]Schedule:[/bold] [green]always-active[/green]")
        console.print(
            "[dim]PRSM_ACTIVE_HOURS not set. Daemon serves the "
            "network 24/7. To opt-out during specific hours, set "
            "PRSM_ACTIVE_HOURS in your systemd unit (e.g., "
            "'22:00-08:00' for overnight only).[/dim]"
        )
        return

    tz = ZoneInfo(window.tz_name)
    now = datetime.now(tz)
    is_active = window.is_active(now)

    if output_format == "json":
        click.echo(_json.dumps({
            "configured": True,
            "window": window.render(),
            "start": window.start.strftime("%H:%M"),
            "end": window.end.strftime("%H:%M"),
            "timezone": window.tz_name,
            "now_local": now.strftime("%H:%M:%S %Z"),
            "is_currently_active": is_active,
        }, indent=2))
        return

    # Rich text output
    color = "green" if is_active else "yellow"
    status_label = "ACTIVE" if is_active else "INACTIVE"
    console.print(
        f"[bold]Schedule:[/bold] [{color}]{window.render()}[/{color}]"
    )
    console.print(
        f"[bold]Now ({window.tz_name}):[/bold] "
        f"{now.strftime('%H:%M:%S')}"
    )
    console.print(
        f"[bold]Status:[/bold] [{color}]{status_label}[/{color}]"
    )
    if is_active:
        console.print(
            "[dim]Daemon is serving the network. Inference "
            "dispatch + discovery announces are active.[/dim]"
        )
    else:
        console.print(
            "[dim]Daemon is OUTSIDE its active window. Inference "
            "requests return 503 with Retry-After: 60. Discovery "
            "announces are skipped — peers will evict this node "
            "from their routing pool within ~60s.[/dim]"
        )


@node.command("claim-rewards")
@click.option(
    "--stake-id", "stake_id", default=None,
    help="Specific stake to claim from. Omit to claim ALL stakes.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
def node_claim_rewards_cli(
    stake_id: Optional[str], output_format: str,
    api_url_override: Optional[str],
):
    """Sprint 770 — manually claim accumulated staking rewards.

    Triggers an immediate claim_rewards via the daemon's
    /staking/claim-rewards endpoint. Useful when:
    - Operator wants to claim before the auto-claim threshold
      (sprint 765's PRSM_AUTO_CLAIM_THRESHOLD_FTNS) is reached
    - Operator wants to test the claim path
    - Auto-claim is disabled and the operator needs a one-shot

    Returns the total amount claimed + per-stake breakdown.
    Exits non-zero on failure.
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/staking/claim-rewards"
    params = {"stake_id": stake_id} if stake_id else None
    try:
        resp = _httpx.post(endpoint, params=params, timeout=30.0)
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]Claim failed ({resp.status_code}):[/red] "
                f"{resp.text}"
            )
        raise SystemExit(1)
    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return
    total = data.get("total_rewards_claimed", "0")
    stakes = data.get("stakes_processed", [])
    console.print(
        f"[bold]Claimed:[/bold] [green]{total}[/green] FTNS"
    )
    if stakes:
        console.print(
            f"[dim]From {len(stakes)} stake(s).[/dim]"
        )
    else:
        console.print(
            "[dim]No stakes had accumulated rewards above the "
            "minimum claim threshold.[/dim]"
        )


@node.command("preemption-status")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
def node_preemption_status_cli(
    output_format: str,
    api_url_override: Optional[str],
):
    """Sprint 776 — show cloud-spot preemption detector status.

    Reports whether the daemon's PreemptionDetector is wired,
    what backend (aws/gcp), and whether a preemption signal has
    been received.

    Returns 503 when the detector is not configured (env unset
    or construction failed at startup).

    Exit codes:
      0 — detector wired; status returned
      1 — daemon answered but detector not configured (503)
      2 — daemon unreachable
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/admin/preemption/status"
    try:
        resp = _httpx.get(endpoint, timeout=10.0)
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[yellow]PreemptionDetector not configured "
                f"({resp.status_code}).[/yellow] "
                f"Set [bold]PRSM_PREEMPTION_DETECTOR=aws|gcp[/bold] "
                f"in your systemd unit and restart to enable. "
                f"Detail: {resp.text}"
            )
        raise SystemExit(1)
    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return
    preempted = data.get("preempted", False)
    backend = data.get("backend", "?")
    interval = data.get("poll_interval_seconds", "?")
    flag_render = (
        "[red]SIGNALED — node is draining[/red]"
        if preempted
        else "[green]clear (no preemption)[/green]"
    )
    console.print(
        f"[bold]Preemption detector:[/bold] backend=[cyan]{backend}"
        f"[/cyan], poll_interval={interval}s"
    )
    console.print(f"  Status: {flag_render}")


@node.command("stake-info")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--identity-file", "identity_file",
    default=None,
    help="Path to identity.json (default: ~/.prsm/identity.json). "
    "Required only when validating an operator_delegation against "
    "this daemon's node_id.",
)
def node_stake_info_cli(
    output_format: str, identity_file: Optional[str],
) -> None:
    """Sprint 795 — show operator's on-chain stake + delegation snapshot.

    Reports operator address, delegation validity, on-chain
    stake amount (wei + FTNS), and the contract being read.
    Always exits 0 — informational. Operators see config gaps
    in the output rather than as command failures.
    """
    import json as _json
    import os as _os
    from decimal import Decimal
    op_addr = (_os.environ.get("PRSM_OPERATOR_ADDRESS") or "").strip()

    payload: Dict[str, Any] = {
        "operator_address": op_addr or None,
        "delegation_status": "absent",
        "delegation_node_id": None,
        "stake_bond_address": (
            _os.environ.get("PRSM_STAKE_BOND_ADDRESS") or None
        ),
        "rpc_url": (
            _os.environ.get("PRSM_BASE_RPC_URL")
            or "https://mainnet.base.org"
        ),
        "on_chain_stake_wei": 0,
        "on_chain_stake_ftns": "0",
        "reader_error": None,
    }

    # No operator address → just emit + bail with a hint.
    if not op_addr:
        if output_format == "json":
            click.echo(_json.dumps(payload, indent=2))
            return
        console.print(
            "[yellow]PRSM_OPERATOR_ADDRESS is unset.[/yellow] "
            "Set it in your systemd unit or shell env to enable "
            "stake-tier eligibility:\n"
            "  [bold]export PRSM_OPERATOR_ADDRESS=0x...[/bold]"
        )
        return

    # Delegation status
    deleg_raw = _os.environ.get("PRSM_OPERATOR_DELEGATION") or ""
    deleg_raw = deleg_raw.strip()
    if deleg_raw:
        try:
            from prsm.node.identity import load_node_identity
            from pathlib import Path as _Path

            blob = _json.loads(deleg_raw)
            ident_path = _Path(
                identity_file
                or (_Path.home() / ".prsm" / "identity.json")
            )
            identity = load_node_identity(ident_path)
            if identity is None:
                payload["delegation_status"] = "identity-missing"
            else:
                from prsm.node.operator_delegation import (
                    verify_operator_delegation_blob,
                )
                ok = verify_operator_delegation_blob(
                    node_id=identity.node_id,
                    operator_address=op_addr,
                    delegation=blob,
                )
                payload["delegation_status"] = (
                    "valid" if ok else "invalid"
                )
                payload["delegation_node_id"] = identity.node_id
        except Exception as exc:
            payload["delegation_status"] = f"parse-error: {exc}"

    # Stake amount via OnChainStakeReader
    try:
        from prsm.node.onchain_stake_reader import OnChainStakeReader
        reader = OnChainStakeReader()
        wei = int(reader.stake_amount_for(op_addr) or 0)
        payload["on_chain_stake_wei"] = wei
        payload["on_chain_stake_ftns"] = str(
            Decimal(wei) / Decimal(10**18),
        )
    except Exception as exc:
        payload["reader_error"] = str(exc)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    # Text mode
    console.print(
        f"[bold]Stake snapshot for [cyan]{op_addr}[/cyan]:[/bold]"
    )
    deleg = payload["delegation_status"]
    deleg_color = (
        "green" if deleg == "valid"
        else "red" if deleg == "invalid"
        else "yellow"
    )
    console.print(
        f"  Delegation:  [{deleg_color}]{deleg}[/{deleg_color}]"
    )
    if payload["delegation_node_id"]:
        console.print(
            f"  Node ID:     [dim]{payload['delegation_node_id']}[/dim]"
        )
    console.print(
        f"  Stake:       [green]{payload['on_chain_stake_ftns']}[/green]"
        f" FTNS ([dim]{payload['on_chain_stake_wei']} wei[/dim])"
    )
    console.print(
        f"  Contract:    [dim]"
        f"{payload['stake_bond_address'] or '<PRSM_STAKE_BOND_ADDRESS unset>'}"
        f"[/dim]"
    )
    console.print(f"  RPC:         [dim]{payload['rpc_url']}[/dim]")
    if payload["reader_error"]:
        console.print(
            f"[yellow]Unable to read on-chain stake:[/yellow] "
            f"{payload['reader_error']}"
        )


@node.command("smoke-test")
@click.option(
    "--no-pool", "skip_pool", is_flag=True, default=False,
    help="Skip the /admin/parallax/pool/snapshot probe. Useful on "
    "single-node dev where the DHT pool is empty by design.",
)
@click.option(
    "--prompt", "prompt_override", default=None,
    help="Inference prompt override. Default: 'The capital of France is'",
)
@click.option(
    "--model", "model_id", default=None,
    help="model_id for the inference call. When unset (default), "
    "smoke-test auto-detects from GET /compute/models — picks "
    "gpt2 if present, else the first listed model. Pass an "
    "explicit value to skip auto-detection.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
def node_smoke_test_cli(
    skip_pool: bool, prompt_override: Optional[str],
    model_id: Optional[str], output_format: str,
    api_url_override: Optional[str],
):
    """Sprint 771 — automated smoke test from runbook §8.

    Probes two surfaces:
      1. GET /admin/parallax/pool/snapshot — DHT-backed pool has
         peers (skipped with --no-pool for single-node dev).
      2. POST /compute/inference — full signed-receipt roundtrip.

    Exit codes:
      0 — both checks passed
      1 — daemon answered but a check failed (no peers, inference
          error, missing settler_signature, etc.)
      2 — daemon unreachable
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    result: Dict[str, Any] = {
        "ok": False,
        "pool": {"ok": None, "skipped": skip_pool},
        "inference": {"ok": None, "signed": None},
    }

    pool_gpu_count = None
    if not skip_pool:
        pool_endpoint = f"{url}/admin/parallax/pool/snapshot"
        try:
            pr = _httpx.get(pool_endpoint, timeout=10.0)
        except Exception as exc:
            if output_format == "json":
                result["pool"]["ok"] = False
                result["pool"]["error"] = (
                    f"daemon unreachable: {exc}"
                )
                click.echo(_json.dumps(result, indent=2))
            else:
                console.print(
                    f"[red]Daemon unreachable at {pool_endpoint}"
                    f"[/red] — {exc}"
                )
            raise SystemExit(2)
        if pr.status_code != 200:
            result["pool"]["ok"] = False
            result["pool"]["status"] = pr.status_code
            result["pool"]["detail"] = pr.text
            if output_format == "json":
                click.echo(_json.dumps(result, indent=2))
            else:
                console.print(
                    f"[red]FAIL[/red] pool snapshot "
                    f"({pr.status_code}): {pr.text}"
                )
            raise SystemExit(1)
        pdata = pr.json()
        pool_gpu_count = pdata.get("gpu_count", 0)
        result["pool"]["ok"] = pool_gpu_count > 0
        result["pool"]["gpu_count"] = pool_gpu_count
        result["pool"]["pool_kind"] = pdata.get("pool_kind")
        if pool_gpu_count == 0:
            if output_format == "json":
                click.echo(_json.dumps(result, indent=2))
            else:
                console.print(
                    "[red]FAIL[/red] pool snapshot: gpu_count=0 "
                    "(no peers in DHT pool)"
                )
            raise SystemExit(1)
        if output_format == "text":
            console.print(
                f"[green]PASS[/green] pool snapshot: "
                f"gpu_count={pool_gpu_count}, "
                f"pool_kind={pdata.get('pool_kind')}"
            )

    # Sprint 828 — auto-detect model when --model not passed.
    # Sprint 824 made the gpt2-vs-mock-catalog error message
    # actionable, but the friction remained: every fresh
    # mock-executor operator hit the same wall + had to re-run
    # with --model. Now we GET /compute/models first when
    # --model is unset, prefer gpt2 if present (preserves prod
    # default), else pick the first listed model + log the
    # substitution. Explicit --model bypasses this entirely.
    resolved_model = model_id
    auto_detected = False
    if resolved_model is None:
        try:
            mr = _httpx.get(f"{url}/compute/models", timeout=10.0)
            if mr.status_code == 200:
                mdata = mr.json() or {}
                models = mdata.get("models") or []
                # Wire format tolerated: list of bare strings
                # (production daemon — confirmed live) OR list of
                # dicts {model_id: ...} (older test fixtures).
                # Live-verified 2026-05-24 mock daemon returns
                # bare strings.
                model_ids = []
                for m in models:
                    if isinstance(m, str):
                        model_ids.append(m)
                    elif isinstance(m, dict) and m.get("model_id"):
                        model_ids.append(m["model_id"])
                if "gpt2" in model_ids:
                    resolved_model = "gpt2"
                elif model_ids:
                    resolved_model = model_ids[0]
                    auto_detected = True
        except Exception:
            pass
        if resolved_model is None:
            resolved_model = "gpt2"
    if auto_detected and output_format == "text":
        console.print(
            f"[dim]Auto-detected --model="
            f"[bold]{resolved_model}[/bold] from daemon catalog "
            f"(gpt2 not present).[/dim]"
        )

    inf_endpoint = f"{url}/compute/inference"
    body = {
        "prompt": prompt_override or "The capital of France is",
        "model_id": resolved_model,
        "budget_ftns": 1.0,
        "privacy_tier": "none",
        "content_tier": "A",
        "max_tokens": 1,
    }
    try:
        ir = _httpx.post(inf_endpoint, json=body, timeout=120.0)
    except Exception as exc:
        result["inference"]["ok"] = False
        result["inference"]["error"] = (
            f"daemon unreachable: {exc}"
        )
        if output_format == "json":
            click.echo(_json.dumps(result, indent=2))
        else:
            console.print(
                f"[red]Daemon unreachable at {inf_endpoint}[/red]"
                f" — {exc}"
            )
        raise SystemExit(2)
    if ir.status_code != 200:
        result["inference"]["ok"] = False
        result["inference"]["status"] = ir.status_code
        result["inference"]["detail"] = ir.text
        if output_format == "json":
            click.echo(_json.dumps(result, indent=2))
        else:
            console.print(
                f"[red]FAIL[/red] inference ({ir.status_code}): "
                f"{ir.text}"
            )
        raise SystemExit(1)
    idata = ir.json()
    receipt = idata.get("receipt") or {}
    sig = receipt.get("settler_signature")
    result["inference"]["ok"] = bool(idata.get("success"))
    result["inference"]["signed"] = bool(sig)
    result["inference"]["output"] = idata.get("output")
    # Sprint 824 — server returns 200 with success=false + an
    # "error" field for application-level failures (e.g. unknown
    # model_id). Surface those distinctly from the "missing
    # signature" path so operators see the actionable message
    # instead of a §7-invariant-violation false alarm.
    if not idata.get("success"):
        err = idata.get("error") or "inference failed"
        result["inference"]["error"] = err
        if output_format == "json":
            click.echo(_json.dumps(result, indent=2))
        else:
            console.print(
                f"[red]FAIL[/red] inference: {err}"
            )
            # When the error is unknown-model, suggest the
            # --model flag so operators can re-run against the
            # actual catalog.
            if "Unknown model_id" in err:
                console.print(
                    "[dim]Hint: pass [bold]--model <id>[/bold] "
                    "matching your daemon's catalog "
                    "(run [bold]prsm compute models[/bold] to "
                    "list).[/dim]"
                )
        raise SystemExit(1)
    if not sig:
        if output_format == "json":
            click.echo(_json.dumps(result, indent=2))
        else:
            console.print(
                "[red]FAIL[/red] inference: receipt missing "
                "settler_signature (§7 verifiable-inference "
                "invariant violated)"
            )
        raise SystemExit(1)

    result["ok"] = True
    if output_format == "json":
        click.echo(_json.dumps(result, indent=2))
        return
    console.print(
        f"[green]PASS[/green] inference: signed receipt "
        f"(settler_signature present, output={idata.get('output')!r})"
    )


@node.command("auto-claim")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--runtime", "fetch_runtime", is_flag=True, default=False,
    help="Also fetch runtime counters from the running daemon "
    "via /admin/auto-claim/status (sprint 769). Requires daemon "
    "reachable on the loopback URL (or PRSM_ADMIN_REMOTE_ALLOWED=1 "
    "behind a proxy with auth).",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL (default: stored credential or 127.0.0.1:8000)",
)
def node_auto_claim_cli(
    output_format: str, fetch_runtime: bool,
    api_url_override: Optional[str],
):
    """Sprint 765-769 — show the auto-claim worker config + runtime.

    Reads `PRSM_AUTO_CLAIM_THRESHOLD_FTNS` + `PRSM_AUTO_CLAIM_INTERVAL_S`
    and reports whether auto-claim is enabled.

    When enabled, the daemon periodically checks accumulated FTNS
    rewards and calls claim_rewards once they cross the threshold.
    Operators set + restart the daemon — env-driven, not runtime-
    mutable. To change cadence/threshold, edit your systemd unit's
    Environment= lines and restart.

    Use `--runtime` to additionally fetch the running daemon's
    worker counters (total claimed this session, attempts,
    failures) via /admin/auto-claim/status. Requires the daemon
    to be reachable on the loopback URL.
    """
    import json as _json
    from prsm.node.auto_claim import resolve_auto_claim_config_from_env

    try:
        cfg = resolve_auto_claim_config_from_env()
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "enabled": False,
                "error": str(exc),
            }))
            raise SystemExit(1)
        console.print(f"[red]Auto-claim config error:[/red] {exc}")
        raise SystemExit(1)

    # Optionally fetch runtime counters from the live daemon.
    runtime_payload: Optional[Dict[str, Any]] = None
    if fetch_runtime:
        import httpx as _httpx
        url = _api_url_from_creds(api_url_override)
        endpoint = f"{url}/admin/auto-claim/status"
        try:
            resp = _httpx.get(endpoint, timeout=5.0)
            if resp.status_code == 200:
                runtime_payload = resp.json()
            elif resp.status_code == 503:
                runtime_payload = {
                    "available": False,
                    "reason": resp.json().get("detail", "no worker"),
                }
            else:
                runtime_payload = {
                    "available": False,
                    "reason": (
                        f"daemon responded {resp.status_code}: "
                        f"{resp.text[:200]}"
                    ),
                }
        except Exception as exc:
            runtime_payload = {
                "available": False,
                "reason": f"daemon unreachable: {exc}",
            }

    if output_format == "json":
        payload = {
            "enabled": cfg.enabled,
            "threshold_ftns": str(cfg.threshold_ftns),
            "interval_seconds": cfg.interval_seconds,
        }
        if runtime_payload is not None:
            payload["runtime"] = runtime_payload
        click.echo(_json.dumps(payload, indent=2))
        return

    if not cfg.enabled:
        console.print(
            "[bold]Auto-claim:[/bold] [dim]disabled[/dim]"
        )
        console.print(
            "[dim]PRSM_AUTO_CLAIM_THRESHOLD_FTNS not set (or 0). "
            "Daemon does not auto-claim. To opt in, set "
            "PRSM_AUTO_CLAIM_THRESHOLD_FTNS=100 (or similar) in "
            "your systemd unit + restart.[/dim]"
        )
        return

    console.print(
        f"[bold]Auto-claim:[/bold] [green]enabled[/green]"
    )
    console.print(
        f"[bold]Threshold:[/bold] "
        f"[cyan]{cfg.threshold_ftns}[/cyan] FTNS"
    )
    console.print(
        f"[bold]Interval:[/bold] "
        f"[cyan]{cfg.interval_seconds}[/cyan] seconds"
    )
    console.print(
        "[dim]Daemon checks accumulated rewards every "
        f"{cfg.interval_seconds:.0f}s; claims when total reaches "
        f"{cfg.threshold_ftns} FTNS.[/dim]"
    )

    if runtime_payload is not None:
        if runtime_payload.get("available") is False:
            console.print(
                f"\n[yellow]Runtime counters unavailable:[/yellow] "
                f"{runtime_payload.get('reason', 'unknown')}"
            )
        else:
            console.print(
                f"\n[bold]Runtime (this session):[/bold]"
            )
            console.print(
                f"  Cumulative claimed: "
                f"[cyan]{runtime_payload.get('total_claimed_ftns', '0')}[/cyan] FTNS"
            )
            console.print(
                f"  Claim attempts: "
                f"[cyan]{runtime_payload.get('claim_attempts', 0)}[/cyan]"
            )
            failures = runtime_payload.get('claim_failures', 0)
            color = "red" if failures > 0 else "green"
            console.print(
                f"  Claim failures: [{color}]{failures}[/{color}]"
            )


@node.command("device-profile")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--suggest", "suggest_preset", is_flag=True, default=False,
    help="Print a 'polite-neighbor' systemd Environment= block for "
    "consumer-device operators (MacBook, gaming PC).",
)
def node_device_profile_cli(output_format: str, suggest_preset: bool):
    """Sprint 764 — show the 4-knob consumer-device UX stack.

    Surfaces all consumer-device controls at a glance:

      1. PRSM_ACTIVE_HOURS (sprint 755-758) — time-of-day window
      2. PRSM_STORAGE_UPLOAD_MBPS + DOWNLOAD_MBPS (sprint 761) —
         bandwidth caps
      3. PRSM_NODE_NICE (sprint 762) — CPU priority
      4. PRSM_ACTIVE_ONLY_ON_AC (sprint 763) — battery awareness

    Each knob composes with the others. Operators on personal
    devices (MacBook, gaming PC) typically want all 4 set;
    operators on dedicated servers leave them unset.

    `--suggest` prints a sane "polite-neighbor" preset they can
    paste into their systemd unit.
    """
    import json as _json
    import os as _os
    from prsm.node.schedule import resolve_active_window_from_env

    if suggest_preset:
        preset = """# Sprint 764 — polite-neighbor preset for consumer-device
# operators (MacBook, gaming PC). Copy-paste into your
# systemd unit's [Service] section. Adjust the timezone +
# hours to match your work schedule.

Environment=PRSM_ACTIVE_HOURS=22:00-08:00
Environment=PRSM_ACTIVE_TIMEZONE=America/New_York
Environment=PRSM_ACTIVE_ONLY_ON_AC=1
Environment=PRSM_NODE_NICE=10
Environment=PRSM_STORAGE_UPLOAD_MBPS=10
Environment=PRSM_STORAGE_DOWNLOAD_MBPS=100
"""
        click.echo(preset)
        return

    # Read current state of all 4 knobs.
    try:
        window = resolve_active_window_from_env()
        window_repr = window.render() if window else None
        window_error = None
    except ValueError as exc:
        window = None
        window_repr = None
        window_error = str(exc)

    upload_mbps = _os.environ.get("PRSM_STORAGE_UPLOAD_MBPS", "").strip()
    download_mbps = _os.environ.get(
        "PRSM_STORAGE_DOWNLOAD_MBPS", "",
    ).strip()
    node_nice = _os.environ.get("PRSM_NODE_NICE", "").strip()
    only_ac = _os.environ.get(
        "PRSM_ACTIVE_ONLY_ON_AC", "",
    ).strip().lower() in ("1", "true", "yes")

    if output_format == "json":
        click.echo(_json.dumps({
            "active_hours": window_repr,
            "active_hours_error": window_error,
            "active_only_on_ac": only_ac,
            "node_nice": node_nice or "0 (default)",
            "storage_upload_mbps": upload_mbps or "0 (unlimited)",
            "storage_download_mbps": download_mbps or "0 (unlimited)",
        }, indent=2))
        return

    # Rich text output.
    console.print("[bold]PRSM consumer-device profile:[/bold]\n")

    # Scheduling
    if window_error:
        console.print(
            f"[red]✗[/red] Active hours: [red]config error: {window_error}[/red]"
        )
    elif window_repr:
        console.print(
            f"[green]✓[/green] Active hours: [cyan]{window_repr}[/cyan]"
        )
    else:
        console.print(
            "[dim]–[/dim] Active hours: [dim]unset (always-active)[/dim]"
        )

    # Battery
    if only_ac:
        console.print(
            "[green]✓[/green] Battery awareness: "
            "[cyan]only-on-AC[/cyan]"
        )
    else:
        console.print(
            "[dim]–[/dim] Battery awareness: "
            "[dim]unset (active regardless of power)[/dim]"
        )

    # CPU priority
    if node_nice and node_nice not in ("0", "+0"):
        console.print(
            f"[green]✓[/green] CPU politeness: "
            f"[cyan]nice +{node_nice}[/cyan]"
        )
    else:
        console.print(
            "[dim]–[/dim] CPU politeness: "
            "[dim]unset (default priority)[/dim]"
        )

    # Bandwidth
    if upload_mbps or download_mbps:
        up = upload_mbps or "unlimited"
        down = download_mbps or "unlimited"
        console.print(
            f"[green]✓[/green] Bandwidth caps: "
            f"[cyan]up={up} Mbps, down={down} Mbps[/cyan]"
        )
    else:
        console.print(
            "[dim]–[/dim] Bandwidth caps: [dim]unset (unlimited)[/dim]"
        )

    console.print(
        "\n[dim]Run `prsm node device-profile --suggest` for a "
        "polite-neighbor preset.[/dim]"
    )


@node.command("streams")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL (default: stored credential or 127.0.0.1:8000)",
)
def node_streams_cli(output_format: str, api_url_override: Optional[str]):
    """Sprint 722 — read in-flight remote token-streams.

    After sprints 711 (wire protocol), 713 (bounded receive queue),
    719 (sender binding), 720 (disconnect cleanup), and 721 (request
    size limit), operators have NO direct visibility into what those
    machines are doing in production. This command queries
    `/admin/parallax/streams` on the running daemon and renders:

      - active stream count
      - per-stream queue depth + maxsize (sprint 713 back-pressure)
      - whether each stream's queue is full (back-pressure engaged)
      - expected_sender prefix (sprint 719 hijack defense)
      - operator-tunable env values currently in effect

    Read-only — no daemon-state mutation. Safe to run on the live
    fleet. Empty list when daemon is idle (typical between
    inferences).
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/admin/parallax/streams"
    try:
        resp = _httpx.get(endpoint, timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"unreachable: {type(exc).__name__}: {exc}",
                "url": endpoint,
            }))
        else:
            console.print(
                f"[red]Cannot reach daemon at {endpoint}[/red] — "
                f"{type(exc).__name__}: {exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]Daemon responded {resp.status_code}[/red]: "
                f"{resp.text}"
            )
        raise SystemExit(1)
    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return

    streams = data.get("streams", [])
    count = data.get("count", 0)
    qmax = data.get("queue_maxsize", "?")
    reqmax = data.get("request_max_bytes", "?")
    console.print(
        f"\n[bold]In-flight remote token-streams:[/bold] {count}"
    )
    console.print(
        f"PRSM_CHAIN_STREAM_QUEUE_MAXSIZE = [yellow]{qmax}[/yellow]"
    )
    console.print(
        f"PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES = [yellow]{reqmax}[/yellow]"
    )
    if not streams:
        console.print(
            "[dim]No active streams (daemon idle — typical between"
            " inferences).[/dim]\n"
        )
        return
    table = Table(title="Active token-streams (sprint 722)")
    table.add_column("stream_id", style="cyan", no_wrap=True)
    table.add_column("peer", style="magenta", no_wrap=True)
    table.add_column("queue", justify="right")
    table.add_column("max", justify="right")
    table.add_column("full?", justify="center")
    for s in streams:
        full_marker = (
            "[red]YES[/red]" if s.get("queue_full") else "[green]no[/green]"
        )
        table.add_row(
            s.get("stream_id_prefix", "") + "...",
            (s.get("expected_sender_prefix", "") or "?") + "...",
            str(s.get("queue_depth", "?")),
            str(s.get("queue_maxsize", "?")),
            full_marker,
        )
    console.print(table)


# ── Sprint 584 — RPC probe ───────────────────────────────────────


@node.command("rpc-probe")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def rpc_probe_cli(output_format: str):
    """Probe whether PRSM_BASE_RPC_URL is reachable + responds.

    Sprint 584 — completes the §7 production preflight trifecta
    with sprints 581 (anchor) + 583 (stake-bond). Tests RPC
    reachability via eth_chainId JSON-RPC call so operators can
    distinguish "wrong contract address" from "unreachable RPC"
    when 581/583 fail with construction_failed.
    """
    import json
    import os as _os
    import httpx as _httpx

    rpc_url = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    outcome = "ok"
    error = None
    chain_id_hex = None
    try:
        resp = _httpx.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_chainId",
                "params": [],
                "id": 1,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            outcome = "error"
            error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        else:
            try:
                body = resp.json()
            except ValueError as exc:
                outcome = "error"
                error = f"non-JSON response: {exc}"
            else:
                chain_id_hex = body.get("result")
                if chain_id_hex is None:
                    outcome = "error"
                    error = f"no 'result' field: {body!r}"[:200]
    except _httpx.HTTPError as exc:
        outcome = "unreachable"
        error = f"{type(exc).__name__}: {exc}"

    payload = {
        "PRSM_BASE_RPC_URL": rpc_url,
        "outcome": outcome,
        "chain_id_hex": chain_id_hex,
        "error": error,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2))
        raise SystemExit(0 if outcome == "ok" else 1)

    if outcome == "ok":
        console.print(
            f"[green]✓ ok[/green] — RPC at [cyan]{rpc_url}[/cyan] "
            f"responded with chain_id=[magenta]{chain_id_hex}[/magenta]"
        )
        return
    console.print(
        f"[red]✗ {outcome}[/red]: {error}\n"
        f"[dim]rpc_url={rpc_url!r}[/dim]\n"
        f"[dim]Check the URL is reachable (try `curl -s {rpc_url}`) "
        f"and that any auth tokens (Infura/Alchemy key) are valid.[/dim]"
    )
    raise SystemExit(1)


# ── Sprint 581 — anchor probe ────────────────────────────────────


@node.command("anchor-probe")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def anchor_probe_cli(output_format: str):
    """Probe whether PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS wires a
    working PublisherKeyAnchorClient.

    Operator preflight before flipping PRSM_PARALLAX_TRUST_STACK_KIND
    =production. Surfaces:
      - The env value (or <unset>)
      - Configured RPC URL
      - Construction outcome: ok / unset / construction_failed
      - Error detail when construction fails

    Sprint 581 — closes the slow "set env, restart daemon, grep
    logs" feedback loop with a one-shot operator-side check.
    """
    import json
    import os as _os
    anchor_addr = (
        _os.environ.get("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "") or ""
    ).strip()
    rpc_url = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    outcome = "ok"
    error = None
    if not anchor_addr:
        outcome = "unset"
    else:
        try:
            from prsm.security.publisher_key_anchor.client import (
                PublisherKeyAnchorClient,
            )
            PublisherKeyAnchorClient(
                contract_address=anchor_addr,
                rpc_url=rpc_url,
            )
        except Exception as exc:  # noqa: BLE001
            outcome = "construction_failed"
            error = f"{type(exc).__name__}: {exc}"

    payload = {
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": anchor_addr or None,
        "PRSM_BASE_RPC_URL": rpc_url,
        "outcome": outcome,
        "error": error,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2))
        raise SystemExit(0 if outcome == "ok" else 1)

    if outcome == "ok":
        console.print(
            f"[green]✓ ok[/green] — PublisherKeyAnchorClient constructed "
            f"against [cyan]{anchor_addr}[/cyan] on "
            f"[magenta]{rpc_url}[/magenta]"
        )
        return
    if outcome == "unset":
        console.print(
            "[yellow]PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS is unset[/yellow] "
            "— required to flip PRSM_PARALLAX_TRUST_STACK_KIND=production. "
            "Set it to the deployed Phase-3.x.3 anchor contract address."
        )
        raise SystemExit(1)
    # construction_failed
    console.print(
        f"[red]✗ construction_failed[/red]: {error}\n"
        f"[dim]anchor_addr={anchor_addr!r}, rpc_url={rpc_url!r}[/dim]\n"
        f"[dim]Check the address resolves to a deployed contract on "
        f"the RPC chain, and that the RPC endpoint is reachable.[/dim]"
    )
    raise SystemExit(1)


# ── Sprint 583 — stake-bond probe ────────────────────────────────


@node.command("stake-bond-probe")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def stake_bond_probe_cli(output_format: str):
    """Probe whether PRSM_STAKE_BOND_ADDRESS wires a working
    StakeManagerClient.

    Sprint 583 — mirror of sprint-581 anchor-probe for the
    sibling §7 production env var (sprint 561). Operator
    preflight before flipping PRSM_PARALLAX_TRUST_STACK_KIND
    =production with stake-weighted trust.
    """
    import json
    import os as _os
    stake_addr = (
        _os.environ.get("PRSM_STAKE_BOND_ADDRESS", "") or ""
    ).strip()
    rpc_url = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    outcome = "ok"
    error = None
    if not stake_addr:
        outcome = "unset"
    else:
        try:
            from prsm.economy.web3.stake_manager import (
                StakeManagerClient,
            )
            StakeManagerClient(
                contract_address=stake_addr,
                rpc_url=rpc_url,
            )
        except Exception as exc:  # noqa: BLE001
            outcome = "construction_failed"
            error = f"{type(exc).__name__}: {exc}"

    payload = {
        "PRSM_STAKE_BOND_ADDRESS": stake_addr or None,
        "PRSM_BASE_RPC_URL": rpc_url,
        "outcome": outcome,
        "error": error,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2))
        raise SystemExit(0 if outcome == "ok" else 1)

    if outcome == "ok":
        console.print(
            f"[green]✓ ok[/green] — StakeManagerClient constructed "
            f"against [cyan]{stake_addr}[/cyan] on "
            f"[magenta]{rpc_url}[/magenta]"
        )
        return
    if outcome == "unset":
        console.print(
            "[yellow]PRSM_STAKE_BOND_ADDRESS is unset[/yellow] "
            "— required for sprint-561 real stake-weighted trust. "
            "Without it, production trust-stack falls back to "
            "ZeroStakeLookup placeholder (every node treated as "
            "zero stake)."
        )
        raise SystemExit(1)
    console.print(
        f"[red]✗ construction_failed[/red]: {error}\n"
        f"[dim]stake_addr={stake_addr!r}, rpc_url={rpc_url!r}[/dim]\n"
        f"[dim]Check the address resolves to a deployed StakeBond "
        f"contract and the RPC endpoint is reachable.[/dim]"
    )
    raise SystemExit(1)


# ── Sprint 579 — trust-stack observability ──────────────────────


_TRUST_STACK_ENV_VARS = [
    ("PRSM_PARALLAX_TRUST_STACK_KIND",
     ("mock", "production"), "mock",
     "Top-level trust-stack kind. mock=permissive (dev); "
     "production=4-component verification (sprints 558-562)."),
    ("PRSM_PARALLAX_PROFILE_SOURCE_KIND",
     ("in_memory", "dht"), "in_memory",
     "Inner ProfileSource (sprint 576). dht hooks Phase 2 "
     "ProfileDHT integration; Phase 1 falls back to in_memory."),
    ("PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND",
     ("logging", "onchain"), "logging",
     "Consensus mismatch submitter (sprint 577). onchain hooks "
     "Phase 2 ChallengeRecord → Phase 7.1x ABI dispatch; Phase 1 "
     "falls back to logging."),
    ("PRSM_PARALLAX_CHAIN_EXECUTOR_KIND",
     ("stub", "rpc"), "stub",
     "Inference chain executor (sprint 578). rpc hooks Phase 2 "
     "make_rpc_chain_executor wiring; Phase 1 falls back to stub."),
]


def _resolve_trust_stack_entry(name, valid, default):
    """Return (kind, env_value, status) for one trust-stack env var."""
    import os as _os
    raw = (_os.environ.get(name, "") or "").strip()
    env_value = raw if raw else None
    val = (raw or default).lower()
    if val in valid:
        # Phase 1: only the FIRST valid kind is fully active; the
        # second one is the Phase 2 hook that currently falls back.
        if val == default:
            status = "active (default)" if env_value is None else "active"
        else:
            # Phase 2 hook kinds — surface as "phase 2 pending"
            # so operators don't assume they got the real thing.
            status = "phase 2 pending (falls back to default)"
        return val, env_value, status
    return default, env_value, f"unknown_falls_back_to_{default}"


@node.command("trust-stack")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format: 'text' (Rich table) or 'json' (agent-parseable)",
)
def trust_stack_cli(output_format: str):
    """Inspect §7 trust-stack Phase-1 env config.

    Reports the 4 ParallaxScheduledExecutor component env vars
    + their effective kinds. Useful for verifying which Phase 2
    hooks an operator has opted into.

    Sprint 579 closes the observability gap left by sprints
    558-562/576/577/578 — operators no longer need to spelunk
    daemon startup logs to see effective trust-stack config.
    """
    import json
    entries = {}
    for (name, valid, default, descr) in _TRUST_STACK_ENV_VARS:
        kind, env_value, status = _resolve_trust_stack_entry(
            name, valid, default,
        )
        entries[name] = {
            "kind": kind,
            "env_value": env_value,
            "status": status,
            "valid": list(valid),
            "default": default,
            "description": descr,
        }

    if output_format == "json":
        click.echo(json.dumps(entries, indent=2))
        return

    # no_wrap on all columns + console.width override prevents Rich
    # auto-truncation in narrow terminals or test harness output.
    from rich.console import Console as _RichConsole
    _wide = _RichConsole(width=140)
    table = Table(title="Trust-stack Phase-1 env config (sprint 579)")
    table.add_column("Env var", style="cyan", no_wrap=True)
    table.add_column("Effective kind", style="green", no_wrap=True)
    table.add_column("Env value", style="magenta", no_wrap=True)
    table.add_column("Status", style="blue", no_wrap=True)
    for name, e in entries.items():
        table.add_row(
            name,
            e["kind"],
            (e["env_value"] or "<unset>"),
            e["status"],
        )
    _wide.print(table)
    _wide.print(
        "\n[dim]Sprints 558-562/576/577/578 plumb four Phase-1 "
        "hooks for §7 trust-stack. Phase 2 lands real impl per "
        "component additively.[/dim]"
    )


# ── Sprint 574 — fleet-ops CLI quartet ───────────────────────────


def _api_base() -> str:
    """Resolve the local daemon's API base URL.

    Sprint 574 — honor PRSM_API_PORT env var FIRST so operators
    running daemons on non-default ports (e.g., bootstrap-us
    droplet at 8002 alongside bootstrap-server-v2) get the right
    target without editing NodeConfig. Falls back to NodeConfig
    loaded value (default 8000) when env unset.
    """
    import os
    env_port = (os.environ.get("PRSM_API_PORT", "") or "").strip()
    if env_port:
        try:
            port = int(env_port)
        except ValueError:
            port = 0
        if port > 0:
            return f"http://127.0.0.1:{port}"
    from prsm.node.config import NodeConfig
    cfg = NodeConfig.load()
    return f"http://127.0.0.1:{cfg.api_port}"


def _daemon_down_message():
    console.print(
        "Could not reach the daemon. Is it running? "
        "Start it with [cyan]prsm node start[/cyan].",
        style="yellow",
    )


@node.command()
@click.argument("address")
def dial(address: str):
    """Dial a peer by address (host:port or ws://host:port).

    Sprint 574 — wraps POST /peers/connect. Useful when auto-dial
    sweep (sprint 573) failed or when joining a peer not in the
    bootstrap registry.
    """
    import httpx
    try:
        resp = httpx.post(
            f"{_api_base()}/peers/connect",
            json={"address": address},
            timeout=30.0,
        )
    except httpx.HTTPError:
        _daemon_down_message()
        raise SystemExit(1)

    if resp.status_code == 200:
        data = resp.json()
        console.print(
            f"[green]✓ Connected[/green] to "
            f"[cyan]{data.get('peer_id', '?')[:16]}...[/cyan] "
            f"at [magenta]{data.get('address', address)}[/magenta]"
        )
        return

    if resp.status_code == 502:
        console.print(
            f"[yellow]Could not connect to {address}[/yellow] "
            f"(502: transport returned None — peer unreachable, "
            f"firewall, or handshake failure).",
        )
    elif resp.status_code == 503:
        console.print(
            "[yellow]Daemon transport not initialized yet "
            "(503).[/yellow] Try again in a moment."
        )
    else:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        console.print(
            f"[red]dial failed[/red] (HTTP {resp.status_code}): {detail}"
        )
    raise SystemExit(1)


@node.command()
@click.argument("cid")
@click.option(
    "--output", "-o", "output_path", default=None,
    help="Write decoded content to this file instead of stdout.",
)
def fetch(cid: str, output_path):
    """Fetch content by CID from the network.

    Sprint 574 — wraps GET /content/retrieve. Base64-decodes the
    response and writes to stdout (default) or --output file.
    """
    import base64
    import httpx
    try:
        resp = httpx.get(
            f"{_api_base()}/content/retrieve/{cid}",
            timeout=60.0,
        )
    except httpx.HTTPError:
        _daemon_down_message()
        raise SystemExit(1)

    if resp.status_code != 200:
        console.print(
            f"[red]fetch failed[/red] (HTTP {resp.status_code})"
        )
        raise SystemExit(1)

    data = resp.json()
    status = data.get("status")
    if status != "success":
        err = data.get("error") or status or "unknown"
        console.print(f"[yellow]{status or 'not_found'}:[/yellow] {err}")
        raise SystemExit(1)

    try:
        payload = base64.b64decode(data.get("data", ""))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]base64 decode failed[/red]: {exc}")
        raise SystemExit(1)

    if output_path:
        with open(output_path, "wb") as fh:
            fh.write(payload)
        console.print(
            f"[green]✓ Wrote {len(payload)} bytes[/green] to "
            f"[cyan]{output_path}[/cyan] "
            f"(filename={data.get('filename', '?')})"
        )
    else:
        # Render as text if it decodes cleanly; raw bytes otherwise
        try:
            console.print(payload.decode("utf-8"))
        except UnicodeDecodeError:
            console.print(f"[dim]{len(payload)} bytes (binary)[/dim]")
            click.echo(payload, nl=False)


@node.command()
@click.argument("file_path")
def share(file_path: str):
    """Upload a file's text contents to the network + print the CID.

    Sprint 574 — wraps POST /content/upload. Output is the CID so
    operators can pipe to ``prsm node dial`` / share workflows.
    """
    import httpx
    try:
        if file_path == "-":
            import sys
            text = sys.stdin.read()
        else:
            with open(file_path, "r", encoding="utf-8") as fh:
                text = fh.read()
    except OSError as exc:
        console.print(f"[red]cannot read[/red] {file_path}: {exc}")
        raise SystemExit(1)

    try:
        resp = httpx.post(
            f"{_api_base()}/content/upload",
            json={"text": text},
            timeout=60.0,
        )
    except httpx.HTTPError:
        _daemon_down_message()
        raise SystemExit(1)

    if resp.status_code != 200:
        console.print(
            f"[red]share failed[/red] (HTTP {resp.status_code}): "
            f"{resp.text}"
        )
        raise SystemExit(1)

    data = resp.json()
    cid = data.get("cid", "")
    console.print(
        f"[green]✓ Shared[/green] [cyan]{cid}[/cyan] "
        f"({data.get('size_bytes', 0)} bytes, "
        f"filename={data.get('filename', '?')})"
    )


@node.command()
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format: 'text' (human) or 'json' (agent-parseable)")
def info(output_format: str):
    """Show node identity and configuration.

    Use --format json for machine-readable output (AI agent consumption).
    """
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity

    config = NodeConfig.load()
    identity = load_node_identity(config.identity_path)

    if not identity:
        if output_format == "json":
            _agent_error("No node identity found. Run: prsm setup")
        console.print("No node identity found. Run 'prsm setup' first.", style="yellow")
        return

    # Sprint 534 F58 fix: surface BOTH primary + fallback
    # bootstrap lists so operators see the full failover chain
    # (not just the single dead bootstrap1 that pre-sprint-534
    # configs persisted).
    fallback_nodes = list(
        getattr(config, "bootstrap_fallback_nodes", []) or []
    )
    data = {
        "ok": True,
        "node_id": identity.node_id,
        "display_name": config.display_name,
        "public_key_b64": identity.public_key_b64,
        "roles": [r.value for r in config.roles],
        "p2p_port": config.p2p_port,
        "api_port": config.api_port,
        "data_dir": config.data_dir,
        "bootstrap_nodes": config.bootstrap_nodes or [],
        "bootstrap_fallback_nodes": fallback_nodes,
    }

    if output_format == "json":
        _agent_output(data)
        return

    table = Table(title="PRSM Node Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Node ID", data["node_id"])
    table.add_row("Display Name", data["display_name"])
    table.add_row("Public Key", data["public_key_b64"][:32] + "...")
    table.add_row("Roles", ", ".join(data["roles"]))
    table.add_row("P2P Port", str(data["p2p_port"]))
    table.add_row("API Port", str(data["api_port"]))
    table.add_row("Data Dir", data["data_dir"])
    table.add_row(
        "Bootstrap Primary",
        ", ".join(data["bootstrap_nodes"]) or "none",
    )
    if fallback_nodes:
        table.add_row(
            "Bootstrap Fallback",
            ", ".join(fallback_nodes),
        )
    console.print(table)


@node.command("configure")
@click.option("--show", is_flag=True, help="Show current configuration and exit")
@click.option("--cpu", default=None, type=int, help="CPU allocation % (10-90)")
@click.option("--memory", default=None, type=int, help="RAM allocation % (10-90)")
@click.option("--storage", default=None, type=float, help="Storage to pledge in GB")
@click.option("--jobs", default=None, type=int, help="Max concurrent compute jobs")
@click.option("--gpu-pct", default=None, type=int, help="GPU allocation % (10-100)")
@click.option("--upload-limit", default=None, type=float, help="Upload bandwidth limit in Mbps (0=unlimited)")
@click.option("--active-hours", default=None, type=str, help="Active hours range (e.g., '22-8' for 10pm-8am, 'off' for always on)")
@click.option("--active-days", default=None, type=str, help="Active days (e.g., 'mon,tue,wed' or 'weekdays' or 'weekends')")
def configure(show: bool, cpu: Optional[int], memory: Optional[int], storage: Optional[float],
              jobs: Optional[int], gpu_pct: Optional[int], upload_limit: Optional[float],
              active_hours: Optional[str], active_days: Optional[str]):
    """Configure node resource settings.
    
    Without arguments, runs interactively to prompt for settings.
    With --show, prints current configuration.
    With specific flags, updates only those settings.
    """
    from prsm.node.config import NodeConfig
    
    config = NodeConfig.load()
    
    # Handle --show flag — redirect to new prsm config system
    if show:
        # Try the new PRSMConfig system first, fall back to legacy display
        try:
            from prsm.cli_modules.config_schema import PRSMConfig
            cfg = PRSMConfig.load()
            console.print()
            console.print("  PRSM Configuration", style="bold cyan")
            console.print("  [dim]Note: Also run `prsm config show` for the full display", style="dim")
            console.print(f"  Role:      {cfg.node_role.value}  (new config)")
            console.print(f"  CPU:       {cfg.cpu_pct}%")
            console.print(f"  Memory:    {cfg.memory_pct}%")
            console.print(f"  Storage:   {cfg.storage_gb} GB")
            console.print(f"  P2P Port:  {cfg.p2p_port}")
            console.print(f"  API Port:  {cfg.api_port}")
            console.print(f"  MCP:       {'enabled' if cfg.mcp_server_enabled else 'disabled'}")
            console.print(f"  Config:    {PRSMConfig.config_path()}")
            console.print()
            return
        except Exception:
            pass
        _show_configuration(config)
        return
    
    # Check if any flags were provided
    has_flags = any(v is not None for v in [cpu, memory, storage, jobs, gpu_pct, upload_limit, active_hours, active_days])
    
    if has_flags:
        # Update only specified fields
        changes = []
        
        if cpu is not None:
            if not 10 <= cpu <= 90:
                raise click.BadParameter("CPU allocation must be between 10-90%")
            old_val = config.cpu_allocation_pct
            config.cpu_allocation_pct = cpu
            changes.append(f"CPU allocation: {old_val}% → {cpu}%")
        
        if memory is not None:
            if not 10 <= memory <= 90:
                raise click.BadParameter("Memory allocation must be between 10-90%")
            old_val = config.memory_allocation_pct
            config.memory_allocation_pct = memory
            changes.append(f"Memory allocation: {old_val}% → {memory}%")
        
        if storage is not None:
            if storage <= 0:
                raise click.BadParameter("Storage must be a positive value in GB")
            old_val = config.storage_gb
            config.storage_gb = storage
            changes.append(f"Storage pledged: {old_val} GB → {storage} GB")
        
        if jobs is not None:
            if jobs < 1:
                raise click.BadParameter("Max concurrent jobs must be at least 1")
            old_val = config.max_concurrent_jobs
            config.max_concurrent_jobs = jobs
            changes.append(f"Max concurrent jobs: {old_val} → {jobs}")
        
        if gpu_pct is not None:
            if not 10 <= gpu_pct <= 100:
                raise click.BadParameter("GPU allocation must be between 10-100%")
            old_val = config.gpu_allocation_pct
            config.gpu_allocation_pct = gpu_pct
            changes.append(f"GPU allocation: {old_val}% → {gpu_pct}%")
        
        if upload_limit is not None:
            if upload_limit < 0:
                raise click.BadParameter("Upload limit cannot be negative")
            old_val = config.upload_mbps_limit
            config.upload_mbps_limit = upload_limit
            old_str = f"{old_val} Mbps" if old_val > 0 else "unlimited"
            new_str = f"{upload_limit} Mbps" if upload_limit > 0 else "unlimited"
            changes.append(f"Upload limit: {old_str} → {new_str}")
        
        if active_hours is not None:
            start, end = parse_active_hours(active_hours)
            old_start, old_end = config.active_hours_start, config.active_hours_end
            config.active_hours_start = start
            config.active_hours_end = end
            old_str = f"{old_start:02d}-{old_end:02d}" if old_start is not None and old_end is not None else "always on"
            new_str = f"{start:02d}-{end:02d}" if start is not None and end is not None else "always on"
            changes.append(f"Active hours: {old_str} → {new_str}")
        
        if active_days is not None:
            days = parse_active_days(active_days)
            old_days = config.active_days
            config.active_days = days
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            old_str = ", ".join(day_names[d] for d in old_days) if old_days else "every day"
            new_str = ", ".join(day_names[d] for d in days) if days else "every day"
            changes.append(f"Active days: {old_str} → {new_str}")
        
        config.save()
        
        console.print("✅ Configuration updated:", style="bold green")
        for change in changes:
            console.print(f"   {change}")
        console.print(f"\n   Config saved to {config.config_path}")
        
    else:
        # Interactive mode
        _run_interactive_configure(config)


def _show_configuration(config: "NodeConfig") -> None:
    """Display current node configuration in human-readable format."""
    from prsm.node.compute_provider import detect_resources
    
    # Detect actual system resources
    resources = detect_resources()
    
    # Format role display
    role_display = " + ".join(r.value for r in config.roles)
    
    console.print()
    console.print("PRSM Node Resource Configuration", style="bold magenta")
    console.print("=" * 50)
    console.print(f"  Role:             {role_display}", style="cyan")
    
    # CPU allocation
    cpu_offered = round(resources.cpu_count * config.cpu_allocation_pct / 100, 1)
    console.print(f"  CPU allocation:   {config.cpu_allocation_pct}% of {resources.cpu_count} cores → {cpu_offered:.1f} cores offered")
    
    # Memory allocation
    mem_offered = round(resources.memory_total_gb * config.memory_allocation_pct / 100, 1)
    console.print(f"  RAM allocation:   {config.memory_allocation_pct}% of {resources.memory_total_gb:.1f} GB → {mem_offered:.1f} GB offered")
    
    # Concurrent jobs
    console.print(f"  Concurrent jobs:  {config.max_concurrent_jobs} slots")
    
    # GPU info
    if resources.gpu_available:
        gpu_offered = round(resources.gpu_memory_gb * config.gpu_allocation_pct / 100, 1)
        console.print(f"  GPU:              {resources.gpu_name} ({resources.gpu_memory_gb:.1f} GB) — {config.gpu_allocation_pct}% → {gpu_offered:.1f} GB offered")
    else:
        console.print(f"  GPU:              not detected")
    
    # Storage
    console.print(f"  Storage pledged:  {config.storage_gb:.1f} GB")
    
    # Upload limit
    if config.upload_mbps_limit > 0:
        console.print(f"  Upload limit:     {config.upload_mbps_limit:.1f} Mbps")
    else:
        console.print(f"  Upload limit:     unlimited")
    
    # Active hours
    if config.active_hours_start is not None and config.active_hours_end is not None:
        console.print(f"  Active hours:     {config.active_hours_start:02d}:00 - {config.active_hours_end:02d}:00")
    else:
        console.print(f"  Active hours:     always on")
    
    # Active days
    if config.active_days:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        days_str = ", ".join(day_names[d] for d in config.active_days)
        console.print(f"  Active days:      {days_str}")
    else:
        console.print(f"  Active days:      every day")
    
    console.print()


def _run_interactive_configure(config: "NodeConfig") -> None:
    """Deprecated: redirects to the new `prsm config` system.

    Runs the legacy config wizard to update the old NodeConfig (kept for
    backward compat), then also runs the new PRSMConfig wizard so both
    configs stay in sync.
    """
    from prsm.node.compute_provider import detect_resources

    console.print()
    console.print("  NOTE: prsm node configure (interactive) is deprecated.", style="yellow")
    console.print("  Tip: Use `prsm config` for the full experience.", style="dim")
    console.print()

    # Run the legacy interactive prompts to update NodeConfig
    resources = detect_resources()
    console.print("  System: {} cores, {:.1f} GB RAM".format(
        resources.cpu_count, resources.memory_total_gb), style="dim")
    console.print()

    cpu = click.prompt("  CPU allocation for compute jobs (%)", default=config.cpu_allocation_pct, type=int)
    config.cpu_allocation_pct = max(10, min(90, cpu))
    mem = click.prompt("  Memory allocation for compute jobs (%)", default=config.memory_allocation_pct, type=int)
    config.memory_allocation_pct = max(10, min(90, mem))
    storage = click.prompt("  Storage to pledge (GB)", default=config.storage_gb, type=float)
    config.storage_gb = storage
    jobs = click.prompt("  Max concurrent compute jobs", default=config.max_concurrent_jobs, type=int)
    config.max_concurrent_jobs = max(1, jobs)
    if resources.gpu_available:
        gpu = click.prompt("  GPU allocation %", default=config.gpu_allocation_pct, type=int)
        config.gpu_allocation_pct = max(10, min(100, gpu))

    config.save()
    console.print(f"\n  ✅ Saved to {config.config_path}", style="green")

    # Also sync to the new config system
    try:
        from prsm.cli_modules.migration import migrate_if_needed
        migrate_if_needed()
    except Exception:
        pass


@node.command()
def benchmark():
    """Run hardware profiler and display compute tier classification."""
    from prsm.compute.wasm.profiler import HardwareProfiler
    profiler = HardwareProfiler()
    profile = profiler.detect()
    click.echo(f"Hardware Profile:")
    # Sprint 533 F51 fix: omit MHz if reading was junk (0).
    # Apple Silicon and some platforms don't expose real freq;
    # showing "0 MHz" or "4 MHz" is worse than just core count.
    if profile.cpu_freq_mhz and profile.cpu_freq_mhz >= 100:
        click.echo(
            f"  CPU: {profile.cpu_cores} cores @ "
            f"{profile.cpu_freq_mhz:.0f} MHz"
        )
    else:
        click.echo(f"  CPU: {profile.cpu_cores} cores")
    click.echo(f"  GPU: {profile.gpu_name or 'None detected'}")
    if profile.gpu_vram_gb > 0:
        click.echo(f"  VRAM: {profile.gpu_vram_gb:.1f} GB")
    click.echo(f"  RAM: {profile.ram_total_gb:.1f} GB total, {profile.ram_available_gb:.1f} GB available")
    click.echo(f"  TFLOPS: {profile.tflops_fp32:.2f} FP32")
    click.echo(f"  Compute Tier: {profile.compute_tier.value.upper()}")
    click.echo(f"  Thermal: {profile.thermal_class.value}")



# ============================================================================
# COMPUTE COMMANDS
# ============================================================================

@main.group()
def compute():
    """Compute job management commands"""
    pass


@compute.command()
@click.option('--prompt', required=True, help='Prompt to process')
@click.option('--model', default='nwtn', help='Model to use (default: nwtn)')
@click.option('--max-tokens', default=1000, type=int, help='Maximum tokens in response')
@click.option('--budget', type=float, help='Maximum FTNS to spend')
def submit(prompt: str, model: str, max_tokens: int, budget: Optional[float]):
    """Submit a compute job to the local node."""
    from prsm.compute.jobs_store import create_job

    job = create_job(prompt=prompt, model=model, max_tokens=max_tokens, budget=budget)

    console.print(f"[bold green]✅ Job submitted[/bold green]")
    console.print(f"   Job ID:  {job['job_id']}")
    console.print(f"   Model:   {job['model']}")
    console.print(f"   Status:  {job['status']}")
    if job.get('budget'):
        console.print(f"   Budget:  {job['budget']} FTNS")
    console.print()
    console.print("  Jobs are processed by your local P2P node.", style="dim")
    console.print("  Check status with:  prsm compute status " + job['job_id'], style="dim")


@compute.command()
@click.argument('job_id')
def status(job_id: str):
    """Get status of a compute job."""
    from prsm.compute.jobs_store import get_job

    job = get_job(job_id)
    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        raise SystemExit(1)

    console.print(f"\n[bold]Job Status: {job_id}[/bold]\n")
    table = Table(show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Status", job.get("status", "unknown"))
    table.add_row("Model", job.get("model", "—"))
    table.add_row("Prompt", (job.get("prompt", "") or "")[:80])
    if job.get("result"):
        table.add_row("Result", str(job["result"])[:120])
    if job.get("error"):
        table.add_row("Error", job["error"])
    if job.get("budget"):
        table.add_row("Budget", f"{job['budget']} FTNS")
    console.print(table)


@compute.command()
@click.argument('job_id')
def result(job_id: str):
    """Get result of a completed compute job."""
    from prsm.compute.jobs_store import get_job

    job = get_job(job_id)
    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        raise SystemExit(1)

    if job.get("status") == "completed" and job.get("result"):
        console.print("[bold green]📄 Job Result:[/bold green]")
        console.print(job["result"])
    else:
        console.print(f"[yellow]Job is {job.get('status', 'unknown')} — no result yet.[/yellow]")
        if job.get("error"):
            console.print(f"[red]Error: {job['error']}[/red]")


@compute.command()
@click.argument('job_id')
def cancel(job_id: str):
    """Cancel a running compute job."""
    from prsm.compute.jobs_store import update_job, get_job

    job = get_job(job_id)
    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        raise SystemExit(1)

    if job.get("status") in ("completed", "failed"):
        console.print(f"[yellow]Job is already {job['status']} — cannot cancel.[/yellow]")
        return

    update_job(job_id, status="cancelled")
    console.print(f"[green]✅ Job {job_id} cancelled[/green]")


@compute.command("list")
@click.option('--limit', default=10, type=int, help='Maximum number of jobs to list')
def list_compute_jobs(limit: int):
    """List recent compute jobs."""
    from prsm.compute.jobs_store import list_jobs as _list

    jobs = _list(limit=limit)
    if not jobs:
        console.print("No compute jobs yet.", style="dim")
        console.print("Submit one with:  prsm compute submit --prompt 'your question'", style="dim")
        return

    table = Table(title="Recent Compute Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Model", style="blue")
    table.add_column("Created", style="green")
    for job in jobs:
        created = job.get("created_at", "?")
        if isinstance(created, (int, float)):
            import datetime
            created = datetime.datetime.fromtimestamp(created).isoformat()[:19]
        table.add_row(
            (job.get("job_id") or "?")[:16],
            job.get("status", "?"),
            job.get("model", "—"),
            str(created)[:19],
        )
    console.print(table)


@compute.command("infer")
@click.option(
    "--prompt", "prompt", required=True,
    help="Prompt text to send to the inference path.",
)
@click.option(
    "--model", "model_id", default="gpt2",
    help="model_id (default: gpt2)",
)
@click.option(
    "--max-tokens", "max_tokens", default=8, type=int,
    help="Max output tokens (default: 8)",
)
@click.option(
    "--budget", "budget_ftns", default=1.0, type=float,
    help="Max FTNS to spend (default: 1.0)",
)
@click.option(
    "--privacy-tier", "privacy_tier",
    type=click.Choice(["none", "standard", "high", "maximum"]),
    default="none",
    help="Privacy tier (default: none)",
)
@click.option(
    "--content-tier", "content_tier",
    type=click.Choice(["A", "B", "C"]), default="A",
    help="Content tier (default: A)",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--verify-receipt", "verify_receipt", is_flag=True, default=False,
    help="Locally verify the receipt signature against "
    "--verify-pubkey-b64. Exits 1 on verify failure.",
)
@click.option(
    "--verify-pubkey-b64", "verify_pubkey_b64", default=None,
    help="Base64 public key for --verify-receipt. Use the "
    "operator's published pubkey (e.g., from "
    "/admin/parallax/pool/snapshot or /info).",
)
@click.option(
    "--stream", "do_stream", is_flag=True, default=False,
    help="Sprint 803 — consume the SSE event stream at "
    "/compute/inference/stream. Tokens print inline as they "
    "arrive; the final receipt is surfaced on stream end.",
)
def compute_infer_cli(
    prompt: str, model_id: str, max_tokens: int,
    budget_ftns: float, privacy_tier: str, content_tier: str,
    api_url_override: Optional[str], output_format: str,
    verify_receipt: bool, verify_pubkey_b64: Optional[str],
    do_stream: bool,
) -> None:
    """Sprint 802 — user-facing verifiable-inference CLI.

    POSTs to /compute/inference with the canonical body shape
    (prompt + model_id + budget_ftns + privacy_tier +
    content_tier + max_tokens) and displays the output + cost +
    signed-receipt summary.

    Pair with --verify-receipt + --verify-pubkey-b64 to locally
    re-run the canonical signing-payload check via sprint
    706/707's standalone verifier. Useful in CI / scripted
    flows that pin an operator's published pubkey.

    Exit 0 success, 1 daemon error OR verify failed, 2 unreachable.
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    body = {
        "prompt": prompt,
        "model_id": model_id,
        "budget_ftns": budget_ftns,
        "privacy_tier": privacy_tier,
        "content_tier": content_tier,
        "max_tokens": max_tokens,
    }

    # Sprint 803 — streaming branch
    if do_stream:
        stream_endpoint = f"{url}/compute/inference/stream"
        tokens_collected = []
        result_payload = None
        try:
            with _httpx.stream(
                "POST", stream_endpoint,
                json=body, timeout=120.0,
                headers=_node_api_key_headers(),  # sp1199 — auth on a keyed node
            ) as resp:
                if resp.status_code != 200:
                    # Sprint 825 — httpx streaming responses
                    # require .read() before .text is accessed.
                    # Without this, the error path raises
                    # "Attempted to access streaming response
                    # content, without having called read()" +
                    # the operator sees a confusing "unreachable"
                    # message instead of the actual server
                    # status detail.
                    try:
                        resp.read()
                        detail_text = resp.text
                    except Exception:
                        detail_text = (
                            f"<unable to read response body for "
                            f"status {resp.status_code}>"
                        )
                    if output_format == "json":
                        click.echo(_json.dumps({
                            "ok": False,
                            "status": resp.status_code,
                            "detail": detail_text,
                        }))
                    else:
                        console.print(
                            f"[red]Inference stream failed "
                            f"({resp.status_code}):[/red] {detail_text}"
                        )
                    raise SystemExit(1)
                # Parse SSE event/data lines
                current_event = None
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("event: "):
                        current_event = line[len("event: "):].strip()
                    elif line.startswith("data: "):
                        try:
                            payload = _json.loads(line[len("data: "):])
                        except Exception:
                            continue
                        if current_event == "token":
                            tokens_collected.append(payload)
                            if output_format == "text":
                                # Print delta inline, no newline
                                click.echo(
                                    payload.get("text_delta", ""),
                                    nl=False,
                                )
                        elif current_event == "result":
                            result_payload = payload
                        elif current_event == "error":
                            if output_format == "json":
                                click.echo(_json.dumps({
                                    "ok": False,
                                    "error": payload.get(
                                        "detail",
                                        "stream error",
                                    ),
                                }))
                            else:
                                console.print(
                                    f"\n[red]Stream error:[/red] "
                                    f"{payload.get('detail', payload)}"
                                )
                            raise SystemExit(1)
        except SystemExit:
            raise
        except Exception as exc:
            if output_format == "json":
                click.echo(_json.dumps({
                    "ok": False,
                    "error": f"daemon unreachable: {exc}",
                }))
            else:
                console.print(
                    f"[red]Daemon unreachable at {stream_endpoint}"
                    f"[/red] — {exc}"
                )
            raise SystemExit(2)

        # End of stream → render terminal state
        if output_format == "json":
            combined = dict(result_payload or {})
            combined["tokens"] = tokens_collected
            click.echo(_json.dumps(combined, indent=2))
            return

        # Text mode: tokens already printed inline; finish with
        # cost + receipt summary
        click.echo("")  # close the inline token line
        if result_payload:
            cost = result_payload.get(
                "ftns_charged", result_payload.get("cost_ftns", "?"),
            )
            console.print(f"[dim]Cost: {cost} FTNS[/dim]")
            receipt = result_payload.get("receipt") or {}
            sig = receipt.get("settler_signature", "")
            if sig:
                console.print(
                    f"[dim]Signed receipt: settler="
                    f"{(receipt.get('settler_node_id') or '?')[:12]}…, "
                    f"sig present[/dim]"
                )
        return

    endpoint = f"{url}/compute/inference"
    try:
        resp = _httpx.post(
            endpoint, json=body, timeout=120.0,
            headers=_node_api_key_headers(),  # sp1199 — auth on a keyed node
        )
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]Inference failed ({resp.status_code}):"
                f"[/red] {resp.text}"
            )
        raise SystemExit(1)
    data = resp.json()

    # Optional receipt verification
    verify_msg = None
    if verify_receipt:
        if not verify_pubkey_b64:
            console.print(
                "[yellow]--verify-receipt set but "
                "--verify-pubkey-b64 missing.[/yellow] "
                "Pass the operator's published pubkey to enable "
                "local verification."
            )
            raise SystemExit(1)
        try:
            from prsm.compute.inference.models import (
                InferenceReceipt,
            )
            from prsm.compute.inference.receipt import verify_receipt as _verify
            receipt = InferenceReceipt.from_dict(
                data.get("receipt") or {},
            )
            verify_msg = bool(_verify(receipt, public_key_b64=verify_pubkey_b64))
        except Exception as exc:
            verify_msg = False
            data["verify_error"] = str(exc)

    if output_format == "json":
        if verify_receipt:
            data["receipt_verified"] = bool(verify_msg)
        click.echo(_json.dumps(data, indent=2))
        if verify_receipt and not verify_msg:
            raise SystemExit(1)
        return

    # Text mode
    out = data.get("output", "")
    # Sprint 825 — cost lives at receipt.cost_ftns for parallax
    # responses. Sprint 802's original lookup only checked top-
    # level keys + rendered "?" against real daemons. Now fall
    # through receipt.cost_ftns before giving up.
    receipt = data.get("receipt") or {}
    cost = data.get(
        "ftns_charged",
        data.get(
            "cost_ftns",
            receipt.get("cost_ftns", "?"),
        ),
    )
    console.print(f"[bold]Output:[/bold] {out}")
    console.print(f"[dim]Cost: {cost} FTNS[/dim]")
    if receipt:
        sig = receipt.get("settler_signature", "")
        if sig:
            console.print(
                f"[dim]Signed receipt: settler="
                f"{receipt.get('settler_node_id', '?')[:12]}…, "
                f"sig present[/dim]"
            )
    if verify_receipt:
        if verify_msg:
            console.print(
                "[green]Receipt verified[/green] against the "
                "supplied pubkey."
            )
        else:
            console.print(
                "[red]Receipt verification failed[/red] — "
                "supplied pubkey does not match the signer."
            )
            raise SystemExit(1)


@compute.command("pay-infer")
@click.option(
    "--prompt", "prompt", required=True,
    help="Prompt text to send to the inference path.",
)
@click.option(
    "--model", "model_id", default="gpt2",
    help="model_id (default: gpt2)",
)
@click.option(
    "--max-tokens", "max_tokens", default=8, type=int,
    help="Max output tokens (default: 8)",
)
@click.option(
    "--budget", "budget_ftns", default=1.0, type=float,
    help="FTNS budget for the request (default: 1.0)",
)
@click.option(
    "--max-spend", "max_spend_ftns", default=None, type=float,
    help="Ceiling you authorize the provider to charge "
    "(default: --budget). The signed authorization caps the "
    "charge at this amount.",
)
@click.option(
    "--privacy-tier", "privacy_tier",
    type=click.Choice(["none", "standard", "high", "maximum"]),
    default="none",
    help="Privacy tier (default: none)",
)
@click.option(
    "--content-tier", "content_tier",
    type=click.Choice(["A", "B", "C"]), default="A",
    help="Content tier (default: A)",
)
@click.option(
    "--provider-address", "provider_address", default=None,
    help="Operator payee address (default: discover via GET /info)",
)
@click.option(
    "--network", "network_name",
    type=click.Choice(["mainnet", "testnet"]), default="testnet",
    help="Network the EscrowPool + chain_id live on (default: "
    "testnet = Base Sepolia 84532; mainnet = Base 8453)",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--verify-pubkey-b64", "verify_pubkey_b64", default=None,
    help="Base64 operator public key — when set, the returned "
    "receipt is verified inline (adds receipt_verified).",
)
@click.option(
    "--dry-run", "dry_run", is_flag=True, default=False,
    help="Preflight only: check the live preconditions (signing "
    "key, operator advertises a payee, escrow funded ≥ max-spend, "
    "chain_id) and print PASS/WARN/FAIL. Signs + broadcasts "
    "NOTHING. Exit 0 if all PASS, else 1.",
)
@click.option(
    "--stream", "do_stream", is_flag=True, default=False,
    help="Stream tokens as they generate (sp1310 pay_and_infer_stream) "
    "instead of waiting for the full unary response. The payment "
    "authorization + on-chain settlement are identical.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def compute_pay_infer_cli(
    prompt: str, model_id: str, max_tokens: int, budget_ftns: float,
    max_spend_ftns: Optional[float], privacy_tier: str, content_tier: str,
    provider_address: Optional[str], network_name: str,
    api_url_override: Optional[str], verify_pubkey_b64: Optional[str],
    dry_run: bool, do_stream: bool, output_format: str,
) -> None:
    """Sprint 1192 — pay for one inference from the terminal (requester-payment).

    Wraps the sp1189 SDK: signs an EIP-712 PaymentAuthorization bound to THIS
    request with your wallet key, and POSTs it so the provider settles A→B from
    your on-chain EscrowPool balance. The operator must run with
    PRSM_REQUESTER_PAYMENT=1; you must have escrow balance ≥ the charge (fund it
    with `prsm wallet deposit`).

    \b
    Key source (signs the authorization): PRIVATE_KEY env, else
    FTNS_WALLET_PRIVATE_KEY. NEVER passed on the command line.

    \b
    Example:
        export PRIVATE_KEY=0x...                       # your wallet
        prsm wallet deposit --amount 5 --network testnet
        prsm compute pay-infer --prompt "Hello" --budget 1 --network testnet

    Exit 0 success, 1 daemon/authorization error, 2 unreachable.
    """
    import json as _json
    from prsm.sdk.client import PRSMClient
    ctx = _wallet_load_signer(network_name)
    requester_key = ctx.get("private_key")
    if not requester_key:
        console.print(
            "❌ no signing key — set PRIVATE_KEY (or FTNS_WALLET_PRIVATE_KEY) "
            "to your wallet's private key. pay-infer signs a payment "
            "authorization with it.", style="red")
        raise SystemExit(1)
    chain_id = 84532 if network_name == "testnet" else 8453
    url = _api_url_from_creds(api_url_override)
    spend_ceiling = max_spend_ftns if max_spend_ftns is not None else budget_ftns

    if dry_run:
        # Preflight ONLY — verify the live preconditions, sign + broadcast nothing.
        from decimal import Decimal
        import httpx as _httpx
        addr = ctx.get("address")
        checks = []  # (ok: True|False|None, line)
        checks.append((True, f"signing key present → requester {addr}"))

        payee = provider_address
        if payee:
            checks.append((True, f"provider address (explicit): {payee}"))
        else:
            info = None
            try:
                info = _httpx.get(f"{url}/info", timeout=10.0,
                                  headers=_node_api_key_headers()).json()
            except Exception as exc:  # noqa: BLE001
                checks.append((False, f"cannot reach daemon /info at {url}: {exc}"))
            if info is not None:
                payee = (info or {}).get("operator_address")
                if payee:
                    checks.append((True, f"operator advertises payee: {payee}"))
                else:
                    checks.append((False,
                        "operator published no payment address (operator_address "
                        "absent from /info) — the operator isn't accepting requester "
                        "payment, or pass --provider-address"))
                # sp1196 — does the operator actually ACCEPT requester payment?
                # operator_address alone doesn't guarantee it (PRSM_REQUESTER_PAYMENT
                # may be off → a paid request gets a 402). Absent key = older node.
                accepted = (info or {}).get("requester_payment_accepted")
                if accepted is True:
                    checks.append((True, "operator accepts requester payment"))
                elif accepted is False:
                    checks.append((False,
                        "operator does NOT accept requester payment "
                        "(PRSM_REQUESTER_PAYMENT is off on the node) — a paid request "
                        "would be rejected (402)"))
                else:
                    checks.append((None,
                        "operator does not advertise payment acceptance (older node); "
                        "proceeding on operator_address"))

        try:
            from prsm.config.networks import resolve_endpoints
            from prsm.economy.web3.escrow_pool_client import EscrowPoolClient
            ep = resolve_endpoints(network_name)
            client = EscrowPoolClient(ep.rpc_url, ep.escrow_pool, ep.ftns_token)
            bal = Decimal(_run_async(client.balance_of(addr))) / (Decimal(10) ** 18)
            if bal >= Decimal(str(spend_ceiling)):
                checks.append((True,
                    f"escrow balance {bal} FTNS ≥ max-spend {spend_ceiling}"))
            else:
                checks.append((False,
                    f"escrow balance {bal} FTNS < max-spend {spend_ceiling} — fund it: "
                    f"prsm wallet deposit --amount {spend_ceiling} --network {network_name}"))
        except Exception as exc:  # noqa: BLE001
            checks.append((None,
                f"could not read escrow balance ({exc}); ensure web3 + an RPC are available"))

        if output_format == "json":
            import json as _json2
            click.echo(_json2.dumps({
                "ok": not any(ok is False for ok, _ in checks),
                "checks": [{"level": ("pass" if ok else "fail" if ok is False else "warn"),
                            "detail": line} for ok, line in checks],
            }))
        else:
            console.print("\n[bold]pay-infer preflight[/bold]")
            for ok, line in checks:
                tag = ("[green]PASS[/green]" if ok is True
                       else "[red]FAIL[/red]" if ok is False else "[yellow]WARN[/yellow]")
                console.print(f"  {tag} {line}")
        if any(ok is False for ok, _ in checks):
            if output_format != "json":
                console.print(
                    "\n[red]preflight FAIL[/red] — resolve the above before paying.\n")
            raise SystemExit(1)
        if output_format != "json":
            console.print("\n[green]preflight PASS[/green] — ready to pay.\n")
        return

    if do_stream:
        # sp1310 — paid STREAMING: print tokens live, then the charge/verify footer.
        # Same signed authorization + on-chain settlement as the unary path.
        async def _go_stream():
            client = PRSMClient(
                base_url=url,
                api_key=(os.environ.get("PRSM_NODE_API_KEY") or "").strip())
            terminal = None
            try:
                async for ev in client.pay_and_infer_stream(
                    prompt,
                    requester_key=requester_key,
                    provider_address=provider_address,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    budget_ftns=budget_ftns,
                    max_spend_ftns=spend_ceiling,
                    privacy_tier=privacy_tier,
                    content_tier=content_tier,
                    chain_id=chain_id,
                    verify_pubkey_b64=verify_pubkey_b64,
                ):
                    if ev.get("type") == "token" and output_format != "json":
                        import sys as _sys
                        _sys.stdout.write(ev.get("text_delta", ""))
                        _sys.stdout.flush()
                    elif ev.get("type") in ("result", "error"):
                        terminal = ev
            finally:
                await client.close()
            return terminal

        try:
            terminal = _run_async(_go_stream())
        except ValueError as exc:
            console.print(f"❌ {exc}", style="red")
            raise SystemExit(1)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if any(s in msg.lower()
                   for s in ("connect", "refused", "unreachable", "timeout")):
                console.print(f"❌ cannot reach daemon at {url}: {exc}", style="red")
                raise SystemExit(2)
            console.print(f"❌ pay-infer (stream) failed: {exc}", style="red")
            raise SystemExit(1)

        terminal = terminal or {
            "type": "error", "detail": "stream ended with no terminal event"}
        if terminal.get("type") == "error":
            detail = terminal.get("detail") or terminal
            if output_format == "json":
                click.echo(_json.dumps({"ok": False, "detail": detail}))
            else:
                console.print(f"\n[red]Payment/stream error:[/red] {detail}")
            raise SystemExit(1)
        if output_format == "json":
            click.echo(_json.dumps(terminal))
            return
        console.print()  # newline after the streamed tokens
        charged = terminal.get("ftns_charged")
        if charged is not None:
            console.print(
                f"[dim]charged: {charged} FTNS (settled from your escrow)[/dim]")
        if "receipt_verified" in terminal:
            ok = terminal["receipt_verified"]
            console.print(
                f"[dim]receipt_verified: "
                f"{'[green]yes[/green]' if ok else '[red]no[/red]'}[/dim]")
        console.print()
        return

    async def _go():
        # sp1199 — pass the node API key so /compute/inference (protected) authenticates
        # on a keyed node; PRSMClient sends it as a Bearer token.
        client = PRSMClient(
            base_url=url, api_key=(os.environ.get("PRSM_NODE_API_KEY") or "").strip())
        try:
            return await client.pay_and_infer(
                prompt,
                requester_key=requester_key,
                provider_address=provider_address,
                model_id=model_id,
                max_tokens=max_tokens,
                budget_ftns=budget_ftns,
                max_spend_ftns=spend_ceiling,
                privacy_tier=privacy_tier,
                content_tier=content_tier,
                chain_id=chain_id,
                verify_pubkey_b64=verify_pubkey_b64,
            )
        finally:
            await client.close()

    try:
        result = _run_async(_go())
    except ValueError as exc:
        # e.g. operator published no payment address + none supplied
        console.print(f"❌ {exc}", style="red")
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 — connection / signing failure
        msg = str(exc)
        if any(s in msg.lower() for s in ("connect", "refused", "unreachable", "timeout")):
            console.print(f"❌ cannot reach daemon at {url}: {exc}", style="red")
            raise SystemExit(2)
        console.print(f"❌ pay-infer failed: {exc}", style="red")
        raise SystemExit(1)

    result = result or {}
    # A rejected authorization comes back as the server's 402 body (FastAPI
    # {"detail": ...}) — no output/success. Detect + surface it as an error.
    succeeded = bool(result.get("success")) or ("output" in result)
    if not succeeded:
        detail = result.get("detail") or result.get("error") or result
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "detail": detail}))
        else:
            console.print("[red]Payment authorization rejected:[/red]")
            console.print(f"   {detail}")
        raise SystemExit(1)

    if output_format == "json":
        click.echo(_json.dumps(result))
        return
    console.print(f"\n[bold]{result.get('output', '')}[/bold]")
    charged = result.get("ftns_charged")
    if charged is not None:
        console.print(f"\n[dim]charged: {charged} FTNS (settled from your escrow)[/dim]")
    if "receipt_verified" in result:
        ok = result["receipt_verified"]
        console.print(
            f"[dim]receipt_verified: "
            f"{'[green]yes[/green]' if ok else '[red]no[/red]'}[/dim]")
    console.print()


@compute.command("pay-infer-multistage")
@click.option("--prompt", "prompt", required=True, help="Prompt text.")
@click.option("--model", "model_id", required=True,
              help="model_id (a big model that shards cross-host, e.g. Qwen/Qwen2.5-7B-Instruct)")
@click.option("--max-tokens", "max_tokens", default=8, type=int, help="Max output tokens (default: 8)")
@click.option("--budget", "budget_ftns", default=1.0, type=float,
              help="FTNS spending CAP (default: 1.0). The quote returns the deterministic price; "
                   "a price above this is not settleable.")
@click.option("--privacy-tier", "privacy_tier",
              type=click.Choice(["none", "standard", "high", "maximum"]), default="none")
@click.option("--content-tier", "content_tier", type=click.Choice(["A", "B", "C"]), default="A")
@click.option("--network", "network_name", type=click.Choice(["mainnet", "testnet"]),
              default="testnet", help="Network for chain_id (testnet=Base Sepolia 84532, mainnet=8453)")
@click.option("--api-url", "api_url_override", default=None, help="Override daemon URL")
@click.option("--verify-pubkey-b64", "verify_pubkey_b64", default=None,
              help="Base64 operator public key — verify the returned receipt inline.")
@click.option("--quote-only", "quote_only", is_flag=True, default=False,
              help="Preview the multi-stage quote (price + payees) and STOP — sign + POST nothing.")
@click.option("--stream", "do_stream", is_flag=True, default=False,
              help="Stream tokens live (pay_and_infer_multistage_stream). Only for multi-stage "
                   "models that DON'T slice-load — sliced big models can't stream (use unary).")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
def compute_pay_infer_multistage_cli(
    prompt: str, model_id: str, max_tokens: int, budget_ftns: float,
    privacy_tier: str, content_tier: str, network_name: str,
    api_url_override: Optional[str], verify_pubkey_b64: Optional[str],
    quote_only: bool, do_stream: bool, output_format: str,
) -> None:
    """Sprint 1330 (S5) — pay for a big-model MULTI-STAGE (cross-host sliced) inference.

    Wraps the SDK ``pay_and_infer_multistage``: quotes the planned stage→node payee set + the
    deterministic price, signs ONE per-stage PaymentAuthorization over it, and POSTs the paid
    request so each stage node self-settles its share from your escrow (Design A). For a request
    that routes to a single node, use ``pay-infer`` instead.

    \b
    Key source (signs the auth): PRIVATE_KEY env, else FTNS_WALLET_PRIVATE_KEY. NEVER on the CLI.
    \b
    Example:
        export PRIVATE_KEY=0x...
        prsm wallet deposit --amount 5 --network testnet
        prsm compute pay-infer-multistage --prompt "Hi" --model Qwen/Qwen2.5-7B-Instruct --network testnet

    Exit 0 success, 1 daemon/authorization/not-settleable error, 2 unreachable.
    """
    import json as _json
    from prsm.sdk.client import PRSMClient
    url = _api_url_from_creds(api_url_override)
    chain_id = 84532 if network_name == "testnet" else 8453

    if quote_only:
        async def _q():
            client = PRSMClient(base_url=url, api_key=(os.environ.get("PRSM_NODE_API_KEY") or "").strip())
            try:
                return await client._post("/compute/inference/quote-multistage", {
                    "model_id": model_id, "prompt": prompt,
                    "max_tokens": max_tokens, "budget_ftns": budget_ftns})
            finally:
                await client.close()
        try:
            q = _run_async(_q()) or {}
        except Exception as exc:  # noqa: BLE001
            console.print(f"❌ cannot reach daemon at {url}: {exc}", style="red")
            raise SystemExit(2)
        if output_format == "json":
            click.echo(_json.dumps(q)); return
        console.print(f"[bold]multi-stage quote[/bold] (multi_stage={q.get('multi_stage')}, "
                      f"settleable={q.get('settleable')})")
        if q.get("price_ftns") is not None:
            console.print(f"  price: {q['price_ftns']} FTNS | budget cap: {budget_ftns} | "
                          f"stages: {q.get('stage_count')}")
        for a, s in (q.get("payees") or []):
            console.print(f"  payee {a}: {int(s)/1e18} FTNS")
        if q.get("reason"):
            console.print(f"  reason: {q['reason']}")
        return

    ctx = _wallet_load_signer(network_name)
    requester_key = ctx.get("private_key")
    if not requester_key:
        console.print("❌ no signing key — set PRIVATE_KEY (or FTNS_WALLET_PRIVATE_KEY).", style="red")
        raise SystemExit(1)

    if do_stream:
        async def _go_stream():
            client = PRSMClient(base_url=url, api_key=(os.environ.get("PRSM_NODE_API_KEY") or "").strip())
            terminal = None
            try:
                async for ev in client.pay_and_infer_multistage_stream(
                    prompt, requester_key=requester_key, model_id=model_id,
                    max_tokens=max_tokens, budget_ftns=budget_ftns,
                    privacy_tier=privacy_tier, content_tier=content_tier,
                    chain_id=chain_id, verify_pubkey_b64=verify_pubkey_b64):
                    if ev.get("type") == "token" and output_format != "json":
                        import sys as _sys
                        _sys.stdout.write(ev.get("text_delta", "")); _sys.stdout.flush()
                    elif ev.get("type") in ("result", "error"):
                        terminal = ev
            finally:
                await client.close()
            return terminal
        try:
            terminal = _run_async(_go_stream())
        except ValueError as exc:
            console.print(f"❌ {exc}", style="red")
            raise SystemExit(1)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if any(s in msg.lower() for s in ("connect", "refused", "unreachable", "timeout")):
                console.print(f"❌ cannot reach daemon at {url}: {exc}", style="red")
                raise SystemExit(2)
            console.print(f"❌ pay-infer-multistage (stream) failed: {exc}", style="red")
            raise SystemExit(1)
        terminal = terminal or {}
        if output_format == "json":
            click.echo(_json.dumps(terminal))
            raise SystemExit(0 if terminal.get("type") == "result" else 1)
        if terminal.get("type") == "error":
            console.print(f"\n[red]stream error:[/red] {terminal.get('detail', terminal)}")
            raise SystemExit(1)
        mq = terminal.get("multistage_quote") or {}
        if mq.get("price_ftns") is not None:
            console.print(f"\n[dim]settled {mq['price_ftns']} FTNS across {mq.get('stage_count')} "
                          f"stage node(s) from your escrow[/dim]")
        if "receipt_verified" in terminal:
            ok = terminal["receipt_verified"]
            console.print(f"[dim]receipt_verified: "
                          f"{'[green]yes[/green]' if ok else '[red]no[/red]'}[/dim]")
        console.print()
        return

    async def _go():
        client = PRSMClient(base_url=url, api_key=(os.environ.get("PRSM_NODE_API_KEY") or "").strip())
        try:
            return await client.pay_and_infer_multistage(
                prompt, requester_key=requester_key, model_id=model_id,
                max_tokens=max_tokens, budget_ftns=budget_ftns,
                privacy_tier=privacy_tier, content_tier=content_tier,
                chain_id=chain_id, verify_pubkey_b64=verify_pubkey_b64)
        finally:
            await client.close()

    try:
        result = _run_async(_go())
    except ValueError as exc:
        # not multi-stage / not settleable / price>budget — actionable
        console.print(f"❌ {exc}", style="red")
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if any(s in msg.lower() for s in ("connect", "refused", "unreachable", "timeout")):
            console.print(f"❌ cannot reach daemon at {url}: {exc}", style="red")
            raise SystemExit(2)
        console.print(f"❌ pay-infer-multistage failed: {exc}", style="red")
        raise SystemExit(1)

    result = result or {}
    succeeded = bool(result.get("success")) or ("output" in result)
    if not succeeded:
        detail = result.get("detail") or result.get("error") or result
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "detail": detail}))
        else:
            console.print("[red]Paid multi-stage inference rejected:[/red]")
            console.print(f"   {detail}")
        raise SystemExit(1)

    if output_format == "json":
        click.echo(_json.dumps(result)); return
    console.print(f"\n[bold]{result.get('output', '')}[/bold]")
    mq = result.get("multistage_quote") or {}
    if mq.get("price_ftns") is not None:
        console.print(f"\n[dim]settled {mq['price_ftns']} FTNS across {mq.get('stage_count')} "
                      f"stage node(s) from your escrow[/dim]")
    if "receipt_verified" in result:
        ok = result["receipt_verified"]
        console.print(f"[dim]receipt_verified: "
                      f"{'[green]yes[/green]' if ok else '[red]no[/red]'}[/dim]")
    console.print()


@compute.command("models")
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def compute_models_cli(
    api_url_override: Optional[str], output_format: str,
) -> None:
    """Sprint 807 — list available inference model_ids.

    Wraps GET /compute/models. Users discover what models the
    daemon's executor accepts so they can pass --model to
    `prsm compute infer`.

    Exit codes:
      0 — listed (even when empty — operator may have started
          the daemon without wiring any models)
      1 — 503 (executor not initialized) + actionable hint
      2 — daemon unreachable
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/compute/models"
    try:
        resp = _httpx.get(endpoint, timeout=10.0)
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]models query failed ({resp.status_code}):"
                f"[/red] {resp.text}"
            )
            if resp.status_code == 503:
                console.print(
                    "[dim]Hint: set [bold]PRSM_INFERENCE_EXECUTOR="
                    "parallax[/bold] (or =mock for local testing) "
                    "and restart the daemon.[/dim]"
                )
        raise SystemExit(1)
    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return

    models = data.get("models", [])
    count = data.get("count", len(models))
    if not models:
        console.print(
            "[dim]0 models available.[/dim] The daemon's "
            "inference executor is wired but reports no "
            "supported models — operator likely hasn't loaded "
            "any HF checkpoints (`PRSM_PARALLAX_HF_MODEL_ID=...`)."
        )
        return
    console.print(f"[bold]{count} model(s) available:[/bold]")
    for m in models:
        console.print(f"  • [cyan]{m}[/cyan]")


@compute.command("verify-receipt")
@click.option(
    "--file", "receipt_file",
    type=click.Path(exists=False, dir_okay=False),
    required=True,
    help="Path to a saved InferenceReceipt JSON file (the "
    "output of `compute infer --format json` is "
    "directly compatible).",
)
@click.option(
    "--pubkey-b64", "pubkey_b64", required=False, default=None,
    help="Base64-encoded Ed25519 public key of the expected signer. OPTIONAL "
    "(sp1255): if omitted, the key EMBEDDED in the receipt is used and is bound "
    "to settler_node_id (node_id == sha256(pubkey)[:32]) so it can't be swapped. "
    "Pass it explicitly to additionally pin the signer to a key you already trust.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def compute_verify_receipt_cli(
    receipt_file: str, pubkey_b64: str, output_format: str,
) -> None:
    """Sprint 804 — pure-offline receipt verifier.

    Loads a saved InferenceReceipt JSON file + verifies the
    Ed25519 signature against the supplied pubkey. No daemon
    required; runs anywhere Python + cryptography are
    installed. Use for offline audits, CI gates, and
    independent third-party verification.

    Exit 0 verified, 1 on any failure (file missing, JSON
    parse error, receipt-schema error, signature mismatch).
    """
    import json as _json
    from pathlib import Path as _Path

    path = _Path(receipt_file)
    if not path.exists():
        msg = f"File not found: {receipt_file}"
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    try:
        raw = path.read_text()
        blob = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        msg = f"Failed to parse JSON: {exc}"
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)
    except OSError as exc:
        msg = f"Read failed: {exc}"
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    # A user passing the FULL inference response (from
    # sprint 802 `compute infer --format json`) gets a dict
    # with `receipt` nested inside. Accept both shapes.
    if "receipt" in blob and isinstance(blob["receipt"], dict):
        blob = blob["receipt"]

    try:
        from prsm.compute.inference.models import InferenceReceipt
        from prsm.compute.inference.receipt import (
            verify_receipt as _verify,
        )
        receipt = InferenceReceipt.from_dict(blob)
    except (KeyError, TypeError, ValueError) as exc:
        msg = (
            f"Receipt missing required fields or invalid: {exc}"
        )
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        msg = f"Receipt parse failed: {exc}"
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    # sp1255 — when no pubkey is supplied, verify against the key embedded in the
    # receipt (bound to settler_node_id). Require the receipt to actually carry one.
    _pk_source = "supplied" if pubkey_b64 else "embedded"
    if not pubkey_b64 and not receipt.settler_pubkey_b64:
        msg = ("Receipt carries no embedded settler pubkey (pre-sp1255 receipt); "
               "pass --pubkey-b64 <the signer's published key> to verify.")
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)
    try:
        ok = bool(_verify(receipt, public_key_b64=pubkey_b64))
    except Exception as exc:  # noqa: BLE001
        msg = f"verify_receipt raised: {exc}"
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    if output_format == "json":
        click.echo(_json.dumps({
            "ok": True,
            "verified": ok,
            "job_id": receipt.job_id,
            "settler_node_id": receipt.settler_node_id,
            "pubkey_source": _pk_source,
        }, indent=2))
        if not ok:
            raise SystemExit(1)
        return

    if ok:
        console.print(
            f"[green]Receipt verified[/green] — "
            f"job_id={receipt.job_id}, settler="
            f"{receipt.settler_node_id[:16]}… "
            f"({_pk_source} pubkey)"
        )
        return
    console.print(
        f"[red]Receipt verification failed[/red] — the {_pk_source} "
        f"pubkey does not match/bind to the signer for "
        f"job_id={receipt.job_id}."
    )
    raise SystemExit(1)


@compute.command("run")
@click.option('--prompt', default=None, help='Prompt to process (legacy NWTN path)')
@click.option('--query', default=None, help='Query for full forge pipeline (Rings 1-10)')
@click.option('--budget', default=10.0, type=float, help='FTNS budget for the job')
@click.option('--privacy', default='standard', type=click.Choice(['none', 'standard', 'high', 'maximum']), help='Privacy level for confidential compute')
@click.option('--api', default='http://127.0.0.1:8000', help='Node API URL')
def compute_run(prompt: str, query: str, budget: float, privacy: str, api: str):
    """Submit a compute job to your running daemon.

    Two modes:

      --prompt: Legacy path — routes directly to NWTN orchestrator.

      --query:  Full forge pipeline (Rings 1-10) — decomposes query via LLM,
                dispatches WASM mobile agents to edge nodes, aggregates results,
                settles FTNS payments, applies differential privacy.

    Requires a running daemon (`prsm node start`).
    """
    import httpx

    if not prompt and not query:
        console.print("[red]Either --prompt or --query is required[/red]")
        raise SystemExit(1)

    # Determine which path to use
    if query:
        # Enforce minimum budget for forge pipeline
        if budget <= 0:
            console.print("[red]FTNS budget is required for forge pipeline execution.[/red]")
            console.print("  PRSM's distributed compute network requires FTNS tokens to pay")
            console.print("  compute providers and data owners for their resources.")
            console.print()
            console.print("  [dim]Get a cost estimate first:[/dim]")
            console.print(f'    prsm compute quote "{query}"')
            console.print()
            console.print("  [dim]Then run with a budget:[/dim]")
            console.print(f'    prsm compute run --query "{query}" --budget 1.0')
            raise SystemExit(1)

        # Full forge pipeline (Rings 1-10)
        endpoint = f"{api}/compute/forge"
        payload = {
            "query": query,
            "budget_ftns": budget,
            "privacy_level": privacy,
        }
        console.print(f"[bold]Forge Pipeline[/bold] (Rings 1-10)")
        console.print(f"  Query: {query}")
        console.print(f"  Budget: {budget:.2f} FTNS")
        console.print(f"  Privacy: {privacy}")
        console.print()
    else:
        # Legacy NWTN path
        endpoint = f"{api}/compute/query"
        payload = {"prompt": prompt, "budget": budget}

    try:
        resp = httpx.post(endpoint, json=payload, timeout=120.0)
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to node API at {api}[/red]")
        console.print("  [dim]Start your daemon first: prsm node start[/dim]")
        raise SystemExit(1)

    if resp.status_code == 200:
        data = resp.json()

        if query:
            # Forge result display
            route = data.get("route", "unknown")
            console.print(f"[bold green]Result[/bold green] (route: {route}):")
            console.print(f"  {data.get('response', str(data.get('result', data)))}")
            console.print()
            console.print(f"  [dim]Job ID: {data.get('job_id', '?')}[/dim]")
            console.print(f"  [dim]Route: {route}[/dim]")
            console.print(f"  [dim]Budget: {data.get('budget_ftns', 0):.2f} FTNS[/dim]")
            traces = data.get("traces_collected", 0)
            if traces:
                console.print(f"  [dim]Training traces collected: {traces}[/dim]")
        else:
            # Legacy result display
            console.print(f"[bold green]Result:[/bold green]")
            text = data.get("response", str(data.get("result", data)))
            console.print(f"  {text}")
            cost = data.get("ftns_charged", data.get("ftns_cost"))
            if cost is not None:
                console.print(f"  [dim]Cost: {cost:.6f} FTNS[/dim]")
            job_id = data.get("job_id")
            if job_id:
                console.print(f"  [dim]Job ID: {job_id}[/dim]")

    elif resp.status_code == 503:
        console.print(f"[red]Service not available[/red]")
        detail = resp.json().get("detail", "") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        console.print(f"  [dim]{detail[:300]}[/dim]")
        raise SystemExit(1)
    elif resp.status_code == 404:
        console.print(f"[red]API endpoint not found on daemon[/red]")
        console.print("  [dim]Upgrade: pip install --upgrade prsm-network[/dim]")
        raise SystemExit(1)
    else:
        console.print(f"[red]Request failed: HTTP {resp.status_code}[/red]")
        if resp.text:
            console.print(f"  [dim]{resp.text[:300]}[/dim]")
        raise SystemExit(1)


@compute.command("quote")
@click.argument("query")
@click.option("--shards", default=3, help="Estimated number of data shards")
@click.option("--tier", default="t2", help="Hardware tier (t1-t4)")
def compute_quote(query, shards, tier):
    """Get a cost estimate for a compute query."""
    from prsm.economy.pricing.engine import PricingEngine

    engine = PricingEngine()
    quote = engine.quote_swarm_job(
        shard_count=shards,
        hardware_tier=tier,
        estimated_pcu_per_shard=50.0,
    )

    click.echo(f"Cost Quote for: {query}")
    click.echo(f"  Compute: {quote.compute_cost} FTNS")
    click.echo(f"  Data: {quote.data_cost} FTNS")
    click.echo(f"  Network Fee: {quote.network_fee} FTNS")
    click.echo(f"  Total: {quote.total} FTNS")


@main.command("demo")
def demo():
    """DEPRECATED: Ring 1-10 demo removed in v1.6.0.

    Use `prsm node start` to launch a real daemon, or `prsm demo-multinode`
    for a multi-node P2P walkthrough.
    """
    # Sprint 533 F46 fix: previously the help-text promised a Ring 1-10
    # demo that the command no longer runs. First-impression UX
    # blocker for new users. Updated docstring + actionable error +
    # nonzero exit so misuse is signaled clearly.
    console.print(
        "[yellow]⚠️  `prsm demo` was removed in v1.6.0 scope alignment.[/yellow]\n"
        "[bold]Try instead:[/bold]\n"
        "  • [cyan]prsm node start[/cyan]        — launch a real daemon\n"
        "  • [cyan]prsm demo-multinode[/cyan]    — multi-node P2P walkthrough\n"
        "  • [cyan]prsm setup[/cyan]             — first-run config wizard\n"
    )
    raise SystemExit(1)


@main.command("mcp-server")
def mcp_server_cmd():
    """Start the PRSM MCP server for LLM tool access.

    Exposes 17 PRSM tools to any MCP-compatible LLM (Claude, Gemini, etc.)
    via stdio protocol.

    Configure in Claude Desktop (~/.claude/claude_desktop_config.json):

        {"mcpServers": {"prsm": {"command": "python", "args": ["-m", "prsm.mcp_entry"]}}}
    """
    import subprocess, sys
    # Use the clean entry point to avoid stdout noise
    proc = subprocess.run([sys.executable, "-m", "prsm.mcp_entry"])
    raise SystemExit(proc.returncode)


@main.command("demo-multinode")
@click.option("--nodes", default=3, type=int, help="Number of nodes to spawn")
def compute_demo(nodes: int):
    """Run a multi-node P2P demonstration.

    Spawns local nodes and demonstrates:
    1. Escrow creation (FTNS locked before job runs)
    2. Job offer broadcast via gossip
    3. Cross-node job acceptance and execution
    4. Result consensus (multiple providers must agree)
    5. Payment release to winning provider
    """
    import asyncio
    from prsm.node.multinode_demo import MultiNodeDemo

    async def _run():
        demo = MultiNodeDemo()
        await demo.run()

    asyncio.run(_run())


# ============================================================================
# FTNS COMMANDS
# ============================================================================

@main.group()
def ftns():
    """FTNS token management commands"""
    pass


@ftns.command()
@click.option("--api-url", default=None, help="PRSM API URL (default: from stored credentials)")
def balance(api_url: str) -> None:
    """Show your FTNS token balance.

    Sprint 831 — F29 fix: pre-831 this command targeted
    /api/v1/ftns/balance (legacy ftns_api router) which is NOT
    mounted on the production daemon (see sprint 830's deferred-
    router allow-list). Every operator running `prsm ftns
    balance` got a 404 with no actionable message.

    Sprint 831 switches to the working inline /balance endpoint
    (defined at node/api.py:2266) which returns the operator's
    on-ledger FTNS balance + recent transactions in one shot.
    The response shape differs from FTNSBalanceResponse — the
    inline endpoint reports {wallet_id, balance,
    recent_transactions:[]} (no available/locked split because
    the ledger doesn't track lock state at this layer).
    """
    import httpx

    url = _api_url_from_creds(api_url)
    headers = _auth_headers() or {}

    try:
        response = httpx.get(
            f"{url}/balance",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm node start", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        table = Table(title="FTNS Balance")
        table.add_column("Type", style="cyan")
        table.add_column("Amount (FTNS)", style="green", justify="right")
        table.add_row("Balance", f"{data.get('balance', 0):.6f}")
        console.print(table)
        wallet_id = data.get("wallet_id") or "?"
        console.print(f"\n[dim]Wallet (node_id): {wallet_id}[/dim]")
        recent = data.get("recent_transactions") or []
        if recent:
            console.print(
                f"[dim]Recent transactions: "
                f"{len(recent)} (newest first)[/dim]"
            )
    elif response.status_code == 503:
        console.print(
            "[red]FAIL[/red] /balance returned 503 — daemon "
            "ledger not initialized. Run [bold]prsm node "
            "start[/bold] to bring the daemon up.",
        )
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(
            f"❌ Failed: HTTP {response.status_code} "
            f"{response.text[:120]}",
            style="red",
        )
        raise SystemExit(1)


@ftns.command()
@click.option("--to",          required=True,             help="Recipient user ID")
@click.option("--amount",      required=True, type=float, help="Amount in FTNS")
@click.option("--description", default="",                help="Optional transfer note")
@click.option("--api-url",     default=None,              help="PRSM API URL (default: from stored credentials)")
def transfer(to: str, amount: float, description: str, api_url: str) -> None:
    """Transfer FTNS tokens to another user."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    if amount <= 0:
        console.print("❌ Amount must be positive", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    console.print(f"💸 Transferring {amount:.6f} FTNS → {to}...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/ftns/transfer",
            # Correct request body: 'recipient' and 'description', not 'to_address'/'memo'
            json={"recipient": to, "amount": amount, "description": description},
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print("✅ Transfer successful!", style="bold green")
        # Correct response fields: 'status', 'amount', 'recipient' (no 'transaction_id'/'fee')
        console.print(f"   Recipient : {data.get('recipient')}")
        console.print(f"   Amount    : {data.get('amount', 0):.6f} FTNS")
        console.print(f"   Status    : {data.get('status')}")
    elif response.status_code == 402:
        console.print("❌ Insufficient FTNS balance", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Transfer failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@ftns.command("transfer-onchain")
@click.option("--to",      required=True,             help="Recipient Ethereum address (0x…)")
@click.option("--amount",  required=True, type=float, help="Amount in FTNS")
@click.option("--api-url", default=None,              help="PRSM API URL (default: from stored credentials)")
def transfer_onchain(to: str, amount: float, api_url: str) -> None:
    """Transfer FTNS tokens on-chain to an Ethereum address.

    Unlike `prsm ftns transfer` (which moves FTNS between user IDs
    on the off-chain DAG ledger), this command broadcasts a real
    ERC-20 Transfer on Base mainnet using the daemon's loaded
    FTNS_WALLET_PRIVATE_KEY. Requires daemon to be started with
    PRSM_ONCHAIN_FTNS=1 + FTNS_WALLET_PRIVATE_KEY set.
    """
    import httpx

    if amount <= 0:
        console.print("❌ Amount must be positive (> 0)", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    console.print(
        f"🔗 Broadcasting on-chain transfer: {amount:.6f} FTNS → {to}...",
        style="bold blue",
    )

    try:
        response = httpx.post(
            f"{url}/wallet/transfer/onchain",
            json={"to_address": to, "amount_ftns": amount},
            timeout=90.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print("✅ Transfer confirmed on-chain!", style="bold green")
        console.print(f"   tx_hash      : 0x{data.get('tx_hash', '').lstrip('0x')}")
        console.print(f"   block_number : {data.get('block_number')}")
        console.print(f"   status       : {data.get('status')}")
        console.print(f"   from         : {data.get('from_address')}")
        console.print(f"   to           : {data.get('to_address')}")
        console.print(f"   amount_ftns  : {data.get('amount_ftns')}")
    elif response.status_code == 503:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = response.text[:300]
        console.print("❌ On-chain ledger not available:", style="red")
        console.print(f"   {detail}")
        raise SystemExit(1)
    elif response.status_code == 422:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = response.text[:300]
        console.print(f"❌ Invalid input: {detail}", style="red")
        raise SystemExit(1)
    else:
        console.print(
            f"❌ Transfer failed: HTTP {response.status_code}", style="red",
        )
        console.print(f"   {response.text[:300]}")
        raise SystemExit(1)


@ftns.command()
@click.option("--amount",      required=True, type=float, help="FTNS to stake")
@click.option("--lock-days",   default=30,    type=int,   help="Lock duration in days (default: 30)")
@click.option("--api-url",     default=None,              help="PRSM API URL (default: from stored credentials)")
def stake(amount: float, lock_days: int, api_url: str) -> None:
    """
    Stake FTNS tokens for governance voting power.

    Staked tokens are locked for the specified duration and grant
    proportional voting power on governance proposals.
    """
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    console.print(f"🔒 Staking {amount:.6f} FTNS for {lock_days} days...", style="bold blue")

    try:
        # Correct endpoint: /api/v1/governance/stake (not /api/v1/ftns/stake)
        # Correct param name: lock_duration_days (not lock_period)
        response = httpx.post(
            f"{url}/api/v1/governance/stake",
            json={"amount": amount, "lock_duration_days": lock_days},
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        staking = data.get("data", {}).get("staking", {})
        console.print("✅ Staking successful!", style="bold green")
        console.print(f"   Staked        : {staking.get('staked_amount', amount)} FTNS")
        console.print(f"   Voting power  : {staking.get('voting_power', '?')}")
        console.print(f"   Unlock date   : {staking.get('unlock_date', '?')[:10]}")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 400:
        detail = response.json().get("detail", response.text)
        console.print(f"❌ Staking failed: {detail}", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Staking failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@ftns.command()
@click.option("--limit",   default=20, type=int, help="Transactions to show (max 100)")
@click.option("--search",  default=None,          help="Filter by description or transaction ID")
@click.option("--onchain", is_flag=True, default=False, help="Show on-chain TX (broadcast by daemon) instead of off-chain DAG ledger")
@click.option("--stats", is_flag=True, default=False, help="With --onchain: print aggregate stats instead of full TX list")
@click.option("--inbound", is_flag=True, default=False, help="With --onchain: show INBOUND Transfer events (FTNS sent TO this wallet) — sprint 512")
@click.option("--lookback-blocks", default=100000, type=int, help="With --onchain --inbound: how many blocks back to scan (default 100000 ~= 56hrs on Base)")
@click.option("--api-url", default=None,           help="PRSM API URL (default: from stored credentials)")
def history(limit: int, search: Optional[str], onchain: bool, stats: bool, inbound: bool, lookback_blocks: int, api_url: str) -> None:
    """Show your FTNS transaction history.

    Default: off-chain DAG ledger (user-to-user FTNS moves).
    With --onchain: real Base mainnet TX broadcast by this daemon's
    loaded FTNS_WALLET_PRIVATE_KEY (sprint-498 endpoint).
    With --onchain --stats: compact aggregate summary.
    With --onchain --inbound: FTNS received from external parties.
    """
    import httpx

    if onchain and inbound and stats:
        url = _api_url_from_creds(api_url)
        try:
            r = httpx.get(
                f"{url}/wallet/transactions/onchain/inbound/stats",
                params={"lookback_blocks": lookback_blocks},
                timeout=30.0,
            )
        except httpx.ConnectError:
            console.print(f"❌ Cannot connect to {url}", style="red")
            raise SystemExit(1)
        if r.status_code == 503:
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text[:300]
            console.print("❌ Inbound stats unavailable:", style="red")
            console.print(f"   {detail}")
            raise SystemExit(1)
        if r.status_code != 200:
            console.print(f"❌ HTTP {r.status_code}", style="red")
            raise SystemExit(1)
        d = r.json()
        console.print(f"\n[bold]Inbound FTNS stats[/bold] for {d.get('recipient')}")
        console.print(f"  count             : {d.get('count')}")
        console.print(f"  total received    : [green]{d.get('total_inbound_ftns', 0):.6f}[/green] FTNS")
        console.print(f"  first inbound blk : {d.get('first_inbound_block', '—')}")
        console.print(f"  last inbound blk  : {d.get('last_inbound_block', '—')}")
        console.print(f"  scan window       : blocks {d.get('from_block')}-{d.get('to_block')}\n")
        return

    if onchain and inbound:
        url = _api_url_from_creds(api_url)
        try:
            r = httpx.get(
                f"{url}/wallet/transactions/onchain/inbound",
                params={"lookback_blocks": lookback_blocks},
                timeout=30.0,
            )
        except httpx.ConnectError:
            console.print(f"❌ Cannot connect to {url}", style="red")
            raise SystemExit(1)
        if r.status_code == 503:
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text[:300]
            console.print("❌ Inbound scan unavailable:", style="red")
            console.print(f"   {detail}")
            raise SystemExit(1)
        if r.status_code != 200:
            console.print(f"❌ HTTP {r.status_code}: {r.text[:200]}", style="red")
            raise SystemExit(1)
        d = r.json()
        transfers = d.get("transfers", [])
        count = d.get("count", 0)
        console.print(
            f"\n🔗 Inbound FTNS for [bold]{d.get('recipient')}[/bold]  "
            f"[dim]({count} in blocks {d.get('from_block')}-{d.get('to_block')})[/dim]",
        )
        if not transfers:
            console.print("No inbound transfers in this window.", style="dim")
            return
        table = Table(title=f"Inbound FTNS  (count={count})")
        table.add_column("tx_hash",  style="dim",    max_width=18)
        table.add_column("block",    style="cyan",   justify="right")
        table.add_column("from",     style="white",  max_width=14)
        table.add_column("amount",   style="green",  justify="right")
        for t in transfers[: min(limit, 100)]:
            tx_hash = t.get("tx_hash") or ""
            table.add_row(
                (tx_hash[:16] + "…") if tx_hash else "—",
                str(t.get("block_number") or "—"),
                (t.get("from_address") or "—")[:14],
                f"{t.get('amount_ftns', 0):.6f}",
            )
        console.print(table)
        return

    if onchain and stats:
        url = _api_url_from_creds(api_url)
        try:
            r = httpx.get(
                f"{url}/wallet/transactions/onchain/stats",
                timeout=10.0,
            )
        except httpx.ConnectError:
            console.print(f"❌ Cannot connect to {url}", style="red")
            raise SystemExit(1)
        if r.status_code == 503:
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text[:300]
            console.print("❌ On-chain ledger not available:", style="red")
            console.print(f"   {detail}")
            raise SystemExit(1)
        if r.status_code != 200:
            console.print(f"❌ HTTP {r.status_code}", style="red")
            raise SystemExit(1)
        d = r.json()
        import datetime as _dt

        def _fmt_ts(ts):
            if ts is None:
                return "—"
            return _dt.datetime.fromtimestamp(ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        console.print(f"\n[bold]On-chain TX stats[/bold] for {d.get('address')}")
        console.print(f"  total       : {d.get('total_count')}")
        console.print(f"  confirmed   : [green]{d.get('confirmed_count')}[/green]")
        console.print(f"  pending     : [yellow]{d.get('pending_count')}[/yellow]")
        console.print(f"  rejected    : [red]{d.get('rejected_count')}[/red]")
        console.print(f"  total sent  : {d.get('total_ftns_sent', 0):.6f} FTNS  (confirmed only)")
        console.print(f"  first tx    : {_fmt_ts(d.get('first_tx_at'))}")
        console.print(f"  last tx     : {_fmt_ts(d.get('last_tx_at'))}")
        console.print(f"  [dim]scope:[/dim] {d.get('scope', '')}\n")
        return

    if onchain:
        url = _api_url_from_creds(api_url)
        try:
            response = httpx.get(
                f"{url}/wallet/transactions/onchain",
                timeout=10.0,
            )
        except httpx.ConnectError:
            console.print(f"❌ Cannot connect to {url}", style="red")
            raise SystemExit(1)

        if response.status_code == 503:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = response.text[:300]
            console.print("❌ On-chain ledger not available:", style="red")
            console.print(f"   {detail}")
            raise SystemExit(1)
        if response.status_code != 200:
            console.print(f"❌ Failed: HTTP {response.status_code}", style="red")
            raise SystemExit(1)

        data = response.json()
        txs = data.get("transactions", [])
        count = data.get("count", 0)
        addr = data.get("connected_address", "?")
        scope = data.get("scope", "")

        console.print(
            f"🔗 On-chain TX for [bold]{addr}[/bold]  "
            f"[dim]({count} total — {scope})[/dim]",
        )
        if not txs:
            console.print(
                "No on-chain transactions in this session.",
                style="dim",
            )
            return

        table = Table(title=f"On-chain FTNS TX  (count={count})")
        table.add_column("tx_hash",   style="dim",     max_width=18)
        table.add_column("block",     style="cyan",    justify="right")
        table.add_column("from",      style="white",   max_width=14)
        table.add_column("to",        style="white",   max_width=14)
        table.add_column("amount",    style="green",   justify="right")
        table.add_column("status",    style="magenta")

        for tx in txs[: min(limit, 100)]:
            tx_hash = (tx.get("tx_hash") or "")
            table.add_row(
                (tx_hash[:16] + "…") if tx_hash else "—",
                str(tx.get("block_number") or "—"),
                (tx.get("from_address") or "—")[:14],
                (tx.get("to_address") or "—")[:14],
                f"{tx.get('amount_ftns', 0):.6f}",
                tx.get("status", "?"),
            )
        console.print(table)
        return

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    params: dict = {"limit": min(limit, 100)}
    if search:
        params["search"] = search

    try:
        # Correct endpoint: /transactions, not /history
        response = httpx.get(
            f"{url}/api/v1/ftns/transactions",
            params=params,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    if response.status_code != 200:
        console.print(f"❌ Failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)

    data = response.json()
    transactions = data.get("transactions", [])

    if not transactions:
        console.print("No transactions found.", style="dim")
        return

    table = Table(title=f"FTNS History — {data.get('user_id', '?')}")
    table.add_column("ID",          style="dim",     max_width=14)
    table.add_column("Type",        style="cyan")
    table.add_column("Amount",      style="green",   justify="right")
    table.add_column("Description", style="white",   max_width=36)
    table.add_column("Status",      style="magenta")
    table.add_column("Time",        style="blue")

    for tx in transactions:
        amount = float(tx.get("amount", 0))
        table.add_row(
            str(tx.get("transaction_id", ""))[:12] + "…",
            tx.get("transaction_type", "?"),
            f"{amount:.6f}",
            (tx.get("description") or "")[:36],
            tx.get("status", "?"),
            str(tx.get("timestamp", ""))[:19],
        )
    console.print(table)


@ftns.command("yield-estimate")
@click.option("--hours", default=8, help="Hours per day available for compute")
@click.option("--stake", default=0, type=float, help="FTNS staked")
def ftns_yield_estimate(hours, stake):
    """Estimate daily/monthly FTNS earnings based on your hardware."""
    from prsm.compute.wasm.profiler import HardwareProfiler
    from prsm.economy.pricing.engine import PricingEngine
    from prsm.economy.pricing.models import ProsumerTier

    profiler = HardwareProfiler()
    profile = profiler.detect()
    tier = ProsumerTier.from_stake(stake)
    engine = PricingEngine()

    estimate = engine.yield_estimate(
        hardware_tier=profile.compute_tier.value,
        tflops=profile.tflops_fp32,
        hours_per_day=hours,
        prosumer_tier=tier,
    )

    click.echo(f"Yield Estimate:")
    click.echo(f"  Hardware: {profile.compute_tier.value.upper()} ({profile.tflops_fp32:.1f} TFLOPS)")
    click.echo(f"  Stake: {stake:.0f} FTNS ({tier.label})")
    click.echo(f"  Yield Boost: {estimate['yield_boost']}x")
    click.echo(f"  Daily: {float(estimate['daily_ftns']):.2f} FTNS")
    click.echo(f"  Monthly: {float(estimate['monthly_ftns']):.2f} FTNS")


# ============================================================================
# SETTLEMENT COMMANDS (under ftns group)
# ============================================================================

def _get_api_url() -> str:
    """Return the PRSM API URL — env var override > loopback default.

    Settlement subcommands read this when the caller omits --api-url.
    Uses PRSM_API_URL env var if set, else 127.0.0.1:8000 which matches
    the default in other CLI subcommands (e.g. `start --api-port 8000`).
    """
    return os.environ.get("PRSM_API_URL", "http://127.0.0.1:8000")


@ftns.group()
def settle():
    """On-chain batch settlement commands."""
    pass


@settle.command("status")
@click.option("--api-url", default=None, help="PRSM API URL")
def settle_status(api_url: str) -> None:
    """Show batch settlement queue status."""
    import httpx
    from rich.console import Console
    from rich.table import Table
    console = Console()
    url = api_url or _get_api_url()
    try:
        resp = httpx.get(f"{url}/settlement/stats", timeout=10)
        data = resp.json()
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return

    table = Table(title="Batch Settlement Status", show_header=True)
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Mode", str(data.get("mode", "unknown")))
    table.add_row("Queue Size", str(data.get("queue_size", 0)))
    table.add_row("Pending FTNS", f"{data.get('pending_amount', 0):.6f}")
    table.add_row("Flush Interval", f"{data.get('flush_interval', 0)}s")
    table.add_row("Flush Threshold", f"{data.get('flush_threshold', 0)} FTNS")
    table.add_row("Total Settled", str(data.get("total_settled", 0)))
    table.add_row("Gas Txs Saved", str(data.get("gas_txs_saved", 0)))
    ago = data.get("last_flush_ago")
    table.add_row("Last Flush", f"{ago:.0f}s ago" if ago else "never")
    console.print(table)


@settle.command("pending")
@click.option("--api-url", default=None, help="PRSM API URL")
def settle_pending(api_url: str) -> None:
    """List pending un-settled transfers."""
    import httpx
    from rich.console import Console
    from rich.table import Table
    console = Console()
    url = api_url or _get_api_url()
    try:
        resp = httpx.get(f"{url}/settlement/pending", timeout=10)
        data = resp.json()
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return

    pending = data.get("pending", [])
    if not pending:
        console.print("  No pending transfers.")
        return

    table = Table(title=f"Pending Transfers ({len(pending)})", show_header=True)
    table.add_column("TX ID")
    table.add_column("To")
    table.add_column("Amount")
    table.add_column("Age")
    for p in pending:
        table.add_row(
            str(p.get("tx_id", "")),
            str(p.get("to", "")),
            f"{p.get('amount', 0):.6f}",
            f"{p.get('age_seconds', 0):.0f}s",
        )
    console.print(table)


@settle.command("flush")
@click.option("--api-url", default=None, help="PRSM API URL")
def settle_flush(api_url: str) -> None:
    """Manually trigger batch settlement (flush all pending transfers on-chain)."""
    import httpx
    from rich.console import Console
    console = Console()
    url = api_url or _get_api_url()
    console.print("  Flushing settlement queue...")
    try:
        resp = httpx.post(f"{url}/settlement/flush", timeout=120)
        data = resp.json()
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return

    settled = data.get("settled_count", 0)
    net = data.get("net_transfers", 0)
    amount = data.get("total_amount", 0)
    duration = data.get("duration_seconds", 0)
    hashes = data.get("tx_hashes", [])
    errors = data.get("errors", [])

    if settled == 0:
        console.print("  Nothing to settle.")
        return

    console.print(f"  Settled {settled} transfers → {net} on-chain txs")
    console.print(f"  Total: {amount:.6f} FTNS in {duration:.1f}s")
    if hashes:
        for h in hashes:
            console.print(f"  TX: {h}")
    if errors:
        for e in errors:
            console.print(f"  [red]Error:[/red] {e}")


@settle.command("history")
@click.option("--api-url", default=None, help="PRSM API URL")
@click.option("--limit", default=10, help="Number of records")
def settle_history(api_url: str, limit: int) -> None:
    """Show recent settlement history."""
    import httpx
    from rich.console import Console
    from rich.table import Table
    console = Console()
    url = api_url or _get_api_url()
    try:
        resp = httpx.get(f"{url}/settlement/history", params={"limit": limit}, timeout=10)
        data = resp.json()
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return

    history = data.get("history", [])
    if not history:
        console.print("  No settlement history.")
        return

    table = Table(title="Settlement History", show_header=True)
    table.add_column("Settled")
    table.add_column("Net TXs")
    table.add_column("Amount")
    table.add_column("Duration")
    table.add_column("Errors")
    for h in history:
        table.add_row(
            str(h.get("settled_count", 0)),
            str(h.get("net_transfers", 0)),
            f"{h.get('total_amount', 0):.6f}",
            f"{h.get('duration_seconds', 0):.1f}s",
            str(len(h.get("errors", []))),
        )
    console.print(table)


# ============================================================================
# BRIDGE COMMANDS (under ftns group)
# ============================================================================

@ftns.group()
def bridge():
    """FTNS token bridge commands for cross-chain transfers."""
    pass


@bridge.command()
@click.option('--amount', required=True, type=float, help='Amount of FTNS to deposit')
@click.option('--address', required=True, help='Destination on-chain address')
@click.option('--chain', default=137, type=int, help='Destination chain ID (default: 137 for Polygon)')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def deposit(amount: float, address: str, chain: int, api_url: str):
    """
    Deposit FTNS tokens from local balance to external chain.
    
    Burns local FTNS and initiates bridge transfer to mint tokens on the destination chain.
    The transaction will go through validation, processing, and confirmation stages.
    """
    import httpx
    
    console.print(f"🌉 Depositing {amount} FTNS to chain {chain}...", style="bold blue")
    console.print(f"   Destination address: {address}")
    
    try:
        response = httpx.post(
            f"{api_url}/bridge/deposit",
            json={
                "amount": amount,
                "chain_address": address,
                "destination_chain": chain
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            tx = data.get('transaction', {})
            console.print(f"✅ Bridge deposit initiated!", style="bold green")
            console.print(f"   Transaction ID: {tx.get('transaction_id')}")
            console.print(f"   Status: {tx.get('status')}")
            console.print(f"   Amount: {int(tx.get('amount', 0)) / 10**18:.4f} FTNS")
            console.print(f"   Fee: {int(tx.get('fee_amount', 0)) / 10**18:.4f} FTNS")
            console.print(f"   Created: {tx.get('created_at', 'N/A')[:19]}")
            
            if tx.get('status') == 'completed':
                console.print(f"   🎉 Transaction completed!", style="green")
            elif tx.get('status') == 'pending':
                console.print(f"   ⏳ Transaction pending - check status with: prsm ftns bridge status", style="yellow")
        else:
            error_detail = response.json().get('detail', response.text)
            console.print(f"❌ Bridge deposit failed: {response.status_code}", style="red")
            console.print(f"   {error_detail}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@bridge.command()
@click.option('--amount', required=True, type=float, help='Amount of FTNS to withdraw')
@click.option('--address', required=True, help='Source on-chain address')
@click.option('--chain', default=137, type=int, help='Source chain ID (default: 137 for Polygon)')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def withdraw(amount: float, address: str, chain: int, api_url: str):
    """
    Withdraw FTNS tokens from external chain to local balance.
    
    Locks on-chain FTNS and initiates bridge transfer to mint local FTNS to your account.
    The transaction will go through validation, processing, and confirmation stages.
    """
    import httpx
    
    console.print(f"🌉 Withdrawing {amount} FTNS from chain {chain}...", style="bold blue")
    console.print(f"   Source address: {address}")
    
    try:
        response = httpx.post(
            f"{api_url}/bridge/withdraw",
            json={
                "amount": amount,
                "chain_address": address,
                "source_chain": chain
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            tx = data.get('transaction', {})
            console.print(f"✅ Bridge withdraw initiated!", style="bold green")
            console.print(f"   Transaction ID: {tx.get('transaction_id')}")
            console.print(f"   Status: {tx.get('status')}")
            console.print(f"   Amount: {int(tx.get('amount', 0)) / 10**18:.4f} FTNS")
            console.print(f"   Fee: {int(tx.get('fee_amount', 0)) / 10**18:.4f} FTNS")
            console.print(f"   Created: {tx.get('created_at', 'N/A')[:19]}")
            
            if tx.get('status') == 'completed':
                console.print(f"   🎉 Transaction completed!", style="green")
            elif tx.get('status') == 'pending':
                console.print(f"   ⏳ Transaction pending - check status with: prsm ftns bridge status", style="yellow")
        else:
            error_detail = response.json().get('detail', response.text)
            console.print(f"❌ Bridge withdraw failed: {response.status_code}", style="red")
            console.print(f"   {error_detail}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@bridge.command()
@click.option('--tx-id', help='Specific transaction ID to look up')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def status(tx_id: str, api_url: str):
    """
    Get bridge status and pending operations.
    
    Without --tx-id: Shows overall bridge statistics and pending transactions.
    With --tx-id: Shows status of a specific bridge transaction.
    """
    import httpx
    
    try:
        if tx_id:
            # Get specific transaction status
            response = httpx.get(f"{api_url}/bridge/transactions/{tx_id}", timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                tx = data.get('transaction', {})
                
                table = Table(title=f"Bridge Transaction: {tx_id[:20]}...")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Transaction ID", tx.get('transaction_id', 'N/A'))
                table.add_row("Direction", tx.get('direction', 'N/A').upper())
                table.add_row("Status", tx.get('status', 'N/A'))
                table.add_row("Amount", f"{int(tx.get('amount', 0)) / 10**18:.4f} FTNS")
                table.add_row("Fee", f"{int(tx.get('fee_amount', 0)) / 10**18:.4f} FTNS")
                table.add_row("Chain Address", tx.get('chain_address', 'N/A'))
                table.add_row("Source Chain", str(tx.get('source_chain', 'N/A')))
                table.add_row("Dest Chain", str(tx.get('destination_chain', 'N/A')))
                table.add_row("Created", tx.get('created_at', 'N/A')[:19] if tx.get('created_at') else 'N/A')
                table.add_row("Updated", tx.get('updated_at', 'N/A')[:19] if tx.get('updated_at') else 'N/A')
                
                if tx.get('completed_at'):
                    table.add_row("Completed", tx.get('completed_at')[:19])
                if tx.get('source_tx_hash'):
                    table.add_row("Source TX Hash", tx.get('source_tx_hash')[:20] + "...")
                if tx.get('destination_tx_hash'):
                    table.add_row("Dest TX Hash", tx.get('destination_tx_hash')[:20] + "...")
                if tx.get('error_message'):
                    table.add_row("Error", tx.get('error_message'), style="red")
                
                console.print(table)
            elif response.status_code == 404:
                console.print(f"❌ Transaction not found: {tx_id}", style="red")
            else:
                console.print(f"❌ Failed to get transaction: {response.status_code}", style="red")
        else:
            # Get overall bridge status
            response = httpx.get(f"{api_url}/bridge/status", timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                stats = data.get('stats', {})
                limits = data.get('limits', {})
                pending = data.get('pending_transactions', [])
                
                # Stats table
                stats_table = Table(title="Bridge Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")
                
                stats_table.add_row("Total Deposited", f"{int(stats.get('total_deposited', 0)) / 10**18:.4f} FTNS")
                stats_table.add_row("Total Withdrawn", f"{int(stats.get('total_withdrawn', 0)) / 10**18:.4f} FTNS")
                stats_table.add_row("Total Fees Collected", f"{int(stats.get('total_fees_collected', 0)) / 10**18:.4f} FTNS")
                stats_table.add_row("Pending Transactions", str(stats.get('pending_transactions', 0)))
                stats_table.add_row("Completed Transactions", str(stats.get('completed_transactions', 0)))
                stats_table.add_row("Failed Transactions", str(stats.get('failed_transactions', 0)))
                
                console.print(stats_table)
                
                # Limits table
                if limits:
                    limits_table = Table(title="Bridge Limits")
                    limits_table.add_column("Limit", style="cyan")
                    limits_table.add_column("Value", style="magenta")
                    
                    limits_table.add_row("Minimum Amount", f"{int(limits.get('min_amount', 0)) / 10**18:.4f} FTNS")
                    limits_table.add_row("Maximum Amount", f"{int(limits.get('max_amount', 0)) / 10**18:.4f} FTNS")
                    limits_table.add_row("Daily Limit", f"{int(limits.get('daily_limit', 0)) / 10**18:.4f} FTNS")
                    limits_table.add_row("Fee (BPS)", str(limits.get('fee_bps', 0)))
                    
                    console.print(limits_table)
                
                # Pending transactions
                if pending:
                    pending_table = Table(title=f"Pending Transactions ({len(pending)})")
                    pending_table.add_column("TX ID", style="dim")
                    pending_table.add_column("Direction", style="cyan")
                    pending_table.add_column("Amount", style="green")
                    pending_table.add_column("Status", style="magenta")
                    pending_table.add_column("Created", style="blue")
                    
                    for tx in pending[:10]:  # Show max 10
                        pending_table.add_row(
                            tx.get('transaction_id', 'N/A')[:20] + "...",
                            tx.get('direction', 'N/A').upper(),
                            f"{int(tx.get('amount', 0)) / 10**18:.4f}",
                            tx.get('status', 'N/A'),
                            tx.get('created_at', 'N/A')[:19]
                        )
                    
                    console.print(pending_table)
                    if len(pending) > 10:
                        console.print(f"   ... and {len(pending) - 10} more pending transactions", style="dim")
            else:
                console.print(f"❌ Failed to get bridge status: {response.status_code}", style="red")
                
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@bridge.command()
@click.option('--limit', default=20, type=int, help='Maximum transactions to show')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def history(limit: int, api_url: str):
    """Show bridge transaction history for the current user."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/bridge/transactions?limit={limit}", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get('transactions', [])
            
            if not transactions:
                console.print("No bridge transactions found.", style="dim")
                return
            
            table = Table(title="Bridge Transaction History")
            table.add_column("TX ID", style="dim")
            table.add_column("Direction", style="cyan")
            table.add_column("Amount", style="green")
            table.add_column("Fee", style="yellow")
            table.add_column("Status", style="magenta")
            table.add_column("Created", style="blue")
            
            for tx in transactions:
                table.add_row(
                    tx.get('transaction_id', 'N/A')[:16] + "...",
                    tx.get('direction', 'N/A').upper(),
                    f"{int(tx.get('amount', 0)) / 10**18:.4f}",
                    f"{int(tx.get('fee_amount', 0)) / 10**18:.4f}",
                    tx.get('status', 'N/A'),
                    tx.get('created_at', 'N/A')[:19]
                )
            
            console.print(table)
        else:
            console.print(f"❌ Failed to get bridge history: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")




# ============================================================================
# STORAGE COMMANDS
# ============================================================================

@main.group()
def storage():
    """Content storage management commands"""
    pass


def _server_detail_or(response, fallback: str) -> str:
    """Return the server's error ``detail`` (a FastAPI HTTPException body) when
    present, else ``fallback``. sp1027 — CLI error messages must surface the
    actionable server reason (e.g. "ContentPublisher … libtorrent not installed")
    instead of a generic guess; a hard-coded message buried the real cause at the
    Tier-1 live bench."""
    detail = None
    try:
        detail = (response.json() or {}).get("detail")
    except Exception:  # noqa: BLE001 - non-JSON / empty body
        detail = None
    if not (isinstance(detail, str) and detail.strip()):
        # sp1072 — fall back to parsing the raw body text as JSON. Some responses (and
        # test doubles) expose the body via ``.text`` without a working ``.json()``;
        # without this the actionable server detail is silently dropped for the
        # generic fallback (the bug the sp832 actionable-503 test was guarding).
        try:
            import json as _json
            body = getattr(response, "text", None)
            if isinstance(body, str) and body.strip():
                d2 = (_json.loads(body) or {}).get("detail")
                if isinstance(d2, str) and d2.strip():
                    detail = d2
        except Exception:  # noqa: BLE001
            pass
    return detail if isinstance(detail, str) and detail.strip() else fallback


@storage.command()
@click.argument("file-path", type=click.Path(exists=True, readable=True))
@click.option("--description",   default="",    help="Content description")
@click.option(
    "--royalty-rate",
    default=0.01, type=float,
    help="FTNS earned per access (0.001–0.1, default: 0.01)"
)
@click.option(
    "--parent-cids",
    default="",
    help="Comma-separated CIDs of content this file derives from"
)
@click.option("--replicas",   default=3, type=int, help="Replication factor (1–10)")
@click.option("--api-url",    default=None,         help="PRSM API URL (default: from stored credentials)")
@click.option("--semantic-shard", is_flag=True, default=False, help="Semantically shard the dataset by content similarity before uploading")
def upload(
    file_path: str,
    description: str,
    royalty_rate: float,
    parent_cids: str,
    replicas: int,
    api_url: str,
    semantic_shard: bool,
) -> None:
    """
    Upload a file to ContentStore and register provenance for royalty collection.

    The file is stored in the native ContentStore and a provenance record is
    created in the platform database. When other users access this content,
    they pay the configured royalty rate to your FTNS balance.

    \b
    Examples:
        prsm storage upload model_weights.pt --royalty-rate 0.05
        prsm storage upload dataset.csv --description "Training data Q1 2026"
        prsm storage upload paper.pdf --parent-cids QmAbc...,QmDef...
    """
    import httpx

    # Sprint 832 — F29 fix: the inline /content/upload endpoint
    # doesn't require auth headers (see `prsm content publish`
    # + `prsm node share` which work without login). Pre-832
    # this command hard-failed on missing login because the
    # legacy /api/v1/content/upload required it. Pass whatever
    # headers we have (possibly empty) and let the server decide.
    headers = _auth_headers() or {}
    url = _api_url_from_creds(api_url)
    file_path_obj = Path(file_path)
    file_size = file_path_obj.stat().st_size

    if semantic_shard:
        console.print(f"[bold]Semantic sharding enabled[/bold]")
        from prsm.data.shard_models import SemanticShard, SemanticShardManifest

        # Read file and create simple shards based on line count
        # (Real implementation would use embeddings for clustering)
        with open(file_path, 'rb') as f:
            content = f.read()

        file_size = len(content)
        # Split into chunks of ~1MB
        chunk_size = 1024 * 1024  # 1MB
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunks.append(content[i:i + chunk_size])

        if not chunks:
            chunks = [content]

        shards = []
        for i, chunk in enumerate(chunks):
            shard = SemanticShard(
                shard_id=f"shard-{i:04d}",
                parent_dataset=file_path_obj.name,
                cid=f"pending-upload-{i}",
                centroid=[float(i) / max(len(chunks), 1)],  # Placeholder centroid
                record_count=len(chunk),
                size_bytes=len(chunk),
                keywords=[file_path_obj.stem, f"shard-{i}"],
            )
            shards.append(shard)

        manifest = SemanticShardManifest(
            dataset_id=file_path_obj.stem,
            total_records=file_size,
            total_size_bytes=file_size,
            shards=shards,
        )

        console.print(f"  Shards: {len(shards)}")
        console.print(f"  Total size: {file_size:,} bytes")
        console.print(f"  Manifest: {manifest.dataset_id}")
        console.print()

    console.print(
        f"📤 Uploading {file_path_obj.name} ({file_size:,} bytes)...",
        style="bold blue"
    )
    if parent_cids:
        console.print(f"   Derivative of: {parent_cids}", style="dim")

    # Sprint 832 — F29 fix: pre-832 this command POSTed multipart
    # to /api/v1/content/upload (legacy content_api router,
    # unmounted on production daemon per sprint 830). Every
    # operator running `prsm storage upload` got a 404.
    # Sprint 832 switches to the inline /content/upload endpoint
    # (node/api.py:7654) which accepts JSON ContentUploadRequest
    # {text, filename, royalty_rate, parent_cids, replicas} —
    # the same wire format `prsm content publish` (sprint 806)
    # and `prsm node share` (sprint 574) use successfully.
    #
    # Binary content: inline endpoint takes a `text` field (UTF-8
    # str). If the file isn't UTF-8-decodable, redirect operators
    # at sprint-817's `prsm content publish-shard` which accepts
    # base64 binary via /content/upload/shard.
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except UnicodeDecodeError:
        console.print(
            f"[red]Binary file detected — inline /content/"
            f"upload accepts UTF-8 text only.[/red]"
        )
        console.print(
            "[yellow]Use [bold]prsm content publish-shard[/bold] "
            "for binary uploads (base64 path).[/yellow]"
        )
        raise SystemExit(1)

    body: Dict[str, Any] = {
        "text": text,
        "filename": file_path_obj.name,
        "royalty_rate": float(royalty_rate),
        "replicas": int(replicas),
    }
    if parent_cids:
        # Inline endpoint expects parent_cids as a list (sprint
        # 821); CLI accepts a comma-separated string.
        body["parent_cids"] = [
            c.strip() for c in parent_cids.split(",") if c.strip()
        ]

    try:
        with console.status("[bold green]Uploading to ContentStore..."):
            response = httpx.post(
                f"{url}/content/upload",
                json=body,
                headers=headers or {},
                timeout=120.0,
            )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print(
            "💡 Start the server: prsm node start",
            style="yellow",
        )
        raise SystemExit(1)

    if response.status_code == 200:
        result = response.json()

        table = Table(title="Upload Result")
        table.add_column("Property",  style="cyan")
        table.add_column("Value",     style="green")
        table.add_row("CID",          result.get("cid", "?"))
        table.add_row("Filename",     result.get("filename", "?"))
        table.add_row("Size",         f"{result.get('size_bytes', 0):,} bytes")
        table.add_row("Royalty Rate", f"{result.get('royalty_rate', 0):.4f} FTNS/access")
        console.print(table)

        if result.get("parent_cids"):
            console.print(
                f"\n[dim]Derivative of: {', '.join(result['parent_cids'])}[/dim]"
            )

        # Emit CID on its own line — easy to capture in scripts.
        console.print(f"\n[bold]CID: {result.get('cid')}[/bold]")

    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 503:
        # sp1027 — surface the server's actual detail (e.g. the libtorrent hint)
        # rather than a generic guess that buries the real cause.
        # sp1072 — surface the server's actionable detail AND the node-start hint
        # together (previously the detail REPLACED the hint, so a "Content uploader
        # not initialized" 503 lost the "run prsm node start" remedy).
        console.print(
            "[red]FAIL[/red] /content/upload returned 503 — "
            + _server_detail_or(response, "Content uploader not available.")
            + " Is the daemon up? Run: prsm node start."
        )
        raise SystemExit(1)
    elif response.status_code == 413:
        # Sprint 333 — inline endpoint enforces
        # PRSM_MAX_UPLOAD_BYTES; >100MB MUST use publish-shard.
        console.print(
            "[red]FAIL[/red] file exceeds upload size cap. Set "
            "[bold]PRSM_MAX_UPLOAD_BYTES[/bold] in your daemon "
            "env to raise the cap (up to 100MB), or use "
            "[bold]prsm content publish-shard[/bold] for "
            "larger content.",
        )
        raise SystemExit(1)
    else:
        console.print(
            f"❌ Upload failed: HTTP {response.status_code}",
            style="red",
        )
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@storage.command()
@click.argument('cid')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def download(cid: str, output: str, api_url: str):
    """Download content from ContentStore by content ID.

    Sprint 833 — F29 fix: pre-833 this command hit
    /api/v1/storage/{cid}/download (legacy storage_api router,
    unmounted on production daemon per sprint 830). Every
    operator got a bare 404. Sprint 833 switches to inline
    /content/retrieve/{cid} which returns JSON
    {status, data: base64, filename, size_bytes} — same shape
    `prsm content fetch` (sprint 805) consumes.
    """
    import base64
    import httpx

    console.print(f"📥 Downloading {cid}...", style="bold blue")

    try:
        response = httpx.get(
            f"{api_url}/content/retrieve/{cid}", timeout=60.0,
        )
    except httpx.ConnectError:
        console.print(
            "❌ Cannot connect to PRSM server. "
            "Run [bold]prsm node start[/bold] to bring up the daemon.",
            style="red",
        )
        raise SystemExit(1)

    if response.status_code != 200:
        console.print(
            f"[red]Download failed[/red] (HTTP "
            f"{response.status_code}): "
            f"{response.text[:200]}",
        )
        raise SystemExit(1)

    data = response.json()
    status = data.get("status")
    if status != "success":
        err = data.get("error") or status or "unknown"
        console.print(
            f"[yellow]{status or 'not_found'}:[/yellow] {err}",
        )
        raise SystemExit(1)

    try:
        payload = base64.b64decode(data.get("data", ""))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]base64 decode failed[/red]: {exc}")
        raise SystemExit(1)

    if output:
        with open(output, 'wb') as f:
            f.write(payload)
        console.print(
            f"✅ Downloaded to {output} ({len(payload)} bytes, "
            f"filename={data.get('filename', '?')})",
            style="green",
        )
    else:
        try:
            console.print(payload.decode('utf-8'))
        except UnicodeDecodeError:
            console.print(
                f"Binary content ({len(payload)} bytes)",
                style="dim",
            )


@storage.command()
@click.argument('cid')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def info(cid: str, api_url: str):
    """Get information about stored content.

    Sprint 833 — F29 fix: pre-833 this command hit
    /api/v1/storage/{cid} (legacy storage_api router, inert per
    sp830). Switches to inline /content/retrieve/{cid} which
    carries metadata (filename, size_bytes, content_hash, status)
    alongside the data — we ignore the data payload + render
    metadata only.
    """
    import httpx

    try:
        response = httpx.get(
            f"{api_url}/content/retrieve/{cid}", timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(
            "❌ Cannot connect to PRSM server. "
            "Run [bold]prsm node start[/bold] to bring up the daemon.",
            style="red",
        )
        raise SystemExit(1)

    if response.status_code != 200:
        console.print(
            f"[red]Info lookup failed[/red] (HTTP "
            f"{response.status_code}): "
            f"{response.text[:200]}",
        )
        raise SystemExit(1)

    data = response.json()
    status = data.get("status")
    if status != "success":
        err = data.get("error") or status or "unknown"
        console.print(
            f"[yellow]{status or 'not_found'}:[/yellow] {err}",
        )
        raise SystemExit(1)

    table = Table(title=f"Storage Info: {cid}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("CID", cid)
    table.add_row("Filename", data.get('filename', 'N/A'))
    table.add_row("Size", f"{data.get('size_bytes', 0)} bytes")
    table.add_row(
        "Content Hash", data.get('content_hash', 'N/A')[:32] + "...",
    )
    table.add_row("Status", str(status))
    if data.get('providers_tried') is not None:
        table.add_row(
            "Providers Tried", str(data.get('providers_tried')),
        )
    console.print(table)


@storage.command()
@click.argument('cid')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def pin(cid: str, api_url: str):
    """Pin content for persistent storage.

    Sprint 834 — F29 fix: pre-834 the CLI hit
    /api/v1/storage/{cid}/pin which has NEVER existed
    (no storage_api router file). Sprint 834 adds inline
    /content/{cid}/pin (node/api.py) wired to
    StorageProvider.pin_content + switches the CLI to it.

    Replication factor option dropped — the inline endpoint
    promotes a single-node pin (GC-protected); cross-node
    replication is handled by sprint-263's replica management.
    """
    import httpx

    console.print(f"📌 Pinning {cid}...", style="bold blue")

    try:
        response = httpx.post(
            f"{api_url}/content/{cid}/pin", timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(
            "❌ Cannot connect to PRSM server. "
            "Run [bold]prsm node start[/bold].",
            style="red",
        )
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print(f"✅ Content pinned!", style="bold green")
        console.print(f"   CID: {data.get('cid')}")
        console.print(f"   Size: {data.get('size_bytes', 0)} bytes")
        return
    if response.status_code == 404:
        console.print(
            f"[yellow]CID not present locally.[/yellow] Upload "
            f"or retrieve [bold]{cid}[/bold] before pinning.",
        )
        raise SystemExit(1)
    if response.status_code == 503:
        console.print(
            "[red]FAIL[/red] Storage provider not initialized. "
            "Run [bold]prsm node start[/bold].",
        )
        raise SystemExit(1)
    console.print(
        f"❌ Pinning failed: HTTP {response.status_code} "
        f"{response.text[:200]}",
        style="red",
    )
    raise SystemExit(1)


@storage.command()
@click.option('--limit', default=20, type=int, help='Maximum results')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def pins(limit: int, api_url: str):
    """List pinned content.

    Sprint 834 — F29 fix: switches from phantom
    /api/v1/storage/pins to inline /storage/pinned-stats
    (existing surface; sprint 263). Response shape changed: the
    inline endpoint returns {pinned: [...], count} where each
    entry has cid/size_bytes/pinned_at/successful_challenges/
    failed_challenges rather than the legacy
    {replication, monthly_cost} columns.
    """
    import httpx

    try:
        response = httpx.get(
            f"{api_url}/storage/pinned-stats", timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(
            "❌ Cannot connect to PRSM server. "
            "Run [bold]prsm node start[/bold].",
            style="red",
        )
        raise SystemExit(1)

    if response.status_code == 503:
        console.print(
            "[red]FAIL[/red] Storage provider not initialized.",
            style="red",
        )
        raise SystemExit(1)
    if response.status_code != 200:
        console.print(
            f"❌ Failed to list pins: HTTP {response.status_code}",
            style="red",
        )
        raise SystemExit(1)

    data = response.json()
    entries = data.get('pinned', [])[:limit]
    if not entries:
        console.print("No pinned content.", style="dim")
        return

    table = Table(title="Pinned Content")
    table.add_column("CID", style="cyan")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Pinned At", style="magenta")
    table.add_column("Challenges ✓/✗", style="blue", justify="right")

    for p in entries:
        cid = p.get('cid', 'N/A')
        cid_display = cid if len(cid) <= 24 else cid[:20] + "..."
        table.add_row(
            cid_display,
            f"{p.get('size_bytes', 0)} bytes",
            str(p.get('pinned_at', '?')),
            f"{p.get('successful_challenges', 0)} / "
            f"{p.get('failed_challenges', 0)}",
        )
    console.print(table)


# ============================================================================
# GOVERNANCE COMMANDS
# ============================================================================

@main.group()
def governance():
    """Governance and voting commands"""
    pass


@governance.command()
@click.option('--limit', default=10, type=int, help='Maximum proposals to show')
@click.option('--status', type=click.Choice(['draft', 'active', 'voting', 'passed', 'rejected', 'executed']), help='Filter by status')
@click.option('--api-url', default=None, help='PRSM API URL')
def proposals(limit: int, status: str, api_url: str):
    """List governance proposals."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    params = {"limit": limit}
    if status:
        params["status_filter"] = status
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/governance/proposals",
            params=params,
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            proposals = outer.get("data", {}).get("proposals", [])
            
            if not proposals:
                console.print("No proposals found.", style="dim")
                return
            
            table = Table(title="Governance Proposals")
            table.add_column("ID", style="dim")
            table.add_column("Title", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Type", style="blue")
            table.add_column("Votes", style="green")
            
            for prop in proposals:
                votes = f"✓{prop.get('votes_for', 0):.0f} ✗{prop.get('votes_against', 0):.0f}"
                table.add_row(
                    prop.get('proposal_id', 'N/A')[:12] + "...",
                    prop.get('title', 'N/A')[:30],
                    prop.get('status', 'N/A'),
                    prop.get('proposal_type', 'N/A'),
                    votes
                )
            console.print(table)
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        elif response.status_code == 404:
            # Sprint 533 F49 fix: surface that the governance route
            # isn't wired on this daemon. Operators wanting governance
            # today use /admin/upgrade/* (sprint-394-era surface).
            console.print(
                "[yellow]⚠️  Governance endpoint not wired on this daemon.[/yellow]\n"
                "Current governance surface lives at `/admin/upgrade/*`. Use:\n"
                "  • [cyan]curl /admin/upgrade/{proposal_id}[/cyan]  — get proposal\n"
                "  • [cyan]curl /admin/upgrade/{id}/compose-upgrade[/cyan]\n"
                "  • [cyan]curl /admin/upgrade/{id}/compose-rollback[/cyan]\n"
                "The `prsm governance ...` CLI commands wrap a future "
                "proposal-list endpoint that isn't deployed in this build."
            )
        else:
            console.print(f"❌ Failed to list proposals: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.argument('proposal-id')
@click.option('--api-url', default=None, help='PRSM API URL')
def proposal(proposal_id: str, api_url: str):
    """Get details of a specific proposal."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/governance/proposals/{proposal_id}",
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("proposal", {})
            results = outer.get("data", {}).get("results")
            
            console.print(f"\n📋 {data.get('title')}", style="bold cyan")
            console.print(f"   ID: {data.get('proposal_id')}")
            console.print(f"   Status: {data.get('status')}")
            console.print(f"   Type: {data.get('proposal_type')}")
            console.print(f"   Proposer: {data.get('proposer_id', 'N/A')[:16]}...")
            console.print()
            console.print("Description:", style="bold")
            console.print(data.get('description', 'N/A'))
            console.print()
            
            table = Table(title="Voting Results")
            table.add_column("Choice", style="cyan")
            table.add_column("Votes (FTNS)", style="green")
            table.add_row("For", f"{data.get('votes_for', 0):.2f}")
            table.add_row("Against", f"{data.get('votes_against', 0):.2f}")
            console.print(table)
            
            if results:
                console.print(f"\n   Results:", style="bold")
                console.print(f"   - Passed: {results.get('passed', False)}")
                console.print(f"   - Quorum met: {results.get('quorum_met', False)}")
            
            console.print(f"\n   Quorum required: {data.get('quorum', 0) * 100:.1f}%")
            console.print(f"   Threshold: {data.get('threshold', 0) * 100:.1f}%")
            console.print(f"   Voting ends: {data.get('voting_ends', 'N/A')}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        elif response.status_code == 404:
            console.print(f"❌ Proposal not found: {proposal_id}", style="red")
        else:
            console.print(f"❌ Failed to get proposal: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.argument('proposal-id')
@click.option('--choice', required=True, type=click.Choice(['for', 'against']), help='Vote choice')
@click.option('--rationale', help='Rationale for your vote')
@click.option('--api-url', default=None, help='PRSM API URL')
def vote(proposal_id: str, choice: str, rationale: str, api_url: str):
    """Cast a vote on a proposal."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    # Convert choice to boolean vote_choice
    vote_choice = (choice == "for")
    
    console.print(f"🗳️  Casting vote '{choice}' on proposal {proposal_id}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/governance/vote",
            json={
                "proposal_id": proposal_id,
                "vote_choice": vote_choice,
                "rationale": rationale
            },
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("vote", {})
            console.print(f"✅ Vote cast successfully!", style="bold green")
            console.print(f"   Proposal ID: {data.get('proposal_id')}")
            console.print(f"   Vote choice: {'For' if data.get('vote_choice') else 'Against'}")
            console.print(f"   Cast at: {data.get('cast_at', 'N/A')}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Failed to cast vote: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.option('--title', required=True, help='Proposal title (min 10 characters)')
@click.option('--description', required=True, help='Proposal description (min 100 characters)')
@click.option('--type', 'proposal_type', required=True,
              type=click.Choice(['safety', 'economic', 'technical', 'governance',
                                'parameter_change', 'constitutional', 'emergency',
                                'operational', 'community']),
              help='Type of proposal')
@click.option('--api-url', default=None, help='PRSM API URL')
def create_proposal(title: str, description: str, proposal_type: str, api_url: str):
    """Create a new governance proposal."""
    import httpx
    
    # Client-side validation
    if len(title) < 10:
        console.print("❌ Title must be at least 10 characters", style="red")
        return
    
    if len(description) < 100:
        console.print("❌ Description must be at least 100 characters", style="red")
        return
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    console.print(f"📝 Creating proposal: {title}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/governance/proposals",
            json={
                "title": title,
                "description": description,
                "proposal_type": proposal_type
            },
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("proposal", {})
            console.print(f"✅ Proposal created!", style="bold green")
            console.print(f"   Proposal ID: {data.get('proposal_id')}")
            console.print(f"   Title: {data.get('title')}")
            console.print(f"   Status: {data.get('status')}")
            console.print(f"   Voting ends: {data.get('voting_ends')}")
            console.print(f"   Created at: {data.get('created_at')}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        elif response.status_code == 422:
            console.print("❌ Validation error:", style="red")
            error_data = response.json()
            console.print(f"   {error_data.get('detail', 'Unknown validation error')}")
        else:
            console.print(f"❌ Failed to create proposal: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.option('--api-url', default=None, help='PRSM API URL')
def voting_power(api_url: str):
    """Show your voting power and governance status."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/governance/status",
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("governance_status", {})
            
            if not data:
                console.print("❌ Governance not activated. Run: prsm governance activate", style="yellow")
                return
            
            console.print(f"🗳️  Your governance status:", style="bold green")
            console.print(f"   Voting power: {data.get('voting_power', 0):.2f} FTNS")
            console.print(f"   Participant tier: {data.get('participant_tier', 'N/A')}")
            console.print(f"   Active: {data.get('is_active', False)}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Failed to get governance status: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.option('--tier', required=True,
              type=click.Choice(['community', 'contributor', 'expert', 'delegate', 'council_member', 'core_team']),
              help='Participant tier for governance')
@click.option('--api-url', default=None, help='PRSM API URL')
def activate(tier: str, api_url: str):
    """Activate governance participation."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    console.print(f"🗳️  Activating governance participation as {tier}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/governance/activate",
            json={
                "participant_tier": tier,
                "auto_stake_percentage": 0.5
            },
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("activation", {})
            console.print(f"✅ Governance activated!", style="bold green")
            console.print(f"   Participant tier: {data.get('participant_tier')}")
            console.print(f"   Voting power: {data.get('voting_power', 0):.2f} FTNS")
            console.print(f"   Auto-stake: {data.get('auto_stake_percentage', 0) * 100:.0f}%")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Failed to activate governance: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")



# ============================================================================
# TORRENT COMMANDS (Phase 1)
# ============================================================================

def _fmt_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_rate(n: float) -> str:
    """Format bytes per second as human-readable rate."""
    return f"{_fmt_bytes(int(n))}/s"


@main.group()
def torrent():
    """BitTorrent P2P distribution commands"""
    pass


@torrent.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", help="Human-readable name (default: filename)")
@click.option("--piece-length", type=int, default=262144, help="Piece size in bytes")
@click.option("--provenance-id", help="Link to existing PRSM provenance record")
@click.option("--no-seed", is_flag=True, help="Create torrent file only, do not start seeding")
@click.option("--output-torrent", type=click.Path(), help="Save .torrent file to disk")
@click.option("--api-url", default=None, help="PRSM API URL")
def create(path: str, name: Optional[str], piece_length: int, provenance_id: Optional[str],
           no_seed: bool, output_torrent: Optional[str], api_url: str):
    """Create a new torrent from a local file or directory and begin seeding."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    path_obj = Path(path)

    console.print(f"🔗 Creating torrent from {path_obj.name}...", style="bold blue")

    payload = {
        "content_path": str(path_obj.absolute()),
        "piece_length": piece_length,
    }
    if name:
        payload["name"] = name
    if provenance_id:
        payload["provenance_id"] = provenance_id

    try:
        response = httpx.post(
            f"{url}/api/v1/torrents/create",
            json=payload,
            headers=headers,
            timeout=60.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        infohash = data.get("infohash", "")

        from rich.panel import Panel
        content = f"""Infohash: {infohash}
Name:     {data.get('name', 'N/A')}
Size:     {_fmt_bytes(data.get('size_bytes', 0))}
Pieces:   {data.get('num_pieces', 0)} × {_fmt_bytes(data.get('piece_length', piece_length))}
Magnet:   magnet:?xt=urn:btih:{infohash}&dn={data.get('name', '')}
Status:   {'Created only' if no_seed else 'Seeding ✅'}"""

        console.print(Panel(content, title="🔗 Torrent Created", border_style="green"))

        if output_torrent:
            console.print(f"   Torrent file: {output_torrent}", style="dim")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 503:
        console.print("⚠️  BitTorrent service unavailable", style="yellow")
        raise SystemExit(1)
    else:
        console.print(f"❌ Create failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@torrent.command()
@click.argument("source")
@click.option("--save-path", type=click.Path(), help="Where to save downloaded files")
@click.option("--seed-mode", is_flag=True, help="Skip downloading — assume data already present")
@click.option("--download", is_flag=True, help="Begin downloading immediately")
@click.option("--api-url", default=None, help="PRSM API URL")
def add(source: str, save_path: Optional[str], seed_mode: bool, download: bool, api_url: str):
    """Add an existing torrent from magnet URI or .torrent file path."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    console.print("🔗 Adding torrent...", style="bold blue")

    payload = {
        "source": source,
        "seed_mode": seed_mode,
    }
    if save_path:
        payload["save_path"] = save_path

    try:
        response = httpx.post(
            f"{url}/api/v1/torrents/add",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        infohash = data.get("infohash", "")[:16]

        console.print(f"✅ Torrent added: {infohash}...", style="bold green")
        console.print(f"   Name:   {data.get('name', 'N/A')}")
        console.print(f"   Status: {data.get('state', 'Added')}")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 503:
        console.print("⚠️  BitTorrent service unavailable", style="yellow")
        raise SystemExit(1)
    else:
        console.print(f"❌ Add failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@torrent.command("list")
@click.option("--seeding", is_flag=True, help="Show only torrents this node is seeding")
@click.option("--available", is_flag=True, help="Show only network-announced torrents")
@click.option("--limit", type=int, default=50, help="Max results")
@click.option("--api-url", default=None, help="PRSM API URL")
def list_torrents(seeding: bool, available: bool, limit: int, api_url: str):
    """List all torrents known to this node."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/torrents",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        torrents = data.get("torrents", [])

        if not torrents:
            console.print("No torrents found.", style="yellow")
            return

        table = Table(title="Torrents")
        table.add_column("Infohash", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Size", style="blue")
        table.add_column("State", style="magenta")
        table.add_column("Progress", style="yellow")
        table.add_column("Peers", style="dim")
        table.add_column("↑ Rate", style="red")
        table.add_column("↓ Rate", style="blue")

        for t in torrents[:limit]:
            infohash = t.get("infohash", "")
            short_hash = infohash[:16] + "..." if len(infohash) > 16 else infohash
            progress = t.get("progress", 0) * 100

            table.add_row(
                short_hash,
                t.get("name", "N/A")[:30],
                _fmt_bytes(t.get("size_bytes", 0)),
                t.get("state", "?"),
                f"{progress:.1f}%",
                str(t.get("seeders", 0) + t.get("leechers", 0)),
                _fmt_rate(t.get("upload_rate", 0)),
                _fmt_rate(t.get("download_rate", 0)),
            )

        console.print(table)
        console.print(f"\n📊 Total: {data.get('total', len(torrents))} torrents", style="blue")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ List failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--api-url", default=None, help="PRSM API URL")
def status(infohash: str, api_url: str):
    """Show detailed live status for a specific torrent."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/torrents/{infohash}",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        progress = data.get("progress", 0) * 100

        from rich.panel import Panel

        # Build progress bar
        progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))

        eta_str = ""
        eta = data.get("eta_seconds", 0)
        if eta > 0:
            mins, secs = divmod(int(eta), 60)
            eta_str = f"  (ETA: {mins}m {secs}s)" if mins else f"  (ETA: {secs}s)"

        content = f"""Infohash:     {data.get('infohash', 'N/A')}
Name:         {data.get('name', 'N/A')}
Size:         {_fmt_bytes(data.get('size_bytes', 0))}
State:        {data.get('state', 'Unknown')}
Progress:     [{progress_bar}] {progress:.1f}%{eta_str}
Download:     {_fmt_rate(data.get('download_rate', 0))}
Upload:       {_fmt_rate(data.get('upload_rate', 0))}
Peers:        {data.get('seeders', 0) + data.get('leechers', 0)} connected
Uploaded:     {_fmt_bytes(data.get('bytes_uploaded', 0))}
Downloaded:   {_fmt_bytes(data.get('bytes_downloaded', 0))}
Magnet URI:   magnet:?xt=urn:btih:{data.get('infohash', '')}"""

        if data.get("error"):
            content += f"\nError:        ⚠️ {data.get('error')}"

        console.print(Panel(content, title="🔗 Torrent Status", border_style="blue"))
    elif response.status_code == 404:
        console.print(f"❌ Torrent not found: {infohash}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Status failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--api-url", default=None, help="PRSM API URL")
def seed(infohash: str, api_url: str):
    """Start seeding a torrent."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.post(
            f"{url}/api/v1/torrents/{infohash}/seed",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        console.print(f"✅ Now seeding: {infohash[:16]}...", style="bold green")
        console.print("   Announced to PRSM network.", style="dim")
    elif response.status_code == 501:
        console.print("⚠️  Seeding control not yet implemented", style="yellow")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Seed failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--delete-files", is_flag=True, help="Also delete local data files")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("--api-url", default=None, help="PRSM API URL")
def unseed(infohash: str, delete_files: bool, yes: bool, api_url: str):
    """Stop seeding a torrent."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    if delete_files and not yes:
        if not click.confirm("This will delete local files. Continue?", default=False):
            console.print("Cancelled.", style="yellow")
            raise SystemExit(0)

    try:
        response = httpx.delete(
            f"{url}/api/v1/torrents/{infohash}/seed",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        console.print(f"✅ Stopped seeding: {infohash[:16]}...", style="bold green")
        if delete_files:
            console.print("   Files deleted.", style="dim")
        else:
            console.print("   ⚠️  Files retained at download location", style="yellow")
    elif response.status_code == 400:
        console.print("⚠️  Could not stop seeding (not found or minimum seed time not met)", style="yellow")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Unseed failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--save-path", type=click.Path(), help="Destination directory")
@click.option("--timeout", type=float, default=3600.0, help="Max seconds to wait")
@click.option("--no-wait", is_flag=True, help="Fire-and-forget (return request_id immediately)")
@click.option("--api-url", default=None, help="PRSM API URL")
def download(infohash: str, save_path: Optional[str], timeout: float, no_wait: bool, api_url: str):
    """Download a torrent's content."""
    import httpx
    import time

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    payload = {
        "infohash": infohash,
        "save_path": save_path or str(Path.home() / ".prsm" / "torrents"),
        "timeout": timeout,
    }

    console.print("🔗 Starting download...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/torrents/{infohash}/download",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        request_id = data.get("request_id", "")

        if no_wait or data.get("status") == "completed":
            console.print(f"✅ Download request: {request_id}", style="bold green")
            if data.get("status") == "completed":
                console.print(f"   Downloaded: {_fmt_bytes(data.get('bytes_downloaded', 0))}")
            return

        # Poll for progress
        console.print(f"   Request ID: {request_id}")
        console.print("   Polling progress...", style="dim")

        start_time = time.time()
        with console.status("[bold green]Downloading...") as status:
            while time.time() - start_time < timeout:
                poll = httpx.get(
                    f"{url}/api/v1/torrents/{infohash}/download/{request_id}",
                    headers=headers,
                    timeout=10.0,
                )
                if poll.status_code == 200:
                    poll_data = poll.json()
                    progress = poll_data.get("progress", 0) * 100
                    status.update(f"[bold green]Downloading... {progress:.1f}%")

                    if poll_data.get("status") == "completed":
                        console.print(f"\n✅ Download complete!", style="bold green")
                        console.print(f"   Downloaded: {_fmt_bytes(poll_data.get('bytes_downloaded', 0))}")
                        return
                    elif poll_data.get("status") == "failed":
                        console.print(f"\n❌ Download failed: {poll_data.get('error', 'Unknown error')}", style="red")
                        raise SystemExit(1)

                time.sleep(2)

        console.print("\n⚠️  Download timed out", style="yellow")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 503:
        console.print("⚠️  BitTorrent requester not available", style="yellow")
        raise SystemExit(1)
    else:
        console.print(f"❌ Download failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--api-url", default=None, help="PRSM API URL")
def peers(infohash: str, api_url: str):
    """List peers currently connected for a torrent."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/torrents/{infohash}/peers",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        peers_data = response.json()

        if not peers_data:
            console.print("No peers connected.", style="yellow")
            return

        table = Table(title="Connected Peers")
        table.add_column("IP:Port", style="cyan")
        table.add_column("Client", style="green")
        table.add_column("Downloaded", style="blue")
        table.add_column("Uploaded", style="magenta")
        table.add_column("Seed?", style="yellow")

        for p in peers_data:
            table.add_row(
                f"{p.get('ip', '?')}:{p.get('port', '?')}",
                p.get("client", "?")[:15],
                _fmt_bytes(p.get("downloaded", 0)),
                _fmt_bytes(p.get("uploaded", 0)),
                "✅" if p.get("is_seed") else "",
            )

        console.print(table)
        console.print(f"\n📊 Total: {len(peers_data)} peers", style="blue")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Peers failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@torrent.command()
@click.option("--api-url", default=None, help="PRSM API URL")
def stats(api_url: str):
    """Show aggregate BitTorrent stats for this node."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/torrents/stats",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()

        if not data.get("available"):
            console.print("⚠️  BitTorrent client not available", style="yellow")
            return

        from rich.panel import Panel

        provider = data.get("provider", {})
        requester = data.get("requester", {})

        content = f"""Active torrents:   {provider.get('active_torrents', 0)} seeding, {requester.get('active_downloads', 0)} downloading
Total uploaded:   {_fmt_bytes(provider.get('total_uploaded_bytes', 0))}
Total downloaded: {_fmt_bytes(requester.get('total_downloaded_bytes', 0))}
Upload rate:       {_fmt_rate(provider.get('upload_rate', 0))}
Download rate:     {_fmt_rate(requester.get('download_rate', 0))}
FTNS earned:       {provider.get('total_rewards', 0):.4f}"""

        console.print(Panel(content, title="🔗 BitTorrent Stats", border_style="green"))
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Stats failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


# ============================================================================

# ============================================================================
# MONITORING COMMANDS (Phase 6)
# ============================================================================

@main.group()
def monitor():
    """System monitoring commands"""
    pass


@monitor.command()
@click.option("--watch", is_flag=True, help="Refresh every 5 seconds")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON for scripting")
@click.option("--api-url", default=None, help="PRSM API URL")
def health_cmd(watch: bool, json_output: bool, api_url: str):
    """Full health check across all PRSM subsystems."""
    import httpx
    import time

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    def fetch_health():
        try:
            return httpx.get(
                f"{url}/api/v1/health",
                headers=headers,
                timeout=10.0,
            )
        except httpx.ConnectError:
            return None

    def display_health(response):
        if response is None:
            console.print("❌ Cannot connect to API", style="red")
            return

        if response.status_code == 200:
            data = response.json()

            if json_output:
                import json
                console.print(json.dumps(data, indent=2))
                return

            from rich.panel import Panel

            status_icons = {
                "healthy": "✅",
                "degraded": "⚠️ ",
                "unhealthy": "❌",
            }

            services = data.get("services", {})
            lines = []
            for name, info in services.items():
                status = info.get("status", "unknown")
                icon = status_icons.get(status, "?")
                detail = info.get("detail", "")
                line = f"{name:15s}: {icon} {status.capitalize()}"
                if detail:
                    line += f"  ({detail})"
                lines.append(line)

            overall = data.get("overall", "unknown")
            overall_icon = status_icons.get(overall, "?")

            content = "\n".join(lines) + f"""
──────────────────────────────────────
Pipeline overall:  {overall_icon} {overall.capitalize()}"""

            console.print(Panel(content, title="🏥 System Health", border_style="green"))
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Health check failed: HTTP {response.status_code}", style="red")

    if watch:
        try:
            while True:
                console.clear()
                display_health(fetch_health())
                console.print("\n[dim]Press Ctrl+C to exit[/dim]")
                time.sleep(5)
        except KeyboardInterrupt:
            console.print("\n👋 Monitoring stopped", style="yellow")
    else:
        display_health(fetch_health())


@monitor.command()
@click.option("--period", type=click.Choice(["1h", "6h", "24h", "7d"]), default="1h", help="Time window")
@click.option("--watch", is_flag=True, help="Auto-refresh every 10 seconds")
@click.option("--api-url", default=None, help="PRSM API URL")
def metrics(period: str, watch: bool, api_url: str):
    """Show key performance metrics for the local node."""
    import httpx
    import time

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    def fetch_metrics():
        try:
            return httpx.get(
                f"{url}/api/v1/monitoring/metrics",
                params={"period": period},
                headers=headers,
                timeout=10.0,
            )
        except httpx.ConnectError:
            return None

    def display_metrics(response):
        if response is None:
            console.print("❌ Cannot connect to API", style="red")
            return

        if response.status_code == 200:
            data = response.json()

            from rich.panel import Panel

            compute = data.get("compute", {})
            network = data.get("network", {})
            economy = data.get("economy", {})

            content = f"""Period: Last {period}
────────── Compute ──────────
Queries served:      {compute.get('queries_served', 0)}
Avg query latency:   {compute.get('avg_latency_ms', 0)}ms
CPU usage:           {compute.get('cpu_percent', 0):.1f}%
Memory usage:        {compute.get('memory_used_gb', 0):.1f} GB / {compute.get('memory_total_gb', 0):.1f} GB
────────── Network ──────────
P2P bandwidth ↑:     {_fmt_bytes(network.get('p2p_upload_bytes', 0))}
P2P bandwidth ↓:     {_fmt_bytes(network.get('p2p_download_bytes', 0))}
BT seeded:           {_fmt_bytes(network.get('bt_seeded_bytes', 0))}
BT downloaded:       {_fmt_bytes(network.get('bt_downloaded_bytes', 0))}
────────── Economy ──────────
FTNS earned:         +{economy.get('ftns_earned', 0):.4f}
FTNS spent:           -{economy.get('ftns_spent', 0):.4f}"""

            console.print(Panel(content, title="📊 Node Metrics", border_style="blue"))
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Metrics failed: HTTP {response.status_code}", style="red")

    if watch:
        try:
            while True:
                console.clear()
                display_metrics(fetch_metrics())
                console.print("\n[dim]Press Ctrl+C to exit[/dim]")
                time.sleep(10)
        except KeyboardInterrupt:
            console.print("\n👋 Monitoring stopped", style="yellow")
    else:
        display_metrics(fetch_metrics())


@monitor.command()
@click.option("--level", type=click.Choice(["debug", "info", "warning", "error"]), default="info",
              help="Log level filter")
@click.option("--service", help="Filter by service: api, nwtn, storage, bittorrent, node")
@click.option("--limit", type=int, default=50, help="Lines to show")
@click.option("--follow", "-f", is_flag=True, help="Stream new log entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def logs(level: str, service: Optional[str], limit: int, follow: bool, api_url: str):
    """Stream or retrieve recent structured log output."""
    import httpx
    import time

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    params = {"level": level, "limit": limit}
    if service:
        params["service"] = service

    def fetch_logs():
        try:
            return httpx.get(
                f"{url}/api/v1/monitoring/logs",
                params=params,
                headers=headers,
                timeout=10.0,
            )
        except httpx.ConnectError:
            return None

    def display_logs(response):
        if response is None:
            console.print("❌ Cannot connect to API", style="red")
            return

        if response.status_code == 200:
            data = response.json()
            logs_list = data.get("logs", [])

            for log in logs_list:
                timestamp = log.get("timestamp", "")[:19]
                log_level = log.get("level", "info").upper()
                log_service = log.get("service", "?")
                message = log.get("message", "")

                level_colors = {
                    "DEBUG": "dim",
                    "INFO": "blue",
                    "WARNING": "yellow",
                    "ERROR": "red",
                }
                level_color = level_colors.get(log_level, "white")

                console.print(f"[dim]{timestamp}[/dim] [{level_color}]{log_level:8s}[/{level_color}] [{log_service}] {message}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Logs failed: HTTP {response.status_code}", style="red")

    if follow:
        try:
            while True:
                console.clear()
                display_logs(fetch_logs())
                console.print("\n[dim]Press Ctrl+C to exit[/dim]")
                time.sleep(2)
        except KeyboardInterrupt:
            console.print("\n👋 Log streaming stopped", style="yellow")
    else:
        display_logs(fetch_logs())


# ============================================================================
# WORKFLOW COMMANDS (Phase 7)
# ============================================================================

@main.group()
def workflow():
    """Workflow scheduling commands"""
    pass


@workflow.command()
@click.option("--name", required=True, help="Workflow name")
@click.option("--description", help="Description")
@click.option("--trigger", required=True, help="Cron expression or 'manual'")
@click.option("--steps-file", type=click.Path(exists=True), help="JSON file describing workflow steps")
@click.option("--budget", type=float, help="Max FTNS per run")
@click.option("--api-url", default=None, help="PRSM API URL")
def create_cmd(name: str, description: Optional[str], trigger: str, steps_file: Optional[str],
               budget: Optional[float], api_url: str):
    """Create a new automated workflow."""
    import httpx
    import json

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    payload = {
        "name": name,
        "trigger": trigger,
    }
    if description:
        payload["description"] = description
    if budget:
        payload["max_budget"] = budget

    if steps_file:
        steps_path = Path(steps_file)
        try:
            steps = json.loads(steps_path.read_text())
            payload["steps"] = steps
        except json.JSONDecodeError:
            console.print("❌ Invalid JSON in steps file", style="red")
            raise SystemExit(1)

    console.print(f"📋 Creating workflow '{name}'...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/workflows",
            json=payload,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print(f"✅ Workflow created: {data.get('workflow_id', 'N/A')}", style="bold green")
        if trigger != "manual":
            console.print(f"   Next run: {data.get('next_run', 'N/A')}", style="dim")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Create failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@workflow.command("list")
@click.option("--status", "status_filter", help="Filter: active, paused, failed")
@click.option("--limit", type=int, default=20, help="Number of entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def list_workflows(status_filter: Optional[str], limit: int, api_url: str):
    """List all workflows for the current user."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    params = {"limit": limit}
    if status_filter:
        params["status"] = status_filter

    try:
        response = httpx.get(
            f"{url}/api/v1/workflows",
            params=params,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        workflows = data.get("workflows", [])

        if not workflows:
            console.print("No workflows found.", style="yellow")
            return

        table = Table(title="📋 Workflows")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Trigger", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Last Run", style="dim")
        table.add_column("Next Run", style="blue")
        table.add_column("Runs", style="dim", justify="right")

        for w in workflows:
            workflow_id = w.get("workflow_id", "?")[:12]
            table.add_row(
                workflow_id,
                w.get("name", "?")[:20],
                w.get("trigger", "?"),
                w.get("status", "?"),
                w.get("last_run", "Never")[:10] if w.get("last_run") else "Never",
                w.get("next_run", "N/A")[:10] if w.get("next_run") else "N/A",
                str(w.get("run_count", 0)),
            )

        console.print(table)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ List failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@workflow.command()
@click.argument("workflow-id")
@click.option("--follow", "-f", is_flag=True, help="Stream execution logs until completion")
@click.option("--api-url", default=None, help="PRSM API URL")
def run(workflow_id: str, follow: bool, api_url: str):
    """Manually trigger a workflow run."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    console.print(f"▶️  Starting workflow {workflow_id[:12]}...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/workflows/{workflow_id}/run",
            headers=headers,
            timeout=30.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print(f"✅ Workflow run started: {data.get('run_id', 'N/A')}", style="bold green")
        console.print(f"   Status: {data.get('status', 'running')}")
    elif response.status_code == 404:
        console.print(f"❌ Workflow not found: {workflow_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Run failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@workflow.command()
@click.argument("workflow-id")
@click.option("--run-id", help="Specific run (default: most recent)")
@click.option("--limit", type=int, default=100, help="Log lines")
@click.option("--api-url", default=None, help="PRSM API URL")
def logs_cmd(workflow_id: str, run_id: Optional[str], limit: int, api_url: str):
    """Show execution logs for a workflow."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    # If no run_id, get the most recent run
    if not run_id:
        run_id = "latest"

    params = {"limit": limit}

    try:
        response = httpx.get(
            f"{url}/api/v1/workflows/{workflow_id}/runs/{run_id}/logs",
            params=params,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        logs_list = data.get("logs", [])

        if not logs_list:
            console.print("No logs found.", style="yellow")
            return

        console.print(f"📋 Workflow {workflow_id[:12]} - Run {run_id[:12] if run_id != 'latest' else 'latest'}",
                     style="bold blue")

        for log in logs_list:
            timestamp = log.get("timestamp", "")[:19]
            level = log.get("level", "info").upper()
            message = log.get("message", "")
            step = log.get("step", "")

            level_colors = {
                "DEBUG": "dim",
                "INFO": "blue",
                "WARNING": "yellow",
                "ERROR": "red",
            }
            level_color = level_colors.get(level, "white")

            step_str = f"[{step}] " if step else ""
            console.print(f"[dim]{timestamp}[/dim] [{level_color}]{level:8s}[/{level_color}] {step_str}{message}")
    elif response.status_code == 404:
        console.print(f"❌ Workflow or run not found: {workflow_id}/{run_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Logs failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)



@main.group()
def mcp():
    """MCP server for AI agent integration."""
    pass


@mcp.command("start")
@click.option("--host", default="localhost", show_default=True, help="Host to bind the MCP server")
@click.option("--port", default=9100, show_default=True, help="Port for the MCP server")
def mcp_start(host: str, port: int):
    """Start the PRSM MCP server for AI agent integration.

    Exposes all installed PRSM skill tools as MCP-compatible endpoints
    that AI agents (Hermes, OpenClaw, Claude Desktop) can discover and invoke.
    """
    from prsm.cli_modules.mcp_server import is_fastmcp_available, start_mcp_server

    if not is_fastmcp_available():
        # Sprint 538 F70 fix: point operators to the PRSM-canonical
        # install path (`.[mcp]` extra) instead of generic
        # `pip install fastmcp`. The `[mcp]` extra also pulls the
        # official `mcp>=1.27.0` SDK that fastmcp wraps + ensures
        # version compat. Generic install can mismatch.
        # F70b: escape `[mcp]` for Rich (square brackets = markup).
        console.print(
            "✗ fastmcp is required to run the MCP server.\n"
            "  Install via PRSM extra (recommended — pins compatible\n"
            "  versions of mcp + fastmcp):\n"
            "    pip install -e '.\\[mcp]'\n"
            "  Or standalone (may not match PRSM's pinned versions):\n"
            "    pip install 'fastmcp>=2.0.0' 'mcp>=1.27.0'",
            style="red",
        )
        raise SystemExit(1)

    console.print(f"◇ Starting PRSM MCP server on {host}:{port}...", style="bold")
    console.print(f"  Endpoint: http://{host}:{port}/mcp", style="dim")
    console.print("  Press Ctrl+C to stop.\n", style="dim")
    start_mcp_server(host=host, port=port)


@mcp.command("config-snippet")
@click.option("--host", default="localhost", show_default=True, help="Host for the snippet")
@click.option("--port", default=9100, show_default=True, help="Port for the snippet")
def mcp_config_snippet(host: str, port: int):
    """Print a YAML config snippet for Hermes/OpenClaw MCP client setup.

    Add the printed snippet to your AI client's configuration file
    to connect it to the PRSM MCP server.
    """
    from prsm.cli_modules.mcp_server import get_config_snippet

    console.print("Add this to your Hermes or OpenClaw config:\n", style="dim")
    snippet = get_config_snippet(host=host, port=port)
    console.print(snippet)


@mcp.command("status")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def mcp_status_cmd(output_format: str):
    """Show MCP server status.

    Reports whether the MCP server is running based on config AND
    actual port-bind state.
    """
    from prsm.cli_modules.config_schema import PRSMConfig

    cfg = PRSMConfig.load()
    enabled = cfg.mcp_server_enabled
    port = cfg.mcp_server_port

    # Sprint 534 F60 fix: probe the actual port instead of trusting
    # config-says-enabled. Pre-fix: `prsm mcp status` reported
    # "enabled (:9100)" while the port was closed — misleading
    # for operators trying to integrate Claude Desktop / Gemini CLI.
    running = False
    if enabled:
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                running = s.connect_ex(("127.0.0.1", port)) == 0
        except Exception:
            running = False

    data = {
        "ok": True,
        "mcp_enabled": enabled,
        "mcp_running": running,
        "mcp_port": port,
        "config_path": str(PRSMConfig.config_path()),
    }

    if output_format == "json":
        _agent_output(data)
        return

    if enabled and running:
        console.print(
            f"\n  {ICONS['success']} MCP Server: running (:{port})",
        )
        console.print(
            f"  Config:  {PRSMConfig.config_path()}",
            style=THEME.dim,
        )
    elif enabled and not running:
        console.print(
            f"\n  ⚠️  MCP Server: enabled in config but NOT listening on :{port}",
            style="yellow",
        )
        console.print(
            f"  Start:   prsm mcp start", style=THEME.dim,
        )
        console.print(
            f"  Config:  {PRSMConfig.config_path()}",
            style=THEME.dim,
        )
    else:
        console.print(f"\n  {ICONS['info']} MCP Server: disabled")
        console.print(
            f"  Enable:  prsm config set mcp_server_enabled true",
            style=THEME.dim,
        )


# ---------------------------------------------------------------------------
# config CLI group (Phase 2)
# ---------------------------------------------------------------------------

@main.group()
def config():
    """Configuration management commands."""
    pass


@config.command("show")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format: 'text' (human) or 'json' (agent-parseable)")
def config_show(output_format: str):
    """Display current PRSM configuration.

    Shows all settings from ~/.prsm/config.yaml grouped by category.
    Use --format json for machine-readable output.
    """
    from prsm.cli_modules.config_schema import PRSMConfig

    cfg = PRSMConfig.load()

    data = {
        "ok": True,
        "config_path": str(PRSMConfig.config_path()),
        "config": cfg.model_dump(),
    }

    if output_format == "json":
        _agent_output(data)
        return

    # Rich table output
    console.print(f"\n[bold]PRSM Configuration[/bold]", style="cyan")
    console.print(f"[dim]Source: {PRSMConfig.config_path()}[/dim]\n")

    # Node Identity
    table = Table(title="Node Identity", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("display_name", cfg.display_name)
    table.add_row("node_role", cfg.node_role.value)
    console.print(table)

    # Resources
    table = Table(title="Resources", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("cpu_pct", f"{cfg.cpu_pct}%")
    table.add_row("memory_pct", f"{cfg.memory_pct}%")
    table.add_row("gpu_pct", f"{cfg.gpu_pct}%")
    table.add_row("storage_gb", f"{cfg.storage_gb} GB")
    table.add_row("max_concurrent_jobs", str(cfg.max_concurrent_jobs))
    if cfg.upload_mbps_limit > 0:
        table.add_row("upload_mbps_limit", f"{cfg.upload_mbps_limit} Mbps")
    else:
        table.add_row("upload_mbps_limit", "unlimited")
    if cfg.active_hours_start is not None and cfg.active_hours_end is not None:
        table.add_row("active_hours", f"{cfg.active_hours_start:02d}:00 - {cfg.active_hours_end:02d}:00")
    if cfg.active_days:
        table.add_row("active_days", ", ".join(str(d) for d in cfg.active_days))
    console.print(table)

    # Network
    table = Table(title="Network", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("p2p_port", str(cfg.p2p_port))
    table.add_row("api_port", str(cfg.api_port))
    # Sprint 533 F52 fix: detect known-stale bootstrap_nodes
    # placeholder patterns and surface a migration hint. The
    # `/dns4/.../p2p/QmPRSM1` pattern was a pre-sprint-148 default
    # using fake PeerIDs; canonical is now `wss://bootstrap-eu.
    # prsm-network.com:8765` (sprint-385 DNS fleet).
    _stale = bool(
        cfg.bootstrap_nodes
        and any(
            "QmPRSM" in b or "bootstrap1.prsm.network" in b
            for b in cfg.bootstrap_nodes
        )
    )
    if cfg.bootstrap_nodes:
        table.add_row("bootstrap_nodes", ", ".join(cfg.bootstrap_nodes))
    else:
        table.add_row("bootstrap_nodes", "[dim]none configured[/dim]")
    console.print(table)
    if _stale:
        console.print(
            "[yellow]⚠️  bootstrap_nodes contains pre-sprint-148 "
            "placeholder values (QmPRSM1 / bootstrap1.prsm.network).[/yellow]\n"
            "[dim]Run `prsm setup --reset` or `prsm config reset` to "
            "refresh from canonical fleet (wss://bootstrap-eu/apac.prsm-network.com:8765).[/dim]"
        )

    # API Keys
    table = Table(title="API Keys", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("has_openai_key", "✓" if cfg.has_openai_key else "✗")
    table.add_row("has_anthropic_key", "✓" if cfg.has_anthropic_key else "✗")
    table.add_row("has_huggingface_token", "✓" if cfg.has_huggingface_token else "✗")
    console.print(table)

    # AI Integration
    table = Table(title="AI Integration", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("mcp_server_enabled", "✓" if cfg.mcp_server_enabled else "✗")
    table.add_row("mcp_server_port", str(cfg.mcp_server_port))
    console.print(table)

    # FTNS Wallet
    if cfg.wallet_address:
        table = Table(title="FTNS Wallet", show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("wallet_address", cfg.wallet_address)
        console.print(table)

    # Meta
    console.print(f"\n[dim]setup_completed: {cfg.setup_completed} | setup_version: {cfg.setup_version}[/dim]")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value.

    Updates ~/.prsm/config.yaml with the new value.
    Example: prsm config set cpu_pct 70
    """
    from prsm.cli_modules.config_schema import PRSMConfig, NodeRole

    cfg = PRSMConfig.load()

    # Check if key exists
    valid_fields = list(cfg.model_dump().keys())
    if key not in valid_fields:
        console.print(f"[red]Error: Unknown configuration key '{key}'[/red]")
        console.print(f"\n[dim]Valid keys:[/dim]")
        for f in sorted(valid_fields):
            console.print(f"  • {f}")
        raise SystemExit(1)

    # Get old value
    old_value = getattr(cfg, key)

    # Parse and set new value based on field type
    try:
        # Get the field type from the model
        field_info = type(cfg).model_fields.get(key)
        if field_info:
            field_type = field_info.annotation
            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is type(None) | type(str):
                # This is Optional[X], extract X
                import typing
                args = typing.get_args(field_type)
                if args:
                    field_type = args[0]

            # Convert value based on type
            if field_type == bool or (hasattr(field_type, "__origin__") and field_type.__origin__ is bool):
                # Boolean parsing
                if value.lower() in ("true", "1", "yes", "on"):
                    new_value = True
                elif value.lower() in ("false", "0", "no", "off"):
                    new_value = False
                else:
                    raise ValueError(f"Cannot convert '{value}' to boolean")
            elif field_type == int or (isinstance(field_type, type) and issubclass(field_type, int)):
                new_value = int(value)
            elif field_type == float or (isinstance(field_type, type) and issubclass(field_type, float)):
                new_value = float(value)
            elif field_type == NodeRole:
                new_value = NodeRole(value.lower())
            elif field_type == list or (hasattr(field_type, "__origin__") and field_type.__origin__ is list):
                # Parse comma-separated list
                new_value = [v.strip() for v in value.split(",") if v.strip()]
            else:
                # Default to string
                new_value = value
        else:
            new_value = value

        setattr(cfg, key, new_value)

        # Validate via Pydantic
        cfg.model_validate(cfg.model_dump())

        # Save
        cfg.save()

        console.print(f"[green]✓ Updated {key}[/green]")
        console.print(f"  Before: {old_value!r}")
        console.print(f"  After:  {new_value!r}")

    except ValueError as e:
        console.print(f"[red]Error: Invalid value for '{key}': {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Error setting configuration: {e}[/red]")
        raise SystemExit(1)


@config.command("reset")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def config_reset(yes: bool):
    """Delete the configuration file.

    Removes ~/.prsm/config.yaml. Run 'prsm setup' to reconfigure.
    """
    from prsm.cli_modules.config_schema import PRSMConfig

    config_path = PRSMConfig.config_path()

    if not config_path.exists():
        console.print("[dim]No configuration file to delete.[/dim]")
        return

    if not yes:
        if not click.confirm(f"Delete {config_path}?"):
            console.print("[dim]Cancelled.[/dim]")
            return

    config_path.unlink()
    console.print(f"[green]✓ Deleted {config_path}[/green]")


@config.command("validate")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format: 'text' (human) or 'json' (agent-parseable)")
def config_validate(output_format: str):
    """Validate configuration and check port availability.

    Runs Pydantic validation and checks that p2p_port and api_port are available.
    Exit code: 0 if all pass/warn, 1 if any fail.
    """
    from prsm.cli_modules.config_schema import PRSMConfig

    cfg = PRSMConfig.load()
    checks = []
    any_fail = False

    # 1. Pydantic validation
    try:
        cfg.model_validate(cfg.model_dump())
        checks.append({
            "name": "Pydantic validation",
            "status": "PASS",
            "details": "All fields valid",
        })
    except Exception as e:
        checks.append({
            "name": "Pydantic validation",
            "status": "FAIL",
            "details": str(e),
        })
        any_fail = True

    # 2. P2P port availability
    p2p_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    p2p_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        p2p_sock.bind(("127.0.0.1", cfg.p2p_port))
        checks.append({
            "name": "P2P port availability",
            "status": "PASS",
            "details": f"Port {cfg.p2p_port} is available",
        })
    except Exception as e:
        checks.append({
            "name": "P2P port availability",
            "status": "WARN",
            "details": f"Port {cfg.p2p_port} may be in use: {e}",
        })
    finally:
        p2p_sock.close()

    # 3. API port availability
    api_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    api_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        api_sock.bind(("127.0.0.1", cfg.api_port))
        checks.append({
            "name": "API port availability",
            "status": "PASS",
            "details": f"Port {cfg.api_port} is available",
        })
    except Exception as e:
        checks.append({
            "name": "API port availability",
            "status": "WARN",
            "details": f"Port {cfg.api_port} may be in use: {e}",
        })
    finally:
        api_sock.close()

    # 4. MCP port availability (if enabled)
    if cfg.mcp_server_enabled:
        mcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            mcp_sock.bind(("127.0.0.1", cfg.mcp_server_port))
            checks.append({
                "name": "MCP port availability",
                "status": "PASS",
                "details": f"Port {cfg.mcp_server_port} is available",
            })
        except Exception as e:
            checks.append({
                "name": "MCP port availability",
                "status": "WARN",
                "details": f"Port {cfg.mcp_server_port} may be in use: {e}",
            })
        finally:
            mcp_sock.close()

    # 5. Config file exists
    checks.append({
        "name": "Config file exists",
        "status": "PASS" if PRSMConfig.exists() else "WARN",
        "details": str(PRSMConfig.config_path()) if PRSMConfig.exists() else "Run 'prsm setup' to create",
    })

    # Output
    result = {
        "ok": not any_fail,
        "config_path": str(PRSMConfig.config_path()),
        "checks": checks,
    }

    if output_format == "json":
        _agent_output(result)
        raise SystemExit(1 if any_fail else 0)

    # Rich output
    console.print("\n[bold]Configuration Validation[/bold]\n")
    table = Table(show_header=True)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    for check in checks:
        status = check["status"]
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(status, status)
        status_style = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(status, "white")
        table.add_row(check["name"], f"[{status_style}]{icon} {status}[/{status_style}]", check["details"])

    console.print(table)

    if any_fail:
        console.print("\n[red]Validation failed. Fix errors above.[/red]")
        raise SystemExit(1)
    else:
        console.print("\n[green]All checks passed or warned.[/green]")


@config.command("path")
def config_path_cmd():
    """Print the configuration file path.

    Useful for scripts and automation.
    """
    from prsm.cli_modules.config_schema import PRSMConfig
    console.print(str(PRSMConfig.config_path()))


@config.command("get")
@click.argument("key")
def config_get(key: str):
    """Get a single config value printed to stdout.

    Useful for shell scripts: VALUE=$(prsm config get cpu_pct)
    """
    from prsm.cli_modules.config_schema import PRSMConfig
    import enum

    cfg = PRSMConfig.load()
    if not hasattr(cfg, key):
        console.print(f"[red]Unknown config key: {key}[/red]")
        console.print(f"Available keys: {', '.join(sorted(k for k in cfg.model_fields if not k.startswith('_')))}")
        raise SystemExit(1)

    val = getattr(cfg, key)
    if isinstance(val, enum.Enum):
        print(val.value)
    elif isinstance(val, list):
        print(",".join(str(v) for v in val))
    elif val is None:
        print("")
    else:
        print(val)


@config.command("export")
def config_export():
    """Export current configuration as YAML to stdout.

    Useful for backups or transferring config between nodes.
    """
    from prsm.cli_modules.config_schema import PRSMConfig
    import yaml

    cfg = PRSMConfig.load()
    data = cfg.model_dump(mode='json')
    yaml.safe_dump(data, sys.stdout, default_flow_style=False, sort_keys=False)


@config.command("import")
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def config_import(filepath: Path, yes: bool):
    """Import configuration from a YAML file.

    Overwrites the current config at ~/.prsm/config.yaml.
    """
    from prsm.cli_modules.config_schema import PRSMConfig
    import yaml

    if not yes:
        if not click.confirm(f"Import config from {filepath}? This will overwrite current settings."):
            console.print("[dim]Cancelled.[/dim]")
            return

    try:
        data = yaml.safe_load(filepath.read_text()) or {}
        cfg = PRSMConfig(**data)
        cfg.save()
        console.print(f"[green]{ICONS['success']} Config imported from {filepath}[/green]")
        console.print(f"  Saved to {PRSMConfig.config_path()}")
    except Exception as e:
        console.print(f"[red]{ICONS['error']} Import failed: {e}[/red]")
        raise SystemExit(1)


@config.command("wizard")
def config_wizard():
    """Re-enter the full interactive setup wizard.

    Alias for `prsm setup`.
    """
    from prsm.cli_modules.setup_wizard import run_setup_wizard
    run_setup_wizard()


# ---------------------------------------------------------------------------
# daemon CLI group — deprecated, delegates to `prsm node`
# ---------------------------------------------------------------------------

@main.group()
def daemon():
    """DEPRECATED: use `prsm node` instead.

    All daemon commands are now available under `prsm node`:
      prsm node start --background   # start in background
      prsm node stop                 # stop background node
      prsm node restart              # restart background node
      prsm node status               # show status
      prsm node logs -f              # follow logs
      prsm node install              # system service
    """
    console.print()
    console.print("  prsm daemon is deprecated.", style="yellow")
    console.print("  Use `prsm node` commands instead:", style="yellow")
    console.print("    prsm node start --background  -- Start in background", style="dim")
    console.print("    prsm node stop                -- Stop background node", style="dim")
    console.print("    prsm node restart             -- Restart background node", style="dim")
    console.print("    prsm node status              -- Show node status", style="dim")
    console.print("    prsm node logs -f             -- Follow logs live", style="dim")
    console.print()


@daemon.command("start")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def daemon_start(host: str, port: int):
    """DEPRECATED: use `prsm node start --background`."""
    from prsm.cli_modules.daemon import daemon_start as _start
    _start(host=host, port=port)


@daemon.command("stop")
@click.option("--timeout", default=10, help="Seconds to wait for graceful shutdown")
def daemon_stop(timeout: int):
    """DEPRECATED: use `prsm node stop`."""
    from prsm.cli_modules.daemon import daemon_stop as _stop
    _stop(timeout=timeout)


@daemon.command("restart")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--timeout", default=10, help="Seconds to wait for graceful shutdown")
def daemon_restart(host: str, port: int, timeout: int):
    """DEPRECATED: use `prsm node restart`."""
    from prsm.cli_modules.daemon import daemon_restart as _restart
    _restart(host=host, port=port, timeout=timeout)


@daemon.command("status")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def daemon_status(output_format: str):
    """DEPRECATED: use `prsm node status`."""
    from prsm.cli_modules.daemon import daemon_status as _status
    _status(output_format=output_format)


@daemon.command("logs")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
def daemon_logs(lines: int, follow: bool):
    """DEPRECATED: use `prsm node logs`."""
    from prsm.cli_modules.daemon import daemon_logs as _logs
    _logs(lines=lines, follow=follow)


@daemon.command("install")
@click.option("--dry-run", is_flag=True, help="Print service file without installing")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def daemon_install(dry_run: bool, host: str, port: int):
    """DEPRECATED: use `prsm node install`."""
    from prsm.cli_modules.daemon import daemon_service_install as _install
    _install(dry_run=dry_run, host=host, port=port)


@daemon.command("uninstall")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def daemon_uninstall(yes: bool):
    """DEPRECATED: use `prsm node uninstall`."""
    from prsm.cli_modules.daemon import daemon_service_uninstall as _uninstall
    _uninstall(yes=yes)


# ---------------------------------------------------------------------------
# wallet CLI group (T6.2)
#
# Surface real on-chain state to users: FTNS balance + claimable royalties +
# claim-flow. Reads from prsm/config/networks.py for per-network contract
# addresses; reads PRIVATE_KEY from env (typically loaded from
# ~/.prsm/<network>-deployer.env). Without PRIVATE_KEY, view-only mode:
# can show balance + claimable for any address but can't claim.
# ---------------------------------------------------------------------------


def _wallet_load_signer(network_name: str) -> dict:
    """Resolve wallet identity for the wallet CLI."""
    import os
    from prsm.config.networks import get_network_config

    cfg = get_network_config(network_name)
    rpc_url = (os.environ.get("BASE_SEPOLIA_RPC_URL")
               if network_name == "testnet"
               else os.environ.get("PRSM_BASE_RPC_URL"))
    if not rpc_url:
        rpc_url = cfg.rpc_url_default

    pk = (os.environ.get("PRIVATE_KEY")
          or os.environ.get("FTNS_WALLET_PRIVATE_KEY")
          or "").strip() or None
    address = None
    if pk:
        try:
            from eth_account import Account
            address = Account.from_key(pk).address
        except Exception as exc:
            console.print(
                f"⚠️  could not derive address from PRIVATE_KEY: {exc}",
                style="yellow")

    return {
        "network": cfg,
        "address": address,
        "private_key": pk,
        "rpc_url": rpc_url,
    }


def _wallet_read_balance_wei(rpc_url: str, ftns_token: str, address: str) -> int:
    """Read FTNS balanceOf(address) using a minimal Web3 call."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    erc20_abi = [{
        "inputs": [{"name": "owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    }]
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(ftns_token), abi=erc20_abi)
    return int(contract.functions.balanceOf(
        Web3.to_checksum_address(address)).call())


def _wallet_read_eth_balance_wei(rpc_url: str, address: str) -> int:
    """Sprint 508: native ETH balance for gas-runway visibility."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    return int(w3.eth.get_balance(Web3.to_checksum_address(address)))


def _wallet_read_inbound_count(
    rpc_url: str,
    ftns_token: str,
    address: str,
    lookback_blocks: int = 10000,
) -> tuple:
    """Sprint 518: aggregate inbound count + total in a
    window using the sprint-512 scan helper. RPC-direct
    so it works without a running daemon.

    Returns (count, total_ftns).
    """
    from web3 import Web3
    from prsm.economy.ftns_onchain import (
        _ERC20_ABI, scan_inbound_transfers,
    )
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(ftns_token),
        abi=_ERC20_ABI,
    )
    latest = w3.eth.block_number
    from_block = max(0, latest - lookback_blocks)
    transfers = scan_inbound_transfers(
        contract,
        recipient=address,
        from_block=from_block,
        to_block=latest,
    )
    total = sum(t.get("amount_ftns", 0.0) for t in transfers)
    return (len(transfers), total)


@main.group()
def content():
    """Content publishing — view uploads, royalties accrued."""
    pass


@content.command("search")
@click.argument("query")
@click.option(
    "--limit", "limit", default=20, type=int,
    help="Max results (server caps at 100; default 20).",
)
@click.option(
    "--min-tier", "min_tier",
    type=click.Choice(["low", "medium", "high"]), default=None,
    help="Filter to creators >= tier (default: no filter, "
    "includes cold-start creators).",
)
@click.option(
    "--exclude-new", "exclude_new", is_flag=True, default=False,
    help="Hide cold-start (TIER_NEW) creators.",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def content_search_cli(
    query: str, limit: int, min_tier: Optional[str],
    exclude_new: bool, api_url_override: Optional[str],
    output_format: str,
) -> None:
    """Sprint 808 — keyword search across the content index.

    Wraps GET /content/search. Returns a list of content rows
    matching QUERY. Use --min-tier to filter to creators with
    a minimum reputation; --exclude-new to hide cold-start.

    Exit codes:
      0 — searched (zero hits is OK, not an error)
      1 — server error (422 bad params, 413 too long, etc.)
      2 — daemon unreachable
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/content/search"
    params: Dict[str, Any] = {"q": query, "limit": limit}
    if min_tier:
        params["min_tier"] = min_tier
    if exclude_new:
        params["exclude_new"] = True
    try:
        resp = _httpx.get(endpoint, params=params, timeout=15.0)
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]search failed ({resp.status_code}):"
                f"[/red] {resp.text}"
            )
        raise SystemExit(1)

    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return

    results = data.get("results", [])
    count = data.get("count", len(results))
    if not results:
        console.print(
            f"[dim]No matches[/dim] for [bold]{query}[/bold]. "
            "Try a broader query or drop --min-tier / "
            "--exclude-new."
        )
        return
    console.print(
        f"[bold]{count} result(s)[/bold] for "
        f"[bold]{query}[/bold]:"
    )
    for r in results:
        tier = r.get("creator_tier", "?")
        fname = r.get("filename", "?")
        cid = r.get("cid", "?")
        console.print(
            f"  • [cyan]{cid}[/cyan]  [bold]{fname}[/bold]  "
            f"[dim](tier={tier})[/dim]"
        )
        # sp1339 — verifiable creator attribution at discovery time (when advertised).
        if r.get("creator_eth_address"):
            console.print(f"      creator: [dim]{r['creator_eth_address']}[/dim]")


@content.command("publish-shard")
@click.argument(
    "file_path", type=click.Path(dir_okay=False),
)
@click.option(
    "--dataset-id", "dataset_id", required=True,
    help="Operator-chosen dataset identifier (used as the "
    "manifest's primary key + display fallback when --title "
    "isn't set).",
)
@click.option(
    "--title", "title", default=None,
    help="Human-readable title (default: dataset_id).",
)
@click.option(
    "--shard-count", "shard_count", default=4, type=int,
    help="Number of semantic shards (default 4; server caps "
    "via PRSM_MAX_SHARD_COUNT, default 1000).",
)
@click.option(
    "--royalty-rate", "royalty_rate", default=0.05, type=float,
    help="FTNS earned per access (server bounds 0.001-0.1).",
)
@click.option(
    "--base-access-fee", "base_access_fee",
    default=5.0, type=float,
    help="Base FTNS fee per dataset access (default 5.0).",
)
@click.option(
    "--per-shard-fee", "per_shard_fee",
    default=0.5, type=float,
    help="Additional FTNS fee per shard fetched (default 0.5).",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def content_publish_shard_cli(
    file_path: str, dataset_id: str, title: Optional[str],
    shard_count: int, royalty_rate: float,
    base_access_fee: float, per_shard_fee: float,
    api_url_override: Optional[str], output_format: str,
) -> None:
    """Sprint 817 — upload a binary dataset via the shard endpoint.

    Reads FILE as bytes (binary-safe), base64-encodes, POSTs to
    /content/upload/shard. Server splits into semantic shards
    + returns per-shard CIDs + manifest CID.

    Use this for binary content or text > 100MB (the regular
    /content/upload endpoint caps at 100MB UTF-8).

    Exit codes:
      0 — uploaded
      1 — file missing / server error / validation error
      2 — daemon unreachable
    """
    import base64 as _b64
    import json as _json
    from pathlib import Path as _Path
    import httpx as _httpx

    path = _Path(file_path)
    if not path.exists():
        msg = f"File not found: {file_path}"
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    try:
        content = path.read_bytes()
    except OSError as exc:
        msg = f"Read failed: {exc}"
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    body: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "title": title or dataset_id,
        "content_b64": _b64.b64encode(content).decode("ascii"),
        "shard_count": shard_count,
        "royalty_rate": royalty_rate,
        "base_access_fee": base_access_fee,
        "per_shard_fee": per_shard_fee,
    }

    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/content/upload/shard"
    try:
        resp = _httpx.post(endpoint, json=body, timeout=300.0)
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]Shard upload failed "
                f"({resp.status_code}):[/red] {resp.text}"
            )
        raise SystemExit(1)

    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return

    shards = data.get("shards", [])
    console.print(
        f"[green]Uploaded shard dataset[/green] "
        f"dataset_id=[cyan]{data.get('dataset_id', dataset_id)}"
        f"[/cyan]"
    )
    console.print(
        f"  manifest_cid: [cyan]"
        f"{data.get('manifest_cid', '?')}[/cyan]"
    )
    console.print(
        f"  shard_count: [bold]{len(shards)}[/bold]"
    )
    for s in shards:
        console.print(
            f"  • shard[{s.get('shard_index')}] "
            f"cid=[cyan]{s.get('cid', '?')}[/cyan]  "
            f"[dim]size_bytes={s.get('size_bytes', '?')}[/dim]"
        )


@content.command("publish")
@click.argument(
    "file_path", type=click.Path(dir_okay=False),
)
@click.option(
    "--filename", "filename_override", default=None,
    help="Override the filename advertised to peers "
    "(default: derived from FILE basename).",
)
@click.option(
    "--replicas", "replicas", default=3, type=int,
    help="Replication factor (0=local-only, max 1000; default 3).",
)
@click.option(
    "--royalty-rate", "royalty_rate", default=None, type=float,
    help="FTNS earned per access (0.001-0.1; default 0.01 "
    "server-side when omitted).",
)
@click.option(
    "--parent-cid", "parent_cids", default=(), multiple=True,
    help="CID of source material this content derives from. "
    "Repeatable for multiple parents.",
)
@click.option(
    "--title", "title", default=None,
    help="Short human title — indexed for topic search so peers can find this by keyword.",
)
@click.option(
    "--description", "description", default=None,
    help="Longer description / abstract — indexed for topic search.",
)
@click.option(
    "--tag", "tags", default=(), multiple=True,
    help="Topic tag (repeatable) — each indexed for topic search.",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def content_publish_cli(
    file_path: str, filename_override: Optional[str],
    replicas: int, royalty_rate: Optional[float],
    parent_cids: tuple, title: Optional[str], description: Optional[str],
    tags: tuple, api_url_override: Optional[str],
    output_format: str,
) -> None:
    """Sprint 806 — upload a text file to the P2P content store.

    Reads FILE as UTF-8, POSTs to /content/upload, returns the
    CID. For binary content use /content/upload/shard (CLI
    wrapper TBD).

    Exit codes:
      0 — uploaded
      1 — file missing / non-UTF-8 / server error
      2 — daemon unreachable
    """
    import json as _json
    from pathlib import Path as _Path
    import httpx as _httpx

    path = _Path(file_path)
    if not path.exists():
        msg = f"File not found: {file_path}"
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        msg = (
            f"{file_path}: not UTF-8 decodable ({exc}). The "
            "/content/upload endpoint takes text; for binary, "
            "use /content/upload/shard."
        )
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)
    except OSError as exc:
        msg = f"Read failed: {exc}"
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    body: Dict[str, Any] = {
        "text": text,
        "filename": filename_override or path.name,
        "replicas": replicas,
        "parent_cids": list(parent_cids),
    }
    if royalty_rate is not None:
        body["royalty_rate"] = royalty_rate
    # sp1340 — descriptive metadata for topic search (indexed network-wide).
    if title:
        body["title"] = title
    if description:
        body["description"] = description
    if tags:
        body["tags"] = list(tags)

    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/content/upload"
    try:
        resp = _httpx.post(
            endpoint, json=body, timeout=120.0,
            headers=_node_api_key_headers(),  # sp1199 — auth on a keyed node
        )
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)

    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]Upload failed ({resp.status_code}):"
                f"[/red] {resp.text}"
            )
        raise SystemExit(1)

    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return

    console.print(
        f"[green]Uploaded[/green] "
        f"filename=[bold]{data.get('filename', path.name)}[/bold] "
        f"cid=[cyan]{data.get('cid', '?')}[/cyan]"
    )
    if data.get("size_bytes") is not None:
        console.print(
            f"  size_bytes: [dim]{data['size_bytes']}[/dim]"
        )
    if data.get("replicas") is not None:
        console.print(
            f"  replicas: [dim]{data['replicas']}[/dim]"
        )


@content.command("fetch")
@click.argument("cid")
@click.option(
    "--output", "output_path", default=None,
    type=click.Path(dir_okay=False),
    help="Write decoded content to PATH. Without this, JSON "
    "mode emits the full server payload (base64 data included); "
    "text mode prints a summary.",
)
@click.option(
    "--timeout", "timeout_s", default=30.0, type=float,
    help="Server-side retrieval timeout (default 30s).",
)
@click.option(
    "--no-verify-hash", "no_verify_hash",
    is_flag=True, default=False,
    help="Skip server-side SHA-256 verification of the "
    "retrieved bytes. Default: verify on.",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def content_fetch_cli(
    cid: str, output_path: Optional[str], timeout_s: float,
    no_verify_hash: bool, api_url_override: Optional[str],
    output_format: str,
) -> None:
    """Sprint 805 — retrieve content from the P2P network by CID.

    Wraps GET /content/retrieve/{cid}. base64-decodes the data
    and writes to --output when set. Exit 0 on success, 1 on
    not_found / server-side error, 2 on daemon unreachable.
    """
    import base64 as _b64
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/content/retrieve/{cid}"
    params: Dict[str, Any] = {"timeout": timeout_s}
    if no_verify_hash:
        params["verify_hash"] = False
    try:
        resp = _httpx.get(endpoint, params=params, timeout=timeout_s + 5.0)
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]Retrieve failed ({resp.status_code}):"
                f"[/red] {resp.text}"
            )
        raise SystemExit(1)

    data = resp.json()
    status = data.get("status", "")
    if status != "success":
        if output_format == "json":
            click.echo(_json.dumps(data, indent=2))
        else:
            if status == "not_found":
                console.print(
                    f"[red]not_found[/red] — CID [cyan]{cid}[/cyan] "
                    f"not retrievable from "
                    f"{data.get('providers_tried', 0)} providers."
                )
            else:
                console.print(
                    f"[red]error[/red] — {data.get('error', status)}"
                )
        raise SystemExit(1)

    # success
    if output_path:
        try:
            content_bytes = _b64.b64decode(
                (data.get("data") or "").encode("ascii"),
            )
        except Exception as exc:
            console.print(
                f"[red]base64 decode failed:[/red] {exc}"
            )
            raise SystemExit(1)
        try:
            from pathlib import Path as _Path
            _Path(output_path).write_bytes(content_bytes)
        except OSError as exc:
            console.print(f"[red]Write failed:[/red] {exc}")
            raise SystemExit(1)

    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return

    # text mode summary
    console.print(
        f"[green]Retrieved[/green] cid=[cyan]{cid}[/cyan]"
    )
    if data.get("filename"):
        console.print(
            f"  filename: [bold]{data['filename']}[/bold]"
        )
    console.print(
        f"  size_bytes: [bold]{data.get('size_bytes', '?')}[/bold]"
    )
    console.print(
        f"  content_hash: [dim]{data.get('content_hash', '?')}[/dim]"
    )
    # sp1338 — verifiable provenance + creator attribution (None for pre-provenance content).
    if data.get("creator_eth_address"):
        console.print(
            f"  creator: [bold]{data['creator_eth_address']}[/bold]"
        )
    if data.get("provenance_hash"):
        console.print(
            f"  provenance_hash: [dim]{data['provenance_hash']}[/dim]"
        )
    console.print(
        f"  providers_tried: "
        f"[dim]{data.get('providers_tried', 0)}[/dim]"
    )
    if output_path:
        console.print(
            f"  wrote → [bold]{output_path}[/bold]"
        )


@content.command("get")
@click.argument("query")
@click.option(
    "--output", "output_path", default=None,
    type=click.Path(dir_okay=False),
    help="Write the retrieved bytes to PATH (base64-decoded).",
)
@click.option(
    "--min-tier", "min_tier",
    type=click.Choice(["low", "medium", "high"]), default=None,
    help="Only consider creators at or above this tier.",
)
@click.option(
    "--api-url", "api_url_override", default=None, help="Override daemon URL",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def content_get_cli(
    query: str, output_path: Optional[str], min_tier: Optional[str],
    api_url_override: Optional[str], output_format: str,
) -> None:
    """Sprint 1341 — one-call find → fetch → verify: search a topic, fetch the top hit,
    verify its integrity client-side, and show who created it.

    Wraps the SDK ``find_and_fetch``: topic-search (sp1339/1340) → retrieve the top match →
    re-hash the bytes (sha256 == content_hash) → surface creator/provenance. The flagship
    data-consumer command. Exit 0 found+verified, 1 nothing found / integrity FAIL, 2 daemon
    unreachable.
    """
    import base64 as _b64
    import json as _json
    from prsm.sdk.client import PRSMClient
    url = _api_url_from_creds(api_url_override)

    async def _go():
        client = PRSMClient(base_url=url,
                            api_key=(os.environ.get("PRSM_NODE_API_KEY") or "").strip())
        try:
            return await client.find_and_fetch(query, min_tier=min_tier)
        finally:
            await client.close()

    try:
        res = _run_async(_go())
    except FileNotFoundError as exc:
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "error": str(exc)}))
        else:
            console.print(f"[yellow]No result:[/yellow] {exc}")
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if any(s in msg.lower() for s in ("connect", "refused", "unreachable", "timeout")):
            console.print(f"[red]Daemon unreachable at {url}[/red] — {exc}")
            raise SystemExit(2)
        console.print(f"[red]content get failed:[/red] {exc}")
        raise SystemExit(1)

    if output_path:
        try:
            from pathlib import Path as _Path
            _Path(output_path).write_bytes(_b64.b64decode((res.get("data") or "").encode()))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Write failed:[/red] {exc}")
            raise SystemExit(1)

    if output_format == "json":
        # drop the (large) base64 payload from the json summary unless it was not written
        summary = {k: v for k, v in res.items() if k != "data" or not output_path}
        click.echo(_json.dumps(summary, indent=2, default=str))
        raise SystemExit(0 if res.get("integrity_verified") else 1)

    ok = res.get("integrity_verified")
    console.print(
        f"[green]Found[/green] [cyan]{res.get('cid')}[/cyan]  "
        f"[bold]{res.get('filename') or '?'}[/bold]  "
        f"[dim](tier={res.get('creator_tier', '?')})[/dim]")
    console.print(
        f"  integrity: {'[green]VERIFIED[/green]' if ok else '[red]FAILED[/red]'}  "
        f"[dim](sha256 == content_hash)[/dim]")
    if res.get("creator_eth_address"):
        console.print(f"  creator: [dim]{res['creator_eth_address']}[/dim]")
    if res.get("provenance_hash"):
        console.print(f"  provenance_hash: [dim]{res['provenance_hash']}[/dim]")
    console.print(f"  size_bytes: [dim]{res.get('size_bytes', '?')}[/dim]")
    if output_path:
        console.print(f"  wrote → [bold]{output_path}[/bold]")
    raise SystemExit(0 if ok else 1)


@content.command("mine")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def content_mine(api_port, output_format, limit):
    """List content this node has uploaded.

    Each entry shows filename, size, royalty rate, access count,
    accumulated FTNS royalties, and on-chain provenance status.
    """
    import json
    import httpx

    url = f"http://127.0.0.1:{api_port}/content/mine?limit={limit}"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code == 503:
        console.print(
            f"[yellow]ContentUploader not configured.[/yellow]\n"
            f"[dim]{resp.json().get('detail', 'unknown')}[/dim]"
        )
        sys.exit(0)
    if resp.status_code != 200:
        console.print(
            f"[red]/content/mine returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)

    body = resp.json()
    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    entries = body.get("entries", [])
    total = body.get("total", 0)
    console.print(
        f"[bold]My Uploaded Content[/bold] "
        f"(showing {len(entries)} of {total}):"
    )
    if not entries:
        console.print(
            f"  [dim]No uploads yet. POST to /content/upload "
            f"or /content/upload/shard.[/dim]"
        )
        return
    for e in entries:
        cid = e.get("content_id", "?")
        if len(cid) > 22:
            cid = cid[:14] + ".." + cid[-6:]
        fn = e.get("filename", "?")
        if len(fn) > 18:
            fn = fn[:15] + "..."
        royalties = e.get("total_royalties", 0.0)
        hits = e.get("access_count", 0)
        size = e.get("size_bytes", 0)
        # Escape brackets so rich doesn't interpret as markup tags
        prov = (
            r"\[chain]" if e.get("provenance_tx_hash")
            else r"\[off]"
        )
        console.print(
            f"  {cid:<22}  {fn:<18}  {size:>8}b  "
            f"{royalties:>9.6f} FTNS  hits={hits:<4}  {prov}"
        )


@main.group()
def wallet():
    """On-chain wallet — view balance, claim royalties."""
    pass


@wallet.command("info")
@click.option(
    "--network", "network_name",
    default="testnet",
    type=click.Choice(["mainnet", "testnet"]),
    help="Network to query (default: testnet)",
)
@click.option(
    "--address",
    default=None,
    help="Override wallet address (default: derive from PRIVATE_KEY env)",
)
def wallet_info(network_name: str, address):
    """Show on-chain wallet state: address, FTNS balance, claimable royalties.

    \b
    Reads from:
      - PRIVATE_KEY env (or FTNS_WALLET_PRIVATE_KEY) — derives address
      - prsm/config/networks.py — for contract addresses
      - BASE_SEPOLIA_RPC_URL (testnet) or PRSM_BASE_RPC_URL (mainnet)

    \b
    Example:
        prsm wallet info --network testnet
        prsm wallet info --network testnet --address 0xabc...
    """
    ctx = _wallet_load_signer(network_name)
    cfg = ctx["network"]

    addr = address or ctx["address"]
    if not addr:
        console.print(
            "❌ no address available — set PRIVATE_KEY env var "
            "or pass --address", style="red")
        raise SystemExit(1)
    if not cfg.ftns_token:
        console.print(
            f"❌ {network_name} FTNS token address not configured in "
            f"prsm/config/networks.py", style="red")
        raise SystemExit(1)

    console.print(f"\n[bold]Wallet on {cfg.name}[/bold] (chainId {cfg.chain_id})")
    console.print(f"Address:        {addr}")
    console.print(f"Explorer:       {cfg.explorer_url}/address/{addr}")
    console.print(f"RPC:            {ctx['rpc_url']}")
    console.print()

    # FTNS balance
    try:
        bal_wei = _wallet_read_balance_wei(ctx['rpc_url'], cfg.ftns_token, addr)
        bal = bal_wei / 1e18
        console.print(f"FTNS balance:   [bold]{bal:,.6f}[/bold] FTNS  "
                      f"({cfg.ftns_token[:10]}…)")
    except Exception as exc:
        console.print(f"FTNS balance:   ⚠️  read failed: {exc}", style="yellow")

    # Sprint 508 — native ETH balance (gas runway).
    try:
        eth_wei = _wallet_read_eth_balance_wei(ctx['rpc_url'], addr)
        eth = eth_wei / 1e18
        if eth < 0.0001:
            status_color = "bold red"
            status_label = "CRITICAL"
        elif eth < 0.0005:
            status_color = "yellow"
            status_label = "LOW"
        else:
            status_color = "bold green"
            status_label = "ok"
        console.print(
            f"ETH balance:    [bold]{eth:.10f}[/bold] ETH  "
            f"[[{status_color}]{status_label}[/{status_color}]]"
        )
        if status_label == "CRITICAL":
            console.print(
                "  ⚠️  Top up ETH now — on-chain TX will start failing.",
                style="bold red",
            )
        elif status_label == "LOW":
            console.print(
                "  ⚠️  Gas is low — plan to top up soon.",
                style="yellow",
            )
    except Exception as exc:
        console.print(f"ETH balance:    ⚠️  read failed: {exc}", style="yellow")

    # Sprint 518 — recent inbound FTNS count (scan Transfer
    # events for `to == address`). Uses sprint-512 helper.
    try:
        inb_count, inb_total = _wallet_read_inbound_count(
            ctx['rpc_url'], cfg.ftns_token, addr,
        )
        if inb_count > 0:
            console.print(
                f"Inbound:        [bold]{inb_count}[/bold] receipts  "
                f"([green]{inb_total:.6f}[/green] FTNS total, last ~10k blocks)"
            )
        else:
            console.print(
                f"Inbound:        0 receipts in last ~10k blocks",
                style="dim",
            )
    except Exception as exc:
        console.print(f"Inbound:        ⚠️  read failed: {exc}", style="yellow")

    # Claimable royalties
    if cfg.royalty_distributor:
        try:
            from prsm.economy.web3.royalty_distributor import (
                RoyaltyDistributorClient,
            )
            client = RoyaltyDistributorClient(
                rpc_url=ctx['rpc_url'],
                distributor_address=cfg.royalty_distributor,
                ftns_token_address=cfg.ftns_token,
                private_key=None,
            )
            claimable_wei = client.claimable(addr)
            claimable = claimable_wei / 1e18
            color = "bold green" if claimable > 0 else "dim"
            console.print(f"Claimable:      [{color}]{claimable:,.6f}[/{color}] "
                          f"FTNS  ({cfg.royalty_distributor[:10]}…)")
            if claimable > 0:
                console.print(f"\n  → run [bold]prsm wallet claim --network "
                              f"{network_name}[/bold] to withdraw")
        except Exception as exc:
            console.print(f"Claimable:      ⚠️  read failed: {exc}",
                          style="yellow")
    else:
        console.print(
            f"Claimable:      (RoyaltyDistributor not configured "
            f"for {network_name})", style="dim")

    console.print()
    if network_name == "testnet":
        for note in cfg.notes:
            console.print(f"  ℹ {note}", style="dim")
    console.print()


@wallet.command("gas-status")
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_gas_status(api_url: str) -> None:
    """Show ETH gas balance for the daemon's loaded wallet.

    Queries /wallet/gas-status on the running daemon. Surfaces
    status=ok|low|critical so operators can top up before
    on-chain TX start failing with cryptic "insufficient funds
    for gas" errors.
    """
    import httpx
    url = _api_url_from_creds(api_url)
    try:
        r = httpx.get(f"{url}/wallet/gas-status", timeout=10.0)
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)
    if r.status_code == 503:
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text[:300]
        console.print("❌ Gas status unavailable:", style="red")
        console.print(f"   {detail}")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"❌ HTTP {r.status_code}: {r.text[:200]}", style="red")
        raise SystemExit(1)
    d = r.json()
    status_color = {
        "ok": "bold green",
        "low": "yellow",
        "critical": "bold red",
        "unavailable": "dim",
    }.get(d.get("status"), "white")
    console.print(f"\n[bold]Operator gas balance[/bold]")
    console.print(f"  address       : {d.get('address')}")
    if d.get("eth_balance") is not None:
        console.print(f"  ETH balance   : {d['eth_balance']:.10f} ETH")
    else:
        console.print(f"  ETH balance   : unavailable")
    console.print(f"  status        : [{status_color}]{d.get('status', '?').upper()}[/{status_color}]")
    console.print(f"  low threshold : {d.get('low_threshold_eth')} ETH")
    console.print(f"  critical thr. : {d.get('critical_threshold_eth')} ETH")
    if d.get("status") == "critical":
        console.print(
            "\n⚠️  Top up ETH now — broadcasts will start failing soon.",
            style="bold red",
        )
    elif d.get("status") == "low":
        console.print(
            "\n⚠️  Plan to top up ETH soon to avoid mid-job failures.",
            style="yellow",
        )
    console.print()


@wallet.command("deposit-info")
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_deposit_info(api_url: str) -> None:
    """Show bridge deposit escrow address + linkage status.

    Sprint 540 Pattern A: daemon-mediated bridge. Deposit on-chain
    FTNS into your off-chain PRSM balance by:
      1. Link your sending ETH address (prsm wallet link-address)
      2. Sign an on-chain ERC-20 transfer to the escrow address
      3. Daemon credits your off-chain balance after detected inbound
    """
    import httpx
    url = _api_url_from_creds(api_url)
    try:
        r = httpx.get(f"{url}/wallet/deposit/info", timeout=10.0)
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)
    if r.status_code == 503:
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text[:300]
        console.print("❌ Deposit flow unavailable:", style="red")
        console.print(f"   {detail}")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"❌ HTTP {r.status_code}", style="red")
        raise SystemExit(1)
    d = r.json()
    console.print("\n[bold]Bridge deposit info[/bold]")
    console.print(f"  escrow address  : [bold]{d.get('escrow_address')}[/bold]")
    console.print(f"  wallet_id       : {d.get('wallet_id')}")
    linked = d.get("linked_eth_address")
    if linked:
        console.print(
            f"  linked ETH addr : [green]{linked}[/green]"
        )
    else:
        console.print(
            "  linked ETH addr : [yellow]NOT LINKED[/yellow]"
        )
        console.print(
            "\n  → Link an address first:",
            style="dim",
        )
        console.print(
            f"    prsm wallet link-address --eth-address 0x...",
            style="cyan",
        )
    console.print(f"  FTNS contract   : {d.get('ftns_token_contract')}")
    console.print(f"  chain_id        : {d.get('chain_id')}")
    console.print()
    console.print(f"[dim]{d.get('instructions', '')}[/dim]")
    console.print()


@wallet.command("link-address")
@click.option(
    "--eth-address", required=True,
    help="0x-prefixed ETH address to link to your wallet_id",
)
@click.option(
    "--wallet-id", default=None,
    help="Wallet to link (default: this node's identity)",
)
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_link_address(
    eth_address: str, wallet_id: str, api_url: str,
) -> None:
    """Link an ETH address for bridge deposits.

    Inbound on-chain FTNS transfers FROM this address will credit
    your off-chain wallet balance automatically (sprint 540).
    """
    import httpx
    url = _api_url_from_creds(api_url)
    # Default wallet_id to this node's identity if unset
    if not wallet_id:
        try:
            info = httpx.get(
                f"{url}/wallet/deposit/info", timeout=5.0,
            ).json()
            wallet_id = info.get("wallet_id")
        except Exception:
            console.print(
                "❌ Could not auto-resolve wallet_id; pass "
                "--wallet-id explicitly", style="red",
            )
            raise SystemExit(1)
    try:
        r = httpx.post(
            f"{url}/wallet/deposit/link",
            json={
                "wallet_id": wallet_id,
                "eth_address": eth_address,
            },
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)
    if r.status_code == 200:
        d = r.json()
        console.print(
            f"✅ Linked [bold]{d['eth_address']}[/bold] → "
            f"wallet_id [bold]{d['wallet_id']}[/bold]",
            style="green",
        )
    elif r.status_code == 422:
        console.print(
            f"❌ Invalid input: {r.json().get('detail', '?')}",
            style="red",
        )
        raise SystemExit(1)
    elif r.status_code == 503:
        console.print(
            f"❌ Service unavailable: {r.json().get('detail', '?')}",
            style="red",
        )
        raise SystemExit(1)
    else:
        console.print(
            f"❌ HTTP {r.status_code}: {r.text[:200]}", style="red",
        )
        raise SystemExit(1)


@wallet.command("deposit")
@click.option(
    "--amount", required=True, type=float,
    help="FTNS to deposit into the on-chain EscrowPool",
)
@click.option(
    "--network", "network_name",
    type=click.Choice(["mainnet", "testnet"]), default="testnet",
    help="Network the EscrowPool lives on (default: testnet = Base Sepolia)",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Skip confirmation prompt",
)
def wallet_deposit(amount: float, network_name: str, yes: bool) -> None:
    """Sprint 1192 — deposit FTNS into the on-chain EscrowPool (self-custodied).

    Funds the requester-payment escrow your address draws from when you
    `prsm compute pay-infer`. Wraps the sp1189 SDK deposit_escrow: approves the
    FTNS allowance then calls EscrowPool.deposit, signed + broadcast with your
    wallet key. You keep custody — withdraw any unspent balance later.

    \b
    Key source (signs the tx): PRIVATE_KEY env, else FTNS_WALLET_PRIVATE_KEY.
    Requires web3 + native gas (ETH on Base) on the signing address.

    \b
    Example:
        export PRIVATE_KEY=0x...
        prsm wallet deposit --amount 5 --network testnet

    Exit 0 success, 1 on error.
    """
    from prsm.sdk.client import PRSMClient
    ctx = _wallet_load_signer(network_name)
    requester_key = ctx.get("private_key")
    addr = ctx.get("address")
    if not requester_key:
        console.print(
            "❌ no signing key — set PRIVATE_KEY (or FTNS_WALLET_PRIVATE_KEY) "
            "to your wallet's private key.", style="red")
        raise SystemExit(1)
    if amount <= 0:
        console.print("❌ --amount must be > 0", style="red")
        raise SystemExit(1)

    console.print(
        f"\nDeposit [bold]{amount} FTNS[/bold] into the EscrowPool on "
        f"[bold]{network_name}[/bold]")
    console.print(f"  from: {addr or '<address from key>'}")
    if not yes and not click.confirm("Sign + broadcast this on-chain deposit?"):
        console.print("aborted", style="yellow")
        raise SystemExit(1)

    async def _go():
        client = PRSMClient()
        try:
            return await client.deposit_escrow(
                requester_key=requester_key,
                amount_ftns=amount,
                network=network_name,
            )
        finally:
            await client.close()

    try:
        tx_hash = _run_async(_go())
    except Exception as exc:  # noqa: BLE001
        console.print(f"❌ deposit failed: {exc}", style="red")
        raise SystemExit(1)
    console.print(f"✅ deposited {amount} FTNS — tx [green]{tx_hash}[/green]")
    console.print(
        "[dim]escrow now funds `prsm compute pay-infer` charges.[/dim]\n")


@wallet.command("faucet")
@click.option(
    "--address", "address", default=None,
    help="Destination address (default: derive from PRIVATE_KEY env)",
)
@click.option(
    "--amount", "amount", default=None, type=float,
    help="FTNS to request (default: the operator's per-request cap)",
)
@click.option(
    "--network", "network_name",
    type=click.Choice(["mainnet", "testnet"]), default="testnet",
    help="Network (default: testnet; the faucet is TESTNET-ONLY)",
)
@click.option("--api-url", "api_url_override", default=None, help="Override daemon URL")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text", help="Output format",
)
def wallet_faucet(
    address, amount, network_name: str,
    api_url_override, output_format: str,
) -> None:
    """Sprint 1193 — request TESTNET FTNS from the on-chain faucet.

    POSTs to the daemon's /ftns/faucet/onchain (sp1190): the faucet wallet sends
    real on-chain testnet FTNS to your address so you can `prsm wallet deposit`
    then `prsm compute pay-infer`. TESTNET-ONLY — the faucet hard-refuses any
    chain other than Base Sepolia, so it never dispenses real-value FTNS.

    \b
    No signing key needed — the faucet operator signs the transfer; this only
    sends your destination address. The default address is derived from
    PRIVATE_KEY / FTNS_WALLET_PRIVATE_KEY if set, else pass --address.

    \b
    Example (end-to-end testnet front door):
        prsm wallet faucet --address 0x... --network testnet
        prsm wallet deposit --amount 5 --network testnet
        prsm compute pay-infer --prompt "Hello" --network testnet

    Exit 0 success, 1 daemon error, 2 unreachable.
    """
    import json as _json
    import httpx
    dest = address
    if not dest:
        dest = _wallet_load_signer(network_name).get("address")
    if not dest:
        console.print(
            "❌ no destination address — pass --address 0x... or set PRIVATE_KEY "
            "(the address is derived from it).", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url_override)
    body = {"destination_address": dest}
    if amount is not None:
        body["amount"] = amount
    try:
        r = httpx.post(f"{url}/ftns/faucet/onchain", json=body, timeout=180.0,
                       headers=_node_api_key_headers())  # sp1199 — /ftns/ is protected
    except httpx.ConnectError:
        console.print(f"❌ cannot connect to {url}", style="red")
        raise SystemExit(2)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text[:300]
        if output_format == "json":
            click.echo(_json.dumps({"ok": False, "status": r.status_code, "detail": detail}))
        else:
            console.print(f"❌ faucet request failed (HTTP {r.status_code}):", style="red")
            console.print(f"   {detail}")
        raise SystemExit(1)
    d = r.json()
    if output_format == "json":
        click.echo(_json.dumps(d))
        return
    console.print(
        f"✅ dispensed [bold]{d.get('dispensed_ftns')} FTNS[/bold] → "
        f"{d.get('recipient')}")
    console.print(f"  tx        : [green]{d.get('tx_hash')}[/green]")
    console.print(f"  faucet    : {d.get('faucet_address')}")
    console.print(f"  network   : {d.get('network')}")
    console.print(
        "[dim]next: `prsm wallet deposit` to fund the EscrowPool, then "
        "`prsm compute pay-infer`.[/dim]\n")


@wallet.command("withdraw")
@click.option(
    "--amount", required=True, type=float,
    help="FTNS to withdraw from off-chain balance to on-chain",
)
@click.option(
    "--to", "to_eth_address", default=None,
    help="Recipient ETH address (default: this wallet's linked addr)",
)
@click.option(
    "--wallet-id", default=None,
    help="Wallet to debit (default: this node's identity)",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Skip confirmation prompt",
)
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_withdraw(
    amount: float, to_eth_address: str, wallet_id: str,
    yes: bool, api_url: str,
) -> None:
    """Withdraw off-chain FTNS to on-chain (Pattern A bridge).

    Debits your off-chain wallet, then signs an on-chain ERC-20
    transfer from the operator escrow to your recipient address.
    Atomicity: if the broadcast fails, the daemon credits a refund
    so your off-chain balance stays whole.
    """
    import httpx
    url = _api_url_from_creds(api_url)
    payload = {"amount_ftns": amount}
    if to_eth_address:
        payload["to_eth_address"] = to_eth_address
    if wallet_id:
        payload["wallet_id"] = wallet_id

    if not yes:
        console.print(
            f"\n[bold]About to withdraw {amount:.6f} FTNS[/bold]"
        )
        if to_eth_address:
            console.print(f"  → to {to_eth_address}")
        else:
            console.print(
                f"  → to (linked address — daemon resolves)",
                style="dim",
            )
        console.print(
            "[yellow]This will broadcast a real on-chain TX. "
            "Continue?[/yellow] (y/N): ",
            end="",
        )
        ans = input().strip().lower()
        if ans not in ("y", "yes"):
            console.print("Cancelled.", style="dim")
            raise SystemExit(0)

    try:
        r = httpx.post(
            f"{url}/wallet/withdraw", json=payload, timeout=90.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if r.status_code == 200:
        d = r.json()
        console.print(
            "✅ Withdraw confirmed on-chain!", style="bold green",
        )
        console.print(f"   tx_hash      : 0x{d.get('tx_hash', '').lstrip('0x')}")
        console.print(f"   block        : {d.get('block_number')}")
        console.print(f"   amount       : {d.get('amount_ftns')} FTNS")
        console.print(f"   to           : {d.get('to_eth_address')}")
        console.print(f"   wallet_id    : {d.get('wallet_id')}")
        console.print(f"   debit_tx_id  : {d.get('debit_tx_id')}")
        return
    detail = ""
    try:
        detail = r.json().get("detail", "")
    except Exception:
        detail = r.text[:300]
    if r.status_code == 402:
        console.print(f"❌ Insufficient off-chain balance:", style="red")
        console.print(f"   {detail}")
    elif r.status_code == 400:
        console.print(f"❌ Invalid request:", style="red")
        console.print(f"   {detail}")
    elif r.status_code == 422:
        console.print(f"❌ Validation error:", style="red")
        console.print(f"   {detail}")
    elif r.status_code == 502:
        console.print(
            f"⚠️  Broadcast failed; off-chain refund issued:",
            style="yellow",
        )
        console.print(f"   {detail}")
    elif r.status_code == 503:
        console.print(f"❌ Service unavailable:", style="red")
        console.print(f"   {detail}")
    else:
        console.print(f"❌ HTTP {r.status_code}: {detail}", style="red")
    raise SystemExit(1)


# ──────────────────────────────────────────────────────────────────
# Sprint 557 — signed withdraw helper for sprint-556 enforcement.
# Operators with requires_user_signature=True on their wallet use
# this command instead of `wallet withdraw` to drive the full
# signed-flow happy path without writing Python.
# ──────────────────────────────────────────────────────────────────


def _build_signed_withdraw_body(
    *,
    amount_ftns: float,
    wallet_id: str,
    to_eth_address: str,
    nonce: int,
    private_key,
    expiry_unix=None,
) -> dict:
    """Build the JSON body for POST /wallet/withdraw with a signed
    EIP-712 payload. Extracted from the CLI command body so tests
    can pin the signing path without Click invocation.

    `expiry_unix` defaults to now + 300 (5-minute window per sprint-
    554 user input).
    """
    import time as _time
    from prsm.economy.withdraw_signature import (
        sign_withdraw_payload,
    )
    if expiry_unix is None:
        expiry_unix = int(_time.time()) + 300
    amount_wei = int(amount_ftns * 1e18)
    payload = {
        "wallet_id": wallet_id,
        "amount_ftns_wei": amount_wei,
        "to_eth_address": to_eth_address,
        "nonce": int(nonce),
        "expiry_unix": int(expiry_unix),
    }
    sig = sign_withdraw_payload(payload, private_key)
    return {
        "amount_ftns": amount_ftns,
        "wallet_id": wallet_id,
        "to_eth_address": to_eth_address,
        "signature": "0x" + sig.hex(),
        "nonce": int(nonce),
        "expiry_unix": int(expiry_unix),
    }


@wallet.command("sign-withdraw")
@click.option(
    "--amount", required=True, type=float,
    help="FTNS to withdraw from off-chain balance to on-chain",
)
@click.option(
    "--to", "to_eth_address", default=None,
    help="Recipient ETH address (default: this wallet's linked addr)",
)
@click.option(
    "--wallet-id", default=None,
    help="Wallet to debit (default: this node's identity)",
)
@click.option(
    "--nonce", default=None, type=int,
    help="Nonce to use (default: fetched from /wallet/deposit/info)",
)
@click.option(
    "--expiry-seconds", default=300, type=int,
    help="Seconds until signature expires (default: 300)",
)
@click.option(
    "--private-key", default=None,
    help=(
        "User's ECDSA private key (default: env PRSM_USER_SIGNING_KEY). "
        "Must recover to the wallet's linked eth_address."
    ),
)
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Skip confirmation prompt",
)
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_sign_withdraw(
    amount: float, to_eth_address: str, wallet_id: str,
    nonce, expiry_seconds: int, private_key, yes: bool,
    api_url: str,
) -> None:
    """Withdraw off-chain FTNS to on-chain WITH a user EIP-712 signature.

    Required when the wallet has requires_user_signature=True (toggle
    via POST /wallet/require-signature). Sprint-556 enforcement at
    /wallet/withdraw verifies the signature recovers to the wallet's
    linked eth_address (sprint 540). Use the same private key you
    used when linking your eth_address.
    """
    import os
    import time as _time
    import httpx

    pk = private_key or os.environ.get("PRSM_USER_SIGNING_KEY", "")
    pk = pk.strip()
    if not pk:
        console.print(
            "❌ No signing key. Pass --private-key or set "
            "PRSM_USER_SIGNING_KEY in env.",
            style="red",
        )
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    # Fetch wallet_id / linked / nonce from /wallet/deposit/info.
    try:
        r = httpx.get(f"{url}/wallet/deposit/info", timeout=15.0)
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(
            f"❌ /wallet/deposit/info returned HTTP {r.status_code}: "
            f"{r.text[:200]}",
            style="red",
        )
        raise SystemExit(1)
    info = r.json()
    resolved_wallet = wallet_id or info.get("wallet_id")
    resolved_to = to_eth_address or info.get("linked_eth_address")
    resolved_nonce = (
        nonce if nonce is not None else info.get("next_withdraw_nonce", 0)
    )
    if not resolved_to:
        console.print(
            "❌ No recipient: pass --to or link an eth_address via "
            "`prsm wallet link-address` first.",
            style="red",
        )
        raise SystemExit(1)

    expiry = int(_time.time()) + int(expiry_seconds)
    body = _build_signed_withdraw_body(
        amount_ftns=amount,
        wallet_id=resolved_wallet,
        to_eth_address=resolved_to,
        nonce=int(resolved_nonce),
        private_key=pk,
        expiry_unix=expiry,
    )

    if not yes:
        console.print(
            f"\n[bold]About to sign + withdraw {amount:.6f} FTNS[/bold]"
        )
        console.print(f"  → to        : {resolved_to}")
        console.print(f"  → wallet_id : {resolved_wallet}")
        console.print(f"  → nonce     : {body['nonce']}")
        console.print(
            f"  → expires   : {expiry} ({expiry_seconds}s window)"
        )
        console.print(
            "[yellow]This will broadcast a real on-chain TX. "
            "Continue?[/yellow] (y/N): ",
            end="",
        )
        ans = input().strip().lower()
        if ans not in ("y", "yes"):
            console.print("Cancelled.", style="dim")
            raise SystemExit(0)

    try:
        r = httpx.post(
            f"{url}/wallet/withdraw", json=body, timeout=90.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if r.status_code == 200:
        d = r.json()
        console.print(
            "✅ Signed withdraw confirmed on-chain!",
            style="bold green",
        )
        console.print(f"   tx_hash        : {d.get('tx_hash')}")
        console.print(f"   block          : {d.get('block_number')}")
        console.print(f"   amount         : {d.get('amount_ftns')} FTNS")
        console.print(f"   to             : {d.get('to_eth_address')}")
        console.print(f"   wallet_id      : {d.get('wallet_id')}")
        console.print(f"   nonce_consumed : {d.get('nonce_consumed')}")
        console.print(f"   debit_tx_id    : {d.get('debit_tx_id')}")
        return

    detail = ""
    try:
        detail = r.json().get("detail", "")
    except Exception:
        detail = r.text[:300]
    if r.status_code == 401:
        console.print(
            "❌ Signature rejected (401):", style="red",
        )
        console.print(f"   {detail}")
        console.print(
            "[dim]Hint: re-run `prsm wallet deposit-info` to read the "
            "current nonce + check that your linked eth_address "
            "matches the key you're signing with.[/dim]"
        )
    elif r.status_code == 402:
        console.print(f"❌ Insufficient off-chain balance: {detail}",
                      style="red")
    elif r.status_code == 502:
        console.print(
            "⚠️  Broadcast failed; off-chain refund issued; "
            "nonce stays consumed (replay safety):",
            style="yellow",
        )
        console.print(f"   {detail}")
    else:
        console.print(
            f"❌ HTTP {r.status_code}: {detail}", style="red",
        )
    raise SystemExit(1)


@wallet.command("claim")
@click.option(
    "--network", "network_name",
    default="testnet",
    type=click.Choice(["mainnet", "testnet"]),
    help="Network to claim from (default: testnet)",
)
@click.confirmation_option(
    prompt="Submit RoyaltyDistributor.claim() transaction?",
    help="Skip confirmation prompt with --yes",
)
def wallet_claim(network_name: str):
    """Withdraw accumulated FTNS royalties via RoyaltyDistributor.claim().

    \b
    The signer's full claimable[address] balance is transferred to their
    address as FTNS, and the mapping entry is zeroed. Reverts on-chain
    if claimable is 0.

    \b
    Required env:
      PRIVATE_KEY or FTNS_WALLET_PRIVATE_KEY — signs the tx
    """
    ctx = _wallet_load_signer(network_name)
    cfg = ctx["network"]

    if not ctx['private_key']:
        console.print("❌ PRIVATE_KEY env var required for claim", style="red")
        raise SystemExit(1)
    if not cfg.royalty_distributor or not cfg.ftns_token:
        console.print(
            f"❌ {network_name}: RoyaltyDistributor or FTNS token "
            f"address missing in prsm/config/networks.py", style="red")
        raise SystemExit(1)

    addr = ctx['address']
    console.print(f"\n[bold]Claiming on {cfg.name}[/bold]")
    console.print(f"From:           {addr}")

    try:
        from prsm.economy.web3.royalty_distributor import (
            RoyaltyDistributorClient,
        )
        client = RoyaltyDistributorClient(
            rpc_url=ctx['rpc_url'],
            distributor_address=cfg.royalty_distributor,
            ftns_token_address=cfg.ftns_token,
            private_key=ctx['private_key'],
        )
        # Pre-flight: check claimable so we surface a clean message
        # instead of an opaque on-chain revert.
        claimable_wei = client.claimable(addr)
        if claimable_wei == 0:
            console.print(
                f"⚠️  claimable[{addr}] is 0 — nothing to claim "
                f"(would revert on-chain). Earn royalties first.",
                style="yellow")
            raise SystemExit(0)
        console.print(f"Claimable:      {claimable_wei / 1e18:,.6f} FTNS")
        console.print()
        console.print("Submitting claim transaction…")
        tx_hash, status = client.claim()
        console.print(f"  Tx hash:       {tx_hash}")
        console.print(f"  Status:        {status}")
        console.print(f"  Explorer:      {cfg.explorer_url}/tx/{tx_hash}")
        console.print(f"\n[bold green]✓ Claim submitted[/bold green]\n")
    except SystemExit:
        raise
    except Exception as exc:
        console.print(
            f"❌ claim failed: {type(exc).__name__}: {exc}", style="red")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Sprint 789 — `prsm wallet devices` subgroup (multi-device arc).
#
# Manages the operator's device roster from the command line.
# `add` mints an EIP-191 delegation; `verify` checks one. The
# `list` command will land once the daemon exposes a binding-list
# HTTP endpoint (sprint 790+).
# ---------------------------------------------------------------------------


@wallet.group("devices")
def wallet_devices() -> None:
    """Manage your operator-account device roster (sprint 789).

    Multi-device operators run multiple daemons under one ETH
    wallet. Each device needs an EIP-191 delegation signed by the
    operator's ETH key — the delegation goes into the new device's
    `PRSM_OPERATOR_DELEGATION` env / file and proves to the network
    that this node_id is authorized under the wallet.
    """


@wallet_devices.command("add")
@click.option(
    "--node-id", "node_id", required=True,
    help="Ed25519 node_id (32-char hex) of the new device. Get this "
    "from the device's identity.json or `prsm node info`.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format. json prints just the delegation blob; "
    "text adds operator-facing deployment guidance.",
)
@click.option(
    "--register", "do_register", is_flag=True, default=False,
    help="Sprint 796 — after minting, POST the delegation to "
    "/api/v1/auth/wallet/bind so the daemon's binding store "
    "records it. Closes the round-trip so the device shows up in "
    "`prsm wallet devices list`. Without this flag, the operator "
    "must manually POST to /bind themselves.",
)
@click.option(
    "--write", "do_write", is_flag=True, default=False,
    help="Sprint 797 — write the delegation JSON to "
    "~/.prsm/operator_delegation.json (the daemon's default "
    "lookup path). chmod 600 since the file is a signing "
    "artifact. Use --write-path to override the destination.",
)
@click.option(
    "--write-path", "write_path", default=None,
    help="Override the destination for --write (otherwise "
    "~/.prsm/operator_delegation.json).",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL (only used with --register).",
)
def wallet_devices_add(
    node_id: str, output_format: str,
    do_register: bool, do_write: bool,
    write_path: Optional[str],
    api_url_override: Optional[str],
) -> None:
    """Mint an EIP-191 delegation authorizing a new device under
    this wallet.

    Reads PRIVATE_KEY from env. Builds the canonical sprint-786
    binding message for (wallet, node_id, now-ISO), EIP-191-signs
    it, and emits the delegation blob.

    Copy the JSON output to the new device's
    `PRSM_OPERATOR_DELEGATION` env or file. Sprint 788's
    verify_operator_delegation_blob will then accept the device's
    operator_address claim across the network.
    """
    import os as _os
    import json as _json
    from datetime import datetime as _dt, timezone as _tz

    pk = (_os.environ.get("PRIVATE_KEY") or "").strip()
    if not pk:
        console.print(
            "[red]PRIVATE_KEY env unset.[/red] Export your "
            "operator wallet's private key + retry:\n"
            "  [bold]export PRIVATE_KEY=0x...[/bold]"
        )
        raise SystemExit(2)

    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        from prsm.interface.onboarding.wallet_binding import (
            build_binding_message,
        )
    except ImportError as exc:
        console.print(f"[red]eth_account import failed:[/red] {exc}")
        raise SystemExit(2)

    try:
        acct = Account.from_key(pk)
    except Exception as exc:
        console.print(
            f"[red]PRIVATE_KEY invalid:[/red] {exc}"
        )
        raise SystemExit(2)

    issued_at_iso = _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    msg = build_binding_message(acct.address, node_id, issued_at_iso)
    signed = acct.sign_message(encode_defunct(text=msg))
    blob = {
        "wallet_address": acct.address,
        "node_id_hex": node_id,
        "issued_at_iso": issued_at_iso,
        "signature": signed.signature.to_0x_hex(),
    }

    # Sprint 797 — optionally persist the blob to a file so the
    # daemon's _merge_operator_delegation picks it up
    # automatically on next start. --write uses
    # ~/.prsm/operator_delegation.json (the daemon's default);
    # --write-path overrides.
    written_path: Optional[str] = None
    if do_write or write_path is not None:
        from pathlib import Path as _Path
        if write_path is not None:
            resolved = _Path(write_path)
        else:
            resolved = (
                _Path.home() / ".prsm" / "operator_delegation.json"
            )
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(_json.dumps(blob, indent=2))
            # chmod 600 — this is a signing artifact, even though
            # the message has already been signed (a leaked file
            # could be replayed by a future-dated reader, and
            # leaks reveal the operator's node_id ↔ wallet binding
            # to anyone reading the disk).
            try:
                resolved.chmod(0o600)
            except OSError:
                pass  # Windows / non-posix: skip
            written_path = str(resolved)
        except OSError as exc:
            console.print(
                f"[red]Failed to write delegation to "
                f"{resolved}:[/red] {exc}"
            )
            raise SystemExit(2)

    # Sprint 796 — optional round-trip register against the
    # daemon's /api/v1/auth/wallet/bind so the binding shows up
    # in `wallet devices list`. Without this flag, the operator
    # has to manually POST the JSON themselves.
    registration_resp: Optional[Dict[str, Any]] = None
    if do_register:
        import httpx as _httpx
        url = _api_url_from_creds(api_url_override)
        endpoint = f"{url}/api/v1/auth/wallet/bind"
        bind_body = {
            "wallet_address": acct.address,
            "node_id_hex": node_id,
            "signature": blob["signature"],
            "issued_at": issued_at_iso,
        }
        try:
            resp = _httpx.post(
                endpoint, json=bind_body, timeout=10.0,
            )
        except Exception as exc:
            # Daemon unreachable — exit 2 BUT still print the
            # delegation so the operator can save it locally +
            # retry the register without re-signing.
            if output_format == "json":
                click.echo(_json.dumps({
                    "delegation": blob,
                    "registration": None,
                    "error": f"daemon unreachable: {exc}",
                }, indent=2))
            else:
                console.print(_json.dumps(blob, indent=2))
                console.print(
                    f"[red]Daemon unreachable at {endpoint}[/red] — "
                    f"{exc}. Delegation above is still valid — save "
                    "it locally and retry with --register once the "
                    "daemon is reachable."
                )
            raise SystemExit(2)
        if resp.status_code != 200:
            if output_format == "json":
                click.echo(_json.dumps({
                    "delegation": blob,
                    "registration": None,
                    "status": resp.status_code,
                    "detail": resp.text,
                }, indent=2))
            else:
                console.print(_json.dumps(blob, indent=2))
                console.print(
                    f"[red]Daemon registration failed "
                    f"({resp.status_code}):[/red] {resp.text}"
                )
            raise SystemExit(1)
        registration_resp = resp.json()

    if output_format == "json":
        if do_register:
            click.echo(_json.dumps({
                "delegation": blob,
                "registration": registration_resp,
            }, indent=2))
        else:
            click.echo(_json.dumps(blob, indent=2))
        return

    console.print(
        f"[bold green]Delegation minted[/bold green] for node "
        f"[cyan]{node_id}[/cyan] under wallet [cyan]{acct.address}"
        f"[/cyan]:"
    )
    console.print(_json.dumps(blob, indent=2))
    if do_register and registration_resp is not None:
        console.print(
            f"\n[bold green]Registered with daemon[/bold green] — "
            f"bound at unix={registration_resp.get('bound_at_unix')}. "
            "Run [bold]prsm wallet devices list[/bold] to confirm."
        )
    elif written_path is not None:
        # Sprint 823 — when --write/--write-path already wrote the
        # file, drop the "Save the JSON above" step (operator
        # would otherwise duplicate-save the same content).
        console.print(
            f"\n[dim]Delegation written to[/dim] "
            f"[bold]{written_path}[/bold]\n"
            "  [dim]Next:[/dim] restart the daemon — sprint-797 "
            "auto-loads [bold]~/.prsm/operator_delegation.json"
            "[/bold] from that path, or set "
            "[bold]PRSM_OPERATOR_DELEGATION_FILE[/bold] to "
            "override.\n"
            "  [dim]Tip:[/dim] add [bold]--register[/bold] to "
            "also auto-record this binding with the daemon."
        )
    else:
        console.print(
            "\n[dim]Deploy to the new device:[/dim]\n"
            "  1. Save the JSON above as e.g. "
            "[bold]~/.prsm/operator_delegation.json[/bold]\n"
            "  2. Export [bold]PRSM_OPERATOR_DELEGATION="
            "$(cat ~/.prsm/operator_delegation.json)[/bold]\n"
            "  3. Restart the daemon. operator_address will now be "
            "trusted across the network.\n"
            "  [dim]Tip:[/dim] add [bold]--write[/bold] to skip "
            "the manual save (writes to "
            "[bold]~/.prsm/operator_delegation.json[/bold] which "
            "sprint-797 auto-loads on daemon start), or "
            "[bold]--register[/bold] to auto-record with the "
            "daemon."
        )


@wallet_devices.command("list")
@click.option(
    "--wallet", "wallet_address", required=True,
    help="Wallet address (0x-prefixed 42-char hex) to list devices for.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL (default: localhost:8000)",
)
def wallet_devices_list(
    wallet_address: str, output_format: str,
    api_url_override: Optional[str],
) -> None:
    """List all node_ids bound to this wallet (multi-device roster).

    Sprint 790 — queries the daemon's
    GET /api/v1/auth/wallet/bindings endpoint and renders one
    row per binding. Exit 0 on success, 2 on daemon-unreachable.
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/api/v1/auth/wallet/bindings"
    try:
        resp = _httpx.get(
            endpoint,
            params={"wallet_address": wallet_address},
            timeout=10.0,
        )
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]bindings query failed "
                f"({resp.status_code}):[/red] {resp.text}"
            )
        raise SystemExit(1)
    bindings = resp.json()

    if output_format == "json":
        click.echo(_json.dumps(bindings, indent=2))
        return

    if not bindings:
        console.print(
            f"[dim]No devices bound to wallet {wallet_address}.[/dim]\n"
            "Use [bold]prsm wallet devices add --node-id <hex>[/bold] "
            "to mint a delegation for a device, then POST it to the "
            "daemon's /api/v1/auth/wallet/bind endpoint."
        )
        return

    console.print(
        f"[bold]{len(bindings)} device(s) bound to "
        f"[cyan]{wallet_address}[/cyan]:[/bold]"
    )
    for b in bindings:
        console.print(
            f"  • [cyan]{b['node_id_hex']}[/cyan]  "
            f"[dim](bound at unix={b['bound_at_unix']})[/dim]"
        )


@wallet_devices.command("earnings")
@click.option(
    "--wallet", "wallet_address", required=True,
    help="Wallet address (0x-prefixed 42-char hex) to query.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
@click.option(
    "--api-url", "api_url_override", default=None,
    help="Override daemon URL",
)
def wallet_devices_earnings(
    wallet_address: str, output_format: str,
    api_url_override: Optional[str],
) -> None:
    """Sprint 792 — per-device FTNS earnings for this wallet.

    Queries GET /api/v1/auth/wallet/devices/earnings; renders a
    table of per-node-id credit + the roster total. Useful for
    spotting underperforming devices.

    Exit 0 on success, 1 on daemon error, 2 on unreachable.
    """
    import json as _json
    import httpx as _httpx
    url = _api_url_from_creds(api_url_override)
    endpoint = f"{url}/api/v1/auth/wallet/devices/earnings"
    try:
        resp = _httpx.get(
            endpoint,
            params={"wallet_address": wallet_address},
            timeout=10.0,
        )
    except Exception as exc:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False,
                "error": f"daemon unreachable: {exc}",
            }))
        else:
            console.print(
                f"[red]Daemon unreachable at {endpoint}[/red] — "
                f"{exc}"
            )
        raise SystemExit(2)
    if resp.status_code != 200:
        if output_format == "json":
            click.echo(_json.dumps({
                "ok": False, "status": resp.status_code,
                "detail": resp.text,
            }))
        else:
            console.print(
                f"[red]earnings query failed "
                f"({resp.status_code}):[/red] {resp.text}"
            )
        raise SystemExit(1)
    data = resp.json()
    if output_format == "json":
        click.echo(_json.dumps(data, indent=2))
        return

    earnings = data.get("earnings_by_node_id", {})
    total = data.get("total_ftns", "0")
    if not earnings:
        console.print(
            f"[dim]No earnings recorded for any device bound to "
            f"{wallet_address}.[/dim]\nEither no devices are "
            "bound (use [bold]prsm wallet devices list[/bold] to "
            "check) or no settled receipts have been produced "
            "yet."
        )
        return
    console.print(
        f"[bold]Per-device earnings for "
        f"[cyan]{wallet_address}[/cyan]:[/bold]"
    )
    for node_id, amount in earnings.items():
        console.print(
            f"  • [cyan]{node_id}[/cyan]  "
            f"[green]{amount}[/green] FTNS"
        )
    console.print(
        f"[bold]Total:[/bold] [green]{total}[/green] FTNS"
    )


@wallet_devices.command("verify")
@click.option(
    "--node-id", "node_id", required=True,
    help="Claimed node_id (32-char hex)",
)
@click.option(
    "--operator", "operator_address", required=True,
    help="Claimed operator_address (0x-prefixed 42-char hex)",
)
@click.option(
    "--delegation-file", "delegation_file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the delegation JSON blob to verify.",
)
def wallet_devices_verify(
    node_id: str, operator_address: str, delegation_file: str,
) -> None:
    """Verify a delegation JSON without requiring a key.

    Exits 0 on PASS, 1 on FAIL. Use this in CI / pre-deploy
    checks to confirm a device's delegation is well-formed
    before shipping the config.
    """
    import json as _json
    from pathlib import Path as _P
    try:
        text = _P(delegation_file).read_text()
        blob = _json.loads(text)
    except Exception as exc:
        console.print(
            f"[red]FAIL[/red] — cannot parse delegation JSON: {exc}"
        )
        raise SystemExit(1)

    from prsm.node.operator_delegation import (
        verify_operator_delegation_blob,
    )
    ok = verify_operator_delegation_blob(
        node_id=node_id,
        operator_address=operator_address,
        delegation=blob,
    )
    if ok:
        console.print(
            f"[green]PASS[/green] — delegation for node "
            f"[cyan]{node_id[:16]}…[/cyan] under operator "
            f"[cyan]{operator_address}[/cyan] verifies cleanly."
        )
        return

    console.print(
        f"[red]FAIL[/red] — delegation does NOT verify for "
        f"(node={node_id[:16]}…, operator={operator_address}). "
        f"Check the signing key, node_id, and operator address "
        f"all match what was originally signed."
    )
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# join-testnet command (T5)
#
# One-command onboarding for new testnet users. Generates a fresh burner
# keypair, persists it to ~/.prsm/testnet-deployer.env (chmod 600), and
# prints next-step guidance for funding + running.
# ---------------------------------------------------------------------------


@main.command("join-testnet")
@click.option(
    "--force", is_flag=True, default=False,
    help="Overwrite existing ~/.prsm/testnet-deployer.env",
)
@click.option(
    "--no-print-key", is_flag=True, default=False,
    help="Don't echo the private key to stdout (still saved to file)",
)
def join_testnet(force: bool, no_print_key: bool):
    """Generate a fresh testnet burner wallet + onboarding env file.

    \b
    Creates ~/.prsm/testnet-deployer.env with:
      - PRIVATE_KEY     (freshly generated, 64-char hex)
      - BASE_SEPOLIA_RPC_URL  (default: https://sepolia.base.org)
      - PRSM_NETWORK    (testnet)
      - PRSM_PROVENANCE_REGISTRY_ADDRESS  (from networks.py)
      - FTNS_TOKEN_ADDRESS  (from networks.py)
      - PRSM_ROYALTY_DISTRIBUTOR_ADDRESS  (from networks.py)
      - PRSM_ONCHAIN_PROVENANCE=1

    \b
    The wallet starts with zero balance. Fund it with Base Sepolia ETH
    (Coinbase Developer Platform faucet, etc.) and request testnet-FTNS
    via the project's #testnet-faucet channel before running.

    \b
    Example:
        prsm join-testnet
        # then fund the printed address...
        source ~/.prsm/testnet-deployer.env
        prsm wallet info --network testnet
    """
    import os
    import stat
    from pathlib import Path
    from eth_account import Account
    from prsm.config.networks import get_network_config

    cfg = get_network_config("testnet")

    env_dir = Path.home() / ".prsm"
    env_dir.mkdir(exist_ok=True)
    env_path = env_dir / "testnet-deployer.env"

    if env_path.exists() and not force:
        console.print(
            f"❌ {env_path} already exists. Pass --force to overwrite.",
            style="red")
        console.print(
            "  (or just `source ~/.prsm/testnet-deployer.env` to reuse "
            "your existing burner)", style="dim")
        raise SystemExit(1)

    # Generate fresh burner
    account = Account.create()
    private_key = account.key.hex()
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    address = account.address

    # Build env file content
    contract_lines = []
    if cfg.ftns_token:
        contract_lines.append(f"export FTNS_TOKEN_ADDRESS={cfg.ftns_token}")
    if cfg.provenance_registry:
        contract_lines.append(
            f"export PRSM_PROVENANCE_REGISTRY_ADDRESS={cfg.provenance_registry}")
    if cfg.royalty_distributor:
        contract_lines.append(
            f"export PRSM_ROYALTY_DISTRIBUTOR_ADDRESS={cfg.royalty_distributor}")

    env_content = "\n".join([
        "# PRSM testnet burner — generated by `prsm join-testnet`",
        "# DO NOT commit. DO NOT use for mainnet. Testnet has zero monetary value.",
        f"# Address: {address}",
        "",
        f"export PRIVATE_KEY={private_key}",
        f"export FTNS_WALLET_PRIVATE_KEY={private_key}",
        f"export BASE_SEPOLIA_RPC_URL={cfg.rpc_url_default}",
        "export PRSM_NETWORK=testnet",
        "export PRSM_ONCHAIN_PROVENANCE=1",
        *contract_lines,
        "",
    ])

    # sp1266 — atomic 0o600 write of the wallet env file (it holds the burner PRIVATE key);
    # write_text-then-chmod left a world-readable TOCTOU window.
    from prsm.node.identity import write_owner_only_file
    write_owner_only_file(env_path, env_content)

    console.print(f"\n[bold green]✓ Testnet wallet created[/bold green]")
    console.print(f"\nAddress:    [bold]{address}[/bold]")
    if not no_print_key:
        console.print(f"Private key: {private_key}", style="dim")
    console.print(f"Env file:   {env_path} (mode 600)")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Fund the wallet with Base Sepolia ETH (gas):")
    console.print(f"     • Coinbase faucet: https://portal.cdp.coinbase.com/faucet")
    console.print(f"     • Verify at: {cfg.explorer_url}/address/{address}")
    console.print()
    console.print("  2. Request testnet-FTNS (ask in project Discord channel)")
    console.print()
    console.print("  3. Activate the env + check status:")
    console.print(f"     source ~/.prsm/testnet-deployer.env")
    console.print(f"     prsm wallet info --network testnet")
    console.print()
    console.print("  4. Start a node + upload content / earn:")
    console.print(f"     prsm node start --network testnet  (T4 — bootstrap "
                  "wiring still pending)")
    console.print(f"     prsm storage upload <file>")
    console.print()
    if cfg.notes:
        console.print("[dim]Reminders:[/dim]")
        for note in cfg.notes:
            console.print(f"  ℹ {note}", style="dim")
    console.print()


@main.group("bootstrap-server")
def bootstrap_server():
    """Manage / probe a bootstrap server you are running.

    Sprint 390 — operator-trifecta third corner.
    Complements `prsm node bootstrap` (this node's
    registration state) and `prsm node bootstrap-test`
    (probes canonical fleet from your perspective) with
    a probe of your OWN bootstrap droplet's HTTP control
    surface.
    """
    pass


@bootstrap_server.command("status")
@click.option(
    "--host", default="127.0.0.1", show_default=True,
    help="Bootstrap server host (default localhost; pass a "
         "hostname / IP for remote probes).",
)
@click.option(
    "--port", default=8000, type=int, show_default=True,
    help="Bootstrap server API port (BootstrapConfig.api_port).",
)
@click.option(
    "--timeout", default=5.0, type=float, show_default=True,
    help="HTTP timeout in seconds.",
)
@click.option(
    "--detailed", is_flag=True,
    help="Also fetch /health/detailed (sprint 392 "
         "per-subsystem readiness probe) and render the "
         "subsystem table.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def bootstrap_server_status(host, port, timeout, detailed, output_format):
    """One-screen ops summary of a running bootstrap server.

    Hits /health and /metrics on the bootstrap server's
    HTTP API and renders a color-coded summary. Defaults
    to localhost:8000 — SSH to your droplet and run this.
    Pass --host for remote probes.
    """
    import asyncio
    import json as _json

    # Late-import so click pulls in nothing at import time
    from prsm.cli_helpers import bootstrap_server_probe as bsp_module

    probe = asyncio.run(
        bsp_module.fetch_server_status(
            host=host, port=port, timeout_seconds=timeout,
            include_subsystems=detailed,
        )
    )

    if output_format == "json":
        console.print(_json.dumps(probe.to_dict(), indent=2))
        sys.exit(0 if probe.status == bsp_module.ProbeStatus.OK else 1)

    # ── Text rendering ────────────────────────────────────
    status_markers = {
        bsp_module.ProbeStatus.OK: "[green]✓ healthy[/green]",
        bsp_module.ProbeStatus.PARTIAL: "[yellow]⚠ partial — metrics unavailable[/yellow]",
        bsp_module.ProbeStatus.CONNECT_FAIL: "[red]✗ connect refused[/red]",
        bsp_module.ProbeStatus.TIMEOUT: "[red]✗ timeout[/red]",
        bsp_module.ProbeStatus.HTTP_ERROR: "[red]✗ http error[/red]",
        bsp_module.ProbeStatus.UNKNOWN: "[red]✗ unknown[/red]",
    }
    marker = status_markers.get(probe.status, "[red]?[/red]")
    console.print(
        f"[bold]PRSM Bootstrap Server Status[/bold] — {marker}"
    )
    console.print(
        f"  target: [cyan]{probe.host}:{probe.port}[/cyan]"
    )
    if probe.error:
        console.print(f"  error:  [red]{probe.error}[/red]")

    if probe.health:
        console.print()
        console.print("[bold]Health[/bold]")
        for k, v in probe.health.items():
            console.print(f"  {k}: {v}")

    if probe.metrics:
        console.print()
        console.print("[bold]Metrics[/bold]")
        # Render flat scalars first, label-dicts last
        flat = {k: v for k, v in probe.metrics.items()
                if not isinstance(v, dict)}
        labeled = {k: v for k, v in probe.metrics.items()
                   if isinstance(v, dict)}
        for k, v in flat.items():
            console.print(f"  {k}: {v}")
        for k, label_dict in labeled.items():
            if not label_dict:
                continue
            console.print(f"  {k}:")
            for label, value in label_dict.items():
                console.print(f"    {label}: {value}")

    if probe.health_detailed:
        console.print()
        agg = probe.health_detailed.get("status", "?")
        agg_color = {
            "healthy": "green",
            "degraded": "yellow",
            "unhealthy": "red",
        }.get(agg, "white")
        console.print(
            f"[bold]Subsystems[/bold] — aggregate: "
            f"[{agg_color}]{agg}[/{agg_color}]"
        )
        for sub_name, sub_data in (
            probe.health_detailed.get("subsystems") or {}
        ).items():
            sub_status = sub_data.get("status", "?")
            sub_color = {
                "healthy": "green",
                "degraded": "yellow",
                "stale": "red",
            }.get(sub_status, "white")
            age = sub_data.get("last_heartbeat_age_seconds")
            age_str = (
                f"{age:.0f}s" if isinstance(age, (int, float))
                else "—"
            )
            console.print(
                f"  {sub_name}: "
                f"[{sub_color}]{sub_status}[/{sub_color}] "
                f"(age {age_str})"
            )

    sys.exit(0 if probe.status == bsp_module.ProbeStatus.OK else 1)


@node.command("fiat-readiness")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_fiat_readiness(output_format):
    """Probe Phase 5 fiat-surface activation readiness.

    Sprint 422 — wraps sprint-285's
    `check_fiat_surface_health()` for operator CLI use.
    Run this BEFORE attempting Phase 5 activation (see
    `docs/operations/phase-5-fiat-surface-activation-
    runbook.md`) to verify your env is ready.

    Exit code: 0 = OK or WARN-only findings; non-zero =
    at least one ERROR finding (activation will fail).

    Default `text` format renders a color-coded findings
    table with remediation hints. `json` format for ops
    automation (parseable on both success + failure).
    """
    import json as _json
    import os

    from prsm.economy.web3.fiat_surface_health import (
        check_fiat_surface_health,
    )

    findings = check_fiat_surface_health(os.environ)

    has_error = any(
        getattr(f.severity, "value", str(f.severity)).lower()
        == "error"
        for f in findings
    )
    has_warn = any(
        getattr(f.severity, "value", str(f.severity)).lower()
        == "warn"
        for f in findings
    )
    if has_error:
        overall = "error"
    elif has_warn:
        overall = "warn"
    else:
        overall = "ok"

    if output_format == "json":
        payload = {
            "overall_status": overall,
            "findings": [
                {
                    "severity": getattr(
                        f.severity, "value", str(f.severity),
                    ),
                    "cause": f.cause,
                    "remediation": f.remediation,
                }
                for f in findings
            ],
        }
        # Plain stdout — Rich's console.print injects ANSI
        # control chars that break JSON parsers downstream.
        click.echo(_json.dumps(payload, indent=2))
        sys.exit(0 if not has_error else 1)

    # Text rendering
    if overall == "ok":
        console.print(
            "[green]✓ Phase 5 fiat surface ready — OK[/green] "
            "[dim](no findings)[/dim]"
        )
        sys.exit(0)

    marker_map = {
        "error": "[red]✗ ERROR[/red]",
        "warn": "[yellow]⚠ WARN[/yellow]",
    }
    overall_marker = marker_map.get(
        overall, "[white]?[/white]"
    )
    console.print(
        f"[bold]Phase 5 fiat-readiness[/bold] — "
        f"{overall_marker}"
    )
    console.print()
    for f in findings:
        sev = getattr(f.severity, "value", str(f.severity)).lower()
        marker = marker_map.get(sev, "[white]?[/white]")
        console.print(f"{marker} [bold]{f.cause}[/bold]")
        console.print(f"  [dim]{f.remediation}[/dim]")
        console.print()

    sys.exit(0 if not has_error else 1)


# ──────────────────────────────────────────────────────────────
# Sprint 638 — `prsm node models list` discovers what's registered.
# Pre-638 operators using `prsm node infer` had to know a model_id
# ahead of time; nothing surfaced the local registry contents from
# the CLI. This command reads PRSM_MODEL_REGISTRY_ROOT, enumerates
# every model + dumps manifest metadata (publisher, shard count,
# published_at, layer ranges). Local-only for now — peer-advertised
# models will come in a follow-on sprint when there's a protocol-
# level "what do you serve?" message.
# ──────────────────────────────────────────────────────────────


@node.command("models")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--registry-root", default=None,
    help="Override PRSM_MODEL_REGISTRY_ROOT for one-off probes.",
)
def node_models_list_cli(output_format: str, registry_root: Optional[str]):
    """List models in the local FilesystemModelRegistry.

    Sprint 638 — closes the discovery gap from sprint 633's
    `prsm node infer`. Operators previously had to know a model_id
    ahead of time; this command surfaces every registered model
    with its publisher identity, shard count, and per-shard layer
    coverage so operators can pick a target for inference.

    Default registry root is read from PRSM_MODEL_REGISTRY_ROOT
    (the same path the daemon uses at runtime); override with
    --registry-root for one-off probes against alternate roots.

    Exit code: 0 when at least one model is present; 1 when the
    registry is empty or unreachable so scripts can branch on it.
    """
    import json as _json
    import os as _os
    import sys as _sys

    from prsm.compute.model_registry.registry import (
        FilesystemModelRegistry,
    )

    root = registry_root or _os.environ.get("PRSM_MODEL_REGISTRY_ROOT", "")
    if not root:
        if output_format == "json":
            click.echo(_json.dumps({
                "error": "registry_root_unset",
                "models": [],
            }, indent=2))
        else:
            console.print(
                "[red]✗ Registry root not configured[/red]\n"
                "[dim]Set PRSM_MODEL_REGISTRY_ROOT or pass "
                "--registry-root to point at a FilesystemModelRegistry "
                "directory.[/dim]"
            )
        _sys.exit(1)

    try:
        registry = FilesystemModelRegistry(root=root)
    except Exception as exc:  # noqa: BLE001
        if output_format == "json":
            click.echo(_json.dumps({
                "error": f"{type(exc).__name__}: {exc}",
                "registry_root": root,
                "models": [],
            }, indent=2))
        else:
            console.print(
                f"[red]✗ Failed to open registry at {root!r}[/red]: "
                f"{type(exc).__name__}: {exc}"
            )
        _sys.exit(1)

    model_ids = registry.list_models()
    summaries = []
    for model_id in model_ids:
        try:
            manifest = registry.get_manifest(model_id)
        except Exception as exc:  # noqa: BLE001
            summaries.append({
                "model_id": model_id,
                "error": f"manifest load failed: "
                         f"{type(exc).__name__}: {exc}",
            })
            continue
        layer_ranges = []
        for s in manifest.shards:
            layer_ranges.append(list(s.layer_range))
        summaries.append({
            "model_id": manifest.model_id,
            "model_name": manifest.model_name,
            "publisher_node_id": manifest.publisher_node_id,
            "total_shards": manifest.total_shards,
            "published_at": manifest.published_at,
            "schema_version": manifest.schema_version,
            "layer_ranges": layer_ranges,
        })

    if output_format == "json":
        click.echo(_json.dumps({
            "registry_root": root,
            "total": len(summaries),
            "models": summaries,
        }, indent=2))
        _sys.exit(0 if summaries else 1)

    if not summaries:
        console.print(
            f"[yellow]No models registered[/yellow] at "
            f"[cyan]{root}[/cyan]\n"
            f"[dim]Register a model via the registry API or "
            f"copy a published-manifest directory into this root.[/dim]"
        )
        _sys.exit(1)

    console.print(
        f"[dim]Registry root: {root}[/dim]\n"
        f"[bold]{len(summaries)} model(s) registered:[/bold]"
    )
    for s in summaries:
        if "error" in s:
            console.print(
                f"  [red]✗ {s['model_id']}[/red]: {s['error']}"
            )
            continue
        # Defense: layer_ranges may all be (0,0) for legacy
        # manifests pre-sprint 627; render compactly when sentinel.
        non_sentinel = [r for r in s["layer_ranges"] if r != [0, 0]]
        layer_summary = (
            ", ".join(f"[{r[0]}, {r[1]})" for r in non_sentinel)
            if non_sentinel
            else "(layer ranges unset — pre-sprint-627 manifest)"
        )
        console.print(
            f"  [green]●[/green] [bold]{s['model_id']}[/bold] "
            f"([dim]name={s['model_name']!r}[/dim])\n"
            f"    publisher: [cyan]{s['publisher_node_id'][:16]}...[/cyan]\n"
            f"    shards: {s['total_shards']}  "
            f"layers: {layer_summary}\n"
            f"    schema_v{s['schema_version']}, "
            f"published_at {s['published_at']:.0f}"
        )
    _sys.exit(0)


# ──────────────────────────────────────────────────────────────
# Sprint 633 — `prsm node infer` operator CLI for the live P2P
# inference path (sprint 628's multi-token GPT-2 demo wrapped as
# a first-class operator command). Closes the dogfood gap where
# the headline demo required hand-running a hardcoded Python
# script — now any operator with a running fleet + registered
# model can drive the full mainnet-anchor-verified inference
# loop directly from the CLI.
# ──────────────────────────────────────────────────────────────


@node.command("infer")
@click.option(
    "--prompt", required=True, type=str,
    help="Initial prompt text",
)
@click.option(
    "--model", default="gpt2", show_default=True,
    help="Model id (must be registered + available on at least one peer)",
)
@click.option(
    "--max-tokens", "-n", type=int, default=10, show_default=True,
    help="Number of tokens to generate",
)
@click.option(
    "--stage-peer-id", default=None,
    help="Peer ID of the layer-stage server. If omitted, uses the "
    "first connected peer. Ignored when --stages is set.",
)
@click.option(
    "--stages", "stages_spec", multiple=True,
    help="Sprint 668 — multi-stage chain spec. Format: "
    "'lo-hi:peer_id' (e.g., '0-6:peer-A'). Pass --stages multiple "
    "times for multi-stage. Settler mints distinct HandoffTokens "
    "per stage with chain_stage_index ascending; each stage signs "
    "its output. Single-host smoke test: '--stages 0-6:peer-X "
    "--stages 6-12:peer-X' (same peer twice). When --stages is "
    "set, --stage-peer-id is ignored.",
)
@click.option(
    "--api", "api_url", default="http://127.0.0.1:8000", show_default=True,
    help="Local PRSM daemon API URL",
)
@click.option(
    "--timeout", type=float, default=120.0, show_default=True,
    help="Per-token request timeout (seconds)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--save-receipts", "save_receipts_path", type=click.Path(),
    default=None,
    help="Append one JSON-line per token to this file. Each line "
    "carries the stage signature, stage_node_id, request_id, and "
    "activation hash — enough for offline verification against the "
    "PublisherKeyAnchor.",
)
@click.option(
    "--output-file", "output_file", type=click.Path(),
    default=None,
    help="Sprint 666 — write ONLY the generated text (no per-token "
    "logs, no receipts) to this file. The text is the full prompt "
    "+ generated tokens. Separate from --save-receipts so automation "
    "pipelines can grab clean text without parsing log lines.",
)
@click.option(
    "--temperature", type=float, default=None,
    help="Sprint 639 — sampling temperature. Omit (default) for "
    "greedy argmax (deterministic, audit-chain-friendly). Values "
    "between 0 and ~2 produce increasingly random output. The "
    "receipt's `sampling_mode` field records the choice so the "
    "downstream `verify-receipts --check-chain` knows to skip "
    "the argmax-vs-next_token_id invariant for non-greedy runs.",
)
@click.option(
    "--top-k", type=int, default=None,
    help="Sprint 639 — restrict sampling to the top-K tokens by "
    "logit magnitude. Pairs with --temperature; combining the two "
    "is the typical 'creative but not chaotic' setting. Ignored "
    "when --temperature is unset.",
)
@click.option(
    "--seed", type=int, default=None,
    help="Sprint 639 — RNG seed for sampling. Recorded in the "
    "receipt so a verifier with the same seed can re-derive the "
    "sample deterministically (a partial mitigation for the "
    "audit-chain weakening that comes with non-greedy sampling).",
)
@click.option(
    "--warm-up/--no-warm-up", default=False,
    help="Sprint 643 — fire a throw-away forward to warm the stage "
    "peer's HF model cache before the real run begins. Eliminates "
    "the ~15s first-token cold-start hit so wall-clock matches "
    "steady-state per-token latency. Warm-up tokens are NOT "
    "written to --save-receipts.",
)
@click.option(
    "--no-seed-warning", is_flag=True, default=False,
    help="Sprint 651 — suppress the audit-chain-weakness warning "
    "when --temperature is set without --seed. Use this only for "
    "intentionally non-audited runs (e.g., exploratory generation "
    "where you don't plan to verify-receipts --strict).",
)
@click.option(
    "--stop", "stop_strings", multiple=True,
    help="Sprint 664 — stop generation early when any of these "
    "strings appears in the generated tail. Multi-value: pass "
    "multiple times to stop on any-of. Common: --stop '.' --stop "
    "$'\\n' --stop '###'. Stops AFTER appending the matched token "
    "so the stop marker appears in the output. Greedy + sampled "
    "modes both honor stop strings.",
)
@click.option(
    "--stop-on-eos/--no-stop-on-eos", default=True,
    help="Sprint 664 — stop generation when the model emits its "
    "EOS token (gpt2: <|endoftext|>, id 50256; llama: </s>). "
    "Default on; --no-stop-on-eos lets you keep generating past "
    "the model's natural end (rarely useful — generates noise).",
)
@click.option(
    "--incremental/--no-incremental", default=False,
    help="Sprint 662 — engage the KV-cache fast path (sprints 654-660). "
    "Stable request_id + decode_mode=INCREMENTAL across the run; "
    "after the cold first request, each subsequent token forwards "
    "only the 1 new position through cached attention. Live-attested "
    "~5x per-token speedup vs default PREFILL path. Requires the "
    "stage peer's daemon to have PRSM_PARALLAX_KV_CACHE_ENABLED=1 "
    "AND a runner supporting run_layer_range_incremental (sprint 656 "
    "added HuggingFaceLayerSliceRunner support). Receipts record "
    "decode_mode='incremental' so sprint-661 C3 invariant doesn't "
    "false-positive on the shared request_id.",
)
def node_infer_cli(
    prompt: str,
    model: str,
    max_tokens: int,
    stage_peer_id: Optional[str],
    api_url: str,
    timeout: float,
    output_format: str,
    save_receipts_path: Optional[str],
    temperature: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    warm_up: bool,
    no_seed_warning: bool,
    incremental: bool,
    stop_strings: tuple,
    stop_on_eos: bool,
    output_file: Optional[str],
    stages_spec: tuple,
):
    """Generate tokens via the live PRSM P2P inference path.

    Wraps the sprint-628 demo: prompt → tokenize+embed locally →
    sign + ship each forward to a stage peer → receive logits →
    argmax → append → repeat. Every dispatch verifies the
    settler's pubkey against the live PublisherKeyAnchor on Base
    mainnet (sprint 621 deploy).

    Required prerequisites (this command does NOT bootstrap them):
      • Local daemon running with PRSM_PARALLAX_TRUST_STACK_KIND=production
      • At least one peer connected that serves the requested model
        via LayerStageServer (PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=huggingface)
      • The model registered on both sides (filesystem registry)

    Example:
        prsm node infer --prompt "The capital of France is" -n 10
    """
    import base64
    import json as _json
    import sys as _sys
    import time as _time

    import httpx as _httpx
    import numpy as _np

    from prsm.compute.chain_rpc.protocol import (
        ContentTier, HandoffToken, PrivacyLevel, RunLayerSliceRequest,
        encode_message, parse_message,
    )
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity

    # ── Resolve stage peer ──
    try:
        peers_resp = _httpx.get(f"{api_url}/peers", timeout=5.0)
        peers_resp.raise_for_status()
    except Exception as exc:
        console.print(
            f"[red]✗ Failed to reach local daemon at {api_url}[/red]: "
            f"{type(exc).__name__}: {exc}\n"
            f"[dim]Start the daemon first: prsm node start[/dim]"
        )
        _sys.exit(1)
    peers_data = peers_resp.json()
    connected = peers_data.get("connected", [])
    # Sprint 668 — parse --stages spec if provided. Format:
    # 'lo-hi:peer_id'. Validates each entry; sets up `stages` list
    # of (lo, hi, peer_id) tuples that the generation loop uses.
    # When --stages is provided, --stage-peer-id is ignored.
    stages_list = []
    if stages_spec:
        for spec in stages_spec:
            if ":" not in spec or "-" not in spec.split(":", 1)[0]:
                console.print(
                    f"[red]✗ Invalid --stages spec[/red] {spec!r}; "
                    f"format is 'lo-hi:peer_id' (e.g., '0-6:peer-A')"
                )
                _sys.exit(1)
            range_part, peer = spec.split(":", 1)
            try:
                lo_str, hi_str = range_part.split("-", 1)
                lo, hi = int(lo_str), int(hi_str)
            except ValueError:
                console.print(
                    f"[red]✗ Invalid --stages layer range[/red] in "
                    f"{spec!r}; lo/hi must be integers"
                )
                _sys.exit(1)
            if lo >= hi or lo < 0:
                console.print(
                    f"[red]✗ Invalid --stages layer range[/red] in "
                    f"{spec!r}; require 0 <= lo < hi"
                )
                _sys.exit(1)
            stages_list.append((lo, hi, peer.strip()))
        if output_format == "text":
            console.print(
                f"[dim]Multi-stage chain: {len(stages_list)} stages — "
                f"{', '.join(f'[{s[0]},{s[1]}):{s[2][:8]}' for s in stages_list)}"
                f"[/dim]"
            )
    elif stage_peer_id is None:
        if not connected:
            console.print(
                "[red]✗ No connected peers[/red]. Need at least one "
                "peer to serve the inference stage.\n"
                "[dim]Check fleet symmetry with `prsm node info` or "
                "wait for bootstrap-mediated discovery.[/dim]"
            )
            _sys.exit(1)
        stage_peer_id = connected[0].get("peer_id")
        if not stage_peer_id:
            console.print("[red]✗ First connected peer has no peer_id[/red]")
            _sys.exit(1)

    # ── Load settler identity ──
    cfg = NodeConfig.load()
    settler = load_node_identity(cfg.identity_path)

    # ── Load tokenizer + embedding layer ──
    # Lazy import — operators without HF installed should still see a
    # clean error rather than a top-level ModuleNotFoundError at CLI
    # boot.
    try:
        import torch as _torch
        from transformers import (
            AutoModelForCausalLM as _AutoModel,
            AutoTokenizer as _AutoTok,
        )
    except ImportError as exc:
        console.print(
            f"[red]✗ HuggingFace deps missing[/red]: {exc}\n"
            f"[dim]Install with: pip install transformers torch[/dim]"
        )
        _sys.exit(1)

    if output_format == "text":
        console.print(
            f"[dim]Loading {model} for tokenize + embed...[/dim]"
        )
    try:
        # sp1285 — pin to an immutable HF commit when PRSM_MODEL_REVISIONS configures one
        # (supply-chain: don't trust the model repo's mutable default branch).
        from prsm.compute.inference.local_inference import resolve_model_revision
        _rev = resolve_model_revision(model)
        tok = _AutoTok.from_pretrained(model, revision=_rev)
        hf_model = _AutoModel.from_pretrained(
            model, torch_dtype=_torch.float32, revision=_rev,
        ).eval()
    except Exception as exc:
        console.print(
            f"[red]✗ Failed to load HF model {model!r}[/red]: "
            f"{type(exc).__name__}: {exc}"
        )
        _sys.exit(1)

    # ── Receipt sink (sprint 634) ──
    # When --save-receipts is set, open the file in append-mode at
    # the top of the loop. Each per-token write is a single JSON
    # line so the file is friendly to `jq` + grep-based post-
    # processing. Append rather than truncate so multi-run audit
    # trails accumulate naturally.
    import hashlib as _hashlib
    import os as _os_for_dir
    from pathlib import Path as _Path
    receipts_fh = None
    if save_receipts_path:
        save_p = _Path(save_receipts_path)
        try:
            save_p.parent.mkdir(parents=True, exist_ok=True)
            # Sprint 642 — auto-gzip when path ends in .gz. ~50%
            # size reduction on the activation_blob_b64 payload.
            # Operator gets compression by adding ".gz" to the
            # path; no separate flag needed.
            if save_p.suffix == ".gz":
                import gzip as _gzip
                receipts_fh = _gzip.open(
                    save_p, "at", encoding="utf-8",
                )
            else:
                receipts_fh = save_p.open("a", encoding="utf-8")
        except OSError as exc:
            console.print(
                f"[red]✗ Cannot open receipts file "
                f"{save_receipts_path!r}[/red]: {exc}"
            )
            _sys.exit(1)
        if output_format == "text":
            console.print(
                f"[dim]Recording per-token receipts to "
                f"{save_receipts_path}[/dim]"
            )

    # ── Audit-rigor advisory (sprint 651) ─────────────────
    # Symmetrical with sprint-650 verify-receipts --strict: warn
    # at GENERATION time so operators who'll audit later see the
    # weakness coming. --no-seed-warning suppresses for intentional
    # non-audited runs.
    if (
        temperature is not None
        and seed is None
        and not no_seed_warning
        and output_format == "text"
    ):
        console.print(
            "[yellow]⚠ Audit-chain advisory[/yellow]: "
            "--temperature is set but --seed is not. "
            "The resulting receipts will be UNVERIFIABLE_NON_GREEDY_NO_SEED "
            "under `prsm node verify-receipts --strict`. "
            "[dim]Pass --seed N for a fully-verifiable audit chain, "
            "or --no-seed-warning to suppress this advisory.[/dim]"
        )

    # ── Warm-up (sprint 643) ──────────────────────────────
    # Send one throw-away forward so the stage peer's HF model
    # cache is hot before we time the real run. Without this, the
    # first generation token takes ~15s while the droplet loads
    # gpt2-124M from disk. Doesn't count toward max_tokens; not
    # written to --save-receipts.
    if warm_up:
        if output_format == "text":
            console.print(
                f"[dim]Warming up stage peer's HF cache "
                f"(throw-away forward)...[/dim]"
            )
        warm_t0 = _time.time()
        try:
            warm_input_ids = tok.encode("test", return_tensors="pt")
            with _torch.no_grad():
                we = hf_model.transformer.wte(warm_input_ids)
                wp = hf_model.transformer.wpe(
                    _torch.arange(warm_input_ids.shape[-1]).unsqueeze(0),
                )
                warm_act = (we + wp).numpy()
            warm_req_id = f"prsm-cli-infer-warmup-{int(_time.time())}"
            warm_deadline = _time.time() + timeout
            warm_token = HandoffToken.sign(
                identity=settler, request_id=warm_req_id,
                chain_stage_index=0, chain_total_stages=1,
                deadline_unix=warm_deadline,
            )
            warm_n_layers = getattr(
                getattr(hf_model, "config", None),
                "num_hidden_layers",
                getattr(hf_model.config, "n_layer", 12),
            )
            warm_request = RunLayerSliceRequest(
                request_id=warm_req_id, model_id=model,
                layer_range=(0, warm_n_layers),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=warm_act.tobytes(),
                activation_shape=tuple(warm_act.shape),
                activation_dtype=str(warm_act.dtype),
                upstream_token=warm_token, deadline_unix=warm_deadline,
            )
            warm_bytes = encode_message(warm_request)
            _httpx.post(
                f"{api_url}/admin/chain-exec-ping",
                json={
                    "peer_id": stage_peer_id,
                    "payload_b64": base64.b64encode(
                        warm_bytes,
                    ).decode("ascii"),
                    "timeout": timeout - 5.0,
                },
                timeout=timeout,
            )
            warm_dt = _time.time() - warm_t0
            if output_format == "text":
                console.print(
                    f"[dim]  warm-up completed in {warm_dt:.1f}s[/dim]"
                )
        except Exception as exc:  # noqa: BLE001
            # Warm-up failure shouldn't kill the run — just log and
            # carry on; the real loop will hit the cold-start hit
            # like pre-643 behavior.
            console.print(
                f"[yellow]⚠ warm-up failed; continuing without it: "
                f"{type(exc).__name__}: {exc}[/yellow]"
            )

    # ── Generation loop ──
    text = prompt
    per_token_records = []
    overall_t0 = _time.time()
    # Sprint 662 — KV-cache fast path setup. When --incremental is
    # set, we maintain a stable request_id (used as the cache key
    # on the server) AND track token IDs explicitly so we can ship
    # ONLY the new token's embedding on hot iterations.
    if incremental:
        from prsm.compute.chain_rpc.protocol import DecodeMode
        cache_request_id = (
            f"prsm-cli-incremental-{int(_time.time())}-"
            f"{settler.node_id[:8]}"
        )
        token_ids_so_far = tok.encode(text, return_tensors="pt").squeeze(0).tolist()
    for step in range(max_tokens):
        step_t0 = _time.time()
        if incremental:
            cur_token_count = len(token_ids_so_far)
            with _torch.no_grad():
                if step == 0:
                    # Cold cache: send full prefix
                    send_ids = _torch.tensor([token_ids_so_far])
                    pos_offset = 0
                else:
                    # Hot cache: send only the last appended token
                    send_ids = _torch.tensor([[token_ids_so_far[-1]]])
                    pos_offset = cur_token_count - 1
                te = hf_model.transformer.wte(send_ids)
                positions = _torch.arange(
                    pos_offset, pos_offset + send_ids.shape[-1],
                ).unsqueeze(0)
                pe = hf_model.transformer.wpe(positions)
                activation = (te + pe).numpy()
        else:
            input_ids = tok.encode(text, return_tensors="pt")
            with _torch.no_grad():
                te = hf_model.transformer.wte(input_ids)
                pe = hf_model.transformer.wpe(
                    _torch.arange(input_ids.shape[-1]).unsqueeze(0),
                )
                activation = (te + pe).numpy()

        # Sprint 662 — request_id stable across the run when
        # --incremental (cache key); fresh per token otherwise.
        if incremental:
            request_id = cache_request_id
        else:
            request_id = f"prsm-cli-infer-step{step}-{int(_time.time())}"
        deadline = _time.time() + timeout
        # Sprint 668 — multi-stage chain. When --stages is set we
        # build a chain definition and iterate stage-by-stage,
        # threading each stage's output as the next stage's input.
        # Single-stage (default) path keeps the existing behavior.
        n_layers = getattr(
            getattr(hf_model, "config", None), "num_hidden_layers",
            getattr(hf_model.config, "n_layer", 12),
        )
        if stages_list:
            effective_stages = stages_list
        else:
            effective_stages = [(0, int(n_layers), stage_peer_id)]
        chain_total = len(effective_stages)

        current_activation = activation
        resp = None  # filled by final stage
        # Sprint 670 — collect per-stage responses for the
        # receipt's stage_chain array (multi-stage audit).
        stage_responses: list = []
        for stage_idx, (lo, hi, peer_id) in enumerate(effective_stages):
            ho_token = HandoffToken.sign(
                identity=settler, request_id=request_id,
                chain_stage_index=stage_idx,
                chain_total_stages=chain_total,
                deadline_unix=deadline,
            )
            request_kwargs = dict(
                request_id=request_id, model_id=model,
                layer_range=(lo, hi),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=current_activation.tobytes(),
                activation_shape=tuple(current_activation.shape),
                activation_dtype=str(current_activation.dtype),
                upstream_token=ho_token, deadline_unix=deadline,
            )
            if incremental:
                request_kwargs["decode_mode"] = DecodeMode.INCREMENTAL
            request = RunLayerSliceRequest(**request_kwargs)
            req_bytes = encode_message(request)
            try:
                r = _httpx.post(
                    f"{api_url}/admin/chain-exec-ping",
                    json={
                        "peer_id": peer_id,
                        "payload_b64": base64.b64encode(req_bytes).decode("ascii"),
                        "timeout": timeout - 5.0,
                    },
                    timeout=timeout,
                )
            except Exception as exc:
                console.print(
                    f"[red]✗ step {step} stage {stage_idx}: "
                    f"chain-exec-ping failed[/red]: "
                    f"{type(exc).__name__}: {exc}"
                )
                _sys.exit(1)
            if r.status_code != 200:
                console.print(
                    f"[red]✗ step {step} stage {stage_idx}: "
                    f"HTTP {r.status_code}[/red]: {r.text[:300]}"
                )
                _sys.exit(1)
            resp_bytes = base64.b64decode(r.json()["response_b64"])
            try:
                resp = parse_message(resp_bytes)
            except Exception as exc:
                console.print(
                    f"[red]✗ step {step} stage {stage_idx}: "
                    f"parse_message failed[/red]: "
                    f"{type(exc).__name__}: {exc}"
                )
                _sys.exit(1)
            if not hasattr(resp, "activation_blob"):
                msg = getattr(resp, "message", "?")
                code = getattr(resp, "code", "?")
                console.print(
                    f"[red]✗ step {step} stage {stage_idx}: "
                    f"StageError[/red] code={code} message={msg}"
                )
                _sys.exit(1)
            # Decode response activation as input for the next
            # stage (or as final logits if this is the tail stage).
            current_activation = _np.frombuffer(
                resp.activation_blob, dtype=resp.activation_dtype,
            ).reshape(resp.activation_shape)
            # Sprint 670/671 — accumulate per-stage entry for the
            # multi-stage receipt's stage_chain array. Sprint 671
            # adds activation_blob_b64 + tee_attestation_b64 so
            # each stage's signature is self-verifiable (matches
            # the top-level audit-chain pattern from sprint 635).
            stage_responses.append({
                "stage_index": stage_idx,
                "layer_range": [lo, hi],
                "peer_id": peer_id,
                "stage_node_id": resp.stage_node_id,
                "stage_signature_b64": resp.stage_signature_b64,
                "activation_shape": list(resp.activation_shape),
                "activation_dtype": resp.activation_dtype,
                "activation_sha256": _hashlib.sha256(
                    bytes(resp.activation_blob),
                ).hexdigest(),
                "activation_blob_b64": base64.b64encode(
                    bytes(resp.activation_blob),
                ).decode("ascii"),
                "tee_attestation_b64": base64.b64encode(
                    bytes(resp.tee_attestation),
                ).decode("ascii"),
                "tee_type": getattr(
                    resp.tee_type, "value", str(resp.tee_type),
                ),
                "epsilon_spent": resp.epsilon_spent,
                "protocol_version": resp.protocol_version,
                "duration_seconds": resp.duration_seconds,
            })
        # After the per-stage loop, `resp` is the FINAL stage's
        # signed response. Final activation = logits at the tail
        # stage (server applies ln_f + lm_head when
        # is_final_stage=True, which it computes from the layer
        # range covering the model's last layer).
        logits = current_activation
        # Sprint 639 — sampling. The `sampling_mode` string is also
        # written into the receipt so verify-receipts --check-chain
        # knows whether to apply the argmax↔next_token_id invariant
        # (C5). Greedy mode commits the full chain-of-custody chain;
        # temperature/top-k weakens it to "stage signed over these
        # logits + operator recorded a sample drawn from this
        # distribution" — verifier with the seed can re-derive.
        # Sprint 641 — single sampling helper used by both CLI infer
        # and verify-receipts replay. Drift-free by construction.
        from prsm.cli_modules.sampling import (
            sample_token_from_logits, format_sampling_mode,
        )
        last_logits = logits[0, -1, :]
        next_id = sample_token_from_logits(
            last_logits,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            step=step,
        )
        sampling_mode = format_sampling_mode(
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        next_token = tok.decode([next_id])
        text += next_token
        if incremental:
            # Sprint 662 — track token IDs for the next iteration's
            # send-just-the-new-token slice. Re-tokenizing text
            # doesn't always grow by exactly +1 (BPE merges).
            token_ids_so_far.append(next_id)
        step_dt = _time.time() - step_t0

        # Sprint 664 — stop conditions. Check AFTER appending so
        # the stop marker (EOS token or stop string) appears in
        # the output text.
        stop_reason: Optional[str] = None
        if stop_on_eos:
            eos_id = getattr(
                getattr(hf_model, "config", None), "eos_token_id", None,
            )
            if eos_id is not None and next_id == eos_id:
                stop_reason = f"eos_token ({eos_id})"
        if not stop_reason and stop_strings:
            # Match on the FULL generated tail (text minus initial
            # prompt). Single-token matching would miss multi-token
            # stop strings like "###" that tokenize to multiple ids.
            generated_tail = text[len(prompt):]
            for stop_str in stop_strings:
                if stop_str and stop_str in generated_tail:
                    stop_reason = f"stop string {stop_str!r}"
                    break

        # Sprint 634 — record signed receipt for this token. The
        # stage_signature is over the canonical signing payload
        # (chain_rpc.protocol.RunLayerSliceResponse.signing_payload);
        # any party with the stage_node_id's pubkey from the
        # PublisherKeyAnchor can rebuild that payload + verify
        # offline. We also hash the activation_blob (logits) so
        # the receipt commits to the exact output bytes the
        # token was sampled from.
        if receipts_fh is not None:
            logits_sha256 = _hashlib.sha256(
                resp.activation_blob,
            ).hexdigest()
            # Sprint 635: include the bytes the stage_signature was
            # COMPUTED over, so receipts are self-verifying offline.
            # Without these, the activation_sha256 is observation-
            # only (proves the operator saw these specific bytes
            # come back, but can't independently reconstruct the
            # signing payload). With them, anyone can rebuild the
            # canonical signing payload + verify Ed25519 against
            # the stage_node_id's pubkey (from PublisherKeyAnchor).
            #
            # Cost: gpt2 logits are ~1.3MB per token → ~13MB per
            # 10-token run base64-encoded. Operators with size
            # concerns can post-process (gzip the file → ~50%).
            receipt_record = {
                "step": step,
                "wall_unix": _time.time(),
                "request_id": resp.request_id,
                "settler_node_id": settler.node_id,
                "stage_node_id": resp.stage_node_id,
                "stage_signature_b64": resp.stage_signature_b64,
                "model_id": model,
                "layer_range": [0, int(n_layers)],
                "activation_shape": list(resp.activation_shape),
                "activation_dtype": resp.activation_dtype,
                "activation_sha256": logits_sha256,
                "activation_blob_b64": base64.b64encode(
                    bytes(resp.activation_blob),
                ).decode("ascii"),
                "tee_attestation_b64": base64.b64encode(
                    bytes(resp.tee_attestation),
                ).decode("ascii"),
                "duration_seconds": resp.duration_seconds,
                "epsilon_spent": resp.epsilon_spent,
                "tee_type": getattr(
                    resp.tee_type, "value",
                    str(resp.tee_type),
                ),
                "protocol_version": resp.protocol_version,
                "next_token_id": next_id,
                "next_token_text": next_token,
                # Sprint 639 — record sampling mode so the
                # verify-receipts --check-chain (sprint 637) can
                # decide whether the C5 argmax-vs-next_token_id
                # invariant applies. Greedy mode lets C5 fire;
                # non-greedy modes skip C5 (with the seed in the
                # mode string, a verifier could re-derive but that's
                # out of sprint 639's scope).
                "sampling_mode": sampling_mode,
                # Sprint 661/662 — decode_mode field tells C3
                # whether request_id duplicates are legitimate.
                # --incremental shares request_id (cache key);
                # PREFILL gets fresh ids per token.
                "decode_mode": (
                    "incremental" if incremental else "prefill"
                ),
                # Sprint 670 — multi-stage chain provenance.
                # When --stages is used (len > 1), each token's
                # receipt embeds one stage_chain entry per stage
                # so the audit chain captures EVERY stage's
                # signature, not just the final. Sprint 671 will
                # extend verify-receipts to validate each entry
                # against its respective stage_node_id pubkey.
                # Single-stage runs (len == 1) record the field
                # too for schema consistency.
                "stage_chain": stage_responses,
            }
            try:
                receipts_fh.write(_json.dumps(receipt_record) + "\n")
                receipts_fh.flush()
            except OSError as exc:
                # Don't kill the run for an audit-write failure; just
                # surface + carry on. Operator can decide on retry.
                console.print(
                    f"[yellow]⚠ step {step}: receipt write failed: "
                    f"{exc}[/yellow]"
                )
        per_token_records.append({
            "step": step, "token_id": next_id,
            "token_text": next_token, "elapsed_s": step_dt,
            "seq_len_before": (
                len(token_ids_so_far) - 1 if incremental
                else input_ids.shape[-1]
            ),
        })
        if output_format == "text":
            console.print(
                f"  [dim][step {step:2d}][/dim] +token "
                f"{next_id:6d} [cyan]{next_token!r:>14s}[/cyan]  "
                f"([dim]{step_dt:.1f}s[/dim])"
            )
            # Sprint 665 — force flush so operators piping the
            # output through `head`, `tee`, or another process
            # see tokens as they're generated, not in a buffered
            # batch at end-of-run. Rich's default buffering when
            # stdout is not a TTY would otherwise delay visibility
            # by minutes for long generations.
            try:
                console.file.flush()
            except Exception:  # noqa: BLE001
                pass
        # Sprint 664 — break the loop on stop condition.
        if stop_reason is not None:
            if output_format == "text":
                console.print(
                    f"  [dim]→ stopped early: {stop_reason}[/dim]"
                )
            break

    overall_dt = _time.time() - overall_t0
    # Sprint 634 — close receipts sink cleanly. Failure here is
    # already-fsync'd so it doesn't lose tokens; just log.
    if receipts_fh is not None:
        try:
            receipts_fh.close()
        except OSError as exc:
            console.print(
                f"[yellow]⚠ receipts file close failed: {exc}[/yellow]"
            )
    # Sprint 666 — write generated text to --output-file when set.
    # Operators use this for automation pipelines that want clean
    # text (no log lines, no receipt JSON).
    if output_file:
        try:
            out_p = _Path(output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(text)
        except OSError as exc:
            console.print(
                f"[yellow]⚠ output-file write failed: "
                f"{output_file!r}: {exc}[/yellow]"
            )
    if output_format == "json":
        click.echo(_json.dumps({
            "prompt": prompt,
            "model": model,
            "stage_peer_id": stage_peer_id,
            "max_tokens": max_tokens,
            "generated_text": text,
            "elapsed_s": overall_dt,
            "per_token": per_token_records,
            "receipts_path": save_receipts_path,
        }, indent=2))
        return
    console.print()
    console.print(
        f"[green]🎯 Generated[/green] "
        f"({max_tokens} tokens in {overall_dt:.1f}s):"
    )
    console.print(f"  [bold]{text!r}[/bold]")
    if save_receipts_path:
        console.print(
            f"[dim]{max_tokens} signed receipt(s) saved to "
            f"{save_receipts_path}[/dim]"
        )
    if output_file:
        console.print(
            f"[dim]Generated text saved to {output_file}[/dim]"
        )


@node.command("verify-receipts")
@click.argument(
    "receipts_path", type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--check-chain", is_flag=True, default=False,
    help="Sprint 637 — also assert chain-of-custody invariants "
    "across receipts: settler/model consistency, request_id "
    "uniqueness, wall_unix monotonicity, next_token_id matches "
    "argmax of activation_blob. Defends against post-generation "
    "tampering that doesn't invalidate per-token signatures.",
)
@click.option(
    "--strict", is_flag=True, default=False,
    help="Sprint 650 — escalate non-greedy receipts that lack a "
    "seed in sampling_mode from silent-skip (sprint 640 default) "
    "to a UNVERIFIABLE_NON_GREEDY_NO_SEED finding. Operators who "
    "want a tamper-proof audit chain pass --strict so weak runs "
    "(temperature without seed) are surfaced rather than waved "
    "through. Greedy runs are unaffected (always fully verified).",
)
def node_verify_receipts_cli(
    receipts_path: str, output_format: str, check_chain: bool,
    strict: bool,
):
    """Verify per-token signed receipts written by `prsm node infer
    --save-receipts`.

    Sprint 635 — closes the Vision §7 truth-surfacing loop. Each
    receipt's stage_signature_b64 was produced by the stage node
    over the canonical signing payload (chain_rpc.protocol
    RunLayerSliceResponse.signing_payload). This command:

      1. Reads each JSON line from `receipts_path`.
      2. Looks up the stage_node_id's pubkey via the live
         PublisherKeyAnchor (using `_build_anchor_or_none()`).
      3. Reconstructs RunLayerSliceResponse from the receipt's
         fields (activation_blob_b64, tee_attestation_b64, etc.).
      4. Calls verify_with_anchor with expected_stage_node_id =
         the receipt's stage_node_id.
      5. Reports pass/fail per line + an aggregate summary.

    Exit code: 0 if every receipt verified, non-zero otherwise.

    Requires sprint 635+ receipts (activation_blob_b64 field).
    Pre-sprint-635 receipts (sha256 only) will be marked
    UNVERIFIABLE — the receipt proved observation but the
    activation bytes weren't persisted for signature reconstruction.
    """
    import json as _json
    import sys as _sys

    from prsm.cli_modules.receipt_verify import verify_receipts_file
    from prsm.node.inference_wiring import _build_anchor_or_none

    anchor = _build_anchor_or_none()
    if anchor is None:
        console.print(
            "[red]✗ No anchor available[/red]: cannot resolve "
            "stage pubkeys for signature verification.\n"
            "[dim]Set PRSM_NETWORK=mainnet (sprint 629 default "
            "honors networks.py) or pass an explicit "
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS.[/dim]"
        )
        _sys.exit(2)

    # Sprint 636 — verification core lives in cli_modules.receipt_verify
    # for direct unit testing. CLI is a thin renderer over the results.
    results = verify_receipts_file(
        receipts_path, anchor=anchor, check_chain=check_chain,
        strict=strict,
    )

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_total = sum(1 for r in results if r["status"] != "CHAIN_ONLY")
    # Sprint 637 — chain_findings is attached to the last result;
    # extract for rendering + exit-code computation.
    chain_findings = []
    for r in results:
        if "chain_findings" in r:
            chain_findings = r["chain_findings"]
            break
    chain_ok = len(chain_findings) == 0
    # Sprint 672 — multi-stage stage_chain results affect overall
    # exit code. ANY stage failing → not overall_ok.
    stage_chain_failures = 0
    for r in results:
        for sresult in r.get("stage_chain_results", []) or []:
            if sresult.get("status") not in ("OK",):
                stage_chain_failures += 1
    overall_ok = (
        n_ok == n_total
        and (chain_ok if check_chain else True)
        and stage_chain_failures == 0
    )

    if output_format == "json":
        click.echo(_json.dumps({
            "receipts_path": receipts_path,
            "total": n_total,
            "verified": n_ok,
            "check_chain": check_chain,
            "chain_ok": chain_ok if check_chain else None,
            "chain_findings": chain_findings if check_chain else None,
            "results": results,
        }, indent=2))
        _sys.exit(0 if overall_ok else 1)

    for r in results:
        if r.get("status") == "CHAIN_ONLY":
            continue
        if r["status"] == "OK":
            console.print(
                f"  [green]✓[/green] line {r['line']:3d}: "
                f"[dim]{r['stage_node_id'][:16]}...[/dim] "
                f"token=[cyan]{r.get('next_token_text', '?')!r}[/cyan]"
            )
        else:
            console.print(
                f"  [red]✗[/red] line {r['line']:3d}: "
                f"[bold]{r['status']}[/bold] — "
                f"{r.get('reason', '?')}"
            )
        # Sprint 672 — render per-stage results when multi-stage.
        stage_results = r.get("stage_chain_results") or []
        if len(stage_results) > 1:
            for sresult in stage_results:
                status = sresult.get("status", "?")
                sid = sresult.get("stage_node_id") or "?"
                stage_idx = sresult.get("stage_index", "?")
                marker = (
                    "[green]✓[/green]" if status == "OK"
                    else "[red]✗[/red]"
                )
                if status == "OK":
                    console.print(
                        f"    {marker} stage {stage_idx} "
                        f"[dim]{sid[:16]}...[/dim]"
                    )
                else:
                    console.print(
                        f"    {marker} stage {stage_idx} "
                        f"[bold]{status}[/bold] — "
                        f"{sresult.get('reason', '?')}"
                    )
    console.print()
    if n_ok == n_total:
        console.print(
            f"[green]🎯 {n_ok}/{n_total} receipts verified[/green] "
            f"against the live anchor."
        )
        if stage_chain_failures == 0:
            # Tally total per-stage signatures verified for the
            # multi-stage path summary.
            total_stages_verified = sum(
                len(r.get("stage_chain_results") or [])
                for r in results
            )
            if total_stages_verified > 0:
                console.print(
                    f"[green]🔗 {total_stages_verified} per-stage "
                    f"signature(s) verified[/green] (multi-stage "
                    f"audit chain complete)."
                )
    else:
        console.print(
            f"[red]✗ {n_ok}/{n_total} receipts verified[/red]; "
            f"{n_total - n_ok} failed."
        )
    # Sprint 637 — render chain findings
    if check_chain:
        if chain_ok:
            console.print(
                "[green]🔗 chain-of-custody invariants OK[/green] — "
                "settler/model consistency, request_id uniqueness, "
                "wall_unix monotonicity, argmax↔next_token_id match"
            )
        else:
            console.print(
                f"[red]✗ {len(chain_findings)} chain-of-custody "
                f"invariant(s) violated:[/red]"
            )
            for f in chain_findings:
                console.print(
                    f"  [red]●[/red] [bold]{f['kind']}[/bold]: "
                    f"{f['message']} "
                    f"[dim](line(s): {f['line_indices']})[/dim]"
                )
    _sys.exit(0 if overall_ok else 1)


# ──────────────────────────────────────────────────────────────
# Sprint 434 — `prsm node incident ...` CLI trifecta gap
#
# The /admin/incident/* REST surface (sprint <pre-roadmap>) +
# `prsm_incident` MCP tool exist; the CLI lane was the gap per
# PRSM_Testing.md §13 "Operator-trifecta gaps". This block adds
# the read-only triage commands an operator needs at incident
# time: list active incidents, view one in detail, print the
# canonical playbook. The mutating commands (open / advance /
# log-event) need more thought about input-parameter UX and
# stay deferred — operators can hit those via the REST surface
# or `prsm_incident` MCP.
# ──────────────────────────────────────────────────────────────


@node.group("incident", invoke_without_command=False)
def node_incident_group():
    """Incident-response triage commands (read-only).

    Maps to /admin/incident/* REST endpoints. For the
    mutating actions (open / advance / log-event), use
    `prsm_incident` MCP or the REST surface directly.
    """


@node_incident_group.command("list")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--severity", default=None,
    help=(
        "Filter by severity: s0 (catastrophic), s1 "
        "(critical), s2 (high), s3 (low)."
    ),
)
@click.option(
    "--phase", default=None,
    help=(
        "Filter by phase: detected, triaged, contained, "
        "mitigated, postmortem."
    ),
)
def node_incident_list(api_port, output_format, severity, phase):
    """List active incidents on this node.

    Wraps GET /admin/incident. Color-codes by severity in
    text mode; emits JSON-stable payload for ops automation.
    """
    import json as _json
    import urllib.parse
    import urllib.request

    params = {}
    if severity:
        params["severity"] = severity
    if phase:
        params["phase"] = phase
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    url = f"http://127.0.0.1:{api_port}/admin/incident{qs}"

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        # Sprint 647 — F25 sweep: surface FastAPI HTTPException detail
        from prsm.cli_modules.http_errors import render_http_error
        click.echo(render_http_error(exc, "incidents"), err=True)
        sys.exit(2)

    records = payload.get("records", [])
    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    count = payload.get("count", len(records))
    if count == 0:
        console.print(
            "[green]✓ No active incidents.[/green]"
        )
        return

    sev_color = {
        "s0": "red",          # catastrophic
        "s1": "red",          # critical
        "s2": "yellow",       # high
        "s3": "cyan",         # low
    }
    console.print(
        f"[bold]Active incidents ({count}):[/bold]\n"
    )
    for r in records:
        sev = (r.get("severity") or "").lower()
        color = sev_color.get(sev, "white")
        console.print(
            f"[{color}]{sev.upper():<8}[/{color}] "
            f"[bold]{r.get('incident_id', '?')}[/bold] "
            f"phase=[dim]{r.get('phase', '?')}[/dim] "
            f"kind=[dim]{r.get('kind', '?')}[/dim]"
        )
        title = r.get("title") or r.get("description")
        if title:
            console.print(f"  [dim]{title}[/dim]")


@node_incident_group.command("details")
@click.argument("incident_id")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_incident_details(incident_id, api_port, output_format):
    """Show full detail for a single incident.

    Wraps GET /admin/incident/{incident_id}.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/incident/"
        f"{incident_id}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            click.echo(
                f"Incident {incident_id!r} not found.", err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: {exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed to fetch: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    console.print(
        f"[bold]Incident {payload.get('incident_id', '?')}[/bold]"
    )
    console.print(
        f"  severity:  {payload.get('severity', '?')}"
    )
    console.print(
        f"  phase:     {payload.get('phase', '?')}"
    )
    console.print(
        f"  kind:      {payload.get('kind', '?')}"
    )
    if payload.get("title"):
        console.print(f"  title:     {payload['title']}")
    if payload.get("description"):
        console.print(
            f"  desc:      {payload['description']}"
        )
    events = payload.get("events") or []
    if events:
        console.print(f"\n  Events ({len(events)}):")
        for e in events:
            console.print(
                f"    - [dim]{e.get('timestamp', '?')}[/dim] "
                f"{e.get('event_type', '?')}: "
                f"{e.get('description', '')}"
            )


@node_incident_group.command("playbook")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--severity", default=None,
    help="Filter playbook to one severity level.",
)
def node_incident_playbook(api_port, output_format, severity):
    """Show the canonical incident-response playbook.

    Wraps GET /admin/incident/playbook. Per Vision §14:
    the playbook is published BEFORE any incident — this
    command makes it readable from the operator terminal.
    """
    import json as _json
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/incident/playbook"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        from prsm.cli_modules.http_errors import render_http_error
        click.echo(render_http_error(exc, "incident playbook"), err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    decision_tree = payload.get("decision_tree", [])
    if severity:
        decision_tree = [
            d for d in decision_tree
            if d.get("severity", "").lower() == severity.lower()
        ]
    console.print("[bold]Incident Response Playbook[/bold]\n")
    console.print(
        f"[bold]Decision tree ({len(decision_tree)} entries):"
        "[/bold]"
    )
    for d in decision_tree:
        sev = d.get("severity", "?")
        phase = d.get("phase", "?")
        recs = d.get("recommendations", [])
        console.print(
            f"\n  [bold]{sev.upper()}[/bold] / [dim]{phase}"
            f"[/dim]"
        )
        for rec in recs:
            console.print(f"    • {rec}")


# ──────────────────────────────────────────────────────────────
# Sprint 435 — `prsm node insurance ...` CLI trifecta gap
#
# Two endpoints on the REST surface:
#   GET  /admin/insurance-fund/status — current fund state
#   POST /admin/insurance-fund/compose-recovery — produce a
#        multi-sig-uploadable transfer tx payload
#
# Both safe in CLI form: status is read-only; compose-recovery
# only PRODUCES the tx bytes (does NOT execute — Foundation Safe
# holds the actual transfer privilege per Vision §14). Operator
# pipes the JSON output into the multi-sig signing tool of choice.
# ──────────────────────────────────────────────────────────────


@node.group("insurance", invoke_without_command=False)
def node_insurance_group():
    """Insurance-fund triage + recovery-tx composition.

    Maps to /admin/insurance-fund/* REST endpoints.
    """


@node_insurance_group.command("status")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_insurance_status(api_port, output_format):
    """Show current insurance-fund status (balance, recent
    recoveries, etc.).

    Wraps GET /admin/insurance-fund/status. 503 if the
    tracker isn't wired on this node.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{api_port}/admin/insurance-fund/status"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            click.echo(
                "Insurance fund tracker not initialized on "
                "this node.",
                err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: {exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed to fetch: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    console.print("[bold]Insurance Fund Status[/bold]")
    for k, v in payload.items():
        console.print(f"  {k}: [cyan]{v}[/cyan]")


@node_insurance_group.command("compose-recovery")
@click.option(
    "--recipient", required=True,
    help="0x-prefixed Ethereum address to receive funds.",
)
@click.option(
    "--amount-wei", required=True, type=int,
    help="Amount to transfer, in wei (integer).",
)
@click.option(
    "--reason", required=True,
    help=(
        "Recovery reason (logged + included in the audit "
        "trail; e.g., 'Sprint 435 user-X reimbursement')."
    ),
)
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="json",
    show_default=True,
    help=(
        "Default JSON so operators can pipe directly into "
        "multi-sig signing tools."
    ),
)
def node_insurance_compose_recovery(
    recipient, amount_wei, reason, api_port, output_format,
):
    """Compose a multi-sig-uploadable insurance-fund recovery tx.

    Does NOT execute — produces the tx-payload bytes that the
    Foundation Safe operator uploads to their multi-sig
    signing tool. Per Vision §14 invariant: PRSM never executes
    the transfer directly; Foundation Safe holds the privilege.

    Output defaults to JSON so the payload can be piped into
    `safe-cli` or similar. Use --format text for human-readable
    summary.
    """
    import json as _json
    import urllib.error
    import urllib.request

    body = {
        "recipient": recipient,
        "amount_wei": amount_wei,
        "reason": reason,
    }
    url = (
        f"http://127.0.0.1:{api_port}/admin/insurance-fund/"
        f"compose-recovery"
    )
    req = urllib.request.Request(
        url, data=_json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            tx = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (422, 503):
            click.echo(
                f"Compose failed: {exc.read().decode()[:300]}",
                err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: {exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        # Default JSON output → operator can pipe to safe-cli.
        # click.echo (not console.print) keeps ANSI clean.
        click.echo(_json.dumps(tx, indent=2))
        return

    console.print(
        "[bold]Recovery TX composed[/bold] "
        "[dim](not executed — multi-sig must sign)[/dim]\n"
    )
    for k, v in tx.items():
        console.print(f"  {k}: [cyan]{v}[/cyan]")


# ──────────────────────────────────────────────────────────────
# Sprint 436 — `prsm node tee ...` CLI trifecta gap closure
#
# Two endpoints on the REST surface:
#   GET  /admin/tee-policy/node-status — this node's own
#        attestation tier (operators use this to pre-screen
#        before dispatching workloads)
#   POST /admin/tee-policy/evaluate — evaluate an
#        attestation blob against a policy
# ──────────────────────────────────────────────────────────────


@node.group("tee", invoke_without_command=False)
def node_tee_group():
    """TEE attestation + policy evaluation commands.

    Maps to /admin/tee-policy/* REST endpoints.
    """


@node_tee_group.command("status")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_tee_status(api_port, output_format):
    """Show this node's TEE attestation tier.

    Wraps GET /admin/tee-policy/node-status. Enterprises
    use this to pre-screen which nodes are eligible to
    participate in a given workload before dispatching.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/tee-policy/"
        f"node-status"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        from prsm.cli_modules.http_errors import render_http_error
        click.echo(render_http_error(exc, "TEE node status"), err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    tier = payload.get("effective_tier", "?")
    vendor = payload.get("vendor", "?")
    verified = payload.get("vendor_verified", False)
    tier_color = {
        "tee-hardware": "green",
        "tee-software": "yellow",
        "none": "red",
    }.get(tier, "white")
    verified_marker = (
        "[green]✓ vendor-verified[/green]" if verified
        else "[yellow]⚠ not vendor-verified[/yellow]"
    )
    console.print("[bold]TEE Node Status[/bold]")
    console.print(
        f"  effective_tier: [{tier_color}]{tier}[/{tier_color}]"
    )
    console.print(f"  vendor: [cyan]{vendor}[/cyan]")
    console.print(f"  {verified_marker}")
    if payload.get("diagnostic"):
        console.print(
            f"  diagnostic: [dim]{payload['diagnostic']}[/dim]"
        )


@node_tee_group.command("evaluate")
@click.option(
    "--policy-file", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a JSON file containing the TEEPolicy "
        "(min_tier, allowed_vendors, require_vendor_verified, "
        "etc.). See prsm.enterprise.tee_policy.TEEPolicy for "
        "the full schema."
    ),
)
@click.option(
    "--attestation-b64", default=None,
    help=(
        "Base64-encoded attestation blob to evaluate. If "
        "omitted, the policy is evaluated against a missing-"
        "attestation case (useful for pre-flight policy "
        "validation)."
    ),
)
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_tee_evaluate(
    policy_file, attestation_b64, api_port, output_format,
):
    """Evaluate an attestation blob against a TEE policy.

    Wraps POST /admin/tee-policy/evaluate. Returns the
    evaluation result with effective_tier, policy_passed,
    reasons, etc.
    """
    import json as _json
    import urllib.error
    import urllib.request

    try:
        with open(policy_file) as f:
            policy = _json.load(f)
    except _json.JSONDecodeError as exc:
        click.echo(
            f"Policy file is not valid JSON: {exc}", err=True,
        )
        sys.exit(1)

    body = {"policy": policy}
    if attestation_b64:
        body["attestation_b64"] = attestation_b64

    url = (
        f"http://127.0.0.1:{api_port}/admin/tee-policy/evaluate"
    )
    req = urllib.request.Request(
        url, data=_json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            result = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 422:
            click.echo(
                f"Invalid policy or attestation: "
                f"{exc.read().decode()[:300]}",
                err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: "
            f"{exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(result, indent=2))
        return

    passed = result.get("policy_passed", False)
    marker = (
        "[green]✓ POLICY PASSED[/green]" if passed
        else "[red]✗ POLICY FAILED[/red]"
    )
    console.print(f"[bold]TEE Policy Evaluation[/bold] — {marker}\n")
    console.print(
        f"  effective_tier: "
        f"[cyan]{result.get('effective_tier', '?')}[/cyan]"
    )
    if result.get("vendor"):
        console.print(f"  vendor: {result['vendor']}")
    reasons = result.get("reasons") or []
    if reasons:
        console.print("\n  Reasons:")
        for r in reasons:
            console.print(f"    • {r}")


# ──────────────────────────────────────────────────────────────
# Sprint 437 — `prsm node federated ...` + `prsm node pipeline ...`
# CLI trifecta gap closure (read-only admin triage).
#
# Both surfaces are deep (~7 endpoints each — list/details/execute/
# round/aggregate/issue-round/update). This sprint closes the
# read-only triage path (list + details) — the operator subset
# needed at incident time. Mutating commands stay deferred per
# the sprint-434 incident-CLI pattern.
# ──────────────────────────────────────────────────────────────


def _node_admin_list_details(
    *, group_name, list_path, details_path_template,
    api_port, output_format, filter_status,
    json_records_key,
    record_id_field,
):
    """Shared shape between federated + pipeline list commands.

    Both follow: GET /admin/<group>/job[?status=X] returns a
    {jobs|records: [...]} envelope; details endpoint takes a
    job_id path param.
    """
    import json as _json
    import urllib.parse
    import urllib.request

    qs = ""
    if filter_status:
        qs = "?" + urllib.parse.urlencode(
            {"status": filter_status},
        )
    url = f"http://127.0.0.1:{api_port}{list_path}{qs}"

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        # Sprint 647 — F25 sweep: delegate to shared helper that
        # decodes FastAPI HTTPException detail from the response
        # body. Sprint 646's inline implementation lived here;
        # consolidated for DRY + so future fixes apply everywhere.
        from prsm.cli_modules.http_errors import render_http_error
        click.echo(
            render_http_error(exc, f"{group_name} jobs"), err=True,
        )
        sys.exit(2)

    records = payload.get(json_records_key, [])
    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    if not records:
        console.print(
            f"[green]✓ No active {group_name} jobs.[/green]"
        )
        return

    console.print(
        f"[bold]{group_name.capitalize()} jobs "
        f"({len(records)}):[/bold]\n"
    )
    for r in records:
        rid = r.get(record_id_field, "?")
        status = r.get("status", "?")
        console.print(
            f"  [bold]{rid}[/bold] "
            f"status=[cyan]{status}[/cyan]"
        )


@node.group("federated", invoke_without_command=False)
def node_federated_group():
    """Federated-learning admin triage commands (read-only).

    Maps to /admin/federated/* REST endpoints. Mutating
    commands (issue-round, aggregate, update) deferred —
    operators use the REST surface or `prsm_federated_learning`
    MCP for those.
    """


@node_federated_group.command("list")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--status", default=None,
    help="Filter by JobStatus (pending, active, completed, ...).",
)
def node_federated_list(api_port, output_format, status):
    """List federated-learning jobs on this node."""
    _node_admin_list_details(
        group_name="federated",
        list_path="/admin/federated/job",
        details_path_template="/admin/federated/job/{}",
        api_port=api_port, output_format=output_format,
        filter_status=status,
        json_records_key="jobs",
        record_id_field="job_id",
    )


@node_federated_group.command("details")
@click.argument("job_id")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_federated_details(job_id, api_port, output_format):
    """Show details for one federated-learning job."""
    import json as _json
    import urllib.error
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/federated/"
        f"job/{job_id}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            click.echo(
                f"Federated job {job_id!r} not found.", err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: "
            f"{exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    console.print(
        f"[bold]Federated job {payload.get('job_id', '?')}[/bold]"
    )
    for k, v in payload.items():
        if k == "job_id":
            continue
        if isinstance(v, (list, dict)):
            v_str = _json.dumps(v, indent=2)[:200]
        else:
            v_str = str(v)
        console.print(f"  {k}: [cyan]{v_str}[/cyan]")


@node.group("pipeline", invoke_without_command=False)
def node_pipeline_group():
    """Pipeline-inference admin triage commands (read-only).

    Maps to /admin/inference/pipeline/* REST endpoints.
    """


@node_pipeline_group.command("list")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_pipeline_list(api_port, output_format):
    """List pipeline-inference jobs on this node."""
    _node_admin_list_details(
        group_name="pipeline",
        list_path="/admin/inference/pipeline/job",
        details_path_template="/admin/inference/pipeline/job/{}",
        api_port=api_port, output_format=output_format,
        filter_status=None,  # endpoint doesn't accept filter
        json_records_key="jobs",
        record_id_field="job_id",
    )


@node_pipeline_group.command("details")
@click.argument("job_id")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_pipeline_details(job_id, api_port, output_format):
    """Show details for one pipeline-inference job."""
    import json as _json
    import urllib.error
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/inference/"
        f"pipeline/job/{job_id}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            click.echo(
                f"Pipeline job {job_id!r} not found.", err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: "
            f"{exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    console.print(
        f"[bold]Pipeline job {payload.get('job_id', '?')}[/bold]"
    )
    for k, v in payload.items():
        if k == "job_id":
            continue
        if isinstance(v, (list, dict)):
            v_str = _json.dumps(v, indent=2)[:200]
        else:
            v_str = str(v)
        console.print(f"  {k}: [cyan]{v_str}[/cyan]")


if __name__ == "__main__":
    main()
