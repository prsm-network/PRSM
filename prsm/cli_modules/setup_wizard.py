"""PRSM Setup Wizard — interactive first-run configuration experience.

7-step wizard that detects your system, configures resources, sets up
API keys, network, and AI integration. All settings saved to ~/.prsm/config.yaml.

Usage:
    prsm setup              # full interactive wizard
    prsm setup --minimal    # auto-detect, just confirm
    prsm setup --dry-run    # walk through without saving
    prsm setup --reset      # clear existing config first
"""
import os
import platform
import shutil
import socket
import uuid
from pathlib import Path

import click

from .ui import (
    console,
    show_banner,
    step_header,
    success,
    warning,
    error,
    info,
    muted,
    prompt_text,
    prompt_choice,
    prompt_confirm,
    prompt_number,
    summary_panel,
    resource_bar,
    section,
)
from .config_schema import PRSMConfig, NodeRole

# Sprint 148 — canonical bootstrap source-of-truth lives in
# prsm.node.config. Re-export so tests can pin equivalence and
# other consumers can import from either place without drift.
from prsm.node.config import (
    DEFAULT_BOOTSTRAP_NODES as _NODE_DEFAULT_BOOTSTRAP_NODES,
)
DEFAULT_BOOTSTRAP_NODES = list(_NODE_DEFAULT_BOOTSTRAP_NODES)


# ── System Detection Helpers ─────────────────────────────────────────


def _detect_system() -> dict:
    """Detect system specs. Uses psutil if available, falls back gracefully."""
    sys_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": platform.processor() or "Unknown",
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": 0.0,
        "gpu": "Not detected",
        "disk_total_gb": 0.0,
        "disk_free_gb": 0.0,
    }

    # RAM via psutil
    try:
        import psutil
        mem = psutil.virtual_memory()
        sys_info["ram_gb"] = round(mem.total / (1024 ** 3), 1)
    except ImportError:
        # Fallback: try /proc/meminfo on Linux, sysctl on macOS
        try:
            if sys_info["os"] == "Darwin":
                import subprocess
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                sys_info["ram_gb"] = round(int(out.strip()) / (1024 ** 3), 1)
            elif sys_info["os"] == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            sys_info["ram_gb"] = round(kb / (1024 ** 2), 1)
                            break
        except Exception:
            sys_info["ram_gb"] = 0.0

    # Disk space
    try:
        usage = shutil.disk_usage(Path.home())
        sys_info["disk_total_gb"] = round(usage.total / (1024 ** 3), 1)
        sys_info["disk_free_gb"] = round(usage.free / (1024 ** 3), 1)
    except Exception:
        pass

    # GPU detection
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            sys_info["gpu"] = result.stdout.strip().split("\n")[0]
    except Exception:
        # Try macOS Metal
        if sys_info["os"] == "Darwin" and sys_info["arch"] == "arm64":
            sys_info["gpu"] = "Apple Silicon (Metal)"

    return sys_info


def _check_port(port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def _check_prerequisites(sys_info: dict) -> list:
    """Check prerequisites, return list of (name, ok, message) tuples."""
    checks = []

    # Python version
    py_parts = sys_info["python"].split(".")
    py_ok = int(py_parts[0]) >= 3 and int(py_parts[1]) >= 10
    checks.append(("Python 3.10+", py_ok, f"Python {sys_info['python']}"))

    # RAM
    ram_ok = sys_info["ram_gb"] >= 4.0
    if sys_info["ram_gb"] > 0:
        checks.append(("RAM ≥ 4 GB", ram_ok, f"{sys_info['ram_gb']} GB"))
    else:
        checks.append(("RAM ≥ 4 GB", None, "Could not detect"))

    # Disk
    disk_ok = sys_info["disk_free_gb"] >= 5.0
    if sys_info["disk_free_gb"] > 0:
        checks.append(("Disk ≥ 5 GB free", disk_ok, f"{sys_info['disk_free_gb']} GB free"))
    else:
        checks.append(("Disk ≥ 5 GB free", None, "Could not detect"))

    return checks


def _save_api_key(key_name: str, key_value: str):
    """Append an API key to ~/.prsm/.env."""
    env_path = PRSMConfig.env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing content
    existing = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()

    existing[key_name] = key_value

    with open(env_path, "w") as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")


# ── Wizard Steps ─────────────────────────────────────────────────────


def _step_welcome(config: PRSMConfig, sys_info: dict, minimal: bool) -> dict:
    """Step 1: Welcome + System Detection."""
    from rich.table import Table

    step_header(1, 7, "Welcome & System Detection")
    show_banner()

    info("Tip: Press Enter to accept [default] values throughout this wizard.")
    console.print()
    info("Detecting your system...")
    console.print()

    # System table
    table = Table(show_header=False, border_style="dim white", padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold white")

    table.add_row("OS", f"{sys_info['os']} ({sys_info['arch']})")
    table.add_row("CPU", f"{sys_info['cpu_model']} ({sys_info['cpu_cores']} cores)")
    if sys_info["ram_gb"] > 0:
        table.add_row("RAM", f"{sys_info['ram_gb']} GB")
    table.add_row("GPU", sys_info["gpu"])
    if sys_info["disk_total_gb"] > 0:
        table.add_row("Disk", f"{sys_info['disk_free_gb']} GB free / {sys_info['disk_total_gb']} GB total")
    table.add_row("Python", sys_info["python"])

    console.print(table)
    console.print()

    # Prerequisites
    prereqs = _check_prerequisites(sys_info)
    for name, ok, msg in prereqs:
        if ok is True:
            success(f"{name}: {msg}")
        elif ok is False:
            error(f"{name}: {msg}")
        else:
            warning(f"{name}: {msg}")

    console.print()

    # Node display name
    default_name = f"prsm-{platform.node().split('.')[0].lower()[:16]}"
    if minimal:
        config.display_name = default_name
        info(f"Node name: {default_name}")
    else:
        config.display_name = prompt_text("Node display name", default=default_name)

    return sys_info


def _step_role(config: PRSMConfig, minimal: bool):
    """Step 2: Intent / Role Selection."""
    step_header(2, 7, "Role Selection")

    if minimal:
        config.node_role = NodeRole.FULL
        info("Role: Full Node (contribute & consume)")
        return

    choices = [
        {
            "label": "contributor",
            "value": "contributor",
            "hint": "Earn FTNS by sharing compute & storage",
        },
        {
            "label": "consumer",
            "value": "consumer",
            "hint": "Use AI services, curate data, run queries",
        },
        {
            "label": "full",
            "value": "full",
            "hint": "Both — contribute resources and use the network",
        },
    ]

    role = prompt_choice("What would you like to do with PRSM?", choices, default=2)
    config.node_role = NodeRole(role)

    if config.node_role == NodeRole.CONTRIBUTOR:
        info("You'll share compute/storage and earn FTNS tokens.")
    elif config.node_role == NodeRole.CONSUMER:
        info("You'll use the network for AI services and data curation.")
    else:
        info("Full node — you'll contribute resources and use the network.")


def _step_resources(config: PRSMConfig, sys_info: dict, minimal: bool):
    """Step 3: Resource Allocation."""
    step_header(3, 7, "Resource Allocation")

    if config.node_role == NodeRole.CONSUMER:
        info("Resource allocation skipped (consumer-only mode).")
        config.cpu_pct = 10
        config.memory_pct = 10
        config.gpu_pct = 0
        config.storage_gb = 1.0
        config.max_concurrent_jobs = 1
        return

    # Smart defaults based on system
    ram = sys_info["ram_gb"] if sys_info["ram_gb"] > 0 else 8.0
    default_storage = min(max(sys_info["disk_free_gb"] * 0.05, 5.0), 100.0) if sys_info["disk_free_gb"] > 0 else 10.0
    default_jobs = max(1, min(sys_info["cpu_cores"] // 2, 8))
    has_gpu = sys_info["gpu"] != "Not detected"

    if minimal:
        config.cpu_pct = 50
        config.memory_pct = 50
        config.gpu_pct = 80 if has_gpu else 0
        config.storage_gb = round(default_storage, 1)
        config.max_concurrent_jobs = default_jobs
    else:
        info("Configure how much of your system PRSM can use.")
        info(f"System: {sys_info['cpu_cores']} cores, {ram} GB RAM, {sys_info['disk_free_gb']} GB free disk")
        console.print()

        config.cpu_pct = prompt_number("CPU allocation %", default=50, min_val=10, max_val=90)
        resource_bar("CPU", config.cpu_pct)

        config.memory_pct = prompt_number("Memory allocation %", default=50, min_val=10, max_val=90)
        resource_bar("Memory", config.memory_pct)

        if has_gpu:
            config.gpu_pct = prompt_number("GPU allocation %", default=80, min_val=0, max_val=100)
            resource_bar("GPU", config.gpu_pct)
        else:
            config.gpu_pct = 0
            muted("No GPU detected — skipping GPU allocation.")

        config.storage_gb = float(prompt_number(
            f"Storage pledge (GB, {sys_info['disk_free_gb']} GB available)",
            default=int(default_storage),
            min_val=1,
            max_val=int(sys_info["disk_free_gb"]) if sys_info["disk_free_gb"] > 0 else 1000,
        ))
        disk_pct = int((config.storage_gb / sys_info["disk_free_gb"]) * 100) if sys_info["disk_free_gb"] > 0 else 0
        resource_bar("Storage", disk_pct)

        config.max_concurrent_jobs = prompt_number(
            "Max concurrent jobs", default=default_jobs, min_val=1, max_val=20
        )

    console.print()
    success("Resource allocation configured.")
    resource_bar("CPU", config.cpu_pct)
    resource_bar("Memory", config.memory_pct)
    if has_gpu:
        resource_bar("GPU", config.gpu_pct)
    info(f"Storage: {config.storage_gb} GB | Max jobs: {config.max_concurrent_jobs}")


def _step_api_keys(config: PRSMConfig, minimal: bool):
    """Step 4: API Keys & Wallet."""
    step_header(4, 7, "API Keys & Wallet")

    if minimal:
        info("API key setup skipped in minimal mode.")
        info("Run 'prsm setup' later to configure API keys.")
        return

    info("API keys are optional but enable richer AI capabilities.")
    info("Keys are stored in ~/.prsm/.env (never in config.yaml).")
    console.print()

    # OpenRouter
    console.print()
    if prompt_confirm("Configure OpenRouter API key? (primary LLM access — supports 200+ models including Claude, Gemini, Llama)", default=True):
        key = prompt_text("OpenRouter API key (starts with 'sk-or-v1-')", default="")
        if key and key.startswith("sk-or-v1-"):
            _save_api_key("OPENROUTER_API_KEY", key)
            config.has_openrouter_key = True
            success("OpenRouter key saved.")
            # Ask which model to use
            console.print()
            info("OpenRouter supports 200+ models. Which would you like to use?")
            console.print("  [dim]Popular choices (leave blank for default):[/dim]")
            console.print("  [dim]  google/gemini-2.0-flash-lite-001  (fast, cheap)[/dim]")
            console.print("  [dim]  google/gemini-flash-1.5-8b         (free tier)[/dim]")
            console.print("  [dim]  anthropic/claude-3-5-sonnet        (quality)[/dim]")
            console.print("  [dim]  meta-llama/llama-3.3-70b-instruct  (open-weight)[/dim]")
            model = prompt_text("OpenRouter model (leave blank for google/gemini-2.0-flash-lite-001)", default="")
            if model:
                _save_api_key("OPENROUTER_DEFAULT_MODEL", model)
                info(f"Model set to: {model}")
            else:
                info("Using default model: google/gemini-2.0-flash-lite-001")
        elif key:
            warning("Key doesn't look like an OpenRouter key (should start with 'sk-or-v1-').")
            if prompt_confirm("Save anyway?", default=False):
                _save_api_key("OPENROUTER_API_KEY", key)
                config.has_openrouter_key = True

    # OpenAI
    if prompt_confirm("Configure OpenAI API key? (for embeddings & semantic search, not your chat model)", default=False):
        key = prompt_text("OpenAI API key — for embeddings & semantic search (optional)", default="")
        if key and key.startswith("sk-"):
            _save_api_key("OPENAI_API_KEY", key)
            config.has_openai_key = True
            success("OpenAI key saved.")
        elif key:
            warning("Key doesn't look like an OpenAI key (should start with 'sk-').")
            if prompt_confirm("Save anyway?", default=False):
                _save_api_key("OPENAI_API_KEY", key)
                config.has_openai_key = True

    # Anthropic
    if prompt_confirm("Configure Anthropic API key? (for safety checks & content moderation)", default=False):
        key = prompt_text("Anthropic API key — for safety checks & content moderation (optional)", default="")
        if key and key.startswith("sk-ant-"):
            _save_api_key("ANTHROPIC_API_KEY", key)
            config.has_anthropic_key = True
            success("Anthropic key saved.")
        elif key:
            warning("Key doesn't look like an Anthropic key (should start with 'sk-ant-').")
            if prompt_confirm("Save anyway?", default=False):
                _save_api_key("ANTHROPIC_API_KEY", key)
                config.has_anthropic_key = True

    # HuggingFace
    if prompt_confirm("Configure HuggingFace token? (for downloading open-source models)", default=False):
        token = prompt_text("HuggingFace token", default="")
        if token and token.startswith("hf_"):
            _save_api_key("HUGGINGFACE_TOKEN", token)
            config.has_huggingface_token = True
            success("HuggingFace token saved.")
        elif token:
            warning("Token doesn't look like a HuggingFace token (should start with 'hf_').")
            if prompt_confirm("Save anyway?", default=False):
                _save_api_key("HUGGINGFACE_TOKEN", token)
                config.has_huggingface_token = True

    # FTNS Wallet
    console.print()
    info("FTNS wallet for earning/spending tokens on the network.")
    if config.wallet_address:
        info(f"Existing wallet: {config.wallet_address}")
        if not prompt_confirm("Keep existing wallet?", default=True):
            config.wallet_address = None

    if not config.wallet_address:
        choices = [
            {"label": "Generate new wallet", "value": "new", "hint": "Create a fresh FTNS address"},
            {"label": "Import existing", "value": "import", "hint": "Enter an existing wallet address"},
            {"label": "Skip for now", "value": "skip", "hint": "Set up wallet later"},
        ]
        wallet_choice = prompt_choice("FTNS Wallet", choices, default=0)
        if wallet_choice == "new":
            config.wallet_address = f"ftns_{uuid.uuid4().hex[:32]}"
            success(f"Generated wallet: {config.wallet_address}")
        elif wallet_choice == "import":
            addr = prompt_text("Wallet address")
            config.wallet_address = addr
            success(f"Wallet imported: {addr}")
        else:
            info("Wallet setup skipped.")


def _step_network(config: PRSMConfig, minimal: bool):
    """Step 4: Network Configuration."""
    step_header(4, 7, "Network Configuration")

    # Sprint 148 — read from canonical source instead of hardcoding.
    # Pre-fix the wizard pinned `bootstrap1.prsm.network` (wrong domain)
    # in `/dns4/.../p2p/QmPRSM1` multiaddr (wrong format + placeholder
    # PeerID). DEFAULT_BOOTSTRAP_NODES is the real `wss://...com:8765`
    # URL the bootstrap server actually serves on.
    default_bootstrap = list(DEFAULT_BOOTSTRAP_NODES)

    if minimal:
        config.p2p_port = 9001
        config.api_port = 8000
        config.bootstrap_nodes = default_bootstrap

        info("Network requires 2 ports:")
        muted("1. P2P port — for node-to-node communication")
        muted("2. API port — for local HTTP API access")
        console.print()

        p2p_ok = _check_port(config.p2p_port)
        api_ok = _check_port(config.api_port)

        if p2p_ok:
            success(f"P2P port {config.p2p_port}: available")
        else:
            warning(f"P2P port {config.p2p_port}: in use (may conflict)")

        if api_ok:
            success(f"API port {config.api_port}: available")
        else:
            warning(f"API port {config.api_port}: in use (may conflict)")

        info(f"Bootstrap nodes: {len(default_bootstrap)} configured")
        return

    # Port context
    info("Network requires 2 ports:")
    muted("1. P2P port — for node-to-node communication")
    muted("2. API port — for local HTTP API access")
    console.print()

    # P2P port
    while True:
        config.p2p_port = prompt_number("P2P port", default=9001, min_val=1024, max_val=65535)
        if _check_port(config.p2p_port):
            success(f"Port {config.p2p_port}: available")
            break
        else:
            warning(f"Port {config.p2p_port} is in use.")
            if prompt_confirm("Use it anyway?", default=False):
                break

    # API port
    while True:
        config.api_port = prompt_number("API port", default=8000, min_val=1024, max_val=65535)
        if config.api_port == config.p2p_port:
            error("API port must differ from P2P port.")
            continue
        if _check_port(config.api_port):
            success(f"Port {config.api_port}: available")
            break
        else:
            warning(f"Port {config.api_port} is in use.")
            if prompt_confirm("Use it anyway?", default=False):
                break

    # Bootstrap nodes
    config.bootstrap_nodes = default_bootstrap
    info(f"Default bootstrap nodes: {len(default_bootstrap)} configured")
    if prompt_confirm("Add custom bootstrap node? (advanced)", default=False):
        node = prompt_text("Bootstrap node multiaddr")
        if node:
            config.bootstrap_nodes.append(node)
            success(f"Added: {node}")


def _step_ai_integration(config: PRSMConfig, minimal: bool):
    """Step 5: AI Integration (MCP Server)."""
    step_header(5, 7, "AI Integration")

    if minimal:
        config.mcp_server_enabled = True
        config.mcp_server_port = 9100
        info("MCP server: enabled on port 9100")
        return

    info("The MCP server lets AI assistants (Hermes, Claude, etc.) use PRSM.")
    console.print()

    config.mcp_server_enabled = prompt_confirm("Enable MCP server?", default=True)

    if config.mcp_server_enabled:
        config.mcp_server_port = prompt_number(
            "MCP server port", default=9100, min_val=1024, max_val=65535
        )

        if _check_port(config.mcp_server_port):
            success(f"MCP port {config.mcp_server_port}: available")
        else:
            warning(f"MCP port {config.mcp_server_port}: in use")

        # Show config snippet
        console.print()
        info("Add this to your AI assistant's MCP config:")
        snippet = (
            f'  prsm:\n'
            f'    transport: streamable-http\n'
            f'    url: http://localhost:{config.mcp_server_port}/mcp'
        )
        from rich.panel import Panel
        console.print(Panel(
            snippet,
            title="  MCP Config Snippet  ",
            title_align="left",
            border_style="dim white",
            padding=(1, 2),
        ))
    else:
        info("MCP server disabled. Enable later with 'prsm config set mcp_server_enabled true'.")


def _step_review(config: PRSMConfig, dry_run: bool) -> bool:
    """Step 7: Review & Launch. Returns True if user confirms."""
    step_header(7, 7, "Review & Launch")

    # Build summary
    items = {
        "Node Name": config.display_name,
        "Role": config.node_role.value,
        "CPU Allocation": f"{config.cpu_pct}%",
        "Memory Allocation": f"{config.memory_pct}%",
        "GPU Allocation": f"{config.gpu_pct}%",
        "Storage Pledge": f"{config.storage_gb} GB",
        "Max Jobs": str(config.max_concurrent_jobs),
        "P2P Port": str(config.p2p_port),
        "API Port": str(config.api_port),
        "Bootstrap Nodes": str(len(config.bootstrap_nodes)),
        "MCP Server": f"{'enabled' if config.mcp_server_enabled else 'disabled'}"
                      + (f" (:{config.mcp_server_port})" if config.mcp_server_enabled else ""),
        "FTNS Wallet": config.wallet_address or "Not configured",
        "OpenRouter Key": "✓ configured" if config.has_openrouter_key else "—",
        "OpenAI Key": "✓ configured" if config.has_openai_key else "—",
        "Anthropic Key": "✓ configured" if config.has_anthropic_key else "—",
        "HuggingFace": "✓ configured" if config.has_huggingface_token else "—",
    }

    summary_panel("PRSM Configuration Summary", items)

    # Resource bars
    console.print()
    resource_bar("CPU", config.cpu_pct)
    resource_bar("Memory", config.memory_pct)
    if config.gpu_pct > 0:
        resource_bar("GPU", config.gpu_pct)
    console.print()

    if dry_run:
        warning("Dry run — nothing will be saved.")
        return True

    if not prompt_confirm("Everything look good? Save configuration?", default=True):
        warning("Setup cancelled. Run 'prsm setup' to try again.")
        return False

    # Save
    path = config.save()
    config.setup_completed = True
    config.save()
    success(f"Configuration saved to {path}")

    # Next steps
    console.print()
    section("Next Steps", icon="arrow")
    info("prsm serve          — Start the PRSM API server")
    info("prsm status         — Check system status")
    info("prsm setup --reset  — Reconfigure from scratch")
    info("prsm node start     — Start a PRSM node")
    console.print()

    return True


# ── Main Entry Point ─────────────────────────────────────────────────


def run_setup_wizard(dry_run: bool = False, minimal: bool = False, reset: bool = False):
    """Run the PRSM setup wizard.

    Args:
        dry_run: Walk through without saving any files.
        minimal: Auto-detect everything, just ask for confirmation.
        reset: Clear existing config before starting.
    """
    try:
        # Handle reset
        if reset:
            if PRSMConfig.exists():
                if prompt_confirm("Reset existing configuration?", default=False):
                    PRSMConfig().reset()
                    success("Configuration cleared.")
                else:
                    info("Reset cancelled.")
                    return
            else:
                info("No existing configuration to reset.")

        # Check for existing config
        if PRSMConfig.exists() and not reset:
            existing = PRSMConfig.load()
            if existing.setup_completed:
                warning("PRSM is already configured.")
                # sp1395 — --dry-run is a non-interactive PREVIEW and --minimal runs with defaults;
                # neither should block on an interactive "re-run?" prompt. Only ask in the plain
                # interactive path.
                if (not dry_run and not minimal
                        and not prompt_confirm("Re-run setup wizard?", default=False)):
                    info("Use 'prsm setup --reset' to start fresh.")
                    return

        # Initialize config
        config = PRSMConfig()

        # Detect system
        sys_info = _detect_system()

        # Run steps (7 steps total). sp1395 — --dry-run is a non-interactive PREVIEW, so run the
        # steps with defaults (like --minimal) instead of prompting; nothing is saved anyway.
        noninteractive = minimal or dry_run
        _step_welcome(config, sys_info, noninteractive)
        _step_role(config, noninteractive)
        _step_resources(config, sys_info, noninteractive)
        _step_api_keys(config, noninteractive)
        _step_network(config, noninteractive)
        _step_ai_integration(config, noninteractive)
        _step_review(config, dry_run)

    except click.Abort:
        console.print()
        warning("Setup interrupted. Run 'prsm setup' to continue later.")
    except KeyboardInterrupt:
        console.print()
        warning("Setup interrupted. Run 'prsm setup' to continue later.")
    except Exception as e:
        console.print()
        error(f"Setup error: {e}")
        muted("Please report this issue at https://github.com/PRSM/prsm/issues")
        raise
