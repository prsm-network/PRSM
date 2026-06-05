"""Sprint 575 — F29 fix: default bootstrap URL must resolve.

User renamed bootstrap1.prsm-network.com → bootstrap-us.prsm-network.com
in DigitalOcean + Cloudflare (per reference_bootstrap_us_dns_rename memory
entry on 2026-05-19). The old DNS no longer resolves.

But the following code paths still defaulted to the dead hostname:
- ``prsm/node/config.py:32`` — BOOTSTRAP_PRIMARY env-var fallback
- ``prsm/node/discovery.py:136`` — _DEFAULT_BOOTSTRAP class constant
- ``prsm/interface/api/onboarding_router.py`` — 4 UI default values
- ``prsm/interface/api/templates/onboarding/network.html`` — placeholder

Result: every new operator booting ``prsm node start`` today FAILS
initial bootstrap because the resolver returns nothing. Daemon then
either waits forever or falls back to a sibling region (the
sprint-375 fallback list saved bacon for some, not all).

Sprint 575 = surgical hostname update. No behavior change — just
the production-correct default. After the update, ``prsm node start``
with no flags reaches a working bootstrap-server.

(Docstrings + audit RFP markdown stay on bootstrap1 — those are
historical references. Operational defaults only.)
"""
from __future__ import annotations


def test_node_config_default_bootstrap_uses_bootstrap_us():
    """NodeConfig's BOOTSTRAP_PRIMARY fallback must be bootstrap-us
    (resolves to 159.203.129.218) — not bootstrap1 (dead DNS).
    """
    from prsm.node.config import NodeConfig
    cfg = NodeConfig.load()
    bootstraps = cfg.bootstrap_nodes
    # First entry is the PRIMARY — must be the live hostname
    assert bootstraps, "bootstrap_nodes empty — default missing"
    primary = bootstraps[0]
    assert "bootstrap-us" in primary or "bootstrap-eu" in primary or "bootstrap-apac" in primary, (
        f"Default bootstrap_nodes primary {primary!r} still on "
        f"legacy hostname — operators booting today fail DNS"
    )
    assert "bootstrap1.prsm-network.com" not in primary, (
        f"bootstrap1 hostname no longer resolves; default must "
        f"point at a live name. Got: {primary!r}"
    )


def test_discovery_default_constant_uses_live_hostname():
    """PeerDiscovery's _DEFAULT_BOOTSTRAP local in __init__ likewise.
    Checked via source-text grep since it's a function-local var.
    """
    import prsm.node.discovery as _mod
    import inspect
    src = inspect.getsource(_mod)
    dead = "wss://bootstrap1.prsm-network.com:8765"
    assert dead not in src, (
        f"prsm/node/discovery.py still ships {dead!r} as a default "
        f"— operators using sprint 575+ defaults hit dead DNS"
    )


def test_onboarding_router_defaults_use_live_hostname():
    """Onboarding UI defaults likewise. Read the source module
    text since the values are Form defaults / dict literals that
    aren't directly callable.
    """
    import prsm.interface.api.onboarding_router as _mod
    import inspect
    src = inspect.getsource(_mod)
    # Source MAY still have "bootstrap1" in a comment / migration
    # note, but the default *values* must not.
    # Crude check: no occurrence of the dead URL as a string literal.
    dead = "wss://bootstrap1.prsm-network.com:8765"
    assert dead not in src, (
        f"Onboarding router still ships {dead!r} as a default — "
        f"new operators via UI flow hit dead DNS"
    )


def test_legacy_bootstrap1_in_stored_config_auto_migrates(tmp_path):
    """An operator's stored config.json with legacy bootstrap1
    entries must be auto-migrated to bootstrap-us on NodeConfig.load().
    Without this, every existing operator gets stranded on dead DNS
    after `prsm upgrade`.
    """
    import json
    from prsm.node.config import NodeConfig

    stored = tmp_path / "node_config.json"
    stored.write_text(json.dumps({
        "bootstrap_nodes": [
            "wss://bootstrap1.prsm-network.com:8765",
            "wss://bootstrap-eu.prsm-network.com:8765",
        ],
    }))

    cfg = NodeConfig.load(path=stored)
    assert cfg.bootstrap_nodes[0] == "wss://bootstrap-us.prsm-network.com:8765"
    # EU fallback preserved as-is
    assert cfg.bootstrap_nodes[1] == "wss://bootstrap-eu.prsm-network.com:8765"


def test_operator_cloud_init_default_uses_live_bootstrap_host():
    """sp575 follow-on (Tier-1 audit gap 4): the turnkey operator cloud-init
    script feeds its BOOTSTRAP_URL to `prsm node start --bootstrap`, which
    BYPASSES the config-file bootstrap1->bootstrap-us auto-migration. So a
    stale default here strands every first-boot turnkey node on dead DNS —
    the single most important first-boot promise (fleet connectivity).
    """
    from pathlib import Path
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "operator-node-cloud-init.sh"
    )
    assert script.is_file(), f"cloud-init script missing at {script}"
    text = script.read_text(encoding="utf-8")
    assert "bootstrap1.prsm-network.com" not in text, (
        "operator-node-cloud-init.sh still references dead bootstrap1; its "
        "--bootstrap override bypasses the config auto-migration, so "
        "first-boot turnkey nodes hit dead DNS"
    )
    assert "bootstrap-us.prsm-network.com" in text, (
        "operator-node-cloud-init.sh must default to the live bootstrap-us host"
    )


def test_smoke_test_fleet_script_uses_live_bootstrap_host():
    """sp575 follow-on (Tier-1 audit gap 4): the fleet smoke-test probes a
    hardcoded canonical host list; it must probe the live bootstrap-us, not
    the dead bootstrap1.
    """
    from pathlib import Path
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "smoke-test-bootstrap-fleet.sh"
    )
    assert script.is_file(), f"smoke-test script missing at {script}"
    text = script.read_text(encoding="utf-8")
    assert "bootstrap1.prsm-network.com" not in text, (
        "smoke-test-bootstrap-fleet.sh still probes dead bootstrap1"
    )
    assert "bootstrap-us.prsm-network.com" in text, (
        "smoke-test-bootstrap-fleet.sh must probe the live bootstrap-us host"
    )


def test_onboarding_template_placeholder_uses_live_hostname():
    """The network.html placeholder operators see in the UI must
    not be the dead URL either.
    """
    from pathlib import Path
    template = (
        Path(__file__).parent.parent.parent
        / "prsm" / "interface" / "api" / "templates"
        / "onboarding" / "network.html"
    )
    if not template.exists():
        return  # not authoritative — skip
    text = template.read_text(encoding="utf-8")
    assert "wss://bootstrap1.prsm-network.com:8765" not in text, (
        "Onboarding network.html placeholder still on dead bootstrap1 "
        "URL — operators copying the placeholder hit dead DNS"
    )
